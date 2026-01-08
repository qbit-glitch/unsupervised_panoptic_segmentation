#!/usr/bin/env python
"""
Train SpectralDiffusion on a subset of a dataset.

Usage:
    python train_subset.py --dataset clevr --subset-pct 0.01 --epochs 10
    python train_subset.py --dataset coco --subset-pct 0.05 --epochs 20
"""
import sys
import os
import random
import argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from train import parse_args, create_dataloader, create_model, train_epoch, evaluate


def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    """Cosine schedule with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    # Parse subset-specific args
    parser = argparse.ArgumentParser(description="Train on dataset subset")
    parser.add_argument("--dataset", type=str, required=True, choices=["synthetic", "clevr", "clevr_masks", "coco"])
    parser.add_argument("--val-dataset", type=str, default=None, choices=["synthetic", "clevr", "clevr_masks", "coco"],
                       help="Validation dataset (default: same as --dataset). Use clevr_masks for proper ARI.")
    parser.add_argument("--subset-pct", type=float, default=0.01, help="Fraction of dataset to use (default: 0.01 = 1%%)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-slots", type=int, default=11)
    parser.add_argument("--save-dir", type=str, default="./outputs")
    parser.add_argument("--debug", action="store_true")
    subset_args = parser.parse_args()
    
    # Create args namespace with required fields (don't use parse_args from train.py)
    args = argparse.Namespace(
        dataset=subset_args.dataset,
        val_dataset=subset_args.val_dataset,
        epochs=subset_args.epochs,
        batch_size=subset_args.batch_size,
        lr=subset_args.lr,
        num_slots=subset_args.num_slots,
        data_dir='../datasets',
        mixed_precision=True,
        image_size=[128, 128],
        device='mps',
        wandb=False,
        output_dir=subset_args.save_dir,
        exp_name=f'{subset_args.dataset}_{int(subset_args.subset_pct*100)}pct',
        log_interval=10,
        eval_interval=1,
        warmup_epochs=2,
        weight_decay=1e-5,
        grad_clip=1.0,
        use_mamba=True,
        init_mode='spectral',
        lambda_spec=0.1,
        lambda_ident=0.01,
        debug=subset_args.debug,
        fast_dev_run=False,
        num_workers=0,  # Avoid multiprocessing issues on MPS
        full_model=False,
        use_dino=False,
    )
    
    print('='*60)
    print(f'{subset_args.dataset.upper()} {int(subset_args.subset_pct*100)}% Training')
    if args.val_dataset:
        print(f'Validation: {args.val_dataset.upper()}')
    print('='*60)
    print(f'Epochs: {args.epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.lr}')
    print(f'Num slots: {args.num_slots}')
    print('='*60)
    
    # Load data
    # Use val_dataset for validation if specified (for proper ARI computation)
    print('\nLoading dataset...')
    train_loader_full = create_dataloader(args, 'train')
    val_loader_full = create_dataloader(args, 'val', dataset_override=args.val_dataset)
    
    # Create subset
    train_size = len(train_loader_full.dataset)
    val_size = len(val_loader_full.dataset)
    train_subset_size = max(100, int(train_size * subset_args.subset_pct))
    val_subset_size = max(50, int(val_size * subset_args.subset_pct))
    
    train_indices = random.sample(range(train_size), min(train_subset_size, train_size))
    val_indices = random.sample(range(val_size), min(val_subset_size, val_size))
    
    train_subset = Subset(train_loader_full.dataset, train_indices)
    val_subset = Subset(val_loader_full.dataset, val_indices)
    
    # Use num_workers=0 to avoid multiprocessing issues on MPS
    train_loader = DataLoader(
        train_subset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,
        drop_last=True
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    print(f'Training samples: {len(train_subset)} ({subset_args.subset_pct*100:.1f}% of {train_size})')
    print(f'Validation samples: {len(val_subset)}')
    print(f'Batches per epoch: {len(train_loader)}')
    
    # Create model
    print('\nCreating model...')
    model = create_model(args)
    if args.mixed_precision:
        model = model.to(torch.float32)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {num_params:,}')
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print('\nStarting training...')
    best_ari = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, args, epoch)
        val_metrics = evaluate(model, val_loader, args)
        
        print(f'\nEpoch {epoch}/{args.epochs}:')
        print(f'  Train loss: {train_metrics["train_loss"]:.4f}')
        if 'train_recon' in train_metrics:
            print(f'  Recon: {train_metrics["train_recon"]:.4f}')
        if 'train_ident' in train_metrics:
            print(f'  Ident: {train_metrics["train_ident"]:.4f}')
        print(f'  Val loss: {val_metrics["val_loss"]:.4f}')
        print(f'  Val ARI: {val_metrics["val_ari"]:.4f}')
        
        if val_metrics['val_ari'] > best_ari:
            best_ari = val_metrics['val_ari']
            print(f'  -> New best ARI!')
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_ari': best_ari,
            }, output_dir / 'best_model.pt')
    
    print(f'\n{"="*60}')
    print(f'Training complete!')
    print(f'Best ARI: {best_ari:.4f}')
    print(f'Model saved to: {output_dir / "best_model.pt"}')
    print(f'{"="*60}')


if __name__ == "__main__":
    main()
