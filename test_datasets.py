"""
Test script for 1% of each dataset for 3 epochs.
"""
import sys
import os
import random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from train import (
    parse_args, create_dataloader, create_model, 
    train_epoch, evaluate
)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine schedule with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def test_dataset(dataset_name, train_pct=0.01, val_pct=0.01, epochs=3):
    """Run 1% test on a dataset."""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name.upper()} - {int(train_pct*100)}% for {epochs} epochs")
    print(f"{'='*60}")
    
    # Setup args
    args = parse_args()
    args.dataset = dataset_name
    args.data_dir = '../datasets'
    args.epochs = epochs
    args.batch_size = 8
    args.mixed_precision = True
    args.num_slots = 11 if dataset_name == 'clevr' else 8
    args.image_size = [128, 128]
    args.device = 'mps'
    args.debug = False
    args.fast_dev_run = False
    args.wandb = False
    args.output_dir = './outputs'
    args.exp_name = f'{dataset_name}_1pct_test'
    args.log_interval = 10
    args.eval_interval = 1
    args.save_interval = 5
    args.warmup_epochs = 1
    args.lr = 1e-4
    args.weight_decay = 1e-5
    args.grad_clip = 1.0
    args.use_mamba = True
    args.init_mode = 'spectral'
    args.lambda_spec = 0.1
    args.lambda_ident = 0.01
    args.backbone = 'base'
    args.use_diffusion = True
    args.config = None
    args.lr_scheduler = 'cosine'
    
    # Create full dataloaders first
    print(f"Loading {dataset_name} dataset...")
    train_loader_full = create_dataloader(args, 'train')
    val_loader_full = create_dataloader(args, 'val')
    
    # Get sizes
    train_size = len(train_loader_full.dataset)
    val_size = len(val_loader_full.dataset)
    
    # Calculate subset sizes
    train_subset_size = max(100, int(train_size * train_pct))
    val_subset_size = max(50, int(val_size * val_pct))
    
    # Create subsets
    train_indices = random.sample(range(train_size), min(train_subset_size, train_size))
    val_indices = random.sample(range(val_size), min(val_subset_size, val_size))
    
    train_subset = Subset(train_loader_full.dataset, train_indices)
    val_subset = Subset(val_loader_full.dataset, val_indices)
    
    # Create subset dataloaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        drop_last=True
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    print(f"Using {len(train_subset)} train samples ({train_subset_size}/{train_size})")
    print(f"Using {len(val_subset)} val samples ({val_subset_size}/{val_size})")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(args)
    if args.mixed_precision and args.device == 'mps':
        model = model.to(torch.float32)
        print("Model converted to bfloat16")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Training loop
    print("\nStarting training...")
    best_ari = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, args, epoch)
        val_metrics = evaluate(model, val_loader, args)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train loss: {train_metrics['train_loss']:.4f}")
        if 'train_recon' in train_metrics:
            print(f"  Recon loss: {train_metrics['train_recon']:.4f}")
        if 'train_ident' in train_metrics:
            print(f"  Ident loss: {train_metrics['train_ident']:.4f}")
        print(f"  Val loss: {val_metrics['val_loss']:.4f}")
        print(f"  Val ARI: {val_metrics['val_ari']:.4f}")
        
        if val_metrics['val_ari'] > best_ari:
            best_ari = val_metrics['val_ari']
            print(f"  -> New best ARI!")
    
    print(f"\n{dataset_name.upper()} Complete! Best ARI: {best_ari:.4f}")
    return best_ari


if __name__ == "__main__":
    results = {}
    
    # Test each dataset
    for dataset in ['synthetic', 'clevr', 'coco']:
        try:
            ari = test_dataset(dataset, train_pct=0.01, val_pct=0.01, epochs=3)
            results[dataset] = ari
        except Exception as e:
            print(f"\nError testing {dataset}: {e}")
            results[dataset] = None
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - 1% Dataset Tests (3 epochs)")
    print("="*60)
    for dataset, ari in results.items():
        if ari is not None:
            print(f"  {dataset.upper():12s}  ARI: {ari:.4f}")
        else:
            print(f"  {dataset.upper():12s}  FAILED")
    print("="*60)
