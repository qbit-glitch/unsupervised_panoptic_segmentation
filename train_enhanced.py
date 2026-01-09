#!/usr/bin/env python
"""
Enhanced Model Training Script

Trains the enhanced model with spectral initialization.
Target: ARI > 0.7 (up from 0.395 baseline)
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))

from models.enhanced_model import (
    EnhancedSlotAttentionModel,
    compute_ari_sklearn
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--num-slots", type=int, default=9)
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--subset-percent", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--use-spectral", action="store_true", default=True)
    parser.add_argument("--no-spectral", dest="use_spectral", action="store_false")
    parser.add_argument("--use-mamba", action="store_true", default=False,
                       help="Use MambaSlotAttention for O(N) linear complexity")
    return parser.parse_args()


def create_dataloaders(args):
    from data.clevr import CLEVRDataset
    
    clevr_root = "../datasets/clevr/CLEVR_v1.0"
    image_size = (args.image_size, args.image_size)
    
    full_dataset = CLEVRDataset(
        root_dir=clevr_root,
        split="val",
        image_size=image_size,
        max_objects=args.num_slots,
        return_masks=True,
    )
    
    n = len(full_dataset)
    n_subset = int(n * args.subset_percent)
    
    train_size = int(0.8 * n_subset)
    
    indices = torch.randperm(n)[:n_subset].tolist()
    train_dataset = Subset(full_dataset, indices[:train_size])
    val_dataset = Subset(full_dataset, indices[train_size:])
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader


def warmup_lr(step, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def train_epoch(model, train_loader, optimizer, epoch, step, args):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch['image'].to(args.device).float()
        
        lr = warmup_lr(step, args.warmup_steps, args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.zero_grad()
        
        outputs = model(images, return_loss=True)
        loss = outputs['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        step += 1
        
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{lr:.2e}"
        )
    
    return total_loss / len(train_loader), step


@torch.no_grad()
def evaluate(model, val_loader, args):
    model.eval()
    
    all_aris = []
    total_loss = 0.0
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        images = batch['image'].to(args.device).float()
        
        outputs = model(images, return_loss=True)
        total_loss += outputs['loss'].item()
        
        if 'mask' in batch:
            gt_masks = batch['mask']
            ari = compute_ari_sklearn(outputs['masks'], gt_masks)
            all_aris.append(ari)
    
    return {
        'loss': total_loss / len(val_loader),
        'ari': sum(all_aris) / len(all_aris) if all_aris else 0.0,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("=" * 60)
    print("ENHANCED MODEL TRAINING")
    print("=" * 60)
    print(f"Spectral Init: {args.use_spectral}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Num Slots: {args.num_slots}")
    print(f"Subset: {args.subset_percent * 100:.1f}%")
    print(f"Use Mamba: {args.use_mamba}")
    print("=" * 60)
    
    model = EnhancedSlotAttentionModel(
        num_slots=args.num_slots,
        num_iterations=args.num_iterations,
        slot_dim=64,
        hidden_dim=128,
        image_size=(args.image_size, args.image_size),
        use_spectral_init=args.use_spectral,
        use_mamba=args.use_mamba
    ).to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print(f"Actual num_slots: {model.num_slots}")
    
    train_loader, val_loader = create_dataloaders(args)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_ari = -1.0
    step = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss, step = train_epoch(model, train_loader, optimizer, epoch, step, args)
        val_metrics = evaluate(model, val_loader, args)
        
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val ARI: {val_metrics['ari']:.4f}")
        
        if val_metrics['ari'] > best_ari:
            best_ari = val_metrics['ari']
            torch.save(model.state_dict(), f"{args.output_dir}/enhanced_best_{timestamp}.pt")
            print(f"  ✓ New best ARI!")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **val_metrics
        })
        
        # Early exit if target reached
        if best_ari > 0.7:
            print(f"\n🎉 Target ARI > 0.7 reached! Best: {best_ari:.4f}")
            break
    
    with open(f"{args.output_dir}/enhanced_history_{timestamp}.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"Best ARI: {best_ari:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
