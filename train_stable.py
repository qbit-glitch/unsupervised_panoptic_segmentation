#!/usr/bin/env python
"""
Stable Training Script - Based on debug_2.md fixes

Key fixes from debug_2.md:
1. Use FULL dataset (not 10% subset)
2. Add diversity loss to prevent slot collapse
3. Slower learning rate (0.0001 instead of 0.0004)
4. Longer warmup (20 epochs instead of 1000 steps)
5. Monitor slot diversity (should stay < 0.5)
6. Detect and warn on ARI collapse

Target: ARI > 0.85 by epoch 100-200
"""

import os
import sys
import torch
import torch.nn as nn
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
from utils.multi_gpu import (
    setup_device, wrap_model, get_model_state,
    count_parameters, add_multi_gpu_args
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)  # Slower LR
    parser.add_argument("--warmup-epochs", type=int, default=20)  # Epoch-based warmup
    parser.add_argument("--weight-decay", type=float, default=0.01)  # Regularization
    parser.add_argument("--num-slots", type=int, default=11)  # Odd number
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--subset-percent", type=float, default=1.0)  # FULL dataset
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--use-spectral", action="store_true", default=True)
    parser.add_argument("--no-spectral", dest="use_spectral", action="store_false")
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    add_multi_gpu_args(parser)  # Add --multi-gpu and --gpu-ids
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
    
    # Use pin_memory only for CUDA
    pin_memory = args.device == "cuda"
    num_workers = getattr(args, 'num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=num_workers, drop_last=True,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def compute_diversity_loss(slots):
    """
    Compute diversity loss to prevent slot collapse.
    Minimizes similarity between different slots.
    
    Args:
        slots: [B, K, D] slot representations
    Returns:
        loss: scalar diversity loss (should be minimized)
    """
    slots_norm = F.normalize(slots, dim=-1)
    similarity = torch.bmm(slots_norm, slots_norm.transpose(1, 2))
    K = slots.shape[1]
    mask = ~torch.eye(K, dtype=torch.bool, device=slots.device)
    off_diag = similarity[:, mask]
    return off_diag.mean()


def check_slot_diversity(slots):
    """
    Check if slots are diverse (not collapsed).
    
    Returns:
        avg_similarity: average off-diagonal similarity
                       Good: < 0.3, Collapsed: > 0.7
    """
    with torch.no_grad():
        slots_norm = F.normalize(slots, dim=-1)
        similarity = torch.bmm(slots_norm, slots_norm.transpose(1, 2))
        K = slots.shape[1]
        mask = ~torch.eye(K, dtype=torch.bool, device=slots.device)
        off_diag = similarity[:, mask]
        return off_diag.mean().item()


def warmup_lr(epoch, warmup_epochs, base_lr):
    """Epoch-based warmup (more stable than step-based)."""
    if epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    return base_lr


def train_epoch(model, train_loader, optimizer, epoch, args):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_diversity = 0.0
    
    # Epoch-based LR warmup
    lr = warmup_lr(epoch, args.warmup_epochs, args.lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch['image'].to(args.device).float()
        
        optimizer.zero_grad()
        
        outputs = model(images, return_loss=True)
        recon_loss = outputs['loss']
        
        # Add diversity loss to prevent slot collapse
        diversity_loss = compute_diversity_loss(outputs['slots'])
        
        # Total loss with regularization
        loss = recon_loss + args.diversity_weight * diversity_loss
        
        loss.backward()
        
        # Gradient clipping (critical for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_diversity += diversity_loss.item()
        
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            div=f"{diversity_loss.item():.3f}",
            lr=f"{lr:.2e}"
        )
    
    n_batches = len(train_loader)
    return {
        'total': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'diversity': total_diversity / n_batches,
    }


@torch.no_grad()
def evaluate(model, val_loader, args):
    model.eval()
    
    all_aris = []
    total_loss = 0.0
    all_slot_sims = []
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        images = batch['image'].to(args.device).float()
        
        outputs = model(images, return_loss=True)
        total_loss += outputs['loss'].item()
        
        # Check slot diversity
        slot_sim = check_slot_diversity(outputs['slots'])
        all_slot_sims.append(slot_sim)
        
        if 'mask' in batch:
            gt_masks = batch['mask']
            ari = compute_ari_sklearn(outputs['masks'], gt_masks)
            all_aris.append(ari)
    
    return {
        'loss': total_loss / len(val_loader),
        'ari': sum(all_aris) / len(all_aris) if all_aris else 0.0,
        'slot_similarity': sum(all_slot_sims) / len(all_slot_sims) if all_slot_sims else 0.0,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device and multi-GPU
    use_multi_gpu, gpu_ids = setup_device(args)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("=" * 60)
    print("STABLE TRAINING (debug_2.md fixes)")
    print("=" * 60)
    print(f"Device: {args.device}")
    if use_multi_gpu:
        print(f"Multi-GPU: Enabled ({len(gpu_ids)} GPUs: {gpu_ids})")
    print(f"Spectral Init: {args.use_spectral}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"LR: {args.lr} (slower)")
    print(f"Warmup Epochs: {args.warmup_epochs}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Diversity Weight: {args.diversity_weight}")
    print(f"Num Slots: {args.num_slots}")
    print(f"Subset: {args.subset_percent * 100:.1f}%")
    print("=" * 60)
    
    model = EnhancedSlotAttentionModel(
        num_slots=args.num_slots,
        num_iterations=args.num_iterations,
        slot_dim=64,
        hidden_dim=128,
        image_size=(args.image_size, args.image_size),
        use_spectral_init=args.use_spectral
    ).to(args.device)
    
    # Wrap model for multi-GPU
    model = wrap_model(model, use_multi_gpu, gpu_ids)
    
    num_params = count_parameters(model)
    print(f"Parameters: {num_params:,}")
    
    train_loader, val_loader = create_dataloaders(args)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # AdamW with weight decay (better regularization)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    best_ari = -1.0
    ari_history = []
    history = []
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, epoch, args)
        val_metrics = evaluate(model, val_loader, args)
        
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_metrics['total']:.4f} (recon: {train_metrics['recon']:.4f}, div: {train_metrics['diversity']:.3f})")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val ARI: {val_metrics['ari']:.4f}")
        print(f"  Slot Similarity: {val_metrics['slot_similarity']:.4f} (good: <0.5, collapse: >0.7)")
        
        # Warn if slots are collapsing
        if val_metrics['slot_similarity'] > 0.7:
            print("  ⚠️ WARNING: Slots may be collapsing! Similarity > 0.7")
        
        # Track ARI for collapse detection
        ari_history.append(val_metrics['ari'])
        
        # Detect ARI collapse (>20% drop from best)
        if epoch > 10 and val_metrics['ari'] < best_ari * 0.8:
            print(f"  ⚠️ WARNING: ARI dropped >20% from best ({best_ari:.4f} -> {val_metrics['ari']:.4f})")
        
        if val_metrics['ari'] > best_ari:
            best_ari = val_metrics['ari']
            torch.save(model.state_dict(), f"{args.output_dir}/stable_best_{timestamp}.pt")
            print(f"  ✓ New best ARI!")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['total'],
            'train_recon': train_metrics['recon'],
            'train_diversity': train_metrics['diversity'],
            **val_metrics
        })
        
        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_ari': best_ari,
            }, f"{args.output_dir}/stable_checkpoint_{timestamp}_ep{epoch}.pt")
        
        # Early exit if target reached
        if best_ari > 0.85:
            print(f"\n🎉 Target ARI > 0.85 reached! Best: {best_ari:.4f}")
            break
    
    with open(f"{args.output_dir}/stable_history_{timestamp}.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"Best ARI: {best_ari:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
