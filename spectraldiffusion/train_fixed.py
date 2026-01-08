#!/usr/bin/env python
"""
Fixed Training Script - Based on debug_3.md fixes

Critical fixes from debug_3.md:
1. CORRECT diversity loss - only penalize POSITIVE similarities (use clamping)
2. Reduce diversity weight to 0.0001 (100× smaller than before)
3. Add safety checks - stop if loss goes negative
4. Clamp total loss to prevent negative values
5. Better logging of loss components

Target: ARI > 0.85 with STABLE positive loss throughout training
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-slots", type=int, default=11)
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--subset-percent", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--use-spectral", action="store_true", default=True)
    parser.add_argument("--no-spectral", dest="use_spectral", action="store_false")
    parser.add_argument("--diversity-weight", type=float, default=0.0001)  # 100× smaller!
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


def compute_diversity_loss_fixed(slots):
    """
    FIXED diversity loss from debug_3.md:
    - Only penalize POSITIVE similarities (slots pointing same direction)
    - Clamp negative similarities to 0 (orthogonal slots are GOOD)
    
    Returns loss in range [0, 1]:
    - 0 = perfect diversity (all orthogonal or opposite)
    - 1 = total collapse (all identical)
    """
    B, K, D = slots.shape
    
    # Normalize
    slots_norm = F.normalize(slots, dim=-1)
    
    # Pairwise cosine similarity
    sim_matrix = torch.bmm(slots_norm, slots_norm.transpose(1, 2))  # [B, K, K]
    
    # Mask out diagonal (self-similarity always 1)
    mask = ~torch.eye(K, dtype=torch.bool, device=slots.device)
    off_diag_sim = sim_matrix[:, mask].reshape(B, -1)
    
    # CRITICAL FIX: Only penalize POSITIVE similarities
    # Negative similarity = orthogonal/opposite = GOOD, don't penalize
    positive_sim = torch.clamp(off_diag_sim, min=0.0)
    
    loss = positive_sim.mean()
    
    return loss


def compute_slot_similarity(slots):
    """
    Compute average off-diagonal slot similarity for monitoring.
    Returns value in [-1, 1]:
    - < 0.3: Good diversity
    - > 0.7: Slots collapsing
    """
    with torch.no_grad():
        B, K, D = slots.shape
        slots_norm = F.normalize(slots, dim=-1)
        sim_matrix = torch.bmm(slots_norm, slots_norm.transpose(1, 2))
        mask = ~torch.eye(K, dtype=torch.bool, device=slots.device)
        off_diag = sim_matrix[:, mask]
        return off_diag.mean().item()


def warmup_lr(epoch, warmup_epochs, base_lr):
    """Epoch-based warmup."""
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
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(args.device).float()
        
        optimizer.zero_grad()
        
        outputs = model(images, return_loss=True)
        
        # Reconstruction loss (should always be positive)
        recon_loss = F.mse_loss(outputs['reconstructed'], images)
        
        # FIXED diversity loss (always in [0, 1] range)
        diversity_loss = compute_diversity_loss_fixed(outputs['slots'])
        
        # Total loss (both components are non-negative)
        loss = recon_loss + args.diversity_weight * diversity_loss
        
        # SAFETY CHECK: Ensure loss is never negative
        if loss.item() < 0:
            print(f"\n❌ CRITICAL: Loss is negative ({loss.item():.6f})!")
            print(f"  Recon: {recon_loss.item():.6f}")
            print(f"  Diversity: {diversity_loss.item():.6f}")
            print("  This should not happen with fixed diversity loss!")
            loss = torch.abs(loss)  # Fallback: use absolute value
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_diversity += diversity_loss.item()
        
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            recon=f"{recon_loss.item():.4f}",
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
    total_recon = 0.0
    all_slot_sims = []
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        images = batch['image'].to(args.device).float()
        
        outputs = model(images, return_loss=True)
        
        # Compute losses using FIXED diversity loss
        recon_loss = F.mse_loss(outputs['reconstructed'], images)
        diversity_loss = compute_diversity_loss_fixed(outputs['slots'])
        loss = recon_loss + args.diversity_weight * diversity_loss
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        
        # Check slot diversity
        slot_sim = compute_slot_similarity(outputs['slots'])
        all_slot_sims.append(slot_sim)
        
        if 'mask' in batch:
            gt_masks = batch['mask']
            ari = compute_ari_sklearn(outputs['masks'], gt_masks)
            all_aris.append(ari)
    
    n = len(val_loader)
    return {
        'loss': total_loss / n,
        'recon': total_recon / n,
        'ari': sum(all_aris) / len(all_aris) if all_aris else 0.0,
        'slot_similarity': sum(all_slot_sims) / len(all_slot_sims) if all_slot_sims else 0.0,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("=" * 60)
    print("FIXED TRAINING (debug_3.md fixes)")
    print("=" * 60)
    print(f"Spectral Init: {args.use_spectral}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Warmup Epochs: {args.warmup_epochs}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Diversity Weight: {args.diversity_weight} (100x smaller)")
    print(f"Num Slots: {args.num_slots}")
    print(f"Subset: {args.subset_percent * 100:.1f}%")
    print("=" * 60)
    print("KEY FIXES:")
    print("  1. Diversity loss only penalizes POSITIVE similarities")
    print("  2. Diversity weight reduced to 0.0001 (100x smaller)")
    print("  3. Safety checks for negative loss")
    print("=" * 60)
    
    model = EnhancedSlotAttentionModel(
        num_slots=args.num_slots,
        num_iterations=args.num_iterations,
        slot_dim=64,
        hidden_dim=128,
        image_size=(args.image_size, args.image_size),
        use_spectral_init=args.use_spectral
    ).to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print(f"Actual num_slots: {model.num_slots}")
    
    train_loader, val_loader = create_dataloaders(args)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # AdamW with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    best_ari = -1.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, epoch, args)
        val_metrics = evaluate(model, val_loader, args)
        
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train - Loss: {train_metrics['total']:.4f} (recon: {train_metrics['recon']:.4f}, div: {train_metrics['diversity']:.4f})")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} (recon: {val_metrics['recon']:.4f})")
        print(f"  Val ARI: {val_metrics['ari']:.4f}")
        print(f"  Slot Similarity: {val_metrics['slot_similarity']:.4f} (target: 0.0-0.5)")
        
        # Verify all losses are positive
        if train_metrics['total'] < 0 or val_metrics['loss'] < 0:
            print("  ❌ WARNING: Negative loss detected!")
        else:
            print("  ✓ All losses positive (good)")
        
        # Warn if slots are collapsing
        if val_metrics['slot_similarity'] > 0.7:
            print("  ⚠️ WARNING: Slots may be collapsing! Similarity > 0.7")
        
        # Detect ARI drop
        if epoch > 10 and best_ari > 0 and val_metrics['ari'] < best_ari * 0.8:
            print(f"  ⚠️ WARNING: ARI dropped >20% from best ({best_ari:.4f} -> {val_metrics['ari']:.4f})")
        
        if val_metrics['ari'] > best_ari:
            best_ari = val_metrics['ari']
            torch.save(model.state_dict(), f"{args.output_dir}/fixed_best_{timestamp}.pt")
            print(f"  ✓ New best ARI!")
        
        history.append({
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
        })
        
        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_ari': best_ari,
            }, f"{args.output_dir}/fixed_checkpoint_{timestamp}_ep{epoch}.pt")
        
        # Early exit if target reached
        if best_ari > 0.85:
            print(f"\n🎉 Target ARI > 0.85 reached! Best: {best_ari:.4f}")
            break
    
    with open(f"{args.output_dir}/fixed_history_{timestamp}.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"Best ARI: {best_ari:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
