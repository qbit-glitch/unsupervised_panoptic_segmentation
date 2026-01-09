#!/usr/bin/env python
"""
Working Baseline Training Script

Follows debug_1.md recommendations:
- lr=4e-4 with warmup
- batch_size=64 (or max GPU can handle)
- epochs=500 for full CLEVR
- gradient clipping max_norm=1.0

Expected ARI progression:
- Epoch 10: ~0.1-0.2
- Epoch 50: ~0.4-0.6
- Epoch 100: ~0.7-0.8
- Epoch 500: ~0.85-0.92
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

from models.working_baseline import (
    WorkingSlotAttentionModel,
    compute_ari_sklearn
)
from utils.multi_gpu import (
    setup_device, wrap_model, get_model_state, 
    count_parameters, add_multi_gpu_args
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--num-slots", type=int, default=7)
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--subset-percent", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--data-dir", type=str, default="../datasets",
                       help="Path to datasets directory")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(\"--diversity-weight\", type=float, default=0.001,
                       help=\"Weight for diversity loss to prevent slot collapse\")
    add_multi_gpu_args(parser)  # Add --multi-gpu and --gpu-ids
    return parser.parse_args()


def compute_diversity_loss(slots):
    """
    Compute diversity loss to prevent slot collapse.
    Only penalizes POSITIVE similarity (slots pointing same direction).
    
    Args:
        slots: [B, K, D] slot representations
    Returns:
        diversity_loss: scalar in [0, 1] range (never negative)
    """
    # Handle DataParallel - slots might be gathered from multiple GPUs
    if slots.dim() == 4:
        # [num_gpus, B, K, D] -> [B*num_gpus, K, D]
        slots = slots.reshape(-1, slots.shape[-2], slots.shape[-1])
    
    B, K, D = slots.shape
    
    # Normalize slots
    slots_norm = F.normalize(slots, dim=-1)  # [B, K, D]
    
    # Compute pairwise cosine similarity
    sim = torch.bmm(slots_norm, slots_norm.transpose(1, 2))  # [B, K, K]
    
    # Mask diagonal (self-similarity)
    mask = ~torch.eye(K, dtype=torch.bool, device=slots.device)
    
    # Get off-diagonal similarities
    off_diag_sim = sim[:, mask].reshape(B, -1)
    
    # KEY FIX: Only penalize POSITIVE similarity (per debug_3.md)
    # Negative sim = orthogonal slots = good, don't penalize
    positive_sim = torch.clamp(off_diag_sim, min=0.0)
    
    # Loss: mean of positive similarities
    diversity_loss = positive_sim.mean()
    
    return diversity_loss


def create_dataloaders(args):
    """Create CLEVR dataloaders."""
    from data.clevr import CLEVRDataset
    import os
    
    clevr_root = os.path.join(args.data_dir, "clevr/CLEVR_v1.0")
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
    val_size = n_subset - train_size
    
    indices = torch.randperm(n)[:n_subset].tolist()
    train_dataset = Subset(full_dataset, indices[:train_size])
    val_dataset = Subset(full_dataset, indices[train_size:])
    
    # Use pin_memory only for CUDA
    pin_memory = args.device == "cuda"
    num_workers = getattr(args, 'num_workers', 4)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader


def warmup_lr(step, warmup_steps, base_lr):
    """Linear warmup."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def train_epoch(model, train_loader, optimizer, epoch, step, args):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    total_div_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch['image'].to(args.device).float()
        
        # Warmup LR
        lr = warmup_lr(step, args.warmup_steps, args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.zero_grad()
        
        outputs = model(images, return_loss=True)
        recon_loss = outputs['loss']
        
        # DataParallel gathers loss from each GPU into a vector, take mean
        if recon_loss.dim() > 0:
            recon_loss = recon_loss.mean()
        
        # Compute diversity loss to prevent slot collapse
        slots = outputs['slots']
        div_loss = compute_diversity_loss(slots)
        if div_loss.dim() > 0:
            div_loss = div_loss.mean()
        
        # Total loss with diversity regularization
        loss = recon_loss + args.diversity_weight * div_loss
        
        loss.backward()
        
        # CRITICAL: Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += recon_loss.item()
        total_div_loss += div_loss.item()
        step += 1
        
        pbar.set_postfix(loss=f"{recon_loss.item():.4f}", div=f"{div_loss.item():.4f}", lr=f"{lr:.2e}")
    
    return total_loss / len(train_loader), step


@torch.no_grad()
def evaluate(model, val_loader, args):
    """Evaluate model."""
    model.eval()
    
    all_aris = []
    total_loss = 0.0
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        images = batch['image'].to(args.device).float()
        
        outputs = model(images, return_loss=True)
        loss = outputs['loss']
        if loss.dim() > 0:
            loss = loss.mean()
        total_loss += loss.item()
        
        # ARI
        if 'mask' in batch:
            gt_masks = batch['mask']
            ari = compute_ari_sklearn(outputs['masks'], gt_masks)
            all_aris.append(ari)
    
    metrics = {
        'loss': total_loss / len(val_loader),
        'ari': sum(all_aris) / len(all_aris) if all_aris else 0.0,
    }
    
    return metrics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device and multi-GPU
    use_multi_gpu, gpu_ids = setup_device(args)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("=" * 60)
    print("WORKING BASELINE TRAINING")
    print("=" * 60)
    print(f"Device: {args.device}")
    if use_multi_gpu:
        print(f"Multi-GPU: Enabled ({len(gpu_ids)} GPUs: {gpu_ids})")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Warmup Steps: {args.warmup_steps}")
    print(f"Num Slots: {args.num_slots}")
    print(f"Subset: {args.subset_percent * 100:.1f}%")
    print("=" * 60)
    
    # Model
    model = WorkingSlotAttentionModel(
        num_slots=args.num_slots,
        num_iterations=args.num_iterations,
        slot_dim=64,
        hidden_dim=128,
        image_size=(args.image_size, args.image_size)
    ).to(args.device)
    
    # Wrap model for multi-GPU
    model = wrap_model(model, use_multi_gpu, gpu_ids)
    
    num_params = count_parameters(model)
    print(f"Parameters: {num_params:,}")
    
    # Data
    train_loader, val_loader = create_dataloaders(args)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training
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
        
        # Get model state (handles DataParallel wrapper)
        model_state = get_model_state(model)
        
        if val_metrics['ari'] > best_ari:
            best_ari = val_metrics['ari']
            torch.save(model_state, f"{args.output_dir}/baseline_best_{timestamp}.pt")
            print(f"  ✓ New best ARI!")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **val_metrics
        })
    
    # Save history
    with open(f"{args.output_dir}/baseline_history_{timestamp}.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"Best ARI: {best_ari:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
