#!/usr/bin/env python
"""
Stage-3 Training Script: Mamba-Slot Attention

Trains the MambaSlotAttentionModel on CLEVR dataset.
Can optionally load pretrained Stage-1 encoder/decoder weights.

Key features:
- Mamba-2 based slot attention (O(K) complexity)
- Identifiability loss (GMM prior regularization)
- Compatible with Stage-1 pretrained weights
- ARI monitoring throughout training

Expected performance:
- Expected ARI: 0.50-0.55 (similar to Stage-1 but 5× faster)
- Inference speed: 5× faster than standard slot attention

Usage:
    # Train from scratch
    python train_stage3.py --epochs 100 --batch-size 32
    
    # Continue from Stage-1 weights
    python train_stage3.py --epochs 100 --stage1-ckpt outputs/baseline_best_20260105_062929.pt
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
import time

sys.path.insert(0, str(Path(__file__).parent))

from models.mamba_baseline import (
    MambaSlotAttentionModel,
    compute_ari_sklearn
)
from utils.multi_gpu import (
    setup_device, wrap_model, get_model_state,
    count_parameters, add_multi_gpu_args, get_model_module
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-3: Mamba-Slot Attention Training")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    
    # Model
    parser.add_argument("--num-slots", type=int, default=7)
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--use-mamba", action="store_true", default=True)
    parser.add_argument("--no-mamba", dest="use_mamba", action="store_false",
                        help="Disable Mamba (ablation)")
    
    # Loss weights
    parser.add_argument("--lambda-ident", type=float, default=0.01,
                        help="Weight for identifiability loss")
    
    # Data
    parser.add_argument("--subset-percent", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default="./datasets",
                        help="Path to datasets directory")
    
    # Device
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"])
    add_multi_gpu_args(parser)  # Add --multi-gpu and --gpu-ids
    
    # Checkpoints
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--stage1-ckpt", type=str, default=None,
                        help="Path to Stage-1 checkpoint to load encoder/decoder weights")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=5)
    
    return parser.parse_args()


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
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        drop_last=True,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
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
    total_recon_loss = 0.0
    total_ident_loss = 0.0
    batch_times = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        batch_start = time.time()
        
        images = batch['image'].to(args.device).float()
        
        # Warmup LR
        lr = warmup_lr(step, args.warmup_steps, args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.zero_grad()
        
        outputs = model(images, return_loss=True)
        loss = outputs['loss']
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        total_loss += outputs['loss_dict']['total']
        total_recon_loss += outputs['loss_dict']['recon']
        total_ident_loss += outputs['loss_dict']['ident']
        step += 1
        
        if batch_idx % args.log_interval == 0:
            pbar.set_postfix(
                loss=f"{outputs['loss_dict']['total']:.4f}",
                recon=f"{outputs['loss_dict']['recon']:.4f}",
                ident=f"{outputs['loss_dict']['ident']:.4f}",
                lr=f"{lr:.2e}",
                ms=f"{batch_time*1000:.0f}"
            )
    
    n_batches = len(train_loader)
    avg_batch_time = sum(batch_times) / len(batch_times) * 1000
    
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'ident_loss': total_ident_loss / n_batches,
        'avg_batch_ms': avg_batch_time,
    }, step


@torch.no_grad()
def evaluate(model, val_loader, args):
    """Evaluate model."""
    model.eval()
    
    all_aris = []
    total_loss = 0.0
    total_recon_loss = 0.0
    eval_times = []
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        batch_start = time.time()
        
        images = batch['image'].to(args.device).float()
        
        outputs = model(images, return_loss=True)
        total_loss += outputs['loss_dict']['total']
        total_recon_loss += outputs['loss_dict']['recon']
        
        eval_times.append(time.time() - batch_start)
        
        # ARI
        if 'mask' in batch:
            gt_masks = batch['mask']
            ari = compute_ari_sklearn(outputs['masks'], gt_masks)
            all_aris.append(ari)
    
    n_batches = len(val_loader)
    
    metrics = {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'ari': sum(all_aris) / len(all_aris) if all_aris else 0.0,
        'avg_eval_ms': sum(eval_times) / len(eval_times) * 1000,
    }
    
    return metrics


def check_slot_diversity(model, val_loader, args):
    """Check slot diversity to detect collapse."""
    model.eval()
    
    all_sims = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(args.device).float()
            outputs = model(images, return_loss=False)
            
            slots = outputs['slots']  # [B, K, D]
            slots_norm = F.normalize(slots, dim=-1)
            sim = torch.bmm(slots_norm, slots_norm.transpose(1, 2))
            
            # Get off-diagonal elements
            K = slots.shape[1]
            mask = ~torch.eye(K, dtype=torch.bool, device=args.device)
            off_diag = sim[:, mask].mean().item()
            all_sims.append(off_diag)
            
            if len(all_sims) >= 5:  # Only check a few batches
                break
    
    return sum(all_sims) / len(all_sims)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device and multi-GPU
    use_multi_gpu, gpu_ids = setup_device(args)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("=" * 60)
    print("STAGE-3: MAMBA-SLOT ATTENTION TRAINING")
    print("=" * 60)
    print(f"Device: {args.device}")
    if use_multi_gpu:
        print(f"Multi-GPU: Enabled ({len(gpu_ids)} GPUs: {gpu_ids})")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Warmup Steps: {args.warmup_steps}")
    print(f"Num Slots: {args.num_slots}")
    print(f"Use Mamba: {args.use_mamba}")
    print(f"Lambda Ident: {args.lambda_ident}")
    print(f"Subset: {args.subset_percent * 100:.1f}%")
    if args.stage1_ckpt:
        print(f"Stage-1 Checkpoint: {args.stage1_ckpt}")
    print("=" * 60)
    
    # Create model
    model = MambaSlotAttentionModel(
        num_slots=args.num_slots,
        num_iterations=args.num_iterations,
        slot_dim=64,
        hidden_dim=128,
        image_size=(args.image_size, args.image_size),
        use_mamba=args.use_mamba,
        lambda_ident=args.lambda_ident,
    ).to(args.device)
    
    # Load Stage-1 weights if provided
    if args.stage1_ckpt and os.path.exists(args.stage1_ckpt):
        model.load_stage1_weights(args.stage1_ckpt)
    
    # Wrap model for multi-GPU
    model = wrap_model(model, use_multi_gpu, gpu_ids)
    
    # Count parameters
    model_module = get_model_module(model)
    total_params = sum(p.numel() for p in model_module.parameters())
    trainable_params = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_ari = -1.0
    step = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics, step = train_epoch(model, train_loader, optimizer, epoch, step, args)
        
        # Evaluate
        if epoch % args.eval_interval == 0 or epoch == 1:
            val_metrics = evaluate(model, val_loader, args)
            slot_sim = check_slot_diversity(model, val_loader, args)
            
            print(f"\nEpoch {epoch}/{args.epochs}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} (recon: {train_metrics['recon_loss']:.4f}, ident: {train_metrics['ident_loss']:.4f})")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val ARI: {val_metrics['ari']:.4f}")
            print(f"  Slot Similarity: {slot_sim:.4f} ({'!! COLLAPSE' if slot_sim > 0.7 else 'OK'})")
            print(f"  Train Speed: {train_metrics['avg_batch_ms']:.1f}ms/batch")
            print(f"  Eval Speed: {val_metrics['avg_eval_ms']:.1f}ms/batch")
            
            # Save best model (handle DataParallel wrapper)
            model_state = get_model_state(model)
            
            if val_metrics['ari'] > best_ari:
                best_ari = val_metrics['ari']
                save_path = f"{args.output_dir}/stage3_mamba_best_{timestamp}.pt"
                torch.save(model_state, save_path)
                print(f"  ✓ New best ARI! Saved to {save_path}")
            
            history.append({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_recon_loss': train_metrics['recon_loss'],
                'train_ident_loss': train_metrics['ident_loss'],
                'val_loss': val_metrics['loss'],
                'ari': val_metrics['ari'],
                'slot_similarity': slot_sim,
                'train_ms': train_metrics['avg_batch_ms'],
            })
        else:
            print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}")
            history.append({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_recon_loss': train_metrics['recon_loss'],
                'train_ident_loss': train_metrics['ident_loss'],
            })
    
    # Save final model
    model_state = get_model_state(model)
    final_path = f"{args.output_dir}/stage3_mamba_final_{timestamp}.pt"
    torch.save(model_state, final_path)
    
    # Save history
    history_path = f"{args.output_dir}/stage3_mamba_history_{timestamp}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config = vars(args)
    config['best_ari'] = best_ari
    config['total_params'] = total_params
    config_path = f"{args.output_dir}/stage3_mamba_config_{timestamp}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 60)
    print("STAGE-3 TRAINING COMPLETE")
    print(f"Best ARI: {best_ari:.4f}")
    print(f"Final model: {final_path}")
    print(f"Best model: {args.output_dir}/stage3_mamba_best_{timestamp}.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
