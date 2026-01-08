#!/usr/bin/env python
"""
SpectralDiffusion Full Training Script

Complete end-to-end training following icml_2027_solution.md:
- Phase 1: CLEVR validation (ARI > 0.92 target)
- Phase 2: Real-world datasets (COCO-Stuff, Cityscapes)

All 5 stages enabled:
1. DINOv2 frozen backbone
2. Multi-scale spectral initialization
3. Mamba-Slot attention with GMM prior
4. Adaptive slot pruning
5. Latent diffusion decoder (optional)

Usage:
    # CLEVR training (Phase 1)
    python train_full.py --phase clevr --epochs 50 --batch-size 8
    
    # With diffusion decoder
    python train_full.py --phase clevr --epochs 50 --use-diffusion
    
    # COCO-Stuff training (Phase 2)
    python train_full.py --phase coco --epochs 30
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

# Force float32
torch.set_default_dtype(torch.float32)

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(description="SpectralDiffusion Full Training")
    
    # Phase selection
    parser.add_argument("--phase", type=str, default="clevr",
                       choices=["clevr", "coco", "cityscapes", "all"],
                       help="Training phase/dataset")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    
    # Architecture
    parser.add_argument("--num-slots", type=int, default=12)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--image-size", type=int, nargs=2, default=[128, 128])
    
    # Components
    parser.add_argument("--use-dino", action="store_true", default=True,
                       help="Use DINOv2 backbone (default: True)")
    parser.add_argument("--use-diffusion", action="store_true", default=False,
                       help="Use diffusion decoder (slower but better)")
    parser.add_argument("--use-mamba", action="store_true", default=True,
                       help="Use Mamba-Slot attention")
    
    # Loss weights (from icml_2027_solution.md)
    parser.add_argument("--lambda-diff", type=float, default=1.0)
    parser.add_argument("--lambda-spec", type=float, default=0.1)
    parser.add_argument("--lambda-ident", type=float, default=0.01)
    parser.add_argument("--lambda-recon", type=float, default=10.0)
    
    # Data
    parser.add_argument("--data-dir", type=str, default="../datasets")
    parser.add_argument("--subset-percent", type=float, default=1.0,
                       help="Percentage of dataset to use (1.0 = full)")
    
    # Hardware
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--num-workers", type=int, default=0)
    
    # Checkpointing
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    return parser.parse_args()


def create_model(args):
    """Create the full SpectralDiffusion model."""
    from models.spectral_diffusion_v2 import SpectralDiffusionV2
    
    model = SpectralDiffusionV2(
        image_size=tuple(args.image_size),
        num_slots=args.num_slots,
        feature_dim=args.feature_dim,
        use_dino=args.use_dino,
        use_mamba=args.use_mamba,
        use_diffusion_decoder=args.use_diffusion,
        init_mode="spectral",
        diffusion_steps=50,
        lambda_diff=args.lambda_diff,
        lambda_spec=args.lambda_spec,
        lambda_ident=args.lambda_ident,
        lambda_recon=args.lambda_recon,
    )
    
    return model.to(args.device).float()


def create_dataloaders(args):
    """Create train and val dataloaders based on phase."""
    image_size = tuple(args.image_size)
    
    if args.phase == "clevr":
        from data.clevr import CLEVRDataset
        clevr_root = os.path.join(args.data_dir, "clevr/CLEVR_v1.0")
        
        # Use val split (has masks)
        full_dataset = CLEVRDataset(
            root_dir=clevr_root,
            split="val",
            image_size=image_size,
            max_objects=args.num_slots,
            return_masks=True,
        )
        
        # Split 80/20
        n = len(full_dataset)
        train_size = int(0.8 * n)
        train_dataset = Subset(full_dataset, list(range(train_size)))
        val_dataset = Subset(full_dataset, list(range(train_size, n)))
        
    elif args.phase == "coco":
        from data.coco import COCOStuffDataset
        coco_root = os.path.join(args.data_dir, "coco_stuff")
        
        train_dataset = COCOStuffDataset(
            root_dir=coco_root,
            split="train",
            image_size=image_size,
        )
        val_dataset = COCOStuffDataset(
            root_dir=coco_root,
            split="val",
            image_size=image_size,
        )
        
    elif args.phase == "cityscapes":
        from data.cityscapes import CityscapesDataset
        cityscapes_root = os.path.join(args.data_dir, "cityscapes")
        
        train_dataset = CityscapesDataset(
            root_dir=cityscapes_root,
            split="train",
            image_size=image_size,
        )
        val_dataset = CityscapesDataset(
            root_dir=cityscapes_root,
            split="val",
            image_size=image_size,
        )
    else:
        raise ValueError(f"Unknown phase: {args.phase}")
    
    # Apply subset
    if args.subset_percent < 1.0:
        train_n = max(1, int(len(train_dataset) * args.subset_percent))
        val_n = max(1, int(len(val_dataset) * args.subset_percent))
        
        train_indices = torch.randperm(len(train_dataset))[:train_n].tolist()
        val_indices = torch.randperm(len(val_dataset))[:val_n].tolist()
        
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    return train_loader, val_loader


def compute_ari(pred_masks, gt_masks):
    """Compute Adjusted Rand Index."""
    pred_labels = pred_masks.argmax(dim=1).flatten().cpu()
    
    if gt_masks.dim() == 4:
        gt_labels = gt_masks.argmax(dim=1).flatten().cpu()
    else:
        gt_labels = gt_masks.flatten().cpu()
    
    N = len(pred_labels)
    if N == 0:
        return 0.0
    
    pred_ids = pred_labels.unique()
    gt_ids = gt_labels.unique()
    
    contingency = torch.zeros(len(pred_ids), len(gt_ids))
    for i, p in enumerate(pred_ids):
        for j, g in enumerate(gt_ids):
            contingency[i, j] = ((pred_labels == p) & (gt_labels == g)).sum().float()
    
    a = contingency.sum(dim=1)
    b = contingency.sum(dim=0)
    
    def comb2(n):
        return n * (n - 1) / 2
    
    sum_comb_c = comb2(contingency).sum()
    sum_comb_a = comb2(a).sum()
    sum_comb_b = comb2(b).sum()
    
    n_comb = comb2(torch.tensor(N, dtype=torch.float))
    expected = sum_comb_a * sum_comb_b / max(n_comb, 1e-8)
    max_index = (sum_comb_a + sum_comb_b) / 2
    
    if max_index - expected == 0:
        return 1.0 if sum_comb_c == expected else 0.0
    
    ari = (sum_comb_c - expected) / (max_index - expected)
    return ari.item()


def train_epoch(model, train_loader, optimizer, scheduler, epoch, args):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    loss_components = {}
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(args.device).float()
        
        optimizer.zero_grad()
        
        outputs = model(images, return_loss=True)
        loss = outputs['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Track loss components
        if 'loss_dict' in outputs:
            for k, v in outputs['loss_dict'].items():
                loss_components[k] = loss_components.get(k, 0.0) + v
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    scheduler.step()
    
    n_batches = len(train_loader)
    avg_loss = total_loss / n_batches
    avg_components = {k: v / n_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_components


@torch.no_grad()
def evaluate(model, val_loader, args):
    """Evaluate model."""
    model.eval()
    
    all_aris = []
    total_recon_loss = 0.0
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        images = batch['image'].to(args.device).float()
        
        outputs = model(images, return_loss=False)
        masks = outputs['masks']
        reconstructed = outputs['reconstructed']
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, images).item()
        total_recon_loss += recon_loss
        
        # ARI if GT masks available
        if 'mask' in batch:
            gt_masks = batch['mask'].to(args.device)
            ari = compute_ari(masks, gt_masks)
            all_aris.append(ari)
    
    n_batches = len(val_loader)
    metrics = {
        'recon_loss': total_recon_loss / n_batches,
    }
    
    if all_aris:
        metrics['ari'] = sum(all_aris) / len(all_aris)
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, args, path):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'args': vars(args),
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"spectral_{args.phase}_{timestamp}"
    
    print("=" * 70)
    print("SpectralDiffusion Full Training")
    print("=" * 70)
    print(f"Phase: {args.phase}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.image_size}")
    print(f"Num Slots: {args.num_slots}")
    print(f"DINOv2: {args.use_dino}")
    print(f"Diffusion Decoder: {args.use_diffusion}")
    print(f"Device: {args.device}")
    print(f"Subset: {args.subset_percent * 100:.1f}%")
    print("=" * 70)
    
    # Create model
    print("\n[1/4] Creating model...")
    model = create_model(args)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    # Resume if specified
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Create dataloaders
    print("\n[2/4] Loading data...")
    train_loader, val_loader = create_dataloaders(args)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Optimizer and scheduler
    print("\n[3/4] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )
    
    # Training loop
    print("\n[4/4] Starting training...")
    print("-" * 70)
    
    best_ari = -1.0
    training_history = []
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, loss_components = train_epoch(
            model, train_loader, optimizer, scheduler, epoch, args
        )
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, args)
        
        epoch_time = time.time() - epoch_start
        
        # Log results
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Recon Loss: {val_metrics['recon_loss']:.4f}")
        
        if 'ari' in val_metrics:
            print(f"  Val ARI: {val_metrics['ari']:.4f}")
            
            if val_metrics['ari'] > best_ari:
                best_ari = val_metrics['ari']
                # Save best model
                best_path = os.path.join(args.output_dir, f"{run_name}_best.pt")
                save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, args, best_path)
        
        # Log loss components
        if loss_components:
            print(f"  Loss breakdown: {', '.join(f'{k}={v:.4f}' for k, v in loss_components.items())}")
        
        # Save history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **val_metrics,
            **loss_components,
        })
        
        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"{run_name}_epoch{epoch}.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, args, ckpt_path)
    
    # Final save
    final_path = os.path.join(args.output_dir, f"{run_name}_final.pt")
    save_checkpoint(model, optimizer, scheduler, args.epochs, val_metrics, args, final_path)
    
    # Save training history
    history_path = os.path.join(args.output_dir, f"{run_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best ARI: {best_ari:.4f}")
    print(f"Final checkpoint: {final_path}")
    print(f"Training history: {history_path}")
    
    # Target check
    if args.phase == "clevr":
        target = 0.92
        if best_ari >= target:
            print(f"✅ TARGET ACHIEVED: ARI {best_ari:.4f} >= {target}")
        else:
            print(f"❌ Target not met: ARI {best_ari:.4f} < {target}")
            print(f"   Gap: {target - best_ari:.4f}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
