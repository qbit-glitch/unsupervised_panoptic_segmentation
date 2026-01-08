#!/usr/bin/env python
"""
End-to-End Training Script for SpectralDiffusion

Trains on multiple datasets as per icml_2027_solution.md:
- Phase 1: Synthetic (validation of architecture)
- Phase 2: CLEVR (object discovery, ARI > 0.92 target)
- Phase 3: COCO-Stuff (real-world, mIoU target)

Usage:
    python train_end_to_end.py --dataset clevr --epochs 5 --device mps
    python train_end_to_end.py --dataset coco --epochs 10 --device mps
    python train_end_to_end.py --dataset all --epochs 5 --device mps
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import time
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from data.clevr import CLEVRDataset, SyntheticShapesDataset
from data.coco import COCOStuffDataset
from torch.utils.data import DataLoader, Subset


def parse_args():
    parser = argparse.ArgumentParser(description="SpectralDiffusion End-to-End Training")
    parser.add_argument("--dataset", type=str, default="all",
                       choices=["synthetic", "clevr", "coco", "all"],
                       help="Dataset to train on")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--num-slots", type=int, default=12)
    parser.add_argument("--image-size", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--subset-percent", type=float, default=0.01,
                       help="Percentage of dataset to use (for quick testing)")
    parser.add_argument("--data-dir", type=str, default="../datasets")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    return parser.parse_args()


def compute_ari(pred_labels: torch.Tensor, gt_labels: torch.Tensor) -> torch.Tensor:
    """Compute Adjusted Rand Index."""
    pred = pred_labels.flatten()
    gt = gt_labels.flatten()
    
    N = len(pred)
    if N == 0:
        return torch.tensor(0.0)
    
    # Contingency table
    pred_ids = pred.unique()
    gt_ids = gt.unique()
    
    contingency = torch.zeros(len(pred_ids), len(gt_ids))
    for i, p in enumerate(pred_ids):
        for j, g in enumerate(gt_ids):
            contingency[i, j] = ((pred == p) & (gt == g)).sum().float()
    
    # Row and column sums
    a = contingency.sum(dim=1)
    b = contingency.sum(dim=0)
    
    # Combinatorial terms
    def comb2(n):
        return n * (n - 1) / 2
    
    sum_comb_c = comb2(contingency).sum()
    sum_comb_a = comb2(a).sum()
    sum_comb_b = comb2(b).sum()
    
    expected = sum_comb_a * sum_comb_b / comb2(torch.tensor(N, dtype=torch.float))
    max_index = (sum_comb_a + sum_comb_b) / 2
    
    if max_index - expected == 0:
        return torch.tensor(1.0) if sum_comb_c == expected else torch.tensor(0.0)
    
    ari = (sum_comb_c - expected) / (max_index - expected)
    return ari


def compute_metrics(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> dict:
    """Compute ARI, mIoU, PQ metrics."""
    metrics = {}
    
    pred_labels = pred_masks.argmax(dim=1)  # [B, H, W]
    gt_labels = gt_masks.argmax(dim=1) if gt_masks.dim() == 4 else gt_masks
    
    B = pred_labels.shape[0]
    
    # ARI
    aris = []
    for b in range(B):
        ari = compute_ari(pred_labels[b], gt_labels[b])
        aris.append(ari.item())
    metrics['ari'] = np.mean(aris)
    
    # mIoU
    ious = []
    for b in range(B):
        pred = pred_labels[b].flatten()
        gt = gt_labels[b].flatten()
        
        for c in gt.unique():
            pred_mask = (pred == c)
            gt_mask = (gt == c)
            intersection = (pred_mask & gt_mask).sum().float()
            union = (pred_mask | gt_mask).sum().float()
            if union > 0:
                ious.append((intersection / union).item())
    
    metrics['miou'] = np.mean(ious) if ious else 0.0
    
    return metrics


def create_dataloaders(args, dataset_name):
    """Create train and val dataloaders for a dataset."""
    image_size = tuple(args.image_size)
    
    if dataset_name == "synthetic":
        train_dataset = SyntheticShapesDataset(
            num_samples=1000,
            image_size=image_size,
            max_objects=min(args.num_slots, 8),
        )
        val_dataset = SyntheticShapesDataset(
            num_samples=200,
            image_size=image_size,
            max_objects=min(args.num_slots, 8),
        )
    
    elif dataset_name == "clevr":
        clevr_root = os.path.join(args.data_dir, "clevr/CLEVR_v1.0")
        # Use val split since it has masks
        full_dataset = CLEVRDataset(
            root_dir=clevr_root,
            split="val",
            image_size=image_size,
            max_objects=args.num_slots,
            return_masks=True,
        )
        # Split into train/val (80/20)
        n = len(full_dataset)
        train_size = int(0.8 * n)
        train_dataset = Subset(full_dataset, list(range(train_size)))
        val_dataset = Subset(full_dataset, list(range(train_size, n)))
    
    elif dataset_name == "coco":
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
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Apply subset percentage
    if args.subset_percent < 1.0:
        train_size = max(1, int(len(train_dataset) * args.subset_percent))
        val_size = max(1, int(len(val_dataset) * args.subset_percent))
        
        train_indices = torch.randperm(len(train_dataset))[:train_size].tolist()
        val_indices = torch.randperm(len(val_dataset))[:val_size].tolist()
        
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch['image'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images, return_loss=True)
        loss = outputs['loss']
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix(loss=loss.item())
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate model."""
    model.eval()
    
    all_aris = []
    all_mious = []
    total_recon_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        images = batch['image'].to(device)
        outputs = model(images, return_loss=True)
        
        masks = outputs['masks']
        reconstructed = outputs['reconstructed']
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, images).item()
        total_recon_loss += recon_loss
        
        # Compute metrics if GT masks available
        if 'mask' in batch:
            gt_masks = batch['mask'].to(device)
            metrics = compute_metrics(masks, gt_masks)
            all_aris.append(metrics['ari'])
            all_mious.append(metrics['miou'])
        
        num_batches += 1
    
    results = {
        'recon_loss': total_recon_loss / max(num_batches, 1),
    }
    
    if all_aris:
        results['ari'] = np.mean(all_aris)
    if all_mious:
        results['miou'] = np.mean(all_mious)
    
    return results


def main():
    args = parse_args()
    
    print("=" * 60)
    print("SpectralDiffusion End-to-End Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Subset: {args.subset_percent * 100:.1f}%")
    print(f"Device: {args.device}")
    print(f"Image size: {args.image_size}")
    print(f"Num slots: {args.num_slots}")
    print("=" * 60)
    
    # Create model
    from train import create_model
    
    class ModelArgs:
        pass
    
    model_args = ModelArgs()
    model_args.device = args.device
    model_args.num_slots = args.num_slots
    model_args.image_size = args.image_size
    model_args.use_mamba = True
    model_args.init_mode = "spectral"
    model_args.full_model = False
    model_args.use_dino = False
    model_args.backbone = "base"
    model_args.lambda_spec = 0.1
    model_args.lambda_ident = 0.01
    model_args.lambda_diff = 1.0
    
    print("\nCreating model...")
    model = create_model(model_args)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Datasets to train on
    if args.dataset == "all":
        datasets = ["synthetic", "clevr"]  # Skip COCO for now as it's slower
    else:
        datasets = [args.dataset]
    
    # Training loop
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Training on: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        try:
            train_loader, val_loader = create_dataloaders(args, dataset_name)
            print(f"Train samples: {len(train_loader.dataset)}")
            print(f"Val samples: {len(val_loader.dataset)}")
            
            dataset_results = {'train_losses': [], 'val_metrics': []}
            
            for epoch in range(1, args.epochs + 1):
                # Train
                train_loss = train_epoch(model, train_loader, optimizer, args.device, epoch)
                dataset_results['train_losses'].append(train_loss)
                
                # Evaluate
                val_metrics = evaluate(model, val_loader, args.device)
                dataset_results['val_metrics'].append(val_metrics)
                
                # Print epoch results
                print(f"\nEpoch {epoch}/{args.epochs}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Recon Loss: {val_metrics['recon_loss']:.4f}")
                if 'ari' in val_metrics:
                    print(f"  ARI: {val_metrics['ari']:.4f}")
                if 'miou' in val_metrics:
                    print(f"  mIoU: {val_metrics['miou']:.4f}")
            
            all_results[dataset_name] = dataset_results
            
        except Exception as e:
            print(f"Error with {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    for name, results in all_results.items():
        print(f"\n{name.upper()}:")
        if results['train_losses']:
            print(f"  Final Train Loss: {results['train_losses'][-1]:.4f}")
        if results['val_metrics']:
            final = results['val_metrics'][-1]
            print(f"  Final Recon Loss: {final['recon_loss']:.4f}")
            if 'ari' in final:
                print(f"  Final ARI: {final['ari']:.4f}")
            if 'miou' in final:
                print(f"  Final mIoU: {final['miou']:.4f}")
    
    # Save checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        args.output_dir, 
        f"spectral_diffusion_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'results': all_results,
        'args': vars(args),
    }, checkpoint_path)
    print(f"\nCheckpoint saved to: {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("End-to-End Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
