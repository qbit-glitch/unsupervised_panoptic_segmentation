#!/usr/bin/env python
"""
Train SpectralDiffusion V2 - Complete Implementation

Uses the full model with:
- SlotConditionedDiffusion decoder
- GMM prior for identifiability
- Full 5-component loss
- All float32
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

# Force float32
torch.set_default_dtype(torch.float32)

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="clevr",
                       choices=["synthetic", "clevr", "coco"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--num-slots", type=int, default=12)
    parser.add_argument("--image-size", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--subset-percent", type=float, default=0.01)
    parser.add_argument("--data-dir", type=str, default="../datasets")
    parser.add_argument("--use-diffusion", action="store_true", default=False,
                       help="Use diffusion decoder (slower but better)")
    parser.add_argument("--use-dino", action="store_true", default=False)
    return parser.parse_args()


def create_dataloader(args):
    """Create dataloaders."""
    from torch.utils.data import DataLoader, Subset
    
    if args.dataset == "synthetic":
        from data.clevr import SyntheticShapesDataset
        train_ds = SyntheticShapesDataset(
            num_samples=1000,
            image_size=tuple(args.image_size),
            max_objects=min(args.num_slots, 8),
        )
        val_ds = SyntheticShapesDataset(
            num_samples=200,
            image_size=tuple(args.image_size),
            max_objects=min(args.num_slots, 8),
        )
    elif args.dataset == "clevr":
        from data.clevr import CLEVRDataset
        clevr_root = os.path.join(args.data_dir, "clevr/CLEVR_v1.0")
        full_ds = CLEVRDataset(
            root_dir=clevr_root,
            split="val",  # Has masks
            image_size=tuple(args.image_size),
            max_objects=args.num_slots,
            return_masks=True,
        )
        n = len(full_ds)
        train_size = int(0.8 * n)
        train_ds = Subset(full_ds, list(range(train_size)))
        val_ds = Subset(full_ds, list(range(train_size, n)))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Apply subset
    if args.subset_percent < 1.0:
        train_size = max(1, int(len(train_ds) * args.subset_percent))
        val_size = max(1, int(len(val_ds) * args.subset_percent))
        train_ds = Subset(train_ds, list(range(train_size)))
        val_ds = Subset(val_ds, list(range(val_size)))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, val_loader


def compute_ari(pred_masks, gt_masks):
    """Compute Adjusted Rand Index."""
    pred_labels = pred_masks.argmax(dim=1).flatten()
    gt_labels = gt_masks.argmax(dim=1).flatten() if gt_masks.dim() == 4 else gt_masks.flatten()
    
    N = len(pred_labels)
    if N == 0:
        return 0.0
    
    # Simple ARI computation
    pred_ids = pred_labels.unique()
    gt_ids = gt_labels.unique()
    
    contingency = torch.zeros(len(pred_ids), len(gt_ids), device=pred_labels.device)
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
    expected = sum_comb_a * sum_comb_b / max(comb2(torch.tensor(N, dtype=torch.float)), 1e-8)
    max_index = (sum_comb_a + sum_comb_b) / 2
    
    if max_index - expected == 0:
        return 1.0 if sum_comb_c == expected else 0.0
    
    ari = (sum_comb_c - expected) / (max_index - expected)
    return ari.item()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("SpectralDiffusion V2 Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Diffusion decoder: {args.use_diffusion}")
    print(f"DINOv2: {args.use_dino}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Create model
    from models.spectral_diffusion_v2 import SpectralDiffusionV2
    
    model = SpectralDiffusionV2(
        image_size=tuple(args.image_size),
        num_slots=args.num_slots,
        feature_dim=256,
        use_dino=args.use_dino,
        use_mamba=True,
        use_diffusion_decoder=args.use_diffusion,
        init_mode="spectral",
        diffusion_steps=50,  # Fewer steps for faster training
    ).to(args.device).float()
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    # Dataloaders
    train_loader, val_loader = create_dataloader(args)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training
    best_ari = -1.0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_diff = 0.0
        train_recon = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            images = batch['image'].to(args.device).float()
            
            optimizer.zero_grad()
            outputs = model(images, return_loss=True)
            
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            loss_dict = outputs.get('loss_dict', {})
            train_diff += loss_dict.get('diff', 0.0)
            train_recon += loss_dict.get('recon', 0.0)
            
            pbar.set_postfix(loss=loss.item())
        
        scheduler.step()
        
        n_batches = len(train_loader)
        avg_loss = train_loss / n_batches
        avg_diff = train_diff / n_batches
        avg_recon = train_recon / n_batches
        
        # Evaluate
        model.eval()
        all_aris = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(args.device).float()
                outputs = model(images, return_loss=False)
                masks = outputs['masks']
                
                if 'mask' in batch:
                    gt_masks = batch['mask'].to(args.device)
                    ari = compute_ari(masks, gt_masks)
                    all_aris.append(ari)
        
        avg_ari = sum(all_aris) / len(all_aris) if all_aris else 0.0
        
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Diff Loss: {avg_diff:.4f}")
        print(f"  Recon Loss: {avg_recon:.4f}")
        print(f"  Val ARI: {avg_ari:.4f}")
        
        if avg_ari > best_ari:
            best_ari = avg_ari
    
    # Save final checkpoint
    os.makedirs("./outputs", exist_ok=True)
    checkpoint_path = f"./outputs/spectral_v2_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_ari': best_ari,
        'args': vars(args),
    }, checkpoint_path)
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best ARI: {best_ari:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
