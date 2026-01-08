"""
Evaluation Script for SpectralDiffusion

Usage:
    python evaluate.py --checkpoint path/to/checkpoint.pt --dataset clevr
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent))

from data.clevr import CLEVRDataset, SyntheticShapesDataset
from data.coco import COCOStuffDataset
from utils.metrics import (
    PanopticMetrics,
    compute_ari,
    compute_foreground_ari,
    compute_pq,
    compute_miou,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SpectralDiffusion")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="synthetic",
                       choices=["synthetic", "clevr", "coco", "pascal_voc"])
    parser.add_argument("--data-dir", type=str, default="./datasets")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--num-vis", type=int, default=10,
                       help="Number of samples to visualize")
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model - for now use simplified version
    from train import create_model, parse_args as parse_train_args
    
    # Create dummy args
    class Args:
        def __init__(self):
            self.backbone = "base"
            self.num_slots = 12
            self.use_mamba = True
            self.use_diffusion = False
            self.init_mode = "spectral"
            self.image_size = [128, 128]
            self.device = device
    
    args = Args()
    model = create_model(args)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def create_dataloader(args) -> DataLoader:
    """Create evaluation dataloader."""
    image_size = (128, 128)  # Default
    
    if args.dataset == "synthetic":
        dataset = SyntheticShapesDataset(
            num_samples=500,
            image_size=image_size,
        )
    elif args.dataset == "clevr":
        dataset = CLEVRDataset(
            root_dir=os.path.join(args.data_dir, "clevr/CLEVR_v1.0"),
            split=args.split,
            image_size=image_size,
        )
    elif args.dataset == "coco":
        split = "val2017" if args.split == "val" else "train2017"
        dataset = COCOStuffDataset(
            root_dir=os.path.join(args.data_dir, "coco_stuff"),
            split=split,
            image_size=image_size,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, args) -> Dict[str, float]:
    """Run evaluation."""
    model.eval()
    
    all_ari = []
    all_fg_ari = []
    all_pq = []
    all_sq = []
    all_rq = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(args.device)
        
        # Forward pass
        outputs = model(images, return_loss=False)
        masks = outputs['masks']  # [B, K, H, W]
        
        # Convert to label maps
        pred_labels = masks.argmax(dim=1)  # [B, H, W]
        
        # Compute metrics if ground truth available
        if 'mask' in batch:
            gt_masks = batch['mask'].to(args.device)
            gt_labels = gt_masks.argmax(dim=1)
            
            # ARI
            ari = compute_ari(pred_labels, gt_labels)
            all_ari.extend(ari.cpu().numpy().tolist())
            
            # FG-ARI
            fg_ari = compute_foreground_ari(pred_labels, gt_labels)
            all_fg_ari.extend(fg_ari.cpu().numpy().tolist())
            
            # PQ
            pq_result = compute_pq(masks, gt_masks)
            all_pq.append(pq_result['PQ'])
            all_sq.append(pq_result['SQ'])
            all_rq.append(pq_result['RQ'])
        
        pbar.set_postfix({
            'ARI': f"{np.mean(all_ari) if all_ari else 0:.4f}",
        })
    
    results = {
        'ARI': np.mean(all_ari) if all_ari else 0,
        'ARI_std': np.std(all_ari) if all_ari else 0,
        'FG_ARI': np.mean(all_fg_ari) if all_fg_ari else 0,
        'PQ': np.mean(all_pq) if all_pq else 0,
        'SQ': np.mean(all_sq) if all_sq else 0,
        'RQ': np.mean(all_rq) if all_rq else 0,
        'num_samples': len(all_ari),
    }
    
    return results


def visualize_results(
    model: nn.Module,
    dataloader: DataLoader,
    args,
    num_samples: int = 10,
):
    """Generate and save visualizations."""
    import matplotlib.pyplot as plt
    
    output_dir = Path(args.output_dir) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    sample_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if sample_count >= num_samples:
            break
        
        images = batch['image'].to(args.device)
        
        with torch.no_grad():
            outputs = model(images, return_loss=False)
        
        masks = outputs['masks']  # [B, K, H, W]
        
        for i in range(min(len(images), num_samples - sample_count)):
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            # Original image
            img = images[i].cpu().permute(1, 2, 0).numpy()
            axes[0, 0].imshow(img)
            axes[0, 0].set_title("Input Image")
            axes[0, 0].axis('off')
            
            # Predicted segmentation
            pred = masks[i].argmax(dim=0).cpu().numpy()
            axes[0, 1].imshow(pred, cmap='tab20')
            axes[0, 1].set_title("Predicted Segmentation")
            axes[0, 1].axis('off')
            
            # Ground truth if available
            if 'mask' in batch:
                gt = batch['mask'][i].argmax(dim=0).cpu().numpy()
                axes[0, 2].imshow(gt, cmap='tab20')
                axes[0, 2].set_title("Ground Truth")
                axes[0, 2].axis('off')
            
            # Individual slot masks
            for j in range(min(5, masks.shape[1])):
                ax = axes[1, j] if j < 4 else axes[0, 3]
                mask_j = masks[i, j].cpu().numpy()
                ax.imshow(mask_j, cmap='viridis')
                ax.set_title(f"Slot {j}")
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"sample_{sample_count}.png", dpi=150)
            plt.close()
            
            sample_count += 1
    
    print(f"Saved {sample_count} visualizations to {output_dir}")


def main():
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SpectralDiffusion Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, args.device)
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader = create_dataloader(args)
    print(f"Evaluation samples: {len(dataloader.dataset)}")
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluate(model, dataloader, args)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Save results
    results_path = output_dir / f"results_{args.dataset}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Generate visualizations
    if args.save_visualizations:
        print("\nGenerating visualizations...")
        visualize_results(model, dataloader, args, args.num_vis)
    
    # Check against targets
    print("\n" + "=" * 60)
    print("TARGET COMPARISON")
    print("=" * 60)
    
    if args.dataset == "clevr" or args.dataset == "synthetic":
        target_ari = 0.92
        achieved = results.get('ARI', 0)
        status = "✓" if achieved >= target_ari else "✗"
        print(f"ARI: {achieved:.4f} / {target_ari:.2f} [{status}]")
    
    if args.dataset == "coco":
        target_pq = 0.38
        achieved = results.get('PQ', 0)
        status = "✓" if achieved >= target_pq else "✗"
        print(f"PQ: {achieved:.4f} / {target_pq:.2f} [{status}]")


if __name__ == "__main__":
    main()
