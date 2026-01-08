#!/usr/bin/env python
"""
End-to-End 1% Test: Training + Evaluation + Inference

Quick validation that SpectralDiffusion model architecture works properly.
Trains on 1% of available datasets for 1 epoch and reports metrics.

Metrics reported:
- ARI (Adjusted Rand Index) - for CLEVR/Synthetic
- mIoU (mean Intersection over Union) - for COCO/Pascal
- PQ (Panoptic Quality) = SQ * RQ
- Reconstruction Loss
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from train import create_model, compute_ari
from data.clevr import SyntheticShapesDataset, CLEVRDataset


def compute_miou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> float:
    """Compute mean IoU between predicted and ground truth masks."""
    pred_labels = pred_masks.argmax(dim=1)  # [B, H, W]
    gt_labels = gt_masks.argmax(dim=1)  # [B, H, W]
    
    B = pred_labels.shape[0]
    total_iou = 0.0
    
    for b in range(B):
        pred = pred_labels[b].flatten()
        gt = gt_labels[b].flatten()
        
        # Compute IoU for each predicted class
        pred_classes = torch.unique(pred)
        gt_classes = torch.unique(gt)
        
        if len(gt_classes) == 0:
            continue
            
        ious = []
        for c in gt_classes:
            pred_mask = (pred == c)
            gt_mask = (gt == c)
            intersection = (pred_mask & gt_mask).sum().float()
            union = (pred_mask | gt_mask).sum().float()
            if union > 0:
                ious.append((intersection / union).item())
        
        if ious:
            total_iou += np.mean(ious)
    
    return total_iou / B


def compute_pq(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> dict:
    """Compute Panoptic Quality = SQ * RQ."""
    pred_labels = pred_masks.argmax(dim=1)
    gt_labels = gt_masks.argmax(dim=1)
    
    B = pred_labels.shape[0]
    total_pq, total_sq, total_rq = 0.0, 0.0, 0.0
    
    for b in range(B):
        pred = pred_labels[b]
        gt = gt_labels[b]
        
        pred_ids = torch.unique(pred)
        gt_ids = torch.unique(gt)
        
        tp, fp, fn = 0, 0, 0
        sq_sum = 0.0
        
        matched_gt = set()
        for pred_id in pred_ids:
            pred_mask = (pred == pred_id)
            best_iou = 0.0
            best_gt_id = None
            
            for gt_id in gt_ids:
                if gt_id.item() in matched_gt:
                    continue
                gt_mask = (gt == gt_id)
                intersection = (pred_mask & gt_mask).sum().float()
                union = (pred_mask | gt_mask).sum().float()
                iou = (intersection / union).item() if union > 0 else 0.0
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_id = gt_id.item()
            
            if best_iou > 0.5:  # Match threshold
                tp += 1
                sq_sum += best_iou
                matched_gt.add(best_gt_id)
            else:
                fp += 1
        
        fn = len(gt_ids) - len(matched_gt)
        
        sq = sq_sum / tp if tp > 0 else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
        pq = sq * rq
        
        total_pq += pq
        total_sq += sq
        total_rq += rq
    
    return {
        'PQ': total_pq / B,
        'SQ': total_sq / B,
        'RQ': total_rq / B,
    }


def run_test(dataset_name: str, model, device: str, num_samples: int = 50, batch_size: int = 4):
    """Run training and evaluation on a dataset."""
    print(f"\n{'='*60}")
    print(f"Testing on: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Create dataset
    if dataset_name == "synthetic":
        dataset = SyntheticShapesDataset(
            num_samples=num_samples,
            image_size=(128, 128),
            max_objects=5,
        )
    elif dataset_name == "clevr":
        dataset = CLEVRDataset(
            root="datasets/clevr",
            split="val",
            image_size=(128, 128),
        )
    else:
        print(f"Dataset {dataset_name} not implemented yet")
        return None
    
    # Limit to 1%
    subset_size = min(num_samples, len(dataset))
    indices = torch.randperm(len(dataset))[:subset_size]
    from torch.utils.data import Subset, DataLoader
    dataset_subset = Subset(dataset, indices.tolist())
    
    loader = DataLoader(
        dataset_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    print(f"Samples: {len(dataset_subset)}")
    
    # Training phase (1 epoch)
    print("\n--- Training Phase (1 epoch) ---")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    train_losses = []
    start_time = time.time()
    
    for batch in tqdm(loader, desc="Training"):
        images = batch['image'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images, return_loss=True)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
    
    train_time = time.time() - start_time
    avg_train_loss = np.mean(train_losses)
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Train Time: {train_time:.1f}s")
    
    # Evaluation phase
    print("\n--- Evaluation Phase ---")
    model.eval()
    
    metrics = {
        'recon_loss': [],
        'ari': [],
        'miou': [],
        'pq': [],
        'sq': [],
        'rq': [],
    }
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch['image'].to(device)
            outputs = model(images, return_loss=True)
            
            masks = outputs['masks']
            reconstructed = outputs['reconstructed']
            
            # Reconstruction loss
            recon_loss = F.mse_loss(reconstructed, images).item()
            metrics['recon_loss'].append(recon_loss)
            
            # ARI if masks available
            if 'mask' in batch:
                gt_masks = batch['mask'].to(device)
                pred_labels = masks.argmax(dim=1)
                gt_labels = gt_masks.argmax(dim=1)
                ari = compute_ari(pred_labels, gt_labels).mean().item()
                metrics['ari'].append(ari)
                
                miou = compute_miou(masks, gt_masks)
                metrics['miou'].append(miou)
                
                pq_scores = compute_pq(masks, gt_masks)
                metrics['pq'].append(pq_scores['PQ'])
                metrics['sq'].append(pq_scores['SQ'])
                metrics['rq'].append(pq_scores['RQ'])
    
    # Inference phase - generate sample outputs
    print("\n--- Inference Phase ---")
    sample_batch = next(iter(loader))
    images = sample_batch['image'][:2].to(device)
    
    with torch.no_grad():
        outputs = model(images, return_loss=False)
        masks = outputs['masks']
        slots = outputs['slots']
    
    print(f"Input shape: {images.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Slots shape: {slots.shape}")
    
    # Check slot diversity
    unique_slots = [len(torch.unique(masks[i].argmax(dim=0))) for i in range(masks.shape[0])]
    print(f"Unique slots per image: {unique_slots}")
    
    # Summarize results
    results = {
        'dataset': dataset_name,
        'samples': len(dataset_subset),
        'train_loss': avg_train_loss,
        'train_time': train_time,
        'recon_loss': np.mean(metrics['recon_loss']),
    }
    
    if metrics['ari']:
        results['ARI'] = np.mean(metrics['ari'])
    if metrics['miou']:
        results['mIoU'] = np.mean(metrics['miou'])
    if metrics['pq']:
        results['PQ'] = np.mean(metrics['pq'])
        results['SQ'] = np.mean(metrics['sq'])
        results['RQ'] = np.mean(metrics['rq'])
    
    return results


def main():
    parser = argparse.ArgumentParser(description="End-to-End 1% Test")
    parser.add_argument("--datasets", nargs="+", default=["synthetic", "clevr"],
                       help="Datasets to test")
    parser.add_argument("--samples", type=int, default=50, help="Samples per dataset (1%)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--use-dino", action="store_true", help="Use DINOv3 backbone")
    parser.add_argument("--full-model", action="store_true", help="Use full model")
    args = parser.parse_args()
    
    print("="*60)
    print("SpectralDiffusion End-to-End 1% Test")
    print("="*60)
    print(f"Datasets: {args.datasets}")
    print(f"Samples per dataset: {args.samples}")
    print(f"Device: {args.device}")
    print(f"Use DINO: {args.use_dino}")
    print("="*60)
    
    # Create model args manually to avoid argparse conflicts
    class ModelArgs:
        pass
    
    model_args = ModelArgs()
    model_args.device = args.device
    model_args.num_slots = 8
    model_args.image_size = [128, 128]
    model_args.use_mamba = True
    model_args.init_mode = "spectral"
    model_args.full_model = args.full_model
    model_args.use_dino = args.use_dino
    model_args.backbone = "base"
    model_args.lambda_spec = 0.1
    model_args.lambda_ident = 0.01
    model_args.lambda_diff = 1.0
    
    print("\nCreating model...")
    model = create_model(model_args)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    # Run tests
    all_results = []
    for dataset_name in args.datasets:
        try:
            results = run_test(
                dataset_name, model, args.device, 
                num_samples=args.samples, 
                batch_size=args.batch_size
            )
            if results:
                all_results.append(results)
        except Exception as e:
            print(f"Error testing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for results in all_results:
        print(f"\n{results['dataset'].upper()}:")
        print(f"  Train Loss: {results['train_loss']:.4f}")
        print(f"  Recon Loss: {results['recon_loss']:.4f}")
        if 'ARI' in results:
            print(f"  ARI: {results['ARI']:.4f}")
        if 'mIoU' in results:
            print(f"  mIoU: {results['mIoU']:.4f}")
        if 'PQ' in results:
            print(f"  PQ: {results['PQ']:.4f} (SQ={results['SQ']:.4f}, RQ={results['RQ']:.4f})")
        print(f"  Time: {results['train_time']:.1f}s")
    
    print("\n" + "="*60)
    print("End-to-End Test Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
