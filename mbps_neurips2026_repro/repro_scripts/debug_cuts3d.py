#!/usr/bin/env python3
"""Diagnostic script for CutS3D tau_knn tuning.

Runs CutS3D on 3 real Cityscapes val images locally (MPS/CPU) with varying
tau_knn values and reports how many masks are found.

Usage:
    python scripts/debug_cuts3d.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mbps_pytorch"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.instance.cuts3d import (
    compute_affinity_matrix,
    spatial_importance_sharpening,
    normalized_cut,
    local_cut_3d,
    extract_pseudo_masks,
)

# --- Config ---
CITYSCAPES = Path("/Users/qbit-glitch/Desktop/datasets/cityscapes")
FEAT_H, FEAT_W, FEAT_DIM = 32, 64, 768
WORK_H, WORK_W = 512, 1024
DEPTH_SCALE = 80.0

# Cityscapes patch-level intrinsics
CS_FX, CS_FY = 70.7, 70.8
CS_CX, CS_CY = 34.3, 16.1

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# --- Pick 3 images ---
feat_dir = CITYSCAPES / "dinov2_features" / "val" / "frankfurt"
depth_dir = CITYSCAPES / "depth_spidepth" / "val" / "frankfurt"

images = []
for feat_path in sorted(feat_dir.glob("*.npy"))[:3]:
    stem = feat_path.stem.replace("_leftImg8bit", "")
    depth_path = depth_dir / f"{stem}.npy"
    if depth_path.exists():
        images.append((feat_path, depth_path, stem))

print(f"Found {len(images)} test images")

# --- Load one image and inspect intermediate values ---
feat_path, depth_path, stem = images[0]
print(f"\nTest image: {stem}")

features = np.load(str(feat_path)).astype(np.float32)  # (2048, 768)
depth_raw = np.load(str(depth_path)).astype(np.float32)
depth_scaled = depth_raw * DEPTH_SCALE  # meters

print(f"Features shape: {features.shape}, norm range: [{np.linalg.norm(features, axis=-1).min():.3f}, {np.linalg.norm(features, axis=-1).max():.3f}]")
print(f"Depth raw: [{depth_raw.min():.3f}, {depth_raw.max():.3f}]")
print(f"Depth scaled (m): [{depth_scaled.min():.1f}, {depth_scaled.max():.1f}]")

features_t = torch.from_numpy(features).to(DEVICE)
depth_t = torch.from_numpy(depth_scaled).to(DEVICE)
if depth_t.shape != (WORK_H, WORK_W):
    depth_t = F.interpolate(depth_t[None, None], (WORK_H, WORK_W), mode="bilinear", align_corners=False)[0, 0]

# Resize depth to patch level
depth_patch = F.interpolate(depth_t[None, None], (FEAT_H, FEAT_W), mode="bilinear", align_corners=False)[0, 0]
print(f"Depth at patch level: [{depth_patch.min().item():.1f}, {depth_patch.max().item():.1f}] m")

# --- Inspect affinity + NCut ---
W = compute_affinity_matrix(features_t)
W_sharp = spatial_importance_sharpening(W, depth_patch)

bipartition, idx_src, idx_snk, fiedler = normalized_cut(W_sharp, tau_ncut=0.0)
fg_count = int(bipartition.sum().item())
print(f"\nNCut bipartition: {fg_count}/{features_t.shape[0]} foreground patches ({100*fg_count/features_t.shape[0]:.1f}%)")
print(f"Fiedler vector range: [{fiedler.min().item():.4f}, {fiedler.max().item():.4f}]")
print(f"Source patch: {idx_src}, Sink patch: {idx_snk}")

# --- Compute 3D distances for adjacent patches ---
from models.instance.cuts3d import pixels_to_3d
points = pixels_to_3d(depth_patch, fx=CS_FX, fy=CS_FY, cx=CS_CX, cy=CS_CY)
points_flat = points.reshape(-1, 3)

# Sample some patch distances
fg_idx = torch.where(bipartition > 0.5)[0]
print(f"\n3D distance analysis (using CS intrinsics fx={CS_FX}):")
print(f"Sample 3D points (first 5 fg patches): {points_flat[fg_idx[:5]].cpu().numpy()}")

# Nearest neighbor distances for fg patches
if len(fg_idx) > 1:
    fg_pts = points_flat[fg_idx]
    diff = fg_pts[:, None, :] - fg_pts[None, :, :]
    dists = torch.sqrt((diff**2).sum(-1) + 1e-8)
    dists = dists + torch.eye(len(fg_idx), device=DEVICE) * 1e8
    nn_dists, _ = dists.topk(min(5, len(fg_idx)-1), dim=-1, largest=False)
    print(f"k-NN distances within FG: [{nn_dists.min().item():.3f}, {nn_dists.median().item():.3f}, {nn_dists.max().item():.3f}] m")
    print(f"  5th percentile: {torch.quantile(nn_dists.flatten(), 0.05).item():.3f} m")
    print(f"  25th percentile: {torch.quantile(nn_dists.flatten(), 0.25).item():.3f} m")
    print(f"  50th percentile (median): {torch.quantile(nn_dists.flatten(), 0.50).item():.3f} m")

# --- Sweep tau_knn ---
print("\n" + "="*60)
print("tau_knn sweep (2 images each, max_instances=5):")
print("="*60)

tau_values = [0.115, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
img_t = torch.zeros((WORK_H, WORK_W, 3), dtype=torch.float32, device=DEVICE)

for tau in tau_values:
    total_masks = 0
    for feat_path, depth_path, stem in images[:2]:
        features = np.load(str(feat_path)).astype(np.float32)
        depth_raw = np.load(str(depth_path)).astype(np.float32)
        depth_scaled = depth_raw * DEPTH_SCALE

        features_t = torch.from_numpy(features).to(DEVICE)
        depth_t = torch.from_numpy(depth_scaled).to(DEVICE)
        if depth_t.shape != (WORK_H, WORK_W):
            depth_t = F.interpolate(depth_t[None, None], (WORK_H, WORK_W), mode="bilinear", align_corners=False)[0, 0]

        with torch.no_grad():
            result = extract_pseudo_masks(
                features_t, depth_t, img_t,
                patch_h=FEAT_H, patch_w=FEAT_W,
                max_instances=5,
                tau_knn=tau,
                min_mask_size=0.005,
                sc_samples=1,  # fast
                use_crf=False,
                fx=CS_FX, fy=CS_FY, cx=CS_CX, cy=CS_CY,
            )
        total_masks += result.num_valid

    print(f"  tau_knn={tau:5.3f}: {total_masks} masks across 2 images")

print("\nDone.")
