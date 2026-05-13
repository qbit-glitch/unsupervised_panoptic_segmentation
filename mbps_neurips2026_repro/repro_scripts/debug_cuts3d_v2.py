#!/usr/bin/env python3
"""CutS3D tau_knn quality test — runs on all 3 local images, checks thing-class hit rate.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/debug_cuts3d_v2.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mbps_pytorch"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.instance.cuts3d import extract_pseudo_masks

CITYSCAPES = Path("/Users/qbit-glitch/Desktop/datasets/cityscapes")
FEAT_H, FEAT_W = 32, 64
WORK_H, WORK_W = 512, 1024
DEPTH_SCALE = 80.0

CS_FX, CS_FY = 70.7, 70.8
CS_CX, CS_CY = 34.3, 16.1

THING_IDS = set(range(11, 19))
CS_NAMES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic_light", 7: "traffic_sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}\n")

# Load all 3 frankfurt images
feat_dir = CITYSCAPES / "dinov2_features" / "val" / "frankfurt"
depth_dir = CITYSCAPES / "depth_spidepth" / "val" / "frankfurt"
sem_dir = CITYSCAPES / "pseudo_semantic_mapped_k80" / "val" / "frankfurt"

entries = []
for feat_path in sorted(feat_dir.glob("*.npy"))[:3]:
    stem = feat_path.stem.replace("_leftImg8bit", "")
    depth_path = depth_dir / f"{stem}.npy"
    sem_path = sem_dir / f"{stem}.png"
    if depth_path.exists() and sem_path.exists():
        entries.append((feat_path, depth_path, sem_path, stem))

print(f"Loaded {len(entries)} images\n")

img_t = torch.zeros((WORK_H, WORK_W, 3), dtype=torch.float32, device=DEVICE)

# Full sweep with quality metrics
tau_values = [0.115, 0.5, 1.0, 2.0, 3.0, 5.0]

print(f"{'tau':>6}  {'total':>6}  {'thing%':>7}  {'per_img':>7}  class breakdown")
print("-" * 65)

for tau in tau_values:
    total_masks = 0
    thing_masks = 0
    class_counts = {i: 0 for i in range(19)}

    for feat_path, depth_path, sem_path, stem in entries:
        # Load inputs
        features = torch.from_numpy(np.load(str(feat_path)).astype(np.float32)).to(DEVICE)
        depth_raw = np.load(str(depth_path)).astype(np.float32)
        depth_t = torch.from_numpy(depth_raw * DEPTH_SCALE).to(DEVICE)
        if depth_t.shape != (WORK_H, WORK_W):
            depth_t = F.interpolate(depth_t[None, None], (WORK_H, WORK_W), mode="bilinear", align_corners=False)[0, 0]
        semantic = np.array(Image.open(str(sem_path)))
        if semantic.shape != (WORK_H, WORK_W):
            semantic = np.array(Image.fromarray(semantic).resize((WORK_W, WORK_H), Image.NEAREST))

        with torch.no_grad():
            result = extract_pseudo_masks(
                features, depth_t, img_t,
                patch_h=FEAT_H, patch_w=FEAT_W,
                max_instances=20,
                tau_knn=tau,
                min_mask_size=0.005,
                sc_samples=2,
                use_crf=False,
                fx=CS_FX, fy=CS_FY, cx=CS_CX, cy=CS_CY,
            )

        for i in range(result.num_valid):
            mask_patch = result.masks[i].cpu().numpy()
            mask_2d = mask_patch.reshape(FEAT_H, FEAT_W)
            mask_full = np.array(
                Image.fromarray(mask_2d.astype(np.float32)).resize((WORK_W, WORK_H), Image.BILINEAR)
            ) > 0.5
            if mask_full.sum() < 100:
                continue

            sem_vals = semantic[mask_full]
            sem_vals = sem_vals[sem_vals < 19]
            if len(sem_vals) == 0:
                continue
            counts = np.bincount(sem_vals, minlength=19)
            cls = int(counts.argmax())
            class_counts[cls] += 1
            total_masks += 1
            if cls in THING_IDS:
                thing_masks += 1

    thing_pct = 100 * thing_masks / max(total_masks, 1)
    per_img = total_masks / max(len(entries), 1)

    # Show top thing classes
    thing_breakdown = {CS_NAMES[c]: class_counts[c] for c in range(11, 19) if class_counts[c] > 0}
    thing_str = " ".join(f"{k}={v}" for k, v in sorted(thing_breakdown.items(), key=lambda x: -x[1]))

    print(f"{tau:>6.3f}  {total_masks:>6}  {thing_pct:>6.1f}%  {per_img:>7.1f}  {thing_str}")

print("\nDone.")
