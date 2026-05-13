#!/usr/bin/env python3
"""Fast CutS3D debug: 1 image, tests semantic-guided extraction.

Uses CPU. Run time: ~60-90s.

Usage:
    python scripts/debug_cuts3d_fast.py
"""
import sys, os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
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
THING_IDS = set(range(11, 19))
CS_NAMES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic_light", 7: "traffic_sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}
CS_FX, CS_FY = 70.7, 70.8
CS_CX, CS_CY = 34.3, 16.1

DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# Load one image
feat_dir = CITYSCAPES / "dinov2_features" / "val" / "frankfurt"
depth_dir = CITYSCAPES / "depth_spidepth" / "val" / "frankfurt"
sem_dir = CITYSCAPES / "pseudo_semantic_mapped_k80" / "val" / "frankfurt"

feat_path = sorted(feat_dir.glob("*.npy"))[0]
stem = feat_path.stem.replace("_leftImg8bit", "")
print(f"Image: {stem}\n")

features = torch.from_numpy(np.load(str(feat_path)).astype(np.float32))
# Keep depth [0,1] — local_cut_3d normalizes 3D coords internally
depth_raw = np.load(str((depth_dir / f"{stem}.npy"))).astype(np.float32)  # [0,1]
depth_t = torch.from_numpy(depth_raw)

# Load semantic and resize to WORK_H×WORK_W
sem_raw = np.array(Image.open(str(sem_dir / f"{stem}.png")))
if sem_raw.shape[:2] != (WORK_H, WORK_W):
    semantic = np.array(Image.fromarray(sem_raw).resize((WORK_W, WORK_H), Image.NEAREST))
else:
    semantic = sem_raw

img_t = torch.zeros((WORK_H, WORK_W, 3), dtype=torch.float32)

# Build thing-class active mask at patch level
sem_patch = np.array(Image.fromarray(semantic).resize((FEAT_W, FEAT_H), Image.NEAREST))
thing_mask_2d = np.zeros((FEAT_H, FEAT_W), dtype=np.float32)
for cls_id in THING_IDS:
    thing_mask_2d[sem_patch == cls_id] = 1.0
initial_active = torch.from_numpy(thing_mask_2d.reshape(-1))
n_thing_patches = int(thing_mask_2d.sum())
print(f"Thing-class patches: {n_thing_patches}/{FEAT_H*FEAT_W} ({100*n_thing_patches/(FEAT_H*FEAT_W):.1f}%)")

# ---- Full extraction with thing-guided active mask ----
import time
print("\nRunning extract_pseudo_masks (thing-guided, tau=0.115, max=20, sc=1)...")
t0 = time.time()
with torch.no_grad():
    result = extract_pseudo_masks(
        features, depth_t, img_t,
        patch_h=FEAT_H, patch_w=FEAT_W,
        max_instances=20,
        tau_knn=0.115,
        min_mask_size=0.005,
        max_mask_size=0.25,
        sc_samples=1,
        use_crf=False,
        fx=CS_FX, fy=CS_FY, cx=CS_CX, cy=CS_CY,
        initial_active=initial_active,
    )
elapsed = time.time() - t0
print(f"Time: {elapsed:.1f}s, num_valid: {result.num_valid}")

thing_count = 0
print("\nMask breakdown:")
for i in range(result.num_valid):
    mask_patch = result.masks[i].numpy()
    mask_2d = mask_patch.reshape(FEAT_H, FEAT_W)
    mask_full = np.array(Image.fromarray(mask_2d.astype(np.float32)).resize((WORK_W, WORK_H), Image.BILINEAR)) > 0.5
    score = float(result.scores[i].item())
    px_count = int(mask_full.sum())

    sem_vals = semantic[mask_full]
    sem_vals = sem_vals[sem_vals < 19]
    cls = int(np.bincount(sem_vals, minlength=19).argmax()) if len(sem_vals) > 0 else -1
    is_thing = cls in THING_IDS
    if is_thing:
        thing_count += 1
    print(f"  Mask {i}: {CS_NAMES.get(cls,'?')} (cls={cls}), thing={is_thing}, px={px_count}, score={score:.3f}")

print(f"\nThing-class masks: {thing_count}/{result.num_valid}")
print("Done.")
