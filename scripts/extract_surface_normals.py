#!/usr/bin/env python3
"""Extract surface normals from DepthPro depth maps for DCFA-X.

Computes Sobel gradients on depth maps to estimate surface normals (nx, ny, nz),
then downsamples to the CAUSE patch grid for adapter training.

Output:
    cityscapes/cause_codes_90d/{split}/{city}/{stem}_normals.npy  # (ph, pw, 3)

Usage:
    python scripts/extract_surface_normals.py \
        --cityscapes_root /path/to/cityscapes
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import sobel
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mbps_pytorch.generate_depth_overclustered_semantics import get_cityscapes_images

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO,
)
logger = logging.getLogger(__name__)


def compute_surface_normals(depth_2d: np.ndarray) -> np.ndarray:
    """Estimate surface normals from a depth map via Sobel gradients.

    Args:
        depth_2d: (H, W) depth map.

    Returns:
        (H, W, 3) unit normals (nx, ny, nz).
    """
    gx = sobel(depth_2d, axis=1).astype(np.float32)
    gy = sobel(depth_2d, axis=0).astype(np.float32)
    nz = np.ones_like(gx)
    normals = np.stack([-gx, -gy, nz], axis=-1)  # (H, W, 3)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True).clip(min=1e-8)
    return (normals / norm).astype(np.float32)


def downsample_to_patch_grid(
    array_hwc: np.ndarray, ph: int, pw: int,
) -> np.ndarray:
    """Downsample (H, W, C) array to (ph, pw, C) via adaptive avg pool."""
    c = array_hwc.shape[-1]
    t = torch.from_numpy(array_hwc).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    ds = F.adaptive_avg_pool2d(t, (ph, pw))  # (1, C, ph, pw)
    return ds.squeeze(0).permute(1, 2, 0).numpy()  # (ph, pw, C)


def extract_normals(
    cityscapes_root: str,
    depth_subdir: str,
    output_subdir: str,
    split: str,
    crop_size: int = 322,
    patch_size: int = 14,
) -> None:
    """Extract surface normals for all images in a split."""
    images = get_cityscapes_images(cityscapes_root, split)
    logger.info("Extracting normals for %s: %d images", split, len(images))

    out_dir = os.path.join(cityscapes_root, output_subdir, split)

    for entry in tqdm(images, desc=f"Normals {split}"):
        city = entry["city"]
        stem = entry["stem"]
        city_dir = os.path.join(out_dir, city)
        os.makedirs(city_dir, exist_ok=True)

        normals_path = os.path.join(city_dir, f"{stem}_normals.npy")
        if os.path.isfile(normals_path):
            continue

        depth_npy_path = os.path.join(
            cityscapes_root, depth_subdir, split, city, f"{stem}.npy",
        )
        if not os.path.isfile(depth_npy_path):
            logger.warning("Depth not found: %s", depth_npy_path)
            continue

        depth = np.load(depth_npy_path).astype(np.float32)
        normals = compute_surface_normals(depth)  # (H, W, 3)

        # Compute patch grid size (same as extract_cause_codes.py)
        orig_h, orig_w = depth.shape
        scale = crop_size / min(orig_h, orig_w)
        new_h = (int(round(orig_h * scale)) // patch_size) * patch_size
        new_w = (int(round(orig_w * scale)) // patch_size) * patch_size
        ph = new_h // patch_size
        pw = new_w // patch_size

        normals_ds = downsample_to_patch_grid(normals, ph, pw)  # (ph, pw, 3)
        np.save(normals_path, normals_ds)

    logger.info("Saved normals to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract surface normals from depth")
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--depth_subdir", type=str, default="depth_depthpro")
    parser.add_argument("--output_subdir", type=str, default="cause_codes_90d")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    args = parser.parse_args()

    for split in args.splits:
        extract_normals(
            args.cityscapes_root, args.depth_subdir,
            args.output_subdir, split,
        )


if __name__ == "__main__":
    main()
