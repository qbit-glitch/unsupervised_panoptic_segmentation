#!/usr/bin/env python3
"""Generate instance training targets from pseudo-instance labels.

Reads: cups_pseudo_labels_v3/{stem}_instance.png (uint16, instance IDs 0-N)
       cups_pseudo_labels_v3/{stem}_semantic.png  (uint8, CUPS 27-class IDs)

Writes per image (at --target_h x --target_w resolution, default 256x512):
  instance_targets/{split}/{city}/{stem}_center.npy   (H, W) float32 heatmap
  instance_targets/{split}/{city}/{stem}_offset.npy   (2, H, W) float32 (dy, dx)
  instance_targets/{split}/{city}/{stem}_boundary.npy  (H, W) uint8 binary

At 256x512: ~1.5MB per image → ~4.5GB for 2975 train images.
At 1024x2048: ~26MB per image → ~77GB (too large).

Usage:
  python mbps_pytorch/generate_instance_targets.py \
    --cityscapes_root /path/to/cityscapes \
    --instance_subdir cups_pseudo_labels_v3 \
    --output_subdir instance_targets \
    --split train --target_h 256 --target_w 512
"""

import argparse
import os
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

# CUPS 27-class → Cityscapes 19-class trainID mapping
# Thing classes in trainID space: 11-18 (person, rider, car, truck, bus, train, motorcycle, bicycle)
_CUPS27_TO_TRAINID = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
    10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18,
    # Extended CUPS classes (19-26) map to 255 (ignore)
    19: 255, 20: 255, 21: 255, 22: 255, 23: 255, 24: 255, 25: 255, 26: 255,
}

_THING_TRAIN_IDS = set(range(11, 19))


def generate_targets_for_image(instance_path, semantic_path=None, min_area=50,
                               target_h=256, target_w=512):
    """Generate center heatmap, offset map, boundary map for one image.

    Args:
        instance_path: Path to uint16 instance PNG (0=bg, 1-N=instances)
        semantic_path: Path to semantic PNG (optional, used to filter thing-only)
        min_area: Minimum instance area in pixels (at original resolution)
        target_h, target_w: Output resolution (default 256x512 to save disk)

    Returns:
        center_map: (target_h, target_w) float32 Gaussian heatmap
        offset_map: (2, target_h, target_w) float32 (dy, dx) from pixel to center
        boundary_map: (target_h, target_w) uint8 binary boundary mask
    """
    inst_orig = np.array(Image.open(instance_path), dtype=np.int32)
    # Resize instance map to target resolution (NEAREST to preserve IDs)
    inst = np.array(
        Image.fromarray(inst_orig.astype(np.uint16)).resize((target_w, target_h), Image.NEAREST),
        dtype=np.int32
    )
    H, W = inst.shape
    # Scale min_area to target resolution
    scale_factor = (target_h * target_w) / (inst_orig.shape[0] * inst_orig.shape[1])
    min_area_scaled = max(int(min_area * scale_factor), 10)

    center_map = np.zeros((H, W), dtype=np.float32)
    offset_map = np.zeros((2, H, W), dtype=np.float32)
    boundary_map = np.zeros((H, W), dtype=np.uint8)

    for uid in np.unique(inst):
        if uid == 0:
            continue  # background / stuff

        mask = inst == uid
        area = mask.sum()
        if area < min_area_scaled:
            continue

        # Centroid
        ys, xs = np.where(mask)
        cy, cx = ys.mean(), xs.mean()

        # Center heatmap (Gaussian, sigma proportional to instance size)
        sigma = max(np.sqrt(area) / 10.0, 4.0)
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        gaussian = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        # Only place gaussian within the instance mask (prevents bleeding into other instances)
        center_map = np.maximum(center_map, gaussian * mask.astype(np.float32))

        # Offset map (dy, dx from each pixel to its center)
        offset_map[0][mask] = cy - ys  # dy
        offset_map[1][mask] = cx - xs  # dx

        # Boundary: pixels at instance edge (erosion-based)
        eroded = ndimage.binary_erosion(mask, iterations=1)
        boundary = mask & ~eroded
        # Dilate by 1px for robustness
        boundary = ndimage.binary_dilation(boundary, iterations=1) & mask
        boundary_map[boundary] = 1

    return center_map, offset_map, boundary_map


def main():
    parser = argparse.ArgumentParser(description="Generate instance targets from pseudo-labels")
    parser.add_argument("--cityscapes_root", type=str, required=True,
                        help="Path to cityscapes dataset root")
    parser.add_argument("--instance_subdir", type=str, default="cups_pseudo_labels_v3",
                        help="Subdirectory containing instance PNGs")
    parser.add_argument("--output_subdir", type=str, default="instance_targets",
                        help="Output subdirectory for .npy targets")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--min_area", type=int, default=50,
                        help="Minimum instance area in pixels (at original resolution)")
    parser.add_argument("--target_h", type=int, default=256,
                        help="Output height (default 256, saves disk vs 1024)")
    parser.add_argument("--target_w", type=int, default=512,
                        help="Output width (default 512, saves disk vs 2048)")
    args = parser.parse_args()

    root = Path(args.cityscapes_root)
    instance_dir = root / args.instance_subdir
    output_dir = root / args.output_subdir

    # Find all images for this split
    img_dir = root / "leftImg8bit" / args.split
    images = sorted(img_dir.glob("*/*_leftImg8bit.png"))
    print(f"Found {len(images)} images in {img_dir}")

    generated = 0
    skipped = 0

    for img_path in tqdm(images, desc=f"Generating {args.split} targets"):
        city = img_path.parent.name
        stem = img_path.name.replace("_leftImg8bit.png", "")

        # Instance label path — try multiple naming conventions and directory structures
        inst_path = None
        for candidate in [
            instance_dir / f"{stem}_leftImg8bit_instance.png",         # CUPS flat
            instance_dir / f"{stem}_instance.png",                      # flat alt
            instance_dir / args.split / city / f"{stem}_leftImg8bit_instance.png",  # city subdirs
            instance_dir / args.split / city / f"{stem}_instance.png",              # city subdirs alt
            instance_dir / city / f"{stem}_leftImg8bit_instance.png",   # city subdirs no split
            instance_dir / city / f"{stem}_instance.png",               # city subdirs no split alt
        ]:
            if candidate.exists():
                inst_path = candidate
                break
        if inst_path is None:
            skipped += 1
            continue

        # Semantic label path (optional)
        sem_path = instance_dir / f"{stem}_leftImg8bit_semantic.png"
        if not sem_path.exists():
            sem_path = None

        # Generate targets
        center, offset, boundary = generate_targets_for_image(
            inst_path, sem_path, min_area=args.min_area,
            target_h=args.target_h, target_w=args.target_w
        )

        # Save
        out_city_dir = output_dir / args.split / city
        out_city_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_city_dir / f"{stem}_center.npy", center)
        np.save(out_city_dir / f"{stem}_offset.npy", offset)
        np.save(out_city_dir / f"{stem}_boundary.npy", boundary)
        generated += 1

    print(f"\nDone: {generated} generated, {skipped} skipped (no instance label)")
    print(f"Targets saved to {output_dir}")


if __name__ == "__main__":
    main()
