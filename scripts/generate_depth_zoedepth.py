#!/usr/bin/env python3
"""Generate monocular depth maps for Cityscapes train images using MiDaS.

Runs MiDaS DPT-Large (PyTorch) on GPU to produce depth maps needed by
CutS3D's LocalCut 3D k-NN graph construction.

Usage:
    python scripts/generate_depth_zoedepth.py \
        --data-dir /media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes \
        --output-dir /media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/depth_zoedepth \
        --image-size 128 256
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Generate MiDaS depth maps")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Cityscapes root (contains leftImg8bit/)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for depth .npy files")
    parser.add_argument("--image-size", type=int, nargs=2, default=[128, 256],
                        help="Target image size [H, W]")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from this city")
    args = parser.parse_args()

    img_h, img_w = args.image_size
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  MiDaS Depth Generation")
    print(f"{'='*60}")
    print(f"  Device:      {device}")
    print(f"  Image size:  {img_h}x{img_w}")
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Load MiDaS DPT-Large model and transforms
    print("\nLoading MiDaS DPT-Large model...")
    sys.stdout.flush()
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
    model = model.to(device)
    model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.dpt_transform
    print("  MiDaS loaded")

    # Find all cities
    img_root = Path(args.data_dir) / "leftImg8bit" / args.split
    cities = sorted([d.name for d in img_root.iterdir() if d.is_dir()])
    total_all = sum(len(list((img_root / c).glob("*_leftImg8bit.png"))) for c in cities)
    print(f"  Found {len(cities)} cities, {total_all} total images")

    # Handle resume
    if args.resume_from and args.resume_from in cities:
        idx = cities.index(args.resume_from)
        cities = cities[idx:]
        print(f"  Resuming from {args.resume_from} ({len(cities)} remaining)")

    total_images = 0
    total_time = 0.0

    for city_idx, city in enumerate(cities):
        city_img_dir = img_root / city
        out_city_dir = Path(args.output_dir) / args.split / city
        out_city_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted(city_img_dir.glob("*_leftImg8bit.png"))

        # Skip already-done images
        todo = []
        for img_path in img_files:
            base = img_path.name.replace("_leftImg8bit.png", "")
            depth_path = out_city_dir / f"{base}.npy"
            if not depth_path.exists():
                todo.append((img_path, base, depth_path))

        if len(todo) == 0:
            print(f"  [{city_idx+1}/{len(cities)}] {city}: all {len(img_files)} done, skipping")
            continue

        print(f"\n  [{city_idx+1}/{len(cities)}] {city}: {len(todo)}/{len(img_files)} to process")
        sys.stdout.flush()

        city_start = time.time()

        for s_idx, (img_path, base, depth_path) in enumerate(todo):
            # Load image as RGB numpy
            img = np.array(Image.open(img_path).convert("RGB"))

            # MiDaS transform and inference
            input_batch = transform(img).to(device)
            with torch.no_grad():
                prediction = model(input_batch)
                # Resize to target resolution
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

            depth = prediction.cpu().numpy().astype(np.float32)
            np.save(depth_path, depth)

            total_images += 1

            if s_idx == 0 or (s_idx + 1) % 20 == 0 or s_idx == len(todo) - 1:
                elapsed = time.time() - city_start
                print(f"    {s_idx+1}/{len(todo)} images, {elapsed:.1f}s")
                sys.stdout.flush()

        city_time = time.time() - city_start
        total_time += city_time
        avg = total_time / total_images if total_images > 0 else 0
        remaining = sum(len(list((img_root / c).glob("*_leftImg8bit.png")))
                        for c in cities[city_idx+1:])
        eta = avg * remaining
        print(
            f"    Done: {len(todo)} images in {city_time:.1f}s "
            f"({city_time/max(len(todo),1):.2f}s/img), "
            f"ETA ~{eta/60:.0f}min"
        )
        sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"  Depth Generation Complete")
    print(f"{'='*60}")
    print(f"  Total images:  {total_images}")
    print(f"  Total time:    {total_time:.1f}s ({total_time/3600:.2f}h)")
    if total_images > 0:
        print(f"  Avg per image: {total_time/total_images:.2f}s")
    print(f"  Output dir:    {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
