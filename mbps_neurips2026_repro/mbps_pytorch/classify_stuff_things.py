#!/usr/bin/env python3
"""Classify semantic clusters into stuff vs things — fully unsupervised.

Uses semantic pseudo-labels + depth maps (no GT labels, no instance masks).

Key signal: depth_split_ratio — for each class, how many MORE connected
components appear after removing depth-edge pixels from the class mask.
  - Things (cars, people): adjacent objects at different depths → depth edges
    split merged blobs into individual instances → high split ratio (>1.5x)
  - Stuff (road, sky): single continuous region, smooth depth → depth edges
    don't create new splits → low split ratio (~1.0x)
  - Already-separate stuff (pole, traffic sign): each object is its own CC
    already, depth edges don't add new splits → low split ratio (~1.0x)

Usage:
    python mbps_pytorch/classify_stuff_things.py \
        --semantic_dir /data/cityscapes/pseudo_semantic_dinov3/train \
        --depth_dir /data/cityscapes/depth_dav3/train \
        --output_path /data/cityscapes/pseudo_semantic_dinov3/stuff_things.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel
from tqdm import tqdm


def compute_cluster_statistics(
    semantic_dir: str,
    depth_dir: str,
    num_classes: int,
    max_images: int = None,
    grad_threshold: float = 0.05,
    depth_blur_sigma: float = 1.0,
) -> dict:
    """Compute per-cluster statistics including depth-split ratio.

    For each class and each image:
      1. Compute CCs from semantic mask alone (original_cc_count)
      2. Remove depth-edge pixels from class mask, re-compute CCs (split_cc_count)
      3. depth_split_ratio = split_cc_count / original_cc_count

    Things have high split_ratio because adjacent same-class objects get
    separated by depth edges. Stuff and already-separate objects have ~1.0.
    """
    semantic_dir = Path(semantic_dir)
    depth_dir = Path(depth_dir)

    semantic_files = sorted(semantic_dir.rglob("*.png"))
    if max_images is not None:
        semantic_files = semantic_files[:max_images]
    print(f"Computing statistics over {len(semantic_files)} images...")

    # Per-cluster accumulators
    cluster_pixels = np.zeros(num_classes, dtype=np.int64)
    cluster_region_counts = [[] for _ in range(num_classes)]
    cluster_split_counts = [[] for _ in range(num_classes)]
    cluster_region_sizes = [[] for _ in range(num_classes)]
    cluster_region_mean_depths = [[] for _ in range(num_classes)]
    cluster_intra_depth_vars = [[] for _ in range(num_classes)]
    cluster_max_region_fracs = [[] for _ in range(num_classes)]

    for sem_path in tqdm(semantic_files, desc="Computing statistics"):
        sem = np.array(Image.open(sem_path))
        H, W = sem.shape

        # Find matching depth map
        rel = sem_path.relative_to(semantic_dir)
        depth_path = depth_dir / rel.parent / f"{sem_path.stem}.npy"
        if not depth_path.exists():
            continue

        depth = np.load(str(depth_path)).astype(np.float64)
        if depth.shape != (H, W):
            depth = np.array(
                Image.fromarray(depth.astype(np.float32)).resize((W, H), Image.BILINEAR)
            ).astype(np.float64)

        image_area = H * W

        # Compute depth edges (same as depth-guided instance generation)
        depth_smooth = gaussian_filter(depth, sigma=depth_blur_sigma)
        gx = sobel(depth_smooth, axis=1)
        gy = sobel(depth_smooth, axis=0)
        grad_mag = np.sqrt(gx**2 + gy**2)
        depth_edges = grad_mag > grad_threshold

        for k in range(num_classes):
            mask = sem == k
            n_pixels = int(mask.sum())
            if n_pixels == 0:
                continue

            cluster_pixels[k] += n_pixels

            # Original CCs (semantic mask only)
            labeled, num_regions = ndimage.label(mask)
            cluster_region_counts[k].append(num_regions)

            # Depth-split CCs (remove depth edges, then re-label)
            split_mask = mask & ~depth_edges
            _, num_split = ndimage.label(split_mask)
            cluster_split_counts[k].append(num_split)

            # Max region fraction (largest CC / total class pixels)
            if num_regions > 0:
                region_sizes = ndimage.sum(mask, labeled, range(1, num_regions + 1))
                max_frac = float(max(region_sizes)) / n_pixels
                cluster_max_region_fracs[k].append(max_frac)

            # Per-region statistics
            for region_id in range(1, num_regions + 1):
                region_mask = labeled == region_id
                region_size = int(region_mask.sum())
                cluster_region_sizes[k].append(region_size / image_area)

                region_depths = depth[region_mask]
                if len(region_depths) > 1:
                    cluster_region_mean_depths[k].append(float(region_depths.mean()))
                    cluster_intra_depth_vars[k].append(float(region_depths.var()))

    # Compute summary statistics
    stats = {}
    for k in range(num_classes):
        total = int(cluster_pixels[k])
        if total == 0:
            stats[k] = {
                "avg_region_count": 0.0,
                "avg_split_count": 0.0,
                "depth_split_ratio": 1.0,
                "avg_relative_size": 0.0,
                "depth_spread": 0.0,
                "intra_depth_var": 0.0,
                "max_region_fraction": 1.0,
                "pixel_count": 0,
            }
            continue

        avg_rc = float(np.mean(cluster_region_counts[k])) if cluster_region_counts[k] else 0.0
        avg_sc = float(np.mean(cluster_split_counts[k])) if cluster_split_counts[k] else 0.0
        split_ratio = avg_sc / max(avg_rc, 1.0)
        avg_rs = float(np.mean(cluster_region_sizes[k])) if cluster_region_sizes[k] else 0.0
        depth_spread = float(np.std(cluster_region_mean_depths[k])) if len(cluster_region_mean_depths[k]) > 1 else 0.0
        intra_dv = float(np.mean(cluster_intra_depth_vars[k])) if cluster_intra_depth_vars[k] else 0.0
        max_rf = float(np.mean(cluster_max_region_fracs[k])) if cluster_max_region_fracs[k] else 1.0

        stats[k] = {
            "avg_region_count": avg_rc,
            "avg_split_count": avg_sc,
            "depth_split_ratio": split_ratio,
            "avg_relative_size": avg_rs,
            "depth_spread": depth_spread,
            "intra_depth_var": intra_dv,
            "max_region_fraction": max_rf,
            "pixel_count": total,
        }

    return stats


def classify_stuff_things(
    stats: dict,
    num_classes: int,
    n_things: int = 8,
    num_images: int = 200,
    big_stuff_size_thresh: float = 0.02,
    big_stuff_pix_thresh: float = 5000.0,
) -> dict:
    """Two-stage stuff/things classification using depth-split ratio.

    Stage 1: Force "big stuff" — classes with large per-region size AND
    high per-image pixel coverage. These are amorphous background classes
    (road, building, vegetation, sky) that have artificially high split
    ratios due to internal depth complexity, not actual object boundaries.

    Stage 2: Among remaining classes, rank by depth_split_ratio. Classes
    where depth edges create the most new CCs (splitting adjacent objects)
    are things. Top n_things → things, rest → stuff.

    With n_things=8: captures person, car, truck, rider, motorcycle + 3 stuff
    With n_things=12: captures all 8 GT Cityscapes things (recommended)
    """
    # Stage 1: identify "big stuff" (large coverage + large regions)
    big_stuff = set()
    for k in range(num_classes):
        s = stats[k]
        if s["pixel_count"] == 0:
            continue
        pix_per_img = s["pixel_count"] / max(num_images, 1)
        if (s["avg_relative_size"] > big_stuff_size_thresh
                and pix_per_img > big_stuff_pix_thresh):
            big_stuff.add(k)

    # Stage 2: rank remaining by split_ratio
    scores = {}
    for k in range(num_classes):
        s = stats[k]
        if s["pixel_count"] == 0:
            scores[k] = -999.0
        elif k in big_stuff:
            scores[k] = -500.0  # forced stuff
        else:
            scores[k] = s["depth_split_ratio"]

    # Top n_things by score → things
    valid = [(k, s) for k, s in scores.items() if s > -999]
    valid.sort(key=lambda x: -x[1])

    thing_set = set()
    for i, (k, s) in enumerate(valid):
        if s <= -500:
            break  # don't make forced-stuff into things
        if i < n_things:
            thing_set.add(k)

    classification = {}
    for k in range(num_classes):
        if scores[k] <= -999:
            classification[k] = {"label": "stuff", "score": 0.0}
        else:
            label = "thing" if k in thing_set else "stuff"
            classification[k] = {"label": label, "score": scores[k]}

    return classification


def main():
    parser = argparse.ArgumentParser(
        description="Classify semantic clusters as stuff vs things (unsupervised)"
    )
    parser.add_argument(
        "--semantic_dir", type=str, required=True,
        help="Directory with pseudo semantic label PNGs"
    )
    parser.add_argument(
        "--depth_dir", type=str, required=True,
        help="Directory with depth map NPYs"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to save stuff_things.json"
    )
    parser.add_argument(
        "--num_classes", type=int, default=19,
        help="Number of semantic classes/clusters"
    )
    parser.add_argument(
        "--n_things", type=int, default=8,
        help="Number of thing classes to select from non-big-stuff classes "
             "(default: 8; use 12 to capture all GT Cityscapes things)"
    )
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="Limit images processed (for speed)"
    )
    parser.add_argument(
        "--grad_threshold", type=float, default=0.05,
        help="Depth gradient edge threshold (default: 0.05)"
    )
    parser.add_argument(
        "--depth_blur", type=float, default=1.0,
        help="Gaussian blur sigma on depth before gradient (default: 1.0)"
    )

    CS_NAMES = [
        "road", "sidewalk", "building", "wall", "fence",
        "pole", "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck",
        "bus", "train", "motorcycle", "bicycle",
    ]

    args = parser.parse_args()

    # Compute statistics
    stats = compute_cluster_statistics(
        semantic_dir=args.semantic_dir,
        depth_dir=args.depth_dir,
        num_classes=args.num_classes,
        max_images=args.max_images,
        grad_threshold=args.grad_threshold,
        depth_blur_sigma=args.depth_blur,
    )

    num_images = len(list(Path(args.semantic_dir).rglob("*.png")))
    if args.max_images is not None:
        num_images = min(num_images, args.max_images)

    # Classify
    classification = classify_stuff_things(
        stats=stats,
        num_classes=args.num_classes,
        n_things=args.n_things,
        num_images=num_images,
    )

    # Print results
    print(f"\nStuff-Things Classification ({args.num_classes} classes, "
          f"top {args.n_things} by score -> things):")
    print("-" * 110)
    print(f"  {'ID':>3}  {'Name':>15}  {'Label':>5}  {'Score':>7}  "
          f"{'Regions':>7}  {'SplitCC':>7}  {'Ratio':>6}  "
          f"{'RelSize':>8}  {'MaxRF':>6}  {'DepSpd':>7}")
    print("-" * 110)

    sorted_classes = sorted(range(args.num_classes),
                           key=lambda k: classification[k]["score"], reverse=True)

    n_stuff = 0
    n_things_count = 0
    gt_stuff = set(range(0, 11))
    gt_things = set(range(11, 19))
    correct = 0

    for k in sorted_classes:
        c = classification[k]
        s = stats[k]
        label = c["label"]
        name = CS_NAMES[k] if k < len(CS_NAMES) else f"cluster_{k}"

        if label == "stuff":
            n_stuff += 1
        else:
            n_things_count += 1

        gt_label = "thing" if k in gt_things else "stuff"
        match = "Y" if label == gt_label else "N"
        if label == gt_label:
            correct += 1

        print(f"  {k:3d}  {name:>15}  {label:>5}  {c['score']:7.3f}  "
              f"{s['avg_region_count']:7.1f}  {s['avg_split_count']:7.1f}  "
              f"{s['depth_split_ratio']:6.2f}  "
              f"{s['avg_relative_size']:8.4f}  {s['max_region_fraction']:6.3f}  "
              f"{s['depth_spread']:7.4f}  "
              f"{match} (GT: {gt_label})")

    print(f"\nResult: {n_stuff} stuff, {n_things_count} things")
    print(f"Accuracy vs GT split: {correct}/{args.num_classes} "
          f"({100*correct/args.num_classes:.0f}%)")

    stuff_ids = sorted([k for k in range(args.num_classes)
                       if classification[k]["label"] == "stuff"])
    thing_ids = sorted([k for k in range(args.num_classes)
                       if classification[k]["label"] == "thing"])
    print(f"Stuff IDs: {stuff_ids}")
    print(f"Thing IDs: {thing_ids}")

    # Save
    output = {
        "classification": {str(k): v for k, v in classification.items()},
        "statistics": {str(k): v for k, v in stats.items()},
        "stuff_ids": stuff_ids,
        "thing_ids": thing_ids,
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
