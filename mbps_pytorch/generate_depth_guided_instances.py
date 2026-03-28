#!/usr/bin/env python3
"""Generate instance pseudo-labels via depth-guided connected components.

Uses DA3 depth maps + DINOv3 semantic pseudo-labels to produce object-level
instance masks. Depth gradient edges split adjacent same-class objects that
simple connected components would merge.

Algorithm per image:
  1. Load semantic labels (trainIDs 0-18) and DA3 depth map
  2. Compute depth gradient magnitude (Sobel filter)
  3. For each thing class (from stuff_things.json or default trainID 11-18):
     a. Get binary class mask from semantic labels
     b. Remove pixels at depth discontinuities (gradient > threshold)
     c. Connected components on the split mask
     d. Filter by minimum area, reclaim boundary pixels via dilation
  4. Save instances as NPZ (compatible with evaluate_cascade_pseudolabels.py)

Usage:
    python mbps_pytorch/generate_depth_guided_instances.py \
        --semantic_dir /data/cityscapes/pseudo_semantic_dinov3/train \
        --depth_dir /data/cityscapes/depth_dav3/train \
        --output_dir /data/cityscapes/pseudo_instance_depth/train \
        --stuff_things /data/cityscapes/pseudo_semantic_dinov3/stuff_things.json
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Cityscapes GT thing class trainIDs (fallback if no stuff_things.json)
DEFAULT_THING_IDS = set(range(11, 19))

# Cityscapes trainID → name mapping (all 19 classes)
CS_NAMES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic_light", 7: "traffic_sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}


def load_thing_ids(stuff_things_path):
    """Load thing class IDs from stuff_things.json.

    Returns set of integer trainIDs classified as things.
    """
    with open(stuff_things_path, "r") as f:
        data = json.load(f)
    thing_ids = set(data["thing_ids"])
    logger.info(f"Loaded {len(thing_ids)} thing IDs from {stuff_things_path}: "
                f"{sorted(thing_ids)} "
                f"({', '.join(CS_NAMES.get(i, str(i)) for i in sorted(thing_ids))})")
    return thing_ids

# Default working resolution (matches Cityscapes depth map native resolution)
# Override via --work_size CLI argument for other datasets (e.g., COCO: 512 512)
WORK_H, WORK_W = 512, 1024


def depth_guided_instances(semantic, depth, thing_ids=DEFAULT_THING_IDS,
                           grad_threshold=0.05, min_area=100,
                           dilation_iters=3, depth_blur_sigma=1.0):
    """Split thing regions using depth gradient edges.

    Args:
        semantic: (H, W) uint8 trainID map
        depth: (H, W) float32 depth map [0, 1]
        thing_ids: set of trainIDs for thing classes
        grad_threshold: depth gradient magnitude threshold for edges
        min_area: minimum pixel area for a valid instance
        dilation_iters: iterations for boundary pixel reclamation
        depth_blur_sigma: Gaussian blur sigma on depth before gradient

    Returns:
        List of (mask, class_id, score) tuples, sorted by area descending.
        Scores normalized to [0, 1].
    """
    # Smooth depth to suppress noise
    if depth_blur_sigma > 0:
        depth_smooth = gaussian_filter(depth.astype(np.float64), sigma=depth_blur_sigma)
    else:
        depth_smooth = depth.astype(np.float64)

    # Depth gradient magnitude (Sobel)
    gx = sobel(depth_smooth, axis=1)
    gy = sobel(depth_smooth, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    depth_edges = grad_mag > grad_threshold

    # Track which pixels are already assigned to an instance
    # (prevents overlapping instances from dilation)
    assigned = np.zeros(semantic.shape, dtype=bool)

    instances = []
    for cls in sorted(thing_ids):
        cls_mask = semantic == cls
        if cls_mask.sum() < min_area:
            continue

        # Remove depth edges from class mask
        split_mask = cls_mask & ~depth_edges

        # Connected components on split mask
        labeled, n_cc = ndimage.label(split_mask)

        # Collect CCs sorted by area (largest first for priority in reclamation)
        cc_list = []
        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = int(cc_mask.sum())
            if area >= min_area:
                cc_list.append((cc_id, cc_mask, area))
        cc_list.sort(key=lambda x: -x[2])

        # Reclaim boundary pixels for each CC
        for cc_id, cc_mask, area in cc_list:
            if dilation_iters > 0:
                dilated = ndimage.binary_dilation(cc_mask, iterations=dilation_iters)
                # Claim pixels that: belong to this class AND not yet assigned
                reclaimed = dilated & cls_mask & ~assigned
                # Include the original CC pixels too
                final_mask = cc_mask | reclaimed
            else:
                final_mask = cc_mask

            final_area = float(final_mask.sum())
            if final_area < min_area:
                continue

            assigned |= final_mask
            instances.append((final_mask, cls, final_area))

    # Sort by area descending, normalize scores to [0, 1]
    instances.sort(key=lambda x: -x[2])
    if instances:
        max_area = instances[0][2]
        instances = [(m, c, s / max_area) for m, c, s in instances]

    return instances


def process_single_image(semantic_path, depth_path, thing_ids=DEFAULT_THING_IDS,
                         grad_threshold=0.05, min_area=100, dilation_iters=3,
                         depth_blur_sigma=1.0):
    """Process one image pair and return instances.

    Returns:
        List of (mask, class_id, score) at WORK_H x WORK_W resolution,
        or empty list if no instances found.
    """
    # Load semantic labels
    semantic_full = np.array(Image.open(semantic_path))
    # Resize to working resolution (nearest neighbor preserves class IDs)
    if semantic_full.shape != (WORK_H, WORK_W):
        semantic = np.array(
            Image.fromarray(semantic_full).resize((WORK_W, WORK_H), Image.NEAREST)
        )
    else:
        semantic = semantic_full

    # Load depth map
    depth = np.load(depth_path)
    if depth.shape != (WORK_H, WORK_W):
        # Resize depth with bilinear interpolation
        depth = np.array(
            Image.fromarray(depth).resize((WORK_W, WORK_H), Image.BILINEAR)
        )

    return depth_guided_instances(
        semantic, depth,
        thing_ids=thing_ids,
        grad_threshold=grad_threshold,
        min_area=min_area,
        dilation_iters=dilation_iters,
        depth_blur_sigma=depth_blur_sigma,
    )


def save_instances(instances, output_path, h=WORK_H, w=WORK_W):
    """Save instances in NPZ format compatible with evaluate_cascade_pseudolabels.py."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not instances:
        # Save empty
        np.savez_compressed(
            str(output_path),
            masks=np.zeros((0, h * w), dtype=bool),
            scores=np.zeros((0,), dtype=np.float32),
            num_valid=0,
            h_patches=h,
            w_patches=w,
        )
        # Visualization
        vis_path = str(output_path).replace(".npz", "_instance.png")
        Image.fromarray(np.zeros((h, w), dtype=np.uint16)).save(vis_path)
        return

    num_instances = len(instances)
    masks = np.zeros((num_instances, h * w), dtype=bool)
    scores = np.zeros(num_instances, dtype=np.float32)

    for i, (mask, cls, score) in enumerate(instances):
        masks[i] = mask.ravel()
        scores[i] = score

    np.savez_compressed(
        str(output_path),
        masks=masks,
        scores=scores,
        num_valid=num_instances,
        h_patches=h,
        w_patches=w,
    )

    # Visualization: instance ID map (0 = background, 1+ = instances)
    vis = np.zeros((h, w), dtype=np.uint16)
    for i, (mask, cls, score) in enumerate(instances):
        vis[mask] = i + 1

    vis_path = str(output_path).replace(".npz", "_instance.png")
    Image.fromarray(vis).save(vis_path)


def find_image_pairs(semantic_dir, depth_dir):
    """Find matching semantic + depth file pairs across city subdirectories."""
    semantic_dir = Path(semantic_dir)
    depth_dir = Path(depth_dir)
    pairs = []

    for sem_path in sorted(semantic_dir.rglob("*.png")):
        stem = sem_path.stem
        # Find matching depth file
        rel = sem_path.relative_to(semantic_dir)
        depth_path = depth_dir / rel.parent / f"{stem}.npy"
        if depth_path.exists():
            pairs.append((sem_path, depth_path))
        else:
            logger.warning(f"No depth map for {sem_path.name}")

    return pairs


def process_dataset(semantic_dir, depth_dir, output_dir, thing_ids=DEFAULT_THING_IDS,
                    grad_threshold=0.05, min_area=100, dilation_iters=3,
                    depth_blur_sigma=1.0, limit=None):
    """Process all images in the dataset."""
    pairs = find_image_pairs(semantic_dir, depth_dir)
    logger.info(f"Found {len(pairs)} semantic+depth pairs")

    if limit is not None:
        pairs = pairs[:limit]
        logger.info(f"Limited to {limit} images")

    output_dir = Path(output_dir)
    total_instances = 0
    instance_counts = []
    per_class_counts = {cls: 0 for cls in sorted(thing_ids)}
    t0 = time.time()

    for sem_path, depth_path in tqdm(pairs, desc="Generating instances"):
        instances = process_single_image(
            sem_path, depth_path,
            thing_ids=thing_ids,
            grad_threshold=grad_threshold,
            min_area=min_area,
            dilation_iters=dilation_iters,
            depth_blur_sigma=depth_blur_sigma,
        )

        # Output path: preserve city subdirectory structure
        rel = sem_path.relative_to(Path(semantic_dir))
        out_path = output_dir / rel.parent / f"{sem_path.stem}.npz"
        save_instances(instances, out_path)

        n = len(instances)
        total_instances += n
        instance_counts.append(n)
        for _, cls, _ in instances:
            per_class_counts[cls] += 1

    elapsed = time.time() - t0
    n_images = len(pairs)
    avg_instances = total_instances / max(n_images, 1)

    # Save stats
    stats = {
        "total_images": n_images,
        "total_instances": total_instances,
        "avg_instances_per_image": round(avg_instances, 2),
        "max_instances_per_image": max(instance_counts) if instance_counts else 0,
        "min_instances_per_image": min(instance_counts) if instance_counts else 0,
        "per_class_counts": {
            CS_NAMES.get(cls, f"class_{cls}"): count
            for cls, count in sorted(per_class_counts.items())
        },
        "config": {
            "grad_threshold": grad_threshold,
            "min_area": min_area,
            "dilation_iters": dilation_iters,
            "depth_blur_sigma": depth_blur_sigma,
        },
        "elapsed_seconds": round(elapsed, 1),
    }
    stats_path = output_dir / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Done in {elapsed:.1f}s ({elapsed/max(n_images,1):.3f}s/image)")
    logger.info(f"Total instances: {total_instances}, avg: {avg_instances:.1f}/image")
    logger.info(f"Per-class: {stats['per_class_counts']}")
    logger.info(f"Stats saved to {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate depth-guided instance pseudo-labels"
    )
    parser.add_argument(
        "--semantic_dir", type=str, required=True,
        help="Path to semantic pseudo-labels (city subdirs with PNGs)",
    )
    parser.add_argument(
        "--depth_dir", type=str, required=True,
        help="Path to depth maps (city subdirs with NPYs)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for NPZ + visualization PNGs",
    )
    parser.add_argument(
        "--grad_threshold", type=float, default=0.05,
        help="Depth gradient magnitude threshold for edge detection (default: 0.05)",
    )
    parser.add_argument(
        "--min_area", type=int, default=100,
        help="Minimum instance area in pixels at 512x1024 (default: 100)",
    )
    parser.add_argument(
        "--dilation_iters", type=int, default=3,
        help="Boundary pixel reclamation dilation iterations (default: 3)",
    )
    parser.add_argument(
        "--depth_blur", type=float, default=1.0,
        help="Gaussian blur sigma on depth before gradient (default: 1.0)",
    )
    parser.add_argument(
        "--stuff_things", type=str, default=None,
        help="Path to stuff_things.json (from classify_stuff_things.py). "
             "If provided, thing_ids are loaded from this file. "
             "If not provided, defaults to Cityscapes GT thing IDs (11-18).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N images (for testing)",
    )
    parser.add_argument(
        "--work_size", type=int, nargs=2, default=None,
        help="Working resolution (H W). Default: 512 1024 for Cityscapes. "
             "Use 512 512 for COCO/COCONUT.",
    )

    args = parser.parse_args()

    # Override working resolution if specified
    global WORK_H, WORK_W
    if args.work_size:
        WORK_H, WORK_W = args.work_size
        logger.info(f"Using custom work size: {WORK_H}x{WORK_W}")

    # Load thing IDs from stuff_things.json or use defaults
    if args.stuff_things:
        thing_ids = load_thing_ids(args.stuff_things)
    else:
        thing_ids = DEFAULT_THING_IDS
        logger.info(f"No --stuff_things provided, using default thing IDs: {sorted(thing_ids)}")

    process_dataset(
        semantic_dir=args.semantic_dir,
        depth_dir=args.depth_dir,
        output_dir=args.output_dir,
        thing_ids=thing_ids,
        grad_threshold=args.grad_threshold,
        min_area=args.min_area,
        dilation_iters=args.dilation_iters,
        depth_blur_sigma=args.depth_blur,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
