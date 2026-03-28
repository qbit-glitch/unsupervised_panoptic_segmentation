#!/usr/bin/env python3
"""Generate instance pseudo-labels via adaptive depth-layered decomposition.

Replaces brittle Sobel gradient thresholding with principled depth quantization.
Depth is discretized into adaptive quantile-based bins, then connected components
are run per (class, depth_bin) pair. Adjacent-bin instances with high feature
similarity can optionally be merged.

Algorithm per image:
  1. Load CAUSE-TR semantic map (27-class), map to 19 trainIDs
  2. Load depth map (SPIdepth or DAv3)
  3. Compute adaptive depth bins from thing-class pixel depths (quantile-based)
  4. For each thing class, for each depth bin:
     - Compute (class_mask & depth_bin_mask) -> connected components
     - Keep CCs with area >= min_area
  5. Optionally merge instances across adjacent depth bins if spatially
     overlapping and DINOv3 feature-similar
  6. Save instances as NPZ

Usage:
    python mbps_pytorch/generate_depth_layer_instances.py \
        --cityscapes_root /path/to/cityscapes \
        --split val --cause27 --num_bins 15

    # With cross-bin merging using DINOv3 features:
    python mbps_pytorch/generate_depth_layer_instances.py \
        --cityscapes_root /path/to/cityscapes \
        --split val --cause27 --merge_adjacent \
        --feature_subdir dinov3_features
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
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Constants ───

WORK_H, WORK_W = 512, 1024
IGNORE_LABEL = 255

CS_NAMES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic_light", 7: "traffic_sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}

THING_IDS = set(range(11, 19))

# CAUSE 27-class → 19 trainID mapping
_CAUSE27_TO_TRAINID = np.full(256, IGNORE_LABEL, dtype=np.uint8)
for _c27, _t19 in {
    0: 0, 1: 1, 2: 255, 3: 255, 4: 2, 5: 3, 6: 4,
    7: 255, 8: 255, 9: 255, 10: 5, 11: 5, 12: 6, 13: 7,
    14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14,
    21: 15, 22: 13, 23: 14, 24: 16, 25: 17, 26: 18,
}.items():
    _CAUSE27_TO_TRAINID[_c27] = _t19


def depth_layer_instances(semantic_19, depth, thing_ids=THING_IDS,
                          num_bins=15, min_area=500):
    """Split thing regions using adaptive depth quantile binning.

    Args:
        semantic_19: (H, W) uint8, Cityscapes 19-class trainIDs
        depth: (H, W) float32, monocular depth
        thing_ids: set of thing trainIDs
        num_bins: number of depth quantile bins
        min_area: minimum instance pixel area

    Returns:
        list of (mask, class_id, score) tuples, sorted by area descending
    """
    h, w = semantic_19.shape

    # Build thing mask and get depth values for adaptive binning
    thing_mask = np.isin(semantic_19, list(thing_ids))
    thing_depths = depth[thing_mask]

    if len(thing_depths) < 100:
        # Not enough thing pixels — fall back to simple CC per class
        instances = []
        for cls in sorted(thing_ids):
            cls_mask = semantic_19 == cls
            if cls_mask.sum() < min_area:
                continue
            labeled, n_cc = ndimage.label(cls_mask)
            for cc_id in range(1, n_cc + 1):
                cc_mask = labeled == cc_id
                area = cc_mask.sum()
                if area >= min_area:
                    instances.append((cc_mask, cls, float(area)))
        return instances

    # Compute adaptive depth bins (quantile-based)
    bin_edges = np.quantile(thing_depths, np.linspace(0, 1, num_bins + 1))
    bin_edges = np.unique(bin_edges)  # remove duplicates from flat depth regions
    n_actual_bins = len(bin_edges) - 1

    if n_actual_bins < 2:
        # Depth is nearly constant — fall back to simple CC
        instances = []
        for cls in sorted(thing_ids):
            cls_mask = semantic_19 == cls
            if cls_mask.sum() < min_area:
                continue
            labeled, n_cc = ndimage.label(cls_mask)
            for cc_id in range(1, n_cc + 1):
                cc_mask = labeled == cc_id
                area = cc_mask.sum()
                if area >= min_area:
                    instances.append((cc_mask, cls, float(area)))
        return instances

    # Assign each pixel to a depth bin
    bin_map = np.digitize(depth, bin_edges) - 1  # 0-indexed
    bin_map = np.clip(bin_map, 0, n_actual_bins - 1)

    # Per (class, bin) connected components
    instances = []
    for cls in sorted(thing_ids):
        cls_mask = semantic_19 == cls
        if cls_mask.sum() < min_area:
            continue
        for b in range(n_actual_bins):
            region = cls_mask & (bin_map == b)
            if region.sum() < min_area:
                continue
            labeled, n_cc = ndimage.label(region)
            for cc_id in range(1, n_cc + 1):
                cc_mask = labeled == cc_id
                area = cc_mask.sum()
                if area >= min_area:
                    instances.append((cc_mask, cls, float(area)))

    return instances


def merge_adjacent_bins(instances, features=None, merge_sim_thresh=0.85,
                        grid_h=32, grid_w=64):
    """Merge instances from adjacent depth bins if spatially overlapping
    and feature-similar.

    Uses union-find to merge compatible instances.
    """
    if features is None or len(instances) < 2:
        return instances

    from sklearn.preprocessing import normalize as l2_normalize

    n = len(instances)
    patch_h = WORK_H // grid_h
    patch_w = WORK_W // grid_w

    # Precompute mean feature per instance
    feat_f32 = features.astype(np.float32)
    feat_norm = l2_normalize(feat_f32, axis=1)  # (n_patches, dim)

    inst_feats = []
    for mask, cls, score in instances:
        # Get patches overlapping with this instance
        patch_indices = []
        for r in range(grid_h):
            for c in range(grid_w):
                patch_region = mask[r * patch_h:(r + 1) * patch_h,
                                   c * patch_w:(c + 1) * patch_w]
                if patch_region.any():
                    patch_indices.append(r * grid_w + c)
        if patch_indices:
            inst_feats.append(feat_norm[patch_indices].mean(axis=0))
        else:
            inst_feats.append(np.zeros(feat_f32.shape[1], dtype=np.float32))

    inst_feats = np.array(inst_feats)
    inst_feats = l2_normalize(inst_feats, axis=1)

    # Union-find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Check all pairs of same-class instances for merging
    struct = ndimage.generate_binary_structure(2, 1)  # 4-connected
    for i in range(n):
        mask_i, cls_i, _ = instances[i]
        dilated_i = ndimage.binary_dilation(mask_i, struct, iterations=2)
        for j in range(i + 1, n):
            mask_j, cls_j, _ = instances[j]
            if cls_i != cls_j:
                continue
            # Check spatial adjacency
            if not (dilated_i & mask_j).any():
                continue
            # Check feature similarity
            sim = np.dot(inst_feats[i], inst_feats[j])
            if sim > merge_sim_thresh:
                union(i, j)

    # Group by root
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # Merge instances in each group
    merged = []
    for root, members in groups.items():
        mask_union = np.zeros((WORK_H, WORK_W), dtype=bool)
        cls = instances[members[0]][1]
        for idx in members:
            mask_union |= instances[idx][0]
        area = float(mask_union.sum())
        merged.append((mask_union, cls, area))

    return merged


def process_single_image(semantic_path, depth_path, thing_ids,
                         cause27=False, num_bins=15, min_area=500,
                         merge_adjacent=False, feature_path=None,
                         merge_sim_thresh=0.85):
    """Process one image and return instances.

    Returns:
        masks: (N, H, W) bool
        scores: (N,) float32
        boxes: (N, 4) float32
    """
    # Load semantic map
    sem_img = Image.open(semantic_path)
    sem_np = np.array(sem_img.resize((WORK_W, WORK_H), Image.NEAREST), dtype=np.uint8)

    if cause27:
        sem_19 = _CAUSE27_TO_TRAINID[sem_np]
    else:
        sem_19 = sem_np

    # Load depth map
    depth = np.load(depth_path).astype(np.float32)
    if depth.shape != (WORK_H, WORK_W):
        depth_img = Image.fromarray(depth)
        depth = np.array(depth_img.resize((WORK_W, WORK_H), Image.BILINEAR), dtype=np.float32)

    # Normalize depth to [0, 1]
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        depth = (depth - d_min) / (d_max - d_min)

    # Generate instances
    instances = depth_layer_instances(
        sem_19, depth, thing_ids=thing_ids,
        num_bins=num_bins, min_area=min_area,
    )

    # Optional merge across adjacent bins
    if merge_adjacent and feature_path is not None and os.path.exists(feature_path):
        features = np.load(feature_path).astype(np.float32)
        instances = merge_adjacent_bins(
            instances, features=features,
            merge_sim_thresh=merge_sim_thresh,
        )

    if not instances:
        return (np.zeros((0, WORK_H, WORK_W), dtype=bool),
                np.array([], dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32))

    # Sort by area descending
    instances.sort(key=lambda x: x[2], reverse=True)

    masks = np.stack([inst[0] for inst in instances], axis=0)
    areas = np.array([inst[2] for inst in instances], dtype=np.float32)
    max_area = areas.max() if len(areas) > 0 else 1.0
    scores = areas / max_area

    # Compute bounding boxes
    boxes = np.zeros((len(masks), 4), dtype=np.float32)
    for i, m in enumerate(masks):
        ys, xs = np.where(m)
        if len(ys) > 0:
            boxes[i] = [xs.min(), ys.min(), xs.max(), ys.max()]

    return masks, scores, boxes


def main():
    parser = argparse.ArgumentParser("Adaptive depth-layer instance segmentation")
    parser.add_argument("--cityscapes_root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--semantic_subdir", default="pseudo_semantic_cause")
    parser.add_argument("--depth_subdir", default="depth_spidepth")
    parser.add_argument("--feature_subdir", default=None,
                        help="DINOv3 features for cross-bin merging")
    parser.add_argument("--output_dir", default="pseudo_instances_depth_layer")
    parser.add_argument("--num_bins", type=int, default=15)
    parser.add_argument("--min_area", type=int, default=500)
    parser.add_argument("--cause27", action="store_true")
    parser.add_argument("--merge_adjacent", action="store_true",
                        help="Merge instances across adjacent bins using feature similarity")
    parser.add_argument("--merge_sim_thresh", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    root = args.cityscapes_root
    sem_dir = os.path.join(root, args.semantic_subdir, args.split)
    depth_dir = os.path.join(root, args.depth_subdir, args.split)
    out_dir = os.path.join(root, args.output_dir, args.split)
    feat_dir = (os.path.join(root, args.feature_subdir, args.split)
                if args.feature_subdir else None)

    thing_ids = THING_IDS

    # Discover images
    image_list = []
    for city in sorted(os.listdir(sem_dir)):
        city_sem = os.path.join(sem_dir, city)
        if not os.path.isdir(city_sem):
            continue
        for fname in sorted(os.listdir(city_sem)):
            if not fname.endswith(".png"):
                continue
            stem = fname.replace(".png", "")
            sem_path = os.path.join(city_sem, fname)
            depth_path = os.path.join(depth_dir, city, stem + ".npy")
            feat_path = None
            if feat_dir:
                # Try with _leftImg8bit suffix first (DINOv2 naming)
                feat_path_try = os.path.join(feat_dir, city, stem + "_leftImg8bit.npy")
                if os.path.exists(feat_path_try):
                    feat_path = feat_path_try
                else:
                    feat_path = os.path.join(feat_dir, city, stem + ".npy")
            if not os.path.exists(depth_path):
                continue
            image_list.append((city, stem, sem_path, depth_path, feat_path))

    if args.limit:
        image_list = image_list[:args.limit]

    logger.info(f"Processing {len(image_list)} images, num_bins={args.num_bins}, "
                f"min_area={args.min_area}, merge={args.merge_adjacent}")

    os.makedirs(out_dir, exist_ok=True)
    total_instances = 0
    per_class_counts = {cls: 0 for cls in sorted(thing_ids)}
    t0 = time.time()

    for city, stem, sem_path, depth_path, feat_path in tqdm(image_list, desc="Depth layers"):
        masks, scores, boxes = process_single_image(
            sem_path, depth_path, thing_ids,
            cause27=args.cause27,
            num_bins=args.num_bins,
            min_area=args.min_area,
            merge_adjacent=args.merge_adjacent,
            feature_path=feat_path,
            merge_sim_thresh=args.merge_sim_thresh,
        )

        # Save NPZ
        city_out = os.path.join(out_dir, city)
        os.makedirs(city_out, exist_ok=True)
        np.savez_compressed(
            os.path.join(city_out, stem + ".npz"),
            masks=masks, scores=scores, boxes=boxes,
            num_valid=len(masks),
        )
        total_instances += len(masks)

        # Optional visualization
        if args.visualize and len(masks) > 0:
            vis = np.zeros((WORK_H, WORK_W, 3), dtype=np.uint8)
            rng = np.random.RandomState(42)
            colors = rng.randint(50, 255, size=(len(masks), 3))
            for i in range(len(masks)):
                vis[masks[i]] = colors[i]
            vis_img = Image.fromarray(vis)
            vis_img.save(os.path.join(city_out, stem + "_vis.png"))

    elapsed = time.time() - t0
    n = len(image_list)
    avg = total_instances / max(n, 1)
    logger.info(f"Done. {total_instances} instances from {n} images "
                f"({avg:.1f} avg/img, {elapsed:.1f}s, {n / max(elapsed, 0.01):.1f} img/s)")

    # Save stats
    stats = {
        "split": args.split,
        "num_images": n,
        "total_instances": total_instances,
        "avg_instances_per_image": round(avg, 2),
        "num_bins": args.num_bins,
        "min_area": args.min_area,
        "merge_adjacent": args.merge_adjacent,
        "elapsed_s": round(elapsed, 1),
    }
    stats_path = os.path.join(root, args.output_dir, f"stats_{args.split}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
