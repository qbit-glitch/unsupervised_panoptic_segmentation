#!/usr/bin/env python3
"""Generate instance pseudo-labels via DINOv2 intra-class feature clustering.

Uses pixel-level connected components on the semantic map for initial instance
separation, then selectively applies HDBSCAN on DINOv2 patch features to split
large regions that likely contain multiple merged objects.

Algorithm per image:
  1. Load CAUSE-TR semantic map (27-class), map to 19 trainIDs
  2. Load pre-extracted DINOv3 ViT-B/16 patch features (32x64 grid, 768-dim)
  3. For each thing class:
     a. Pixel-level connected components on semantic_19 == cls
     b. Small CCs (< split_threshold): keep as single instance
     c. Large CCs (>= split_threshold): cluster overlapping patches with
        HDBSCAN to split merged objects, assign pixels by patch cluster
  4. Save instances as NPZ (compatible with evaluate_cascade_pseudolabels.py)

Usage:
    # Quick test (3 images):
    python mbps_pytorch/generate_dino_cluster_instances.py \
        --cityscapes_root /path/to/cityscapes \
        --split val --cause27 --limit 3 --visualize

    # Full val set:
    python mbps_pytorch/generate_dino_cluster_instances.py \
        --cityscapes_root /path/to/cityscapes \
        --split val --cause27

    # With depth conditioning:
    python mbps_pytorch/generate_dino_cluster_instances.py \
        --cityscapes_root /path/to/cityscapes \
        --split val --cause27 --depth_subdir depth_spidepth
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
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Constants ───

WORK_H, WORK_W = 512, 1024
IGNORE_LABEL = 255

# Cityscapes trainID names
CS_NAMES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic_light", 7: "traffic_sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}

# Standard Cityscapes thing trainIDs
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


# ─── Grid Inference ───

def infer_grid(n_patches):
    """Infer (grid_h, grid_w) from total patch count assuming ~1:2 aspect ratio."""
    for gh, gw in [(32, 64), (48, 96), (64, 128), (128, 256), (256, 512)]:
        if gh * gw == n_patches:
            return gh, gw
    # Fallback: try to factor as ~1:2 aspect ratio
    for gh in range(16, 512):
        gw = n_patches // gh
        if gh * gw == n_patches and abs(gw / gh - 2.0) < 0.1:
            return gh, gw
    raise ValueError(f"Cannot infer grid from {n_patches} patches")


def patches_to_pixel_mask(patch_indices, grid_w, patch_h, patch_w):
    """Convert flat patch indices to a pixel-level boolean mask."""
    mask = np.zeros((WORK_H, WORK_W), dtype=bool)
    for idx in patch_indices:
        r, c = divmod(idx, grid_w)
        mask[r * patch_h:(r + 1) * patch_h,
             c * patch_w:(c + 1) * patch_w] = True
    return mask


# ─── Per-Class Algorithm ───

def get_patch_features_for_cc(cc_pixel_mask, feat_norm, grid_h, grid_w,
                               patch_h, patch_w, depth=None, depth_weight=2.0):
    """Extract patch features for patches that overlap a pixel-level CC.

    Args:
        cc_pixel_mask: (WORK_H, WORK_W) bool — pixel mask of connected component
        feat_norm: (grid_h, grid_w, feat_dim) float32 L2-normalized features
        grid_h, grid_w: patch grid dimensions
        patch_h, patch_w: pixel size of each patch
        depth: (WORK_H, WORK_W) float32 normalized depth, or None
        depth_weight: weight for depth feature

    Returns:
        patch_indices: (K,) int — flat indices into (grid_h, grid_w) grid
        X: (K, D) float32 — feature matrix for clustering
    """
    feat_dim = feat_norm.shape[2]

    # Find patches that have significant overlap with the CC
    patch_overlap = np.zeros((grid_h, grid_w), dtype=np.float32)
    for r in range(grid_h):
        for c in range(grid_w):
            block = cc_pixel_mask[r * patch_h:(r + 1) * patch_h,
                                   c * patch_w:(c + 1) * patch_w]
            patch_overlap[r, c] = block.sum() / (patch_h * patch_w)

    # Only include patches with >= 20% overlap with the CC
    valid_patches = patch_overlap >= 0.2
    if valid_patches.sum() == 0:
        return np.array([], dtype=int), np.zeros((0, feat_dim), dtype=np.float32)

    rows, cols = np.where(valid_patches)
    patch_indices = rows * grid_w + cols

    # Extract features
    F = feat_norm[rows, cols]  # (K, feat_dim)
    parts = [F]

    # Add depth if available
    if depth is not None:
        patch_depth = np.zeros(len(rows), dtype=np.float32)
        for i, (r, c) in enumerate(zip(rows, cols)):
            block = depth[r * patch_h:(r + 1) * patch_h,
                          c * patch_w:(c + 1) * patch_w]
            cc_block = cc_pixel_mask[r * patch_h:(r + 1) * patch_h,
                                      c * patch_w:(c + 1) * patch_w]
            if cc_block.any():
                patch_depth[i] = block[cc_block].mean()
            else:
                patch_depth[i] = block.mean()
        parts.append(depth_weight * patch_depth.reshape(-1, 1))

    X = np.hstack(parts).astype(np.float32)
    return patch_indices, X


def assign_pixels_to_clusters(cc_pixel_mask, patch_indices, labels,
                               grid_h, grid_w, patch_h, patch_w):
    """Assign each pixel in a CC to the cluster of its overlapping patch.

    Args:
        cc_pixel_mask: (WORK_H, WORK_W) bool — pixel mask of CC
        patch_indices: (K,) int — flat indices of clustered patches
        labels: (K,) int — cluster label per patch
        grid_h, grid_w: patch grid dimensions
        patch_h, patch_w: pixel size of each patch

    Returns:
        Dict[int, ndarray] — cluster_label → pixel mask (WORK_H, WORK_W) bool
    """
    # Build a label map at patch level
    patch_label_map = np.full((grid_h, grid_w), -1, dtype=np.int32)
    for idx, lbl in zip(patch_indices, labels):
        r = idx // grid_w
        c = idx % grid_w
        patch_label_map[r, c] = lbl

    # Assign each pixel to its patch's cluster
    cluster_masks = {}
    pixel_ys, pixel_xs = np.where(cc_pixel_mask)

    if len(pixel_ys) == 0:
        return cluster_masks

    # Vectorized: compute patch row/col for all pixels at once
    patch_rs = pixel_ys // patch_h
    patch_cs = pixel_xs // patch_w

    # Clip to grid bounds
    patch_rs = np.clip(patch_rs, 0, grid_h - 1)
    patch_cs = np.clip(patch_cs, 0, grid_w - 1)

    pixel_labels = patch_label_map[patch_rs, patch_cs]

    for lbl in np.unique(pixel_labels):
        if lbl < 0:
            continue
        mask = np.zeros((WORK_H, WORK_W), dtype=bool)
        sel = pixel_labels == lbl
        mask[pixel_ys[sel], pixel_xs[sel]] = True
        cluster_masks[lbl] = mask

    return cluster_masks


def dino_cluster_instances(semantic_19, features, depth=None,
                           thing_ids=THING_IDS,
                           depth_weight=2.0,
                           min_cluster_size=15, min_area=500,
                           split_threshold=3000):
    """Find instances via pixel-level CCs + selective DINOv2 feature clustering.

    For each thing class:
      1. Pixel-level connected components on semantic_19 == cls
      2. Small CCs (< split_threshold pixels): keep as single instance
      3. Large CCs (>= split_threshold): cluster overlapping patches with HDBSCAN
         to split merged objects; assign pixels by their patch's cluster label

    Args:
        semantic_19: (WORK_H, WORK_W) uint8 trainID map (0-18, 255=ignore)
        features: (n_patches, feat_dim) float32 patch features
        depth: (WORK_H, WORK_W) float32 depth map normalized [0,1], or None
        thing_ids: set of trainIDs to process as thing classes
        depth_weight: weight for normalized mean depth per patch
        min_cluster_size: HDBSCAN min_cluster_size parameter (higher = less fragmentation)
        min_area: minimum pixel area for a valid instance
        split_threshold: only attempt clustering on CCs with more pixels than this

    Returns:
        List of (mask, class_id, score) tuples sorted by area descending.
    """
    n_patches_total, feat_dim = features.shape
    grid_h, grid_w = infer_grid(n_patches_total)

    patch_h = WORK_H // grid_h
    patch_w = WORK_W // grid_w
    logger.debug(f"Grid: {grid_h}x{grid_w}, patch: {patch_h}x{patch_w}px, feat_dim: {feat_dim}")

    # Reshape and normalize features
    feat_norm = normalize(features, norm="l2", axis=1).reshape(grid_h, grid_w, feat_dim)

    instances = []

    for cls in sorted(thing_ids):
        cls_pixel_mask = (semantic_19 == cls)
        if cls_pixel_mask.sum() < min_area:
            continue

        # Pixel-level connected components
        labeled, n_cc = ndimage.label(cls_pixel_mask)

        for cc_id in range(1, n_cc + 1):
            cc_mask = (labeled == cc_id)
            area = float(cc_mask.sum())

            if area < min_area:
                continue

            # Small CC → single instance (no splitting needed)
            if area < split_threshold:
                instances.append((cc_mask, cls, area))
                continue

            # Large CC → try feature clustering to split merged objects
            patch_indices, X = get_patch_features_for_cc(
                cc_mask, feat_norm, grid_h, grid_w, patch_h, patch_w,
                depth, depth_weight
            )
            n_patches = len(patch_indices)

            # If too few patches for clustering, keep as single instance
            if n_patches < min_cluster_size * 2:
                instances.append((cc_mask, cls, area))
                continue

            # Adaptive min_cluster_size: at least min_cluster_size, at most n/3
            effective_mcs = max(min_cluster_size, n_patches // 3)

            clusterer = HDBSCAN(
                min_cluster_size=effective_mcs,
                min_samples=2,
                metric="euclidean",
                n_jobs=1,
            )
            labels = clusterer.fit_predict(X)

            # Assign noise points to nearest valid cluster
            noise_mask = labels == -1
            if noise_mask.any() and not noise_mask.all():
                from sklearn.neighbors import NearestNeighbors
                valid_idx = np.where(~noise_mask)[0]
                noise_idx = np.where(noise_mask)[0]
                nn = NearestNeighbors(n_neighbors=1).fit(X[valid_idx])
                _, nearest = nn.kneighbors(X[noise_idx])
                labels[noise_idx] = labels[valid_idx[nearest.ravel()]]

            # Count actual clusters
            unique_labels = np.unique(labels[labels >= 0])

            # If only 1 cluster (or all noise), keep as single instance
            if len(unique_labels) <= 1:
                instances.append((cc_mask, cls, area))
                continue

            # Split: assign pixels to cluster by their patch's label
            cluster_masks = assign_pixels_to_clusters(
                cc_mask, patch_indices, labels,
                grid_h, grid_w, patch_h, patch_w
            )

            for lbl, sub_mask in cluster_masks.items():
                sub_area = float(sub_mask.sum())
                if sub_area >= min_area:
                    instances.append((sub_mask, cls, sub_area))

    # Sort by area descending, normalize scores to [0, 1]
    instances.sort(key=lambda x: -x[2])
    if instances:
        max_area = instances[0][2]
        instances = [(m, c, a / max_area) for m, c, a in instances]

    return instances


# ─── Global Clustering Algorithm ───

def global_cluster_instances(features, semantic_19=None, depth=None,
                              thing_ids=THING_IDS,
                              spatial_weight=2.0, depth_weight=1.0,
                              min_cluster_size=20, min_area=500):
    """Discover instances by clustering ALL patches globally (class-agnostic).

    Instead of working per-class on semantic CCs, cluster all patches at once
    so that adjacent same-class objects (e.g. two cars) get separate clusters
    based on feature similarity.

    Args:
        features: (n_patches, feat_dim) float32 patch features
        semantic_19: (WORK_H, WORK_W) uint8 trainID map for classification
        depth: (WORK_H, WORK_W) float32 depth map normalized [0,1], or None
        thing_ids: set of trainIDs to keep as thing instances
        spatial_weight: weight for normalized spatial coordinates
        depth_weight: weight for depth features
        min_cluster_size: HDBSCAN min_cluster_size
        min_area: minimum pixel area for a valid instance

    Returns:
        List of (mask, class_id, score) tuples sorted by area descending.
    """
    n_patches_total, feat_dim = features.shape
    grid_h, grid_w = infer_grid(n_patches_total)
    patch_h = WORK_H // grid_h
    patch_w = WORK_W // grid_w
    logger.debug(f"Global mode — Grid: {grid_h}x{grid_w}, patch: {patch_h}x{patch_w}px")

    # L2-normalize features
    feat_norm = normalize(features, norm="l2", axis=1)

    # Spatial coordinates normalized to [0, 1]
    indices = np.arange(n_patches_total)
    rows = indices // grid_w
    cols = indices % grid_w
    spatial = np.stack([rows / grid_h, cols / grid_w], axis=1).astype(np.float32)

    # Build feature matrix
    parts = [feat_norm]
    if spatial_weight > 0:
        parts.append(spatial_weight * spatial)

    if depth is not None and depth_weight > 0:
        # Compute mean depth per patch
        patch_depth = np.zeros(n_patches_total, dtype=np.float32)
        for i in range(n_patches_total):
            r, c = divmod(i, grid_w)
            block = depth[r * patch_h:(r + 1) * patch_h,
                          c * patch_w:(c + 1) * patch_w]
            patch_depth[i] = block.mean()
        # Z-score normalize
        d_mean, d_std = patch_depth.mean(), patch_depth.std()
        if d_std > 1e-6:
            patch_depth = (patch_depth - d_mean) / d_std
        parts.append(depth_weight * patch_depth.reshape(-1, 1))

    X = np.hstack(parts).astype(np.float32)

    # HDBSCAN clustering
    effective_min_samples = max(2, min_cluster_size // 3)
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=effective_min_samples,
        metric="euclidean",
        n_jobs=1,
    )
    labels = clusterer.fit_predict(X)

    # Collect instances from each cluster
    instances = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # skip noise

    for lbl in unique_labels:
        patch_idx = np.where(labels == lbl)[0]

        # Build pixel mask from patches
        raw_mask = patches_to_pixel_mask(patch_idx, grid_w, patch_h, patch_w)

        # Split spatially disjoint parts
        labeled_ccs, n_ccs = ndimage.label(raw_mask)

        for cc_id in range(1, n_ccs + 1):
            cc_mask = (labeled_ccs == cc_id)

            # Classify via majority vote from semantic map
            if semantic_19 is not None:
                class_pixels = semantic_19[cc_mask]
                class_pixels = class_pixels[class_pixels != IGNORE_LABEL]
                if len(class_pixels) == 0:
                    continue
                cls = int(np.bincount(class_pixels).argmax())

                # Only keep thing classes
                if cls not in thing_ids:
                    continue

                # Intersect with semantic class for pixel-level boundary refinement
                refined_mask = cc_mask & (semantic_19 == cls)
                area = float(refined_mask.sum())
                if area < min_area:
                    continue
                instances.append((refined_mask, cls, area))
            else:
                area = float(cc_mask.sum())
                if area < min_area:
                    continue
                instances.append((cc_mask, 0, area))

    # Sort by area descending, normalize scores to [0, 1]
    instances.sort(key=lambda x: -x[2])
    if instances:
        max_area = instances[0][2]
        instances = [(m, c, a / max_area) for m, c, a in instances]

    return instances


# ─── Feature-CC Algorithm ───

def feature_cc_instances(features, semantic_19=None, depth=None,
                          thing_ids=THING_IDS,
                          sim_threshold=0.6, depth_weight=1.0,
                          min_area=500):
    """Discover instances via feature similarity edges + connected components.

    Build a 4-connected adjacency graph on the patch grid. Cut edges where
    cosine similarity between neighboring patches falls below sim_threshold.
    Run connected components on the remaining graph. Each CC is a candidate
    instance, classified via majority vote from the semantic map.

    Args:
        features: (n_patches, feat_dim) float32 patch features
        semantic_19: (WORK_H, WORK_W) uint8 trainID map for classification
        depth: (WORK_H, WORK_W) float32 depth map normalized [0,1], or None
        thing_ids: set of trainIDs to keep as thing instances
        sim_threshold: cosine similarity threshold — edges below this are cut
        depth_weight: if > 0 and depth provided, also cut edges with large
                      depth discontinuities (|d1-d2| > 1/depth_weight)
        min_area: minimum pixel area for a valid instance

    Returns:
        List of (mask, class_id, score) tuples sorted by area descending.
    """
    n_patches_total, feat_dim = features.shape
    grid_h, grid_w = infer_grid(n_patches_total)
    patch_h = WORK_H // grid_h
    patch_w = WORK_W // grid_w
    logger.debug(f"Feature-CC mode — Grid: {grid_h}x{grid_w}, "
                 f"patch: {patch_h}x{patch_w}px, sim_thresh: {sim_threshold}")

    # L2-normalize features → cosine similarity = dot product
    feat_norm = normalize(features, norm="l2", axis=1).reshape(grid_h, grid_w, feat_dim)

    # Compute mean depth per patch (optional)
    patch_depth = None
    if depth is not None and depth_weight > 0:
        patch_depth = np.zeros((grid_h, grid_w), dtype=np.float32)
        for r in range(grid_h):
            for c in range(grid_w):
                patch_depth[r, c] = depth[r * patch_h:(r + 1) * patch_h,
                                          c * patch_w:(c + 1) * patch_w].mean()

    # Build adjacency: label map via flood-fill on similarity graph
    # Use scipy.ndimage.label with a custom structure won't work directly,
    # so we build an explicit connected-component via union-find.
    parent = np.arange(grid_h * grid_w, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    depth_thresh = 1.0 / depth_weight if depth_weight > 0 else float("inf")

    # Check horizontal edges (right neighbor)
    for r in range(grid_h):
        for c in range(grid_w - 1):
            sim = np.dot(feat_norm[r, c], feat_norm[r, c + 1])
            if sim >= sim_threshold:
                # Check depth discontinuity
                if patch_depth is not None:
                    if abs(patch_depth[r, c] - patch_depth[r, c + 1]) > depth_thresh:
                        continue
                union(r * grid_w + c, r * grid_w + c + 1)

    # Check vertical edges (bottom neighbor)
    for r in range(grid_h - 1):
        for c in range(grid_w):
            sim = np.dot(feat_norm[r, c], feat_norm[r + 1, c])
            if sim >= sim_threshold:
                if patch_depth is not None:
                    if abs(patch_depth[r, c] - patch_depth[r + 1, c]) > depth_thresh:
                        continue
                union(r * grid_w + c, (r + 1) * grid_w + c)

    # Collect components
    from collections import defaultdict
    components = defaultdict(list)
    for idx in range(grid_h * grid_w):
        components[find(idx)].append(idx)

    # Build instances from components
    instances = []
    for root, patch_indices in components.items():
        # Build pixel mask
        raw_mask = patches_to_pixel_mask(patch_indices, grid_w, patch_h, patch_w)

        # Classify via majority vote from semantic map
        if semantic_19 is not None:
            class_pixels = semantic_19[raw_mask]
            class_pixels = class_pixels[class_pixels != IGNORE_LABEL]
            if len(class_pixels) == 0:
                continue
            cls = int(np.bincount(class_pixels).argmax())

            # Only keep thing classes
            if cls not in thing_ids:
                continue

            # Intersect with semantic class for boundary refinement
            refined_mask = raw_mask & (semantic_19 == cls)

            # Split into pixel-level CCs (a single patch-CC might span
            # disjoint semantic regions after intersection)
            labeled_ccs, n_ccs = ndimage.label(refined_mask)
            for cc_id in range(1, n_ccs + 1):
                cc_mask = (labeled_ccs == cc_id)
                area = float(cc_mask.sum())
                if area >= min_area:
                    instances.append((cc_mask, cls, area))
        else:
            area = float(raw_mask.sum())
            if area >= min_area:
                instances.append((raw_mask, 0, area))

    # Sort by area descending, normalize scores to [0, 1]
    instances.sort(key=lambda x: -x[2])
    if instances:
        max_area = instances[0][2]
        instances = [(m, c, a / max_area) for m, c, a in instances]

    return instances


# ─── I/O ───

def process_single_image(semantic_path, feature_path, depth_path=None,
                         cause27=False, global_mode=False, feature_cc=False,
                         **kwargs):
    """Process a single image and return instances.

    Returns:
        List of (mask, class_id, score) tuples at WORK_H x WORK_W resolution.
    """
    # Load semantic map, downsample to working resolution
    semantic_full = np.array(Image.open(semantic_path))
    if semantic_full.shape != (WORK_H, WORK_W):
        semantic = np.array(
            Image.fromarray(semantic_full).resize((WORK_W, WORK_H), Image.NEAREST)
        )
    else:
        semantic = semantic_full

    # Map CAUSE 27-class to 19 trainIDs
    if cause27:
        semantic = _CAUSE27_TO_TRAINID[semantic]

    # Load features — shape (n_patches, feat_dim), grid inferred automatically
    features = np.load(feature_path).astype(np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {features.shape}")

    # Load depth (optional)
    depth = None
    if depth_path is not None and Path(depth_path).exists():
        depth = np.load(depth_path).astype(np.float32)
        if depth.shape != (WORK_H, WORK_W):
            depth = np.array(
                Image.fromarray(depth).resize((WORK_W, WORK_H), Image.BILINEAR)
            )
        # Normalize to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

    if feature_cc:
        return feature_cc_instances(features, semantic, depth, **kwargs)
    elif global_mode:
        return global_cluster_instances(features, semantic, depth, **kwargs)
    else:
        return dino_cluster_instances(semantic, features, depth, **kwargs)


def save_instances(instances, output_path, h=WORK_H, w=WORK_W):
    """Save instances as NPZ in 3D mask format (N, H, W)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not instances:
        np.savez_compressed(
            str(output_path),
            masks=np.zeros((0, h, w), dtype=bool),
            scores=np.zeros(0, dtype=np.float32),
            num_valid=0,
        )
        return

    N = len(instances)
    masks = np.zeros((N, h, w), dtype=bool)
    scores = np.zeros(N, dtype=np.float32)

    for i, (mask, cls, score) in enumerate(instances):
        masks[i] = mask
        scores[i] = score

    np.savez_compressed(
        str(output_path),
        masks=masks,
        scores=scores,
        num_valid=N,
    )


def visualize_instances(img_path, instances, save_path, h=WORK_H, w=WORK_W):
    """Save overlay visualization of clustered instances."""
    img = np.array(Image.open(img_path).resize((w, h)))
    vis = img.astype(float)

    if not instances:
        Image.fromarray(img).save(save_path)
        return

    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(len(instances), 3))

    for i, (mask, cls, score) in enumerate(instances):
        vis[mask] = vis[mask] * 0.5 + colors[i] * 0.5

    Image.fromarray(np.clip(vis, 0, 255).astype(np.uint8)).save(save_path)


def find_triples(cityscapes_root, split, semantic_subdir, feature_subdir,
                 depth_subdir=None):
    """Find matching (semantic, feature, depth) file triples across cities."""
    sem_dir = Path(cityscapes_root) / semantic_subdir / split
    feat_dir = Path(cityscapes_root) / feature_subdir / split
    depth_dir = Path(cityscapes_root) / depth_subdir / split if depth_subdir else None
    img_dir = Path(cityscapes_root) / "leftImg8bit" / split

    triples = []
    for sem_path in sorted(sem_dir.rglob("*.png")):
        stem = sem_path.stem
        city = sem_path.parent.name

        # Feature files may have _leftImg8bit suffix
        feat_path = feat_dir / city / f"{stem}.npy"
        if not feat_path.exists():
            feat_path = feat_dir / city / f"{stem}_leftImg8bit.npy"
        if not feat_path.exists():
            logger.warning(f"No features for {stem}")
            continue

        dep_path = None
        if depth_dir:
            dep_path = depth_dir / city / f"{stem}.npy"
            if not dep_path.exists():
                dep_path = depth_dir / city / f"{stem}_leftImg8bit.npy"
            if not dep_path.exists():
                dep_path = None

        img_path = img_dir / city / f"{stem}_leftImg8bit.png"
        if not img_path.exists():
            img_path = img_dir / city / f"{stem}.png"

        triples.append({
            "semantic": sem_path,
            "features": feat_path,
            "depth": dep_path,
            "image": img_path if img_path.exists() else None,
            "city": city,
            "stem": stem,
        })

    return triples


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(
        description="Generate instances via DINOv2 intra-class feature clustering"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--semantic_subdir", type=str, default="pseudo_semantic_cause")
    parser.add_argument("--feature_subdir", type=str, default="dinov3_features")
    parser.add_argument("--depth_subdir", type=str, default=None,
                        help="Depth map subdirectory (optional, e.g., depth_spidepth)")
    parser.add_argument("--output_dir", type=str, default="pseudo_instances_dino_cluster")
    parser.add_argument("--cause27", action="store_true",
                        help="Map CAUSE 27-class IDs to 19 trainIDs")
    parser.add_argument("--depth_weight", type=float, default=2.0,
                        help="Weight for depth in clustering")
    parser.add_argument("--min_cluster_size", type=int, default=15,
                        help="HDBSCAN min_cluster_size (higher=less fragmentation)")
    parser.add_argument("--min_area", type=int, default=500,
                        help="Minimum pixel area for a valid instance")
    parser.add_argument("--split_threshold", type=int, default=3000,
                        help="Only cluster CCs larger than this (pixels)")
    parser.add_argument("--global_mode", action="store_true",
                        help="Global HDBSCAN: cluster all patches, then classify")
    parser.add_argument("--feature_cc", action="store_true",
                        help="Feature-CC: edge-cut by cosine similarity + connected components")
    parser.add_argument("--sim_threshold", type=float, default=0.6,
                        help="Cosine similarity threshold for feature-CC edge cuts")
    parser.add_argument("--spatial_weight", type=float, default=2.0,
                        help="Weight for spatial coords in global mode")
    parser.add_argument("--visualize", action="store_true",
                        help="Save overlay visualizations")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N images")
    args = parser.parse_args()

    # Find file triples
    triples = find_triples(
        args.cityscapes_root, args.split,
        args.semantic_subdir, args.feature_subdir, args.depth_subdir,
    )
    if args.limit:
        triples = triples[:args.limit]

    logger.info(f"Found {len(triples)} images for {args.split} split")
    mode_str = "FEATURE-CC" if args.feature_cc else ("GLOBAL" if args.global_mode else "per-class")
    logger.info(f"Mode: {mode_str}, min_area={args.min_area}, "
                f"depth={'yes' if args.depth_subdir else 'no'}")
    if args.feature_cc:
        logger.info(f"  sim_threshold={args.sim_threshold}, depth_weight={args.depth_weight}")
    elif args.global_mode:
        logger.info(f"  spatial_weight={args.spatial_weight}, depth_weight={args.depth_weight}, "
                     f"min_cluster_size={args.min_cluster_size}")
    else:
        logger.info(f"  depth_weight={args.depth_weight}, min_cluster_size={args.min_cluster_size}, "
                     f"split_threshold={args.split_threshold}")

    # Output directories
    output_base = Path(args.output_dir) / args.split
    output_base.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        vis_dir = Path(args.output_dir) / "vis" / args.split
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Process
    stats = {
        "total_images": 0,
        "total_instances": 0,
        "instances_per_image": [],
        "time_per_image": [],
    }

    for item in tqdm(triples, desc="DINOv2 clustering"):
        t0 = time.time()

        if args.feature_cc:
            instances = process_single_image(
                semantic_path=item["semantic"],
                feature_path=item["features"],
                depth_path=item["depth"],
                cause27=args.cause27,
                feature_cc=True,
                thing_ids=THING_IDS,
                sim_threshold=args.sim_threshold,
                depth_weight=args.depth_weight,
                min_area=args.min_area,
            )
        elif args.global_mode:
            instances = process_single_image(
                semantic_path=item["semantic"],
                feature_path=item["features"],
                depth_path=item["depth"],
                cause27=args.cause27,
                global_mode=True,
                thing_ids=THING_IDS,
                spatial_weight=args.spatial_weight,
                depth_weight=args.depth_weight,
                min_cluster_size=args.min_cluster_size,
                min_area=args.min_area,
            )
        else:
            instances = process_single_image(
                semantic_path=item["semantic"],
                feature_path=item["features"],
                depth_path=item["depth"],
                cause27=args.cause27,
                thing_ids=THING_IDS,
                depth_weight=args.depth_weight,
                min_cluster_size=args.min_cluster_size,
                min_area=args.min_area,
                split_threshold=args.split_threshold,
            )

        # Save NPZ
        city_dir = output_base / item["city"]
        city_dir.mkdir(parents=True, exist_ok=True)
        npz_path = city_dir / f"{item['stem']}.npz"
        save_instances(instances, npz_path)

        # Visualization
        if args.visualize and item["image"] is not None:
            vis_city_dir = vis_dir / item["city"]
            vis_city_dir.mkdir(parents=True, exist_ok=True)
            vis_path = vis_city_dir / f"{item['stem']}_overlay.png"
            visualize_instances(item["image"], instances, vis_path)

        elapsed = time.time() - t0
        n_inst = len(instances)
        stats["total_images"] += 1
        stats["total_instances"] += n_inst
        stats["instances_per_image"].append(n_inst)
        stats["time_per_image"].append(elapsed)

        if stats["total_images"] <= 5 or stats["total_images"] % 100 == 0:
            tqdm.write(
                f"  {item['stem']}: {n_inst} instances, {elapsed:.1f}s"
            )

    # Summary
    if stats["total_images"] > 0:
        inst_arr = np.array(stats["instances_per_image"])
        time_arr = np.array(stats["time_per_image"])
        print(f"\n{'=' * 60}")
        print(f"DINOv2 Feature Clustering — Instance Generation Complete")
        print(f"{'=' * 60}")
        print(f"  Images processed:        {stats['total_images']}")
        print(f"  Total instances:         {stats['total_instances']}")
        print(f"  Avg instances/image:     {inst_arr.mean():.1f}")
        print(f"  Median instances/image:  {np.median(inst_arr):.0f}")
        print(f"  Min/Max instances:       {inst_arr.min()}/{inst_arr.max()}")
        print(f"  Avg time/image:          {time_arr.mean():.1f}s")
        print(f"  Output:                  {output_base}")

        summary = {
            "config": {
                "semantic_subdir": args.semantic_subdir,
                "feature_subdir": args.feature_subdir,
                "depth_subdir": args.depth_subdir,
                "cause27": args.cause27,
                "depth_weight": args.depth_weight,
                "min_cluster_size": args.min_cluster_size,
                "min_area": args.min_area,
                "split_threshold": args.split_threshold,
                "split": args.split,
            },
            "total_images": stats["total_images"],
            "total_instances": stats["total_instances"],
            "avg_instances_per_image": float(inst_arr.mean()),
            "median_instances_per_image": float(np.median(inst_arr)),
            "min_instances": int(inst_arr.min()),
            "max_instances": int(inst_arr.max()),
            "avg_time_per_image": float(time_arr.mean()),
        }
        stats_path = Path(args.output_dir) / f"stats_{args.split}.json"
        with open(stats_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Stats saved:             {stats_path}")


if __name__ == "__main__":
    main()
