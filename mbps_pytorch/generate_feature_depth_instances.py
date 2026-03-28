#!/usr/bin/env python3
"""Generate instance pseudo-labels via DINOv2 feature gradients + depth edges.

Extends depth-guided splitting with appearance-based boundary detection using
pre-computed DINOv2 ViT-B/14 features. Feature spatial gradients detect
boundaries between co-planar objects (e.g., adjacent pedestrians at same depth)
that depth Sobel edges miss.

Algorithm per image:
  1. Load semantic labels, depth map, and DINOv2 features
  2. Compute depth gradient edges (Sobel, as in depth-guided baseline)
  3. Compute DINOv2 feature gradient edges:
     - Reshape features to (32, 64, 768) spatial grid
     - Compute L2 norm of spatial differences (appearance boundary strength)
     - Upsample to working resolution (512, 1024)
  4. Combine depth + feature edges (union, weighted, or product mode)
  5. Per-class connected components on (class_mask & ~combined_edges)
  6. Filter by min_area, reclaim boundary pixels via dilation
  7. Save as NPZ

Usage:
    # Quick test (5 images)
    python mbps_pytorch/generate_feature_depth_instances.py \
        --cityscapes_root /path/to/cityscapes --split val \
        --alpha 0.5 --feat_threshold 0.5 --combine_mode weighted \
        --limit 5

    # Full val with union mode (most aggressive splitting)
    python mbps_pytorch/generate_feature_depth_instances.py \
        --cityscapes_root /path/to/cityscapes --split val \
        --combine_mode union --depth_threshold 0.20 --feat_threshold 0.5 \
        --min_area 1000
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
from scipy.ndimage import gaussian_filter, sobel, zoom
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

WORK_H, WORK_W = 512, 1024
FEAT_H, FEAT_W = 32, 64
FEAT_DIM = 768
DEFAULT_THING_IDS = set(range(11, 19))

CS_NAMES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic_light", 7: "traffic_sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}

IGNORE_LABEL = 255

_CAUSE27_TO_TRAINID = np.full(256, IGNORE_LABEL, dtype=np.uint8)
for _c27, _t19 in {
    0: 0, 1: 1, 2: 255, 3: 255, 4: 2, 5: 3, 6: 4,
    7: 255, 8: 255, 9: 255, 10: 5, 11: 5, 12: 6, 13: 7,
    14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14,
    21: 15, 22: 13, 23: 14, 24: 16, 25: 17, 26: 18,
}.items():
    _CAUSE27_TO_TRAINID[_c27] = _t19


def compute_feature_edges(features, threshold=0.5, blur_sigma=0.0):
    """Compute boundary map from DINOv2 spatial feature gradients.

    Args:
        features: (N, D) float array where N = FEAT_H * FEAT_W, D = FEAT_DIM
        threshold: normalized gradient magnitude threshold for edges
        blur_sigma: Gaussian blur on gradient magnitude before thresholding

    Returns:
        feat_edges: (WORK_H, WORK_W) bool — feature-based edge map
        feat_grad_norm: (WORK_H, WORK_W) float32 — normalized gradient magnitude
    """
    feat = features.astype(np.float32).reshape(FEAT_H, FEAT_W, FEAT_DIM)

    # Spatial gradients: L2 norm of feature differences between adjacent patches
    # Horizontal gradients (32, 63, 768) → L2 norm → (32, 63)
    dx = feat[:, 1:, :] - feat[:, :-1, :]
    grad_x = np.linalg.norm(dx, axis=2)

    # Vertical gradients (31, 64, 768) → L2 norm → (31, 64)
    dy = feat[1:, :, :] - feat[:-1, :, :]
    grad_y = np.linalg.norm(dy, axis=2)

    # Pad back to (FEAT_H, FEAT_W) by taking max of neighbors
    grad_x_full = np.zeros((FEAT_H, FEAT_W), dtype=np.float32)
    grad_x_full[:, :-1] = np.maximum(grad_x_full[:, :-1], grad_x)
    grad_x_full[:, 1:] = np.maximum(grad_x_full[:, 1:], grad_x)

    grad_y_full = np.zeros((FEAT_H, FEAT_W), dtype=np.float32)
    grad_y_full[:-1, :] = np.maximum(grad_y_full[:-1, :], grad_y)
    grad_y_full[1:, :] = np.maximum(grad_y_full[1:, :], grad_y)

    # Combined gradient magnitude
    feat_grad = np.sqrt(grad_x_full ** 2 + grad_y_full ** 2)

    # Normalize to [0, 1]
    g_min, g_max = feat_grad.min(), feat_grad.max()
    if g_max > g_min:
        feat_grad_norm = (feat_grad - g_min) / (g_max - g_min)
    else:
        feat_grad_norm = np.zeros_like(feat_grad)

    # Optional blur before upsampling
    if blur_sigma > 0:
        feat_grad_norm = gaussian_filter(feat_grad_norm, sigma=blur_sigma)

    # Upsample to working resolution using bilinear interpolation
    scale_h = WORK_H / FEAT_H
    scale_w = WORK_W / FEAT_W
    feat_grad_full = zoom(feat_grad_norm, (scale_h, scale_w), order=1)

    # Ensure exact shape (zoom can be off by 1 pixel)
    if feat_grad_full.shape != (WORK_H, WORK_W):
        feat_grad_full = np.array(
            Image.fromarray(feat_grad_full).resize((WORK_W, WORK_H), Image.BILINEAR)
        )

    feat_edges = feat_grad_full > threshold
    return feat_edges, feat_grad_full


def feature_depth_instances(semantic, depth, features, thing_ids=DEFAULT_THING_IDS,
                            depth_threshold=0.20, feat_threshold=0.5,
                            alpha=0.5, combine_mode="weighted",
                            min_area=1000, dilation_iters=3,
                            depth_blur_sigma=1.0, feat_blur_sigma=0.0):
    """Split thing regions using combined depth + feature gradient edges.

    Args:
        semantic: (H, W) uint8 trainID map
        depth: (H, W) float32 depth map [0, 1]
        features: (N, D) DINOv2 features (N=FEAT_H*FEAT_W, D=FEAT_DIM)
        thing_ids: set of trainIDs for thing classes
        depth_threshold: depth gradient threshold (0.20 = best known)
        feat_threshold: feature gradient threshold (normalized)
        alpha: depth edge weight in weighted mode (0=features only, 1=depth only)
        combine_mode: "union", "weighted", or "product"
        min_area: minimum instance pixel area
        dilation_iters: boundary reclamation dilation iterations
        depth_blur_sigma: Gaussian blur on depth before gradient
        feat_blur_sigma: Gaussian blur on feature gradient before upsampling

    Returns:
        List of (mask, class_id, score) tuples, sorted by area descending.
    """
    # --- Depth edges (same as depth-guided baseline) ---
    if depth_blur_sigma > 0:
        depth_smooth = gaussian_filter(depth.astype(np.float64), sigma=depth_blur_sigma)
    else:
        depth_smooth = depth.astype(np.float64)

    gx = sobel(depth_smooth, axis=1)
    gy = sobel(depth_smooth, axis=0)
    depth_grad = np.sqrt(gx ** 2 + gy ** 2)

    # Normalize depth gradient to [0, 1]
    d_min, d_max = depth_grad.min(), depth_grad.max()
    if d_max > d_min:
        depth_grad_norm = (depth_grad - d_min) / (d_max - d_min)
    else:
        depth_grad_norm = np.zeros_like(depth_grad)

    depth_edges = depth_grad > depth_threshold

    # --- Feature edges ---
    feat_edges, feat_grad_norm = compute_feature_edges(
        features, threshold=feat_threshold, blur_sigma=feat_blur_sigma
    )

    # --- Combine edge maps ---
    if combine_mode == "union":
        combined_edges = depth_edges | feat_edges
    elif combine_mode == "weighted":
        combined_score = alpha * depth_grad_norm + (1.0 - alpha) * feat_grad_norm
        # Use geometric mean of thresholds as combined threshold
        combined_threshold = alpha * depth_threshold + (1.0 - alpha) * feat_threshold
        combined_edges = combined_score > combined_threshold
    elif combine_mode == "product":
        combined_score = depth_grad_norm * feat_grad_norm
        combined_edges = combined_score > (depth_threshold * feat_threshold)
    else:
        raise ValueError(f"Unknown combine_mode: {combine_mode}")

    # --- Per-class connected components (same as depth-guided baseline) ---
    assigned = np.zeros(semantic.shape, dtype=bool)
    instances = []

    for cls in sorted(thing_ids):
        cls_mask = semantic == cls
        if cls_mask.sum() < min_area:
            continue

        split_mask = cls_mask & ~combined_edges
        labeled, n_cc = ndimage.label(split_mask)

        cc_list = []
        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = int(cc_mask.sum())
            if area >= min_area:
                cc_list.append((cc_id, cc_mask, area))
        cc_list.sort(key=lambda x: -x[2])

        for cc_id, cc_mask, area in cc_list:
            if dilation_iters > 0:
                dilated = ndimage.binary_dilation(cc_mask, iterations=dilation_iters)
                reclaimed = dilated & cls_mask & ~assigned
                final_mask = cc_mask | reclaimed
            else:
                final_mask = cc_mask

            final_area = float(final_mask.sum())
            if final_area < min_area:
                continue

            assigned |= final_mask
            instances.append((final_mask, cls, final_area))

    # Sort by area descending, normalize scores
    instances.sort(key=lambda x: -x[2])
    if instances:
        max_area = instances[0][2]
        instances = [(m, c, s / max_area) for m, c, s in instances]

    return instances


def save_instances(instances, output_path, h=WORK_H, w=WORK_W):
    """Save instances in NPZ format compatible with evaluate_cascade_pseudolabels.py."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not instances:
        np.savez_compressed(
            str(output_path),
            masks=np.zeros((0, h * w), dtype=bool),
            scores=np.zeros((0,), dtype=np.float32),
            num_valid=0,
            h_patches=h,
            w_patches=w,
        )
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate instance pseudo-labels via DINOv2 feature + depth edges"
    )
    parser.add_argument("--cityscapes_root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--semantic_subdir", default="pseudo_semantic_mapped_k80")
    parser.add_argument("--depth_subdir", default="depth_spidepth")
    parser.add_argument("--feature_subdir", default="dinov2_features")
    parser.add_argument("--output_subdir", default="pseudo_instance_feat_depth")
    parser.add_argument("--cause27", action="store_true",
                        help="Map CAUSE 27-class labels to 19 trainIDs")

    # Edge detection parameters
    parser.add_argument("--depth_threshold", type=float, default=0.20,
                        help="Depth Sobel gradient threshold (default: 0.20, best known)")
    parser.add_argument("--feat_threshold", type=float, default=0.5,
                        help="Feature gradient threshold (normalized, default: 0.5)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Depth weight in weighted mode (0=feat only, 1=depth only)")
    parser.add_argument("--combine_mode", default="weighted",
                        choices=["union", "weighted", "product"],
                        help="How to combine depth and feature edges")

    # Instance parameters
    parser.add_argument("--min_area", type=int, default=1000)
    parser.add_argument("--dilation_iters", type=int, default=3)
    parser.add_argument("--depth_blur", type=float, default=1.0)
    parser.add_argument("--feat_blur", type=float, default=0.0,
                        help="Gaussian blur on feature gradient before upsampling")
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    root = args.cityscapes_root
    sem_dir = os.path.join(root, args.semantic_subdir, args.split)
    depth_dir = os.path.join(root, args.depth_subdir, args.split)
    feat_dir = os.path.join(root, args.feature_subdir, args.split)
    out_dir = os.path.join(root, args.output_subdir, args.split)

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

            # DINOv2 features have _leftImg8bit suffix
            feat_path = os.path.join(feat_dir, city, stem + "_leftImg8bit.npy")
            if not os.path.exists(feat_path):
                # Fallback: try without suffix
                feat_path = os.path.join(feat_dir, city, stem + ".npy")

            if not os.path.exists(depth_path):
                continue
            if not os.path.exists(feat_path):
                logger.warning(f"No features for {stem}, skipping")
                continue

            image_list.append((city, stem, sem_path, depth_path, feat_path))

    if args.limit:
        image_list = image_list[:args.limit]

    logger.info(f"Processing {len(image_list)} images | "
                f"depth_thresh={args.depth_threshold}, feat_thresh={args.feat_threshold}, "
                f"alpha={args.alpha}, mode={args.combine_mode}, min_area={args.min_area}")

    os.makedirs(out_dir, exist_ok=True)
    total_instances = 0
    per_class_counts = {cls: 0 for cls in sorted(DEFAULT_THING_IDS)}
    t0 = time.time()

    for city, stem, sem_path, depth_path, feat_path in tqdm(image_list, desc="Feature+Depth"):
        # Load semantic map
        sem_img = np.array(Image.open(sem_path))
        if sem_img.shape != (WORK_H, WORK_W):
            sem_img = np.array(
                Image.fromarray(sem_img).resize((WORK_W, WORK_H), Image.NEAREST)
            )
        if args.cause27:
            sem_19 = _CAUSE27_TO_TRAINID[sem_img]
        else:
            sem_19 = sem_img

        # Load depth map
        depth = np.load(depth_path).astype(np.float32)
        if depth.shape != (WORK_H, WORK_W):
            depth = np.array(
                Image.fromarray(depth).resize((WORK_W, WORK_H), Image.BILINEAR),
                dtype=np.float32,
            )

        # Load DINOv2 features
        features = np.load(feat_path)  # (2048, 768) float16

        # Generate instances
        instances = feature_depth_instances(
            sem_19, depth, features,
            thing_ids=DEFAULT_THING_IDS,
            depth_threshold=args.depth_threshold,
            feat_threshold=args.feat_threshold,
            alpha=args.alpha,
            combine_mode=args.combine_mode,
            min_area=args.min_area,
            dilation_iters=args.dilation_iters,
            depth_blur_sigma=args.depth_blur,
            feat_blur_sigma=args.feat_blur,
        )

        # Save NPZ
        city_out = os.path.join(out_dir, city)
        out_path = os.path.join(city_out, stem + ".npz")
        save_instances(instances, out_path)

        total_instances += len(instances)
        for _, cls, _ in instances:
            per_class_counts[cls] += 1

    elapsed = time.time() - t0
    n = len(image_list)
    avg = total_instances / max(n, 1)

    logger.info(f"Done. {total_instances} instances from {n} images "
                f"({avg:.1f} avg/img, {elapsed:.1f}s)")
    logger.info(f"Per-class: {{{', '.join(f'{CS_NAMES[c]}: {v}' for c, v in sorted(per_class_counts.items()))}}}")

    # Save stats
    stats = {
        "split": args.split,
        "num_images": n,
        "total_instances": total_instances,
        "avg_instances_per_image": round(avg, 2),
        "config": {
            "depth_threshold": args.depth_threshold,
            "feat_threshold": args.feat_threshold,
            "alpha": args.alpha,
            "combine_mode": args.combine_mode,
            "min_area": args.min_area,
            "dilation_iters": args.dilation_iters,
            "depth_blur": args.depth_blur,
            "feat_blur": args.feat_blur,
            "feature_subdir": args.feature_subdir,
        },
        "per_class_counts": {
            CS_NAMES.get(c, str(c)): v for c, v in sorted(per_class_counts.items())
        },
        "elapsed_s": round(elapsed, 1),
    }
    stats_path = os.path.join(root, args.output_subdir, f"stats_{args.split}.json")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
