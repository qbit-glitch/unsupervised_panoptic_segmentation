"""Shared utilities for instance decomposition methods."""

import numpy as np
from scipy import ndimage
from PIL import Image

WORK_H, WORK_W = 512, 1024


def dilation_reclaim(instances, semantic, thing_ids, min_area=100,
                     dilation_iters=3):
    """Apply dilation-based boundary reclamation to instance masks.

    Args:
        instances: List of (mask, class_id, area) — raw instances before reclaim.
        semantic: (H,W) uint8 trainID map.
        thing_ids: set of thing class trainIDs.
        min_area: drop instances smaller than this after reclaim.
        dilation_iters: dilation iterations for reclaiming boundary pixels.

    Returns:
        List of (mask, class_id, score) with scores normalized to [0,1].
    """
    if not instances:
        return []

    # Sort by area descending for priority
    instances.sort(key=lambda x: -x[2])
    assigned = np.zeros(semantic.shape, dtype=bool)
    result = []

    for mask, cls, area in instances:
        if dilation_iters > 0:
            dilated = ndimage.binary_dilation(mask, iterations=dilation_iters)
            cls_mask = semantic == cls
            reclaimed = dilated & cls_mask & ~assigned
            final_mask = mask | reclaimed
        else:
            final_mask = mask

        final_area = float(final_mask.sum())
        if final_area < min_area:
            continue

        assigned |= final_mask
        result.append((final_mask, cls, final_area))

    # Normalize scores
    result.sort(key=lambda x: -x[2])
    if result:
        max_area = result[0][2]
        result = [(m, c, s / max_area) for m, c, s in result]

    return result


def load_features(feat_path, target_h=32, target_w=64):
    """Load pre-extracted DINOv2 features.

    Args:
        feat_path: path to .npy file (N_patches, 768) float16.
        target_h, target_w: expected patch grid dimensions.

    Returns:
        features: (target_h, target_w, 768) float32.
    """
    feats = np.load(str(feat_path)).astype(np.float32)
    n_patches, dim = feats.shape
    if n_patches == target_h * target_w:
        return feats.reshape(target_h, target_w, dim)
    # Fallback: try to infer grid
    hw = int(np.sqrt(n_patches * target_w / target_h))
    ww = n_patches // hw
    return feats.reshape(hw, ww, dim)


def upsample_features(features_grid, target_h=WORK_H, target_w=WORK_W):
    """Upsample (h, w, C) feature grid to (target_h, target_w, C) via bilinear."""
    h, w, C = features_grid.shape
    if h == target_h and w == target_w:
        return features_grid
    # Use PIL for each channel group (faster than scipy for large C)
    result = np.zeros((target_h, target_w, C), dtype=np.float32)
    # Process in chunks of 64 channels
    chunk = 64
    for i in range(0, C, chunk):
        end = min(i + chunk, C)
        for c in range(i, end):
            result[:, :, c] = np.array(
                Image.fromarray(features_grid[:, :, c]).resize(
                    (target_w, target_h), Image.BILINEAR
                )
            )
    return result


def cosine_similarity_regions(features_grid, mask_a, mask_b):
    """Compute mean cosine similarity between two pixel regions.

    Args:
        features_grid: (h, w, C) feature array at any resolution.
        mask_a, mask_b: (H, W) bool masks at features_grid resolution.

    Returns:
        float cosine similarity in [-1, 1].
    """
    h, w, C = features_grid.shape
    feat_a = features_grid[mask_a].mean(axis=0)
    feat_b = features_grid[mask_b].mean(axis=0)
    norm_a = np.linalg.norm(feat_a) + 1e-8
    norm_b = np.linalg.norm(feat_b) + 1e-8
    return float(np.dot(feat_a, feat_b) / (norm_a * norm_b))
