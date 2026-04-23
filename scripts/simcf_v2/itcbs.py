"""Step F -- ITCBS: Info-Theoretic Cluster Boundary Sharpening.

Reassigns boundary pixels by maximizing local mutual information
between pixel features and class assignments:

    y_p* = argmax_c  sum_{q in N(p)} log P(c | f_q)

where P(c | f) = softmax(-||f - mu_c||^2 / T) and mu_c is the per-class
feature centroid. Only pixels within a morphological boundary zone are
candidates; reassignment requires a minimum MI gain.

Target: Reduce road/sidewalk confusion at class boundaries.
"""

import logging

import numpy as np
from PIL import Image
from scipy import ndimage

logger = logging.getLogger(__name__)

FEAT_H, FEAT_W = 32, 64
NUM_CLASSES = 19


def compute_class_centroids(
    features: np.ndarray,
    semantic: np.ndarray,
    cluster_to_class: np.ndarray,
) -> np.ndarray:
    """Compute per-class mean feature vectors for a single image.

    Args:
        features: (N_patches, D) feature vectors.
        semantic: (H, W) uint8 cluster IDs.
        cluster_to_class: (256,) cluster -> trainID LUT.

    Returns:
        (NUM_CLASSES, D) mean feature per class. Zeros for absent classes.
    """
    D = features.shape[1]
    feat_2d = features.reshape(FEAT_H, FEAT_W, -1)
    sem_small = np.array(
        Image.fromarray(semantic).resize((FEAT_W, FEAT_H), Image.NEAREST)
    )
    mapped = cluster_to_class[sem_small]

    centroids = np.zeros((NUM_CLASSES, D), dtype=np.float64)
    for cls in range(NUM_CLASSES):
        mask = mapped == cls
        if mask.any():
            centroids[cls] = feat_2d[mask].mean(axis=0).astype(np.float64)
    return centroids


def step_f(
    semantic: np.ndarray,
    features: np.ndarray,
    cluster_to_class: np.ndarray,
    class_centroids: np.ndarray,
    num_clusters: int = 80,
    boundary_px: int = 2,
    temperature: float = 0.1,
    min_gain: float = 0.5,
) -> int:
    """Sharpen class boundaries via local MI maximization.

    Modifies semantic in-place.

    Args:
        semantic: (H, W) uint8 cluster IDs.
        features: (N_patches, D) L2-normalized DINOv3 features.
        cluster_to_class: (256,) cluster -> trainID LUT.
        class_centroids: (NUM_CLASSES, D) per-class mean features (global).
        num_clusters: Number of k-means clusters.
        boundary_px: Dilation width of boundary zone at patch resolution.
        temperature: Softmax temperature for P(c|f).
        min_gain: Minimum MI gain to trigger reassignment.

    Returns:
        Number of pixels changed.
    """
    H, W = semantic.shape
    D = features.shape[1]
    feat_2d = features.reshape(FEAT_H, FEAT_W, -1)

    # Map semantic to classes at patch resolution
    sem_small = np.array(
        Image.fromarray(semantic).resize((FEAT_W, FEAT_H), Image.NEAREST)
    )
    mapped_small = cluster_to_class[sem_small]

    # Detect boundary patches: any neighbor has a different valid class
    boundary = np.zeros((FEAT_H, FEAT_W), dtype=bool)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(mapped_small, dy, axis=0), dx, axis=1)
            # Mask out wrapped edges to avoid false boundaries
            valid_mask = np.ones((FEAT_H, FEAT_W), dtype=bool)
            if dy == -1:
                valid_mask[-1, :] = False
            elif dy == 1:
                valid_mask[0, :] = False
            if dx == -1:
                valid_mask[:, -1] = False
            elif dx == 1:
                valid_mask[:, 0] = False
            boundary |= (
                (mapped_small != shifted)
                & (mapped_small < NUM_CLASSES)
                & (shifted < NUM_CLASSES)
                & valid_mask
            )

    # Dilate boundary zone
    if boundary_px > 1:
        struct = ndimage.generate_binary_structure(2, 1)
        boundary = ndimage.binary_dilation(
            boundary, structure=struct, iterations=boundary_px - 1
        )

    if not boundary.any():
        return 0

    # Precompute log P(c|f) for all patches: log softmax(-||f - mu_c||^2 / T)
    feat_flat = feat_2d.reshape(-1, D).astype(np.float64)
    sq_dists = np.sum(
        (feat_flat[:, None, :] - class_centroids[None, :, :]) ** 2, axis=2
    )  # (N_patches, NUM_CLASSES)
    logits = -sq_dists / temperature
    logits -= logits.max(axis=1, keepdims=True)  # numerical stability
    log_probs = logits - np.log(
        np.sum(np.exp(logits), axis=1, keepdims=True) + 1e-10
    )
    log_probs_2d = log_probs.reshape(FEAT_H, FEAT_W, NUM_CLASSES)

    # Per-class best cluster LUT (first cluster mapping to each class)
    class_best_cluster = _build_class_cluster_lut(
        semantic, cluster_to_class, num_clusters
    )

    # Process boundary patches
    n_changed = 0
    boundary_indices = np.argwhere(boundary)

    for idx in boundary_indices:
        r, c = idx[0], idx[1]
        current_cls = int(mapped_small[r, c])
        if current_cls >= NUM_CLASSES:
            continue

        # Local 3x3 neighborhood log-probs
        r_lo, r_hi = max(0, r - 1), min(FEAT_H, r + 2)
        c_lo, c_hi = max(0, c - 1), min(FEAT_W, c + 2)
        local_lp = log_probs_2d[r_lo:r_hi, c_lo:c_hi]  # (h, w, C)

        # Sum log-probs over neighborhood per class
        class_scores = local_lp.sum(axis=(0, 1))  # (C,)

        # Only consider classes present in neighborhood
        nb_classes = set(mapped_small[r_lo:r_hi, c_lo:c_hi].flatten())
        nb_classes.discard(255)
        if len(nb_classes) < 2:
            continue

        best_cls = int(class_scores.argmax())
        if best_cls == current_cls or best_cls not in nb_classes:
            continue

        gain = class_scores[best_cls] - class_scores[current_cls]
        if gain < min_gain:
            continue

        best_cluster = class_best_cluster[best_cls]
        if best_cluster == 255:
            continue

        # Reassign at full resolution for this patch's footprint
        scale_h, scale_w = H / FEAT_H, W / FEAT_W
        fr_lo = int(r * scale_h)
        fr_hi = int((r + 1) * scale_h)
        fc_lo = int(c * scale_w)
        fc_hi = int((c + 1) * scale_w)

        patch_region = semantic[fr_lo:fr_hi, fc_lo:fc_hi]
        old_cls_mask = cluster_to_class[patch_region] == current_cls
        if old_cls_mask.any():
            patch_region[old_cls_mask] = best_cluster
            n_changed += int(old_cls_mask.sum())

    return n_changed


def _build_class_cluster_lut(
    semantic: np.ndarray,
    cluster_to_class: np.ndarray,
    num_clusters: int,
) -> np.ndarray:
    """Find the most frequent cluster per class in this image.

    Returns:
        (NUM_CLASSES,) uint8 array of best cluster IDs (255 = none).
    """
    lut = np.full(NUM_CLASSES, 255, dtype=np.uint8)
    flat = semantic.flatten()
    flat = flat[flat < num_clusters]
    if len(flat) == 0:
        return lut

    cluster_counts = np.bincount(flat, minlength=num_clusters)
    for cls in range(NUM_CLASSES):
        cls_mask = cluster_to_class[:num_clusters] == cls
        if not cls_mask.any():
            continue
        cls_counts = cluster_counts[:num_clusters].copy()
        cls_counts[~cls_mask] = 0
        if cls_counts.max() > 0:
            lut[cls] = int(cls_counts.argmax())
    return lut
