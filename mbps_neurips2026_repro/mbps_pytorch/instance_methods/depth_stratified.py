"""Depth-Stratified DINOv2 Spectral Clustering for instance decomposition.

Separates depth and feature signals: depth creates "layers" (quantile bins),
DINOv2 features split each layer into instances via agglomerative clustering.

This fixes Joint NCut's failure (PQ_things=17.90) where mixing depth+features
in a joint affinity caused depth to dominate. Here, depth only defines the
stratification; instance discrimination is purely appearance-based.

Targets co-planar objects: people at the same depth land in the same bin,
then DINOv2 features distinguish them by clothing/pose/context.
"""

import numpy as np
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist

from .utils import dilation_reclaim

THING_IDS = set(range(11, 19))


def _quantile_depth_bins(
    depth_values: np.ndarray,
    n_bins: int,
) -> list:
    """Compute depth quantile bin intervals.

    Args:
        depth_values: 1D array of depth values within a class region.
        n_bins: number of bins.

    Returns:
        List of (low, high) tuples defining bin intervals.
    """
    if len(depth_values) < 2 or n_bins < 1:
        return [(depth_values.min(), depth_values.max() + 1e-8)]

    quantiles = np.quantile(depth_values, np.linspace(0, 1, n_bins + 1))
    # Deduplicate collapsed quantiles
    unique_q = np.unique(quantiles)
    if len(unique_q) < 2:
        return [(unique_q[0], unique_q[0] + 1e-8)]

    bins = []
    for i in range(len(unique_q) - 1):
        bins.append((unique_q[i], unique_q[i + 1]))
    return bins


def _cluster_patches(
    patch_features: np.ndarray,
    sim_threshold: float,
) -> np.ndarray:
    """Agglomerative clustering on cosine similarity.

    Args:
        patch_features: (P, D) L2-normalized feature vectors.
        sim_threshold: cosine similarity cutoff for merging.

    Returns:
        labels: (P,) int cluster labels starting from 0.
    """
    P = patch_features.shape[0]
    if P <= 1:
        return np.zeros(P, dtype=int)

    # Cosine distance = 1 - cosine_similarity
    distances = pdist(patch_features, metric="cosine")
    # Clip NaN/inf from potential zero-norm vectors
    distances = np.nan_to_num(distances, nan=1.0, posinf=1.0, neginf=0.0)

    Z = linkage(distances, method="average")
    # Cut at distance = 1 - sim_threshold
    labels = fcluster(Z, t=1.0 - sim_threshold, criterion="distance")
    return labels - 1  # fcluster returns 1-based labels


def depth_stratified_instances(
    semantic: np.ndarray,
    depth: np.ndarray,
    thing_ids: set = THING_IDS,
    n_depth_bins: int = 5,
    sim_threshold: float = 0.65,
    min_area: int = 1000,
    dilation_iters: int = 3,
    depth_blur_sigma: float = 1.0,
    features: np.ndarray = None,
) -> list:
    """Instance decomposition via depth-stratified feature clustering.

    Args:
        semantic: (H,W) uint8 trainID map.
        depth: (H,W) float32 [0,1].
        thing_ids: set of thing class trainIDs.
        n_depth_bins: number of depth quantile bins per class.
        sim_threshold: cosine similarity threshold for clustering.
        min_area: minimum instance area.
        dilation_iters: boundary reclamation iterations.
        depth_blur_sigma: Gaussian blur sigma for depth.
        features: (fh, fw, C) DINOv2 feature grid. Required.

    Returns:
        List of (mask, class_id, score).
    """
    if features is None:
        raise ValueError("Depth-stratified method requires DINOv2 features")

    H, W = semantic.shape
    fh, fw, C = features.shape

    # Smooth depth for stable binning
    if depth_blur_sigma > 0:
        depth_smooth = gaussian_filter(
            depth.astype(np.float64), sigma=depth_blur_sigma
        ).astype(np.float32)
    else:
        depth_smooth = depth.astype(np.float32)

    # Precompute pixel-to-patch mapping
    # patch (py, px) covers pixels [py*scale_y : (py+1)*scale_y, ...]
    scale_y = H / fh
    scale_x = W / fw

    # L2-normalize features for cosine similarity
    feat_norms = np.linalg.norm(features, axis=2, keepdims=True)
    feat_normalized = features / (feat_norms + 1e-8)

    raw_instances = []

    for cls in sorted(thing_ids):
        cls_mask = semantic == cls
        n_cls_pixels = cls_mask.sum()
        if n_cls_pixels < min_area:
            continue

        # Get depth values for this class
        depth_vals = depth_smooth[cls_mask]
        bins = _quantile_depth_bins(depth_vals, n_depth_bins)

        for q_low, q_high in bins:
            # Pixels in this depth bin AND this class
            if q_high == bins[-1][1]:
                # Last bin: inclusive upper bound
                bin_mask = cls_mask & (depth_smooth >= q_low) & (
                    depth_smooth <= q_high
                )
            else:
                bin_mask = cls_mask & (depth_smooth >= q_low) & (
                    depth_smooth < q_high
                )

            bin_pixel_count = bin_mask.sum()
            if bin_pixel_count < min_area:
                continue

            # Find unique patches that overlap this bin
            ys, xs = np.where(bin_mask)
            pys = np.clip((ys / scale_y).astype(int), 0, fh - 1)
            pxs = np.clip((xs / scale_x).astype(int), 0, fw - 1)
            patch_coords = set(zip(pys.tolist(), pxs.tolist()))

            if len(patch_coords) < 2:
                # Single patch or empty: entire bin = one instance
                raw_instances.append((bin_mask.copy(), cls, float(bin_pixel_count)))
                continue

            # Extract patch features
            patch_list = sorted(patch_coords)
            patch_features = np.array(
                [feat_normalized[py, px] for py, px in patch_list]
            )

            # Cluster patches by appearance
            labels = _cluster_patches(patch_features, sim_threshold)
            n_clusters = labels.max() + 1

            # Build pixel-level masks for each cluster
            # Create patch-to-cluster lookup
            patch_to_cluster = {}
            for idx, (py, px) in enumerate(patch_list):
                patch_to_cluster[(py, px)] = labels[idx]

            for cluster_id in range(n_clusters):
                # Pixels whose patch belongs to this cluster
                cluster_patches = {
                    k for k, v in patch_to_cluster.items() if v == cluster_id
                }
                # Build mask: pixel in bin_mask AND its patch is in this cluster
                cluster_mask = np.zeros((H, W), dtype=bool)
                for py, px in cluster_patches:
                    y0 = int(py * scale_y)
                    y1 = min(int((py + 1) * scale_y), H)
                    x0 = int(px * scale_x)
                    x1 = min(int((px + 1) * scale_x), W)
                    cluster_mask[y0:y1, x0:x1] = True

                # Intersect with actual bin mask (pixel-level precision)
                cluster_mask &= bin_mask

                area = float(cluster_mask.sum())
                if area < min_area:
                    continue

                # Split disconnected components
                labeled, n_cc = ndimage.label(cluster_mask)
                for cc_id in range(1, n_cc + 1):
                    cc_mask = labeled == cc_id
                    cc_area = float(cc_mask.sum())
                    if cc_area >= min_area:
                        raw_instances.append((cc_mask, cls, cc_area))

    return dilation_reclaim(
        raw_instances, semantic, thing_ids,
        min_area=min_area, dilation_iters=dilation_iters,
    )
