"""Approach #1: DINOv2 Feature Gradient Edges + Depth Fusion.

Computes spatial gradients of PCA-reduced DINOv2 features to detect
boundaries between visually distinct objects (even at the same depth).
Fuses feature edges with depth Sobel edges, then applies standard CC.

Targets co-planar objects (person, car) where depth alone fails.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel
from sklearn.decomposition import PCA

from .utils import dilation_reclaim, upsample_features

THING_IDS = set(range(11, 19))


def _compute_feature_gradients(
    features: np.ndarray,
    pca_dim: int = 64,
    target_h: int = 512,
    target_w: int = 1024,
) -> np.ndarray:
    """Compute spatial gradient magnitude of PCA-reduced features.

    Args:
        features: (fh, fw, C) DINOv2 feature grid (e.g., 32x64x768).
        pca_dim: number of PCA components for dimensionality reduction.
        target_h, target_w: output resolution for gradient map.

    Returns:
        grad_mag: (target_h, target_w) feature gradient magnitude.
    """
    fh, fw, C = features.shape

    # PCA reduce: (fh*fw, C) -> (fh*fw, pca_dim)
    flat = features.reshape(-1, C)
    pca = PCA(n_components=min(pca_dim, C, flat.shape[0]))
    reduced = pca.fit_transform(flat)  # (fh*fw, pca_dim)
    reduced = reduced.reshape(fh, fw, -1)

    # Compute per-channel spatial gradients at feature resolution
    grad_sum = np.zeros((fh, fw), dtype=np.float64)
    for c in range(reduced.shape[2]):
        gx = sobel(reduced[:, :, c].astype(np.float64), axis=1)
        gy = sobel(reduced[:, :, c].astype(np.float64), axis=0)
        grad_sum += gx ** 2 + gy ** 2
    grad_mag = np.sqrt(grad_sum)

    # Normalize to [0, 1]
    gmax = grad_mag.max()
    if gmax > 0:
        grad_mag /= gmax

    # Upsample to target resolution
    from PIL import Image
    grad_mag = np.array(
        Image.fromarray(grad_mag.astype(np.float32)).resize(
            (target_w, target_h), Image.BILINEAR
        )
    )
    return grad_mag


def feature_edge_cc_instances(
    semantic: np.ndarray,
    depth: np.ndarray,
    thing_ids: set = THING_IDS,
    feat_grad_threshold: float = 0.15,
    depth_grad_threshold: float = 0.03,
    fusion_mode: str = "union",
    pca_dim: int = 64,
    min_area: int = 1000,
    dilation_iters: int = 3,
    depth_blur_sigma: float = 1.0,
    features: np.ndarray = None,
) -> list:
    """Instance decomposition via DINOv2 feature gradients + depth edges.

    Args:
        semantic: (H,W) uint8 trainID map.
        depth: (H,W) float32 [0,1].
        thing_ids: set of thing class trainIDs.
        feat_grad_threshold: threshold for feature gradient edge map.
        depth_grad_threshold: threshold for depth Sobel edge map.
        fusion_mode: how to combine edges — "union", "intersection",
            "max_response", or "weighted_sum".
        pca_dim: PCA components for feature reduction.
        min_area: minimum instance area.
        dilation_iters: boundary reclamation iterations.
        depth_blur_sigma: Gaussian blur sigma for depth.
        features: (fh, fw, C) DINOv2 feature grid. Required.

    Returns:
        List of (mask, class_id, score).
    """
    if features is None:
        raise ValueError("Feature edge method requires DINOv2 features")

    H, W = semantic.shape

    # Compute depth edges (standard Sobel)
    if depth_blur_sigma > 0:
        depth_smooth = gaussian_filter(
            depth.astype(np.float64), sigma=depth_blur_sigma
        )
    else:
        depth_smooth = depth.astype(np.float64)

    gx = sobel(depth_smooth, axis=1)
    gy = sobel(depth_smooth, axis=0)
    depth_grad = np.sqrt(gx ** 2 + gy ** 2)

    # Normalize depth gradient to [0, 1]
    dmax = depth_grad.max()
    if dmax > 0:
        depth_grad_norm = depth_grad / dmax
    else:
        depth_grad_norm = depth_grad

    # Compute feature edges
    feat_grad = _compute_feature_gradients(
        features, pca_dim=pca_dim, target_h=H, target_w=W
    )

    # Fuse edge maps
    if fusion_mode == "union":
        edges = (depth_grad > depth_grad_threshold) | (
            feat_grad > feat_grad_threshold
        )
    elif fusion_mode == "intersection":
        edges = (depth_grad > depth_grad_threshold) & (
            feat_grad > feat_grad_threshold
        )
    elif fusion_mode == "max_response":
        combined = np.maximum(depth_grad_norm, feat_grad)
        threshold = min(depth_grad_threshold, feat_grad_threshold)
        edges = combined > threshold
    elif fusion_mode == "weighted_sum":
        combined = 0.5 * depth_grad_norm + 0.5 * feat_grad
        threshold = 0.5 * (depth_grad_threshold + feat_grad_threshold)
        edges = combined > threshold
    else:
        raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

    # Standard CC pipeline on fused edges
    instances = []
    for cls in sorted(thing_ids):
        cls_mask = semantic == cls
        if cls_mask.sum() < min_area:
            continue

        split_mask = cls_mask & ~edges
        labeled, n_cc = ndimage.label(split_mask)

        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = float(cc_mask.sum())
            if area >= min_area:
                instances.append((cc_mask, cls, area))

    return dilation_reclaim(
        instances, semantic, thing_ids,
        min_area=min_area, dilation_iters=dilation_iters,
    )
