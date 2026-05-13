"""Adaptive Depth-Feature Edge Fusion for instance decomposition.

Uses depth Sobel edges where depth gradient is strong (informative),
and DINOv2 feature gradient edges where depth is flat (uninformative).
This fixes Feature Edge CC's catastrophic failure (PQ_things=0.0) by
gating feature edges with a depth confidence map.

Key insight: feature edges are useful where depth fails (co-planar
objects), harmful where depth succeeds (over-segmentation).
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel
from scipy.special import expit

from .feature_edge_cc import _compute_feature_gradients
from .utils import dilation_reclaim

THING_IDS = set(range(11, 19))


def adaptive_edge_instances(
    semantic: np.ndarray,
    depth: np.ndarray,
    thing_ids: set = THING_IDS,
    depth_grad_threshold: float = 0.03,
    feat_grad_threshold: float = 0.30,
    depth_conf_temperature: float = 0.05,
    depth_conf_center: float = 0.03,
    fusion_mode: str = "soft",
    pca_dim: int = 64,
    min_area: int = 1000,
    dilation_iters: int = 3,
    depth_blur_sigma: float = 1.0,
    features: np.ndarray = None,
) -> list:
    """Instance decomposition via adaptive depth-feature edge fusion.

    Feature edges are gated by depth confidence: suppressed where depth
    gradient is strong, activated where depth is flat.

    Args:
        semantic: (H,W) uint8 trainID map.
        depth: (H,W) float32 [0,1].
        thing_ids: set of thing class trainIDs.
        depth_grad_threshold: threshold for final edge map.
        feat_grad_threshold: threshold for feature edges (hard mode only).
        depth_conf_temperature: sigmoid sharpness for depth confidence.
        depth_conf_center: sigmoid center point (matches depth edge scale).
        fusion_mode: "soft" (weighted gating) or "hard" (binary switch).
        pca_dim: PCA components for feature gradient computation.
        min_area: minimum instance area.
        dilation_iters: boundary reclamation iterations.
        depth_blur_sigma: Gaussian blur sigma for depth.
        features: (fh, fw, C) DINOv2 feature grid. Required.

    Returns:
        List of (mask, class_id, score).
    """
    if features is None:
        raise ValueError("Adaptive edge method requires DINOv2 features")

    H, W = semantic.shape

    # --- Depth edge computation ---
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
    depth_edge_norm = depth_grad / (dmax + 1e-8)

    # --- Depth confidence map ---
    # High where depth gradient is strong, low where flat
    depth_conf = expit(
        (depth_edge_norm - depth_conf_center) / (depth_conf_temperature + 1e-8)
    )

    # --- Feature edge computation (reuse from feature_edge_cc) ---
    feat_edge = _compute_feature_gradients(
        features, pca_dim=pca_dim, target_h=H, target_w=W
    )

    # --- Adaptive fusion ---
    if fusion_mode == "soft":
        # Feature edges suppressed where depth is confident
        gated_feat = (1.0 - depth_conf) * feat_edge
        combined = np.maximum(depth_edge_norm, gated_feat)
        edges = combined > depth_grad_threshold
    elif fusion_mode == "hard":
        depth_edges = depth_edge_norm > depth_grad_threshold
        feat_edges = feat_edge > feat_grad_threshold
        depth_flat = depth_conf < 0.5
        edges = depth_edges | (depth_flat & feat_edges)
    else:
        raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

    # --- Standard CC pipeline ---
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
