"""Approach #3: Learned Depth Edge Detector (inference only).

Loads a trained edge prediction ConvNet and uses its per-pixel
boundary probability map in place of Sobel edges for CC decomposition.

Training script: mbps_pytorch/train_learned_edge.py
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter

from .utils import dilation_reclaim, upsample_features

THING_IDS = set(range(11, 19))


def _prepare_edge_input(
    depth: np.ndarray,
    features: np.ndarray,
    pca_dim: int = 64,
) -> np.ndarray:
    """Prepare multi-channel input for edge prediction network.

    Channels: depth (1) + Sobel gx,gy (2) + PCA-reduced DINOv2 (pca_dim).

    Args:
        depth: (H, W) float32 depth map.
        features: (fh, fw, C) DINOv2 features.
        pca_dim: PCA dimensionality for features.

    Returns:
        input_tensor: (1, C_in, H, W) float32 tensor for the network.
    """
    import torch
    from scipy.ndimage import sobel
    from sklearn.decomposition import PCA

    H, W = depth.shape

    # Depth channel
    depth_ch = depth.astype(np.float32)[np.newaxis]  # (1, H, W)

    # Sobel gradient channels
    depth_f64 = depth.astype(np.float64)
    gx = sobel(depth_f64, axis=1).astype(np.float32)
    gy = sobel(depth_f64, axis=0).astype(np.float32)
    grad_ch = np.stack([gx, gy])  # (2, H, W)

    # PCA-reduced DINOv2 features upsampled to (H, W)
    fh, fw, C = features.shape
    flat = features.reshape(-1, C)
    n_comp = min(pca_dim, C, flat.shape[0])
    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(flat).reshape(fh, fw, n_comp)

    # Upsample to full resolution
    feat_up = upsample_features(reduced, target_h=H, target_w=W)
    feat_ch = feat_up.transpose(2, 0, 1)  # (pca_dim, H, W)

    # Concatenate all channels
    channels = np.concatenate([depth_ch, grad_ch, feat_ch], axis=0)
    return torch.from_numpy(channels).unsqueeze(0).float()  # (1, C_in, H, W)


def learned_edge_cc_instances(
    semantic: np.ndarray,
    depth: np.ndarray,
    thing_ids: set = THING_IDS,
    edge_threshold: float = 0.5,
    min_area: int = 1000,
    dilation_iters: int = 3,
    depth_blur_sigma: float = 1.0,
    features: np.ndarray = None,
    model=None,
    pca_dim: int = 64,
) -> list:
    """Instance decomposition using learned edge prediction.

    Args:
        semantic: (H,W) uint8 trainID map.
        depth: (H,W) float32 [0,1].
        thing_ids: set of thing class trainIDs.
        edge_threshold: probability threshold for edge map.
        min_area: minimum instance area.
        dilation_iters: boundary reclamation iterations.
        depth_blur_sigma: unused (model handles its own smoothing).
        features: (fh, fw, C) DINOv2 feature grid. Required.
        model: trained EdgePredictor model. Required.
        pca_dim: PCA components for feature reduction.

    Returns:
        List of (mask, class_id, score).
    """
    if features is None:
        raise ValueError("Learned edge method requires DINOv2 features")
    if model is None:
        raise ValueError("Learned edge method requires a trained model")

    import torch

    H, W = semantic.shape

    # Prepare input and predict
    input_tensor = _prepare_edge_input(depth, features, pca_dim=pca_dim)
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        edge_logits = model(input_tensor)  # (1, 1, H, W)
        edge_prob = torch.sigmoid(edge_logits).squeeze().cpu().numpy()

    # Threshold to binary edge map
    edges = edge_prob > edge_threshold

    # Standard CC pipeline on learned edges
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
