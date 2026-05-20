"""Depth-aware SLIC superpixel generator for the SGM adapter.

Stacks ``[RGB, alpha * depth, beta * PCA_8(DINO)] -> R^{12}`` and runs SLIC on the
12-channel feature volume so the resulting superpixels respect colour,
depth, and semantic edges simultaneously.

Sibling to ``superpixel_affinity.py`` — does **not** modify it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.segmentation import slic
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepthAwareSlicConfig:
    """Hyper-parameters for depth-aware SLIC."""

    n_segments: int = 1500
    compactness: float = 10.0
    sigma: float = 1.0
    alpha_slic: float = 1.0
    beta_slic: float = 0.5
    dino_pca_dim: int = 8
    pca_subsample_stride: int = 50


def _upsample_dino(dino_features: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinearly upsample patch-grid DINO features to image resolution.

    Args:
        dino_features: ``(H_p, W_p, D)`` float32.

    Returns:
        ``(H, W, D)`` float32.
    """
    tensor = torch.from_numpy(dino_features).permute(2, 0, 1).unsqueeze(0).float()
    tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()


def _pca_reduce(features_hw_d: np.ndarray, dim: int, stride: int) -> np.ndarray:
    """Per-image PCA on flattened DINO features.

    Subsampling stride avoids O(HW * D) fits when D == 768.
    """
    h, w, d = features_hw_d.shape
    flat = features_hw_d.reshape(h * w, d)
    fit_data = flat[::stride] if stride > 1 else flat
    pca = PCA(n_components=dim, random_state=0).fit(fit_data)
    reduced = pca.transform(flat).astype(np.float32)
    return reduced.reshape(h, w, dim)


def compute_depth_aware_slic(
    image: np.ndarray,
    depth: np.ndarray,
    dino_features: Optional[np.ndarray],
    config: DepthAwareSlicConfig = DepthAwareSlicConfig(),
) -> np.ndarray:
    """Compute depth-aware SLIC superpixels.

    Args:
        image: ``(H, W, 3)`` uint8 RGB.
        depth: ``(H, W)`` float32 normalised to ``[0, 1]``.
        dino_features: Optional ``(H_p, W_p, D)`` float32. If ``None`` we fall
            back to RGB + depth only (8-D stack).
        config: SLIC + stacking hyper-parameters.

    Returns:
        ``(H, W)`` int32 superpixel ID map in ``[0, K)``.
    """
    if image.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {image.dtype}")
    if depth.ndim != 2 or depth.shape != image.shape[:2]:
        raise ValueError(
            f"depth shape {depth.shape} must match image first two dims {image.shape[:2]}"
        )

    height, width = image.shape[:2]
    rgb = image.astype(np.float32) / 255.0
    depth_scaled = (config.alpha_slic * depth[..., None]).astype(np.float32)
    channels = [rgb, depth_scaled]

    if dino_features is not None:
        dino_up = _upsample_dino(dino_features, height, width)
        dino_pca = _pca_reduce(dino_up, config.dino_pca_dim, config.pca_subsample_stride)
        channels.append(config.beta_slic * dino_pca)

    stacked = np.concatenate(channels, axis=-1)  # (H, W, 3 + 1 + dino_pca_dim)

    labels = slic(
        stacked,
        n_segments=config.n_segments,
        compactness=config.compactness,
        sigma=config.sigma,
        channel_axis=-1,
        start_label=0,
        convert2lab=False,
    )
    return labels.astype(np.int32)


def compute_or_load_slic(
    image_path: Path,
    depth_path: Path,
    dino_path: Optional[Path],
    cache_path: Optional[Path],
    config: DepthAwareSlicConfig = DepthAwareSlicConfig(),
) -> np.ndarray:
    """Cache-aware wrapper around :func:`compute_depth_aware_slic`.

    Args:
        image_path: PNG/JPG image.
        depth_path: ``.npy`` depth map ``(H, W)``.
        dino_path: Optional ``.npy`` DINO patch features ``(H_p, W_p, D)``.
        cache_path: Optional ``.npy`` output cache; load if present, else save.

    Returns:
        ``(H, W)`` int32 superpixel ID map.
    """
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        return cached.astype(np.int32)

    image_pil = Image.open(image_path).convert("RGB")
    depth = np.load(depth_path).astype(np.float32)
    dh, dw = depth.shape
    if image_pil.size != (dw, dh):
        # Bring the image down to the depth/working resolution; SLIC labels
        # are then cached at the same resolution used by the trainer.
        image_pil = image_pil.resize((dw, dh), Image.BILINEAR)
    image = np.array(image_pil)
    dino = np.load(dino_path).astype(np.float32) if dino_path is not None else None
    if dino is not None and dino.ndim == 2:
        # Flat patches (N_patches, D); reshape to (H_p, W_p, D).
        # Cityscapes default: 2048 = 32 * 64.
        side = int(np.sqrt(dino.shape[0]))
        if side * side == dino.shape[0]:
            dino = dino.reshape(side, side, -1)
        else:
            hp = 32
            wp = dino.shape[0] // hp
            if hp * wp != dino.shape[0]:
                raise ValueError(
                    f"DINO patch tensor shape {dino.shape} not reshapeable to (H_p, W_p, D)"
                )
            dino = dino.reshape(hp, wp, -1)
    labels = compute_depth_aware_slic(image, depth, dino, config)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, labels.astype(np.int32))
    return labels


def superpixel_count(labels: Union[np.ndarray, torch.Tensor]) -> int:
    """Number of unique superpixel IDs in the label map."""
    if isinstance(labels, torch.Tensor):
        return int(labels.max().item()) + 1
    return int(labels.max()) + 1


__all__ = [
    "DepthAwareSlicConfig",
    "compute_depth_aware_slic",
    "compute_or_load_slic",
    "superpixel_count",
]
