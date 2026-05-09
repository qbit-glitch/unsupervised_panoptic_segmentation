"""Depth-guided contrastive loss with boundary emphasis and repulsion.

Extends the base depth_guided_correlation_loss from stego_loss.py with:
- Sobel edge detection on depth for boundary emphasis
- Repulsion term across depth discontinuities
- Scale-aware sigma for multi-resolution features
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_loss
from ...models.semantic.stego_loss import compute_cosine_similarity

logger = logging.getLogger(__name__)

# Sobel kernels for depth edge detection (3x3)
_SOBEL_X = torch.tensor(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
).view(1, 1, 3, 3)

_SOBEL_Y = torch.tensor(
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
).view(1, 1, 3, 3)


def _detect_depth_edges(
    depth: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Detect edges in depth map using Sobel operator.

    Args:
        depth: Depth map of shape (B, 1, H, W).
        threshold: Edge magnitude threshold (relative to max).

    Returns:
        Binary edge mask of shape (B, 1, H, W), values in [0, 1].
    """
    sobel_x = _SOBEL_X.to(device=depth.device, dtype=depth.dtype)
    sobel_y = _SOBEL_Y.to(device=depth.device, dtype=depth.dtype)

    grad_x = F.conv2d(depth, sobel_x, padding=1)
    grad_y = F.conv2d(depth, sobel_y, padding=1)
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    # Normalize per-sample and threshold
    b = magnitude.shape[0]
    mag_flat = magnitude.reshape(b, -1)
    max_mag = mag_flat.max(dim=-1, keepdim=True).values.reshape(b, 1, 1, 1)
    max_mag = max_mag.clamp(min=1e-8)
    norm_mag = magnitude / max_mag

    return (norm_mag > threshold).float()


def _flatten_depth_to_patches(
    depth: torch.Tensor,
    num_patches: int,
) -> torch.Tensor:
    """Downsample depth map to match patch count via adaptive pooling.

    Args:
        depth: Depth map of shape (B, 1, H, W).
        num_patches: Target number of spatial positions N.

    Returns:
        Flattened depth of shape (B, N).
    """
    b = depth.shape[0]
    # Infer spatial dims: assume roughly square-ish or 1:2 ratio
    h_w_ratio = depth.shape[2] / depth.shape[3]
    patch_w = max(1, int((num_patches / h_w_ratio) ** 0.5))
    patch_h = max(1, num_patches // patch_w)
    # Adjust to get exact count
    while patch_h * patch_w < num_patches:
        patch_h += 1

    pooled = F.adaptive_avg_pool2d(depth, (patch_h, patch_w))  # (B, 1, pH, pW)
    flat = pooled.reshape(b, -1)  # (B, pH*pW)

    # Trim or pad to exact num_patches
    if flat.shape[1] >= num_patches:
        return flat[:, :num_patches]
    pad = torch.zeros(b, num_patches - flat.shape[1], device=depth.device, dtype=depth.dtype)
    return torch.cat([flat, pad], dim=1)


def _flatten_edges_to_patches(
    edge_mask: torch.Tensor,
    num_patches: int,
) -> torch.Tensor:
    """Downsample edge mask to match patch count via max pooling.

    Uses max pooling so any edge pixel within a patch region marks
    that patch as boundary-adjacent.

    Args:
        edge_mask: Binary edge mask (B, 1, H, W).
        num_patches: Target number of spatial positions N.

    Returns:
        Flattened edge indicator of shape (B, N), values in {0, 1}.
    """
    b = edge_mask.shape[0]
    h_w_ratio = edge_mask.shape[2] / edge_mask.shape[3]
    patch_w = max(1, int((num_patches / h_w_ratio) ** 0.5))
    patch_h = max(1, num_patches // patch_w)
    while patch_h * patch_w < num_patches:
        patch_h += 1

    pooled = F.adaptive_max_pool2d(edge_mask, (patch_h, patch_w))
    flat = pooled.reshape(b, -1)

    if flat.shape[1] >= num_patches:
        return flat[:, :num_patches]
    pad = torch.zeros(b, num_patches - flat.shape[1], device=edge_mask.device, dtype=edge_mask.dtype)
    return torch.cat([flat, pad], dim=1)


@register_loss("depth_contrastive")
class DepthContrastiveLoss(nn.Module):
    """Enhanced depth-guided contrastive loss.

    Combines attraction for same-depth pairs with repulsion across
    depth discontinuities. Boundary-adjacent pairs receive boosted
    weighting to sharpen segmentation edges.

    Args:
        sigma_d: Depth similarity bandwidth.
        num_pairs: Number of random pixel pairs to sample per image.
        boundary_weight: Multiplier for boundary-adjacent pairs.
        repulsion_margin: Cosine similarity threshold for repulsion.
        edge_threshold: Relative threshold for Sobel edge detection.
        depth_disc_threshold: Depth weight threshold below which
            pairs are treated as cross-discontinuity (repulsion).
    """

    def __init__(
        self,
        sigma_d: float = 0.5,
        num_pairs: int = 2048,
        boundary_weight: float = 2.0,
        repulsion_margin: float = 0.3,
        edge_threshold: float = 0.1,
        depth_disc_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.sigma_d = sigma_d
        self.num_pairs = num_pairs
        self.boundary_weight = boundary_weight
        self.repulsion_margin = repulsion_margin
        self.edge_threshold = edge_threshold
        self.depth_disc_threshold = depth_disc_threshold

    def forward(
        self,
        adapted_features: torch.Tensor,
        original_features: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Compute depth contrastive loss.

        Args:
            adapted_features: Adapter output (B, N, D), L2-normalized.
            original_features: Original DINOv3 features (B, N_orig, D_orig).
                Used only for scale detection; loss operates on adapted.
            depth: Depth map (B, 1, H, W).

        Returns:
            Scalar loss value.
        """
        b, n, d = adapted_features.shape
        device = adapted_features.device

        # Compute scale-aware sigma
        sigma = self._scale_aware_sigma(n, original_features.shape[1])

        # Flatten depth and edges to patch resolution
        depth_flat = _flatten_depth_to_patches(depth, n)  # (B, N)
        edge_mask = _detect_depth_edges(depth, threshold=self.edge_threshold)
        edge_flat = _flatten_edges_to_patches(edge_mask, n)  # (B, N)

        total_loss = torch.tensor(0.0, device=device)

        for i in range(b):
            loss_i = self._per_sample_loss(
                features=adapted_features[i],
                depth_vals=depth_flat[i],
                edge_vals=edge_flat[i],
                sigma=sigma,
            )
            total_loss = total_loss + loss_i

        return total_loss / b

    def _scale_aware_sigma(
        self,
        n_adapted: int,
        n_original: int,
    ) -> float:
        """Compute sigma scaled by relative resolution.

        Coarser features (fewer patches) use larger sigma to account
        for each patch covering a wider spatial area.

        Args:
            n_adapted: Number of patches in adapted features.
            n_original: Number of patches in original features.

        Returns:
            Adjusted sigma value.
        """
        if n_adapted >= n_original or n_original == 0:
            return self.sigma_d
        scale_ratio = (n_original / max(n_adapted, 1)) ** 0.5
        return self.sigma_d * scale_ratio

    def _per_sample_loss(
        self,
        features: torch.Tensor,
        depth_vals: torch.Tensor,
        edge_vals: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Compute loss for a single sample in the batch.

        Args:
            features: (N, D) L2-normalized features.
            depth_vals: (N,) flattened depth values.
            edge_vals: (N,) binary boundary indicators.
            sigma: Depth bandwidth for this scale.

        Returns:
            Scalar loss for this sample.
        """
        n = features.shape[0]
        device = features.device

        # Sample random pairs
        idx_i = torch.randint(0, n, (self.num_pairs,), device=device)
        idx_j = torch.randint(0, n, (self.num_pairs,), device=device)

        # Depth weights: w_ij = exp(-|d_i - d_j|^2 / 2*sigma^2)
        depth_diff = depth_vals[idx_i] - depth_vals[idx_j]
        w_ij = torch.exp(-depth_diff ** 2 / (2.0 * sigma ** 2))

        # Cosine similarity between paired features
        feat_i = features[idx_i]  # (num_pairs, D)
        feat_j = features[idx_j]  # (num_pairs, D)
        cos_sim = (feat_i * feat_j).sum(dim=-1)  # (num_pairs,)
        # Features are already L2-normalized, but clamp for safety
        cos_sim = cos_sim.clamp(-1.0, 1.0)

        # Boundary emphasis: boost weight when either pixel is near an edge
        is_boundary = (edge_vals[idx_i] + edge_vals[idx_j]).clamp(max=1.0)
        boundary_boost = 1.0 + (self.boundary_weight - 1.0) * is_boundary
        w_ij = w_ij * boundary_boost

        # Split into attraction (similar depth) and repulsion (discontinuity)
        attract_mask = w_ij >= self.depth_disc_threshold
        repel_mask = ~attract_mask

        # Attraction: w_ij * (1 - cos_sim)^2
        attract_loss = w_ij * (1.0 - cos_sim) ** 2
        attract_loss = attract_loss * attract_mask.float()

        # Repulsion: max(0, cos_sim - margin)^2
        repel_term = F.relu(cos_sim - self.repulsion_margin) ** 2
        repel_loss = (1.0 - w_ij) * repel_term * repel_mask.float()

        return (attract_loss + repel_loss).mean()
