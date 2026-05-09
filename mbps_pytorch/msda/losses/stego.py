"""STEGO correspondence loss wrapper for MSDA adapter features.

Wraps the existing stego_loss() from stego_loss.py, adapting it for
the MSDA interface where adapted features may have different spatial
resolution than the original DINOv3 teacher features.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_loss
from ...models.semantic.stego_loss import find_knn_pairs, stego_loss

logger = logging.getLogger(__name__)


def _downsample_features_to_match(
    features: torch.Tensor,
    target_n: int,
    spatial_h: Optional[int] = None,
    spatial_w: Optional[int] = None,
) -> torch.Tensor:
    """Downsample (B, N, D) features to (B, target_n, D).

    Reshapes to spatial grid, applies adaptive average pooling,
    and flattens back to sequence form.

    Args:
        features: Input features (B, N, D).
        target_n: Target number of spatial positions.
        spatial_h: Known spatial height. If None, inferred.
        spatial_w: Known spatial width. If None, inferred.

    Returns:
        Downsampled features (B, target_n, D).
    """
    b, n, d = features.shape
    if n == target_n:
        return features

    # Infer spatial dimensions if not provided
    if spatial_h is None or spatial_w is None:
        # Assume 1:2 (H:W) ratio like Cityscapes patches
        spatial_w = max(1, int(n ** 0.5 * (2 ** 0.5)))
        spatial_h = max(1, n // spatial_w)
        # Adjust to cover all tokens
        while spatial_h * spatial_w < n:
            spatial_h += 1

    # Compute target spatial dims preserving aspect ratio
    ratio = spatial_h / max(spatial_w, 1)
    target_w = max(1, int((target_n / ratio) ** 0.5))
    target_h = max(1, target_n // target_w)
    while target_h * target_w < target_n:
        target_h += 1

    # Reshape to spatial, pool, reshape back
    # Pad if n < spatial_h * spatial_w
    total = spatial_h * spatial_w
    if n < total:
        pad = torch.zeros(b, total - n, d, device=features.device, dtype=features.dtype)
        features_padded = torch.cat([features, pad], dim=1)
    else:
        features_padded = features[:, :total]

    spatial = features_padded.reshape(b, spatial_h, spatial_w, d)
    spatial = spatial.permute(0, 3, 1, 2)  # (B, D, H, W)
    pooled = F.adaptive_avg_pool2d(spatial, (target_h, target_w))
    result = pooled.permute(0, 2, 3, 1).reshape(b, target_h * target_w, d)

    return result[:, :target_n]


@register_loss("stego")
class StegoCorrespondenceLoss(nn.Module):
    """STEGO InfoNCE loss adapted for the MSDA pipeline.

    Uses original DINOv3 features as the teacher signal for KNN
    positive pair mining, then computes InfoNCE on the adapter's
    output features.

    When the adapter produces a different number of spatial tokens
    than the original backbone (e.g., from multi-scale pooling),
    adapted features are downsampled to match before computing loss.

    Args:
        temperature: InfoNCE temperature.
        knn_k: Number of nearest neighbors for positive mining.
        num_negatives: Number of negative samples in InfoNCE.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        knn_k: int = 7,
        num_negatives: int = 64,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.knn_k = knn_k
        self.num_negatives = num_negatives

    def forward(
        self,
        adapted_features: torch.Tensor,
        original_features: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Compute STEGO correspondence loss.

        Args:
            adapted_features: Adapter output (B, N, D), L2-normalized.
            original_features: Original DINOv3 features (B, N_orig, D_orig).
                Used as teacher for KNN positive pair mining.
            depth: Depth map (B, 1, H, W). Unused; kept for interface
                consistency with other MSDA losses.

        Returns:
            Scalar STEGO loss.
        """
        n_adapted = adapted_features.shape[1]
        n_original = original_features.shape[1]

        # If spatial resolutions differ, downsample adapted to match original
        if n_adapted != n_original:
            logger.debug(
                "Downsampling adapted features from N=%d to N=%d "
                "to match original resolution",
                n_adapted,
                n_original,
            )
            aligned_features = _downsample_features_to_match(
                adapted_features, target_n=n_original
            )
        else:
            aligned_features = adapted_features

        # Delegate to existing stego_loss: uses original_features as DINO
        # teacher for KNN, computes InfoNCE on aligned adapter features
        return stego_loss(
            semantic_codes=aligned_features,
            dino_features=original_features,
            temperature=self.temperature,
            knn_k=self.knn_k,
            num_negatives=self.num_negatives,
        )
