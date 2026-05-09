"""Hybrid loss combining depth contrastive and SwAV clustering.

Weighted sum of DepthContrastiveLoss and SwAVSinkhornLoss for
joint depth-aware contrastive learning and prototype clustering.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from . import register_loss
from .depth_contrastive import DepthContrastiveLoss
from .swav_sinkhorn import SwAVSinkhornLoss

logger = logging.getLogger(__name__)


def _split_kwargs(
    kwargs: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split kwargs into depth-loss and swav-loss specific groups.

    Args:
        kwargs: Combined keyword arguments.

    Returns:
        Tuple of (depth_kwargs, swav_kwargs).
    """
    depth_keys = {
        "sigma_d", "num_pairs", "boundary_weight",
        "repulsion_margin", "edge_threshold", "depth_disc_threshold",
    }
    swav_keys = {
        "num_prototypes", "output_dim", "teacher_dim",
        "temperature", "sinkhorn_iters", "epsilon",
        "projection_hidden",
    }

    depth_kwargs = {k: v for k, v in kwargs.items() if k in depth_keys}
    swav_kwargs = {k: v for k, v in kwargs.items() if k in swav_keys}

    unknown = set(kwargs.keys()) - depth_keys - swav_keys
    if unknown:
        logger.warning("HybridLoss ignoring unknown kwargs: %s", unknown)

    return depth_kwargs, swav_kwargs


@register_loss("hybrid")
class HybridLoss(nn.Module):
    """Weighted combination of depth contrastive and SwAV losses.

    Enables joint training with depth-aware contrastive learning
    (local spatial coherence) and SwAV prototype clustering (global
    semantic structure).

    Args:
        alpha: Weight for depth contrastive loss.
        beta: Weight for SwAV Sinkhorn loss.
        **kwargs: Passed through to sub-losses. Keys matching
            DepthContrastiveLoss params go to depth loss; keys
            matching SwAVSinkhornLoss params go to SwAV loss.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        depth_kwargs, swav_kwargs = _split_kwargs(kwargs)

        logger.debug(
            "HybridLoss: alpha=%.2f, beta=%.2f, "
            "depth_kwargs=%s, swav_kwargs=%s",
            alpha, beta, depth_kwargs, swav_kwargs,
        )

        self.depth_loss = DepthContrastiveLoss(**depth_kwargs)
        self.swav_loss = SwAVSinkhornLoss(**swav_kwargs)

    def forward(
        self,
        adapted_features: torch.Tensor,
        original_features: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted hybrid loss.

        Args:
            adapted_features: Adapter output (B, N, D), L2-normalized.
            original_features: Original DINOv3 features (B, N_orig, D_orig).
            depth: Depth map (B, 1, H, W).

        Returns:
            Scalar combined loss: alpha * L_depth + beta * L_swav.
        """
        l_depth = self.depth_loss(adapted_features, original_features, depth)
        l_swav = self.swav_loss(adapted_features, original_features, depth)

        combined = self.alpha * l_depth + self.beta * l_swav

        logger.debug(
            "HybridLoss: L_depth=%.4f, L_swav=%.4f, combined=%.4f",
            l_depth.item(), l_swav.item(), combined.item(),
        )

        return combined
