"""Semantic Segmentation Loss.

Combines STEGO correspondence loss with DepthG depth-guided loss.

L_semantic = L_stego + λ_depthg · L_depthg
"""

from __future__ import annotations

from typing import Dict, Optional

import jax
import jax.numpy as jnp

from mbps.models.semantic.stego_loss import (
    depth_guided_correlation_loss,
    stego_loss,
)


class SemanticLoss:
    """Combined semantic segmentation loss.

    Args:
        lambda_depthg: Weight for depth-guided correlation loss.
        stego_temperature: Temperature for STEGO InfoNCE.
        knn_k: Number of nearest neighbors for STEGO.
        depth_sigma: Bandwidth for depth-guided correlation.
    """

    def __init__(
        self,
        lambda_depthg: float = 0.3,
        stego_temperature: float = 0.1,
        knn_k: int = 7,
        depth_sigma: float = 0.5,
    ):
        self.lambda_depthg = lambda_depthg
        self.stego_temperature = stego_temperature
        self.knn_k = knn_k
        self.depth_sigma = depth_sigma

    def __call__(
        self,
        semantic_codes: jnp.ndarray,
        dino_features: jnp.ndarray,
        depth: Optional[jnp.ndarray] = None,
        key: Optional[jax.Array] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Compute combined semantic loss.

        Args:
            semantic_codes: Codes of shape (B, N, 90).
            dino_features: DINO features of shape (B, N, 384).
            depth: Optional depth values of shape (B, N).
            key: PRNG key for stochastic operations.

        Returns:
            Dictionary with loss components and total.
        """
        losses = {}

        # STEGO correspondence loss
        l_stego = stego_loss(
            semantic_codes,
            dino_features,
            temperature=self.stego_temperature,
            knn_k=self.knn_k,
        )
        losses["stego"] = l_stego

        # DepthG depth-guided correlation loss
        if depth is not None:
            l_depthg = depth_guided_correlation_loss(
                semantic_codes,
                depth,
                sigma_d=self.depth_sigma,
                key=key,
            )
            losses["depthg"] = l_depthg
        else:
            l_depthg = jnp.array(0.0)
            losses["depthg"] = l_depthg

        # Total
        losses["total"] = l_stego + self.lambda_depthg * l_depthg

        return losses
