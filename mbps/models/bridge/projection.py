"""Adaptive Projection Bridge (APB).

Projects semantic codes (90-dim) and DINO features (384-dim) into a
shared bridge dimension (192-dim) for Mamba2 fusion. Includes learned
inverse projections back to original dimensions.

Mathematical specification:
    S' = W_s · S + b_s  where W_s ∈ ℝ^{D_b × D_s}
    F' = W_f · F + b_f  where W_f ∈ ℝ^{D_b × D_f}
    L_align = ||S'||_F + ||F'||_F (Frobenius norm reg)
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


class AdaptiveProjectionBridge(nn.Module):
    """Adaptive Projection Bridge (APB).

    Projects heterogeneous feature streams into a common dimension
    for cross-modal fusion.

    Attributes:
        semantic_dim: Input semantic code dimension (90).
        feature_dim: Input DINO feature dimension (384).
        bridge_dim: Shared bridge dimension (192).
    """

    semantic_dim: int = 90
    feature_dim: int = 384
    bridge_dim: int = 192

    @nn.compact
    def __call__(
        self,
        semantic_codes: jnp.ndarray,
        dino_features: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Project features to bridge dimension.

        Args:
            semantic_codes: Shape (B, N, 90).
            dino_features: Shape (B, N, 384).

        Returns:
            Tuple of:
                - semantic_proj: Shape (B, N, 192).
                - feature_proj: Shape (B, N, 192).
                - align_loss: Scalar alignment regularization.
        """
        # Semantic projection: 90 → 192
        sem_proj = nn.Dense(
            self.bridge_dim, name="semantic_proj"
        )(semantic_codes)
        sem_proj = nn.LayerNorm(name="sem_norm")(sem_proj)

        # Feature projection: 384 → 192
        feat_proj = nn.Dense(
            self.bridge_dim, name="feature_proj"
        )(dino_features)
        feat_proj = nn.LayerNorm(name="feat_norm")(feat_proj)

        # Alignment loss: encourage projected features to have similar norms
        sem_norm = jnp.mean(jnp.sum(sem_proj**2, axis=-1))
        feat_norm = jnp.mean(jnp.sum(feat_proj**2, axis=-1))
        align_loss = jnp.abs(sem_norm - feat_norm)

        return sem_proj, feat_proj, align_loss


class InverseProjection(nn.Module):
    """Inverse projection from bridge dimension back to original.

    Projects fused features from D_b back to original dimensions.

    Attributes:
        output_dim: Target output dimension.
    """

    output_dim: int = 384

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Inverse project from bridge dimension.

        Args:
            x: Fused features of shape (B, N, D_b).

        Returns:
            Projected features of shape (B, N, output_dim).
        """
        x = nn.Dense(self.output_dim, name="inv_proj")(x)
        x = nn.LayerNorm(name="inv_norm")(x)
        return x
