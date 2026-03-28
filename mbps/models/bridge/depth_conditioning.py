"""Unified Depth Conditioning Module (UDCM).

Encodes monocular depth via sinusoidal positional encoding and
modulates semantic/feature branch representations through an
FiLM-style gating mechanism.

Mathematical specification:
    γ(D) = [sin(2^k π D), cos(2^k π D)]_{k=0}^{K-1}  (sinusoidal encoding)
    X_conditioned = X ⊙ (1 + W_γ · γ(D)) + W_β · γ(D)  (FiLM modulation)
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


def sinusoidal_depth_encoding(
    depth: jnp.ndarray,
    freq_bands: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
) -> jnp.ndarray:
    """Compute sinusoidal positional encoding of depth values.

    γ(D) = [sin(2^k π D), cos(2^k π D)] for k in freq_bands.

    Args:
        depth: Depth values of shape (B, N) or (B, H, W), in [0, 1].
        freq_bands: Frequency bands for encoding.

    Returns:
        Depth encoding of shape (B, ..., 2*len(freq_bands)).
    """
    encodings = []
    for freq in freq_bands:
        encodings.append(jnp.sin(freq * jnp.pi * depth))
        encodings.append(jnp.cos(freq * jnp.pi * depth))
    return jnp.stack(encodings, axis=-1)


class UnifiedDepthConditioning(nn.Module):
    """Unified Depth Conditioning Module (UDCM).

    Conditions both semantic and feature branches on monocular
    depth information using FiLM (Feature-wise Linear Modulation).

    Attributes:
        bridge_dim: Bridge feature dimension (192).
        freq_bands: Sinusoidal encoding frequencies.
        depth_mlp_dims: Hidden dims for depth processing MLP.
    """

    bridge_dim: int = 192
    freq_bands: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    depth_mlp_dims: Optional[Tuple[int, ...]] = None

    @nn.compact
    def __call__(
        self,
        depth: jnp.ndarray,
        semantic_proj: jnp.ndarray,
        feature_proj: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Condition features on depth.

        Args:
            depth: Depth values of shape (B, N), in [0, 1].
            semantic_proj: Projected semantic of shape (B, N, D_b).
            feature_proj: Projected features of shape (B, N, D_b).

        Returns:
            Tuple of:
                - sem_conditioned: Depth-conditioned semantic (B, N, D_b).
                - feat_conditioned: Depth-conditioned features (B, N, D_b).
                - depth_loss: Depth consistency regularization loss.
        """
        # Auto-scale MLP dims with bridge_dim if not explicitly set
        mlp_dims = self.depth_mlp_dims or (64, self.bridge_dim)

        # Sinusoidal depth encoding
        depth_enc = sinusoidal_depth_encoding(depth, self.freq_bands)
        # (B, N, 2*num_freqs)

        # Process depth encoding through MLP
        d = depth_enc
        for i, dim in enumerate(mlp_dims):
            d = nn.Dense(dim, name=f"depth_mlp_{i}")(d)
            d = jax.nn.relu(d)

        # Ensure output dim matches bridge_dim
        d = nn.Dense(self.bridge_dim, name="depth_proj")(d)

        # FiLM modulation parameters
        # Semantic branch: γ_s, β_s — clamped to prevent feature explosion
        gamma_s = jnp.clip(
            nn.Dense(self.bridge_dim, name="gamma_s")(d) + 1.0, 0.1, 5.0
        )
        beta_s = jnp.clip(
            nn.Dense(self.bridge_dim, name="beta_s")(d), -5.0, 5.0
        )

        # Feature branch: γ_f, β_f — clamped to prevent feature explosion
        gamma_f = jnp.clip(
            nn.Dense(self.bridge_dim, name="gamma_f")(d) + 1.0, 0.1, 5.0
        )
        beta_f = jnp.clip(
            nn.Dense(self.bridge_dim, name="beta_f")(d), -5.0, 5.0
        )

        # Apply FiLM: x_cond = x * gamma + beta
        sem_conditioned = semantic_proj * gamma_s + beta_s
        feat_conditioned = feature_proj * gamma_f + beta_f

        # Depth consistency loss: nearby depth → similar modulation
        # Encourage smooth depth conditioning
        depth_grad_x = jnp.diff(depth, axis=-1)
        mod_grad_x_s = jnp.diff(
            jnp.mean(gamma_s, axis=-1), axis=-1
        )
        depth_loss = jnp.mean(
            jnp.abs(mod_grad_x_s)
            * jnp.exp(-jnp.clip(jnp.abs(depth_grad_x), 0.0, 10.0) * 5.0)
        )

        return sem_conditioned, feat_conditioned, depth_loss
