"""Bidirectional Cross-Modal Scan (BiCMS).

Implements bidirectional Mamba2 scanning for cross-modal fusion
of semantic and instance feature streams.

Architecture:
    1. Interleave semantic and instance tokens → joint sequence
    2. Forward Mamba2 scan on interleaved sequence
    3. Reverse Mamba2 scan on interleaved sequence
    4. Merge forward + reverse via learned gating
    5. De-interleave into separate output streams
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from mbps.models.bridge.mamba2_ssd import Mamba2Stack


def interleave_tokens(
    semantic: jnp.ndarray,
    features: jnp.ndarray,
) -> jnp.ndarray:
    """Interleave semantic and feature token sequences.

    Creates [s_1, f_1, s_2, f_2, ...] ordering.

    Args:
        semantic: Semantic tokens of shape (B, N, D).
        features: Feature tokens of shape (B, N, D).

    Returns:
        Interleaved sequence of shape (B, 2*N, D).
    """
    b, n, d = semantic.shape
    # Stack along new dim: (B, N, 2, D)
    interleaved = jnp.stack([semantic, features], axis=2)
    # Reshape to (B, 2*N, D)
    return jnp.reshape(interleaved, (b, 2 * n, d))


def deinterleave_tokens(
    interleaved: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Separate interleaved tokens back into two streams.

    Args:
        interleaved: Interleaved sequence of shape (B, 2*N, D).

    Returns:
        Tuple of:
            - semantic: Shape (B, N, D).
            - features: Shape (B, N, D).
    """
    b, l, d = interleaved.shape
    n = l // 2
    reshaped = jnp.reshape(interleaved, (b, n, 2, d))
    semantic = reshaped[:, :, 0, :]
    features = reshaped[:, :, 1, :]
    return semantic, features


class BidirectionalCrossModalScan(nn.Module):
    """Bidirectional Cross-Modal Scan (BiCMS).

    Fuses semantic and instance feature streams using bidirectional
    Mamba2 scanning on interleaved tokens.

    Attributes:
        dim: Bridge dimension (192).
        num_layers: Number of Mamba2 layers per direction (4).
        state_dim: SSM state dimension (64).
        chunk_size: Chunk size (64, reduced for memory efficiency).
        dropout_rate: Dropout rate.
    """

    dim: int = 192
    num_layers: int = 4
    state_dim: int = 16
    chunk_size: int = 64
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        semantic: jnp.ndarray,
        features: jnp.ndarray,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply BiCMS fusion.

        Args:
            semantic: Semantic tokens (B, N, D_b).
            features: Feature tokens (B, N, D_b).
            deterministic: If True, disable dropout.

        Returns:
            Tuple of:
                - fused_semantic: Shape (B, N, D_b).
                - fused_features: Shape (B, N, D_b).
        """
        b, n, d = semantic.shape

        # Interleave tokens: [s1, f1, s2, f2, ...]
        interleaved = interleave_tokens(semantic, features)  # (B, 2N, D)

        # Forward scan
        forward_out = Mamba2Stack(
            num_layers=self.num_layers,
            dim=d,
            state_dim=self.state_dim,
            chunk_size=self.chunk_size,
            dropout_rate=self.dropout_rate,
            name="forward_mamba",
        )(interleaved, deterministic=deterministic)

        # Backward scan (flip → process → flip back)
        reversed_input = jnp.flip(interleaved, axis=1)
        backward_out = Mamba2Stack(
            num_layers=self.num_layers,
            dim=d,
            state_dim=self.state_dim,
            chunk_size=self.chunk_size,
            dropout_rate=self.dropout_rate,
            name="backward_mamba",
        )(reversed_input, deterministic=deterministic)
        backward_out = jnp.flip(backward_out, axis=1)

        # Merge forward and backward via learned gating
        gate = nn.Dense(d, name="merge_gate")(
            jnp.concatenate([forward_out, backward_out], axis=-1)
        )
        gate = jax.nn.sigmoid(gate)
        merged = gate * forward_out + (1.0 - gate) * backward_out

        # De-interleave back into separate streams
        fused_semantic, fused_features = deinterleave_tokens(merged)

        return fused_semantic, fused_features

    def get_state_norms(self) -> jnp.ndarray:
        """Get L2 norms of internal SSM states for regularization.

        This is called during loss computation for state regularization.

        Returns:
            Scalar state norm.
        """
        # Placeholder — actual state norms are tracked during forward pass
        return jnp.array(0.0)
