"""Bidirectional Cross-Modal Scan (BiCMS).

Implements bidirectional Mamba2 scanning for cross-modal fusion
of semantic and instance feature streams.

Architecture:
    1. Interleave semantic and instance tokens -> joint sequence
    2. Forward Mamba2 scan on interleaved sequence
    3. Reverse Mamba2 scan on interleaved sequence
    4. Merge forward + reverse via learned gating
    5. De-interleave into separate output streams
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from mbps_pytorch.models.bridge.mamba2_ssd import Mamba2Stack


def interleave_tokens(
    semantic: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
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
    interleaved = torch.stack([semantic, features], dim=2)
    # Reshape to (B, 2*N, D)
    return interleaved.reshape(b, 2 * n, d)


def deinterleave_tokens(
    interleaved: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    reshaped = interleaved.reshape(b, n, 2, d)
    semantic = reshaped[:, :, 0, :]
    features = reshaped[:, :, 1, :]
    return semantic, features


class BidirectionalCrossModalScan(nn.Module):
    """Bidirectional Cross-Modal Scan (BiCMS).

    Fuses semantic and instance feature streams using bidirectional
    Mamba2 scanning on interleaved tokens.

    Args:
        dim: Bridge dimension (192).
        num_layers: Number of Mamba2 layers per direction (4).
        state_dim: SSM state dimension (64).
        chunk_size: GPU-aligned chunk size (128).
        dropout_rate: Dropout rate.
    """

    def __init__(
        self,
        dim: int = 192,
        num_layers: int = 4,
        state_dim: int = 64,
        chunk_size: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.dropout_rate = dropout_rate

        # Forward Mamba2 stack
        self.forward_mamba = Mamba2Stack(
            num_layers=num_layers,
            dim=dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            dropout_rate=dropout_rate,
        )

        # Backward Mamba2 stack
        self.backward_mamba = Mamba2Stack(
            num_layers=num_layers,
            dim=dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            dropout_rate=dropout_rate,
        )

        # Merge gate: takes concatenated forward+backward (2*dim) -> dim
        self.merge_gate = nn.Linear(2 * dim, dim)

    def forward(
        self,
        semantic: torch.Tensor,
        features: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        forward_out = self.forward_mamba(
            interleaved, deterministic=deterministic
        )

        # Backward scan (flip -> process -> flip back)
        reversed_input = torch.flip(interleaved, dims=[1])
        backward_out = self.backward_mamba(
            reversed_input, deterministic=deterministic
        )
        backward_out = torch.flip(backward_out, dims=[1])

        # Merge forward and backward via learned gating
        gate = self.merge_gate(
            torch.cat([forward_out, backward_out], dim=-1)
        )
        gate = torch.sigmoid(gate)
        merged = gate * forward_out + (1.0 - gate) * backward_out

        # De-interleave back into separate streams
        fused_semantic, fused_features = deinterleave_tokens(merged)

        return fused_semantic, fused_features

    def get_state_norms(self) -> torch.Tensor:
        """Get L2 norms of internal SSM states for regularization.

        This is called during loss computation for state regularization.

        Returns:
            Scalar state norm.
        """
        # Placeholder -- actual state norms are tracked during forward pass
        return torch.tensor(0.0)
