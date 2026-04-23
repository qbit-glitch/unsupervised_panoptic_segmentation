"""Frozen-Feature Depth Adapter.

Learns a small residual correction to frozen CAUSE 90D codes conditioned on
depth, via a skip connection. Zero-initialized final layer ensures the adapter
starts as identity — it can only help, never hurt.

Supports variable depth input (1D raw or 16D sinusoidal), hidden dimension,
and number of hidden layers.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn


SINUSOIDAL_FREQS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


def sinusoidal_depth_encode(depth_flat: torch.Tensor) -> torch.Tensor:
    """Sinusoidal positional encoding of depth values.

    Args:
        depth_flat: (B, N) or (B, N, 1) raw depth values in [0, 1].

    Returns:
        (B, N, 16) sinusoidal-encoded depth features.
    """
    if depth_flat.dim() == 3:
        depth_flat = depth_flat.squeeze(-1)  # (B, N)
    encodings: List[torch.Tensor] = []
    for freq in SINUSOIDAL_FREQS:
        encodings.append(torch.sin(freq * np.pi * depth_flat))
        encodings.append(torch.cos(freq * np.pi * depth_flat))
    return torch.stack(encodings, dim=-1)  # (B, N, 16)


class DepthAdapter(nn.Module):
    """MLP adapter: frozen 90D codes + depth -> adjusted 90D codes.

    Attributes:
        code_dim: Input/output code dimension (90).
        depth_dim: Depth feature dimension (1 for raw, 16 for sinusoidal).
        hidden_dim: Hidden layer dimension.
        num_layers: Number of hidden layers (minimum 1).
    """

    def __init__(
        self,
        code_dim: int = 90,
        depth_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.depth_dim = depth_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers: List[nn.Module] = []
        in_dim = code_dim + depth_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.out_linear = nn.Linear(hidden_dim, code_dim)

        # Zero-init final layer: adapter starts as identity
        nn.init.zeros_(self.out_linear.weight)
        nn.init.zeros_(self.out_linear.bias)

    def forward(self, codes: torch.Tensor, depth: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute depth-adjusted codes via skip connection.

        Args:
            codes: Frozen CAUSE codes of shape (B, N, code_dim).
            depth: Depth values of shape (B, N, depth_dim) or (B, N).
            **kwargs: Ignored (accepted for interface compatibility with v2 variants).

        Returns:
            Adjusted codes of shape (B, N, code_dim).
        """
        if depth.dim() == 2:
            depth = depth.unsqueeze(-1)  # (B, N) -> (B, N, 1)

        x = torch.cat([codes, depth], dim=-1)  # (B, N, code_dim + depth_dim)
        x = self.mlp(x)
        residual = self.out_linear(x)  # (B, N, code_dim), starts at zero

        return codes + residual
