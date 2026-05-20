"""Lightweight superpixel affinity adapter for learnable instances.

This module is intentionally separate from the existing dense
``AdaptiveInstanceNet`` path.  It implements the small edge MLP used by the
superpixel-graph ablation:

    frozen DCFA/DINO/CLIP/depth features -> superpixel graph -> edge logits

Instances are recovered outside the model by thresholding predicted merge
probabilities and taking connected components over same-class superpixels.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SuperpixelAffinityAdapter(nn.Module):
    """Tiny MLP that predicts whether two adjacent superpixels should merge.

    Args:
        input_dim: Pairwise descriptor dimension.
        hidden_dim: MLP width. 128 gives a small adapter even with DINO/CLIP
            descriptors.
        num_layers: Number of hidden linear layers.
        dropout: Dropout between hidden layers.
        use_layer_norm: Whether to normalize hidden activations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.10,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """Return raw merge logits for ``edge_features``.

        Args:
            edge_features: ``(..., input_dim)`` float tensor.

        Returns:
            ``(...,)`` raw logits. Apply ``sigmoid`` for merge probabilities.
        """
        logits = self.net(edge_features).squeeze(-1)
        return logits


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross entropy with optional per-edge weights."""
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if weights is not None:
        loss = loss * weights
        return loss.sum() / weights.sum().clamp_min(1.0)
    return loss.mean()


__all__ = ["SuperpixelAffinityAdapter", "weighted_bce_with_logits"]
