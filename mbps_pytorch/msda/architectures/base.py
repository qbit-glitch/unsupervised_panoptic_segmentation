"""Base adapter class for MSDA architectures."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AdapterConfig:
    """Configuration for MSDA adapter architectures."""

    input_dim: int = 1024
    output_dim: int = 128
    spatial_h: int = 32
    spatial_w: int = 64
    depth_embed_dim: int = 16
    num_scales: int = 3
    hidden_dim: int = 512
    num_blocks: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    num_slots: int = 200
    slot_iters: int = 7


class DepthFiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on depth.

    Uses 1x1 convolutions instead of Linear to avoid permute operations
    that create non-contiguous tensors (which fail on MPS backward).
    """

    def __init__(self, depth_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(depth_dim, feature_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim * 2, 1),
        )

    def forward(
        self, x: torch.Tensor, depth_embed: torch.Tensor
    ) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            x: Feature map (B, C, H, W).
            depth_embed: Depth embedding (B, D, H, W).

        Returns:
            Modulated features (B, C, H, W).
        """
        params = self.net(depth_embed)  # (B, 2*C, H, W)
        gamma, beta = params.chunk(2, dim=1)  # each (B, C, H, W)
        return x * (1.0 + gamma) + beta


def sinusoidal_depth_encoding(
    depth: torch.Tensor, dim: int = 16
) -> torch.Tensor:
    """Encode scalar depth values with sinusoidal positional encoding.

    Args:
        depth: Depth map of shape (B, 1, H, W) or (B, H, W).
        dim: Embedding dimension (must be even).

    Returns:
        Depth embeddings of shape (B, dim, H, W).
    """
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)

    freqs = torch.pow(
        10000.0,
        -torch.arange(0, dim, 2, device=depth.device, dtype=depth.dtype) / dim,
    )
    freqs = freqs.view(1, -1, 1, 1)  # (1, dim//2, 1, 1)

    args = depth * freqs  # (B, dim//2, H, W)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim, H, W)


class ResBlock(nn.Module):
    """Residual convolutional block with optional FiLM conditioning."""

    def __init__(
        self,
        channels: int,
        depth_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
        self.film = DepthFiLM(depth_dim, channels) if depth_dim > 0 else None

    def forward(
        self, x: torch.Tensor, depth_embed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        if self.film is not None and depth_embed is not None:
            out = self.film(out, depth_embed)
        return torch.relu(out + residual)


class BaseAdapter(nn.Module, abc.ABC):
    """Abstract base class for all MSDA adapter architectures."""

    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__()
        self.cfg = cfg

    @abc.abstractmethod
    def forward(
        self,
        features: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Transform DINOv3 features using depth conditioning.

        Args:
            features: DINOv3 features (B, N, D) where N=H*W patches.
            depth: Depth map (B, 1, H_d, W_d) at original resolution.

        Returns:
            Transformed features (B, N_out, D_out) L2-normalized.
        """

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
