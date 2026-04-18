"""Spatial Prior Module (ViT-Adapter, ICLR 2023, Chen et al.).

4-stage convolutional stem producing (c2, c3, c4) at strides (4, 8, 16)
from raw RGB, projected to the ViT embedding dim so they can be cross-
attended with DINOv3 patch tokens in the Injector/Extractor.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

__all__ = ["SpatialPriorModule"]


def _conv_bn_relu(cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SpatialPriorModule(nn.Module):
    """Convolutional stem emitting c2 (stride 4), c3 (stride 8), c4 (stride 16)."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 768, hidden: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            _conv_bn_relu(in_channels, hidden, k=3, s=2, p=1),  # /2
            _conv_bn_relu(hidden, hidden, k=3, s=1, p=1),
            _conv_bn_relu(hidden, hidden, k=3, s=1, p=1),
            _conv_bn_relu(hidden, hidden, k=3, s=2, p=1),  # /4
        )
        self.conv2 = _conv_bn_relu(hidden, hidden * 2, k=3, s=2, p=1)  # /8
        self.conv3 = _conv_bn_relu(hidden * 2, hidden * 4, k=3, s=2, p=1)  # /16
        self.proj2 = nn.Conv2d(hidden, embed_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(hidden * 2, embed_dim, kernel_size=1)
        self.proj4 = nn.Conv2d(hidden * 4, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)                       # B, hidden, H/4, W/4
        c2 = self.proj2(x)                     # B, embed, H/4, W/4
        x2 = self.conv2(x)                     # B, hidden*2, H/8, W/8
        c3 = self.proj3(x2)                    # B, embed, H/8, W/8
        x3 = self.conv3(x2)                    # B, hidden*4, H/16, W/16
        c4 = self.proj4(x3)                    # B, embed, H/16, W/16
        return c2, c3, c4
