"""DINOv3-to-DINOv2 feature alignment adapter for CAUSE-TR Mode B retraining.

The CAUSE-TR codebook (2048, 768) and cluster_probe (27, 90) were trained against
DINOv2 ViT-B/14 features. When we swap the backbone to DINOv3 ViT-B/16, the 768-d
features live in a different (but same-dimensional) manifold. A small linear layer
initialized to identity can rotate/scale DINOv3 features into the codebook's
geometry without disturbing it at step 0.

This module is trainable; the codebook and cluster_probe downstream are frozen.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

InitMode = Literal["identity", "kaiming"]


class DINOv3ToDINOv2Adapter(nn.Module):
    """Linear(768->768) alignment adapter, identity-initialized by default.

    A single shared instance feeds both student and EMA-teacher paths in the
    Segment_TR forward — adding an EMA twin would create a student/teacher gap
    that the frozen codebook cannot absorb.

    Parameter count: 768*768 + 768 = 590,592.
    """

    def __init__(self, dim: int = 768, bias: bool = True, init: InitMode = "identity") -> None:
        super().__init__()
        self.dim = dim
        self.fc = nn.Linear(dim, dim, bias=bias)
        if init == "identity":
            nn.init.eye_(self.fc.weight)
            if bias:
                nn.init.zeros_(self.fc.bias)
        elif init == "kaiming":
            nn.init.kaiming_normal_(self.fc.weight)
            if bias:
                nn.init.zeros_(self.fc.bias)
        else:
            raise ValueError(f"Unknown init mode: {init}")
        logger.info(
            "DINOv3ToDINOv2Adapter built: dim=%d bias=%s init=%s params=%d",
            dim, bias, init, self.num_parameters(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, dim) -> (B, N, dim)."""
        return self.fc(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["DINOv3ToDINOv2Adapter"]
