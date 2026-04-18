"""ViT-Adapter wrapper (Chen et al., ICLR 2023).

Frozen DINOv3 backbone + SPM + num_blocks (Injector, Extractor) pairs.
Final output is a 4-level FPN-style dict {p2, p3, p4, p5} at strides
(4, 8, 16, 32), each with pyramid_channels=256.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .extractor import Extractor
from .injector import Injector
from .spm import SpatialPriorModule

__all__ = ["ViTAdapter"]


class ViTAdapter(nn.Module):
    """Frozen ViT backbone + SPM + alternating Injector/Extractor blocks.

    Emits a 4-level FPN dict ``{p2, p3, p4, p5}`` at strides (4, 8, 16, 32),
    each with ``pyramid_channels`` channels, ready for the MSDeformAttn
    pixel decoder.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int = 768,
        num_blocks: int = 4,
        num_heads: int = 8,
        pyramid_channels: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.spm = SpatialPriorModule(in_channels=3, embed_dim=embed_dim)
        self.injectors = nn.ModuleList([Injector(embed_dim, num_heads) for _ in range(num_blocks)])
        self.extractors = nn.ModuleList([Extractor(embed_dim, num_heads) for _ in range(num_blocks)])
        # Project embed_dim -> pyramid_channels per level.
        self.proj_p2 = nn.Conv2d(embed_dim, pyramid_channels, kernel_size=1)
        self.proj_p3 = nn.Conv2d(embed_dim, pyramid_channels, kernel_size=1)
        self.proj_p4 = nn.Conv2d(embed_dim, pyramid_channels, kernel_size=1)
        # p5 is a strided conv on p4.
        self.to_p5 = nn.Conv2d(pyramid_channels, pyramid_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        vit = self.backbone(x)                       # B, embed, H/16, W/16
        # DINOv3ViTBackbone subclasses detectron2.Backbone, which mandates
        # a Dict[str, Tensor] return. Unwrap to the single feature tensor.
        # Mocks in unit tests may return a raw tensor, so stay permissive.
        if isinstance(vit, dict):
            vit = vit.get("dinov3", next(iter(vit.values())))
        c2, c3, c4 = self.spm(x)                     # strides 4/8/16
        for inj, ext in zip(self.injectors, self.extractors):
            vit = inj(vit_feat=vit, c2=c2, c3=c3, c4=c4)
            # Extractor output overwrites c2/c3/c4 intentionally: SPM seeds
            # the pyramid, each block refines it with ViT context.
            c2, c3, c4 = ext(c2=c2, c3=c3, c4=c4, vit_feat=vit)
        p2 = self.proj_p2(c2)
        p3 = self.proj_p3(c3)
        p4 = self.proj_p4(c4)
        p5 = self.to_p5(p4)
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}
