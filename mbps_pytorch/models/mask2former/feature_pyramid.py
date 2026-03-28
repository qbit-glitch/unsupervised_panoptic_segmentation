"""Simple Feature Pyramid for ViT → multi-scale features.

Converts single-scale ViT patch features from multiple intermediate layers
into a 4-level feature pyramid at strides {4, 8, 16, 32}, following ViTDet
(Li et al., ECCV 2022).

Input: 4 feature maps from DINOv3 ViT-L/16 layers [4, 11, 17, 23],
       each (B, 1024, H/16, W/16) = (B, 1024, 32, 64) for 512x1024 images.
Output: {
    "1/4":  (B, 256, H/4,  W/4)  = (B, 256, 128, 256)
    "1/8":  (B, 256, H/8,  W/8)  = (B, 256,  64, 128)
    "1/16": (B, 256, H/16, W/16) = (B, 256,  32,  64)
    "1/32": (B, 256, H/32, W/32) = (B, 256,  16,  32)
}
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class SimpleFeaturePyramid(nn.Module):
    """Convert multi-layer ViT features to a multi-scale FPN.

    Args:
        in_dim: ViT feature dimension (1024 for ViT-L).
        out_dim: Output FPN channel dimension (256).
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim

        # 1/4 scale: project + 4x upsample (two 2x ConvTranspose)
        self.adapter_1_4 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
            nn.ConvTranspose2d(out_dim, out_dim, kernel_size=2, stride=2),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
            nn.ConvTranspose2d(out_dim, out_dim, kernel_size=2, stride=2),
        )

        # 1/8 scale: project + 2x upsample
        self.adapter_1_8 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
            nn.ConvTranspose2d(out_dim, out_dim, kernel_size=2, stride=2),
        )

        # 1/16 scale: project only (native ViT resolution)
        self.adapter_1_16 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.GroupNorm(32, out_dim),
        )

        # 1/32 scale: project + 2x downsample
        self.adapter_1_32 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, multi_layer_features: List[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert 4 ViT layer features to multi-scale pyramid.

        Args:
            multi_layer_features: List of 4 tensors, each (B, C, H/16, W/16).
                From ViT layers [4, 11, 17, 23].

        Returns:
            Dict with keys "1/4", "1/8", "1/16", "1/32", each (B, out_dim, H_i, W_i).
        """
        assert len(multi_layer_features) == 4, f"Expected 4 features, got {len(multi_layer_features)}"

        f4, f11, f17, f23 = multi_layer_features

        return {
            "1/4": self.adapter_1_4(f4),
            "1/8": self.adapter_1_8(f11),
            "1/16": self.adapter_1_16(f17),
            "1/32": self.adapter_1_32(f23),
        }
