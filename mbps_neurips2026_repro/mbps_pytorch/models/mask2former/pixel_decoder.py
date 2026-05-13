"""FPN Pixel Decoder for Mask2Former (MPS-compatible).

Replaces MSDeformAttnPixelDecoder which requires CUDA custom ops.
Uses standard top-down FPN with lateral connections.

Input: Multi-scale features from SimpleFeaturePyramid
Output:
    - mask_features: (B, mask_dim, H/4, W/4) for mask prediction
    - multi_scale_features: list of 3 feature maps at 1/8, 1/16, 1/32
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FPNPixelDecoder(nn.Module):
    """FPN-based pixel decoder (MPS-compatible, no deformable attention).

    Args:
        feature_dim: Input/output feature dimension from SimpleFeaturePyramid.
        mask_dim: Output dimension for mask features.
    """

    def __init__(self, feature_dim: int = 256, mask_dim: int = 256):
        super().__init__()

        # Lateral connections (1x1 conv to align channels)
        self.lateral_convs = nn.ModuleDict({
            "1/4": nn.Conv2d(feature_dim, feature_dim, 1),
            "1/8": nn.Conv2d(feature_dim, feature_dim, 1),
            "1/16": nn.Conv2d(feature_dim, feature_dim, 1),
            "1/32": nn.Conv2d(feature_dim, feature_dim, 1),
        })

        # Output convolutions (3x3 conv after FPN fusion)
        self.output_convs = nn.ModuleDict({
            "1/4": nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.GroupNorm(32, feature_dim),
                nn.ReLU(inplace=True),
            ),
            "1/8": nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.GroupNorm(32, feature_dim),
                nn.ReLU(inplace=True),
            ),
            "1/16": nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.GroupNorm(32, feature_dim),
                nn.ReLU(inplace=True),
            ),
            "1/32": nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.GroupNorm(32, feature_dim),
                nn.ReLU(inplace=True),
            ),
        })

        # Final mask feature projection (1/4 scale → mask features)
        self.mask_feature_proj = nn.Sequential(
            nn.Conv2d(feature_dim, mask_dim, 3, padding=1),
            nn.GroupNorm(32, mask_dim),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, features: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Apply FPN top-down pathway.

        Args:
            features: Dict with "1/4", "1/8", "1/16", "1/32" feature maps.

        Returns:
            mask_features: (B, mask_dim, H/4, W/4) for mask dot product.
            multi_scale_features: List of 3 features at [1/32, 1/16, 1/8]
                (low-res to high-res, matching Mask2Former decoder convention).
        """
        # Lateral projections
        lat = {k: self.lateral_convs[k](features[k]) for k in features}

        # Top-down pathway (coarse → fine)
        fpn = {}
        fpn["1/32"] = self.output_convs["1/32"](lat["1/32"])

        fpn["1/16"] = self.output_convs["1/16"](
            lat["1/16"] + F.interpolate(fpn["1/32"], size=lat["1/16"].shape[-2:],
                                         mode="bilinear", align_corners=False)
        )
        fpn["1/8"] = self.output_convs["1/8"](
            lat["1/8"] + F.interpolate(fpn["1/16"], size=lat["1/8"].shape[-2:],
                                        mode="bilinear", align_corners=False)
        )
        fpn["1/4"] = self.output_convs["1/4"](
            lat["1/4"] + F.interpolate(fpn["1/8"], size=lat["1/4"].shape[-2:],
                                        mode="bilinear", align_corners=False)
        )

        mask_features = self.mask_feature_proj(fpn["1/4"])

        # Multi-scale features for transformer decoder (low-res → high-res)
        multi_scale_features = [fpn["1/32"], fpn["1/16"], fpn["1/8"]]

        return mask_features, multi_scale_features
