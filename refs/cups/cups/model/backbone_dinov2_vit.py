"""DINOv2 ViT-B/14 backbone wrapper for Detectron2.

Wraps DINOv2 ViT-B/14 (with registers) as a Detectron2 Backbone,
then pairs it with SimpleFeaturePyramid for multi-scale FPN features
compatible with Cascade Mask R-CNN and Panoptic FPN.

Architecture:
    Image (B, 3, H, W)  [H,W must be divisible by 14]
    → DINOv2 ViT-B/14 (frozen)
    → patch tokens (B, H/14 * W/14, 768)
    → reshape to (B, 768, H/14, W/14)
    → SimpleFeaturePyramid → {p2, p3, p4, p5} all 256-dim
"""

from __future__ import annotations

import logging
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import Backbone, BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec

logger = logging.getLogger(__name__)


class DINOv2ViTBackbone(Backbone):
    """DINOv2 ViT-B/14 with registers, wrapped as Detectron2 Backbone.

    Loads the model via torch.hub and outputs a single-scale spatial
    feature map that can be fed into SimpleFeaturePyramid.

    Args:
        model_name: torch.hub model name (default: dinov2_vitb14_reg).
        freeze: If True, freeze all backbone parameters.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14_reg",
        freeze: bool = True,
    ):
        super().__init__()
        # Load DINOv2 via torch.hub
        self.vit = torch.hub.load(
            "facebookresearch/dinov2", model_name, verbose=False,
        )
        self.embed_dim: int = self.vit.embed_dim  # 768
        self.patch_size: int = self.vit.patch_size  # 14
        self.num_register_tokens: int = getattr(self.vit, "num_register_tokens", 0)  # 4
        self._out_feature = "dinov2"

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad_(False)
            self.vit.eval()
            logger.info(
                f"DINOv2 backbone frozen: {model_name}, "
                f"embed_dim={self.embed_dim}, patch_size={self.patch_size}, "
                f"registers={self.num_register_tokens}"
            )

        self._freeze = freeze

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract spatial feature map from DINOv2.

        Args:
            x: Input images (B, 3, H, W). H, W come from ImageList (already
               padded to size_divisibility). We pad to patch_size internally,
               then resize output to H/16 x W/16 to match the reported stride.

        Returns:
            Dict with single feature: {"dinov2": (B, 768, H/16, W/16)}.
        """
        B, C, H, W = x.shape
        ps = self.patch_size  # 14

        # Pad to nearest multiple of patch_size for ViT
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        H_pad, W_pad = H + pad_h, W + pad_w
        h_p, w_p = H_pad // ps, W_pad // ps

        # Forward through DINOv2 — get patch tokens only
        if self._freeze:
            with torch.no_grad():
                features = self.vit.forward_features(x)
        else:
            features = self.vit.forward_features(x)

        patch_tokens = features["x_norm_patchtokens"]  # (B, N_patches, 768)

        # Reshape to spatial feature map at ViT's native resolution
        feat_map = patch_tokens.reshape(B, h_p, w_p, self.embed_dim)
        feat_map = feat_map.permute(0, 3, 1, 2).contiguous()  # (B, 768, h_p, w_p)

        # Resize to match the reported stride of 16.
        # ViT produces H_pad/14 patches, but we report stride=16 for FPN
        # compatibility. Resize to H/16 x W/16 so that SimpleFeaturePyramid's
        # scale_factors produce feature maps at exact strides {4,8,16,32},
        # making all head spatial operations (upsample, loss) size-consistent.
        target_h = H // 16
        target_w = W // 16
        if feat_map.shape[2] != target_h or feat_map.shape[3] != target_w:
            feat_map = F.interpolate(
                feat_map, size=(target_h, target_w),
                mode="bilinear", align_corners=False,
            )

        return {self._out_feature: feat_map}

    @property
    def size_divisibility(self) -> int:
        # Report 16 to align with FPN expectations; internal padding handles 14
        return 16

    def output_shape(self) -> Dict[str, ShapeSpec]:
        """Return output feature shapes.

        Reports stride=16 (not actual 14) so SimpleFeaturePyramid computes
        log2-contiguous strides: 4, 8, 16, 32 -> p2, p3, p4, p5.
        The actual spatial resolution (H/14) is close enough for ROI Align.

        Note: This is a method (not @property) to match Detectron2 Backbone API.
        """
        return {
            self._out_feature: ShapeSpec(
                channels=self.embed_dim,
                stride=16,
            )
        }

    def train(self, mode: bool = True):
        """Override train to keep backbone in eval if frozen."""
        super().train(mode)
        if self._freeze:
            self.vit.eval()
        return self


def build_dinov2_vitb_fpn_backbone(
    out_channels: int = 256,
    freeze: bool = True,
    model_name: str = "dinov2_vitb14_reg",
) -> Backbone:
    """Build DINOv2 ViT-B/14 + SimpleFeaturePyramid backbone.

    Returns a Backbone that outputs multi-scale FPN features:
        p2 (~stride 4), p3 (~stride 8), p4 (~stride 14), p5 (~stride 28)

    Args:
        out_channels: FPN output channels (default 256).
        freeze: Whether to freeze the DINOv2 backbone.
        model_name: DINOv2 model name.

    Returns:
        SimpleFeaturePyramid wrapping DINOv2ViTBackbone.
    """
    from detectron2.modeling.backbone.vit import SimpleFeaturePyramid
    from detectron2.modeling.backbone.fpn import LastLevelMaxPool

    backbone = DINOv2ViTBackbone(model_name=model_name, freeze=freeze)

    fpn = SimpleFeaturePyramid(
        net=backbone,
        in_feature="dinov2",
        out_channels=out_channels,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        norm="LN",
        top_block=LastLevelMaxPool(),
    )

    logger.info(
        f"Built DINOv2+SimpleFeaturePyramid: {model_name}, "
        f"out_channels={out_channels}, freeze={freeze}"
    )
    return fpn
