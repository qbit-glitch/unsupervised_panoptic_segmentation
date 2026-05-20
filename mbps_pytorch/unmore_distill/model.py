"""Compact DINOv3-S student for unMORE instance distillation."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


class DINOv3SmallFPNBackbone(nn.Module):
    """DINOv3-S feature extractor with a lightweight same-resolution FPN.

    Timm exposes DINOv3-S as a ViT/EVA backbone. Intermediate blocks have the
    same patch stride, so this module fuses several depths, then creates a small
    pyramid by up/downsampling the fused feature map.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3",
        pretrained: bool = False,
        out_indices: Tuple[int, ...] = (3, 6, 9, 11),
        fpn_dim: int = 192,
        train_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.body = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            dynamic_img_size=True,
        )
        in_channels = [info["num_chs"] for info in self.body.feature_info]
        self.laterals = nn.ModuleList([nn.Conv2d(c, fpn_dim, 1) for c in in_channels])
        self.smooth = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=fpn_dim),
            nn.Conv2d(fpn_dim, fpn_dim, 1),
            nn.GroupNorm(16, fpn_dim),
            nn.GELU(),
        )
        self.out_channels = fpn_dim

        if not train_backbone:
            for param in self.body.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.body(x)
        fused = None
        for feat, lateral in zip(features, self.laterals):
            projected = lateral(feat)
            fused = projected if fused is None else fused + projected
        assert fused is not None
        fused = self.smooth(fused / len(features))

        p3 = F.interpolate(fused, scale_factor=2.0, mode="bilinear", align_corners=False)
        p4 = fused
        p5 = F.max_pool2d(fused, kernel_size=2, stride=2)
        p6 = F.max_pool2d(p5, kernel_size=2, stride=2)
        return OrderedDict([("0", p3), ("1", p4), ("2", p5), ("3", p6)])


def build_unmore_dinov3s_maskrcnn(
    pretrained_backbone: bool = False,
    train_backbone: bool = True,
    fpn_dim: int = 192,
    min_size: int = 512,
    max_size: int = 896,
    image_mean: Iterable[float] = (0.485, 0.456, 0.406),
    image_std: Iterable[float] = (0.229, 0.224, 0.225),
) -> MaskRCNN:
    """Build a class-agnostic Mask R-CNN student around DINOv3-S."""

    backbone = DINOv3SmallFPNBackbone(
        pretrained=pretrained_backbone,
        train_backbone=train_backbone,
        fpn_dim=fpn_dim,
    )

    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64), (32, 64, 128), (64, 128, 256), (128, 256, 512)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=14,
        sampling_ratio=2,
    )

    return MaskRCNN(
        backbone,
        num_classes=2,  # background + class-agnostic object
        min_size=min_size,
        max_size=max_size,
        image_mean=list(image_mean),
        image_std=list(image_std),
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool,
        box_detections_per_img=100,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
