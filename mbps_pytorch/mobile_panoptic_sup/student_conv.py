"""Mobile conv panoptic student: RepViT-M1.5 + BiFPN + semantic head.

Phase-0 semantic-only (thing instances via connected components at inference,
see ``panoptic_postprocess_conv``). Pure conv -> trivially ONNX-exportable; the
phone-CPU-safe foil to the EoMT student. A center/offset or MaskConver head is a
Phase-2 upgrade.
"""
from __future__ import annotations

import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # mbps_pytorch on path
from train_mobile_panoptic import BiFPN  # noqa: E402  (reuse; module has no import side effects)


class ConvStudent(nn.Module):
    def __init__(self, num_classes: int = 133, fpn_dim: int = 128,
                 pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "repvit_m1_5", pretrained=pretrained, features_only=True)
        chans = self.backbone.feature_info.channels()        # [64, 128, 256, 512]
        self.neck = BiFPN(chans, fpn_dim=fpn_dim, num_repeats=2)
        self.sem_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)        # finest first
        fpn = self.neck(feats)          # per-level, fpn_dim channels
        return self.sem_head(fpn[0])    # 1/4-res logits [B, num_classes, H/4, W/4]
