# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry

from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "SEM_SEG_HEADS_REGISTRY",
    "CustomSemSegFPNHead",
    "build_sem_seg_head",
]

SEM_SEG_HEADS_REGISTRY = Registry("SEM_SEG_HEADS")
SEM_SEG_HEADS_REGISTRY.__doc__ = """
Registry for semantic segmentation heads, which make semantic segmentation predictions
from feature maps.
"""


def build_sem_seg_head(cfg, input_shape):
    """Build a semantic segmentation head from `cfg.MODEL.SEM_SEG_HEAD.NAME`."""
    name = cfg.MODEL.SEM_SEG_HEAD.NAME
    return SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)


@SEM_SEG_HEADS_REGISTRY.register()
class CustomSemSegFPNHead(nn.Module):
    """A semantic segmentation head described in :paper:`PanopticFPN`.

    It takes a list of FPN features as input, and applies a sequence of
    3x3 convs and upsampling to scale all of them to the stride defined by
    ``common_stride``. Then these features are added and used to make final
    predictions by another 1x1 conv layer.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        conv_dims: int,
        common_stride: int,
        loss_weight: float = 1.0,
        norm: Optional[Union[str, Callable]] = None,
        class_weight: Optional[Tuple[float, ...]] = None,
        ignore_value: int = -1,
        stuff_kd_weight: float = 0.0,
        kd_temperature: float = 2.0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            conv_dims: number of output channels for the intermediate conv layers.
            common_stride: the common stride that all features will be upscaled to
            loss_weight: loss weight
            norm (str or callable): normalization for all conv layers
            class_weight: Class-wise weighting factors
            ignore_value: category id to be ignored during training.
            stuff_kd_weight: weight for stuff-preservation KD loss. 0.0 disables.
            kd_temperature: temperature for softening pseudo-label targets in KD.
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        if not len(input_shape):
            raise ValueError("SemSegFPNHead(input_shape=) cannot be empty!")
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = common_stride
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.stuff_kd_weight = stuff_kd_weight
        self.kd_temperature = kd_temperature

        self.scale_heads = []
        for in_feature, stride, channels in zip(self.in_features, feature_strides, feature_channels):
            head_ops = []
            head_length = max(1, int(np.log2(stride) - np.log2(self.common_stride)))
            for k in range(head_length):
                norm_module = get_norm(norm, conv_dims)
                conv = Conv2d(
                    channels if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if stride != self.common_stride:
                    head_ops.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "input_shape": {k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES},
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "conv_dims": cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            "common_stride": cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            "norm": cfg.MODEL.SEM_SEG_HEAD.NORM,
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "class_weight": cfg.MODEL.SEM_SEG_HEAD.CLASS_WEIGHT,
            "stuff_kd_weight": getattr(cfg.MODEL.SEM_SEG_HEAD, "STUFF_KD_WEIGHT", 0.0),
            "kd_temperature": getattr(cfg.MODEL.SEM_SEG_HEAD, "KD_TEMPERATURE", 2.0),
        }

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor] = None,
        pixel_weights: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        pseudo_onehot: Optional[torch.Tensor] = None,
    ):
        """Forward pass for semantic segmentation head.

        Args:
            features: dict of FPN features.
            targets: (B, H, W) ground truth semantic labels, or None for inference.
            pixel_weights: optional (B, H, W) per-pixel loss weights from M5
                confidence-weighted loss mitigation. None preserves original behavior.
            depth: unused in base class, accepted for API compatibility with
                DepthFiLMSemSegHead subclass.
            pseudo_onehot: optional (B, C, H, W) one-hot pseudo-label targets
                for stuff-preservation KD loss.

        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = self.layers(features)
        if self.training:
            return None, self.losses(
                x, targets, pixel_weights=pixel_weights, pseudo_onehot=pseudo_onehot,
            )
        else:
            x = F.interpolate(x, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
            return x, {}

    def layers(self, features):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        x = self.predictor(x)
        return x

    def losses(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pixel_weights: Optional[torch.Tensor] = None,
        pseudo_onehot: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute semantic segmentation loss with optional per-pixel weighting.

        Args:
            predictions: (B, C, H', W') logits from the FPN head.
            targets: (B, H, W) ground truth class indices.
            pixel_weights: optional (B, H, W) per-pixel confidence weights
                from M5 mitigation. When None, uses standard mean reduction.
            pseudo_onehot: optional (B, C, H, W) one-hot pseudo-label targets
                for stuff-preservation KD loss. Only used when stuff_kd_weight > 0.

        Returns:
            Dict of loss tensors.
        """
        predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=False,
        )

        class_weight = (
            torch.tensor(self.class_weight, device=predictions.device)
            if self.class_weight is not None
            else None
        )

        if pixel_weights is not None:
            # M5: Confidence-weighted loss — use reduction="none" and apply weights
            # Resize pixel_weights to match predictions spatial size
            if pixel_weights.shape[-2:] != predictions.shape[-2:]:
                pixel_weights = F.interpolate(
                    pixel_weights.unsqueeze(1).float(),
                    size=predictions.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            loss_unreduced = F.cross_entropy(
                predictions,
                targets,
                reduction="none",
                ignore_index=self.ignore_value,
                weight=class_weight,
            )  # (B, H, W)
            valid = (targets != self.ignore_value).float()
            weighted = loss_unreduced * pixel_weights * valid
            denom = valid.sum().clamp(min=1.0)
            loss = weighted.sum() / denom
        else:
            loss = F.cross_entropy(
                predictions,
                targets,
                reduction="mean",
                ignore_index=self.ignore_value,
                weight=class_weight,
            )
        losses = {"loss_sem_seg": loss * self.loss_weight}

        # Stuff-preservation KD loss: KL divergence on stuff pixels only
        if self.stuff_kd_weight > 0.0 and pseudo_onehot is not None:
            kd_loss = self._compute_stuff_kd_loss(predictions, targets, pseudo_onehot)
            losses["loss_stuff_kd"] = kd_loss * self.stuff_kd_weight

        return losses

    def _compute_stuff_kd_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pseudo_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL-divergence KD loss on stuff pixels only.

        Stuff pixels are those with target class < 11 (stuff classes in CUPS
        27-class mapping) and not equal to ignore_value.

        Args:
            predictions: (B, C, H, W) logits at full resolution.
            targets: (B, H, W) ground truth class indices.
            pseudo_onehot: (B, C, H, W) one-hot pseudo-label targets.

        Returns:
            Scalar KD loss tensor.
        """
        t = self.kd_temperature
        num_classes = predictions.shape[1]

        # Resize pseudo_onehot to match predictions spatial size if needed
        if pseudo_onehot.shape[-2:] != predictions.shape[-2:]:
            pseudo_onehot = F.interpolate(
                pseudo_onehot.float(),
                size=predictions.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        # Align channel count (pseudo_onehot may differ from prediction head)
        if pseudo_onehot.shape[1] != num_classes:
            c_oh = pseudo_onehot.shape[1]
            if c_oh < num_classes:
                pseudo_onehot = F.pad(pseudo_onehot, (0, 0, 0, 0, 0, num_classes - c_oh))
            else:
                pseudo_onehot = pseudo_onehot[:, :num_classes]

        # Stuff mask: class > 0 (things=0, void=255), not ignored
        stuff_mask = (targets > 0) & (targets != self.ignore_value)
        if stuff_mask.sum() == 0:
            return predictions.sum() * 0.0  # zero loss, preserves grad graph

        # Student: softened log-probabilities
        log_student = F.log_softmax(predictions / t, dim=1)
        # Teacher: softened pseudo-label distribution
        teacher = F.softmax(pseudo_onehot / t, dim=1)

        # Per-pixel KL divergence: (B, H, W) — sum over class dimension
        kl_per_pixel = F.kl_div(log_student, teacher, reduction="none").sum(dim=1)

        # Mask to stuff pixels only and average
        kl_stuff = (kl_per_pixel * stuff_mask.float()).sum()
        kd_loss = kl_stuff / stuff_mask.float().sum().clamp(min=1.0)
        # Scale by T^2 (standard KD practice)
        kd_loss = kd_loss * (t ** 2)

        return kd_loss
