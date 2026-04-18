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
        aux_weights: Optional[Dict[str, float]] = None,
        aux_params: Optional[Dict[str, float]] = None,
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
            aux_weights: optional dict of P1-P4 aux-loss weights keyed by
                {"lovasz","boundary","stego","depth_smooth","gated_crf","neco"}.
                Missing keys default to 0.0 (disabled).
            aux_params: optional dict of hyperparameters consumed by aux losses
                (boundary_dilate_px, stego_temperature, etc.).
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
        self.aux_weights: Dict[str, float] = dict(aux_weights or {})
        self.aux_params: Dict[str, float] = dict(aux_params or {})
        # Registry of aux-loss callables. Imported lazily here (not at module
        # load) so test fixtures that do not need aux losses stay lightweight.
        self._aux_fns: Dict[str, Callable[[torch.Tensor, torch.Tensor, dict], torch.Tensor]] = {}
        if any(w > 0.0 for w in self.aux_weights.values()):
            from cups.losses import build_aux_losses

            self._aux_fns = build_aux_losses()

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
        head_cfg = cfg.MODEL.SEM_SEG_HEAD
        aux_weights = {
            "lovasz": getattr(head_cfg, "LOVASZ_WEIGHT", 0.0),
            "boundary": getattr(head_cfg, "BOUNDARY_WEIGHT", 0.0),
            "stego": getattr(head_cfg, "STEGO_WEIGHT", 0.0),
            "depth_smooth": getattr(head_cfg, "DEPTH_SMOOTH_WEIGHT", 0.0),
            "gated_crf": getattr(head_cfg, "GATED_CRF_WEIGHT", 0.0),
            "neco": getattr(head_cfg, "NECO_WEIGHT", 0.0),
        }
        aux_params = {
            "boundary_dilate_px": getattr(head_cfg, "BOUNDARY_DILATE_PX", 3),
            "boundary_ce_mult": getattr(head_cfg, "BOUNDARY_CE_MULT", 2.0),
            "stego_temperature": getattr(head_cfg, "STEGO_TEMPERATURE", 0.1),
            "stego_knn_k": getattr(head_cfg, "STEGO_KNN_K", 7),
            "stego_feature_source": getattr(head_cfg, "STEGO_FEATURE_SOURCE", "fpn_p2"),
            "depth_smooth_alpha": getattr(head_cfg, "DEPTH_SMOOTH_ALPHA", 10.0),
            "gated_crf_kernel": getattr(head_cfg, "GATED_CRF_KERNEL", 5),
            "gated_crf_rgb_sigma": getattr(head_cfg, "GATED_CRF_RGB_SIGMA", 0.1),
            "neco_k": getattr(head_cfg, "NECO_K", 5),
        }
        return {
            "input_shape": {k: v for k, v in input_shape.items() if k in head_cfg.IN_FEATURES},
            "ignore_value": head_cfg.IGNORE_VALUE,
            "num_classes": head_cfg.NUM_CLASSES,
            "conv_dims": head_cfg.CONVS_DIM,
            "common_stride": head_cfg.COMMON_STRIDE,
            "norm": head_cfg.NORM,
            "loss_weight": head_cfg.LOSS_WEIGHT,
            "class_weight": head_cfg.CLASS_WEIGHT,
            "stuff_kd_weight": getattr(head_cfg, "STUFF_KD_WEIGHT", 0.0),
            "kd_temperature": getattr(head_cfg, "KD_TEMPERATURE", 2.0),
            "aux_weights": aux_weights,
            "aux_params": aux_params,
        }

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor] = None,
        pixel_weights: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        pseudo_onehot: Optional[torch.Tensor] = None,
        ctx: Optional[Dict[str, torch.Tensor]] = None,
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
            ctx: optional context dict with auxiliary tensors consumed by
                Pass-1 to Pass-4 aux losses (``depth``, ``rgb``,
                ``dino_features``). Missing keys raise KeyError only inside the
                specific aux loss that requires them.

        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = self.layers(features)
        if self.training:
            return None, self.losses(
                x,
                targets,
                pixel_weights=pixel_weights,
                pseudo_onehot=pseudo_onehot,
                ctx=ctx,
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
        ctx: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute semantic segmentation loss with optional per-pixel weighting.

        Args:
            predictions: (B, C, H', W') logits from the FPN head.
            targets: (B, H, W) ground truth class indices.
            pixel_weights: optional (B, H, W) per-pixel confidence weights
                from M5 mitigation. When None, uses standard mean reduction.
            pseudo_onehot: optional (B, C, H, W) one-hot pseudo-label targets
                for stuff-preservation KD loss. Only used when stuff_kd_weight > 0.
            ctx: optional context dict carrying auxiliary tensors consumed by
                the P1-P4 aux losses (``depth``, ``rgb``, ``dino_features``).
                Keys are only validated by the specific aux losses that need
                them. Passing ``None`` or an empty dict disables all aux
                terms regardless of their weights.

        Returns:
            Dict of loss tensors.
        """
        predictions_low = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
        predictions = F.interpolate(
            predictions_low,
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

        # P1-P4 auxiliary losses (Lovász, boundary CE, STEGO, depth smoothness,
        # Gated-CRF, NeCo). Each is controlled by a non-zero weight in
        # ``self.aux_weights``; disabled terms are skipped entirely so the
        # compute cost is zero when the aux head is unused.
        if self._aux_fns and any(w > 0.0 for w in self.aux_weights.values()):
            aux_ctx: Dict[str, object] = {
                **(ctx or {}),
                "logits_up": predictions,
                "logits_low": predictions_low,
                "targets": targets,
                "class_weight": class_weight,
                "ignore_index": self.ignore_value,
                "params": self.aux_params,
            }
            name_map = (
                ("lovasz", "loss_lovasz", "lovasz_softmax"),
                ("boundary", "loss_boundary", "boundary_ce"),
                ("stego", "loss_stego", "stego_corr"),
                ("depth_smooth", "loss_depth_smooth", "depth_smoothness"),
                ("gated_crf", "loss_gated_crf", "gated_crf"),
                ("neco", "loss_neco", "neco"),
            )
            for weight_key, out_key, fn_key in name_map:
                w = float(self.aux_weights.get(weight_key, 0.0))
                if w <= 0.0:
                    continue
                fn = self._aux_fns[fn_key]
                losses[out_key] = fn(predictions, targets, aux_ctx) * w

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
