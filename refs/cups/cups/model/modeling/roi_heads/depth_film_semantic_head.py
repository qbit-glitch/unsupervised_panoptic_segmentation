# Copyright (c) Facebook, Inc. and its affiliates.
"""Depth-conditioned semantic segmentation head with FiLM modulation.

Extends CustomSemSegFPNHead with a DepthEncoder that computes sinusoidal +
Sobel features from raw depth, and per-FPN-scale FiLM generators that
modulate intermediate features via gamma/beta affine transforms.
"""

import logging
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from torch import nn
from torch.nn import functional as F

from .semantic_seg import CustomSemSegFPNHead, SEM_SEG_HEADS_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["DepthEncoder", "DepthFiLMSemSegHead"]


class DepthEncoder(nn.Module):
    """Encode raw depth into multi-channel features (sinusoidal + Sobel).

    Input:  (B, 1, H, W) raw depth map.
    Output: (B, out_channels, H, W) depth features.

    Default out_channels=15:
        - 12 channels: sinusoidal encoding (6 freq_bands x sin + cos)
        - 2 channels: Sobel gradient magnitude (gx, gy)
        - 1 channel: raw depth (normalized)
    """

    def __init__(
        self,
        out_channels: int = 15,
        freq_bands: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
    ) -> None:
        """Initialize DepthEncoder.

        Args:
            out_channels: total output channels (must match
                2*len(freq_bands) + 2 + 1).
            freq_bands: frequencies for sinusoidal positional encoding.
        """
        super().__init__()
        self.freq_bands = freq_bands
        expected_channels = 2 * len(freq_bands) + 2 + 1
        if out_channels != expected_channels:
            logger.warning(
                "out_channels=%d does not match expected=%d "
                "(2*%d freq_bands + 2 Sobel + 1 raw). Using expected.",
                out_channels, expected_channels, len(freq_bands),
            )
        self.out_channels = expected_channels

        # Fixed Sobel kernels (not learnable)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """Compute depth features.

        Args:
            depth: (B, 1, H, W) raw depth map.

        Returns:
            (B, out_channels, H, W) depth features.
        """
        # Sinusoidal encoding: (B, 2*num_freqs, H, W)
        sin_features = []
        for freq in self.freq_bands:
            sin_features.append(torch.sin(freq * depth))
            sin_features.append(torch.cos(freq * depth))

        # Sobel gradients: (B, 1, H, W) each
        gx = F.conv2d(depth, self.sobel_x, padding=1)
        gy = F.conv2d(depth, self.sobel_y, padding=1)

        # Concatenate: sinusoidal + Sobel gx + Sobel gy + raw depth
        features = torch.cat(sin_features + [gx, gy, depth], dim=1)
        return features


@SEM_SEG_HEADS_REGISTRY.register()
class DepthFiLMSemSegHead(CustomSemSegFPNHead):
    """Semantic segmentation head with depth FiLM conditioning at each FPN scale.

    Inherits all behavior from CustomSemSegFPNHead (CE loss, stuff KD loss,
    pixel weighting). When depth is None, behaves identically to the base class.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        depth_channels: int = 15,
        num_classes: int,
        conv_dims: int,
        common_stride: int,
        loss_weight: float = 1.0,
        norm: Optional[Union[str, Callable]] = None,
        class_weight: Optional[Tuple[float, ...]] = None,
        ignore_value: int = -1,
        stuff_kd_weight: float = 0.0,
        kd_temperature: float = 2.0,
    ) -> None:
        """Initialize DepthFiLMSemSegHead.

        Args:
            input_shape: shapes (channels and stride) of the input features.
            depth_channels: number of channels produced by DepthEncoder.
            num_classes: number of classes to predict.
            conv_dims: number of output channels for intermediate conv layers.
            common_stride: the common stride that all features will be upscaled to.
            loss_weight: loss weight.
            norm: normalization for all conv layers.
            class_weight: class-wise weighting factors.
            ignore_value: category id to be ignored during training.
            stuff_kd_weight: weight for stuff-preservation KD loss.
            kd_temperature: temperature for KD softening.
        """
        super().__init__(
            input_shape,
            num_classes=num_classes,
            conv_dims=conv_dims,
            common_stride=common_stride,
            loss_weight=loss_weight,
            norm=norm,
            class_weight=class_weight,
            ignore_value=ignore_value,
            stuff_kd_weight=stuff_kd_weight,
            kd_temperature=kd_temperature,
        )
        self.depth_encoder = DepthEncoder(out_channels=depth_channels)

        # FiLM generators: one per FPN scale
        # Each produces gamma + beta (2 * conv_dims channels)
        self.film_generators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.depth_encoder.out_channels, conv_dims, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(conv_dims, conv_dims * 2, 1),
            )
            for _ in self.in_features
        ])
        logger.info(
            "DepthFiLMSemSegHead: %d FPN scales, depth_channels=%d, conv_dims=%d",
            len(self.in_features), self.depth_encoder.out_channels, conv_dims,
        )

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        """Build config dict from detectron2 config.

        Args:
            cfg: detectron2 CfgNode.
            input_shape: dict of ShapeSpec from backbone.

        Returns:
            Dict of keyword arguments for __init__.
        """
        ret = super().from_config(cfg, input_shape)
        ret["depth_channels"] = getattr(
            cfg.MODEL.SEM_SEG_HEAD, "DEPTH_CHANNELS", 15,
        )
        return ret

    def layers(
        self,
        features: Dict[str, torch.Tensor],
        depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute FPN feature aggregation with optional depth FiLM modulation.

        Args:
            features: dict of FPN features keyed by level name.
            depth: optional (B, 1, H, W) raw depth tensor.

        Returns:
            (B, num_classes, H', W') prediction logits.
        """
        depth_feat: Optional[torch.Tensor] = None
        if depth is not None:
            depth_feat = self.depth_encoder(depth)

        x: Optional[torch.Tensor] = None
        for i, f in enumerate(self.in_features):
            feat_i = self.scale_heads[i](features[f])
            if depth_feat is not None:
                d = F.interpolate(
                    depth_feat,
                    size=feat_i.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                film_params = self.film_generators[i](d)
                gamma, beta = film_params.chunk(2, dim=1)
                feat_i = feat_i * (1.0 + gamma) + beta
            if x is None:
                x = feat_i
            else:
                x = x + feat_i

        return self.predictor(x)

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor] = None,
        pixel_weights: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        pseudo_onehot: Optional[torch.Tensor] = None,
    ):
        """Forward pass with optional depth conditioning.

        Args:
            features: dict of FPN features.
            targets: (B, H, W) ground truth semantic labels, or None for inference.
            pixel_weights: optional (B, H, W) per-pixel confidence weights.
            depth: optional (B, 1, H, W) raw depth tensor for FiLM conditioning.
            pseudo_onehot: optional (B, C, H, W) one-hot pseudo-label targets
                for stuff-preservation KD loss.

        Returns:
            In training, returns (None, dict of losses).
            In inference, returns (CxHxW logits, {}).
        """
        x = self.layers(features, depth=depth)
        if self.training:
            return None, self.losses(
                x, targets, pixel_weights=pixel_weights, pseudo_onehot=pseudo_onehot,
            )
        else:
            x = F.interpolate(
                x,
                scale_factor=self.common_stride,
                mode="bilinear",
                align_corners=False,
            )
            return x, {}
