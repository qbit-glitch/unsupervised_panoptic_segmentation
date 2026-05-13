"""Mask2Former for Panoptic Segmentation with DINOv3 ViT-L/16 Backbone.

Full model: frozen DINOv3 ViT-L/16 → SimpleFeaturePyramid → FPN PixelDecoder
→ Mask2Former TransformerDecoder → class + mask predictions.

Only the decoder and pyramid are trained (~16M params).
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn

from .feature_pyramid import SimpleFeaturePyramid
from .pixel_decoder import FPNPixelDecoder
from .transformer_decoder import MultiScaleMaskedTransformerDecoder


class DINOv3Backbone(nn.Module):
    """Frozen DINOv3 ViT-L/16 backbone with multi-layer feature extraction.

    Loads from HuggingFace and extracts features from 4 intermediate layers
    [4, 11, 17, 23], reshaped to 2D spatial feature maps.

    Args:
        model_name: HuggingFace model name.
        layer_indices: Which transformer blocks to extract features from.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        layer_indices: tuple[int, ...] = (4, 11, 17, 23),
    ):
        super().__init__()
        self.layer_indices = layer_indices

        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # ViT-L/16: embed_dim=1024, patch_size=16, num_registers=4
        self.embed_dim = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        # DINOv3 has CLS + register tokens to skip
        num_registers = getattr(self.model.config, "num_register_tokens", 0)
        self.skip_tokens = 1 + num_registers  # CLS + registers

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-layer features as 2D spatial maps.

        Args:
            pixel_values: (B, 3, H, W) normalized images.

        Returns:
            List of 4 tensors, each (B, embed_dim, H/patch, W/patch).
        """
        B, _, H, W = pixel_values.shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        # Forward through ViT with hidden state output
        outputs = self.model(
            pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states  # (num_layers+1,) each (B, N, D)

        features = []
        for idx in self.layer_indices:
            # +1 because hidden_states[0] is the embedding output
            feat = hidden_states[idx + 1]
            # Remove CLS + register tokens
            feat = feat[:, self.skip_tokens:, :]
            # Reshape to spatial: (B, N, D) → (B, D, H/p, W/p)
            feat = feat.transpose(1, 2).reshape(B, self.embed_dim, h_patches, w_patches)
            features.append(feat)

        return features


class Mask2FormerPanoptic(nn.Module):
    """Full Mask2Former for unsupervised panoptic segmentation.

    Architecture:
        DINOv3 ViT-L/16 (frozen) → SimpleFeaturePyramid → FPN PixelDecoder
        → Mask2Former TransformerDecoder → {pred_logits, pred_masks}

    Args:
        backbone: Frozen DINOv3Backbone (or any module returning 4 feature maps).
        num_classes: Number of semantic classes (19 for Cityscapes).
        hidden_dim: Transformer hidden dimension.
        num_queries: Number of learnable object queries.
        nheads: Attention heads.
        dim_feedforward: FFN dimension.
        dec_layers: Decoder layers.
        backbone_dim: Backbone feature dimension (1024 for ViT-L).
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 19,
        hidden_dim: int = 256,
        num_queries: int = 100,
        nheads: int = 8,
        dim_feedforward: int = 1024,
        dec_layers: int = 9,
        backbone_dim: int = 1024,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        self.feature_pyramid = SimpleFeaturePyramid(
            in_dim=backbone_dim, out_dim=hidden_dim,
        )
        self.pixel_decoder = FPNPixelDecoder(
            feature_dim=hidden_dim, mask_dim=hidden_dim,
        )
        self.transformer_decoder = MultiScaleMaskedTransformerDecoder(
            in_channels=hidden_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            mask_dim=hidden_dim,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            images: (B, 3, H, W) normalized input images.

        Returns:
            Dict with:
                pred_logits: (B, Q, num_classes+1) class logits.
                pred_masks: (B, Q, H/4, W/4) mask logits.
                aux_outputs: List of dicts from intermediate decoder layers.
        """
        # Backbone: frozen feature extraction
        multi_layer_features = self.backbone(images)

        # Multi-scale pyramid from ViT features
        pyramid_features = self.feature_pyramid(multi_layer_features)

        # Pixel decoder: FPN top-down + mask features
        mask_features, multi_scale_features = self.pixel_decoder(pyramid_features)

        # Transformer decoder: queries → class + mask predictions
        outputs = self.transformer_decoder(multi_scale_features, mask_features)

        return outputs

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Get only trainable parameters (excludes frozen backbone)."""
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
