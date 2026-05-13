"""DINOv3 ViT-B/16 backbone wrapper for Detectron2.

Uses the OFFICIAL facebookresearch/dinov3 implementation directly.
No custom ViT, RoPE, or attention code -- everything from Meta's repo.

Architecture:
    Image (B, 3, H, W)  [H,W must be divisible by 16]
    -> Official DINOv3 ViT-B/16 (frozen)
    -> patch tokens (B, H/16 * W/16, 768)
    -> reshape to (B, 768, H/16, W/16)
    -> SimpleFeaturePyramid -> {p2, p3, p4, p5} all 256-dim
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import Backbone
from detectron2.layers import ShapeSpec

logger = logging.getLogger(__name__)

# --- Add official dinov3 repo to sys.path ---
_CUPS_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_CUPS_ROOT = os.path.abspath(os.path.join(_CUPS_MODEL_DIR, "..", ".."))
# Try project root first (4 levels up), then cups root (2 levels up)
_PROJECT_ROOT_CANDIDATE = os.path.abspath(os.path.join(_CUPS_MODEL_DIR, "..", "..", "..", ".."))
if os.path.isdir(os.path.join(_PROJECT_ROOT_CANDIDATE, "refs", "dinov3")):
    _PROJECT_ROOT = _PROJECT_ROOT_CANDIDATE
else:
    _PROJECT_ROOT = _CUPS_ROOT
_DINOV3_ROOT = os.path.join(_PROJECT_ROOT, "refs", "dinov3")
if _DINOV3_ROOT not in sys.path:
    sys.path.insert(0, _DINOV3_ROOT)


def _load_official_dinov3_vitb16() -> nn.Module:
    """Load official DINOv3 ViT-B/16 from facebookresearch/dinov3 repo.

    Loads pre-converted weights from weights/dinov3_vitb16_official.pth
    (converted from HuggingFace, verified cosine=1.0 match).
    Falls back to downloading from Meta's CDN if local weights not found.
    """
    from dinov3.hub.backbones import dinov3_vitb16

    local_pth = os.path.join(_PROJECT_ROOT, "weights", "dinov3_vitb16_official.pth")

    if os.path.exists(local_pth):
        logger.info("Loading official DINOv3 ViT-B/16 from local: %s", local_pth)
        model = dinov3_vitb16(pretrained=False)
        sd = torch.load(local_pth, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)
    else:
        logger.info("Downloading official DINOv3 ViT-B/16 from Meta CDN...")
        model = dinov3_vitb16(pretrained=True)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Official DINOv3 ViT-B/16: %.1fM params", n_params / 1e6)
    return model


class DINOv3ViTBackbone(Backbone):
    """DINOv3 ViT-B/16 with registers, wrapped as Detectron2 Backbone.

    Uses the official facebookresearch/dinov3 implementation.
    Outputs a single-scale spatial feature map for SimpleFeaturePyramid.

    Args:
        model_name: Unused (kept for API compatibility). Weights loaded from official repo.
        freeze: If True, freeze all backbone parameters (default True).
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        freeze: bool = True,
        lora_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.embed_dim: int = 768
        self.patch_size: int = 16
        self.num_register_tokens: int = 4
        self._out_feature = "dinov3"
        self._freeze = freeze
        self._lora_config = lora_config

        # Load official model
        self.vit = _load_official_dinov3_vitb16()

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad_(False)
            self.vit.train(False)
            logger.info(
                "DINOv3 backbone frozen: "
                "embed_dim=%d, patch_size=%d, registers=%d",
                self.embed_dim, self.patch_size, self.num_register_tokens,
            )

        # Inject DoRA/LoRA adapters if config provided
        if lora_config is not None:
            from cups.model.lora import DoRAConfig, inject_dora_into_model

            dora_cfg = DoRAConfig(
                rank=lora_config.get("RANK", 4),
                alpha=lora_config.get("ALPHA", 4.0),
                dropout=lora_config.get("DROPOUT", 0.05),
                late_block_start=lora_config.get("LATE_BLOCK_START", 6),
            )
            variant = lora_config.get("VARIANT", "dora")
            adapted = inject_dora_into_model(self.vit, dora_cfg, variant=variant)
            logger.info(
                "DoRA injected into DINOv3 backbone: %d layers, %d trainable params",
                len(adapted), sum(adapted.values()),
            )

    def _set_conv_dora_spatial_dims(self, h_p: int, w_p: int) -> None:
        """Set spatial dims on all ConvDoRALinear modules before forward.

        ConvDoRALinear needs (h_patches, w_patches) to reshape the LoRA
        bottleneck features into a 2D grid for depthwise convolution.
        This must be called before each forward pass since image sizes
        can vary (multi-scale augmentation).
        """
        from cups.model.lora import ConvDoRALinear

        for module in self.vit.modules():
            if isinstance(module, ConvDoRALinear):
                module._spatial_dims = (h_p, w_p)

    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return patch tokens (B, N, 768).

        Uses forward_features() which returns a dict with x_norm_patchtokens.
        """
        feats = self.vit.forward_features(x)
        return feats["x_norm_patchtokens"]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract spatial feature map from DINOv3.

        Args:
            x: Input images (B, 3, H, W). H,W already padded to size_divisibility
               by Detectron2's ImageList before this call.

        Returns:
            Dict with single feature: {"dinov3": (B, 768, H/16, W/16)}.
        """
        B, C, H, W = x.shape
        ps = self.patch_size  # 16

        # Pad to nearest multiple of patch_size (safety, should be no-op)
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        H_pad, W_pad = H + pad_h, W + pad_w
        h_p = H_pad // ps
        w_p = W_pad // ps

        # Set spatial dims for ConvDoRALinear modules (varies with input size)
        if self._lora_config is not None:
            self._set_conv_dora_spatial_dims(h_p, w_p)

        if self._freeze and self._lora_config is None:
            # Fully frozen: no gradients needed anywhere
            with torch.no_grad():
                patch_tokens = self._extract_patch_tokens(x)
        else:
            # Either unfrozen or LoRA active: gradients must flow
            patch_tokens = self._extract_patch_tokens(x)

        # (B, h_p * w_p, 768) -> (B, 768, h_p, w_p)
        feat_map = patch_tokens.reshape(B, h_p, w_p, self.embed_dim)
        feat_map = feat_map.permute(0, 3, 1, 2).contiguous()

        # Resize handles any residual rounding
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
        return 16

    def output_shape(self) -> Dict[str, ShapeSpec]:
        """Report stride=16 so SimpleFeaturePyramid produces strides 4,8,16,32."""
        return {
            self._out_feature: ShapeSpec(
                channels=self.embed_dim,
                stride=16,
            )
        }

    def train(self, mode: bool = True):
        """Keep backbone in inference mode when frozen."""
        super().train(mode)
        if self._freeze:
            self.vit.train(False)
        return self


def _make_vit_patch_pyramid_cls():
    """Build a SimpleFeaturePyramid subclass lazily.

    Subclassing is done at call time so detectron2 can be imported only
    when a DINOv3 backbone is actually constructed (keeps unit tests that
    stub out the full FPN lightweight).
    """
    from detectron2.modeling.backbone.vit import SimpleFeaturePyramid

    class DINOv3FeaturePyramid(SimpleFeaturePyramid):
        """SimpleFeaturePyramid that also emits pre-FPN DINOv3 patch tokens.

        Downstream aux losses (STEGO correspondence, NeCo) need stride-16
        patch-level features that still carry the self-supervised DINOv3
        structure. The FPN-P2 output is heavily transformed and at stride-4
        — useful for Cascade RoI heads but lossy for correspondence.
        """

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            bottom_up_features = self.net(x)
            features = bottom_up_features[self.in_feature]
            results = []
            for stage in self.stages:
                results.append(stage(features))
            if self.top_block is not None:
                if self.top_block.in_feature in bottom_up_features:
                    top_block_in_feature = bottom_up_features[self.top_block.in_feature]
                else:
                    top_block_in_feature = results[
                        self._out_features.index(self.top_block.in_feature)
                    ]
                results.extend(self.top_block(top_block_in_feature))
            assert len(self._out_features) == len(results)
            out = {f: res for f, res in zip(self._out_features, results)}
            out["vit_patch"] = features
            return out

    return DINOv3FeaturePyramid


def build_dinov3_vitb_fpn_backbone(
    out_channels: int = 256,
    freeze: bool = True,
    model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    lora_config: Optional[Dict] = None,
) -> Backbone:
    """Build DINOv3 ViT-B/16 + SimpleFeaturePyramid backbone.

    Uses official facebookresearch/dinov3 implementation.

    Args:
        out_channels: FPN output channels (default 256).
        freeze: Whether to freeze the DINOv3 backbone (default True).
        model_name: Unused (kept for API compat).
        lora_config: DoRA/LoRA config dict. Keys: RANK, ALPHA, DROPOUT,
            LATE_BLOCK_START, VARIANT. None means no adaptation.

    Returns:
        SimpleFeaturePyramid subclass wrapping DINOv3ViTBackbone that
        exposes both FPN levels (p2-p5) and the pre-FPN ``vit_patch``
        tensor at stride-16.
    """
    from detectron2.modeling.backbone.fpn import LastLevelMaxPool

    Pyramid = _make_vit_patch_pyramid_cls()

    backbone = DINOv3ViTBackbone(
        model_name=model_name, freeze=freeze, lora_config=lora_config,
    )

    fpn = Pyramid(
        net=backbone,
        in_feature="dinov3",
        out_channels=out_channels,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        norm="LN",
        top_block=LastLevelMaxPool(),
    )

    logger.info(
        "Built DINOv3+SimpleFeaturePyramid (official repo), "
        "out_channels=%d, freeze=%s, lora=%s",
        out_channels, freeze,
        lora_config.get("VARIANT", "dora") if lora_config else "none",
    )
    return fpn


# ---------------------------------------------------------------------------
# DINOv3 ViT-L/16 backbone (embed_dim=1024, 307M params)
# ---------------------------------------------------------------------------

def _load_official_dinov3_vitl16() -> nn.Module:
    """Load official DINOv3 ViT-L/16 from facebookresearch/dinov3 repo.

    Loads pre-converted weights from weights/dinov3_vitl16_official.pth if present.
    Falls back to downloading from Meta's CDN if local weights not found.
    """
    from dinov3.hub.backbones import dinov3_vitl16

    local_pth = os.path.join(_PROJECT_ROOT, "weights", "dinov3_vitl16_official.pth")

    if os.path.exists(local_pth):
        logger.info("Loading official DINOv3 ViT-L/16 from local: %s", local_pth)
        model = dinov3_vitl16(pretrained=False)
        sd = torch.load(local_pth, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)
    else:
        logger.info("Downloading official DINOv3 ViT-L/16 from Meta CDN...")
        model = dinov3_vitl16(pretrained=True)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Official DINOv3 ViT-L/16: %.1fM params", n_params / 1e6)
    return model


class DINOv3ViTLBackbone(Backbone):
    """DINOv3 ViT-L/16 with registers, wrapped as Detectron2 Backbone.

    Identical interface to DINOv3ViTBackbone but embed_dim=1024 (ViT-L vs ViT-B).
    Uses official facebookresearch/dinov3 implementation.

    Args:
        model_name: Unused (kept for API compatibility).
        freeze: If True, freeze all backbone parameters (default True).
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16",
        freeze: bool = True,
        lora_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.embed_dim: int = 1024
        self.patch_size: int = 16
        self._out_feature = "dinov3"
        self._freeze = freeze
        self._lora_config = lora_config

        self.vit = _load_official_dinov3_vitl16()

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad_(False)
            self.vit.train(False)
            logger.info(
                "DINOv3 ViT-L/16 backbone frozen: embed_dim=%d, patch_size=%d",
                self.embed_dim, self.patch_size,
            )

        if lora_config is not None:
            from cups.model.lora import DoRAConfig, inject_dora_into_model

            dora_cfg = DoRAConfig(
                rank=lora_config.get("RANK", 4),
                alpha=lora_config.get("ALPHA", 4.0),
                dropout=lora_config.get("DROPOUT", 0.05),
                late_block_start=lora_config.get("LATE_BLOCK_START", 6),
            )
            variant = lora_config.get("VARIANT", "dora")
            adapted = inject_dora_into_model(self.vit, dora_cfg, variant=variant)
            logger.info(
                "DoRA injected into DINOv3 ViT-L backbone: %d layers, %d params",
                len(adapted), sum(adapted.values()),
            )

    def _set_conv_dora_spatial_dims(self, h_p: int, w_p: int) -> None:
        """Set spatial dims on all ConvDoRALinear modules before forward."""
        from cups.model.lora import ConvDoRALinear

        for module in self.vit.modules():
            if isinstance(module, ConvDoRALinear):
                module._spatial_dims = (h_p, w_p)

    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit.forward_features(x)["x_norm_patchtokens"]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape
        ps = self.patch_size

        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        H_pad, W_pad = H + pad_h, W + pad_w
        h_p = H_pad // ps
        w_p = W_pad // ps

        # Set spatial dims for ConvDoRALinear modules (varies with input size)
        if self._lora_config is not None:
            self._set_conv_dora_spatial_dims(h_p, w_p)

        use_no_grad = self._freeze and self._lora_config is None
        ctx = torch.no_grad() if use_no_grad else contextlib.nullcontext()
        with ctx:
            patch_tokens = self._extract_patch_tokens(x)

        feat_map = patch_tokens.reshape(B, h_p, w_p, self.embed_dim)
        feat_map = feat_map.permute(0, 3, 1, 2).contiguous()

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
        return 16

    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {
            self._out_feature: ShapeSpec(
                channels=self.embed_dim,
                stride=16,
            )
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze:
            self.vit.train(False)
        return self


def build_dinov3_vitl_fpn_backbone(
    out_channels: int = 256,
    freeze: bool = True,
    model_name: str = "facebook/dinov3-vitl16",
    lora_config: Optional[Dict] = None,
) -> Backbone:
    """Build DINOv3 ViT-L/16 + SimpleFeaturePyramid backbone.

    Same interface as build_dinov3_vitb_fpn_backbone() but uses ViT-L/16 (1024-dim).

    Args:
        out_channels: FPN output channels (default 256).
        freeze: Whether to freeze the DINOv3 backbone (default True).
        model_name: Unused (kept for API compat).
        lora_config: DoRA/LoRA config dict. None means no adaptation.

    Returns:
        SimpleFeaturePyramid subclass wrapping DINOv3ViTLBackbone that
        exposes both FPN levels (p2-p5) and the pre-FPN ``vit_patch``
        tensor at stride-16.
    """
    from detectron2.modeling.backbone.fpn import LastLevelMaxPool

    Pyramid = _make_vit_patch_pyramid_cls()

    backbone = DINOv3ViTLBackbone(
        model_name=model_name, freeze=freeze, lora_config=lora_config,
    )

    fpn = Pyramid(
        net=backbone,
        in_feature="dinov3",
        out_channels=out_channels,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        norm="LN",
        top_block=LastLevelMaxPool(),
    )

    logger.info(
        "Built DINOv3 ViT-L/16 + SimpleFeaturePyramid (official repo), "
        "out_channels=%d, freeze=%s, lora=%s",
        out_channels, freeze,
        lora_config.get("VARIANT", "dora") if lora_config else "none",
    )
    return fpn
