"""
Lightweight Panoptic Segmentation Architectures for RepViT-M0.9.

Implements 4 architecture variants:
  1. PanopticDeepLab - Bottom-up center/offset (primary)
  2. KMaXDeepLab - k-means cross-attention masks
  3. MaskConver - Pure convolution center-based
  4. Mask2FormerLite - 2-layer transformer decoder

All share: RepViT-M0.9 backbone + BiFPN feature pyramid.

Usage:
    model = build_model("panoptic_deeplab", num_classes=19, fpn_type="bifpn")
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ---------------------------------------------------------------------------
# BiFPN: Bidirectional Feature Pyramid Network
# ---------------------------------------------------------------------------

class BiFPNBlock(nn.Module):
    """Single BiFPN layer with weighted bidirectional fusion."""

    def __init__(self, fpn_dim: int, num_levels: int = 4, epsilon: float = 1e-4):
        super().__init__()
        self.num_levels = num_levels
        self.epsilon = epsilon

        # Top-down fusion weights (learnable, softmax-normalized)
        self.w_td = nn.Parameter(torch.ones(num_levels - 1, 2))
        # Bottom-up fusion weights
        self.w_bu = nn.Parameter(torch.ones(num_levels - 1, 3))

        # Depthwise separable convs after each fusion
        self.td_convs = nn.ModuleList([
            DepthwiseSeparableConv(fpn_dim, fpn_dim)
            for _ in range(num_levels - 1)
        ])
        self.bu_convs = nn.ModuleList([
            DepthwiseSeparableConv(fpn_dim, fpn_dim)
            for _ in range(num_levels - 1)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: list of (B, C, H_i, W_i) from coarsest to finest
        Returns:
            list of fused features at same resolutions
        """
        assert len(features) == self.num_levels

        # Top-down pass: coarse to fine
        td_feats = [None] * self.num_levels
        td_feats[-1] = features[-1]  # coarsest unchanged

        for i in range(self.num_levels - 2, -1, -1):
            w = F.relu(self.w_td[i])
            w = w / (w.sum() + self.epsilon)
            up = F.interpolate(td_feats[i + 1], size=features[i].shape[2:],
                               mode='bilinear', align_corners=False)
            td_feats[i] = self.td_convs[i](w[0] * features[i] + w[1] * up)

        # Bottom-up pass: fine to coarse
        bu_feats = [None] * self.num_levels
        bu_feats[0] = td_feats[0]  # finest from top-down

        for i in range(1, self.num_levels):
            w = F.relu(self.w_bu[i - 1])
            w = w / (w.sum() + self.epsilon)
            down = F.interpolate(bu_feats[i - 1], size=features[i].shape[2:],
                                 mode='bilinear', align_corners=False)
            bu_feats[i] = self.bu_convs[i - 1](
                w[0] * features[i] + w[1] * td_feats[i] + w[2] * down
            )

        return bu_feats


class BiFPN(nn.Module):
    """Multi-layer BiFPN with lateral projections."""

    def __init__(self, in_channels_list: List[int], fpn_dim: int = 128,
                 num_layers: int = 2):
        super().__init__()
        self.fpn_dim = fpn_dim

        # Lateral 1x1 projections to unify channel dims
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, fpn_dim, 1, bias=False),
                nn.GroupNorm(1, fpn_dim),
            )
            for c in in_channels_list
        ])

        # Stacked BiFPN blocks
        self.bifpn_layers = nn.ModuleList([
            BiFPNBlock(fpn_dim, num_levels=len(in_channels_list))
            for _ in range(num_layers)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Project to uniform channels (finest to coarsest order)
        projected = [lat(f) for lat, f in zip(self.laterals, features)]

        # Run BiFPN layers
        for bifpn in self.bifpn_layers:
            projected = bifpn(projected)

        return projected


class SimpleFPN(nn.Module):
    """Simple top-down FPN (baseline)."""

    def __init__(self, in_channels_list: List[int], fpn_dim: int = 128):
        super().__init__()
        self.fpn_dim = fpn_dim
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, fpn_dim, 1) for c in in_channels_list
        ])
        self.smooths = nn.ModuleList([
            DepthwiseSeparableConv(fpn_dim, fpn_dim)
            for _ in in_channels_list
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        lats = [l(f) for l, f in zip(self.laterals, features)]
        # Top-down
        for i in range(len(lats) - 2, -1, -1):
            up = F.interpolate(lats[i + 1], size=lats[i].shape[2:],
                               mode='bilinear', align_corners=False)
            lats[i] = lats[i] + up
        return [s(l) for s, l in zip(self.smooths, lats)]


class PANetFPN(nn.Module):
    """Path Aggregation Network FPN."""

    def __init__(self, in_channels_list: List[int], fpn_dim: int = 128):
        super().__init__()
        self.fpn_dim = fpn_dim
        # Top-down
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, fpn_dim, 1) for c in in_channels_list
        ])
        self.td_smooths = nn.ModuleList([
            DepthwiseSeparableConv(fpn_dim, fpn_dim)
            for _ in in_channels_list
        ])
        # Bottom-up augmentation
        self.bu_convs = nn.ModuleList([
            DepthwiseSeparableConv(fpn_dim, fpn_dim)
            for _ in range(len(in_channels_list) - 1)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        lats = [l(f) for l, f in zip(self.laterals, features)]
        # Top-down
        for i in range(len(lats) - 2, -1, -1):
            up = F.interpolate(lats[i + 1], size=lats[i].shape[2:],
                               mode='bilinear', align_corners=False)
            lats[i] = lats[i] + up
        td = [s(l) for s, l in zip(self.td_smooths, lats)]
        # Bottom-up
        bu = [td[0]]
        for i in range(1, len(td)):
            down = F.interpolate(bu[-1], size=td[i].shape[2:],
                                 mode='bilinear', align_corners=False)
            bu.append(self.bu_convs[i - 1](td[i] + down))
        return bu


def build_fpn(fpn_type: str, in_channels_list: List[int],
              fpn_dim: int = 128) -> nn.Module:
    """Factory for FPN variants."""
    if fpn_type == "bifpn":
        return BiFPN(in_channels_list, fpn_dim, num_layers=2)
    elif fpn_type == "simple":
        return SimpleFPN(in_channels_list, fpn_dim)
    elif fpn_type == "panet":
        return PANetFPN(in_channels_list, fpn_dim)
    else:
        raise ValueError(f"Unknown FPN type: {fpn_type}")


# ---------------------------------------------------------------------------
# Common Building Blocks
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution with GroupNorm + GELU."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(in_dim, in_dim, kernel_size, padding=kernel_size // 2,
                            groups=in_dim, bias=False)
        self.pw = nn.Conv2d(in_dim, out_dim, 1, bias=False)
        self.norm = nn.GroupNorm(1, out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pw(self.dw(x))))


class LightweightASPP(nn.Module):
    """Lightweight ASPP with depthwise separable convolutions."""

    def __init__(self, in_dim: int, out_dim: int,
                 rates: Tuple[int, ...] = (6, 12, 18)):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.GroupNorm(1, out_dim),
            nn.GELU(),
        )
        self.atrous = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, in_dim, 3, padding=r, dilation=r,
                          groups=in_dim, bias=False),
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
                nn.GroupNorm(1, out_dim),
                nn.GELU(),
            )
            for r in rates
        ])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.GroupNorm(1, out_dim),
            nn.GELU(),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(out_dim * (2 + len(rates)), out_dim, 1, bias=False),
            nn.GroupNorm(1, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        feats = [self.conv1x1(x)]
        for atrous in self.atrous:
            feats.append(atrous(x))
        pool = F.interpolate(self.pool(x), size=(h, w), mode='bilinear',
                             align_corners=False)
        feats.append(pool)
        return self.merge(torch.cat(feats, dim=1))


# ---------------------------------------------------------------------------
# Architecture 1: Panoptic-DeepLab (Bottom-Up, Primary)
# ---------------------------------------------------------------------------

class PanopticDeepLabHead(nn.Module):
    """Panoptic-DeepLab dual-decoder head.

    Produces:
        - Semantic logits (num_classes channels)
        - Center heatmap (1 channel, sigmoid)
        - Offset map (2 channels, dx/dy to center)
    """

    def __init__(self, fpn_dim: int = 128, num_classes: int = 19,
                 aspp_rates: Tuple[int, ...] = (3, 6, 9)):
        super().__init__()
        # Semantic decoder
        self.sem_aspp = LightweightASPP(fpn_dim, fpn_dim, aspp_rates)
        self.sem_head = nn.Sequential(
            DepthwiseSeparableConv(fpn_dim, fpn_dim),
            nn.Conv2d(fpn_dim, num_classes, 1),
        )

        # Instance decoder (shared stem + separate heads)
        self.inst_aspp = LightweightASPP(fpn_dim, fpn_dim, aspp_rates)
        self.center_head = nn.Sequential(
            DepthwiseSeparableConv(fpn_dim, fpn_dim),
            nn.Conv2d(fpn_dim, 1, 1),
        )
        self.offset_head = nn.Sequential(
            DepthwiseSeparableConv(fpn_dim, fpn_dim),
            nn.Conv2d(fpn_dim, 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.sem_head[-1], self.center_head[-1], self.offset_head[-1]]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)

    def forward(self, fpn_feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        finest = fpn_feats[0]
        # Semantic
        sem_feat = self.sem_aspp(finest)
        logits = self.sem_head(sem_feat)
        # Instance
        inst_feat = self.inst_aspp(finest)
        center = torch.sigmoid(self.center_head(inst_feat))
        offset = self.offset_head(inst_feat)
        return {"logits": logits, "center": center, "offset": offset}


class PanopticDeepLab(nn.Module):
    """Full Panoptic-DeepLab model: RepViT + FPN + dual heads."""

    def __init__(self, backbone_name: str = "repvit_m0_9.dist_450e_in1k",
                 num_classes: int = 19, fpn_dim: int = 128,
                 fpn_type: str = "bifpn", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                          features_only=True)
        channels = self.backbone.feature_info.channels()
        self.fpn = build_fpn(fpn_type, channels, fpn_dim)
        self.head = PanopticDeepLabHead(fpn_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        fpn_feats = self.fpn(features)
        return self.head(fpn_feats)


# ---------------------------------------------------------------------------
# Architecture 2: kMaX-DeepLab (k-Means Cross-Attention)
# ---------------------------------------------------------------------------

class KMeansClusterHead(nn.Module):
    """kMaX cluster head: k-means cross-attention for mask prediction.

    Instead of Hungarian matching, uses iterative k-means updates
    where cluster centers attend to pixel features.
    """

    def __init__(self, fpn_dim: int = 128, num_classes: int = 19,
                 num_queries: int = 128, num_heads: int = 4,
                 num_layers: int = 2):
        super().__init__()
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.d_model = fpn_dim

        # Learnable cluster centers
        self.cluster_centers = nn.Embedding(num_queries, fpn_dim)

        # Cross-attention layers (cluster centers attend to pixels)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(fpn_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(fpn_dim) for _ in range(num_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fpn_dim, fpn_dim * 2),
                nn.GELU(),
                nn.Linear(fpn_dim * 2, fpn_dim),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(fpn_dim) for _ in range(num_layers)
        ])

        # Prediction heads
        self.class_head = nn.Linear(fpn_dim, num_classes + 1)  # +1 for no-object
        self.mask_embed = nn.Sequential(
            nn.Linear(fpn_dim, fpn_dim),
            nn.GELU(),
            nn.Linear(fpn_dim, fpn_dim),
        )

        # Semantic head (separate)
        self.sem_head = nn.Sequential(
            DepthwiseSeparableConv(fpn_dim, fpn_dim),
            nn.Conv2d(fpn_dim, num_classes, 1),
        )

    def forward(self, fpn_feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        finest = fpn_feats[0]
        B, C, H, W = finest.shape

        # Pixel features as KV
        pixel_feats = finest.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # Cluster centers as Q
        queries = self.cluster_centers.weight.unsqueeze(0).expand(B, -1, -1)

        # Iterative k-means cross-attention
        for cross_attn, norm, ffn, ffn_norm in zip(
                self.cross_attn_layers, self.norms, self.ffns, self.ffn_norms):
            # Cross-attention: clusters attend to pixels
            q = norm(queries)
            attn_out, _ = cross_attn(q, pixel_feats, pixel_feats)
            queries = queries + attn_out
            # FFN
            queries = queries + ffn(ffn_norm(queries))

        # Predictions
        class_logits = self.class_head(queries)  # (B, Q, C+1)
        mask_embeds = self.mask_embed(queries)  # (B, Q, C)

        # Mask predictions via dot product
        masks = torch.einsum('bqc,bchw->bqhw', mask_embeds, finest)

        # Semantic (direct prediction for stuff classes)
        sem_logits = self.sem_head(finest)

        return {
            "logits": sem_logits,
            "mask_logits": masks,
            "class_logits": class_logits,
            "queries": queries,
        }


class KMaXDeepLab(nn.Module):
    """kMaX-DeepLab: RepViT + FPN + k-means cluster head."""

    def __init__(self, backbone_name: str = "repvit_m0_9.dist_450e_in1k",
                 num_classes: int = 19, fpn_dim: int = 128,
                 fpn_type: str = "bifpn", num_queries: int = 128,
                 pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                          features_only=True)
        channels = self.backbone.feature_info.channels()
        self.fpn = build_fpn(fpn_type, channels, fpn_dim)
        self.head = KMeansClusterHead(fpn_dim, num_classes, num_queries)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        fpn_feats = self.fpn(features)
        return self.head(fpn_feats)


# ---------------------------------------------------------------------------
# Architecture 3: MaskConver (Pure Convolution)
# ---------------------------------------------------------------------------

class MaskConverHead(nn.Module):
    """MaskConver: Pure convolution center-based panoptic head.

    No attention, no transformers. Uses center prediction to unify
    stuff and things, then generates masks via dynamic convolutions.
    """

    def __init__(self, fpn_dim: int = 128, num_classes: int = 19,
                 num_masks: int = 128, kernel_dim: int = 8):
        super().__init__()
        self.num_masks = num_masks
        self.kernel_dim = kernel_dim

        # Shared feature extractor (ConvNeXt-style)
        self.shared = nn.Sequential(
            DepthwiseSeparableConv(fpn_dim, fpn_dim),
            DepthwiseSeparableConv(fpn_dim, fpn_dim),
        )

        # Semantic head
        self.sem_head = nn.Conv2d(fpn_dim, num_classes, 1)

        # Center heatmap (unified stuff + things)
        self.center_head = nn.Sequential(
            DepthwiseSeparableConv(fpn_dim, fpn_dim),
            nn.Conv2d(fpn_dim, 1, 1),
        )

        # Dynamic kernel predictor (per-center)
        self.kernel_head = nn.Sequential(
            DepthwiseSeparableConv(fpn_dim, fpn_dim),
            nn.Conv2d(fpn_dim, kernel_dim, 1),
        )

        # Mask feature branch
        self.mask_branch = nn.Sequential(
            DepthwiseSeparableConv(fpn_dim, fpn_dim),
            nn.Conv2d(fpn_dim, kernel_dim, 1),
        )

        # Offset head
        self.offset_head = nn.Sequential(
            DepthwiseSeparableConv(fpn_dim, fpn_dim),
            nn.Conv2d(fpn_dim, 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.sem_head, self.center_head[-1], self.offset_head[-1]]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)

    def forward(self, fpn_feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        finest = fpn_feats[0]
        shared = self.shared(finest)

        logits = self.sem_head(shared)
        center = torch.sigmoid(self.center_head(shared))
        offset = self.offset_head(shared)

        # Dynamic masks: kernel features dot mask features
        kernel_feats = self.kernel_head(shared)  # (B, K, H, W)
        mask_feats = self.mask_branch(finest)  # (B, K, H, W)

        return {
            "logits": logits,
            "center": center,
            "offset": offset,
            "kernel_feats": kernel_feats,
            "mask_feats": mask_feats,
        }


class MaskConver(nn.Module):
    """MaskConver: RepViT + FPN + pure conv panoptic head."""

    def __init__(self, backbone_name: str = "repvit_m0_9.dist_450e_in1k",
                 num_classes: int = 19, fpn_dim: int = 128,
                 fpn_type: str = "bifpn", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                          features_only=True)
        channels = self.backbone.feature_info.channels()
        self.fpn = build_fpn(fpn_type, channels, fpn_dim)
        self.head = MaskConverHead(fpn_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        fpn_feats = self.fpn(features)
        return self.head(fpn_feats)


# ---------------------------------------------------------------------------
# Architecture 4: Mask2Former-Lite (2-Layer Transformer Decoder)
# ---------------------------------------------------------------------------

class Mask2FormerLiteHead(nn.Module):
    """Mask2Former with only 2 decoder layers for mobile deployment."""

    def __init__(self, fpn_dim: int = 128, num_classes: int = 19,
                 num_queries: int = 100, num_heads: int = 4,
                 num_layers: int = 2):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = fpn_dim

        # Learnable queries
        self.query_embed = nn.Embedding(num_queries, fpn_dim)
        self.query_feat = nn.Embedding(num_queries, fpn_dim)

        # Masked cross-attention + self-attention layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(fpn_dim, num_heads,
                                                    batch_first=True),
                'cross_norm': nn.LayerNorm(fpn_dim),
                'self_attn': nn.MultiheadAttention(fpn_dim, num_heads,
                                                   batch_first=True),
                'self_norm': nn.LayerNorm(fpn_dim),
                'ffn': nn.Sequential(
                    nn.Linear(fpn_dim, fpn_dim * 2),
                    nn.GELU(),
                    nn.Linear(fpn_dim * 2, fpn_dim),
                ),
                'ffn_norm': nn.LayerNorm(fpn_dim),
            }))

        # Prediction heads
        self.class_head = nn.Linear(fpn_dim, num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(fpn_dim, fpn_dim),
            nn.GELU(),
            nn.Linear(fpn_dim, fpn_dim),
        )

        # Semantic head (auxiliary)
        self.sem_head = nn.Sequential(
            DepthwiseSeparableConv(fpn_dim, fpn_dim),
            nn.Conv2d(fpn_dim, num_classes, 1),
        )

    def forward(self, fpn_feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        finest = fpn_feats[0]
        B, C, H, W = finest.shape

        # Pixel features
        pixel_feats = finest.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # Initialize queries
        queries = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)

        # Decoder layers
        for layer in self.layers:
            # Cross-attention (queries attend to pixels)
            q = layer['cross_norm'](queries)
            q_out, _ = layer['cross_attn'](q, pixel_feats, pixel_feats)
            queries = queries + q_out

            # Self-attention (queries attend to each other)
            q = layer['self_norm'](queries)
            q_out, _ = layer['self_attn'](q, q, q)
            queries = queries + q_out

            # FFN
            queries = queries + layer['ffn'](layer['ffn_norm'](queries))

        # Predictions
        class_logits = self.class_head(queries)  # (B, Q, C+1)
        mask_embeds = self.mask_embed(queries)  # (B, Q, C)
        mask_logits = torch.einsum('bqc,bchw->bqhw', mask_embeds, finest)

        # Semantic (auxiliary)
        sem_logits = self.sem_head(finest)

        return {
            "logits": sem_logits,
            "mask_logits": mask_logits,
            "class_logits": class_logits,
        }


class Mask2FormerLite(nn.Module):
    """Mask2Former-Lite: RepViT + FPN + 2-layer transformer decoder."""

    def __init__(self, backbone_name: str = "repvit_m0_9.dist_450e_in1k",
                 num_classes: int = 19, fpn_dim: int = 128,
                 fpn_type: str = "bifpn", num_queries: int = 100,
                 pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                          features_only=True)
        channels = self.backbone.feature_info.channels()
        self.fpn = build_fpn(fpn_type, channels, fpn_dim)
        self.head = Mask2FormerLiteHead(fpn_dim, num_classes, num_queries)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        fpn_feats = self.fpn(features)
        return self.head(fpn_feats)


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

ARCH_REGISTRY = {
    "panoptic_deeplab": PanopticDeepLab,
    "kmax_deeplab": KMaXDeepLab,
    "maskconver": MaskConver,
    "mask2former_lite": Mask2FormerLite,
}


def build_model(arch: str, backbone_name: str = "repvit_m0_9.dist_450e_in1k",
                num_classes: int = 19, fpn_dim: int = 128,
                fpn_type: str = "bifpn", pretrained: bool = True,
                **kwargs) -> nn.Module:
    """Build a panoptic model by name.

    Args:
        arch: One of "panoptic_deeplab", "kmax_deeplab", "maskconver",
              "mask2former_lite"
        backbone_name: timm model name
        num_classes: Number of semantic classes
        fpn_dim: FPN feature dimension
        fpn_type: "bifpn", "simple", or "panet"
        pretrained: Load pretrained backbone
    """
    if arch not in ARCH_REGISTRY:
        raise ValueError(f"Unknown arch: {arch}. Choose from {list(ARCH_REGISTRY)}")

    cls = ARCH_REGISTRY[arch]
    model = cls(backbone_name=backbone_name, num_classes=num_classes,
                fpn_dim=fpn_dim, fpn_type=fpn_type, pretrained=pretrained,
                **kwargs)

    # Print parameter counts
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_p = sum(p.numel() for p in model.backbone.parameters())
    fpn_p = sum(p.numel() for p in model.fpn.parameters())
    head_p = total - backbone_p - fpn_p
    print(f"[Model] {arch} + {fpn_type}")
    print(f"  Backbone: {backbone_p / 1e6:.2f}M | FPN: {fpn_p / 1e6:.2f}M | "
          f"Head: {head_p / 1e6:.2f}M | Total: {total / 1e6:.2f}M "
          f"({trainable / 1e6:.2f}M trainable)")

    return model


# ---------------------------------------------------------------------------
# Panoptic Inference Utilities
# ---------------------------------------------------------------------------

def panoptic_inference_center_offset(
    sem_logits: torch.Tensor,
    center_map: torch.Tensor,
    offset_map: torch.Tensor,
    thing_ids: set,
    center_threshold: float = 0.1,
    nms_kernel: int = 7,
    min_area: int = 50,
    max_offset_dist: float = 30.0,
) -> Tuple[torch.Tensor, dict]:
    """Bottom-up panoptic inference from center/offset predictions.

    Returns:
        pan_map: (H, W) int32 with segment IDs
        segments: dict mapping segment_id -> class_trainID
    """
    import numpy as np
    from scipy import ndimage

    # To numpy
    sem_pred = sem_logits.argmax(dim=0).cpu().numpy()  # (H, W)
    center_np = center_map.squeeze(0).cpu().numpy()  # (H, W)
    offset_np = offset_map.cpu().numpy()  # (2, H, W)
    H, W = sem_pred.shape

    pan_map = np.zeros((H, W), dtype=np.int32)
    segments = {}
    seg_id = 1

    # Stuff classes: one segment per class
    for cls in range(sem_logits.shape[0]):
        if cls not in thing_ids:
            mask = sem_pred == cls
            if mask.sum() >= 64:
                pan_map[mask] = seg_id
                segments[seg_id] = cls
                seg_id += 1

    # Things: NMS on center heatmap
    thing_mask = np.isin(sem_pred, list(thing_ids))
    if thing_mask.sum() < min_area:
        return pan_map, segments

    # NMS via max pooling
    center_t = torch.from_numpy(center_np).unsqueeze(0).unsqueeze(0).float()
    pooled = F.max_pool2d(center_t, nms_kernel, stride=1,
                          padding=nms_kernel // 2)
    nms_mask = (center_t == pooled).squeeze().numpy()
    peaks = (center_np > center_threshold) & nms_mask & thing_mask

    peak_ys, peak_xs = np.where(peaks)

    # Limit peaks to top-K by confidence to avoid O(N*P) blowup
    if len(peak_ys) > 200:
        peak_scores = center_np[peak_ys, peak_xs]
        top_k = np.argsort(peak_scores)[-200:]
        peak_ys, peak_xs = peak_ys[top_k], peak_xs[top_k]

    if len(peak_ys) == 0:
        # Fallback to connected components
        labeled, n = ndimage.label(thing_mask)
        for i in range(1, n + 1):
            inst_mask = labeled == i
            if inst_mask.sum() >= min_area:
                cls = int(np.median(sem_pred[inst_mask]))
                pan_map[inst_mask] = seg_id
                segments[seg_id] = cls
                seg_id += 1
        return pan_map, segments

    # Vote each thing pixel to nearest center
    ys, xs = np.where(thing_mask)
    voted_y = ys + offset_np[0, ys, xs]
    voted_x = xs + offset_np[1, ys, xs]

    # Vectorized assignment: compute all distances at once
    # peaks: (N_peaks, 2), voted: (N_pixels, 2)
    peak_coords = np.stack([peak_ys, peak_xs], axis=1).astype(np.float32)  # (P, 2)
    voted_coords = np.stack([voted_y, voted_x], axis=1).astype(np.float32)  # (N, 2)

    # Batch distance: (N, P)
    dists = np.linalg.norm(
        voted_coords[:, None, :] - peak_coords[None, :, :], axis=2
    )
    nearest_peak = dists.argmin(axis=1)  # (N,)
    nearest_dist = dists[np.arange(len(nearest_peak)), nearest_peak]  # (N,)

    # Assign pixels to their nearest peak (if within max_offset_dist)
    for pi in range(len(peak_ys)):
        assigned = (nearest_peak == pi) & (nearest_dist < max_offset_dist)
        if assigned.sum() < min_area:
            continue
        inst_pixels_y = ys[assigned]
        inst_pixels_x = xs[assigned]
        cls = int(np.median(sem_pred[inst_pixels_y, inst_pixels_x]))
        if cls in thing_ids:
            pan_map[inst_pixels_y, inst_pixels_x] = seg_id
            segments[seg_id] = cls
            seg_id += 1

    return pan_map, segments


def panoptic_inference_mask_cls(
    sem_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    thing_ids: set,
    mask_threshold: float = 0.5,
    overlap_threshold: float = 0.8,
    min_area: int = 50,
) -> Tuple[torch.Tensor, dict]:
    """Mask-classification panoptic inference (for kMaX, Mask2Former).

    Returns:
        pan_map: (H, W) int32 with segment IDs
        segments: dict mapping segment_id -> class_trainID
    """
    import numpy as np

    H, W = sem_logits.shape[1:]
    num_classes = sem_logits.shape[0]

    # Upsample masks if needed
    if mask_logits.shape[1:] != (H, W):
        mask_logits = F.interpolate(
            mask_logits.unsqueeze(0), size=(H, W),
            mode='bilinear', align_corners=False
        ).squeeze(0)

    # Class predictions (exclude no-object class)
    cls_probs = F.softmax(class_logits, dim=-1)  # (Q, C+1)
    scores, pred_classes = cls_probs[:, :-1].max(dim=-1)  # (Q,), (Q,)

    # Mask probabilities
    mask_probs = mask_logits.sigmoid()  # (Q, H, W)

    # Sort by score
    sorted_idx = scores.argsort(descending=True)

    pan_map = np.zeros((H, W), dtype=np.int32)
    segments = {}
    seg_id = 1
    used = np.zeros((H, W), dtype=bool)

    for idx in sorted_idx:
        cls = pred_classes[idx].item()
        score = scores[idx].item()
        mask = (mask_probs[idx] > mask_threshold).cpu().numpy()

        # Remove already assigned pixels
        mask = mask & ~used
        area = mask.sum()

        if area < min_area:
            continue
        if score < 0.3:
            continue

        # Check overlap with existing segments
        if used.any() and (mask & used).sum() / (area + 1e-8) > overlap_threshold:
            continue

        pan_map[mask] = seg_id
        segments[seg_id] = cls
        used |= mask
        seg_id += 1

    # Fill unassigned pixels with stuff predictions from semantic head
    sem_pred = sem_logits.argmax(dim=0).cpu().numpy()
    unassigned = pan_map == 0
    for cls in range(num_classes):
        if cls not in thing_ids:
            mask = unassigned & (sem_pred == cls)
            if mask.sum() >= 64:
                pan_map[mask] = seg_id
                segments[seg_id] = cls
                seg_id += 1

    return pan_map, segments
