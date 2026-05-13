"""AdaptiveInstanceNet: Conv2d-based adaptive instance decomposition.

Replaces the fixed depth-gradient threshold (τ=0.10) with a learned,
spatially-adaptive split predictor using depthwise-separable Conv2d blocks.

Takes DINOv2 features + depth + semantic probabilities/clusters and predicts:
  1. Per-pixel split probability (instance boundary map)
  2. Per-pixel instance embedding (for grouping/merging)

References:
    - Method 2 from mamba_bridge_pseudo_label_refinement.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Depthwise-separable Conv2d block with residual connection.

    GroupNorm → DW-Conv 3×3 → GELU → PW-Conv 1×1 → residual
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(16, dim)
        self.dw_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.act = nn.GELU()
        self.pw_conv = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.dw_conv(x)
        x = self.act(x)
        x = self.pw_conv(x)
        return res + x


class AdaptiveInstanceNet(nn.Module):
    """Conv2d-based adaptive instance decomposition network.

    Architecture:
        [DINOv2(768), depth(1), depth_grads(2), semantic inputs]
        → concat → 1×1 proj → hidden_dim
        → N × ConvBlock (depthwise-separable + residual)
        → split_head → split_logit (1, H, W)
        → embed_head → instance_embed (embed_dim, H, W)

    Args:
        feature_dim: DINOv2 feature dimension (768)
        depth_channels: depth (1) + sobel_x + sobel_y (2) = 3
        semantic_dim: semantic input channels (CAUSE-27, trainID-19, or k clusters)
        hidden_dim: internal processing dimension
        embed_dim: instance embedding dimension
        num_blocks: number of low-resolution ConvBlock layers
        use_highres_fusion: enable a 128x256 branch for Sobel/SAM3 priors
        fusion_channels: high-resolution fusion input channels
        fusion_hidden_dim: high-resolution prior encoder width
        fusion_blocks: number of high-resolution ConvBlock layers
    """

    def __init__(
        self,
        feature_dim: int = 768,
        depth_channels: int = 3,
        semantic_dim: int = 27,
        hidden_dim: int = 256,
        embed_dim: int = 32,
        num_blocks: int = 6,
        use_highres_fusion: bool = False,
        fusion_channels: int = 9,
        fusion_hidden_dim: int = 64,
        fusion_blocks: int = 2,
        thing_classes: int = 0,
        **kwargs,  # absorb unused Mamba args for CLI compatibility
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_highres_fusion = use_highres_fusion
        self.thing_classes = thing_classes

        # Input fusion
        total_input = feature_dim + depth_channels + semantic_dim
        self.input_proj = nn.Sequential(
            nn.Conv2d(total_input, hidden_dim, 1, bias=False),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
        )

        # Conv processing blocks
        self.blocks = nn.ModuleList([
            ConvBlock(hidden_dim) for _ in range(num_blocks)
        ])

        if use_highres_fusion:
            self.fusion_encoder = nn.Sequential(
                nn.Conv2d(fusion_channels, fusion_hidden_dim, 3, padding=1, bias=False),
                nn.GroupNorm(8, fusion_hidden_dim),
                nn.GELU(),
                nn.Conv2d(fusion_hidden_dim, fusion_hidden_dim, 3, padding=1, bias=False),
                nn.GroupNorm(8, fusion_hidden_dim),
                nn.GELU(),
            )
            self.highres_fuse = nn.Sequential(
                nn.Conv2d(hidden_dim + fusion_hidden_dim, hidden_dim, 1, bias=False),
                nn.GroupNorm(16, hidden_dim),
                nn.GELU(),
            )
            self.highres_blocks = nn.ModuleList([
                ConvBlock(hidden_dim) for _ in range(fusion_blocks)
            ])
        else:
            self.fusion_encoder = None
            self.highres_fuse = None
            self.highres_blocks = nn.ModuleList()

        # Split probability head: per-pixel boundary prediction
        self.split_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

        # Instance embedding head: per-pixel grouping embeddings
        self.embed_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, 1),
        )

        # Optional background-aware thing classifier.
        # Class 0 is non-thing/background; classes 1..8 map to Cityscapes
        # thing trainIDs 11..18.
        if thing_classes > 0:
            self.thing_head = nn.Sequential(
                nn.Conv2d(hidden_dim, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, thing_classes, 1),
            )
        else:
            self.thing_head = None

    def forward(
        self,
        dinov2_features: torch.Tensor,
        depth: torch.Tensor,
        depth_grads: torch.Tensor,
        cause_logits: torch.Tensor,
        fusion_inputs: torch.Tensor | None = None,
    ):
        """
        Args:
            dinov2_features: (B, 768, H, W) DINOv2 patch features
            depth: (B, 1, H, W) normalized depth [0, 1]
            depth_grads: (B, 2, H, W) Sobel_x, Sobel_y of depth
            cause_logits: (B, semantic_dim, H, W) semantic probabilities
            fusion_inputs: optional high-resolution fused
                [DepthPro, Sobel, SAM3] priors

        Returns:
            split_logit: (B, 1, H, W) raw logits for boundary
            instance_embed: (B, embed_dim, H, W) instance grouping embeddings
        """
        x = torch.cat([dinov2_features, depth, depth_grads, cause_logits], dim=1)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        if self.use_highres_fusion:
            if fusion_inputs is None:
                h, w = x.shape[-2] * 4, x.shape[-1] * 4
                fusion_inputs = x.new_zeros(
                    x.shape[0], self.fusion_encoder[0].in_channels, h, w)
            x = F.interpolate(
                x, size=fusion_inputs.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            fusion = self.fusion_encoder(fusion_inputs)
            x = self.highres_fuse(torch.cat([x, fusion], dim=1))
            for block in self.highres_blocks:
                x = block(x)

        split_logit = self.split_head(x)
        instance_embed = self.embed_head(x)
        if self.thing_head is not None:
            thing_logits = self.thing_head(x)
            return split_logit, instance_embed, thing_logits

        return split_logit, instance_embed
