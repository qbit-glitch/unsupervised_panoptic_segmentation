"""AdaptiveInstanceNet: Conv2d-based adaptive instance decomposition.

Replaces the fixed depth-gradient threshold (τ=0.10) with a learned,
spatially-adaptive split predictor using depthwise-separable Conv2d blocks.

Takes DINOv2 features + depth + CAUSE semantics and predicts:
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
        [DINOv2(768), depth(1), depth_grads(2), cause_logits(27)]
        → concat → 1×1 proj → hidden_dim
        → N × ConvBlock (depthwise-separable + residual)
        → split_head → split_logit (1, H, W)
        → embed_head → instance_embed (embed_dim, H, W)

    Args:
        feature_dim: DINOv2 feature dimension (768)
        depth_channels: depth (1) + sobel_x + sobel_y (2) = 3
        semantic_dim: CAUSE 27-class logits
        hidden_dim: internal processing dimension
        embed_dim: instance embedding dimension
        num_blocks: number of ConvBlock layers
    """

    def __init__(
        self,
        feature_dim: int = 768,
        depth_channels: int = 3,
        semantic_dim: int = 27,
        hidden_dim: int = 256,
        embed_dim: int = 32,
        num_blocks: int = 6,
        **kwargs,  # absorb unused Mamba args for CLI compatibility
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

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

    def forward(
        self,
        dinov2_features: torch.Tensor,
        depth: torch.Tensor,
        depth_grads: torch.Tensor,
        cause_logits: torch.Tensor,
    ):
        """
        Args:
            dinov2_features: (B, 768, H, W) DINOv2 patch features
            depth: (B, 1, H, W) normalized depth [0, 1]
            depth_grads: (B, 2, H, W) Sobel_x, Sobel_y of depth
            cause_logits: (B, 27, H, W) CAUSE semantic logits/probabilities

        Returns:
            split_logit: (B, 1, H, W) raw logits for boundary (apply sigmoid for prob)
            instance_embed: (B, embed_dim, H, W) instance grouping embeddings
        """
        x = torch.cat([dinov2_features, depth, depth_grads, cause_logits], dim=1)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        split_logit = self.split_head(x)
        instance_embed = self.embed_head(x)

        return split_logit, instance_embed
