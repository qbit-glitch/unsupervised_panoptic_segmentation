"""Architecture A: Multi-Scale Conv Adapter with depth FiLM conditioning."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_arch
from .base import AdapterConfig, BaseAdapter, ResBlock, sinusoidal_depth_encoding


class ConvScaleBlock(nn.Module):
    """Stack of ResBlocks at a single spatial scale with DepthFiLM."""

    def __init__(
        self,
        channels: int,
        depth_embed_dim: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock(channels, depth_dim=depth_embed_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])

    def forward(
        self, x: torch.Tensor, depth_embed: torch.Tensor
    ) -> torch.Tensor:
        """Run through residual blocks with depth conditioning.

        Args:
            x: Feature map (B, C, H, W).
            depth_embed: Sinusoidal depth embedding (B, D, H, W).

        Returns:
            Processed features (B, C, H, W).
        """
        for block in self.blocks:
            x = block(x, depth_embed)
        return x


@register_arch("conv")
class ConvAdapter(BaseAdapter):
    """Multi-scale convolutional adapter with transposed-conv upsampling.

    Pipeline:
        DINOv3 (B, 2048, 1024) -> reshape (B, 1024, 32, 64)
        -> 1x1 Conv stem: 1024 -> hidden_dim
        -> TransposedConv: (hidden_dim, 32, 64) -> (hidden_dim/2, 64, 128)
                        -> (hidden_dim/4, 128, 256)
        -> Scale 1: num_blocks x ResBlock(hidden_dim) @ 32x64 + DepthFiLM
        -> Scale 2: num_blocks x ResBlock(hidden_dim/2) @ 64x128 + DepthFiLM
        -> Scale 3: num_blocks x ResBlock(hidden_dim/4) @ 128x256 + DepthFiLM
        -> Downsample all to 32x64, concat
        -> 1x1 Conv -> output_dim
        -> L2-normalize -> (B, 2048, output_dim)
    """

    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__(cfg)

        s1_ch = cfg.hidden_dim
        s2_ch = cfg.hidden_dim // 2
        s3_ch = cfg.hidden_dim // 4
        scale_channels = [s1_ch, s2_ch, s3_ch]

        # Stem: reduce input_dim -> hidden_dim
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.input_dim, s1_ch, kernel_size=1),
            nn.BatchNorm2d(s1_ch),
            nn.ReLU(inplace=True),
        )

        # Upsampling between scales
        self.upsample_1_to_2 = nn.Sequential(
            nn.ConvTranspose2d(s1_ch, s2_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(s2_ch),
            nn.ReLU(inplace=True),
        )
        self.upsample_2_to_3 = nn.Sequential(
            nn.ConvTranspose2d(s2_ch, s3_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(s3_ch),
            nn.ReLU(inplace=True),
        )

        # Per-scale processing
        self.scale_blocks = nn.ModuleList([
            ConvScaleBlock(ch, cfg.depth_embed_dim, cfg.num_blocks, cfg.dropout)
            for ch in scale_channels
        ])

        # Downsamplers for scales 2 and 3 back to 32x64
        self.downsample_2 = nn.Sequential(
            nn.Conv2d(s2_ch, s2_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(s2_ch),
            nn.ReLU(inplace=True),
        )
        self.downsample_3 = nn.Sequential(
            nn.Conv2d(s3_ch, s3_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(s3_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(s3_ch, s3_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(s3_ch),
            nn.ReLU(inplace=True),
        )

        total_ch = sum(scale_channels)
        self.projection = nn.Conv2d(total_ch, cfg.output_dim, kernel_size=1)

    def _get_depth_embeds(
        self,
        depth: torch.Tensor,
        spatial_sizes: List[tuple[int, int]],
    ) -> List[torch.Tensor]:
        """Resize depth to each scale and compute sinusoidal embeddings.

        Args:
            depth: Raw depth (B, 1, H_d, W_d).
            spatial_sizes: List of (H, W) for each scale.

        Returns:
            List of depth embeddings, each (B, depth_embed_dim, H_s, W_s).
        """
        embeds: List[torch.Tensor] = []
        for h, w in spatial_sizes:
            d = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
            embeds.append(sinusoidal_depth_encoding(d, self.cfg.depth_embed_dim))
        return embeds

    def forward(
        self, features: torch.Tensor, depth: torch.Tensor
    ) -> torch.Tensor:
        """Transform DINOv3 features with multi-scale depth-conditioned conv.

        Args:
            features: DINOv3 features (B, N, D) where N=2048, D=1024.
            depth: Depth map (B, 1, H_d, W_d).

        Returns:
            Transformed features (B, 2048, output_dim), L2-normalized.
        """
        B = features.shape[0]
        h, w = self.cfg.spatial_h, self.cfg.spatial_w

        x = features.permute(0, 2, 1).contiguous().reshape(B, self.cfg.input_dim, h, w)
        x = self.stem(x)

        spatial_sizes = [(h, w), (h * 2, w * 2), (h * 4, w * 4)]
        depth_embeds = self._get_depth_embeds(depth, spatial_sizes)

        # Scale 1: 32x64
        s1 = self.scale_blocks[0](x, depth_embeds[0])

        # Scale 2: 64x128
        s2 = self.upsample_1_to_2(s1)
        s2 = self.scale_blocks[1](s2, depth_embeds[1])

        # Scale 3: 128x256
        s3 = self.upsample_2_to_3(s2)
        s3 = self.scale_blocks[2](s3, depth_embeds[2])

        # Downsample all to 32x64
        s2_down = self.downsample_2(s2)
        s3_down = self.downsample_3(s3)

        merged = torch.cat([s1, s2_down, s3_down], dim=1)
        out = self.projection(merged)

        out = out.reshape(B, self.cfg.output_dim, -1).permute(0, 2, 1).contiguous()
        return F.normalize(out, dim=-1)
