"""Architecture B: Transformer Pyramid adapter with multi-scale processing."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AdapterConfig, BaseAdapter, DepthFiLM, ResBlock, sinusoidal_depth_encoding
from . import register_arch


class LearnablePositionalEncoding2D(nn.Module):
    """Learnable 2D positional encoding for spatial grids."""

    def __init__(self, dim: int, h: int, w: int) -> None:
        super().__init__()
        self.h = h
        self.w = w
        self.pos_embed = nn.Parameter(torch.randn(1, dim, h, w) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to (B, C, H, W) features."""
        return x + self.pos_embed


class DepthCrossAttention(nn.Module):
    """Cross-attention where features attend to depth positional encoding."""

    def __init__(self, d_model: int, num_heads: int, depth_dim: int = 16) -> None:
        super().__init__()
        self.depth_proj = nn.Linear(depth_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, depth_pe: torch.Tensor
    ) -> torch.Tensor:
        """Apply depth cross-attention.

        Args:
            x: Sequence features (B, N, D).
            depth_pe: Depth positional encoding (B, depth_dim, H, W).

        Returns:
            Cross-attended features (B, N, D).
        """
        b, n, d = x.shape
        # Flatten depth PE to sequence: (B, depth_dim, H, W) -> (B, H*W, D)
        depth_kv = depth_pe.flatten(2).permute(0, 2, 1).contiguous()  # (B, H*W, depth_dim)
        depth_kv = self.depth_proj(depth_kv)  # (B, H*W, D)

        residual = x
        x_normed = self.norm(x)
        attn_out, _ = self.cross_attn(x_normed, depth_kv, depth_kv)
        return residual + attn_out


class PyramidScale(nn.Module):
    """Single scale of the transformer pyramid with self-attention + depth cross-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        h: int,
        w: int,
        dropout: float = 0.1,
        depth_dim: int = 16,
    ) -> None:
        super().__init__()
        self.h = h
        self.w = w
        self.pos_enc = LearnablePositionalEncoding2D(d_model, h, w)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.depth_cross_attn = DepthCrossAttention(d_model, num_heads, depth_dim)

    def forward(self, x: torch.Tensor, depth_pe: torch.Tensor) -> torch.Tensor:
        """Process one pyramid scale.

        Args:
            x: Feature map (B, C, H, W).
            depth_pe: Depth PE resized to this scale (B, depth_dim, H, W).

        Returns:
            Processed feature map (B, C, H, W).
        """
        x = self.pos_enc(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).permute(0, 2, 1).contiguous()  # (B, H*W, C)

        tokens = self.self_attn(tokens)
        tokens = self.depth_cross_attn(tokens, depth_pe)

        return tokens.permute(0, 2, 1).contiguous().reshape(b, c, h, w)


class ConvScale(nn.Module):
    """Conv-only scale for high-resolution (too many tokens for attention)."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        depth_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock(channels, depth_dim=depth_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, depth_pe: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, depth_pe)
        return x


@register_arch("transformer")
class TransformerPyramid(BaseAdapter):
    """Multi-scale transformer pyramid with FPN merge.

    Scale 1 (32x64, 1024-dim): full transformer + depth cross-attn
    Scale 2 (64x128, 512-dim): half transformer + depth cross-attn
    Scale 3 (128x256, 256-dim): conv-only with DepthFiLM (32K tokens)
    """

    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__(cfg)
        depth_dim = cfg.depth_embed_dim

        # Scale 1: native resolution (32×64, 1024-dim)
        self.scale1 = PyramidScale(
            d_model=cfg.input_dim,
            num_heads=cfg.num_heads,
            num_blocks=cfg.num_blocks,
            h=cfg.spatial_h,
            w=cfg.spatial_w,
            dropout=cfg.dropout,
            depth_dim=depth_dim,
        )

        # Scale 2: upsampled (64×128, 512-dim)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(cfg.input_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.scale2 = PyramidScale(
            d_model=512,
            num_heads=max(1, cfg.num_heads // 2),
            num_blocks=max(1, cfg.num_blocks // 2),
            h=cfg.spatial_h * 2,
            w=cfg.spatial_w * 2,
            dropout=cfg.dropout,
            depth_dim=depth_dim,
        )

        # Scale 3: high-res (128×256, 256-dim), conv-only
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(cfg.input_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.scale3 = ConvScale(
            channels=256,
            num_blocks=cfg.num_blocks,
            depth_dim=depth_dim,
            dropout=cfg.dropout,
        )

        # FPN merge: downsample all to 32×64, concat, project
        merge_dim = cfg.input_dim + 512 + 256  # 1792
        self.merge_proj = nn.Sequential(
            nn.Conv2d(merge_dim, cfg.output_dim, 1),
            nn.BatchNorm2d(cfg.output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, features: torch.Tensor, depth: torch.Tensor
    ) -> torch.Tensor:
        """Transform features through multi-scale pyramid.

        Args:
            features: DINOv3 features (B, 2048, 1024), L2-normalized.
            depth: Depth map (B, 1, 512, 1024).

        Returns:
            Transformed features (B, 2048, output_dim), L2-normalized.
        """
        b = features.shape[0]
        h, w = self.cfg.spatial_h, self.cfg.spatial_w

        # Reshape to spatial: (B, 2048, 1024) -> (B, 1024, 32, 64)
        x = features.permute(0, 2, 1).contiguous().reshape(b, self.cfg.input_dim, h, w)

        # Depth PE at each scale
        depth_s1 = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
        depth_pe_s1 = sinusoidal_depth_encoding(depth_s1, self.cfg.depth_embed_dim)

        depth_s2 = F.interpolate(depth, size=(h * 2, w * 2), mode="bilinear", align_corners=False)
        depth_pe_s2 = sinusoidal_depth_encoding(depth_s2, self.cfg.depth_embed_dim)

        depth_s3 = F.interpolate(depth, size=(h * 4, w * 4), mode="bilinear", align_corners=False)
        depth_pe_s3 = sinusoidal_depth_encoding(depth_s3, self.cfg.depth_embed_dim)

        # Process each scale
        out1 = self.scale1(x, depth_pe_s1)  # (B, 1024, 32, 64)

        x2 = self.upsample2(x)  # (B, 512, 64, 128)
        out2 = self.scale2(x2, depth_pe_s2)

        x3 = self.upsample3(x)  # (B, 256, 128, 256)
        out3 = self.scale3(x3, depth_pe_s3)

        # FPN merge: downsample to 32×64 and concatenate
        out2_down = F.adaptive_avg_pool2d(out2, (h, w))  # (B, 512, 32, 64)
        out3_down = F.adaptive_avg_pool2d(out3, (h, w))  # (B, 256, 32, 64)
        merged = torch.cat([out1, out2_down, out3_down], dim=1)  # (B, 1792, 32, 64)

        out = self.merge_proj(merged)  # (B, output_dim, 32, 64)

        # Reshape back to sequence and L2-normalize
        out = out.flatten(2).permute(0, 2, 1).contiguous()  # (B, 2048, output_dim)
        return F.normalize(out, p=2, dim=-1)
