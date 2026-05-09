"""Architecture D: Conv + Transformer Hybrid with depth cross-attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_arch
from .base import AdapterConfig, BaseAdapter, ResBlock, sinusoidal_depth_encoding


def sinusoidal_2d_pos_encoding(
    h: int, w: int, dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Generate 2D sinusoidal positional encoding.

    Args:
        h: Height in patches.
        w: Width in patches.
        dim: Embedding dimension (split half for y, half for x).
        device: Target device.
        dtype: Target dtype.

    Returns:
        Positional encoding (1, H*W, dim).
    """
    half = dim // 2
    y_pos = torch.arange(h, device=device, dtype=dtype).unsqueeze(1).expand(h, w)
    x_pos = torch.arange(w, device=device, dtype=dtype).unsqueeze(0).expand(h, w)

    freqs = torch.pow(
        10000.0,
        -torch.arange(0, half, 2, device=device, dtype=dtype) / half,
    )

    y_flat = y_pos.reshape(-1, 1)
    x_flat = x_pos.reshape(-1, 1)
    freqs = freqs.unsqueeze(0)

    pe_y = torch.cat([torch.sin(y_flat * freqs), torch.cos(y_flat * freqs)], dim=-1)
    pe_x = torch.cat([torch.sin(x_flat * freqs), torch.cos(x_flat * freqs)], dim=-1)

    pe = torch.cat([pe_y, pe_x], dim=-1)
    return pe[:, :dim].unsqueeze(0)


class DepthCrossAttention(nn.Module):
    """Cross-attention where queries are spatial tokens and keys/values are depth."""

    def __init__(self, d_model: int, num_heads: int, depth_dim: int) -> None:
        super().__init__()
        self.depth_proj = nn.Linear(depth_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, depth_embed: torch.Tensor
    ) -> torch.Tensor:
        """Apply depth cross-attention.

        Args:
            x: Spatial tokens (B, N, D).
            depth_embed: Depth embedding (B, N, depth_dim).

        Returns:
            Attended features (B, N, D).
        """
        kv = self.depth_proj(depth_embed)
        attn_out, _ = self.cross_attn(x, kv, kv)
        return self.norm(x + attn_out)


class TransformerBlockWithDepth(nn.Module):
    """Transformer encoder block followed by depth cross-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        depth_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.depth_cross_attn = DepthCrossAttention(d_model, num_heads, depth_dim)

    def forward(
        self, x: torch.Tensor, depth_embed: torch.Tensor
    ) -> torch.Tensor:
        """Self-attention then depth cross-attention.

        Args:
            x: Tokens (B, N, D).
            depth_embed: Depth features (B, N, depth_dim).

        Returns:
            Processed tokens (B, N, D).
        """
        x = self.self_attn_block(x)
        x = self.depth_cross_attn(x, depth_embed)
        return x


class ConvBranch(nn.Module):
    """High-resolution conv branch operating at 64x128 and 128x256."""

    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__()
        mid_ch = cfg.hidden_dim
        high_ch = cfg.hidden_dim // 2

        # Stem reduces from input_dim to hidden_dim before upsampling
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.input_dim, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

        self.upsample_to_mid = nn.Sequential(
            nn.ConvTranspose2d(mid_ch, mid_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.upsample_to_high = nn.Sequential(
            nn.ConvTranspose2d(mid_ch, high_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(high_ch),
            nn.ReLU(inplace=True),
        )

        self.mid_blocks = nn.ModuleList([
            ResBlock(mid_ch, depth_dim=cfg.depth_embed_dim, dropout=cfg.dropout)
            for _ in range(cfg.num_blocks)
        ])
        self.high_blocks = nn.ModuleList([
            ResBlock(high_ch, depth_dim=cfg.depth_embed_dim, dropout=cfg.dropout)
            for _ in range(cfg.num_blocks)
        ])

        self.downsample_mid = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.downsample_high = nn.Sequential(
            nn.Conv2d(high_ch, high_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(high_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(high_ch, high_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(high_ch),
            nn.ReLU(inplace=True),
        )

        self.out_channels = mid_ch + high_ch

    def forward(
        self,
        x: torch.Tensor,
        depth: torch.Tensor,
        depth_embed_dim: int,
    ) -> torch.Tensor:
        """Run high-res conv processing.

        Args:
            x: Input features (B, C, 32, 64) at input_dim channels.
            depth: Raw depth (B, 1, H_d, W_d).
            depth_embed_dim: Dimension for sinusoidal encoding.

        Returns:
            Concatenated downsampled conv features (B, out_channels, 32, 64).
        """
        x = self.stem(x)

        mid = self.upsample_to_mid(x)
        mid_h, mid_w = mid.shape[2], mid.shape[3]
        d_mid = F.interpolate(depth, size=(mid_h, mid_w), mode="bilinear", align_corners=False)
        d_mid_embed = sinusoidal_depth_encoding(d_mid, depth_embed_dim)
        for block in self.mid_blocks:
            mid = block(mid, d_mid_embed)

        high = self.upsample_to_high(mid)
        high_h, high_w = high.shape[2], high.shape[3]
        d_high = F.interpolate(depth, size=(high_h, high_w), mode="bilinear", align_corners=False)
        d_high_embed = sinusoidal_depth_encoding(d_high, depth_embed_dim)
        for block in self.high_blocks:
            high = block(high, d_high_embed)

        mid_down = self.downsample_mid(mid)
        high_down = self.downsample_high(high)

        return torch.cat([mid_down, high_down], dim=1)


class TransformerBranch(nn.Module):
    """Low-resolution transformer branch at 32x64 with depth cross-attention."""

    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__()
        d_model = cfg.hidden_dim

        self.input_proj = nn.Linear(cfg.input_dim, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlockWithDepth(
                d_model=d_model,
                num_heads=cfg.num_heads,
                depth_dim=cfg.depth_embed_dim,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.num_blocks)
        ])

        self.out_dim = d_model

    def forward(
        self,
        x: torch.Tensor,
        depth_embed_flat: torch.Tensor,
        pos_enc: torch.Tensor,
    ) -> torch.Tensor:
        """Run transformer with positional encoding and depth cross-attn.

        Args:
            x: Tokens (B, N, D) at input_dim.
            depth_embed_flat: Depth embedding (B, N, depth_dim).
            pos_enc: 2D positional encoding (1, N, hidden_dim).

        Returns:
            Processed tokens (B, N, hidden_dim).
        """
        x = self.input_proj(x)
        x = x + pos_enc
        for block in self.blocks:
            x = block(x, depth_embed_flat)
        return x


@register_arch("conv_transformer")
class ConvTransformerAdapter(BaseAdapter):
    """Conv + Transformer hybrid with FPN merge.

    Conv branch: stem(1024->hidden_dim) -> TransposedConv upsample -> ResBlocks
    Transformer branch: proj(1024->hidden_dim) -> self-attn + depth cross-attn
    FPN merge: concat conv + transformer -> 1x1 conv -> output_dim
    """

    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__(cfg)

        self.conv_branch = ConvBranch(cfg)
        self.transformer_branch = TransformerBranch(cfg)

        merge_ch = self.conv_branch.out_channels + self.transformer_branch.out_dim
        self.fpn_merge = nn.Sequential(
            nn.Conv2d(merge_ch, merge_ch // 2, 1),
            nn.BatchNorm2d(merge_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(merge_ch // 2, cfg.output_dim, 1),
        )

        self._pos_cache: torch.Tensor | None = None
        self._pos_cache_key: tuple[int, int, str] | None = None

    def _get_pos_encoding(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Get cached 2D positional encoding."""
        key = (h, w, str(device))
        if self._pos_cache is None or self._pos_cache_key != key:
            self._pos_cache = sinusoidal_2d_pos_encoding(
                h, w, self.cfg.hidden_dim, device, dtype
            )
            self._pos_cache_key = key
        return self._pos_cache

    def forward(
        self, features: torch.Tensor, depth: torch.Tensor
    ) -> torch.Tensor:
        """Transform features through parallel conv + transformer branches.

        Args:
            features: DINOv3 features (B, N, D) where N=2048, D=1024.
            depth: Depth map (B, 1, H_d, W_d).

        Returns:
            Transformed features (B, 2048, output_dim), L2-normalized.
        """
        B = features.shape[0]
        h, w = self.cfg.spatial_h, self.cfg.spatial_w

        x_spatial = features.permute(0, 2, 1).contiguous().reshape(B, self.cfg.input_dim, h, w)

        # Conv branch: high-res processing
        conv_out = self.conv_branch(x_spatial, depth, self.cfg.depth_embed_dim)

        # Transformer branch: low-res with depth cross-attention
        d_low = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
        d_low_embed = sinusoidal_depth_encoding(d_low, self.cfg.depth_embed_dim)
        d_low_flat = d_low_embed.reshape(B, self.cfg.depth_embed_dim, -1).permute(0, 2, 1).contiguous()

        pos_enc = self._get_pos_encoding(h, w, features.device, features.dtype)
        trans_out = self.transformer_branch(features, d_low_flat, pos_enc)
        trans_out = trans_out.permute(0, 2, 1).contiguous().reshape(
            B, self.transformer_branch.out_dim, h, w
        )

        # FPN merge
        merged = torch.cat([conv_out, trans_out], dim=1)
        out = self.fpn_merge(merged)

        out = out.reshape(B, self.cfg.output_dim, -1).permute(0, 2, 1).contiguous()
        return F.normalize(out, dim=-1)
