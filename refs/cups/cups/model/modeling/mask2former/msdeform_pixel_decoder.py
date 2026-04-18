"""MSDeformAttn-style pixel decoder (pure-torch fallback).

Fuses multi-scale pyramid levels via a stack of transformer encoder
layers operating on concatenated flattened tokens, then up-samples a
mask feature map at stride 4.

p2 is NOT included in self-attention (stride-4 token count is
prohibitive). It participates only via a 1x1 lateral projection into
the final mask feature.

TODO(oom): dense self-attention over concatenated p3+p4+p5 tokens
scales as O(N^2). At 640x1280 crops N~16k, so per-layer attention
matrix is ~1-2 GB per head-batch. Switch to
F.scaled_dot_product_attention (flash backend) or window/local
attention before full-scale training.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MSDeformAttnPixelDecoder"]


def _level_embedding(num_levels: int, dim: int) -> nn.Parameter:
    """Learnable level embedding, one vector per pyramid scale."""
    return nn.Parameter(torch.randn(num_levels, dim) * 0.02)


class _MSALayer(nn.Module):
    """Single encoder layer: self-attention over all flattened tokens."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(self, in_channels: int = 256, hidden_dim: int = 256, mask_dim: int = 256, num_layers: int = 6, num_heads: int = 8) -> None:
        super().__init__()
        # mask_conv uses GroupNorm(32, mask_dim); guard divisibility up-front.
        assert mask_dim % 32 == 0, f"mask_dim={mask_dim} must be divisible by 32 for GroupNorm"
        self.hidden_dim = hidden_dim
        self.mask_dim = mask_dim
        # Project each pyramid level (already hidden_dim) + add level embedding.
        self.level_embed = _level_embedding(num_levels=4, dim=hidden_dim)
        self.layers = nn.ModuleList([_MSALayer(hidden_dim, num_heads) for _ in range(num_layers)])
        # Upsample fused p3 -> p2 resolution to produce mask feature.
        self.lateral_p3 = nn.Conv2d(hidden_dim, mask_dim, kernel_size=1)
        self.lateral_p2 = nn.Conv2d(in_channels, mask_dim, kernel_size=1)
        self.mask_conv = nn.Sequential(
            nn.Conv2d(mask_dim, mask_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, mask_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mask_dim, mask_dim, kernel_size=1),
        )

    def _flatten_levels(self, feats: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[int, int, int]]]:
        """Flatten {p2..p5} (we use p3, p4, p5 for attention) and add level emb."""
        tokens = []
        shapes: List[Tuple[int, int, int]] = []
        for lvl, key in enumerate(["p3", "p4", "p5"]):
            x = feats[key]
            B, C, H, W = x.shape
            shapes.append((B, H, W))
            t = x.flatten(2).transpose(1, 2)                # B, H*W, C
            t = t + self.level_embed[lvl]
            tokens.append(t)
        x_cat = torch.cat(tokens, dim=1)                    # B, sum(H*W), C
        return x_cat, shapes

    def _split_levels(self, x_cat: torch.Tensor, shapes: List[Tuple[int, int, int]]) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        offset = 0
        for (B, H, W) in shapes:
            n = H * W
            t = x_cat[:, offset : offset + n, :]
            out.append(t.transpose(1, 2).reshape(B, -1, H, W))
            offset += n
        return out

    def forward(self, feats: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x_cat, shapes = self._flatten_levels(feats)
        for layer in self.layers:
            x_cat = layer(x_cat)
        multi_scale = self._split_levels(x_cat, shapes)     # list of [p3, p4, p5]
        # Build mask feature at stride 4: upsample p3 + add lateral p2.
        p3 = multi_scale[0]
        p2 = feats["p2"]
        p3_up = F.interpolate(self.lateral_p3(p3), size=p2.shape[-2:], mode="bilinear", align_corners=False)
        mask_feat = self.mask_conv(p3_up + self.lateral_p2(p2))
        return mask_feat, multi_scale
