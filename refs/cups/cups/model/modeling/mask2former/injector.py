"""Injector: DINOv3 patch tokens (Q) attend to SPM features (K, V).

Adds spatial priors from the convolutional stem back into the frozen ViT
tokens. Followed optionally by an MLP. Residual gating controls how much
of the injected signal is kept (learnable scalar, initialized at 0 so
early training behaves like identity).
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["Injector"]


def _flatten_bchw(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    return x.flatten(2).transpose(1, 2)  # B, H*W, C


class Injector(nn.Module):
    """Cross-attention block where ViT tokens query concatenated SPM features."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 8, mlp_ratio: float = 0.5, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        # Residual gate initialized at 0 (preserves ViT behavior early).
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        vit_feat: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
    ) -> torch.Tensor:
        B, C, Hv, Wv = vit_feat.shape
        q = _flatten_bchw(vit_feat)                      # B, Hv*Wv, C
        kv = torch.cat(
            [_flatten_bchw(c2), _flatten_bchw(c3), _flatten_bchw(c4)],
            dim=1,
        )                                                # B, sum(H*W), C
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm, need_weights=False)
        q = q + self.gate * attn_out
        q = q + self.gate * self.mlp(q_norm)
        return q.transpose(1, 2).reshape(B, C, Hv, Wv)
