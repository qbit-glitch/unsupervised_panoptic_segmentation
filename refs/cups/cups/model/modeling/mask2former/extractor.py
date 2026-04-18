"""Extractor: SPM features (Q) attend to DINOv3 patch tokens (K, V)."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .injector import _flatten_bchw

__all__ = ["Extractor"]


class Extractor(nn.Module):
    """Cross-attention block: flattened SPM features query ViT patch tokens.

    Mirrors :class:`Injector`'s design (pre-LN, gated residual, parallel MLP
    branch off ``q_norm``). The ``gate`` scalar is zero-initialised so the
    block is identity at step 0 and learns how much ViT context to absorb.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 8, mlp_ratio: float = 0.5, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
        vit_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shapes = [c2.shape, c3.shape, c4.shape]
        q_flat = torch.cat(
            [_flatten_bchw(c2), _flatten_bchw(c3), _flatten_bchw(c4)],
            dim=1,
        )
        kv_flat = _flatten_bchw(vit_feat)
        q_norm = self.norm_q(q_flat)
        kv_norm = self.norm_kv(kv_flat)
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm, need_weights=False)
        q = q_flat + self.gate * attn_out
        q = q + self.gate * self.mlp(q_norm)
        # Split back to three levels.
        out = []
        offset = 0
        for shape in shapes:
            B, C, H, W = shape
            n = H * W
            out.append(q[:, offset : offset + n, :].transpose(1, 2).reshape(B, C, H, W))
            offset += n
        return tuple(out)  # type: ignore[return-value]
