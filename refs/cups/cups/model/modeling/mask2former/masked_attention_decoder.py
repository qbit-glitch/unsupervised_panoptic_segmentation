"""Masked-attention transformer decoder (Mask2Former).

- L layers of (cross-attn with mask, self-attn, FFN) applied to queries.
- Cycles through multi_scale[0], [1], [2] as cross-attn memory.
- Emits per-layer (pred_logits, pred_masks); last is main, rest are aux.
- Hooks:
    * depth_bias query pool (N2) -> decoder takes optional `depth` kwarg.
    * return_query_embeds -> also return final query embeddings (B, Q, C)
      for N3/N4 losses.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MaskedAttentionDecoder"]


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2)


class _CrossAttnLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(
        self,
        q: torch.Tensor,
        memory: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Cross-attention with optional masked-attention on background queries.
        h = self.cross_norm(q)
        a, _ = self.cross_attn(h, memory, memory, attn_mask=attn_mask, need_weights=False)
        q = q + a
        # Self-attention over queries.
        h = self.self_norm(q)
        a, _ = self.self_attn(h, h, h, need_weights=False)
        q = q + a
        # Feed-forward.
        q = q + self.ff(self.ff_norm(q))
        return q


class MaskedAttentionDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_classes: int = 20,
        num_layers: int = 9,
        num_heads: int = 8,
        query_pool: Optional[nn.Module] = None,
        droppath: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.query_pool = query_pool
        self.layers = nn.ModuleList([_CrossAttnLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)   # +1 for no-object
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.droppath = droppath

    def _pred(self, q: torch.Tensor, mask_feat: torch.Tensor):
        logits = self.class_embed(q)                         # B, Q, K+1
        mask_embed = self.mask_embed(q)                       # B, Q, C
        masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feat)
        return logits, masks, mask_embed

    def forward(
        self,
        mask_feat: torch.Tensor,
        multi_scale: List[torch.Tensor],
        depth: Optional[torch.Tensor] = None,
        return_query_embeds: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B = mask_feat.shape[0]
        q = self.query_pool(batch_size=B, depth=depth) if self.query_pool is not None else None
        assert q is not None, "query_pool must be provided"
        aux_outputs: List[Dict[str, torch.Tensor]] = []
        _, init_masks, _ = self._pred(q, mask_feat)
        # Initial attn mask from first prediction (sets binary threshold).
        for i, layer in enumerate(self.layers):
            level = multi_scale[i % len(multi_scale)]
            memory = _flatten(level)                         # B, N, C
            # Mask-attention: compute binary mask from current prediction
            # at this level's resolution to gate cross-attention.
            with torch.no_grad():
                pred_at_level = F.interpolate(init_masks, size=level.shape[-2:], mode="bilinear", align_corners=False)
                attn_mask = (pred_at_level.sigmoid() < 0.5)
                attn_mask = attn_mask.flatten(2).detach()
                attn_mask = attn_mask.repeat_interleave(layer.cross_attn.num_heads, dim=0)
                attn_mask = attn_mask.where(attn_mask.sum(-1, keepdim=True) != attn_mask.shape[-1], torch.zeros_like(attn_mask))
            q = layer(q, memory, attn_mask=attn_mask)
            logits, masks, _ = self._pred(q, mask_feat)
            # DropPath during training (simple stochastic depth).
            if self.training and self.droppath > 0.0 and torch.rand(1).item() < self.droppath:
                continue
            if i < self.num_layers - 1:
                aux_outputs.append({"pred_logits": logits, "pred_masks": masks})
            init_masks = masks
        out = {"pred_logits": logits, "pred_masks": masks, "aux_outputs": aux_outputs}
        if return_query_embeds:
            out["query_embeds"] = q
        return out
