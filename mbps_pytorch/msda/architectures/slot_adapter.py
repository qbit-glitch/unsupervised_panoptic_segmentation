"""Architecture C: Slot Attention adapter with object-centric bottleneck."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AdapterConfig, BaseAdapter, DepthFiLM, ResBlock, sinusoidal_depth_encoding
from . import register_arch


class SlotAttentionModule(nn.Module):
    """Iterative slot attention with GRU updates and MLP refinement."""

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        input_dim: int,
        num_iters: int = 7,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iters = num_iters
        self.eps = eps

        # Slot initialization parameters (Gaussian)
        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.02)
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # Attention projections
        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.proj_q = nn.Linear(slot_dim, slot_dim)
        self.proj_k = nn.Linear(input_dim, slot_dim)
        self.proj_v = nn.Linear(input_dim, slot_dim)

        # GRU update
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # MLP refinement
        self.norm_mlp = nn.LayerNorm(slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim * 4, slot_dim),
        )

        self._scale = slot_dim ** -0.5

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run iterative slot attention.

        Args:
            features: Input features (B, N, D_in).

        Returns:
            slots: Final slot representations (B, K, D_slot).
            attn_weights: Last-iteration attention weights (B, N, K).
        """
        b, n, _ = features.shape

        # Initialize slots from learned Gaussian
        sigma = self.slot_log_sigma.exp()
        slots = self.slot_mu + sigma * torch.randn(
            b, self.num_slots, self.slot_dim, device=features.device
        )

        features_normed = self.norm_input(features)
        k = self.proj_k(features_normed)  # (B, N, D_slot)
        v = self.proj_v(features_normed)  # (B, N, D_slot)

        attn_weights = None
        for _ in range(self.num_iters):
            slots_prev = slots
            slots_normed = self.norm_slots(slots)
            q = self.proj_q(slots_normed)  # (B, K, D_slot)

            # Attention: slots compete for features (softmax over slots)
            logits = torch.bmm(k, q.transpose(1, 2)) * self._scale  # (B, N, K)
            attn_weights = F.softmax(logits, dim=-1)  # (B, N, K)

            # Weighted mean aggregation per slot
            attn_norm = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + self.eps)
            updates = torch.bmm(attn_norm.transpose(1, 2), v)  # (B, K, D_slot)

            # GRU update
            slots = self.gru(
                updates.reshape(b * self.num_slots, self.slot_dim),
                slots_prev.reshape(b * self.num_slots, self.slot_dim),
            ).reshape(b, self.num_slots, self.slot_dim)

            # MLP refinement with residual
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn_weights


@register_arch("slot")
class SlotAdapter(BaseAdapter):
    """Slot attention adapter projecting features through object-centric bottleneck.

    Encodes spatial features to 128x256, runs slot attention to discover
    object-like groupings, then broadcasts slot features back to pixels
    via attention weights.
    """

    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__(cfg)
        hd = cfg.hidden_dim
        depth_dim = cfg.depth_embed_dim
        out_h, out_w = cfg.spatial_h * 4, cfg.spatial_w * 4  # 128×256

        # Conv encoder: 1024 -> hidden_dim at 128×256
        self.encoder = nn.Sequential(
            nn.Conv2d(cfg.input_dim, hd, 1),
            nn.BatchNorm2d(hd),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hd, hd, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hd),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hd, hd, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hd),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            ResBlock(hd, depth_dim=depth_dim, dropout=cfg.dropout),
            ResBlock(hd, depth_dim=depth_dim, dropout=cfg.dropout),
        )

        # Depth positional encoding projection
        self.depth_proj = nn.Linear(depth_dim, hd)

        # Slot attention
        self.slot_attn = SlotAttentionModule(
            num_slots=cfg.num_slots,
            slot_dim=hd,
            input_dim=hd,
            num_iters=cfg.slot_iters,
        )

        # Output projection from slot_dim to output_dim
        self.out_proj = nn.Linear(hd, cfg.output_dim)

        self._out_h = out_h
        self._out_w = out_w

    def forward(
        self, features: torch.Tensor, depth: torch.Tensor
    ) -> torch.Tensor:
        """Transform features through slot attention bottleneck.

        Args:
            features: DINOv3 features (B, 2048, 1024), L2-normalized.
            depth: Depth map (B, 1, 512, 1024).

        Returns:
            Per-pixel slot features (B, 32768, output_dim), L2-normalized.
        """
        b = features.shape[0]
        h, w = self.cfg.spatial_h, self.cfg.spatial_w
        out_h, out_w = self._out_h, self._out_w

        # Reshape to spatial: (B, 2048, 1024) -> (B, 1024, 32, 64)
        x = features.permute(0, 2, 1).contiguous().reshape(b, self.cfg.input_dim, h, w)

        # Encode and upsample to 128×256
        x = self.encoder(x)       # (B, hd, 32, 64)
        x = self.upsample(x)      # (B, hd, 128, 256)

        # Depth PE at output resolution for ResBlock FiLM
        depth_resized = F.interpolate(
            depth, size=(out_h, out_w), mode="bilinear", align_corners=False,
        )
        depth_pe = sinusoidal_depth_encoding(depth_resized, self.cfg.depth_embed_dim)

        # ResBlocks with depth FiLM conditioning
        for block in self.res_blocks:
            x = block(x, depth_pe)

        # Flatten to sequence
        x_flat = x.flatten(2).permute(0, 2, 1).contiguous()  # (B, 32768, hd)

        # Add depth positional encoding to features
        depth_pe_flat = depth_pe.flatten(2).permute(0, 2, 1).contiguous()  # (B, 32768, depth_dim)
        x_flat = x_flat + self.depth_proj(depth_pe_flat)

        # Slot attention
        slots, attn_weights = self.slot_attn(x_flat)  # slots: (B, K, hd), attn: (B, 32768, K)

        # Project slots to output dim
        slot_features = self.out_proj(slots)  # (B, K, output_dim)

        # Broadcast: each pixel gets its assigned slot's features (weighted by attn)
        # attn_weights: (B, 32768, K), slot_features: (B, K, output_dim)
        out = torch.bmm(attn_weights, slot_features)  # (B, 32768, output_dim)

        return F.normalize(out, p=2, dim=-1)
