"""Depth-aware SGM Adapter.

Cross-attention fusion of frozen DINOv2 features with DepthPro depth, followed
by a small decoder that emits per-thing-class foreground probability maps used
by the superpixel-guided mask losses in
:mod:`mbps_pytorch.losses.superpixel_sgm`.

Sibling to the existing ``superpixel_affinity_adapter.py`` (edge MLP) — does
**not** replace it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from mbps_pytorch.models.semantic.depth_adapter import sinusoidal_depth_encode


@dataclass(frozen=True)
class SGMAdapterConfig:
    """Hyper-parameters for the SGM adapter."""

    d_dino: int = 768
    d_depth: int = 16
    d_fusion: int = 768
    n_fusion_layers: int = 3
    n_heads: int = 8
    n_thing: int = 8
    patch_h: int = 32
    patch_w: int = 64
    use_concat_fusion: bool = False  # True -> cheap MLP fallback


class _XAttnBlock(nn.Module):
    """One cross-attention block: norm -> MHA(q=x, kv=depth) -> norm -> FFN."""

    def __init__(self, d_out: int, n_heads: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_out)
        self.norm_kv = nn.LayerNorm(d_out)
        self.attn = nn.MultiheadAttention(d_out, n_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_out)
        self.ffn = nn.Sequential(
            nn.Linear(d_out, 4 * d_out),
            nn.GELU(),
            nn.Linear(4 * d_out, d_out),
        )

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(x)
        k = v = self.norm_kv(kv)
        attn_out, _ = self.attn(q, k, v, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class FusionXAttn(nn.Module):
    """Depth-conditioned cross-attention fusion of DINOv2 patch features."""

    def __init__(self, config: SGMAdapterConfig) -> None:
        super().__init__()
        self.proj_dino = nn.Linear(config.d_dino, config.d_fusion)
        self.proj_depth = nn.Linear(config.d_depth, config.d_fusion)
        self.blocks = nn.ModuleList(
            [_XAttnBlock(config.d_fusion, config.n_heads) for _ in range(config.n_fusion_layers)]
        )

    def forward(self, f_dino: torch.Tensor, depth_patch: torch.Tensor) -> torch.Tensor:
        """Fuse DINO features with depth via cross-attention.

        Args:
            f_dino: ``(B, N, d_dino)`` frozen patch features.
            depth_patch: ``(B, N)`` or ``(B, N, d_depth)`` depth values in
                ``[0, 1]`` or already sinusoidally encoded.

        Returns:
            ``(B, N, d_fusion)`` fused features.
        """
        if depth_patch.dim() == 2:
            depth_enc = sinusoidal_depth_encode(depth_patch)
        else:
            depth_enc = depth_patch
        x = self.proj_dino(f_dino)
        kv = self.proj_depth(depth_enc)
        for block in self.blocks:
            x = block(x, kv)
        return x


class FusionConcat(nn.Module):
    """Cheap concat+MLP fusion fallback (~50K params)."""

    def __init__(self, config: SGMAdapterConfig) -> None:
        super().__init__()
        in_dim = config.d_dino + config.d_depth
        hidden = config.d_fusion
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, f_dino: torch.Tensor, depth_patch: torch.Tensor) -> torch.Tensor:
        if depth_patch.dim() == 2:
            depth_enc = sinusoidal_depth_encode(depth_patch)
        else:
            depth_enc = depth_patch
        x = torch.cat([f_dino, depth_enc], dim=-1)
        return self.mlp(x)


class SGMHead(nn.Module):
    """Decoder: fused patch features -> per-thing-class foreground probability."""

    def __init__(self, config: SGMAdapterConfig) -> None:
        super().__init__()
        self.patch_h = config.patch_h
        self.patch_w = config.patch_w
        self.up1 = nn.ConvTranspose2d(config.d_fusion, 192, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(192, 96, 4, 2, 1)
        self.mid = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Conv2d(32, config.n_thing, kernel_size=1)

    def forward(self, fbar: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            fbar: ``(B, N, d_fusion)`` with ``N == patch_h * patch_w``.
            out_hw: target ``(H, W)`` for the foreground maps.

        Returns:
            ``(B, n_thing, H, W)`` sigmoid foreground probabilities.
        """
        b, n, d = fbar.shape
        expected = self.patch_h * self.patch_w
        if n != expected:
            raise ValueError(f"Expected N={expected}, got {n}")
        x = fbar.transpose(1, 2).reshape(b, d, self.patch_h, self.patch_w)
        x = F.gelu(self.up1(x))
        x = F.gelu(self.up2(x))
        x = self.mid(x)
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        x = self.head(x)
        return torch.sigmoid(x)


class SGMAdapter(nn.Module):
    """Fusion + SGM head wrapper."""

    def __init__(self, config: SGMAdapterConfig) -> None:
        super().__init__()
        self.config = config
        if config.use_concat_fusion:
            self.fusion: nn.Module = FusionConcat(config)
        else:
            self.fusion = FusionXAttn(config)
        self.head = SGMHead(config)

    def forward(
        self,
        f_dino: torch.Tensor,
        depth_patch: torch.Tensor,
        out_hw: Tuple[int, int] = (512, 1024),
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            f_dino: ``(B, N, d_dino)``.
            depth_patch: ``(B, N)`` raw depth in ``[0, 1]``, or
                ``(B, N, d_depth)`` pre-encoded depth features.
            out_hw: output spatial resolution.

        Returns:
            ``(B, n_thing, H, W)`` foreground probabilities in ``[0, 1]``.
        """
        fbar = self.fusion(f_dino, depth_patch)
        return self.head(fbar, out_hw)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = [
    "SGMAdapterConfig",
    "FusionXAttn",
    "FusionConcat",
    "SGMHead",
    "SGMAdapter",
]
