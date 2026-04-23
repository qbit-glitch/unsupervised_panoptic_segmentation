"""DCFA v2 Adapter Variants for ablation and DCFA-X combined architecture.

All variants share the same interface:
    forward(codes, depth, **kwargs) → adjusted_codes

Variants:
    DepthAdapterFiLM       — B1: FiLM conditioning (depth→γ,β modulation)
    DepthAdapterCrossAttn  — A1: Cross-attention with DINOv2 768D
    DepthAdapterDeep       — B2: 4-layer bottleneck MLP
    DepthAdapterWindowAttn — B3: Local 3×3 patch neighborhood attention
    DepthAdapterX          — Combined: FiLM + cross-attention + fusion MLP
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from mbps_pytorch.models.semantic.depth_adapter import (
    DepthAdapter,
    sinusoidal_depth_encode,
)

# ─── Registry ─────────────────────────────────────────────────────────────────

ADAPTER_REGISTRY: Dict[str, Type[nn.Module]] = {
    "v3": DepthAdapter,
}


def register_adapter(name: str):
    """Decorator to register an adapter variant."""
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        ADAPTER_REGISTRY[name] = cls
        return cls
    return decorator


def create_adapter(name: str, **kwargs) -> nn.Module:
    """Factory to create adapter by name.

    Args:
        name: Adapter variant name (v3, film, cross_attn, deep, window_attn, x).
        **kwargs: Passed to adapter constructor.

    Returns:
        Adapter module instance.
    """
    if name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown adapter: {name}. Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    return ADAPTER_REGISTRY[name](**kwargs)


# ─── B1: FiLM Conditioning ───────────────────────────────────────────────────

@register_adapter("film")
class DepthAdapterFiLM(nn.Module):
    """FiLM conditioning: depth features modulate codes via learned (gamma, beta).

    Instead of concatenating depth to codes, depth produces multiplicative
    and additive modulation factors per code dimension.
    """

    def __init__(
        self,
        code_dim: int = 90,
        depth_dim: int = 16,
        hidden_dim: int = 384,
        num_layers: int = 2,
        geo_dim: int = 0,
    ) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.depth_dim = depth_dim + geo_dim  # depth + optional normals/gradients

        # FiLM generator: depth features → (gamma, beta)
        self.film_net = nn.Sequential(
            nn.Linear(self.depth_dim, 64),
            nn.ReLU(),
            nn.Linear(64, code_dim * 2),  # gamma + beta
        )

        # MLP on modulated codes
        layers = []
        in_dim = code_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.out_linear = nn.Linear(hidden_dim, code_dim)
        nn.init.zeros_(self.out_linear.weight)
        nn.init.zeros_(self.out_linear.bias)

    def forward(
        self, codes: torch.Tensor, depth: torch.Tensor, **kwargs,
    ) -> torch.Tensor:
        if depth.dim() == 2:
            depth = depth.unsqueeze(-1)

        film_params = self.film_net(depth)  # (B, N, 2*code_dim)
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma = gamma + 1.0  # center around 1 for identity init
        codes_mod = gamma * codes + beta

        x = self.mlp(codes_mod)
        residual = self.out_linear(x)
        return codes + residual


# ─── A1: Cross-Attention with DINOv2 768D ────────────────────────────────────

@register_adapter("cross_attn")
class DepthAdapterCrossAttn(nn.Module):
    """Cross-attention: queries from codes, keys/values from DINOv2 768D.

    Recovers information lost in the 768→90 projection by attending
    back to the original DINOv2 features.
    """

    def __init__(
        self,
        code_dim: int = 90,
        depth_dim: int = 16,
        dino_dim: int = 768,
        d_attn: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.depth_dim = depth_dim
        self.d_attn = d_attn

        # Depth-conditioned query projection
        self.wq = nn.Linear(code_dim + depth_dim, d_attn)
        self.wk = nn.Linear(dino_dim, d_attn)
        self.wv = nn.Linear(dino_dim, d_attn)

        # Fusion MLP: [codes; attn_out] → residual
        self.fusion = nn.Sequential(
            nn.Linear(code_dim + d_attn, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, code_dim),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(
        self,
        codes: torch.Tensor,
        depth: torch.Tensor,
        dino768: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if dino768 is None:
            raise ValueError("DepthAdapterCrossAttn requires dino768 input")
        if depth.dim() == 2:
            depth = depth.unsqueeze(-1)

        q_input = torch.cat([codes, depth], dim=-1)  # (B, N, code_dim+depth_dim)
        q = self.wq(q_input)    # (B, N, d_attn)
        k = self.wk(dino768)    # (B, N, d_attn)
        v = self.wv(dino768)    # (B, N, d_attn)

        # Per-patch attention (each patch attends to itself — N=1 per position)
        # This is effectively a learned projection from 768D conditioned on codes
        scale = math.sqrt(self.d_attn)
        attn_score = (q * k).sum(dim=-1, keepdim=True) / scale  # (B, N, 1)
        attn_weight = torch.sigmoid(attn_score)  # gating, not softmax (N=1)
        attn_out = attn_weight * v  # (B, N, d_attn)

        fused = torch.cat([codes, attn_out], dim=-1)  # (B, N, code_dim+d_attn)
        residual = self.fusion(fused)
        return codes + residual


# ─── B2: Deeper Bottleneck MLP ───────────────────────────────────────────────

@register_adapter("deep")
class DepthAdapterDeep(nn.Module):
    """4-layer MLP with bottleneck for deeper depth-code interaction."""

    def __init__(
        self,
        code_dim: int = 90,
        depth_dim: int = 16,
        hidden_dim: int = 384,
        bottleneck_dim: int = 64,
    ) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.depth_dim = depth_dim
        in_dim = code_dim + depth_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),  # compress
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, hidden_dim),  # expand
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.out_linear = nn.Linear(hidden_dim, code_dim)
        nn.init.zeros_(self.out_linear.weight)
        nn.init.zeros_(self.out_linear.bias)

    def forward(
        self, codes: torch.Tensor, depth: torch.Tensor, **kwargs,
    ) -> torch.Tensor:
        if depth.dim() == 2:
            depth = depth.unsqueeze(-1)
        x = torch.cat([codes, depth], dim=-1)
        x = self.mlp(x)
        residual = self.out_linear(x)
        return codes + residual


# ─── B3: Local Window Attention ──────────────────────────────────────────────

@register_adapter("window_attn")
class DepthAdapterWindowAttn(nn.Module):
    """3x3 local patch attention before MLP for spatial context.

    Requires spatial dimensions (ph, pw) to reshape flat patches into a grid.
    """

    def __init__(
        self,
        code_dim: int = 90,
        depth_dim: int = 16,
        hidden_dim: int = 384,
        num_layers: int = 2,
        window_size: int = 3,
    ) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.depth_dim = depth_dim
        self.window_size = window_size
        feat_dim = code_dim + depth_dim

        # Window attention: simple local aggregation
        self.attn_proj = nn.Linear(feat_dim, feat_dim)
        self.attn_gate = nn.Linear(feat_dim, 1)

        # MLP on attended features
        layers = []
        in_dim = feat_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.out_linear = nn.Linear(hidden_dim, code_dim)
        nn.init.zeros_(self.out_linear.weight)
        nn.init.zeros_(self.out_linear.bias)

    def _local_attention(
        self, feat_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Apply local window attention on 2D feature map.

        Args:
            feat_2d: (B, C, H, W) feature map.

        Returns:
            (B, C, H, W) locally attended features.
        """
        b, c, h, w = feat_2d.shape
        pad = self.window_size // 2

        # Pad and unfold into local windows
        padded = F.pad(feat_2d, (pad, pad, pad, pad), mode="reflect")
        # Use unfold to get (B, C, H, W, ws, ws) local windows
        windows = padded.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)
        # windows: (B, C, H, W, ws, ws)

        # Reshape for attention: (B, H*W, ws*ws, C)
        ws2 = self.window_size ** 2
        windows = windows.permute(0, 2, 3, 4, 5, 1)  # (B, H, W, ws, ws, C)
        windows = windows.reshape(b, h * w, ws2, c)

        # Center patch as query, all window patches as keys
        center = windows[:, :, ws2 // 2:ws2 // 2 + 1, :]  # (B, HW, 1, C)

        # Gate: how much to attend to each neighbor
        gates = self.attn_gate(windows)  # (B, HW, ws2, 1)
        gates = F.softmax(gates, dim=2)

        # Weighted sum of projected window features
        proj = self.attn_proj(windows)  # (B, HW, ws2, C)
        attended = (gates * proj).sum(dim=2)  # (B, HW, C)

        return attended.reshape(b, h, w, c).permute(0, 3, 1, 2)  # (B, C, H, W)

    def forward(
        self,
        codes: torch.Tensor,
        depth: torch.Tensor,
        spatial_shape: Optional[tuple] = None,
        **kwargs,
    ) -> torch.Tensor:
        if depth.dim() == 2:
            depth = depth.unsqueeze(-1)

        b, n, _ = codes.shape
        feat = torch.cat([codes, depth], dim=-1)  # (B, N, code_dim+depth_dim)

        # Reshape to 2D for local attention
        if spatial_shape is not None:
            ph, pw = spatial_shape
        else:
            raise ValueError(
                "spatial_shape required for window_attn (non-square patch grids)"
            )
        feat_2d = feat.reshape(b, ph, pw, -1).permute(0, 3, 1, 2)  # (B, C, ph, pw)

        attended = self._local_attention(feat_2d)  # (B, C, ph, pw)
        attended = attended.permute(0, 2, 3, 1).reshape(b, n, -1)  # (B, N, C)

        x = self.mlp(attended)
        residual = self.out_linear(x)
        return codes + residual


# ─── DCFA-X: Combined Architecture ──────────────────────────────────────────

@register_adapter("x")
class DepthAdapterX(nn.Module):
    """DCFA-X: FiLM + cross-attention + fusion MLP.

    Step 1: FiLM — depth+normals modulate codes via (gamma, beta).
    Step 2: Cross-attention — recover info from 768D DINOv2 features.
    Step 3: Fusion MLP — merge FiLM'd codes + attention output.
    Step 4: Skip connection — codes + zero_init(residual).
    """

    def __init__(
        self,
        code_dim: int = 90,
        depth_dim: int = 16,
        geo_dim: int = 0,
        dino_dim: int = 768,
        d_attn: int = 64,
        fusion_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.depth_dim = depth_dim
        self.geo_dim = geo_dim

        # Step 1: FiLM generator (depth + normals → gamma, beta)
        film_in = depth_dim + geo_dim
        self.film_net = nn.Sequential(
            nn.Linear(film_in, 64),
            nn.ReLU(),
            nn.Linear(64, code_dim * 2),
        )

        # Step 2: Cross-attention (Q=FiLM'd codes, KV=DINOv2 768D)
        self.d_attn = d_attn
        self.wq = nn.Linear(code_dim, d_attn)
        self.wk = nn.Linear(dino_dim, d_attn)
        self.wv = nn.Linear(dino_dim, d_attn)

        # Step 3: Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(code_dim + d_attn, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
        )
        self.out_linear = nn.Linear(fusion_hidden, code_dim)
        nn.init.zeros_(self.out_linear.weight)
        nn.init.zeros_(self.out_linear.bias)

    def forward(
        self,
        codes: torch.Tensor,
        depth: torch.Tensor,
        dino768: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if dino768 is None:
            raise ValueError("DepthAdapterX requires dino768 input")
        if depth.dim() == 2:
            depth = depth.unsqueeze(-1)

        # Step 1: FiLM conditioning
        geo_parts = [depth]
        if normals is not None:
            geo_parts.append(normals)
        geo = torch.cat(geo_parts, dim=-1)  # (B, N, depth_dim + geo_dim)

        film_params = self.film_net(geo)  # (B, N, 2*code_dim)
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma = gamma + 1.0  # identity-centered
        codes_film = gamma * codes + beta

        # Step 2: Cross-attention
        q = self.wq(codes_film)  # (B, N, d_attn)
        k = self.wk(dino768)     # (B, N, d_attn)
        v = self.wv(dino768)     # (B, N, d_attn)

        scale = math.sqrt(self.d_attn)
        attn_score = (q * k).sum(dim=-1, keepdim=True) / scale
        attn_weight = torch.sigmoid(attn_score)
        attn_out = attn_weight * v  # (B, N, d_attn)

        # Step 3: Fusion
        fused = torch.cat([codes_film, attn_out], dim=-1)
        x = self.fusion(fused)
        residual = self.out_linear(x)

        # Step 4: Skip connection
        return codes + residual
