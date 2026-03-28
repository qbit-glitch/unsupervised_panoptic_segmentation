"""CSCMRefineNet: Cross-Modal Refinement Network for semantic pseudo-labels.

Refines CAUSE-TR 27-class semantic pseudo-labels using cross-modal
processing of DINOv2 features conditioned on depth geometry.

Supports two block types:
  - "conv": Depthwise-separable Conv2d blocks (fast, stable, local context)
  - "mamba": VisionMamba2/GatedDeltaNet SSM blocks (long-range, slower)

Architecture:
    DINOv2 (768) → SemanticProjection → sem_proj (bridge_dim)
    DINOv2 (768) + depth → DepthFeatureProjection → depth_proj (bridge_dim)
    [sem_proj, depth_proj] → N × CoupledBlock → [sem_refined, _]
    sem_refined → head → refined_logits (27)

References:
    - Coupled Mamba (Li et al., NeurIPS 2024)
    - DFormerv2 (Yin et al., CVPR 2025)
    - FiLM (Perez et al., AAAI 2018)
    - GatedDeltaNet (Yang et al., ICLR 2025)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .mamba2 import VisionMamba2


class SemanticProjection(nn.Module):
    """Project DINOv2 features to bridge dimension for semantic stream."""

    def __init__(self, feature_dim: int = 768, bridge_dim: int = 192):
        super().__init__()
        self.conv = nn.Conv2d(feature_dim, bridge_dim, 1, bias=False)
        self.norm = nn.GroupNorm(1, bridge_dim)  # instance-norm style
        self.act = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, feature_dim, H, W) → (B, bridge_dim, H, W)"""
        return self.act(self.norm(self.conv(features)))


class DepthFeatureProjection(nn.Module):
    """Project DINOv2 features with depth FiLM conditioning.

    Encodes depth via sinusoidal positional encoding + Sobel gradients,
    then modulates projected DINOv2 features via FiLM (gamma/beta).
    """

    def __init__(
        self,
        feature_dim: int = 768,
        bridge_dim: int = 192,
        depth_freq_bands: int = 6,
    ):
        super().__init__()
        self.bridge_dim = bridge_dim

        # Feature projection
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feature_dim, bridge_dim, 1, bias=False),
            nn.GroupNorm(1, bridge_dim),
            nn.GELU(),
        )

        # Depth encoding: sinusoidal + Sobel gradients → FiLM params
        # Input: sin/cos for each freq band + raw depth + grad_x + grad_y
        depth_input_dim = 2 * depth_freq_bands + 3
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_input_dim, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, bridge_dim * 2, 1),  # gamma + beta for FiLM
        )

        # Pre-compute frequency bands
        self.register_buffer(
            "freq_bands",
            torch.tensor([2**i * math.pi for i in range(depth_freq_bands)]),
        )

    def forward(
        self,
        features: torch.Tensor,
        depth: torch.Tensor,
        depth_grads: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, 768, H, W) DINOv2 patch features
            depth: (B, 1, H, W) normalized depth [0, 1]
            depth_grads: (B, 2, H, W) Sobel_x, Sobel_y of depth
        Returns:
            (B, bridge_dim, H, W) depth-conditioned feature projection
        """
        feat_proj = self.feat_proj(features)

        # Sinusoidal depth encoding
        freqs = self.freq_bands  # (F,)
        d_expanded = depth * freqs[None, :, None, None]  # (B, F, H, W)
        depth_enc = torch.cat([
            torch.sin(d_expanded),
            torch.cos(d_expanded),
            depth,
            depth_grads,
        ], dim=1)  # (B, 2F+3, H, W)

        # FiLM conditioning
        film_params = self.depth_encoder(depth_enc)  # (B, 2*bridge_dim, H, W)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma.clamp(-2.0, 2.0)
        beta = beta.clamp(-2.0, 2.0)

        return feat_proj * (1.0 + gamma) + beta


class _ASPPLiteConv(nn.Module):
    """ASPP-lite: parallel dilated depthwise convolutions at rates {1, 3, 5}.

    Provides multi-scale receptive fields: rate-1 preserves local detail
    for thing boundaries, rates 3 and 5 capture broader context for stuff.
    Inspired by ASPP (Chen et al., TPAMI 2018) with depthwise-separable
    convolutions for parameter efficiency.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.dw1 = nn.Conv2d(d_model, d_model, 3, padding=1, dilation=1, groups=d_model)
        self.dw3 = nn.Conv2d(d_model, d_model, 3, padding=3, dilation=3, groups=d_model)
        self.dw5 = nn.Conv2d(d_model, d_model, 3, padding=5, dilation=5, groups=d_model)
        self.merge = nn.Conv2d(d_model * 3, d_model, 1)
        self.norm = nn.GroupNorm(1, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.dw1(x)
        h3 = self.dw3(x)
        h5 = self.dw5(x)
        merged = self.merge(torch.cat([h1, h3, h5], dim=1))
        return self.act(self.norm(merged))


class UpsampleBilinear(nn.Module):
    """Strategy A: Parameter-free bilinear 4x upsampling."""

    def __init__(self, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )


class UpsampleTransposedConv(nn.Module):
    """Strategy B: 2-stage learnable transposed convolution (2x + 2x = 4x)."""

    def __init__(self, channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                channels, channels, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.ConvTranspose2d(
                channels, channels, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class UpsamplePixelShuffle(nn.Module):
    """Strategy C: Sub-pixel convolution (PixelShuffle) 4x upsampling."""

    def __init__(self, channels: int, scale_factor: int = 4):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * scale_factor**2, 1, bias=False),
            nn.PixelShuffle(scale_factor),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


def _build_upsample(strategy: str, channels: int) -> nn.Module:
    """Factory for upsampling modules."""
    if strategy == "bilinear":
        return UpsampleBilinear(scale_factor=4)
    elif strategy == "transposed_conv":
        return UpsampleTransposedConv(channels)
    elif strategy == "pixel_shuffle":
        return UpsamplePixelShuffle(channels)
    elif strategy == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown upsample strategy: {strategy}")


class CoupledConvBlock(nn.Module):
    """Coupled dual-chain Conv2d block for cross-modal feature fusion.

    Two depthwise-separable conv streams (semantic and depth-feature) with
    learnable cross-chain gating. Operates natively in (B, C, H, W) format
    — no flatten/unflatten overhead.

    When use_aspp=True, replaces the single 3×3 depthwise conv with
    parallel dilated convolutions at rates {1, 3, 5} (ASPP-lite).
    """

    def __init__(
        self,
        d_model: int = 192,
        coupling_strength: float = 0.1,
        use_aspp: bool = False,
        **kwargs,  # accept and ignore Mamba-specific args
    ):
        super().__init__()

        # Pre-norm
        self.norm_sem = nn.GroupNorm(1, d_model)
        self.norm_depth = nn.GroupNorm(1, d_model)

        # Cross-chain coupling: 1×1 conv + sigmoid gate
        self.cross_d2s = nn.Conv2d(d_model, d_model, 1)
        self.cross_s2d = nn.Conv2d(d_model, d_model, 1)
        self.alpha = nn.Parameter(torch.tensor(coupling_strength))
        self.beta = nn.Parameter(torch.tensor(coupling_strength))

        if use_aspp:
            # ASPP-lite: multi-scale dilated convolutions + pointwise
            self.sem_conv = nn.Sequential(
                _ASPPLiteConv(d_model),
                nn.Conv2d(d_model, d_model * 2, 1),
                nn.GELU(),
                nn.Conv2d(d_model * 2, d_model, 1),
            )
            self.depth_conv = nn.Sequential(
                _ASPPLiteConv(d_model),
                nn.Conv2d(d_model, d_model * 2, 1),
                nn.GELU(),
                nn.Conv2d(d_model * 2, d_model, 1),
            )
        else:
            # Standard: depthwise 3×3 + pointwise expand-contract
            self.sem_conv = nn.Sequential(
                nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model),
                nn.GroupNorm(1, d_model),
                nn.GELU(),
                nn.Conv2d(d_model, d_model * 2, 1),
                nn.GELU(),
                nn.Conv2d(d_model * 2, d_model, 1),
            )
            self.depth_conv = nn.Sequential(
                nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model),
                nn.GroupNorm(1, d_model),
                nn.GELU(),
                nn.Conv2d(d_model, d_model * 2, 1),
                nn.GELU(),
                nn.Conv2d(d_model * 2, d_model, 1),
            )

    def forward(
        self, sem: torch.Tensor, depth_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sem: (B, D, H, W) semantic stream
            depth_feat: (B, D, H, W) depth-feature stream
        Returns:
            (refined_sem, refined_depth_feat) — same shapes
        """
        # Pre-norm
        sem_n = self.norm_sem(sem)
        depth_n = self.norm_depth(depth_feat)

        # Cross-chain modulation
        sem_input = sem_n + self.alpha * torch.sigmoid(self.cross_d2s(depth_n))
        depth_input = depth_n + self.beta * torch.sigmoid(self.cross_s2d(sem_n))

        # Conv processing + residual
        sem_out = sem + self.sem_conv(sem_input)
        depth_out = depth_feat + self.depth_conv(depth_input)

        return sem_out, depth_out


class CoupledMambaBlock(nn.Module):
    """Coupled dual-chain SSM for cross-modal feature fusion.

    Two separate VisionMamba2 chains (semantic and depth-feature) with
    learnable cross-chain state coupling. Each chain's input is augmented
    by a gated projection of the partner chain's features.

    Reference: Coupled Mamba (Li et al., NeurIPS 2024)
    """

    def __init__(
        self,
        d_model: int = 192,
        layer_type: str = "gated_delta_net",
        scan_mode: str = "bidirectional",
        coupling_strength: float = 0.1,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 256,
    ):
        super().__init__()
        self.d_model = d_model

        # Two independent VisionMamba2 streams
        mamba_kwargs = dict(
            d_model=d_model, scan_mode=scan_mode, layer_type=layer_type,
            d_state=d_state, d_conv=d_conv, expand=expand,
            headdim=headdim, chunk_size=chunk_size,
        )
        self.sem_mamba = VisionMamba2(**mamba_kwargs)
        self.depth_mamba = VisionMamba2(**mamba_kwargs)

        # Cross-chain coupling: gated projection from partner → self
        self.cross_depth_to_sem = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.cross_sem_to_depth = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # Learnable coupling strength (initialized small for stable training)
        self.alpha = nn.Parameter(torch.tensor(coupling_strength))
        self.beta = nn.Parameter(torch.tensor(coupling_strength))

        # Pre-norm
        self.norm_sem = nn.LayerNorm(d_model)
        self.norm_depth = nn.LayerNorm(d_model)

        # Post-SSM FFN
        self.ffn_sem = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ffn_depth = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self, sem: torch.Tensor, depth_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sem: (B, D, H, W) semantic stream
            depth_feat: (B, D, H, W) depth-feature stream
        Returns:
            (refined_sem, refined_depth_feat) — same shapes
        """
        B, D, H, W = sem.shape

        # Flatten to (B, L, D) for LayerNorm + cross-coupling
        sem_flat = sem.permute(0, 2, 3, 1).reshape(B, H * W, D)
        depth_flat = depth_feat.permute(0, 2, 3, 1).reshape(B, H * W, D)

        # Pre-norm
        sem_normed = self.norm_sem(sem_flat)
        depth_normed = self.norm_depth(depth_flat)

        # Cross-chain modulation
        cross_d2s = self.alpha * self.cross_depth_to_sem(depth_normed)
        cross_s2d = self.beta * self.cross_sem_to_depth(sem_normed)

        sem_input = sem_normed + cross_d2s
        depth_input = depth_normed + cross_s2d

        # Reshape to (B, D, H, W) for VisionMamba2
        sem_input = sem_input.reshape(B, H, W, D).permute(0, 3, 1, 2)
        depth_input = depth_input.reshape(B, H, W, D).permute(0, 3, 1, 2)

        # Clamp inputs to safe range for SSD backward (exp-cumsum overflows
        # with |x| > ~2.0 in the pure PyTorch reference implementation)
        sem_input = sem_input.clamp(-2.0, 2.0)
        depth_input = depth_input.clamp(-2.0, 2.0)

        # Independent SSM processing with coupled inputs
        sem_out = self.sem_mamba(sem_input)
        depth_out = self.depth_mamba(depth_input)

        # Residual + FFN
        sem_out_flat = sem_out.permute(0, 2, 3, 1).reshape(B, H * W, D)
        depth_out_flat = depth_out.permute(0, 2, 3, 1).reshape(B, H * W, D)

        sem_refined = sem_flat + sem_out_flat
        sem_refined = sem_refined + self.ffn_sem(sem_refined)

        depth_refined = depth_flat + depth_out_flat
        depth_refined = depth_refined + self.ffn_depth(depth_refined)

        # Reshape back to (B, D, H, W)
        sem_refined = sem_refined.reshape(B, H, W, D).permute(0, 3, 1, 2)
        depth_refined = depth_refined.reshape(B, H, W, D).permute(0, 3, 1, 2)

        return sem_refined, depth_refined


class WindowedAttentionBlock(nn.Module):
    """Windowed self-attention block (Swin-style) with cross-modal gating.

    Partitions feature maps into non-overlapping windows and applies
    multi-head self-attention within each window. Alternating blocks use
    shifted windows for cross-window information flow.

    Cross-modal gating follows the same alpha/beta sigmoid pattern as
    CoupledConvBlock for consistency.
    """

    def __init__(
        self,
        d_model: int = 192,
        window_size: int = 8,
        num_heads: int = 4,
        shift: bool = False,
        coupling_strength: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.shift = shift
        self.shift_size = window_size // 2 if shift else 0
        self.scale = self.head_dim ** -0.5

        # Pre-norm
        self.norm_sem = nn.GroupNorm(1, d_model)
        self.norm_depth = nn.GroupNorm(1, d_model)

        # Cross-chain coupling
        self.cross_d2s = nn.Conv2d(d_model, d_model, 1)
        self.cross_s2d = nn.Conv2d(d_model, d_model, 1)
        self.alpha = nn.Parameter(torch.tensor(coupling_strength))
        self.beta = nn.Parameter(torch.tensor(coupling_strength))

        # QKV projections (separate for each stream)
        self.qkv_sem = nn.Linear(d_model, d_model * 3)
        self.qkv_depth = nn.Linear(d_model, d_model * 3)
        self.out_sem = nn.Linear(d_model, d_model)
        self.out_depth = nn.Linear(d_model, d_model)

        # Relative position bias
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(num_heads, (2 * window_size - 1) * (2 * window_size - 1))
        )
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        # Precompute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, ws, ws)
        coords_flat = coords.reshape(2, -1)  # (2, ws*ws)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, ws*ws, ws*ws)
        rel_coords[0] += window_size - 1
        rel_coords[1] += window_size - 1
        rel_coords[0] *= 2 * window_size - 1
        rel_pos_index = rel_coords.sum(0)  # (ws*ws, ws*ws)
        self.register_buffer("rel_pos_index", rel_pos_index)

        # FFN
        self.ffn_sem = nn.Sequential(
            nn.GroupNorm(1, d_model),
            nn.Conv2d(d_model, d_model * 2, 1),
            nn.GELU(),
            nn.Conv2d(d_model * 2, d_model, 1),
        )
        self.ffn_depth = nn.Sequential(
            nn.GroupNorm(1, d_model),
            nn.Conv2d(d_model, d_model * 2, 1),
            nn.GELU(),
            nn.Conv2d(d_model * 2, d_model, 1),
        )

    def _window_partition(
        self, x: torch.Tensor, H: int, W: int
    ) -> Tuple[torch.Tensor, int, int]:
        """Partition (B, H*W, C) into windows of (B*nW, ws*ws, C). Pads if needed."""
        B, _, C = x.shape
        ws = self.window_size

        # Pad H, W to multiples of window_size
        Hp = math.ceil(H / ws) * ws
        Wp = math.ceil(W / ws) * ws
        x = x.reshape(B, H, W, C)
        if Hp != H or Wp != W:
            x = F.pad(x, (0, 0, 0, Wp - W, 0, Hp - H))
        # (B, Hp, Wp, C) -> (B, Hp//ws, ws, Wp//ws, ws, C) -> (B*nW, ws*ws, C)
        nH, nW = Hp // ws, Wp // ws
        x = x.reshape(B, nH, ws, nW, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * nH * nW, ws * ws, C)
        return x, Hp, Wp

    def _window_unpartition(
        self, x: torch.Tensor, B: int, Hp: int, Wp: int, H: int, W: int
    ) -> torch.Tensor:
        """Reverse window partition: (B*nW, ws*ws, C) -> (B, H*W, C)."""
        ws = self.window_size
        C = x.shape[-1]
        nH, nW = Hp // ws, Wp // ws
        x = x.reshape(B, nH, nW, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        # Crop padding
        if Hp != H or Wp != W:
            x = x[:, :H, :W, :].contiguous()
        return x.reshape(B, H * W, C)

    def _windowed_attention(
        self,
        x: torch.Tensor,
        qkv_proj: nn.Linear,
        out_proj: nn.Linear,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Apply windowed multi-head self-attention.

        Args:
            x: (B, H*W, C)
        Returns:
            (B, H*W, C)
        """
        B = x.shape[0]
        ws = self.window_size

        # Cyclic shift
        if self.shift_size > 0:
            x_2d = x.reshape(B, H, W, self.d_model)
            x_2d = torch.roll(x_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x = x_2d.reshape(B, H * W, self.d_model)

        # Partition into windows
        x_win, Hp, Wp = self._window_partition(x, H, W)  # (B*nW, ws*ws, C)
        nW_total = x_win.shape[0]

        # QKV
        qkv = qkv_proj(x_win).reshape(nW_total, ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*nW, heads, ws*ws, head_dim)
        q, k, v = qkv.unbind(0)

        # Attention with relative position bias
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*nW, heads, ws*ws, ws*ws)
        bias = self.rel_pos_bias[:, self.rel_pos_index.view(-1)].reshape(
            self.num_heads, ws * ws, ws * ws
        )
        attn = attn + bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(nW_total, ws * ws, self.d_model)
        out = out_proj(out)

        # Unpartition
        out = self._window_unpartition(out, B, Hp, Wp, H, W)  # (B, H*W, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            out_2d = out.reshape(B, H, W, self.d_model)
            out_2d = torch.roll(out_2d, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            out = out_2d.reshape(B, H * W, self.d_model)

        return out

    def forward(
        self, sem: torch.Tensor, depth_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sem: (B, D, H, W) semantic stream
            depth_feat: (B, D, H, W) depth-feature stream
        Returns:
            (refined_sem, refined_depth_feat) — same shapes
        """
        B, D, H, W = sem.shape

        # Pre-norm
        sem_n = self.norm_sem(sem)
        depth_n = self.norm_depth(depth_feat)

        # Cross-chain modulation (in spatial domain)
        sem_input = sem_n + self.alpha * torch.sigmoid(self.cross_d2s(depth_n))
        depth_input = depth_n + self.beta * torch.sigmoid(self.cross_s2d(sem_n))

        # Flatten to (B, H*W, D) for attention
        sem_flat = sem_input.permute(0, 2, 3, 1).reshape(B, H * W, D)
        depth_flat = depth_input.permute(0, 2, 3, 1).reshape(B, H * W, D)

        # Windowed self-attention
        sem_attn = self._windowed_attention(sem_flat, self.qkv_sem, self.out_sem, H, W)
        depth_attn = self._windowed_attention(depth_flat, self.qkv_depth, self.out_depth, H, W)

        # Reshape back to (B, D, H, W) + residual
        sem_attn = sem_attn.reshape(B, H, W, D).permute(0, 3, 1, 2)
        depth_attn = depth_attn.reshape(B, H, W, D).permute(0, 3, 1, 2)

        sem_out = sem + sem_attn
        depth_out = depth_feat + depth_attn

        # FFN + residual
        sem_out = sem_out + self.ffn_sem(sem_out)
        depth_out = depth_out + self.ffn_depth(depth_out)

        return sem_out, depth_out


class MambaOutBlock(nn.Module):
    """MambaOut block: gated CNN without SSM recurrence.

    Keeps Mamba's gated convolution structure (gate * value) but removes
    the state-space recurrence, using depthwise 7x7 conv instead.

    Reference: Yu et al., "MambaOut: Do We Really Need Mamba for Vision?", 2024

    Each stream:
        GroupNorm → cross-modal gate → gate_path(Conv1x1→DWConv7x7→GN→SiLU)
                                     × value_path(Conv1x1)
        → Conv1x1 → residual
    """

    def __init__(
        self,
        d_model: int = 192,
        expand: int = 2,
        coupling_strength: float = 0.1,
    ):
        super().__init__()
        inner_dim = d_model * expand

        # Pre-norm
        self.norm_sem = nn.GroupNorm(1, d_model)
        self.norm_depth = nn.GroupNorm(1, d_model)

        # Cross-chain coupling
        self.cross_d2s = nn.Conv2d(d_model, d_model, 1)
        self.cross_s2d = nn.Conv2d(d_model, d_model, 1)
        self.alpha = nn.Parameter(torch.tensor(coupling_strength))
        self.beta = nn.Parameter(torch.tensor(coupling_strength))

        # Semantic stream
        self.sem_gate = nn.Sequential(
            nn.Conv2d(d_model, inner_dim, 1, bias=False),
            nn.Conv2d(inner_dim, inner_dim, 7, padding=3, groups=inner_dim, bias=False),
            nn.GroupNorm(1, inner_dim),
            nn.SiLU(),
        )
        self.sem_value = nn.Conv2d(d_model, inner_dim, 1, bias=False)
        self.sem_out = nn.Conv2d(inner_dim, d_model, 1, bias=False)

        # Depth stream
        self.depth_gate = nn.Sequential(
            nn.Conv2d(d_model, inner_dim, 1, bias=False),
            nn.Conv2d(inner_dim, inner_dim, 7, padding=3, groups=inner_dim, bias=False),
            nn.GroupNorm(1, inner_dim),
            nn.SiLU(),
        )
        self.depth_value = nn.Conv2d(d_model, inner_dim, 1, bias=False)
        self.depth_out = nn.Conv2d(inner_dim, d_model, 1, bias=False)

    def forward(
        self, sem: torch.Tensor, depth_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sem: (B, D, H, W) semantic stream
            depth_feat: (B, D, H, W) depth-feature stream
        Returns:
            (refined_sem, refined_depth_feat) — same shapes
        """
        # Pre-norm
        sem_n = self.norm_sem(sem)
        depth_n = self.norm_depth(depth_feat)

        # Cross-chain modulation
        sem_input = sem_n + self.alpha * torch.sigmoid(self.cross_d2s(depth_n))
        depth_input = depth_n + self.beta * torch.sigmoid(self.cross_s2d(sem_n))

        # Gated conv: gate * value
        sem_out = sem + self.sem_out(self.sem_gate(sem_input) * self.sem_value(sem_input))
        depth_out = depth_feat + self.depth_out(
            self.depth_gate(depth_input) * self.depth_value(depth_input)
        )

        return sem_out, depth_out


class CSCMRefineNet(nn.Module):
    """Cross-Modal Refinement Network for semantic pseudo-labels.

    Takes pre-computed DINOv2 features + SPIdepth depth and produces
    refined 27-class semantic logits.

    Args:
        num_classes: number of semantic classes (27 for CAUSE-TR)
        feature_dim: DINOv2 feature dimension (768 for ViT-B/14)
        bridge_dim: internal bridge dimension
        num_blocks: number of coupled blocks
        block_type: "conv" or "mamba"
        layer_type: "mamba2" or "gated_delta_net" (only for block_type="mamba")
        scan_mode: "raster", "bidirectional", or "cross_scan" (only for block_type="mamba")
        coupling_strength: initial alpha/beta for cross-chain coupling
        d_state: SSM state dimension (only for block_type="mamba")
        chunk_size: SSD chunk size (only for block_type="mamba")
    """

    def __init__(
        self,
        num_classes: int = 27,
        feature_dim: int = 768,
        bridge_dim: int = 192,
        num_blocks: int = 4,
        block_type: str = "conv",
        layer_type: str = "gated_delta_net",
        scan_mode: str = "bidirectional",
        coupling_strength: float = 0.1,
        d_state: int = 64,
        chunk_size: int = 32,
        gradient_checkpointing: bool = True,
        use_aspp: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gradient_checkpointing = gradient_checkpointing

        self.sem_proj = SemanticProjection(feature_dim, bridge_dim)
        self.depth_feat_proj = DepthFeatureProjection(feature_dim, bridge_dim)

        if block_type == "conv":
            self.blocks = nn.ModuleList([
                CoupledConvBlock(
                    d_model=bridge_dim,
                    coupling_strength=coupling_strength,
                    use_aspp=use_aspp,
                )
                for _ in range(num_blocks)
            ])
        elif block_type == "mamba":
            self.blocks = nn.ModuleList([
                CoupledMambaBlock(
                    d_model=bridge_dim,
                    layer_type=layer_type,
                    scan_mode=scan_mode,
                    coupling_strength=coupling_strength,
                    d_state=d_state,
                    chunk_size=chunk_size,
                )
                for _ in range(num_blocks)
            ])
        else:
            raise ValueError(f"Unknown block_type: {block_type}")

        # Output head: bridge_dim → num_classes
        # Small random init so model starts slightly perturbed from CAUSE
        # (zero-init creates an inescapable local minimum at the CAUSE identity)
        self.head = nn.Sequential(
            nn.GroupNorm(1, bridge_dim),
            nn.Conv2d(bridge_dim, num_classes, 1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        dinov2_features: torch.Tensor,
        depth: torch.Tensor,
        depth_grads: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            dinov2_features: (B, 768, H, W) — DINOv2 patch features
            depth: (B, 1, H, W) — normalized depth [0, 1]
            depth_grads: (B, 2, H, W) — Sobel gradients of depth
        Returns:
            refined_logits: (B, 27, H, W)
        """
        sem = self.sem_proj(dinov2_features)
        depth_feat = self.depth_feat_proj(dinov2_features, depth, depth_grads)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                sem, depth_feat = checkpoint(
                    block, sem, depth_feat, use_reentrant=False,
                )
            else:
                sem, depth_feat = block(sem, depth_feat)

        refined_logits = self.head(sem)

        return refined_logits


class HiResRefineNet(nn.Module):
    """High-resolution refinement network with 4x upsampled processing.

    Projects DINOv2 features (32x64) to bridge_dim, upsamples to 128x256,
    then applies N coupled blocks at high resolution before the output head.

    Supports multiple block types and upsampling strategies for ablation.

    Args:
        num_classes: number of semantic classes (19 for Cityscapes eval)
        feature_dim: DINOv2 feature dimension (768 for ViT-B/14)
        bridge_dim: internal bridge dimension
        num_blocks: number of coupled blocks at high resolution
        block_type: "conv" | "attention" | "mamba_bidir" | "mamba_spatial" | "mambaout"
        upsample_strategy: "bilinear" | "transposed_conv" | "pixel_shuffle" | "none"
        coupling_strength: initial alpha/beta for cross-chain coupling
        gradient_checkpointing: use gradient checkpointing for blocks
        layer_type: SSM layer type for mamba blocks ("mamba2" or "gated_delta_net")
        d_state: SSM state dimension for mamba blocks
        chunk_size: SSD chunk size for mamba blocks
        window_size: window size for attention blocks
        num_heads: number of attention heads for attention blocks
    """

    def __init__(
        self,
        num_classes: int = 19,
        feature_dim: int = 768,
        bridge_dim: int = 192,
        num_blocks: int = 4,
        block_type: str = "conv",
        upsample_strategy: str = "transposed_conv",
        coupling_strength: float = 0.1,
        gradient_checkpointing: bool = True,
        layer_type: str = "gated_delta_net",
        d_state: int = 64,
        chunk_size: int = 32,
        window_size: int = 8,
        num_heads: int = 4,
        use_aspp: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gradient_checkpointing = gradient_checkpointing

        # Projection: 768 -> bridge_dim at 32x64
        self.sem_proj = SemanticProjection(feature_dim, bridge_dim)
        self.depth_feat_proj = DepthFeatureProjection(feature_dim, bridge_dim)

        # Upsampling: 32x64 -> 128x256 (separate for each stream)
        self.upsample_sem = _build_upsample(upsample_strategy, bridge_dim)
        self.upsample_depth = _build_upsample(upsample_strategy, bridge_dim)

        # Build blocks at high resolution
        blocks = []
        for i in range(num_blocks):
            if block_type == "conv":
                blocks.append(
                    CoupledConvBlock(
                        d_model=bridge_dim,
                        coupling_strength=coupling_strength,
                        use_aspp=use_aspp,
                    )
                )
            elif block_type == "attention":
                blocks.append(
                    WindowedAttentionBlock(
                        d_model=bridge_dim,
                        window_size=window_size,
                        num_heads=num_heads,
                        shift=(i % 2 == 1),
                        coupling_strength=coupling_strength,
                    )
                )
            elif block_type == "mamba_bidir":
                blocks.append(
                    CoupledMambaBlock(
                        d_model=bridge_dim,
                        layer_type=layer_type,
                        scan_mode="bidirectional",
                        coupling_strength=coupling_strength,
                        d_state=d_state,
                        chunk_size=chunk_size,
                    )
                )
            elif block_type == "mamba_spatial":
                blocks.append(
                    CoupledMambaBlock(
                        d_model=bridge_dim,
                        layer_type=layer_type,
                        scan_mode="cross_scan",
                        coupling_strength=coupling_strength,
                        d_state=d_state,
                        chunk_size=chunk_size,
                    )
                )
            elif block_type == "mambaout":
                blocks.append(
                    MambaOutBlock(
                        d_model=bridge_dim,
                        coupling_strength=coupling_strength,
                    )
                )
            else:
                raise ValueError(f"Unknown block_type: {block_type}")
        self.blocks = nn.ModuleList(blocks)

        # Output head: bridge_dim -> num_classes
        self.head = nn.Sequential(
            nn.GroupNorm(1, bridge_dim),
            nn.Conv2d(bridge_dim, num_classes, 1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        dinov2_features: torch.Tensor,
        depth: torch.Tensor,
        depth_grads: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            dinov2_features: (B, 768, H, W) — DINOv2 patch features (e.g. 32x64)
            depth: (B, 1, H, W) — normalized depth [0, 1]
            depth_grads: (B, 2, H, W) — Sobel gradients of depth
        Returns:
            logits: (B, num_classes, H_up, W_up) — at upsampled resolution
        """
        # Project to bridge_dim at input resolution
        sem = self.sem_proj(dinov2_features)
        depth_feat = self.depth_feat_proj(dinov2_features, depth, depth_grads)

        # Upsample both streams
        sem = self.upsample_sem(sem)
        depth_feat = self.upsample_depth(depth_feat)

        # Apply blocks at high resolution
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                sem, depth_feat = checkpoint(
                    block, sem, depth_feat, use_reentrant=False,
                )
            else:
                sem, depth_feat = block(sem, depth_feat)

        # Output head
        logits = self.head(sem)

        return logits


class DepthSkipBlock(nn.Module):
    """Extract boundary features from depth at a target spatial scale.

    Downsamples full-resolution depth to the target scale, computes Sobel
    gradients, and projects [depth, grad_x, grad_y] to skip_dim channels.
    """

    def __init__(self, skip_dim: int = 32, rich: bool = False):
        super().__init__()
        self.rich = rich
        # Input channels: depth + grad_x + grad_y (+ laplacian if rich)
        in_ch = 4 if rich else 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, skip_dim, 3, padding=1, bias=False),
            nn.GroupNorm(1, skip_dim),
            nn.GELU(),
        )
        if rich:
            # Second conv for richer skip features
            self.conv2 = nn.Sequential(
                nn.Conv2d(skip_dim, skip_dim, 3, padding=1, bias=False),
                nn.GroupNorm(1, skip_dim),
                nn.GELU(),
            )
        # Fixed Sobel kernels (not learned)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).reshape(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        if rich:
            # Laplacian kernel for 2nd-order edge detection
            laplacian = torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
            ).reshape(1, 1, 3, 3)
            self.register_buffer("laplacian", laplacian)

    def forward(self, depth_full: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Args:
            depth_full: (B, 1, H_d, W_d) full-resolution depth (e.g. 512x1024)
            target_h, target_w: target spatial dimensions for this decoder stage
        Returns:
            (B, skip_dim, target_h, target_w) boundary features
        """
        # Downsample depth to target scale
        depth = F.interpolate(
            depth_full, size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        )
        # Compute Sobel gradients
        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)
        if self.rich:
            lap = F.conv2d(depth, self.laplacian, padding=1)
            skip_input = torch.cat([depth, grad_x, grad_y, lap], dim=1)  # (B, 4, H, W)
        else:
            skip_input = torch.cat([depth, grad_x, grad_y], dim=1)  # (B, 3, H, W)
        out = self.conv1(skip_input)
        if self.rich:
            out = self.conv2(out)
        return out


class InstanceSkipBlock(nn.Module):
    """Extract instance boundary features at a target spatial scale.

    Downsamples full-resolution instance masks to the target scale, computes
    boundary maps and approximate distance transforms, and projects to
    inst_skip_dim channels.
    """

    def __init__(self, inst_skip_dim: int = 16):
        super().__init__()
        # Input: [boundary_map, distance_transform] = 2 channels
        self.conv = nn.Sequential(
            nn.Conv2d(2, inst_skip_dim, 3, padding=1, bias=False),
            nn.GroupNorm(1, inst_skip_dim),
            nn.GELU(),
        )

    def forward(
        self, instance_full: torch.Tensor, target_h: int, target_w: int
    ) -> torch.Tensor:
        """
        Args:
            instance_full: (B, 1, H, W) int/float instance IDs (0=stuff/bg)
            target_h, target_w: target spatial dimensions for this decoder stage
        Returns:
            (B, inst_skip_dim, target_h, target_w) instance boundary features
        """
        # Downsample instance map via nearest-neighbor (preserves IDs)
        inst = F.interpolate(
            instance_full.float(), size=(target_h, target_w), mode="nearest"
        )

        # Compute boundary: where adjacent pixels have different instance IDs
        diff_h = (inst[:, :, 1:, :] != inst[:, :, :-1, :]).float()
        diff_w = (inst[:, :, :, 1:] != inst[:, :, :, :-1]).float()
        boundary = torch.zeros_like(inst)
        boundary[:, :, 1:, :] += diff_h
        boundary[:, :, :-1, :] += diff_h
        boundary[:, :, :, 1:] += diff_w
        boundary[:, :, :, :-1] += diff_w
        boundary = (boundary > 0).float()  # (B, 1, H, W)

        # Approximate distance transform via iterative dilation
        dist = boundary.clone()
        for _ in range(5):  # 5 iterations ≈ 5-pixel radius
            dist = F.max_pool2d(dist, 3, stride=1, padding=1)
        dist = 1.0 - dist  # Invert: 0 at boundary, 1 far away

        skip_input = torch.cat([boundary, dist], dim=1)  # (B, 2, H, W)
        return self.conv(skip_input)


class CenterOffsetHead(nn.Module):
    """Center heatmap (1-ch) + offset (2-ch) head for instance segmentation.

    Uses separate multi-layer branches for center and offset prediction,
    following Panoptic-DeepLab design. Each branch has 3 conv layers with
    dilated convolutions for larger receptive field (~15px at 128x256).

    GroupNorm (not BatchNorm) for consistency with DepthGuidedUNet.
    ~0.15M params at in_dim=192, hidden_dim=64.
    """

    def __init__(self, in_dim: int = 192, hidden_dim: int = 64):
        super().__init__()
        # Center branch: 3 conv layers with increasing dilation (RF ≈ 15px)
        self.center_branch = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
        )
        self.center_conv = nn.Conv2d(hidden_dim, 1, 1)

        # Offset branch: 3 conv layers with increasing dilation
        self.offset_branch = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
        )
        self.offset_conv = nn.Conv2d(hidden_dim, 2, 1)

        # Init center bias to -4.0 (sigmoid(-4)=0.018) for strong suppression
        # of initial center predictions to avoid false positive explosion
        nn.init.zeros_(self.center_conv.weight)
        nn.init.constant_(self.center_conv.bias, -4.0)
        nn.init.normal_(self.offset_conv.weight, std=0.01)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, in_dim, H, W) shared decoder features
        Returns:
            center: (B, 1, H, W) center heatmap (sigmoid applied)
            offset: (B, 2, H, W) offset predictions (dy, dx)
        """
        center = torch.sigmoid(self.center_conv(self.center_branch(x)))
        offset = self.offset_conv(self.offset_branch(x))
        return center, offset


class DecoderStage(nn.Module):
    """Single UNet decoder stage: upsample 2x + depth skip + fuse + refine.

    Upsamples both streams via transposed convolution, concatenates depth
    skip features with the semantic stream, fuses back to bridge_dim, and
    applies one CoupledConvBlock or WindowedAttentionBlock for refinement.
    """

    def __init__(self, bridge_dim: int = 192, skip_dim: int = 32,
                 coupling_strength: float = 0.1, rich_skip: bool = False,
                 block_type: str = "conv", window_size: int = 8,
                 num_heads: int = 4, shift: bool = False,
                 use_instance: bool = False, inst_skip_dim: int = 16):
        super().__init__()
        self.use_instance = use_instance
        # 2x upsampling for both streams
        self.up_sem = nn.Sequential(
            nn.ConvTranspose2d(bridge_dim, bridge_dim, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, bridge_dim),
            nn.GELU(),
        )
        self.up_depth = nn.Sequential(
            nn.ConvTranspose2d(bridge_dim, bridge_dim, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, bridge_dim),
            nn.GELU(),
        )
        # Depth skip connection
        self.depth_skip = DepthSkipBlock(skip_dim, rich=rich_skip)
        # Instance skip connection (optional)
        if use_instance:
            self.instance_skip = InstanceSkipBlock(inst_skip_dim)
        # Fuse: (bridge_dim + skip_dim [+ inst_skip_dim]) -> bridge_dim
        fuse_in = bridge_dim + skip_dim + (inst_skip_dim if use_instance else 0)
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, bridge_dim, 1, bias=False),
            nn.GroupNorm(1, bridge_dim),
            nn.GELU(),
        )
        # Refinement block at this scale
        if block_type == "attention":
            self.block = WindowedAttentionBlock(
                d_model=bridge_dim,
                window_size=window_size,
                num_heads=num_heads,
                shift=shift,
                coupling_strength=coupling_strength,
            )
        else:
            self.block = CoupledConvBlock(
                d_model=bridge_dim,
                coupling_strength=coupling_strength,
            )

    def forward(
        self,
        sem: torch.Tensor,
        depth_feat: torch.Tensor,
        depth_full: torch.Tensor,
        instance_full: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            sem: (B, bridge_dim, H, W) semantic stream
            depth_feat: (B, bridge_dim, H, W) depth-feature stream
            depth_full: (B, 1, H_d, W_d) full-resolution depth
            instance_full: (B, 1, H_i, W_i) full-resolution instance IDs (optional)
        Returns:
            (sem, depth_feat) at 2x spatial resolution
        """
        # Upsample both streams
        sem = self.up_sem(sem)
        depth_feat = self.up_depth(depth_feat)
        _, _, H, W = sem.shape
        # Depth skip at this scale
        skip = self.depth_skip(depth_full, H, W)
        # Instance skip at this scale (optional)
        if self.use_instance and instance_full is not None:
            inst_skip = self.instance_skip(instance_full, H, W)
            sem = self.fuse(torch.cat([sem, skip, inst_skip], dim=1))
        else:
            sem = self.fuse(torch.cat([sem, skip], dim=1))
        # Refine at this scale
        sem, depth_feat = self.block(sem, depth_feat)
        return sem, depth_feat


class DepthGuidedUNet(nn.Module):
    """Option A: Depth-Guided UNet Decoder for high-resolution refinement.

    Replaces the direct 4x upsampling of HiResRefineNet with a progressive
    2-stage decoder (32x64 -> 64x128 -> 128x256). At each stage, depth
    skip connections inject geometric boundary cues via Sobel gradients.

    Architecture:
        DINOv2 (768) -> projections (192, 32x64)
        -> 2 bottleneck blocks at 32x64
        -> DecoderStage1: 32x64 -> 64x128 + depth skip
        -> DecoderStage2: 64x128 -> 128x256 + depth skip
        -> classification head (192 -> num_classes) at 128x256

    Args:
        num_classes: number of output semantic classes
        feature_dim: DINOv2 feature dimension (768)
        bridge_dim: internal channel dimension (192)
        num_bottleneck_blocks: number of CoupledConvBlocks at 32x64
        skip_dim: channel dimension for depth skip features
        coupling_strength: initial cross-modal coupling strength
        gradient_checkpointing: use gradient checkpointing for memory
        rich_skip: use richer depth skip (2nd conv + Laplacian)
        num_final_blocks: extra CoupledConvBlocks after decoder
        num_decoder_stages: number of 2x upsample stages (2=128x256, 3=256x512)
        block_type: "conv" or "attention" for decoder/bottleneck blocks
        window_size: window size for attention blocks
        num_heads: number of attention heads
    """

    def __init__(
        self,
        num_classes: int = 19,
        feature_dim: int = 768,
        bridge_dim: int = 192,
        num_bottleneck_blocks: int = 2,
        skip_dim: int = 32,
        coupling_strength: float = 0.1,
        gradient_checkpointing: bool = True,
        rich_skip: bool = False,
        num_final_blocks: int = 0,
        num_decoder_stages: int = 2,
        block_type: str = "conv",
        window_size: int = 8,
        num_heads: int = 4,
        use_instance: bool = False,
        inst_skip_dim: int = 16,
        use_instance_heads: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gradient_checkpointing = gradient_checkpointing
        self.num_decoder_stages = num_decoder_stages
        self.use_instance = use_instance
        self.use_instance_heads = use_instance_heads

        # Projection: 768 -> bridge_dim at 32x64
        self.sem_proj = SemanticProjection(feature_dim, bridge_dim)
        self.depth_feat_proj = DepthFeatureProjection(feature_dim, bridge_dim)

        # Bottleneck blocks at 32x64
        bottleneck_blocks = []
        for i in range(num_bottleneck_blocks):
            if block_type == "attention":
                bottleneck_blocks.append(WindowedAttentionBlock(
                    d_model=bridge_dim, window_size=window_size,
                    num_heads=num_heads, shift=(i % 2 == 1),
                    coupling_strength=coupling_strength,
                ))
            else:
                bottleneck_blocks.append(CoupledConvBlock(
                    d_model=bridge_dim, coupling_strength=coupling_strength,
                ))
        self.bottleneck = nn.ModuleList(bottleneck_blocks)

        # Progressive decoder stages (each 2x upsample)
        self.decoder_stages = nn.ModuleList()
        for s in range(num_decoder_stages):
            self.decoder_stages.append(DecoderStage(
                bridge_dim, skip_dim, coupling_strength,
                rich_skip=rich_skip, block_type=block_type,
                window_size=window_size, num_heads=num_heads,
                shift=(s % 2 == 1),
                use_instance=use_instance, inst_skip_dim=inst_skip_dim,
            ))

        # Optional extra blocks at output resolution
        final = []
        for i in range(num_final_blocks):
            if block_type == "attention":
                final.append(WindowedAttentionBlock(
                    d_model=bridge_dim, window_size=window_size,
                    num_heads=num_heads, shift=(i % 2 == 1),
                    coupling_strength=coupling_strength,
                ))
            else:
                final.append(CoupledConvBlock(
                    d_model=bridge_dim, coupling_strength=coupling_strength,
                ))
        self.final_blocks = nn.ModuleList(final)

        # Output head
        self.head = nn.Sequential(
            nn.GroupNorm(1, bridge_dim),
            nn.Conv2d(bridge_dim, num_classes, 1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        # Instance heads (optional center heatmap + offset)
        if use_instance_heads:
            self.instance_head = CenterOffsetHead(bridge_dim)

    def forward(
        self,
        dinov2_features: torch.Tensor,
        depth: torch.Tensor,
        depth_grads: torch.Tensor,
        depth_full: torch.Tensor = None,
        instance_full: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            dinov2_features: (B, 768, H, W) DINOv2 patch features (32x64)
            depth: (B, 1, H, W) depth at patch resolution (32x64) for FiLM
            depth_grads: (B, 2, H, W) Sobel gradients at patch resolution
            depth_full: (B, 1, H_d, W_d) full-resolution depth for skip connections
            instance_full: (B, 1, H_i, W_i) full-resolution instance IDs (optional)
        Returns:
            logits: (B, num_classes, H_out, W_out) where H_out=32*2^stages, W_out=64*2^stages
        """
        # Project to bridge_dim at 32x64
        sem = self.sem_proj(dinov2_features)
        depth_feat = self.depth_feat_proj(dinov2_features, depth, depth_grads)

        # Bottleneck at 32x64
        for block in self.bottleneck:
            if self.gradient_checkpointing and self.training:
                sem, depth_feat = checkpoint(
                    block, sem, depth_feat, use_reentrant=False,
                )
            else:
                sem, depth_feat = block(sem, depth_feat)

        # Progressive decoding with depth skip connections (+ optional instance skip)
        for stage in self.decoder_stages:
            if self.gradient_checkpointing and self.training:
                sem, depth_feat = checkpoint(
                    stage, sem, depth_feat, depth_full, instance_full,
                    use_reentrant=False,
                )
            else:
                sem, depth_feat = stage(sem, depth_feat, depth_full, instance_full)

        # Extra blocks at 128x256
        for block in self.final_blocks:
            if self.gradient_checkpointing and self.training:
                sem, depth_feat = checkpoint(
                    block, sem, depth_feat, use_reentrant=False,
                )
            else:
                sem, depth_feat = block(sem, depth_feat)

        # Classification at 128x256
        logits = self.head(sem)

        if self.use_instance_heads:
            center, offset = self.instance_head(sem)
            return {"semantic": logits, "center": center, "offset": offset}

        return logits
