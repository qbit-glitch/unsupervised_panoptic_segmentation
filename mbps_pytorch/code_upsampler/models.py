"""Small 90D code-space upsamplers for DCFA/CAUSE features.

The modules operate on continuous 90D semantic codes, not labels. They are
designed to be trained with frozen DCFA targets and evaluated with fixed K=80
clustering downstream.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        groups = 8 if out_ch % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvGNAct(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class GuidanceEncoder(nn.Module):
    """Tiny RGB/depth encoder used by both upsampler variants."""

    def __init__(self, in_ch: int = 4, hidden_ch: int = 32, num_blocks: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [ConvGNAct(in_ch, hidden_ch)]
        layers.extend(ResidualBlock(hidden_ch) for _ in range(num_blocks))
        self.net = nn.Sequential(*layers)

    def forward(self, guidance: torch.Tensor) -> torch.Tensor:
        return self.net(guidance)


def coord_feature_channels(num_freqs: int) -> int:
    if num_freqs <= 0:
        return 2
    return 2 + 4 * num_freqs


def make_coord_features(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    num_freqs: int = 4,
) -> torch.Tensor:
    """Return normalized xy + Fourier features at the requested image grid."""
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    feats = [xx, yy]
    for i in range(num_freqs):
        freq = float(2**i) * math.pi
        feats.extend(
            [
                torch.sin(freq * xx),
                torch.cos(freq * xx),
                torch.sin(freq * yy),
                torch.cos(freq * yy),
            ]
        )
    coord = torch.stack(feats, dim=0).unsqueeze(0)
    return coord.expand(batch_size, -1, -1, -1)


def append_coord_features(guidance: torch.Tensor, num_freqs: int) -> torch.Tensor:
    coords = make_coord_features(
        guidance.shape[0],
        guidance.shape[-2],
        guidance.shape[-1],
        guidance.device,
        guidance.dtype,
        num_freqs=num_freqs,
    )
    return torch.cat([guidance, coords], dim=1)


class ResidualCodeUpsampler(nn.Module):
    """Bilinear 90D upsample plus a learned RGB/depth-guided residual."""

    def __init__(
        self,
        code_dim: int = 90,
        guidance_ch: int = 4,
        hidden_ch: int = 64,
        guidance_hidden: int = 32,
        num_blocks: int = 3,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.residual_scale = residual_scale
        self.guidance = GuidanceEncoder(guidance_ch, guidance_hidden)
        layers: list[nn.Module] = [ConvGNAct(code_dim + guidance_hidden, hidden_ch)]
        layers.extend(ResidualBlock(hidden_ch) for _ in range(num_blocks))
        layers.append(nn.Conv2d(hidden_ch, code_dim, 3, padding=1))
        self.refine = nn.Sequential(*layers)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(
        self,
        low_code: torch.Tensor,
        guidance: torch.Tensor,
        output_size: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if output_size is None:
            output_size = guidance.shape[-2:]
        base = F.interpolate(low_code, size=output_size, mode="bilinear", align_corners=False)
        guide = self.guidance(guidance)
        residual = self.refine(torch.cat([base, guide], dim=1))
        return base + self.residual_scale * residual


class DynamicKernelCodeUpsampler(nn.Module):
    """AnyUp-style local reassembly over bilinear-upsampled 90D codes.

    The guidance encoder predicts one local kernel per high-resolution pixel.
    Those weights reassemble nearby 90D code vectors with a shared kernel across
    channels, followed by a small residual refinement.
    """

    def __init__(
        self,
        code_dim: int = 90,
        guidance_ch: int = 4,
        guidance_hidden: int = 32,
        hidden_ch: int = 64,
        kernel_size: int = 3,
        num_blocks: int = 2,
        residual_scale: float = 0.15,
        use_coords: bool = False,
        coord_freqs: int = 4,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.code_dim = code_dim
        self.kernel_size = kernel_size
        self.kernel_elems = kernel_size * kernel_size
        self.residual_scale = residual_scale
        self.use_coords = use_coords
        self.coord_freqs = coord_freqs
        guide_in = guidance_ch + (coord_feature_channels(coord_freqs) if use_coords else 0)
        self.guidance = GuidanceEncoder(guide_in, guidance_hidden)
        self.kernel_head = nn.Conv2d(guidance_hidden, self.kernel_elems, 3, padding=1)
        nn.init.zeros_(self.kernel_head.weight)
        nn.init.zeros_(self.kernel_head.bias)
        center = self.kernel_elems // 2
        with torch.no_grad():
            self.kernel_head.bias[center] = 4.0

        layers: list[nn.Module] = [ConvGNAct(code_dim + guidance_hidden, hidden_ch)]
        layers.extend(ResidualBlock(hidden_ch) for _ in range(num_blocks))
        layers.append(nn.Conv2d(hidden_ch, code_dim, 3, padding=1))
        self.refine = nn.Sequential(*layers)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(
        self,
        low_code: torch.Tensor,
        guidance: torch.Tensor,
        output_size: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if output_size is None:
            output_size = guidance.shape[-2:]
        base = F.interpolate(low_code, size=output_size, mode="bilinear", align_corners=False)
        guide_in = append_coord_features(guidance, self.coord_freqs) if self.use_coords else guidance
        guide = self.guidance(guide_in)
        weights = F.softmax(self.kernel_head(guide), dim=1)

        bsz, code_dim, height, width = base.shape
        patches = F.unfold(
            base,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        patches = patches.view(bsz, code_dim, self.kernel_elems, height, width)
        reassembled = (patches * weights.unsqueeze(1)).sum(dim=2)
        residual = self.refine(torch.cat([reassembled, guide], dim=1))
        return reassembled + self.residual_scale * residual


class AttentiveCodeUpsampler(nn.Module):
    """JAFAR-style local cross-attention over 90D code neighborhoods.

    High-resolution RGB/depth/coordinate queries attend to local windows of
    semantic 90D keys/values. SFT modulation lets the guidance stream sharpen or
    damp code dimensions before the residual refinement head.
    """

    def __init__(
        self,
        code_dim: int = 90,
        guidance_ch: int = 4,
        guidance_hidden: int = 48,
        hidden_ch: int = 64,
        attn_dim: int = 64,
        window_size: int = 5,
        num_blocks: int = 2,
        residual_scale: float = 0.15,
        use_coords: bool = True,
        coord_freqs: int = 4,
    ) -> None:
        super().__init__()
        if window_size % 2 != 1:
            raise ValueError("window_size must be odd")
        self.code_dim = code_dim
        self.window_size = window_size
        self.window_elems = window_size * window_size
        self.residual_scale = residual_scale
        self.use_coords = use_coords
        self.coord_freqs = coord_freqs

        guide_in = guidance_ch + (coord_feature_channels(coord_freqs) if use_coords else 0)
        self.guidance = GuidanceEncoder(guide_in, guidance_hidden)
        self.q_proj = nn.Conv2d(guidance_hidden, attn_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(code_dim, attn_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(code_dim, code_dim, 1, bias=False)
        self.sft = nn.Conv2d(guidance_hidden, 2 * code_dim, 3, padding=1)
        nn.init.zeros_(self.sft.weight)
        nn.init.zeros_(self.sft.bias)

        layers: list[nn.Module] = [ConvGNAct(code_dim + guidance_hidden, hidden_ch)]
        layers.extend(ResidualBlock(hidden_ch) for _ in range(num_blocks))
        layers.append(nn.Conv2d(hidden_ch, code_dim, 3, padding=1))
        self.refine = nn.Sequential(*layers)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(
        self,
        low_code: torch.Tensor,
        guidance: torch.Tensor,
        output_size: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if output_size is None:
            output_size = guidance.shape[-2:]
        base = F.interpolate(low_code, size=output_size, mode="bilinear", align_corners=False)
        guide_in = append_coord_features(guidance, self.coord_freqs) if self.use_coords else guidance
        guide = self.guidance(guide_in)

        query = self.q_proj(guide)
        key_map = self.k_proj(base)
        value_map = self.v_proj(base)
        bsz, attn_dim, height, width = query.shape

        key_patches = F.unfold(
            key_map,
            kernel_size=self.window_size,
            padding=self.window_size // 2,
        ).view(bsz, attn_dim, self.window_elems, height, width)
        value_patches = F.unfold(
            value_map,
            kernel_size=self.window_size,
            padding=self.window_size // 2,
        ).view(bsz, self.code_dim, self.window_elems, height, width)

        logits = (query.unsqueeze(2) * key_patches).sum(dim=1) / math.sqrt(attn_dim)
        weights = F.softmax(logits, dim=1)
        attended = (value_patches * weights.unsqueeze(1)).sum(dim=2)

        gamma, beta = self.sft(guide).chunk(2, dim=1)
        attended = attended * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
        residual = self.refine(torch.cat([attended, guide], dim=1))
        return attended + self.residual_scale * residual
