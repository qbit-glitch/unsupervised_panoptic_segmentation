"""Image-compatible Mamba2 modules with configurable scan strategies.

Provides two main modules:
    - VisionMamba2: single-stream image processing
    - CrossModalMamba2: two-stream cross-modal fusion (BiCMS-style)

Both support 3 scan modes for ablation:
    - "raster": simple row-major flatten
    - "bidirectional": forward + reverse scan with learned gating
    - "cross_scan": VMamba-style 4-way directional scan with learned merge

And 2 layer types:
    - "mamba2": Mamba2 SSD (simple accumulation)
    - "gated_delta_net": GatedDeltaNet (error-correcting delta rule)

Works on CPU, MPS (Apple Silicon), and CUDA.
"""

from typing import Tuple

import torch
import torch.nn as nn

from .mamba2 import Mamba2
from .gated_delta_net import GatedDeltaNet
from .scan import (
    raster_scan, raster_unscan,
    cross_scan, cross_unscan,
    interleave_tokens, deinterleave_tokens,
)


SCAN_MODES = ("raster", "bidirectional", "cross_scan")
LAYER_TYPES = ("mamba2", "gated_delta_net")


def _build_layer(layer_type: str, d_model: int, **kwargs) -> nn.Module:
    """Factory: build a sequence layer by type.

    Both Mamba2 and GatedDeltaNet share the same interface:
        forward(u: (B, L, D)) -> (B, L, D)

    Args:
        layer_type: "mamba2" or "gated_delta_net"
        d_model: input/output dimension
        **kwargs: passed through to the layer constructor.
            Mamba2 uses: d_state, d_conv, expand, headdim, ngroups, chunk_size
            GatedDeltaNet uses: d_conv, chunk_size (ignores Mamba2-specific kwargs)
    """
    if layer_type == "mamba2":
        return Mamba2(d_model=d_model, **kwargs)
    elif layer_type == "gated_delta_net":
        return GatedDeltaNet(d_model=d_model, **kwargs)
    else:
        raise ValueError(f"layer_type must be one of {LAYER_TYPES}, got '{layer_type}'")


class VisionMamba2(nn.Module):
    """Vision-compatible sequence model with configurable 2D scan strategy.

    Accepts either (B, C, H, W) image features or (B, L, C) token sequences.
    When given 4D input, automatically flattens → processes → unflattens.

    Args:
        d_model: feature dimension (C channel dim)
        scan_mode: "raster" | "bidirectional" | "cross_scan"
        layer_type: "mamba2" | "gated_delta_net"
        d_state: SSM state dimension. Default 128.
        d_conv: conv1d kernel width. Default 4.
        expand: expansion factor. Default 2.
        headdim: head dimension. Default 64.
        ngroups: number of head groups. Default 1.
        chunk_size: SSD chunk size. Default 256.
        **layer_kwargs: additional kwargs passed to the layer constructor
    """

    def __init__(
        self,
        d_model: int,
        scan_mode: str = "bidirectional",
        layer_type: str = "mamba2",
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        **layer_kwargs,
    ):
        super().__init__()
        assert scan_mode in SCAN_MODES, f"scan_mode must be one of {SCAN_MODES}, got '{scan_mode}'"
        assert layer_type in LAYER_TYPES, f"layer_type must be one of {LAYER_TYPES}, got '{layer_type}'"
        self.scan_mode = scan_mode
        self.layer_type = layer_type
        self.d_model = d_model

        layer_args = dict(
            d_state=d_state, d_conv=d_conv, expand=expand,
            headdim=headdim, ngroups=ngroups, chunk_size=chunk_size, **layer_kwargs,
        )

        if scan_mode == "raster":
            self.mamba = _build_layer(layer_type, d_model, **layer_args)

        elif scan_mode == "bidirectional":
            self.mamba_fwd = _build_layer(layer_type, d_model, **layer_args)
            self.mamba_bwd = _build_layer(layer_type, d_model, **layer_args)
            self.merge_gate = nn.Linear(2 * d_model, d_model)

        elif scan_mode == "cross_scan":
            # Single layer shared across all 4 directions (batched in B dim)
            self.mamba = _build_layer(layer_type, d_model, **layer_args)
            self.merge_proj = nn.Linear(4 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, H, W) image features or (B, L, C) token sequence
        Returns:
            Same shape as input
        """
        is_image = x.dim() == 4
        if is_image:
            B, C, H, W = x.shape
            tokens = raster_scan(x)  # (B, H*W, C)
        else:
            tokens = x
            B = x.shape[0]

        if self.scan_mode == "raster":
            out = self._forward_raster(tokens)
        elif self.scan_mode == "bidirectional":
            out = self._forward_bidirectional(tokens)
        elif self.scan_mode == "cross_scan":
            out = self._forward_cross_scan(tokens, B, H if is_image else None, W if is_image else None)

        if is_image:
            out = raster_unscan(out, H, W)
        return out

    def _forward_raster(self, tokens: torch.Tensor) -> torch.Tensor:
        """Simple forward scan on raster-ordered tokens."""
        return self.mamba(tokens)

    def _forward_bidirectional(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward + reverse scan with learned gate merge."""
        fwd_out = self.mamba_fwd(tokens)
        bwd_out = self.mamba_bwd(torch.flip(tokens, [1]))
        bwd_out = torch.flip(bwd_out, [1])

        gate = torch.sigmoid(self.merge_gate(
            torch.cat([fwd_out, bwd_out], dim=-1)
        ))
        return gate * fwd_out + (1.0 - gate) * bwd_out

    def _forward_cross_scan(self, tokens: torch.Tensor, B: int,
                            H: int = None, W: int = None) -> torch.Tensor:
        """VMamba-style 4-way cross-scan with learned merge.

        For 4D image input: uses spatial cross_scan (row/col × fwd/bwd).
        For 3D sequence input: uses 2 directions (fwd + bwd) × 2 (repeat for 4-way batch).
        """
        if H is not None and W is not None:
            # Image mode: reconstruct (B, C, H, W) for proper spatial cross-scan
            x_4d = raster_unscan(tokens, H, W)
            scanned = cross_scan(x_4d)  # (4*B, H*W, C)
        else:
            # Sequence mode fallback: fwd, bwd, fwd-shifted, bwd-shifted
            fwd = tokens
            bwd = torch.flip(tokens, [1])
            scanned = torch.cat([fwd, bwd, fwd, bwd], dim=0)  # (4*B, L, C)

        # Process all 4 directions in a single Mamba2 call
        scanned_out = self.mamba(scanned)  # (4*B, L, C)

        if H is not None and W is not None:
            # Unscan: align all directions back to raster spatial order
            dirs = cross_unscan(scanned_out, B, H, W)  # (B, 4, H*W, C)
        else:
            L, C = tokens.shape[1], tokens.shape[2]
            d0, d1, d2, d3 = scanned_out.chunk(4, dim=0)
            d1 = torch.flip(d1, [1])
            d3 = torch.flip(d3, [1])
            dirs = torch.stack([d0, d1, d2, d3], dim=1)  # (B, 4, L, C)

        # Merge 4 directions via learned projection
        # (B, 4, L, C) → (B, L, 4*C) → (B, L, C)
        B_out, _, L_out, C_out = dirs.shape
        dirs_concat = dirs.permute(0, 2, 1, 3).reshape(B_out, L_out, 4 * C_out)
        return self.merge_proj(dirs_concat)


class CrossModalMamba2(nn.Module):
    """Cross-modal fusion with token interleaving (BiCMS-style).

    Takes two feature streams (e.g. semantic + instance), interleaves their
    tokens, processes with a sequence model, then de-interleaves back.

    Accepts either (B, C, H, W) image features or (B, N, D) token sequences.

    Args:
        d_model: feature dimension
        scan_mode: "raster" | "bidirectional" | "cross_scan"
        layer_type: "mamba2" | "gated_delta_net"
        d_state: SSM state dimension. Default 128.
        d_conv: conv1d kernel width. Default 4.
        expand: expansion factor. Default 2.
        headdim: head dimension. Default 64.
        ngroups: number of head groups. Default 1.
        chunk_size: SSD chunk size. Default 256.
        **layer_kwargs: additional kwargs passed to the layer constructor
    """

    def __init__(
        self,
        d_model: int,
        scan_mode: str = "bidirectional",
        layer_type: str = "mamba2",
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        **layer_kwargs,
    ):
        super().__init__()
        assert scan_mode in SCAN_MODES, f"scan_mode must be one of {SCAN_MODES}, got '{scan_mode}'"
        assert layer_type in LAYER_TYPES, f"layer_type must be one of {LAYER_TYPES}, got '{layer_type}'"
        self.scan_mode = scan_mode
        self.layer_type = layer_type
        self.d_model = d_model

        # VisionMamba2 handles the scan strategy; it processes the interleaved sequence
        self.vision_mamba = VisionMamba2(
            d_model=d_model, scan_mode=scan_mode, layer_type=layer_type,
            d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim,
            ngroups=ngroups, chunk_size=chunk_size, **layer_kwargs,
        )

    def forward(
        self,
        semantic: torch.Tensor,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-modal Mamba2 fusion.

        Args:
            semantic: (B, C, H, W) or (B, N, D) — first stream
            features: (B, C, H, W) or (B, N, D) — second stream
        Returns:
            (fused_semantic, fused_features) — same shapes as inputs
        """
        is_image = semantic.dim() == 4
        if is_image:
            B, C, H, W = semantic.shape
            sem_tokens = raster_scan(semantic)   # (B, H*W, C)
            feat_tokens = raster_scan(features)  # (B, H*W, C)
        else:
            sem_tokens = semantic
            feat_tokens = features

        # Interleave: [s1, f1, s2, f2, ...]
        interleaved = interleave_tokens(sem_tokens, feat_tokens)  # (B, 2N, D)

        # Process with scan strategy (VisionMamba2 handles 3D input as sequence)
        fused = self.vision_mamba(interleaved)  # (B, 2N, D)

        # De-interleave
        fused_sem, fused_feat = deinterleave_tokens(fused)  # (B, N, D) each

        if is_image:
            fused_sem = raster_unscan(fused_sem, H, W)
            fused_feat = raster_unscan(fused_feat, H, W)

        return fused_sem, fused_feat
