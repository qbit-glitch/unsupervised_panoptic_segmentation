"""4-Directional Cross-Modal Mamba2 Scanning (VMamba-style).

Processes a 2D feature grid with 4 scan directions:
  H→ (horizontal forward), ←H (horizontal backward),
  V↓ (vertical forward), ↑V (vertical backward)

Each direction runs a BiCMS-style interleaved scan on rows or columns,
then results are merged via learned linear projection.

This gives every patch full 2D spatial context, unlike raster BiCMS
which has weak vertical context (neighbor is 64 tokens away).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from mbps_pytorch.models.bridge.mamba2_ssd import Mamba2Stack
from mbps_pytorch.models.bridge.bicms import interleave_tokens, deinterleave_tokens


class DirectionalBiCMS(nn.Module):
    """Bidirectional cross-modal scan for a single direction.

    Interleaves two streams, runs forward+backward Mamba2, merges via
    learned gating, and de-interleaves.

    Args:
        dim: Bridge dimension.
        num_layers: Mamba2 layers per direction.
        state_dim: SSM state dimension.
        chunk_size: GPU-aligned chunk size.
        dropout_rate: Dropout rate.
    """

    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 2,
        state_dim: int = 64,
        chunk_size: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.forward_mamba = Mamba2Stack(
            num_layers=num_layers, dim=dim,
            state_dim=state_dim, chunk_size=chunk_size,
            dropout_rate=dropout_rate,
        )
        self.backward_mamba = Mamba2Stack(
            num_layers=num_layers, dim=dim,
            state_dim=state_dim, chunk_size=chunk_size,
            dropout_rate=dropout_rate,
        )
        self.merge_gate = nn.Linear(2 * dim, dim)

    def forward(
        self,
        stream_a: torch.Tensor,
        stream_b: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run bidirectional cross-modal scan.

        Args:
            stream_a: First stream (B_scan, L_scan, D).
            stream_b: Second stream (B_scan, L_scan, D).
            deterministic: Disable dropout if True.

        Returns:
            (fused_a, fused_b): Both (B_scan, L_scan, D).
        """
        # Interleave: [a1, b1, a2, b2, ...]
        interleaved = interleave_tokens(stream_a, stream_b)  # (B_scan, 2*L, D)

        # Forward scan
        fwd = self.forward_mamba(interleaved, deterministic=deterministic)

        # Backward scan (flip → process → flip)
        bwd = self.backward_mamba(
            torch.flip(interleaved, dims=[1]), deterministic=deterministic
        )
        bwd = torch.flip(bwd, dims=[1])

        # Merge via sigmoid gating
        gate = torch.sigmoid(
            self.merge_gate(torch.cat([fwd, bwd], dim=-1))
        )
        merged = gate * fwd + (1.0 - gate) * bwd

        # De-interleave
        fused_a, fused_b = deinterleave_tokens(merged)
        return fused_a, fused_b


class FourDirectionalCrossModalScan(nn.Module):
    """4-directional cross-modal Mamba2 scanning.

    Processes 2D spatial grids with horizontal and vertical scans,
    each bidirectional with cross-modal interleaving.

    Args:
        dim: Bridge dimension (256).
        num_layers: Mamba2 layers per direction (2).
        state_dim: SSM state dimension (64).
        chunk_size: GPU chunk size (128).
        dropout_rate: Dropout rate.
        spatial_h: Spatial grid height (32).
        spatial_w: Spatial grid width (64).
    """

    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 2,
        state_dim: int = 64,
        chunk_size: int = 128,
        dropout_rate: float = 0.1,
        spatial_h: int = 32,
        spatial_w: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w

        # Horizontal BiCMS (scans rows left↔right)
        self.h_bicms = DirectionalBiCMS(
            dim=dim, num_layers=num_layers,
            state_dim=state_dim, chunk_size=chunk_size,
            dropout_rate=dropout_rate,
        )

        # Vertical BiCMS (scans columns top↔bottom)
        self.v_bicms = DirectionalBiCMS(
            dim=dim, num_layers=num_layers,
            state_dim=state_dim, chunk_size=chunk_size,
            dropout_rate=dropout_rate,
        )

        # Merge 4 directions (H_fwd+H_bwd merged, V_fwd+V_bwd merged → cat → proj)
        # Each BiCMS already merges fwd/bwd, so we merge H and V
        self.sem_merge = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
        )
        self.inst_merge = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(
        self,
        stream_sem: torch.Tensor,
        stream_inst: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 4-directional cross-modal scanning.

        Args:
            stream_sem: Semantic stream (B, N, D) where N = H*W.
            stream_inst: Instance stream (B, N, D).
            deterministic: Disable dropout if True.

        Returns:
            (fused_sem, fused_inst): Both (B, N, D).
        """
        B, N, D = stream_sem.shape
        H, W = self.spatial_h, self.spatial_w
        assert N == H * W, f"Expected N={H*W}, got {N}"

        # Reshape to 2D grid: (B, H, W, D)
        sem_2d = stream_sem.reshape(B, H, W, D)
        inst_2d = stream_inst.reshape(B, H, W, D)

        # --- HORIZONTAL SCANS (rows) ---
        # Reshape: (B*H, W, D) — each row is an independent sequence
        sem_rows = sem_2d.reshape(B * H, W, D)
        inst_rows = inst_2d.reshape(B * H, W, D)

        h_sem, h_inst = self.h_bicms(sem_rows, inst_rows, deterministic)
        # Reshape back: (B, H, W, D)
        h_sem = h_sem.reshape(B, H, W, D)
        h_inst = h_inst.reshape(B, H, W, D)

        # --- VERTICAL SCANS (columns) ---
        # Transpose to (B, W, H, D), then reshape: (B*W, H, D)
        sem_cols = sem_2d.permute(0, 2, 1, 3).reshape(B * W, H, D)
        inst_cols = inst_2d.permute(0, 2, 1, 3).reshape(B * W, H, D)

        v_sem, v_inst = self.v_bicms(sem_cols, inst_cols, deterministic)
        # Reshape back: (B, W, H, D) → (B, H, W, D)
        v_sem = v_sem.reshape(B, W, H, D).permute(0, 2, 1, 3)
        v_inst = v_inst.reshape(B, W, H, D).permute(0, 2, 1, 3)

        # --- MERGE H + V ---
        # (B, H, W, 2D) → (B, H, W, D) → (B, N, D)
        fused_sem = self.sem_merge(
            torch.cat([h_sem, v_sem], dim=-1)
        ).reshape(B, N, D)

        fused_inst = self.inst_merge(
            torch.cat([h_inst, v_inst], dim=-1)
        ).reshape(B, N, D)

        return fused_sem, fused_inst
