"""Mamba2 Structured State Space Duality (SSD) for GPU.

GPU-optimized implementation of the Mamba2 selective state space model
using chunked matrix multiplications instead of sequential scans.

Key insight: SSD formulates scan as chunked matrix products:
    y_t = Sum_{s=1}^{t} (Prod_{r=s+1}^{t} A_r) B_s x_s

    Chunked into blocks of size P (128, GPU-aligned):
    Y_chunk = M_chunk @ (B_chunk (x) X_chunk) + correction_term

where M is the causal mask weighted by state transitions.

Reference: Mamba2 paper (Gu & Dao, 2024), Section 7.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SSDKernel(nn.Module):
    """Structured State Space Duality (SSD) Kernel.

    GPU-optimized implementation using chunked matrix multiplications
    instead of sequential scans.

    Args:
        dim: Model dimension (D_b = 192).
        state_dim: SSM state dimension (N = 64).
        chunk_size: Chunk size for GPU (P = 128).
        dt_rank: Rank of Delta projection.
    """

    def __init__(
        self,
        dim: int = 192,
        state_dim: int = 64,
        chunk_size: int = 128,
        dt_rank: int = 16,
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.dt_rank = dt_rank

        # Learned SSM parameters
        # A: diagonal state transition (discretized)
        # Initialize as -ones * log(2) for stability
        self.A_log = nn.Parameter(-torch.ones(dim) * torch.log(torch.tensor(2.0)))

        # B, C projections from input
        self.B_proj = nn.Linear(dim, state_dim, bias=False)
        self.C_proj = nn.Linear(dim, state_dim, bias=False)

        # Delta (discretization step) projection
        self.dt_proj = nn.Linear(dim, dim)

        # Input projection: x -> (z, x_bar) where z is gate
        self.x_proj = nn.Linear(dim, 2 * dim)

        # Output gating projection
        self.x_proj_gate = nn.Linear(dim, 2 * dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Apply SSD selective scan.

        Args:
            x: Input sequence of shape (B, L, D).
            deterministic: If True, disable dropout.

        Returns:
            Output sequence of shape (B, L, D).
        """
        b, l, d = x.shape
        n = self.state_dim
        p = self.chunk_size

        # Pad sequence length to multiple of chunk_size
        pad_len = (p - l % p) % p
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        l_padded = x.shape[1]
        num_chunks = l_padded // p

        # A: negative for stability
        A = -torch.exp(self.A_log)  # (D,)

        # Input projection: x -> (x_bar, z) where z is gate
        x_proj = self.x_proj(x)
        x_bar, z = torch.split(x_proj, d, dim=-1)

        # Compute B, C, Delta
        B = self.B_proj(x_bar)      # (B, L, N)
        C = self.C_proj(x_bar)      # (B, L, N)
        dt = F.softplus(self.dt_proj(x_bar))  # (B, L, D)

        # Discretize A: A_bar = exp(Delta * A)
        A_bar = torch.exp(dt * A[None, None, :])  # (B, L, D)

        # =============================================
        # CHUNKED SSD COMPUTATION (GPU-optimized)
        # =============================================

        # Reshape into chunks
        x_chunks = rearrange(x_bar, "b (c p) d -> b c p d", p=p)
        B_chunks = rearrange(B, "b (c p) n -> b c p n", p=p)
        C_chunks = rearrange(C, "b (c p) n -> b c p n", p=p)
        A_bar_chunks = rearrange(A_bar, "b (c p) d -> b c p d", p=p)

        # === INTRA-CHUNK: Matrix multiplication within each chunk ===
        # Build causal mask with state transitions
        # M[i,j] = Prod_{r=j+1}^{i} A_bar[r]  for i >= j, 0 otherwise

        # Cumulative product of A_bar within chunk
        # log-space for numerical stability
        log_A = torch.log(torch.abs(A_bar_chunks) + 1e-8)  # (B, C, P, D)
        log_A_cumsum = torch.cumsum(log_A, dim=2)  # (B, C, P, D)

        # M[i,j] = exp(cumsum[i] - cumsum[j]) for i >= j
        # Shape: (B, C, P, P, D)
        log_M = log_A_cumsum[:, :, :, None, :] - log_A_cumsum[:, :, None, :, :]

        # Causal mask
        causal = torch.tril(torch.ones((p, p), device=x.device))  # (P, P)
        M = torch.exp(log_M) * causal[None, None, :, :, None]  # (B, C, P, P, D)

        # Intra-chunk output via matrix multiply
        # y_intra[i] = Sum_j M[i,j] * (B[j]^T x[j]) projected by C[i]
        # BX = B (x) X -> (B, C, P, N, D) via outer product
        BX = torch.einsum("bcpn,bcpd->bcpnd", B_chunks, x_chunks)  # (B,C,P,N,D)

        # Contract with causal M over positions
        # For each output position i: state = Sum_j M[i,j,d] * BX[j,n,d]
        # This is the key GPU-friendly matmul
        state_intra = torch.einsum(
            "bcijd,bcjnd->bcind", M, BX
        )  # (B, C, P, N, D)

        # Project back with C
        y_intra = torch.einsum("bcpn,bcpnd->bcpd", C_chunks, state_intra)

        # === INTER-CHUNK: State propagation across chunks ===
        # Compute chunk-level state transitions
        # Final state of each chunk propagates to next chunk

        # Chunk-level cumulative A product
        chunk_A = torch.prod(A_bar_chunks, dim=2)  # (B, num_chunks, D)

        # Final state per chunk
        chunk_states = torch.einsum(
            "bcpn,bcpd->bcnd",
            B_chunks,
            x_chunks * A_bar_chunks,
        )  # (B, num_chunks, N, D)

        # Propagate states across chunks using Python for loop
        # (replaces jax.lax.scan)
        propagated_list = []
        prev_state = torch.zeros((b, n, d), device=x.device)  # initial state

        for chunk_idx in range(num_chunks):
            cs = chunk_states[:, chunk_idx]   # (B, N, D)
            ca = chunk_A[:, chunk_idx]         # (B, D)
            new_state = prev_state * ca[:, None, :] + cs
            propagated_list.append(new_state)
            prev_state = new_state

        propagated = torch.stack(propagated_list, dim=1)  # (B, num_chunks, N, D)

        # Inter-chunk correction: add contribution from previous chunks
        # Shift propagated states by 1 (chunk i uses state from chunk i-1)
        prev_states = torch.cat(
            [torch.zeros((b, 1, n, d), device=x.device), propagated[:, :-1]],
            dim=1,
        )  # (B, num_chunks, N, D)

        # Expand prev_state across positions within chunk
        # Apply position-dependent A decay within chunk
        y_inter = torch.einsum(
            "bcpn,bcnd->bcpd",
            C_chunks,
            prev_states,
        )

        # Total output
        y = y_intra + y_inter  # (B, num_chunks, P, D)

        # Reshape back
        y = rearrange(y, "b c p d -> b (c p) d")

        # Remove padding
        y = y[:, :l, :]

        # Gating with SiLU
        y = y * torch.nn.functional.silu(z[:, :l, :])

        # Output projection
        y = self.out_proj(y)

        return y


class Mamba2Block(nn.Module):
    """Single Mamba2 block with SSD kernel + residual + norm.

    Args:
        dim: Model dimension.
        state_dim: SSM state dimension.
        chunk_size: Chunk size for GPU.
        expansion_factor: FFN expansion factor.
        dropout_rate: Dropout rate.
    """

    def __init__(
        self,
        dim: int = 192,
        state_dim: int = 64,
        chunk_size: int = 128,
        expansion_factor: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.expansion_factor = expansion_factor
        self.dropout_rate = dropout_rate

        # Pre-norm + SSD
        self.norm1 = nn.LayerNorm(dim)
        self.ssd = SSDKernel(
            dim=dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
        )
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # Pre-norm + FFN
        self.norm2 = nn.LayerNorm(dim)
        inner_dim = dim * expansion_factor
        self.ffn_up = nn.Linear(dim, inner_dim)
        self.ffn_down = nn.Linear(inner_dim, dim)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.dropout3 = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Apply Mamba2 block.

        Args:
            x: Input of shape (B, L, D).
            deterministic: If True, disable dropout.

        Returns:
            Output of shape (B, L, D).
        """
        # Pre-norm + SSD
        residual = x
        x = self.norm1(x)
        x = self.ssd(x, deterministic=deterministic)
        if not deterministic:
            x = self.dropout1(x)
        x = x + residual

        # Pre-norm + FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn_up(x)
        x = F.gelu(x)
        if not deterministic:
            x = self.dropout2(x)
        x = self.ffn_down(x)
        if not deterministic:
            x = self.dropout3(x)
        x = x + residual

        return x


class Mamba2Stack(nn.Module):
    """Stack of Mamba2 blocks.

    Args:
        num_layers: Number of Mamba2 blocks (4 per SKILL.md).
        dim: Model dimension (192).
        state_dim: SSM state dimension (64).
        chunk_size: Chunk size for GPU (128).
        dropout_rate: Dropout rate.
    """

    def __init__(
        self,
        num_layers: int = 4,
        dim: int = 192,
        state_dim: int = 64,
        chunk_size: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.dropout_rate = dropout_rate

        self.blocks = nn.ModuleList([
            Mamba2Block(
                dim=dim,
                state_dim=state_dim,
                chunk_size=chunk_size,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Apply stack of Mamba2 blocks.

        Args:
            x: Input of shape (B, L, D).
            deterministic: If True, disable dropout.

        Returns:
            Output of shape (B, L, D).
        """
        for block in self.blocks:
            x = block(x, deterministic=deterministic)

        x = self.final_norm(x)
        return x
