"""Mamba2 Structured State Space Duality (SSD) for TPU.

Memory-efficient implementation using chunked matrix multiplications
with lax.scan to process one chunk at a time.

Key insight: SSD formulates scan as chunked matrix products:
    y_t = Σ_{s=1}^{t} (Π_{r=s+1}^{t} A_r) B_s x_s

    Chunked into blocks of size P:
    Y_chunk = M_chunk @ (B_chunk ⊗ X_chunk) + inter_chunk_correction

Memory optimization: The M matrix (P, P, D) is only materialized for
one chunk at a time via lax.scan, reducing peak memory from
O(C × P² × D) to O(P² × D) where C = L/P is number of chunks.

Reference: Mamba2 paper (Gu & Dao, 2024), Section 7.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange


class SSDKernel(nn.Module):
    """Structured State Space Duality (SSD) Kernel.

    Memory-efficient implementation: processes chunks sequentially via
    lax.scan so only one chunk's M matrix lives in memory at a time.

    Attributes:
        dim: Model dimension (D_b = 192).
        state_dim: SSM state dimension (N = 64).
        chunk_size: Chunk size (P = 64, reduced for memory).
        dt_rank: Rank of Δ projection.
    """

    dim: int = 192
    state_dim: int = 16
    chunk_size: int = 64
    dt_rank: int = 16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
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
            x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0)))

        l_padded = x.shape[1]

        # Learned SSM parameters
        # A: diagonal state transition (discretized)
        A_log = self.param(
            "A_log",
            lambda key, shape: -jnp.ones(shape) * jnp.log(2.0),
            (d,),
        )
        A_log_clamped = jnp.clip(A_log, -20.0, 2.0)
        A = -jnp.exp(A_log_clamped)  # (D,) — negative for stability

        # B, C projections from input
        B_proj = nn.Dense(n, use_bias=False, name="B_proj")
        C_proj = nn.Dense(n, use_bias=False, name="C_proj")

        # Δ (discretization step) projection
        dt_proj = nn.Dense(d, use_bias=False, name="dt_proj")

        # dt_bias: initialized via inverse softplus of uniform[0.001, 0.1]
        # per Mamba2 reference — constrains initial dt to small stable values
        dt_bias = self.param(
            "dt_bias",
            lambda key, shape: jnp.log(jnp.expm1(
                jax.random.uniform(key, shape, minval=0.001, maxval=0.1)
            )),
            (d,),
        )

        # Input projection: x → (z, x_bar) where z is gate
        x_proj = nn.Dense(2 * d, name="x_proj")(x)
        x_bar, z = jnp.split(x_proj, 2, axis=-1)

        # Compute B, C, Δ with RMSNorm (per Mamba2 paper Section 7)
        # RMSNorm bounds B, C to unit RMS regardless of input magnitude,
        # preventing the O(P×N) amplification in the SSD matmul chain.
        B = nn.RMSNorm(name="B_norm")(B_proj(x_bar))   # (B, L, N)
        C = nn.RMSNorm(name="C_norm")(C_proj(x_bar))   # (B, L, N)
        dt = jnp.clip(
            jax.nn.softplus(dt_proj(x_bar) + dt_bias[None, None, :]),
            1e-4, 5.0,
        )  # (B, L, D)

        # Discretize A: Ā = exp(Δ * A), clamped for stability
        dt_A = jnp.clip(dt * A[None, None, :], -20.0, 0.0)
        A_bar = jnp.exp(dt_A)  # (B, L, D) — always in (0, 1]

        # =============================================
        # MEMORY-EFFICIENT CHUNKED SSD (via lax.scan)
        # =============================================
        # Process one chunk at a time to avoid materializing the
        # full M tensor (B, C, P, P, D) which can exceed TPU HBM.
        # With lax.scan, only one chunk's M (B, P, P, D) is live.

        # Reshape into chunks: scan axis first for lax.scan
        x_chunks = rearrange(x_bar, "b (c p) d -> c b p d", p=p)
        B_chunks = rearrange(B, "b (c p) n -> c b p n", p=p)
        C_chunks = rearrange(C, "b (c p) n -> c b p n", p=p)
        A_bar_chunks = rearrange(A_bar, "b (c p) d -> c b p d", p=p)

        # Static causal mask (shared across all chunks)
        causal_mask = jnp.tril(jnp.ones((p, p)))  # (P, P)

        def process_chunk(carry, chunk_inputs):
            """Process one chunk: intra-chunk matmul + inter-chunk correction.

            Args:
                carry: Previous chunk's final SSM state (B, N, D).
                chunk_inputs: Tuple of (x_c, B_c, C_c, A_bar_c).

            Returns:
                (new_state, chunk_output) where chunk_output is (B, P, D).
            """
            prev_state = carry  # (B, N, D)
            x_c, B_c, C_c, A_bar_c = chunk_inputs
            # x_c: (B, P, D), B_c: (B, P, N), C_c: (B, P, N), A_bar_c: (B, P, D)

            # --- INTRA-CHUNK: causal matmul within this chunk ---
            # Build M for this chunk only: (B, P, P, D)
            log_a = jnp.log(jnp.clip(A_bar_c, 1e-8, 1.0))  # (B, P, D)
            log_a_cumsum = jnp.cumsum(log_a, axis=1)  # (B, P, D)

            # M[i,j] = exp(cumsum[i] - cumsum[j]) for i >= j
            log_m = (
                log_a_cumsum[:, :, None, :]
                - log_a_cumsum[:, None, :, :]
            )  # (B, P, P, D)
            M = jnp.exp(log_m) * causal_mask[None, :, :, None]  # (B, P, P, D)

            # BX outer product: (B, P, N, D)
            BX = jnp.einsum("bpn,bpd->bpnd", B_c, x_c)

            # Causal contraction: state[i] = Σ_{j<=i} M[i,j] * BX[j]
            state = jnp.einsum("bijd,bjnd->bind", M, BX)  # (B, P, N, D)

            # Intra-chunk output
            y_intra = jnp.einsum("bpn,bpnd->bpd", C_c, state)  # (B, P, D)

            # --- INTER-CHUNK: contribution from previous chunk's state ---
            # Position-dependent A decay within this chunk:
            # decay[p] = Π_{r=0}^{p} A_bar[r] = exp(cumsum[p])
            decay = jnp.exp(log_a_cumsum)  # (B, P, D)

            # Previous state decayed to each position: (B, P, N, D)
            decayed_prev = prev_state[:, None, :, :] * decay[:, :, None, :]

            # Project with C
            y_inter = jnp.einsum("bpn,bpnd->bpd", C_c, decayed_prev)  # (B, P, D)

            # Total chunk output
            y_c = y_intra + y_inter

            # --- UPDATE STATE for next chunk ---
            # New state = prev_state * total_decay + Σ B[p] * x[p] * decay_to_end[p]
            # decay_to_end[p] = exp(cumsum[P-1] - cumsum[p])
            chunk_a_total = jnp.exp(log_a_cumsum[:, -1, :])  # (B, D) total decay
            log_a_to_end = log_a_cumsum[:, -1:, :] - log_a_cumsum  # (B, P, D)
            a_to_end = jnp.exp(log_a_to_end)  # (B, P, D)

            chunk_state = jnp.einsum(
                "bpn,bpd->bnd", B_c, x_c * a_to_end
            )  # (B, N, D)

            new_state = prev_state * chunk_a_total[:, None, :] + chunk_state

            # Guard against NaN accumulation across chunks.
            # If one chunk produces NaN state, all subsequent chunks would
            # also be NaN. Resetting to 0 allows recovery.
            new_state = jnp.where(jnp.isfinite(new_state), new_state, 0.0)
            y_c = jnp.where(jnp.isfinite(y_c), y_c, 0.0)

            return new_state, y_c

        # Run scan over all chunks (one at a time for memory efficiency)
        # jax.checkpoint prevents storing all scan intermediates for backprop,
        # trading 2x compute for O(num_chunks) memory savings.
        _, y_chunks = jax.lax.scan(
            jax.checkpoint(process_chunk),
            jnp.zeros((b, n, d)),  # initial SSM state
            (x_chunks, B_chunks, C_chunks, A_bar_chunks),
        )
        # y_chunks: (num_chunks, B, P, D)

        # Reshape back to sequence
        y = rearrange(y_chunks, "c b p d -> b (c p) d")

        # Remove padding
        y = y[:, :l, :]

        # Guard scan output: if scan produced NaN, fall back to D skip.
        # This lets the D skip path (y = D * x_bar) provide gradients
        # to x_proj even when the scan is numerically unstable.
        y = jnp.where(jnp.isfinite(y), y, 0.0)

        # D skip connection (per Mamba2 reference)
        D = self.param("D", nn.initializers.ones, (d,))
        y = y + D[None, None, :] * x_bar[:, :l, :]

        # RMSNorm on output before gating (per Mamba2 reference)
        y = nn.RMSNorm(name="output_norm")(y) * jax.nn.silu(z[:, :l, :])

        # Output projection
        y = nn.Dense(d, name="out_proj")(y)

        return y


class Mamba2Block(nn.Module):
    """Single Mamba2 block with SSD kernel + residual + norm.

    Attributes:
        dim: Model dimension.
        state_dim: SSM state dimension.
        chunk_size: Chunk size (64, reduced for memory efficiency).
        expansion_factor: FFN expansion factor.
        dropout_rate: Dropout rate.
    """

    dim: int = 192
    state_dim: int = 16
    chunk_size: int = 64
    expansion_factor: int = 2
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Apply Mamba2 block.

        Args:
            x: Input of shape (B, L, D).
            deterministic: If True, disable dropout.

        Returns:
            Output of shape (B, L, D).
        """
        # Pre-norm + SSD
        residual = x
        x = nn.LayerNorm(name="norm1")(x)
        x = SSDKernel(
            dim=self.dim,
            state_dim=self.state_dim,
            chunk_size=self.chunk_size,
            name="ssd",
        )(x, deterministic=deterministic)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + residual

        # Pre-norm + FFN
        residual = x
        x = nn.LayerNorm(name="norm2")(x)
        inner_dim = self.dim * self.expansion_factor
        x = nn.Dense(inner_dim, name="ffn_up")(x)
        x = jax.nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.dim, name="ffn_down")(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + residual

        return x


class Mamba2Stack(nn.Module):
    """Stack of Mamba2 blocks.

    Attributes:
        num_layers: Number of Mamba2 blocks (4 per SKILL.md).
        dim: Model dimension (192).
        state_dim: SSM state dimension (16, reduced for memory).
        chunk_size: Chunk size (64).
        dropout_rate: Dropout rate.
    """

    num_layers: int = 4
    dim: int = 192
    state_dim: int = 16
    chunk_size: int = 64
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Apply stack of Mamba2 blocks.

        Args:
            x: Input of shape (B, L, D).
            deterministic: If True, disable dropout.

        Returns:
            Output of shape (B, L, D).
        """
        for i in range(self.num_layers):
            x = Mamba2Block(
                dim=self.dim,
                state_dim=self.state_dim,
                chunk_size=self.chunk_size,
                dropout_rate=self.dropout_rate,
                name=f"mamba_block_{i}",
            )(x, deterministic=deterministic)

        x = nn.LayerNorm(name="final_norm")(x)
        return x
