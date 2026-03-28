# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Mac-compatible GatedDeltaNet (pure PyTorch, no CUDA/Triton/FLA).
# Ported from official NVlabs/GatedDeltaNet (ICLR 2025).
# All Triton kernels and FLA dependencies replaced with pure PyTorch.
# Works on CPU, MPS (Apple Silicon), and CUDA.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .norm import RMSNormGated


# ---------------------------------------------------------------------------
# Pure PyTorch delta rule kernels
# From refs/GatedDeltaNet/lit_gpt/gated_delta_rule_ops/chunk.py
# ---------------------------------------------------------------------------

def chunk_gated_delta_rule(q, k, v, beta, g, chunk_size=64):
    """Chunked gated delta rule (pure PyTorch).

    Uses WY representation for efficient within-chunk computation.
    Ported from chunk_gated_delta_rule_ref (chunk.py:639-686).

    Arguments:
        q: (B, H, L, D_k) — queries (L2-normalized)
        k: (B, H, L, D_k) — keys (L2-normalized)
        v: (B, H, L, D_v) — values
        beta: (B, H, L) — per-step learning rate (sigmoid, in [0,1])
        g: (B, H, L) — log-space forget gate (negative scalars)
        chunk_size: int — chunk size for block decomposition
    Returns:
        o: (B, H, L, D_v) — output
    """
    q, k, v, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, beta, g])
    decay = g
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]

    q = q * (d_k ** -0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Pad to multiple of chunk_size
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        k_beta = F.pad(k_beta, (0, 0, 0, pad_len))
        decay = F.pad(decay, (0, pad_len))
    l_padded = q.shape[2]

    # Note: diagonal is masked
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)

    q, k, v, k_beta, decay = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size),
        [q, k, v, k_beta, decay.unsqueeze(-1)]
    )
    decay = decay.squeeze(-1).cumsum(-1)

    # Intra-chunk decay mask — clamp to max=0 before exp to prevent
    # gradient explosion from upper triangle entries (positive differences
    # produce huge exp values; even though masked in forward, gradients
    # still flow through unmasked exp on MPS/CPU backends)
    L_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).clamp(max=0).exp()

    # WY representation: compute corrected values within each chunk
    # The for-loop forward substitution computes (I - L)^{-1} where L is
    # strictly lower-triangular.  Replace with batched triangular solve:
    #   (I - L) @ X = I  =>  X = (I - L)^{-1}
    I_cs = torch.eye(chunk_size, dtype=torch.float, device=q.device)

    L1 = ((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
    # (I - (-L1)) = (I + L1) is unit lower-triangular
    M1 = I_cs + L1
    I_expand = I_cs.expand_as(M1)
    attn = torch.linalg.solve_triangular(M1, I_expand, upper=False, unitriangular=True)
    k_cumsum = attn @ v

    # WY for keys (same pattern, no decay mask)
    L2 = ((k_beta @ k.transpose(-1, -2))).masked_fill(mask, 0)
    M2 = I_cs + L2
    attn = torch.linalg.solve_triangular(M2, I_expand, upper=False, unitriangular=True)
    k_cumdecay = attn @ k_beta

    u = k_cumsum  # corrected values
    S = k.new_zeros(b, h, d_k, d_v)  # recurrent state
    o = torch.zeros_like(v)

    mask2 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    num_chunks = l_padded // chunk_size

    for i in range(num_chunks):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], u[:, :, i]

        # Intra-chunk attention
        attn_i = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask2, 0)

        # Inter-chunk: read from state, correct values
        v_prime = (k_cumdecay[:, :, i] * decay[:, :, i, :, None].exp()) @ S
        v_new = v_i - v_prime

        # Output: inter-chunk read + intra-chunk attention
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn_i @ v_new

        # Update state
        S = (S * decay[:, :, i, -1, None, None].exp() +
             (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new)

    o = rearrange(o, 'b h n c d -> b h (n c) d')

    # Remove padding
    if pad_len > 0:
        o = o[:, :, :l, :]

    return o


def recurrent_gated_delta_rule(q, k, v, beta, g):
    """Sequential gated delta rule (pure PyTorch, for testing).

    Step-by-step recurrence — simple but O(L) sequential.
    Ported from recurrent_gated_delta_rule_ref (chunk.py:688-711).

    Arguments:
        q: (B, H, L, D_k)
        k: (B, H, L, D_k)
        v: (B, H, L, D_v)
        beta: (B, H, L)
        g: (B, H, L)
    Returns:
        o: (B, H, L, D_v)
    """
    q, k, v, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, beta, g])
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v, device=v.device, dtype=v.dtype)
    q = q * (d_k ** -0.5)
    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i].clone()
        S = S.clone() * g[:, :, i].exp()[..., None, None]
        beta_i = beta[:, :, i]
        _v = _v - (S.clone() * _k[..., None]).sum(-2)
        _v = _v * beta_i[..., None]
        S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
    return o


# ---------------------------------------------------------------------------
# GatedDeltaNet nn.Module
# ---------------------------------------------------------------------------

class GatedDeltaNet(nn.Module):
    """GatedDeltaNet layer with pure PyTorch backend.

    Drop-in replacement for Mamba2 with the same (B, L, D) → (B, L, D) interface.
    Uses the delta rule for error-correcting memory updates instead of
    Mamba2's simple accumulation.

    Arguments:
        d_model: input/output dimension (hidden_size)
        num_heads: number of attention heads. If None, auto-computed.
        expand_k: key dimension expansion factor. Default 0.75.
        expand_v: value dimension expansion factor. Default 1.5.
        d_conv: causal conv1d kernel width. Default 4.
        use_mamba_gate: if True, use Mamba2-style gate (A_log + dt_bias). Default True.
        gate_logit_normalizer: normalizer for gate logits. Default 16.
        chunk_size: chunk size for chunked delta rule. Default 64.
        layer_idx: optional layer index.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = None,
        expand_k: float = 0.75,
        expand_v: float = 1.5,
        d_conv: int = 4,
        use_mamba_gate: bool = True,
        gate_logit_normalizer: int = 16,
        chunk_size: int = 64,
        layer_idx: int = None,
        # Accept and ignore Mamba2-specific kwargs for API compatibility
        d_state: int = None,
        expand: int = None,
        headdim: int = None,
        ngroups: int = None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.d_conv = d_conv
        self.use_mamba_gate = use_mamba_gate
        self.gate_logit_normalizer = gate_logit_normalizer
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)

        # Auto-compute num_heads if not provided
        if num_heads is None:
            # Try to find a reasonable head count
            for candidate in [8, 4, 16, 6, 2, 1]:
                if self.key_dim % candidate == 0 and self.value_dim % candidate == 0:
                    num_heads = candidate
                    break
            if num_heads is None:
                num_heads = 1
        self.num_heads = num_heads

        assert self.key_dim % num_heads == 0, f"key_dim {self.key_dim} not divisible by num_heads {num_heads}"
        assert self.value_dim % num_heads == 0, f"value_dim {self.value_dim} not divisible by num_heads {num_heads}"
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # Projections
        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.g_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)

        # Gate and beta projections
        self.gk_proj = nn.Linear(d_model, num_heads, bias=not use_mamba_gate)
        self.b_proj = nn.Linear(d_model, num_heads, bias=True)

        # Causal conv1d for Q, K, V (pure PyTorch, no causal_conv1d CUDA)
        self.q_conv1d = nn.Conv1d(
            self.key_dim, self.key_dim, d_conv,
            groups=self.key_dim, padding=d_conv - 1, bias=False,
        )
        self.k_conv1d = nn.Conv1d(
            self.key_dim, self.key_dim, d_conv,
            groups=self.key_dim, padding=d_conv - 1, bias=False,
        )
        self.v_conv1d = nn.Conv1d(
            self.value_dim, self.value_dim, d_conv,
            groups=self.value_dim, padding=d_conv - 1, bias=False,
        )

        self.act = nn.SiLU()

        # Output norm (replaces FusedRMSNormSwishGate)
        self.o_norm = RMSNormGated(self.head_v_dim, eps=1e-5, norm_before_gate=True)

        # Mamba-style gate (identical to Mamba2 init)
        if use_mamba_gate:
            A = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
            self.A_log = nn.Parameter(torch.log(A))
            self.A_log._no_weight_decay = True

            dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
            dt = torch.exp(
                torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            self.dt_bias._no_weight_decay = True

    def forward(self, u):
        """Forward pass.

        Arguments:
            u: (B, L, D) — input sequence
        Returns:
            out: (B, L, D) — output sequence
        """
        batch, seqlen, dim = u.shape

        # Project Q, K, V
        q = self.q_proj(u)  # (B, L, key_dim)
        k = self.k_proj(u)  # (B, L, key_dim)
        v = self.v_proj(u)  # (B, L, value_dim)

        # Causal conv1d (pure PyTorch): transpose → conv → SiLU → truncate → transpose
        q = self.act(self.q_conv1d(q.transpose(1, 2)).transpose(1, 2)[:, :seqlen])
        k = self.act(self.k_conv1d(k.transpose(1, 2)).transpose(1, 2)[:, :seqlen])
        v = self.act(self.v_conv1d(v.transpose(1, 2)).transpose(1, 2)[:, :seqlen])

        # Compute forget gate (gk)
        gk = self.gk_proj(u).float()  # (B, L, num_heads)
        if self.use_mamba_gate:
            gk = -self.A_log.float().exp() * F.softplus(gk + self.dt_bias)
        else:
            gk = F.logsigmoid(gk) / self.gate_logit_normalizer
        gk = gk.transpose(1, 2)  # (B, H, L)

        # Compute beta (per-step learning rate)
        beta = self.b_proj(u).float().sigmoid().transpose(1, 2)  # (B, H, L)

        # Reshape to multi-head: (B, L, dim) → (B, H, L, head_dim)
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)

        # L2 normalize Q, K
        q = F.normalize(q, p=2, dim=-1).to(v.dtype)
        k = F.normalize(k, p=2, dim=-1).to(v.dtype)

        # Core: chunked gated delta rule
        o = chunk_gated_delta_rule(q, k, v, beta, gk, chunk_size=self.chunk_size)

        # Output: norm + gate + projection
        o = rearrange(o, 'b h l d -> b l h d')
        g = self.g_proj(u)  # (B, L, value_dim)
        g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)

        # RMSNorm(o) * SiLU(g) — replaces FusedRMSNormSwishGate
        o = self.o_norm(o, g)
        o = rearrange(o, 'b l h d -> b l (h d)')

        out = self.o_proj(o)
        return out
