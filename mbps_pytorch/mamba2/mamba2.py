# Copyright (c) 2024, Tri Dao, Albert Gu.
# Mac-compatible Mamba2 module (pure PyTorch, no CUDA/Triton).
# Ported from official state-spaces/mamba v2.3.0 modules/mamba2.py.
# Removed: Triton kernels, causal_conv1d, distributed parallelism, HuggingFace mixin.
# Works on CPU, MPS (Apple Silicon), and CUDA.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .ssd import ssd_chunk_scan_combined
from .norm import RMSNormGated, rms_norm_fn


class Mamba2(nn.Module):
    """Mamba2 module with pure PyTorch SSD backend.

    Drop-in replacement for the official Mamba2 class, minus distributed
    parallelism and fused CUDA/Triton kernels. All compute uses standard
    PyTorch ops that work on CPU, MPS, and CUDA.

    Arguments:
        d_model: input/output dimension
        d_state: SSM state dimension (N in paper). Default 128.
        d_conv: causal conv1d kernel width. Default 4.
        conv_init: uniform init range for conv weights, or None for default.
        expand: expansion factor for inner dimension. Default 2.
        headdim: dimension per head (P in paper). Default 64.
        d_ssm: if set, only apply SSM on this many dims; rest uses gated MLP.
        ngroups: number of head groups for B/C (G in paper). Default 1.
        A_init_range: (min, max) for uniform init of A. Default (1, 16).
        D_has_hdim: if True, D has shape (nheads, headdim) instead of (nheads,).
        rmsnorm: if True, apply RMSNorm before output projection. Default True.
        norm_before_gate: RMSNorm placement relative to gating. Default False.
        dt_min/dt_max: range for dt initialization.
        dt_init_floor: floor for dt init.
        dt_limit: (min, max) clamp for dt values.
        bias: if True, use bias in linear projections.
        conv_bias: if True, use bias in conv1d.
        chunk_size: chunk size for SSD block decomposition. Default 256.
        layer_idx: optional layer index (for compatibility).
    """

    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        chunk_size=256,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm
        self.ngroups = ngroups
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A parameter (log-space)
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        # Optional RMSNorm before output projection
        if self.rmsnorm:
            self.norm = RMSNormGated(
                self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // ngroups, **factory_kwargs,
            )

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u):
        """
        Arguments:
            u: (batch, seqlen, d_model)
        Returns:
            out: (batch, seqlen, d_model)
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads,)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        # Split the projection: [z0, x0] (MLP), z (gate), xBC (conv input), dt
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Causal conv1d (pure PyTorch fallback — no causal_conv1d CUDA kernel)
        xBC = self.act(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :seqlen]
        )  # (B, L, d_ssm + 2 * ngroups * d_state)

        # Split into x, B, C
        x, B, C = torch.split(
            xBC,
            [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1
        )

        # Reshape for SSD
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)

        # Core SSD computation
        y = ssd_chunk_scan_combined(
            x, dt, A, B, C,
            chunk_size=self.chunk_size,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            **dt_limit_kwargs,
        )
        y = rearrange(y, "b l h p -> b l (h p)")

        # RMSNorm + gating
        if self.rmsnorm:
            y = self.norm(y, z)

        # Gated MLP branch (if d_ssm < d_inner)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)

        # Output projection
        out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        """Single-step inference (autoregressive decoding).

        Arguments:
            hidden_states: (batch, 1, d_model)
            conv_state: (batch, conv_dim, d_conv) — updated in-place
            ssm_state: (batch, nheads, headdim, d_state) — updated in-place
        Returns:
            out: (batch, 1, d_model)
            conv_state: updated conv state
            ssm_state: updated ssm state
        """
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time"

        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B, d_in_proj)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step (pure PyTorch — shift state and apply conv)
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = xBC
        xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
        if self.conv1d.bias is not None:
            xBC = xBC + self.conv1d.bias
        xBC = self.act(xBC).to(dtype=dtype)

        # Split
        x, B, C = torch.split(
            xBC,
            [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1
        )
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step (pure PyTorch — no selective_state_update Triton kernel)
        assert self.ngroups == 1, "Only support ngroups=1 for step inference"
        # Discretize A and B
        dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
        if self.dt_limit != (0.0, float("inf")):
            dt = dt.clamp(min=self.dt_limit[0], max=self.dt_limit[1])
        dA = torch.exp(dt * A)  # (batch, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
        ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
        y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")

        # Norm + gating
        if not self.rmsnorm:
            y = y * self.act(z)
        else:
            y = self.norm(y, z)

        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """Allocate conv and SSM states for autoregressive inference."""
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0],
            device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state,
            device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state
