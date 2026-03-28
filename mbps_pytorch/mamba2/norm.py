# Copyright (c) 2024, Tri Dao.
# Pure PyTorch RMSNorm with optional gating.
# Ported from official state-spaces/mamba v2.3.0 layernorm_gated.py:rms_norm_ref.
# No Triton — works on CPU, MPS, and CUDA.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def rms_norm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None,
                norm_before_gate=True, upcast=True):
    """RMSNorm with optional gating (pure PyTorch).

    From layernorm_gated.py:rms_norm_ref.

    Arguments:
        x: (..., dim)
        weight: (dim,)
        bias: (dim,) or None
        z: (..., dim) or None — gating input
        eps: float
        group_size: int or None — if set, normalize within groups of this size
        norm_before_gate: if True, RMSNorm(x) * SiLU(z). If False, RMSNorm(x * SiLU(z))
        upcast: if True, compute in float32
    """
    dtype = x.dtype
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


class RMSNormGated(nn.Module):
    """RMSNorm with optional gating, as an nn.Module.

    Drop-in replacement for the Triton-based RMSNormGated from the official Mamba2.
    """

    def __init__(self, d, eps=1e-5, norm_before_gate=True, bias=False,
                 group_size=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.group_size = group_size
        self.weight = nn.Parameter(torch.ones(d, **factory_kwargs))
        self.bias = nn.Parameter(torch.zeros(d, **factory_kwargs)) if bias else None

    def forward(self, x, z=None):
        return rms_norm_fn(
            x, self.weight, bias=self.bias, z=z, eps=self.eps,
            group_size=self.group_size, norm_before_gate=self.norm_before_gate,
        )
