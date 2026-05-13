"""Scaled Low-Rank (SLR) adapters from GDA (CVPR 2024).

Ported from: https://github.com/HSG-AIML/GDA
Reference: "Parameter Efficient Self-Supervised Geospatial Domain Adaptation"

SLR adapters extend low-rank adaptation with learnable input/output channel
scaling vectors, providing more flexible adaptation than plain LoRA.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledLowRankAdapter(nn.Module):
    """SLR adapter for linear layers.

    Adds a low-rank bottleneck (down -> up) plus learnable input/output
    scaling vectors around a frozen linear layer.
    """

    def __init__(
        self,
        linear: nn.Linear,
        hidden_dim: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = linear
        out_dim, in_dim = self.linear.weight.shape
        self.out_dim = out_dim
        self.in_dim = in_dim

        # Freeze original layer
        for p in self.linear.parameters():
            p.requires_grad = False

        # Learnable scaling vectors (similar to IA3 / FiLM)
        self.in_scaler = nn.Parameter(torch.ones(in_dim))
        self.out_scaler = nn.Parameter(torch.ones(out_dim))

        # Low-rank bottleneck
        self.down = nn.Linear(in_dim, hidden_dim, bias=False)
        self.up = nn.Linear(hidden_dim, out_dim, bias=False)

        # Init: up=0 (zero contribution at start), down=normal
        nn.init.zeros_(self.up.weight)
        nn.init.normal_(self.down.weight, std=0.02)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input scaling (non-destructive, like FiLM)
        x_scaled = x * self.in_scaler

        # Frozen base path
        base = self.linear(x_scaled)

        # Low-rank residual path
        x_lr = self.up(self.dropout(self.down(x_scaled)))

        # Combine and apply output scaling
        out = (base + x_lr) * self.out_scaler
        return out

    def trainable_count(self) -> int:
        return (
            self.in_scaler.numel()
            + self.out_scaler.numel()
            + self.down.weight.numel()
            + self.up.weight.numel()
        )

    def merge(self) -> None:
        """Merge SLR weights into base linear for efficient inference."""
        if getattr(self, "merged", False):
            return
        # Store original weight/bias before merge
        self.register_buffer("_original_weight", self.linear.weight.data.clone())
        if self.linear.bias is not None:
            self.register_buffer("_original_bias", self.linear.bias.data.clone())

        # Compute effective weight: W_eff = out_scaler * (W + up @ down) * in_scaler
        W = self.linear.weight.data
        up = self.up.weight.data  # (out, hidden)
        down = self.down.weight.data  # (hidden, in)

        # Effective low-rank update
        delta = up @ down  # (out, in)

        # Apply scaling
        W_eff = (W + delta) * self.out_scaler.unsqueeze(1) * self.in_scaler.unsqueeze(0)
        self.linear.weight.data.copy_(W_eff)

        # Handle bias
        if self.linear.bias is not None:
            b = self.linear.bias.data
            self.linear.bias.data.copy_(b * self.out_scaler)

        self.merged = True

    def unmerge(self) -> None:
        """Restore base linear from merged state."""
        if not getattr(self, "merged", False):
            return
        if hasattr(self, "_original_weight"):
            self.linear.weight.data.copy_(self._original_weight)
        if hasattr(self, "_original_bias"):
            self.linear.bias.data.copy_(self._original_bias)
        self.merged = False


class ScaledLowRankConvAdapter(nn.Module):
    """SLR adapter for Conv2d layers (e.g., patch embed)."""

    def __init__(
        self,
        conv2d: nn.Conv2d,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = conv2d

        # Freeze original
        for p in self.proj.parameters():
            p.requires_grad = False

        # Channel-wise scaler
        self.scaler = nn.Parameter(torch.ones(self.proj.out_channels))

        # Low-rank conv bottleneck
        kernel_size = (4, 4) if conv2d.kernel_size == (16, 16) else (2, 2)
        self.down = nn.Conv2d(
            self.proj.in_channels,
            self.hidden_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )
        self.up = nn.Conv2d(
            self.hidden_dim,
            self.proj.out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

        nn.init.zeros_(self.up.weight)
        nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_lr = self.up(self.down(x))
        x = self.proj(x)
        x = x + x_lr
        return torch.einsum("bdhw,d->bdhw", x, self.scaler)

    def trainable_count(self) -> int:
        return (
            self.scaler.numel()
            + sum(p.numel() for p in self.down.parameters())
            + sum(p.numel() for p in self.up.parameters())
        )


def wrap_with_slr(
    model: nn.Module,
    target_modules: str = ".*attn|.*mlp",
    target_layers: str = "qkv|fc1|fc2|proj",
    hidden_dim: int = 8,
    dropout: float = 0.0,
    also_unfreeze_norm: bool = True,
) -> None:
    """Inject SLR adapters into a model via regex matching (GDA-style).

    Args:
        model: Model to adapt.
        target_modules: Regex matching module names to inject into.
        target_layers: Regex matching layer names within matched modules.
        hidden_dim: Bottleneck dimension for SLR adapters.
        dropout: Dropout on low-rank path.
        also_unfreeze_norm: Whether to unfreeze LayerNorm parameters.
    """
    import re

    for m_name, module in dict(model.named_modules()).items():
        if re.fullmatch(target_modules, m_name):
            children = dict(module.named_children())
            set_as_module = False
            if not children:
                set_as_module = True
                children = {m_name: module}
            for c_name, layer in children.items():
                if re.fullmatch(target_layers, c_name):
                    if isinstance(layer, nn.Linear):
                        adapter = ScaledLowRankAdapter(layer, hidden_dim=hidden_dim, dropout=dropout)
                        if set_as_module:
                            setattr(model, c_name, adapter)
                        else:
                            setattr(module, c_name, adapter)

    if also_unfreeze_norm:
        for n, p in model.named_parameters():
            if "norm" in n:
                p.requires_grad = True
