"""Core LoRA / DoRA / Conv-DoRA layer wrappers.

Reusable across DINOv2, DINOv3, CAUSE-TR, and depth models.
Adapted from refs/cups/cups/model/lora.py with additions for Conv2d support.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Linear adapters
# --------------------------------------------------------------------------- #

class LoRALinear(nn.Module):
    """Standard LoRA-adapted linear layer (Hu et al., ICLR 2022)."""

    def __init__(
        self,
        wrapped: nn.Linear,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_features = wrapped.in_features
        self.out_features = wrapped.out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(wrapped.weight.data.clone(), requires_grad=False)
        if wrapped.bias is not None:
            self.bias = nn.Parameter(wrapped.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        dev = wrapped.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure weight dtype matches input (for mixed-precision training)
        weight = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        base_out = F.linear(x, weight, bias)
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        lora_out = self.lora_dropout(
            F.linear(F.linear(x, lora_A), lora_B) * self.scaling
        )
        return base_out + lora_out

    def trainable_count(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


class DoRALinear(nn.Module):
    """DoRA-adapted linear layer (Liu et al., ICML 2024 Oral).

    Note on normalization: we use row-wise L2 normalization (dim=1) rather than
    column-wise. This is consistent with PyTorch nn.Linear where each row of the
    weight matrix corresponds to one output dimension. The parameter count table
    in Architecture-DINOv2-CAUSE-TR-Adapters.md assumes this row-wise formulation.
    """
    

    def __init__(
        self,
        wrapped: nn.Linear,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_features = wrapped.in_features
        self.out_features = wrapped.out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(wrapped.weight.data.clone(), requires_grad=False)
        if wrapped.bias is not None:
            self.bias = nn.Parameter(wrapped.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.lora_magnitude = nn.Parameter(
            self.weight.data.norm(dim=1, keepdim=True).clone()
        )

        dev = wrapped.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        delta_V = self.lora_dropout(self.scaling * (lora_B @ lora_A))
        V_prime = self.weight.to(x.dtype) + delta_V.to(x.dtype)
        V_norm = V_prime.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_prime = self.lora_magnitude.to(x.dtype) * (V_prime / V_norm.detach())
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, W_prime, bias)

    def trainable_count(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel() + self.lora_magnitude.numel()


class ConvDoRALinear(DoRALinear):
    """Conv-DoRA: DoRA weight-space + DWConv activation-space refinement."""

    def __init__(
        self,
        wrapped: nn.Linear,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(wrapped, rank=rank, alpha=alpha, dropout=dropout)

        dev = wrapped.weight.device
        self.dwconv = nn.Conv2d(
            rank, rank, kernel_size=3, padding=1, groups=rank, bias=False,
        ).to(dev)
        nn.init.zeros_(self.dwconv.weight)
        self.conv_gate = nn.Parameter(torch.zeros(1, device=dev))
        self._spatial_dims: Optional[Tuple[int, int]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        delta_V = self.lora_dropout(self.scaling * (lora_B @ lora_A))
        V_prime = self.weight.to(x.dtype) + delta_V.to(x.dtype)
        V_norm = V_prime.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_prime = self.lora_magnitude.to(x.dtype) * (V_prime / V_norm.detach())
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        dora_out = F.linear(x, W_prime, bias)

        h_p, w_p = self._spatial_dims or (None, None)
        if h_p is not None and w_p is not None:
            B_size = x.shape[0]
            n_patches = h_p * w_p
            seq_len = x.shape[1]
            n_special = seq_len - n_patches

            z = F.linear(x, self.lora_A)
            z_patch = z[:, n_special:, :]
            z_2d = z_patch.transpose(1, 2).reshape(B_size, self.rank, h_p, w_p)
            z_conv = self.dwconv(z_2d)
            z_flat = z_conv.reshape(B_size, self.rank, -1).transpose(1, 2)
            if n_special > 0:
                zeros = torch.zeros(B_size, n_special, self.rank, device=z.device, dtype=z.dtype)
                z_flat = torch.cat([zeros, z_flat], dim=1)
            conv_out = F.linear(z_flat, self.lora_B) * self.scaling
            dora_out = dora_out + self.conv_gate * conv_out

        return dora_out

    def trainable_count(self) -> int:
        base = super().trainable_count()
        conv_params = self.dwconv.weight.numel() + self.conv_gate.numel()
        return base + conv_params


# --------------------------------------------------------------------------- #
# Conv2d adapter (for CAUSE-TR 1x1 convs)
# --------------------------------------------------------------------------- #

class LoRAConv2d(nn.Module):
    """LoRA-adapted Conv2d layer (for 1x1 convolutions in CAUSE-TR head).

    Uses two 1x1 convs as the low-rank decomposition:
        A: (in_channels -> rank, 1x1)
        B: (rank -> out_channels, 1x1)
    """

    def __init__(
        self,
        wrapped: nn.Conv2d,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = wrapped.in_channels
        self.out_channels = wrapped.out_channels
        self.kernel_size = wrapped.kernel_size
        self.stride = wrapped.stride
        self.padding = wrapped.padding
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen base weight and bias
        self.weight = nn.Parameter(wrapped.weight.data.clone(), requires_grad=False)
        if wrapped.bias is not None:
            self.bias = nn.Parameter(wrapped.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        dev = wrapped.weight.device
        self.lora_A = nn.Conv2d(
            self.in_channels, rank, kernel_size=1, bias=False,
        ).to(dev)
        self.lora_B = nn.Conv2d(
            rank, self.out_channels, kernel_size=1, bias=False,
        ).to(dev)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.lora_dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        base_out = F.conv2d(x, weight, bias, self.stride, self.padding)
        lora_out = self.lora_dropout(
            self.lora_B(self.lora_A(x)) * self.scaling
        )
        return base_out + lora_out

    def trainable_count(self) -> int:
        return sum(p.numel() for p in [self.lora_A.weight, self.lora_B.weight])


# --------------------------------------------------------------------------- #
# Generic injection helpers
# --------------------------------------------------------------------------- #

ADAPTER_CLASSES = {
    "lora": LoRALinear,
    "dora": DoRALinear,
    "conv_dora": ConvDoRALinear,
}


def wrap_linear_if_match(
    parent: nn.Module,
    attr_name: str,
    adapter_cls,
    rank: int,
    alpha: float,
    dropout: float,
) -> Optional[int]:
    """Wrap a nn.Linear child with a LoRA/DoRA adapter if it exists.

    Returns:
        Number of trainable params added, or None if not adapted.
    """
    original = getattr(parent, attr_name, None)
    if original is None or not isinstance(original, nn.Linear):
        return None
    adapter = adapter_cls(
        wrapped=original,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    setattr(parent, attr_name, adapter)
    return adapter.trainable_count()


def wrap_conv2d_if_match(
    parent: nn.Module,
    attr_name: str,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> Optional[int]:
    """Wrap a nn.Conv2d child with a LoRAConv2d adapter if it exists."""
    original = getattr(parent, attr_name, None)
    if original is None or not isinstance(original, nn.Conv2d):
        return None
    adapter = LoRAConv2d(
        wrapped=original,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    setattr(parent, attr_name, adapter)
    return adapter.trainable_count()


def freeze_non_adapter_params(model: nn.Module) -> None:
    """Freeze all parameters except adapter (LoRA/DoRA) parameters.

    Uses suffix matching for robustness against accidental substring matches
    in base model parameter names.
    """
    ADAPTER_SUFFIXES = (".lora_A", ".lora_B", ".lora_magnitude", ".dwconv.weight", ".conv_gate",
                        ".lora_A.weight", ".lora_B.weight")
    for name, param in model.named_parameters():
        if any(name.endswith(suffix) for suffix in ADAPTER_SUFFIXES):
            param.requires_grad = True
        else:
            param.requires_grad = False


def count_adapter_params(model: nn.Module) -> int:
    """Count total trainable adapter parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())
