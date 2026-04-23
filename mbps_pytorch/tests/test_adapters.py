"""Smoke tests for LoRA/DoRA adapter injection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn

from mbps_pytorch.models.adapters import (
    LoRALinear,
    DoRALinear,
    ConvDoRALinear,
    LoRAConv2d,
    freeze_non_adapter_params,
    count_adapter_params,
    count_total_params,
)


def test_lora_linear():
    base = nn.Linear(10, 20)
    adapter = LoRALinear(base, rank=4, alpha=4.0)
    x = torch.randn(2, 10)
    y = adapter(x)
    assert y.shape == (2, 20)
    assert adapter.trainable_count() == 4 * 10 + 20 * 4
    print("LoRALinear OK")


def test_dora_linear():
    base = nn.Linear(10, 20)
    adapter = DoRALinear(base, rank=4, alpha=4.0)
    x = torch.randn(2, 10)
    y = adapter(x)
    assert y.shape == (2, 20)
    assert adapter.trainable_count() == 4 * 10 + 20 * 4 + 20
    print("DoRALinear OK")


def test_conv_dora_linear():
    base = nn.Linear(10, 20)
    adapter = ConvDoRALinear(base, rank=4, alpha=4.0)
    adapter._spatial_dims = (4, 4)
    x = torch.randn(2, 16 + 1, 10)
    y = adapter(x)
    assert y.shape == (2, 17, 20)
    print("ConvDoRALinear OK")


def test_lora_conv2d():
    base = nn.Conv2d(3, 16, kernel_size=1)
    adapter = LoRAConv2d(base, rank=4, alpha=4.0)
    x = torch.randn(2, 3, 8, 8)
    y = adapter(x)
    assert y.shape == (2, 16, 8, 8)
    print("LoRAConv2d OK")


def test_freeze_non_adapter_params():
    model = nn.Sequential(
        LoRALinear(nn.Linear(10, 20), rank=4),
        nn.Linear(20, 10),
    )
    # Unfreeze everything first
    for p in model.parameters():
        p.requires_grad = True
    freeze_non_adapter_params(model)
    # After freezing: only lora params require grad in adapter
    assert not model[0].weight.requires_grad
    assert model[0].lora_A.requires_grad
    assert model[0].lora_B.requires_grad
    # Plain linear should be frozen
    assert not model[1].weight.requires_grad
    print("freeze_non_adapter_params OK")


def test_count_params():
    model = nn.Linear(10, 20)
    total = count_total_params(model)
    assert total == 10 * 20 + 20
    print("count_total_params OK")


if __name__ == "__main__":
    test_lora_linear()
    test_dora_linear()
    test_conv_dora_linear()
    test_lora_conv2d()
    test_freeze_non_adapter_params()
    test_count_params()
    print("All adapter smoke tests passed!")
