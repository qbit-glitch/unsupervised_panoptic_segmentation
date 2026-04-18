"""Shared fixtures for Stage-2 Mask2Former tests.

Keeps a tiny deterministic batch (B=2, H=64, W=128) + dummy DINOv3
patch-token tensor so tests run in <1s on CPU.
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def tiny_image_batch() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(2, 3, 64, 128)


@pytest.fixture
def tiny_vit_patch() -> torch.Tensor:
    """(B=2, C=768, H/16=4, W/16=8) — matches tiny_image_batch at stride 16."""
    torch.manual_seed(1)
    return torch.randn(2, 768, 4, 8)


@pytest.fixture
def tiny_gt_panoptic() -> list[dict]:
    """List of 2 gt dicts with (labels, masks) as Mask2Former expects.

    Each sample: labels Long (N,), masks Bool (N, H, W).
    """
    torch.manual_seed(2)
    gts = []
    for _ in range(2):
        N = 3
        labels = torch.tensor([0, 5, 10], dtype=torch.long)
        masks = torch.zeros(N, 64, 128, dtype=torch.bool)
        for i in range(N):
            masks[i, 10 * i : 10 * i + 15, 10 * i : 10 * i + 20] = True
        gts.append({"labels": labels, "masks": masks})
    return gts
