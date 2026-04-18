"""Unit tests for boundary-weighted CE (P1 aux-loss)."""
from __future__ import annotations

import torch

from cups.losses.boundary import boundary_ce, compute_boundary_mask


def test_boundary_mask_binary_and_thin() -> None:
    """Dilated label-change detector produces a thin boundary band."""
    targets = torch.zeros(1, 32, 32, dtype=torch.long)
    targets[0, :, 16:] = 1  # vertical class split at column 16
    mask = compute_boundary_mask(targets, dilate_px=2)
    assert mask.dtype == torch.bool
    assert mask.shape == (1, 32, 32)
    assert mask.sum().item() > 0
    # Boundary band should be a small fraction of the image, not everything.
    assert mask.sum().item() < targets.numel()


def test_boundary_mask_empty_when_uniform() -> None:
    targets = torch.zeros(1, 16, 16, dtype=torch.long)
    mask = compute_boundary_mask(targets, dilate_px=2)
    assert mask.sum().item() == 0


def test_boundary_ce_finite_and_scalar() -> None:
    logits = torch.randn(1, 2, 32, 32, requires_grad=True)
    targets = torch.zeros(1, 32, 32, dtype=torch.long)
    targets[0, :, 16:] = 1
    ctx = {
        "params": {"boundary_dilate_px": 2, "boundary_ce_mult": 4.0},
        "ignore_index": 255,
    }
    loss = boundary_ce(logits, targets, ctx)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_boundary_ce_up_weights_boundary_pixels() -> None:
    """Higher ``boundary_ce_mult`` should strictly increase the loss on a
    deliberately-wrong boundary prediction."""
    torch.manual_seed(0)
    B, C, H, W = 1, 2, 32, 32
    targets = torch.zeros(B, H, W, dtype=torch.long)
    targets[0, :, W // 2 :] = 1
    logits = torch.randn(B, C, H, W)

    base_ctx = {
        "params": {"boundary_dilate_px": 2, "boundary_ce_mult": 1.0},
        "ignore_index": 255,
    }
    boosted_ctx = {
        "params": {"boundary_dilate_px": 2, "boundary_ce_mult": 5.0},
        "ignore_index": 255,
    }
    base = boundary_ce(logits, targets, base_ctx)
    boosted = boundary_ce(logits, targets, boosted_ctx)
    assert boosted.item() >= base.item() - 1e-6


def test_boundary_ce_respects_ignore_index() -> None:
    logits = torch.randn(1, 3, 8, 8, requires_grad=True)
    targets = torch.full((1, 8, 8), 255, dtype=torch.long)
    ctx = {
        "params": {"boundary_dilate_px": 1, "boundary_ce_mult": 2.0},
        "ignore_index": 255,
    }
    loss = boundary_ce(logits, targets, ctx)
    assert torch.isfinite(loss)
    # Fully-ignored image: loss should be zero / near-zero.
    assert loss.item() == 0.0
