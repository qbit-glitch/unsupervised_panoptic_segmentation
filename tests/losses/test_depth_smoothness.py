"""Unit tests for depth-guided logit regularizer (P3 aux-loss)."""
from __future__ import annotations

import torch

from cups.losses.depth_smoothness import depth_smoothness


def test_requires_depth_in_ctx() -> None:
    logits = torch.randn(1, 5, 16, 32, requires_grad=True)
    try:
        depth_smoothness(logits, None, {"params": {"depth_smooth_alpha": 10.0}})
    except KeyError as e:
        assert "depth" in str(e)
    else:
        raise AssertionError("depth_smoothness should raise KeyError when ctx lacks 'depth'")


def test_scalar_output_with_grad() -> None:
    B, C, H, W = 2, 5, 16, 32
    logits = torch.randn(B, C, H, W, requires_grad=True)
    depth = torch.rand(B, 1, H, W)
    ctx = {"depth": depth, "params": {"depth_smooth_alpha": 10.0}}

    loss = depth_smoothness(logits, None, ctx)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_zero_when_logits_are_flat() -> None:
    """If logits are spatially constant, ||grad logits|| == 0 -> loss == 0 regardless of depth."""
    logits = torch.zeros(1, 5, 16, 32, requires_grad=True)
    depth = torch.linspace(0, 1, 32).view(1, 1, 1, 32).expand(1, 1, 16, 32).contiguous()
    ctx = {"depth": depth, "params": {"depth_smooth_alpha": 10.0}}
    loss = depth_smoothness(logits, None, ctx)
    assert loss.item() < 1e-5


def test_penalises_logit_jumps_on_smooth_depth() -> None:
    """Flat depth + large logit step -> positive loss and gradient."""
    logits = torch.zeros(1, 5, 16, 32, requires_grad=True)
    with torch.no_grad():
        logits[0, 0, :, :16] = 10.0
    depth = torch.zeros(1, 1, 16, 32)
    ctx = {"depth": depth, "params": {"depth_smooth_alpha": 10.0}}

    loss = depth_smoothness(logits, None, ctx)

    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_depth_edges_suppress_penalty() -> None:
    """A logit jump co-located with a depth edge should cost less than the same jump on flat depth."""
    H, W = 16, 32
    base_logits = torch.zeros(1, 5, H, W)
    base_logits[0, 0, :, : W // 2] = 10.0

    depth_flat = torch.zeros(1, 1, H, W)
    depth_edge = torch.zeros(1, 1, H, W)
    depth_edge[:, :, :, : W // 2] = 0.0
    depth_edge[:, :, :, W // 2 :] = 1.0

    params = {"depth_smooth_alpha": 10.0}
    loss_flat = depth_smoothness(base_logits.clone().requires_grad_(True), None,
                                  {"depth": depth_flat, "params": params})
    loss_edge = depth_smoothness(base_logits.clone().requires_grad_(True), None,
                                  {"depth": depth_edge, "params": params})
    assert loss_edge.item() < loss_flat.item()


def test_accepts_3d_depth() -> None:
    """A (B, H, W) depth tensor should be reshaped to (B, 1, H, W) transparently."""
    logits = torch.randn(1, 5, 16, 32, requires_grad=True)
    depth_3d = torch.rand(1, 16, 32)
    ctx = {"depth": depth_3d, "params": {"depth_smooth_alpha": 10.0}}
    loss = depth_smoothness(logits, None, ctx)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_depth_resized_to_logit_resolution() -> None:
    """Mismatched depth resolution should be bilinearly resized."""
    logits = torch.randn(1, 5, 16, 32, requires_grad=True)
    depth = torch.rand(1, 1, 64, 128)
    ctx = {"depth": depth, "params": {"depth_smooth_alpha": 10.0}}
    loss = depth_smoothness(logits, None, ctx)
    loss.backward()
    assert torch.isfinite(loss)
