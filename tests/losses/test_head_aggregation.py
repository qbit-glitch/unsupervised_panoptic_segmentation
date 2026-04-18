"""Tests for ``CustomSemSegFPNHead.losses`` aux-loss aggregation hook."""
from __future__ import annotations

import torch

from tests.losses.head_fixture import make_head_with_aux


def _make_inputs(num_classes: int = 19, B: int = 2, H: int = 128, W: int = 256):
    """Return (logits, targets) at pre-upsample / full resolution.

    ``logits`` is at stride=common_stride (4x smaller than targets); the
    head internally upsamples before computing CE. Targets contain ignore
    pixels to exercise the ignore-index branch.
    """
    C = num_classes
    logits = torch.randn(B, C, H // 4, W // 4, requires_grad=True)
    targets = torch.randint(0, C, (B, H, W), dtype=torch.long)
    # Sprinkle ignore pixels so the ignore-index branch executes.
    targets[:, 0, 0] = 255
    return logits, targets


def test_aggregates_all_aux_weights_zero():
    head = make_head_with_aux(num_classes=19, weights=None)
    logits, targets = _make_inputs(num_classes=19)
    out = head.losses(logits, targets, ctx={})
    assert set(out.keys()) == {"loss_sem_seg"}
    assert out["loss_sem_seg"].dim() == 0


def test_passes_ctx_through_untouched():
    """With all aux weights zero, ctx must not raise even if empty/None."""
    head = make_head_with_aux(num_classes=19, weights=None)
    logits, targets = _make_inputs(num_classes=19)
    out_none = head.losses(logits, targets, ctx=None)
    out_empty = head.losses(logits, targets, ctx={})
    assert set(out_none.keys()) == {"loss_sem_seg"}
    assert set(out_empty.keys()) == {"loss_sem_seg"}


def test_aggregates_when_lovasz_enabled():
    """Enabling Lovász adds ``loss_lovasz`` with gradient flow."""
    head = make_head_with_aux(num_classes=19, weights={"LOVASZ_WEIGHT": 0.5})
    logits, targets = _make_inputs(num_classes=19)
    out = head.losses(logits, targets, ctx={})
    assert "loss_sem_seg" in out
    assert "loss_lovasz" in out
    assert out["loss_lovasz"].requires_grad
    assert torch.isfinite(out["loss_lovasz"])


def test_ignore_index_does_not_blow_up_ce():
    """CE with 255 ignore values should produce a finite scalar loss."""
    head = make_head_with_aux(num_classes=19, ignore_value=255)
    logits, targets = _make_inputs(num_classes=19)
    out = head.losses(logits, targets, ctx={})
    assert torch.isfinite(out["loss_sem_seg"])


def test_gradient_flows_to_logits():
    head = make_head_with_aux(num_classes=19)
    logits, targets = _make_inputs(num_classes=19)
    out = head.losses(logits, targets, ctx={})
    out["loss_sem_seg"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
