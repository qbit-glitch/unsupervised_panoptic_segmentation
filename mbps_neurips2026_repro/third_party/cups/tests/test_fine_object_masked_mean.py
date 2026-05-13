"""Regression tests for FineObjectSemanticLoss helper sync.

Covers:
  - _batch_masked_mean returns (logits, valid) so callers can sync parallel
    tensors (cls_labels, iou_weights) with the empty-mask filter.
  - The forward path filters cls_labels/iou_weights with that mask, preventing
    silent class-mispairing when an IoU-passing mask collapses to zero pixels
    after nearest-neighbour resize to logit resolution.
  - Teacher-gated thing_mc_panda assertion holds when student/teacher masks
    are identical.

See plan: docs/plans/2026-05-05_stage4_sam3_alignfix_followups.md
"""
import os
import sys
from typing import List, Optional

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cups.losses.fine_object import _batch_masked_mean, FineObjectSemanticLoss


def test_batch_masked_mean_returns_tuple_of_logits_and_valid():
    logits = torch.randn(4, 8, 8)
    masks = torch.zeros(3, 8, 8, dtype=torch.bool)
    masks[0, 0:4, 0:4] = True
    # masks[1] intentionally all-zero
    masks[2, 4:8, 4:8] = True

    result = _batch_masked_mean(logits, masks)
    assert isinstance(result, tuple) and len(result) == 2
    mean_logits, valid = result
    assert mean_logits.shape == (2, 4)
    assert valid.dtype == torch.bool
    assert valid.tolist() == [True, False, True]


def test_batch_masked_mean_empty_input():
    """Edge case: zero masks. Both outputs are zero-length."""
    logits = torch.randn(4, 8, 8)
    masks = torch.zeros(0, 8, 8, dtype=torch.bool)
    mean_logits, valid = _batch_masked_mean(logits, masks)
    assert mean_logits.shape == (0, 4)
    assert valid.shape == (0,)
    assert valid.dtype == torch.bool


def test_forward_drops_cls_labels_for_collapsed_masks():
    """Silent-mispairing regression test.

    Construct a batch where one IoU-passing mask collapses to zero pixels after
    the nearest-neighbour resize from (80,160) to (20,40). Use cls_labels =
    [stuff, thing] with the stuff mask collapsing — pre-fix the surviving thing
    logits would be paired with cls_labels[0]=stuff_idx and routed to entropy
    (no focal call); post-fix cls_labels is filtered to [thing_idx] and focal CE
    fires once. Counter on _focal_ce makes the difference observable.
    """
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 20, 40)
    masks = torch.zeros(2, 80, 160, dtype=torch.bool)
    masks[0, 0, 0] = True            # collapses (1 px → 0 after resize stride 4)
    masks[1, 0:32, 0:32] = True      # survives
    ious = torch.tensor([0.9, 0.9])
    cls_labels = torch.tensor([4, 0], dtype=torch.long)  # [traffic-sign (stuff), person (thing)]

    seen_focal_rows = []
    import cups.losses.fine_object as fom
    real_focal = fom._focal_ce

    def counting_focal(logits_, targets, gamma, weights, ignore_index=255):
        seen_focal_rows.append(targets.shape[0])
        return real_focal(logits_, targets, gamma, weights, ignore_index)

    fom._focal_ce = counting_focal
    try:
        loss_fn = FineObjectSemanticLoss(mode="thing_focal_only", min_hard_iou=0.0)
        out = loss_fn(
            logits=logits,
            sam_masks_list=[masks],
            sam_iou_list=[ious],
            sam_class_labels_list=[cls_labels],
            min_iou_score=0.0,
        )
    finally:
        fom._focal_ce = real_focal

    assert torch.isfinite(out)
    # Post-fix invariant: the surviving mask is the thing one (cls=0=person).
    # Filtered cls_labels = [0] → routed to focal CE with one row.
    # Pre-fix: surviving row paired with cls_labels[0]=4 (stuff) → entropy
    # path → focal_ce never called → seen_focal_rows == [].
    assert seen_focal_rows == [1], (
        f"Expected one focal CE call with one row; got {seen_focal_rows}. "
        "Pre-fix pre-condition is [] (silent mispairing routes survivor to entropy)."
    )


def test_forward_thing_mc_panda_teacher_valid_match():
    """Defensive assert holds: identical input masks → identical valid tensors."""
    torch.manual_seed(1)
    logits = torch.randn(1, 5, 20, 40)
    teacher_logits = torch.randn(1, 5, 20, 40)
    masks = torch.zeros(2, 80, 160, dtype=torch.bool)
    masks[0, 0:32, 0:32] = True
    masks[1, 0, 0] = True  # collapses
    cls_labels = torch.tensor([0, 0], dtype=torch.long)

    loss_fn = FineObjectSemanticLoss(mode="thing_mc_panda", min_hard_iou=0.0)
    out = loss_fn(
        logits=logits,
        sam_masks_list=[masks],
        sam_iou_list=[torch.tensor([0.9, 0.9])],
        sam_class_labels_list=[cls_labels],
        teacher_logits=teacher_logits,
        min_iou_score=0.0,
    )
    assert torch.isfinite(out)


def test_path_b_focal_only_applies_teacher_gate_when_provided():
    """Path B: thing_focal_only routes through the agreement gate.

    Verifies the gate is APPLIED (the weights passed to `_focal_ce` are
    `iou * gate`, not just `iou`) by spying on `_focal_ce`. The gate's
    effect is on per-mask redistribution inside `_focal_ce`'s
    weighted-average normalization, not on overall magnitude — so we
    assert the weights pattern, not the loss value.
    """
    torch.manual_seed(9)
    C, H, W = 5, 16, 32
    student_logits = torch.randn(1, C, H, W)
    masks = torch.zeros(2, H, W, dtype=torch.bool)
    masks[0, 0:8, 0:16] = True
    masks[1, 8:16, 16:32] = True

    # Teacher gives different predictions per region: high P(thing) on mask 0
    # region (gate_0 ≈ 0), low on mask 1 region (gate_1 ≈ 1).
    teacher = torch.zeros(1, C, H, W)
    teacher[0, 0, 0:8, 0:16] = 10.0   # mask 0 region: P(thing)≈1
    teacher[0, 4, 8:16, 16:32] = 10.0  # mask 1 region: P(thing)≈0

    cls_labels = torch.tensor([0, 0], dtype=torch.long)

    captured_weights: List[Optional[torch.Tensor]] = []
    import cups.losses.fine_object as fom
    real_focal = fom._focal_ce

    def spy_focal(logits_, targets, gamma, weights, ignore_index=255):
        captured_weights.append(
            weights.detach().clone() if weights is not None else None
        )
        return real_focal(logits_, targets, gamma, weights, ignore_index)

    fom._focal_ce = spy_focal
    try:
        loss_fn = FineObjectSemanticLoss(
            mode="thing_focal_only",
            min_hard_iou=0.0,
            common_thing_channel_indices=[0],
            gate_focal_with_teacher=True,
        )
        loss_with_gate = loss_fn(
            logits=student_logits,
            teacher_logits=teacher,
            sam_masks_list=[masks],
            sam_iou_list=[torch.tensor([0.9, 0.9])],
            sam_class_labels_list=[cls_labels],
            min_iou_score=0.0,
        )
    finally:
        fom._focal_ce = real_focal

    # Gate applied → exactly one _focal_ce call with two-mask weights
    assert len(captured_weights) == 1
    w = captured_weights[0]
    assert w is not None and w.shape == (2,)

    # iou_weights would be [0.9, 0.9]. With gate, w = [0.9*g0, 0.9*g1].
    # g0 ≈ 0 (teacher agrees), g1 ≈ 1 (teacher disagrees).
    assert w[0].item() < 0.01, f"mask 0 should have near-zero weight; got {w[0].item():.4f}"
    assert w[1].item() > 0.5, f"mask 1 should have ~iou weight; got {w[1].item():.4f}"
    # Asymmetry: gated weights are NOT proportional to iou alone.
    assert (w[0].item() / w[1].item()) < 0.05, (
        f"Path B should redistribute heavily toward disagreement mask; "
        f"got ratio={w[0].item() / w[1].item():.4f}"
    )
    assert torch.isfinite(loss_with_gate)


def test_path_b_focal_only_disabled_gate_matches_unconditional_focal():
    """When gate_focal_with_teacher=False, focal_only ignores teacher and
    behaves like unconditional focal CE (backward-compat opt-out)."""
    torch.manual_seed(8)
    C, H, W = 5, 16, 32
    student_logits = torch.randn(1, C, H, W)
    teacher_logits = torch.randn(1, C, H, W) * 5.0  # arbitrary teacher
    masks = torch.zeros(1, H, W, dtype=torch.bool)
    masks[0, 4:12, 8:24] = True
    cls_labels = torch.tensor([0], dtype=torch.long)
    common_args = dict(
        logits=student_logits,
        sam_masks_list=[masks],
        sam_iou_list=[torch.tensor([0.9])],
        sam_class_labels_list=[cls_labels],
        min_iou_score=0.0,
    )

    loss_no_gate = FineObjectSemanticLoss(
        mode="thing_focal_only",
        min_hard_iou=0.0,
        gate_focal_with_teacher=False,
    )(teacher_logits=teacher_logits, **common_args)

    loss_no_teacher = FineObjectSemanticLoss(
        mode="thing_focal_only",
        min_hard_iou=0.0,
        gate_focal_with_teacher=True,  # default, but no teacher passed
    )(teacher_logits=None, **common_args)

    assert torch.isclose(loss_no_gate, loss_no_teacher, atol=1e-6), (
        f"Disabling gate should match the no-teacher path. "
        f"got {loss_no_gate.item():.6f} vs {loss_no_teacher.item():.6f}"
    )
