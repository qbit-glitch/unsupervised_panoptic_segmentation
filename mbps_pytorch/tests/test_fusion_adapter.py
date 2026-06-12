"""Tests for the cross-model fusion adapter, pair sampler, and teacher loss."""
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mbps_pytorch.models.semantic.cross_model_adapter import (
    CrossModelAdapter,
    sample_stratified_pairs,
    teacher_guided_correlation_loss,
)


def test_adapter_identity_at_init() -> None:
    """Zero-init output layer -> adapter starts as exact identity (both sides)."""
    for code_dim, cond_dim in ((90, 100), (100, 90)):
        adapter = CrossModelAdapter(code_dim=code_dim, cond_dim=cond_dim,
                                    proj_width=16, hidden_dim=384, num_layers=2)
        codes = torch.randn(2, 50, code_dim)
        cond = torch.randn(2, 50, cond_dim)
        out = adapter(codes, cond)
        assert out.shape == codes.shape
        assert torch.allclose(out, codes), "adapter must start as identity"


def test_sampler_offset_constraints() -> None:
    h, w = 23, 46
    gen = torch.Generator().manual_seed(0)
    (i_s, j_s), (i_l, j_l) = sample_stratified_pairs(
        h, w, n_short=512, n_long=512, r_short=4, r_long=8,
        device=torch.device("cpu"), generator=gen,
    )
    assert i_s.shape == (512,) and i_l.shape == (512,)
    ri, ci = i_s // w, i_s % w
    rj, cj = j_s // w, j_s % w
    linf_s = torch.max((ri - rj).abs(), (ci - cj).abs())
    assert (linf_s >= 1).all() and (linf_s <= 4).all(), "short pairs must have L_inf in [1, 4]"
    ri, ci = i_l // w, i_l % w
    rj, cj = j_l // w, j_l % w
    linf_l = torch.max((ri - rj).abs(), (ci - cj).abs())
    assert (linf_l >= 8).all(), "long pairs must have L_inf >= 8"
    for idx in (i_s, j_s, i_l, j_l):
        assert (idx >= 0).all() and (idx < h * w).all(), "flat indices in range"


def test_teacher_loss_zero_cases() -> None:
    n, ds, dt = 64, 90, 100
    gen = torch.Generator().manual_seed(1)
    idx_i = torch.randint(0, n, (32,), generator=gen)
    idx_j = torch.randint(0, n, (32,), generator=gen)
    # Case 1: zero teacher -> cosine weights are 0/eps-stable -> finite loss
    teacher = torch.randn(n, dt, generator=gen)
    student = torch.randn(n, ds, generator=gen)
    loss = teacher_guided_correlation_loss(student, torch.zeros(n, dt), idx_i, idx_j)
    assert torch.isfinite(loss)
    # Case 2: student pairs identical (cos=1) -> (1-cos)^2 = 0 regardless of teacher
    same_student = torch.ones(n, ds)
    loss2 = teacher_guided_correlation_loss(same_student, teacher, idx_i, idx_j)
    assert loss2.abs().item() < 1e-6


def test_teacher_loss_pulls_weighted_pairs() -> None:
    """High-teacher-similarity pairs with dissimilar student codes -> positive loss."""
    n = 16
    teacher = torch.ones(n, 100)                      # all pairs: teacher cos = 1
    student = torch.randn(n, 90)
    idx_i = torch.arange(8)
    idx_j = torch.arange(8, 16)
    loss = teacher_guided_correlation_loss(student, teacher, idx_i, idx_j)
    assert loss.item() > 0.0


def test_teacher_loss_no_grad_through_teacher() -> None:
    """Teacher weights must be detached — gradients flow only through the student."""
    n = 16
    teacher = torch.randn(n, 100, requires_grad=True)
    student = torch.randn(n, 90, requires_grad=True)
    idx_i = torch.arange(8)
    idx_j = torch.arange(8, 16)
    loss = teacher_guided_correlation_loss(student, teacher, idx_i, idx_j)
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None or teacher.grad.abs().sum() == 0
