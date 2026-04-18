from __future__ import annotations

import torch

from cups.losses.query_consistency import query_consistency_loss


def test_query_consistency_zero_when_student_equals_teacher() -> None:
    q = torch.randn(2, 10, 32)
    dec_out = {"query_embeds": q}
    ctx = {"teacher_query_embeds": q.detach().clone(), "temperature": 0.1}
    loss = query_consistency_loss(dec_out, targets=[], ctx=ctx)
    assert loss.item() < 1e-4


def test_query_consistency_positive_when_different() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 10, 32, requires_grad=True)
    tq = torch.randn(2, 10, 32)
    dec_out = {"query_embeds": q}
    ctx = {"teacher_query_embeds": tq, "temperature": 0.1}
    loss = query_consistency_loss(dec_out, targets=[], ctx=ctx)
    loss.backward()
    assert loss.item() > 0.0
    assert q.grad is not None
