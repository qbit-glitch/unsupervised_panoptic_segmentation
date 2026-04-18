from __future__ import annotations

import torch

from cups.losses.xquery import xquery_loss


def test_xquery_is_zero_when_batch_size_one() -> None:
    q = torch.randn(1, 10, 32)
    dec_out = {"query_embeds": q}
    loss = xquery_loss(dec_out, targets=[], ctx={})
    assert loss.item() == 0.0


def test_xquery_positive_for_batch_size_two() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 10, 32, requires_grad=True)
    dec_out = {"query_embeds": q}
    loss = xquery_loss(dec_out, targets=[], ctx={"temperature": 0.1})
    loss.backward()
    assert loss.item() > 0.0
    assert q.grad is not None
