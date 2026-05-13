"""Unit tests for Lovász-Softmax loss (P1 aux-loss)."""
from __future__ import annotations

import torch

from cups.losses.lovasz import lovasz_softmax


def test_perfect_pred_zero_loss() -> None:
    """Perfect logits -> perfect softmax -> near-zero Lovász loss."""
    B, C, H, W = 2, 5, 16, 32
    targets = torch.randint(0, C, (B, H, W))
    logits = torch.full((B, C, H, W), -10.0)
    for b in range(B):
        flat = targets[b].flatten()
        for i, v in enumerate(flat.tolist()):
            logits[b, int(v), i // W, i % W] = 10.0
    loss = lovasz_softmax(logits, targets, {"ignore_index": 255})
    assert loss.item() < 1e-3


def test_gradient_flows() -> None:
    logits = torch.randn(2, 5, 16, 32, requires_grad=True)
    targets = torch.randint(0, 5, (2, 16, 32))
    loss = lovasz_softmax(logits, targets, {"ignore_index": 255})
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_scalar_output() -> None:
    logits = torch.randn(1, 3, 8, 16)
    targets = torch.randint(0, 3, (1, 8, 16))
    loss = lovasz_softmax(logits, targets, {"ignore_index": 255})
    assert loss.dim() == 0


def test_ignore_index_honored() -> None:
    """All-ignore target should produce a zero loss with grad preserved."""
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    targets = torch.full((1, 4, 4), 255, dtype=torch.long)
    loss = lovasz_softmax(logits, targets, {"ignore_index": 255})
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
