"""Unit tests for the STEGO correspondence shim (P2 aux-loss)."""
from __future__ import annotations

import pytest
import torch

from cups.losses.stego_adapter import stego_corr


def test_requires_dino_features_in_ctx() -> None:
    logits = torch.randn(2, 5, 16, 32)
    targets = torch.randint(0, 5, (2, 16, 32))
    ctx = {"params": {"stego_temperature": 0.1, "stego_knn_k": 4}}
    with pytest.raises(KeyError, match="dino_features"):
        stego_corr(logits, targets, ctx)


def test_returns_scalar_with_grad() -> None:
    B = 2
    Hs = Ws = 8
    D = 32
    logits = torch.randn(B, 5, Hs * 4, Ws * 4, requires_grad=True)
    dino = torch.randn(B, Hs * Ws, D)
    ctx = {
        "dino_features": dino,
        "params": {
            "stego_temperature": 0.1,
            "stego_knn_k": 4,
            "stego_feature_source": "fpn_p2",
        },
    }
    loss = stego_corr(logits, targets=None, ctx=ctx)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_accepts_channel_first_dino_features() -> None:
    """``(B, D, Hs, Ws)`` patch tensors should be handled transparently."""
    B = 1
    Hs = Ws = 8
    D = 16
    logits = torch.randn(B, 5, Hs * 4, Ws * 4, requires_grad=True)
    dino = torch.randn(B, D, Hs, Ws)
    ctx = {
        "dino_features": dino,
        "params": {
            "stego_temperature": 0.1,
            "stego_knn_k": 4,
            "stego_feature_source": "fpn_p2",
        },
    }
    loss = stego_corr(logits, targets=None, ctx=ctx)
    assert loss.dim() == 0
    loss.backward()
    assert logits.grad is not None
