"""Verify the aux-loss registry exposes all six callables."""
from __future__ import annotations

from cups.losses import build_aux_losses


def test_registry_returns_callables() -> None:
    reg = build_aux_losses()
    expected = {
        "lovasz_softmax",
        "boundary_ce",
        "stego_corr",
        "depth_smoothness",
        "gated_crf",
        "neco",
    }
    assert set(reg.keys()) == expected
    for name, fn in reg.items():
        assert callable(fn), f"{name} is not callable"
