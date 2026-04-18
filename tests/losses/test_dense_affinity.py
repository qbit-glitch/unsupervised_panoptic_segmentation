"""Unit tests for dense-affinity aux losses (P4): Gated-CRF and NeCo."""
from __future__ import annotations

import torch

from cups.losses.dense_affinity import gated_crf, neco


# -----------------------------------------------------------------------------
# Gated-CRF
# -----------------------------------------------------------------------------


def test_gated_crf_requires_rgb() -> None:
    logits = torch.randn(1, 5, 16, 32, requires_grad=True)
    ctx = {"params": {"gated_crf_kernel": 3, "gated_crf_rgb_sigma": 0.1}}
    try:
        gated_crf(logits, None, ctx)
    except KeyError as e:
        assert "rgb" in str(e)
    else:
        raise AssertionError("gated_crf should raise KeyError when ctx lacks 'rgb'")


def test_gated_crf_scalar_with_grad() -> None:
    B, C, H, W = 2, 5, 16, 32
    logits = torch.randn(B, C, H, W, requires_grad=True)
    rgb = torch.randn(B, 3, H, W)
    ctx = {"rgb": rgb, "params": {"gated_crf_kernel": 3, "gated_crf_rgb_sigma": 0.1}}

    loss = gated_crf(logits, None, ctx)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_gated_crf_resizes_rgb() -> None:
    logits = torch.randn(1, 5, 16, 32, requires_grad=True)
    rgb = torch.randn(1, 3, 64, 128)
    ctx = {"rgb": rgb, "params": {"gated_crf_kernel": 3, "gated_crf_rgb_sigma": 0.1}}
    loss = gated_crf(logits, None, ctx)
    loss.backward()
    assert torch.isfinite(loss)


def test_gated_crf_smaller_on_smooth_predictions() -> None:
    """Pixel-wise identical predictions should minimise the pairwise term."""
    H, W = 16, 32
    rgb = torch.randn(1, 3, H, W)
    uniform = torch.full((1, 5, H, W), -5.0)
    uniform[:, 0, :, :] = 5.0  # all pixels confidently class 0
    noisy = torch.randn(1, 5, H, W) * 3.0

    ctx = {"rgb": rgb, "params": {"gated_crf_kernel": 3, "gated_crf_rgb_sigma": 0.1}}
    loss_uniform = gated_crf(uniform, None, ctx)
    loss_noisy = gated_crf(noisy, None, ctx)
    assert loss_uniform.item() < loss_noisy.item()


# -----------------------------------------------------------------------------
# NeCo
# -----------------------------------------------------------------------------


def test_neco_requires_dino_features() -> None:
    logits = torch.randn(1, 5, 16, 32, requires_grad=True)
    ctx = {"params": {"neco_k": 3}}
    try:
        neco(logits, None, ctx)
    except KeyError as e:
        assert "dino_features" in str(e)
    else:
        raise AssertionError("neco should raise KeyError when ctx lacks 'dino_features'")


def test_neco_scalar_with_grad() -> None:
    B, C = 2, 5
    Hs = Ws = 8
    D = 16
    logits = torch.randn(B, C, Hs * 4, Ws * 4, requires_grad=True)
    dino = torch.randn(B, Hs * Ws, D)
    ctx = {"dino_features": dino, "params": {"neco_k": 3}}

    loss = neco(logits, None, ctx)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_neco_accepts_channel_first_dino() -> None:
    B, D, Hs, Ws = 1, 16, 8, 8
    logits = torch.randn(B, 5, Hs * 4, Ws * 4, requires_grad=True)
    dino = torch.randn(B, D, Hs, Ws)
    ctx = {"dino_features": dino, "params": {"neco_k": 3}}
    loss = neco(logits, None, ctx)
    loss.backward()
    assert torch.isfinite(loss)


def test_neco_prefers_structurally_aligned_logits() -> None:
    """Logits that mirror DINO neighbourhood structure should cost less."""
    torch.manual_seed(0)
    B = 1
    Hs = Ws = 8
    D = 16
    C = 5

    dino = torch.randn(B, Hs * Ws, D)
    params = {"neco_k": 3}

    # Logits aligned with DINO: project DINO features linearly into C channels.
    proj = torch.randn(D, C)
    aligned = dino @ proj  # (B, N, C)
    aligned_img = aligned.transpose(1, 2).reshape(B, C, Hs, Ws)
    aligned_img = aligned_img.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)

    random = torch.randn(B, C, Hs * 4, Ws * 4)

    ctx = {"dino_features": dino, "params": params}
    loss_aligned = neco(aligned_img, None, ctx)
    loss_random = neco(random, None, ctx)
    assert loss_aligned.item() < loss_random.item()
