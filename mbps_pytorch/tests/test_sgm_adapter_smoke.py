"""Smoke tests for the depth-aware SGM adapter pipeline.

Exercises the four new modules end-to-end on a tiny synthetic image:

* :mod:`mbps_pytorch.instance_methods.depth_aware_slic`
* :mod:`mbps_pytorch.models.instance.sgm_adapter`
* :mod:`mbps_pytorch.losses.superpixel_sgm`

Uses CPU, no real Cityscapes data. Designed to run in well under a minute.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("skimage", reason="Depth-aware SLIC smoke test requires scikit-image")

from mbps_pytorch.instance_methods.depth_aware_slic import (
    DepthAwareSlicConfig,
    compute_depth_aware_slic,
    superpixel_count,
)
from mbps_pytorch.losses.superpixel_sgm import (
    SGMLossConfig,
    compute_sgm_losses,
    depth_aware_edge_weights,
    mst_soft_labels,
    sp_labels_from_coarse,
    superpixel_adjacency,
    superpixel_foreground_prob,
)
from mbps_pytorch.models.instance.sgm_adapter import SGMAdapter, SGMAdapterConfig


# Tiny resolution to keep the smoke test fast: 32x64 input, 4x8 patch grid.
_H, _W = 32, 64
_HP, _WP = 4, 8
_N_THING = 3
_DINO_D = 16


def _make_synthetic_inputs(rng: np.random.Generator):
    """Synthetic image + depth + DINO features + coarse mask for one image."""
    image = (rng.integers(0, 256, size=(_H, _W, 3), dtype=np.int32)).astype(np.uint8)
    depth = rng.uniform(0.0, 1.0, size=(_H, _W)).astype(np.float32)
    dino_patch = rng.standard_normal(size=(_HP, _WP, _DINO_D)).astype(np.float32)

    coarse = np.zeros((_N_THING, _H, _W), dtype=np.uint8)
    # Coarse regions must be substantially larger than one SLIC superpixel
    # (n_segments=24 over 32x64 -> ~85 px/SP) so that at least one SP can be
    # fully-inside, giving positive supervision after sp_labels_from_coarse.
    coarse[0, 2:28, 4:50] = 1   # class 0 has an object (~1200 px)
    coarse[1, 4:30, 14:60] = 1  # class 1 has an object (~1200 px)
    # class 2 left intentionally empty
    return image, depth, dino_patch, coarse


def test_depth_aware_slic_runs_and_returns_int_map():
    rng = np.random.default_rng(0)
    image, depth, dino_patch, _ = _make_synthetic_inputs(rng)
    config = DepthAwareSlicConfig(n_segments=24, dino_pca_dim=4, pca_subsample_stride=1)
    labels = compute_depth_aware_slic(image, depth, dino_patch, config)
    assert labels.shape == (_H, _W)
    assert labels.dtype == np.int32
    k = superpixel_count(labels)
    assert k >= 4, f"expected at least 4 SPs, got {k}"


def test_sgm_adapter_forward_shapes():
    config = SGMAdapterConfig(
        d_dino=_DINO_D, d_depth=16, d_fusion=32, n_fusion_layers=1,
        n_heads=2, n_thing=_N_THING, patch_h=_HP, patch_w=_WP,
        use_concat_fusion=False,
    )
    model = SGMAdapter(config)
    f_dino = torch.randn(1, _HP * _WP, _DINO_D)
    depth_patch = torch.rand(1, _HP * _WP)
    out = model(f_dino, depth_patch, out_hw=(_H, _W))
    assert out.shape == (1, _N_THING, _H, _W)
    assert torch.all((out >= 0) & (out <= 1)), "sigmoid out of range"


def test_sgm_adapter_concat_fusion_shapes():
    config = SGMAdapterConfig(
        d_dino=_DINO_D, d_depth=16, d_fusion=32, n_fusion_layers=1,
        n_heads=2, n_thing=_N_THING, patch_h=_HP, patch_w=_WP,
        use_concat_fusion=True,
    )
    model = SGMAdapter(config)
    f_dino = torch.randn(1, _HP * _WP, _DINO_D)
    depth_patch = torch.rand(1, _HP * _WP)
    out = model(f_dino, depth_patch, out_hw=(_H, _W))
    assert out.shape == (1, _N_THING, _H, _W)


def test_sp_labels_from_coarse_three_states():
    sp = torch.zeros(4, 4, dtype=torch.long)
    sp[:, 2:] = 1
    coarse = torch.zeros(4, 4)
    coarse[:, 0:2] = 1   # all-fg in SP 0
    # SP 1 stays all-bg
    y = sp_labels_from_coarse(coarse, sp, k=2)
    assert y.tolist() == [1, 0]

    # Mixed case for SP 0
    coarse_mixed = torch.zeros(4, 4)
    coarse_mixed[0:2, 0] = 1
    y2 = sp_labels_from_coarse(coarse_mixed, sp, k=2)
    assert y2[0].item() == -1
    assert y2[1].item() == 0


def test_superpixel_foreground_prob_is_in_unit_interval():
    sp = torch.zeros(8, 8, dtype=torch.long)
    sp[:, 4:] = 1
    m_tilde = torch.rand(8, 8)
    delta = torch.ones(8, 8)
    p = superpixel_foreground_prob(m_tilde, delta, sp, k=2)
    assert p.shape == (2,)
    assert torch.all((p >= 0) & (p <= 1))


def test_mst_soft_labels_matches_source_on_disconnected_graph():
    """When edges is empty, soft labels collapse to the input P."""
    P = torch.tensor([0.1, 0.7, 0.3])
    edges = np.zeros((0, 2), dtype=np.int64)
    weights = np.zeros(0, dtype=np.float64)
    hat = mst_soft_labels(P, edges, weights, SGMLossConfig())
    assert torch.allclose(hat, P)


def test_full_loss_runs_and_backprops():
    rng = np.random.default_rng(1)
    image_np, depth_np, dino_patch_np, coarse_np = _make_synthetic_inputs(rng)

    slic_config = DepthAwareSlicConfig(n_segments=24, dino_pca_dim=4, pca_subsample_stride=1)
    sp_labels_np = compute_depth_aware_slic(image_np, depth_np, dino_patch_np, slic_config)

    # Tensors
    image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    depth = torch.from_numpy(depth_np)
    sp_labels = torch.from_numpy(sp_labels_np.astype(np.int64))
    coarse = torch.from_numpy(coarse_np).float()

    dino_full = torch.from_numpy(dino_patch_np).permute(2, 0, 1).unsqueeze(0)
    dino_full = torch.nn.functional.interpolate(
        dino_full, size=(_H, _W), mode="bilinear", align_corners=False
    ).squeeze(0)

    # Trainable per-pixel prediction (skips the adapter; tests losses alone)
    m_tilde = torch.full((_N_THING, _H, _W), 0.5, requires_grad=True)
    losses = compute_sgm_losses(
        m_tilde=m_tilde,
        coarse_mask=coarse,
        image=image,
        depth=depth,
        dino_full=dino_full,
        sp_labels=sp_labels,
        checkpoints=[],
        config=SGMLossConfig(),
    )
    loss = losses["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert m_tilde.grad is not None
    assert torch.isfinite(m_tilde.grad).all()
    assert m_tilde.grad.abs().sum().item() > 0


def test_adaptive_self_training_zero_with_no_checkpoints():
    rng = np.random.default_rng(2)
    image_np, depth_np, dino_patch_np, coarse_np = _make_synthetic_inputs(rng)
    sp_labels_np = compute_depth_aware_slic(
        image_np, depth_np, dino_patch_np,
        DepthAwareSlicConfig(n_segments=16, dino_pca_dim=4, pca_subsample_stride=1),
    )
    image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    depth = torch.from_numpy(depth_np)
    sp_labels = torch.from_numpy(sp_labels_np.astype(np.int64))
    coarse = torch.from_numpy(coarse_np).float()
    dino_full = torch.from_numpy(dino_patch_np).permute(2, 0, 1).unsqueeze(0)
    dino_full = torch.nn.functional.interpolate(
        dino_full, size=(_H, _W), mode="bilinear", align_corners=False
    ).squeeze(0)
    m_tilde = torch.full((_N_THING, _H, _W), 0.7, requires_grad=True)

    losses_empty = compute_sgm_losses(
        m_tilde=m_tilde, coarse_mask=coarse, image=image, depth=depth,
        dino_full=dino_full, sp_labels=sp_labels, checkpoints=[],
        config=SGMLossConfig(),
    )
    assert float(losses_empty["L_ad"]) == 0.0

    losses_ckpt = compute_sgm_losses(
        m_tilde=m_tilde, coarse_mask=coarse, image=image, depth=depth,
        dino_full=dino_full, sp_labels=sp_labels,
        checkpoints=[torch.full((_N_THING, _H, _W), 0.8)],
        config=SGMLossConfig(),
    )
    # Both predictions above threshold -> IoU = 1, weight = 1, |0.7 - 0.8| = 0.1
    assert float(losses_ckpt["L_ad"]) > 0.0


def test_edge_weights_include_depth_and_feature_terms():
    """Constructed graph where colour-only weights would lose depth ordering."""
    mu_c = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    bar_d = torch.tensor([0.0, 1.0, 0.0])      # node 0 and 2 same depth, 1 far
    bar_f = torch.zeros(3, 4)
    edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
    cfg = SGMLossConfig(lambda_d=10.0, lambda_f=0.0)
    w = depth_aware_edge_weights(edges, mu_c, bar_d, bar_f, cfg)
    # edge (0, 2) must be smaller than edges to node 1
    assert w[2] < w[0]
    assert w[2] < w[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
