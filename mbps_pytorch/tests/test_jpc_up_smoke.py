"""Smoke tests for the JPC-Up panoptic pseudo-label module."""

from __future__ import annotations

import torch

from mbps_pytorch.losses.jpc_up_loss import JPCUpLossConfig, compute_jpc_up_losses
from mbps_pytorch.models.panoptic.jpc_up import JPCUpConfig, JointPanopticCouplerUp


def _tiny_config() -> JPCUpConfig:
    return JPCUpConfig(
        appearance_dim=16,
        semantic_dim=10,
        fusion_dim=32,
        num_prototypes=7,
        num_queries=5,
        query_dim=32,
        instance_embed_dim=8,
        low_hw=(4, 8),
        mid_hw=(8, 16),
        high_hw=(16, 32),
        num_coupling_blocks=1,
        num_heads=4,
        up_guidance_ch=7,
    )


def test_jpc_up_forward_shapes_cpu():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = JointPanopticCouplerUp(cfg)
    appearance = torch.randn(2, cfg.appearance_dim, *cfg.low_hw)
    semantic_codes = torch.randn(2, cfg.semantic_dim, *cfg.low_hw)
    depth = torch.rand(2, 1, 32, 64)
    rgb = torch.rand(2, 3, 32, 64)

    out = model(appearance, semantic_codes, depth, rgb)

    assert out.refined_codes.shape == (2, cfg.semantic_dim, *cfg.high_hw)
    assert out.semantic_logits.shape == (2, cfg.num_prototypes, *cfg.high_hw)
    assert out.objectness_logits.shape == (2, 1, *cfg.high_hw)
    assert out.center_logits.shape == (2, 1, *cfg.high_hw)
    assert out.boundary_logits.shape == (2, 1, *cfg.high_hw)
    assert out.instance_embeddings.shape == (2, cfg.instance_embed_dim, *cfg.high_hw)
    assert out.mask_logits.shape == (2, cfg.num_queries, *cfg.high_hw)
    assert out.query_scores.shape == (2, cfg.num_queries)
    assert out.query_embeddings.shape == (2, cfg.num_queries, cfg.instance_embed_dim)
    assert out.mid_codes.shape == (2, cfg.semantic_dim, *cfg.mid_hw)
    assert set(out.depth_maps) == {"low", "mid", "high"}


def test_jpc_up_losses_run_and_backprop():
    torch.manual_seed(1)
    cfg = _tiny_config()
    model = JointPanopticCouplerUp(cfg)
    appearance = torch.randn(1, cfg.appearance_dim, *cfg.low_hw)
    semantic_codes = torch.randn(1, cfg.semantic_dim, *cfg.low_hw)
    depth = torch.rand(1, 1, 32, 64)
    rgb = torch.rand(1, 3, 32, 64)
    out = model(appearance, semantic_codes, depth, rgb)

    semantic_target = torch.randint(0, cfg.num_prototypes, (1, *cfg.high_hw))
    proposal_masks = torch.zeros(1, 3, *cfg.high_hw)
    proposal_masks[:, 0, 2:10, 4:14] = 1.0
    proposal_masks[:, 1, 5:15, 16:28] = 1.0
    proposal_masks[:, 2, 0:4, 0:5] = 1.0
    proposal_scores = torch.tensor([[0.9, 0.7, 0.2]])
    boundary_target = torch.rand(1, 1, *cfg.high_hw)
    center_target = torch.rand(1, 1, *cfg.high_hw)

    losses = compute_jpc_up_losses(
        out,
        semantic_target=semantic_target,
        teacher_codes=semantic_codes,
        proposal_masks=proposal_masks,
        proposal_scores=proposal_scores,
        center_target=center_target,
        boundary_target=boundary_target,
        config=JPCUpLossConfig(min_proposal_score=0.1),
    )

    assert "loss" in losses
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_jpc_up_losses_allow_missing_evidence():
    torch.manual_seed(2)
    cfg = _tiny_config()
    model = JointPanopticCouplerUp(cfg)
    out = model(
        torch.randn(1, cfg.appearance_dim, *cfg.low_hw),
        torch.randn(1, cfg.semantic_dim, *cfg.low_hw),
        torch.rand(1, 1, 32, 64),
        None,
    )

    losses = compute_jpc_up_losses(out)
    assert torch.isfinite(losses["loss"])
    assert float(losses["L_semantic"].detach()) == 0.0
    assert float(losses["L_objectness"].detach()) == 0.0
    assert float(losses["L_proposal_mask"].detach()) == 0.0
