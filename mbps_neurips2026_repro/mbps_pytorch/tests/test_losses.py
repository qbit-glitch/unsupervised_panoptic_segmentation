"""Unit tests for all loss functions (PyTorch).

Tests cover:
    - Semantic loss (STEGO + DepthG)
    - Instance loss (Dice, BCE, unsupervised)
    - Bridge loss (reconstruction, CKA, state reg)
    - Consistency loss (uniform, boundary, DBC)
    - PQ proxy loss
    - Gradient magnitudes
    - No NaN/Inf values
"""

from __future__ import annotations

import sys
import os

import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestSemanticLoss:
    """Test semantic loss functions."""

    def test_semantic_loss_computes(self):
        """SemanticLoss should return dict with total key."""
        from mbps_pytorch.losses.semantic_loss import SemanticLoss

        torch.manual_seed(0)
        loss_fn = SemanticLoss(lambda_depthg=0.3)
        sem_codes = torch.randn(2, 50, 90, device=DEVICE)
        dino_feats = torch.randn(2, 50, 384, device=DEVICE)
        depth = torch.rand(2, 50, device=DEVICE)

        result = loss_fn(sem_codes, dino_feats, depth=depth)

        assert "total" in result
        assert "stego" in result
        assert "depthg" in result

    def test_semantic_loss_no_nan(self):
        """Semantic loss should not produce NaN."""
        from mbps_pytorch.losses.semantic_loss import SemanticLoss

        torch.manual_seed(42)
        loss_fn = SemanticLoss()
        sem_codes = torch.randn(1, 30, 90, device=DEVICE)
        dino_feats = torch.randn(1, 30, 384, device=DEVICE)

        result = loss_fn(sem_codes, dino_feats)
        assert not torch.isnan(result["total"])

    def test_semantic_loss_no_depth(self):
        """SemanticLoss should work without depth (depthg = 0)."""
        from mbps_pytorch.losses.semantic_loss import SemanticLoss

        torch.manual_seed(0)
        loss_fn = SemanticLoss()
        sem_codes = torch.randn(1, 20, 90, device=DEVICE)
        dino_feats = torch.randn(1, 20, 384, device=DEVICE)

        result = loss_fn(sem_codes, dino_feats, depth=None)
        assert float(result["depthg"]) == 0.0


class TestInstanceLoss:
    """Test instance loss functions."""

    def test_dice_loss_range(self):
        """Dice loss should be in [0, 1]."""
        from mbps_pytorch.losses.instance_loss import dice_loss

        pred = torch.zeros(1, 5, 100, device=DEVICE)
        target = torch.ones(1, 5, 100, device=DEVICE)
        loss = dice_loss(pred, target)

        assert float(loss) >= 0.0
        assert float(loss) <= 1.0

    def test_dice_loss_perfect(self):
        """Dice loss should be near 0 for perfect prediction."""
        from mbps_pytorch.losses.instance_loss import dice_loss

        pred = torch.ones(1, 5, 100, device=DEVICE) * 10.0  # sigmoid ~ 1
        target = torch.ones(1, 5, 100, device=DEVICE)
        loss = dice_loss(pred, target)

        assert float(loss) < 0.1

    def test_bce_loss_symmetric(self):
        """BCE with all-zeros pred and all-ones target should equal log(2)."""
        from mbps_pytorch.losses.instance_loss import mask_bce_loss

        pred = torch.zeros(1, 1, 100, device=DEVICE)
        target = torch.ones(1, 1, 100, device=DEVICE)
        loss = mask_bce_loss(pred, target)

        # sigmoid(0)=0.5, BCE = -log(0.5) = log(2) ~ 0.693
        assert abs(float(loss) - np.log(2)) < 0.01

    def test_instance_loss_class(self):
        """InstanceLoss should return dict with total."""
        from mbps_pytorch.losses.instance_loss import InstanceLoss

        torch.manual_seed(0)
        loss_fn = InstanceLoss()
        pred_masks = torch.randn(1, 10, 50, device=DEVICE)
        pred_scores = torch.ones(1, 10, device=DEVICE) * 0.8
        features = torch.randn(1, 50, 384, device=DEVICE)

        result = loss_fn(pred_masks, pred_scores, features)
        assert "total" in result

    def test_instance_loss_no_nan(self):
        """Instance loss should not produce NaN."""
        from mbps_pytorch.losses.instance_loss import InstanceLoss

        torch.manual_seed(0)
        loss_fn = InstanceLoss()
        pred_masks = torch.randn(1, 5, 30, device=DEVICE)
        pred_scores = torch.ones(1, 5, device=DEVICE) * 0.5
        features = torch.randn(1, 30, 64, device=DEVICE)

        result = loss_fn(pred_masks, pred_scores, features)
        assert not torch.isnan(result["total"])


class TestBridgeLoss:
    """Test bridge loss functions."""

    def test_reconstruction_loss(self):
        """Recon loss of identical tensors should be ~0."""
        from mbps_pytorch.losses.bridge_loss import reconstruction_loss

        x = torch.ones(1, 50, 90, device=DEVICE)
        loss = reconstruction_loss(x, x)

        assert abs(float(loss)) < 1e-5

    def test_cka_self_alignment(self):
        """CKA of feature with itself should give -1 (perfect)."""
        from mbps_pytorch.losses.bridge_loss import cka_loss

        torch.manual_seed(0)
        features = torch.randn(1, 50, 32, device=DEVICE)
        loss = cka_loss(features, features)

        assert abs(float(loss) - (-1.0)) < 0.01

    def test_cka_orthogonal(self):
        """CKA of orthogonal features should be near 0."""
        from mbps_pytorch.losses.bridge_loss import cka_loss

        torch.manual_seed(0)
        a = torch.randn(1, 100, 32, device=DEVICE)
        torch.manual_seed(99)
        b = torch.randn(1, 100, 32, device=DEVICE)

        loss = cka_loss(a, b)
        # Should be close to 0 (slightly negative)
        assert float(loss) > -0.5

    def test_state_reg_nonneg(self):
        """State regularization should be non-negative."""
        from mbps_pytorch.losses.bridge_loss import state_regularization_loss

        norms = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        loss = state_regularization_loss(norms)

        assert float(loss) > 0.0

    def test_bridge_loss_class(self):
        """BridgeLoss should return dict with total."""
        from mbps_pytorch.losses.bridge_loss import BridgeLoss

        torch.manual_seed(0)
        loss_fn = BridgeLoss()

        orig_sem = torch.randn(1, 50, 90, device=DEVICE)
        orig_feat = torch.randn(1, 50, 384, device=DEVICE)
        recon_sem = torch.randn(1, 50, 90, device=DEVICE)
        recon_feat = torch.randn(1, 50, 384, device=DEVICE)
        fused_sem = torch.randn(1, 50, 192, device=DEVICE)
        fused_feat = torch.randn(1, 50, 192, device=DEVICE)
        align_loss = torch.tensor(0.1, device=DEVICE)
        state_norms = torch.tensor([0.5, 0.3], device=DEVICE)

        result = loss_fn(
            orig_sem, orig_feat, recon_sem, recon_feat,
            fused_sem, fused_feat, align_loss, state_norms
        )

        assert "total" in result
        assert "recon" in result
        assert "cka" in result
        assert not torch.isnan(result["total"])


class TestConsistencyLoss:
    """Test consistency loss functions."""

    def test_uniformity_loss(self):
        """Uniformity loss should be non-negative."""
        from mbps_pytorch.losses.consistency_loss import uniformity_loss

        torch.manual_seed(0)
        instance_masks = torch.sigmoid(torch.randn(1, 5, 64, device=DEVICE))
        semantic_pred = torch.randint(0, 5, (1, 64), device=DEVICE)

        loss = uniformity_loss(instance_masks, semantic_pred, num_classes=5)
        assert float(loss) >= 0.0

    def test_boundary_alignment_range(self):
        """Boundary alignment loss should be in [0, 1]."""
        from mbps_pytorch.losses.consistency_loss import boundary_alignment_loss

        torch.manual_seed(0)
        sem_pred = torch.randint(0, 5, (1, 64), device=DEVICE)
        inst_masks = torch.sigmoid(torch.randn(1, 5, 64, device=DEVICE))

        loss = boundary_alignment_loss(sem_pred, inst_masks, 8, 8)
        assert float(loss) >= 0.0
        assert float(loss) <= 1.0

    def test_dbc_loss_no_nan(self):
        """DBC loss should not produce NaN."""
        from mbps_pytorch.losses.consistency_loss import depth_boundary_coherence_loss

        torch.manual_seed(0)
        sem_pred = torch.randint(0, 5, (1, 64), device=DEVICE)
        inst_masks = torch.sigmoid(torch.randn(1, 5, 64, device=DEVICE))
        depth = torch.rand(1, 64, device=DEVICE)

        loss = depth_boundary_coherence_loss(sem_pred, inst_masks, depth, 8, 8)
        assert not torch.isnan(loss)

    def test_consistency_loss_class(self):
        """ConsistencyLoss should return dict with all components."""
        from mbps_pytorch.losses.consistency_loss import ConsistencyLoss

        torch.manual_seed(0)
        loss_fn = ConsistencyLoss(num_classes=5)
        sem_pred = torch.randint(0, 5, (1, 64), device=DEVICE)
        inst_masks = torch.sigmoid(torch.randn(1, 5, 64, device=DEVICE))
        depth = torch.rand(1, 64, device=DEVICE)

        result = loss_fn(sem_pred, inst_masks, depth, 8, 8)

        assert "total" in result
        assert "uniform" in result
        assert "boundary" in result
        assert "dbc" in result


class TestPQProxyLoss:
    """Test differentiable PQ proxy loss."""

    def test_pq_proxy_range(self):
        """PQ proxy loss should be in [0, 2] (1 - PQ, PQ in [-1,1])."""
        from mbps_pytorch.losses.pq_proxy_loss import PQProxyLoss

        torch.manual_seed(0)
        pq_loss = PQProxyLoss()
        pred_masks = torch.randn(1, 10, 100, device=DEVICE)
        pred_scores = torch.ones(1, 10, device=DEVICE) * 0.8
        torch.manual_seed(1)
        teacher_masks = torch.randn(1, 10, 100, device=DEVICE)
        teacher_scores = torch.ones(1, 10, device=DEVICE) * 0.9

        result = pq_loss(pred_masks, pred_scores, teacher_masks, teacher_scores)

        assert "total" in result
        assert "pq" in result
        assert float(result["total"]) >= 0.0

    def test_perfect_match(self):
        """PQ proxy should be near 1.0 when pred matches teacher."""
        from mbps_pytorch.losses.pq_proxy_loss import differentiable_pq

        masks = torch.sigmoid(torch.ones(5, 100, device=DEVICE) * 5.0)
        scores = torch.ones(5, device=DEVICE) * 0.9

        pq = differentiable_pq(masks, scores, masks, scores)
        assert float(pq) > 0.8

    def test_pq_no_nan(self):
        """PQ proxy should not produce NaN."""
        from mbps_pytorch.losses.pq_proxy_loss import PQProxyLoss

        torch.manual_seed(0)
        pq_loss = PQProxyLoss()
        pred_masks = torch.randn(1, 5, 50, device=DEVICE)
        pred_scores = torch.ones(1, 5, device=DEVICE) * 0.5
        torch.manual_seed(1)
        teacher_masks = torch.randn(1, 5, 50, device=DEVICE)
        teacher_scores = torch.ones(1, 5, device=DEVICE) * 0.5

        result = pq_loss(pred_masks, pred_scores, teacher_masks, teacher_scores)
        assert not torch.isnan(result["total"])


class TestGradientBalancing:
    """Test gradient projection and balancing."""

    def test_no_conflict_preserves_gradient(self):
        """Non-conflicting gradients should be unchanged."""
        from mbps_pytorch.losses.gradient_balancing import project_conflicting_gradients

        g_sem = torch.tensor([1.0, 0.0, 0.0], device=DEVICE)
        g_inst = torch.tensor([0.0, 1.0, 0.0], device=DEVICE)

        projected = project_conflicting_gradients(g_sem, g_inst)
        np.testing.assert_allclose(
            projected.cpu().numpy(), g_inst.cpu().numpy(), atol=1e-6
        )

    def test_conflicting_projected(self):
        """Conflicting gradient component should be removed."""
        from mbps_pytorch.losses.gradient_balancing import project_conflicting_gradients

        g_sem = torch.tensor([1.0, 0.0], device=DEVICE)
        g_inst = torch.tensor([-1.0, 1.0], device=DEVICE)  # Conflicts along dim 0

        projected = project_conflicting_gradients(g_sem, g_inst)

        # Projected should have non-negative dot with g_sem
        dot = torch.sum(projected * g_sem)
        assert float(dot) >= -1e-6

    def test_gradient_balancer_class(self):
        """GradientBalancer should combine gradient dicts."""
        from mbps_pytorch.losses.gradient_balancing import GradientBalancer

        balancer = GradientBalancer()

        g_sem = {"w": torch.tensor([1.0, 0.0], device=DEVICE)}
        g_inst = {"w": torch.tensor([0.0, 1.0], device=DEVICE)}

        combined = balancer.balance_gradients(g_sem, g_inst, beta=1.0)
        assert "w" in combined
        assert combined["w"].shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
