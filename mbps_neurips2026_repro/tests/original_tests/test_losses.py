"""Unit tests for all loss functions.

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
import unittest

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSemanticLoss(unittest.TestCase):
    """Test semantic loss functions."""

    def test_semantic_loss_computes(self):
        """SemanticLoss should return dict with total key."""
        from mbps.losses.semantic_loss import SemanticLoss

        loss_fn = SemanticLoss(lambda_depthg=0.3)
        rng = jax.random.PRNGKey(0)
        sem_codes = jax.random.normal(rng, (2, 50, 90))
        dino_feats = jax.random.normal(rng, (2, 50, 384))
        depth = jax.random.uniform(rng, (2, 50))

        result = loss_fn(sem_codes, dino_feats, depth=depth, key=rng)

        self.assertIn("total", result)
        self.assertIn("stego", result)
        self.assertIn("depthg", result)

    def test_semantic_loss_no_nan(self):
        """Semantic loss should not produce NaN."""
        from mbps.losses.semantic_loss import SemanticLoss

        loss_fn = SemanticLoss()
        rng = jax.random.PRNGKey(42)
        sem_codes = jax.random.normal(rng, (1, 30, 90))
        dino_feats = jax.random.normal(rng, (1, 30, 384))

        result = loss_fn(sem_codes, dino_feats)
        self.assertFalse(jnp.isnan(result["total"]))

    def test_semantic_loss_no_depth(self):
        """SemanticLoss should work without depth (depthg = 0)."""
        from mbps.losses.semantic_loss import SemanticLoss

        loss_fn = SemanticLoss()
        rng = jax.random.PRNGKey(0)
        sem_codes = jax.random.normal(rng, (1, 20, 90))
        dino_feats = jax.random.normal(rng, (1, 20, 384))

        result = loss_fn(sem_codes, dino_feats, depth=None)
        self.assertEqual(float(result["depthg"]), 0.0)


class TestInstanceLoss(unittest.TestCase):
    """Test instance loss functions."""

    def test_dice_loss_range(self):
        """Dice loss should be in [0, 1]."""
        from mbps.losses.instance_loss import dice_loss

        pred = jnp.zeros((1, 5, 100))
        target = jnp.ones((1, 5, 100))
        loss = dice_loss(pred, target)

        self.assertGreaterEqual(float(loss), 0.0)
        self.assertLessEqual(float(loss), 1.0)

    def test_dice_loss_perfect(self):
        """Dice loss should be near 0 for perfect prediction."""
        from mbps.losses.instance_loss import dice_loss

        pred = jnp.ones((1, 5, 100)) * 10.0  # sigmoid ≈ 1
        target = jnp.ones((1, 5, 100))
        loss = dice_loss(pred, target)

        self.assertLess(float(loss), 0.1)

    def test_bce_loss_symmetric(self):
        """BCE with all-zeros pred and all-ones target should equal log(2)."""
        from mbps.losses.instance_loss import mask_bce_loss

        pred = jnp.zeros((1, 1, 100))
        target = jnp.ones((1, 1, 100))
        loss = mask_bce_loss(pred, target)

        # sigmoid(0)=0.5, BCE = -log(0.5) = log(2) ≈ 0.693
        self.assertAlmostEqual(float(loss), np.log(2), places=3)

    def test_instance_loss_class(self):
        """InstanceLoss should return dict with total."""
        from mbps.losses.instance_loss import InstanceLoss

        loss_fn = InstanceLoss()
        rng = jax.random.PRNGKey(0)
        pred_masks = jax.random.normal(rng, (1, 10, 50))
        pred_scores = jnp.ones((1, 10)) * 0.8
        features = jax.random.normal(rng, (1, 50, 384))

        result = loss_fn(pred_masks, pred_scores, features)
        self.assertIn("total", result)

    def test_instance_loss_no_nan(self):
        """Instance loss should not produce NaN."""
        from mbps.losses.instance_loss import InstanceLoss

        loss_fn = InstanceLoss()
        rng = jax.random.PRNGKey(0)
        pred_masks = jax.random.normal(rng, (1, 5, 30))
        pred_scores = jnp.ones((1, 5)) * 0.5
        features = jax.random.normal(rng, (1, 30, 64))

        result = loss_fn(pred_masks, pred_scores, features)
        self.assertFalse(jnp.isnan(result["total"]))


class TestBridgeLoss(unittest.TestCase):
    """Test bridge loss functions."""

    def test_reconstruction_loss(self):
        """Recon loss of identical tensors should be ~0."""
        from mbps.losses.bridge_loss import reconstruction_loss

        x = jnp.ones((1, 50, 90))
        loss = reconstruction_loss(x, x)

        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_cka_self_alignment(self):
        """CKA of feature with itself should give -1 (perfect)."""
        from mbps.losses.bridge_loss import cka_loss

        rng = jax.random.PRNGKey(0)
        features = jax.random.normal(rng, (1, 50, 32))
        loss = cka_loss(features, features)

        self.assertAlmostEqual(float(loss), -1.0, places=3)

    def test_cka_orthogonal(self):
        """CKA of orthogonal features should be near 0."""
        from mbps.losses.bridge_loss import cka_loss

        rng = jax.random.PRNGKey(0)
        a = jax.random.normal(rng, (1, 100, 32))
        b = jax.random.normal(jax.random.PRNGKey(99), (1, 100, 32))

        loss = cka_loss(a, b)
        # Should be close to 0 (slightly negative)
        self.assertGreater(float(loss), -0.5)

    def test_state_reg_nonneg(self):
        """State regularization should be non-negative."""
        from mbps.losses.bridge_loss import state_regularization_loss

        norms = jnp.array([1.0, 2.0, 3.0])
        loss = state_regularization_loss(norms)

        self.assertGreater(float(loss), 0.0)

    def test_bridge_loss_class(self):
        """BridgeLoss should return dict with total."""
        from mbps.losses.bridge_loss import BridgeLoss

        loss_fn = BridgeLoss()
        rng = jax.random.PRNGKey(0)

        orig_sem = jax.random.normal(rng, (1, 50, 90))
        orig_feat = jax.random.normal(rng, (1, 50, 384))
        recon_sem = jax.random.normal(rng, (1, 50, 90))
        recon_feat = jax.random.normal(rng, (1, 50, 384))
        fused_sem = jax.random.normal(rng, (1, 50, 192))
        fused_feat = jax.random.normal(rng, (1, 50, 192))
        align_loss = jnp.array(0.1)
        state_norms = jnp.array([0.5, 0.3])

        result = loss_fn(
            orig_sem, orig_feat, recon_sem, recon_feat,
            fused_sem, fused_feat, align_loss, state_norms
        )

        self.assertIn("total", result)
        self.assertIn("recon", result)
        self.assertIn("cka", result)
        self.assertFalse(jnp.isnan(result["total"]))


class TestConsistencyLoss(unittest.TestCase):
    """Test consistency loss functions."""

    def test_uniformity_loss(self):
        """Uniformity loss should be non-negative."""
        from mbps.losses.consistency_loss import uniformity_loss

        rng = jax.random.PRNGKey(0)
        instance_masks = jax.nn.sigmoid(jax.random.normal(rng, (1, 5, 64)))
        semantic_pred = jax.random.randint(rng, (1, 64), 0, 5)

        loss = uniformity_loss(instance_masks, semantic_pred, num_classes=5)
        self.assertGreaterEqual(float(loss), 0.0)

    def test_boundary_alignment_range(self):
        """Boundary alignment loss should be in [0, 1]."""
        from mbps.losses.consistency_loss import boundary_alignment_loss

        rng = jax.random.PRNGKey(0)
        sem_pred = jax.random.randint(rng, (1, 64), 0, 5)
        inst_masks = jax.nn.sigmoid(jax.random.normal(rng, (1, 5, 64)))

        loss = boundary_alignment_loss(sem_pred, inst_masks, 8, 8)
        self.assertGreaterEqual(float(loss), 0.0)
        self.assertLessEqual(float(loss), 1.0)

    def test_dbc_loss_no_nan(self):
        """DBC loss should not produce NaN."""
        from mbps.losses.consistency_loss import depth_boundary_coherence_loss

        rng = jax.random.PRNGKey(0)
        sem_pred = jax.random.randint(rng, (1, 64), 0, 5)
        inst_masks = jax.nn.sigmoid(jax.random.normal(rng, (1, 5, 64)))
        depth = jax.random.uniform(rng, (1, 64))

        loss = depth_boundary_coherence_loss(sem_pred, inst_masks, depth, 8, 8)
        self.assertFalse(jnp.isnan(loss))

    def test_consistency_loss_class(self):
        """ConsistencyLoss should return dict with all components."""
        from mbps.losses.consistency_loss import ConsistencyLoss

        loss_fn = ConsistencyLoss(num_classes=5)
        rng = jax.random.PRNGKey(0)
        sem_pred = jax.random.randint(rng, (1, 64), 0, 5)
        inst_masks = jax.nn.sigmoid(jax.random.normal(rng, (1, 5, 64)))
        depth = jax.random.uniform(rng, (1, 64))

        result = loss_fn(sem_pred, inst_masks, depth, 8, 8)

        self.assertIn("total", result)
        self.assertIn("uniform", result)
        self.assertIn("boundary", result)
        self.assertIn("dbc", result)


class TestPQProxyLoss(unittest.TestCase):
    """Test differentiable PQ proxy loss."""

    def test_pq_proxy_range(self):
        """PQ proxy loss should be in [0, 2] (1 - PQ, PQ in [-1,1])."""
        from mbps.losses.pq_proxy_loss import PQProxyLoss

        pq_loss = PQProxyLoss()
        rng = jax.random.PRNGKey(0)
        pred_masks = jax.random.normal(rng, (1, 10, 100))
        pred_scores = jnp.ones((1, 10)) * 0.8
        teacher_masks = jax.random.normal(jax.random.PRNGKey(1), (1, 10, 100))
        teacher_scores = jnp.ones((1, 10)) * 0.9

        result = pq_loss(pred_masks, pred_scores, teacher_masks, teacher_scores)

        self.assertIn("total", result)
        self.assertIn("pq", result)
        self.assertGreaterEqual(float(result["total"]), 0.0)

    def test_perfect_match(self):
        """PQ proxy should be near 1.0 when pred matches teacher."""
        from mbps.losses.pq_proxy_loss import differentiable_pq

        masks = jax.nn.sigmoid(jnp.ones((5, 100)) * 5.0)
        scores = jnp.ones(5) * 0.9

        pq = differentiable_pq(masks, scores, masks, scores)
        self.assertGreater(float(pq), 0.8)

    def test_pq_no_nan(self):
        """PQ proxy should not produce NaN."""
        from mbps.losses.pq_proxy_loss import PQProxyLoss

        pq_loss = PQProxyLoss()
        rng = jax.random.PRNGKey(0)
        pred_masks = jax.random.normal(rng, (1, 5, 50))
        pred_scores = jnp.ones((1, 5)) * 0.5
        teacher_masks = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 50))
        teacher_scores = jnp.ones((1, 5)) * 0.5

        result = pq_loss(pred_masks, pred_scores, teacher_masks, teacher_scores)
        self.assertFalse(jnp.isnan(result["total"]))


class TestGradientBalancing(unittest.TestCase):
    """Test gradient projection and balancing."""

    def test_no_conflict_preserves_gradient(self):
        """Non-conflicting gradients should be unchanged."""
        from mbps.losses.gradient_balancing import project_conflicting_gradients

        g_sem = jnp.array([1.0, 0.0, 0.0])
        g_inst = jnp.array([0.0, 1.0, 0.0])

        projected = project_conflicting_gradients(g_sem, g_inst)
        np.testing.assert_allclose(np.array(projected), np.array(g_inst), atol=1e-6)

    def test_conflicting_projected(self):
        """Conflicting gradient component should be removed."""
        from mbps.losses.gradient_balancing import project_conflicting_gradients

        g_sem = jnp.array([1.0, 0.0])
        g_inst = jnp.array([-1.0, 1.0])  # Conflicts along dim 0

        projected = project_conflicting_gradients(g_sem, g_inst)

        # Projected should have non-negative dot with g_sem
        dot = jnp.sum(projected * g_sem)
        self.assertGreaterEqual(float(dot), -1e-6)

    def test_gradient_balancer_class(self):
        """GradientBalancer should combine gradient dicts."""
        from mbps.losses.gradient_balancing import GradientBalancer

        balancer = GradientBalancer()

        g_sem = {"w": jnp.array([1.0, 0.0])}
        g_inst = {"w": jnp.array([0.0, 1.0])}

        combined = balancer.balance_gradients(g_sem, g_inst, beta=1.0)
        self.assertIn("w", combined)
        self.assertEqual(combined["w"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
