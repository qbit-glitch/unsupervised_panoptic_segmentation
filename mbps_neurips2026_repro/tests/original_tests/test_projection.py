"""Unit tests for Adaptive Projection Bridge.

Tests cover:
    - Forward/inverse projection shapes
    - Reconstruction error
    - CKA alignment computation
    - Alignment loss properties
    - Gradient flow through projections
"""

from __future__ import annotations

import sys
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAdaptiveProjectionBridge(unittest.TestCase):
    """Test the Adaptive Projection Bridge (APB)."""

    def test_output_shapes(self):
        """Projection should output bridge-dim features."""
        from mbps.models.bridge.projection import AdaptiveProjectionBridge

        model = AdaptiveProjectionBridge(
            semantic_dim=90, feature_dim=384, bridge_dim=192
        )
        rng = jax.random.PRNGKey(0)
        sem = jnp.ones((2, 100, 90))
        feat = jnp.ones((2, 100, 384))
        params = model.init(rng, sem, feat)
        sem_p, feat_p, align_loss = model.apply(params, sem, feat)

        self.assertEqual(sem_p.shape, (2, 100, 192))
        self.assertEqual(feat_p.shape, (2, 100, 192))
        self.assertEqual(align_loss.shape, ())

    def test_align_loss_nonnegative(self):
        """Alignment loss should be non-negative."""
        from mbps.models.bridge.projection import AdaptiveProjectionBridge

        model = AdaptiveProjectionBridge()
        rng = jax.random.PRNGKey(0)
        sem = jax.random.normal(rng, (1, 50, 90))
        feat = jax.random.normal(jax.random.PRNGKey(1), (1, 50, 384))
        params = model.init(rng, sem, feat)
        _, _, align_loss = model.apply(params, sem, feat)

        self.assertGreaterEqual(float(align_loss), 0.0)

    def test_gradient_flow(self):
        """Gradients should flow through projection bridge."""
        from mbps.models.bridge.projection import AdaptiveProjectionBridge

        model = AdaptiveProjectionBridge(
            semantic_dim=90, feature_dim=384, bridge_dim=192
        )
        rng = jax.random.PRNGKey(0)
        sem = jax.random.normal(rng, (1, 20, 90))
        feat = jax.random.normal(rng, (1, 20, 384))
        params = model.init(rng, sem, feat)

        def loss_fn(params):
            s, f, a = model.apply(params, sem, feat)
            return jnp.mean(s ** 2) + jnp.mean(f ** 2) + a

        grads = jax.grad(loss_fn)(params)
        grad_norms = jax.tree.map(lambda g: float(jnp.sum(jnp.abs(g))), grads)
        total = sum(jax.tree.leaves(grad_norms))

        self.assertGreater(total, 0.0)

    def test_different_bridge_dims(self):
        """Projection should work with various bridge dimensions."""
        from mbps.models.bridge.projection import AdaptiveProjectionBridge

        for bridge_dim in [128, 192, 256, 384]:
            model = AdaptiveProjectionBridge(
                semantic_dim=90, feature_dim=384, bridge_dim=bridge_dim
            )
            rng = jax.random.PRNGKey(0)
            sem = jnp.ones((1, 10, 90))
            feat = jnp.ones((1, 10, 384))
            params = model.init(rng, sem, feat)
            sem_p, feat_p, _ = model.apply(params, sem, feat)

            self.assertEqual(sem_p.shape[-1], bridge_dim)
            self.assertEqual(feat_p.shape[-1], bridge_dim)


class TestInverseProjection(unittest.TestCase):
    """Test inverse projection from bridge back to original dim."""

    def test_output_shape(self):
        """Inverse projection should output correct dimension."""
        from mbps.models.bridge.projection import InverseProjection

        model = InverseProjection(output_dim=384)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((2, 50, 192))
        params = model.init(rng, x)
        out = model.apply(params, x)

        self.assertEqual(out.shape, (2, 50, 384))

    def test_semantic_inverse(self):
        """Inverse projection for semantic should restore dim 90."""
        from mbps.models.bridge.projection import InverseProjection

        model = InverseProjection(output_dim=90)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 30, 192))
        params = model.init(rng, x)
        out = model.apply(params, x)

        self.assertEqual(out.shape, (1, 30, 90))


class TestReconstructionError(unittest.TestCase):
    """Test round-trip reconstruction quality."""

    def test_reconstruction_error_bounded(self):
        """Reconstruction error should be bounded after training init."""
        from mbps.models.bridge.projection import (
            AdaptiveProjectionBridge,
            InverseProjection,
        )

        rng = jax.random.PRNGKey(0)
        sem = jax.random.normal(rng, (1, 50, 90))
        feat = jax.random.normal(jax.random.PRNGKey(1), (1, 50, 384))

        # Forward projection
        proj = AdaptiveProjectionBridge(
            semantic_dim=90, feature_dim=384, bridge_dim=192
        )
        proj_params = proj.init(rng, sem, feat)
        sem_p, feat_p, _ = proj.apply(proj_params, sem, feat)

        # Inverse projection
        inv_sem = InverseProjection(output_dim=90)
        inv_feat = InverseProjection(output_dim=384)

        inv_sem_params = inv_sem.init(rng, sem_p)
        inv_feat_params = inv_feat.init(rng, feat_p)

        recon_sem = inv_sem.apply(inv_sem_params, sem_p)
        recon_feat = inv_feat.apply(inv_feat_params, feat_p)

        # Reconstruction error should be finite (not checking < 0.05 before training)
        sem_error = float(jnp.mean((sem - recon_sem) ** 2))
        feat_error = float(jnp.mean((feat - recon_feat) ** 2))

        self.assertTrue(np.isfinite(sem_error))
        self.assertTrue(np.isfinite(feat_error))

    def test_cka_after_projection(self):
        """CKA between projected streams should be finite and bounded."""
        from mbps.models.bridge.projection import AdaptiveProjectionBridge
        from mbps.losses.bridge_loss import cka_loss

        rng = jax.random.PRNGKey(0)
        sem = jax.random.normal(rng, (1, 50, 90))
        feat = jax.random.normal(jax.random.PRNGKey(1), (1, 50, 384))

        proj = AdaptiveProjectionBridge()
        params = proj.init(rng, sem, feat)
        sem_p, feat_p, _ = proj.apply(params, sem, feat)

        cka = cka_loss(sem_p, feat_p)

        # CKA loss should be in [-1, 0] (negative CKA)
        self.assertLessEqual(float(cka), 0.1)  # Allow small positive margin
        self.assertGreaterEqual(float(cka), -1.1)


if __name__ == "__main__":
    unittest.main()
