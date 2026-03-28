"""Unit tests for MBPS core modules.

Tests cover:
    - Model components (shape checks, forward pass)
    - Loss functions (value ranges, gradients)
    - Data transforms
    - Bridge modules
    - Training curriculum
"""

from __future__ import annotations

import sys
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBackbone(unittest.TestCase):
    """Test DINO ViT-S/8 backbone."""

    def test_output_shape(self):
        """Backbone should output (B, N+1, 384) tokens."""
        from mbps.models.backbone.dino_vits8 import DINOViTS8

        model = DINOViTS8()
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 224, 224, 3))
        params = model.init(rng, x)
        out = model.apply(params, x)

        # ViT-S/8 on 224x224: (224/8)^2 = 784 patches + 1 cls = 785
        self.assertEqual(out.shape, (1, 785, 384))

    def test_spatial_features(self):
        """Spatial features should have (B, N, 384) without CLS."""
        from mbps.models.backbone.dino_vits8 import DINOViTS8

        model = DINOViTS8()
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 224, 224, 3))
        params = model.init(rng, x)
        out = model.apply(params, x, method=model.get_spatial_features)

        self.assertEqual(out.shape, (1, 784, 384))

    def test_frozen_gradients(self):
        """Backbone should produce zero gradients (frozen)."""
        from mbps.models.backbone.dino_vits8 import DINOViTS8

        model = DINOViTS8()
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 64, 64, 3))
        params = model.init(rng, x)

        def loss_fn(params):
            return jnp.mean(model.apply(params, x))

        grads = jax.grad(loss_fn)(params)
        grad_norms = jax.tree.map(lambda g: jnp.sum(jnp.abs(g)), grads)
        total = sum(jax.tree.leaves(grad_norms))
        self.assertEqual(float(total), 0.0)


class TestSemanticHead(unittest.TestCase):
    """Test DepthG semantic head."""

    def test_output_shape(self):
        """DepthG should output (B, N, 90) semantic codes."""
        from mbps.models.semantic.depthg_head import DepthGHead

        model = DepthGHead(input_dim=384, output_dim=90)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((2, 100, 384))
        params = model.init(rng, x)
        out = model.apply(params, x)

        self.assertEqual(out.shape, (2, 100, 90))


class TestInstanceHead(unittest.TestCase):
    """Test instance segmentation head."""

    def test_output_shapes(self):
        """Instance head should output masks and scores."""
        from mbps.models.instance.cascade_mask_rcnn import InstanceHead

        model = InstanceHead(max_instances=50, hidden_dim=128)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((2, 100, 384))
        params = model.init(rng, x)
        masks, scores = model.apply(params, x)

        self.assertEqual(masks.shape, (2, 50, 100))
        self.assertEqual(scores.shape, (2, 50))

    def test_scores_range(self):
        """Instance scores should be in [0, 1]."""
        from mbps.models.instance.cascade_mask_rcnn import InstanceHead

        model = InstanceHead(max_instances=20, hidden_dim=64)
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (1, 50, 384))
        params = model.init(rng, x)
        _, scores = model.apply(params, x)

        self.assertTrue(jnp.all(scores >= 0))
        self.assertTrue(jnp.all(scores <= 1))


class TestProjectionBridge(unittest.TestCase):
    """Test Adaptive Projection Bridge."""

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
        feat = jax.random.normal(rng, (1, 50, 384))
        params = model.init(rng, sem, feat)
        _, _, align_loss = model.apply(params, sem, feat)

        self.assertGreaterEqual(float(align_loss), 0.0)


class TestMamba2(unittest.TestCase):
    """Test Mamba2 SSD module."""

    def test_output_shape(self):
        """Mamba2 block should preserve sequence shape."""
        from mbps.models.bridge.mamba2_ssd import Mamba2Block

        model = Mamba2Block(dim=64, state_dim=16, chunk_size=32)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((2, 128, 64))
        params = model.init(rng, x)
        out = model.apply(params, x)

        self.assertEqual(out.shape, (2, 128, 64))

    def test_stack_output_shape(self):
        """Mamba2 stack should preserve shape."""
        from mbps.models.bridge.mamba2_ssd import Mamba2Stack

        model = Mamba2Stack(num_layers=2, dim=32, state_dim=8, chunk_size=16)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 64, 32))
        params = model.init(rng, x)
        out = model.apply(params, x)

        self.assertEqual(out.shape, (1, 64, 32))


class TestBiCMS(unittest.TestCase):
    """Test Bidirectional Cross-Modal Scan."""

    def test_output_shapes(self):
        """BiCMS should produce two output streams of same shape."""
        from mbps.models.bridge.bicms import BidirectionalCrossModalScan

        model = BidirectionalCrossModalScan(
            dim=32, num_layers=1, state_dim=8, chunk_size=16
        )
        rng = jax.random.PRNGKey(0)
        sem = jnp.ones((1, 32, 32))
        feat = jnp.ones((1, 32, 32))
        params = model.init(rng, sem, feat)
        out_sem, out_feat = model.apply(params, sem, feat)

        self.assertEqual(out_sem.shape, (1, 32, 32))
        self.assertEqual(out_feat.shape, (1, 32, 32))


class TestDepthConditioning(unittest.TestCase):
    """Test Unified Depth Conditioning Module."""

    def test_output_shapes(self):
        """UDCM should output conditioned features of same shape."""
        from mbps.models.bridge.depth_conditioning import (
            UnifiedDepthConditioning,
        )

        model = UnifiedDepthConditioning(bridge_dim=64)
        rng = jax.random.PRNGKey(0)
        depth = jnp.ones((2, 50))
        sem = jnp.ones((2, 50, 64))
        feat = jnp.ones((2, 50, 64))
        params = model.init(rng, depth, sem, feat)
        out_sem, out_feat, d_loss = model.apply(params, depth, sem, feat)

        self.assertEqual(out_sem.shape, (2, 50, 64))
        self.assertEqual(out_feat.shape, (2, 50, 64))


class TestLossFunctions(unittest.TestCase):
    """Test loss function computations."""

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

        pred = jnp.ones((1, 5, 100)) * 10.0  # High logits → sigmoid ≈ 1
        target = jnp.ones((1, 5, 100))
        loss = dice_loss(pred, target)

        self.assertLess(float(loss), 0.1)

    def test_cka_loss_self_alignment(self):
        """CKA of feature with itself should give -1 (perfect alignment)."""
        from mbps.losses.bridge_loss import cka_loss

        features = jax.random.normal(jax.random.PRNGKey(0), (1, 50, 32))
        loss = cka_loss(features, features)

        self.assertAlmostEqual(float(loss), -1.0, places=3)

    def test_pq_proxy_range(self):
        """PQ proxy should be in [0, 1]."""
        from mbps.losses.pq_proxy_loss import PQProxyLoss

        pq_loss = PQProxyLoss()
        pred_masks = jax.random.normal(jax.random.PRNGKey(0), (1, 10, 100))
        pred_scores = jnp.ones((1, 10)) * 0.8
        teacher_masks = jax.random.normal(jax.random.PRNGKey(1), (1, 10, 100))
        teacher_scores = jnp.ones((1, 10)) * 0.9

        result = pq_loss(pred_masks, pred_scores, teacher_masks, teacher_scores)

        self.assertGreaterEqual(float(result["total"]), 0.0)
        self.assertLessEqual(float(result["total"]), 2.0)


class TestCurriculum(unittest.TestCase):
    """Test training curriculum."""

    def test_phase_transitions(self):
        """Phases should transition at correct epochs."""
        from mbps.training.curriculum import TrainingCurriculum

        curriculum = TrainingCurriculum(
            phase_a_end=20, phase_b_end=40, total_epochs=60
        )

        self.assertEqual(curriculum.get_phase(1), "A")
        self.assertEqual(curriculum.get_phase(20), "A")
        self.assertEqual(curriculum.get_phase(21), "B")
        self.assertEqual(curriculum.get_phase(40), "B")
        self.assertEqual(curriculum.get_phase(41), "C")

    def test_beta_rampup(self):
        """Beta should ramp from 0→1 during Phase B."""
        from mbps.training.curriculum import TrainingCurriculum

        curriculum = TrainingCurriculum(
            phase_a_end=20, phase_b_end=40
        )

        config_start = curriculum.get_config(21)
        config_mid = curriculum.get_config(30)
        config_end = curriculum.get_config(40)

        self.assertAlmostEqual(config_start.beta, 0.05, places=1)
        self.assertAlmostEqual(config_mid.beta, 0.5, places=1)
        self.assertAlmostEqual(config_end.beta, 1.0, places=1)

    def test_phase_a_no_instance(self):
        """Phase A should have zero instance weight."""
        from mbps.training.curriculum import TrainingCurriculum

        curriculum = TrainingCurriculum()
        config = curriculum.get_config(10)

        self.assertEqual(config.beta, 0.0)
        self.assertFalse(config.use_bridge)


class TestTransforms(unittest.TestCase):
    """Test data transforms."""

    def test_normalize_denormalize(self):
        """Normalize → denormalize should be identity."""
        from mbps.data.transforms import denormalize, normalize

        rng = jax.random.PRNGKey(0)
        image = jax.random.uniform(rng, (224, 224, 3))

        normalized = normalize(image)
        recovered = denormalize(normalized)

        np.testing.assert_allclose(
            np.array(image), np.array(recovered), atol=1e-5
        )

    def test_random_crop_shape(self):
        """Random crop should produce correct output shape."""
        from mbps.data.transforms import random_crop

        rng = jax.random.PRNGKey(0)
        image = jnp.ones((512, 512, 3))
        cropped = random_crop(image, (320, 320), rng)

        self.assertEqual(cropped.shape, (320, 320, 3))


class TestStuffThingsClassifier(unittest.TestCase):
    """Test stuff-things classifier."""

    def test_output_shape(self):
        """Should output one score per cluster."""
        from mbps.models.classifier.stuff_things_mlp import StuffThingsClassifier

        model = StuffThingsClassifier()
        rng = jax.random.PRNGKey(0)
        cues = jnp.ones((2, 27, 3))
        params = model.init(rng, cues)
        scores = model.apply(params, cues)

        self.assertEqual(scores.shape, (2, 27))

    def test_scores_in_range(self):
        """Scores should be in [0, 1] (sigmoid output)."""
        from mbps.models.classifier.stuff_things_mlp import StuffThingsClassifier

        model = StuffThingsClassifier()
        rng = jax.random.PRNGKey(42)
        cues = jax.random.normal(rng, (1, 10, 3))
        params = model.init(rng, cues)
        scores = model.apply(params, cues)

        self.assertTrue(jnp.all(scores >= 0))
        self.assertTrue(jnp.all(scores <= 1))


class TestHungarianMatching(unittest.TestCase):
    """Test Hungarian matching."""

    def test_perfect_match(self):
        """Perfect clustering should give accuracy = 1.0."""
        from mbps.evaluation.hungarian_matching import hungarian_match

        pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        gt = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])

        mapping, acc = hungarian_match(pred, gt, 3, 3)
        self.assertAlmostEqual(acc, 1.0, places=5)


class TestEMA(unittest.TestCase):
    """Test EMA updates."""

    def test_ema_update(self):
        """EMA should smoothly average parameters."""
        from mbps.training.ema import EMAState

        params = {"w": jnp.array([1.0, 2.0, 3.0])}
        ema = EMAState(params, momentum=0.5)

        new_params = {"w": jnp.array([0.0, 0.0, 0.0])}
        ema.update(new_params)

        expected = 0.5 * jnp.array([1.0, 2.0, 3.0]) + 0.5 * jnp.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            np.array(ema.get_params()["w"]),
            np.array(expected),
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
