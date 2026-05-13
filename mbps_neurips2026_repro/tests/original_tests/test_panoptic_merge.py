"""Unit tests for panoptic merging module.

Tests cover:
    - No pixel belongs to multiple instances
    - Output format validity
    - Score-based priority ordering
    - Stuff region filling
    - Overlap resolution
"""

from __future__ import annotations

import sys
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPanopticMerge(unittest.TestCase):
    """Test panoptic merge algorithm."""

    def test_no_multi_instance_pixels(self):
        """No pixel should belong to multiple instances."""
        from mbps.models.merger.panoptic_merge import panoptic_merge

        n = 100
        m = 10
        k = 5

        semantic_pred = jnp.zeros(n, dtype=jnp.int32)
        instance_masks = jax.random.normal(jax.random.PRNGKey(0), (m, n))
        instance_scores = jnp.linspace(0.9, 0.4, m)
        thing_clusters = jnp.array([True, True, True, False, False])
        stuff_clusters = jnp.array([False, False, False, True, True])

        _, instance_ids = panoptic_merge(
            semantic_pred, instance_masks, instance_scores,
            thing_clusters, stuff_clusters
        )

        # Each pixel should have exactly one instance ID
        # (0 for stuff, positive for things)
        self.assertEqual(instance_ids.shape, (n,))

        # Check that instance IDs are non-negative
        self.assertTrue(jnp.all(instance_ids >= 0))

    def test_output_format_panoptic_ids(self):
        """Panoptic IDs should encode instance_id * 1000 + class."""
        from mbps.models.merger.panoptic_merge import panoptic_merge

        n = 50
        semantic_pred = jnp.ones(n, dtype=jnp.int32) * 2  # All class 2
        instance_masks = jnp.ones((3, n)) * 5.0  # High logits
        instance_scores = jnp.array([0.9, 0.8, 0.7])
        thing_clusters = jnp.array([False, False, True, False, False])
        stuff_clusters = jnp.array([True, True, False, True, True])

        panoptic_ids, instance_ids = panoptic_merge(
            semantic_pred, instance_masks, instance_scores,
            thing_clusters, stuff_clusters
        )

        # Verify encoding: panoptic_id = instance_id * 1000 + class
        decoded_instance = panoptic_ids // 1000
        decoded_class = panoptic_ids % 1000

        np.testing.assert_array_equal(
            np.array(decoded_instance), np.array(instance_ids)
        )
        np.testing.assert_array_equal(
            np.array(decoded_class), np.array(semantic_pred)
        )

    def test_score_priority(self):
        """Higher-scoring instances should be assigned first."""
        from mbps.models.merger.panoptic_merge import panoptic_merge

        n = 20
        # Two non-overlapping instances
        mask1 = jnp.concatenate([jnp.ones(10) * 5, jnp.ones(10) * -5])
        mask2 = jnp.concatenate([jnp.ones(10) * -5, jnp.ones(10) * 5])
        instance_masks = jnp.stack([mask1, mask2])

        semantic_pred = jnp.zeros(n, dtype=jnp.int32)
        instance_scores = jnp.array([0.9, 0.5])
        thing_clusters = jnp.array([True])
        stuff_clusters = jnp.array([False])

        _, instance_ids = panoptic_merge(
            semantic_pred, instance_masks, instance_scores,
            thing_clusters, stuff_clusters
        )

        # First 10 pixels should have instance_id=1 (higher score)
        # Next 10 should have instance_id=2
        self.assertEqual(int(instance_ids[0]), 1)
        self.assertEqual(int(instance_ids[10]), 2)

    def test_stuff_pixels_zero_instance(self):
        """Stuff-class pixels should have instance_id=0."""
        from mbps.models.merger.panoptic_merge import panoptic_merge

        n = 50
        semantic_pred = jnp.ones(n, dtype=jnp.int32) * 3  # All class 3
        instance_masks = jax.random.normal(jax.random.PRNGKey(0), (5, n))
        instance_scores = jnp.ones(5) * 0.8

        # Class 3 is stuff
        thing_clusters = jnp.array([True, True, False, False, False])
        stuff_clusters = jnp.array([False, False, True, True, True])

        _, instance_ids = panoptic_merge(
            semantic_pred, instance_masks, instance_scores,
            thing_clusters, stuff_clusters
        )

        # All pixels are stuff class, so instance_id should be 0
        self.assertTrue(jnp.all(instance_ids == 0))

    def test_low_score_filtered(self):
        """Instances below score threshold should be excluded."""
        from mbps.models.merger.panoptic_merge import panoptic_merge

        n = 50
        semantic_pred = jnp.zeros(n, dtype=jnp.int32)
        instance_masks = jnp.ones((3, n)) * 5.0
        instance_scores = jnp.array([0.1, 0.1, 0.1])  # All below threshold
        thing_clusters = jnp.array([True])
        stuff_clusters = jnp.array([False])

        _, instance_ids = panoptic_merge(
            semantic_pred, instance_masks, instance_scores,
            thing_clusters, stuff_clusters,
            score_threshold=0.3,
        )

        # All instances filtered → all stuff (id=0)
        self.assertTrue(jnp.all(instance_ids == 0))

    def test_empty_input(self):
        """Should handle case with zero instances gracefully."""
        from mbps.models.merger.panoptic_merge import panoptic_merge

        n = 20
        semantic_pred = jnp.zeros(n, dtype=jnp.int32)
        instance_masks = jnp.zeros((0, n))
        instance_scores = jnp.zeros(0)
        thing_clusters = jnp.array([True])
        stuff_clusters = jnp.array([False])

        # Should not raise
        panoptic_ids, instance_ids = panoptic_merge(
            semantic_pred, instance_masks, instance_scores,
            thing_clusters, stuff_clusters
        )

        self.assertEqual(panoptic_ids.shape, (n,))


class TestBatchPanopticMerge(unittest.TestCase):
    """Test batch panoptic merge."""

    def test_batch_output_shapes(self):
        """Batch merge should return (B, N) outputs."""
        from mbps.models.merger.panoptic_merge import batch_panoptic_merge

        b, n, m, k = 2, 50, 5, 3
        sem_pred = jnp.zeros((b, n), dtype=jnp.int32)
        inst_masks = jax.random.normal(jax.random.PRNGKey(0), (b, m, n))
        inst_scores = jnp.ones((b, m)) * 0.8
        thing_mask = jnp.array([[True, True, False]] * b)
        stuff_mask = jnp.array([[False, False, True]] * b)

        pan_ids, inst_ids = batch_panoptic_merge(
            sem_pred, inst_masks, inst_scores, thing_mask, stuff_mask
        )

        self.assertEqual(pan_ids.shape, (b, n))
        self.assertEqual(inst_ids.shape, (b, n))


if __name__ == "__main__":
    unittest.main()
