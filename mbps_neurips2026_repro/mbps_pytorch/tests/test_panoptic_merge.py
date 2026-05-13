"""Unit tests for panoptic merging module (PyTorch).

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

import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPanopticMerge:
    """Test panoptic merge algorithm."""

    def test_no_multi_instance_pixels(self):
        """No pixel should belong to multiple instances."""
        from mbps_pytorch.models.merger.panoptic_merge import panoptic_merge

        n = 100
        m = 10

        semantic_pred = torch.zeros(n, dtype=torch.int32)
        torch.manual_seed(0)
        instance_masks = torch.randn(m, n)
        instance_scores = torch.linspace(0.9, 0.4, m)
        thing_clusters = torch.tensor([True, True, True, False, False])
        stuff_clusters = torch.tensor([False, False, False, True, True])

        _, instance_ids = panoptic_merge(
            semantic_pred, instance_masks, instance_scores,
            thing_clusters, stuff_clusters
        )

        # Each pixel should have exactly one instance ID
        # (0 for stuff, positive for things)
        assert instance_ids.shape == (n,)

        # Check that instance IDs are non-negative
        assert torch.all(instance_ids >= 0)

    def test_output_format_panoptic_ids(self):
        """Panoptic IDs should encode instance_id * 1000 + class."""
        from mbps_pytorch.models.merger.panoptic_merge import panoptic_merge

        n = 50
        semantic_pred = torch.ones(n, dtype=torch.int32) * 2  # All class 2
        instance_masks = torch.ones(3, n) * 5.0  # High logits
        instance_scores = torch.tensor([0.9, 0.8, 0.7])
        thing_clusters = torch.tensor([False, False, True, False, False])
        stuff_clusters = torch.tensor([True, True, False, True, True])

        panoptic_ids, instance_ids = panoptic_merge(
            semantic_pred, instance_masks, instance_scores,
            thing_clusters, stuff_clusters
        )

        # Verify encoding: panoptic_id = instance_id * 1000 + class
        decoded_instance = panoptic_ids // 1000
        decoded_class = panoptic_ids % 1000

        np.testing.assert_array_equal(
            decoded_instance.numpy(), instance_ids.numpy()
        )
        np.testing.assert_array_equal(
            decoded_class.numpy(), semantic_pred.numpy()
        )

    def test_score_priority(self):
        """Higher-scoring instances should be assigned first."""
        from mbps_pytorch.models.merger.panoptic_merge import panoptic_merge

        n = 20
        # Two non-overlapping instances
        mask1 = torch.cat([torch.ones(10) * 5, torch.ones(10) * -5])
        mask2 = torch.cat([torch.ones(10) * -5, torch.ones(10) * 5])
        instance_masks = torch.stack([mask1, mask2])

        semantic_pred = torch.zeros(n, dtype=torch.int32)
        instance_scores = torch.tensor([0.9, 0.5])
        thing_clusters = torch.tensor([True])
        stuff_clusters = torch.tensor([False])

        _, instance_ids = panoptic_merge(
            semantic_pred, instance_masks, instance_scores,
            thing_clusters, stuff_clusters
        )

        # First 10 pixels should have instance_id=1 (higher score)
        # Next 10 should have instance_id=2
        assert int(instance_ids[0]) == 1
        assert int(instance_ids[10]) == 2

    def test_stuff_pixels_zero_instance(self):
        """Stuff-class pixels should have instance_id=0."""
        from mbps_pytorch.models.merger.panoptic_merge import panoptic_merge

        n = 50
        semantic_pred = torch.ones(n, dtype=torch.int32) * 3  # All class 3
        torch.manual_seed(0)
        instance_masks = torch.randn(5, n)
        instance_scores = torch.ones(5) * 0.8

        # Class 3 is stuff
        thing_clusters = torch.tensor([True, True, False, False, False])
        stuff_clusters = torch.tensor([False, False, True, True, True])

        _, instance_ids = panoptic_merge(
            semantic_pred, instance_masks, instance_scores,
            thing_clusters, stuff_clusters
        )

        # All pixels are stuff class, so instance_id should be 0
        assert torch.all(instance_ids == 0)

    def test_low_score_filtered(self):
        """Instances below score threshold should be excluded."""
        from mbps_pytorch.models.merger.panoptic_merge import panoptic_merge

        n = 50
        semantic_pred = torch.zeros(n, dtype=torch.int32)
        instance_masks = torch.ones(3, n) * 5.0
        instance_scores = torch.tensor([0.1, 0.1, 0.1])  # All below threshold
        thing_clusters = torch.tensor([True])
        stuff_clusters = torch.tensor([False])

        _, instance_ids = panoptic_merge(
            semantic_pred, instance_masks, instance_scores,
            thing_clusters, stuff_clusters,
            score_threshold=0.3,
        )

        # All instances filtered -> all stuff (id=0)
        assert torch.all(instance_ids == 0)

    def test_empty_input(self):
        """Should handle case with zero instances gracefully."""
        from mbps_pytorch.models.merger.panoptic_merge import panoptic_merge

        n = 20
        semantic_pred = torch.zeros(n, dtype=torch.int32)
        instance_masks = torch.zeros(0, n)
        instance_scores = torch.zeros(0)
        thing_clusters = torch.tensor([True])
        stuff_clusters = torch.tensor([False])

        # Should not raise
        panoptic_ids, instance_ids = panoptic_merge(
            semantic_pred, instance_masks, instance_scores,
            thing_clusters, stuff_clusters
        )

        assert panoptic_ids.shape == (n,)


class TestBatchPanopticMerge:
    """Test batch panoptic merge."""

    def test_batch_output_shapes(self):
        """Batch merge should return (B, N) outputs."""
        from mbps_pytorch.models.merger.panoptic_merge import batch_panoptic_merge

        b, n, m = 2, 50, 5
        sem_pred = torch.zeros(b, n, dtype=torch.int32)
        torch.manual_seed(0)
        inst_masks = torch.randn(b, m, n)
        inst_scores = torch.ones(b, m) * 0.8
        thing_mask = torch.tensor([[True, True, False]] * b)
        stuff_mask = torch.tensor([[False, False, True]] * b)

        pan_ids, inst_ids = batch_panoptic_merge(
            sem_pred, inst_masks, inst_scores, thing_mask, stuff_mask
        )

        assert pan_ids.shape == (b, n)
        assert inst_ids.shape == (b, n)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
