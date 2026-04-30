"""Tests for hierarchical density-frozen cluster merging.

Algorithm: greedy agglomerative merging of cluster centroids by cosine
similarity, with the n_freeze lowest-population centroids protected from
being merged. Used to reduce a k=300 over-clustering down to k=80 while
preserving rare modes (dead-class candidates).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


class TestRareModeIdentification:
    """Identify low-population centroids to freeze as rare modes."""

    def test_returns_n_smallest_population_indices(self) -> None:
        """The n_freeze indices with lowest pixel counts are returned."""
        from mbps_pytorch.stage4.hierarchical_merge import identify_rare_modes

        # 5 centroids with pixel counts; the 2 smallest (idx 1, 3) should be picked
        pixel_counts = torch.tensor([1000.0, 5.0, 2000.0, 10.0, 800.0])
        rare = identify_rare_modes(pixel_counts, n_freeze=2)
        assert sorted(rare) == [1, 3]

    def test_returns_empty_when_n_freeze_zero(self) -> None:
        from mbps_pytorch.stage4.hierarchical_merge import identify_rare_modes

        pixel_counts = torch.tensor([1.0, 2.0, 3.0])
        rare = identify_rare_modes(pixel_counts, n_freeze=0)
        assert rare == []

    def test_caps_n_freeze_at_total_clusters(self) -> None:
        from mbps_pytorch.stage4.hierarchical_merge import identify_rare_modes

        pixel_counts = torch.tensor([1.0, 2.0])
        rare = identify_rare_modes(pixel_counts, n_freeze=10)
        assert sorted(rare) == [0, 1]


class TestHierarchicalMerge:
    """End-to-end agglomerative merge with frozen-centroid constraint."""

    def test_reduces_to_target_k(self) -> None:
        """Final cluster count equals target_k."""
        from mbps_pytorch.stage4.hierarchical_merge import hierarchical_merge

        torch.manual_seed(0)
        centroids = torch.randn(20, 16)
        counts = torch.full((20,), 100.0)

        merged_centroids, mapping = hierarchical_merge(
            centroids, counts, target_k=10, frozen_indices=[]
        )
        assert merged_centroids.shape[0] == 10
        assert mapping.shape == (20,)
        assert mapping.max().item() < 10

    def test_frozen_centroids_survive(self) -> None:
        """Centroids in frozen_indices are unchanged in the output."""
        from mbps_pytorch.stage4.hierarchical_merge import hierarchical_merge

        torch.manual_seed(1)
        centroids = torch.randn(10, 8)
        counts = torch.full((10,), 100.0)

        frozen = [2, 7]
        merged_centroids, mapping = hierarchical_merge(
            centroids, counts, target_k=5, frozen_indices=frozen
        )

        # Each frozen centroid must remain bit-exact in the merged set
        for old_idx in frozen:
            new_idx = mapping[old_idx].item()
            assert torch.allclose(merged_centroids[new_idx], centroids[old_idx])

    def test_mapping_is_consistent(self) -> None:
        """If two old centroids map to the same new id, they were merged."""
        from mbps_pytorch.stage4.hierarchical_merge import hierarchical_merge

        torch.manual_seed(2)
        centroids = torch.randn(8, 4)
        counts = torch.full((8,), 50.0)
        merged_centroids, mapping = hierarchical_merge(
            centroids, counts, target_k=4, frozen_indices=[]
        )

        # Every old id must have a valid new id in [0, target_k)
        assert mapping.min().item() >= 0
        assert mapping.max().item() < 4

        # Every new id must have at least one old id mapped to it
        unique_new = mapping.unique()
        assert unique_new.shape[0] == 4

    def test_target_k_below_frozen_raises(self) -> None:
        """Cannot reduce below the number of frozen centroids."""
        from mbps_pytorch.stage4.hierarchical_merge import hierarchical_merge

        centroids = torch.randn(10, 4)
        counts = torch.full((10,), 100.0)

        with pytest.raises(ValueError):
            hierarchical_merge(
                centroids, counts, target_k=2, frozen_indices=[0, 1, 2, 3]
            )

    def test_no_merge_when_target_equals_input_k(self) -> None:
        """When target_k == input k, no merging happens; identity mapping."""
        from mbps_pytorch.stage4.hierarchical_merge import hierarchical_merge

        torch.manual_seed(3)
        centroids = torch.randn(6, 4)
        counts = torch.full((6,), 100.0)
        merged_centroids, mapping = hierarchical_merge(
            centroids, counts, target_k=6, frozen_indices=[]
        )
        assert merged_centroids.shape == centroids.shape
        # Mapping should be a permutation; verify identity-up-to-permutation
        # by checking that every old centroid is preserved bit-exact under its mapping
        for old_idx in range(6):
            assert torch.allclose(
                merged_centroids[mapping[old_idx]], centroids[old_idx]
            )

    def test_most_similar_non_frozen_pair_merges_first(self) -> None:
        """First merge happens between the two most-similar non-frozen centroids."""
        from mbps_pytorch.stage4.hierarchical_merge import hierarchical_merge

        # Centroids: 0 and 1 are nearly identical (most similar pair)
        # Centroids 2 and 3 are far apart
        centroids = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.001, 0.0],   # nearly identical to 0
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        counts = torch.tensor([100.0, 100.0, 100.0, 100.0])
        merged_centroids, mapping = hierarchical_merge(
            centroids, counts, target_k=3, frozen_indices=[]
        )
        # 0 and 1 should map to the same new id
        assert mapping[0].item() == mapping[1].item()
        # 2 and 3 should map to different new ids
        assert mapping[2].item() != mapping[3].item()
