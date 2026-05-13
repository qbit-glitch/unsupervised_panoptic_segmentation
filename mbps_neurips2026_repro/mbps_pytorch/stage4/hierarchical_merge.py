"""Hierarchical density-frozen cluster merging.

Merges k-means centroids by cosine similarity, protecting low-population
centroids from being merged. The "rare modes" (smallest n_freeze populations)
survive untouched, which prevents Hungarian alignment from collapsing
dead-class candidates into their high-density neighbours.

Reference: t-NEB-style maximum-density-path merging (arXiv:2503.15582), with
the addition of explicit rare-mode freezing.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def identify_rare_modes(pixel_counts: torch.Tensor, n_freeze: int) -> list[int]:
    """Return indices of the n_freeze centroids with the smallest pixel counts.

    Args:
        pixel_counts: (k,) tensor of per-cluster pixel populations.
        n_freeze: how many low-population centroids to flag as rare modes.

    Returns:
        Sorted list of indices to freeze. Caps at the total number of
        centroids when n_freeze exceeds it.
    """
    if n_freeze <= 0:
        return []
    k = pixel_counts.shape[0]
    n = min(n_freeze, k)
    smallest = torch.argsort(pixel_counts)[:n]
    return sorted(smallest.tolist())


def _resolve_root(parent: list[int], x: int) -> int:
    """Path-compressing union-find root lookup."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def hierarchical_merge(
    centroids: torch.Tensor,
    counts: torch.Tensor,
    target_k: int,
    frozen_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy agglomerative cosine-similarity merge with frozen-mode protection.

    At each iteration, the two most-similar non-frozen centroids are merged
    via population-weighted averaging. Frozen centroids never participate in
    a merge and survive unchanged.

    Args:
        centroids: (k, d) tensor of cluster centroids.
        counts: (k,) tensor of per-cluster pixel populations.
        target_k: desired final cluster cardinality.
        frozen_indices: list of centroid indices that must NOT be merged.

    Returns:
        Tuple of:
            merged_centroids: (target_k, d) tensor of surviving centroids.
            mapping: (k,) long tensor mapping each old centroid index to its
                new compact index in [0, target_k).

    Raises:
        ValueError: if target_k > input k or target_k < len(frozen_indices).
    """
    if centroids.dim() != 2:
        raise ValueError(f"centroids must be (k, d), got {tuple(centroids.shape)}")
    if counts.dim() != 1 or counts.shape[0] != centroids.shape[0]:
        raise ValueError(
            f"counts shape {tuple(counts.shape)} mismatched with centroids "
            f"k={centroids.shape[0]}"
        )

    k = centroids.shape[0]
    if target_k > k:
        raise ValueError(f"target_k ({target_k}) cannot exceed input k ({k})")
    if target_k < len(frozen_indices):
        raise ValueError(
            f"target_k ({target_k}) cannot be smaller than the number of "
            f"frozen centroids ({len(frozen_indices)})"
        )

    frozen_set = set(int(i) for i in frozen_indices)
    parent = list(range(k))
    # active[i] holds (centroid_tensor, count) for each surviving root i
    active: dict[int, tuple[torch.Tensor, float]] = {
        i: (centroids[i].clone(), float(counts[i].item()))
        for i in range(k)
    }

    while len(active) > target_k:
        # Find the most-similar non-frozen pair via cosine similarity
        active_ids = list(active.keys())
        non_frozen_ids = [i for i in active_ids if i not in frozen_set]
        if len(non_frozen_ids) < 2:
            logger.warning(
                "Only %d non-frozen centroids remain; cannot reduce further.",
                len(non_frozen_ids),
            )
            break

        # Stack non-frozen centroids and compute pairwise cosine similarity
        stacked = torch.stack([active[i][0] for i in non_frozen_ids])
        normalized = F.normalize(stacked, dim=-1)
        sim_matrix = normalized @ normalized.T
        # Mask diagonal to -inf so a centroid is never merged with itself
        sim_matrix.fill_diagonal_(-float("inf"))

        # Take upper triangle so each pair is counted once
        n = sim_matrix.shape[0]
        triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        sim_matrix = torch.where(triu_mask, sim_matrix, torch.tensor(-float("inf")))

        flat_idx = sim_matrix.argmax().item()
        ii, jj = divmod(flat_idx, n)
        i_global = non_frozen_ids[ii]
        j_global = non_frozen_ids[jj]

        # Population-weighted merge
        ci, ni = active[i_global]
        cj, nj = active[j_global]
        total = ni + nj
        merged_centroid = (ci * ni + cj * nj) / total
        active[i_global] = (merged_centroid, total)
        del active[j_global]
        parent[j_global] = i_global

    # Compact root ids to [0, target_k)
    roots = [_resolve_root(parent, i) for i in range(k)]
    unique_roots = sorted(set(roots))
    root_to_new = {r: new_id for new_id, r in enumerate(unique_roots)}

    mapping = torch.tensor(
        [root_to_new[roots[i]] for i in range(k)], dtype=torch.long
    )

    merged_centroids = torch.zeros(
        len(unique_roots), centroids.shape[1], dtype=centroids.dtype
    )
    for r, new_id in root_to_new.items():
        merged_centroids[new_id] = active[r][0]

    return merged_centroids, mapping
