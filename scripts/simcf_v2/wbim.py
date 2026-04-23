"""Step E -- WBIM: Wasserstein Barycentric Instance Merging.

Replaces Step B's cosine similarity with Sliced 1-Wasserstein distance
between per-instance feature distributions:

    SW_1(mu_a, mu_b) = E_theta[W_1(theta^T # mu_a, theta^T # mu_b)]

Uses L random projections onto 1D for efficient approximation. Merges
instances with low distributional distance, which is more robust than
comparing mean vectors when feature distributions are multimodal.

Target: Reduce car FP from over-aggressive cosine merging.
"""

import logging

import numpy as np
from PIL import Image
from scipy import ndimage

logger = logging.getLogger(__name__)

FEAT_H, FEAT_W = 32, 64
NUM_CLASSES = 19


def sliced_wasserstein_1d(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    n_projections: int = 50,
    seed: int = 42,
) -> float:
    """Compute Sliced 1-Wasserstein distance between two point sets.

    Args:
        samples_a: (n_a, D) feature vectors.
        samples_b: (n_b, D) feature vectors.
        n_projections: Number of random projections.
        seed: Random seed for reproducibility.

    Returns:
        Approximate SW_1 distance (non-negative).
    """
    D = samples_a.shape[1]
    rng = np.random.RandomState(seed)

    # Random unit directions
    directions = rng.randn(n_projections, D)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8

    proj_a = samples_a @ directions.T  # (n_a, L)
    proj_b = samples_b @ directions.T  # (n_b, L)

    n_a, n_b = len(samples_a), len(samples_b)
    n_common = max(n_a, n_b)
    total_w1 = 0.0

    for l_idx in range(n_projections):
        a_sorted = np.sort(proj_a[:, l_idx])
        b_sorted = np.sort(proj_b[:, l_idx])

        # Resample to common length via linear interpolation
        if n_a != n_common:
            a_sorted = np.interp(
                np.linspace(0, 1, n_common),
                np.linspace(0, 1, n_a),
                a_sorted,
            )
        if n_b != n_common:
            b_sorted = np.interp(
                np.linspace(0, 1, n_common),
                np.linspace(0, 1, n_b),
                b_sorted,
            )

        total_w1 += np.mean(np.abs(a_sorted - b_sorted))

    return total_w1 / n_projections


def step_e(
    semantic: np.ndarray,
    instance: np.ndarray,
    features: np.ndarray,
    cluster_to_class: np.ndarray,
    n_projections: int = 50,
    sw_threshold: float = 0.3,
    dilate_px: int = 3,
) -> tuple:
    """Merge adjacent same-class instances using Sliced Wasserstein distance.

    Args:
        semantic: (H, W) uint8 cluster IDs.
        instance: (H, W) uint16 instance IDs.
        features: (N_patches, D) L2-normalized DINOv3 features.
        cluster_to_class: (256,) cluster -> trainID LUT.
        n_projections: Number of random projections for SW_1.
        sw_threshold: Max SW_1 distance to trigger merge.
        dilate_px: Dilation pixels for adjacency.

    Returns:
        (modified_instance, n_merges)
    """
    inst_small = np.array(
        Image.fromarray(instance).resize((FEAT_W, FEAT_H), Image.NEAREST)
    )
    feat_2d = features.reshape(FEAT_H, FEAT_W, -1)
    mapped_sem = cluster_to_class[semantic]

    instance_ids = np.unique(instance)
    instance_ids = instance_ids[instance_ids > 0]
    if len(instance_ids) < 2:
        return instance, 0

    # Per-instance: majority class + feature set
    inst_class = {}
    inst_feat_sets = {}

    for iid in instance_ids:
        mask = instance == iid
        tids = mapped_sem[mask]
        valid = tids[tids < NUM_CLASSES]
        if len(valid) == 0:
            continue
        inst_class[iid] = int(np.bincount(valid, minlength=NUM_CLASSES).argmax())

        mask_s = inst_small == iid
        if mask_s.sum() < 2:  # need >= 2 patches for distribution
            continue
        inst_feat_sets[iid] = feat_2d[mask_s]  # (n_patches, D)

    # Build adjacency via dilation overlap
    struct = ndimage.generate_binary_structure(2, 1)
    adjacency = set()
    for iid in instance_ids:
        if iid not in inst_class:
            continue
        mask = instance == iid
        dilated = ndimage.binary_dilation(
            mask, structure=struct, iterations=dilate_px
        )
        border = dilated & ~mask
        for nb in np.unique(instance[border]):
            if nb == 0 or nb == iid or nb not in inst_class:
                continue
            adjacency.add((min(iid, nb), max(iid, nb)))

    # Merge criterion: same class + low Wasserstein distance
    merge_pairs = []
    for i, j in adjacency:
        if inst_class.get(i) != inst_class.get(j):
            continue
        if i not in inst_feat_sets or j not in inst_feat_sets:
            continue
        sw_dist = sliced_wasserstein_1d(
            inst_feat_sets[i], inst_feat_sets[j], n_projections
        )
        if sw_dist < sw_threshold:
            merge_pairs.append((i, j))

    if not merge_pairs:
        return instance, 0

    # Union-find with path compression
    parent = {iid: iid for iid in instance_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in merge_pairs:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi

    # Renumber contiguously
    new_instance = np.zeros_like(instance)
    root_to_new = {}
    next_id = 1
    for iid in sorted(instance_ids):
        root = find(iid)
        if root not in root_to_new:
            root_to_new[root] = next_id
            next_id += 1
        new_instance[instance == iid] = root_to_new[root]

    n_merges = len(instance_ids) - len(root_to_new)
    return new_instance, n_merges
