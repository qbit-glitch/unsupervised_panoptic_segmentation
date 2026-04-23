"""Step H -- GSID: Grassmannian Subspace Instance Discrimination.

Alternative to Step B's cosine similarity for instance merging.
Computes Grassmannian distance between per-instance feature subspaces:

    d_G(V_a, V_b) = sqrt(sum_i theta_i^2)
    theta_i = arccos(sigma_i(V_a^T V_b))

where V_a, V_b are rank-r right singular vector bases of the centered
feature matrices. Small d_G means the instances lie in similar feature
subspaces -> merge. More discriminative than mean-vector cosine for
instances with diverse features (e.g., rider on bicycle).

Target: Fix rider regression from Step B by better discrimination.
"""

import logging

import numpy as np
from PIL import Image
from scipy import ndimage

logger = logging.getLogger(__name__)

FEAT_H, FEAT_W = 32, 64
NUM_CLASSES = 19


def grassmannian_distance(
    V_a: np.ndarray, V_b: np.ndarray
) -> float:
    """Grassmannian distance between two subspaces.

    Args:
        V_a: (D, r_a) orthonormal basis for subspace A.
        V_b: (D, r_b) orthonormal basis for subspace B.

    Returns:
        Grassmannian distance (non-negative). 0 = identical subspaces.
    """
    r = min(V_a.shape[1], V_b.shape[1])
    M = V_a[:, :r].T @ V_b[:, :r]  # (r, r)
    sigmas = np.linalg.svd(M, compute_uv=False)
    sigmas = np.clip(sigmas, -1.0, 1.0)
    angles = np.arccos(sigmas)
    return float(np.sqrt(np.sum(angles**2)))


def step_h(
    semantic: np.ndarray,
    instance: np.ndarray,
    features: np.ndarray,
    cluster_to_class: np.ndarray,
    subspace_rank: int = 5,
    grass_threshold: float = 0.8,
    min_patches: int = 6,
    dilate_px: int = 3,
) -> tuple:
    """Merge adjacent same-class instances using Grassmannian subspace distance.

    Args:
        semantic: (H, W) uint8 cluster IDs.
        instance: (H, W) uint16 instance IDs.
        features: (N_patches, D) L2-normalized DINOv3 features.
        cluster_to_class: (256,) cluster -> trainID LUT.
        subspace_rank: SVD rank for subspace representation.
        grass_threshold: Max Grassmannian distance to merge.
        min_patches: Min patches per instance for subspace.
        dilate_px: Dilation for adjacency.

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

    # Per-instance: majority class + subspace basis
    inst_class = {}
    inst_subspace = {}

    for iid in instance_ids:
        mask = instance == iid
        tids = mapped_sem[mask]
        valid = tids[tids < NUM_CLASSES]
        if len(valid) == 0:
            continue
        inst_class[iid] = int(
            np.bincount(valid, minlength=NUM_CLASSES).argmax()
        )

        mask_s = inst_small == iid
        n_patches = int(mask_s.sum())
        if n_patches < min_patches:
            continue

        # Centered feature matrix -> SVD -> right singular vectors
        F = feat_2d[mask_s].astype(np.float64)
        F = F - F.mean(axis=0, keepdims=True)
        r = min(subspace_rank, n_patches - 1, F.shape[1])
        if r < 1:
            continue

        try:
            _, _, Vt = np.linalg.svd(F, full_matrices=False)
            inst_subspace[iid] = Vt[:r, :].T  # (D, r) column basis
        except np.linalg.LinAlgError:
            continue

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

    # Merge criterion: same class + small Grassmannian distance
    merge_pairs = []
    for i, j in adjacency:
        if inst_class.get(i) != inst_class.get(j):
            continue
        if i not in inst_subspace or j not in inst_subspace:
            # Fall back to cosine for instances too small for subspace
            continue
        d_g = grassmannian_distance(inst_subspace[i], inst_subspace[j])
        if d_g < grass_threshold:
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
