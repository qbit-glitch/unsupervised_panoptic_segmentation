"""Step D -- SDAIR: Spectral Depth-Aware Instance Refinement.

Splits over-merged instances using spectral graph partitioning with a
depth-feature product kernel:

    w_ij = exp(-||f_i - f_j||^2 / 2*sigma_f^2) * exp(-|d_i - d_j|^2 / 2*sigma_d^2)

For each sufficiently large instance:
  1. Build affinity matrix using the product kernel
  2. Compute normalized graph Laplacian L_norm = I - D^{-1/2} W D^{-1/2}
  3. Find Fiedler vector (2nd eigenvector of L_norm)
  4. Split along Fiedler sign if Fiedler value < threshold
  5. Recurse on children until stable

Target: Person PQ 2.6 -> 10+ by separating co-planar pedestrians.
"""

import logging

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

FEAT_H, FEAT_W = 32, 64


def _spectral_bipartition(
    feat_vecs: np.ndarray,
    depth_vals: np.ndarray,
    sigma_f: float,
    sigma_d: float,
    fiedler_threshold: float,
    min_split_ratio: float,
) -> tuple:
    """Attempt spectral bipartition of a patch set.

    Args:
        feat_vecs: (n, D) L2-normalized feature vectors.
        depth_vals: (n,) depth values.
        sigma_f: Feature kernel bandwidth.
        sigma_d: Depth kernel bandwidth.
        fiedler_threshold: Max Fiedler value to trigger split.
        min_split_ratio: Min fraction in smaller group to accept split.

    Returns:
        (should_split, group_labels) where group_labels[i] in {0, 1}.
    """
    n = len(feat_vecs)
    if n < 4:
        return False, None

    # Affinity matrix: product of feature and depth Gaussian kernels
    f_sq = cdist(feat_vecs, feat_vecs, metric="sqeuclidean")
    d_sq = (depth_vals[:, None] - depth_vals[None, :]) ** 2
    W = np.exp(-f_sq / (2 * sigma_f**2)) * np.exp(-d_sq / (2 * sigma_d**2))
    np.fill_diagonal(W, 0.0)

    # Normalized graph Laplacian: L_norm = I - D^{-1/2} W D^{-1/2}
    D_diag = W.sum(axis=1)
    D_inv_sqrt = np.where(D_diag > 1e-10, 1.0 / np.sqrt(D_diag), 0.0)
    L_norm = np.eye(n) - (D_inv_sqrt[:, None] * W * D_inv_sqrt[None, :])
    L_norm = 0.5 * (L_norm + L_norm.T)  # symmetrize for numerical stability

    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

    fiedler_val = eigenvalues[1]
    fiedler_vec = eigenvectors[:, 1]

    if fiedler_val >= fiedler_threshold:
        return False, None

    # Bipartition along Fiedler vector sign
    groups = (fiedler_vec >= 0).astype(np.int32)
    ratio = min(groups.sum(), n - groups.sum()) / n

    if ratio < min_split_ratio:
        return False, None

    return True, groups


def _recursive_split(
    patch_indices: np.ndarray,
    feat_vecs: np.ndarray,
    depth_vals: np.ndarray,
    sigma_f: float,
    sigma_d: float,
    fiedler_threshold: float,
    min_split_ratio: float,
    min_patches: int,
    max_depth: int,
    current_depth: int,
) -> list:
    """Recursively split patches until stable.

    Returns:
        List of patch index arrays (one per final group).
    """
    if current_depth >= max_depth or len(patch_indices) < min_patches:
        return [patch_indices]

    should_split, groups = _spectral_bipartition(
        feat_vecs, depth_vals, sigma_f, sigma_d,
        fiedler_threshold, min_split_ratio,
    )

    if not should_split:
        return [patch_indices]

    result = []
    for gid in [0, 1]:
        mask = groups == gid
        sub_splits = _recursive_split(
            patch_indices[mask], feat_vecs[mask], depth_vals[mask],
            sigma_f, sigma_d, fiedler_threshold, min_split_ratio,
            min_patches, max_depth, current_depth + 1,
        )
        result.extend(sub_splits)

    return result


NUM_CLASSES = 19
# Stuff classes should never be split (road, sidewalk, building, wall,
# fence, pole, traffic light, traffic sign, vegetation, terrain, sky)
_STUFF_IDS = set(range(0, 11))


def step_d(
    semantic: np.ndarray,
    instance: np.ndarray,
    features: np.ndarray,
    depth: np.ndarray,
    cluster_to_class: np.ndarray | None = None,
    min_instance_patches: int = 5,
    sigma_f: float = 0.3,
    sigma_d: float = 0.1,
    fiedler_threshold: float = 0.15,
    min_split_ratio: float = 0.15,
    max_recursion: int = 3,
) -> tuple:
    """Split over-merged instances via spectral depth-aware partitioning.

    Args:
        semantic: (H, W) uint8 cluster IDs (read-only).
        instance: (H, W) uint16 instance IDs.
        features: (N_patches, D) L2-normalized DINOv3 features.
        depth: (H, W) float depth map.
        cluster_to_class: (256,) cluster -> trainID LUT. If provided,
            stuff-class instances are skipped (only things are split).
        min_instance_patches: Min patches at 32x64 to consider splitting.
        sigma_f: Feature kernel bandwidth.
        sigma_d: Depth kernel bandwidth.
        fiedler_threshold: Max Fiedler eigenvalue to trigger split.
        min_split_ratio: Min size ratio for the smaller group.
        max_recursion: Max recursion depth per instance.

    Returns:
        (modified_instance, n_splits)
    """
    H, W = instance.shape
    feat_2d = features.reshape(FEAT_H, FEAT_W, -1)

    # Downsample to patch resolution
    inst_small = np.array(
        Image.fromarray(instance).resize((FEAT_W, FEAT_H), Image.NEAREST)
    )
    depth_small = np.array(
        Image.fromarray(depth.astype(np.float32)).resize(
            (FEAT_W, FEAT_H), Image.BILINEAR
        )
    ).astype(np.float64)

    # Precompute mapped semantics for stuff filtering
    mapped_sem = None
    if cluster_to_class is not None:
        mapped_sem = cluster_to_class[semantic]

    instance_ids = np.unique(instance)
    instance_ids = instance_ids[instance_ids > 0]

    new_instance = instance.copy()
    next_id = int(instance_ids.max()) + 1 if len(instance_ids) > 0 else 1
    n_splits = 0

    for iid in instance_ids:
        # Skip stuff-class instances (road, sky, etc. should not be split)
        if mapped_sem is not None:
            mask = instance == iid
            tids = mapped_sem[mask]
            valid = tids[tids < NUM_CLASSES]
            if len(valid) > 0:
                majority_cls = int(
                    np.bincount(valid, minlength=NUM_CLASSES).argmax()
                )
                if majority_cls in _STUFF_IDS:
                    continue

        patch_mask = inst_small == iid
        n_patches = int(patch_mask.sum())

        if n_patches < min_instance_patches:
            continue

        # Extract per-patch features and depth
        patch_indices = np.argwhere(patch_mask)  # (n, 2)
        feat_vecs = feat_2d[patch_indices[:, 0], patch_indices[:, 1]]
        depth_vals = depth_small[patch_indices[:, 0], patch_indices[:, 1]]

        splits = _recursive_split(
            patch_indices, feat_vecs, depth_vals,
            sigma_f, sigma_d, fiedler_threshold, min_split_ratio,
            min_instance_patches, max_recursion, current_depth=0,
        )

        if len(splits) <= 1:
            continue

        # Map splits back to full resolution
        full_mask = new_instance == iid
        scale_h = H / FEAT_H
        scale_w = W / FEAT_W

        for group_idx, group_patches in enumerate(splits):
            if group_idx == 0:
                continue  # first group keeps original ID

            # Build full-resolution mask from patch coordinates
            group_mask_full = np.zeros((H, W), dtype=bool)
            for pi in group_patches:
                r_lo = int(pi[0] * scale_h)
                r_hi = int((pi[0] + 1) * scale_h)
                c_lo = int(pi[1] * scale_w)
                c_hi = int((pi[1] + 1) * scale_w)
                group_mask_full[r_lo:r_hi, c_lo:c_hi] = True

            assign_mask = full_mask & group_mask_full
            if assign_mask.any():
                new_instance[assign_mask] = next_id
                next_id += 1
                n_splits += 1

    return new_instance, n_splits
