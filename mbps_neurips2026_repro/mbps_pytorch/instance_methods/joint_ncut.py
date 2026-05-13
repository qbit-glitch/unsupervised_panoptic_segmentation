"""Approach #2: Depth-Feature Joint Normalized Cut.

Builds a joint affinity matrix from DA3 depth and DINOv2 features,
then applies recursive 2-way normalized cut (Fiedler vector bipartition)
to decompose thing-class regions into instances.

Inspired by DiffCut (NeurIPS 2024) and Shi & Malik (2000).
"""

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from .utils import dilation_reclaim

THING_IDS = set(range(11, 19))


def _build_joint_affinity(
    depth_ds: np.ndarray,
    features_ds: np.ndarray,
    cls_mask: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    sigma_d: float = None,
    sigma_f: float = None,
) -> tuple:
    """Build sparse joint affinity matrix for 4-connected pixels.

    A[i,j] = exp(-alpha * |d_i-d_j|^2 / sigma_d^2
                 - beta * ||f_i-f_j||^2 / sigma_f^2)

    Args:
        depth_ds: (h, w) depth at working resolution.
        features_ds: (h, w, C) features at working resolution.
        cls_mask: (h, w) bool mask for this class.
        alpha: depth weight.
        beta: feature weight.
        sigma_d: depth bandwidth (None = median heuristic).
        sigma_f: feature bandwidth (None = median heuristic).

    Returns:
        affinity: (N, N) sparse CSR affinity matrix.
        pixel_indices: (N, 2) array of (y, x) coords.
    """
    h, w = depth_ds.shape
    ys, xs = np.where(cls_mask)
    N = len(ys)

    if N < 2:
        return None, np.stack([ys, xs], axis=1) if N > 0 else np.zeros((0, 2))

    idx_map = np.full((h, w), -1, dtype=np.int32)
    idx_map[ys, xs] = np.arange(N, dtype=np.int32)

    depths = depth_ds[ys, xs]
    feats = features_ds[ys, xs]

    # Collect neighbor pairs for sigma estimation
    all_i, all_j = [], []
    for dy, dx in [(0, 1), (1, 0)]:
        ny, nx = ys + dy, xs + dx
        valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        i_idx = np.where(valid)[0]
        j_idx = idx_map[ny[valid], nx[valid]]
        in_mask = j_idx >= 0
        all_i.append(i_idx[in_mask])
        all_j.append(j_idx[in_mask])

    if not all_i or sum(len(x) for x in all_i) == 0:
        return None, np.stack([ys, xs], axis=1)

    i_all = np.concatenate(all_i)
    j_all = np.concatenate(all_j)

    d_diff_sq = (depths[i_all] - depths[j_all]) ** 2
    f_diff_sq = np.sum((feats[i_all] - feats[j_all]) ** 2, axis=1)

    # Median heuristic for bandwidths
    if sigma_d is None:
        median_d = np.median(d_diff_sq)
        sigma_d = np.sqrt(median_d + 1e-8)
    if sigma_f is None:
        median_f = np.median(f_diff_sq)
        sigma_f = np.sqrt(median_f + 1e-8)

    w_ij = np.exp(
        -alpha * d_diff_sq / (sigma_d ** 2 + 1e-8)
        - beta * f_diff_sq / (sigma_f ** 2 + 1e-8)
    )

    # Build symmetric sparse matrix
    rows = np.concatenate([i_all, j_all])
    cols = np.concatenate([j_all, i_all])
    vals = np.concatenate([w_ij, w_ij])

    aff = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    pixel_indices = np.stack([ys, xs], axis=1)
    return aff, pixel_indices


def _ncut_cost(affinity, labels):
    """Compute normalized cut cost for a bipartition.

    NCut(A, B) = cut(A,B)/assoc(A,V) + cut(A,B)/assoc(B,V)
    """
    mask_a = labels == 0
    mask_b = labels == 1
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return float("inf")

    aff_dense = affinity
    if hasattr(affinity, "toarray"):
        aff_dense = affinity.toarray()

    cut_ab = aff_dense[np.ix_(mask_a, mask_b)].sum()
    assoc_a = aff_dense[mask_a].sum()
    assoc_b = aff_dense[mask_b].sum()

    if assoc_a < 1e-10 or assoc_b < 1e-10:
        return float("inf")

    return cut_ab / assoc_a + cut_ab / assoc_b


def _recursive_ncut(
    affinity,
    pixel_indices: np.ndarray,
    ncut_threshold: float = 0.05,
    min_pixels: int = 10,
    max_depth: int = 8,
    depth: int = 0,
) -> list:
    """Recursively bipartition using normalized cut.

    Args:
        affinity: (N, N) sparse affinity matrix.
        pixel_indices: (N, 2) array of (y, x) coords.
        ncut_threshold: stop splitting if NCut cost > threshold.
        min_pixels: minimum partition size.
        max_depth: maximum recursion depth.
        depth: current recursion depth.

    Returns:
        List of pixel_indices arrays (one per partition).
    """
    N = affinity.shape[0]

    if N < min_pixels * 2 or depth >= max_depth:
        return [pixel_indices]

    # Compute degree matrix and normalized Laplacian
    # L_rw = I - D^{-1} W (random walk Laplacian)
    degrees = np.array(affinity.sum(axis=1)).flatten()
    degrees = np.maximum(degrees, 1e-10)
    D_inv = 1.0 / degrees

    # Find Fiedler vector (2nd smallest eigenvector of L_rw)
    # Equivalent to 2nd largest eigenvector of D^{-1/2} W D^{-1/2}
    D_inv_sqrt = np.sqrt(D_inv)

    # Build D^{-1/2} W D^{-1/2}
    from scipy.sparse import diags
    D_inv_sqrt_sparse = diags(D_inv_sqrt)
    L_sym = D_inv_sqrt_sparse @ affinity @ D_inv_sqrt_sparse

    try:
        # Find top-2 eigenvectors (largest eigenvalues)
        eigenvalues, eigenvectors = eigsh(L_sym, k=2, which="LM")
        # Fiedler vector is the one with the 2nd largest eigenvalue
        fiedler = eigenvectors[:, 0]  # 2nd eigenvector (eigsh returns ascending)
        # Transform back: v = D^{-1/2} u
        fiedler = D_inv_sqrt * fiedler
    except Exception:
        return [pixel_indices]

    # Bipartition by sign of Fiedler vector
    labels = (fiedler >= np.median(fiedler)).astype(int)

    # Check NCut cost
    cost = _ncut_cost(affinity, labels)
    if cost > ncut_threshold:
        return [pixel_indices]

    # Split
    mask_a = labels == 0
    mask_b = labels == 1

    if mask_a.sum() < min_pixels or mask_b.sum() < min_pixels:
        return [pixel_indices]

    results = []

    # Recurse on partition A
    idx_a = np.where(mask_a)[0]
    aff_a = affinity[np.ix_(idx_a, idx_a)]
    parts_a = _recursive_ncut(
        aff_a, pixel_indices[idx_a],
        ncut_threshold=ncut_threshold,
        min_pixels=min_pixels,
        max_depth=max_depth,
        depth=depth + 1,
    )
    results.extend(parts_a)

    # Recurse on partition B
    idx_b = np.where(mask_b)[0]
    aff_b = affinity[np.ix_(idx_b, idx_b)]
    parts_b = _recursive_ncut(
        aff_b, pixel_indices[idx_b],
        ncut_threshold=ncut_threshold,
        min_pixels=min_pixels,
        max_depth=max_depth,
        depth=depth + 1,
    )
    results.extend(parts_b)

    return results


def joint_ncut_instances(
    semantic: np.ndarray,
    depth_map: np.ndarray,
    thing_ids: set = THING_IDS,
    alpha: float = 1.0,
    beta: float = 1.0,
    ncut_threshold: float = 0.05,
    work_resolution: tuple = (32, 64),
    min_area: int = 1000,
    dilation_iters: int = 3,
    depth_blur_sigma: float = 1.0,
    features: np.ndarray = None,
) -> list:
    """Instance decomposition via recursive normalized cut on joint affinity.

    Args:
        semantic: (H,W) uint8 trainID map.
        depth_map: (H,W) float32 [0,1].
        thing_ids: set of thing class trainIDs.
        alpha: depth weight in affinity.
        beta: feature weight in affinity.
        ncut_threshold: NCut cost threshold for stopping recursion.
        work_resolution: (h, w) for tractable spectral computation.
        min_area: minimum instance area.
        dilation_iters: boundary reclamation iterations.
        depth_blur_sigma: Gaussian blur sigma for depth.
        features: (fh, fw, C) DINOv2 feature grid. Required.

    Returns:
        List of (mask, class_id, score).
    """
    if features is None:
        raise ValueError("Joint NCut method requires DINOv2 features")

    H, W = semantic.shape
    wh, ww = work_resolution

    # Downsample to working resolution
    from PIL import Image as PILImage

    depth_ds = np.array(
        PILImage.fromarray(depth_map.astype(np.float32)).resize(
            (ww, wh), PILImage.BILINEAR
        )
    )
    sem_ds = np.array(
        PILImage.fromarray(semantic).resize((ww, wh), PILImage.NEAREST)
    )

    # Resize features to working resolution
    fh, fw, C = features.shape
    if fh == wh and fw == ww:
        feat_ds = features
    else:
        from scipy.ndimage import zoom
        zoom_y, zoom_x = wh / fh, ww / fw
        feat_ds = zoom(features, (zoom_y, zoom_x, 1), order=1).astype(
            np.float32
        )

    raw_instances = []
    scale_y, scale_x = H / wh, W / ww
    min_pixels = max(5, min_area // int(scale_y * scale_x))

    for cls in sorted(thing_ids):
        cls_mask = sem_ds == cls
        n_pixels = cls_mask.sum()
        if n_pixels < min_pixels * 2:
            continue

        affinity, pixel_indices = _build_joint_affinity(
            depth_ds, feat_ds, cls_mask,
            alpha=alpha, beta=beta,
        )

        if affinity is None or affinity.shape[0] < min_pixels * 2:
            continue

        # Recursive NCut
        partitions = _recursive_ncut(
            affinity, pixel_indices,
            ncut_threshold=ncut_threshold,
            min_pixels=min_pixels,
        )

        # Convert partitions to full-resolution masks
        for part_pixels in partitions:
            if len(part_pixels) < min_pixels:
                continue

            full_mask = np.zeros((H, W), dtype=bool)
            for py, px in part_pixels:
                y0 = int(py * scale_y)
                y1 = int((py + 1) * scale_y)
                x0 = int(px * scale_x)
                x1 = int((px + 1) * scale_x)
                full_mask[y0:y1, x0:x1] = True
            full_mask &= (semantic == cls)

            # Connected components to handle disconnected patches
            labeled, n_cc = ndimage.label(full_mask)
            for cc_id in range(1, n_cc + 1):
                cc_mask = labeled == cc_id
                area = float(cc_mask.sum())
                if area >= min_area:
                    raw_instances.append((cc_mask, cls, area))

    return dilation_reclaim(
        raw_instances, semantic, thing_ids,
        min_area=min_area, dilation_iters=dilation_iters,
    )
