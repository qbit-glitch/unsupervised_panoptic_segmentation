"""Method 2: Mumford-Shah energy minimization in depth-feature space.

E(Π) = Σ_i [α·Var_depth(R_i) + β·Var_feature(R_i)] + γ·|∂Π|

Solved via spectral clustering on an affinity graph constructed from depth
and DINOv2 feature similarity. Graph-cut (alpha-expansion) is an alternative
if pymaxflow is available.
"""

import numpy as np
from scipy import ndimage
from sklearn.cluster import SpectralClustering

from .utils import dilation_reclaim

THING_IDS = set(range(11, 19))


def _build_affinity_matrix(depth_ds, features_ds, cls_mask,
                           alpha=1.0, beta=0.1, sigma_depth=0.05,
                           sigma_feat=1.0):
    """Build affinity matrix for pixels within a class mask (vectorized).

    A[i,j] = exp(-alpha * |d_i - d_j|^2 / sigma_d^2
                 - beta * ||f_i - f_j||^2 / sigma_f^2)

    Only for 4-connected neighbors (sparse).

    Args:
        depth_ds: (h, w) depth at working resolution.
        features_ds: (h, w, C) features at working resolution.
        cls_mask: (h, w) bool mask for this class.

    Returns:
        affinity: (N, N) sparse-compatible affinity matrix.
        pixel_indices: (N, 2) array of (y, x) coords.
    """
    from scipy.sparse import coo_matrix

    h, w = depth_ds.shape
    ys, xs = np.where(cls_mask)
    N = len(ys)

    if N == 0:
        return np.zeros((0, 0)), np.zeros((0, 2), dtype=int)

    # Map (y,x) -> index
    idx_map = np.full((h, w), -1, dtype=np.int32)
    idx_map[ys, xs] = np.arange(N, dtype=np.int32)

    depths = depth_ds[ys, xs]
    feats = features_ds[ys, xs]  # (N, C)

    # Vectorized neighbor lookup for all 4 directions
    row_list, col_list, val_list = [], [], []
    for dy, dx in [(0, 1), (1, 0)]:  # Only 2 directions (symmetric)
        ny = ys + dy
        nx = xs + dx
        valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        i_idx = np.where(valid)[0]
        j_idx = idx_map[ny[valid], nx[valid]]
        # Filter to pairs where neighbor is also in cls_mask
        in_mask = j_idx >= 0
        i_idx = i_idx[in_mask]
        j_idx = j_idx[in_mask]

        if len(i_idx) == 0:
            continue

        d_diff = (depths[i_idx] - depths[j_idx]) ** 2
        f_diff = np.sum((feats[i_idx] - feats[j_idx]) ** 2, axis=1)
        w_ij = np.exp(
            -alpha * d_diff / (sigma_depth ** 2 + 1e-8)
            - beta * f_diff / (sigma_feat ** 2 + 1e-8)
        )

        # Add both directions
        row_list.extend([i_idx, j_idx])
        col_list.extend([j_idx, i_idx])
        val_list.extend([w_ij, w_ij])

    if row_list:
        rows = np.concatenate(row_list)
        cols = np.concatenate(col_list)
        vals = np.concatenate(val_list)
        aff = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    else:
        aff = coo_matrix((N, N)).tocsr()

    pixel_indices = np.stack([ys, xs], axis=1)
    return aff, pixel_indices


def mumford_shah_instances(semantic, depth, thing_ids=THING_IDS,
                           alpha=1.0, beta=0.1, gamma=2.0,
                           sigma_depth=0.05, sigma_feat=1.0,
                           n_clusters=20, work_resolution=(64, 128),
                           min_area=1000, dilation_iters=3,
                           depth_blur_sigma=1.0, features=None):
    """Mumford-Shah energy instance decomposition via spectral clustering.

    Args:
        semantic: (H,W) uint8 trainID map.
        depth: (H,W) float32 [0,1].
        thing_ids: set of thing class trainIDs.
        alpha: depth variance weight.
        beta: feature variance weight.
        gamma: boundary length penalty (unused in spectral, controls n_clusters).
        sigma_depth: depth kernel bandwidth.
        sigma_feat: feature kernel bandwidth.
        n_clusters: max number of instances per class (spectral clustering k).
        work_resolution: (h, w) to downsample for tractability.
        min_area: minimum instance area.
        dilation_iters: boundary reclamation.
        depth_blur_sigma: unused.
        features: (fh, fw, C) feature grid. Required.

    Returns:
        List of (mask, class_id, score).
    """
    if features is None:
        raise ValueError("Mumford-Shah method requires features")

    H, W = semantic.shape
    wh, ww = work_resolution

    # Downsample everything to work resolution
    from PIL import Image
    depth_ds = np.array(
        Image.fromarray(depth).resize((ww, wh), Image.BILINEAR)
    )
    sem_ds = np.array(
        Image.fromarray(semantic).resize((ww, wh), Image.NEAREST)
    )

    fh, fw, C = features.shape
    if fh == wh and fw == ww:
        feat_ds = features
    else:
        # Use scipy zoom for efficient multi-channel resize
        from scipy.ndimage import zoom
        zoom_y, zoom_x = wh / fh, ww / fw
        feat_ds = zoom(features, (zoom_y, zoom_x, 1), order=1).astype(np.float32)

    raw_instances = []
    for cls in sorted(thing_ids):
        cls_mask = sem_ds == cls
        n_pixels = cls_mask.sum()
        if n_pixels < 10:
            continue

        k = min(n_clusters, max(2, n_pixels // 50))

        # Subsample if too many pixels (spectral clustering is O(N^3))
        max_pixels = 2000
        if n_pixels > max_pixels:
            # Strided subsample — preserves local connectivity better
            # than random sampling (keeps 4-connected structure intact)
            all_ys, all_xs = np.where(cls_mask)
            stride = max(1, n_pixels // max_pixels)
            idx = np.arange(0, n_pixels, stride)[:max_pixels]
            sub_mask = np.zeros_like(cls_mask)
            sub_mask[all_ys[idx], all_xs[idx]] = True
            work_mask = sub_mask
            n_pixels = len(idx)
        else:
            work_mask = cls_mask

        aff, pixel_indices = _build_affinity_matrix(
            depth_ds, feat_ds, work_mask,
            alpha=alpha, beta=beta,
            sigma_depth=sigma_depth, sigma_feat=sigma_feat,
        )

        if aff.shape[0] < 2:
            continue

        try:
            import warnings
            sc = SpectralClustering(
                n_clusters=k, affinity="precomputed",
                assign_labels="kmeans", random_state=42,
                n_init=1,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                                        message="Graph is not fully connected")
                labels = sc.fit_predict(aff.toarray())
        except Exception:
            continue

        # Convert back to full resolution masks
        scale_y, scale_x = H / wh, W / ww
        for label_id in range(k):
            cluster_pixels = pixel_indices[labels == label_id]
            if len(cluster_pixels) == 0:
                continue

            full_mask = np.zeros((H, W), dtype=bool)
            for py, px in cluster_pixels:
                y0, y1 = int(py * scale_y), int((py + 1) * scale_y)
                x0, x1 = int(px * scale_x), int((px + 1) * scale_x)
                full_mask[y0:y1, x0:x1] = True
            full_mask &= (semantic == cls)

            # Connected components
            labeled, n_cc = ndimage.label(full_mask)
            for cc_id in range(1, n_cc + 1):
                cc_mask = labeled == cc_id
                area = float(cc_mask.sum())
                if area >= min_area:
                    raw_instances.append((cc_mask, cls, area))

    return dilation_reclaim(raw_instances, semantic, thing_ids,
                            min_area=min_area,
                            dilation_iters=dilation_iters)
