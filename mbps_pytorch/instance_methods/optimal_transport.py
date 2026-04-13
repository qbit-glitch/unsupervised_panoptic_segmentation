"""Method 5: Optimal Transport instance decomposition (Sinkhorn).

Assigns pixels to instance prototypes via entropic OT. Prototypes are
discovered per-class via K-means on [features, depth, position].
"""

import numpy as np
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans

from .utils import dilation_reclaim

THING_IDS = set(range(11, 19))


def _sinkhorn(a, b, C, epsilon=0.1, n_iters=100):
    """Log-domain Sinkhorn-Knopp algorithm for entropic OT.

    Uses log-domain computation to avoid numerical overflow with large
    cost matrices.

    Args:
        a: (N,) source distribution.
        b: (K,) target distribution.
        C: (N, K) cost matrix.
        epsilon: entropic regularization.
        n_iters: number of iterations.

    Returns:
        T: (N, K) transport plan.
    """
    log_a = np.log(a + 1e-30)
    log_b = np.log(b + 1e-30)
    log_K = -C / (epsilon + 1e-10)

    # Log-domain Sinkhorn iterations
    f = np.zeros_like(a)  # dual variable for rows
    g = np.zeros_like(b)  # dual variable for cols
    for _ in range(n_iters):
        # g = log_b - logsumexp(log_K + f[:, None], axis=0)
        log_Kf = log_K + f[:, None]
        g = log_b - _logsumexp(log_Kf, axis=0)
        # f = log_a - logsumexp(log_K + g[None, :], axis=1)
        log_Kg = log_K + g[None, :]
        f = log_a - _logsumexp(log_Kg, axis=1)

    # T[i,j] = exp(f[i] + log_K[i,j] + g[j])
    T = np.exp(f[:, None] + log_K + g[None, :])
    return T


def _logsumexp(x, axis):
    """Numerically stable logsumexp."""
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max_squeezed = np.squeeze(x_max, axis=axis)
    return x_max_squeezed + np.log(np.sum(np.exp(x - x_max), axis=axis) + 1e-30)


def sinkhorn_instances(semantic, depth, thing_ids=THING_IDS,
                       K_proto=15, epsilon=0.1, depth_scale=10.0,
                       min_area=1000, dilation_iters=3,
                       depth_blur_sigma=1.0, features=None):
    """OT-based instance decomposition.

    Args:
        semantic: (H,W) uint8 trainID map.
        depth: (H,W) float32 [0,1].
        thing_ids: set of thing class trainIDs.
        K_proto: number of instance prototypes per class.
        epsilon: entropic regularization strength.
        depth_scale: weight of depth in cost matrix.
        min_area: minimum instance area.
        dilation_iters: boundary reclamation.
        depth_blur_sigma: unused (kept for interface).
        features: (h, w, C) feature grid. Required.

    Returns:
        List of (mask, class_id, score).
    """
    if features is None:
        raise ValueError("OT method requires features")

    fh, fw, C = features.shape
    H, W = semantic.shape

    # Build position grid at feature resolution
    yy, xx = np.mgrid[0:fh, 0:fw]
    pos = np.stack([yy / fh, xx / fw], axis=-1)  # (fh, fw, 2)

    # Pool depth to feature resolution
    from PIL import Image
    depth_ds = np.array(
        Image.fromarray(depth).resize((fw, fh), Image.BILINEAR)
    )
    sem_ds = np.array(
        Image.fromarray(semantic).resize((fw, fh), Image.NEAREST)
    )

    raw_instances = []
    for cls in sorted(thing_ids):
        cls_mask = sem_ds == cls
        n_cls = cls_mask.sum()
        if n_cls < 3:
            continue

        # Build descriptors
        feats_cls = features[cls_mask]  # (n_cls, C)
        depth_cls = depth_ds[cls_mask][:, None] * depth_scale  # (n_cls, 1)
        pos_cls = pos[cls_mask]  # (n_cls, 2)
        desc = np.concatenate([feats_cls, depth_cls, pos_cls], axis=1)

        k = min(K_proto, n_cls)
        if k < 2:
            # Single instance for this class
            full_mask = semantic == cls
            area = float(full_mask.sum())
            if area >= min_area:
                raw_instances.append((full_mask, cls, area))
            continue

        # K-means prototypes
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=min(256, n_cls),
                                n_init=1, max_iter=50, random_state=42)
        kmeans.fit(desc)
        prototypes = kmeans.cluster_centers_  # (k, D)

        # Cost matrix
        cost = np.sum((desc[:, None, :] - prototypes[None, :, :]) ** 2,
                      axis=2)  # (n_cls, k)

        # Sinkhorn
        a = np.ones(n_cls) / n_cls
        b = np.ones(k) / k
        T = _sinkhorn(a, b, cost, epsilon=epsilon)

        # Assignment
        assignments = T.argmax(axis=1)  # (n_cls,)

        # Convert to full-res pixel masks
        cls_pixels = np.argwhere(cls_mask)  # (n_cls, 2) in ds coords
        scale_y, scale_x = H / fh, W / fw

        for proto_id in range(k):
            proto_mask = assignments == proto_id
            if proto_mask.sum() == 0:
                continue

            # Map ds coords to full res
            full_mask = np.zeros((H, W), dtype=bool)
            for py, px in cls_pixels[proto_mask]:
                y0, y1 = int(py * scale_y), int((py + 1) * scale_y)
                x0, x1 = int(px * scale_x), int((px + 1) * scale_x)
                full_mask[y0:y1, x0:x1] = True
            full_mask &= (semantic == cls)

            # Connected components (handle non-contiguous)
            labeled, n_cc = ndimage.label(full_mask)
            for cc_id in range(1, n_cc + 1):
                cc_mask = labeled == cc_id
                area = float(cc_mask.sum())
                if area >= min_area:
                    raw_instances.append((cc_mask, cls, area))

    return dilation_reclaim(raw_instances, semantic, thing_ids,
                            min_area=min_area,
                            dilation_iters=dilation_iters)
