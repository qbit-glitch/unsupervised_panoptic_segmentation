"""Method 4: Contrastive Depth-Feature Embedding + HDBSCAN.

Learns a projection head that maps [DINOv2 features + depth + position] into
a 128-dim embedding where same-instance patches are close and different-instance
patches are far. Depth discontinuities provide weak self-supervision.
Inference: cluster embeddings with HDBSCAN per thing class.

NOTE: Training is done via train_contrastive_embed.py. This module handles
inference only (with a pre-trained checkpoint) or raw clustering without training.
"""

import numpy as np
from scipy import ndimage
from sklearn.cluster import HDBSCAN

from .utils import dilation_reclaim

THING_IDS = set(range(11, 19))


def _raw_feature_cluster(features, sem_ds, cls, hdbscan_min_cluster=5,
                         hdbscan_min_samples=3):
    """Cluster raw features for a single class using HDBSCAN.

    Args:
        features: (h, w, C) feature grid.
        sem_ds: (h, w) semantic map at feature resolution.
        cls: class ID to cluster.
        hdbscan_min_cluster: HDBSCAN min_cluster_size.
        hdbscan_min_samples: HDBSCAN min_samples.

    Returns:
        List of (pixel_indices, label) where pixel_indices are (N,2) yx coords.
    """
    cls_mask = sem_ds == cls
    if cls_mask.sum() < hdbscan_min_cluster:
        return []

    ys, xs = np.where(cls_mask)
    feats = features[ys, xs]  # (N, C)

    # L2 normalize
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    feats_norm = feats / norms

    clusterer = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster,
        min_samples=hdbscan_min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(feats_norm)

    results = []
    for label_id in set(labels):
        if label_id == -1:  # noise
            continue
        mask = labels == label_id
        coords = np.stack([ys[mask], xs[mask]], axis=1)
        results.append((coords, label_id))

    return results


def contrastive_instances(semantic, depth, thing_ids=THING_IDS,
                          hdbscan_min_cluster=5, hdbscan_min_samples=3,
                          min_area=1000, dilation_iters=3,
                          depth_blur_sigma=1.0, depth_weight=2.0,
                          pos_weight=0.5, features=None,
                          model=None):
    """Contrastive embedding + HDBSCAN instance decomposition.

    If model is None, falls back to raw feature clustering (CE-raw variant).

    Args:
        semantic: (H,W) uint8 trainID map.
        depth: (H,W) float32 [0,1].
        thing_ids: set of thing class trainIDs.
        hdbscan_min_cluster: HDBSCAN min_cluster_size.
        hdbscan_min_samples: HDBSCAN min_samples.
        min_area: minimum instance area.
        dilation_iters: boundary reclamation.
        depth_blur_sigma: unused.
        depth_weight: scale factor for depth in input vector.
        pos_weight: scale factor for position in input vector.
        features: (h, w, C) feature grid. Required.
        model: trained ContrastiveProjectionHead or None.

    Returns:
        List of (mask, class_id, score).
    """
    if features is None:
        raise ValueError("Contrastive method requires features")

    fh, fw, C = features.shape
    H, W = semantic.shape

    # Downsample semantic to feature resolution
    from PIL import Image
    sem_ds = np.array(
        Image.fromarray(semantic).resize((fw, fh), Image.NEAREST)
    )

    # If model provided, compute learned embeddings
    if model is not None:
        depth_ds = np.array(
            Image.fromarray(depth).resize((fw, fh), Image.BILINEAR)
        )
        yy, xx = np.mgrid[0:fh, 0:fw]
        pos = np.stack([yy / fh, xx / fw], axis=-1)

        import torch
        device = next(model.parameters()).device
        x = np.concatenate([
            features,
            depth_ds[:, :, None] * depth_weight,
            pos * pos_weight,
        ], axis=-1)  # (fh, fw, C+3)
        x_flat = torch.from_numpy(
            x.reshape(-1, x.shape[-1])
        ).float().to(device)

        with torch.no_grad():
            embeddings = model(x_flat).cpu().numpy()  # (fh*fw, embed_dim)
        embed_grid = embeddings.reshape(fh, fw, -1)
        cluster_features = embed_grid
    else:
        cluster_features = features

    raw_instances = []
    scale_y, scale_x = H / fh, W / fw

    for cls in sorted(thing_ids):
        clusters = _raw_feature_cluster(
            cluster_features, sem_ds, cls,
            hdbscan_min_cluster=hdbscan_min_cluster,
            hdbscan_min_samples=hdbscan_min_samples,
        )

        for coords, label_id in clusters:
            # Convert to full-res mask
            full_mask = np.zeros((H, W), dtype=bool)
            for py, px in coords:
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
