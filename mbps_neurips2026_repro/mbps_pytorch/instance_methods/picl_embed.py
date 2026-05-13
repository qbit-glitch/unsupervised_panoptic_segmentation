"""PICL inference: project DINOv2 features through trained PICL head → HDBSCAN.

Unlike the old contrastive_embed.py (which failed because features were trained
with depth-proximity pairs), PICL features are trained with pseudo-instance-mask
pairs, making them genuinely instance-discriminative.

Training: mbps_pytorch/train_picl.py
Evaluation: mbps_pytorch/eval_picl.py
"""

import numpy as np
from scipy import ndimage
from sklearn.cluster import HDBSCAN

from .utils import dilation_reclaim

THING_IDS = set(range(11, 19))


def picl_instances(
    semantic: np.ndarray,
    depth: np.ndarray,
    thing_ids: set = THING_IDS,
    hdbscan_min_cluster: int = 5,
    hdbscan_min_samples: int = 3,
    min_area: int = 1000,
    dilation_iters: int = 3,
    depth_weight: float = 2.0,
    pos_weight: float = 0.5,
    features: np.ndarray = None,
    model=None,
) -> list:
    """Instance decomposition via PICL-projected features + HDBSCAN.

    Projects DINOv2 patch features through a trained PICL head (instance-mask
    contrastive learning) into a 128-dim space where same-instance patches
    cluster together. HDBSCAN then finds the clusters per thing class.

    If model is None, falls back to raw DINOv2 feature clustering (CE-raw).

    Args:
        semantic:            (H, W) uint8 trainID map.
        depth:               (H, W) float32 depth [0, 1].
        thing_ids:           Set of thing class trainIDs.
        hdbscan_min_cluster: HDBSCAN min_cluster_size.
        hdbscan_min_samples: HDBSCAN min_samples.
        min_area:            Minimum instance area in pixels.
        dilation_iters:      Boundary reclamation iterations.
        depth_weight:        Scale for depth in input vector (must match training).
        pos_weight:          Scale for position in input vector (must match training).
        features:            (fh, fw, C) DINOv2 feature grid. Required.
        model:               Trained PICLProjectionHead or None.

    Returns:
        List of (mask: np.ndarray(H,W) bool, class_id: int, score: float).
    """
    if features is None:
        raise ValueError("picl_instances requires DINOv2 features")

    fh, fw, C = features.shape
    H, W = semantic.shape

    # Downsample semantic to feature resolution
    from PIL import Image
    sem_ds = np.array(
        Image.fromarray(semantic).resize((fw, fh), Image.NEAREST)
    )

    # Build cluster_features: PICL-projected or raw DINOv2
    if model is not None:
        depth_ds = np.array(
            Image.fromarray(depth).resize((fw, fh), Image.BILINEAR)
        )
        yy, xx = np.mgrid[0:fh, 0:fw]
        pos = np.stack([yy / fh, xx / fw], axis=-1)  # (fh,fw,2)

        import torch
        device = next(model.parameters()).device

        x = np.concatenate([
            features,
            depth_ds[:, :, None] * depth_weight,
            pos * pos_weight,
        ], axis=-1)  # (fh, fw, C+3)
        x_flat = torch.from_numpy(x.reshape(-1, x.shape[-1])).float().to(device)

        with torch.no_grad():
            embeddings = model(x_flat).cpu().numpy()  # (fh*fw, embed_dim)

        cluster_features = embeddings.reshape(fh, fw, -1)
    else:
        cluster_features = features  # fallback: raw DINOv2

    raw_instances = []
    scale_y = H / fh
    scale_x = W / fw

    for cls in sorted(thing_ids):
        cls_mask = sem_ds == cls
        n_patches = cls_mask.sum()
        if n_patches < hdbscan_min_cluster:
            continue

        ys, xs = np.where(cls_mask)
        feats = cluster_features[ys, xs]  # (N, embed_dim)

        # L2 normalise (already normalised for PICL, but harmless for raw)
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        feats_norm = feats / norms

        clusterer = HDBSCAN(
            min_cluster_size=hdbscan_min_cluster,
            min_samples=hdbscan_min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(feats_norm)

        for label_id in set(labels):
            if label_id == -1:  # noise
                continue

            cluster_mask_feat = labels == label_id
            feat_ys = ys[cluster_mask_feat]
            feat_xs = xs[cluster_mask_feat]

            # Upscale to full resolution
            full_mask = np.zeros((H, W), dtype=bool)
            for py, px in zip(feat_ys, feat_xs):
                y0 = int(py * scale_y)
                y1 = int((py + 1) * scale_y)
                x0 = int(px * scale_x)
                x1 = int((px + 1) * scale_x)
                full_mask[y0:y1, x0:x1] = True
            full_mask &= (semantic == cls)

            # Split into connected components
            labeled, n_cc = ndimage.label(full_mask)
            for cc_id in range(1, n_cc + 1):
                cc_mask = labeled == cc_id
                area = float(cc_mask.sum())
                if area >= min_area:
                    raw_instances.append((cc_mask, cls, area))

    return dilation_reclaim(
        raw_instances, semantic, thing_ids,
        min_area=min_area,
        dilation_iters=dilation_iters,
    )
