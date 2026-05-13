"""Two-stage instance decomposition: depth oversegmentation + learned merge.

Stage 1: Sobel+CC at low threshold -> over-segmented fragments
Stage 2: Pairwise merge predictor (learned or feature-based) groups fragments

The merge predictor can be:
  - A trained MergePredictor model (with PCA-reduced features)
  - A simple cosine similarity threshold (training-free baseline)
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel

from .utils import dilation_reclaim, cosine_similarity_regions

THING_IDS = set(range(11, 19))


def _oversegment_sobel_cc(
    semantic: np.ndarray,
    depth: np.ndarray,
    thing_ids: set,
    grad_threshold: float = 0.10,
    min_area: int = 200,
    depth_blur_sigma: float = 1.0,
) -> list:
    """Stage 1: Over-segment with low-threshold Sobel+CC.

    Returns list of (mask, class_id, area) WITHOUT dilation reclaim.
    """
    if depth_blur_sigma > 0:
        depth_smooth = gaussian_filter(
            depth.astype(np.float64), sigma=depth_blur_sigma
        )
    else:
        depth_smooth = depth.astype(np.float64)

    gx = sobel(depth_smooth, axis=1)
    gy = sobel(depth_smooth, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    depth_edges = grad_mag > grad_threshold

    fragments = []
    for cls in sorted(thing_ids):
        cls_mask = semantic == cls
        if cls_mask.sum() < min_area:
            continue

        split_mask = cls_mask & ~depth_edges
        labeled, n_cc = ndimage.label(split_mask)

        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = int(cc_mask.sum())
            if area >= min_area:
                fragments.append((cc_mask, cls, float(area)))

    return fragments


def _find_adjacent_pairs(fragments: list, adjacency_dilation: int = 5) -> list:
    """Find pairs of adjacent same-class fragments.

    Args:
        fragments: list of (mask, cls, area) tuples.
        adjacency_dilation: dilation iterations to bridge depth-edge gaps.

    Returns list of (i, j) index pairs where i < j.
    """
    n = len(fragments)
    if n < 2:
        return []

    H, W = fragments[0][0].shape
    inst_map = np.full((H, W), -1, dtype=np.int32)
    for i, (mask, cls, area) in enumerate(fragments):
        inst_map[mask] = i

    struct = ndimage.generate_binary_structure(2, 1)
    pairs = set()

    for i in range(n):
        mask_i, cls_i, _ = fragments[i]
        dilated = ndimage.binary_dilation(
            mask_i, structure=struct, iterations=adjacency_dilation
        )
        border = dilated & ~mask_i
        neighbors = np.unique(inst_map[border])
        for j in neighbors:
            if j < 0 or j == i:
                continue
            if fragments[j][1] != cls_i:
                continue
            pair = (min(i, j), max(i, j))
            pairs.add(pair)

    return list(pairs)


def _extract_pairwise_descriptor(
    frag_a: tuple,
    frag_b: tuple,
    features: np.ndarray,
    depth: np.ndarray,
    pca=None,
) -> np.ndarray:
    """Extract pairwise descriptor for a merge candidate.

    Args:
        frag_a: (mask, cls, area) tuple.
        frag_b: (mask, cls, area) tuple.
        features: (fh, fw, C) feature grid at patch resolution.
        depth: (H, W) depth map at full resolution.
        pca: sklearn PCA object or None.

    Returns:
        descriptor: (D,) float32 feature vector.
    """
    mask_a, cls_a, area_a = frag_a
    mask_b, cls_b, area_b = frag_b
    H, W = mask_a.shape
    fh, fw, C = features.shape

    from PIL import Image

    # Downsample masks to feature resolution
    def _ds_mask(m: np.ndarray) -> np.ndarray:
        return np.array(
            Image.fromarray(m.astype(np.uint8)).resize((fw, fh), Image.NEAREST)
        ).astype(bool)

    mask_a_ds = _ds_mask(mask_a)
    mask_b_ds = _ds_mask(mask_b)

    # Mean features per fragment
    feat_a = features[mask_a_ds].mean(axis=0) if mask_a_ds.sum() > 0 else np.zeros(C)
    feat_b = features[mask_b_ds].mean(axis=0) if mask_b_ds.sum() > 0 else np.zeros(C)

    # Apply PCA if available
    if pca is not None:
        feat_a = pca.transform(feat_a.reshape(1, -1))[0]
        feat_b = pca.transform(feat_b.reshape(1, -1))[0]

    feat_diff = np.abs(feat_a - feat_b)

    # Cosine similarity
    norm_a = np.linalg.norm(feat_a) + 1e-8
    norm_b = np.linalg.norm(feat_b) + 1e-8
    cos_sim = np.dot(feat_a, feat_b) / (norm_a * norm_b)

    # Depth statistics
    depth_a = depth[mask_a].mean() if mask_a.sum() > 0 else 0.0
    depth_b = depth[mask_b].mean() if mask_b.sum() > 0 else 0.0
    depth_diff = abs(depth_a - depth_b)

    # Spatial statistics
    log_area_a = np.log1p(area_a)
    log_area_b = np.log1p(area_b)

    ys_a, xs_a = np.where(mask_a)
    ys_b, xs_b = np.where(mask_b)
    cy_a, cx_a = ys_a.mean(), xs_a.mean()
    cy_b, cx_b = ys_b.mean(), xs_b.mean()
    diag = np.sqrt(H ** 2 + W ** 2)
    centroid_dist = np.sqrt((cy_a - cy_b) ** 2 + (cx_a - cx_b) ** 2) / diag

    # Boundary feature similarity (features at shared boundary region)
    dilated_a = ndimage.binary_dilation(mask_a, iterations=2)
    dilated_b = ndimage.binary_dilation(mask_b, iterations=2)
    boundary_region = dilated_a & dilated_b & ~mask_a & ~mask_b
    boundary_ds = _ds_mask(boundary_region)
    if boundary_ds.sum() > 0 and mask_a_ds.sum() > 0 and mask_b_ds.sum() > 0:
        # Average cosine sim from boundary patches to each fragment
        bnd_feat = features[boundary_ds].mean(axis=0)
        if pca is not None:
            bnd_feat = pca.transform(bnd_feat.reshape(1, -1))[0]
        norm_bnd = np.linalg.norm(bnd_feat) + 1e-8
        bnd_cos = 0.5 * (
            np.dot(bnd_feat, feat_a) / (norm_bnd * norm_a)
            + np.dot(bnd_feat, feat_b) / (norm_bnd * norm_b)
        )
    else:
        bnd_cos = cos_sim

    descriptor = np.concatenate([
        feat_a.astype(np.float32),
        feat_b.astype(np.float32),
        feat_diff.astype(np.float32),
        np.array([cos_sim, depth_a, depth_b, depth_diff,
                  log_area_a, log_area_b, centroid_dist, bnd_cos],
                 dtype=np.float32),
    ])
    return descriptor


def _apply_union_find(fragments: list, merge_pairs: list) -> list:
    """Merge fragments using union-find.

    Args:
        fragments: list of (mask, cls, area).
        merge_pairs: list of (i, j) pairs to merge.

    Returns:
        Merged list of (mask, cls, area).
    """
    n = len(fragments)
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    for i, j in merge_pairs:
        union(i, j)

    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    merged = []
    for root, members in groups.items():
        mask_merged = fragments[members[0]][0].copy()
        cls = fragments[members[0]][1]
        for mi in members[1:]:
            mask_merged |= fragments[mi][0]
        area = float(mask_merged.sum())
        merged.append((mask_merged, cls, area))

    return merged


def learned_merge_instances(
    semantic: np.ndarray,
    depth: np.ndarray,
    thing_ids: set = THING_IDS,
    grad_threshold: float = 0.10,
    merge_threshold: float = 0.5,
    min_area: int = 1000,
    dilation_iters: int = 3,
    depth_blur_sigma: float = 1.0,
    features: np.ndarray = None,
    model=None,
    pca=None,
    mode: str = "feature_cosine",
) -> list:
    """Two-stage instance decomposition: oversegment + merge.

    Args:
        semantic: (H,W) uint8 trainID map.
        depth: (H,W) float32 [0,1].
        thing_ids: set of thing class trainIDs.
        grad_threshold: Sobel threshold for Stage 1 oversegmentation.
        merge_threshold: merge probability/similarity threshold for Stage 2.
            For mode='feature_cosine': cosine similarity threshold (0.6-0.9).
            For mode='learned': sigmoid probability threshold (0.3-0.7).
        min_area: minimum instance area after merge.
        dilation_iters: boundary reclamation iterations.
        depth_blur_sigma: Gaussian blur sigma for depth.
        features: (fh, fw, C) feature grid. Required.
        model: trained MergePredictor (for mode='learned') or None.
        pca: fitted PCA object (for mode='learned') or None.
        mode: 'feature_cosine' (training-free) or 'learned' (trained model).

    Returns:
        List of (mask, class_id, score).
    """
    if features is None:
        raise ValueError("Learned merge method requires features")

    # Stage 1: Over-segment with low threshold
    frag_min_area = max(min_area // 4, 100)
    fragments = _oversegment_sobel_cc(
        semantic, depth, thing_ids,
        grad_threshold=grad_threshold,
        min_area=frag_min_area,
        depth_blur_sigma=depth_blur_sigma,
    )

    if len(fragments) < 2:
        return dilation_reclaim(
            fragments, semantic, thing_ids,
            min_area=min_area, dilation_iters=dilation_iters,
        )

    # Find adjacent same-class pairs
    pairs = _find_adjacent_pairs(fragments)

    if not pairs:
        return dilation_reclaim(
            fragments, semantic, thing_ids,
            min_area=min_area, dilation_iters=dilation_iters,
        )

    # Stage 2: Decide which pairs to merge
    merge_pairs = []

    if mode == "feature_cosine":
        # Training-free: merge if cosine similarity > threshold
        fh, fw, C = features.shape
        from PIL import Image

        for i, j in pairs:
            mask_i_ds = np.array(
                Image.fromarray(
                    fragments[i][0].astype(np.uint8)
                ).resize((fw, fh), Image.NEAREST)
            ).astype(bool)
            mask_j_ds = np.array(
                Image.fromarray(
                    fragments[j][0].astype(np.uint8)
                ).resize((fw, fh), Image.NEAREST)
            ).astype(bool)

            if mask_i_ds.sum() == 0 or mask_j_ds.sum() == 0:
                continue

            sim = cosine_similarity_regions(features, mask_i_ds, mask_j_ds)
            if sim > merge_threshold:
                merge_pairs.append((i, j))

    elif mode == "learned":
        if model is None:
            raise ValueError("mode='learned' requires a trained model")

        import torch
        device = next(model.parameters()).device
        model.eval()

        descriptors = []
        pair_indices = []
        for i, j in pairs:
            desc = _extract_pairwise_descriptor(
                fragments[i], fragments[j], features, depth, pca
            )
            descriptors.append(desc)
            pair_indices.append((i, j))

        if descriptors:
            X = torch.from_numpy(np.stack(descriptors)).float().to(device)
            with torch.no_grad():
                logits = model(X).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()

            for k, (i, j) in enumerate(pair_indices):
                if probs[k] > merge_threshold:
                    merge_pairs.append((i, j))

    # Apply merges
    if merge_pairs:
        fragments = _apply_union_find(fragments, merge_pairs)

    return dilation_reclaim(
        fragments, semantic, thing_ids,
        min_area=min_area, dilation_iters=dilation_iters,
    )
