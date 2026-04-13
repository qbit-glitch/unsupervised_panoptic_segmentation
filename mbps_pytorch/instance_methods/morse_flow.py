"""Method 6: Morse/Gradient Flow Decomposition.

Uses watershed transform (the numerical Morse-theoretic decomposition) to find
depth basins of attraction, then optionally merges adjacent basins with similar
DINOv2 features. Unlike Sobel+CC which asks "is the gradient above a threshold?"
(binary, local), watershed asks "where does each pixel's gradient flow converge?"
(global, structural).

The h-minima transform suppresses shallow local minima before watershed,
analogous to persistence thresholding in TDA.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage.morphology import h_minima, local_minima
from skimage.measure import label as sk_label

from .utils import dilation_reclaim, cosine_similarity_regions

THING_IDS = set(range(11, 19))


def _compute_watershed_basins(depth, depth_blur_sigma=1.0,
                              min_basin_depth=0.03):
    """Compute watershed basins from depth map.

    Args:
        depth: (H,W) float32 [0,1] depth map.
        depth_blur_sigma: Gaussian blur before watershed.
        min_basin_depth: h-minima suppression — basins shallower than this
            are merged into neighbors. Higher = fewer, larger basins.

    Returns:
        basin_map: (H,W) int32 basin labels (1-indexed, 0 = no basin).
        n_basins: number of basins.
    """
    # Smooth depth
    if depth_blur_sigma > 0:
        depth_smooth = gaussian_filter(depth.astype(np.float64),
                                       sigma=depth_blur_sigma)
    else:
        depth_smooth = depth.astype(np.float64)

    # Suppress shallow minima via h-minima transform
    if min_basin_depth > 0:
        depth_suppressed = h_minima(depth_smooth, h=min_basin_depth)
    else:
        depth_suppressed = depth_smooth

    # Find local minima as markers
    minima = local_minima(depth_suppressed)
    markers, n_markers = ndimage.label(minima)

    if n_markers == 0:
        # Fallback: single basin covering everything
        return np.ones(depth.shape, dtype=np.int32), 1

    # Watershed on depth field (negate: watershed finds catchment basins
    # of the inverted surface, so we negate to find basins of the original)
    basin_map = watershed(depth_smooth, markers=markers)

    return basin_map, n_markers


def _feature_merge_adjacent(instances, features_grid, merge_threshold=0.80):
    """Merge adjacent same-class instances with similar features.

    Args:
        instances: List of (mask, class_id, area).
        features_grid: (h, w, C) feature array at a resolution compatible
            with the masks (can be lower-res if masks are downsampled).
        merge_threshold: cosine similarity threshold for merging.

    Returns:
        Merged list of (mask, class_id, area).
    """
    if merge_threshold <= 0 or not instances or features_grid is None:
        return instances

    # Group by class
    by_class = {}
    for mask, cls, area in instances:
        by_class.setdefault(cls, []).append([mask, cls, area])

    merged = []
    for cls, cls_instances in by_class.items():
        # Iterative merge until stable
        changed = True
        while changed:
            changed = False
            n = len(cls_instances)
            if n < 2:
                break

            # Find first mergeable pair
            merge_i, merge_j = -1, -1
            best_sim = -1
            for i in range(n):
                for j in range(i + 1, n):
                    # Check adjacency: dilate one, check overlap with other
                    dilated_i = ndimage.binary_dilation(cls_instances[i][0],
                                                       iterations=2)
                    if not np.any(dilated_i & cls_instances[j][0]):
                        continue
                    # Downsample masks to feature resolution
                    fh, fw = features_grid.shape[:2]
                    mh, mw = cls_instances[i][0].shape
                    if fh != mh or fw != mw:
                        from PIL import Image
                        mask_i_ds = np.array(
                            Image.fromarray(cls_instances[i][0].astype(np.uint8))
                            .resize((fw, fh), Image.NEAREST)
                        ).astype(bool)
                        mask_j_ds = np.array(
                            Image.fromarray(cls_instances[j][0].astype(np.uint8))
                            .resize((fw, fh), Image.NEAREST)
                        ).astype(bool)
                    else:
                        mask_i_ds = cls_instances[i][0]
                        mask_j_ds = cls_instances[j][0]

                    if mask_i_ds.sum() == 0 or mask_j_ds.sum() == 0:
                        continue

                    sim = cosine_similarity_regions(features_grid,
                                                   mask_i_ds, mask_j_ds)
                    if sim > merge_threshold and sim > best_sim:
                        best_sim = sim
                        merge_i, merge_j = i, j

            if merge_i >= 0:
                # Merge j into i
                cls_instances[merge_i][0] = (cls_instances[merge_i][0] |
                                             cls_instances[merge_j][0])
                cls_instances[merge_i][2] = float(
                    cls_instances[merge_i][0].sum()
                )
                cls_instances.pop(merge_j)
                changed = True

        merged.extend(cls_instances)

    return [(m, c, a) for m, c, a in merged]


def morse_flow_instances(semantic, depth, thing_ids=THING_IDS,
                         min_basin_depth=0.03, merge_threshold=0.80,
                         min_area=1000, dilation_iters=3,
                         depth_blur_sigma=1.0, features=None):
    """Morse/watershed instance decomposition + optional feature merge.

    Args:
        semantic: (H,W) uint8 trainID map.
        depth: (H,W) float32 depth [0,1].
        thing_ids: set of thing class trainIDs.
        min_basin_depth: h-minima suppression threshold. Higher = fewer basins.
        merge_threshold: cosine similarity for feature-based merge (0 = off).
        min_area: minimum instance area.
        dilation_iters: boundary reclamation iterations.
        depth_blur_sigma: Gaussian blur sigma.
        features: (h, w, C) feature grid or None. Needed for feature merge.

    Returns:
        List of (mask, class_id, score).
    """
    # Compute watershed basins
    basin_map, n_basins = _compute_watershed_basins(
        depth, depth_blur_sigma=depth_blur_sigma,
        min_basin_depth=min_basin_depth
    )

    # Extract instances: intersect basins with thing-class semantic masks
    raw_instances = []
    for cls in sorted(thing_ids):
        cls_mask = semantic == cls
        if cls_mask.sum() < min_area:
            continue

        # Find unique basins overlapping this class
        cls_basins = np.unique(basin_map[cls_mask])
        for basin_id in cls_basins:
            if basin_id == 0:
                continue
            instance_mask = (basin_map == basin_id) & cls_mask
            area = float(instance_mask.sum())
            if area >= min_area:
                raw_instances.append((instance_mask, cls, area))

    # Feature-based merge of adjacent similar instances
    if features is not None and merge_threshold > 0:
        raw_instances = _feature_merge_adjacent(
            raw_instances, features, merge_threshold
        )

    # Dilation reclaim + normalize scores
    return dilation_reclaim(raw_instances, semantic, thing_ids,
                            min_area=min_area,
                            dilation_iters=dilation_iters)
