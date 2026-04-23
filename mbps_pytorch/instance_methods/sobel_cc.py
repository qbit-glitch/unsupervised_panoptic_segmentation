"""Baseline: Sobel + Connected Components instance decomposition."""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel

THING_IDS = set(range(11, 19))


def sobel_cc_instances(semantic, depth, thing_ids=THING_IDS,
                       grad_threshold=0.03, min_area=1000,
                       dilation_iters=3, depth_blur_sigma=1.0,
                       features=None, min_area_ratio=None,
                       use_adaptive_threshold=False, threshold_percentile=95):
    """Baseline Sobel+CC depth-guided instance decomposition.

    This is the existing method, refactored for the unified interface.

    Args:
        min_area: absolute minimum area in pixels (legacy)
        min_area_ratio: minimum area as fraction of image area (e.g., 0.0005 = 0.05%)
    
    NOTE: depth_blur_sigma controls Gaussian smoothing before Sobel.
    For high-resolution depth maps with sharp boundaries (e.g., DA3),
    consider sigma=0.5 or sigma=0 (no blur). For noisy depth (e.g.,
    zero-shot DA2), sigma=1.0 may be appropriate.
    """
    img_area = semantic.shape[0] * semantic.shape[1]
    if min_area_ratio is not None:
        effective_min_area = max(min_area, int(min_area_ratio * img_area))
    else:
        effective_min_area = min_area

    if depth_blur_sigma > 0:
        depth_smooth = gaussian_filter(depth.astype(np.float64),
                                       sigma=depth_blur_sigma)
    else:
        depth_smooth = depth.astype(np.float64)

    gx = sobel(depth_smooth, axis=1)
    gy = sobel(depth_smooth, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    if use_adaptive_threshold:
        # Normalize by depth range and use percentile
        depth_range = depth_smooth.max() - depth_smooth.min()
        if depth_range > 1e-6:
            grad_mag_norm = grad_mag / depth_range
            adaptive_tau = np.percentile(grad_mag_norm[grad_mag_norm > 0], threshold_percentile)
            depth_edges = grad_mag_norm > max(adaptive_tau, grad_threshold)
        else:
            depth_edges = grad_mag > grad_threshold
    else:
        depth_edges = grad_mag > grad_threshold

    assigned = np.zeros(semantic.shape, dtype=bool)
    instances = []

    for cls in sorted(thing_ids):
        cls_mask = semantic == cls
        if cls_mask.sum() < effective_min_area:
            continue

        split_mask = cls_mask & ~depth_edges
        labeled, n_cc = ndimage.label(split_mask)

        cc_list = []
        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = int(cc_mask.sum())
            if area >= effective_min_area:
                cc_list.append((cc_mask, area))
        # Ascending order prevents large instances from dilating first and
        # reclaiming boundary pixels that rightfully belong to smaller neighbors.
        cc_list.sort(key=lambda x: x[1])  # ascending by area

        for cc_mask, area in cc_list:
            if dilation_iters > 0:
                dilated = ndimage.binary_dilation(cc_mask,
                                                  iterations=dilation_iters)
                reclaimed = dilated & cls_mask & ~assigned
                final_mask = cc_mask | reclaimed
            else:
                final_mask = cc_mask

            final_area = float(final_mask.sum())
            if final_area < effective_min_area:
                continue

            assigned |= final_mask
            instances.append((final_mask, cls, final_area))

    instances.sort(key=lambda x: -x[2])
    if instances:
        max_area = instances[0][2]
        instances = [(m, c, s / max_area) for m, c, s in instances]

    return instances
