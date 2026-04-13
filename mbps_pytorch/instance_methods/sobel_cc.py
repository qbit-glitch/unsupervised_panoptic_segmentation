"""Baseline: Sobel + Connected Components instance decomposition."""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel

THING_IDS = set(range(11, 19))


def sobel_cc_instances(semantic, depth, thing_ids=THING_IDS,
                       grad_threshold=0.20, min_area=1000,
                       dilation_iters=3, depth_blur_sigma=1.0,
                       features=None):
    """Baseline Sobel+CC depth-guided instance decomposition.

    This is the existing method, refactored for the unified interface.
    """
    if depth_blur_sigma > 0:
        depth_smooth = gaussian_filter(depth.astype(np.float64),
                                       sigma=depth_blur_sigma)
    else:
        depth_smooth = depth.astype(np.float64)

    gx = sobel(depth_smooth, axis=1)
    gy = sobel(depth_smooth, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    depth_edges = grad_mag > grad_threshold

    assigned = np.zeros(semantic.shape, dtype=bool)
    instances = []

    for cls in sorted(thing_ids):
        cls_mask = semantic == cls
        if cls_mask.sum() < min_area:
            continue

        split_mask = cls_mask & ~depth_edges
        labeled, n_cc = ndimage.label(split_mask)

        cc_list = []
        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = int(cc_mask.sum())
            if area >= min_area:
                cc_list.append((cc_mask, area))
        cc_list.sort(key=lambda x: -x[1])

        for cc_mask, area in cc_list:
            if dilation_iters > 0:
                dilated = ndimage.binary_dilation(cc_mask,
                                                  iterations=dilation_iters)
                reclaimed = dilated & cls_mask & ~assigned
                final_mask = cc_mask | reclaimed
            else:
                final_mask = cc_mask

            final_area = float(final_mask.sum())
            if final_area < min_area:
                continue

            assigned |= final_mask
            instances.append((final_mask, cls, final_area))

    instances.sort(key=lambda x: -x[2])
    if instances:
        max_area = instances[0][2]
        instances = [(m, c, s / max_area) for m, c, s in instances]

    return instances
