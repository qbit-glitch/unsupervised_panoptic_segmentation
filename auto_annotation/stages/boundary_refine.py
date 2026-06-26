#!/usr/bin/env python3
"""SAM-snap boundary refinement.

The highest-leverage trick for precise boundaries (see deep-research findings):
DINOv3/open-vocab semantics are patch-grid-soft, but SAM masks are pixel-crisp. For
each crisp SAM mask we overwrite the enclosed region with its MAJORITY semantic
label, so boundaries snap to SAM's edges. Masks are applied largest-first, so smaller
masks (fine detail) win on overlap.

Pass SAM "everything"-mode class-agnostic masks for stuff boundaries; the thing masks
from SAM3 already double as crisp snap regions.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["snap_to_sam"]


def snap_to_sam(label_map: np.ndarray, sam_masks: List[np.ndarray],
                min_area: int = 50) -> np.ndarray:
    """Snap semantic boundaries to crisp SAM masks via in-mask majority vote.

    Args:
        label_map: int32 HxW semantic class ids (modified copy returned).
        sam_masks: list of bool HxW class-agnostic masks (e.g. SAM everything-mode).
        min_area: ignore masks smaller than this many pixels (noise).

    Returns:
        A new int32 label map with boundaries snapped to mask edges.
    """
    out = label_map.copy()
    # largest first so small masks override and keep fine detail
    ordered = sorted(
        (m for m in sam_masks if m.dtype == bool and int(m.sum()) >= min_area),
        key=lambda m: int(m.sum()),
        reverse=True,
    )
    snapped = 0
    for mask in ordered:
        vals = out[mask]
        if vals.size == 0:
            continue
        counts = np.bincount(vals.astype(np.int64))
        majority = int(counts.argmax())
        # only relabel if the mask is not already homogeneous (cheap skip)
        if counts[majority] != vals.size:
            out[mask] = majority
            snapped += 1
    logger.debug("SAM-snap: relabeled %d / %d masks", snapped, len(ordered))
    return out
