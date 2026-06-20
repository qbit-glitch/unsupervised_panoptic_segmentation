"""Semantic argmax -> panoptic via connected components for thing classes."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from auto_annotation import taxonomy_coco as T

DIV = 1000


def semantic_to_panoptic(sem: np.ndarray) -> tuple[np.ndarray, dict]:
    """(HxW argmax class-idx map) -> (class*1000+inst map, {seg_id: cat_idx}).

    Stuff classes -> one segment (inst 0); thing classes -> one segment per
    connected component; everything else (e.g. void) stays VOID.
    """
    pan = np.full(sem.shape, T.VOID_IDX * DIV, dtype=np.int32)
    seg2cat: dict = {}
    for cls in np.unique(sem):
        cls = int(cls)
        region = sem == cls
        if cls in T.STUFF_IDXS:
            sid = cls * DIV
            pan[region] = sid
            seg2cat[sid] = cls
        elif cls in T.THING_IDXS:
            lab, n = ndimage.label(region)
            for inst in range(1, n + 1):
                sid = cls * DIV + inst
                pan[lab == inst] = sid
                seg2cat[sid] = cls
    return pan, seg2cat
