"""Convert ``class*1000+inst`` panoptic maps to COCO-panoptic (rgb2id PNG + json).

Lets EoMT's ``coco_panoptic`` loader and ``coco_eval`` consume auto-labels in the
same format as COCO GT, so nothing downstream needs a bespoke reader.
"""
from __future__ import annotations

import numpy as np

from auto_annotation import taxonomy_coco as T

DIV = 1000


def _id2rgb(seg_id: int) -> tuple[int, int, int]:
    return seg_id % 256, (seg_id // 256) % 256, (seg_id // 65536) % 256


def pan_to_coco(pan: np.ndarray, stem: str) -> tuple[np.ndarray, dict]:
    """(class*1000+inst map) -> (rgb2id-encoded uint8 PNG array, COCO ann dict)."""
    h, w = pan.shape
    rgb = np.zeros((h, w, 3), np.uint8)
    segments = []
    for seg_id in np.unique(pan):
        cls = int(seg_id) // DIV
        if cls == T.VOID_IDX:
            continue
        mask = pan == seg_id
        rgb[mask] = _id2rgb(int(seg_id))
        segments.append({
            "id": int(seg_id),
            "category_id": cls,                 # already contiguous 0..132
            "isthing": int(T.is_thing(cls)),
            "area": int(mask.sum()),
        })
    ann = {"file_name": f"{stem}.png", "image_id": stem, "segments_info": segments}
    return rgb, ann
