"""COCO-133 panoptic eval glue: contiguous class map, GT loader, PQ.

Reuses the per-image greedy-IoU PQ from ``evaluate_cross_dataset`` and the
single-source taxonomy from ``auto_annotation.taxonomy_coco``. Runs in ``.venv``.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

_MBPS_PYTORCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_MBPS_PYTORCH))            # for evaluate_cross_dataset
sys.path.insert(0, str(_MBPS_PYTORCH.parent))     # for auto_annotation
from auto_annotation import taxonomy_coco as T    # noqa: E402

logger = logging.getLogger(__name__)

COCO_ROOT = Path("/Volumes/code_files/datasets/coco")
_GT_JSON = COCO_ROOT / "annotations/panoptic_val2017.json"
_GT_PNG_DIR = COCO_ROOT / "annotations/panoptic_val2017"
VAL_IMG_DIR = COCO_ROOT / "val2017"


@lru_cache(maxsize=1)
def _gt_data() -> dict:
    return json.loads(_GT_JSON.read_text())


def coco_contiguous_maps() -> tuple[dict, dict, set, set]:
    """(category_id -> 0..132 sorted-by-id, idx -> name, thing idxs, stuff idxs)."""
    cats = sorted(_gt_data()["categories"], key=lambda c: c["id"])
    catid2idx = {c["id"]: i for i, c in enumerate(cats)}
    idx2name = {i: c.name for i, c in T.COCO_CLASSES.items()}
    return catid2idx, idx2name, set(T.THING_IDXS), set(T.STUFF_IDXS)


def _rgb2id(png_path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(png_path).convert("RGB")).astype(np.int64)
    return arr[..., 0] + arr[..., 1] * 256 + arr[..., 2] * 65536


def load_coco_gt(image_id: int) -> tuple[np.ndarray, dict]:
    """Return (segment-id map HxW, {seg_id: contiguous_cat_idx}) for a val image."""
    ann = next(a for a in _gt_data()["annotations"] if a["image_id"] == image_id)
    catid2idx, *_ = coco_contiguous_maps()
    seg_map = _rgb2id(_GT_PNG_DIR / ann["file_name"])
    seg2cat = {s["id"]: catid2idx[s["category_id"]] for s in ann["segments_info"]}
    return seg_map, seg2cat


def eval_pq(preds: list[tuple[np.ndarray, dict]],
            gts: list[tuple[np.ndarray, dict]]) -> dict:
    """Per-image greedy IoU>0.5 PQ aggregated over a list of (seg_map, {sid:cat})."""
    from evaluate_cross_dataset import compute_pq, summarize_pq  # noqa: WPS433

    _, idx2name, things, stuff = coco_contiguous_maps()
    acc_tp: dict = defaultdict(int)
    acc_iou: dict = defaultdict(float)
    acc_fp: dict = defaultdict(int)
    acc_fn: dict = defaultdict(int)
    for (pred_pan, pred_seg), (gt_pan, gt_seg) in zip(preds, gts):
        tp, iou, fp, fn = compute_pq(gt_seg, pred_seg, gt_pan, pred_pan)
        for k, v in tp.items():
            acc_tp[k] += v
        for k, v in iou.items():
            acc_iou[k] += v
        for k, v in fp.items():
            acc_fp[k] += v
        for k, v in fn.items():
            acc_fn[k] += v
    return summarize_pq(acc_tp, acc_iou, acc_fp, acc_fn, idx2name, stuff, things)
