"""Panoptic Quality (PQ) Evaluation Metric.

PQ = SQ · RQ = (Σ IoU(p,g))/|TP| · |TP|/(|TP|+½|FP|+½|FN|)

Also computes per-class PQ, PQ^Th (things), PQ^St (stuff).
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import jax.numpy as jnp
import numpy as np


class PQResult(NamedTuple):
    """Panoptic Quality results."""

    pq: float
    sq: float
    rq: float
    pq_per_class: np.ndarray
    pq_things: float
    pq_stuff: float
    num_categories: int


def compute_panoptic_quality(
    pred_panoptic: np.ndarray,
    gt_panoptic: np.ndarray,
    pred_segments: list[dict],
    gt_segments: list[dict],
    thing_classes: set[int],
    stuff_classes: set[int],
    iou_threshold: float = 0.5,
    label_divisor: int = 1000,
) -> PQResult:
    """Compute Panoptic Quality metric.

    Args:
        pred_panoptic: Predicted panoptic IDs (H, W).
        gt_panoptic: Ground truth panoptic IDs (H, W).
        pred_segments: List of dicts with 'id', 'category_id'.
        gt_segments: List of dicts with 'id', 'category_id'.
        thing_classes: Set of thing class IDs.
        stuff_classes: Set of stuff class IDs.
        iou_threshold: IoU threshold for matching (0.5).
        label_divisor: Divisor for panoptic encoding.

    Returns:
        PQResult with PQ, SQ, RQ, per-class, things, stuff.
    """
    all_classes = thing_classes | stuff_classes
    num_classes = len(all_classes)

    pq_per_class = np.zeros(max(all_classes) + 1)
    tp_per_class = np.zeros(max(all_classes) + 1)
    fp_per_class = np.zeros(max(all_classes) + 1)
    fn_per_class = np.zeros(max(all_classes) + 1)
    iou_per_class = np.zeros(max(all_classes) + 1)

    # Build segment-to-pixel mappings
    pred_ids = np.unique(pred_panoptic)
    gt_ids = np.unique(gt_panoptic)

    # For each ground truth segment, find matching prediction
    gt_matched = set()
    pred_matched = set()

    for gt_seg in gt_segments:
        gt_id = gt_seg["id"]
        gt_cat = gt_seg["category_id"]

        if gt_cat not in all_classes:
            continue

        gt_mask = pred_panoptic == gt_id  # Hmm, should be gt_panoptic
        gt_mask_correct = gt_panoptic == gt_id
        gt_area = np.sum(gt_mask_correct)

        if gt_area == 0:
            continue

        best_iou = 0.0
        best_pred_id = None

        for pred_seg in pred_segments:
            pred_id = pred_seg["id"]
            pred_cat = pred_seg["category_id"]

            if pred_cat != gt_cat:
                continue

            pred_mask = pred_panoptic == pred_id
            intersection = np.sum(pred_mask & gt_mask_correct)
            union = np.sum(pred_mask | gt_mask_correct)

            if union == 0:
                continue

            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
                best_pred_id = pred_id

        if best_iou > iou_threshold:
            tp_per_class[gt_cat] += 1
            iou_per_class[gt_cat] += best_iou
            gt_matched.add(gt_id)
            pred_matched.add(best_pred_id)
        else:
            fn_per_class[gt_cat] += 1

    # Count FP: unmatched predictions
    for pred_seg in pred_segments:
        if pred_seg["id"] not in pred_matched:
            pred_cat = pred_seg["category_id"]
            if pred_cat in all_classes:
                fp_per_class[pred_cat] += 1

    # Compute per-class PQ
    for c in all_classes:
        tp = tp_per_class[c]
        fp = fp_per_class[c]
        fn = fn_per_class[c]
        sum_iou = iou_per_class[c]

        if tp + 0.5 * fp + 0.5 * fn > 0:
            sq = sum_iou / (tp + 1e-8)
            rq = tp / (tp + 0.5 * fp + 0.5 * fn)
            pq_per_class[c] = sq * rq
        else:
            pq_per_class[c] = 0.0

    # Aggregate
    thing_pq = [pq_per_class[c] for c in thing_classes if tp_per_class[c] + fn_per_class[c] + fp_per_class[c] > 0]
    stuff_pq = [pq_per_class[c] for c in stuff_classes if tp_per_class[c] + fn_per_class[c] + fp_per_class[c] > 0]

    overall_pq = np.mean(
        [pq_per_class[c] for c in all_classes if tp_per_class[c] + fn_per_class[c] + fp_per_class[c] > 0]
    ) if len(all_classes) > 0 else 0.0

    total_tp = np.sum(tp_per_class)
    total_iou = np.sum(iou_per_class)
    overall_sq = total_iou / (total_tp + 1e-8)
    overall_rq = total_tp / (
        total_tp + 0.5 * np.sum(fp_per_class) + 0.5 * np.sum(fn_per_class) + 1e-8
    )

    return PQResult(
        pq=float(overall_pq),
        sq=float(overall_sq),
        rq=float(overall_rq),
        pq_per_class=pq_per_class,
        pq_things=float(np.mean(thing_pq)) if thing_pq else 0.0,
        pq_stuff=float(np.mean(stuff_pq)) if stuff_pq else 0.0,
        num_categories=num_classes,
    )
