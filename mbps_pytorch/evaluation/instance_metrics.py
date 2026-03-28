"""Instance Segmentation Metrics.

Computes Average Precision (AP) at various IoU thresholds
for instance segmentation evaluation.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Tuple

import numpy as np


class InstanceResult(NamedTuple):
    """Instance segmentation evaluation results."""

    ap: float          # AP @ IoU=0.5
    ap_50: float       # Same as ap (IoU=0.5)
    ap_75: float       # AP @ IoU=0.75
    ap_mean: float     # AP averaged over IoU=[0.5:0.05:0.95]
    precision: np.ndarray   # Precision at each IoU threshold
    recall: np.ndarray      # Recall at each IoU threshold


def compute_mask_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """Compute IoU between two binary masks.

    Args:
        pred_mask: Predicted binary mask of shape (H, W) or (N,).
        gt_mask: Ground truth binary mask of shape (H, W) or (N,).

    Returns:
        IoU score.
    """
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask | gt_mask)
    return float(intersection / (union + 1e-8))


def compute_ap(
    pred_masks: np.ndarray,
    pred_scores: np.ndarray,
    gt_masks: np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute Average Precision at a single IoU threshold.

    Args:
        pred_masks: Predicted binary masks of shape (M_p, N).
        pred_scores: Confidence scores of shape (M_p,).
        gt_masks: Ground truth binary masks of shape (M_g, N).
        iou_threshold: IoU threshold for TP/FP classification.

    Returns:
        Tuple of (AP, precision_array, recall_array).
    """
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return 0.0, np.array([]), np.array([])

    num_gt = len(gt_masks)

    # Sort predictions by score (descending)
    sorted_idx = np.argsort(-pred_scores)
    pred_masks = pred_masks[sorted_idx]
    pred_scores = pred_scores[sorted_idx]

    gt_matched = np.zeros(num_gt, dtype=bool)
    tp = np.zeros(len(pred_masks))
    fp = np.zeros(len(pred_masks))

    for i, pred_mask in enumerate(pred_masks):
        best_iou = 0.0
        best_gt = -1

        for j, gt_mask in enumerate(gt_masks):
            if gt_matched[j]:
                continue

            iou = compute_mask_iou(pred_mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt = j

        if best_iou >= iou_threshold and best_gt >= 0:
            tp[i] = 1
            gt_matched[best_gt] = True
        else:
            fp[i] = 1

    # Cumulative TP and FP
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    precision = cum_tp / (cum_tp + cum_fp + 1e-8)
    recall = cum_tp / (num_gt + 1e-8)

    # AP via all-point interpolation
    ap = _compute_ap_from_pr(precision, recall)

    return ap, precision, recall


def _compute_ap_from_pr(
    precision: np.ndarray,
    recall: np.ndarray,
) -> float:
    """Compute AP from precision-recall curve using all-point interpolation.

    Args:
        precision: Precision values at each detection.
        recall: Recall values at each detection.

    Returns:
        Average precision.
    """
    if len(precision) == 0:
        return 0.0

    # Prepend sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute monotonically decreasing precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Find points where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1

    # Sum (delta_recall * precision) at change points
    ap = float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))
    return ap


def compute_ap_range(
    pred_masks: np.ndarray,
    pred_scores: np.ndarray,
    gt_masks: np.ndarray,
    iou_thresholds: np.ndarray | None = None,
) -> InstanceResult:
    """Compute AP at multiple IoU thresholds (COCO-style).

    Args:
        pred_masks: Predicted binary masks of shape (M_p, N).
        pred_scores: Confidence scores of shape (M_p,).
        gt_masks: Ground truth binary masks of shape (M_g, N).
        iou_thresholds: Array of IoU thresholds. Default [0.5:0.05:0.95].

    Returns:
        InstanceResult with AP at various thresholds.
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    aps = np.zeros(len(iou_thresholds))
    precisions = []
    recalls = []

    for i, threshold in enumerate(iou_thresholds):
        ap, prec, rec = compute_ap(pred_masks, pred_scores, gt_masks, threshold)
        aps[i] = ap
        precisions.append(prec)
        recalls.append(rec)

    # Standard AP metrics
    ap_50 = aps[0] if len(aps) > 0 else 0.0  # IoU=0.5
    ap_75_idx = np.argmin(np.abs(iou_thresholds - 0.75))
    ap_75 = aps[ap_75_idx] if len(aps) > ap_75_idx else 0.0
    ap_mean = float(np.mean(aps))

    return InstanceResult(
        ap=float(ap_50),
        ap_50=float(ap_50),
        ap_75=float(ap_75),
        ap_mean=ap_mean,
        precision=aps,  # AP at each threshold
        recall=np.array(iou_thresholds),
    )


def compute_ap_batch(
    pred_masks_list: list[np.ndarray],
    pred_scores_list: list[np.ndarray],
    gt_masks_list: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> float:
    """Compute AP averaged over a batch of images.

    Args:
        pred_masks_list: List of predicted mask arrays per image.
        pred_scores_list: List of score arrays per image.
        gt_masks_list: List of GT mask arrays per image.
        iou_threshold: IoU threshold for matching.

    Returns:
        Mean AP over all images.
    """
    aps = []
    for pred_masks, pred_scores, gt_masks in zip(
        pred_masks_list, pred_scores_list, gt_masks_list
    ):
        ap, _, _ = compute_ap(pred_masks, pred_scores, gt_masks, iou_threshold)
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0
