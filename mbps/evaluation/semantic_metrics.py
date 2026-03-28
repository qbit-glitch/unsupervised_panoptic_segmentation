"""Semantic Segmentation Metrics.

Computes mean Intersection-over-Union (mIoU) and pixel accuracy
for unsupervised semantic segmentation evaluation.

Uses Hungarian matching to map predicted clusters to GT classes
before computing IoU.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np

from mbps.evaluation.hungarian_matching import hungarian_match


class SemanticResult(NamedTuple):
    """Semantic segmentation evaluation results."""

    miou: float
    pixel_accuracy: float
    per_class_iou: np.ndarray
    per_class_accuracy: np.ndarray
    mapping: dict


def compute_miou(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    num_pred_clusters: int,
    num_gt_classes: int,
    ignore_label: int = 255,
    mapping: Optional[dict] = None,
) -> SemanticResult:
    """Compute mean IoU for unsupervised semantic segmentation.

    If no mapping is provided, uses Hungarian matching to find the
    optimal cluster-to-class assignment.

    Args:
        predictions: Predicted cluster IDs of shape (N,) or (H, W).
        ground_truth: Ground truth class IDs of shape (N,) or (H, W).
        num_pred_clusters: Number of predicted clusters.
        num_gt_classes: Number of ground truth classes.
        ignore_label: Label to ignore in computation.
        mapping: Optional pre-computed cluster-to-class mapping.

    Returns:
        SemanticResult with mIoU, accuracy, per-class metrics, mapping.
    """
    pred_flat = predictions.ravel()
    gt_flat = ground_truth.ravel()

    # Compute mapping if not provided
    if mapping is None:
        mapping, _ = hungarian_match(
            pred_flat, gt_flat, num_pred_clusters, num_gt_classes, ignore_label
        )

    # Apply mapping
    valid = gt_flat != ignore_label
    mapped_pred = np.full_like(pred_flat, fill_value=ignore_label)
    for pred_c, gt_c in mapping.items():
        mapped_pred[pred_flat == pred_c] = gt_c

    mapped_valid = mapped_pred[valid]
    gt_valid = gt_flat[valid]

    # Per-class IoU and accuracy
    per_class_iou = np.zeros(num_gt_classes)
    per_class_accuracy = np.zeros(num_gt_classes)
    class_present = np.zeros(num_gt_classes, dtype=bool)

    for c in range(num_gt_classes):
        pred_c = mapped_valid == c
        gt_c = gt_valid == c

        if not np.any(gt_c):
            continue

        class_present[c] = True
        intersection = np.sum(pred_c & gt_c)
        union = np.sum(pred_c | gt_c)

        per_class_iou[c] = intersection / (union + 1e-8)
        per_class_accuracy[c] = intersection / (np.sum(gt_c) + 1e-8)

    # Aggregate over present classes only
    if np.any(class_present):
        miou = float(np.mean(per_class_iou[class_present]))
        mean_accuracy = float(np.mean(per_class_accuracy[class_present]))
    else:
        miou = 0.0
        mean_accuracy = 0.0

    # Overall pixel accuracy
    pixel_accuracy = float(np.mean(mapped_valid == gt_valid)) if gt_valid.size > 0 else 0.0

    return SemanticResult(
        miou=miou,
        pixel_accuracy=pixel_accuracy,
        per_class_iou=per_class_iou,
        per_class_accuracy=per_class_accuracy,
        mapping=mapping,
    )


def compute_miou_batch(
    predictions_list: list[np.ndarray],
    ground_truth_list: list[np.ndarray],
    num_pred_clusters: int,
    num_gt_classes: int,
    ignore_label: int = 255,
) -> SemanticResult:
    """Compute mIoU over a batch of images with a shared mapping.

    Concatenates all predictions and ground truth, then computes
    a single Hungarian matching for the entire dataset.

    Args:
        predictions_list: List of predicted label arrays.
        ground_truth_list: List of ground truth label arrays.
        num_pred_clusters: Number of predicted clusters.
        num_gt_classes: Number of ground truth classes.
        ignore_label: Label to ignore.

    Returns:
        SemanticResult with dataset-level mIoU.
    """
    all_pred = np.concatenate([p.ravel() for p in predictions_list])
    all_gt = np.concatenate([g.ravel() for g in ground_truth_list])

    return compute_miou(
        all_pred, all_gt, num_pred_clusters, num_gt_classes, ignore_label
    )
