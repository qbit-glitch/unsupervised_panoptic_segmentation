"""Hungarian Matching for Unsupervised Evaluation.

Maps predicted cluster IDs to ground-truth class IDs via
linear assignment (Hungarian algorithm) to compute accuracy.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_match(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    num_pred_clusters: int,
    num_gt_classes: int,
    ignore_label: int = 255,
) -> Tuple[dict, float]:
    """Find optimal cluster-to-class mapping via Hungarian algorithm.

    Args:
        pred_labels: Predicted cluster IDs (N,).
        gt_labels: Ground truth class IDs (N,).
        num_pred_clusters: Number of predicted clusters.
        num_gt_classes: Number of ground truth classes.
        ignore_label: Label to ignore in computation.

    Returns:
        Tuple of:
            - mapping: Dict mapping pred_cluster -> gt_class.
            - accuracy: Overall accuracy with optimal mapping.
    """
    # Build cost matrix (negative intersection)
    valid = gt_labels != ignore_label
    pred_valid = pred_labels[valid]
    gt_valid = gt_labels[valid]

    cost_matrix = np.zeros((num_pred_clusters, num_gt_classes))

    for pred_c in range(num_pred_clusters):
        for gt_c in range(num_gt_classes):
            # Count intersection
            cost_matrix[pred_c, gt_c] = -np.sum(
                (pred_valid == pred_c) & (gt_valid == gt_c)
            )

    # Run Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build mapping
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[r] = c

    # Compute accuracy
    mapped_pred = np.zeros_like(pred_valid)
    for pred_c, gt_c in mapping.items():
        mapped_pred[pred_valid == pred_c] = gt_c

    accuracy = np.mean(mapped_pred == gt_valid) if gt_valid.size > 0 else 0.0

    return mapping, float(accuracy)


def compute_miou(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    mapping: dict,
    num_classes: int,
    ignore_label: int = 255,
) -> Tuple[float, np.ndarray]:
    """Compute mean IoU after Hungarian matching.

    Args:
        pred_labels: Predicted cluster IDs (N,).
        gt_labels: Ground truth class IDs (N,).
        mapping: Cluster-to-class mapping from hungarian_match.
        num_classes: Number of GT classes.
        ignore_label: Label to ignore.

    Returns:
        Tuple of (mIoU, per_class_iou).
    """
    valid = gt_labels != ignore_label

    # Map predictions to GT class space
    mapped = np.full_like(pred_labels, fill_value=ignore_label)
    for pred_c, gt_c in mapping.items():
        mapped[pred_labels == pred_c] = gt_c

    mapped_valid = mapped[valid]
    gt_valid = gt_labels[valid]

    per_class_iou = np.zeros(num_classes)

    for c in range(num_classes):
        pred_c = mapped_valid == c
        gt_c = gt_valid == c
        intersection = np.sum(pred_c & gt_c)
        union = np.sum(pred_c | gt_c)
        per_class_iou[c] = intersection / (union + 1e-8)

    miou = np.mean(per_class_iou[per_class_iou > 0])
    return float(miou), per_class_iou
