"""Differentiable Panoptic Quality (PQ) Proxy Loss.

L_PQ ~= 1 - PQ_differentiable

Uses soft matching between predicted and EMA-teacher segments
to create a differentiable approximation of PQ for end-to-end training.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


def soft_iou(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
) -> torch.Tensor:
    """Compute soft IoU between two sets of masks.

    Args:
        pred_masks: Predicted mask probs of shape (M_p, N).
        target_masks: Target mask probs of shape (M_t, N).

    Returns:
        IoU matrix of shape (M_p, M_t).
    """
    # Intersection: (M_p, M_t)
    intersection = torch.einsum("mn,tn->mt", pred_masks, target_masks)

    # Union
    pred_sum = torch.sum(pred_masks, dim=-1)[:, None]    # (M_p, 1)
    target_sum = torch.sum(target_masks, dim=-1)[None, :]  # (1, M_t)
    union = pred_sum + target_sum - intersection + 1e-8

    return intersection / union


def differentiable_pq(
    pred_masks: torch.Tensor,
    pred_scores: torch.Tensor,
    teacher_masks: torch.Tensor,
    teacher_scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Compute differentiable PQ proxy.

    PQ = (sum IoU(matched)) / (|TP| + 0.5|FP| + 0.5|FN|)

    Uses soft matching instead of hard bipartite matching.

    Args:
        pred_masks: Predicted mask probs (M_p, N).
        pred_scores: Prediction scores (M_p,).
        teacher_masks: EMA teacher mask probs (M_t, N).
        teacher_scores: Teacher scores (M_t,).
        iou_threshold: IoU threshold for matching.

    Returns:
        Differentiable PQ score.
    """
    # Compute IoU matrix
    iou_matrix = soft_iou(pred_masks, teacher_masks)  # (M_p, M_t)

    # Soft matching: for each pred, find best teacher match
    max_iou_per_pred = torch.max(iou_matrix, dim=-1).values  # (M_p,)
    max_iou_per_teacher = torch.max(iou_matrix, dim=0).values  # (M_t,)

    # Soft TP: IoU above threshold, weighted by scores
    soft_tp_contrib = torch.where(
        max_iou_per_pred > iou_threshold,
        max_iou_per_pred * pred_scores,
        torch.zeros_like(max_iou_per_pred),
    )

    # Count TP, FP, FN
    tp = torch.sum((max_iou_per_pred > iou_threshold).float())
    fp = torch.sum((max_iou_per_pred <= iou_threshold).float())
    fn = torch.sum((max_iou_per_teacher <= iou_threshold).float())

    # PQ
    sum_iou = torch.sum(soft_tp_contrib)
    pq = sum_iou / (tp + 0.5 * fp + 0.5 * fn + 1e-8)

    return pq


class PQProxyLoss(nn.Module):
    """Differentiable PQ Proxy Loss.

    Uses EMA teacher predictions as pseudo ground truth.

    Args:
        iou_threshold: IoU threshold for matching.
    """

    def __init__(self, iou_threshold: float = 0.5):
        super().__init__()
        self.iou_threshold = iou_threshold

    def forward(
        self,
        pred_masks: torch.Tensor,
        pred_scores: torch.Tensor,
        teacher_masks: torch.Tensor,
        teacher_scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute PQ proxy loss.

        Args:
            pred_masks: Student mask logits (B, M, N).
            pred_scores: Student scores (B, M).
            teacher_masks: Teacher mask logits (B, M, N).
            teacher_scores: Teacher scores (B, M).

        Returns:
            Dict with PQ loss.
        """
        b = pred_masks.shape[0]
        total_pq = torch.tensor(0.0, device=pred_masks.device)

        for i in range(b):
            pred_probs = torch.sigmoid(pred_masks[i])
            teacher_probs = torch.sigmoid(teacher_masks[i])

            pq = differentiable_pq(
                pred_probs,
                pred_scores[i],
                teacher_probs,
                teacher_scores[i],
                self.iou_threshold,
            )
            total_pq = total_pq + pq

        avg_pq = total_pq / b
        return {
            "pq": avg_pq,
            "total": 1.0 - avg_pq,  # Minimize 1 - PQ
        }
