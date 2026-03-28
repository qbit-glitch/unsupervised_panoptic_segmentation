"""Differentiable Panoptic Quality (PQ) Proxy Loss.

L_PQ ≈ 1 - PQ_differentiable

Uses soft matching between predicted and EMA-teacher segments
to create a differentiable approximation of PQ for end-to-end training.
"""

from __future__ import annotations

from typing import Dict, Optional

import jax
import jax.numpy as jnp


def soft_iou(
    pred_masks: jnp.ndarray,
    target_masks: jnp.ndarray,
) -> jnp.ndarray:
    """Compute soft IoU between two sets of masks.

    Args:
        pred_masks: Predicted mask probs of shape (M_p, N).
        target_masks: Target mask probs of shape (M_t, N).

    Returns:
        IoU matrix of shape (M_p, M_t).
    """
    # Intersection: (M_p, M_t)
    intersection = jnp.einsum("mn,tn->mt", pred_masks, target_masks)

    # Union
    pred_sum = jnp.sum(pred_masks, axis=-1)[:, None]    # (M_p, 1)
    target_sum = jnp.sum(target_masks, axis=-1)[None, :]  # (1, M_t)
    union = pred_sum + target_sum - intersection + 1e-8

    return intersection / union


def differentiable_pq(
    pred_masks: jnp.ndarray,
    pred_scores: jnp.ndarray,
    teacher_masks: jnp.ndarray,
    teacher_scores: jnp.ndarray,
    iou_threshold: float = 0.5,
) -> jnp.ndarray:
    """Compute differentiable PQ proxy.

    PQ = (Σ IoU(matched)) / (|TP| + 0.5|FP| + 0.5|FN|)

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
    max_iou_per_pred = jnp.max(iou_matrix, axis=-1)  # (M_p,)
    max_iou_per_teacher = jnp.max(iou_matrix, axis=0)  # (M_t,)

    # Soft TP: IoU above threshold, weighted by scores
    soft_tp_contrib = jnp.where(
        max_iou_per_pred > iou_threshold,
        max_iou_per_pred * pred_scores,
        jnp.zeros_like(max_iou_per_pred),
    )

    # Count TP, FP, FN
    tp = jnp.sum(max_iou_per_pred > iou_threshold)
    fp = jnp.sum(max_iou_per_pred <= iou_threshold)
    fn = jnp.sum(max_iou_per_teacher <= iou_threshold)

    # PQ
    sum_iou = jnp.sum(soft_tp_contrib)
    pq = sum_iou / (tp + 0.5 * fp + 0.5 * fn + 1e-8)

    return pq


class PQProxyLoss:
    """Differentiable PQ Proxy Loss.

    Uses EMA teacher predictions as pseudo ground truth.

    Args:
        iou_threshold: IoU threshold for matching.
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def __call__(
        self,
        pred_masks: jnp.ndarray,
        pred_scores: jnp.ndarray,
        teacher_masks: jnp.ndarray,
        teacher_scores: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """Compute PQ proxy loss.

        Args:
            pred_masks: Student mask logits (B, M, N).
            pred_scores: Student scores (B, M).
            teacher_masks: Teacher mask logits (B, M, N).
            teacher_scores: Teacher scores (B, M).

        Returns:
            Dict with PQ loss.
        """
        def _single_pq(pred_m, pred_s, teacher_m, teacher_s):
            pred_probs = jax.nn.sigmoid(pred_m)
            teacher_probs = jax.nn.sigmoid(teacher_m)
            return differentiable_pq(
                pred_probs, pred_s, teacher_probs, teacher_s,
                self.iou_threshold,
            )

        pq_values = jax.vmap(_single_pq)(
            pred_masks, pred_scores, teacher_masks, teacher_scores
        )
        avg_pq = jnp.mean(pq_values)
        return {
            "pq": avg_pq,
            "total": 1.0 - avg_pq,  # Minimize 1 - PQ
        }
