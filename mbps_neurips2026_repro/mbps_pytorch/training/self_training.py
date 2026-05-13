"""Self-Training with Confidence-Weighted Pseudo-Labels.

Phase D of training: uses EMA teacher to generate high-confidence
predictions as training targets for multiple refinement rounds.

Algorithm:
    for r = 1 to R:
        pseudo_labels = teacher.predict(dataset, conf > tau_r)
        retrain model on pseudo_labels for E_r epochs
        update teacher theta_ema
        tau_{r+1} = tau_r + 0.05
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


logger = logging.getLogger(__name__)


class PseudoLabelGenerator:
    """Generate confidence-weighted pseudo-labels from EMA teacher.

    Args:
        confidence_threshold: Initial confidence threshold.
        threshold_increment: Per-round threshold increase.
        confidence_alpha: Balance between semantic and instance confidence.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        threshold_increment: float = 0.05,
        confidence_alpha: float = 0.5,
    ):
        self.threshold = confidence_threshold
        self.increment = threshold_increment
        self.alpha = confidence_alpha
        self.current_round = 0

    def generate(
        self,
        teacher_semantic: torch.Tensor,
        teacher_instance_masks: torch.Tensor,
        teacher_instance_scores: torch.Tensor,
        teacher_panoptic: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Generate pseudo-labels from teacher predictions.

        Args:
            teacher_semantic: Softmax semantic probs (B, N, K).
            teacher_instance_masks: Instance mask logits (B, M, N).
            teacher_instance_scores: Instance scores (B, M).
            teacher_panoptic: Panoptic IDs (B, N).

        Returns:
            Dict with pseudo-labels and confidence weights.
        """
        # Semantic confidence: max class probability per pixel
        sem_conf, sem_pred = torch.max(teacher_semantic, dim=-1)  # (B, N)

        # Instance confidence: mask score * max mask prob
        inst_conf = torch.max(
            teacher_instance_scores[:, :, None] * torch.sigmoid(teacher_instance_masks),
            dim=1,
        ).values  # (B, N)

        # Combined confidence
        pixel_conf = self.alpha * sem_conf + (1.0 - self.alpha) * inst_conf

        # Apply threshold
        valid_mask = pixel_conf > self.threshold

        return {
            "semantic_pseudo": sem_pred,
            "panoptic_pseudo": teacher_panoptic,
            "confidence": pixel_conf,
            "valid_mask": valid_mask,
        }

    def advance_round(self) -> None:
        """Advance to next self-training round.

        Increases confidence threshold.
        """
        self.current_round += 1
        self.threshold += self.increment
        logger.info(
            f"Self-training round {self.current_round}: "
            f"threshold = {self.threshold:.3f}"
        )


def confidence_weighted_loss(
    loss: torch.Tensor,
    confidence: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Weight loss by pseudo-label confidence.

    L_weighted = sum(w_i * L_i) / sum(w_i)

    Args:
        loss: Per-pixel loss of shape (B, N).
        confidence: Confidence weights of shape (B, N).
        valid_mask: Boolean mask for valid pixels (B, N).

    Returns:
        Confidence-weighted scalar loss.
    """
    weights = confidence * valid_mask.float()
    weighted_loss = torch.sum(loss * weights) / (torch.sum(weights) + 1e-8)
    return weighted_loss


class SelfTrainer:
    """Self-training manager for Phase D.

    Args:
        num_rounds: Number of self-training rounds (R=3).
        epochs_per_round: Epochs per round (E_r=5).
        initial_threshold: Initial confidence threshold (tau_1=0.7).
        threshold_increment: Per-round threshold increase (delta_tau=0.05).
    """

    def __init__(
        self,
        num_rounds: int = 3,
        epochs_per_round: int = 5,
        initial_threshold: float = 0.7,
        threshold_increment: float = 0.05,
    ):
        self.num_rounds = num_rounds
        self.epochs_per_round = epochs_per_round
        self.label_generator = PseudoLabelGenerator(
            confidence_threshold=initial_threshold,
            threshold_increment=threshold_increment,
        )

    def get_pseudo_labels(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Generate pseudo-labels for current round.

        Args:
            teacher_outputs: EMA teacher predictions.

        Returns:
            Pseudo-labels with confidence weights.
        """
        return self.label_generator.generate(
            teacher_semantic=teacher_outputs["semantic_probs"],
            teacher_instance_masks=teacher_outputs["instance_masks"],
            teacher_instance_scores=teacher_outputs["instance_scores"],
            teacher_panoptic=teacher_outputs["panoptic_ids"],
        )

    def advance_round(self) -> bool:
        """Advance to next round.

        Returns:
            True if more rounds remain, False if done.
        """
        self.label_generator.advance_round()
        return self.label_generator.current_round < self.num_rounds
