"""Self-Training with Confidence-Weighted Pseudo-Labels.

Phase D of training: uses EMA teacher to generate high-confidence
predictions as training targets for multiple refinement rounds.

Algorithm:
    for r = 1 to R:
        pseudo_labels = teacher.predict(dataset, conf > τ_r)
        retrain model on pseudo_labels for E_r epochs
        update teacher θ_ema
        τ_{r+1} = τ_r + 0.05
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging


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
        teacher_semantic: jnp.ndarray,
        teacher_instance_masks: jnp.ndarray,
        teacher_instance_scores: jnp.ndarray,
        teacher_panoptic: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """Generate pseudo-labels from teacher predictions.

        Args:
            teacher_semantic: Softmax semantic probs (B, N, K).
            teacher_instance_masks: Instance mask probs (B, M, N).
            teacher_instance_scores: Instance scores (B, M).
            teacher_panoptic: Panoptic IDs (B, N).

        Returns:
            Dict with pseudo-labels and confidence weights.
        """
        # Semantic confidence: max class probability per pixel
        sem_conf = jnp.max(teacher_semantic, axis=-1)  # (B, N)
        sem_pred = jnp.argmax(teacher_semantic, axis=-1)  # (B, N)

        # Instance confidence: mask score * max mask prob
        inst_max_prob = jnp.max(teacher_instance_masks, axis=-1)  # (B, N)
        # ... but actually per-instance confidence
        inst_conf = jnp.max(
            teacher_instance_scores[:, :, None] * jax.nn.sigmoid(teacher_instance_masks),
            axis=1,
        )  # (B, N)

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
        logging.info(
            f"Self-training round {self.current_round}: "
            f"threshold = {self.threshold:.3f}"
        )


def confidence_weighted_loss(
    loss: jnp.ndarray,
    confidence: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Weight loss by pseudo-label confidence.

    L_weighted = Σ w_i · L_i / Σ w_i

    Args:
        loss: Per-pixel loss of shape (B, N).
        confidence: Confidence weights of shape (B, N).
        valid_mask: Boolean mask for valid pixels (B, N).

    Returns:
        Confidence-weighted scalar loss.
    """
    weights = confidence * valid_mask.astype(jnp.float32)
    weighted_loss = jnp.sum(loss * weights) / (jnp.sum(weights) + 1e-8)
    return weighted_loss


class SelfTrainer:
    """Self-training manager for Phase D.

    Args:
        num_rounds: Number of self-training rounds (R=3).
        epochs_per_round: Epochs per round (E_r=5).
        initial_threshold: Initial confidence threshold (τ_1=0.7).
        threshold_increment: Per-round threshold increase (Δτ=0.05).
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
        teacher_outputs: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
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
