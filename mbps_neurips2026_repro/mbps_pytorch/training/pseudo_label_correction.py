"""Adaptive pseudo-label correction inspired by Uni-UVPT (NeurIPS 2023).

Ported from: https://github.com/huawei-noah/noah-research/tree/master/uni-uvpt
Reference: "Universal Unsupervised Visual Prompt Tuning for Source-Free Domain
            Adaptive Semantic Segmentation"

Monitors pseudo-label quality during training and adaptively corrects noisy
labels using the early-learning phenomenon.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PseudoLabelCorrector:
    """Online pseudo-label correction with IoU curve monitoring.

    Based on Uni-UVPT's adaptive pseudo-label correction strategy:
    1. Maintain a queue of per-class IoU estimates
    2. Monitor the slope of the IoU curve over time
    3. When slope difference exceeds threshold, trigger correction
    4. Correct noisy labels by re-assigning low-confidence predictions
    """

    def __init__(
        self,
        num_classes: int = 27,
        queue_length: int = 1000,
        curve_update_interval: int = 200,
        slope_diff_threshold: float = 0.95,
        trustable_quantile: float = 0.66,
        warmup_iters: int = 1000,
    ):
        self.num_classes = num_classes
        self.queue_length = queue_length
        self.curve_update_interval = curve_update_interval
        self.slope_diff_threshold = slope_diff_threshold
        self.trustable_quantile = trustable_quantile
        self.warmup_iters = warmup_iters

        # Per-class IoU history queues
        self.iou_queues: Dict[int, Deque[float]] = {
            i: deque(maxlen=queue_length) for i in range(num_classes)
        }

        # Correction state
        self.corrected_classes: set[int] = set()
        self.iteration = 0
        self.last_update_iter = 0

    def update(
        self,
        predictions: torch.Tensor,
        pseudo_labels: torch.Tensor,
        confidences: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update IoU tracking and optionally correct pseudo-labels.

        Args:
            predictions: (B, C, H, W) model predictions (logits or probs).
            pseudo_labels: (B, H, W) current pseudo-labels.
            confidences: Optional (B, H, W) confidence scores.

        Returns:
            corrected_labels: (B, H, W) potentially corrected pseudo-labels.
        """
        self.iteration += 1

        if self.iteration < self.warmup_iters:
            return pseudo_labels

        # Compute per-class IoU against predictions
        pred_labels = predictions.argmax(dim=1) if predictions.dim() == 4 else predictions
        self._update_iou_queues(pred_labels, pseudo_labels)

        # Check if it's time to evaluate correction
        if (self.iteration - self.last_update_iter) >= self.curve_update_interval:
            self._evaluate_correction()
            self.last_update_iter = self.iteration

        # Apply correction to noisy classes
        if self.corrected_classes:
            return self._correct_labels(predictions, pseudo_labels, confidences)

        return pseudo_labels

    def _update_iou_queues(
        self,
        pred_labels: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> None:
        """Compute per-class IoU and append to queues."""
        for cls in range(self.num_classes):
            pred_mask = (pred_labels == cls).float()
            gt_mask = (gt_labels == cls).float()

            intersection = (pred_mask * gt_mask).sum().item()
            union = ((pred_mask + gt_mask) > 0).float().sum().item()

            if union > 0:
                iou = intersection / union
                self.iou_queues[cls].append(iou)

    def _evaluate_correction(self) -> None:
        """Evaluate whether to correct pseudo-labels for each class."""
        for cls in range(self.num_classes):
            queue = self.iou_queues[cls]
            if len(queue) < self.queue_length // 2:
                continue

            # Split queue into two halves and compute slopes
            arr = np.array(queue)
            mid = len(arr) // 2
            first_half = arr[:mid]
            second_half = arr[mid:]

            if len(first_half) < 2 or len(second_half) < 2:
                continue

            # Linear regression slopes
            x1 = np.arange(len(first_half))
            x2 = np.arange(len(second_half))
            slope1 = np.polyfit(x1, first_half, 1)[0] if len(first_half) > 1 else 0
            slope2 = np.polyfit(x2, second_half, 1)[0] if len(second_half) > 1 else 0

            # If slope drops significantly, mark class for correction
            if abs(slope2 - slope1) > self.slope_diff_threshold:
                self.corrected_classes.add(cls)
                logger.info(
                    "Class %d marked for correction (slope1=%.4f, slope2=%.4f)",
                    cls, slope1, slope2,
                )

    def _correct_labels(
        self,
        predictions: torch.Tensor,
        pseudo_labels: torch.Tensor,
        confidences: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Correct pseudo-labels for noisy classes."""
        corrected = pseudo_labels.clone()

        # Get prediction probabilities
        if predictions.dim() == 4:
            probs = F.softmax(predictions, dim=1)
        else:
            probs = predictions

        for cls in self.corrected_classes:
            # Find pixels currently labeled as this class
            mask = (pseudo_labels == cls)
            if not mask.any():
                continue

            # Compute trustable threshold from confidence distribution
            if confidences is not None:
                cls_conf = confidences[mask]
                if cls_conf.numel() > 0:
                    threshold = torch.quantile(cls_conf, self.trustable_quantile)
                    # Keep only trustable pixels
                    trustable = mask & (confidences >= threshold)
                    corrected[~trustable & mask] = 255  # 255 = ignore / unlabeled
            else:
                # Use prediction entropy as proxy for confidence
                cls_probs = probs[:, cls, :, :]
                threshold = torch.quantile(cls_probs[mask], self.trustable_quantile)
                trustable = mask & (cls_probs >= threshold)
                corrected[~trustable & mask] = 255

        return corrected

    def reset(self) -> None:
        """Reset all tracking state."""
        self.iou_queues = {
            i: deque(maxlen=self.queue_length) for i in range(self.num_classes)
        }
        self.corrected_classes.clear()
        self.iteration = 0
        self.last_update_iter = 0


class SimplePseudoLabelFilter:
    """Simpler alternative: entropy-based filtering without curve monitoring.

    Filters pseudo-labels by per-pixel prediction entropy.
    """

    def __init__(self, entropy_threshold: float = 1.5):
        self.entropy_threshold = entropy_threshold

    def __call__(
        self,
        logits: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Filter high-entropy pseudo-labels.

        Args:
            logits: (B, C, H, W) prediction logits.
            pseudo_labels: (B, H, W) pseudo-labels.

        Returns:
            filtered_labels: (B, H, W) with high-entropy pixels set to 255.
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)  # (B, H, W)

        mask = entropy < self.entropy_threshold
        filtered = pseudo_labels.clone()
        filtered[~mask] = 255  # ignore index

        return filtered
