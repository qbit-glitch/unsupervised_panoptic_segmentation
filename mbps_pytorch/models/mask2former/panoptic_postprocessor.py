"""Panoptic post-processing for Mask2Former inference.

Converts per-query class logits + mask logits into a unified panoptic map.
- Stuff classes: merge all queries predicting the same stuff class.
- Thing classes: each query becomes a separate instance.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch.nn import functional as F


class PanopticResult(NamedTuple):
    panoptic_map: torch.Tensor  # (H, W) int64, panoptic IDs
    segments_info: list[dict]   # Per-segment: {id, category_id, is_thing, score}


class PanopticPostProcessor:
    """Convert Mask2Former outputs to panoptic predictions.

    Args:
        thing_classes: Set of thing class IDs.
        stuff_classes: Set of stuff class IDs.
        score_threshold: Minimum confidence to keep a query.
        overlap_threshold: Maximum overlap fraction for thing instance filtering.
        label_divisor: Multiplier for panoptic ID encoding (category * divisor + instance).
    """

    def __init__(
        self,
        thing_classes: set[int],
        stuff_classes: set[int],
        score_threshold: float = 0.5,
        overlap_threshold: float = 0.8,
        label_divisor: int = 1000,
    ):
        self.thing_classes = thing_classes
        self.stuff_classes = stuff_classes
        self.score_threshold = score_threshold
        self.overlap_threshold = overlap_threshold
        self.label_divisor = label_divisor

    @torch.no_grad()
    def __call__(
        self,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        target_size: tuple[int, int],
    ) -> list[PanopticResult]:
        """Post-process a batch of predictions.

        Args:
            pred_logits: (B, Q, num_classes+1) class logits.
            pred_masks: (B, Q, H/4, W/4) mask logits.
            target_size: (H, W) output resolution.

        Returns:
            List of B PanopticResult.
        """
        B = pred_logits.shape[0]
        num_classes = pred_logits.shape[-1] - 1  # Exclude no-object class

        # Upsample masks to target resolution
        masks = F.interpolate(
            pred_masks, size=target_size,
            mode="bilinear", align_corners=False,
        )  # (B, Q, H, W)

        results = []
        for b in range(B):
            result = self._process_single(
                pred_logits[b], masks[b], num_classes,
            )
            results.append(result)

        return results

    def _process_single(
        self,
        logits: torch.Tensor,   # (Q, C+1)
        masks: torch.Tensor,    # (Q, H, W)
        num_classes: int,
    ) -> PanopticResult:
        """Process a single image."""
        H, W = masks.shape[-2:]

        # Get class probabilities (exclude no-object for scoring)
        probs = logits.softmax(-1)  # (Q, C+1)
        scores, labels = probs[:, :-1].max(-1)  # (Q,) scores and class labels

        # Filter by score
        keep = scores > self.score_threshold
        scores = scores[keep]
        labels = labels[keep]
        mask_logits = masks[keep]  # (K, H, W)

        if len(scores) == 0:
            return PanopticResult(
                panoptic_map=torch.zeros(H, W, dtype=torch.int64, device=masks.device),
                segments_info=[],
            )

        # Binary masks via sigmoid threshold
        mask_probs = mask_logits.sigmoid()  # (K, H, W)

        # Sort by score (highest first)
        sorted_idx = scores.argsort(descending=True)
        scores = scores[sorted_idx]
        labels = labels[sorted_idx]
        mask_probs = mask_probs[sorted_idx]

        panoptic_map = torch.zeros(H, W, dtype=torch.int64, device=masks.device)
        segments_info = []
        current_id = 0
        used_pixels = torch.zeros(H, W, dtype=torch.bool, device=masks.device)

        # Stuff accumulator: collect masks per stuff class
        stuff_masks: dict[int, torch.Tensor] = {}

        for i in range(len(scores)):
            cat_id = labels[i].item()
            score = scores[i].item()
            binary_mask = mask_probs[i] > 0.5

            if cat_id in self.thing_classes:
                # Thing: check overlap with already placed segments
                overlap = (binary_mask & used_pixels).sum().item()
                mask_area = binary_mask.sum().item()
                if mask_area == 0:
                    continue
                if overlap / mask_area > self.overlap_threshold:
                    continue

                # Place this instance
                current_id += 1
                seg_id = cat_id * self.label_divisor + current_id
                new_pixels = binary_mask & ~used_pixels
                panoptic_map[new_pixels] = seg_id
                used_pixels |= new_pixels

                segments_info.append({
                    "id": seg_id,
                    "category_id": cat_id,
                    "is_thing": True,
                    "score": score,
                    "area": new_pixels.sum().item(),
                })

            elif cat_id in self.stuff_classes:
                # Stuff: accumulate masks for same class
                if cat_id not in stuff_masks:
                    stuff_masks[cat_id] = torch.zeros(H, W, device=masks.device)
                stuff_masks[cat_id] = torch.max(stuff_masks[cat_id], mask_probs[i])

        # Place stuff segments
        for cat_id, soft_mask in stuff_masks.items():
            binary_mask = (soft_mask > 0.5) & ~used_pixels
            if binary_mask.sum() == 0:
                continue

            current_id += 1
            seg_id = cat_id * self.label_divisor + current_id
            panoptic_map[binary_mask] = seg_id
            used_pixels |= binary_mask

            segments_info.append({
                "id": seg_id,
                "category_id": cat_id,
                "is_thing": False,
                "score": soft_mask[binary_mask].mean().item(),
                "area": binary_mask.sum().item(),
            })

        return PanopticResult(
            panoptic_map=panoptic_map,
            segments_info=segments_info,
        )
