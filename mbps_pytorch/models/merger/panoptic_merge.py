"""Panoptic Merger Module.

Combines semantic segmentation, instance masks, and stuff-things
classification into a unified panoptic segmentation output.

Implements Algorithm 9 from the technical report.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import torch


class PanopticOutput(NamedTuple):
    """Panoptic segmentation output container."""

    panoptic_seg: torch.Tensor   # (H, W) with encoded panoptic IDs
    semantic_seg: torch.Tensor   # (H, W) semantic class per pixel
    instance_seg: torch.Tensor   # (H, W) instance ID per pixel (0=stuff)
    segments_info: list          # metadata per segment


def panoptic_merge(
    semantic_pred: torch.Tensor,
    instance_masks: torch.Tensor,
    instance_scores: torch.Tensor,
    thing_clusters: torch.Tensor,
    stuff_clusters: torch.Tensor,
    overlap_threshold: float = 0.5,
    score_threshold: float = 0.3,
    min_stuff_area: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge semantic and instance predictions into panoptic output.

    Implements Algorithm 9: PANOPTIC-MERGING from the technical report.

    Args:
        semantic_pred: Semantic class predictions of shape (N,).
        instance_masks: Instance mask logits of shape (M, N).
        instance_scores: Instance confidence scores of shape (M,).
        thing_clusters: Boolean mask for thing clusters of shape (K,).
        stuff_clusters: Boolean mask for stuff clusters of shape (K,).
        overlap_threshold: Max overlap for accepting an instance.
        score_threshold: Min score for accepting an instance.
        min_stuff_area: Minimum pixel count for stuff segments.

    Returns:
        Tuple of:
            - panoptic_ids: Per-pixel panoptic IDs of shape (N,).
              Encoded as: panoptic_id = instance_id * max_classes + semantic_class
            - instance_ids: Per-pixel instance IDs of shape (N,).
              0 for stuff pixels, >0 for thing instances.
    """
    n = semantic_pred.shape[0]
    m = instance_masks.shape[0]
    k = max(int(semantic_pred.max().item()) + 1, 1)

    device = semantic_pred.device

    # Initialize outputs
    instance_ids = torch.zeros(n, dtype=torch.int32, device=device)
    used_pixels = torch.zeros(n, dtype=torch.bool, device=device)

    # Convert mask logits to probabilities
    mask_probs = torch.sigmoid(instance_masks)  # (M, N)

    # Sort instances by score (descending)
    sorted_idx = torch.argsort(instance_scores, descending=True)

    # Counter for unique instance IDs
    instance_counter = 0

    # Process thing instances (highest score first)
    for rank in range(m):
        i = sorted_idx[rank]
        score = instance_scores[i]

        # Skip low-confidence instances
        if score < score_threshold:
            continue

        mask_i = mask_probs[i] > 0.5  # Binary mask

        # Get majority semantic class
        class_counts = torch.zeros(k, device=device)
        for c in range(k):
            class_counts[c] = torch.sum((semantic_pred == c) & mask_i).float()
        majority_class = torch.argmax(class_counts)

        # Check if majority class is a thing class
        if majority_class < thing_clusters.shape[0] and thing_clusters[majority_class]:
            # Check overlap with already assigned pixels
            overlap = torch.sum(mask_i & used_pixels).float() / (
                torch.sum(mask_i).float() + 1e-8
            )

            if overlap < overlap_threshold:
                instance_counter += 1
                valid_pixels = mask_i & ~used_pixels
                instance_ids = torch.where(
                    valid_pixels,
                    torch.tensor(instance_counter, dtype=torch.int32, device=device),
                    instance_ids,
                )
                used_pixels = used_pixels | valid_pixels

    # Stuff regions: fill unassigned pixels
    # Stuff pixels get instance_id = 0 (already default)

    # Encode panoptic IDs: id * label_divisor + class
    label_divisor = 1000
    panoptic_ids = instance_ids * label_divisor + semantic_pred.to(torch.int32)

    return panoptic_ids, instance_ids


def batch_panoptic_merge(
    semantic_pred: torch.Tensor,
    instance_masks: torch.Tensor,
    instance_scores: torch.Tensor,
    thing_mask: torch.Tensor,
    stuff_mask: torch.Tensor,
    overlap_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch version of panoptic merge.

    Args:
        semantic_pred: Shape (B, N) -- per-pixel semantic classes.
        instance_masks: Shape (B, M, N) -- instance mask logits.
        instance_scores: Shape (B, M) -- instance scores.
        thing_mask: Shape (B, K) -- boolean mask for thing clusters.
        stuff_mask: Shape (B, K) -- boolean mask for stuff clusters.
        overlap_threshold: Max overlap threshold.

    Returns:
        Tuple of:
            - panoptic_ids: Shape (B, N).
            - instance_ids: Shape (B, N).
    """
    b = semantic_pred.shape[0]
    results_pan = []
    results_inst = []

    for i in range(b):
        pan, inst = panoptic_merge(
            semantic_pred[i],
            instance_masks[i],
            instance_scores[i],
            thing_mask[i],
            stuff_mask[i],
            overlap_threshold=overlap_threshold,
        )
        results_pan.append(pan)
        results_inst.append(inst)

    return torch.stack(results_pan), torch.stack(results_inst)
