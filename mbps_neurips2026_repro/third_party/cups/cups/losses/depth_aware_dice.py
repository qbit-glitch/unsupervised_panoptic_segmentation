"""Depth-aware proposal consistency loss for the mask head.

Penalizes foreground proposals whose bounding-box region spans a large depth
range, which is a strong signal that the proposal merges two separate objects
at different depths (e.g., two pedestrians at different distances).

Reference: inspired by Depth-Aware instance losses (ISPRS 2024, arXiv:2405.10947),
adapted to work on proposal crops rather than predicted mask logits so it can be
added as a pure auxiliary loss without modifying Detectron2 mask head internals.

Usage (from CustomStandardROIHeads._forward_mask):
    loss = proposal_depth_variance_loss(instances, depth_maps, weight=0.1)
    if loss is not None:
        losses["loss_mask_depth"] = loss
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from detectron2.structures import Instances


def proposal_depth_variance_loss(
    instances: List[Instances],
    depth_maps: torch.Tensor,
    weight: float = 0.1,
    min_area: int = 64,
) -> Optional[torch.Tensor]:
    """Compute mean depth variance over foreground proposal crops.

    For each foreground proposal, crops the depth map to the proposal's
    bounding box and computes the depth standard deviation.  High std → the
    proposal spans two depth planes → loss penalises it.

    Args:
        instances: list[Instances] of length B (one per image), each with
            .proposal_boxes and a .gt_classes field (after label assignment).
            Only proposals with gt_classes >= 0 (foreground) are considered.
        depth_maps: (B, 1, H, W) depth tensor, values in [0, 1].
        weight: scalar multiplier applied to the returned loss.
        min_area: skip proposals whose crop area (pixels) is below this.

    Returns:
        Scalar loss tensor, or None if no foreground proposals are found.
    """
    if weight == 0.0:
        return None

    B, _, H, W = depth_maps.shape
    variances: List[torch.Tensor] = []

    for img_idx, inst in enumerate(instances):
        if not inst.has("proposal_boxes") or not inst.has("gt_classes"):
            continue
        fg_mask = inst.gt_classes >= 0
        if not fg_mask.any():
            continue

        boxes = inst.proposal_boxes.tensor[fg_mask]  # (N_fg, 4)
        depth_img = depth_maps[img_idx]  # (1, H, W)

        for box in boxes:
            x1, y1, x2, y2 = box.long()
            x1 = x1.clamp(0, W - 1)
            y1 = y1.clamp(0, H - 1)
            x2 = x2.clamp(x1 + 1, W)
            y2 = y2.clamp(y1 + 1, H)

            crop = depth_img[:, y1:y2, x1:x2]  # (1, h, w)
            if crop.numel() < min_area:
                continue

            # Depth std within crop (0 = uniform depth = one plane)
            variances.append(crop.std())

    if not variances:
        return None

    return weight * torch.stack(variances).mean()
