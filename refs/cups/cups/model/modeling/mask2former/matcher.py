"""Hungarian bipartite matcher for Mask2Former.

Adapted from the reference implementation in
refs/cutler/videocutler/mask2former/modeling/matcher.py with no video
logic and with an optional point sampling to keep cost matrix memory
bounded (num_points random points per mask).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

__all__ = ["HungarianMatcher"]


def _sigmoid_focal_cost(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Batched focal BCE cost between Q logit-vectors and N 0/1-targets."""
    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    focal_pos = alpha * focal_pos
    focal_neg = (1 - alpha) * focal_neg
    # inputs: (Q, P), targets: (N, P) -> (Q, N)
    return focal_pos @ targets.T + focal_neg @ (1 - targets).T


def _dice_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = inputs.sigmoid()
    numerator = 2.0 * (inputs @ targets.T)
    denominator = inputs.sum(-1, keepdim=True) + targets.sum(-1).unsqueeze(0)
    return 1 - (numerator + 1) / (denominator + 1)


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 2.0, cost_mask: float = 5.0, cost_dice: float = 5.0, num_points: int = 12544) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        pred_logits = outputs["pred_logits"]          # B, Q, K
        pred_masks = outputs["pred_masks"]             # B, Q, H, W
        B, Q, K = pred_logits.shape
        indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for b in range(B):
            tgt_labels = targets[b]["labels"]           # N
            tgt_masks = targets[b]["masks"].float()     # N, H, W
            if tgt_labels.numel() == 0:
                indices.append(
                    (
                        torch.empty(0, dtype=torch.long, device=pred_masks.device),
                        torch.empty(0, dtype=torch.long, device=pred_masks.device),
                    )
                )
                continue
            # Sample K' points per mask for tractable matching.
            _, H, W = tgt_masks.shape
            pnts = torch.randint(0, H * W, (self.num_points,), device=pred_masks.device)
            out_mask = pred_masks[b].flatten(1)[:, pnts]        # Q, P
            tgt_mask = tgt_masks.flatten(1)[:, pnts]            # N, P
            # Class cost: -softmax[c_gt] (Mask2Former uses raw prob, not log-prob).
            prob = pred_logits[b].softmax(-1)                   # Q, K
            cost_class = -prob[:, tgt_labels]                   # Q, N
            cost_mask = _sigmoid_focal_cost(out_mask, tgt_mask) # Q, N
            cost_dice = _dice_cost(out_mask, tgt_mask)          # Q, N
            C = (
                self.cost_class * cost_class
                + self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
            ).cpu()
            row, col = linear_sum_assignment(C.numpy())
            indices.append((torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long)))
        return indices
