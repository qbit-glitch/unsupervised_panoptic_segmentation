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
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        num_points: int = 12544,
        num_stuff_classes: int = 0,
        num_queries_stuff: int = 0,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points
        self.num_stuff_classes = num_stuff_classes
        self.num_queries_stuff = num_queries_stuff
        self.decoupled = (num_stuff_classes > 0 and num_queries_stuff > 0)

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
            # Align pred to target resolution, then sample K' shared points.
            _, H, W = tgt_masks.shape
            src_b = F.interpolate(
                pred_masks[b].unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(1)
            pnts = torch.randint(0, H * W, (self.num_points,), device=pred_masks.device)
            out_mask = src_b.flatten(1)[:, pnts]                # Q, P
            tgt_mask = tgt_masks.flatten(1)[:, pnts]            # N, P
            prob = pred_logits[b].softmax(-1)                   # Q, K

            if self.decoupled:
                # Split queries and targets by type
                stuff_q_idx = torch.arange(self.num_queries_stuff, device=pred_masks.device)
                thing_q_idx = torch.arange(self.num_queries_stuff, Q, device=pred_masks.device)
                stuff_t_mask = tgt_labels < self.num_stuff_classes
                thing_t_mask = ~stuff_t_mask

                src = []
                tgt = []

                # Match stuff queries to stuff targets
                if stuff_q_idx.numel() > 0 and stuff_t_mask.any():
                    sq = stuff_q_idx
                    st = stuff_t_mask.nonzero(as_tuple=True)[0]
                    cost_c = -prob[sq][:, tgt_labels[st]]
                    cost_m = _sigmoid_focal_cost(out_mask[sq], tgt_mask[st])
                    cost_d = _dice_cost(out_mask[sq], tgt_mask[st])
                    C = (self.cost_class * cost_c + self.cost_mask * cost_m + self.cost_dice * cost_d).cpu()
                    row, col = linear_sum_assignment(C.numpy())
                    src.append(sq[row])
                    tgt.append(st[col])

                # Match thing queries to thing targets
                if thing_q_idx.numel() > 0 and thing_t_mask.any():
                    tq = thing_q_idx
                    tt = thing_t_mask.nonzero(as_tuple=True)[0]
                    # Thing class logits are offset by num_stuff_classes in the decoder,
                    # but pred_logits already concatenates stuff+thing logits.
                    # We need to index the correct columns for thing targets.
                    cost_c = -prob[tq][:, tgt_labels[tt]]
                    cost_m = _sigmoid_focal_cost(out_mask[tq], tgt_mask[tt])
                    cost_d = _dice_cost(out_mask[tq], tgt_mask[tt])
                    C = (self.cost_class * cost_c + self.cost_mask * cost_m + self.cost_dice * cost_d).cpu()
                    row, col = linear_sum_assignment(C.numpy())
                    src.append(tq[row])
                    tgt.append(tt[col])

                if len(src) == 0:
                    indices.append((torch.empty(0, dtype=torch.long, device=pred_masks.device),
                                    torch.empty(0, dtype=torch.long, device=pred_masks.device)))
                else:
                    indices.append((torch.cat(src), torch.cat(tgt)))
            else:
                # Standard shared matching (backward compatible)
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
