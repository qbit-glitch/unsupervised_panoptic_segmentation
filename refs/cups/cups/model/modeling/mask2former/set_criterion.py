"""Mask2Former SetCriterion (Hungarian + focal + dice + deep supervision).

Closely follows the reference implementation
(refs/cutler/videocutler/mask2former/modeling/criterion.py) but drops the
background-removal/video-specific paths and simplifies point sampling.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .matcher import HungarianMatcher

__all__ = ["SetCriterion"]


def _dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    num = 2.0 * (inputs * targets).sum(-1)
    den = inputs.sum(-1) + targets.sum(-1)
    return 1 - (num + 1) / (den + 1)


def _sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    return loss.mean(1).sum() / max(inputs.shape[0], 1)


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float = 0.1,
        losses: Sequence[str] = ("labels", "masks"),
        num_points: int = 12544,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = dict(weight_dict)
        self.eos_coef = eos_coef
        self.losses = tuple(losses)
        self.num_points = num_points
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _get_src_permutation_idx(self, indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        src_logits = outputs["pred_logits"]                    # B, Q, K+1 (we let last be "no-object")
        B, Q, K1 = src_logits.shape
        target_classes = torch.full((B, Q), self.num_classes, dtype=torch.long, device=src_logits.device)
        for b, (src, tgt) in enumerate(indices):
            target_classes[b, src] = targets[b]["labels"][tgt]
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_masks(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]              # (sum N_b), H, W
        tgt_masks = torch.cat(
            [t["masks"][idx].float() for t, (_, idx) in zip(targets, indices)], dim=0
        )                                                       # (sum N_b), H_gt, W_gt
        if src_masks.numel() == 0:
            zero = src_masks.sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero}
        # Align shapes (resize src to tgt)
        src_masks = F.interpolate(src_masks.unsqueeze(1), size=tgt_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        # Point sampling for mask loss (memory-safe)
        H, W = tgt_masks.shape[-2:]
        pnts = torch.randint(0, H * W, (self.num_points,), device=src_masks.device)
        src_pt = src_masks.flatten(1)[:, pnts]
        tgt_pt = tgt_masks.flatten(1)[:, pnts]
        loss_mask = _sigmoid_focal_loss(src_pt, tgt_pt)
        loss_dice = _dice_loss(src_pt, tgt_pt).mean()
        return {"loss_mask": loss_mask, "loss_dice": loss_dice}

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)
        losses: Dict[str, torch.Tensor] = {}
        for loss_name in self.losses:
            if loss_name == "labels":
                losses.update(self.loss_labels(outputs_without_aux, targets, indices))
            elif loss_name == "masks":
                losses.update(self.loss_masks(outputs_without_aux, targets, indices))
        # Deep supervision on aux outputs.
        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                idx = self.matcher(aux, targets)
                for loss_name in self.losses:
                    if loss_name == "labels":
                        for k, v in self.loss_labels(aux, targets, idx).items():
                            losses[f"{k}_{i}"] = v
                    elif loss_name == "masks":
                        for k, v in self.loss_masks(aux, targets, idx).items():
                            losses[f"{k}_{i}"] = v
        # Apply weight dict.
        weighted = {}
        for k, v in losses.items():
            base = k.rsplit("_", 1)[0] if k.rsplit("_", 1)[-1].isdigit() else k
            w = self.weight_dict.get(base, 1.0)
            weighted[k] = v * w
        return weighted
