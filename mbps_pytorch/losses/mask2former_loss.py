"""Mask2Former Loss with Hungarian Matching.

Implements bipartite matching between predicted queries and ground truth segments,
then computes:
    - Cross-entropy loss for class predictions (with no-object class)
    - Point-sampled binary cross-entropy for mask predictions
    - Point-sampled Dice loss for mask predictions
    - Deep supervision: loss applied at all decoder layer outputs
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def dice_loss(
    inputs: torch.Tensor, targets: torch.Tensor, num_masks: int,
) -> torch.Tensor:
    """Dice loss on point-sampled masks.

    Args:
        inputs: (N_matched, num_points) sigmoid-activated mask predictions.
        targets: (N_matched, num_points) binary ground truth.
        num_masks: Total masks across batch (for normalization).
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / (num_masks + 1e-8)


def sigmoid_ce_loss(
    inputs: torch.Tensor, targets: torch.Tensor, num_masks: int,
) -> torch.Tensor:
    """Sigmoid BCE loss on point-sampled masks.

    Args:
        inputs: (N_matched, num_points) raw mask logits.
        targets: (N_matched, num_points) binary ground truth.
        num_masks: Total masks across batch (for normalization).
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / (num_masks + 1e-8)


def sample_points_from_masks(
    masks: torch.Tensor, num_points: int,
) -> torch.Tensor:
    """Sample random flat indices for point-based mask loss.

    Args:
        masks: (N, H, W) mask tensor.
        num_points: Number of points to sample.

    Returns:
        (N, num_points) flat indices into H*W.
    """
    N, H, W = masks.shape
    total = H * W
    # Sample random flat indices
    indices = torch.randint(0, total, (N, num_points), device=masks.device)
    return indices


def gather_from_masks(
    masks: torch.Tensor, indices: torch.Tensor,
) -> torch.Tensor:
    """Gather values from masks at given flat indices.

    Args:
        masks: (N, H, W) mask tensor.
        indices: (N, P) flat indices.

    Returns:
        (N, P) sampled values.
    """
    N = masks.shape[0]
    flat = masks.reshape(N, -1)  # (N, H*W)
    return torch.gather(flat, 1, indices)  # (N, P)


class Mask2FormerCriterion(nn.Module):
    """Hungarian matching + mask classification loss for Mask2Former.

    Args:
        num_classes: Number of semantic classes (19 for Cityscapes).
        eos_coef: Weight for the no-object class in CE loss (0.1).
        cost_class: Matching cost weight for class probability.
        cost_mask: Matching cost weight for mask BCE.
        cost_dice: Matching cost weight for mask Dice.
        weight_class: Loss weight for class CE.
        weight_mask: Loss weight for mask BCE.
        weight_dice: Loss weight for mask Dice.
        num_points: Number of points for point-sampled mask loss.
        deep_supervision: Apply loss at all decoder layer outputs.
    """

    def __init__(
        self,
        num_classes: int = 19,
        eos_coef: float = 0.1,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        weight_class: float = 2.0,
        weight_mask: float = 5.0,
        weight_dice: float = 5.0,
        num_points: int = 12544,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.weight_class = weight_class
        self.weight_mask = weight_mask
        self.weight_dice = weight_dice
        self.num_points = num_points
        self.deep_supervision = deep_supervision

        # Class weights: lower weight for no-object class
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Compute total loss with deep supervision.

        Args:
            outputs: {pred_logits: (B,Q,C+1), pred_masks: (B,Q,H,W), aux_outputs: [...]}
            targets: List of B dicts, each with:
                labels: (M,) int64 class indices.
                masks: (M, H, W) binary float masks at model output resolution.

        Returns:
            Dict of loss components + total_loss.
        """
        # Final layer loss
        losses = self._compute_loss(outputs, targets)

        # Deep supervision: auxiliary losses from intermediate layers
        if self.deep_supervision and "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_losses = self._compute_loss(aux, targets)
                for k, v in aux_losses.items():
                    losses[f"{k}_aux_{i}"] = v

        # Total
        total = torch.tensor(0.0, device=losses["loss_ce"].device)
        for k, v in losses.items():
            if k.startswith("loss_ce"):
                total = total + self.weight_class * v
            elif k.startswith("loss_mask"):
                total = total + self.weight_mask * v
            elif k.startswith("loss_dice"):
                total = total + self.weight_dice * v

        losses["total_loss"] = total
        return losses

    def _compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Compute loss for one decoder output layer."""
        pred_logits = outputs["pred_logits"]  # (B, Q, C+1)
        pred_masks = outputs["pred_masks"]    # (B, Q, H, W)
        B = pred_logits.shape[0]

        # Hungarian matching
        indices = self._hungarian_match(pred_logits, pred_masks, targets)

        # Class loss
        loss_ce = self._loss_class(pred_logits, targets, indices)

        # Mask losses (point-sampled)
        loss_mask, loss_dice = self._loss_masks(pred_masks, targets, indices)

        return {
            "loss_ce": loss_ce,
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
        }

    @torch.no_grad()
    def _hungarian_match(
        self,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Bipartite matching between predictions and targets.

        Returns:
            List of B (pred_indices, target_indices) tuples.
        """
        B, Q = pred_logits.shape[:2]
        indices = []

        for b in range(B):
            tgt_labels = targets[b]["labels"]   # (M,)
            tgt_masks = targets[b]["masks"]     # (M, H_t, W_t)
            M = len(tgt_labels)

            if M == 0:
                indices.append((
                    torch.tensor([], dtype=torch.int64, device=pred_logits.device),
                    torch.tensor([], dtype=torch.int64, device=pred_logits.device),
                ))
                continue

            # Class cost: -P(target_class)
            probs = pred_logits[b].softmax(-1)  # (Q, C+1)
            cost_class = -probs[:, tgt_labels]  # (Q, M)

            # Mask costs: point-sampled via flat indexing (MPS-compatible)
            H_m, W_m = pred_masks.shape[-2:]
            num_match_points = min(self.num_points, H_m * W_m)
            # Shared random indices for this image
            pt_indices = torch.randint(0, H_m * W_m, (num_match_points,), device=pred_masks.device)

            # Sample predicted masks at points: (Q, P)
            pred_flat = pred_masks[b].reshape(Q, -1)  # (Q, H*W)
            pred_sampled = pred_flat[:, pt_indices]     # (Q, P)

            # Sample target masks at points (resize to pred mask resolution first)
            tgt_masks_resized = F.interpolate(
                tgt_masks.unsqueeze(1).float(),
                size=(H_m, W_m),
                mode="bilinear", align_corners=False,
            ).squeeze(1)  # (M, H, W)
            tgt_flat = tgt_masks_resized.reshape(M, -1)  # (M, H*W)
            tgt_sampled = tgt_flat[:, pt_indices]          # (M, P)

            # BCE cost: compute per-pair (Q, M)
            pred_sigmoid = pred_sampled.sigmoid()  # (Q, P)
            # Expand for pairwise: (Q, 1, P) vs (1, M, P)
            pred_expand = pred_sampled[:, None, :].expand(-1, M, -1)  # (Q, M, P)
            tgt_expand = tgt_sampled[None, :, :].expand(Q, -1, -1)    # (Q, M, P)
            cost_mask = F.binary_cross_entropy_with_logits(
                pred_expand, tgt_expand, reduction="none"
            ).mean(-1)  # (Q, M)

            # Dice cost: compute per-pair (Q, M)
            pred_sig_expand = pred_sigmoid[:, None, :].expand(-1, M, -1)  # (Q, M, P)
            numerator = 2 * (pred_sig_expand * tgt_expand).sum(-1)
            denominator = pred_sig_expand.sum(-1) + tgt_expand.sum(-1)
            cost_dice_val = 1 - (numerator + 1) / (denominator + 1)  # (Q, M)

            # Total cost
            cost = (self.cost_class * cost_class
                    + self.cost_mask * cost_mask
                    + self.cost_dice * cost_dice_val)

            cost_np = cost.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            indices.append((
                torch.tensor(row_ind, dtype=torch.int64, device=pred_logits.device),
                torch.tensor(col_ind, dtype=torch.int64, device=pred_logits.device),
            ))

        return indices

    def _loss_class(
        self,
        pred_logits: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Cross-entropy loss for class predictions."""
        B, Q = pred_logits.shape[:2]

        # Target labels: no-object for unmatched queries
        target_classes = torch.full(
            (B, Q), self.num_classes,
            dtype=torch.int64, device=pred_logits.device,
        )
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[b, pred_idx] = targets[b]["labels"][tgt_idx]

        loss = F.cross_entropy(
            pred_logits.flatten(0, 1),  # (B*Q, C+1)
            target_classes.flatten(),    # (B*Q,)
            weight=self.empty_weight,
        )
        return loss

    def _loss_masks(
        self,
        pred_masks: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Point-sampled mask BCE + Dice loss for matched predictions."""
        device = pred_masks.device
        total_masks = sum(len(idx[0]) for idx in indices)

        if total_masks == 0:
            return (torch.tensor(0.0, device=device),
                    torch.tensor(0.0, device=device))

        all_pred = []
        all_tgt = []

        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            matched_pred = pred_masks[b, pred_idx]  # (K, H, W)
            tgt_masks = targets[b]["masks"][tgt_idx]  # (K, H_t, W_t)

            # Resize targets to pred resolution
            tgt_resized = F.interpolate(
                tgt_masks.unsqueeze(1).float(),
                size=pred_masks.shape[-2:],
                mode="bilinear", align_corners=False,
            ).squeeze(1)  # (K, H, W)

            all_pred.append(matched_pred)
            all_tgt.append(tgt_resized)

        all_pred = torch.cat(all_pred, dim=0)  # (N_total, H, W)
        all_tgt = torch.cat(all_tgt, dim=0)    # (N_total, H, W)

        # Point sample via flat indexing (MPS-compatible, no grid_sample)
        N, H_m, W_m = all_pred.shape
        num_pts = min(self.num_points, H_m * W_m)
        pt_indices = torch.randint(0, H_m * W_m, (N, num_pts), device=device)

        pred_sampled = torch.gather(all_pred.reshape(N, -1), 1, pt_indices)  # (N, P)
        tgt_sampled = torch.gather(all_tgt.reshape(N, -1), 1, pt_indices)    # (N, P)

        loss_mask = sigmoid_ce_loss(pred_sampled, tgt_sampled, total_masks)
        loss_dice = dice_loss(pred_sampled.sigmoid(), tgt_sampled, total_masks)

        return loss_mask, loss_dice
