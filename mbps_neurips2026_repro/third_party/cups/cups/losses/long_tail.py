from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class EQLv2Loss(nn.Module):
    """Softmax-head-compatible EQLv2-style foreground BCE loss.

    The CUPS ROI head is a softmax classifier with an explicit background
    column. EQLv2 is gradient-balancing BCE over foreground classes, so this
    module applies BCE to the foreground logits and treats background RoIs as
    all-negative foreground targets.
    """

    def __init__(
        self,
        num_classes: int,
        gamma: float = 12.0,
        mu: float = 0.8,
        alpha: float = 4.0,
        momentum: float = 0.99,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.gamma = float(gamma)
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("pos_grad", torch.zeros(self.num_classes))
        self.register_buffer("neg_grad", torch.zeros(self.num_classes))

    def _weights(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ratio = self.pos_grad.to(device) / (self.neg_grad.to(device) + self.eps)
        neg_w = torch.sigmoid(self.gamma * (ratio - self.mu))
        pos_w = 1.0 + self.alpha * (1.0 - neg_w)
        return pos_w.detach(), neg_w.detach()

    @torch.no_grad()
    def _collect_grad(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prob = logits.sigmoid()
        grad = (prob - target).abs()
        pos = (grad * target).sum(dim=0)
        neg = (grad * (1.0 - target)).sum(dim=0)
        self.pos_grad.mul_(self.momentum).add_(pos.detach().to(self.pos_grad.device), alpha=1.0 - self.momentum)
        self.neg_grad.mul_(self.momentum).add_(neg.detach().to(self.neg_grad.device), alpha=1.0 - self.momentum)

    def forward(
        self,
        scores: torch.Tensor,
        gt_classes: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if scores.numel() == 0:
            loss = scores.sum() * 0.0
            return loss if reduction != "none" else scores.new_zeros((0,))

        logits = scores[:, : self.num_classes]
        valid = gt_classes >= 0
        logits = logits[valid]
        gt_classes = gt_classes[valid].long()
        if weights is not None:
            weights = weights[valid]
        if logits.numel() == 0:
            loss_vec = scores.new_zeros((0,))
            return loss_vec if reduction == "none" else scores.sum() * 0.0

        target = logits.new_zeros((logits.shape[0], self.num_classes))
        fg = (gt_classes >= 0) & (gt_classes < self.num_classes)
        if fg.any():
            target[fg, gt_classes[fg]] = 1.0

        if self.training:
            self._collect_grad(logits.detach(), target.detach())

        pos_w, neg_w = self._weights(logits.device)
        cls_weight = target * pos_w.view(1, -1) + (1.0 - target) * neg_w.view(1, -1)
        loss_vec = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss_vec = (loss_vec * cls_weight).sum(dim=1)
        if weights is not None:
            loss_vec = loss_vec * weights.to(loss_vec.device)

        if reduction == "none":
            return loss_vec
        if loss_vec.numel() == 0:
            return scores.sum() * 0.0
        return loss_vec.mean()


class SeesawSoftmaxLoss(nn.Module):
    """Small Seesaw-loss implementation for CUPS ablations."""

    def __init__(self, num_classes: int, p: float = 0.8, q: float = 2.0, eps: float = 1e-2) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.p = float(p)
        self.q = float(q)
        self.eps = float(eps)
        self.register_buffer("cum_samples", torch.zeros(self.num_classes + 1))

    @torch.no_grad()
    def _update_counts(self, gt_classes: torch.Tensor) -> None:
        valid = (gt_classes >= 0) & (gt_classes <= self.num_classes)
        if valid.any():
            counts = torch.bincount(gt_classes[valid].detach().cpu(), minlength=self.num_classes + 1).float()
            self.cum_samples.add_(counts)

    def forward(
        self,
        scores: torch.Tensor,
        gt_classes: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if scores.numel() == 0:
            loss = scores.sum() * 0.0
            return loss if reduction != "none" else scores.new_zeros((0,))
        valid = gt_classes >= 0
        scores = scores[valid]
        gt_classes = gt_classes[valid].long()
        if weights is not None:
            weights = weights[valid]
        if scores.numel() == 0:
            loss_vec = scores.new_zeros((0,))
            return loss_vec if reduction == "none" else scores.sum() * 0.0

        if self.training:
            self._update_counts(gt_classes)

        sample_counts = self.cum_samples.to(scores.device).clamp_min(self.eps)
        target_counts = sample_counts[gt_classes].unsqueeze(1)
        mitigation = torch.ones_like(scores)
        class_counts = sample_counts.view(1, -1)
        mitigation = torch.where(class_counts > target_counts, (target_counts / class_counts).pow(self.p), mitigation)

        probs = F.softmax(scores.detach(), dim=1)
        target_probs = probs.gather(1, gt_classes.clamp_max(self.num_classes).view(-1, 1)).clamp_min(self.eps)
        compensation = torch.ones_like(scores)
        compensation = torch.where(probs > target_probs, (probs / target_probs).pow(self.q), compensation)

        seesaw_weights = mitigation * compensation
        one_hot = F.one_hot(gt_classes.clamp_max(self.num_classes), num_classes=self.num_classes + 1).bool()
        seesaw_weights = torch.where(one_hot, torch.ones_like(seesaw_weights), seesaw_weights)
        logits = scores + seesaw_weights.clamp_min(self.eps).log()
        loss_vec = F.cross_entropy(logits, gt_classes, reduction="none")
        if weights is not None:
            loss_vec = loss_vec * weights.to(loss_vec.device)

        if reduction == "none":
            return loss_vec
        return loss_vec.mean() if loss_vec.numel() else scores.sum() * 0.0


class LDAMSemanticLoss(nn.Module):
    """LDAM for dense semantic logits.

    Margins are larger for rarer classes and are subtracted only from the
    target-class logit before the scaled cross-entropy term.
    """

    def __init__(
        self,
        num_classes: int,
        class_freq: Sequence[float] | torch.Tensor,
        max_margin: float = 0.5,
        s: float = 30.0,
        class_weight: Optional[Sequence[float]] = None,
        ignore_index: int = -1,
    ) -> None:
        super().__init__()
        freq = np.asarray(list(class_freq), dtype=np.float32)
        if freq.size == 0:
            freq = np.ones(num_classes, dtype=np.float32)
        if freq.size < num_classes:
            freq = np.pad(freq, (0, num_classes - freq.size), constant_values=float(freq.mean()))
        freq = freq[:num_classes]
        margins = 1.0 / np.power(freq + 1e-6, 0.25)
        margins = margins * (float(max_margin) / max(float(margins.max()), 1e-12))
        self.register_buffer("margins", torch.from_numpy(margins).float())
        self.s = float(s)
        self.ignore_index = int(ignore_index)
        self.class_weight = tuple(float(v) for v in class_weight) if class_weight is not None else None

    def adjust_logits(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_value: int | None = None,
    ) -> torch.Tensor:
        ignore_index = self.ignore_index if ignore_value is None else int(ignore_value)
        _, num_classes, _, _ = logits.shape
        valid = targets != ignore_index
        t_safe = targets.clamp(0, num_classes - 1)
        margins = self.margins.to(logits.device)[t_safe] * valid.to(logits.dtype)
        margin_map = torch.zeros_like(logits)
        margin_map.scatter_(1, t_safe.unsqueeze(1), margins.unsqueeze(1))
        return self.s * (logits - margin_map)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
        class_weight: Optional[torch.Tensor] = None,
        ignore_value: int | None = None,
        **_: object,
    ) -> torch.Tensor:
        ignore_index = self.ignore_index if ignore_value is None else int(ignore_value)
        adj_logits = self.adjust_logits(logits, targets, ignore_value=ignore_index)
        weight = class_weight
        if weight is None and self.class_weight is not None:
            weight = torch.tensor(self.class_weight, device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(
            adj_logits,
            targets,
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
        )
