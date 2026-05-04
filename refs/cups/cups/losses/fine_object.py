"""Fine-object semantic loss using pre-computed SAM mask priors.

Encourages the semantic head to make confident predictions inside SAM-detected
fine-grained object regions (bicycle, motorcycle, rider, pole, etc.).

Architecture note
-----------------
The CUPS semantic head has k_stuff + 1 channels during training, where k_stuff
is the number of stuff pseudo-class clusters (~65 for k=80 overclustering).
**Channel 0** is the unified "thing region" class — ALL thing pseudo-class
pixels are remapped to channel 0 during dataloader processing
(see pseudo_label_dataset.py line 433: things_classes → 0, stuff_classes → 1..S).
The panoptic combine function (panoptic_fpn.py) explicitly skips semantic_label==0
and treats it as the thing region.  Channels 1..S are individual stuff pseudo-classes.
The individual per-thing-class identity (bicycle vs car vs person) lives in the
INSTANCE HEAD (Cascade Mask R-CNN ROI classifier), not the semantic head.

Consequence for SAM3 supervision
---------------------------------
- For SAM3 thing-class masks (bicycle, motorcycle, car, person, ...):
  Target channel 0 (the unified thing channel).  This is the ONLY correct
  semantic-head target for thing classes.  The previous ``class_specific`` mode
  that mapped bicycle→26, person→17, etc. was wrong: those indices hit arbitrary
  stuff pseudo-class channels, not the Cityscapes classes.

- For SAM3 stuff-class masks (pole, traffic sign, traffic light, guard rail):
  Two sub-modes:
    ``entropy``:  Minimize prediction entropy inside the mask — channel-agnostic,
                  but still forces the model to commit to some stuff channel.
                  Safe to use without knowing the Hungarian assignment.
    ``class_specific`` (future):  Cross-entropy toward the specific stuff
                  pseudo-class channel that the Hungarian assignment maps to
                  the given Cityscapes class.  Requires ``stuff_channel_map``
                  extracted from the Stage-3 checkpoint (see
                  scripts/extract_cups_class_mapping.py).

Modes
-----
``"thing_entropy_stuff_entropy"`` (default, safe):
    Thing masks → thing_coverage (target channel 0 = unified thing channel).
    Stuff masks → entropy.

``"thing_focal_stuff_entropy"`` (recommended):
    Thing masks → IoU-weighted focal CE toward channel 0 (unified thing channel).
    Stuff masks → entropy.
    Adds focal modulation to amplify gradient on rare/dead thing classes and
    IoU soft-weighting so high-confidence SAM3 masks contribute more.

``"entropy"`` (legacy):
    All masks → entropy.  Fully class-agnostic.

``"thing_coverage"`` (legacy):
    All masks → thing_coverage.  Only valid when all masks are thing classes.

``"class_specific"`` (deprecated / wrong for k=80 CUPS heads):
    Kept for backward compatibility but will raise a warning if the semantic
    head channel count ≠ 27.  Use ``thing_focal_stuff_entropy`` instead.

``"thing_mc_panda"`` (Exp 1 — MC-PanDA ECCV 2024 + S2C CVPR 2024):
    Like ``thing_focal_only`` but gates each mask's loss by
    ``(1 - teacher_P(thing_channel))``.  Well-predicted thing regions
    (car, person) contribute near-zero gradient; uncertain/rare regions
    (bicycle, motorcycle) get full gradient.  Requires ``teacher_logits``
    in ``forward()``.

``"thing_focal_per_class_freq"`` (Exp 2 — EFL CVPR 2022):
    Like ``thing_focal_only`` but scales gamma per SAM3 class by
    ``1 + (1 - sigmoid((n_c - mu) / sigma))``; rare classes get higher
    effective gamma.  Requires ``class_frequencies`` in constructor.

``"thing_focal_stuff_kd"`` (Exp 3 — Incrementer CVPR 2023):
    Thing masks → focal CE (same as ``thing_focal_only``).
    Non-SAM pixels → KL(teacher||student) weighted by teacher stuff
    confidence — an anti-forgetting anchor for stuff predictions.
    Requires ``teacher_logits`` in ``forward()``.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ── SAM3 class metadata ───────────────────────────────────────────────────────
#
# Whether each SAM3 class is a "thing" (instance) class or a "stuff" (semantic)
# class in the Cityscapes taxonomy.
#
# Thing classes → target the unified "thing region" channel (last channel of the
#   semantic head).  This is the only valid semantic-head supervision for things.
#
# Stuff classes → entropy mode until Hungarian mapping is available.
#
# SAM3 FINE_CLASS_NAMES order:
#   0=person  1=bicycle  2=motorcycle  3=rider
#   4=traffic sign  5=traffic light
#   6=truck  7=bus  8=train
#   9=guard rail  10=caravan  11=trailer  12=car  13=pole
#
_SAM3_IS_THING: List[bool] = [
    True,   # 0: person         (Cityscapes thing)
    True,   # 1: bicycle        (Cityscapes thing)
    True,   # 2: motorcycle     (Cityscapes thing)
    True,   # 3: rider          (Cityscapes thing)
    False,  # 4: traffic sign   (Cityscapes stuff)
    False,  # 5: traffic light  (Cityscapes stuff)
    True,   # 6: truck          (Cityscapes thing)
    True,   # 7: bus            (Cityscapes thing)
    True,   # 8: train          (Cityscapes thing)
    False,  # 9: guard rail     (Cityscapes void/stuff — not a standard instance)
    True,   # 10: caravan       (Cityscapes void/thing — treat as thing)
    True,   # 11: trailer       (Cityscapes void/thing — treat as thing)
    True,   # 12: car           (Cityscapes thing)
    False,  # 13: pole          (Cityscapes stuff)
]

# Legacy mapping — only valid for a 27-channel semantic head (NOT the standard
# k=80 CUPS head which has ~66 channels).  Kept for reference and backward
# compatibility only.  DO NOT use when num_channels != 27.
_SAM3_TO_CUPS27_LEGACY: List[int] = [
    17,  # 0: person
    26,  # 1: bicycle
    25,  # 2: motorcycle
    18,  # 3: rider
    13,  # 4: traffic sign
    12,  # 5: traffic light
    20,  # 6: truck
    21,  # 7: bus
    24,  # 8: train
     7,  # 9: guard rail
    22,  # 10: caravan
    23,  # 11: trailer
    19,  # 12: car
    10,  # 13: pole
]

_VALID_MODES = (
    "entropy",
    "thing_coverage",
    "thing_entropy_stuff_entropy",
    "thing_focal_stuff_entropy",
    "thing_focal_only",
    "class_specific",             # deprecated for k=80 heads
    "thing_mc_panda",             # Exp 1: MC-PanDA teacher gating
    "thing_focal_per_class_freq", # Exp 2: Equalized focal loss
    "thing_focal_stuff_kd",       # Exp 3: Incrementer-style KD anchor
)

# Modes that skip stuff masks entirely (stuff either left untouched or handled
# by a separate mechanism like KD on non-SAM pixels).
_SKIP_STUFF_MODES = frozenset({
    "thing_focal_only",
    "thing_mc_panda",
    "thing_focal_per_class_freq",
    "thing_focal_stuff_kd",
})


class FineObjectSemanticLoss(nn.Module):
    """Semantic-head loss on SAM fine-grained mask regions.

    Args:
        mode: One of the modes described in the module docstring.
              Recommended: ``"thing_focal_stuff_entropy"``.
        thing_class_idx: Semantic class index for the unified thing-region class.
            Pass ``-1`` for auto-detect (resolves to channel 0, which is the thing
            channel in all CUPS k=80 heads — things map to 0, stuff to 1..S).
        stuff_channel_map: Optional dict mapping SAM3 class index → semantic head
            channel index for stuff classes.  Only used when mode contains
            ``class_specific`` stuff supervision.  Extract from Stage-3
            checkpoint via ``scripts/extract_cups_class_mapping.py``.
        focal_gamma: Focal loss exponent for thing-class CE.  Default 2.0.
            Use 3.0 for very rare classes (bicycle, motorcycle, train).
        use_iou_weighting: If True, weight each mask's loss by its SAM3 IoU
            score (soft-weighting instead of hard threshold).
        min_hard_iou: Hard-discard threshold.  Masks with IoU below this are
            dropped entirely regardless of soft-weighting.
    """

    def __init__(
        self,
        mode: str = "thing_focal_stuff_entropy",
        thing_class_idx: int = -1,
        stuff_channel_map: Optional[Dict[int, int]] = None,
        focal_gamma: float = 2.0,
        use_iou_weighting: bool = True,
        min_hard_iou: float = 0.10,
        # Exp 1 (thing_mc_panda): teacher-gated focal CE
        common_thing_channel_indices: Optional[List[int]] = None,
        teacher_logit_weight: float = 1.0,
        # Exp 2 (thing_focal_per_class_freq): equalized focal gamma
        class_frequencies: Optional[List[float]] = None,
        gamma_scale_factor: float = 1.0,
        # Exp 3 (thing_focal_stuff_kd): KD distillation on non-SAM pixels
        stuff_kd_lambda: float = 0.1,
        stuff_channel_start: int = 1,
        # Legacy args — kept for backward compat
        sam3_to_cups: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown mode {mode!r}. Use one of {_VALID_MODES}.")
        self.mode = mode
        self.thing_class_idx = thing_class_idx
        self.stuff_channel_map = stuff_channel_map or {}
        self.focal_gamma = focal_gamma
        self.use_iou_weighting = use_iou_weighting
        self.min_hard_iou = min_hard_iou
        # Exp 1
        self.common_thing_channel_indices: List[int] = common_thing_channel_indices or []
        self.teacher_logit_weight = teacher_logit_weight
        # Exp 2
        self.class_frequencies: Optional[List[float]] = class_frequencies
        self.gamma_scale_factor = gamma_scale_factor
        # Exp 3
        self.stuff_kd_lambda = stuff_kd_lambda
        self.stuff_channel_start = stuff_channel_start
        # Legacy
        self._sam3_to_cups_legacy: List[int] = (
            sam3_to_cups if sam3_to_cups is not None else _SAM3_TO_CUPS27_LEGACY
        )

    def forward(
        self,
        logits: Tensor,
        sam_masks_list: List[Optional[Tensor]],
        sam_iou_list: List[Optional[Tensor]],
        sam_class_labels_list: Optional[List[Optional[Tensor]]] = None,
        min_iou_score: float = 0.10,
        teacher_logits: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the fine-object semantic loss for one batch.

        Args:
            logits: ``(B, C, H, W)`` semantic logits.
            sam_masks_list: Per-image ``(N_i, H, W)`` bool tensors or ``None``.
            sam_iou_list: Per-image ``(N_i,)`` float IoU confidence or ``None``.
            sam_class_labels_list: Per-image ``(N_i,)`` int32 SAM3 class indices
                (0..13).  Required for non-entropy modes; ignored otherwise.
            min_iou_score: Hard-discard threshold (overrides ``min_hard_iou``
                init arg when called).  Masks with IoU below this are skipped.

        Returns:
            Scalar loss tensor (zero-gradient guard when no valid masks).
        """
        C = logits.shape[1]
        thing_idx = self.thing_class_idx
        if thing_idx == -1:
            # Channel 0 = unified thing class in all CUPS k=80 heads.
            # (things_classes → target 0, stuff_classes → target 1..S in the dataloader)
            thing_idx = 0

        # Warn if legacy class_specific mode is used with a non-27-channel head
        if self.mode == "class_specific" and C != 27:
            warnings.warn(
                f"FineObjectSemanticLoss mode='class_specific' expects 27 semantic "
                f"head channels but got {C}.  This targets WRONG channels for k=80 "
                f"CUPS heads.  Use mode='thing_focal_stuff_entropy' instead.",
                UserWarning,
                stacklevel=2,
            )

        per_image_losses: List[Tensor] = []
        hard_iou = max(min_iou_score, self.min_hard_iou)

        for b, (masks, ious) in enumerate(zip(sam_masks_list, sam_iou_list)):
            if masks is None or masks.shape[0] == 0:
                continue

            logits_b = logits[b]   # (C, H, W)
            h, w = logits_b.shape[1:]

            # Resize SAM masks to logit resolution
            if masks.shape[1:] != (h, w):
                masks = F.interpolate(
                    masks.float().unsqueeze(1),
                    size=(h, w),
                    mode="nearest",
                ).squeeze(1).bool()

            # Hard-discard low-quality masks; build soft IoU weights
            if ious is not None and ious.numel() > 0:
                keep = ious >= hard_iou
                iou_weights = ious[keep].clamp(0.0, 1.0) if self.use_iou_weighting else None
            else:
                keep = torch.ones(masks.shape[0], dtype=torch.bool, device=masks.device)
                iou_weights = None

            masks = masks[keep]
            if masks.shape[0] == 0:
                continue

            # Pull and filter class labels in sync with the keep mask
            cls_labels: Optional[Tensor] = None
            if sam_class_labels_list is not None:
                raw_labels = sam_class_labels_list[b]
                if raw_labels is not None and raw_labels.numel() > 0:
                    cls_labels = raw_labels[keep]

            # Mean logit vector per mask — (M, C)
            mean_logits = _batch_masked_mean(logits_b, masks)
            if mean_logits.shape[0] == 0:
                continue

            # Prepare per-image teacher logits for teacher-gated modes
            teacher_b: Optional[Tensor] = None
            mean_teacher_logits: Optional[Tensor] = None
            if teacher_logits is not None and self.mode in (
                "thing_mc_panda", "thing_focal_stuff_kd"
            ):
                teacher_b = teacher_logits[b]
                if teacher_b.shape[1:] != (h, w):
                    teacher_b = F.interpolate(
                        teacher_b.unsqueeze(0).float(),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                mean_teacher_logits = _batch_masked_mean(teacher_b, masks)

            loss = self._compute_loss(
                mean_logits, cls_labels, thing_idx, iou_weights, logits.device,
                mean_teacher_logits=mean_teacher_logits,
            )
            if loss is not None:
                per_image_losses.append(loss)

            # Exp 3: KD distillation on non-SAM pixels (anti-forgetting anchor)
            if self.mode == "thing_focal_stuff_kd" and teacher_b is not None:
                union_mask = masks.any(dim=0)  # (H, W)
                kd = self._stuff_kd_loss(logits_b, teacher_b, union_mask, logits.device)
                per_image_losses.append(kd * self.stuff_kd_lambda)

        if not per_image_losses:
            return logits.sum() * 0.0

        return torch.stack(per_image_losses).mean()

    # ── Internal dispatch ─────────────────────────────────────────────────────

    def _compute_loss(
        self,
        mean_logits: Tensor,      # (M, C)
        cls_labels: Optional[Tensor],   # (M,) SAM3 class indices or None
        thing_idx: int,
        iou_weights: Optional[Tensor],  # (M,) or None
        device: torch.device,
        mean_teacher_logits: Optional[Tensor] = None,  # (M, C) for mc_panda / stuff_kd
    ) -> Optional[Tensor]:
        if self.mode in ("entropy", "thing_coverage"):
            return self._legacy_loss(mean_logits, thing_idx, iou_weights)

        if self.mode == "class_specific":
            return self._legacy_class_specific(mean_logits, cls_labels, iou_weights, device)

        # Split thing vs stuff per mask (all remaining modes)
        if self.mode in (
            "thing_entropy_stuff_entropy", "thing_focal_stuff_entropy", "thing_focal_only",
            "thing_mc_panda", "thing_focal_per_class_freq", "thing_focal_stuff_kd",
        ):
            return self._split_thing_stuff_loss(
                mean_logits, cls_labels, thing_idx, iou_weights, device,
                mean_teacher_logits=mean_teacher_logits,
            )

        return None

    def _split_thing_stuff_loss(
        self,
        mean_logits: Tensor,
        cls_labels: Optional[Tensor],
        thing_idx: int,
        iou_weights: Optional[Tensor],
        device: torch.device,
        mean_teacher_logits: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        M = mean_logits.shape[0]
        parts: List[Tensor] = []

        if cls_labels is None or cls_labels.numel() == 0:
            return _entropy_loss(mean_logits, iou_weights)

        sam_idx = cls_labels.to(device=device, dtype=torch.long)
        valid = sam_idx >= 0

        is_thing_mask = torch.zeros(M, dtype=torch.bool, device=device)
        for m in range(M):
            if valid[m] and sam_idx[m] < len(_SAM3_IS_THING):
                is_thing_mask[m] = _SAM3_IS_THING[sam_idx[m]]

        # ── Thing masks ────────────────────────────────────────────────────
        thing_sel = is_thing_mask & valid
        if thing_sel.any():
            thing_logits = mean_logits[thing_sel]
            thing_w = iou_weights[thing_sel] if iou_weights is not None else None
            targets = torch.full(
                (thing_sel.sum(),), thing_idx, dtype=torch.long, device=device,
            )

            if self.mode == "thing_mc_panda":
                teacher_thing = (
                    mean_teacher_logits[thing_sel]
                    if mean_teacher_logits is not None else None
                )
                if teacher_thing is not None:
                    loss_t = self._mc_panda_thing_loss(
                        thing_logits, teacher_thing, thing_idx, thing_w, device
                    )
                else:
                    loss_t = _focal_ce(thing_logits, targets, self.focal_gamma, thing_w)
            elif self.mode == "thing_focal_per_class_freq":
                loss_t = self._freq_aware_focal_loss(
                    thing_logits, sam_idx[thing_sel], thing_idx, thing_w, device
                )
            elif self.mode in (
                "thing_focal_stuff_entropy", "thing_focal_only", "thing_focal_stuff_kd"
            ):
                loss_t = _focal_ce(thing_logits, targets, self.focal_gamma, thing_w)
            else:  # thing_entropy_stuff_entropy
                loss_t = _weighted_ce(thing_logits, targets, thing_w)
            parts.append(loss_t)

        # ── Stuff masks ────────────────────────────────────────────────────
        # Modes in _SKIP_STUFF_MODES leave stuff predictions entirely untouched
        # (or handle them via a separate KD mechanism on non-SAM pixels).
        stuff_sel = (~is_thing_mask) & valid
        if stuff_sel.any() and self.mode not in _SKIP_STUFF_MODES:
            stuff_logits = mean_logits[stuff_sel]
            stuff_w = iou_weights[stuff_sel] if iou_weights is not None else None

            if self.stuff_channel_map:
                parts.append(
                    self._stuff_class_specific(
                        stuff_logits, sam_idx[stuff_sel], stuff_w, device
                    )
                )
            else:
                parts.append(_entropy_loss(stuff_logits, stuff_w))

        if not parts:
            return None
        return torch.stack(parts).mean()

    def _stuff_class_specific(
        self,
        stuff_logits: Tensor,   # (M_stuff, C)
        sam_idx: Tensor,        # (M_stuff,) SAM3 class indices for stuff masks
        iou_weights: Optional[Tensor],
        device: torch.device,
    ) -> Tensor:
        """CE toward correct stuff pseudo-class channel (needs stuff_channel_map)."""
        targets = torch.full(
            (stuff_logits.shape[0],), 255, dtype=torch.long, device=device
        )
        for m in range(stuff_logits.shape[0]):
            cls = int(sam_idx[m].item())
            if cls in self.stuff_channel_map:
                targets[m] = self.stuff_channel_map[cls]

        valid = targets != 255
        if not valid.any():
            return _entropy_loss(stuff_logits, iou_weights)

        return _weighted_ce(stuff_logits[valid], targets[valid],
                            iou_weights[valid] if iou_weights is not None else None)

    def _legacy_loss(
        self,
        mean_logits: Tensor,
        thing_idx: int,
        iou_weights: Optional[Tensor],
    ) -> Tensor:
        if self.mode == "entropy":
            return _entropy_loss(mean_logits, iou_weights)
        # thing_coverage
        targets = torch.full(
            (mean_logits.shape[0],), thing_idx,
            dtype=torch.long, device=mean_logits.device,
        )
        return _weighted_ce(mean_logits, targets, iou_weights)

    def _legacy_class_specific(
        self,
        mean_logits: Tensor,
        cls_labels: Optional[Tensor],
        iou_weights: Optional[Tensor],
        device: torch.device,
    ) -> Tensor:
        if cls_labels is None or cls_labels.numel() == 0:
            return _entropy_loss(mean_logits, iou_weights)
        cups_map = torch.tensor(self._sam3_to_cups_legacy, dtype=torch.long, device=device)
        sam_idx = cls_labels.to(device=device, dtype=torch.long)
        valid = sam_idx >= 0
        sam_idx_clamped = sam_idx.clamp(min=0)
        cups_targets = cups_map[sam_idx_clamped]
        if not valid.any():
            return _entropy_loss(mean_logits, iou_weights)
        return _weighted_ce(
            mean_logits[valid], cups_targets[valid],
            iou_weights[valid] if iou_weights is not None else None,
        )

    # ── New literature-grounded methods (Exp 1-3) ─────────────────────────────

    def _mc_panda_thing_loss(
        self,
        thing_logits: Tensor,            # (M_thing, C) student mean logits
        teacher_thing_logits: Tensor,    # (M_thing, C) teacher mean logits
        thing_idx: int,
        iou_weights: Optional[Tensor],   # (M_thing,) or None
        device: torch.device,
    ) -> Tensor:
        """MC-PanDA + S2C gated focal CE (ECCV 2024 / CVPR 2024).

        Masks where teacher already confidently predicts the thing channel
        contribute near-zero gradient.  Rare/uncertain masks get full gradient.
        """
        teacher_probs = teacher_thing_logits.softmax(dim=-1)  # (M, C)

        if self.common_thing_channel_indices:
            common_idx = torch.tensor(
                self.common_thing_channel_indices, dtype=torch.long, device=device
            )
            common_conf = teacher_probs[:, common_idx].max(dim=-1).values  # (M,)
        else:
            # Default: gate by teacher's confidence in the unified thing channel
            common_conf = teacher_probs[:, thing_idx]

        mc_gate = (1.0 - common_conf * self.teacher_logit_weight).clamp(0.0, 1.0)
        combined_w = mc_gate if iou_weights is None else (mc_gate * iou_weights)

        targets = torch.full(
            (thing_logits.shape[0],), thing_idx, dtype=torch.long, device=device,
        )
        return _focal_ce(thing_logits, targets, self.focal_gamma, combined_w)

    def _freq_aware_focal_loss(
        self,
        thing_logits: Tensor,      # (M_thing, C)
        sam_thing_labels: Tensor,  # (M_thing,) SAM3 class indices (thing masks only)
        thing_idx: int,
        iou_weights: Optional[Tensor],
        device: torch.device,
    ) -> Tensor:
        """Equalized focal loss with per-class frequency-based gamma (CVPR 2022).

        Rare SAM3 classes (bicycle, motorcycle) get higher effective gamma;
        common classes (person, car) get lower gamma.
        """
        targets = torch.full(
            (thing_logits.shape[0],), thing_idx, dtype=torch.long, device=device,
        )
        if self.class_frequencies is None or len(self.class_frequencies) == 0:
            return _focal_ce(thing_logits, targets, self.focal_gamma, iou_weights)

        freq = torch.tensor(self.class_frequencies, dtype=torch.float, device=device)
        mu = freq.mean()
        sigma = freq.std().clamp(min=1e-6)

        # Per-mask effective gamma: higher for rare, lower for common classes
        gammas = thing_logits.new_full((thing_logits.shape[0],), self.focal_gamma)
        for m in range(thing_logits.shape[0]):
            cls = int(sam_thing_labels[m].item())
            if 0 <= cls < len(self.class_frequencies):
                e_c = 1.0 - torch.sigmoid((freq[cls] - mu) / sigma)
                gammas[m] = self.focal_gamma * (1.0 + float(e_c) * self.gamma_scale_factor)

        log_p = F.log_softmax(thing_logits, dim=1)                         # (M, C)
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)          # (M,)
        pt = log_pt.exp()
        nll = -(1.0 - pt).pow(gammas) * log_pt                             # (M,)

        if iou_weights is not None:
            return (iou_weights * nll).sum() / iou_weights.sum().clamp(min=1e-6)
        return nll.mean()

    def _stuff_kd_loss(
        self,
        logits_b: Tensor,           # (C, H, W) student
        teacher_logits_b: Tensor,   # (C, H, W) teacher
        sam_union_mask: Tensor,     # (H, W) bool — True inside any SAM mask
        device: torch.device,
    ) -> Tensor:
        """Incrementer-style KL distillation on non-SAM pixels (CVPR 2023).

        Anchors stuff predictions to the teacher outside SAM mask regions,
        weighted by teacher confidence on stuff channels.
        """
        non_sam = ~sam_union_mask  # (H, W)
        if not non_sam.any():
            return logits_b.sum() * 0.0

        C = logits_b.shape[0]
        s_flat = logits_b.permute(1, 2, 0).reshape(-1, C)           # (HW, C)
        t_flat = teacher_logits_b.permute(1, 2, 0).reshape(-1, C)   # (HW, C)
        ns_flat = non_sam.reshape(-1)                                 # (HW,)

        s_ns = s_flat[ns_flat]   # (N_ns, C) student non-SAM logits
        t_ns = t_flat[ns_flat]   # (N_ns, C) teacher non-SAM logits

        t_probs = t_ns.softmax(dim=-1)              # (N_ns, C)
        s_log_probs = F.log_softmax(s_ns, dim=-1)   # (N_ns, C)

        # KL(t || s) per pixel
        kl = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=-1)  # (N_ns,)

        # Weight by teacher's confidence on stuff channels
        if self.stuff_channel_start < C:
            sim_weight = t_probs[:, self.stuff_channel_start:].max(dim=-1).values
            return (kl * sim_weight).mean()
        return kl.mean()


# ── Loss primitives ───────────────────────────────────────────────────────────

def _entropy_loss(logits: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    """Mean prediction entropy (minimized)."""
    probs = logits.softmax(dim=1)
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)   # (M,)
    if weights is not None:
        return (weights * entropy).sum() / weights.sum().clamp(min=1e-6)
    return entropy.mean()


def _weighted_ce(
    logits: Tensor,
    targets: Tensor,
    weights: Optional[Tensor] = None,
    ignore_index: int = 255,
) -> Tensor:
    """Cross-entropy with optional per-sample IoU weighting."""
    if weights is None:
        return F.cross_entropy(logits, targets, ignore_index=ignore_index)
    # Manual weighted CE
    log_p = F.log_softmax(logits, dim=1)   # (M, C)
    valid = targets != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    nll = -log_p[valid].gather(1, targets[valid].unsqueeze(1)).squeeze(1)  # (M',)
    w = weights[valid]
    return (w * nll).sum() / w.sum().clamp(min=1e-6)


def _focal_ce(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    weights: Optional[Tensor] = None,
    ignore_index: int = 255,
) -> Tensor:
    """IoU-weighted focal cross-entropy.

    Loss = -w_iou * (1 - p_t)^gamma * log(p_t)
    where p_t = softmax(logits)[target_class].
    """
    log_p = F.log_softmax(logits, dim=1)   # (M, C)
    valid = targets != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    lv = log_p[valid]
    tv = targets[valid].unsqueeze(1)
    log_pt = lv.gather(1, tv).squeeze(1)   # (M',)
    pt = log_pt.exp()
    focal_weight = (1.0 - pt).pow(gamma)
    nll = -focal_weight * log_pt
    if weights is not None:
        w = weights[valid]
        return (w * nll).sum() / w.sum().clamp(min=1e-6)
    return nll.mean()


# ── Spatial helper ────────────────────────────────────────────────────────────

def _batch_masked_mean(logits: Tensor, masks: Tensor) -> Tensor:
    """Mean logit vector inside each mask without looping.

    Args:
        logits: ``(C, H, W)``
        masks:  ``(M, H, W)`` bool

    Returns:
        ``(M', C)`` mean logit vectors where M' <= M (empty masks dropped).
    """
    M = masks.shape[0]
    if M == 0:
        return logits.new_zeros(0, logits.shape[0])

    flat_masks = masks.reshape(M, -1).float()           # (M, HW)
    flat_logits = logits.reshape(logits.shape[0], -1)   # (C, HW)

    pixel_counts = flat_masks.sum(dim=1).clamp(min=1.0)
    summed = flat_masks @ flat_logits.T                 # (M, C)
    mean_logits = summed / pixel_counts.unsqueeze(1)

    valid = flat_masks.sum(dim=1) > 0
    return mean_logits[valid]
