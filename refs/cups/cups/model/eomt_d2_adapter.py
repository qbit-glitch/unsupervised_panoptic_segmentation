"""Detectron2-interface adapter that lets the EoMT (DINOv2 ViT-B + masked
attention queries) network run inside the ORIGINAL, unmodified CUPS Stage-3
self-training pipeline (cups/pl_model_self.py + train_self.py).

The adapter speaks both protocols:

CUPS / Detectron2 side (what SelfSupervisedModel expects of ``self.model``):
  * train forward:  model(list of {"image", "sem_seg", "instances"}) -> loss dict
  * eval forward:   model(list of {"image"}) -> list of
        {"panoptic_seg": (seg_map LongTensor[H,W], segments_info), "sem_seg": (S+1,H,W)}
    with the CUPS conventions: stuff segments carry category_id = semantic
    channel index in [1, S]; thing segments carry category_id = thing index
    in [0, T); semantic channel 0 is the merged "thing" channel.
  * ``model.sem_seg_head.predictor.out_channels`` == S + 1 (sanitize guard).

EoMT side: the wrapped network is the stock EoMT module trained in Stage-2 on
the unified k=80 pseudo-class space; the adapter translates between the
unified 80-class space and CUPS's split (thing-index / stuff-channel) spaces.

The teacher wrapper ``EoMTWithTTA`` mirrors CUPS ``PanopticFPNWithTTA``:
multi-scale (config MODEL.TTA_SCALES) + horizontal-flip inference with
per-query logit averaging, exposing the wrapped model under ``.model`` (the
attribute CUPS's EMA update relies on).

Run with PYTHONPATH including BOTH refs/cups and refs/eomt (the loss is
imported from refs/eomt/training/mask_classification_loss.py).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

log = logging.getLogger(__name__)


class _Predictor:
    def __init__(self, out_channels: int) -> None:
        self.out_channels = out_channels


class _SemSegHeadShim:
    """Duck-typed stand-in for D2's sem_seg_head (only out_channels is read)."""

    def __init__(self, out_channels: int) -> None:
        self.predictor = _Predictor(out_channels)


class EoMTPanopticShim(nn.Module):
    """EoMT network behind a Detectron2 panoptic-model interface.

    Args:
        eomt: The EoMT nn.Module (models.eomt.EoMT) with Stage-2 weights.
        stuff_class_ids: Sorted unified-space (k=80) ids that are stuff;
            stuff semantic channel ``s`` (1-based) maps to
            ``stuff_class_ids[s - 1]``.
        thing_class_ids: Sorted unified-space ids that are things; thing
            index ``t`` maps to ``thing_class_ids[t]``.
        confidence_threshold: Absolute query score floor for panoptic
            inference (CUPS MODEL.TTA_INFERENCE_CONFIDENCE_THRESHOLD).
        overlap_threshold: EoMT panoptic-merge overlap pruning threshold.
        min_mask_pixels: Drop training masks smaller than this (CUPS
            augmentations drop instances below 5 px).
        loss_*: MaskClassificationLoss coefficients (EoMT defaults).
    """

    def __init__(
        self,
        eomt: nn.Module,
        stuff_class_ids: Sequence[int],
        thing_class_ids: Sequence[int],
        confidence_threshold: float = 0.5,
        overlap_threshold: float = 0.8,
        min_mask_pixels: int = 5,
        loss_num_points: int = 12544,
        loss_oversample_ratio: float = 3.0,
        loss_importance_sample_ratio: float = 0.75,
        loss_mask_coefficient: float = 5.0,
        loss_dice_coefficient: float = 5.0,
        loss_class_coefficient: float = 2.0,
        loss_no_object_coefficient: float = 0.1,
        droploss_enabled: bool = False,
        droploss_stuff_overlap_max: float = 0.5,
    ) -> None:
        super().__init__()
        # The attribute name deliberately contains "head": CUPS Stage-3's
        # head-only optimizer selects parameters by ("head" in name) and
        # ("norm" not in name). Backbone parameters are additionally frozen
        # via requires_grad by the builder, so only the EoMT decoder
        # (queries, last masked-attn blocks, class/mask heads, upscale)
        # receives updates — the EoMT analog of CUPS's frozen-backbone,
        # heads-only fine-tune.
        self.eomt_head = eomt

        self.stuff_class_ids: List[int] = [int(c) for c in stuff_class_ids]
        self.thing_class_ids: List[int] = [int(c) for c in thing_class_ids]
        self.num_classes: int = len(self.stuff_class_ids) + len(self.thing_class_ids)
        self._thing_set = set(self.thing_class_ids)
        # unified-space id -> thing index / stuff channel (1-based)
        self._thing_pos = {c: i for i, c in enumerate(self.thing_class_ids)}
        self._stuff_channel = {c: i + 1 for i, c in enumerate(self.stuff_class_ids)}

        self.confidence_threshold = float(confidence_threshold)
        self.overlap_threshold = float(overlap_threshold)
        self.min_mask_pixels = int(min_mask_pixels)

        self.sem_seg_head = _SemSegHeadShim(len(self.stuff_class_ids) + 1)

        # Imported lazily from the refs/eomt tree (PYTHONPATH).
        from training.mask_classification_loss import MaskClassificationLoss

        self.criterion = MaskClassificationLoss(
            num_points=loss_num_points,
            oversample_ratio=loss_oversample_ratio,
            importance_sample_ratio=loss_importance_sample_ratio,
            mask_coefficient=loss_mask_coefficient,
            dice_coefficient=loss_dice_coefficient,
            class_coefficient=loss_class_coefficient,
            num_labels=self.num_classes,
            no_object_coefficient=loss_no_object_coefficient,
            droploss_enabled=droploss_enabled,
            droploss_stuff_overlap_max=droploss_stuff_overlap_max,
        )

    # -- plumbing ------------------------------------------------------------

    def _set_grid(self, h: int, w: int) -> None:
        patch_embed = self.eomt_head.encoder.backbone.patch_embed
        ps = int(patch_embed.patch_size[0])
        patch_embed.grid_size = (h // ps, w // ps)

    def forward_logits(self, imgs: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """Raw EoMT forward on a [B,3,H,W] float image batch in [0,1]."""
        self._set_grid(int(imgs.shape[-2]), int(imgs.shape[-1]))
        return self.eomt_head(imgs)

    # -- D2-style forward ------------------------------------------------------

    def forward(self, batched_inputs: List[Dict[str, Any]]):
        if self.training:
            return self._forward_train(batched_inputs)
        return self._forward_eval(batched_inputs)

    # -- training: D2 pseudo-labels -> EoMT mask-classification loss ---------

    def _forward_train(self, batched_inputs: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        imgs = torch.stack([s["image"].float() for s in batched_inputs])
        targets = [self._d2_sample_to_target(s) for s in batched_inputs]

        mask_logits_per_layer, class_logits_per_layer = self.forward_logits(imgs)

        losses: Dict[str, Tensor] = {}
        num_layers = len(mask_logits_per_layer)
        for i, (mask_logits, class_logits) in enumerate(
            zip(mask_logits_per_layer, class_logits_per_layer)
        ):
            layer_losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets,
            )
            suffix = "" if i == num_layers - 1 else f"_aux{i}"
            for key, value in layer_losses.items():
                if "mask" in key:
                    value = value * self.criterion.mask_coefficient
                elif "dice" in key:
                    value = value * self.criterion.dice_coefficient
                elif "cross_entropy" in key:
                    value = value * self.criterion.class_coefficient
                losses[f"{key}{suffix}"] = value
        return losses

    def _d2_sample_to_target(self, sample: Dict[str, Any]) -> Dict[str, Tensor]:
        sem: Tensor = sample["sem_seg"]
        instances = sample["instances"]
        device = sem.device
        h, w = sem.shape[-2:]

        # Confident-stuff region for the DropLoss stuff-gate: pixels whose CUPS
        # pseudo sem-channel is a known stuff class (1..num_stuff). Unmatched
        # queries NOT predominantly on this region are freed from the no-object
        # loss (discovery); queries on stuff keep the penalty (see
        # MaskClassificationLoss). sem==0 is thing-merged, 255 is ignore — both
        # are off-stuff and thus discovery-eligible.
        num_stuff = len(self.stuff_class_ids)
        stuff_region = ((sem >= 1) & (sem <= num_stuff)).reshape(h, w)

        masks: List[Tensor] = []
        labels: List[int] = []

        gt_masks = instances.gt_masks.tensor.bool()
        gt_classes = instances.gt_classes
        for i in range(gt_masks.shape[0]):
            m = gt_masks[i]
            if int(m.sum()) < self.min_mask_pixels:
                continue
            t = int(gt_classes[i])
            if not 0 <= t < len(self.thing_class_ids):
                continue
            masks.append(m)
            labels.append(self.thing_class_ids[t])

        num_stuff = len(self.stuff_class_ids)
        for value in sem.unique().tolist():
            s = int(value)
            if not 1 <= s <= num_stuff:
                continue  # 0 = thing region (covered by instances), 255 = ignore
            m = sem == s
            if int(m.sum()) < self.min_mask_pixels:
                continue
            masks.append(m)
            labels.append(self.stuff_class_ids[s - 1])

        if not masks:
            return {
                "masks": torch.zeros((0, h, w), dtype=torch.bool, device=device),
                "labels": torch.zeros((0,), dtype=torch.long, device=device),
                "stuff_region": stuff_region,
            }
        return {
            "masks": torch.stack(masks),
            "labels": torch.tensor(labels, dtype=torch.long, device=device),
            "stuff_region": stuff_region,
        }

    # -- inference: EoMT logits -> D2 panoptic prediction --------------------

    @torch.no_grad()
    def _forward_eval(self, batched_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        imgs = torch.stack([s["image"].float() for s in batched_inputs])
        H, W = int(imgs.shape[-2]), int(imgs.shape[-1])
        mask_logits_per_layer, class_logits_per_layer = self.forward_logits(imgs)
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], size=(H, W), mode="bilinear", align_corners=False
        )
        class_logits = class_logits_per_layer[-1]
        return [
            self.logits_to_d2_prediction(mask_logits[b], class_logits[b])
            for b in range(imgs.shape[0])
        ]

    @torch.no_grad()
    def logits_to_d2_prediction(
        self, mask_logits: Tensor, class_logits: Tensor
    ) -> Dict[str, Any]:
        """Convert one image's final-layer logits into the D2 panoptic format.

        Panoptic merge follows EoMT's to_per_pixel_preds_panoptic (score floor
        + per-pixel argmax competition + overlap pruning), which is the EoMT
        analog of D2's panoptic inference used by CUPS.

        Args:
            mask_logits: [Q, H, W] (already at output resolution).
            class_logits: [Q, C + 1].
        """
        H, W = mask_logits.shape[-2:]
        device = mask_logits.device
        probs = class_logits.softmax(dim=-1)
        scores, classes = probs.max(dim=-1)

        seg_map = torch.zeros((H, W), dtype=torch.long, device=device)
        segments_info: List[Dict[str, Any]] = []

        keep = classes.ne(class_logits.shape[-1] - 1) & (scores > self.confidence_threshold)
        masks = mask_logits.sigmoid()

        if bool(keep.any()):
            kept_masks = masks[keep]
            kept_scores = scores[keep]
            kept_classes = classes[keep]
            mask_ids = (kept_scores[:, None, None] * kept_masks).argmax(0)

            segment_id = 1
            stuff_segment_ids: Dict[int, int] = {}
            for k in range(kept_masks.shape[0]):
                cls = int(kept_classes[k])
                orig_mask = kept_masks[k] >= 0.5
                new_mask = mask_ids == k
                final_mask = orig_mask & new_mask

                orig_area = int(orig_mask.sum())
                new_area = int(new_mask.sum())
                final_area = int(final_mask.sum())
                if (
                    orig_area == 0
                    or new_area == 0
                    or final_area == 0
                    or new_area / orig_area < self.overlap_threshold
                ):
                    continue

                seg_score = float(kept_scores[k])
                if cls in self._thing_set:
                    seg_map[final_mask] = segment_id
                    segments_info.append(
                        {
                            "id": segment_id,
                            "isthing": True,
                            "category_id": self._thing_pos[cls],
                            "score": seg_score,
                        }
                    )
                    segment_id += 1
                else:
                    if cls in stuff_segment_ids:
                        seg_map[final_mask] = stuff_segment_ids[cls]
                        continue
                    stuff_segment_ids[cls] = segment_id
                    seg_map[final_mask] = segment_id
                    segments_info.append(
                        {
                            "id": segment_id,
                            "isthing": False,
                            "category_id": self._stuff_channel[cls],
                            "score": seg_score,
                        }
                    )
                    segment_id += 1

        # Semantic scores in the CUPS (S+1)-channel convention: channel 0 is
        # the merged thing channel, channels 1..S the stuff classes. Built as
        # the standard query-composition semantic probabilities.
        sem_unified = torch.einsum("qhw, qc -> chw", masks, probs[:, :-1])
        sem = torch.zeros(
            (len(self.stuff_class_ids) + 1, H, W), dtype=sem_unified.dtype, device=device
        )
        thing_ids = torch.tensor(self.thing_class_ids, device=device)
        stuff_ids = torch.tensor(self.stuff_class_ids, device=device)
        sem[0] = sem_unified[thing_ids].sum(dim=0)
        sem[1:] = sem_unified[stuff_ids]

        return {"panoptic_seg": (seg_map, segments_info), "sem_seg": sem}


class EoMTWithTTA(nn.Module):
    """CUPS PanopticFPNWithTTA analog for EoMT: multi-scale + flip averaging.

    Exposes the wrapped shim under ``.model`` (CUPS's EMA update iterates
    ``teacher_model.model.parameters()``). Inference-only.
    """

    def __init__(
        self,
        model: EoMTPanopticShim,
        tta_scales: Sequence[float] = (0.5, 0.75, 1.0),
        tta_flip: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.tta_scales = tuple(float(s) for s in tta_scales)
        self.tta_flip = bool(tta_flip)

    @torch.no_grad()
    def forward(self, batched_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        was_training = self.model.training
        self.model.eval()
        try:
            imgs = torch.stack([s["image"].float() for s in batched_inputs])
            H, W = int(imgs.shape[-2]), int(imgs.shape[-1])
            flips = (False, True) if self.tta_flip else (False,)

            mask_sum: Tensor | None = None
            class_sum: Tensor | None = None
            n_views = 0
            for scale in self.tta_scales:
                h, w = int(round(H * scale)), int(round(W * scale))
                x = F.interpolate(imgs, size=(h, w), mode="bilinear")
                for flip in flips:
                    xi = x.flip(-1) if flip else x
                    mask_logits_per_layer, class_logits_per_layer = self.model.forward_logits(xi)
                    m = mask_logits_per_layer[-1]
                    c = class_logits_per_layer[-1]
                    if flip:
                        m = m.flip(-1)
                    m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
                    mask_sum = m if mask_sum is None else mask_sum + m
                    class_sum = c if class_sum is None else class_sum + c
                    n_views += 1

            mask_logits = mask_sum / n_views
            class_logits = class_sum / n_views
            return [
                self.model.logits_to_d2_prediction(mask_logits[b], class_logits[b])
                for b in range(imgs.shape[0])
            ]
        finally:
            self.model.train(was_training)
