"""Mask2FormerPanoptic: end-to-end meta-arch for Stage-2 M2F.

Training path:
    image -> ViTAdapter -> pixel_decoder -> transformer_decoder -> losses

Eval path (panoptic merge):
    take top-scoring queries, softmax over class, threshold masks,
    greedy assignment onto a (H, W) panoptic map.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import META_ARCH_REGISTRY

__all__ = ["Mask2FormerPanoptic"]

# Label values treated as "ignore" / void when building per-sample targets.
# -1 is the internal unlabeled marker; 255 is the detectron2/Cityscapes convention.
_IGNORE_LABELS: frozenset = frozenset({-1, 255})


@META_ARCH_REGISTRY.register()
class Mask2FormerPanoptic(nn.Module):
    """End-to-end meta-arch. Registered so detectron2 tooling can enumerate it."""

    def __init__(
        self,
        backbone: nn.Module,
        pixel_decoder: nn.Module,
        transformer_decoder: nn.Module,
        criterion: nn.Module,
        num_stuff_classes: int,
        num_thing_classes: int,
        object_mask_threshold: float = 0.4,
        overlap_threshold: float = 0.8,
        aux_loss_hooks: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.transformer_decoder = transformer_decoder
        self.criterion = criterion
        self.num_stuff_classes = num_stuff_classes
        self.num_thing_classes = num_thing_classes
        self.num_classes = num_stuff_classes + num_thing_classes
        self.object_mask_threshold = object_mask_threshold
        self.overlap_threshold = overlap_threshold
        self.aux_loss_hooks = aux_loss_hooks or {}
        # DINOv3 backbone (refs/dinov3/README.md) expects ImageNet-normalized input.
        # The backbone wrapper does NOT normalize internally, so we do it here.
        self.register_buffer(
            "_pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _stack_images(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        """Pad to max H/W in batch and apply ImageNet normalization.

        uint8 inputs are assumed to span [0, 255] and are rescaled to [0, 1]
        before normalization. Float inputs are assumed to already be in [0, 1].
        Padding pixels stay at zero, which is safe because zero is further
        from the normalized distribution centre than any valid pixel — the
        backbone treats it as out-of-distribution per existing behaviour.
        """
        imgs = [s["image"].to(self.device) for s in batch]
        H = max(t.shape[-2] for t in imgs)
        W = max(t.shape[-1] for t in imgs)
        padded = torch.zeros(len(imgs), 3, H, W, device=self.device)
        for i, t in enumerate(imgs):
            x = t.float() / 255.0 if t.dtype == torch.uint8 else t.float()
            # Normalize only the valid region; padding stays at zero.
            h, w = t.shape[-2], t.shape[-1]
            padded[i, :, :h, :w] = (x - self._pixel_mean[0]) / self._pixel_std[0]
        return padded

    def _maybe_depth(self, batch: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
        if all("depth" in s for s in batch):
            return torch.stack([s["depth"].to(self.device) for s in batch], dim=0)
        return None

    def _collect_targets(self, batch: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        """Build SetCriterion-compatible targets in the combined stuff+thing ID space.

        Dataset convention (pseudo_label_dataset.py:430-491):
          - sem_seg value 0      -> thing-region marker (when IGNORE_UNKNOWN_THING_REGIONS=False)
          - sem_seg value [1..S] -> stuff class, 1-indexed
          - sem_seg value 255    -> void / ignore
          - instances.gt_classes -> thing class in [0..T), 0-indexed

        Combined space expected by SetCriterion(num_classes=S+T):
          - [0, S)     -> stuff classes         (sem_seg value c -> label c - 1)
          - [S, S+T)   -> thing classes         (gt_classes c    -> label c + S)
          - S+T        -> "no-object" phi       (handled internally, absent from targets)

        sem_seg==0 is always skipped: thing regions are sourced from `instances`,
        and emitting a union mask as target label 0 would both collide with stuff
        class 0 and double-count things.
        """
        targets: List[Dict[str, torch.Tensor]] = []
        for s in batch:
            if "_m2f_targets" in s:
                targets.append({
                    "labels": s["_m2f_targets"]["labels"].to(self.device),
                    "masks": s["_m2f_targets"]["masks"].to(self.device),
                })
                continue
            labels: List[int] = []
            masks: List[torch.Tensor] = []
            if "instances" in s:
                inst = s["instances"].to(self.device)
                # Thing gt_classes are local [0, T); shift into combined space [S, S+T).
                labels.extend(int(c) + self.num_stuff_classes for c in inst.gt_classes.tolist())
                masks.extend([m.bool() for m in inst.gt_masks.tensor])
            if "sem_seg" in s:
                sem = s["sem_seg"].to(self.device)
                for c in sem.unique():
                    ci = int(c)
                    # Skip void (255, -1) and thing-region marker (0).
                    if ci in _IGNORE_LABELS or ci == 0:
                        continue
                    # Stuff sem_seg is 1-indexed [1, S]; decrement into [0, S).
                    labels.append(ci - 1)
                    masks.append((sem == c))
            if not labels:
                H, W = s["image"].shape[-2:]
                targets.append({"labels": torch.zeros(0, dtype=torch.long, device=self.device),
                                "masks": torch.zeros(0, H, W, dtype=torch.bool, device=self.device)})
            else:
                targets.append({"labels": torch.as_tensor(labels, dtype=torch.long, device=self.device),
                                "masks": torch.stack(masks, dim=0).bool()})
        return targets

    def _panoptic_merge(self, logits: torch.Tensor, masks: torch.Tensor, H: int, W: int) -> Dict[str, torch.Tensor]:
        """Greedy panoptic assembly (one sample)."""
        scores, labels = F.softmax(logits, dim=-1).max(-1)
        mask_probs = masks.sigmoid()
        keep = (labels < self.num_classes) & (scores > self.object_mask_threshold)
        cur_masks = mask_probs[keep]
        cur_scores = scores[keep]
        cur_labels = labels[keep]
        panoptic_seg = torch.zeros((H, W), dtype=torch.int32, device=logits.device)
        sem_seg = torch.zeros((self.num_classes, H, W), dtype=torch.float32, device=logits.device)
        if cur_masks.numel() == 0:
            return {"sem_seg": sem_seg, "panoptic_seg": (panoptic_seg, [])}
        cur_masks = F.interpolate(cur_masks.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
        cur_mask_ids = cur_prob_masks.argmax(0)
        current_id = 0
        # Downstream consumers (prediction_to_label_format at model.py:343,
        # prediction_to_standard_format at model.py:192) expect the Cascade-era
        # category_id space:
        #   stuff: 1-indexed [1, S]    accessed as stuff_classes[cid - 1]
        #   thing: 0-indexed [0, T)    accessed as thing_classes[cid]
        # Our internal criterion uses the combined space [0, S+T); we remap
        # ONLY the segments_info dict so external code sees the expected form.
        # "score" is the softmax class confidence of the winning query.
        segments_info: List[Dict[str, Any]] = []
        for k in range(cur_masks.shape[0]):
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            if mask_area > 0 and original_area > 0 and mask_area / original_area > self.overlap_threshold:
                current_id += 1
                panoptic_seg[cur_mask_ids == k] = current_id
                cls = int(cur_labels[k])
                is_thing = cls >= self.num_stuff_classes
                external_cid = (cls - self.num_stuff_classes) if is_thing else (cls + 1)
                segments_info.append({
                    "id": current_id,
                    "category_id": int(external_cid),
                    "isthing": is_thing,
                    "score": float(cur_scores[k]),
                })
                sem_seg[cls] = torch.maximum(sem_seg[cls], cur_masks[k])
        return {"sem_seg": sem_seg, "panoptic_seg": (panoptic_seg, segments_info)}

    def forward(
        self, batch: List[Dict[str, Any]]
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        images = self._stack_images(batch)
        depth = self._maybe_depth(batch)
        feats = self.backbone(images)
        mask_feat, multi_scale = self.pixel_decoder(feats)
        return_emb = bool(self.aux_loss_hooks.get("return_query_embeds", False))
        dec_out = self.transformer_decoder(mask_feat=mask_feat, multi_scale=multi_scale, depth=depth, return_query_embeds=return_emb)
        if self.training:
            targets = self._collect_targets(batch)
            loss_dict = self.criterion(dec_out, targets)
            ctx = {
                "images": images,
                "depth": depth,
                "teacher_query_embeds": getattr(self, "_teacher_query_embeds", None),
            }
            for loss_key, hook in self.aux_loss_hooks.items():
                if loss_key in ("xquery", "query_consistency"):
                    loss_dict[f"loss_{loss_key}"] = hook(dec_out, targets, ctx)
            return loss_dict
        outputs: List[Dict[str, torch.Tensor]] = []
        # TODO(task-0.17): `images` is the padded batch tensor, so H, W are the
        # max image dims in the batch, not the per-sample original dims. The
        # plan assumes uniform LSJ crops at train/eval time, so this matches;
        # relax when evaluating on native-resolution images.
        H, W = images.shape[-2:]
        for b in range(images.shape[0]):
            outputs.append(self._panoptic_merge(dec_out["pred_logits"][b], dec_out["pred_masks"][b], H, W))
        return outputs
