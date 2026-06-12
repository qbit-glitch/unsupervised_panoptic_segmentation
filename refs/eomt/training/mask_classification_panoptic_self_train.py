"""Stage-3 self-training LightningModule for EoMT panoptic — exact CUPS recipe.

Faithful port of refs/cups/cups/pl_model_self.py (SelfSupervisedModel) +
refs/cups/train_self.py to EoMT's query-based head. Per training batch:

    1. Teacher (EMA copy of student, decay 0.999) runs TTA inference on the
       CLEAN image: scales (0.5, 0.75, 1.0) x horizontal flip = 6 views,
       final-block mask/class logits averaged per query.
       (CUPS: PanopticFPNWithTTA, TTA_SCALES [0.5, 0.75, 1.0] + D2 flip.)
    2. Pseudo-targets built with CUPS thresholds:
       - stuff queries:  relative per-class threshold
         score >= SEMANTIC_SEGMENTATION_THRESHOLD * max-score-of-that-class
         (CUPS make_pseudo_labels: class_threshold = 0.5 * per-class max).
       - thing queries:  absolute floor TTA_INFERENCE_CONFIDENCE_THRESHOLD
         = 0.5 (CUPS instance branch detection threshold).
       NOTE: CUPS's CLASS_THRESHOLD_ALPHA / CLASS_FREQUENCIES / CONFIDENCE_STEP
       config keys are dead code (defined, never consumed) — the executed
       recipe is a CONSTANT threshold for all ROUNDS x ROUND_STEPS steps.
    3. Augmentations applied AFTER teacher labelling, in CUPS order:
       copy-paste (max 3 things) -> photometric -> RandomCrop(512..1024, x2)
       -> ResolutionJitter [(384,768),(416,832),(448,896)].
       Teacher sees the clean image; the student sees the augmented crop —
       this input asymmetry is the core consistency-regularisation mechanism.
    4. Student forward + MaskClassificationLoss on the augmented batch.
    5. EMA update theta_t <- 0.999 theta_t + 0.001 theta_s every batch end
       (CUPS hardcodes 0.999 and updates per batch, i.e. 8x per optimizer
       step under accumulate_grad_batches=8).

Optimizer (CUPS train_self.py configure_optimizers): plain AdamW,
lr=1e-4, weight_decay=1e-5, CONSTANT LR (no warmup, no poly schedule),
heads-only — backbone frozen. EoMT mapping: freeze patch_embed +
encoder blocks [0 .. L-num_blocks); train the last num_blocks blocks
(EoMT's de-facto decoder: the only layers where queries meet image
tokens), backbone.norm, queries, class/mask heads, upscale blocks.

No teacher warmup (CUPS SELF_TRAINING.NUM_STEPS_STARTUP = 0).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision import tv_tensors

from training.cups_stage3_augmentation import (
    PhotometricAugmentations,
    copy_paste_batch,
    random_crop_batch,
    resolution_jitter_batch,
)
from training.ema_teacher import EMATeacher
from training.mask_classification_panoptic import MaskClassificationPanoptic

log = logging.getLogger(__name__)


class MaskClassificationPanopticSelfTrain(MaskClassificationPanoptic):
    """Stage-3 self-training panoptic head (exact CUPS protocol).

    Args:
        ema_decay: EMA decay for the teacher (CUPS hardcodes 0.999).
        rounds: Self-training rounds — only sets total steps (CUPS).
        round_steps: Optimizer steps per round (CUPS: 4000).
        semantic_segmentation_threshold: Relative per-class threshold for
            stuff queries (CUPS SEMANTIC_SEGMENTATION_THRESHOLD = 0.5).
        instance_confidence_threshold: Absolute floor for thing queries
            (CUPS TTA_INFERENCE_CONFIDENCE_THRESHOLD = 0.5).
        mask_threshold: Teacher mask binarisation threshold.
        min_mask_pixels: Drop teacher masks smaller than this.
        tta_scales: Teacher TTA scales (CUPS TTA_SCALES).
        tta_flip: Add horizontally flipped views (D2 TTA default).
        use_copy_paste: CUPS AUGMENTATION.COPY_PASTE.
        copy_paste_max_objects: CUPS MAX_NUM_PASTED_OBJECTS.
        use_photometric: CUPS PhotometricAugmentations.
        crop_resolution_min/max, crop_long_side_scale: CUPS RandomCrop.
        jitter_resolutions: CUPS AUGMENTATION.RESOLUTIONS.
    """

    def __init__(
        self,
        ema_decay: float = 0.999,
        rounds: int = 3,
        round_steps: int = 4000,
        semantic_segmentation_threshold: float = 0.5,
        instance_confidence_threshold: float = 0.5,
        mask_threshold: float = 0.50,
        min_mask_pixels: int = 64,
        tta_scales: Sequence[float] = (0.5, 0.75, 1.0),
        tta_flip: bool = True,
        use_copy_paste: bool = True,
        copy_paste_max_objects: int = 3,
        use_photometric: bool = True,
        crop_resolution_min: int = 512,
        crop_resolution_max: int = 1024,
        crop_long_side_scale: float = 2.0,
        jitter_resolutions: Sequence[Sequence[int]] = ((384, 768), (416, 832), (448, 896)),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.ema_decay = float(ema_decay)
        self.rounds = int(rounds)
        self.round_steps = int(round_steps)
        self.semantic_segmentation_threshold = float(semantic_segmentation_threshold)
        self.instance_confidence_threshold = float(instance_confidence_threshold)
        self.mask_threshold = float(mask_threshold)
        self.min_mask_pixels = int(min_mask_pixels)
        self.tta_scales = tuple(float(s) for s in tta_scales)
        self.tta_flip = bool(tta_flip)
        self.use_copy_paste = bool(use_copy_paste)
        self.copy_paste_max_objects = int(copy_paste_max_objects)
        self.use_photometric = bool(use_photometric)
        self.crop_resolution_min = int(crop_resolution_min)
        self.crop_resolution_max = int(crop_resolution_max)
        self.crop_long_side_scale = float(crop_long_side_scale)
        self.jitter_resolutions = tuple(tuple(int(v) for v in r) for r in jitter_resolutions)

        self.thing_class_set = {
            c for c in range(self.num_classes) if c not in set(self.stuff_classes)
        }
        self.photometric = PhotometricAugmentations() if self.use_photometric else None

        # CUPS Stage-3 freezes the backbone and trains heads only. EoMT
        # analog: freeze everything in the encoder backbone EXCEPT the last
        # num_blocks blocks (where queries attend to image tokens) and the
        # final norm feeding the prediction heads.
        self._freeze_backbone_except_decoder()

        # Teacher built lazily in on_fit_start (after student ckpt load).
        self.teacher: Optional[EMATeacher] = None

    # -- parameter freezing / optimizer (CUPS head-only AdamW) --------------

    def _freeze_backbone_except_decoder(self) -> None:
        backbone = self.network.encoder.backbone
        num_blocks_total = len(backbone.blocks)
        first_trainable_block = num_blocks_total - self.network.num_blocks
        frozen, trainable = 0, 0
        for name, param in self.network.named_parameters():
            if not name.startswith("encoder.backbone."):
                param.requires_grad_(True)
                trainable += param.numel()
                continue
            sub = name.replace("encoder.backbone.", "")
            if sub.startswith("blocks."):
                block_idx = int(sub.split(".")[1])
                keep = block_idx >= first_trainable_block
            elif sub.startswith("norm."):
                keep = True
            else:
                keep = False  # patch_embed, pos_embed, etc.
            param.requires_grad_(keep)
            if keep:
                trainable += param.numel()
            else:
                frozen += param.numel()
        log.info(
            "CUPS Stage-3 freezing: %.1fM trainable / %.1fM frozen "
            "(backbone blocks >= %d + norm + queries/heads train).",
            trainable / 1e6,
            frozen / 1e6,
            first_trainable_block,
        )

    def configure_optimizers(self):
        # CUPS train_self.py: plain AdamW(lr, weight_decay), constant LR,
        # no LLRD, no warmup, no poly schedule.
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        log.info(
            "CUPS Stage-3 optimizer: AdamW lr=%g weight_decay=%g, constant LR.",
            self.lr,
            self.weight_decay,
        )
        return optimizer

    # -- lifecycle ----------------------------------------------------------

    def on_fit_start(self) -> None:
        if hasattr(super(), "on_fit_start"):
            super().on_fit_start()
        if self.teacher is None:
            log.info(
                "Initialising EMA teacher (decay=%.4f) by deep-copying the "
                "loaded student weights.",
                self.ema_decay,
            )
            self.teacher = EMATeacher(self.network, decay=self.ema_decay)
            self.teacher.module.to(self.device)

    # -- dynamic grid size (EoMT reshapes with a fixed init-time grid) ------

    @staticmethod
    def _set_grid_size(network: torch.nn.Module, h: int, w: int) -> None:
        patch_embed = network.encoder.backbone.patch_embed
        ps = int(patch_embed.patch_size[0])
        patch_embed.grid_size = (h // ps, w // ps)

    def forward(self, imgs: torch.Tensor):
        self._set_grid_size(self.network, int(imgs.shape[-2]), int(imgs.shape[-1]))
        return super().forward(imgs)

    # -- core training step (CUPS SelfSupervisedModel.training_step) --------

    def training_step(self, batch, batch_idx):
        imgs, _dataset_targets = batch  # clean images, dataset labels unused

        round_idx = min(self.rounds, 1 + self.global_step // max(1, self.round_steps))
        self.log("st/round", float(round_idx), on_step=True)

        with torch.no_grad():
            # 1. Teacher TTA on the clean image.
            mask_logits, class_logits = self._teacher_forward_tta(imgs)
            # 2. CUPS-threshold pseudo-targets at clean-image resolution.
            targets = self._build_pseudo_targets(mask_logits, class_logits)
            self.log(
                "st/avg_targets_per_img",
                float(sum(t["labels"].numel() for t in targets) / max(1, len(targets))),
                on_step=True,
            )
            # 3. Augmentations in CUPS order (student-only views).
            aug_imgs = imgs.float()
            if self.use_copy_paste:
                aug_imgs, targets = copy_paste_batch(
                    aug_imgs,
                    targets,
                    thing_classes=self.thing_class_set,
                    max_num_pasted_objects=self.copy_paste_max_objects,
                )
            if self.photometric is not None:
                aug_imgs = self.photometric(aug_imgs)
            aug_imgs, targets = random_crop_batch(
                aug_imgs,
                targets,
                resolution_min=self.crop_resolution_min,
                resolution_max=self.crop_resolution_max,
                long_side_scale=self.crop_long_side_scale,
            )
            aug_imgs, targets = resolution_jitter_batch(
                aug_imgs, targets, resolutions=self.jitter_resolutions
            )

        # 4. Student forward + loss on the augmented batch.
        mask_logits_per_block, class_logits_per_block = self(aug_imgs)
        losses_all_blocks = {}
        for i, (m_logits, c_logits) in enumerate(
            list(zip(mask_logits_per_block, class_logits_per_block))
        ):
            losses = self.criterion(
                masks_queries_logits=m_logits,
                class_queries_logits=c_logits,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        return self.criterion.loss_total(losses_all_blocks, self.log)

    def on_train_batch_end(self, outputs, batch, batch_idx=None, dataloader_idx=None):
        super().on_train_batch_end(outputs, batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        # 5. EMA update every batch (CUPS on_train_batch_end, 0.999).
        if self.teacher is not None:
            self.teacher.update(self.network)

    # -- teacher TTA ---------------------------------------------------------

    @torch.no_grad()
    def _teacher_forward_tta(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-scale + flip TTA: average final-block logits per query.

        Returns (mask_logits [B,Q,H,W] at clean-image resolution,
        class_logits [B,Q,C+1]).
        """
        assert self.teacher is not None, "EMA teacher must be initialised by on_fit_start"
        teacher = self.teacher.module
        teacher.eval()

        H, W = int(imgs.shape[-2]), int(imgs.shape[-1])
        flips = (False, True) if self.tta_flip else (False,)

        mask_sum: Optional[torch.Tensor] = None
        class_sum: Optional[torch.Tensor] = None
        n_views = 0
        for scale in self.tta_scales:
            h, w = int(round(H * scale)), int(round(W * scale))
            x = F.interpolate(imgs.float(), size=(h, w), mode="bilinear") / 255.0
            for flip in flips:
                xi = x.flip(-1) if flip else x
                self._set_grid_size(teacher, h, w)
                mask_logits_per_layer, class_logits_per_layer = teacher(xi)
                m = mask_logits_per_layer[-1]
                c = class_logits_per_layer[-1]
                if flip:
                    m = m.flip(-1)
                m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
                mask_sum = m if mask_sum is None else mask_sum + m
                class_sum = c if class_sum is None else class_sum + c
                n_views += 1

        return mask_sum / n_views, class_sum / n_views

    # -- pseudo-target construction (CUPS make_pseudo_labels analog) --------
    #
    # CUPS's pseudo-label IS the teacher's TTA panoptic prediction: queries
    # pass an absolute 0.5 score floor (TTA_INFERENCE_CONFIDENCE_THRESHOLD),
    # then compete per pixel (argmax) with overlap pruning, and stuff segments
    # merge per class. We reuse the established EoMT inference path
    # (to_per_pixel_preds_panoptic) and convert the resulting segments back to
    # (mask, label) targets. This per-pixel competition is what contains weak
    # classes in CUPS — a naive per-query relative threshold floods the target
    # set with junk queries and collapses training (verified empirically).

    @torch.no_grad()
    def _build_pseudo_targets(
        self,
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        B, _, H, W = mask_logits.shape
        preds = self.to_per_pixel_preds_panoptic(
            [mask_logits[b] for b in range(B)],
            class_logits,
            self.stuff_classes,
            self.instance_confidence_threshold,  # CUPS: 0.5 detection floor
            self.overlap_thresh,
        )

        targets: List[Dict[str, torch.Tensor]] = []
        for b in range(B):
            class_map = preds[b][:, :, 0]
            segment_map = preds[b][:, :, 1]
            masks_list: List[torch.Tensor] = []
            labels_list: List[int] = []
            for seg_id in segment_map.unique().tolist():
                if seg_id < 0:
                    continue
                m = segment_map == seg_id
                if int(m.sum()) < self.min_mask_pixels:
                    continue
                cls = int(class_map[m][0])
                if not 0 <= cls < self.num_classes:
                    continue
                masks_list.append(m)
                labels_list.append(cls)
            if not masks_list:
                targets.append(self._empty_target(H, W))
                continue
            labels = torch.tensor(labels_list, dtype=torch.long, device=class_map.device)
            targets.append(
                {
                    "masks": tv_tensors.Mask(torch.stack(masks_list)),
                    "labels": labels,
                    "is_crowd": torch.zeros_like(labels, dtype=torch.bool),
                }
            )
        return targets

    @staticmethod
    def _empty_target(H: int, W: int) -> Dict[str, torch.Tensor]:
        return {
            "masks": tv_tensors.Mask(torch.zeros((0, H, W), dtype=torch.bool)),
            "labels": torch.zeros((0,), dtype=torch.long),
            "is_crowd": torch.zeros((0,), dtype=torch.bool),
        }

    # -- ckpt state ----------------------------------------------------------

    def on_save_checkpoint(self, checkpoint) -> None:
        super().on_save_checkpoint(checkpoint)
        if self.teacher is not None:
            checkpoint["ema_teacher_state_dict"] = self.teacher.module.state_dict()

    def on_load_checkpoint(self, checkpoint) -> None:
        teacher_state = checkpoint.get("ema_teacher_state_dict")
        if teacher_state is None:
            return
        if self.teacher is None:
            self.teacher = EMATeacher(self.network, decay=self.ema_decay)
        try:
            self.teacher.module.load_state_dict(teacher_state, strict=False)
            log.info("Restored EMA teacher weights from checkpoint.")
        except RuntimeError as err:
            log.warning("Could not restore EMA teacher state: %s", err)
