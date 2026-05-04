from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from detectron2.layers import FrozenBatchNorm2d
from detectron2.structures import BitMasks, Boxes, Instances
from detectron2.utils.events import EventStorage
from torch import Tensor
from yacs.config import CfgNode

from cups.augmentation import RandomCrop
from cups.data.utils import get_bounding_boxes, instances_to_masks
from cups.model import panoptic_cascade_mask_r_cnn
from cups.pl_model_pseudo import UnsupervisedModel
from cups.stage4_utils import resolve_stage4_class_ids

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class SelfSupervisedModel(UnsupervisedModel):
    """This class implements the self-supervised model for training a Panoptic Cascade Mask R-CNN."""

    def __init__(
        self,
        model: nn.Module,
        num_thing_pseudo_classes: int,
        num_stuff_pseudo_classes: int,
        config: CfgNode,
        thing_classes: Set[int],
        stuff_classes: Set[int],
        copy_paste_augmentation: nn.Module = nn.Identity(),
        photometric_augmentation: nn.Module = nn.Identity(),
        resolution_jitter_augmentation: nn.Module = nn.Identity(),
        class_names: List[str] | None = None,
        classes_mask: List[bool] | None = None,
        mask_refiner: Any = None,
    ) -> None:
        """Constructor method.

        Args:
            model (nn.Module): Cascade Panoptic Mask R-CNN.
            num_thing_pseudo_classes (int): Number of estimated pseudo thing classes.
            num_thing_pseudo_classes (int): Number of estimated stuff thing classes.
            config (CfgNode): Config object.
            thing_classes (Set[int]): Set of thing classes.
            stuff_classes (Set[int]): Set of stuff classes.
            copy_paste_augmentation (nn.Module): Copy-paste augmentation module.
            photometric_augmentation (nn.Module): Photometric augmentation module.
            resolution_jitter_augmentation (nn.Module): Resolution jitter augmentation module.
            class_to_name (List[str] | None): List containing the name of the semantic classes.
            classes_mask (List[bool] | None): Mask of valid classes in validation set.
            mask_refiner: Optional MaskRefiner for classical mask refinement.
        """
        # Call super constructor
        super(SelfSupervisedModel, self).__init__(
            model=model.model,  # type: ignore
            num_thing_pseudo_classes=num_thing_pseudo_classes,
            num_stuff_pseudo_classes=num_stuff_pseudo_classes,
            config=config,
            thing_classes=thing_classes,
            stuff_classes=stuff_classes,
            copy_paste_augmentation=copy_paste_augmentation,
            photometric_augmentation=photometric_augmentation,
            resolution_jitter_augmentation=resolution_jitter_augmentation,
            class_names=class_names,
            classes_mask=classes_mask,
        )
        # Init ema model
        self.teacher_model: nn.Module = copy.deepcopy(model)
        # Init crop module
        if config.DATA.DATASET == "kitti":
            self.crop_module: nn.Module = RandomCrop(resolution_max=368, resolution_min=288, long_side_scale=3.369)
        else:
            self.crop_module = RandomCrop()
        # Set self-training round
        self.round: int = 1
        # Mask refiner (classical refinement)
        self.mask_refiner = mask_refiner
        self.stage4_ids = resolve_stage4_class_ids(
            config,
            self.hparams.thing_pseudo_classes,
            self.hparams.stuff_pseudo_classes,
        )

        # ── Fine-object SAM supervision (Stage-4 fine-tuning) ──────────────
        fo_cfg = getattr(config.SELF_TRAINING, "FINE_OBJECT", None)
        self._fo_enabled: bool = (
            fo_cfg is not None and getattr(fo_cfg, "ENABLED", False)
        )
        self._fo_logits_cache: Dict[str, Any] = {}
        self._fo_load_labels: Optional[set] = None
        if self._fo_enabled:
            from cups.losses.fine_object import FineObjectSemanticLoss

            # Build stuff_channel_map if provided (from extract_cups_class_mapping.py output)
            raw_stuff_map = getattr(fo_cfg, "STUFF_CHANNEL_MAP", None)
            stuff_channel_map = dict(raw_stuff_map) if raw_stuff_map else {}

            # Parse new params for Exp 1-3 modes
            raw_common_idx = getattr(fo_cfg, "COMMON_THING_CHANNEL_INDICES", None)
            common_idx = list(raw_common_idx) if raw_common_idx else []
            raw_class_freq = getattr(fo_cfg, "CLASS_FREQUENCIES_SAM3", None)
            class_freq = list(raw_class_freq) if raw_class_freq else None

            self._fo_loss = FineObjectSemanticLoss(
                mode=getattr(fo_cfg, "MODE", "thing_focal_stuff_entropy"),
                thing_class_idx=int(getattr(fo_cfg, "THING_CLASS_IDX", -1)),
                stuff_channel_map=stuff_channel_map,
                focal_gamma=float(getattr(fo_cfg, "FOCAL_GAMMA", 2.0)),
                use_iou_weighting=bool(getattr(fo_cfg, "USE_IOU_WEIGHTING", True)),
                min_hard_iou=float(getattr(fo_cfg, "MIN_HARD_IOU", 0.10)),
                # Exp 1
                common_thing_channel_indices=common_idx,
                teacher_logit_weight=float(getattr(fo_cfg, "TEACHER_LOGIT_WEIGHT", 1.0)),
                # Exp 2
                class_frequencies=class_freq,
                gamma_scale_factor=float(getattr(fo_cfg, "GAMMA_SCALE_FACTOR", 1.0)),
                # Exp 3
                stuff_kd_lambda=float(getattr(fo_cfg, "STUFF_KD_LAMBDA", 0.1)),
                stuff_channel_start=int(getattr(fo_cfg, "STUFF_CHANNEL_START", 1)),
            )
            self._fo_weight = float(getattr(fo_cfg, "WEIGHT", 0.1))
            self._fo_masks_dir = Path(fo_cfg.SAM_MASKS_DIR)
            self._fo_min_iou = float(getattr(fo_cfg, "MIN_IOU_SCORE", 0.10))
            self._fo_max_masks = int(getattr(fo_cfg, "MAX_MASKS_PER_IMAGE", 30))
            raw_load_labels = getattr(fo_cfg, "LOAD_CLASS_LABELS", ())
            self._fo_load_labels: Optional[set] = set(raw_load_labels) if raw_load_labels else None
            # Hook student sem_seg predictor to capture logits without touching internals
            self.model.sem_seg_head.predictor.register_forward_hook(
                lambda _m, _inp, out: self._fo_logits_cache.update({"logits": out})
            )
            # Hook teacher sem_seg predictor for teacher-gated modes (mc_panda, stuff_kd)
            self._fo_teacher_logits_cache: Dict[str, Any] = {}
            self.teacher_model.model.sem_seg_head.predictor.register_forward_hook(
                lambda _m, _inp, out: self._fo_teacher_logits_cache.update({"logits": out})
            )
            log.info(
                "FineObjectSemanticLoss enabled (mode=%s, weight=%.3f, dir=%s)",
                fo_cfg.MODE, self._fo_weight, fo_cfg.SAM_MASKS_DIR,
            )

    def forward(self, input: List[Dict[str, Tensor]]) -> List[Dict[str, Any]]:
        """Just wraps the forward pass of the Cascade Panoptic Mask R-CNN.

        Args:
            input (List[Dict[str, Tensor]]): List of inputs (images during inference and images + labels for training)

        Returns:
            output (List[Dict[str, Any]]): Prediction of the model for training the loss.
        """
        output = self.model(input)
        return output  # type: ignore

    def training_step(self, batch: List[Dict[str, Any]], batch_index: int) -> Dict[str, Tensor]:
        """Training step.

        Args:
            batch (List[Dict[str, Any]])): Batch of training data.
            batch_index (int): Batch index.

        Returns:
            loss (Dict[str, Tensor]): Loss value in a dict.
        """
        # Make storage object
        if self.storage is None:
            self.storage = EventStorage(0)
            self.storage.__enter__()
        # Snapshot image names before make_pseudo_labels drops them
        batch_image_names: List[str] = [
            sample.get("image_name", "") for sample in batch
        ]
        # Make pseudo labels
        if self._fo_enabled:
            self._fo_teacher_logits_cache.clear()
        self.teacher_model.eval()
        with torch.no_grad():
            # Make prediction with TTA
            predictions_tta = self.teacher_model(batch)
            # Generate pseudo labels based on TTA prediction
            pseudo_labels = self.make_pseudo_labels(predictions_tta, batch, self.hparams.stuff_pseudo_classes)
            # Apply classical mask refinement (if enabled)
            if self.mask_refiner is not None:
                pseudo_labels = self.mask_refiner.refine_pseudo_labels(pseudo_labels)
            # Perform copy-paste augmentation
            if self.copy_paste_augmentation is not None:
                pseudo_labels = self.copy_paste_augmentation(pseudo_labels, pseudo_labels)
            # Apply photometric augmentations
            pseudo_labels = self.photometric_augmentation(pseudo_labels)
            # Crop data
            pseudo_labels = self.crop_module(pseudo_labels)
            # Perform resolution jitter
            pseudo_labels = self.resolution_jitter_augmentation(pseudo_labels)
            # Sanitize sem_seg: set any out-of-range class IDs to ignore (255)
            num_classes = self.model.sem_seg_head.predictor.out_channels
            for sample in pseudo_labels:
                sem = sample["sem_seg"]
                invalid = (sem < 0) | ((sem >= num_classes) & (sem != 255))
                if invalid.any():
                    sem[invalid] = 255
                sample["sem_seg"] = sem
        # Filter degenerate boxes that RandomCrop/resize aug can introduce.
        # Zero-area boxes (x2<=x1 or y2<=y1) produce log(0)→-inf in the RPN
        # Box2BoxTransform, causing loss_rpn_loc=inf and poisoned gradients.
        for sample in pseudo_labels:
            inst = sample.get("instances")
            if inst is not None and len(inst) > 0:
                boxes = inst.gt_boxes.tensor
                valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                if not valid.all():
                    sample["instances"] = inst[valid]

        # Attach image names so downstream hooks (SAM3 adapter, depth dice) can
        # resolve per-image files from disk during the student forward pass.
        for sample, img_name in zip(pseudo_labels, batch_image_names):
            if img_name and "file_name" not in sample:
                sample["file_name"] = img_name

        # Clear stale logits cache before the student forward pass
        if self._fo_enabled:
            self._fo_logits_cache.clear()

        # Train using self pseudo labels
        loss_dict = self.model(pseudo_labels)

        # ── Fine-object SAM supervision ──────────────────────────────────
        if self._fo_enabled:
            low_logits = self._fo_logits_cache.get("logits")
            if low_logits is not None:
                # Upsample predictor output to full resolution
                common_stride = self.model.sem_seg_head.common_stride
                full_logits = F.interpolate(
                    low_logits.float(),
                    scale_factor=common_stride,
                    mode="bilinear",
                    align_corners=False,
                )
                # Upscale teacher logits if available (for mc_panda / stuff_kd modes)
                teacher_full_logits: Optional[Tensor] = None
                low_teacher = self._fo_teacher_logits_cache.get("logits")
                if low_teacher is not None:
                    with torch.no_grad():
                        teacher_full_logits = F.interpolate(
                            low_teacher.float(),
                            scale_factor=common_stride,
                            mode="bilinear",
                            align_corners=False,
                        ).detach()
                sam_masks, sam_ious, sam_cls = self._load_sam_masks(batch_image_names, full_logits.device)
                fo_loss = self._fo_loss(
                    full_logits, sam_masks, sam_ious, sam_cls, self._fo_min_iou,
                    teacher_logits=teacher_full_logits,
                )
                loss_dict["loss_fine_object"] = fo_loss * self._fo_weight
                self.log("losses/fine_object", fo_loss, sync_dist=True)

        # Compute sum of losses
        loss: Tensor = sum(loss_dict.values())
        # Log final loss
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        # Log all losses
        for key, value in loss_dict.items():
            self.log("losses/" + key, value, sync_dist=True)
        # Log media
        if ((self.global_step) % self.hparams.config.TRAINING.LOG_MEDIA_N_STEPS) == 0:
            # Make inference prediction
            self.model.eval()
            with torch.no_grad():
                prediction = self.model([{"image": sample["image"]} for sample in pseudo_labels])
            self.model.train()
            self.log_visualizations(pseudo_labels, prediction)
        return {"loss": loss}

    def make_pseudo_labels(
        self,
        predictions: List[Dict[str, Tuple[Tensor, List[Dict[str, Any]]]]],
        images: List[Dict[str, Any]],
        stuff_classes: Tuple[int, ...],
    ) -> List[Dict[str, Any]]:
        """Function generates pseudo labels from TTA predictions.

        Args:
            predictions (List[Dict[str, Tuple[Tensor, List[Dict[str, Any]]]]]): TTA predictions.
            images (List[Tensor]): Corresponding original images.
            stuff_classes (Tuple[int, ...]): Semantic stuff classes.

        Returns:
            pseudo_labels (List[Dict[str, Any]]): Pseudo labels.
        """
        # Make output list
        pseudo_labels = []
        # Iterate over batch size
        for sample, image in zip(predictions, images):
            # Make weights for semantic and instance segmentation
            weight_semantic = (
                torch.ones(
                    sample["panoptic_seg"][0].amax().item() + 1,  # type: ignore
                    device=sample["panoptic_seg"][0].device,  # type: ignore
                    dtype=torch.long,
                )
                * 255.0
            )
            weight_instance = torch.zeros(  # type: ignore
                sample["panoptic_seg"][0].amax().item() + 1,  # type: ignore
                device=sample["panoptic_seg"][0].device,  # type: ignore
                dtype=torch.long,  # type: ignore
            )
            # Init object semantics
            object_semantics = []
            # Fill weights and get object semantics
            for object in sample["panoptic_seg"][1]:
                if object["isthing"]:
                    weight_semantic[object["id"]] = 0
                    weight_instance[object["id"]] = weight_instance.amax() + 1
                    object_semantics.append(object["category_id"])
                else:
                    weight_semantic[object["id"]] = object["category_id"]
            # Get instance map
            instance = torch.embedding(indices=sample["panoptic_seg"][0], weight=weight_instance.view(-1, 1)).squeeze()
            # Get raw semantic segmentation
            semantic_segmentation_raw = sample["sem_seg"]
            # Get max class scores
            max_class_scores = semantic_segmentation_raw.amax(dim=(1, 2), keepdim=True)  # type: ignore
            # Compute class threshold. Stage-4 lowers thresholds only for
            # resolved rare stuff classes while keeping common classes strict.
            stage4_cfg = getattr(self.hparams.config, "STAGE4", None)
            if stage4_cfg is not None and getattr(stage4_cfg, "ENABLED", False):
                factors = torch.full(
                    (semantic_segmentation_raw.shape[0], 1, 1),
                    float(getattr(stage4_cfg, "TAU_COMMON", 0.70)),
                    device=semantic_segmentation_raw.device,
                    dtype=semantic_segmentation_raw.dtype,
                )
                for rare_cls in self.stage4_ids.rare_stuff_targets:
                    if 0 <= rare_cls < factors.shape[0]:
                        factors[rare_cls] = float(getattr(stage4_cfg, "TAU_RARE", 0.25))
                class_threshold = max_class_scores * factors
            else:
                class_threshold = max_class_scores * self.hparams.config.SELF_TRAINING.SEMANTIC_SEGMENTATION_THRESHOLD
            # Make semantic pseudo label
            semantic_segmentation = torch.where(  # type: ignore
                semantic_segmentation_raw > class_threshold, semantic_segmentation_raw, 0.0  # type: ignore
            )
            semantic_segmentation_pseudo = semantic_segmentation.argmax(dim=0)
            semantic_segmentation_pseudo[semantic_segmentation.sum(dim=0) == 0] = 255
            # M5: Compute per-pixel confidence weights from teacher softmax
            confidence_weights = None
            lora_cfg = getattr(self.hparams.config.MODEL, "LORA", None)
            if (
                lora_cfg is not None
                and hasattr(lora_cfg, "MITIGATIONS")
                and getattr(lora_cfg.MITIGATIONS.CONFIDENCE_WEIGHTED_LOSS, "ENABLED", False)
            ):
                m5_cfg = lora_cfg.MITIGATIONS.CONFIDENCE_WEIGHTED_LOSS
                temp = m5_cfg.TEMPERATURE
                min_w = m5_cfg.MIN_WEIGHT
                # semantic_segmentation_raw: (C, H, W) logits
                conf = (semantic_segmentation_raw / temp).softmax(dim=0).max(dim=0).values
                confidence_weights = conf.clamp(min=min_w)  # (H, W)

            # Construct output
            img_hw = tuple(image["image"].shape[1:])
            if instance.amax() > 0.0:
                instance_masks = instances_to_masks(instance)
                boxes = get_bounding_boxes(instance)
                # Degenerate boxes (x2<=x1 or y2<=y1) produce log(0)→-inf in
                # the RPN Box2BoxTransform and cause loss_rpn_loc=inf.  Drop them.
                valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                if not valid.all():
                    instance_masks = instance_masks[valid]
                    boxes = boxes[valid]
                    keep = valid.tolist()
                    object_semantics = [s for s, v in zip(object_semantics, keep) if v]
                object_semantics_tensor: Tensor = torch.tensor(object_semantics, device=instance.device)

            if instance.amax() > 0.0 and boxes.shape[0] > 0:
                sample_dict = {
                    "image": image["image"].squeeze(),
                    "sem_seg": semantic_segmentation_pseudo.long(),
                    "instances": Instances(
                        image_size=img_hw,
                        gt_masks=BitMasks(instance_masks),
                        gt_boxes=Boxes(boxes),
                        gt_classes=object_semantics_tensor,
                    ),
                }
            else:
                sample_dict = {
                    "image": image["image"].squeeze(),
                    "sem_seg": semantic_segmentation_pseudo.long(),
                    "instances": Instances(
                        image_size=img_hw,
                        gt_masks=BitMasks(torch.zeros(0, *img_hw).bool()),
                        gt_boxes=Boxes(torch.zeros(0, 4).long()),
                        gt_classes=torch.zeros(0).long(),
                    ),
                }
            if confidence_weights is not None:
                sample_dict["confidence_weights"] = confidence_weights
            if (
                stage4_cfg is not None
                and getattr(stage4_cfg, "ENABLED", False)
                and float(getattr(stage4_cfg, "REPLAY_WEIGHT", 0.0)) > 0.0
            ):
                sample_dict["pseudo_onehot"] = semantic_segmentation_raw.detach()
            pseudo_labels.append(sample_dict)
        return pseudo_labels

    def validation_step(
        self,
        batch: Tuple[List[Dict[str, Tensor]], Tensor, List[str]],
        batch_index: int,
    ) -> None:
        """Validation step — PQ metrics (via super) + training losses on val pseudo-labels.

        Losses are computed by running teacher→pseudo-label→student(train mode, no_grad)
        on the val batch, identical to the training pipeline but without augmentation/cropping.
        """
        super().validation_step(batch, batch_index)

        images, _panoptic_labels, image_names = batch

        # Build teacher input (attach image_name for SAM mask loading)
        teacher_input = [
            {"image": img["image"], "image_name": name}
            for img, name in zip(images, image_names)
        ]

        with torch.no_grad():
            self.teacher_model.eval()
            predictions_tta = self.teacher_model(teacher_input)
            pseudo_labels = self.make_pseudo_labels(
                predictions_tta, teacher_input, self.hparams.stuff_pseudo_classes
            )
            # Sanitize sem_seg (same guard as training_step)
            num_classes = self.model.sem_seg_head.predictor.out_channels
            for sample in pseudo_labels:
                sem = sample["sem_seg"]
                invalid = (sem < 0) | ((sem >= num_classes) & (sem != 255))
                if invalid.any():
                    sem[invalid] = 255
                sample["sem_seg"] = sem

            if self._fo_enabled:
                self._fo_logits_cache.clear()

            # Student forward in train mode to get losses (fresh EventStorage avoids
            # interfering with the mid-training storage that may still be open)
            prev_training = self.model.training
            self.model.train()
            with EventStorage(0):
                val_loss_dict: Dict[str, Tensor] = self.model(pseudo_labels)
            self.model.train(prev_training)

            # Fine-object SAM loss (mirrors training_step logic)
            if self._fo_enabled:
                low_logits = self._fo_logits_cache.get("logits")
                if low_logits is not None:
                    common_stride = self.model.sem_seg_head.common_stride
                    full_logits = F.interpolate(
                        low_logits.float(),
                        scale_factor=common_stride,
                        mode="bilinear",
                        align_corners=False,
                    )
                    sam_masks, sam_ious, sam_cls = self._load_sam_masks(image_names, full_logits.device)
                    fo_loss = self._fo_loss(full_logits, sam_masks, sam_ious, sam_cls, self._fo_min_iou)
                    val_loss_dict["loss_fine_object"] = fo_loss * self._fo_weight

        # Accumulate per-loss scalars for epoch-end averaging
        if not hasattr(self, "_val_loss_accum"):
            self._val_loss_accum: Dict[str, List[float]] = {}
        for k, v in val_loss_dict.items():
            self._val_loss_accum.setdefault(k, []).append(v.item())

    def on_validation_epoch_end(self) -> None:
        """PQ metrics (via super) then per-loss averages — logged to W&B and printed."""
        super().on_validation_epoch_end()

        # ── DDP checkpoint fix ─────────────────────────────────────────────
        # Lightning 2.6 requires the monitored metric ("pq_val") to be present
        # in callback_metrics on ALL ranks for ModelCheckpoint to save.
        # The base class logs pq_val with rank_zero_only=True, so rank-1 never
        # sees it and the checkpoint callback silently skips every save after
        # the first one.  Fix: broadcast rank-0's value to all ranks, then
        # re-log so every rank has the same pq_val.
        pq_cb = self.trainer.callback_metrics.get("pq_val", torch.tensor(0.0))
        pq_tensor = torch.tensor(float(pq_cb), device=self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(pq_tensor, src=0)
        # Directly write into callback_metrics on all ranks so ModelCheckpoint
        # sees the real PQ value before its own on_validation_epoch_end runs.
        # Do NOT call self.log() again — super() already logged pq_val and
        # Lightning 2.6 forbids logging the same metric twice with different args.
        self.trainer.callback_metrics["pq_val"] = pq_tensor
        # ── end DDP checkpoint fix ─────────────────────────────────────────

        if not getattr(self, "_val_loss_accum", {}):
            return

        is_global_zero = getattr(self.trainer, "is_global_zero", True)

        # Average across batches
        avg_losses: Dict[str, float] = {
            k: sum(v) / len(v) for k, v in self._val_loss_accum.items()
        }
        total_loss: float = sum(avg_losses.values())

        # Log each loss and total to W&B
        for k, v in avg_losses.items():
            self.log(f"val_losses/{k}", v, rank_zero_only=True, sync_dist=False)
        self.log("val_losses/total", total_loss, rank_zero_only=True, sync_dist=False)

        # Print CSV-style summary to stdout (same style as PQ line above)
        if is_global_zero:
            keys_sorted = sorted(avg_losses.keys())
            header = ", ".join(keys_sorted) + ", total_loss"
            values = ", ".join(f"{avg_losses[k]:.4f}" for k in keys_sorted) + f", {total_loss:.4f}"
            print("\nVal Losses: " + header)
            print(values)

        self._val_loss_accum = {}

    def _load_sam_masks(
        self,
        image_names: List[str],
        device: torch.device,
    ) -> Tuple[
        List[Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
    ]:
        """Load pre-computed SAM fine masks for a batch.

        Returns:
            Tuple of (masks_list, ious_list, class_labels_list).
            Each entry is ``(N, H, W)`` bool / ``(N,)`` float / ``(N,)`` int64,
            or ``None`` when no mask file exists for that image.
        """
        masks_list: List[Optional[torch.Tensor]] = []
        ious_list: List[Optional[torch.Tensor]] = []
        cls_list: List[Optional[torch.Tensor]] = []

        for img_name in image_names:
            if not img_name:
                masks_list.append(None); ious_list.append(None); cls_list.append(None)
                continue

            path = Path(img_name)
            city = path.parent.name  # works for both leftImg8bit and leftImg8bit_sequence layouts
            stem = path.stem.replace("_leftImg8bit_sequence", "").replace("_leftImg8bit", "")
            mask_path = self._fo_masks_dir / city / f"{stem}_fine_masks.npz"

            if not mask_path.exists():
                masks_list.append(None); ious_list.append(None); cls_list.append(None)
                continue

            data = np.load(str(mask_path))
            masks_np: np.ndarray = data["masks"]       # (N, H, W) bool
            ious_np: np.ndarray = data["iou_scores"]   # (N,) float
            cls_np: np.ndarray = data.get("class_labels", np.full(len(masks_np), -1, dtype=np.int32))

            if masks_np.shape[0] == 0:
                masks_list.append(None); ious_list.append(None); cls_list.append(None)
                continue

            # Optionally filter to a whitelist of SAM3 class labels (e.g. stuff-only).
            if self._fo_load_labels is not None:
                keep_cls = np.isin(cls_np, list(self._fo_load_labels))
                masks_np = masks_np[keep_cls]
                ious_np = ious_np[keep_cls]
                cls_np = cls_np[keep_cls]
                if masks_np.shape[0] == 0:
                    masks_list.append(None); ious_list.append(None); cls_list.append(None)
                    continue

            # Keep top-K highest-confidence masks to cap memory
            if masks_np.shape[0] > self._fo_max_masks:
                top_k = np.argsort(ious_np)[::-1][: self._fo_max_masks]
                masks_np = masks_np[top_k]
                ious_np = ious_np[top_k]
                cls_np = cls_np[top_k]

            masks_list.append(torch.from_numpy(masks_np).to(device))
            ious_list.append(torch.from_numpy(ious_np.astype(np.float32)).to(device))
            cls_list.append(torch.from_numpy(cls_np.astype(np.int64)).to(device))

        return masks_list, ious_list, cls_list

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Updates teacher model via EMA (unless disabled for ablation).

        Args:
            outputs (Any): Unused.
            batch (Any): Unused.
            batch_idx (Any): Unused.
        """
        # Skip EMA if disabled (Exp 13: test LoRA implicit smoothing)
        if getattr(self.hparams.config.SELF_TRAINING, "DISABLE_EMA", False):
            return

        # Perform EMA update
        for train_parameter, teacher_parameter in zip(  # type: ignore
            self.model.parameters(), self.teacher_model.model.parameters()  # type: ignore
        ):  # type: ignore
            teacher_parameter.data.mul_(0.999).add_((1.0 - 0.999) * train_parameter.data)

        # M3: Spectral norm ball projection on magnitude vectors
        lora_cfg = getattr(self.hparams.config.MODEL, "LORA", None)
        if (
            lora_cfg is not None
            and getattr(lora_cfg, "ENABLED", False)
            and hasattr(lora_cfg, "MITIGATIONS")
            and getattr(lora_cfg.MITIGATIONS.SPECTRAL_NORM_BALL, "ENABLED", False)
        ):
            from cups.model.lora import spectral_norm_project

            backbone = ProgressiveLoRACallback._find_vit_backbone(self.model)
            if backbone is not None:
                delta = lora_cfg.MITIGATIONS.SPECTRAL_NORM_BALL.DELTA
                n_proj = spectral_norm_project(backbone, delta)
                if n_proj > 0 and self.global_step % 100 == 0:
                    log.info("M3: projected %d magnitude vectors (step %d)", n_proj, self.global_step)

    def on_train_epoch_end(self) -> None:
        """Stuff to perform at the end of the epoch."""
        # Just close the storage object
        self.storage.__exit__(None, None, None)  # type: ignore
        # Set storage to Nona
        self.storage = None

    def configure_optimizers(self):
        """Builds the models' optimizer (+ optional M1 cosine warmup scheduler).

        When LoRA is enabled, delegates to the parent class DoRA-aware optimizer
        which creates 6 param groups with differential learning rates. Otherwise
        falls back to head-only optimizer.

        Returns:
            optimizer or dict with optimizer + lr_scheduler.
        """
        lora_cfg = getattr(self.hparams.config.MODEL, "LORA", None)
        lora_enabled = lora_cfg is not None and getattr(lora_cfg, "ENABLED", False)

        if lora_enabled:
            # Use parent's DoRA-aware optimizer (6 param groups)
            optimizer = super().configure_optimizers()
            if isinstance(optimizer, dict):
                optimizer = optimizer["optimizer"]
        else:
            # Head-only optimizer (original behavior)
            parameters = [
                parameter for name, parameter in self.model.named_parameters()
                if ("head" in name) and ("norm" not in name)
            ]
            if self.hparams.config.TRAINING.OPTIMIZER == "sgd":
                optimizer = torch.optim.SGD(
                    params=parameters,
                    lr=self.hparams.config.TRAINING.SGD.LEARNING_RATE,
                    weight_decay=self.hparams.config.TRAINING.SGD.WEIGHT_DECAY,
                    momentum=self.hparams.config.TRAINING.SGD.MOMENTUM,
                )
                log.info("SGD used.")
            else:
                optimizer = torch.optim.AdamW(
                    params=parameters,
                    lr=self.hparams.config.TRAINING.ADAMW.LEARNING_RATE,
                    weight_decay=self.hparams.config.TRAINING.ADAMW.WEIGHT_DECAY,
                    betas=self.hparams.config.TRAINING.ADAMW.BETAS,
                )
                log.info("AdamW used.")

        # M1: Cosine LR warmup for LoRA param groups
        if (
            lora_enabled
            and hasattr(lora_cfg, "MITIGATIONS")
            and getattr(lora_cfg.MITIGATIONS.COSINE_WARMUP, "ENABLED", False)
        ):
            warmup_steps = lora_cfg.MITIGATIONS.COSINE_WARMUP.WARMUP_STEPS
            # Build per-group lambda: head groups (idx 0,1) get 1.0, LoRA groups ramp
            lora_group_names = {"dora_A", "dora_B", "dora_magnitude", "dora_conv"}

            def make_lambda(group_name: str):
                is_lora = group_name in lora_group_names
                return lambda step: (
                    0.5 * (1.0 - math.cos(math.pi * min(step, warmup_steps) / warmup_steps))
                    if is_lora else 1.0
                )

            lambdas = [
                make_lambda(g.get("name", "")) for g in optimizer.param_groups
            ]
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
            log.info("M1: Cosine warmup scheduler enabled (%d steps)", warmup_steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return optimizer


class ProgressiveLoRACallback:
    """Lightning callback for progressive LoRA expansion across self-training rounds.

    Detects round boundaries via global_step and triggers rank expansion
    and/or layer coverage expansion. Inspired by Filatov & Kindulov (2023)
    who showed progressive rank/coverage across self-training rounds
    stabilizes training and beats EMA momentum networks.

    Usage: Add to trainer callbacks when LORA.PROGRESSIVE.ENABLED is True.

    Args:
        config: Full training config (needs SELF_TRAINING.ROUND_STEPS,
            MODEL.LORA.PROGRESSIVE.*).
        variant: LoRA variant ("conv_dora", "dora", "lora").
    """

    def __init__(self, config: CfgNode, variant: str = "conv_dora") -> None:
        self.round_steps = config.SELF_TRAINING.ROUND_STEPS
        self.variant = variant
        self._current_round = 0  # 0-based

        prog = config.MODEL.LORA.PROGRESSIVE
        from cups.model.lora import MitigationConfig, ProgressiveDoRAConfig

        self.prog_config = ProgressiveDoRAConfig(
            ranks=tuple(prog.RANKS),
            alphas=tuple(prog.ALPHAS),
            late_block_starts=tuple(prog.COVERAGES),
            variant=variant,
            dropout=config.MODEL.LORA.DROPOUT,
            lr_a=config.MODEL.LORA.LR_A,
            lr_b=config.MODEL.LORA.LR_B,
            magnitude_wd=config.MODEL.LORA.MAGNITUDE_WD,
        )

        # Load mitigation config
        if hasattr(config.MODEL.LORA, "MITIGATIONS"):
            self.mitigation_cfg = MitigationConfig.from_cfg(config)
        else:
            self.mitigation_cfg = MitigationConfig()

        # M2: Magnitude warmup state
        self._mag_frozen_at: int | None = None  # step when magnitude was frozen

        # M4: SWA state
        self._swa_accumulator = None
        if self.mitigation_cfg.swa_enabled:
            from cups.model.lora import SWAAccumulator
            self._swa_accumulator = SWAAccumulator()

        log.info(
            "ProgressiveLoRACallback: %d rounds, ranks=%s, coverages=%s",
            self.prog_config.num_rounds,
            self.prog_config.ranks,
            self.prog_config.late_block_starts,
        )

    def _freeze_magnitudes(self, model: nn.Module) -> int:
        """M2: Freeze all lora_magnitude params. Returns count."""
        from cups.model.lora import DoRALinear
        count = 0
        for module in model.modules():
            if isinstance(module, DoRALinear):
                module.lora_magnitude.requires_grad_(False)
                count += 1
        return count

    def _unfreeze_magnitudes(self, model: nn.Module) -> int:
        """M2: Unfreeze all lora_magnitude params. Returns count."""
        from cups.model.lora import DoRALinear
        count = 0
        for module in model.modules():
            if isinstance(module, DoRALinear):
                module.lora_magnitude.requires_grad_(True)
                count += 1
        return count

    def on_train_batch_start(
        self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int,
    ) -> None:
        """Check for round boundary, handle M2/M4 mitigations, expand LoRA."""
        step = trainer.global_step

        # --- M2: Magnitude warmup — unfreeze after N steps ---
        if (
            self.mitigation_cfg.magnitude_warmup_enabled
            and self._mag_frozen_at is not None
        ):
            elapsed = step - self._mag_frozen_at
            if elapsed >= self.mitigation_cfg.magnitude_warmup_freeze_steps:
                n_unfrozen = self._unfreeze_magnitudes(pl_module.model)
                log.info(
                    "M2: Unfroze %d magnitude vectors at step %d (frozen for %d steps)",
                    n_unfrozen, step, elapsed,
                )
                self._mag_frozen_at = None

        # --- M4: SWA accumulation during last fraction of round ---
        if self._swa_accumulator is not None:
            steps_in_round = step - (self._current_round * self.round_steps)
            swa_start = int(self.round_steps * (1.0 - self.mitigation_cfg.swa_fraction))
            if steps_in_round >= swa_start:
                backbone = self._find_vit_backbone(pl_module.model)
                if backbone is not None:
                    self._swa_accumulator.update(backbone)

        # --- Round boundary detection ---
        new_round = min(step // self.round_steps, self.prog_config.num_rounds - 1)

        if new_round <= self._current_round:
            return  # no round change

        from cups.model.lora import (
            expand_lora_rank,
            expand_lora_coverage,
        )

        old_round = self._current_round
        self._current_round = new_round
        old_cfg = self.prog_config.get_dora_config(old_round)
        new_cfg = self.prog_config.get_dora_config(new_round)

        # Find the backbone (ViT) inside the model
        backbone = self._find_vit_backbone(pl_module.model)
        if backbone is None:
            log.warning("ProgressiveLoRACallback: cannot find ViT backbone.")
            return

        log.info(
            "=== Progressive LoRA: round %d → %d (step %d) ===",
            old_round + 1, new_round + 1, step,
        )

        # M4: Apply SWA BEFORE rank expansion (critical ordering)
        if self._swa_accumulator is not None and self._swa_accumulator.count > 0:
            self._swa_accumulator.apply(backbone)
            self._swa_accumulator.reset()

        # Step 1: Expand rank if needed
        if new_cfg.rank > old_cfg.rank:
            n_expanded = expand_lora_rank(
                backbone, new_rank=new_cfg.rank, new_alpha=new_cfg.alpha,
            )
            log.info("  Rank: %d → %d (%d adapters)", old_cfg.rank, new_cfg.rank, n_expanded)

        # Step 2: Expand layer coverage if needed
        if new_cfg.late_block_start < old_cfg.late_block_start:
            newly = expand_lora_coverage(
                backbone, new_cfg, variant=self.variant,
            )
            log.info("  Coverage: late_block_start %d → %d (+%d layers)",
                     old_cfg.late_block_start, new_cfg.late_block_start, len(newly))

        # Step 3: Rebuild optimizer to include new parameters
        self._rebuild_optimizer(trainer, pl_module)

        # M2: Freeze magnitudes at start of new round
        if self.mitigation_cfg.magnitude_warmup_enabled:
            n_frozen = self._freeze_magnitudes(pl_module.model)
            self._mag_frozen_at = step
            log.info(
                "M2: Froze %d magnitude vectors at round %d start (step %d, will unfreeze after %d steps)",
                n_frozen, new_round + 1, step,
                self.mitigation_cfg.magnitude_warmup_freeze_steps,
            )

    @staticmethod
    def _find_vit_backbone(model: nn.Module) -> nn.Module | None:
        """Walk model hierarchy to find the ViT with .blocks attribute."""
        # Common path: model.backbone.net.vit (Detectron2 + SimpleFeaturePyramid)
        for attr_path in [
            ["backbone", "net", "vit"],
            ["backbone", "bottom_up", "net", "vit"],
            ["backbone", "vit"],
            ["vit"],
        ]:
            obj = model
            for attr in attr_path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "blocks"):
                return obj
        # Fallback: walk all modules
        for m in model.modules():
            if hasattr(m, "blocks") and isinstance(getattr(m, "blocks"), nn.ModuleList):
                return m
        return None

    @staticmethod
    def _rebuild_optimizer(trainer: Any, pl_module: Any) -> None:
        """Rebuild optimizer (+ optional scheduler) for new LoRA parameters."""
        result = pl_module.configure_optimizers()
        if isinstance(result, dict):
            trainer.optimizers = [result["optimizer"]]
            # Also rebuild scheduler if M1 cosine warmup is active
            if "lr_scheduler" in result:
                sched_cfg = result["lr_scheduler"]
                trainer.lr_scheduler_configs = [sched_cfg]
                log.info("  Optimizer + scheduler rebuilt with updated parameter groups")
            else:
                log.info("  Optimizer rebuilt with updated parameter groups")
        else:
            trainer.optimizers = [result]
            log.info("  Optimizer rebuilt with updated parameter groups")


class AdaptiveDelayedStartCallback:
    """M6: Activate LoRA when head loss converges rather than at a fixed step.

    Tracks an EMA of the training loss. When the smoothed loss drops below
    tau * initial_loss, LoRA parameters are unfrozen. Falls back to unfreezing
    at max_wait_steps if the threshold is never reached.

    Args:
        config: Full training config with MODEL.LORA.MITIGATIONS.ADAPTIVE_DELAYED_START.
    """

    def __init__(self, config: CfgNode) -> None:
        m6 = config.MODEL.LORA.MITIGATIONS.ADAPTIVE_DELAYED_START
        self._tau: float = m6.TAU
        self._max_wait: int = m6.MAX_WAIT_STEPS
        self._initial_loss: float | None = None
        self._loss_ema: float | None = None
        self._ema_alpha: float = 0.95  # smoothing factor
        self._lora_active: bool = False

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int,
    ) -> None:
        """Check loss convergence and activate LoRA if threshold met."""
        if self._lora_active:
            return

        step = trainer.global_step
        current_loss = trainer.callback_metrics.get("loss", None)
        if current_loss is None:
            return
        current_loss = float(current_loss)

        # Initialize on first step
        if self._initial_loss is None:
            self._initial_loss = current_loss
            self._loss_ema = current_loss

        # Update EMA
        self._loss_ema = (
            self._ema_alpha * self._loss_ema + (1 - self._ema_alpha) * current_loss
        )

        # Check activation conditions
        threshold_met = self._loss_ema < self._tau * self._initial_loss
        max_wait_met = step >= self._max_wait

        if threshold_met or max_wait_met:
            self._activate_lora(pl_module.model)
            self._lora_active = True
            reason = (
                f"loss EMA {self._loss_ema:.4f} < {self._tau} * {self._initial_loss:.4f}"
                if threshold_met else f"max_wait {self._max_wait} steps reached"
            )
            log.info("M6: LoRA activated at step %d (%s)", step, reason)

    @staticmethod
    def _activate_lora(model: nn.Module) -> int:
        """Unfreeze all LoRA adapter parameters. Returns count."""
        from cups.model.lora import DoRALinear, LoRALinear
        count = 0
        for module in model.modules():
            if isinstance(module, (DoRALinear, LoRALinear)):
                for param in module.parameters():
                    if not param.requires_grad:
                        param.requires_grad_(True)
                        count += 1
        log.info("M6: Unfroze %d LoRA parameters", count)
        return count


def build_model_self(
    config: CfgNode,
    thing_pseudo_classes: Tuple[int, ...] | None,
    stuff_pseudo_classes: Tuple[int, ...] | None,
    thing_classes: Set[int],
    stuff_classes: Set[int],
    copy_paste_augmentation: nn.Module = nn.Identity(),
    photometric_augmentation: nn.Module = nn.Identity(),
    resolution_jitter_augmentation: nn.Module = nn.Identity(),
    class_weights: Tuple[float, ...] | None = None,
    class_names: List[str] | None = None,
    classes_mask: List[bool] | None = None,
    freeze_bn: bool = True,
    mask_refiner: Any = None,
) -> SelfSupervisedModel:
    """Function to build the model.

    Args:
        config (CfgNode): Config object.
        thing_pseudo_classes (int): Estimated pseudo thing classes.
        stuff_pseudo_classes (int): Estimated stuff thing classes.
        copy_paste_augmentation (nn.Module): Copy-paste augmentation module.
        class_weights (Tuple[float, ...] | None): Semantic class weight. Default None.
        freeze_bn (bool): If true BN layers are frozen.

    Returns:
        model (UnsupervisedTrainer): Unsupervised trainer.
    """
    # Check parameters
    if thing_pseudo_classes is None or stuff_pseudo_classes is None:
        assert config.MODEL.CHECKPOINT is not None, "If thing stuff split is not given checkpoint needs the be given."
    # Load checkpoint if utilized
    if config.MODEL.CHECKPOINT is not None:
        checkpoint_raw = torch.load(config.MODEL.CHECKPOINT, map_location="cpu", weights_only=False)
        hp = checkpoint_raw.get("hyper_parameters", {}) if isinstance(checkpoint_raw, dict) else {}
        if thing_pseudo_classes is None and "thing_pseudo_classes" in hp:
            thing_pseudo_classes = tuple(hp["thing_pseudo_classes"])
        if stuff_pseudo_classes is None and "stuff_pseudo_classes" in hp:
            stuff_pseudo_classes = tuple(hp["stuff_pseudo_classes"])
        checkpoint = checkpoint_raw
        # Case if we have a lighting checkpoint
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]
            checkpoint = {key.replace("model.", ""): item for key, item in checkpoint.items()}
        # Case if we have a Detectron2 (U2Seg) checkpoint
        else:
            checkpoint = checkpoint["model"]
        # Get number of classes based on model weights
        num_clusters_stuffs: int = int(checkpoint["sem_seg_head.predictor.bias"].shape[0] - 1)
        num_clusters_things: int = int(checkpoint["roi_heads.mask_head.predictor.bias"].shape[0])
    else:
        num_clusters_things = len(thing_pseudo_classes)  # type: ignore
        num_clusters_stuffs = len(stuff_pseudo_classes)  # type: ignore
    stage4_cfg = getattr(config, "STAGE4", None)
    stage4_ids = resolve_stage4_class_ids(config, thing_pseudo_classes, stuff_pseudo_classes)
    # Init model — route based on backbone type
    backbone_type = getattr(config.MODEL, "BACKBONE_TYPE", "resnet50")
    if backbone_type == "dinov2_vitb":
        from cups.model.model_vitb import panoptic_cascade_mask_r_cnn_vitb
        log.info("Self-training: Using DINOv2 ViT-B/14 backbone")
        model: nn.Module = panoptic_cascade_mask_r_cnn_vitb(
            num_clusters_things=num_clusters_things,
            num_clusters_stuffs=num_clusters_stuffs + 1,
            confidence_threshold=config.MODEL.INFERENCE_CONFIDENCE_THRESHOLD,
            tta_detection_threshold=config.MODEL.TTA_INFERENCE_CONFIDENCE_THRESHOLD,
            class_weights=class_weights,
            use_tta=True,
            tta_scales=config.MODEL.TTA_SCALES,
            default_size=config.DATA.CROP_RESOLUTION,
            drop_loss_iou_threshold=config.TRAINING.DROP_LOSS_IOU_THRESHOLD,
            use_drop_loss=config.SELF_TRAINING.USE_DROP_LOSS,
            freeze_backbone=getattr(config.MODEL, "DINOV2_FREEZE", True),
            stage4_cfg=stage4_cfg,
            stage4_ids=stage4_ids,
        )
    elif backbone_type == "dinov3_vitb":
        from cups.model.model_vitb import panoptic_cascade_mask_r_cnn_dinov3
        log.info("Self-training: Using DINOv3 ViT-B/16 backbone")
        # Build WITHOUT TTA first so checkpoint can be loaded into the raw model.
        # The TTA wrapper's cfg has ResNet config which causes key mismatches.
        model: nn.Module = panoptic_cascade_mask_r_cnn_dinov3(
            num_clusters_things=num_clusters_things,
            num_clusters_stuffs=num_clusters_stuffs + 1,
            confidence_threshold=config.MODEL.INFERENCE_CONFIDENCE_THRESHOLD,
            tta_detection_threshold=config.MODEL.TTA_INFERENCE_CONFIDENCE_THRESHOLD,
            class_weights=class_weights,
            use_tta=False,
            tta_scales=config.MODEL.TTA_SCALES,
            default_size=config.DATA.CROP_RESOLUTION,
            drop_loss_iou_threshold=config.TRAINING.DROP_LOSS_IOU_THRESHOLD,
            use_drop_loss=config.SELF_TRAINING.USE_DROP_LOSS,
            freeze_backbone=getattr(config.MODEL, "DINOV2_FREEZE", True),
            stuff_kd_weight=getattr(config.MODEL.SEM_SEG_HEAD, "STUFF_KD_WEIGHT", 0.0),
            kd_temperature=getattr(config.MODEL.SEM_SEG_HEAD, "KD_TEMPERATURE", 2.0),
            sem_seg_head_name=(
                "DepthFiLMSemSegHead"
                if getattr(config.MODEL.SEM_SEG_HEAD, "USE_DEPTH_FILM", False)
                else "CustomSemSegFPNHead"
            ),
            depth_channels=getattr(config.MODEL.SEM_SEG_HEAD, "DEPTH_CHANNELS", 15),
            stage4_cfg=stage4_cfg,
            stage4_ids=stage4_ids,
            depth_dice_weight=getattr(
                getattr(config.MODEL, "ROI_MASK_HEAD", None), "DEPTH_DICE_WEIGHT", 0.0
            ),
            sam3_mask_adapter=getattr(
                getattr(config.MODEL, "ROI_BOX_HEAD", None), "SAM3_MASK_ADAPTER", False
            ),
            sam3_adapter_dim=getattr(
                getattr(config.MODEL, "ROI_BOX_HEAD", None), "SAM3_ADAPTER_DIM", 256
            ),
            sam3_n_max_masks=getattr(
                getattr(config.MODEL, "ROI_BOX_HEAD", None), "SAM3_N_MAX_MASKS", 20
            ),
            sam3_masks_dir=getattr(
                getattr(config.MODEL, "ROI_BOX_HEAD", None), "SAM3_MASKS_DIR", ""
            ),
        )
        # Load checkpoint BEFORE TTA wrapping
        if config.MODEL.CHECKPOINT is not None:
            log.info(f"Loading checkpoint into DINOv3 model: {config.MODEL.CHECKPOINT}")
            # Filter out teacher keys from self-training checkpoints
            student_checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith("teacher_")}
            if len(student_checkpoint) < len(checkpoint):
                log.info(f"Filtered {len(checkpoint) - len(student_checkpoint)} teacher keys from checkpoint")
            missing, unexpected = model.load_state_dict(student_checkpoint, strict=False)
            if unexpected:
                log.info(f"Ignoring {len(unexpected)} unexpected keys (e.g. seesaw_loss buffers): {unexpected[:3]}")
        # Now wrap with TTA
        from detectron2.config import get_cfg as _get_cfg
        from cups.model.modeling.meta_arch.panoptic_fpn_tta import PanopticFPNWithTTA
        _cfg = _get_cfg()
        _cfg.set_new_allowed(True)
        import os
        _cfg.merge_from_file(os.path.join(os.path.dirname(__file__), "model", "Panoptic-Cascade-Mask-R-CNN.yaml"))
        _cfg.TEST.AUG.MIN_SIZES = tuple(
            int(config.DATA.CROP_RESOLUTION[0] * s) for s in config.MODEL.TTA_SCALES
        )
        _cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.MODEL.INFERENCE_CONFIDENCE_THRESHOLD
        _cfg.TEST.INSTANCE_SCORE_THRESH = config.MODEL.TTA_INFERENCE_CONFIDENCE_THRESHOLD
        _cfg.freeze()
        model = PanopticFPNWithTTA(_cfg, model)
        log.info("TTA model wrapped after checkpoint loading.")
    else:
        model: nn.Module = panoptic_cascade_mask_r_cnn(
            load_dino=config.MODEL.USE_DINO,
            num_clusters_things=num_clusters_things,
            num_clusters_stuffs=num_clusters_stuffs + 1,  # Stuff classes plus single thing classes
            confidence_threshold=config.MODEL.INFERENCE_CONFIDENCE_THRESHOLD,
            class_weights=class_weights,
            use_tta=True,
            tta_detection_threshold=config.MODEL.TTA_INFERENCE_CONFIDENCE_THRESHOLD,
            use_drop_loss=config.SELF_TRAINING.USE_DROP_LOSS,
            tta_scales=config.MODEL.TTA_SCALES,
            default_size=config.DATA.CROP_RESOLUTION,
            drop_loss_iou_threshold=config.TRAINING.DROP_LOSS_IOU_THRESHOLD,
            stage4_cfg=stage4_cfg,
            stage4_ids=stage4_ids,
        )
    # Apply checkpoint (skip for dinov3_vitb — already loaded above before TTA wrapping)
    if config.MODEL.CHECKPOINT is not None and backbone_type != "dinov3_vitb":
        # Filter out teacher keys from self-training checkpoints
        student_checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith("teacher_")}
        if len(student_checkpoint) < len(checkpoint):
            log.info(f"Filtered {len(checkpoint) - len(student_checkpoint)} teacher keys from checkpoint")
        log.info(f"Checkpoint loaded from {config.MODEL.CHECKPOINT}.")
        model.model.load_state_dict(student_checkpoint)  # type: ignore
    # Freeze BN layer
    if freeze_bn:
        log.info("Freeze batch norm layers")
        model = FrozenBatchNorm2d.convert_frozen_batchnorm(model)
    # Init trainer
    model: SelfSupervisedModel = SelfSupervisedModel(
        model=model,
        num_thing_pseudo_classes=num_clusters_things,
        num_stuff_pseudo_classes=num_clusters_stuffs,
        config=config,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        copy_paste_augmentation=copy_paste_augmentation,
        photometric_augmentation=photometric_augmentation,
        resolution_jitter_augmentation=resolution_jitter_augmentation,
        class_names=class_names,
        classes_mask=classes_mask,
        mask_refiner=mask_refiner,
    )
    return model
