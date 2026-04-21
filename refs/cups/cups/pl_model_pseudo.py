from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

import torch.nn as nn
import torch.optim
import torchvision.transforms as tf
from detectron2.utils.events import EventStorage
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm, rank_zero_only
from torch import Tensor
from torchvision.ops._utils import split_normalization_params
from yacs.config import CfgNode

from cups.metrics import PanopticQualitySemanticMatching
from cups.model import (
    filter_predictions,
    panoptic_cascade_mask_r_cnn,
    panoptic_cascade_mask_r_cnn_vitb,
    prediction_to_label_format,
    prediction_to_standard_format,
)
from cups.model.modeling.mask2former.ema import EMAModel
from cups.model.modeling.mask2former.swa import average_state_dicts
from cups.visualization import (
    save_image,
    save_object_proposals,
    save_panoptic_segmentation,
    save_panoptic_segmentation_overlay,
    save_semantic_segmentation,
)

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class UnsupervisedModel(LightningModule):
    """This class implements the unsupervised model for training a Panoptic Cascade Mask R-CNN."""

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
        """
        # Call super constructor
        super(UnsupervisedModel, self).__init__()
        # Save model
        self.model: nn.Module = model
        # Save augmentation modules
        self.copy_paste_augmentation: nn.Module = copy_paste_augmentation
        self.photometric_augmentation: nn.Module = photometric_augmentation
        self.resolution_jitter_augmentation: nn.Module = resolution_jitter_augmentation
        # Make thing and stuff classes
        classes = range(num_stuff_pseudo_classes + num_thing_pseudo_classes)
        stuff_pseudo_classes = tuple(classes[:num_stuff_pseudo_classes])
        thing_pseudo_classes = tuple(classes[num_stuff_pseudo_classes:])
        # Omit class names if needed
        if classes_mask is not None:
            class_names = [name for index, name in enumerate(class_names) if classes_mask[index]]  # type: ignore
        # Save hyperparameters
        self.save_hyperparameters(
            {
                "config": config,
                "thing_pseudo_classes": thing_pseudo_classes,
                "stuff_pseudo_classes": stuff_pseudo_classes,
                "class_names": class_names,
                "classes_mask": classes_mask,
            }
        )
        # config
        self.config = config
        # Init storage object
        self.storage: EventStorage | None = None
        # Init metrics
        self.panoptic_quality: PanopticQualitySemanticMatching = PanopticQualitySemanticMatching(
            things=thing_classes,
            stuffs=stuff_classes,
            num_clusters=len(thing_pseudo_classes) + len(stuff_pseudo_classes),
            things_prototype=set(thing_pseudo_classes) if config.VALIDATION.ADHERE_THING_STUFF else None,
            stuffs_prototype=set(stuff_pseudo_classes) if config.VALIDATION.ADHERE_THING_STUFF else None,
            cache_device=config.VALIDATION.CACHE_DEVICE,
            classes_mask=classes_mask,
            sync_on_compute=False,
            dist_sync_on_step=False,
        )
        # Set parameters for second validation
        self.plot_validation_samples: bool = False
        self.assignments: Tensor | None = None
        # Set parameters for copy-paste augmentation
        self.prediction_temp: Dict | None = None
        # Get color template
        if config.DATA.DATASET == "cityscapes" and config.DATA.NUM_CLASSES == 27:
            self.color_template: str = "cityscapes"
        elif config.DATA.DATASET == "mots":
            self.color_template = "mots"
        else:
            self.color_template = "cityscapes_19"
        # --- EMA teacher (G1 / N4 / Stage-3) --------------------------------
        self._ema_enabled: bool = bool(getattr(getattr(config.MODEL, "EMA", None), "ENABLED", False))
        self._ema_decay: float = float(getattr(getattr(config.MODEL, "EMA", None), "DECAY", 0.9998))
        self.ema_teacher: EMAModel | None = EMAModel(self.model, decay=self._ema_decay) if self._ema_enabled else None
        self._ema_backup: Dict[str, Tensor] | None = None
        # --- SWA averaging (G2) ---------------------------------------------
        self._swa_enabled: bool = bool(getattr(getattr(config.MODEL, "SWA", None), "ENABLED", False))
        self._swa_num_ckpts: int = int(getattr(getattr(config.MODEL, "SWA", None), "NUM_CKPTS", 5))
        self._swa_start_fraction: float = float(getattr(getattr(config.MODEL, "SWA", None), "START_FRACTION", 0.75))
        self._swa_state_dicts: List[Dict[str, Tensor]] = []
        # --- N4 query-consistency: teacher forward cache --------------------
        self._query_consistency_weight: float = float(
            getattr(getattr(config.MODEL, "MASK2FORMER", None), "QUERY_CONSISTENCY_WEIGHT", 0.0)
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
        # Perform copy-paste augmentation
        if self.copy_paste_augmentation is not None:
            if self.global_step < self.hparams.config.AUGMENTATION.NUM_STEPS_STARTUP:
                batch_aug = self.copy_paste_augmentation(batch, deepcopy(batch))
            else:
                if self.prediction_temp is not None:
                    batch_aug = self.copy_paste_augmentation(self.prediction_temp, deepcopy(batch))
                else:
                    batch_aug = deepcopy(batch)
        else:
            batch_aug = deepcopy(batch)
        # Augmented batch
        augmented_batch = self.photometric_augmentation(self.resolution_jitter_augmentation(batch_aug))
        # --- N4: teacher forward to populate query embeddings on student model ---
        # The aux_loss_hooks["query_consistency"] reads ctx["teacher_query_embeds"]
        # via Mask2FormerPanoptic; we stash the EMA-teacher's pred_query_embeds on
        # the student model as _teacher_query_embeds so the hook can pull them.
        if self._query_consistency_weight > 0.0 and self.ema_teacher is not None:
            self._populate_teacher_query_embeds(augmented_batch)
        # Get losses
        loss_dict = self(augmented_batch)
        # Clear stashed teacher embeddings so they don't leak across steps.
        if hasattr(self.model, "_teacher_query_embeds"):
            self.model._teacher_query_embeds = None
        # Compute sum of losses
        loss: Tensor = sum(loss_dict.values())
        # Log final loss
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        # Log all losses
        for key, value in loss_dict.items():
            self.log("losses/" + key, value, sync_dist=True)
        # Make inference prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model([{"image": sample["image"]} for sample in batch])
            self.prediction_temp = prediction_to_label_format(  # type: ignore
                prediction,
                [sample["image"] for sample in batch],
                confidence_threshold=self.hparams.config.AUGMENTATION.CONFIDENCE,
            )
            self.prediction_temp = filter_predictions(self.prediction_temp, batch)  # type: ignore
        self.model.train()
        # Log media
        if ((self.global_step) % self.hparams.config.TRAINING.LOG_MEDIA_N_STEPS) == 0:
            self.log_visualizations(batch_aug, prediction)
        return {"loss": loss}

    @rank_zero_only
    def log_visualizations(self, batch: List[Dict[str, Any]], prediction: List[Dict[str, Any]]) -> None:
        """Logs visualizations.

        Args:
            batch (List[Dict[str, Any]])): Batch of training data.
            prediction (List[Dict[str, Any]]): Inference predictions.
        """
        # Get object proposals
        object_proposal_pseudo: Tensor = (
            (
                batch[0]["instances"].gt_masks.tensor.cpu().float()
                * torch.arange(1, batch[0]["instances"].gt_masks.tensor.shape[0] + 1).view(-1, 1, 1)
            )
            .sum(dim=0)
            .long()
        )
        # Get raw semantic prediction
        semantic_prediction: Tensor = prediction[0]["sem_seg"].argmax(dim=0)
        # Convert panoptic prediction to standard format
        panoptic_prediction: Tensor = prediction_to_standard_format(
            prediction[0]["panoptic_seg"],
            stuff_classes=self.hparams.stuff_pseudo_classes,
            thing_classes=self.hparams.thing_pseudo_classes,
        )
        # Log image
        self.logger.log_image(key="training_image", images=[save_image(batch[0]["image"].cpu(), path=None)])
        # Log training panoptic prediction
        self.logger.log_image(
            key="training_panoptic_prediction",
            images=[
                save_panoptic_segmentation_overlay(
                    panoptic_prediction.cpu(),
                    batch[0]["image"].cpu(),
                    path=None,
                    dataset="pseudo",
                    bounding_boxes=True,
                )
            ],
        )
        # Log raw semantic prediction
        self.logger.log_image(
            key="training_raw_semantic_predictions",
            images=[
                save_semantic_segmentation(
                    semantic_prediction.cpu(),
                    path=None,
                    dataset="pseudo",
                )
            ],
        )
        # Log pseudo label
        self.logger.log_image(
            key="training_semantic_pseudo_label",
            images=[
                save_semantic_segmentation(
                    batch[0]["sem_seg"].cpu(),
                    path=None,
                    dataset="pseudo",
                )
            ],
        )
        # Log object proposals
        self.logger.log_image(
            key="training_object_proposal_pseudo_label",
            images=[
                save_object_proposals(
                    object_proposal_pseudo,
                    path=None,
                )
            ],
        )

    def validation_step(self, batch: Tuple[List[Dict[str, Tensor]], Tensor, List[str]], batch_index: int) -> None:
        """Validation step.

        Args:
            batch (Tuple[List[Dict[str, Tensor]], Tensor, List[str]]): Batch of training data.
            batch_index (int): Batch index.
        """
        # Get data
        images, panoptic_labels, image_names = batch
        # Semantic segmentation eval resize
        # # Make prediction
        prediction = self(images)
        # Convert to standard panoptic format
        panoptic_predictions: Tensor = torch.stack(
            [
                prediction_to_standard_format(
                    prediction[index]["panoptic_seg"],
                    stuff_classes=self.hparams.stuff_pseudo_classes,
                    thing_classes=self.hparams.thing_pseudo_classes,
                )
                for index in range(len(prediction))
            ],
            dim=0,
        )
        # We plot the samples if we have the assignments and plotting is enabled
        if self.assignments is not None and self.plot_validation_samples:
            # Remap prediction using assignments
            panoptic_predictions_remapped: Tensor = self.panoptic_quality.map_to_target(
                panoptic_predictions, self.assignments
            )
            for index, prediction in enumerate(panoptic_predictions_remapped):
                self.logger.log_image(
                    key="validation_prediction",
                    images=[save_panoptic_segmentation(prediction, path=None, dataset=self.color_template)],
                )
                save_panoptic_segmentation(
                    prediction,
                    path=os.path.join(self.logger.save_dir, image_names[index] + "_prediction.png"),
                    dataset=self.color_template,
                )
        # Update metric
        if self.config.VALIDATION.SEMSEG_CENTER_CROP_SIZE is not None:
            # Apply resize and center crop for semantic segmentation evaluation
            size = self.config.VALIDATION.SEMSEG_CENTER_CROP_SIZE
            resized_sem_pred = tf.CenterCrop(size)(
                tf.Resize(size, interpolation=tf.InterpolationMode.NEAREST)(panoptic_predictions[..., 0])
            )
            resized_sem_label = tf.CenterCrop(size)(
                tf.Resize(size, interpolation=tf.InterpolationMode.NEAREST)(panoptic_labels[..., 0])
            )
            zeros = torch.zeros_like(resized_sem_pred)
            resized_sem_pred = torch.stack((resized_sem_pred, zeros), dim=-1)
            resized_sem_label = torch.stack((resized_sem_label, zeros), dim=-1)
            self.panoptic_quality.update(resized_sem_pred, resized_sem_label)
        else:
            self.panoptic_quality.update(panoptic_predictions, panoptic_labels)

    def on_validation_epoch_end(self) -> None:
        """Accumulate metric after validation."""
        # Compute metrics
        (
            pq,
            sq,
            rq,
            pq_c,
            sq_c,
            rq_c,
            pq_t,
            sq_t,
            rq_t,
            pq_s,
            sq_s,
            rq_s,
            miou,
            acc,
            assignments,
            predictions,
        ) = self.panoptic_quality.compute()
        # Save assignments
        self.assignments = assignments
        # Log metrics (sync_dist=False: PQ metric caches on CPU, NCCL can't reduce CPU tensors)
        self.log("pq_val", pq, rank_zero_only=True, sync_dist=False)
        self.log("sq_val", sq, rank_zero_only=True, sync_dist=False)
        self.log("rq_val", rq, rank_zero_only=True, sync_dist=False)
        self.log("pq_t_val", pq_t, rank_zero_only=True, sync_dist=False)
        self.log("sq_t_val", sq_t, rank_zero_only=True, sync_dist=False)
        self.log("rq_t_val", rq_t, rank_zero_only=True, sync_dist=False)
        self.log("pq_s_val", pq_s, rank_zero_only=True, sync_dist=False)
        self.log("sq_s_val", sq_s, rank_zero_only=True, sync_dist=False)
        self.log("rq_s_val", rq_s, rank_zero_only=True, sync_dist=False)
        self.log("miou_val", miou, rank_zero_only=True, sync_dist=False)
        self.log("acc_val", acc, rank_zero_only=True, sync_dist=False)
        # Log class-wise metrics
        for index in range(pq_c.shape[0]):
            if self.hparams.class_names is None:
                self.log(f"val/pq_{str(index).zfill(2)}_val", pq_c[index], rank_zero_only=True, sync_dist=False)
                self.log(f"val/sq_{str(index).zfill(2)}_val", sq_c[index], rank_zero_only=True, sync_dist=False)
                self.log(f"val/rq_{str(index).zfill(2)}_val", rq_c[index], rank_zero_only=True, sync_dist=False)
            else:
                self.log(
                    f"val/pq_{self.hparams.class_names[index]}_val", pq_c[index], rank_zero_only=True, sync_dist=False
                )
                self.log(
                    f"val/sq_{self.hparams.class_names[index]}_val", sq_c[index], rank_zero_only=True, sync_dist=False
                )
                self.log(
                    f"val/rq_{self.hparams.class_names[index]}_val", rq_c[index], rank_zero_only=True, sync_dist=False
                )
        # Log final prediction (only for training and if we not already plot the samples)
        if not self.plot_validation_samples and self.logger is not None:
            self.logger.log_image(
                key="validation_prediction",
                images=[
                    save_panoptic_segmentation(sample.cpu(), path=None, dataset=self.color_template)
                    for sample in predictions
                ],
            )
        # print results to copy to excel
        pq = pq.item()
        sq = sq.item()
        rq = rq.item()
        pq_t = pq_t.item()
        sq_t = sq_t.item()
        rq_t = rq_t.item()
        pq_s = pq_s.item()
        sq_s = sq_s.item()
        rq_s = rq_s.item()
        acc = acc.item()
        miou = miou.item()
        print("\nPQ, SQ, RQ, PQ_things, SQ_things, RQ_things, PQ_stuffs, SQ_stuffs, RQ_stuffs, Acc, mIoU")
        print("; ".join(map(str, [pq, sq, rq, pq_t, sq_t, rq_t, pq_s, sq_s, rq_s, acc, miou])))
        # Reset metric
        self.panoptic_quality.reset()

    def on_validation_end(self) -> None:  # type: ignore[override]
        """Restore student weights after EMA validation (counterpart to on_validation_epoch_start)."""
        if self.ema_teacher is not None and self._ema_backup is not None:
            for name, p in self.model.named_parameters():
                if name in self._ema_backup:
                    p.data.copy_(self._ema_backup[name])
            self._ema_backup = None

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Things to do before the optimizer step.

        Args:
            optimizer (torch.optim.Optimizer): Unused.
        """
        # Compute gradient norms
        gradient_norms = grad_norm(self.model, norm_type=2)
        # Log gradient norms
        self.log_dict(gradient_norms)

    @torch.no_grad()
    def _populate_teacher_query_embeds(self, augmented_batch: List[Dict[str, Any]]) -> None:
        """Run the EMA teacher (weights swapped in) and stash pred_query_embeds on
        the student so the N4 query_consistency hook can read them via ctx.

        Bypasses the meta-arch's panoptic-merge eval path to capture the raw
        decoder dict (which contains pred_query_embeds when requested).
        """
        assert self.ema_teacher is not None
        # Requires Mask2FormerPanoptic-style meta-arch (backbone/pixel_decoder/transformer_decoder).
        if not all(hasattr(self.model, attr) for attr in ("backbone", "pixel_decoder", "transformer_decoder")):
            return
        # Swap EMA shadow into the student temporarily for the teacher forward.
        backup = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
            if name in self.ema_teacher.shadow
        }
        self.ema_teacher.copy_to(self.model)
        was_training = self.model.training
        self.model.eval()
        try:
            # Use the meta-arch's own stacking helpers so shapes match.
            images = self.model._stack_images(augmented_batch)  # (B, 3, H, W)
            depth = self.model._maybe_depth(augmented_batch)
            feats = self.model.backbone(images)
            mask_feat, multi_scale = self.model.pixel_decoder(feats)
            teacher_dec_out = self.model.transformer_decoder(
                mask_feat=mask_feat,
                multi_scale=multi_scale,
                depth=depth,
                return_query_embeds=True,
            )
            # MaskedAttentionDecoder returns {"pred_logits", "pred_masks", "aux_outputs", "query_embeds"}.
            teacher_embeds: Tensor | None = teacher_dec_out.get("query_embeds") if isinstance(teacher_dec_out, dict) else None
            self.model._teacher_query_embeds = teacher_embeds
        finally:
            # Restore student weights + mode.
            for name, p in self.model.named_parameters():
                if name in backup:
                    p.data.copy_(backup[name])
            if was_training:
                self.model.train()

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:  # type: ignore[override]
        """EMA update + SWA buffering after every optimizer step."""
        # EMA teacher: update after each student step
        if self.ema_teacher is not None:
            self.ema_teacher.update(self.model)
        # SWA buffering: save N evenly-spaced snapshots in the last START_FRACTION window
        if self._swa_enabled:
            total_steps = self._estimate_total_steps()
            start_step = int(self._swa_start_fraction * total_steps) if total_steps > 0 else 0
            if total_steps > 0 and self.global_step >= start_step and self._swa_num_ckpts > 0:
                remaining = max(1, total_steps - start_step)
                stride = max(1, remaining // self._swa_num_ckpts)
                if ((self.global_step - start_step) % stride == 0
                        and len(self._swa_state_dicts) < self._swa_num_ckpts):
                    self._swa_state_dicts.append(
                        {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}
                    )
                    log.info(
                        f"[SWA] cached ckpt {len(self._swa_state_dicts)}/{self._swa_num_ckpts} "
                        f"at step {self.global_step}"
                    )

    def on_train_end(self) -> None:  # type: ignore[override]
        """Average SWA buffered state dicts and save; no-op otherwise."""
        if self._swa_enabled and self._swa_state_dicts:
            avg = average_state_dicts(self._swa_state_dicts)
            save_dir = getattr(self.logger, "save_dir", None) if self.logger is not None else None
            out_dir = save_dir or getattr(self.hparams.config.SYSTEM, "LOG_PATH", ".")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "swa_averaged.ckpt")
            torch.save({"state_dict": {f"model.{k}": v for k, v in avg.items()}}, out_path)
            log.info(f"[SWA] wrote averaged weights ({len(self._swa_state_dicts)} ckpts) to {out_path}")

    def on_validation_epoch_start(self) -> None:  # type: ignore[override]
        """Swap in EMA teacher weights for validation (G1 / N4 path)."""
        if self.ema_teacher is not None:
            self._ema_backup = {
                name: p.detach().clone() for name, p in self.model.named_parameters()
                if name in self.ema_teacher.shadow
            }
            self.ema_teacher.copy_to(self.model)

    def _estimate_total_steps(self) -> int:
        """Best-effort estimate of total optimizer steps for SWA scheduling."""
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return 0
        max_steps = getattr(trainer, "max_steps", -1)
        if max_steps is not None and max_steps > 0:
            return int(max_steps)
        # Fallback: derive from max_epochs × estimated_stepping_batches
        est = getattr(trainer, "estimated_stepping_batches", None)
        if est is not None and est > 0:
            return int(est)
        return 0

    def on_train_epoch_end(self) -> None:
        """Stuff to perform at the end of the epoch."""
        # Just close the storage object
        self.storage.__exit__(None, None, None)  # type: ignore
        # Set storage to Nona
        self.storage = None

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Builds the models' optimizer.

        When DoRA/LoRA is enabled, uses differential learning rates
        (LoRA+, Hayou et al., ICML 2024) with 5 parameter groups:
            1. Head (non-norm) — head_lr, head_wd
            2. Head (norm)     — head_lr, 0.0
            3. DoRA B          — lr_b, 0.0
            4. DoRA magnitude  — lr_b, magnitude_wd
            5. DoRA A          — lr_a, 0.0

        Returns:
            optimizer (torch.optim.Optimizer): Optimizer of the model.
        """
        # Check if DoRA/LoRA is active
        lora_enabled = (
            hasattr(self.hparams.config.MODEL, "LORA")
            and getattr(self.hparams.config.MODEL.LORA, "ENABLED", False)
        )

        if lora_enabled and self.hparams.config.TRAINING.OPTIMIZER != "sgd":
            # Use DoRA-aware param groups with differential LR
            from cups.model.lora import DoRAConfig, get_dora_param_groups

            lora_section = self.hparams.config.MODEL.LORA
            dora_cfg = DoRAConfig(
                rank=getattr(lora_section, "RANK", 4),
                alpha=getattr(lora_section, "ALPHA", 4.0),
                dropout=getattr(lora_section, "DROPOUT", 0.05),
                late_block_start=getattr(lora_section, "LATE_BLOCK_START", 6),
                lr_a=getattr(lora_section, "LR_A", 1e-5),
                lr_b=getattr(lora_section, "LR_B", 5e-5),
                magnitude_wd=getattr(lora_section, "MAGNITUDE_WD", 1e-3),
            )
            parameters = get_dora_param_groups(
                model=self.model,
                config=dora_cfg,
                head_lr=self.hparams.config.TRAINING.ADAMW.LEARNING_RATE,
                head_wd=self.hparams.config.TRAINING.ADAMW.WEIGHT_DECAY,
            )
            optimizer = torch.optim.AdamW(
                params=parameters,
                lr=self.hparams.config.TRAINING.ADAMW.LEARNING_RATE,
                betas=self.hparams.config.TRAINING.ADAMW.BETAS,
            )
            log.info("AdamW with DoRA differential LR (LoRA+) used.")
            return optimizer

        # Standard optimizer path (no LoRA)
        parameter_groups = split_normalization_params(self.model)
        if self.hparams.config.TRAINING.OPTIMIZER == "sgd":
            parameters = [
                {"params": parameter, "weight_decay": weight_decay}
                for parameter, weight_decay in zip(
                    parameter_groups, (0.0, self.hparams.config.TRAINING.SGD.WEIGHT_DECAY)
                )
                if parameter
            ]
            optimizer: torch.optim.Optimizer = torch.optim.SGD(
                params=parameters,
                lr=self.hparams.config.TRAINING.SGD.LEARNING_RATE,
                weight_decay=self.hparams.config.TRAINING.SGD.WEIGHT_DECAY,
                momentum=self.hparams.config.TRAINING.SGD.MOMENTUM,
            )
            log.info("SGD used.")
        else:
            parameters = [
                {"params": parameter, "weight_decay": weight_decay}
                for parameter, weight_decay in zip(
                    parameter_groups, (0.0, self.hparams.config.TRAINING.ADAMW.WEIGHT_DECAY)
                )
                if parameter
            ]
            optimizer = torch.optim.AdamW(
                params=parameters,
                lr=self.hparams.config.TRAINING.ADAMW.LEARNING_RATE,
                weight_decay=self.hparams.config.TRAINING.ADAMW.WEIGHT_DECAY,
                betas=self.hparams.config.TRAINING.ADAMW.BETAS,
            )
            log.info("AdamW used.")
        # Linear LR warm-up for Mask2Former cold-start. The M2F paper uses a
        # short linear ramp because 100 random-init queries otherwise produce
        # destabilising gradient spikes in the first few hundred steps. We
        # gate behind TRAINING.WARMUP_STEPS so pre-existing configs (Cascade
        # Mask R-CNN, LoRA experiments) keep their original no-scheduler
        # behaviour by default.
        warmup_steps = int(getattr(self.hparams.config.TRAINING, "WARMUP_STEPS", 0))
        if warmup_steps > 0:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps,
            )
            log.info("Linear LR warm-up: %d steps (start_factor=1e-3 -> 1.0).", warmup_steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }
        return optimizer


def _build_mask2former_model(
    config: CfgNode,
    num_stuff_classes: int | None = None,
    num_thing_classes: int | None = None,
    class_weights: Tuple[float, ...] | None = None,
):
    """Build Mask2Former panoptic model (Stage-2 M2F meta-arch).

    Deferred import avoids circular deps between ``pl_model_pseudo`` and
    ``cups.model.model_mask2former``.

    Args:
        config: Full CUPS config node.
        num_stuff_classes: Number of pseudo stuff classes (from dataloader).
        num_thing_classes: Number of pseudo thing classes (from dataloader).
        class_weights: Per-class CE weights of length num_stuff+num_thing.
            If None, SetCriterion uses uniform weights. If provided, rare
            thing classes get upweighted (essential for k=80 pseudo-labels
            where some thing classes have <0.1 instances per crop).

    Returns:
        Constructed ``Mask2FormerPanoptic`` ``nn.Module``.
    """
    from cups.model.model_mask2former import build_mask2former_vitb
    return build_mask2former_vitb(
        config,
        num_stuff_classes=num_stuff_classes,
        num_thing_classes=num_thing_classes,
        class_weights=class_weights,
    )


def build_model_pseudo(
    config: CfgNode,
    thing_pseudo_classes: Tuple[int, ...] | None = None,
    stuff_pseudo_classes: Tuple[int, ...] | None = None,
    thing_classes: Optional[Set[int]] = None,
    stuff_classes: Optional[Set[int]] = None,
    copy_paste_augmentation: Optional[nn.Module] = nn.Identity(),
    photometric_augmentation: nn.Module = nn.Identity(),
    resolution_jitter_augmentation: nn.Module = nn.Identity(),
    class_weights: Tuple[float, ...] | None = None,
    use_tta: bool = False,
    class_names: List[str] | None = None,
    classes_mask: List[bool] | None = None,
) -> UnsupervisedModel:
    """Function to build the model.

    Args:
        config (CfgNode): Config object.
        thing_pseudo_classes (int): Estimated pseudo thing classes.
        stuff_pseudo_classes (int): Estimated stuff thing classes.
        copy_paste_augmentation (nn.Module): Copy-paste augmentation module.
        class_weights (Tuple[float, ...] | None): Semantic class weight. Default None.
        use_tta (bool): If true TTA is used.
        class_to_name (List[str] | None): List containing the name of the semantic classes.
        classes_mask (List[bool] | None): Mask of valid classes in validation set.

    Returns:
        model (UnsupervisedTrainer): Unsupervised trainer.
    """
    # NEW branch: Mask2FormerPanoptic meta-arch.
    # Use bare-name ``_build_mask2former_model`` so ``monkeypatch.setattr`` on
    # the ``cups.pl_model_pseudo`` module attribute intercepts the call.
    if getattr(config.MODEL, "META_ARCH", "Cascade") == "Mask2FormerPanoptic":
        if thing_pseudo_classes is None or stuff_pseudo_classes is None:
            raise ValueError(
                "Mask2FormerPanoptic meta-arch requires thing_pseudo_classes "
                "and stuff_pseudo_classes from the training dataloader."
            )
        num_clusters_things = len(thing_pseudo_classes)
        num_clusters_stuffs = len(stuff_pseudo_classes)
        m2f_model = _build_mask2former_model(
            config,
            num_stuff_classes=num_clusters_stuffs,
            num_thing_classes=num_clusters_things,
            class_weights=class_weights,
        )
        return UnsupervisedModel(
            model=m2f_model,
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
        )
    # Check parameters
    if thing_pseudo_classes is None or stuff_pseudo_classes is None:
        assert config.MODEL.CHECKPOINT is not None, "If thing stuff split is not given checkpoint needs the be given."
    # Load checkpoint if utilized
    if config.MODEL.CHECKPOINT is not None:
        checkpoint = torch.load(config.MODEL.CHECKPOINT, map_location="cpu", weights_only=False)
        # Case if we have a lighting checkpoint
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]
            checkpoint = {
                key.replace("model.", ""): item
                for key, item in checkpoint.items()
                if "teacher_model." not in key and not key.startswith("teacher_")
            }
        # Case if we have a Detectron2 (U2Seg) checkpoint
        else:
            checkpoint = checkpoint["model"]
        # Get number of classes based on model weights
        num_clusters_stuffs: int = int(checkpoint["sem_seg_head.predictor.bias"].shape[0] - 1)
        num_clusters_things: int = int(checkpoint["roi_heads.mask_head.predictor.bias"].shape[0])
        # Log info about checkpoint
        log.info(f"Checkpoint loaded from {config.MODEL.CHECKPOINT}.")
    else:
        num_clusters_things = len(thing_pseudo_classes)  # type: ignore
        num_clusters_stuffs = len(stuff_pseudo_classes)  # type: ignore
    # Init model — route based on backbone type
    backbone_type = getattr(config.MODEL, "BACKBONE_TYPE", "resnet50")
    if backbone_type == "dinov2_vitb":
        log.info("Using DINOv2 ViT-B/14 backbone")
        model: nn.Module = panoptic_cascade_mask_r_cnn_vitb(
            num_clusters_things=num_clusters_things,
            num_clusters_stuffs=num_clusters_stuffs + 1,
            confidence_threshold=config.MODEL.INFERENCE_CONFIDENCE_THRESHOLD,
            tta_detection_threshold=config.MODEL.TTA_INFERENCE_CONFIDENCE_THRESHOLD,
            class_weights=class_weights,
            use_tta=use_tta,
            tta_scales=config.MODEL.TTA_SCALES,
            default_size=config.DATA.CROP_RESOLUTION,
            drop_loss_iou_threshold=config.TRAINING.DROP_LOSS_IOU_THRESHOLD,
            use_drop_loss=config.TRAINING.DROP_LOSS,
            freeze_backbone=getattr(config.MODEL, "DINOV2_FREEZE", True),
        )
    elif backbone_type == "dinov3_vitb":
        from cups.model.model_vitb import panoptic_cascade_mask_r_cnn_dinov3
        # Extract LoRA config if present
        lora_cfg = None
        if hasattr(config.MODEL, "LORA") and getattr(config.MODEL.LORA, "ENABLED", False):
            lora_cfg = {
                "VARIANT": getattr(config.MODEL.LORA, "VARIANT", "dora"),
                "RANK": getattr(config.MODEL.LORA, "RANK", 4),
                "ALPHA": getattr(config.MODEL.LORA, "ALPHA", 4.0),
                "DROPOUT": getattr(config.MODEL.LORA, "DROPOUT", 0.05),
                "LATE_BLOCK_START": getattr(config.MODEL.LORA, "LATE_BLOCK_START", 6),
            }
            log.info("DoRA/LoRA enabled: %s", lora_cfg)
        log.info("Using DINOv3 ViT-B/16 backbone")
        model: nn.Module = panoptic_cascade_mask_r_cnn_dinov3(
            num_clusters_things=num_clusters_things,
            num_clusters_stuffs=num_clusters_stuffs + 1,
            confidence_threshold=config.MODEL.INFERENCE_CONFIDENCE_THRESHOLD,
            tta_detection_threshold=config.MODEL.TTA_INFERENCE_CONFIDENCE_THRESHOLD,
            class_weights=class_weights,
            use_tta=use_tta,
            tta_scales=config.MODEL.TTA_SCALES,
            default_size=config.DATA.CROP_RESOLUTION,
            drop_loss_iou_threshold=config.TRAINING.DROP_LOSS_IOU_THRESHOLD,
            use_drop_loss=config.TRAINING.DROP_LOSS,
            freeze_backbone=getattr(config.MODEL, "DINOV2_FREEZE", True),
            lora_config=lora_cfg,
            stuff_kd_weight=getattr(config.MODEL.SEM_SEG_HEAD, "STUFF_KD_WEIGHT", 0.0),
            kd_temperature=getattr(config.MODEL.SEM_SEG_HEAD, "KD_TEMPERATURE", 2.0),
            sem_seg_head_name=(
                "DepthFiLMSemSegHead"
                if getattr(config.MODEL.SEM_SEG_HEAD, "USE_DEPTH_FILM", False)
                else "CustomSemSegFPNHead"
            ),
            depth_channels=getattr(config.MODEL.SEM_SEG_HEAD, "DEPTH_CHANNELS", 15),
        )
    elif backbone_type == "dinov3_vitl":
        from cups.model.model_vitb import panoptic_cascade_mask_r_cnn_dinov3_vitl
        # Extract LoRA config if present (same as ViT-B path)
        lora_cfg_l = None
        if hasattr(config.MODEL, "LORA") and getattr(config.MODEL.LORA, "ENABLED", False):
            lora_cfg_l = {
                "VARIANT": getattr(config.MODEL.LORA, "VARIANT", "dora"),
                "RANK": getattr(config.MODEL.LORA, "RANK", 4),
                "ALPHA": getattr(config.MODEL.LORA, "ALPHA", 4.0),
                "DROPOUT": getattr(config.MODEL.LORA, "DROPOUT", 0.05),
                "LATE_BLOCK_START": getattr(config.MODEL.LORA, "LATE_BLOCK_START", 6),
            }
        log.info("Using DINOv3 ViT-L/16 backbone")
        model: nn.Module = panoptic_cascade_mask_r_cnn_dinov3_vitl(
            num_clusters_things=num_clusters_things,
            num_clusters_stuffs=num_clusters_stuffs + 1,
            confidence_threshold=config.MODEL.INFERENCE_CONFIDENCE_THRESHOLD,
            tta_detection_threshold=config.MODEL.TTA_INFERENCE_CONFIDENCE_THRESHOLD,
            class_weights=class_weights,
            use_tta=use_tta,
            tta_scales=config.MODEL.TTA_SCALES,
            default_size=config.DATA.CROP_RESOLUTION,
            drop_loss_iou_threshold=config.TRAINING.DROP_LOSS_IOU_THRESHOLD,
            use_drop_loss=config.TRAINING.DROP_LOSS,
            freeze_backbone=getattr(config.MODEL, "DINOV2_FREEZE", True),
            lora_config=lora_cfg_l,
        )
    else:
        model: nn.Module = panoptic_cascade_mask_r_cnn(
            load_dino=config.MODEL.USE_DINO,
            num_clusters_things=num_clusters_things,
            num_clusters_stuffs=num_clusters_stuffs + 1,  # Stuff classes plus single thing classes
            confidence_threshold=config.MODEL.INFERENCE_CONFIDENCE_THRESHOLD,
            tta_detection_threshold=config.MODEL.TTA_INFERENCE_CONFIDENCE_THRESHOLD,
            class_weights=class_weights,
            use_tta=use_tta,
            tta_scales=config.MODEL.TTA_SCALES,
            default_size=config.DATA.CROP_RESOLUTION,
            drop_loss_iou_threshold=config.TRAINING.DROP_LOSS_IOU_THRESHOLD,
            use_drop_loss=config.TRAINING.DROP_LOSS,
        )
    # Apply checkpoint
    if config.MODEL.CHECKPOINT is not None:
        if use_tta:
            model.model.load_state_dict(checkpoint)  # type: ignore
        else:
            model.load_state_dict(checkpoint)
    # Init trainer
    model: UnsupervisedModel = UnsupervisedModel(
        model=model,
        num_thing_pseudo_classes=num_clusters_things,
        num_stuff_pseudo_classes=num_clusters_stuffs,
        config=config,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        copy_paste_augmentation=copy_paste_augmentation,  # type: ignore
        photometric_augmentation=photometric_augmentation,
        resolution_jitter_augmentation=resolution_jitter_augmentation,
        class_names=class_names,
        classes_mask=classes_mask,
    )
    return model
