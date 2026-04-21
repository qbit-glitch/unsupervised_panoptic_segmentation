from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Set, Tuple

import math

import torch.nn as nn
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
        # A2: Log per-class thresholding once
        self._logged_class_threshold: bool = False

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
        # Make pseudo labels
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
        # Train using self pseudo labels
        loss_dict = self.model(pseudo_labels)
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
            # AMR-ST: Asymmetric Multi-Round Self-Training
            st_cfg = self.hparams.config.SELF_TRAINING
            base_thresh = st_cfg.SEMANTIC_SEGMENTATION_THRESHOLD
            use_amr = getattr(st_cfg, "AMR_ST_ENABLED", False)
            if use_amr:
                # TTA logit sharpening for rare classes
                rare_classes = getattr(st_cfg, "AMR_ST_RARE_CLASSES", [3, 6, 12, 16, 17])
                rare_temp = getattr(st_cfg, "AMR_ST_RARE_TEMP", 0.7)
                tau_freq = getattr(st_cfg, "AMR_ST_TAU_FREQ", 0.7)
                tau_rare = getattr(st_cfg, "AMR_ST_TAU_RARE", 0.35)
                # Round-wise adaptive lowering
                round_decay = getattr(st_cfg, "AMR_ST_ROUND_DECAY", 0.1)
                current_tau_rare = max(0.15, tau_rare - (self.round - 1) * round_decay)

                # Sharpen rare-class logits
                sem_logits_boosted = semantic_segmentation_raw.clone()
                for rc in rare_classes:
                    if rc < sem_logits_boosted.shape[0]:
                        sem_logits_boosted[rc] = sem_logits_boosted[rc] / rare_temp

                # Class-dependent thresholding
                threshold_map = torch.full_like(sem_logits_boosted, tau_freq)
                for rc in rare_classes:
                    if rc < threshold_map.shape[0]:
                        threshold_map[rc] = current_tau_rare

                # Apply threshold: pixel must exceed its class threshold AND be max class
                semantic_segmentation = torch.where(
                    sem_logits_boosted > threshold_map, sem_logits_boosted, 0.0
                )
                semantic_segmentation_pseudo = semantic_segmentation.argmax(dim=0)
                semantic_segmentation_pseudo[semantic_segmentation.sum(dim=0) == 0] = 255
                if not self._logged_class_threshold:
                    log.info(
                        "AMR-ST: asymmetric thresholds (freq=%.2f, rare=%.2f, round=%d, temp=%.2f)",
                        tau_freq, current_tau_rare, self.round, rare_temp,
                    )
                    self._logged_class_threshold = True
            else:
                # A2: Per-class threshold adjustment based on class frequency
                class_freqs = st_cfg.get("CLASS_FREQUENCIES", None)
                num_stuff_classes = max_class_scores.shape[0]
                if class_freqs and len(class_freqs) == num_stuff_classes:
                    class_freqs = torch.tensor(class_freqs, device=semantic_segmentation_raw.device, dtype=torch.float32)
                    freq_ratio = class_freqs / class_freqs.max()  # [0, 1]
                    alpha = getattr(st_cfg, "CLASS_THRESHOLD_ALPHA", 0.3)
                    per_class_factor = (freq_ratio ** alpha).view(-1, 1, 1)
                    class_threshold = max_class_scores * base_thresh * per_class_factor
                    if not self._logged_class_threshold:
                        log.info(
                            "A2: Using class-aware pseudo-label thresholding (alpha=%.2f, %d classes)",
                            alpha,
                            len(class_freqs),
                        )
                        self._logged_class_threshold = True
                else:
                    if class_freqs and len(class_freqs) != num_stuff_classes:
                        if not self._logged_class_threshold:
                            log.warning(
                                "A2: CLASS_FREQUENCIES length (%d) does not match stuff classes (%d). "
                                "Falling back to global threshold.",
                                len(class_freqs),
                                num_stuff_classes,
                            )
                            self._logged_class_threshold = True
                    class_threshold = max_class_scores * base_thresh
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
            if instance.amax() > 0.0:
                # Make instance masks
                instance_masks = instances_to_masks(instance)
                # Make object semantics
                object_semantics_tensor: Tensor = torch.tensor(object_semantics, device=instance.device)
                sample_dict = {
                    "image": image["image"].squeeze(),
                    "sem_seg": semantic_segmentation_pseudo.long(),
                    "instances": Instances(
                        image_size=tuple(image["image"].shape[1:]),
                        gt_masks=BitMasks(instance_masks),
                        gt_boxes=Boxes(get_bounding_boxes(instance)),
                        gt_classes=object_semantics_tensor,
                    ),
                }
            else:
                sample_dict = {
                    "image": image["image"].squeeze(),
                    "sem_seg": semantic_segmentation_pseudo.long(),
                    "instances": Instances(
                        image_size=tuple(image["image"].shape),
                        gt_masks=BitMasks(torch.zeros(0, *image["image"].shape[1:]).bool()),
                        gt_boxes=Boxes(torch.zeros(0, 4).long()),
                        gt_classes=torch.zeros(0).long(),
                    ),
                }
            if confidence_weights is not None:
                sample_dict["confidence_weights"] = confidence_weights
            pseudo_labels.append(sample_dict)
        return pseudo_labels

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
        checkpoint = torch.load(config.MODEL.CHECKPOINT, map_location="cpu", weights_only=False)
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
            use_seesaw_loss=getattr(config.MODEL.ROI_BOX_HEAD, "USE_SEESAW_LOSS", False),
            seesaw_p=getattr(config.MODEL.ROI_BOX_HEAD, "SEESAW_P", 0.8),
            seesaw_q=getattr(config.MODEL.ROI_BOX_HEAD, "SEESAW_Q", 2.0),
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
            use_seesaw_loss=getattr(config.MODEL.ROI_BOX_HEAD, "USE_SEESAW_LOSS", False),
            seesaw_p=getattr(config.MODEL.ROI_BOX_HEAD, "SEESAW_P", 0.8),
            seesaw_q=getattr(config.MODEL.ROI_BOX_HEAD, "SEESAW_Q", 2.0),
        )
        # Load checkpoint BEFORE TTA wrapping
        if config.MODEL.CHECKPOINT is not None:
            log.info(f"Loading checkpoint into DINOv3 model: {config.MODEL.CHECKPOINT}")
            # Filter out teacher keys from self-training checkpoints
            student_checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith("teacher_")}
            if len(student_checkpoint) < len(checkpoint):
                log.info(f"Filtered {len(checkpoint) - len(student_checkpoint)} teacher keys from checkpoint")
            missing_keys, unexpected_keys = model.load_state_dict(student_checkpoint, strict=False)
            if missing_keys:
                log.warning(f"Missing keys when loading checkpoint: {missing_keys}")
            if unexpected_keys:
                log.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
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
