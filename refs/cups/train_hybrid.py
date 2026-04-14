"""Hybrid training: Phase 1 (supervised on disk pseudo-labels) + Phase 2 (self-training with TTA).

Chains Stage-2 and Stage-3 into one script. No pre-trained checkpoint needed.

Usage:
    python train_hybrid.py --experiment_config_file configs/train_hybrid_local.yaml --disable_wandb
    python train_hybrid.py --experiment_config_file configs/train_hybrid_local.yaml --disable_wandb --skip_phase1
"""
import gc
import logging
import multiprocessing
import os
import resource
import sys
from argparse import REMAINDER, ArgumentParser
from datetime import datetime
from typing import Any, Dict

# Python 3.14+ defaults to 'forkserver'
if sys.version_info >= (3, 14):
    multiprocessing.set_start_method("fork", force=True)

import torch
import torch.nn
import torch.serialization
if hasattr(torch.serialization, "add_safe_globals"):
    from yacs.config import CfgNode as _CfgNode
    torch.serialization.add_safe_globals([_CfgNode])

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from yacs.config import CfgNode

import cups
from cups.augmentation import (
    CopyPasteAugmentation,
    PhotometricAugmentations,
    ResolutionJitter,
    get_pseudo_label_augmentations,
)
from cups.data import (
    CITYSCAPES_CLASSNAMES,
    CITYSCAPES_STUFF_CLASSES,
    CITYSCAPES_THING_CLASSES,
    CityscapesPanopticValidation,
    CityscapesSelfTraining,
    PseudoLabelDataset,
    StepDataset,
    collate_function_validation,
)
from cups.mask_refinement import MaskRefiner
from cups.utils import RTPTCallback

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (16384, rlimit[1]))

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

torch.set_float32_matmul_precision("medium")


def _identity_collate(x):
    """Identity collate function (pickleable, unlike lambda)."""
    return x


def configure():
    """Parse args and load config."""
    parser = ArgumentParser()
    parser.add_argument("--cuda_visible_devices", default=None, type=str)
    parser.add_argument("config", default=None, nargs=REMAINDER)
    parser.add_argument("--disable_wandb", default=False, action="store_true")
    parser.add_argument("--experiment_config_file", default=None, type=str)
    parser.add_argument(
        "--skip_phase1", default=False, action="store_true",
        help="Skip Phase 1; use MODEL.CHECKPOINT for Phase 2 directly.",
    )
    args = parser.parse_args()
    args_dict = vars(args)
    if (cuda_visible_devices := args_dict.pop("cuda_visible_devices")) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if args_dict.pop("disable_wandb"):
        os.environ["WANDB_MODE"] = "disabled"
    experiment_config_file = args_dict.pop("experiment_config_file")
    skip_phase1 = args_dict.pop("skip_phase1")
    config = cups.get_default_config(
        experiment_config_file=experiment_config_file,
        command_line_arguments=args.config,
    )
    return config, skip_phase1


def run_phase1(config, experiment_path, run_name):
    """Phase 1: Supervised training on disk pseudo-labels.

    Returns:
        checkpoint_path: Path to Detectron2-format .pth file.
        thing_pseudo_classes: Tuple of thing pseudo-class IDs from dataset.
        stuff_pseudo_classes: Tuple of stuff pseudo-class IDs from dataset.
    """
    log.info("=" * 60)
    log.info("PHASE 1: Supervised training on disk pseudo-labels")
    log.info(f"Steps: {config.TRAINING.STEPS}, Batch size: {config.TRAINING.BATCH_SIZE}")
    log.info("=" * 60)

    # Dataset
    training_dataset = PseudoLabelDataset(
        root=config.DATA.ROOT,
        root_pseudo=config.DATA.ROOT_PSEUDO,
        return_detectron2_format=True,
        ground_truth_scale=config.DATA.SCALE,
        crop_resolution=config.DATA.CROP_RESOLUTION,
        thing_stuff_threshold=config.DATA.THING_STUFF_THRESHOLD,
        ignore_unknown_thing_regions=config.DATA.IGNORE_UNKNOWN_THING_REGIONS,
        augmentations=get_pseudo_label_augmentations(config.DATA.CROP_RESOLUTION),
        dataset=config.DATA.DATASET,
        only_use_non_empty_samples=True,
    )

    validation_dataset = CityscapesPanopticValidation(
        root=config.DATA.ROOT_VAL,
        crop_resolution=(512, 1024),
        num_classes=27,
        resize_scale=0.5,
    )

    thing_classes = CITYSCAPES_THING_CLASSES
    stuff_classes = CITYSCAPES_STUFF_CLASSES
    class_names = CITYSCAPES_CLASSNAMES

    log.info(
        f"{len(training_dataset)} training samples, {len(validation_dataset)} validation samples. "
        f"Things: {training_dataset.things_classes}, Stuff: {training_dataset.stuff_classes}"
    )

    # Gradient accumulation
    accum = getattr(config.TRAINING, "ACCUMULATE_GRAD_BATCHES", 1)
    effective_batch = config.TRAINING.BATCH_SIZE * config.SYSTEM.NUM_GPUS * accum
    log.info(f"Effective batch size: {config.TRAINING.BATCH_SIZE} × {config.SYSTEM.NUM_GPUS} GPUs × {accum} accum = {effective_batch}")

    # DataLoaders
    num_workers = config.SYSTEM.NUM_WORKERS
    training_data_loader = DataLoader(
        dataset=StepDataset(
            training_dataset,
            steps=config.TRAINING.STEPS * config.SYSTEM.NUM_GPUS * config.TRAINING.BATCH_SIZE * accum,
        ),
        batch_size=config.TRAINING.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_identity_collate,
        drop_last=True,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=6 if num_workers > 0 else None,
    )
    validation_data_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=min(config.TRAINING.BATCH_SIZE, 2),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_function_validation,
        drop_last=False,
        pin_memory=False,
    )

    # Model (from scratch, no checkpoint)
    model = cups.build_model_pseudo(
        config=config,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        thing_pseudo_classes=training_dataset.things_classes,
        stuff_pseudo_classes=training_dataset.stuff_classes,
        class_weights=(
            tuple(
                (1.0 / (torch.tensor(training_dataset.class_distribution)
                         * len(training_dataset.class_distribution))).tolist()
            ) if config.TRAINING.CLASS_WEIGHTING else None
        ),
        copy_paste_augmentation=(
            CopyPasteAugmentation(
                thing_class=len(training_dataset.stuff_classes),
                max_num_pasted_objects=config.AUGMENTATION.MAX_NUM_PASTED_OBJECTS,
            ) if config.AUGMENTATION.COPY_PASTE else None
        ),
        photometric_augmentation=PhotometricAugmentations(),
        resolution_jitter_augmentation=ResolutionJitter(
            scales=None, resolutions=config.AUGMENTATION.RESOLUTIONS,
        ),
        class_names=class_names,
    )

    # Strategy
    if config.SYSTEM.NUM_GPUS > 1:
        use_gloo = getattr(config.VALIDATION, "CACHE_DEVICE", "cuda") == "cpu"
        strategy = DDPStrategy(
            process_group_backend="gloo" if use_gloo else "nccl",
            find_unused_parameters=True,
        )
    else:
        strategy = "auto"

    # Logger
    logger = WandbLogger(
        name="phase1_" + run_name,
        log_model=False,
        save_dir=experiment_path,
        project="Unsupervised Panoptic Segmentation",
    )

    # Trainer
    trainer = Trainer(
        default_root_dir=experiment_path,
        accelerator=config.SYSTEM.ACCELERATOR,
        devices=config.SYSTEM.NUM_GPUS,
        num_nodes=config.SYSTEM.NUM_NODES,
        strategy=strategy,
        precision=config.TRAINING.PRECISION,
        max_steps=config.TRAINING.STEPS,
        min_steps=config.TRAINING.STEPS,
        accumulate_grad_batches=accum,
        callbacks=[
            RTPTCallback(name_initials="SB", experiment_name="Hybrid_P1"),
            TQDMProgressBar(refresh_rate=1),
            ModelCheckpoint(
                filename="hybrid_phase1_{step:06d}",
                every_n_train_steps=config.TRAINING.VAL_EVERY_N_STEPS,
                save_last=True,
                save_top_k=-1,
            ),
            LearningRateMonitor(logging_interval="step", log_momentum=True),
        ],
        logger=logger,
        log_every_n_steps=config.TRAINING.LOG_EVERT_N_STEPS,
        gradient_clip_algorithm=config.TRAINING.GRADIENT_CLIP_ALGORITHM,
        gradient_clip_val=config.TRAINING.GRADIENT_CLIP_VAL,
        check_val_every_n_epoch=None,
        val_check_interval=config.TRAINING.VAL_EVERY_N_STEPS,
        num_sanity_val_steps=0,
    )

    # Train Phase 1
    trainer.fit(
        model=model,
        train_dataloaders=training_data_loader,
        val_dataloaders=validation_data_loader,
    )

    # Save raw Detectron2 state dict (avoids Lightning key prefix issues)
    raw_state_dict = model.model.state_dict()
    checkpoint_path = os.path.join(experiment_path, "phase1_detectron2.pth")
    torch.save({"model": raw_state_dict}, checkpoint_path)
    log.info(f"Phase 1 complete. Detectron2 checkpoint saved to: {checkpoint_path}")

    # Store pseudo-class info for Phase 2
    thing_pseudo_classes = training_dataset.things_classes
    stuff_pseudo_classes = training_dataset.stuff_classes

    # Cleanup
    del model, trainer, training_data_loader, validation_data_loader, training_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return checkpoint_path, thing_pseudo_classes, stuff_pseudo_classes


def run_phase2(config, experiment_path, run_name, checkpoint_path):
    """Phase 2: Self-training with TTA teacher."""
    total_steps = config.SELF_TRAINING.ROUND_STEPS * config.SELF_TRAINING.ROUNDS
    log.info("=" * 60)
    log.info("PHASE 2: Self-training with TTA teacher")
    log.info(f"Checkpoint: {checkpoint_path}")
    log.info(f"Rounds: {config.SELF_TRAINING.ROUNDS}, Steps/round: {config.SELF_TRAINING.ROUND_STEPS}, Total: {total_steps}")
    log.info("=" * 60)

    # Set checkpoint path in config
    config.defrost()
    config.MODEL.CHECKPOINT = checkpoint_path
    config.freeze()

    # Dataset (raw images only — teacher generates pseudo-labels via TTA)
    training_dataset = CityscapesSelfTraining(
        root=config.DATA.ROOT,
        split="train",
        resize_scale=config.DATA.SCALE,
        crop_resolution=config.DATA.CROP_RESOLUTION,
        only_train_samples=False,
    )

    validation_dataset = CityscapesPanopticValidation(
        root=config.DATA.ROOT_VAL,
        crop_resolution=(512, 1024),
        num_classes=27,
        resize_scale=0.5,
    )

    thing_classes = CITYSCAPES_THING_CLASSES
    stuff_classes = CITYSCAPES_STUFF_CLASSES

    log.info(f"{len(training_dataset)} training samples, {len(validation_dataset)} validation samples.")

    # DataLoaders
    prefetch = None if config.SYSTEM.NUM_WORKERS == 0 else (
        2 if config.SYSTEM.NUM_WORKERS <= 2 else 6
    )
    training_data_loader = DataLoader(
        dataset=training_dataset,
        batch_size=config.TRAINING.BATCH_SIZE,
        shuffle=True,
        num_workers=config.SYSTEM.NUM_WORKERS,
        collate_fn=_identity_collate,
        drop_last=True,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=prefetch,
    )
    validation_data_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=config.SYSTEM.NUM_WORKERS,
        collate_fn=collate_function_validation,
        drop_last=False,
        pin_memory=False,
    )

    # Mask refiner
    mask_refiner = None
    if getattr(config, "MASK_REFINEMENT", None) is not None and config.MASK_REFINEMENT.ENABLE:
        mask_refiner = MaskRefiner.from_config(config)
        log.info("Mask refinement enabled.")

    # Model (loads Phase 1 checkpoint, creates TTA teacher)
    model = cups.build_model_self(
        config=config,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        thing_pseudo_classes=None,  # inferred from checkpoint weights
        stuff_pseudo_classes=None,
        class_weights=None,
        photometric_augmentation=PhotometricAugmentations(),
        freeze_bn=True,
        resolution_jitter_augmentation=ResolutionJitter(
            scales=None, resolutions=config.AUGMENTATION.RESOLUTIONS,
        ),
        mask_refiner=mask_refiner,
    )

    # Copy-paste augmentation
    model.copy_paste_augmentation = (
        CopyPasteAugmentation(
            thing_class=len(model.hparams.stuff_pseudo_classes),
            max_num_pasted_objects=config.AUGMENTATION.MAX_NUM_PASTED_OBJECTS,
        ) if config.AUGMENTATION.COPY_PASTE else None
    )

    # Logger
    logger = WandbLogger(
        name="phase2_self_" + run_name,
        log_model=False,
        save_dir=experiment_path,
        project="Unsupervised Panoptic Segmentation",
    )

    # Trainer
    trainer = Trainer(
        default_root_dir=experiment_path,
        accelerator=config.SYSTEM.ACCELERATOR,
        devices=config.SYSTEM.NUM_GPUS,
        num_nodes=config.SYSTEM.NUM_NODES,
        strategy=(
            config.SYSTEM.DISTRIBUTED_BACKEND if config.SYSTEM.NUM_GPUS == 1
            else "ddp_find_unused_parameters_true"
        ),
        precision=config.TRAINING.PRECISION,
        max_steps=total_steps,
        min_steps=total_steps,
        callbacks=[
            RTPTCallback(name_initials="SB", experiment_name="Hybrid_P2"),
            TQDMProgressBar(refresh_rate=1),
            ModelCheckpoint(
                filename="hybrid_phase2_{step:06d}",
                every_n_train_steps=config.TRAINING.VAL_EVERY_N_STEPS,
                save_last=True,
                save_top_k=-1,
            ),
        ],
        logger=logger,
        log_every_n_steps=config.TRAINING.LOG_EVERT_N_STEPS,
        gradient_clip_algorithm=config.TRAINING.GRADIENT_CLIP_ALGORITHM,
        gradient_clip_val=config.TRAINING.GRADIENT_CLIP_VAL,
        check_val_every_n_epoch=None,
        val_check_interval=config.TRAINING.VAL_EVERY_N_STEPS,
        num_sanity_val_steps=0,
    )

    # Train Phase 2
    trainer.fit(
        model=model,
        train_dataloaders=training_data_loader,
        val_dataloaders=validation_data_loader,
    )

    log.info("Phase 2 complete. Hybrid training finished.")


def main():
    config, skip_phase1 = configure()
    log.info(config)
    seed_everything(config.SYSTEM.SEED)

    # Experiment directory
    if config.SYSTEM.RUN_NAME is not None:
        run_name = config.SYSTEM.RUN_NAME
    else:
        run_name = "hybrid_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experiment_path = os.path.join(
        os.getcwd() if config.SYSTEM.LOG_PATH is None else config.SYSTEM.LOG_PATH,
        "experiments",
        run_name,
    )
    os.makedirs(experiment_path, exist_ok=True)

    if skip_phase1:
        assert config.MODEL.CHECKPOINT is not None, \
            "--skip_phase1 requires MODEL.CHECKPOINT to be set in config or via CLI"
        checkpoint_path = config.MODEL.CHECKPOINT
        log.info(f"Skipping Phase 1. Using checkpoint: {checkpoint_path}")
    else:
        checkpoint_path, _, _ = run_phase1(config, experiment_path, run_name)

    run_phase2(config, experiment_path, run_name, checkpoint_path)


if __name__ == "__main__":
    main()
