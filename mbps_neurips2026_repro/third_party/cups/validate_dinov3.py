"""Validate DINOv3 Stage-3 checkpoint on Cityscapes val."""
import os
import sys
import logging
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
from yacs.config import CfgNode

import cups
from cups.data import (
    CITYSCAPES_STUFF_CLASSES,
    CITYSCAPES_THING_CLASSES,
    CityscapesPanopticValidation,
    collate_function_validation,
)
from cups.augmentation import PhotometricAugmentations, ResolutionJitter

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

torch.set_float32_matmul_precision("medium")


def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment_config_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root_val", default=None, type=str, help="Override DATA.ROOT_VAL for local eval.")
    parser.add_argument("--accelerator", default=None, type=str, help="Override SYSTEM.ACCELERATOR.")
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = "disabled"

    # Load config
    config = cups.get_default_config(experiment_config_file=args.experiment_config_file)
    config.defrost()
    config.MODEL.CHECKPOINT = args.checkpoint
    if args.root_val is not None:
        config.DATA.ROOT_VAL = args.root_val
    if args.accelerator is not None:
        config.SYSTEM.ACCELERATOR = args.accelerator
        config.SYSTEM.NUM_GPUS = 1
    config.freeze()
    log.info(config)

    seed_everything(config.SYSTEM.SEED)

    # Validation dataset
    validation_dataset = CityscapesPanopticValidation(
        root=config.DATA.ROOT_VAL,
        crop_resolution=(512, 1024),
        num_classes=27,
        resize_scale=0.5,
    )
    validation_data_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_function_validation,
        drop_last=False,
    )
    log.info(f"{len(validation_dataset)} validation samples.")

    # Build model with checkpoint
    model = cups.build_model_self(
        config=config,
        thing_classes=CITYSCAPES_THING_CLASSES,
        stuff_classes=CITYSCAPES_STUFF_CLASSES,
        thing_pseudo_classes=None,
        stuff_pseudo_classes=None,
        class_weights=None,
        photometric_augmentation=PhotometricAugmentations(),
        freeze_bn=True,
        resolution_jitter_augmentation=ResolutionJitter(
            scales=None,
            resolutions=config.AUGMENTATION.RESOLUTIONS,
        ),
    )

    trainer = Trainer(
        accelerator=config.SYSTEM.ACCELERATOR,
        devices=config.SYSTEM.NUM_GPUS,
        precision=config.TRAINING.PRECISION,
        logger=False,
    )

    trainer.validate(model=model, dataloaders=validation_data_loader)


if __name__ == "__main__":
    main()
