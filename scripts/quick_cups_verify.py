#!/usr/bin/env python3
"""Verify CUPS Stage-3 PQ using trainer.validate() — same as training.

Uses GPU for everything, exactly replicates training validation.

Usage (on remote):
    export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:$PYTHONPATH
    cd /media/santosh/Kuldeep/panoptic_segmentation
    python -u scripts/quick_cups_verify.py
"""

import os
import sys
import torch
from pathlib import Path

REPO_ROOT = Path("/media/santosh/Kuldeep/panoptic_segmentation")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "refs" / "cups"))

import cups
from cups.data import (
    CITYSCAPES_THING_CLASSES, CITYSCAPES_STUFF_CLASSES, CITYSCAPES_CLASSNAMES,
    CityscapesPanopticValidation, collate_function_validation,
)
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

# Disable wandb
os.environ["WANDB_MODE"] = "disabled"


def main():
    # Load config
    config = cups.get_default_config(
        experiment_config_file=str(REPO_ROOT / "refs/cups/configs/train_cityscapes_vitb.yaml")
    )
    config.defrost()
    config.MODEL.CHECKPOINT = str(
        REPO_ROOT / "experiments/experiments/cups_vitb_k80_stage3"
        / "Unsupervised Panoptic Segmentation/sh17g3sl/checkpoints"
        / "ups_checkpoint_step=000100.ckpt"
    )
    config.freeze()

    # Extract pseudo classes from checkpoint
    ckpt = torch.load(config.MODEL.CHECKPOINT, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    thing_pseudo = tuple(hp.get("thing_pseudo_classes", tuple(range(65, 80))))
    stuff_pseudo = tuple(hp.get("stuff_pseudo_classes", tuple(range(0, 65))))
    print(f"thing_pseudo: {thing_pseudo}")
    print(f"stuff_pseudo: {stuff_pseudo[:5]}...{stuff_pseudo[-3:]}")
    del ckpt

    # Build model — pass pseudo classes so model hparams are correct
    model = cups.build_model_pseudo(
        config=config,
        thing_classes=CITYSCAPES_THING_CLASSES,
        stuff_classes=CITYSCAPES_STUFF_CLASSES,
        thing_pseudo_classes=thing_pseudo,
        stuff_pseudo_classes=stuff_pseudo,
        class_weights=None,
        copy_paste_augmentation=torch.nn.Identity(),
        resolution_jitter_augmentation=torch.nn.Identity(),
        photometric_augmentation=torch.nn.Identity(),
        use_tta=False,
        class_names=CITYSCAPES_CLASSNAMES,
        classes_mask=None,
    )
    print(f"Model loaded. hparams.thing_pseudo: {model.hparams.thing_pseudo_classes}")

    # Skip logger.log_image() call in on_validation_epoch_end (we have no logger)
    model.plot_validation_samples = True

    # Validation dataset (same as train_self.py hardcodes)
    val_dataset = CityscapesPanopticValidation(
        root=config.DATA.ROOT,
        crop_resolution=(512, 1024),
        num_classes=27,
        resize_scale=0.5,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_function_validation,
        drop_last=False,
        pin_memory=False,
    )
    print(f"Val dataset: {len(val_dataset)} images")

    # Use Trainer.validate() — GPU, same as training
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )

    print("\nRunning trainer.validate() (GPU, same as training)...")
    results = trainer.validate(model=model, dataloaders=val_loader)
    print(f"\nResults: {results}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
