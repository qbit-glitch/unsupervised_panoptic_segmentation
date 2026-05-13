#!/usr/bin/env python3
"""Validate CUPS Stage-3 checkpoint (DINOv2 or DINOv3 backbone).

Usage (local CPU, DINOv2):
    export PYTHONPATH=/path/to/mbps_panoptic_segmentation/refs/cups:$PYTHONPATH
    export WANDB_MODE=disabled
    python -u scripts/validate_cups_stage3.py

Usage (Anydesk GPU, DINOv2 with TTA fix):
    python -u scripts/validate_cups_stage3.py \
        --config refs/cups/configs/val_cups_stage3_vitb_k80_local.yaml \
        --checkpoint /path/to/cups_vitb_k80_stage3/ups_checkpoint_step=000100.ckpt \
        --accelerator gpu

Usage (Anydesk GPU, DINOv3):
    python -u scripts/validate_cups_stage3.py \
        --config refs/cups/configs/val_cups_stage3_dinov3_vitb_k80_anydesk.yaml \
        --checkpoint /path/to/cups_dinov3_stage3/checkpoint.ckpt \
        --accelerator gpu \
        --num_things 15 --num_stuffs 65
"""

import argparse
import os
import sys

os.environ.setdefault("WANDB_MODE", "disabled")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CUPS_ROOT = os.path.join(REPO_ROOT, "refs", "cups")
sys.path.insert(0, CUPS_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Validate CUPS Stage-3 checkpoint")
    parser.add_argument(
        "--config",
        default=os.path.join(CUPS_ROOT, "configs", "val_cups_stage3_vitb_k80_local.yaml"),
        help="CUPS config YAML file",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(
            REPO_ROOT, "checkpoints", "cups_vitb_k80_stage3",
            "ups_checkpoint_step=000100.ckpt",
        ),
        help="Path to Stage-3 checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--accelerator",
        default=None,
        help="Accelerator override: cpu, gpu, mps. Defaults to config value.",
    )
    parser.add_argument(
        "--num_things", type=int, default=15,
        help="Number of thing pseudo-classes in checkpoint (default 15 for k=80)",
    )
    parser.add_argument(
        "--num_stuffs", type=int, default=65,
        help="Number of stuff pseudo-classes in checkpoint (default 65 for k=80)",
    )
    parser.add_argument(
        "--precision", default="32-true",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        help="Trainer precision (default: 32-true; use 16-mixed on GPU to halve VRAM)",
    )
    parser.add_argument(
        "--data_root", default=None,
        help="Override DATA.ROOT and DATA.ROOT_VAL (useful when running on a different machine)",
    )
    parser.add_argument(
        "--run_name", default=None,
        help="WandB run name (defaults to config RUN_NAME)",
    )
    args = parser.parse_args()

    import torch
    import cups
    from cups.data import (
        CITYSCAPES_THING_CLASSES,
        CITYSCAPES_STUFF_CLASSES,
        CityscapesPanopticValidation,
        collate_function_validation,
    )
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import TQDMProgressBar
    from pytorch_lightning.loggers import WandbLogger
    from torch.utils.data import DataLoader

    print(f"Loading config: {args.config}")
    config = cups.get_default_config(
        experiment_config_file=args.config,
        command_line_arguments=[],
    )
    # Override dataset root if specified (for running on different machines)
    if args.data_root:
        config.defrost()
        config.DATA.ROOT = args.data_root
        config.DATA.ROOT_VAL = args.data_root
        config.freeze()

    print(f"Dataset root: {config.DATA.ROOT}")
    print(f"Backbone type: {getattr(config.MODEL, 'BACKBONE_TYPE', 'resnet50')}")
    print(f"TTA scales: {config.MODEL.TTA_SCALES}")
    print(f"Checkpoint: {args.checkpoint}")

    # Determine accelerator
    accel = args.accelerator or config.SYSTEM.ACCELERATOR
    print(f"Accelerator: {accel}")

    # Val dataset
    validation_dataset = CityscapesPanopticValidation(
        root=config.DATA.ROOT_VAL,
        resize_scale=config.DATA.VAL_SCALE,
        crop_resolution=config.DATA.CROP_RESOLUTION,
        num_classes=config.DATA.NUM_CLASSES,
    )
    print(f"Val dataset: {len(validation_dataset)} images")

    validation_data_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_function_validation,
        drop_last=False,
        pin_memory=False,
    )

    # Build Stage-3 self-training model.
    # Pass pseudo class lists directly to skip checkpoint loading in build_model_self.
    from cups.augmentation import PhotometricAugmentations, ResolutionJitter

    thing_pseudo_classes = list(range(args.num_things))
    stuff_pseudo_classes = list(range(args.num_stuffs))

    import inspect
    _build_kwargs = dict(
        config=config,
        thing_classes=CITYSCAPES_THING_CLASSES,
        stuff_classes=CITYSCAPES_STUFF_CLASSES,
        thing_pseudo_classes=thing_pseudo_classes,
        stuff_pseudo_classes=stuff_pseudo_classes,
        class_weights=None,
        photometric_augmentation=PhotometricAugmentations(),
        freeze_bn=True,
        resolution_jitter_augmentation=ResolutionJitter(
            scales=None,
            resolutions=config.AUGMENTATION.RESOLUTIONS,
        ),
    )
    # mask_refiner added in newer versions of cups; pass only if accepted
    if "mask_refiner" in inspect.signature(cups.build_model_self).parameters:
        _build_kwargs["mask_refiner"] = None
    model = cups.build_model_self(**_build_kwargs)

    run_name = args.run_name or getattr(config.SYSTEM, "RUN_NAME", "cups_stage3_val")
    out_dir = f"experiments/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    logger = WandbLogger(
        name=run_name,
        save_dir=out_dir,
        project="Unsupervised Panoptic Segmentation",
        log_model=False,
    )

    trainer = Trainer(
        default_root_dir=out_dir,
        accelerator=accel,
        devices=1,
        precision=args.precision,
        callbacks=[TQDMProgressBar(refresh_rate=10)],
        logger=logger,
        num_sanity_val_steps=0,
    )

    # Load checkpoint manually (weights_only=False needed for yacs.config.CfgNode).
    # SelfSupervisedModel has self.model (student) and self.teacher_model (EMA teacher).
    # State dict keys: model.* and teacher_model.* — load directly with strict=False.
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    print("Checkpoint loaded successfully.")

    print("\nRunning validation...")
    trainer.validate(model=model, dataloaders=validation_data_loader)


if __name__ == "__main__":
    main()
