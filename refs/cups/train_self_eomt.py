"""CUPS Stage-3 self-training with the DINOv2 + EoMT network.

Identical to train_self.py — same datasets, same SelfSupervisedModel, same
augmentations, same trainer/optimizer/checkpointing — except the model
builder constructs the EoMT network behind a Detectron2-interface adapter
(cups/model/eomt_d2_adapter.py) instead of the Panoptic Cascade Mask R-CNN.

Requires PYTHONPATH to include BOTH refs/cups and refs/eomt.
"""

import copy
import logging
import os
import resource
from argparse import REMAINDER, ArgumentParser
from datetime import datetime
from typing import Any, Dict

import torch
import torch.nn
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from yacs.config import CfgNode

import cups
from cups.augmentation import (
    CopyPasteAugmentation,
    PhotometricAugmentations,
    ResolutionJitter,
)
from cups.data import (
    CITYSCAPES_STUFF_CLASSES,
    CITYSCAPES_THING_CLASSES,
    CityscapesPanopticValidation,
    CityscapesSelfTraining,
    collate_function_validation,
)
from cups.model.eomt_d2_adapter import EoMTPanopticShim, EoMTWithTTA
from cups.pl_model_self import SelfSupervisedModel
from cups.utils import RTPTCallback


def _identity_collate(x):
    """Identity collate function (pickleable, unlike lambda)."""
    return x


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (16384, rlimit[1]))

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

torch.set_float32_matmul_precision("medium")

# k=80 unified pseudo-class split (identical to the EoMT Stage-2 config
# refs/eomt/configs/dinov3/cityscapes/panoptic/eomt_base_640_santosh*.yaml).
STUFF_CLASSES_80 = [
    0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 28, 30, 31, 33, 34, 35, 36, 37, 38, 39,
    42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    60, 61, 62, 63, 64, 65, 68, 69, 70, 71, 72, 73, 74, 75, 76,
    77, 79,
]
THING_CLASSES_80 = sorted(set(range(80)) - set(STUFF_CLASSES_80))
EOMT_BACKBONE_NAME = "facebook/dinov2-base"
EOMT_NUM_QUERIES = 100
EOMT_NUM_BLOCKS = 3


def build_eomt_network(config: CfgNode) -> torch.nn.Module:
    """Build the EoMT network and load the Stage-2 Lightning checkpoint."""
    from models.eomt import EoMT  # refs/eomt (PYTHONPATH)
    from models.vit import ViT

    encoder = ViT(
        img_size=tuple(config.DATA.CROP_RESOLUTION),
        backbone_name=EOMT_BACKBONE_NAME,
    )
    network = EoMT(
        encoder=encoder,
        num_classes=len(STUFF_CLASSES_80) + len(THING_CLASSES_80),
        num_q=EOMT_NUM_QUERIES,
        num_blocks=EOMT_NUM_BLOCKS,
        masked_attn_enabled=True,
    )

    assert config.MODEL.CHECKPOINT is not None, "Stage-2 EoMT checkpoint required."
    checkpoint = torch.load(config.MODEL.CHECKPOINT, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    network_state = {
        key.replace("network.", "", 1): value
        for key, value in state_dict.items()
        if key.startswith("network.")
    }
    missing, unexpected = network.load_state_dict(network_state, strict=False)
    log.info(
        "EoMT Stage-2 checkpoint loaded from %s (%d keys; missing=%d unexpected=%d).",
        config.MODEL.CHECKPOINT,
        len(network_state),
        len(missing),
        len(unexpected),
    )
    if missing:
        log.warning("Missing keys (first 5): %s", missing[:5])
    if unexpected:
        log.warning("Unexpected keys (first 5): %s", unexpected[:5])

    # CUPS Stage-3 freezes the backbone (DINOV2_FREEZE) and trains heads
    # only. EoMT analog: freeze patch embed + all encoder blocks except the
    # last EOMT_NUM_BLOCKS (the de-facto decoder where queries attend to
    # image tokens). The CUPS head-only optimizer additionally excludes any
    # parameter whose name contains "norm".
    backbone = network.encoder.backbone
    first_trainable = len(backbone.blocks) - EOMT_NUM_BLOCKS
    frozen, trainable = 0, 0
    for name, param in network.named_parameters():
        if name.startswith("encoder.backbone."):
            sub = name.replace("encoder.backbone.", "")
            if sub.startswith("blocks."):
                keep = int(sub.split(".")[1]) >= first_trainable
            elif sub.startswith("norm."):
                keep = True
            else:
                keep = False
            param.requires_grad_(keep)
        else:
            param.requires_grad_(True)
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()
    log.info(
        "EoMT freezing: %.1fM trainable / %.1fM frozen (blocks >= %d + heads).",
        trainable / 1e6,
        frozen / 1e6,
        first_trainable,
    )
    return network


def build_model_self_eomt(
    config: CfgNode,
    thing_classes,
    stuff_classes,
    photometric_augmentation: torch.nn.Module,
    resolution_jitter_augmentation: torch.nn.Module,
) -> SelfSupervisedModel:
    """Builds the ORIGINAL SelfSupervisedModel around the EoMT adapter."""
    network = build_eomt_network(config)
    shim = EoMTPanopticShim(
        eomt=network,
        stuff_class_ids=STUFF_CLASSES_80,
        thing_class_ids=THING_CLASSES_80,
        confidence_threshold=config.MODEL.TTA_INFERENCE_CONFIDENCE_THRESHOLD,
    )
    wrapper = EoMTWithTTA(shim, tta_scales=config.MODEL.TTA_SCALES, tta_flip=True)
    model = SelfSupervisedModel(
        model=wrapper,
        num_thing_pseudo_classes=len(THING_CLASSES_80),
        num_stuff_pseudo_classes=len(STUFF_CLASSES_80),
        config=config,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        photometric_augmentation=photometric_augmentation,
        resolution_jitter_augmentation=resolution_jitter_augmentation,
        class_names=None,
        classes_mask=None,
        mask_refiner=None,
    )
    return model


def configure() -> CfgNode:
    """Function loads default config, experiment config, and parses command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cuda_visible_devices",
        default=None,
        type=str,
        help="Sets the visible cuda devices.",
    )
    parser.add_argument(
        "config",
        help="Modify config options using the command-line",
        default=None,
        nargs=REMAINDER,
    )
    parser.add_argument(
        "--disable_wandb",
        default=False,
        action="store_true",
        help="Binary flag. If set run will not be tracked with Weights and Biases.",
    )
    parser.add_argument("--experiment_config_file", default=None, type=str, help="Path to experiment config file.")
    parser.add_argument("--ckpt_path", default=None, type=str, help="Path to Lightning checkpoint to resume training.")
    args = parser.parse_args()
    args_dict: Dict[str, Any] = vars(args)
    if (cuda_visible_devices := args_dict.pop("cuda_visible_devices")) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if args_dict.pop("disable_wandb"):
        os.environ["WANDB_MODE"] = "disabled"
    ckpt_path = args_dict.pop("ckpt_path")
    experiment_config_file = args_dict.pop("experiment_config_file")
    config: CfgNode = cups.get_default_config(
        experiment_config_file=experiment_config_file, command_line_arguments=args.config
    )
    return config, ckpt_path


def main() -> None:
    # Get config
    config, ckpt_path = configure()
    # Print config
    log.info(config)
    # Set seed
    seed_everything(config.SYSTEM.SEED)
    # Make datasets (Cityscapes only — KITTI path omitted for the EoMT variant)
    training_dataset = CityscapesSelfTraining(
        root=config.DATA.ROOT,
        split="train",
        resize_scale=config.DATA.SCALE,
        crop_resolution=config.DATA.CROP_RESOLUTION,
        only_train_samples=False,
        depth_subdir=getattr(config.DATA, "DEPTH_SUBDIR", ""),
    )
    validation_dataset = CityscapesPanopticValidation(
        root=config.DATA.ROOT_VAL,
        crop_resolution=(512, 1024),
        num_classes=27,
        resize_scale=0.5,
    )
    thing_classes = CITYSCAPES_THING_CLASSES
    stuff_classes = CITYSCAPES_STUFF_CLASSES
    # Print dataset length
    log.info(f"{len(training_dataset)} training samples and {len(validation_dataset)} validation samples detected.")
    # Make data loaders
    prefetch = None if config.SYSTEM.NUM_WORKERS == 0 else (2 if config.SYSTEM.NUM_WORKERS <= 2 else 6)
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
    # Init model (EoMT behind the D2 adapter, original SelfSupervisedModel)
    model: LightningModule = build_model_self_eomt(
        config=config,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        photometric_augmentation=PhotometricAugmentations(),
        resolution_jitter_augmentation=ResolutionJitter(
            scales=None,
            resolutions=config.AUGMENTATION.RESOLUTIONS,
        ),
    )
    # Init copy-paste augmentation
    model.copy_paste_augmentation = (
        CopyPasteAugmentation(
            thing_class=len(model.hparams.stuff_pseudo_classes),
            max_num_pasted_objects=config.AUGMENTATION.MAX_NUM_PASTED_OBJECTS,
        )
        if config.AUGMENTATION.COPY_PASTE
        else None
    )
    # Print model
    log.info(model)
    # Init experiments folder since W&B otherwise warns and uses temp
    os.makedirs(
        os.path.join(os.getcwd() if config.SYSTEM.LOG_PATH is None else config.SYSTEM.LOG_PATH, "experiments"),
        exist_ok=True,
    )
    if config.SYSTEM.RUN_NAME is not None:
        run_name = config.SYSTEM.RUN_NAME
    else:
        run_name = "pseudo_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experiment_path = os.path.join(
        os.getcwd() if config.SYSTEM.LOG_PATH is None else config.SYSTEM.LOG_PATH,
        "experiments",
        run_name,
    )
    os.makedirs(experiment_path, exist_ok=True)
    # Init logger
    logger = WandbLogger(
        name="self_" + run_name,
        log_model=False,
        save_dir=experiment_path,
        project="Unsupervised Panoptic Segmentation",
    )
    # Init trainer
    trainer: Trainer = Trainer(
        default_root_dir=experiment_path,
        accelerator=config.SYSTEM.ACCELERATOR,
        devices=config.SYSTEM.NUM_GPUS,
        num_nodes=config.SYSTEM.NUM_NODES,
        strategy=(
            config.SYSTEM.DISTRIBUTED_BACKEND if config.SYSTEM.NUM_GPUS == 1 else "ddp_find_unused_parameters_true"
        ),
        precision=config.TRAINING.PRECISION,
        accumulate_grad_batches=getattr(config.TRAINING, "ACCUMULATE_GRAD_BATCHES", 1),
        max_steps=config.SELF_TRAINING.ROUND_STEPS * config.SELF_TRAINING.ROUNDS,
        min_steps=config.SELF_TRAINING.ROUND_STEPS * config.SELF_TRAINING.ROUNDS,
        callbacks=[
            RTPTCallback(name_initials="CR&OH", experiment_name="UPS_Self_EoMT"),
            TQDMProgressBar(refresh_rate=1),
            ModelCheckpoint(
                filename="best_pq_{step:06d}",
                monitor="pq_val",
                mode="max",
                save_top_k=6,
                save_last=True,
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
    # Perform training
    trainer.fit(
        model=model,
        train_dataloaders=training_data_loader,
        val_dataloaders=validation_data_loader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
