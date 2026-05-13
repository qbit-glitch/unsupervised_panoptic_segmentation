"""MBPS Training Script (PyTorch).

Usage:
    python scripts/train.py --config configs/cityscapes.yaml
    python scripts/train.py --config configs/coco_stuff27.yaml --resume PATH

Launches full 4-phase curriculum training with:
    - Phase A: Semantic-only (epochs 1-20)
    - Phase B: + Instance with gradient projection (epochs 21-40)
    - Phase C: Full model with bridge + consistency + PQ (epochs 41-60)
    - Phase D: Self-training refinement
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbps_pytorch.data.datasets import get_dataset, collate_batch
from mbps_pytorch.data.transforms import TrainTransform
from mbps_pytorch.models.mbps_model import MBPSModel
from mbps_pytorch.training.trainer import MBPSTrainer

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and merge YAML config with defaults.

    Args:
        config_path: Path to dataset-specific config.

    Returns:
        Merged configuration dict.
    """
    # Load default config
    default_path = os.path.join(
        os.path.dirname(config_path), "default.yaml"
    )
    with open(default_path) as f:
        config = yaml.safe_load(f)

    # Override with dataset-specific config
    with open(config_path) as f:
        override = yaml.safe_load(f)

    def deep_merge(base, override):
        for k, v in override.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                deep_merge(base[k], v)
            else:
                base[k] = v

    deep_merge(config, override)
    return config


def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
    steps_per_epoch: int,
) -> tuple:
    """Create optimizer with learning rate schedule.

    Args:
        model: PyTorch model.
        config: Training configuration.
        steps_per_epoch: Number of training steps per epoch.

    Returns:
        Tuple of (optimizer, lr_scheduler).
    """
    lr = config["training"]["learning_rate"]
    warmup_epochs = config["training"].get("warmup_epochs", 5)
    total_epochs = config["training"]["total_epochs"]
    weight_decay = config["training"].get("weight_decay", 0.01)
    gradient_clip = config["training"].get("gradient_clip", 1.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Cosine schedule with linear warmup
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler, gradient_clip


def create_train_data(config: Dict[str, Any]) -> DataLoader:
    """Create training data pipeline.

    Args:
        config: Data configuration.

    Returns:
        PyTorch DataLoader.
    """
    data_config = config["data"]
    training_config = config.get("training", {})
    batch_size = data_config.get(
        "batch_size", training_config.get("batch_size", 8)
    )

    # Python dataset with PyTorch DataLoader
    logger.info("Using Python dataset loader with PyTorch DataLoader")
    transform = TrainTransform(
        crop_size=tuple(data_config.get("crop_size", data_config["image_size"])),
    )
    dataset_name = data_config.get(
        "dataset", data_config.get("dataset_name", "cityscapes")
    )
    dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_config["data_dir"],
        depth_dir=data_config.get("depth_dir", ""),
        split="train",
        image_size=tuple(data_config["image_size"]),
        transforms=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        collate_fn=collate_batch,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader


def setup_ddp(rank: int, world_size: int) -> None:
    """Initialize DDP process group.

    Args:
        rank: Process rank.
        world_size: Total number of processes.
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="MBPS Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num_devices",
        type=int,
        default=None,
        help="Number of GPU devices (None = all available)",
    )
    parser.add_argument(
        "--vm_name",
        type=str,
        default=os.environ.get("MBPS_VM_NAME", "local"),
        help="VM identifier for W&B and checkpoints",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=os.environ.get("MBPS_EXPERIMENT", "default"),
        help="Experiment name (e.g. cityscapes_full, ablation_no_mamba)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        help="Ablation config to merge (e.g. configs/ablations/no_mamba.yaml)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for DDP (set by torchrun)",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)

    # Merge ablation config if specified
    if args.ablation and os.path.exists(args.ablation):
        with open(args.ablation) as f:
            ablation_override = yaml.safe_load(f)

        def deep_merge(base, override):
            for k, v in override.items():
                if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                    deep_merge(base[k], v)
                else:
                    base[k] = v

        deep_merge(config, ablation_override)
        logger.info(f"Merged ablation config: {args.ablation}")

    # Determine device setup
    if torch.cuda.is_available():
        num_devices = args.num_devices or torch.cuda.device_count()
        device = torch.device("cuda", args.local_rank)
    else:
        num_devices = 1
        device = torch.device("cpu")

    # DDP setup for multi-GPU
    use_ddp = num_devices > 1 and torch.cuda.is_available()
    if use_ddp:
        setup_ddp(args.local_rank, num_devices)

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Auto-compute global batch size
    per_device_batch = config["data"].get(
        "batch_size", config["training"].get("batch_size", 8)
    )
    global_batch = per_device_batch * num_devices
    logger.info(
        f"Batch size: {per_device_batch} per device x {num_devices} devices "
        f"= {global_batch} global"
    )

    # Override checkpoint dir if configured
    ckpt_base = config.get("checkpointing", {}).get("checkpoint_dir", "checkpoints")
    config.setdefault("checkpointing", {})["checkpoint_dir"] = os.path.join(
        ckpt_base, args.experiment, args.vm_name
    )
    logger.info(
        f"Checkpoints -> {config['checkpointing']['checkpoint_dir']}"
    )

    logger.info(f"Using {num_devices} device(s): {device}")
    logger.info(f"Config: {args.config}")

    # Create model
    arch = config["architecture"]
    mamba_cfg = arch.get("mamba", {})
    model = MBPSModel(
        num_classes=arch.get("num_classes", config["data"].get("num_classes", 27)),
        semantic_dim=arch.get("semantic_dim", 90),
        feature_dim=arch.get("backbone_dim", 384),
        bridge_dim=arch.get("bridge_dim", 192),
        max_instances=arch.get("max_instances", 100),
        mamba_layers=mamba_cfg.get("num_layers", 4),
        mamba_state_dim=mamba_cfg.get("state_dim", 64),
        chunk_size=mamba_cfg.get("chunk_size", 128),
        use_depth_conditioning=arch.get("use_depth_conditioning", True),
        use_mamba_bridge=arch.get("use_mamba_bridge", True),
        use_bidirectional=mamba_cfg.get("use_bidirectional", True),
    )
    model = model.to(device)

    # Wrap in DDP for multi-GPU
    if use_ddp:
        model = DDP(model, device_ids=[args.local_rank])

    # Create training data
    train_data = create_train_data(config)

    # Create trainer (handles optimizer internally)
    trainer = MBPSTrainer(
        config=config,
        model=model,
        device=device,
    )

    # Initialize EMA and training state
    trainer.create_train_state()

    # Resume from checkpoint if requested
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    # Initialize W&B if enabled
    if config.get("logging", {}).get("use_wandb", False):
        trainer.init_wandb(
            vm_name=args.vm_name,
            experiment_name=args.experiment,
        )

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {param_count:,}")
    logger.info(f"Trainable parameters: {trainable_count:,}")

    # Train
    trainer.train(train_loader=train_data)

    logger.info("Training complete!")

    # Cleanup DDP
    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
