"""MBPS Training Script.

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
import os
import sys
from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import yaml
from absl import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbps.data.datasets import get_dataset
from mbps.data.tfrecord_utils import create_tfrecord_dataset
from mbps.data.transforms import TrainTransform
from mbps.models.mbps_model import MBPSModel
from mbps.training.trainer import MBPSTrainer, TrainState, unreplicate


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


def _nan_to_zero() -> optax.GradientTransformation:
    """Replace NaN/inf values in gradients with zero.

    Prevents NaN from one loss component from corrupting all params.
    This is critical for Phase C where untrained bridge params can
    produce unstable gradients through the SSD matmul chain.
    """
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        updates = jax.tree.map(
            lambda g: jnp.where(jnp.isfinite(g), g, 0.0), updates
        )
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def create_optimizer(config: Dict[str, Any]) -> optax.GradientTransformation:
    """Create optimizer with learning rate schedule.

    Args:
        config: Training configuration.

    Returns:
        Optax optimizer.
    """
    lr = config["training"]["learning_rate"]
    warmup_epochs = config["training"].get("warmup_epochs", 5)
    total_epochs = config["training"]["total_epochs"]
    weight_decay = config["training"].get("weight_decay", 0.01)

    # Cosine schedule with warmup
    schedule = optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=0.0,
                end_value=lr,
                transition_steps=warmup_epochs * 100,  # approx steps
            ),
            optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=(total_epochs - warmup_epochs) * 100,
            ),
        ],
        boundaries=[warmup_epochs * 100],
    )

    optimizer = optax.chain(
        _nan_to_zero(),  # Replace NaN/inf grads with 0 before clipping
        optax.clip_by_global_norm(config["training"].get("gradient_clip", 1.0)),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=weight_decay,
        ),
    )

    return optimizer


def create_train_data(
    config: Dict[str, Any],
    num_shards: int = 1,
    shard_index: int = 0,
):
    """Create training data pipeline.

    Args:
        config: Data configuration.
        num_shards: Number of data shards (for multi-host training).
        shard_index: This host's shard index.

    Returns:
        TFRecord dataset iterator or Python dataset.
    """
    data_config = config["data"]
    training_config = config.get("training", {})
    tfrecord_dir = data_config.get("tfrecord_dir")
    batch_size = data_config.get("batch_size", training_config.get("batch_size", 8))

    # Check GCS config for tfrecord_dir override
    gcs_tfrecord = config.get("gcs", {}).get("tfrecord_dir")
    if gcs_tfrecord:
        tfrecord_dir = gcs_tfrecord

    # Pseudo mask config
    pseudo_mask_dir = data_config.get("pseudo_mask_dir")
    gcs_pseudo_mask = config.get("gcs", {}).get("pseudo_mask_dir")
    if gcs_pseudo_mask:
        pseudo_mask_dir = gcs_pseudo_mask
    has_pseudo_masks = pseudo_mask_dir is not None
    max_instances = config.get("architecture", {}).get("max_instances", 5)
    image_size = tuple(data_config["image_size"])

    if tfrecord_dir and tf.io.gfile.exists(tfrecord_dir):
        # Use TFRecord pipeline for TPU (works with local and gs:// paths)
        logging.info(f"Using TFRecord pipeline from {tfrecord_dir}")
        if has_pseudo_masks:
            logging.info(f"Pseudo masks enabled (from TFRecords)")

        tfrecord_pattern = os.path.join(tfrecord_dir, "*.tfrecord")
        dataset = create_tfrecord_dataset(
            tfrecord_pattern=tfrecord_pattern,
            batch_size=batch_size,
            image_size=image_size,
            shuffle=True,
            has_pseudo_masks=has_pseudo_masks,
            max_instances=max_instances,
            num_shards=num_shards,
            shard_index=shard_index,
        )
        return dataset
    else:
        # Fallback: Python dataset
        logging.info("Using Python dataset loader")
        if has_pseudo_masks:
            logging.info(f"Pseudo masks from {pseudo_mask_dir}")
        transform = TrainTransform(
            image_size=image_size,
            crop_size=tuple(data_config.get("crop_size", data_config["image_size"])),
        )
        dataset_name = data_config.get("dataset", data_config.get("dataset_name", "cityscapes"))
        dataset = get_dataset(
            dataset_name=dataset_name,
            data_dir=data_config["data_dir"],
            depth_dir=data_config.get("depth_dir", ""),
            split="train",
            image_size=image_size,
            transforms=transform,
            pseudo_mask_dir=pseudo_mask_dir,
            max_instances=max_instances,
        )
        return dataset


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
        help="Number of TPU/GPU devices (None = all available)",
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
        "--start_epoch",
        type=int,
        default=1,
        help="Epoch to resume training from (skips earlier epochs)",
    )
    parser.add_argument(
        "--coordinator_address",
        type=str,
        default=None,
        help="IP:port of coordinator for multi-host training (e.g. 10.0.0.1:1234). "
             "If not set, single-host mode is used.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Total number of hosts in multi-host training",
    )
    parser.add_argument(
        "--process_id",
        type=int,
        default=0,
        help="This host's process ID (0-indexed)",
    )
    args = parser.parse_args()

    # Initialize multi-host distributed runtime (MUST be before ANY jax calls)
    if args.coordinator_address is not None:
        jax.distributed.initialize(
            coordinator_address=args.coordinator_address,
            num_processes=args.num_processes,
            process_id=args.process_id,
        )
        logging.info(
            f"Multi-host initialized: process {jax.process_index()} of "
            f"{jax.process_count()}, {jax.local_device_count()} local devices, "
            f"{jax.device_count()} total devices"
        )

    # Setup
    logging.set_verbosity(logging.INFO)
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
        logging.info(f"Merged ablation config: {args.ablation}")

    num_devices = args.num_devices or jax.local_device_count()

    # Auto-compute global batch size
    per_device_batch = config["data"].get(
        "batch_size", config["training"].get("batch_size", 8)
    )
    global_batch = per_device_batch * jax.device_count()
    logging.info(
        f"Batch size: {per_device_batch} per device x {jax.device_count()} devices "
        f"({jax.process_count()} hosts x {jax.local_device_count()} local) "
        f"= {global_batch} global"
    )

    # Override checkpoint dir with GCS path if configured
    gcs_ckpt_dir = config.get("gcs", {}).get("checkpoint_dir")
    if gcs_ckpt_dir:
        config["checkpointing"]["checkpoint_dir"] = os.path.join(
            gcs_ckpt_dir, args.experiment, args.vm_name
        )
        logging.info(
            f"Checkpoints → GCS: {config['checkpointing']['checkpoint_dir']}"
        )

    logging.info(
        f"Process {jax.process_index()}: using {num_devices} local devices: "
        f"{jax.local_devices()}"
    )
    logging.info(f"Config: {args.config}")

    rng = jax.random.PRNGKey(args.seed)

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

    # Create optimizer
    optimizer = create_optimizer(config)

    # Create trainer (with multi-device pmap support)
    trainer = MBPSTrainer(
        config=config, model=model, optimizer=optimizer,
        num_devices=num_devices,
    )

    # Initialize W&B if enabled (rank-0 only in multi-host)
    if config.get("logging", {}).get("use_wandb", False) and jax.process_index() == 0:
        trainer.init_wandb(
            vm_name=args.vm_name,
            experiment_name=args.experiment,
        )

    # Create training data (with multi-host sharding)
    train_data = create_train_data(
        config,
        num_shards=jax.process_count(),
        shard_index=jax.process_index(),
    )

    # Initialize or restore state
    if args.resume:
        logging.info(f"Resuming from {args.resume}")
        ckpt = trainer.ckpt_manager.load(args.resume)

        # Init with random params to get tree structure
        rng, init_rng = jax.random.split(rng)
        dummy_input = {
            "image": jnp.zeros(
                (1,) + tuple(config["data"]["image_size"]) + (3,)
            ),
            "depth": jnp.zeros(
                (1,) + tuple(config["data"]["image_size"])
            ),
        }
        state = trainer.create_train_state(init_rng, dummy_input)

        # Restore checkpoint params into init tree structure
        from mbps.training.checkpointing import _flatten_pytree

        def _restore_params_from_flat(init_tree, flat_ckpt, prefix):
            """Replace init tree leaves with checkpoint values."""
            init_flat = _flatten_pytree(init_tree, prefix)
            init_leaves, treedef = jax.tree_util.tree_flatten(init_tree)
            flat_keys = list(init_flat.keys())
            assert len(flat_keys) == len(init_leaves), (
                f"Mismatch: {len(flat_keys)} keys vs {len(init_leaves)} leaves"
            )
            restored = []
            matched = 0
            for key, leaf in zip(flat_keys, init_leaves):
                if key in flat_ckpt:
                    restored.append(jnp.array(flat_ckpt[key]))
                    matched += 1
                else:
                    restored.append(leaf)
            logging.info(
                f"Restored {matched}/{len(init_leaves)} {prefix} from checkpoint"
            )
            return treedef.unflatten(restored)

        devices = jax.local_devices()[:num_devices]

        # Restore model params
        restored_params = _restore_params_from_flat(
            unreplicate(state.params), ckpt["params"], "params"
        )
        state.params = jax.device_put_replicated(restored_params, devices)

        # Restore EMA params if available
        if ckpt.get("ema_params"):
            restored_ema = _restore_params_from_flat(
                unreplicate(state.ema_params), ckpt["ema_params"], "ema"
            )
            state.ema_params = jax.device_put_replicated(restored_ema, devices)

        # Restore epoch
        meta = ckpt.get("metadata", {})
        if meta.get("epoch") and args.start_epoch == 1:
            args.start_epoch = meta["epoch"] + 1
            logging.info(f"Resuming from epoch {args.start_epoch}")

        logging.info("Checkpoint restored successfully")
    else:
        rng, init_rng = jax.random.split(rng)
        image_size = tuple(config["data"]["image_size"])
        dummy_input = {
            "image": jnp.zeros((1,) + image_size + (3,)),
            "depth": jnp.zeros((1,) + image_size),
        }
        state = trainer.create_train_state(init_rng, dummy_input)
        logging.info("Initialized model from scratch")

    # Count parameters (unreplicate to get single-device copy)
    param_count = sum(
        p.size for p in jax.tree.leaves(unreplicate(state.params))
    )
    logging.info(f"Total parameters: {param_count:,}")

    # For multi-host: compute fixed steps_per_epoch so all workers stay in sync.
    # Each worker's TF pipeline produces batches of `per_device_batch`.
    # With file-level sharding, workers may have uneven data — .repeat() +
    # fixed step count keeps them synchronized.
    steps_per_epoch = 0
    if jax.process_count() > 1:
        # Cityscapes: 2975 train images; COCO: ~118K
        # Each worker reads ~total/num_processes images.
        # steps = images_per_worker / per_device_batch
        total_images = config["data"].get("num_train_images", 2975)
        images_per_worker = total_images // jax.process_count()
        steps_per_epoch = max(1, images_per_worker // per_device_batch)
        logging.info(
            f"Multi-host: {steps_per_epoch} steps/epoch "
            f"({total_images} images / {jax.process_count()} workers / "
            f"{per_device_batch} batch)"
        )

    # Train
    final_state = trainer.train(
        state=state,
        train_data=train_data,
        rng=rng,
        start_epoch=args.start_epoch,
        steps_per_epoch=steps_per_epoch,
    )

    # Save final checkpoint (unreplicate from device axis, rank-0 only)
    if jax.process_index() == 0:
        trainer.ckpt_manager.save(
            epoch=config["training"]["total_epochs"],
            params=unreplicate(final_state.params),
            opt_state=unreplicate(final_state.opt_state),
            ema_params=unreplicate(final_state.ema_params),
        )
    logging.info(f"Training complete on process {jax.process_index()}!")


if __name__ == "__main__":
    main()
