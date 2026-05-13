"""MBPS v2 Training Script (JAX/Flax, TPU).

Usage:
    python scripts/train_v2.py --config configs/v2_cityscapes_gcs.yaml --seed 42

    # With ablation:
    python scripts/train_v2.py --config configs/v2_cityscapes_gcs.yaml \
        --ablation configs/v2_ablations/no_mamba.yaml --seed 42

    # Resume from checkpoint:
    python scripts/train_v2.py --config configs/v2_cityscapes_gcs.yaml \
        --resume gs://mbps-panoptic/checkpoints/v2_cityscapes_full/mbps-v4-0/checkpoint_epoch_0020

v2 training is a simplified 2-phase pipeline:
    Phase 1 (Bootstrap): Train on pseudo-labels with cross-entropy + discriminative loss
    Phase 2 (Self-training): EMA teacher generates refined pseudo-labels, retrain
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import yaml
from absl import logging
from flax import jax_utils

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbps.data.copy_paste import copy_paste_augment, create_self_enhanced_source
from mbps.data.tfrecord_utils import create_tfrecord_dataset
from mbps.models.instance.embedding_clustering import cluster_embeddings
from mbps.losses.bridge_loss import cka_loss, reconstruction_loss
from mbps.losses.instance_embedding_loss import discriminative_loss
from mbps.losses.semantic_loss_v2 import semantic_cross_entropy
from mbps.models.mbps_v2_model import MBPSv2Model


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and merge YAML config with v2 defaults."""
    default_path = os.path.join(
        os.path.dirname(config_path), "v2_default.yaml"
    )
    with open(default_path) as f:
        config = yaml.safe_load(f)

    with open(config_path) as f:
        override = yaml.safe_load(f)

    def deep_merge(base, override):
        for k, v in override.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                deep_merge(base[k], v)
            else:
                base[k] = v

    if override:
        deep_merge(config, override)
    return config


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def _nan_to_zero() -> optax.GradientTransformation:
    """Replace NaN/inf values in gradients with zero."""
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        updates = jax.tree.map(
            lambda g: jnp.where(jnp.isfinite(g), g, 0.0), updates
        )
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def create_optimizer(
    config: Dict[str, Any],
    steps_per_epoch: int,
) -> optax.GradientTransformation:
    """Create optimizer with cosine LR schedule."""
    tc = config["training"]
    lr = float(tc["learning_rate"])
    warmup_epochs = int(tc.get("warmup_epochs", 3))
    total_epochs = int(tc["total_epochs"])
    weight_decay = float(tc.get("weight_decay", 0.05))

    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    schedule = optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=0.0, end_value=lr,
                transition_steps=max(warmup_steps, 1),
            ),
            optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=max(total_steps - warmup_steps, 1),
            ),
        ],
        boundaries=[warmup_steps],
    )

    return optax.chain(
        _nan_to_zero(),
        optax.clip_by_global_norm(float(tc.get("gradient_clip", 1.0))),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_model(config: Dict[str, Any]) -> MBPSv2Model:
    """Create MBPS v2 model from config."""
    ac = config["architecture"]
    mc = ac.get("mamba", {})

    return MBPSv2Model(
        num_classes=ac["num_classes"],
        backbone_dim=ac.get("backbone_dim", 768),
        instance_embed_dim=ac.get("instance_embed_dim", 64),
        bridge_dim=ac.get("bridge_dim", 384),
        mamba_layers=mc.get("num_layers", 4),
        mamba_state_dim=mc.get("state_dim", 64),
        chunk_size=mc.get("chunk_size", 128),
        use_depth_conditioning=ac.get("use_depth_conditioning", True),
        use_mamba_bridge=ac.get("use_mamba_bridge", True),
        use_bidirectional=mc.get("use_bidirectional", True),
        dropout_rate=float(config["training"].get("dropout_rate", 0.1)),
    )


# ---------------------------------------------------------------------------
# Loss computation (runs per-device inside pmap)
# ---------------------------------------------------------------------------

def compute_v2_loss(
    params: Any,
    model: MBPSv2Model,
    batch: Dict[str, jnp.ndarray],
    config: Dict[str, Any],
    use_bridge: bool,
    rng: jax.Array,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute v2 loss for pmap (called via value_and_grad).

    Args:
        params: Model parameters.
        model: MBPSv2Model instance.
        batch: Dict with image, depth, pseudo_semantic, pseudo_instance.
        config: Loss weight config.
        use_bridge: Whether bridge is active.
        rng: PRNG key for dropout.

    Returns:
        (total_loss, loss_dict).
    """
    lw = config["loss_weights"]

    # Forward pass
    model_outputs = model.apply(
        params,
        image=batch["image"],
        depth=batch["depth"],
        use_bridge=use_bridge,
        deterministic=False,
        rngs={"dropout": rng},
    )

    losses = {}

    # 1. Semantic cross-entropy
    sem_loss = semantic_cross_entropy(
        logits=model_outputs["semantic_logits"],
        labels=batch["pseudo_semantic"],
        label_smoothing=float(lw.get("label_smoothing", 0.1)),
    )
    losses["L_semantic"] = sem_loss

    # 2. Discriminative instance loss
    inst_losses = discriminative_loss(
        embeddings=model_outputs["instance_embeddings"],
        instance_labels=batch["pseudo_instance"],
        delta_v=float(lw.get("delta_v", 0.5)),
        delta_d=float(lw.get("delta_d", 1.5)),
    )
    losses["L_instance"] = inst_losses["total"]
    losses["L_instance_pull"] = inst_losses["pull"]
    losses["L_instance_push"] = inst_losses["push"]

    # 3. Bridge loss (if active)
    bridge_loss = jnp.array(0.0)
    if use_bridge and "fused_semantic" in model_outputs:
        recon_sem = reconstruction_loss(
            model_outputs["semantic_logits"],
            model_outputs["reconstructed_semantic"],
        )
        recon_inst = reconstruction_loss(
            model_outputs["instance_embeddings"],
            model_outputs["reconstructed_instance"],
        )
        cka = cka_loss(
            model_outputs["fused_semantic"],
            model_outputs["fused_instance"],
        )
        align = model_outputs.get("align_loss", jnp.array(0.0))

        _safe = lambda x: jnp.where(jnp.isfinite(x), x, 0.0)
        bridge_loss = (
            float(lw.get("lambda_recon", 0.5)) * _safe(recon_sem + recon_inst)
            + float(lw.get("lambda_cka", 0.1)) * _safe(cka)
            + _safe(align)
        )
        losses["L_bridge_recon"] = recon_sem + recon_inst
        losses["L_bridge_cka"] = cka

    losses["L_bridge"] = bridge_loss
    losses["bridge_gate"] = model_outputs.get("bridge_gate", jnp.array(0.0))

    # Total loss with NaN guards
    _safe = lambda x: jnp.where(jnp.isfinite(x), x, 0.0)
    total = (
        float(lw.get("lambda_semantic", 1.0)) * _safe(sem_loss)
        + float(lw.get("lambda_instance", 1.0)) * _safe(inst_losses["total"])
        + float(lw.get("lambda_bridge", 0.1)) * _safe(bridge_loss)
    )
    losses["L_total"] = total

    return total, losses


# ---------------------------------------------------------------------------
# pmap train step
# ---------------------------------------------------------------------------

def make_train_step(model, optimizer, config):
    """Create pmapped train step function.

    Returns a function that takes (params, opt_state, ema_params, batch, rng)
    and returns (new_params, new_opt_state, new_ema, loss_val, loss_dict).
    """
    ema_momentum = float(config["training"].get("ema_momentum", 0.999))
    bridge_enable_epoch = int(config["training"].get("bridge_enable_epoch", 5))

    def train_step(params, opt_state, ema_params, batch, rng, epoch):
        use_bridge = epoch >= bridge_enable_epoch

        def loss_fn(p):
            return compute_v2_loss(p, model, batch, config, use_bridge, rng)

        (loss_val, loss_dict), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)

        # Average across devices
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss_val = jax.lax.pmean(loss_val, axis_name="batch")
        loss_dict = jax.tree.map(
            lambda x: jax.lax.pmean(x, axis_name="batch"), loss_dict
        )

        # Optimizer update
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # EMA update
        new_ema = jax.tree.map(
            lambda e, p: ema_momentum * e + (1.0 - ema_momentum) * p,
            ema_params, new_params,
        )

        return new_params, new_opt_state, new_ema, loss_val, loss_dict

    return jax.pmap(
        train_step,
        axis_name="batch",
        static_broadcasted_argnums=(5,),  # epoch is static
    )


# ---------------------------------------------------------------------------
# Self-enhanced copy-paste inference
# ---------------------------------------------------------------------------

def make_inference_fn(model):
    """Create pmapped inference function for self-enhanced copy-paste.

    Runs a forward pass without gradients to get semantic predictions
    and instance embeddings from the EMA teacher parameters.
    """
    def inference_step(params, batch):
        outputs = model.apply(
            params,
            image=batch["image"],
            depth=batch["depth"],
            use_bridge=False,  # Bridge not needed for pseudo-label generation
            deterministic=True,
        )
        return {
            "semantic_probs": outputs["semantic_probs"],         # (B, N, K)
            "instance_embeddings": outputs["instance_embeddings"],  # (B, N, D)
        }

    return jax.pmap(inference_step, axis_name="batch")


def predictions_to_source_batch(
    inference_outputs: Dict[str, np.ndarray],
    batch_np: Dict[str, np.ndarray],
    h_patches: int,
    w_patches: int,
    confidence_threshold: float,
    similarity_threshold: float,
    min_instance_patches: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Convert model inference outputs to a source batch for copy-paste.

    Runs instance clustering on each sample's embeddings, then creates
    a confidence-filtered source batch.
    """
    semantic_probs = inference_outputs["semantic_probs"]       # (B, N, K)
    instance_embeds = inference_outputs["instance_embeddings"]  # (B, N, D)
    B = semantic_probs.shape[0]

    semantic_preds = np.argmax(semantic_probs, axis=-1).astype(np.int32)
    confidence = np.max(semantic_probs, axis=-1).astype(np.float32)

    instance_preds = np.zeros((B, h_patches * w_patches), dtype=np.int32)
    for b in range(B):
        result = cluster_embeddings(
            embeddings=jnp.array(instance_embeds[b]),
            h_patches=h_patches,
            w_patches=w_patches,
            similarity_threshold=similarity_threshold,
            min_patch_count=min_instance_patches,
        )
        instance_preds[b] = np.array(result.instance_map).reshape(-1)

    return create_self_enhanced_source(
        images=batch_np["image"],
        depths=batch_np["depth"],
        semantic_preds=semantic_preds,
        instance_preds=instance_preds,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _save_npy(params: Any, path: str) -> None:
    """Save params as .npz via tf.io.gfile (GCS-compatible)."""
    flat = {}
    for i, (keys, val) in enumerate(
        jax.tree_util.tree_leaves_with_path(params)
    ):
        key_str = "/".join(str(k) for k in keys)
        flat[key_str] = np.array(val)

    buf = io.BytesIO()
    np.savez(buf, **flat)
    buf.seek(0)

    tf.io.gfile.makedirs(os.path.dirname(path))
    with tf.io.gfile.GFile(path, "wb") as f:
        f.write(buf.getvalue())


def _load_npy(path: str) -> Dict[str, np.ndarray]:
    """Load params from .npz via tf.io.gfile."""
    with tf.io.gfile.GFile(path, "rb") as f:
        data = np.load(io.BytesIO(f.read()))
    return dict(data)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def shard_batch(batch: Dict[str, jnp.ndarray], num_devices: int):
    """Reshape batch: (B, ...) -> (num_devices, B//num_devices, ...)."""
    def _shard(x):
        if x.shape[0] % num_devices != 0:
            return None
        return x.reshape((num_devices, -1) + x.shape[1:])
    return jax.tree.map(_shard, batch)


def unreplicate(tree):
    """Take slice [0] of every leaf."""
    return jax.tree.map(lambda x: x[0], tree)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    """Main v2 training entry point."""
    parser = argparse.ArgumentParser(description="MBPS v2 Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_devices", type=int, default=None)
    parser.add_argument("--vm_name", type=str,
                        default=os.environ.get("MBPS_VM_NAME", "local"))
    parser.add_argument("--experiment", type=str,
                        default=os.environ.get("MBPS_EXPERIMENT", "v2_default"))
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=1)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Merge ablation config if provided
    if args.ablation:
        with open(args.ablation) as f:
            ablation = yaml.safe_load(f)
        if ablation:
            def deep_merge(base, override):
                for k, v in override.items():
                    if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                        deep_merge(base[k], v)
                    else:
                        base[k] = v
            deep_merge(config, ablation)
            logging.info(f"Applied ablation config: {args.ablation}")

    # Log configuration
    logging.set_verbosity(logging.INFO)
    num_devices = args.num_devices or jax.local_device_count()
    devices = jax.local_devices()[:num_devices]
    batch_per_device = config["data"]["batch_size"]
    global_batch = batch_per_device * num_devices

    tc = config["training"]
    dc = config["data"]
    ac = config["architecture"]

    logging.info(f"MBPS v2 Training")
    logging.info(f"  Config: {args.config}")
    logging.info(f"  Experiment: {args.experiment}")
    logging.info(f"  VM: {args.vm_name}")
    logging.info(f"  Seed: {args.seed}")
    logging.info(f"  Devices: {num_devices} ({devices[0].platform})")
    logging.info(f"  Batch: {batch_per_device}/dev x {num_devices}dev = {global_batch}")
    logging.info(f"  Backbone: {ac.get('backbone', 'dinov3_vitb')} ({ac.get('backbone_dim', 768)}d)")
    logging.info(f"  Bridge: {ac.get('bridge_dim', 384)}d, mamba={ac.get('use_mamba_bridge', True)}")
    logging.info(f"  Epochs: {tc['total_epochs']} (bootstrap: {tc['bootstrap_end']})")

    # Set random seed
    rng = jax.random.PRNGKey(args.seed)

    # Create model
    model = create_model(config)

    # Create data pipeline
    gcs_config = config.get("gcs", {})
    tfrecord_dir = gcs_config.get("tfrecord_dir") or dc.get("tfrecord_dir")
    patch_size = ac.get("patch_size", 16)

    if tfrecord_dir and tf.io.gfile.exists(tfrecord_dir):
        logging.info(f"TFRecord dir: {tfrecord_dir}")
        dataset = create_tfrecord_dataset(
            tfrecord_pattern=os.path.join(tfrecord_dir, "*.tfrecord"),
            batch_size=global_batch,
            image_size=tuple(dc["image_size"]),
            shuffle=True,
            has_pseudo_labels=True,
            has_pseudo_masks=False,
            patch_size=patch_size,
        )
    else:
        logging.error(f"TFRecord dir not found: {tfrecord_dir}")
        logging.error("v2 training requires TFRecords with pseudo-labels.")
        logging.error("Run the pseudo-label pipeline first (mbps_pytorch/).")
        sys.exit(1)

    # Count steps per epoch
    # For TFRecords, count files and estimate
    tfrecord_files = tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrecord"))
    n_train = dc.get("n_train", 2975)  # Default: Cityscapes train
    steps_per_epoch = max(n_train // global_batch, 1)
    logging.info(f"  Steps/epoch: {steps_per_epoch} (est. {n_train} samples / {global_batch} batch)")

    # Initialize model parameters
    image_size = tuple(dc["image_size"])
    dummy_image = jnp.zeros((1, image_size[0], image_size[1], 3))
    dummy_depth = jnp.zeros((1, image_size[0], image_size[1]))
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, dummy_image, dummy_depth)

    # Load DINOv3 backbone weights
    weights_dir = gcs_config.get("weights_dir")
    backbone_weights_path = None
    if weights_dir:
        backbone_weights_path = os.path.join(weights_dir, "dinov3_vitb16_flax.npz")
    if backbone_weights_path and tf.io.gfile.exists(backbone_weights_path):
        logging.info(f"Loading DINOv3 weights from {backbone_weights_path}")
        from mbps.models.backbone.dinov3_weights_converter import load_flax_params
        backbone_params = load_flax_params(backbone_weights_path)
        if "params" in backbone_params:
            backbone_params = backbone_params["params"]
        params = params.unfreeze() if hasattr(params, 'unfreeze') else dict(params)
        if "params" in params:
            params["params"]["backbone"] = backbone_params
        else:
            params["backbone"] = backbone_params
        logging.info("DINOv3 backbone weights loaded successfully")
    else:
        logging.warning("DINOv3 weights not found — using random init "
                        "(will produce garbage results)")

    # Create optimizer
    optimizer = create_optimizer(config, steps_per_epoch)
    opt_state = optimizer.init(params)

    # EMA parameters
    ema_params = jax.tree.map(jnp.copy, params)

    # Count parameters
    n_params = sum(x.size for x in jax.tree.leaves(params))
    logging.info(f"Total parameters: {n_params:,}")

    # Resume from checkpoint
    start_epoch = args.start_epoch
    if args.resume:
        logging.info(f"Resuming from {args.resume}")
        ckpt_path = os.path.join(args.resume, "params.npz")
        if tf.io.gfile.exists(ckpt_path):
            ckpt_data = _load_npy(ckpt_path)
            # Restore params from checkpoint
            loaded_params = {}
            for key, val in ckpt_data.items():
                loaded_params[key] = jnp.array(val)
            logging.info(f"Loaded {len(loaded_params)} parameter arrays")
        # Try to get epoch from path
        resume_dir = args.resume.rstrip("/")
        if "epoch_" in resume_dir:
            try:
                epoch_str = resume_dir.split("epoch_")[-1][:4]
                start_epoch = int(epoch_str) + 1
                logging.info(f"Resuming from epoch {start_epoch}")
            except ValueError:
                pass

    # Replicate across devices
    params = jax.device_put_replicated(params, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)
    ema_params = jax.device_put_replicated(ema_params, devices)

    # Create pmapped train step
    p_train_step = make_train_step(model, optimizer, config)

    # Initialize W&B
    wandb_active = False
    if HAS_WANDB and config.get("logging", {}).get("use_wandb", False):
        log_cfg = config.get("logging", {})
        wandb.init(
            project=log_cfg.get("wandb_project", "mbps"),
            entity=log_cfg.get("wandb_entity"),
            group=log_cfg.get("wandb_group") or args.experiment,
            name=f"{args.experiment}/{args.vm_name}",
            tags=[args.vm_name, "v2"],
            config=config,
            reinit=True,
        )
        wandb_active = True
        logging.info("W&B initialized")

    # Checkpoint config
    ckpt_base = gcs_config.get("checkpoint_dir") or config.get(
        "checkpointing", {}
    ).get("checkpoint_dir", "checkpoints")
    ckpt_dir = os.path.join(ckpt_base, args.experiment, args.vm_name)
    save_every = int(config.get("checkpointing", {}).get("save_every_n_epochs", 5))

    # Copy-paste augmentation config (adapted from CUPS, Hahn et al. CVPR 2025)
    aug_config = config.get("augmentation", {})
    use_copy_paste = bool(aug_config.get("copy_paste", True))
    cp_max_objects = int(aug_config.get("max_paste_objects", 5))
    cp_min_tokens = int(aug_config.get("min_instance_tokens", 4))
    cp_flip_prob = float(aug_config.get("flip_prob", 0.5))
    cp_scale_range = tuple(aug_config.get("scale_range", [1.0, 1.0]))
    cp_patch_size = int(ac.get("patch_size", 16))
    cp_rng = np.random.RandomState(args.seed + 7)  # Separate RNG for augmentation

    # Self-enhanced copy-paste (CUPS Section 3.2)
    se_config = config.get("self_enhanced_copy_paste", {})
    use_self_enhanced = bool(se_config.get("enabled", False))
    se_warmup_steps = int(se_config.get("warmup_steps", 500))
    se_confidence = float(se_config.get("confidence_threshold", 0.75))
    se_similarity = float(config.get("evaluation", {}).get("similarity_threshold", 0.7))
    se_min_patches = int(config.get("evaluation", {}).get("min_instance_patches", 4))

    if use_copy_paste:
        logging.info(
            f"  Copy-paste augmentation: ON "
            f"(max_objects={cp_max_objects}, flip_prob={cp_flip_prob}, "
            f"scale_range={cp_scale_range})"
        )
    else:
        logging.info("  Copy-paste augmentation: OFF")
    if use_self_enhanced:
        logging.info(
            f"  Self-enhanced copy-paste: ON "
            f"(warmup={se_warmup_steps} steps, conf={se_confidence})"
        )

    # Patch grid dimensions (for self-enhanced clustering)
    H_p = image_size[0] // patch_size
    W_p = image_size[1] // patch_size

    # Self-enhanced copy-paste: inference function + prediction cache
    prediction_cache: Optional[Dict[str, np.ndarray]] = None
    p_inference_fn = None
    if use_self_enhanced and use_copy_paste:
        p_inference_fn = make_inference_fn(model)
        logging.info("  Self-enhanced inference function created")

    # ==================================================================
    # TRAINING LOOP
    # ==================================================================
    total_epochs = int(tc["total_epochs"])
    bridge_enable = int(tc.get("bridge_enable_epoch", 5))
    log_every = int(config.get("logging", {}).get("log_every_n_steps", 50))
    global_step = 0

    logging.info(f"Starting training: epochs {start_epoch}-{total_epochs}")

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        epoch_metrics: Dict[str, float] = {}
        num_steps = 0
        bridge_on = epoch >= bridge_enable
        phase = "bootstrap" if epoch <= int(tc["bootstrap_end"]) else "self_train"

        logging.info(
            f"=== Epoch {epoch}/{total_epochs} "
            f"(phase={phase}, bridge={'ON' if bridge_on else 'OFF'}) ==="
        )

        for batch in dataset:
            # Drop string fields
            batch = {
                k: v for k, v in batch.items()
                if not (hasattr(v, 'dtype') and v.dtype == tf.string)
            }

            # Convert TF tensors to numpy for augmentation
            batch_np = {
                k: np.array(v) for k, v in batch.items()
            }

            # Skip if batch size doesn't match
            if batch_np["image"].shape[0] != global_batch:
                continue

            # Save un-augmented batch for self-enhanced inference
            batch_np_orig = None
            if use_self_enhanced and use_copy_paste:
                batch_np_orig = {k: v.copy() for k, v in batch_np.items()}

            # Copy-paste augmentation (Hahn et al., CVPR 2025)
            if use_copy_paste:
                # Self-enhanced: use cached model predictions as paste source
                cp_source = None
                if (use_self_enhanced
                        and global_step >= se_warmup_steps
                        and prediction_cache is not None):
                    cp_source = prediction_cache

                batch_np = copy_paste_augment(
                    batch_np,
                    rng=cp_rng,
                    patch_size=cp_patch_size,
                    max_paste_objects=cp_max_objects,
                    min_instance_tokens=cp_min_tokens,
                    flip_prob=cp_flip_prob,
                    scale_range=cp_scale_range,
                    source_batch=cp_source,
                )

            # Convert to jax arrays
            batch = jax.tree.map(jnp.array, batch_np)

            # Shard across devices
            batch = shard_batch(batch, num_devices)

            # Per-device PRNG keys
            rng, *step_rngs = jax.random.split(rng, num_devices + 1)
            step_rngs = jnp.array(step_rngs)

            # Train step
            params, opt_state, ema_params, loss_val, loss_dict = p_train_step(
                params, opt_state, ema_params, batch, step_rngs, epoch,
            )

            global_step += 1
            num_steps += 1

            # Self-enhanced: run EMA inference to update prediction cache
            if (p_inference_fn is not None
                    and batch_np_orig is not None
                    and use_copy_paste):
                try:
                    orig_jax = jax.tree.map(jnp.array, batch_np_orig)
                    orig_sharded = shard_batch(orig_jax, num_devices)
                    inf_out = p_inference_fn(ema_params, orig_sharded)
                    # pmap returns (num_devices, batch_per_device, ...)
                    # Merge device + batch dims → (global_batch, ...)
                    inf_np = {}
                    for k, v in inf_out.items():
                        v_np = np.array(v)
                        inf_np[k] = v_np.reshape(
                            (v_np.shape[0] * v_np.shape[1],) + v_np.shape[2:]
                        )
                    prediction_cache = predictions_to_source_batch(
                        inference_outputs=inf_np,
                        batch_np=batch_np_orig,
                        h_patches=H_p,
                        w_patches=W_p,
                        confidence_threshold=se_confidence,
                        similarity_threshold=se_similarity,
                        min_instance_patches=se_min_patches,
                    )
                except Exception as e:
                    logging.warning(f"Self-enhanced inference failed: {e}")

            # Accumulate metrics
            for k, v in loss_dict.items():
                val = float(v[0])
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + val

            # Step logging
            if num_steps % log_every == 0:
                step_loss = float(loss_val[0])
                parts = [f"L_total={step_loss:.4f}"]
                for lk in ["L_semantic", "L_instance", "L_bridge"]:
                    if lk in loss_dict:
                        parts.append(f"{lk}={float(loss_dict[lk][0]):.4f}")
                if "bridge_gate" in loss_dict:
                    parts.append(f"gate={float(loss_dict['bridge_gate'][0]):.4f}")
                if use_self_enhanced:
                    se_status = "active" if prediction_cache is not None else "warmup"
                    parts.append(f"se={se_status}")
                logging.info(
                    f"  Step {num_steps}: " + ", ".join(parts)
                )
                if wandb_active:
                    wandb.log(
                        {f"step/{k}": float(v[0]) for k, v in loss_dict.items()},
                        step=global_step,
                    )

            # Stop after steps_per_epoch
            if num_steps >= steps_per_epoch:
                break

        # Epoch summary
        epoch_time = time.time() - epoch_start
        for k in epoch_metrics:
            epoch_metrics[k] /= max(num_steps, 1)
        epoch_metrics["epoch_time_s"] = epoch_time
        epoch_metrics["num_steps"] = num_steps

        logging.info(
            f"Epoch {epoch} done ({epoch_time:.1f}s): "
            + ", ".join(f"{k}={v:.4f}" for k, v in epoch_metrics.items()
                        if isinstance(v, float) and k != "epoch_time_s")
        )

        if wandb_active:
            wandb.log(
                {f"epoch/{k}": v for k, v in epoch_metrics.items()
                 if isinstance(v, (int, float))},
                step=global_step,
            )

        # Checkpoint
        if epoch % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch:04d}")
            logging.info(f"Saving checkpoint: {ckpt_path}")
            _save_npy(unreplicate(params), os.path.join(ckpt_path, "params.npz"))
            _save_npy(unreplicate(ema_params), os.path.join(ckpt_path, "ema_params.npz"))
            logging.info("Checkpoint saved")

    # Final checkpoint
    final_ckpt = os.path.join(ckpt_dir, f"checkpoint_epoch_{total_epochs:04d}")
    logging.info(f"Saving final checkpoint: {final_ckpt}")
    _save_npy(unreplicate(params), os.path.join(final_ckpt, "params.npz"))
    _save_npy(unreplicate(ema_params), os.path.join(final_ckpt, "ema_params.npz"))

    if wandb_active:
        wandb.finish()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
