#!/usr/bin/env python3
"""MBPS Baseline Runner for TPU.

Simple script to run a smoke test or short baseline training on TPUs.
Uses jax.pmap for data-parallel training across all available devices.

Usage:
    # Dry run (init model, run one forward pass, exit)
    python scripts/run_baseline.py --dry-run --config configs/cityscapes.yaml

    # Short smoke test (3 epochs with synthetic data)
    python scripts/run_baseline.py --config configs/cityscapes.yaml --epochs 3 --synthetic

    # Full baseline run (uses all TPUs)
    python scripts/run_baseline.py --config configs/cityscapes.yaml

    # Specify number of devices
    python scripts/run_baseline.py --config configs/cityscapes.yaml --num-devices 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from absl import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbps.models.mbps_model import MBPSModel
from mbps.losses import (
    SemanticLoss,
    InstanceLoss,
    BridgeLoss,
    ConsistencyLoss,
    PQProxyLoss,
)
from mbps.training.curriculum import TrainingCurriculum


# ---------------------------------------------------------------------------
# Multi-device utilities
# ---------------------------------------------------------------------------

def shard_array(x, num_devices):
    """Reshape (B, ...) to (num_devices, B // num_devices, ...)."""
    assert x.shape[0] % num_devices == 0, (
        f"Batch dim {x.shape[0]} not divisible by {num_devices}"
    )
    return x.reshape((num_devices, -1) + x.shape[1:])


def unreplicate(tree):
    """Take slice [0] of every leaf (device-0 copy)."""
    return jax.tree.map(lambda x: x[0], tree)


# ---------------------------------------------------------------------------
# Config / model helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and merge YAML config with defaults."""
    default_path = os.path.join(os.path.dirname(config_path), "default.yaml")
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

    deep_merge(config, override)
    return config


def create_synthetic_batch(config: Dict[str, Any], batch_size: int = 2):
    """Create a synthetic batch for testing pipeline."""
    image_size = tuple(config["data"]["image_size"])
    batch = {
        "image": np.random.randn(batch_size, image_size[0], image_size[1], 3).astype(
            np.float32
        ),
        "depth": np.random.randn(batch_size, image_size[0], image_size[1]).astype(
            np.float32
        ),
    }
    return {k: jnp.array(v) for k, v in batch.items()}


def print_device_info(num_devices: int = None):
    """Print JAX device information."""
    devices = jax.devices()
    num_devices = num_devices or len(devices)
    print("\n" + "=" * 60)
    print("  MBPS Baseline Runner — Device Info")
    print("=" * 60)
    print(f"  JAX version:      {jax.__version__}")
    print(f"  Backend:          {jax.default_backend()}")
    print(f"  Total devices:    {len(devices)}")
    print(f"  Using devices:    {num_devices}")
    for i in range(num_devices):
        print(f"    Device {i}:       {devices[i]}")
    if num_devices > 1:
        print(f"  Data parallelism: ENABLED (pmap across {num_devices} devices)")
    else:
        print(f"  Data parallelism: DISABLED (single device)")
    print("=" * 60 + "\n")


def create_model_from_config(config: Dict[str, Any]) -> MBPSModel:
    """Create MBPSModel from config dict."""
    arch = config["architecture"]
    mamba_cfg = arch.get("mamba", {})
    return MBPSModel(
        num_classes=arch.get(
            "num_classes", config["data"].get("num_classes", 27)
        ),
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


def dry_run(config: Dict[str, Any]):
    """Initialize model, run one forward pass, print info, and exit."""
    print("\n  DRY RUN: Initializing model and running one forward pass...\n")

    model = create_model_from_config(config)
    print(f"  Model created: {model.__class__.__name__}")
    print(f"    num_classes:     {model.num_classes}")
    print(f"    semantic_dim:    {model.semantic_dim}")
    print(f"    feature_dim:     {model.feature_dim}")
    print(f"    bridge_dim:      {model.bridge_dim}")
    print(f"    max_instances:   {model.max_instances}")
    print(f"    mamba_layers:    {model.mamba_layers}")
    print(f"    mamba_state_dim: {model.mamba_state_dim}")
    print(f"    chunk_size:      {model.chunk_size}")
    print(f"    depth_cond:      {model.use_depth_conditioning}")
    print(f"    mamba_bridge:    {model.use_mamba_bridge}")
    print(f"    bidirectional:   {model.use_bidirectional}")

    # Use smaller image for dry-run to save memory
    image_size = tuple(config["data"]["image_size"])
    dry_h = min(image_size[0], 256)
    dry_w = min(image_size[1], 512)

    print(f"\n  Creating dummy input: ({dry_h}, {dry_w})...")
    rng = jax.random.PRNGKey(0)
    dummy_image = jnp.zeros((1, dry_h, dry_w, 3))
    dummy_depth = jnp.zeros((1, dry_h, dry_w))

    print("  Initializing model parameters...")
    t0 = time.time()
    params = model.init(
        rng, image=dummy_image, depth=dummy_depth, use_bridge=True, deterministic=True
    )
    init_time = time.time() - t0

    param_count = sum(p.size for p in jax.tree.leaves(params))
    print(f"  Total parameters: {param_count:,}")
    print(f"  Init time:        {init_time:.2f}s")

    # Forward pass
    print("\n  Running forward pass...")
    t0 = time.time()
    outputs = model.apply(
        params, image=dummy_image, depth=dummy_depth, use_bridge=True, deterministic=True
    )
    fwd_time = time.time() - t0

    print(f"  Forward time:     {fwd_time:.2f}s")
    print("\n  Output shapes:")
    for k, v in sorted(outputs.items()):
        if isinstance(v, jnp.ndarray):
            print(f"    {k:30s} {str(v.shape):20s} ({v.dtype})")
        else:
            print(f"    {k:30s} {v}")

    print("\n  Dry run complete! Model is functional.\n")
    return True


def compute_mbps_loss(
    model, params, image, depth, epoch, rng,
    curriculum, semantic_loss_fn, instance_loss_fn,
    bridge_loss_fn, consistency_loss_fn, pq_loss_fn,
    num_classes, ema_params=None,
):
    """Compute the full MBPS loss with all components and curriculum weights.

    Returns (total_loss, loss_dict) where loss_dict contains per-component values.
    """
    phase_config = curriculum.get_config(epoch)
    weights = curriculum.get_loss_weights(epoch)

    # Forward pass
    model_output = model.apply(
        params,
        image=image,
        depth=depth,
        use_bridge=phase_config.use_bridge,
        deterministic=False,
        rngs={"dropout": rng},
    )

    loss_dict = {}
    total_loss = jnp.array(0.0)

    # Downsample depth to token resolution (matching backbone output grid)
    patch_size = 8
    b = depth.shape[0]
    spatial_h = image.shape[1] // patch_size
    spatial_w = image.shape[2] // patch_size
    # Average-pool depth from (B, H, W) -> (B, h_tokens, w_tokens) -> (B, N)
    depth_resized = jax.image.resize(
        depth[:, :, :, None],  # (B, H, W, 1)
        (b, spatial_h, spatial_w, 1),
        method="bilinear",
    )[:, :, :, 0]  # (B, h_tokens, w_tokens)
    depth_tokens = depth_resized.reshape(b, -1)  # (B, N) at token resolution

    # -- L_semantic (always active) --
    sem_losses = semantic_loss_fn(
        semantic_codes=model_output["semantic_codes"],
        dino_features=model_output["dino_features"],
        depth=depth_tokens,
        key=rng,
    )
    loss_dict["L_semantic"] = sem_losses["total"]
    total_loss += weights["alpha"] * sem_losses["total"]

    # -- L_instance (Phase B onwards) --
    if weights["beta"] > 0:
        inst_losses = instance_loss_fn(
            pred_masks=model_output["instance_masks"],
            pred_scores=model_output["instance_scores"],
            features=model_output["dino_features"],
        )
        loss_dict["L_instance"] = inst_losses["total"]
        total_loss += weights["beta"] * inst_losses["total"]
    else:
        loss_dict["L_instance"] = jnp.array(0.0)

    # -- L_bridge (Phase C onwards) --
    if weights["gamma"] > 0 and "fused_semantic" in model_output:
        bridge_losses = bridge_loss_fn(
            original_semantic=model_output["semantic_codes"],
            original_features=model_output["dino_features"],
            reconstructed_semantic=model_output.get(
                "reconstructed_semantic", model_output["semantic_codes"]
            ),
            reconstructed_features=model_output.get(
                "reconstructed_features", model_output["dino_features"]
            ),
            fused_semantic=model_output["fused_semantic"],
            fused_features=model_output["fused_features"],
            align_loss=model_output.get("align_loss", jnp.array(0.0)),
        )
        loss_dict["L_bridge"] = bridge_losses["total"]
        total_loss += weights["gamma"] * bridge_losses["total"]
    else:
        loss_dict["L_bridge"] = jnp.array(0.0)

    # -- L_consistency (Phase C onwards) --
    if weights["delta"] > 0:
        sem_pred = jnp.argmax(model_output["semantic_codes"], axis=-1)
        mask_probs = jax.nn.sigmoid(model_output["instance_masks"])

        cons_losses = consistency_loss_fn(
            semantic_pred=sem_pred,
            instance_masks=mask_probs,
            depth=depth_tokens,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
        )
        loss_dict["L_consistency"] = cons_losses["total"]
        total_loss += weights["delta"] * cons_losses["total"]
    else:
        loss_dict["L_consistency"] = jnp.array(0.0)

    # -- L_pq (Phase C, needs EMA teacher) --
    if weights["epsilon"] > 0 and ema_params is not None:
        teacher_output = model.apply(
            ema_params,
            image=image,
            depth=depth,
            use_bridge=True,
            deterministic=True,
        )
        pq_losses = pq_loss_fn(
            pred_masks=model_output["instance_masks"],
            pred_scores=model_output["instance_scores"],
            teacher_masks=teacher_output["instance_masks"],
            teacher_scores=teacher_output["instance_scores"],
        )
        loss_dict["L_pq"] = pq_losses["total"]
        total_loss += weights["epsilon"] * pq_losses["total"]
    else:
        loss_dict["L_pq"] = jnp.array(0.0)

    loss_dict["L_total"] = total_loss
    return total_loss, loss_dict


def smoke_test(config: Dict[str, Any], epochs: int = 3, batch_size: int = 2,
               use_real_data: bool = False, num_devices: int = 1):
    """Run a short training loop with proper MBPS losses and pmap parallelism.

    Uses:
        - TrainingCurriculum for phase transitions (A->B->C)
        - SemanticLoss (STEGO + DepthG)
        - InstanceLoss (unsupervised mask coherence)
        - BridgeLoss (reconstruction + CKA + state reg)
        - ConsistencyLoss (uniformity + boundary + DBC)
        - PQProxyLoss (differentiable PQ with EMA teacher)
        - jax.pmap for data-parallel training across all devices
    """
    devices = jax.devices()[:num_devices]
    global_batch_size = batch_size * num_devices
    data_mode = "REAL DATA" if use_real_data else "synthetic data"

    print(f"\n  SMOKE TEST: Running {epochs} epoch(s) with {data_mode}...")
    print(f"  Data parallelism: {num_devices} device(s)")
    print(f"  Per-device batch: {batch_size}, Global batch: {global_batch_size}\n")

    # -- Load real dataset if requested --
    dataset = None
    if use_real_data:
        try:
            from mbps.data.datasets import get_dataset, collate_batch
            data_cfg = config["data"]
            dataset = get_dataset(
                dataset_name=data_cfg["dataset"],
                data_dir=data_cfg["data_dir"],
                depth_dir=data_cfg["depth_dir"],
                split="train",
                image_size=tuple(data_cfg["image_size"]),
                subset_fraction=data_cfg.get("subset_fraction"),
            )
            print(f"  Loaded dataset: {data_cfg['dataset']} ({len(dataset)} samples)")
        except Exception as e:
            print(f"  Failed to load dataset: {e}")
            print("  Falling back to synthetic data")
            dataset = None

    model = create_model_from_config(config)
    num_classes = config["architecture"].get(
        "num_classes", config["data"].get("num_classes", 27)
    )

    # -- Curriculum --
    train_cfg = config["training"]
    curriculum = TrainingCurriculum(
        phase_a_end=train_cfg.get("phase_a_end", 2),
        phase_b_end=train_cfg.get("phase_b_end", 4),
        total_epochs=train_cfg.get("total_epochs", epochs),
    )

    # -- Loss functions --
    semantic_loss_fn = SemanticLoss()
    instance_loss_fn = InstanceLoss()
    bridge_loss_fn = BridgeLoss()
    consistency_loss_fn = ConsistencyLoss(num_classes=num_classes)
    pq_loss_fn = PQProxyLoss()

    # -- Optimizer --
    lr = train_cfg["learning_rate"]
    optimizer = optax.chain(
        optax.clip_by_global_norm(train_cfg.get("gradient_clip", 1.0)),
        optax.adamw(
            learning_rate=lr,
            weight_decay=train_cfg.get("weight_decay", 0.01),
        ),
    )

    # -- Initialization --
    image_size = tuple(config["data"]["image_size"])
    if dataset is not None:
        test_h, test_w = image_size[0], image_size[1]
    else:
        test_h = min(image_size[0], 128)
        test_w = min(image_size[1], 256)
    rng = jax.random.PRNGKey(42)

    dummy_image = jnp.zeros((batch_size, test_h, test_w, 3))
    dummy_depth = jnp.zeros((batch_size, test_h, test_w))

    print(f"  Image size: ({test_h}, {test_w})")
    print(f"  Phases: A(1-{curriculum.phase_a_end}), "
          f"B({curriculum.phase_a_end+1}-{curriculum.phase_b_end}), "
          f"C({curriculum.phase_b_end+1}-{epochs})")
    print("  Initializing model...")

    rng, init_rng = jax.random.split(rng)
    params = model.init(
        init_rng, image=dummy_image, depth=dummy_depth,
        use_bridge=True, deterministic=True,
    )
    opt_state = optimizer.init(params)
    ema_params = jax.tree.map(jnp.copy, params)  # EMA copy

    param_count = sum(p.size for p in jax.tree.leaves(params))
    print(f"  Parameters: {param_count:,}")

    # -- Replicate across devices for pmap --
    params = jax.device_put_replicated(params, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)
    ema_params = jax.device_put_replicated(ema_params, devices)
    print(f"  Parameters replicated to {num_devices} device(s)")

    # -- Define pmapped train step --
    ema_decay = 0.999

    def train_step_fn(params, opt_state, ema_params, image, depth, epoch, rng):
        """Single train step executed on each device via pmap."""
        def loss_wrapper(p):
            return compute_mbps_loss(
                model, p, image, depth, epoch, rng,
                curriculum, semantic_loss_fn, instance_loss_fn,
                bridge_loss_fn, consistency_loss_fn, pq_loss_fn,
                num_classes, ema_params=ema_params,
            )

        (loss_val, loss_dict), grads = jax.value_and_grad(
            loss_wrapper, has_aux=True
        )(params)

        # Average gradients and losses across devices
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
            lambda e, p: ema_decay * e + (1 - ema_decay) * p,
            ema_params, new_params,
        )

        return new_params, new_opt_state, new_ema, loss_val, loss_dict

    p_train_step = jax.pmap(
        train_step_fn, axis_name="batch", static_broadcasted_argnums=(5,)
    )

    # -- Training loop --
    if dataset is not None:
        steps_per_epoch = max(1, len(dataset) // global_batch_size)
        steps_per_epoch = min(steps_per_epoch, 50)
    else:
        steps_per_epoch = 5
    total_steps = epochs * steps_per_epoch

    print(f"\n  Training: {epochs} epoch(s) x {steps_per_epoch} steps = {total_steps} steps")
    print("  " + "=" * 80)
    print(f"  {'Epoch':>5s} | {'Phase':>5s} | {'L_total':>9s} | {'L_sem':>9s} | "
          f"{'L_inst':>9s} | {'L_bridge':>9s} | {'L_cons':>9s} | {'L_pq':>9s} | "
          f"{'a':>4s} {'b':>5s} {'g':>5s} {'d':>5s} {'e':>5s} | {'Time':>6s}")
    print("  " + "-" * 80)

    # Pre-generate batch indices for real data
    data_indices = None
    if dataset is not None:
        from mbps.data.datasets import collate_batch
        data_indices = list(range(len(dataset)))

    total_time = 0
    for epoch in range(1, epochs + 1):
        phase = curriculum.get_phase(epoch)
        weights = curriculum.get_loss_weights(epoch)
        epoch_losses = {}
        t_epoch = time.time()

        # Shuffle data indices each epoch
        if data_indices is not None:
            rng_np = np.random.RandomState(epoch)
            rng_np.shuffle(data_indices)

        for step in range(steps_per_epoch):
            rng, data_rng = jax.random.split(rng)

            if dataset is not None:
                # Load global batch from real data
                start_idx = (step * global_batch_size) % len(dataset)
                batch_indices = [
                    data_indices[i % len(data_indices)]
                    for i in range(start_idx, start_idx + global_batch_size)
                ]
                samples = [dataset[i] for i in batch_indices]
                batch = collate_batch(samples)
                image = jnp.array(batch["image"])
                depth = jnp.array(batch["depth"])
            else:
                # Generate random global batch
                image = jax.random.normal(
                    data_rng, (global_batch_size, test_h, test_w, 3)
                )
                depth = jnp.abs(jax.random.uniform(
                    data_rng, (global_batch_size, test_h, test_w)
                ))

            # Shard across devices: (global_batch, ...) -> (num_devices, batch, ...)
            image = shard_array(image, num_devices)
            depth = shard_array(depth, num_devices)

            # Per-device PRNG keys
            rng, *step_rngs = jax.random.split(rng, num_devices + 1)
            step_rngs = jnp.array(step_rngs)

            # Pmapped train step
            params, opt_state, ema_params, loss_val, loss_dict = p_train_step(
                params, opt_state, ema_params, image, depth, epoch, step_rngs
            )

            # Accumulate losses (from device 0)
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + float(v[0])

        epoch_time = time.time() - t_epoch
        total_time += epoch_time

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= steps_per_epoch

        print(
            f"  {epoch:>5d} | {phase:>5s} | "
            f"{epoch_losses.get('L_total', 0):>9.4f} | "
            f"{epoch_losses.get('L_semantic', 0):>9.4f} | "
            f"{epoch_losses.get('L_instance', 0):>9.4f} | "
            f"{epoch_losses.get('L_bridge', 0):>9.4f} | "
            f"{epoch_losses.get('L_consistency', 0):>9.4f} | "
            f"{epoch_losses.get('L_pq', 0):>9.4f} | "
            f"{weights['alpha']:>4.1f} {weights['beta']:>5.2f} "
            f"{weights['gamma']:>5.2f} {weights['delta']:>5.2f} "
            f"{weights['epsilon']:>5.2f} | "
            f"{epoch_time:>5.1f}s"
        )

    print("  " + "=" * 80)
    print(
        f"  Total time: {total_time:.1f}s, "
        f"avg step: {total_time / total_steps:.2f}s"
    )

    # ================================================================
    # EVALUATION: Compute panoptic segmentation metrics
    # ================================================================
    # Unreplicate params for evaluation
    eval_params = unreplicate(params)

    print("\n" + "=" * 60)
    print(f"  Evaluation Metrics (on {data_mode})")
    print("=" * 60)

    eval_metrics = evaluate_model(
        model, eval_params, config, num_classes,
        test_h, test_w, batch_size, rng,
        dataset=dataset, num_devices=num_devices,
    )

    # Print standard metrics table
    print("\n  +-----------------------------------------------------+")
    print("  |         Panoptic Segmentation Metrics               |")
    print("  +-------------+-------------------------+-------------+")
    print("  | Metric      | Description             | Value       |")
    print("  +-------------+-------------------------+-------------+")
    print(f"  | PQ          | Panoptic Quality        | {eval_metrics['PQ']:>9.2f}%  |")
    print(f"  | PQ^Th       | PQ (thing classes)      | {eval_metrics['PQ_Th']:>9.2f}%  |")
    print(f"  | PQ^St       | PQ (stuff classes)      | {eval_metrics['PQ_St']:>9.2f}%  |")
    print(f"  | SQ          | Segmentation Quality    | {eval_metrics['SQ']:>9.2f}%  |")
    print(f"  | RQ          | Recognition Quality     | {eval_metrics['RQ']:>9.2f}%  |")
    print("  +-------------+-------------------------+-------------+")
    print(f"  | mIoU        | Mean IoU (Hungarian)    | {eval_metrics['mIoU']:>9.2f}%  |")
    print(f"  | Accuracy    | Pixel accuracy          | {eval_metrics['Accuracy']:>9.2f}%  |")
    print("  +-------------+-------------------------+-------------+")

    # Per-class IoU breakdown (top-5)
    if "per_class_iou" in eval_metrics and eval_metrics["per_class_iou"] is not None:
        per_class = eval_metrics["per_class_iou"]
        sorted_classes = np.argsort(per_class)[::-1]
        print("\n  Top-5 classes by IoU:")
        for i, c in enumerate(sorted_classes[:5]):
            print(f"    Class {c:>3d}: IoU = {per_class[c]*100:.2f}%")

    print(f"\n  Eval time: {eval_metrics.get('eval_time', 0):.2f}s")

    # Write metrics as JSON
    import json
    metrics_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
    )
    os.makedirs(metrics_path, exist_ok=True)
    metrics_file = os.path.join(metrics_path, "smoke_test_metrics.json")
    json_metrics = {k: v for k, v in eval_metrics.items() if k != "per_class_iou"}
    if "per_class_iou" in eval_metrics and eval_metrics["per_class_iou"] is not None:
        json_metrics["per_class_iou"] = eval_metrics["per_class_iou"].tolist()
    with open(metrics_file, "w") as f:
        json.dump(json_metrics, f, indent=2)
    print(f"  Metrics saved to: {metrics_file}")

    print("\n  Smoke test complete!\n")
    return True


def evaluate_model(
    model, params, config, num_classes,
    img_h, img_w, batch_size, rng,
    dataset=None, num_devices=1,
):
    """Run evaluation and compute standard panoptic/semantic/instance metrics.

    Uses jax.pmap for parallel forward passes across devices.

    Computes:
        - PQ, PQ^Th, PQ^St, SQ, RQ (panoptic segmentation)
        - mIoU, Accuracy (semantic segmentation via Hungarian matching)

    Args:
        dataset: Optional dataset object. When provided, loads real images,
            depth, and semantic labels for evaluation. Otherwise uses
            synthetic random data.
        num_devices: Number of devices for parallel forward passes.
    """
    from mbps.evaluation.panoptic_quality import compute_panoptic_quality
    from mbps.evaluation.hungarian_matching import hungarian_match, compute_miou

    t0 = time.time()

    # -- Multi-device setup --
    if num_devices > 1:
        devices = jax.devices()[:num_devices]
        r_params = jax.device_put_replicated(params, devices)

        @jax.pmap
        def p_forward(p, image, depth):
            return model.apply(
                p, image=image, depth=depth,
                use_bridge=True, deterministic=True,
            )

    global_eval_bs = batch_size * num_devices

    # Class definitions
    dataset_name = config["data"].get("dataset", "")
    if dataset_name == "nyu_depth_v2":
        stuff_classes = set(range(num_classes))
        thing_classes = set()
    elif dataset_name == "pascal_voc":
        stuff_classes = {0}
        thing_classes = set(range(1, num_classes))
    else:
        thing_classes = set(range(11, min(num_classes, 19)))
        stuff_classes = set(range(0, min(11, num_classes)))

    patch_size = 8
    h_tokens = img_h // patch_size
    w_tokens = img_w // patch_size

    # -- Prepare evaluation batches --
    if dataset is not None:
        from mbps.data.datasets import collate_batch
        n_eval = min(len(dataset), 50)
        eval_indices = list(range(n_eval))
        n_batches = max(1, (n_eval + global_eval_bs - 1) // global_eval_bs)
        print(f"  Evaluating on {n_eval} real samples ({n_batches} batches, "
              f"{global_eval_bs} samples/batch across {num_devices} devices)...")
    else:
        n_batches = 1
        print(f"  Evaluating on synthetic data ({global_eval_bs} samples "
              f"across {num_devices} devices)...")

    all_pq, all_sq, all_rq = [], [], []
    all_pq_th, all_pq_st = [], []
    all_miou, all_acc = [], []
    all_per_class = []

    for batch_idx in range(n_batches):
        # -- Load data --
        if dataset is not None:
            start = batch_idx * global_eval_bs
            end = min(start + global_eval_bs, n_eval)
            actual_bs = end - start
            if actual_bs <= 0:
                break

            samples = [dataset[eval_indices[i]] for i in range(start, end)]
            # Pad to global_eval_bs for even sharding
            while len(samples) < global_eval_bs:
                samples.append(samples[-1])

            batch = collate_batch(samples)
            eval_image = jnp.array(batch["image"])
            eval_depth = jnp.array(batch["depth"])

            # Get real semantic labels and downsample to token resolution
            if "semantic_label" in batch:
                gt_sem_full = batch["semantic_label"]
                gt_semantic = np.zeros((actual_bs, h_tokens * w_tokens), dtype=np.int32)
                for b in range(actual_bs):
                    label_2d = gt_sem_full[b]
                    for ty in range(h_tokens):
                        for tx in range(w_tokens):
                            cy = min(ty * patch_size + patch_size // 2, label_2d.shape[0] - 1)
                            cx = min(tx * patch_size + patch_size // 2, label_2d.shape[1] - 1)
                            gt_semantic[b, ty * w_tokens + tx] = label_2d[cy, cx]
                has_real_labels = True
            else:
                np.random.seed(42 + batch_idx)
                gt_semantic = np.random.randint(0, num_classes, (actual_bs, h_tokens * w_tokens))
                has_real_labels = False
        else:
            rng, eval_rng, data_rng = jax.random.split(rng, 3)
            eval_image = jax.random.normal(
                data_rng, (global_eval_bs, img_h, img_w, 3)
            )
            eval_depth = jnp.abs(jax.random.uniform(
                data_rng, (global_eval_bs, img_h, img_w)
            ))
            actual_bs = global_eval_bs
            np.random.seed(42)
            gt_semantic = np.random.randint(
                0, num_classes, (global_eval_bs, h_tokens * w_tokens)
            )
            has_real_labels = False

        # -- Forward pass (with pmap if num_devices > 1) --
        if num_devices > 1:
            img_sharded = shard_array(eval_image, num_devices)
            dep_sharded = shard_array(eval_depth, num_devices)
            outputs = p_forward(r_params, img_sharded, dep_sharded)
            # Gather: (num_devices, per_dev_bs, ...) -> (global_eval_bs, ...)
            outputs = jax.tree.map(
                lambda x: x.reshape((-1,) + x.shape[2:]), outputs
            )
        else:
            outputs = model.apply(
                params,
                image=eval_image[:actual_bs],
                depth=eval_depth[:actual_bs],
                use_bridge=True,
                deterministic=True,
            )

        # -- Semantic predictions --
        semantic_codes = np.array(outputs["semantic_codes"])  # (B, N, C)
        semantic_pred_tokens = np.argmax(semantic_codes, axis=-1)  # (B, N)

        # -- Instance predictions --
        instance_masks = np.array(jax.nn.sigmoid(outputs["instance_masks"]))  # (B, M, N)
        instance_scores = np.array(outputs["instance_scores"])  # (B, M)

        for b in range(actual_bs):
            # -- Semantic metrics (Hungarian matching) --
            pred_flat = semantic_pred_tokens[b]
            gt_flat = gt_semantic[b]

            mapping, accuracy = hungarian_match(
                pred_labels=pred_flat,
                gt_labels=gt_flat,
                num_pred_clusters=num_classes,
                num_gt_classes=num_classes,
            )
            all_acc.append(accuracy)

            miou, per_class_iou = compute_miou(
                pred_labels=pred_flat,
                gt_labels=gt_flat,
                mapping=mapping,
                num_classes=num_classes,
            )
            all_miou.append(miou)
            all_per_class.append(per_class_iou)

            # -- Panoptic segmentation metrics --
            label_divisor = 1000
            score_threshold = 0.1

            # Create panoptic prediction from semantic + instance
            pred_panoptic = np.zeros((h_tokens * w_tokens,), dtype=np.int64)
            pred_segments = []

            # Map predicted clusters to GT classes via Hungarian
            mapped_pred = np.full_like(pred_flat, fill_value=255)
            for pred_c, gt_c in mapping.items():
                mapped_pred[pred_flat == pred_c] = gt_c

            seg_id = 1
            for c in stuff_classes:
                mask = mapped_pred == c
                if np.sum(mask) > 0:
                    panoptic_id = c * label_divisor + seg_id
                    pred_panoptic[mask] = panoptic_id
                    pred_segments.append({
                        "id": panoptic_id,
                        "category_id": c,
                        "isthing": False,
                    })
                    seg_id += 1

            # Add thing instances from instance head
            for m in range(instance_masks.shape[1]):
                if instance_scores[b, m] < score_threshold:
                    continue
                mask = instance_masks[b, m] > 0.5
                if np.sum(mask) == 0:
                    continue

                masked_sem = mapped_pred[mask]
                if masked_sem.size == 0:
                    continue
                valid_sem = masked_sem[masked_sem != 255].astype(int)
                if valid_sem.size == 0:
                    continue
                cat_id = int(np.bincount(valid_sem, minlength=num_classes).argmax())
                if cat_id not in thing_classes:
                    continue

                panoptic_id = cat_id * label_divisor + seg_id
                pred_panoptic[mask] = panoptic_id
                pred_segments.append({
                    "id": panoptic_id,
                    "category_id": cat_id,
                    "isthing": True,
                })
                seg_id += 1

            # Build GT panoptic
            gt_panoptic = np.zeros_like(pred_panoptic)
            gt_segments = []
            gt_seg_id = 1
            for c in stuff_classes | thing_classes:
                mask = gt_flat == c
                if np.sum(mask) > 0:
                    pan_id = c * label_divisor + gt_seg_id
                    gt_panoptic[mask] = pan_id
                    gt_segments.append({
                        "id": pan_id,
                        "category_id": c,
                        "isthing": c in thing_classes,
                    })
                    gt_seg_id += 1

            # Compute PQ
            pq_result = compute_panoptic_quality(
                pred_panoptic=pred_panoptic,
                gt_panoptic=gt_panoptic,
                pred_segments=pred_segments,
                gt_segments=gt_segments,
                thing_classes=thing_classes,
                stuff_classes=stuff_classes,
                iou_threshold=0.5,
                label_divisor=label_divisor,
            )
            all_pq.append(pq_result.pq)
            all_sq.append(pq_result.sq)
            all_rq.append(pq_result.rq)
            all_pq_th.append(pq_result.pq_things)
            all_pq_st.append(pq_result.pq_stuff)

        if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
            print(f"    Eval batch {batch_idx + 1}/{n_batches} | "
                  f"mIoU so far: {np.mean(all_miou)*100:.2f}%")

    eval_time = time.time() - t0

    # Average across all samples
    avg_per_class = np.mean(all_per_class, axis=0) if all_per_class else None

    return {
        "PQ": float(np.mean(all_pq)) * 100,
        "PQ_Th": float(np.mean(all_pq_th)) * 100,
        "PQ_St": float(np.mean(all_pq_st)) * 100,
        "SQ": float(np.mean(all_sq)) * 100,
        "RQ": float(np.mean(all_rq)) * 100,
        "mIoU": float(np.mean(all_miou)) * 100,
        "Accuracy": float(np.mean(all_acc)) * 100,
        "per_class_iou": avg_per_class,
        "eval_time": eval_time,
        "n_samples": len(all_miou),
        "has_real_labels": has_real_labels if dataset is not None else False,
    }


def main():
    parser = argparse.ArgumentParser(description="MBPS Baseline Runner")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Init model, run one forward pass, print info, exit",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for smoke test (no real data needed)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs for smoke test (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size for smoke test (default: 2)",
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        default=None,
        help="Number of TPU/GPU devices (default: all available)",
    )
    args = parser.parse_args()

    logging.set_verbosity(logging.INFO)

    num_devices = args.num_devices or jax.device_count()
    print_device_info(num_devices)

    config = load_config(args.config)
    print(f"  Config: {args.config}")
    print(f"  Dataset: {config['data'].get('dataset', 'unknown')}")

    if args.dry_run:
        success = dry_run(config)
    else:
        success = smoke_test(
            config, epochs=args.epochs, batch_size=args.batch_size,
            use_real_data=not args.synthetic, num_devices=num_devices,
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
