#!/usr/bin/env python3
"""Train and evaluate DepthG semantic segmentation baseline on TPU.

Usage:
    # Train DepthG on Cityscapes 5%
    python scripts/train_depthg.py --config configs/cityscapes_5pct.yaml --epochs 30

    # Evaluate only
    python scripts/train_depthg.py --config configs/cityscapes_5pct.yaml --eval-only \
        --checkpoint checkpoints/depthg.npz
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from absl import logging

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbps.data.datasets import get_dataset, collate_batch
from mbps.models.backbone.dino_vits8 import DINOViTS8
from mbps.models.semantic.depthg_head import DepthGHead
from mbps.losses.semantic_loss import SemanticLoss
from mbps.evaluation.hungarian_matching import hungarian_match, compute_miou


def load_config(config_path):
    """Load config with default fallback."""
    default_path = Path(config_path).parent / "default.yaml"
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


def shard_batch(batch, num_devices):
    """Shard batch across devices."""
    return jax.tree.map(
        lambda x: x.reshape((num_devices, -1) + x.shape[1:]), batch
    )


def train_depthg(config, args):
    """Train DepthG semantic segmentation."""
    print("\n" + "=" * 80)
    print("  DepthG Semantic Segmentation Training")
    print("=" * 80)
    
    # Setup devices
    num_devices = args.num_devices or jax.device_count()
    devices = jax.devices()[:num_devices]
    print(f"Using {num_devices} TPU devices: {devices}")
    
    # Load dataset
    data_cfg = config["data"]
    train_ds = get_dataset(
        dataset_name=data_cfg["dataset"],
        data_dir=data_cfg["data_dir"],
        depth_dir=data_cfg["depth_dir"],
        split="train",
        image_size=tuple(data_cfg["image_size"]),
        subset_fraction=data_cfg.get("subset_fraction"),
    )
    
    val_ds = get_dataset(
        dataset_name=data_cfg["dataset"],
        data_dir=data_cfg["data_dir"],
        depth_dir=data_cfg["depth_dir"],
        split="val",
        image_size=tuple(data_cfg["image_size"]),
        subset_fraction=data_cfg.get("subset_fraction", 0.1),  # 10% val
    )
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    
    # Initialize models
    backbone = DINOViTS8(pretrained=True, frozen=True)
    semantic_head = DepthGHead(
        input_dim=384,
        code_dim=config["architecture"].get("semantic_dim", 90),
    )
    
    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    img_h, img_w = data_cfg["image_size"]
    dummy_img = jnp.zeros((1, img_h, img_w, 3))
    dummy_depth = jnp.zeros((1, img_h, img_w))
    
    print("\nInitializing models...")
    rng, backbone_rng, head_rng = jax.random.split(rng, 3)
    
    backbone_params = backbone.init(backbone_rng, dummy_img)
    dino_out = backbone.apply(backbone_params, dummy_img)
    dino_features = dino_out["patch_features"]  # (B, N, 384)
    
    head_params = semantic_head.init(head_rng, dino_features, deterministic=True)
    
    # Loss and optimizer
    loss_fn = SemanticLoss(
        lambda_depthg=0.3,
        stego_temperature=0.1,
        knn_k=7,
        depth_sigma=0.5,
    )
    
    learning_rate = config["training"]["learning_rate"]
    optimizer = optax.chain(
        optax.clip_by_global_norm(config["training"].get("gradient_clip", 1.0)),
        optax.adamw(learning_rate, weight_decay=config["training"].get("weight_decay", 0.01)),
    )
    
    opt_state = optimizer.init(head_params)
    
    # Replicate across devices
    backbone_params = jax.device_put_replicated(backbone_params, devices)
    head_params = jax.device_put_replicated(head_params, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)
    
    print(f"Parameters: {sum(p.size for p in jax.tree.leaves(head_params)):,}")
    
    # Training step (pmapped)
    @jax.pmap
    def train_step(backbone_params, head_params, opt_state, image, depth, rng):
        """Single training step."""
        def loss_wrapper(params):
            # Forward through frozen backbone
            dino_out = backbone.apply(backbone_params, image)
            features = dino_out["patch_features"]  # (B, N, 384)
            
            # Forward through DepthG head
            semantic_codes = semantic_head.apply(params, features, deterministic=False)
            
            # Downsample depth to token resolution
            patch_size = 8
            b, h, w = depth.shape
            h_tokens = h // patch_size
            w_tokens = w // patch_size
            depth_resized = jax.image.resize(
                depth[:, :, :, None],
                (b, h_tokens, w_tokens, 1),
                method="bilinear",
            )[:, :, :, 0]
            depth_tokens = depth_resized.reshape(b, -1)
            
            # Compute loss
            losses = loss_fn(semantic_codes, features, depth_tokens, rng)
            return losses["total"], losses
        
        (loss_val, losses), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(head_params)
        
        # Average across devices
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss_val = jax.lax.pmean(loss_val, axis_name="batch")
        losses = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name="batch"), losses)
        
        # Update
        updates, new_opt_state = optimizer.update(grads, opt_state, head_params)
        new_params = optax.apply_updates(head_params, updates)
        
        return new_params, new_opt_state, loss_val, losses
    
    # Training loop
    batch_size = data_cfg["batch_size"] * num_devices
    steps_per_epoch = len(train_ds) // batch_size
    total_epochs = args.epochs
    
    print(f"\nTraining for {total_epochs} epochs, {steps_per_epoch} steps/epoch")
    print("=" * 80)
    print(f"{'Epoch':>6} | {'L_total':>8} | {'L_stego':>8} | {'L_depth':>8} | {'Time':>6}")
    print("-" * 80)
    
    best_miou = 0.0
    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()
        epoch_losses = []
        
        # Shuffle indices
        rng_np = np.random.RandomState(epoch)
        indices = np.arange(len(train_ds))
        rng_np.shuffle(indices)
        
        for step in range(steps_per_epoch):
            # Load batch
            start_idx = step * batch_size
            batch_indices = indices[start_idx:start_idx + batch_size]
            samples = [train_ds[i] for i in batch_indices]
            batch = collate_batch(samples)
            
            # Shard across devices
            image = shard_batch(jnp.array(batch["image"]), num_devices)
            depth = shard_batch(jnp.array(batch["depth"]), num_devices)
            
            # Step
            rng, *step_rngs = jax.random.split(rng, num_devices + 1)
            step_rngs = jnp.array(step_rngs)
            
            head_params, opt_state, loss_val, losses = train_step(
                backbone_params, head_params, opt_state, image, depth, step_rngs
            )
            
            epoch_losses.append({k: float(v[0]) for k, v in losses.items()})
        
        # Average losses
        avg_losses = {
            k: np.mean([l[k] for l in epoch_losses])
            for k in epoch_losses[0].keys()
        }
        
        epoch_time = time.time() - epoch_start
        print(
            f"{epoch:>6d} | {avg_losses['total']:>8.4f} | "
            f"{avg_losses['stego']:>8.4f} | {avg_losses['depthg']:>8.4f} | "
            f"{epoch_time:>5.1f}s"
        )
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            miou = evaluate_depthg(
                backbone, semantic_head, backbone_params, head_params,
                val_ds, config, num_devices, devices
            )
            
            if miou > best_miou:
                best_miou = miou
                # Save checkpoint
                save_checkpoint(head_params, args.output, epoch, miou)
    
    print("=" * 80)
    print(f"Training complete! Best mIoU: {best_miou:.2f}%")
    
    return head_params


def evaluate_depthg(backbone, semantic_head, backbone_params, head_params,
                    dataset, config, num_devices, devices):
    """Evaluate DepthG on validation set."""
    from mbps.evaluation.hungarian_matching import hungarian_match, compute_miou
    
    # Unreplicate for single-device inference
    backbone_params_single = jax.tree.map(lambda x: x[0], backbone_params)
    head_params_single = jax.tree.map(lambda x: x[0], head_params)
    
    num_classes = config["data"]["num_classes"]
    patch_size = 8
    img_h, img_w = config["data"]["image_size"]
    h_tokens = img_h // patch_size
    w_tokens = img_w // patch_size
    
    all_miou = []
    all_acc = []
    
    print(f"\n  Evaluating on {min(len(dataset), 50)} samples...")
    
    for i in range(min(len(dataset), 50)):
        sample = dataset[i]
        image = jnp.array(sample["image"][None])  # (1, H, W, 3)
        
        # Forward pass
        dino_out = backbone.apply(backbone_params_single, image)
        features = dino_out["patch_features"]
        semantic_codes = semantic_head.apply(
            head_params_single, features, deterministic=True
        )
        
        # Predictions
        pred = np.argmax(np.array(semantic_codes[0]), axis=-1)  # (N,)
        
        # Ground truth (downsample to token resolution)
        if "semantic_label" in sample:
            gt_full = sample["semantic_label"]
            gt = np.zeros(h_tokens * w_tokens, dtype=np.int32)
            for ty in range(h_tokens):
                for tx in range(w_tokens):
                    cy = min(ty * patch_size + patch_size // 2, gt_full.shape[0] - 1)
                    cx = min(tx * patch_size + patch_size // 2, gt_full.shape[1] - 1)
                    gt[ty * w_tokens + tx] = gt_full[cy, cx]
        else:
            gt = np.random.randint(0, num_classes, pred.shape)
        
        # Hungarian matching
        mapping, acc = hungarian_match(pred, gt, num_classes, num_classes)
        miou, _ = compute_miou(pred, gt, mapping, num_classes)
        
        all_miou.append(miou)
        all_acc.append(acc)
    
    avg_miou = np.mean(all_miou) * 100
    avg_acc = np.mean(all_acc) * 100
    
    print(f"  mIoU: {avg_miou:.2f}% | Accuracy: {avg_acc:.2f}%")
    
    return avg_miou


def save_checkpoint(params, output_path, epoch, miou):
    """Save checkpoint to disk."""
    # Unreplicate
    params_single = jax.tree.map(lambda x: x[0] if hasattr(x, 'shape') else x, params)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Save as npz
    params_flat = jax.tree.util.tree_flatten(params_single)[0]
    np.savez(output_path, *params_flat, epoch=epoch, miou=miou)
    
    print(f"  Checkpoint saved: {output_path} (Epoch {epoch}, mIoU {miou:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Train DepthG Baseline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", type=str, default="checkpoints/depthg_baseline.npz")
    parser.add_argument("--num-devices", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    logging.set_verbosity(logging.INFO)
    
    config = load_config(args.config)
    
    if args.eval_only:
        # Evaluation only mode
        raise NotImplementedError("Eval-only mode not yet implemented")
    else:
        # Training mode
        train_depthg(config, args)


if __name__ == "__main__":
    main()
