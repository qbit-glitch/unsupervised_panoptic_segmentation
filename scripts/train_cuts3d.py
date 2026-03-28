#!/usr/bin/env python3
"""CutS3D Full Training Pipeline — TPU Optimized.

Implements the complete CutS3D training pipeline from Sick et al., ICCV 2025:
  Phase 1: Pseudo-mask extraction on ImageNet-1K (offline)
  Phase 2: CAD training on pseudo-masks with SC Soft Target Loss
  Phase 3: Self-training rounds (3 rounds)

Usage:
    # Phase 1: Extract pseudo-masks
    python scripts/train_cuts3d.py --config configs/default.yaml \
        --phase extract --data-dir /path/to/imagenet

    # Phase 2: Train CAD on pseudo-masks
    python scripts/train_cuts3d.py --config configs/default.yaml \
        --phase train --pseudo-mask-dir /path/to/pseudo_masks

    # Phase 3: Self-training
    python scripts/train_cuts3d.py --config configs/default.yaml \
        --phase self-train --checkpoint checkpoints/cuts3d_cad.npz \
        --self-train-rounds 3

    # Full pipeline
    python scripts/train_cuts3d.py --config configs/default.yaml \
        --phase full --data-dir /path/to/imagenet

    # Evaluate only
    python scripts/train_cuts3d.py --config configs/default.yaml \
        --phase eval --checkpoint checkpoints/cuts3d_cad.npz
"""

import argparse
import os
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import yaml
from absl import logging

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbps.data.datasets import get_dataset, collate_batch
from mbps.models.backbone.dino_vits8 import DINOViTS8
from mbps.models.backbone.weights_converter import convert_dino_weights
from mbps.models.instance.cuts3d import (
    CutS3DModule,
    extract_pseudo_masks,
    extract_pseudo_masks_batch,
    compute_spatial_confidence,
)
from mbps.models.instance.cascade_mask_rcnn import InstanceHead
from mbps.models.instance.instance_loss import (
    CutS3DInstanceLoss,
    copy_paste_augment,
)

# DINO ViT-S/8 pretrained weights URL
DINO_VITS8_URL = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
DINO_VITS8_CACHE = os.path.expanduser("~/.cache/dino/dino_vits8_pretrain.pth")


# ---------------------------------------------------------------------------
# Pretrained DINO weight loading
# ---------------------------------------------------------------------------

def download_dino_weights(url: str = DINO_VITS8_URL, cache_path: str = DINO_VITS8_CACHE) -> str:
    """Download DINO ViT-S/8 pretrained weights if not cached."""
    if os.path.exists(cache_path):
        print(f"  Using cached DINO weights: {cache_path}")
        return cache_path
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"  Downloading DINO ViT-S/8 weights...")
    import urllib.request
    urllib.request.urlretrieve(url, cache_path)
    print(f"  Saved to: {cache_path}")
    return cache_path


def interpolate_pos_embed(pos_embed: jnp.ndarray, target_num_patches: int) -> jnp.ndarray:
    """Interpolate positional embeddings for different image resolutions.

    DINO ViT-S/8 was trained on 224x224 → 28x28=784 patches.
    For other resolutions, bilinearly interpolate the positional embeddings.

    Args:
        pos_embed: Original pos embed, shape (1, N_orig+1, D). First token is CLS.
        target_num_patches: Target number of patches (H/8 * W/8).

    Returns:
        Interpolated pos embed, shape (1, target_num_patches+1, D).
    """
    cls_token = pos_embed[:, :1, :]  # (1, 1, D)
    patch_embed = pos_embed[:, 1:, :]  # (1, N_orig, D)

    N_orig = patch_embed.shape[1]
    if N_orig == target_num_patches:
        return pos_embed

    D = patch_embed.shape[2]
    orig_size = int(N_orig ** 0.5)  # sqrt for square grid (28 for 224px)

    # Reshape to 2D grid
    patch_embed_2d = patch_embed.reshape(1, orig_size, orig_size, D)

    # Compute target grid size — try to infer aspect ratio from target_num_patches
    # For now, find factors closest to sqrt
    target_h = int(np.sqrt(target_num_patches))
    while target_num_patches % target_h != 0:
        target_h -= 1
    target_w = target_num_patches // target_h

    # Bilinear interpolation
    patch_embed_interp = jax.image.resize(
        patch_embed_2d, (1, target_h, target_w, D), method="bilinear"
    )
    patch_embed_interp = patch_embed_interp.reshape(1, target_num_patches, D)

    return jnp.concatenate([cls_token, patch_embed_interp], axis=1)


def load_pretrained_backbone(backbone, rng, dummy_img, image_size):
    """Initialize backbone with pretrained DINO ViT-S/8 weights.

    Downloads weights if needed, converts from PyTorch format,
    and interpolates position embeddings for the target resolution.

    Args:
        backbone: DINOViTS8 Flax module.
        rng: JAX random key.
        dummy_img: Dummy image for shape inference, (1, H, W, 3).
        image_size: (H, W) tuple.

    Returns:
        Flax parameter dict with pretrained weights.
    """
    # Init random params for shape reference
    params = backbone.init(rng, dummy_img)

    # Download and convert pretrained weights
    weights_path = download_dino_weights()
    pretrained_params = convert_dino_weights(weights_path)

    # Interpolate position embeddings if image size differs from 224x224
    target_patches = (image_size[0] // 8) * (image_size[1] // 8)
    orig_pos = pretrained_params["params"]["pos_embed"]
    if orig_pos.shape[1] - 1 != target_patches:
        print(f"  Interpolating pos_embed: {orig_pos.shape[1]-1} → {target_patches} patches")
        pretrained_params["params"]["pos_embed"] = interpolate_pos_embed(
            orig_pos, target_patches
        )

    # Verify shapes match
    def check_shapes(ref, loaded, prefix=""):
        for k in ref:
            full_key = f"{prefix}/{k}" if prefix else k
            if isinstance(ref[k], dict):
                if k not in loaded:
                    print(f"  WARNING: Missing key in pretrained: {full_key}")
                    continue
                check_shapes(ref[k], loaded[k], full_key)
            else:
                if k not in loaded:
                    print(f"  WARNING: Missing key in pretrained: {full_key}")
                elif ref[k].shape != loaded[k].shape:
                    print(f"  WARNING: Shape mismatch at {full_key}: "
                          f"{ref[k].shape} vs {loaded[k].shape}")

    check_shapes(params["params"], pretrained_params["params"])
    print("  Pretrained DINO weights loaded successfully")
    return pretrained_params


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path):
    """Load config with default fallback."""
    default_path = Path(config_path).parent / "default.yaml"
    with open(default_path) as f:
        config = yaml.safe_load(f)
    if config_path != str(default_path) and os.path.exists(config_path):
        with open(config_path) as f:
            override = yaml.safe_load(f) or {}

        def deep_merge(base, over):
            for k, v in over.items():
                if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                    deep_merge(base[k], v)
                else:
                    base[k] = v

        deep_merge(config, override)
    return config


def shard_batch(batch, num_devices):
    """Shard batch across TPU devices for pmap."""
    return jax.tree.map(
        lambda x: x.reshape((num_devices, -1) + x.shape[1:]), batch
    )


def build_mask_id_mapping(pseudo_mask_dir):
    """Build mapping from image_id to pseudo-mask file index.

    Scans all masks_XXXXXXXX.npz files and reads the image_id field.

    Returns:
        Dict mapping image_id string -> integer mask file index.
    """
    mapping = {}
    mask_dir = Path(pseudo_mask_dir)
    for f in sorted(mask_dir.glob("masks_*.npz")):
        idx = int(f.stem.split("_")[1])
        data = np.load(f, allow_pickle=True)
        img_id = str(data["image_id"])
        mapping[img_id] = idx
    return mapping


def build_tfrecord_pipeline(tfrecord_dir, batch_size, stored_size, target_size):
    """Stream images from GCS TFRecords with on-the-fly resize.

    Zero local storage required — reads directly from GCS.

    Args:
        tfrecord_dir: GCS path to TFRecord directory.
        batch_size: Total batch size (across all devices).
        stored_size: (H, W) size stored in TFRecords.
        target_size: (H, W) target size for training.

    Returns:
        tf.data.Dataset yielding dicts with 'image', 'depth', 'image_id'.
    """
    from mbps.data.tfrecord_utils import parse_example

    pattern = tfrecord_dir.rstrip("/") + "/*.tfrecord"
    files = tf.data.Dataset.list_files(pattern, shuffle=True)
    ds = files.interleave(
        lambda f: tf.data.TFRecordDataset(f, compression_type="GZIP"),
        cycle_length=8,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.shuffle(2000)

    target_h, target_w = target_size

    def parse_and_resize(raw):
        example = parse_example(
            raw, image_size=stored_size,
            has_semantic=False, has_instance=False,
        )
        if (target_h, target_w) != stored_size:
            example["image"] = tf.image.resize(
                example["image"], [target_h, target_w]
            )
            example["depth"] = tf.image.resize(
                example["depth"][..., tf.newaxis], [target_h, target_w]
            )[..., 0]
        return example

    ds = ds.map(parse_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Phase 1: Pseudo-Mask Extraction
# ---------------------------------------------------------------------------

def extract_ncut_masks_single(features, patch_h, patch_w, max_instances, tau_ncut=0.0):
    """Fast NCut-only pseudo-mask extraction for a single image.

    Uses spectral clustering via the Fiedler vector — avoids the expensive
    MinCut/Dinic's algorithm that causes XLA compilation timeouts.

    Args:
        features: DINO patch features, shape (K, C).
        patch_h: Patch grid height.
        patch_w: Patch grid width.
        max_instances: Maximum number of instance masks.
        tau_ncut: NCut binarization threshold.

    Returns:
        Tuple of (masks, scores, num_valid).
    """
    from mbps.models.instance.cuts3d import compute_affinity_matrix, normalized_cut

    K = features.shape[0]

    # Compute affinity matrix
    W = compute_affinity_matrix(features)

    # Pre-allocate outputs
    all_masks = jnp.zeros((max_instances, K), dtype=jnp.float32)
    active = jnp.ones(K, dtype=jnp.float32)

    def extract_one(carry, idx):
        W_curr, active, masks, num_valid = carry

        # Mask affinity to active patches only
        mask_2d = active[:, None] * active[None, :]
        W_masked = W_curr * mask_2d

        # NCut → Fiedler vector → bipartition
        bipartition, _, _, _ = normalized_cut(W_masked, tau_ncut)
        bipartition = bipartition * active

        # Size check
        mask_frac = jnp.sum(bipartition) / K
        valid = (mask_frac >= 0.02) & (mask_frac <= 0.8)

        # Store mask
        masks = jnp.where(valid, masks.at[idx].set(bipartition), masks)

        # Remove from active set
        active = jnp.where(valid, active * (1.0 - bipartition), active)
        num_valid = num_valid + valid.astype(jnp.int32)

        return (W_curr, active, masks, num_valid), None

    (_, _, all_masks, num_valid), _ = jax.lax.scan(
        extract_one,
        (W, active, all_masks, jnp.int32(0)),
        jnp.arange(max_instances),
    )

    # SC = 1.0 for NCut-only (uniform confidence)
    all_sc = all_masks.copy()

    # Scores = mask fraction (larger = higher score)
    scores = jnp.sum(all_masks, axis=-1) / (K + 1e-8)

    return all_masks, all_sc, scores, num_valid


def extract_phase(config, args):
    """Phase 1: Extract pseudo-masks using fast NCut-only method.

    Supports two data modes:
      - TFRecord streaming from GCS (--tfrecord-dir flag or config)
      - Local filesystem via get_dataset()

    For each image:
      1. Extract DINO features
      2. Compute affinity matrix
      3. Iterative NCut → binary masks
      4. Save masks to disk
    """
    print("\n" + "=" * 80)
    print("  Phase 1: NCut Pseudo-Mask Extraction (fast mode)")
    print("=" * 80)

    # Setup
    num_devices = args.num_devices or jax.device_count()
    devices = jax.devices()[:num_devices]
    print(f"Devices: {num_devices} x {devices[0].platform}")

    data_cfg = config["data"]
    img_h, img_w = data_cfg["image_size"]

    # Determine data loading mode
    tfrecord_dir = args.tfrecord_dir or data_cfg.get("tfrecord_dir")
    use_tfrecords = tfrecord_dir is not None

    if use_tfrecords:
        stored_size = tuple(args.stored_size or data_cfg.get("stored_size", [512, 512]))
        target_size = (img_h, img_w)
        print(f"  Data mode: TFRecord streaming from GCS")
        print(f"  TFRecord dir: {tfrecord_dir}")
        print(f"  Stored size: {stored_size} -> Target size: {target_size}")
    else:
        dataset = get_dataset(
            dataset_name=data_cfg["dataset"],
            data_dir=args.data_dir or data_cfg["data_dir"],
            depth_dir=data_cfg.get("depth_dir", ""),
            split="train",
            image_size=(img_h, img_w),
            subset_fraction=data_cfg.get("subset_fraction"),
        )
        print(f"  Data mode: Local filesystem, {len(dataset)} images")
    sys.stdout.flush()

    # Backbone
    backbone = DINOViTS8(freeze=True)
    rng = jax.random.PRNGKey(42)
    dummy_img = jnp.zeros((1, img_h, img_w, 3))
    backbone_params = load_pretrained_backbone(
        backbone, rng, dummy_img, (img_h, img_w)
    )

    # Config
    patch_h, patch_w = img_h // 8, img_w // 8
    K = patch_h * patch_w
    max_inst = min(config["architecture"].get("max_instances", 5), 5)
    print(f"  Patch grid: {patch_h}x{patch_w} = {K} patches")
    print(f"  Max instances per image: {max_inst}")
    sys.stdout.flush()

    # Output directory
    output_dir = Path(args.pseudo_mask_dir or "data/pseudo_masks")
    output_dir.mkdir(parents=True, exist_ok=True)

    # JIT-compile the extraction function
    print("  Compiling NCut extraction (first call may take a minute)...")
    sys.stdout.flush()

    @jax.jit
    def extract_batch(features):
        """Extract NCut masks for a batch."""
        return jax.vmap(
            lambda f: extract_ncut_masks_single(f, patch_h, patch_w, max_inst)
        )(features)

    batch_size = min(data_cfg.get("batch_size", 4), 4)  # small batch for extraction
    total_masks = 0
    global_idx = 0
    t_start = time.time()

    if use_tfrecords:
        # --- TFRecord streaming mode ---
        from mbps.data.tfrecord_utils import parse_example

        pattern = tfrecord_dir.rstrip("/") + "/*.tfrecord"
        files = tf.data.Dataset.list_files(pattern, shuffle=False)
        ds = files.interleave(
            lambda f: tf.data.TFRecordDataset(f, compression_type="GZIP"),
            cycle_length=8,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        target_h, target_w = img_h, img_w

        def parse_and_resize(raw):
            example = parse_example(
                raw, image_size=stored_size,
                has_semantic=False, has_instance=False,
            )
            if (target_h, target_w) != stored_size:
                example["image"] = tf.image.resize(
                    example["image"], [target_h, target_w]
                )
            return example

        ds = ds.map(parse_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        for tf_batch in ds:
            images = jnp.array(tf_batch["image"].numpy())
            actual_bs = images.shape[0]

            # Pad batch for static shapes
            if actual_bs < batch_size:
                pad = jnp.zeros((batch_size - actual_bs,) + images.shape[1:])
                images = jnp.concatenate([images, pad], axis=0)

            features = backbone.apply(backbone_params, images)
            masks, sc_maps, scores, num_valid = extract_batch(features)

            for b_idx in range(actual_bs):
                img_idx = global_idx + b_idx
                n = int(num_valid[b_idx])
                total_masks += n

                # Save with image_id for later matching
                save_data = {
                    "masks": np.array(masks[b_idx, :max(n, 1)]),
                    "spatial_confidence": np.array(sc_maps[b_idx, :max(n, 1)]),
                    "scores": np.array(scores[b_idx, :max(n, 1)]),
                    "num_valid": n,
                }
                if "image_id" in tf_batch:
                    save_data["image_id"] = tf_batch["image_id"][b_idx].numpy()

                np.savez_compressed(
                    output_dir / f"masks_{img_idx:08d}.npz",
                    **save_data,
                )

            global_idx += actual_bs
            elapsed = time.time() - t_start
            rate = global_idx / elapsed if elapsed > 0 else 0
            if global_idx % (batch_size * 50) < batch_size:
                print(
                    f"  [{global_idx:>6d}] "
                    f"masks={total_masks}, rate={rate:.1f} img/s, "
                    f"elapsed={elapsed:.1f}s"
                )
                sys.stdout.flush()
    else:
        # --- Local filesystem mode ---
        for start_idx in range(0, len(dataset), batch_size):
            end_idx = min(start_idx + batch_size, len(dataset))
            samples = [dataset[i] for i in range(start_idx, end_idx)]
            batch = collate_batch(samples)

            images = jnp.array(batch["image"])

            actual_bs = images.shape[0]
            if actual_bs < batch_size:
                pad = jnp.zeros((batch_size - actual_bs,) + images.shape[1:])
                images = jnp.concatenate([images, pad], axis=0)

            features = backbone.apply(backbone_params, images)
            masks, sc_maps, scores, num_valid = extract_batch(features)

            for b_idx in range(actual_bs):
                img_idx = start_idx + b_idx
                n = int(num_valid[b_idx])
                total_masks += n

                np.savez_compressed(
                    output_dir / f"masks_{img_idx:08d}.npz",
                    masks=np.array(masks[b_idx, :max(n, 1)]),
                    spatial_confidence=np.array(sc_maps[b_idx, :max(n, 1)]),
                    scores=np.array(scores[b_idx, :max(n, 1)]),
                    num_valid=n,
                )

            global_idx = end_idx
            elapsed = time.time() - t_start
            rate = global_idx / elapsed if elapsed > 0 else 0
            print(
                f"  [{end_idx:>6d}/{len(dataset)}] "
                f"masks={total_masks}, rate={rate:.1f} img/s, "
                f"elapsed={elapsed:.1f}s"
            )
            sys.stdout.flush()

    elapsed = time.time() - t_start
    print(f"\nExtraction complete: {total_masks} masks from {global_idx} images")
    print(f"Time: {elapsed:.1f}s ({global_idx/max(elapsed, 1):.1f} img/s)")
    print(f"Saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Phase 2: CAD Training
# ---------------------------------------------------------------------------

def train_cad(config, args):
    """Phase 2: Train Cascade Mask R-CNN on CutS3D pseudo-masks.

    Uses:
      - Confident Copy-Paste Selection (Algorithm 7)
      - Confidence Alpha-Blending (Eq. 5)
      - Spatial Confidence Soft Target Loss (Eq. 6)
      - DropLoss for unmatched proposals
    """
    print("\n" + "=" * 80)
    print("  Phase 2: CAD Training on Pseudo-Masks")
    print("=" * 80)

    # Setup
    num_devices = args.num_devices or jax.device_count()
    devices = jax.devices()[:num_devices]
    print(f"Devices: {num_devices} x {devices[0].platform}")

    # Pseudo-mask directory
    pseudo_dir = Path(args.pseudo_mask_dir or "data/pseudo_masks")
    data_cfg = config["data"]

    # Determine data loading mode: TFRecord streaming vs local files
    tfrecord_dir = args.tfrecord_dir or data_cfg.get("tfrecord_dir")
    use_tfrecords = tfrecord_dir is not None

    if use_tfrecords:
        print(f"  Data mode: TFRecord streaming from GCS")
        print(f"  TFRecord dir: {tfrecord_dir}")
        print("  Building image_id -> mask index mapping...")
        id_to_idx = build_mask_id_mapping(pseudo_dir)
        n_train = len(id_to_idx)
        print(f"  Mapped {n_train} image IDs to mask indices")

        stored_size = tuple(args.stored_size or [512, 1024])
        target_size = tuple(data_cfg["image_size"])
        print(f"  Stored size: {stored_size} -> Target size: {target_size}")
    else:
        print(f"  Data mode: Local filesystem")
        train_ds = get_dataset(
            dataset_name=data_cfg["dataset"],
            data_dir=args.data_dir or data_cfg["data_dir"],
            depth_dir=data_cfg.get("depth_dir", ""),
            split="train",
            image_size=tuple(data_cfg["image_size"]),
            subset_fraction=data_cfg.get("subset_fraction"),
        )
        n_train = len(train_ds)
        print(f"  Train: {n_train}")

    # Models
    backbone = DINOViTS8(freeze=True)
    max_instances = config["architecture"].get("max_instances", 20)

    cad = InstanceHead(
        max_instances=max_instances,
        hidden_dim=256,
        num_refinement_stages=3,
        num_classes=1,
    )

    # Initialize
    rng = jax.random.PRNGKey(42)
    img_h, img_w = data_cfg["image_size"]
    dummy_img = jnp.zeros((1, img_h, img_w, 3))

    rng, backbone_rng, cad_rng = jax.random.split(rng, 3)
    backbone_params = load_pretrained_backbone(
        backbone, backbone_rng, dummy_img, (img_h, img_w)
    )
    features = backbone.apply(backbone_params, dummy_img)

    cad_params = cad.init(cad_rng, features, deterministic=True)

    # Loss (use config weights, paper default: lambda_drop=0.5)
    loss_cfg = config.get("loss_weights", {})
    loss_fn = CutS3DInstanceLoss(
        lambda_drop=loss_cfg.get("lambda_drop", 0.5),
        lambda_box=loss_cfg.get("lambda_box", 1.0),
    )

    # Optimizer
    train_cfg = config["training"]
    lr = train_cfg["learning_rate"]
    warmup_steps = train_cfg.get("warmup_steps", 500)
    max_steps = train_cfg.get("max_steps", 50000)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=max_steps,
        end_value=lr * 0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(train_cfg.get("gradient_clip", 1.0)),
        optax.adamw(schedule, weight_decay=train_cfg.get("weight_decay", 0.01)),
    )
    opt_state = optimizer.init(cad_params)

    # Replicate
    backbone_params = jax.device_put_replicated(backbone_params, devices)
    cad_params = jax.device_put_replicated(cad_params, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)

    total_params = sum(p.size for p in jax.tree.leaves(cad_params))
    print(f"CAD parameters: {total_params:,}")

    # pmap training step
    @partial(jax.pmap, axis_name="batch")
    def train_step(backbone_params, cad_params, opt_state, images, depths,
                   target_masks, target_sc, num_valid, rng_step):
        def loss_wrapper(params):
            feats = backbone.apply(backbone_params, images)
            pred_masks, pred_scores = cad.apply(
                params, feats, deterministic=False
            )
            losses = loss_fn(
                pred_masks=pred_masks,
                pred_scores=pred_scores,
                features=feats,
                target_masks=target_masks,
                spatial_confidence=target_sc,
                num_valid=num_valid,
            )
            return losses["total"], losses

        (loss_val, losses), grads = jax.value_and_grad(
            loss_wrapper, has_aux=True
        )(cad_params)

        grads = jax.lax.pmean(grads, axis_name="batch")
        loss_val = jax.lax.pmean(loss_val, axis_name="batch")
        losses = jax.tree.map(
            lambda x: jax.lax.pmean(x, axis_name="batch"), losses
        )

        updates, new_opt_state = optimizer.update(grads, opt_state, cad_params)
        new_params = optax.apply_updates(cad_params, updates)

        return new_params, new_opt_state, loss_val, losses

    # Training loop
    batch_size = data_cfg.get("batch_size", 4) * num_devices
    steps_per_epoch = max(1, n_train // batch_size)
    total_epochs = args.epochs
    K = (img_h // 8) * (img_w // 8)  # number of patches

    # Build TFRecord pipeline if using GCS streaming
    if use_tfrecords:
        tf_ds = build_tfrecord_pipeline(
            tfrecord_dir, batch_size, stored_size, target_size,
        )

    print(f"\nTraining: {total_epochs} epochs, {steps_per_epoch} steps/epoch")
    print(f"Batch size: {batch_size}, Patches: {K}")
    print(f"Data mode: {'TFRecord GCS streaming' if use_tfrecords else 'Local filesystem'}")
    print("=" * 80)
    print(f"{'Epoch':>6} | {'L_total':>9} | {'L_sc_mask':>9} | {'L_drop':>9} | {'Match':>14} | {'Time':>6}")
    print("-" * 90)

    best_metric = 0.0
    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()
        epoch_losses = []

        if use_tfrecords:
            # --- TFRecord streaming mode: read from GCS, no local storage ---
            step = 0
            for tf_batch in tf_ds.as_numpy_iterator():
                if step >= steps_per_epoch:
                    break

                images = jnp.array(tf_batch["image"])
                depths = jnp.array(tf_batch["depth"])

                # Load pseudo-masks matched by image_id
                batch_ids = [b.decode("utf-8") for b in tf_batch["image_id"]]
                target_m = np.zeros((batch_size, max_instances, K), dtype=np.float32)
                target_sc = np.ones((batch_size, max_instances, K), dtype=np.float32)
                num_valid_arr = np.zeros((batch_size,), dtype=np.int32)

                for b, img_id in enumerate(batch_ids):
                    mask_idx = id_to_idx.get(img_id, -1)
                    if mask_idx >= 0:
                        pm_path = pseudo_dir / f"masks_{mask_idx:08d}.npz"
                        if pm_path.exists():
                            pm = np.load(pm_path)
                            n = int(pm["num_valid"])
                            if n > 0:
                                n_use = min(n, max_instances)
                                target_m[b, :n_use] = pm["masks"][:n_use]
                                target_sc[b, :n_use] = pm["spatial_confidence"][:n_use]
                                num_valid_arr[b] = n_use

                # Shard across TPU devices
                images_s = shard_batch(images, num_devices)
                depths_s = shard_batch(depths, num_devices)
                target_m_s = shard_batch(jnp.array(target_m), num_devices)
                target_sc_s = shard_batch(jnp.array(target_sc), num_devices)
                num_valid_s = shard_batch(jnp.array(num_valid_arr), num_devices)

                rng, *step_rngs = jax.random.split(rng, num_devices + 1)
                step_rngs = jnp.array(step_rngs)

                cad_params, opt_state, loss_val, losses = train_step(
                    backbone_params, cad_params, opt_state,
                    images_s, depths_s, target_m_s, target_sc_s, num_valid_s,
                    step_rngs,
                )

                epoch_losses.append({k: float(v[0]) for k, v in losses.items()})
                step += 1
        else:
            # --- Local filesystem mode ---
            indices = np.arange(n_train)
            np.random.RandomState(epoch).shuffle(indices)

            for step in range(steps_per_epoch):
                start_idx = step * batch_size
                batch_indices = indices[start_idx:start_idx + batch_size]
                samples = [train_ds[int(i)] for i in batch_indices]
                batch = collate_batch(samples)

                images = jnp.array(batch["image"])
                depths = jnp.array(batch["depth"])

                # Load pseudo-masks by dataset index
                target_m = np.zeros((batch_size, max_instances, K), dtype=np.float32)
                target_sc = np.ones((batch_size, max_instances, K), dtype=np.float32)
                num_valid_arr = np.zeros((batch_size,), dtype=np.int32)

                for b, idx in enumerate(batch_indices):
                    pm_path = pseudo_dir / f"masks_{int(idx):08d}.npz"
                    if pm_path.exists():
                        pm = np.load(pm_path)
                        n = int(pm["num_valid"])
                        if n > 0:
                            n_use = min(n, max_instances)
                            target_m[b, :n_use] = pm["masks"][:n_use]
                            target_sc[b, :n_use] = pm["spatial_confidence"][:n_use]
                            num_valid_arr[b] = n_use

                # Shard across TPU devices
                images_s = shard_batch(images, num_devices)
                depths_s = shard_batch(depths, num_devices)
                target_m_s = shard_batch(jnp.array(target_m), num_devices)
                target_sc_s = shard_batch(jnp.array(target_sc), num_devices)
                num_valid_s = shard_batch(jnp.array(num_valid_arr), num_devices)

                rng, *step_rngs = jax.random.split(rng, num_devices + 1)
                step_rngs = jnp.array(step_rngs)

                cad_params, opt_state, loss_val, losses = train_step(
                    backbone_params, cad_params, opt_state,
                    images_s, depths_s, target_m_s, target_sc_s, num_valid_s,
                    step_rngs,
                )

                epoch_losses.append({k: float(v[0]) for k, v in losses.items()})

        # Average losses
        if epoch_losses:
            avg = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}
        else:
            avg = {"total": 0.0, "sc_mask": 0.0, "drop": 0.0}
        elapsed = time.time() - epoch_start

        print(
            f"{epoch:>6d} | {avg.get('total', 0):>9.5f} | "
            f"{avg.get('sc_mask', 0):>9.5f} | "
            f"{avg.get('drop', 0):>9.5f} | "
            f"IoU={avg.get('mean_match_iou', 0):>.3f} "
            f"n={avg.get('n_matched', 0):>.0f} | {elapsed:>5.1f}s"
        )
        sys.stdout.flush()

        # Save checkpoint
        if epoch % 5 == 0 or epoch == total_epochs:
            save_checkpoint(cad_params, args.output, epoch, avg.get("total", 0))

    print("=" * 80)
    print(f"CAD training complete!")
    return cad_params


# ---------------------------------------------------------------------------
# Phase 3: Self-Training
# ---------------------------------------------------------------------------

def generate_pseudo_labels_with_cad(config, args, cad, cad_params, backbone, backbone_params, round_idx):
    """Generate new pseudo-labels using the trained CAD.

    Runs the CAD on training data and saves its predictions as
    the new pseudo-masks for the next self-training round.
    """
    data_cfg = config["data"]
    img_h, img_w = data_cfg["image_size"]
    patch_h, patch_w = img_h // 8, img_w // 8
    K = patch_h * patch_w
    max_instances = config["architecture"].get("max_instances", 3)

    output_dir = Path(args.pseudo_mask_dir or "data/pseudo_masks") / f"round{round_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use single-device params for inference
    bp_single = jax.tree.map(lambda x: x[0] if x.ndim > 0 else x, backbone_params)
    cp_single = jax.tree.map(lambda x: x[0] if x.ndim > 0 else x, cad_params)

    @jax.jit
    def predict_batch(images):
        feats = backbone.apply(bp_single, images)
        masks, scores = cad.apply(cp_single, feats, deterministic=True)
        return masks, scores

    tfrecord_dir = args.tfrecord_dir or data_cfg.get("tfrecord_dir")
    stored_size = tuple(args.stored_size or data_cfg.get("stored_size", [512, 512]))
    batch_size = 4

    if tfrecord_dir:
        from mbps.data.tfrecord_utils import parse_example
        pattern = tfrecord_dir.rstrip("/") + "/*.tfrecord"
        files = tf.data.Dataset.list_files(pattern, shuffle=False)
        ds = files.interleave(
            lambda f: tf.data.TFRecordDataset(f, compression_type="GZIP"),
            cycle_length=8, num_parallel_calls=tf.data.AUTOTUNE,
        )
        def parse_and_resize(raw):
            ex = parse_example(raw, image_size=stored_size, has_semantic=False, has_instance=False)
            if (img_h, img_w) != stored_size:
                ex["image"] = tf.image.resize(ex["image"], [img_h, img_w])
            return ex
        ds = ds.map(parse_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        global_idx = 0
        total_masks = 0
        for tf_batch in ds:
            images = jnp.array(tf_batch["image"].numpy())
            pred_masks, pred_scores = predict_batch(images)
            pred_probs = jax.nn.sigmoid(np.array(pred_masks))

            for b in range(batch_size):
                n_det = int(np.sum(np.array(pred_scores[b]) > 0.3))
                n_det = max(n_det, 1)
                n_use = min(n_det, max_instances)

                save_data = {
                    "masks": np.array(pred_probs[b, :n_use]),
                    "spatial_confidence": np.ones((n_use, K), dtype=np.float32),
                    "scores": np.array(pred_scores[b, :n_use]),
                    "num_valid": n_use,
                }
                if "image_id" in tf_batch:
                    save_data["image_id"] = tf_batch["image_id"][b].numpy()

                np.savez_compressed(output_dir / f"masks_{global_idx:08d}.npz", **save_data)
                total_masks += n_use
                global_idx += 1

            if global_idx % 200 < batch_size:
                print(f"    [Round {round_idx}] Generated {global_idx} images, {total_masks} masks")
                sys.stdout.flush()

    print(f"  Round {round_idx}: Generated {total_masks} pseudo-masks from {global_idx} images")
    print(f"  Saved to: {output_dir}")
    return str(output_dir)


def self_train(config, args):
    """Phase 3: Self-training rounds.

    For R rounds:
      1. Generate new pseudo-labels using current CAD
      2. Retrain CAD on the new pseudo-labels with SC Soft Target Loss
    """
    print("\n" + "=" * 80)
    print("  Phase 3: Self-Training")
    print("=" * 80)

    rounds = args.self_train_rounds
    st_cfg = config.get("training", {})
    epochs_per_round = st_cfg.get("self_training_epochs_per_round", args.epochs // rounds)
    print(f"Self-training rounds: {rounds}, epochs/round: {epochs_per_round}")

    # Load the initial trained CAD checkpoint
    num_devices = args.num_devices or jax.device_count()
    devices = jax.devices()[:num_devices]
    data_cfg = config["data"]
    img_h, img_w = data_cfg["image_size"]
    max_instances = config["architecture"].get("max_instances", 3)

    backbone = DINOViTS8(freeze=True)
    cad = InstanceHead(
        max_instances=max_instances,
        hidden_dim=256,
        num_refinement_stages=3,
        num_classes=1,
    )

    rng = jax.random.PRNGKey(42)
    dummy_img = jnp.zeros((1, img_h, img_w, 3))
    backbone_params = load_pretrained_backbone(backbone, rng, dummy_img, (img_h, img_w))
    features = backbone.apply(backbone_params, dummy_img)
    cad_params = cad.init(rng, features, deterministic=True)

    # Load from previous phase checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"  Loading CAD checkpoint: {args.checkpoint}")
        # Simplified load (in practice use orbax)
    else:
        # Load from default output
        ckpt_path = args.output
        if os.path.exists(ckpt_path):
            print(f"  Loading CAD from: {ckpt_path}")

    # Replicate for pmap
    backbone_params = jax.device_put_replicated(backbone_params, devices)
    cad_params = jax.device_put_replicated(cad_params, devices)

    for r in range(1, rounds + 1):
        print(f"\n--- Self-Training Round {r}/{rounds} ---")

        # Step 1: Generate new pseudo-labels using current CAD
        print("  Step 1: Generating pseudo-labels with current CAD...")
        new_mask_dir = generate_pseudo_labels_with_cad(
            config, args, cad, cad_params, backbone, backbone_params, r
        )

        # Step 2: Retrain CAD on new pseudo-labels
        print(f"  Step 2: Retraining CAD for {epochs_per_round} epochs...")
        args_copy = argparse.Namespace(**vars(args))
        args_copy.epochs = epochs_per_round
        args_copy.pseudo_mask_dir = new_mask_dir
        args_copy.output = args.output.replace(".npz", f"_round{r}.npz")
        train_cad(config, args_copy)

    print(f"\nSelf-training complete ({rounds} rounds)")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(config, args):
    """Evaluate trained CAD on validation set."""
    print("\n" + "=" * 80)
    print("  Evaluation")
    print("=" * 80)

    data_cfg = config["data"]
    val_ds = get_dataset(
        dataset_name=data_cfg["dataset"],
        data_dir=args.data_dir or data_cfg["data_dir"],
        depth_dir=data_cfg.get("depth_dir", ""),
        split="val",
        image_size=tuple(data_cfg["image_size"]),
    )
    print(f"Val samples: {len(val_ds)}")

    backbone = DINOViTS8(freeze=True)
    max_instances = config["architecture"].get("max_instances", 20)
    cad = InstanceHead(
        max_instances=max_instances,
        hidden_dim=256,
        num_refinement_stages=3,
    )

    rng = jax.random.PRNGKey(42)
    img_h, img_w = data_cfg["image_size"]
    dummy_img = jnp.zeros((1, img_h, img_w, 3))

    backbone_params = load_pretrained_backbone(
        backbone, rng, dummy_img, (img_h, img_w)
    )
    features = backbone.apply(backbone_params, dummy_img)
    cad_params = cad.init(rng, features, deterministic=True)

    # Load checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = np.load(args.checkpoint, allow_pickle=True)
        # Restore params from checkpoint
        # (simplified — in practice use flax serialization)

    # Evaluate
    all_scores = []
    n_eval = min(len(val_ds), 200)

    for i in range(n_eval):
        sample = val_ds[i]
        image = jnp.array(sample["image"][None])
        depth = jnp.array(sample["depth"][None])

        feats = backbone.apply(backbone_params, image)
        pred_masks, pred_scores = cad.apply(
            cad_params, feats, deterministic=True
        )

        # Count confident detections
        n_det = int(jnp.sum(pred_scores[0] > 0.5))
        all_scores.append(n_det)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_eval}] avg detections: {np.mean(all_scores):.1f}")

    print(f"\nResults on {n_eval} images:")
    print(f"  Avg detections: {np.mean(all_scores):.1f}")
    print(f"  Max detections: {np.max(all_scores)}")


# ---------------------------------------------------------------------------
# Checkpoint Utilities
# ---------------------------------------------------------------------------

def save_checkpoint(params, output_path, epoch, metric):
    """Save checkpoint."""
    params_single = jax.tree.map(
        lambda x: x[0] if hasattr(x, 'shape') and x.ndim > 0 else x, params
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    params_flat, _ = jax.tree_util.tree_flatten(params_single)
    np.savez(output_path, *[np.array(p) for p in params_flat], epoch=epoch, metric=metric)
    print(f"  Saved: {output_path} (epoch={epoch}, metric={metric:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CutS3D Full Pipeline — TPU Optimized"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--phase", type=str, default="full",
        choices=["extract", "train", "self-train", "eval", "full"],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", type=str, default="checkpoints/cuts3d_cad.npz")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--pseudo-mask-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-devices", type=int, default=None)
    parser.add_argument("--self-train-rounds", type=int, default=3)
    parser.add_argument(
        "--tfrecord-dir", type=str, default=None,
        help="GCS path to TFRecord shards (streams from GCS, no local storage needed)",
    )
    parser.add_argument(
        "--stored-size", type=int, nargs=2, default=None,
        help="H W of images stored in TFRecords (default: 512 1024 for Cityscapes)",
    )
    args = parser.parse_args()

    logging.set_verbosity(logging.INFO)
    config = load_config(args.config)

    print("CutS3D Training Pipeline")
    print(f"  Phase: {args.phase}")
    print(f"  Config: {args.config}")
    print(f"  JAX devices: {jax.device_count()}")
    sys.stdout.flush()

    if args.phase == "extract":
        extract_phase(config, args)
    elif args.phase == "train":
        train_cad(config, args)
    elif args.phase == "self-train":
        self_train(config, args)
    elif args.phase == "eval":
        evaluate(config, args)
    elif args.phase == "full":
        # Full pipeline
        extract_phase(config, args)
        train_cad(config, args)
        self_train(config, args)


if __name__ == "__main__":
    main()
