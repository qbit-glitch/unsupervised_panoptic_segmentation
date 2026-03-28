#!/usr/bin/env python3
"""AP^mask Evaluation for CutS3D Cascade Mask R-CNN (CAD).

Computes COCO-style AP, AP_50, AP_75 on Cityscapes val set
for paper-comparable instance segmentation metrics.

Usage:
    python scripts/eval_cuts3d_ap.py \
        --config configs/cityscapes_5pct.yaml \
        --checkpoint checkpoints/cuts3d_cad_5pct.npz

    python scripts/eval_cuts3d_ap.py \
        --config configs/cityscapes_5pct.yaml \
        --checkpoint checkpoints/cuts3d_cad_5pct.npz \
        --output results/ap_results.json \
        --score-threshold 0.3 \
        --max-images 50
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
import tensorflow as tf
from PIL import Image

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbps.data.datasets import get_dataset
from mbps.data.tfrecord_utils import parse_example
from mbps.evaluation.instance_metrics import compute_mask_iou, _compute_ap_from_pr
from mbps.models.backbone.dino_vits8 import DINOViTS8
from mbps.models.backbone.weights_converter import convert_dino_weights
from mbps.models.instance.cascade_mask_rcnn import InstanceHead


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path):
    """Load config with default fallback."""
    import yaml

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


# ---------------------------------------------------------------------------
# Pretrained DINO weight loading (copied from train_cuts3d.py)
# ---------------------------------------------------------------------------

DINO_VITS8_URL = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
DINO_VITS8_CACHE = os.path.expanduser("~/.cache/dino/dino_vits8_pretrain.pth")


def download_dino_weights(url=DINO_VITS8_URL, cache_path=DINO_VITS8_CACHE):
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


def interpolate_pos_embed(pos_embed, target_num_patches):
    """Interpolate positional embeddings for different image resolutions."""
    cls_token = pos_embed[:, :1, :]
    patch_embed = pos_embed[:, 1:, :]
    N_orig = patch_embed.shape[1]
    if N_orig == target_num_patches:
        return pos_embed
    D = patch_embed.shape[2]
    orig_size = int(N_orig ** 0.5)
    patch_embed_2d = patch_embed.reshape(1, orig_size, orig_size, D)
    target_h = int(np.sqrt(target_num_patches))
    while target_num_patches % target_h != 0:
        target_h -= 1
    target_w = target_num_patches // target_h
    patch_embed_interp = jax.image.resize(
        patch_embed_2d, (1, target_h, target_w, D), method="bilinear"
    )
    patch_embed_interp = patch_embed_interp.reshape(1, target_num_patches, D)
    return jnp.concatenate([cls_token, patch_embed_interp], axis=1)


def load_pretrained_backbone(backbone, rng, dummy_img, image_size):
    """Initialize backbone with pretrained DINO ViT-S/8 weights."""
    params = backbone.init(rng, dummy_img)
    weights_path = download_dino_weights()
    pretrained_params = convert_dino_weights(weights_path)
    target_patches = (image_size[0] // 8) * (image_size[1] // 8)
    orig_pos = pretrained_params["params"]["pos_embed"]
    if orig_pos.shape[1] - 1 != target_patches:
        print(f"  Interpolating pos_embed: {orig_pos.shape[1]-1} -> {target_patches} patches")
        pretrained_params["params"]["pos_embed"] = interpolate_pos_embed(
            orig_pos, target_patches
        )
    print("  Pretrained DINO weights loaded")
    return pretrained_params


# ---------------------------------------------------------------------------
# Checkpoint Loading
# ---------------------------------------------------------------------------

def load_cad_checkpoint(checkpoint_path, cad, dummy_features, rng):
    """Load CAD parameters from flat npz checkpoint.

    The save_checkpoint function in train_cuts3d.py flattens params via
    jax.tree_util.tree_flatten and saves as arr_0, arr_1, ... We restore
    by re-initializing the model to get the treedef, then unflattening.

    Args:
        checkpoint_path: Path to .npz checkpoint file.
        cad: InstanceHead module.
        dummy_features: Features for shape inference, shape (1, N, D).
        rng: JAX PRNG key.

    Returns:
        Restored Flax parameter dict.
    """
    # Initialize model to get tree structure
    init_params = cad.init(rng, dummy_features, deterministic=True)
    init_flat, treedef = jax.tree_util.tree_flatten(init_params)

    # Load checkpoint
    ckpt = np.load(checkpoint_path, allow_pickle=True)

    # Extract arr_* keys in order
    arr_keys = sorted(
        [k for k in ckpt.files if k.startswith("arr_")],
        key=lambda k: int(k.split("_")[1]),
    )

    # Verify count matches
    num_leaves = len(init_flat)
    num_ckpt = len(arr_keys)
    if num_ckpt != num_leaves:
        raise ValueError(
            f"Checkpoint has {num_ckpt} arrays but model has {num_leaves} leaves. "
            f"Ensure the model architecture matches the checkpoint."
        )

    # Load arrays and verify shapes
    loaded_leaves = []
    for i, (arr_key, ref_leaf) in enumerate(zip(arr_keys, init_flat)):
        arr = jnp.array(ckpt[arr_key])
        if arr.shape != ref_leaf.shape:
            raise ValueError(
                f"Shape mismatch at leaf {i} ({arr_key}): "
                f"checkpoint {arr.shape} vs model {ref_leaf.shape}"
            )
        loaded_leaves.append(arr)

    # Unflatten
    restored_params = treedef.unflatten(loaded_leaves)

    epoch = int(ckpt.get("epoch", -1))
    metric = float(ckpt.get("metric", 0.0))
    print(f"  Loaded checkpoint: epoch={epoch}, train_loss={metric:.4f}")
    print(f"  Restored {num_leaves} parameter arrays")

    return restored_params


# ---------------------------------------------------------------------------
# Ground Truth Instance Mask Extraction
# ---------------------------------------------------------------------------

def extract_gt_instance_masks(instance_label, patch_h, patch_w):
    """Extract GT instance masks at patch resolution from Cityscapes instance_label.

    Cityscapes encoding: values >= 1000 are thing instances
    (id = classId * 1000 + instanceIdx).

    Args:
        instance_label: Instance ID map at image_size, shape (H, W) int32.
        patch_h: Patch grid height (H // 8).
        patch_w: Patch grid width (W // 8).

    Returns:
        Binary masks, shape (num_instances, N) where N = patch_h * patch_w.
        Returns shape (0, N) if no thing instances found.
    """
    N = patch_h * patch_w

    # Downsample to patch resolution using nearest neighbor
    label_pil = Image.fromarray(instance_label.astype(np.int32), mode="I")
    label_resized = np.array(
        label_pil.resize((patch_w, patch_h), Image.NEAREST)
    ).astype(np.int32)

    # Flatten
    label_flat = label_resized.flatten()  # (N,)

    # Extract thing instances (id >= 1000)
    unique_ids = np.unique(label_flat)
    instance_ids = unique_ids[unique_ids >= 1000]

    if len(instance_ids) == 0:
        return np.zeros((0, N), dtype=bool)

    # Create binary masks
    masks = np.stack(
        [(label_flat == inst_id) for inst_id in instance_ids], axis=0
    )  # (num_instances, N)

    # Filter out masks that vanished during downsampling
    valid = masks.sum(axis=1) > 0
    masks = masks[valid]

    return masks


# ---------------------------------------------------------------------------
# Prediction Extraction
# ---------------------------------------------------------------------------

def extract_predictions(mask_logits, scores, score_threshold=0.3, mask_threshold=0.5):
    """Extract filtered binary masks and scores from model output.

    Args:
        mask_logits: Raw mask logits for one image, shape (M, N).
        scores: Confidence scores, shape (M,).
        score_threshold: Minimum score to keep a prediction.
        mask_threshold: Threshold for binarizing mask probabilities.

    Returns:
        Tuple of (pred_masks, pred_scores):
            - pred_masks: Binary masks, shape (num_kept, N) bool.
            - pred_scores: Scores, shape (num_kept,) float32.
    """
    N = mask_logits.shape[1]

    # Filter by score
    keep = scores > score_threshold
    if not np.any(keep):
        return np.zeros((0, N), dtype=bool), np.zeros((0,), dtype=np.float32)

    filtered_logits = mask_logits[keep]
    filtered_scores = scores[keep]

    # Sigmoid in numpy
    mask_probs = 1.0 / (1.0 + np.exp(-filtered_logits.astype(np.float64)))

    # Binarize
    binary_masks = mask_probs > mask_threshold

    # Filter out empty masks
    non_empty = binary_masks.sum(axis=1) > 0
    binary_masks = binary_masks[non_empty]
    filtered_scores = filtered_scores[non_empty]

    return binary_masks, filtered_scores


# ---------------------------------------------------------------------------
# COCO-style Dataset-Level AP
# ---------------------------------------------------------------------------

def compute_dataset_ap_single_threshold(
    all_pred_masks,
    all_pred_scores,
    all_gt_masks,
    iou_threshold=0.5,
):
    """Compute COCO-style pooled AP at a single IoU threshold.

    Pools all predictions across the dataset, sorts globally by score,
    and matches greedily within each image.

    Args:
        all_pred_masks: List of arrays per image, each (M_p_i, N) bool.
        all_pred_scores: List of arrays per image, each (M_p_i,) float.
        all_gt_masks: List of arrays per image, each (M_g_i, N) bool.
        iou_threshold: IoU threshold for TP/FP classification.

    Returns:
        AP value (float).
    """
    # Build global detection list: (score, image_idx, pred_idx_in_image)
    detections = []
    for img_idx, (pred_masks, pred_scores) in enumerate(
        zip(all_pred_masks, all_pred_scores)
    ):
        for pred_idx in range(len(pred_scores)):
            detections.append((
                float(pred_scores[pred_idx]),
                img_idx,
                pred_idx,
            ))

    # Total GT count
    total_gt = sum(len(gt) for gt in all_gt_masks)

    if total_gt == 0 or len(detections) == 0:
        return 0.0

    # Sort by score descending
    detections.sort(key=lambda x: -x[0])

    # Track which GT masks have been matched per image
    gt_matched = [np.zeros(len(gt), dtype=bool) for gt in all_gt_masks]

    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))

    for det_idx, (score, img_idx, pred_idx) in enumerate(detections):
        pred_mask = all_pred_masks[img_idx][pred_idx]
        gt_masks_img = all_gt_masks[img_idx]

        best_iou = 0.0
        best_gt = -1

        for gt_idx in range(len(gt_masks_img)):
            if gt_matched[img_idx][gt_idx]:
                continue
            iou = compute_mask_iou(pred_mask, gt_masks_img[gt_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx

        if best_iou >= iou_threshold and best_gt >= 0:
            tp[det_idx] = 1
            gt_matched[img_idx][best_gt] = True
        else:
            fp[det_idx] = 1

    # PR curve
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precision = cum_tp / (cum_tp + cum_fp + 1e-8)
    recall = cum_tp / total_gt

    # AP via all-point interpolation (COCO standard)
    ap = _compute_ap_from_pr(precision, recall)
    return ap


def compute_dataset_ap_range(
    all_pred_masks,
    all_pred_scores,
    all_gt_masks,
    iou_thresholds=None,
):
    """Compute COCO-style AP at multiple IoU thresholds.

    Args:
        all_pred_masks: List of arrays per image.
        all_pred_scores: List of score arrays per image.
        all_gt_masks: List of GT mask arrays per image.
        iou_thresholds: Array of IoU thresholds. Default [0.5:0.05:0.95].

    Returns:
        Dict with 'AP', 'AP_50', 'AP_75', 'per_threshold'.
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    per_threshold = {}
    for threshold in iou_thresholds:
        ap = compute_dataset_ap_single_threshold(
            all_pred_masks, all_pred_scores, all_gt_masks, threshold
        )
        per_threshold[f"AP@{threshold:.2f}"] = ap

    ap_50 = per_threshold.get("AP@0.50", 0.0)
    ap_75 = per_threshold.get("AP@0.75", 0.0)
    ap_mean = float(np.mean(list(per_threshold.values())))

    return {
        "AP": ap_mean,
        "AP_50": ap_50,
        "AP_75": ap_75,
        "per_threshold": per_threshold,
    }


# ---------------------------------------------------------------------------
# Results Output
# ---------------------------------------------------------------------------

def print_results_table(results, n_images, n_skipped, n_total_preds, n_total_gt):
    """Print formatted results table."""
    print("\n" + "=" * 60)
    print("  AP^mask Evaluation Results - CutS3D CAD")
    print("=" * 60)
    print(f"  Images evaluated:    {n_images}")
    if n_skipped > 0:
        print(f"  Images skipped:      {n_skipped} (no instance labels)")
    print(f"  Total predictions:   {n_total_preds}")
    print(f"  Total GT instances:  {n_total_gt}")
    print("-" * 60)
    print(f"  {'Metric':<20s}  {'Value':>10s}")
    print("-" * 60)
    print(f"  {'AP (COCO)':<20s}  {results['AP'] * 100:>9.2f}%")
    print(f"  {'AP_50':<20s}  {results['AP_50'] * 100:>9.2f}%")
    print(f"  {'AP_75':<20s}  {results['AP_75'] * 100:>9.2f}%")
    print("-" * 60)
    for key, val in sorted(results["per_threshold"].items()):
        print(f"  {key:<20s}  {val * 100:>9.2f}%")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AP^mask Evaluation for CutS3D CAD"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Config YAML path")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to CAD checkpoint (.npz)")
    parser.add_argument("--output", type=str, default="results/ap_results.json",
                        help="JSON output path")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory from config")
    parser.add_argument("--score-threshold", type=float, default=0.3,
                        help="Minimum score to keep predictions")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="Sigmoid threshold for mask binarization")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit evaluation images (for debugging)")
    parser.add_argument("--tfrecord-dir", type=str, default=None,
                        help="GCS path to val TFRecord shards (streams from GCS)")
    parser.add_argument("--stored-size", type=int, nargs=2, default=None,
                        help="H W of images stored in TFRecords (default: 512 1024)")
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------
    config = load_config(args.config)
    data_cfg = config["data"]
    img_h, img_w = data_cfg["image_size"]
    patch_h, patch_w = img_h // 8, img_w // 8
    N = patch_h * patch_w
    max_instances = config["architecture"].get("max_instances", 20)

    print("\n" + "=" * 60)
    print("  CutS3D AP^mask Evaluation")
    print("=" * 60)
    print(f"  Config:          {args.config}")
    print(f"  Checkpoint:      {args.checkpoint}")
    print(f"  Image size:      {img_h}x{img_w}")
    print(f"  Patch grid:      {patch_h}x{patch_w} = {N} tokens")
    print(f"  Max instances:   {max_instances}")
    print(f"  Score threshold: {args.score_threshold}")
    print(f"  Mask threshold:  {args.mask_threshold}")
    print(f"  JAX backend:     {jax.default_backend()}")
    print(f"  JAX devices:     {jax.device_count()}")
    print("=" * 60)
    sys.stdout.flush()

    # ----------------------------------------------------------------
    # Initialize models
    # ----------------------------------------------------------------
    print("\nInitializing models...")

    backbone = DINOViTS8(freeze=True)
    cad = InstanceHead(
        max_instances=max_instances,
        hidden_dim=256,
        num_refinement_stages=3,
        num_classes=1,
    )

    rng = jax.random.PRNGKey(42)
    dummy_img = jnp.zeros((1, img_h, img_w, 3))

    # Load pretrained DINO backbone
    backbone_params = load_pretrained_backbone(
        backbone, rng, dummy_img, (img_h, img_w)
    )

    # Get dummy features for CAD initialization
    dummy_features = backbone.apply(backbone_params, dummy_img)

    # Load trained CAD checkpoint
    print(f"\nLoading CAD checkpoint: {args.checkpoint}")
    cad_params = load_cad_checkpoint(args.checkpoint, cad, dummy_features, rng)

    # ----------------------------------------------------------------
    # JIT compile inference
    # ----------------------------------------------------------------
    print("\nCompiling inference function...")
    sys.stdout.flush()

    @jax.jit
    def forward(bp, cp, image):
        """Forward pass: backbone + CAD."""
        features = backbone.apply(bp, image)
        masks, scores = cad.apply(cp, features, deterministic=True)
        return masks, scores

    # Warmup JIT
    _ = forward(backbone_params, cad_params, dummy_img)
    print("  JIT compilation done")

    # ----------------------------------------------------------------
    # Load validation dataset
    # ----------------------------------------------------------------
    tfrecord_dir = args.tfrecord_dir or data_cfg.get("tfrecord_dir")
    use_tfrecords = tfrecord_dir is not None

    if use_tfrecords:
        # Replace train dir with val dir if user passed train path
        if "/train" in tfrecord_dir:
            tfrecord_dir = tfrecord_dir.replace("/train", "/val")
        print(f"\nLoading val data from TFRecords: {tfrecord_dir}")
        stored_size = tuple(args.stored_size or [512, 1024])

        pattern = tfrecord_dir.rstrip("/") + "/*.tfrecord"
        files = tf.data.Dataset.list_files(pattern, shuffle=False)
        val_tf_ds = files.interleave(
            lambda f: tf.data.TFRecordDataset(f, compression_type="GZIP"),
            cycle_length=2,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        target_h, target_w = img_h, img_w

        def parse_val(raw):
            example = parse_example(
                raw, image_size=stored_size,
                has_semantic=True, has_instance=True,
            )
            if (target_h, target_w) != stored_size:
                example["image"] = tf.image.resize(
                    example["image"], [target_h, target_w]
                )
                # Resize instance labels with nearest neighbor to preserve IDs
                example["instance_label"] = tf.squeeze(tf.image.resize(
                    tf.cast(example["instance_label"][..., tf.newaxis], tf.float32),
                    [target_h, target_w], method="nearest",
                ), axis=-1)
                example["instance_label"] = tf.cast(
                    example["instance_label"], tf.int32
                )
            return example

        val_tf_ds = val_tf_ds.map(parse_val, num_parallel_calls=tf.data.AUTOTUNE)
        val_tf_ds = val_tf_ds.prefetch(tf.data.AUTOTUNE)

        # Count records
        n_val = sum(1 for _ in val_tf_ds)
        print(f"  Val samples: {n_val}")
    else:
        print("\nLoading validation dataset from local files...")
        val_ds = get_dataset(
            dataset_name=data_cfg["dataset"],
            data_dir=args.data_dir or data_cfg["data_dir"],
            depth_dir=data_cfg.get("depth_dir", ""),
            split="val",
            image_size=tuple(data_cfg["image_size"]),
        )
        n_val = len(val_ds)
        print(f"  Val samples: {n_val}")

    # ----------------------------------------------------------------
    # Inference loop
    # ----------------------------------------------------------------
    max_images = args.max_images or n_val
    n_eval = min(n_val, max_images)

    all_pred_masks = []
    all_pred_scores = []
    all_gt_masks = []
    n_images = 0
    n_skipped = 0
    n_total_preds = 0
    n_total_gt = 0

    print(f"\nEvaluating {n_eval} images...")
    print(f"{'Img':>5s} | {'Preds':>6s} | {'GT':>4s} | {'Image ID'}")
    print("-" * 60)
    sys.stdout.flush()

    t_start = time.time()

    if use_tfrecords:
        # TFRecord streaming mode
        for tf_sample in val_tf_ds:
            if n_images + n_skipped >= n_eval:
                break

            sample = {k: v.numpy() for k, v in tf_sample.items()}
            image_id = sample["image_id"].decode("utf-8") if isinstance(
                sample["image_id"], bytes
            ) else str(sample["image_id"])

            # Instance label from TFRecords
            instance_label = sample.get("instance_label")
            if instance_label is None or not np.any(instance_label >= 1000):
                n_skipped += 1
                continue

            image = jnp.array(sample["image"][None])
            masks_out, scores_out = forward(backbone_params, cad_params, image)
            masks_np = np.array(masks_out[0])
            scores_np = np.array(scores_out[0])

            pred_m, pred_s = extract_predictions(
                masks_np, scores_np, args.score_threshold, args.mask_threshold
            )
            gt_m = extract_gt_instance_masks(instance_label, patch_h, patch_w)

            all_pred_masks.append(pred_m)
            all_pred_scores.append(pred_s)
            all_gt_masks.append(gt_m)

            n_images += 1
            n_total_preds += len(pred_s)
            n_total_gt += len(gt_m)

            if n_images <= 5 or n_images % 25 == 0:
                print(
                    f"{n_images:>5d} | {len(pred_s):>6d} | {len(gt_m):>4d} | "
                    f"{image_id}"
                )
                sys.stdout.flush()
    else:
        # Local filesystem mode
        for i in range(n_eval):
            sample = val_ds[i]
            if "instance_label" not in sample:
                n_skipped += 1
                continue

            image = jnp.array(sample["image"][None])
            masks_out, scores_out = forward(backbone_params, cad_params, image)
            masks_np = np.array(masks_out[0])
            scores_np = np.array(scores_out[0])

            pred_m, pred_s = extract_predictions(
                masks_np, scores_np, args.score_threshold, args.mask_threshold
            )
            gt_m = extract_gt_instance_masks(
                sample["instance_label"], patch_h, patch_w
            )

            all_pred_masks.append(pred_m)
            all_pred_scores.append(pred_s)
            all_gt_masks.append(gt_m)

            n_images += 1
            n_total_preds += len(pred_s)
            n_total_gt += len(gt_m)

            if n_images <= 5 or n_images % 25 == 0 or i == n_eval - 1:
                print(
                    f"{n_images:>5d} | {len(pred_s):>6d} | {len(gt_m):>4d} | "
                    f"{sample.get('image_id', str(i))}"
                )
                sys.stdout.flush()

    t_elapsed = time.time() - t_start
    print(f"\nInference: {n_images} images in {t_elapsed:.1f}s "
          f"({t_elapsed / max(n_images, 1):.3f}s/image)")

    if n_images == 0:
        print("ERROR: No images with instance labels found. Cannot compute AP.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Compute AP metrics
    # ----------------------------------------------------------------
    print("\nComputing COCO-style AP metrics...")
    sys.stdout.flush()

    results = compute_dataset_ap_range(
        all_pred_masks, all_pred_scores, all_gt_masks
    )

    # Print results
    print_results_table(results, n_images, n_skipped, n_total_preds, n_total_gt)

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    output = {
        "metrics": {
            "AP": results["AP"],
            "AP_50": results["AP_50"],
            "AP_75": results["AP_75"],
            "per_threshold": results["per_threshold"],
        },
        "config": {
            "checkpoint": args.checkpoint,
            "config_file": args.config,
            "score_threshold": args.score_threshold,
            "mask_threshold": args.mask_threshold,
            "image_size": [img_h, img_w],
            "max_instances": max_instances,
        },
        "stats": {
            "n_images": n_images,
            "n_skipped": n_skipped,
            "n_total_predictions": n_total_preds,
            "n_total_gt_instances": n_total_gt,
            "inference_time_s": round(t_elapsed, 2),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
