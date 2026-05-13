"""MBPS Evaluation Script.

Uses jax.pmap for parallel inference across all available devices.

Usage:
    python scripts/evaluate.py --config configs/cityscapes.yaml --checkpoint PATH
    python scripts/evaluate.py --config configs/cityscapes.yaml --checkpoint PATH --use_crf
    python scripts/evaluate.py --config configs/cityscapes.yaml --checkpoint PATH --num_devices 4

Evaluates trained model on validation set and reports:
    - PQ, SQ, RQ (overall, things, stuff)
    - mIoU (with Hungarian matching)
    - Per-class metrics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import yaml
from absl import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbps.data.datasets import get_dataset
from mbps.data.transforms import EvalTransform
from mbps.evaluation.hungarian_matching import compute_miou, hungarian_match
from mbps.evaluation.panoptic_quality import compute_panoptic_quality
from mbps.models.mbps_model import MBPSModel
from mbps.models.merger.crf_postprocess import crf_inference
from mbps.models.merger.panoptic_merge import panoptic_merge
from mbps.training.checkpointing import CheckpointManager, _flatten_pytree

# ---------------------------------------------------------------------------
# Cityscapes label ID to train ID mapping
# ---------------------------------------------------------------------------
# Original Cityscapes labelIds → trainIds (19 classes, rest = 255 ignore)
CITYSCAPES_ID_TO_TRAINID = {
    7: 0,   # road
    8: 1,   # sidewalk
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    17: 5,  # pole
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10, # sky
    24: 11, # person
    25: 12, # rider
    26: 13, # car
    27: 14, # truck
    28: 15, # bus
    31: 16, # train
    32: 17, # motorcycle
    33: 18, # bicycle
}

def remap_cityscapes_labels(labels: np.ndarray) -> np.ndarray:
    """Remap Cityscapes original label IDs to train IDs (0-18).

    Non-training classes become 255 (ignore).
    """
    remapped = np.full_like(labels, 255)
    for orig_id, train_id in CITYSCAPES_ID_TO_TRAINID.items():
        remapped[labels == orig_id] = train_id
    return remapped


# ---------------------------------------------------------------------------
# Multi-device utilities
# ---------------------------------------------------------------------------

def shard_array(x, num_devices):
    """Reshape (B, ...) to (num_devices, B // num_devices, ...)."""
    assert x.shape[0] % num_devices == 0, (
        f"Batch dim {x.shape[0]} not divisible by {num_devices}"
    )
    return x.reshape((num_devices, -1) + x.shape[1:])


def load_config(config_path: str) -> Dict[str, Any]:
    """Load config (same as train.py)."""
    default_path = os.path.join(
        os.path.dirname(config_path), "default.yaml"
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
    deep_merge(config, override)
    return config


def evaluate_model(
    model: MBPSModel,
    params: Any,
    dataset,
    config: Dict[str, Any],
    use_crf: bool = False,
    num_devices: int = 1,
) -> Dict[str, Any]:
    """Run evaluation on entire dataset with pmap-parallel forward passes.

    Args:
        model: MBPS model.
        params: Model parameters (unreplicated, single-device copy).
        dataset: Evaluation dataset.
        config: Configuration dict.
        use_crf: Whether to apply CRF post-processing.
        num_devices: Number of devices for parallel inference.

    Returns:
        Dict with evaluation metrics.
    """
    # -- Multi-device setup --
    if num_devices > 1:
        devices = jax.local_devices()[:num_devices]
        r_params = jax.device_put_replicated(params, devices)

        @jax.pmap
        def p_forward(p, image, depth):
            return model.apply(
                p, image=image, depth=depth,
                use_bridge=True, deterministic=True,
            )

    all_pred_semantic = []
    all_gt_semantic = []
    all_per_image = []  # Per-image data for PQ computation
    num_samples = 0

    logging.info(
        f"Starting evaluation with {num_devices} device(s)..."
    )
    start_time = time.time()

    # Process in batches of num_devices (1 sample per device)
    dataset_len = len(dataset)
    for batch_start in range(0, dataset_len, num_devices):
        batch_end = min(batch_start + num_devices, dataset_len)
        actual_count = batch_end - batch_start

        # Load samples
        samples = [dataset[i] for i in range(batch_start, batch_end)]
        # Pad to num_devices for even sharding
        while len(samples) < num_devices:
            samples.append(samples[-1])

        # Stack into (num_devices, H, W, 3) and (num_devices, H, W)
        images = jnp.stack([jnp.array(s["image"]) for s in samples])
        depths = jnp.stack([
            jnp.array(s["depth"]) if "depth" in s
            else jnp.zeros(images.shape[1:3])
            for s in samples
        ])

        # Forward pass
        if num_devices > 1:
            # Each device gets 1 sample: (num_devices, 1, H, W, C)
            outputs = p_forward(
                r_params, images[:, None], depths[:, None]
            )
            # Squeeze per-device batch dim: (num_devices, 1, ...) -> (num_devices, ...)
            outputs = jax.tree.map(
                lambda x: x[:, 0] if x.ndim >= 2 else x,
                outputs,
            )
        else:
            outputs = model.apply(
                params,
                image=images[:1],
                depth=depths[:1],
                use_bridge=True,
                deterministic=True,
            )

        # Process each actual sample
        for i in range(actual_count):
            semantic_pred = np.array(outputs["semantic_pred"][i])  # (N,)
            instance_masks = np.array(outputs["instance_masks"][i])  # (M, N)
            instance_scores = np.array(outputs["instance_scores"][i])  # (M,)

            # Optional CRF refinement
            if use_crf:
                h, w = config["data"]["image_size"]
                spatial_h = h // 8
                spatial_w = w // 8

                semantic_logits = np.array(outputs["semantic_probs"][i])
                from jax.image import resize
                image_tokens = np.array(
                    resize(
                        jnp.array(samples[i]["image"]),
                        (spatial_h, spatial_w, 3),
                        method="bilinear",
                    )
                ).reshape(-1, 3)

                refined = crf_inference(
                    jnp.array(np.log(semantic_logits + 1e-8)),
                    jnp.array(image_tokens),
                    spatial_h=spatial_h,
                    spatial_w=spatial_w,
                )
                semantic_pred = np.array(jnp.argmax(refined, axis=-1))

            # Stuff-things classification
            stuff_things_scores = np.array(
                outputs["stuff_things_scores"][i]
            )
            thing_mask = stuff_things_scores > 0.5
            stuff_mask = ~thing_mask

            # Panoptic merge
            pan_ids, inst_ids = panoptic_merge(
                jnp.array(semantic_pred),
                jnp.array(instance_masks),
                jnp.array(instance_scores),
                jnp.array(thing_mask),
                jnp.array(stuff_mask),
            )

            all_pred_semantic.append(semantic_pred)

            # Store per-image panoptic data for PQ
            img_data = {
                "semantic_pred": semantic_pred,
                "inst_ids": np.array(inst_ids),
            }

            # Collect ground truth if available
            sample = samples[i]
            h, w = config["data"]["image_size"]
            sp_h = h // 8
            sp_w = w // 8

            if "semantic_label" in sample:
                gt_semantic = np.array(sample["semantic_label"])
                if gt_semantic.ndim > 1:
                    gt_semantic = gt_semantic.flatten()

                # Remap Cityscapes original IDs to train IDs (0-18)
                dataset_name = config["data"].get(
                    "dataset", config["data"].get("dataset_name", "")
                )
                if "cityscapes" in dataset_name.lower():
                    gt_semantic = remap_cityscapes_labels(gt_semantic)

                # Match token resolution
                if gt_semantic.shape[0] != semantic_pred.shape[0]:
                    from jax.image import resize
                    gt_2d = gt_semantic.reshape(h, w)
                    gt_semantic = np.array(
                        resize(
                            jnp.array(gt_2d)[..., None].astype(jnp.float32),
                            (sp_h, sp_w, 1),
                            method="nearest",
                        )
                    ).squeeze(-1).flatten().astype(np.int32)

                all_gt_semantic.append(gt_semantic)
                img_data["gt_semantic"] = gt_semantic

            # Collect GT instance labels for PQ
            if "instance_label" in sample:
                gt_instance = np.array(sample["instance_label"])
                if gt_instance.ndim > 1:
                    gt_instance = gt_instance.flatten()
                # Downsample to patch resolution
                if gt_instance.shape[0] != semantic_pred.shape[0]:
                    from jax.image import resize
                    gt_inst_2d = gt_instance.reshape(h, w)
                    gt_instance = np.array(
                        resize(
                            jnp.array(gt_inst_2d)[..., None].astype(jnp.float32),
                            (sp_h, sp_w, 1),
                            method="nearest",
                        )
                    ).squeeze(-1).flatten().astype(np.int32)
                img_data["gt_instance"] = gt_instance

            all_per_image.append(img_data)

            num_samples += 1
            if num_samples % 100 == 0:
                logging.info(f"Evaluated {num_samples} samples...")

    eval_time = time.time() - start_time
    logging.info(
        f"Evaluation complete: {num_samples} samples in {eval_time:.1f}s "
        f"({eval_time/max(num_samples,1):.3f}s/sample)"
    )

    results = {"num_samples": num_samples, "eval_time_s": eval_time}

    # Compute metrics if we have GT
    if all_gt_semantic:
        pred_flat = np.concatenate(all_pred_semantic)
        gt_flat = np.concatenate(all_gt_semantic)

        num_classes = config["architecture"]["num_classes"]
        num_pred = int(np.max(pred_flat)) + 1

        # Hungarian matching + mIoU
        mapping, accuracy = hungarian_match(
            pred_flat,
            gt_flat,
            num_pred_clusters=max(num_pred, num_classes),
            num_gt_classes=num_classes,
            ignore_label=config.get("evaluation", {}).get("ignore_label", 255),
        )

        miou, per_class_iou = compute_miou(
            pred_flat, gt_flat, mapping, num_classes,
            ignore_label=config.get("evaluation", {}).get("ignore_label", 255),
        )

        results["accuracy"] = accuracy
        results["miou"] = miou
        results["per_class_iou"] = per_class_iou.tolist()
        results["cluster_mapping"] = {str(k): v for k, v in mapping.items()}

        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"mIoU: {miou:.4f}")

        # ---------------------------------------------------------------
        # Compute PQ, SQ, RQ using the Hungarian mapping
        # ---------------------------------------------------------------
        # Determine stuff/thing class sets
        data_cfg = config.get("data", {})
        num_stuff = len(data_cfg.get("stuff_classes", []))
        num_thing = len(data_cfg.get("thing_classes", []))
        if num_stuff == 0 and num_thing == 0:
            # Fallback: all classes are stuff
            num_stuff = num_classes
        stuff_class_ids = set(range(num_stuff))
        thing_class_ids = set(range(num_stuff, num_stuff + num_thing))
        all_class_ids = stuff_class_ids | thing_class_ids
        label_divisor = 1000
        ignore_label = config.get("evaluation", {}).get("ignore_label", 255)

        # Accumulate per-class TP, FP, FN, IoU across all images
        pq_stat = {
            c: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0}
            for c in all_class_ids
        }

        num_pq_images = 0
        for img_data in all_per_image:
            if "gt_semantic" not in img_data:
                continue

            gt_semantic = img_data["gt_semantic"]
            gt_instance = img_data.get("gt_instance")
            semantic_pred = img_data["semantic_pred"]
            inst_ids = img_data["inst_ids"]

            # Remap predicted semantic clusters to GT class IDs
            mapped_pred = np.full_like(semantic_pred, ignore_label)
            for pred_c, gt_c in mapping.items():
                mapped_pred[semantic_pred == pred_c] = gt_c

            # Build predicted panoptic map
            # Use (inst_ids + 1) so stuff (inst_ids=0) gets segment_id=1,
            # avoiding collision between class_id=0 and ignore sentinel
            pred_pan = (np.array(inst_ids) + 1) * label_divisor + mapped_pred
            # Mark ignore pixels with -1
            pred_pan[mapped_pred == ignore_label] = -1

            pred_segments = []
            for pan_id in np.unique(pred_pan):
                if pan_id < 0:
                    continue
                cat = int(pan_id % label_divisor)
                if cat == ignore_label or cat not in all_class_ids:
                    continue
                area = int(np.sum(pred_pan == pan_id))
                if area == 0:
                    continue
                pred_segments.append({"id": int(pan_id), "category_id": cat})

            # Build GT panoptic map
            gt_pan = np.zeros_like(gt_semantic, dtype=np.int32)
            gt_segments = []
            seg_counter = 1

            for cls_id in sorted(all_class_ids):
                cls_mask = gt_semantic == cls_id
                if not np.any(cls_mask):
                    continue

                if cls_id in stuff_class_ids:
                    # Stuff: one segment per class per image
                    pan_id = seg_counter * label_divisor + cls_id
                    gt_pan[cls_mask] = pan_id
                    gt_segments.append(
                        {"id": pan_id, "category_id": cls_id}
                    )
                    seg_counter += 1
                else:
                    # Thing: use instance labels to separate instances
                    if gt_instance is not None:
                        unique_insts = np.unique(gt_instance[cls_mask])
                        for uid in unique_insts:
                            inst_mask = cls_mask & (gt_instance == uid)
                            if np.sum(inst_mask) == 0:
                                continue
                            pan_id = seg_counter * label_divisor + cls_id
                            gt_pan[inst_mask] = pan_id
                            gt_segments.append(
                                {"id": pan_id, "category_id": cls_id}
                            )
                            seg_counter += 1
                    else:
                        # No instance labels — treat whole class as one
                        pan_id = seg_counter * label_divisor + cls_id
                        gt_pan[cls_mask] = pan_id
                        gt_segments.append(
                            {"id": pan_id, "category_id": cls_id}
                        )
                        seg_counter += 1

            # Match pred segments to GT segments per-image
            img_gt_matched = set()
            img_pred_matched = set()

            for gt_seg in gt_segments:
                gt_id = gt_seg["id"]
                gt_cat = gt_seg["category_id"]
                gt_mask = gt_pan == gt_id
                gt_area = np.sum(gt_mask)
                if gt_area == 0:
                    continue

                best_iou = 0.0
                best_pred_id = None

                for pred_seg in pred_segments:
                    if pred_seg["category_id"] != gt_cat:
                        continue
                    pred_mask = pred_pan == pred_seg["id"]
                    intersection = np.sum(pred_mask & gt_mask)
                    if intersection == 0:
                        continue
                    union = np.sum(pred_mask | gt_mask)
                    iou = intersection / union
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_id = pred_seg["id"]

                if best_iou > 0.5:
                    pq_stat[gt_cat]["tp"] += 1
                    pq_stat[gt_cat]["iou_sum"] += best_iou
                    img_gt_matched.add(gt_id)
                    if best_pred_id is not None:
                        img_pred_matched.add(best_pred_id)
                else:
                    pq_stat[gt_cat]["fn"] += 1

            # Count FP: unmatched predicted segments
            for pred_seg in pred_segments:
                if pred_seg["id"] not in img_pred_matched:
                    cat = pred_seg["category_id"]
                    if cat in all_class_ids:
                        pq_stat[cat]["fp"] += 1

            num_pq_images += 1

        # Compute final PQ, SQ, RQ per class
        pq_per_class = {}
        sq_per_class = {}
        rq_per_class = {}
        for c in all_class_ids:
            tp = pq_stat[c]["tp"]
            fp = pq_stat[c]["fp"]
            fn = pq_stat[c]["fn"]
            iou_sum = pq_stat[c]["iou_sum"]
            denom = tp + 0.5 * fp + 0.5 * fn
            if denom > 0:
                sq_c = iou_sum / max(tp, 1)
                rq_c = tp / denom
                pq_per_class[c] = sq_c * rq_c
                sq_per_class[c] = sq_c
                rq_per_class[c] = rq_c
            else:
                pq_per_class[c] = 0.0
                sq_per_class[c] = 0.0
                rq_per_class[c] = 0.0

        # Aggregate: only over classes that appear in GT
        active_classes = [
            c for c in all_class_ids
            if pq_stat[c]["tp"] + pq_stat[c]["fn"] + pq_stat[c]["fp"] > 0
        ]
        active_stuff = [c for c in active_classes if c in stuff_class_ids]
        active_thing = [c for c in active_classes if c in thing_class_ids]

        pq_val = float(np.mean([pq_per_class[c] for c in active_classes])) if active_classes else 0.0
        sq_val = float(np.mean([sq_per_class[c] for c in active_classes])) if active_classes else 0.0
        rq_val = float(np.mean([rq_per_class[c] for c in active_classes])) if active_classes else 0.0
        pq_stuff = float(np.mean([pq_per_class[c] for c in active_stuff])) if active_stuff else 0.0
        pq_thing = float(np.mean([pq_per_class[c] for c in active_thing])) if active_thing else 0.0

        results["pq"] = pq_val
        results["sq"] = sq_val
        results["rq"] = rq_val
        results["pq_stuff"] = pq_stuff
        results["pq_things"] = pq_thing
        results["pq_per_class"] = {str(c): pq_per_class[c] for c in all_class_ids}
        results["pq_num_images"] = num_pq_images

        logging.info(f"PQ:  {pq_val:.4f}")
        logging.info(f"SQ:  {sq_val:.4f}")
        logging.info(f"RQ:  {rq_val:.4f}")
        logging.info(f"PQ_St: {pq_stuff:.4f}  PQ_Th: {pq_thing:.4f}")
        logging.info(
            f"PQ computed over {num_pq_images} images, "
            f"{len(active_classes)} active classes "
            f"({len(active_stuff)} stuff, {len(active_thing)} thing)"
        )

    return results


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="MBPS Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument(
        "--num_devices",
        type=int,
        default=None,
        help="Number of devices (default: all local devices)",
    )
    parser.add_argument(
        "--multihost",
        action="store_true",
        help="Initialize JAX distributed for TPU pod evaluation",
    )
    args = parser.parse_args()

    # On TPU pods, all workers must participate in JAX distributed init
    # and run the same program. All workers evaluate independently using
    # their local 4 devices; only process 0 saves results.
    _is_multihost = args.multihost
    if _is_multihost:
        jax.distributed.initialize()
        logging.set_verbosity(logging.INFO)
        logging.info(
            f"Process {jax.process_index()}/{jax.process_count()}: "
            f"distributed init done, {jax.local_device_count()} local devices"
        )
    logging.set_verbosity(logging.INFO)
    config = load_config(args.config)

    num_devices = args.num_devices or jax.local_device_count()

    # Print device info
    devices = jax.local_devices()
    print("\n" + "=" * 60)
    print("  MBPS Evaluation")
    print("=" * 60)
    print(f"  JAX version:      {jax.__version__}")
    print(f"  Backend:          {jax.default_backend()}")
    print(f"  Local devices:    {len(devices)}")
    print(f"  Using devices:    {num_devices}")
    for i in range(num_devices):
        print(f"    Device {i}:       {devices[i]}")
    if num_devices > 1:
        print(f"  Parallel eval:    ENABLED (pmap across {num_devices} devices)")
    print("=" * 60 + "\n")

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

    # Initialize model to get param structure
    rng = jax.random.PRNGKey(0)
    image_size = tuple(config["data"]["image_size"])
    dummy_input = {
        "image": jnp.zeros((1,) + image_size + (3,)),
        "depth": jnp.zeros((1,) + image_size),
    }
    init_params = model.init(rng, **dummy_input)

    # Load checkpoint (flat dict)
    ckpt_manager = CheckpointManager(checkpoint_dir="")
    ckpt = ckpt_manager.load(args.checkpoint)
    ckpt_flat = ckpt["params"]
    logging.info(f"Loaded checkpoint from {args.checkpoint}")

    # Restore params by walking init_params tree and replacing from flat ckpt
    def _restore_from_flat(tree, flat_dict, prefix="params"):
        if isinstance(tree, dict):
            return {
                k: _restore_from_flat(v, flat_dict, f"{prefix}_{k}")
                for k, v in tree.items()
            }
        elif isinstance(tree, (list, tuple)):
            return type(tree)(
                _restore_from_flat(v, flat_dict, f"{prefix}_{i}")
                for i, v in enumerate(tree)
            )
        else:
            if prefix in flat_dict:
                return jnp.array(flat_dict[prefix])
            return tree

    params = _restore_from_flat(init_params, ckpt_flat)
    logging.info(f"Restored {len(ckpt_flat)} params from checkpoint")

    # Create eval dataset
    transform = EvalTransform()
    dataset = get_dataset(
        dataset_name=config["data"].get("dataset_name", config["data"].get("dataset", "cityscapes")),
        data_dir=config["data"]["data_dir"],
        depth_dir=config["data"].get("depth_dir", ""),
        split=args.split,
        image_size=tuple(config["data"]["image_size"]),
        transforms=transform,
    )

    logging.info(f"Loaded {len(dataset)} samples for evaluation")

    # Run evaluation
    results = evaluate_model(
        model, params, dataset, config,
        use_crf=args.use_crf, num_devices=num_devices,
    )

    # Only process 0 saves results and prints summary (all workers compute)
    _is_coordinator = not _is_multihost or jax.process_index() == 0

    if _is_coordinator:
        # Override output path with GCS if configured
        gcs_results_dir = config.get("gcs", {}).get("results_dir")
        output_path = args.output
        if gcs_results_dir:
            vm_name = os.environ.get("MBPS_VM_NAME", "local")
            experiment = os.environ.get("MBPS_EXPERIMENT", "default")
            output_path = os.path.join(
                gcs_results_dir, experiment, vm_name, "eval_results.json"
            )

        # Convert numpy types for JSON serialization
        def _json_safe(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        results = json.loads(json.dumps(results, default=_json_safe))

        # Save results (works with both local and gs:// paths)
        output_dir = os.path.dirname(os.path.abspath(output_path)) if not output_path.startswith("gs://") else os.path.dirname(output_path)
        tf.io.gfile.makedirs(output_dir)
        with tf.io.gfile.GFile(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {output_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("  MBPS Evaluation Results")
        print("=" * 60)

        # Print key metrics first
        key_metrics = ["pq", "sq", "rq", "pq_stuff", "pq_things", "miou", "accuracy"]
        for k in key_metrics:
            if k in results:
                print(f"  {k:20s}: {results[k]:.4f}")

        # Print other scalar metrics
        for k, v in results.items():
            if k in key_metrics or k in ("per_class_iou", "cluster_mapping", "pq_per_class"):
                continue
            if isinstance(v, float):
                print(f"  {k:20s}: {v:.4f}")
            elif isinstance(v, (int, str)):
                print(f"  {k:20s}: {v}")

        # Print per-class PQ if available
        if "pq_per_class" in results:
            print("\n  Per-class PQ:")
            data_cfg = config.get("data", {})
            stuff_names = data_cfg.get("stuff_classes", [])
            thing_names = data_cfg.get("thing_classes", [])
            all_names = stuff_names + thing_names
            for cls_str, pq_val in sorted(
                results["pq_per_class"].items(), key=lambda x: int(x[0])
            ):
                cls_id = int(cls_str)
                name = all_names[cls_id] if cls_id < len(all_names) else f"class_{cls_id}"
                kind = "St" if cls_id < len(stuff_names) else "Th"
                print(f"    [{kind}] {name:20s}: {pq_val:.4f}")

        print("=" * 60)
    else:
        logging.info(
            f"Process {jax.process_index()}: evaluation done "
            f"(results saved by process 0 only)"
        )


if __name__ == "__main__":
    main()
