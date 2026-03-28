"""MBPS Evaluation Script (PyTorch).

Uses torch.no_grad() for efficient inference across available devices.

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
import logging
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbps_pytorch.data.datasets import get_dataset
from mbps_pytorch.data.transforms import EvalTransform
from mbps_pytorch.evaluation.hungarian_matching import compute_miou, hungarian_match
from mbps_pytorch.evaluation.panoptic_quality import compute_panoptic_quality
from mbps_pytorch.models.mbps_model import MBPSModel
from mbps_pytorch.models.merger.crf_postprocess import crf_inference
from mbps_pytorch.models.merger.panoptic_merge import panoptic_merge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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
    model: nn.Module,
    dataset,
    config: Dict[str, Any],
    device: torch.device,
    use_crf: bool = False,
    num_devices: int = 1,
) -> Dict[str, Any]:
    """Run evaluation on entire dataset with torch.no_grad() inference.

    Args:
        model: MBPS model (already loaded with weights).
        dataset: Evaluation dataset.
        config: Configuration dict.
        device: Torch device.
        use_crf: Whether to apply CRF post-processing.
        num_devices: Number of devices (for DataParallel).

    Returns:
        Dict with evaluation metrics.
    """
    model.eval()

    # Wrap in DataParallel if multiple GPUs are available
    if num_devices > 1 and torch.cuda.is_available():
        eval_model = nn.DataParallel(model)
    else:
        eval_model = model

    all_pred_semantic = []
    all_gt_semantic = []
    all_pq_results = []
    num_samples = 0

    logger.info(
        f"Starting evaluation with {num_devices} device(s)..."
    )
    start_time = time.time()

    dataset_len = len(dataset)
    for idx in range(dataset_len):
        sample = dataset[idx]

        # Prepare inputs
        image = torch.tensor(
            sample["image"], dtype=torch.float32
        ).unsqueeze(0).to(device)  # (1, H, W, 3)
        depth = torch.tensor(
            sample["depth"] if "depth" in sample
            else np.zeros(image.shape[1:3]),
            dtype=torch.float32,
        ).unsqueeze(0).to(device)  # (1, H, W)

        # Forward pass
        with torch.no_grad():
            outputs = eval_model(
                image=image,
                depth=depth,
                use_bridge=True,
                deterministic=True,
            )

        # Extract predictions
        semantic_pred = outputs["semantic_pred"][0].cpu().numpy()  # (N,)
        instance_masks = outputs["instance_masks"][0].cpu().numpy()  # (M, N)
        instance_scores = outputs["instance_scores"][0].cpu().numpy()  # (M,)

        # Optional CRF refinement
        if use_crf:
            h, w = config["data"]["image_size"]
            n = semantic_pred.shape[0]
            spatial_h = int(n ** 0.5)
            spatial_w = n // spatial_h

            semantic_logits = outputs["semantic_probs"][0].cpu().numpy()
            image_np = sample["image"]
            image_resized = F.interpolate(
                torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).float(),
                size=(spatial_h, spatial_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0).numpy().reshape(-1, 3)

            refined = crf_inference(
                torch.tensor(np.log(semantic_logits + 1e-8)),
                torch.tensor(image_resized),
                spatial_h=spatial_h,
                spatial_w=spatial_w,
            )
            semantic_pred = refined.argmax(dim=-1).numpy()

        # Stuff-things classification
        stuff_things_scores = outputs["stuff_things_scores"][0].cpu().numpy()
        thing_mask = stuff_things_scores > 0.5
        stuff_mask = ~thing_mask

        # Panoptic merge
        pan_ids, inst_ids = panoptic_merge(
            torch.tensor(semantic_pred),
            torch.tensor(instance_masks),
            torch.tensor(instance_scores),
            torch.tensor(thing_mask),
            torch.tensor(stuff_mask),
        )

        all_pred_semantic.append(semantic_pred)

        # Collect ground truth if available
        if "label" in sample:
            gt_semantic = np.array(sample["label"])
            if gt_semantic.ndim > 1:
                gt_semantic = gt_semantic.flatten()
            # Match token resolution
            if gt_semantic.shape[0] != semantic_pred.shape[0]:
                gt_2d = gt_semantic.reshape(
                    config["data"]["image_size"][0],
                    config["data"]["image_size"][1],
                )
                spatial_h = int(semantic_pred.shape[0] ** 0.5)
                spatial_w = semantic_pred.shape[0] // spatial_h
                gt_resized = F.interpolate(
                    torch.tensor(gt_2d).float().unsqueeze(0).unsqueeze(0),
                    size=(spatial_h, spatial_w),
                    mode="nearest",
                ).squeeze().numpy().astype(np.int32).flatten()
                gt_semantic = gt_resized

            all_gt_semantic.append(gt_semantic)

        num_samples += 1
        if num_samples % 100 == 0:
            logger.info(f"Evaluated {num_samples} samples...")

    eval_time = time.time() - start_time
    logger.info(
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

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"mIoU: {miou:.4f}")

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
        help="Number of devices (default: all available)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_config(args.config)

    # Determine device
    if torch.cuda.is_available():
        num_devices = args.num_devices or torch.cuda.device_count()
        device = torch.device("cuda")
    else:
        num_devices = 1
        device = torch.device("cpu")

    # Print device info
    print("\n" + "=" * 60)
    print("  MBPS Evaluation (PyTorch)")
    print("=" * 60)
    print(f"  PyTorch version:  {torch.__version__}")
    print(f"  CUDA available:   {torch.cuda.is_available()}")
    print(f"  Using devices:    {num_devices}")
    if torch.cuda.is_available():
        for i in range(num_devices):
            print(f"    Device {i}:       {torch.cuda.get_device_name(i)}")
    if num_devices > 1:
        print(f"  Parallel eval:    ENABLED (DataParallel across {num_devices} devices)")
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

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "params" in checkpoint:
        model.load_state_dict(checkpoint["params"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Create eval dataset
    transform = EvalTransform(
        image_size=tuple(config["data"]["image_size"]),
    )
    dataset = get_dataset(
        dataset_name=config["data"].get(
            "dataset_name", config["data"].get("dataset", "cityscapes")
        ),
        data_dir=config["data"]["data_dir"],
        depth_dir=config["data"].get("depth_dir", ""),
        split=args.split,
        image_size=tuple(config["data"]["image_size"]),
        transforms=transform,
    )

    logger.info(f"Loaded {len(dataset)} samples for evaluation")

    # Run evaluation
    results = evaluate_model(
        model, dataset, config,
        device=device,
        use_crf=args.use_crf,
        num_devices=num_devices,
    )

    # Save results
    output_path = args.output
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  MBPS Evaluation Results")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        elif isinstance(v, (int, str)):
            print(f"  {k:20s}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
