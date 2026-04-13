#!/usr/bin/env python3
"""Evaluate trained PICL projection head on Cityscapes val.

Loads a trained PICLProjectionHead checkpoint and runs instance decomposition
via picl_instances (PICL features + HDBSCAN) with a hyperparameter sweep.

Usage:
    # Single config
    python mbps_pytorch/eval_picl.py \
        --cityscapes_root /path/to/cityscapes \
        --checkpoint checkpoints/picl/round1/best.pth

    # Sweep HDBSCAN params (12 configs)
    python mbps_pytorch/eval_picl.py \
        --cityscapes_root /path/to/cityscapes \
        --checkpoint checkpoints/picl/round1/best.pth \
        --sweep

    # Quick check on 50 images
    python mbps_pytorch/eval_picl.py \
        --cityscapes_root /path/to/cityscapes \
        --checkpoint checkpoints/picl/round1/best.pth \
        --max_images 50
"""

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from ablate_instance_methods import (
    NUM_CLASSES, THING_IDS,
    discover_files, remap_gt_to_trainids, resize_nearest,
    evaluate_panoptic_single, compute_pq_from_accumulators,
)
from instance_methods import METHODS
from instance_methods.utils import load_features
from train_picl import PICLProjectionHead


SWEEP_GRID = {
    "hdbscan_min_cluster": [3, 5, 8],
    "hdbscan_min_samples": [2, 3, 5],
    "min_area": [500, 1000],
}

DEFAULT_CONFIG = {
    "hdbscan_min_cluster": 5,
    "hdbscan_min_samples": 3,
    "min_area": 1000,
}


def expand_grid(grid_dict: dict) -> list:
    """Expand parameter grid into list of config dicts."""
    keys = list(grid_dict.keys())
    values = list(grid_dict.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained PICLProjectionHead from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = PICLProjectionHead(
        input_dim=cfg.get("input_dim", 771),
        hidden_dim=cfg.get("hidden_dim", 512),
        embed_dim=cfg.get("embed_dim", 128),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Loaded: epoch={ckpt['epoch']}, loss={ckpt['loss']:.4f}, "
          f"PQ_things={ckpt.get('PQ_things', 'N/A')}, "
          f"embed_std={ckpt.get('embed_std', 'N/A'):.4f}")
    print(f"  Pair mining: {cfg.get('pair_mining', 'unknown')}  "
          f"(instance_mask = PICL, depth_proximity = old failed approach)")
    return model, cfg


def run_eval(
    model,
    model_cfg: dict,
    files: list,
    eval_hw: tuple,
    config: dict,
    label: str = "",
) -> dict:
    """Evaluate one HDBSCAN config on all images."""
    from instance_methods.picl_embed import picl_instances

    H, W = eval_hw
    depth_weight = model_cfg.get("depth_weight", 2.0)
    pos_weight = model_cfg.get("pos_weight", 0.5)

    tp_acc = np.zeros(NUM_CLASSES)
    fp_acc = np.zeros(NUM_CLASSES)
    fn_acc = np.zeros(NUM_CLASSES)
    iou_acc = np.zeros(NUM_CLASSES)
    total_instances = 0
    total_time = 0.0
    n_images = 0
    errors = 0

    for sem_path, depth_path, gt_label_path, gt_inst_path, feat_path in tqdm(
        files, desc=label, leave=False
    ):
        pred_sem = np.array(Image.open(sem_path))
        if pred_sem.shape != (H, W):
            pred_sem = resize_nearest(pred_sem, eval_hw)

        depth = np.load(str(depth_path))
        if depth.shape != (H, W):
            depth = np.array(
                Image.fromarray(depth).resize((W, H), Image.BILINEAR)
            )

        features = load_features(str(feat_path)) if feat_path else None

        kwargs = dict(config)
        kwargs["thing_ids"] = THING_IDS
        kwargs["features"] = features
        kwargs["model"] = model
        kwargs["depth_weight"] = depth_weight
        kwargs["pos_weight"] = pos_weight
        kwargs["dilation_iters"] = 3

        t0 = time.time()
        try:
            instances = picl_instances(pred_sem, depth, **kwargs)
        except Exception as e:
            if errors < 3:
                tqdm.write(f"  [WARN] {label}: {type(e).__name__}: {e}")
            errors += 1
            instances = []
        total_time += time.time() - t0

        gt_raw = np.array(Image.open(gt_label_path))
        gt_sem = remap_gt_to_trainids(gt_raw)
        if gt_sem.shape != (H, W):
            gt_sem = resize_nearest(gt_sem, eval_hw)

        gt_inst_map = np.array(Image.open(gt_inst_path), dtype=np.int32)
        if gt_inst_map.shape != (H, W):
            gt_inst_map = np.array(
                Image.fromarray(gt_inst_map).resize((W, H), Image.NEAREST)
            )

        tp, fp, fn, iou_s, n_inst = evaluate_panoptic_single(
            pred_sem, instances, gt_sem, gt_inst_map, eval_hw
        )
        tp_acc += tp
        fp_acc += fp
        fn_acc += fn
        iou_acc += iou_s
        total_instances += n_inst
        n_images += 1

    metrics = compute_pq_from_accumulators(tp_acc, fp_acc, fn_acc, iou_acc)
    metrics["avg_instances"] = round(total_instances / max(n_images, 1), 1)
    metrics["n_images"] = n_images
    metrics["seconds_per_image"] = round(total_time / max(n_images, 1), 3)
    metrics["total_seconds"] = round(total_time, 1)
    metrics["errors"] = errors
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained PICL projection head"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained PICL .pth checkpoint")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep HDBSCAN hyperparameters")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--eval_size", type=int, nargs=2, default=[512, 1024])
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_mapped_k80")
    parser.add_argument("--depth_subdir", type=str, default="depth_spidepth")
    parser.add_argument("--feature_subdir", type=str, default="dinov2_features")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--round_id", type=int, default=1,
                        help="Round number for output filename")
    parser.add_argument("--output_dir", type=str,
                        default="results/ablation_picl")
    args = parser.parse_args()

    cs_root = Path(args.cityscapes_root)
    eval_hw = tuple(args.eval_size)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\n{'='*70}")
    print(f"PICL EVALUATION  |  Round {args.round_id}")
    print(f"{'='*70}")
    print(f"  Checkpoint: {args.checkpoint}")
    model, model_cfg = load_model(args.checkpoint, device)

    files = discover_files(
        cs_root, args.split, args.semantic_subdir,
        args.depth_subdir, args.feature_subdir
    )
    if args.max_images:
        files = files[:args.max_images]
    print(f"  Images: {len(files)}, Device: {device}")

    configs = expand_grid(SWEEP_GRID) if args.sweep else [DEFAULT_CONFIG]
    print(f"  Configs: {len(configs)}")

    all_results = []
    for i, config in enumerate(configs):
        label = (f"[{i+1}/{len(configs)}] picl "
                 f"mc={config['hdbscan_min_cluster']} "
                 f"ms={config['hdbscan_min_samples']} "
                 f"A={config['min_area']}")
        print(f"\n  {label}")
        metrics = run_eval(model, model_cfg, files, eval_hw, config, label=label)
        print(f"    PQ={metrics['PQ']:5.2f}  PQ_st={metrics['PQ_stuff']:5.2f}  "
              f"PQ_th={metrics['PQ_things']:5.2f}  "
              f"inst/img={metrics['avg_instances']:4.1f}  "
              f"({metrics['seconds_per_image']:.2f}s/img)")

        all_results.append({
            "method": "picl",
            "round_id": args.round_id,
            "checkpoint": str(args.checkpoint),
            "training_config": model_cfg,
            "eval_config": config,
            **metrics,
        })

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"picl_round{args.round_id}_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump({
            "method": "picl",
            "round_id": args.round_id,
            "checkpoint": str(args.checkpoint),
            "training_config": model_cfg,
            "results": all_results,
            "n_images": len(files),
            "split": args.split,
        }, f, indent=2)
    print(f"\n  Saved: {out_path}")

    best = max(all_results, key=lambda r: r["PQ_things"])
    print(f"\n  Best PQ_things: {best['PQ_things']:.2f}  "
          f"(config: mc={best['eval_config']['hdbscan_min_cluster']} "
          f"ms={best['eval_config']['hdbscan_min_samples']} "
          f"A={best['eval_config']['min_area']})")

    # Save best PQ_things for shell script to read
    with open(output_dir / f"round{args.round_id}_best_pq_things.txt", "w") as f:
        f.write(str(best["PQ_things"]))


if __name__ == "__main__":
    main()
