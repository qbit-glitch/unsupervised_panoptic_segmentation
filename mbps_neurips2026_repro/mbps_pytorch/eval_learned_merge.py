#!/usr/bin/env python3
"""Evaluate learned merge (and feature-merge baseline) on Cityscapes val.

Two modes:
  1. feature_cosine: training-free cosine similarity merge (Step 0 baseline)
  2. learned: trained MergePredictor model (Step 3)

Usage:
    # Step 0: Feature merge baseline (no training)
    python mbps_pytorch/eval_learned_merge.py \
        --cityscapes_root /path/to/cityscapes \
        --mode feature_cosine --sweep

    # Step 3: Learned merge
    python mbps_pytorch/eval_learned_merge.py \
        --cityscapes_root /path/to/cityscapes \
        --mode learned \
        --checkpoint checkpoints/merge_predictor/best.pth \
        --pca_path checkpoints/merge_predictor/pca.npz \
        --sweep
"""

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from ablate_instance_methods import (
    CS_ID_TO_TRAIN, NUM_CLASSES, STUFF_IDS, THING_IDS,
    CLASS_NAMES, WORK_H, WORK_W,
    discover_files, remap_gt_to_trainids, resize_nearest,
    evaluate_panoptic_single, compute_pq_from_accumulators,
)
from instance_methods.learned_merge import learned_merge_instances
from instance_methods.utils import load_features


SWEEP_GRID_COSINE = {
    "grad_threshold": [0.10, 0.15, 0.20],
    "merge_threshold": [0.60, 0.70, 0.80, 0.90],
    "min_area": [500, 1000],
}

SWEEP_GRID_LEARNED = {
    "grad_threshold": [0.05, 0.10, 0.15],
    "merge_threshold": [0.30, 0.50, 0.70],
    "min_area": [500, 1000],
}

DEFAULT_CONFIG_COSINE = {
    "grad_threshold": 0.10,
    "merge_threshold": 0.80,
    "min_area": 1000,
}

DEFAULT_CONFIG_LEARNED = {
    "grad_threshold": 0.10,
    "merge_threshold": 0.50,
    "min_area": 1000,
}


def expand_grid(grid_dict: dict) -> list:
    """Expand parameter grid into list of config dicts."""
    keys = list(grid_dict.keys())
    values = list(grid_dict.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def load_merge_model(checkpoint_path: str, device):
    """Load trained MergePredictor from checkpoint."""
    import torch
    from train_merge_predictor import MergePredictor

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MergePredictor(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg.get("hidden_dim", 128),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Loaded MergePredictor: epoch={ckpt['epoch']}, "
          f"acc={ckpt.get('val_accuracy', 'N/A')}")
    return model, cfg


def load_pca(pca_path: str):
    """Load fitted PCA from NPZ."""
    from sklearn.decomposition import PCA

    data = np.load(pca_path)
    pca = PCA(n_components=data["components"].shape[0])
    pca.components_ = data["components"]
    pca.mean_ = data["mean"]
    pca.explained_variance_ = data.get(
        "explained_variance", np.ones(pca.components_.shape[0])
    )
    return pca


def run_eval(files, k_to_trainid, eval_hw, config, mode,
             model=None, pca=None, label=""):
    """Evaluate one config on all images."""
    H, W = eval_hw

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
        pred_k = np.array(Image.open(sem_path))
        pred_sem = k_to_trainid[pred_k]
        if pred_sem.shape != (H, W):
            pred_sem = resize_nearest(pred_sem, eval_hw)

        depth = np.load(str(depth_path))
        if depth.shape != (H, W):
            depth = np.array(
                Image.fromarray(depth).resize((W, H), Image.BILINEAR)
            )

        features = None
        if feat_path is not None:
            features = load_features(str(feat_path))

        kwargs = {
            "thing_ids": THING_IDS,
            "features": features,
            "grad_threshold": config["grad_threshold"],
            "merge_threshold": config["merge_threshold"],
            "min_area": config["min_area"],
            "dilation_iters": 3,
            "depth_blur_sigma": 1.0,
            "mode": mode,
            "model": model,
            "pca": pca,
        }

        t0 = time.time()
        try:
            instances = learned_merge_instances(pred_sem, depth, **kwargs)
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
        description="Evaluate learned merge / feature merge baseline"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--mode", type=str, default="feature_cosine",
                        choices=["feature_cosine", "learned"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained MergePredictor .pth (for mode=learned)")
    parser.add_argument("--pca_path", type=str, default=None,
                        help="Path to PCA .npz (for mode=learned)")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--eval_size", type=int, nargs=2, default=[512, 1024])
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_raw_k80")
    parser.add_argument("--depth_subdir", type=str, default="depth_spidepth")
    parser.add_argument("--feature_subdir", type=str,
                        default="dinov2_features")
    parser.add_argument("--centroids_path", type=str, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--output_dir", type=str,
                        default="results/ablation_instance_methods")
    parser.add_argument("--grad_thresholds", type=float, nargs="+",
                        default=None,
                        help="Override sweep grad_threshold values "
                             "(e.g., --grad_thresholds 0.01 0.02 0.03)")
    args = parser.parse_args()

    cs_root = Path(args.cityscapes_root)
    eval_hw = tuple(args.eval_size)

    # Device
    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model and PCA (for learned mode)
    model, pca = None, None
    if args.mode == "learned":
        if not args.checkpoint:
            parser.error("--checkpoint required for mode=learned")
        model, model_cfg = load_merge_model(args.checkpoint, device)
        if args.pca_path:
            pca = load_pca(args.pca_path)

    print(f"\n{'='*70}")
    print(f"LEARNED MERGE EVALUATION — mode={args.mode}")
    print(f"{'='*70}")

    # Load cluster mapping
    centroids_path = args.centroids_path or str(
        cs_root / args.semantic_subdir / "kmeans_centroids.npz"
    )
    data = np.load(centroids_path)
    c2c = data["cluster_to_class"]
    k_to_trainid = np.full(256, 255, dtype=np.uint8)
    for cid, tid in enumerate(c2c):
        k_to_trainid[cid] = int(tid)

    # Discover files
    files = discover_files(
        cs_root, args.split, args.semantic_subdir,
        args.depth_subdir, args.feature_subdir
    )
    if args.max_images:
        files = files[:args.max_images]
    print(f"  Images: {len(files)}, Device: {device}")

    # Build configs
    if args.sweep:
        grid = dict(SWEEP_GRID_COSINE if args.mode == "feature_cosine"
                    else SWEEP_GRID_LEARNED)
        if args.grad_thresholds:
            grid["grad_threshold"] = args.grad_thresholds
        configs = expand_grid(grid)
    else:
        configs = [DEFAULT_CONFIG_COSINE if args.mode == "feature_cosine"
                   else DEFAULT_CONFIG_LEARNED]
    print(f"  Configs: {len(configs)}")

    # Run
    all_results = []
    for i, config in enumerate(configs):
        label = (f"[{i+1}/{len(configs)}] {args.mode} "
                 f"[τ={config['grad_threshold']}, "
                 f"mt={config['merge_threshold']}, "
                 f"A={config['min_area']}]")
        print(f"\n  {label}")
        t0 = time.time()
        metrics = run_eval(
            files, k_to_trainid, eval_hw, config,
            mode=args.mode, model=model, pca=pca, label=label
        )
        dt = time.time() - t0
        print(f"    PQ={metrics['PQ']:5.2f}  PQ_st={metrics['PQ_stuff']:5.2f}  "
              f"PQ_th={metrics['PQ_things']:5.2f}  "
              f"inst/img={metrics['avg_instances']:4.1f}  "
              f"({metrics['seconds_per_image']:.2f}s/img, {dt:.0f}s total)")

        all_results.append({
            "method": f"learned_merge_{args.mode}",
            "config": config,
            **metrics,
        })

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"ablation_learned_merge_{args.mode}_val.json"
    with open(out_path, "w") as f:
        json.dump({
            "method": f"learned_merge_{args.mode}",
            "mode": args.mode,
            "results": all_results,
            "n_images": len(files),
        }, f, indent=2)
    print(f"\n  Saved to: {out_path}")

    # Summary
    best = max(all_results, key=lambda r: r["PQ_things"])
    print(f"\n  Best PQ_things: {best['PQ_things']:.2f} "
          f"(config: {best['config']})")


if __name__ == "__main__":
    main()
