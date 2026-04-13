#!/usr/bin/env python3
"""Mumford-Shah Phase B: validate top configs from Phase A (100 imgs) on full 500 images.

Usage:
    python mbps_pytorch/mumford_shah_phase_b.py \
        --cityscapes_root /path/to/cityscapes
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Reuse everything from the ablation script
sys.path.insert(0, str(Path(__file__).parent))
from instance_methods import METHODS
sys.path.insert(0, str(Path(__file__).parent.parent))
from mbps_pytorch.ablate_instance_methods import (
    DEFAULT_CONFIGS, discover_files, run_single_config,
    CS_ID_TO_TRAIN,
)

# Top 5 configs from Phase A (100 images), ranked by PQ_things
PHASE_B_CONFIGS = [
    {"alpha": 1.0,  "beta": 1.0, "n_clusters": 10, "work_resolution_h": 64,
     "work_resolution_w": 128, "min_area": 1000, "dilation_iters": 3, "depth_blur_sigma": 1.0},
    {"alpha": 0.1,  "beta": 1.0, "n_clusters": 10, "work_resolution_h": 64,
     "work_resolution_w": 128, "min_area": 1000, "dilation_iters": 3, "depth_blur_sigma": 1.0},
    {"alpha": 1.0,  "beta": 1.0, "n_clusters": 20, "work_resolution_h": 64,
     "work_resolution_w": 128, "min_area": 1000, "dilation_iters": 3, "depth_blur_sigma": 1.0},
    {"alpha": 0.1,  "beta": 1.0, "n_clusters": 20, "work_resolution_h": 64,
     "work_resolution_w": 128, "min_area": 1000, "dilation_iters": 3, "depth_blur_sigma": 1.0},
    {"alpha": 10.0, "beta": 1.0, "n_clusters": 10, "work_resolution_h": 64,
     "work_resolution_w": 128, "min_area": 1000, "dilation_iters": 3, "depth_blur_sigma": 1.0},
]


def main():
    parser = argparse.ArgumentParser(description="Mumford-Shah Phase B validation")
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--eval_size", type=int, nargs=2, default=[512, 1024])
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_raw_k80")
    parser.add_argument("--depth_subdir", type=str, default="depth_spidepth")
    parser.add_argument("--feature_subdir", type=str,
                        default="dinov2_features")
    parser.add_argument("--centroids_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str,
                        default="results/ablation_instance_methods")
    args = parser.parse_args()

    eval_hw = tuple(args.eval_size)
    root = Path(args.cityscapes_root)

    # Load centroids
    import numpy as np
    if args.centroids_path:
        cent_path = args.centroids_path
    else:
        cent_path = root / args.semantic_subdir / "kmeans_centroids.npz"
    cent_data = np.load(cent_path)
    centroids = cent_data["centroids"]
    cluster_to_class = cent_data["cluster_to_class"]
    # Build as numpy array for direct indexing: k_to_trainid[pred_k_array]
    n_clusters = len(cluster_to_class)
    k_to_trainid = np.full(256, 255, dtype=np.uint8)
    for k in range(n_clusters):
        k_to_trainid[k] = int(cluster_to_class[k])

    # Discover files
    files = discover_files(root, args.split, args.semantic_subdir,
                           args.depth_subdir, args.feature_subdir)
    print(f"Phase B: {len(files)} images, {len(PHASE_B_CONFIGS)} configs")

    method_fn = METHODS["mumford_shah"]
    all_results = []

    for i, config in enumerate(PHASE_B_CONFIGS):
        label = (f"[{i+1}/{len(PHASE_B_CONFIGS)}] mumford_shah "
                 f"[a={config['alpha']}, b={config['beta']}, "
                 f"k={config['n_clusters']}, A_min={config['min_area']}]")
        print(f"\n  {label}")
        t0 = time.time()
        metrics = run_single_config(
            method_fn, "mumford_shah", files, k_to_trainid, eval_hw,
            config, label=label
        )
        dt = time.time() - t0
        print(f"    PQ={metrics['PQ']:5.2f}  PQ_st={metrics['PQ_stuff']:5.2f}  "
              f"PQ_th={metrics['PQ_things']:5.2f}  "
              f"inst/img={metrics['avg_instances']:4.1f}  "
              f"({metrics['seconds_per_image']:.2f}s/img, {dt:.0f}s total)")

        all_results.append({
            "method": "mumford_shah",
            "config": config,
            **metrics,
        })

    # Save results
    out_path = Path(args.output_dir) / "ablation_mumford_shah_phase_b_val.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"method": "mumford_shah_phase_b", "results": all_results,
                   "n_images": len(files)}, f, indent=2)
    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    main()
