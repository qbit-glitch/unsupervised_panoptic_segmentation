#!/usr/bin/env python3
"""Unified ablation: novel instance decomposition methods.

Evaluates PQ across methods and hyperparameter grids on Cityscapes val.
All methods share the same interface via `instance_methods/`.

Usage:
    # Single method, default params
    python mbps_pytorch/ablate_instance_methods.py \
        --cityscapes_root /path/to/cityscapes \
        --method morse

    # Sweep all configs for a method
    python mbps_pytorch/ablate_instance_methods.py \
        --cityscapes_root /path/to/cityscapes \
        --method morse --sweep

    # All methods, default params only
    python mbps_pytorch/ablate_instance_methods.py \
        --cityscapes_root /path/to/cityscapes \
        --method all

    # All methods, full sweep
    python mbps_pytorch/ablate_instance_methods.py \
        --cityscapes_root /path/to/cityscapes \
        --method all --sweep

    # Quick test on 10 images
    python mbps_pytorch/ablate_instance_methods.py \
        --cityscapes_root /path/to/cityscapes \
        --method morse --max_images 10
"""

import argparse
import json
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ─── Cityscapes Constants ───

CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
    22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
    32: 17, 33: 18,
}
NUM_CLASSES = 19
STUFF_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
THING_IDS = {11, 12, 13, 14, 15, 16, 17, 18}
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]
WORK_H, WORK_W = 512, 1024


# ─── Hyperparameter Grids ───

SWEEP_GRIDS = {
    "sobel_cc": {
        "grad_threshold": [0.10, 0.15, 0.20, 0.25, 0.30],
        "min_area": [500, 1000, 1500],
        "dilation_iters": [3],
        "depth_blur_sigma": [1.0],
    },
    "morse": {
        "min_basin_depth": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
        "merge_threshold": [0.0, 0.5, 0.7, 0.80, 0.85, 0.90, 0.95],
        "min_area": [1000],
        "dilation_iters": [3],
        "depth_blur_sigma": [1.0],
    },
    "tda": {
        "tau_persist": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
        "filtration_mode": ["depth_direct", "gradient_mag"],
        "min_area": [500, 1000],
        "dilation_iters": [3],
        "depth_blur_sigma": [1.0],
    },
    "ot": {
        "K_proto": [5, 10, 15, 20],
        "epsilon": [0.01, 0.05, 0.1],
        "depth_scale": [1.0, 5.0, 10.0],
        "min_area": [500, 1000],
        "dilation_iters": [3],
        "depth_blur_sigma": [1.0],
    },
    "mumford_shah": {
        "alpha": [0.1, 1.0, 10.0],
        "beta": [0.01, 0.1, 1.0],
        "n_clusters": [10, 20],
        "work_resolution_h": [64],
        "work_resolution_w": [128],
        "min_area": [500, 1000],
        "dilation_iters": [3],
        "depth_blur_sigma": [1.0],
    },
    "contrastive": {
        "hdbscan_min_cluster": [3, 5, 8, 12],
        "hdbscan_min_samples": [2, 3, 5],
        "min_area": [500, 1000],
        "dilation_iters": [3],
        "depth_blur_sigma": [1.0],
    },
    "picl": {
        "hdbscan_min_cluster": [3, 5, 8],
        "hdbscan_min_samples": [2, 3, 5],
        "min_area": [500, 1000],
    },
}

# Default configs (one config per method for quick single runs)
DEFAULT_CONFIGS = {
    "sobel_cc": {"grad_threshold": 0.03, "min_area": 1000,
                 "dilation_iters": 3, "depth_blur_sigma": 1.0},
    "morse": {"min_basin_depth": 0.03, "merge_threshold": 0.80,
              "min_area": 1000, "dilation_iters": 3, "depth_blur_sigma": 1.0},
    "tda": {"tau_persist": 0.05, "filtration_mode": "depth_direct",
            "min_area": 1000, "dilation_iters": 3, "depth_blur_sigma": 1.0},
    "ot": {"K_proto": 15, "epsilon": 0.1, "depth_scale": 10.0,
           "min_area": 1000, "dilation_iters": 3, "depth_blur_sigma": 1.0},
    "mumford_shah": {"alpha": 1.0, "beta": 0.1, "n_clusters": 20,
                     "work_resolution_h": 64, "work_resolution_w": 128,
                     "min_area": 1000, "dilation_iters": 3, "depth_blur_sigma": 1.0},
    "contrastive": {"hdbscan_min_cluster": 5, "hdbscan_min_samples": 3,
                    "min_area": 1000, "dilation_iters": 3, "depth_blur_sigma": 1.0},
    "feature_edge_cc": {"feat_grad_threshold": 0.15, "depth_grad_threshold": 0.03,
                        "fusion_mode": "union", "pca_dim": 64,
                        "min_area": 1000, "dilation_iters": 3, "depth_blur_sigma": 1.0},
    "joint_ncut": {"alpha": 1.0, "beta": 1.0, "ncut_threshold": 0.05,
                   "work_resolution_h": 32, "work_resolution_w": 64,
                   "min_area": 1000, "dilation_iters": 3, "depth_blur_sigma": 1.0},
    "learned_edge_cc": {"edge_threshold": 0.5, "pca_dim": 64,
                        "min_area": 1000, "dilation_iters": 3, "depth_blur_sigma": 1.0},
    "plane_decomp": {"patch_size": 16, "normal_angle_threshold": 15.0,
                     "residual_threshold": 0.02,
                     "min_area": 1000, "dilation_iters": 3, "depth_blur_sigma": 1.0},
    "adaptive_edge": {"depth_grad_threshold": 0.03, "feat_grad_threshold": 0.30,
                      "depth_conf_temperature": 0.05, "depth_conf_center": 0.03,
                      "fusion_mode": "soft", "pca_dim": 64,
                      "min_area": 1000, "dilation_iters": 3, "depth_blur_sigma": 1.0},
    "depth_stratified": {"n_depth_bins": 5, "sim_threshold": 0.65,
                         "min_area": 1000, "dilation_iters": 3, "depth_blur_sigma": 1.0},
    "picl": {"hdbscan_min_cluster": 5, "hdbscan_min_samples": 3,
             "min_area": 1000, "dilation_iters": 3},
}


def expand_grid(grid_dict):
    """Expand a parameter grid dict into a list of config dicts."""
    keys = list(grid_dict.keys())
    values = list(grid_dict.values())
    configs = []
    for combo in product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


# ─── Helpers ───

def remap_gt_to_trainids(gt_raw):
    out = np.full_like(gt_raw, 255, dtype=np.uint8)
    for cs_id, train_id in CS_ID_TO_TRAIN.items():
        out[gt_raw == cs_id] = train_id
    return out


def resize_nearest(arr, hw):
    h, w = hw
    return np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST))


# ─── Panoptic Evaluation (from sweep_k50_spidepth.py) ───

def evaluate_panoptic_single(pred_sem, pred_instances, gt_sem, gt_inst_map,
                              eval_hw):
    """Evaluate one image. Returns per-class (tp, fp, fn, iou) arrays."""
    H, W = eval_hw

    # Build predicted panoptic map
    pred_pan = np.zeros((H, W), dtype=np.int32)
    pred_segments = {}
    next_id = 1

    # Stuff from semantic
    for cls in STUFF_IDS:
        mask = pred_sem == cls
        if mask.sum() < 64:
            continue
        pred_pan[mask] = next_id
        pred_segments[next_id] = cls
        next_id += 1

    # Things from instance masks
    for mask, cls, score in pred_instances:
        if cls not in THING_IDS:
            continue
        new_pixels = mask & (pred_pan == 0)
        if new_pixels.sum() < 10:
            continue
        pred_pan[new_pixels] = next_id
        pred_segments[next_id] = cls
        next_id += 1

    # Build GT panoptic map
    gt_pan = np.zeros((H, W), dtype=np.int32)
    gt_segments = {}
    gt_next_id = 1

    for cls in STUFF_IDS:
        mask = gt_sem == cls
        if mask.sum() < 64:
            continue
        gt_pan[mask] = gt_next_id
        gt_segments[gt_next_id] = cls
        gt_next_id += 1

    for uid in np.unique(gt_inst_map):
        if uid < 1000:
            continue
        raw_cls = uid // 1000
        if raw_cls not in CS_ID_TO_TRAIN:
            continue
        train_id = CS_ID_TO_TRAIN[raw_cls]
        if train_id not in THING_IDS:
            continue
        mask = gt_inst_map == uid
        if mask.sum() < 10:
            continue
        gt_pan[mask] = gt_next_id
        gt_segments[gt_next_id] = train_id
        gt_next_id += 1

    # Match segments per category
    tp = np.zeros(NUM_CLASSES)
    fp = np.zeros(NUM_CLASSES)
    fn = np.zeros(NUM_CLASSES)
    iou_sum = np.zeros(NUM_CLASSES)
    matched_pred = set()

    gt_by_cat = defaultdict(list)
    for seg_id, cat in gt_segments.items():
        gt_by_cat[cat].append(seg_id)
    pred_by_cat = defaultdict(list)
    for seg_id, cat in pred_segments.items():
        pred_by_cat[cat].append(seg_id)

    for cat in range(NUM_CLASSES):
        gt_segs = gt_by_cat.get(cat, [])
        pred_segs = pred_by_cat.get(cat, [])
        if not gt_segs and not pred_segs:
            continue

        for gt_id in gt_segs:
            gt_mask = gt_pan == gt_id
            best_iou = 0.0
            best_pred = None
            for pred_id in pred_segs:
                if pred_id in matched_pred:
                    continue
                pred_mask = pred_pan == pred_id
                inter = np.sum(gt_mask & pred_mask)
                union = np.sum(gt_mask | pred_mask)
                if union == 0:
                    continue
                iou_val = inter / union
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_pred = pred_id
            if best_iou > 0.5 and best_pred is not None:
                tp[cat] += 1
                iou_sum[cat] += best_iou
                matched_pred.add(best_pred)
            else:
                fn[cat] += 1

        for pred_id in pred_segs:
            if pred_id not in matched_pred:
                fp[cat] += 1

    return tp, fp, fn, iou_sum, len(pred_instances)


def compute_pq_from_accumulators(tp, fp, fn, iou_sum):
    """Compute PQ/SQ/RQ from accumulated tp/fp/fn/iou arrays."""
    all_pq, stuff_pq, thing_pq = [], [], []
    per_class = {}

    for c in range(NUM_CLASSES):
        t, f_p, f_n = tp[c], fp[c], fn[c]
        if t + f_p + f_n > 0:
            sq = iou_sum[c] / (t + 1e-8)
            rq = t / (t + 0.5 * f_p + 0.5 * f_n)
            pq = sq * rq
        else:
            sq = rq = pq = 0.0

        per_class[CLASS_NAMES[c]] = {
            "PQ": round(pq * 100, 2), "SQ": round(sq * 100, 2),
            "RQ": round(rq * 100, 2), "TP": int(t), "FP": int(f_p), "FN": int(f_n),
        }
        if t + f_p + f_n > 0:
            all_pq.append(pq)
            if c in STUFF_IDS:
                stuff_pq.append(pq)
            else:
                thing_pq.append(pq)

    return {
        "PQ": round(float(np.mean(all_pq)) * 100, 2) if all_pq else 0.0,
        "PQ_stuff": round(float(np.mean(stuff_pq)) * 100, 2) if stuff_pq else 0.0,
        "PQ_things": round(float(np.mean(thing_pq)) * 100, 2) if thing_pq else 0.0,
        "SQ": round(float(np.sum(iou_sum) / (np.sum(tp) + 1e-8)) * 100, 2),
        "RQ": round(float(np.sum(tp) / (np.sum(tp) + 0.5 * np.sum(fp) + 0.5 * np.sum(fn) + 1e-8)) * 100, 2),
        "per_class": per_class,
    }


# ─── Data Loading ───

def discover_files(cityscapes_root, split, semantic_subdir, depth_subdir,
                   feature_subdir=None):
    """Find matching file paths for evaluation.

    Returns list of (sem_path, depth_path, gt_label_path, gt_inst_path, feat_path_or_None).
    """
    root = Path(cityscapes_root)
    sem_dir = root / semantic_subdir / split
    depth_dir = root / depth_subdir / split
    gt_dir = root / "gtFine" / split
    feat_dir = root / feature_subdir / split if feature_subdir else None

    gt_label_files = sorted(gt_dir.rglob("*_gtFine_labelIds.png"))

    files = []
    for gt_label_path in gt_label_files:
        rel = gt_label_path.relative_to(gt_dir)
        base = str(rel).replace("_gtFine_labelIds.png", "")
        city = base.split("/")[0]
        stem = base.split("/")[-1]

        gt_inst_path = gt_dir / (base + "_gtFine_instanceIds.png")
        if not gt_inst_path.exists():
            continue

        # Semantic: try stem.png then stem_leftImg8bit.png
        sem_path = sem_dir / city / f"{stem}.png"
        if not sem_path.exists():
            sem_path = sem_dir / city / f"{stem}_leftImg8bit.png"
        if not sem_path.exists():
            continue

        # Depth: try stem.npy then stem_leftImg8bit.npy
        depth_path = depth_dir / city / f"{stem}.npy"
        if not depth_path.exists():
            depth_path = depth_dir / city / f"{stem}_leftImg8bit.npy"
        if not depth_path.exists():
            continue

        # Features (optional)
        feat_path = None
        if feat_dir is not None:
            feat_path = feat_dir / city / f"{stem}.npy"
            if not feat_path.exists():
                feat_path = feat_dir / city / f"{stem}_leftImg8bit.npy"
            if not feat_path.exists():
                feat_path = None

        files.append((sem_path, depth_path, gt_label_path, gt_inst_path, feat_path))

    return files


# ─── Method Dispatch ───

def build_method_kwargs(method_name, config, features=None, model=None):
    """Build kwargs for a method call from config dict + features."""
    kwargs = dict(config)
    kwargs["thing_ids"] = THING_IDS

    # Handle work_resolution tuple for mumford_shah and joint_ncut
    if method_name in ("mumford_shah", "joint_ncut"):
        h = kwargs.pop("work_resolution_h", 128)
        w = kwargs.pop("work_resolution_w", 256)
        kwargs["work_resolution"] = (h, w)

    # Add features for methods that need them
    if method_name in ("morse", "ot", "mumford_shah", "contrastive",
                       "feature_edge_cc", "joint_ncut", "learned_edge_cc",
                       "adaptive_edge", "depth_stratified", "picl"):
        kwargs["features"] = features
    elif method_name == "tda":
        kwargs["features"] = None  # TDA doesn't use features

    # Add model for methods that need a trained model
    if method_name == "learned_edge_cc" and model is not None:
        kwargs["model"] = model
    elif method_name == "picl" and model is not None:
        kwargs["model"] = model

    return kwargs


NEEDS_FEATURES = {"morse", "ot", "mumford_shah", "contrastive",
                  "feature_edge_cc", "joint_ncut", "learned_edge_cc",
                  "adaptive_edge", "depth_stratified", "picl"}


# ─── Run a Single Configuration ───

def run_single_config(method_fn, method_name, files, k_to_trainid, eval_hw,
                      config, label="", model=None):
    """Evaluate one method+config on all images. Returns metrics dict."""
    from instance_methods.utils import load_features

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
        # Load and remap semantics
        pred_k = np.array(Image.open(sem_path))
        pred_sem = k_to_trainid[pred_k]
        if pred_sem.shape != (H, W):
            pred_sem = resize_nearest(pred_sem, eval_hw)

        # Load depth
        depth = np.load(str(depth_path))
        if depth.shape != (H, W):
            depth = np.array(
                Image.fromarray(depth).resize((W, H), Image.BILINEAR)
            )

        # Load features if needed
        features = None
        if method_name in NEEDS_FEATURES and feat_path is not None:
            features = load_features(str(feat_path))

        # Build method kwargs
        kwargs = build_method_kwargs(method_name, config, features=features,
                                     model=model)

        # Run instance method
        t0 = time.time()
        try:
            instances = method_fn(pred_sem, depth, **kwargs)
        except Exception as e:
            if errors < 3:
                tqdm.write(f"  [WARN] {label}: {type(e).__name__}: {e}")
            errors += 1
            instances = []
        total_time += time.time() - t0

        # Load GT
        gt_raw = np.array(Image.open(gt_label_path))
        gt_sem = remap_gt_to_trainids(gt_raw)
        if gt_sem.shape != (H, W):
            gt_sem = resize_nearest(gt_sem, eval_hw)

        gt_inst_map = np.array(Image.open(gt_inst_path), dtype=np.int32)
        if gt_inst_map.shape != (H, W):
            gt_inst_map = np.array(
                Image.fromarray(gt_inst_map).resize((W, H), Image.NEAREST)
            )

        # Evaluate
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


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(
        description="Unified ablation: novel instance decomposition methods"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--method", type=str, required=True,
                        choices=["sobel_cc", "morse", "tda", "ot",
                                 "mumford_shah", "contrastive",
                                 "learned_merge", "feature_edge_cc",
                                 "joint_ncut", "learned_edge_cc",
                                 "plane_decomp", "adaptive_edge",
                                 "depth_stratified", "picl", "all"],
                        help="Method to evaluate (or 'all')")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full hyperparameter sweep")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val"])
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
    parser.add_argument("--learned_edge_dir", type=str,
                        default="checkpoints/learned_edge",
                        help="Directory with trained learned edge model")
    parser.add_argument("--picl_checkpoint", type=str,
                        default="checkpoints/picl/round1/best.pth",
                        help="Path to trained PICL checkpoint (.pth)")
    args = parser.parse_args()

    cs_root = Path(args.cityscapes_root)
    eval_hw = tuple(args.eval_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import method registry
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from instance_methods import METHODS

    # Determine which methods to run
    if args.method == "all":
        methods_to_run = list(METHODS.keys())
    else:
        methods_to_run = [args.method]

    # Load cluster → trainID mapping
    centroids_path = args.centroids_path or str(
        cs_root / args.semantic_subdir / "kmeans_centroids.npz"
    )
    data = np.load(centroids_path)
    cluster_to_class = data["cluster_to_class"]
    k = len(cluster_to_class)

    k_to_trainid = np.full(256, 255, dtype=np.uint8)
    for cid, tid in enumerate(cluster_to_class):
        k_to_trainid[cid] = int(tid)

    # Discover files
    need_features = any(m in NEEDS_FEATURES for m in methods_to_run)
    feat_subdir = args.feature_subdir if need_features else None
    files = discover_files(cs_root, args.split, args.semantic_subdir,
                           args.depth_subdir, feat_subdir)

    if args.max_images:
        files = files[:args.max_images]

    print(f"\n{'='*70}")
    print(f"INSTANCE METHOD ABLATION STUDY")
    print(f"{'='*70}")
    print(f"  Methods: {', '.join(methods_to_run)}")
    print(f"  Sweep: {args.sweep}")
    print(f"  Split: {args.split}, Images: {len(files)}")
    print(f"  Semantics: {args.semantic_subdir} (k={k})")
    print(f"  Depth: {args.depth_subdir}")
    print(f"  Features: {args.feature_subdir}")
    print(f"  Eval resolution: {eval_hw[0]}x{eval_hw[1]}")
    print(f"  Output: {output_dir}")

    if need_features:
        n_with_feat = sum(1 for f in files if f[4] is not None)
        n_without = len(files) - n_with_feat
        print(f"  Feature coverage: {n_with_feat}/{len(files)} images"
              f" ({n_without} missing)")

    # Load learned edge model if needed
    learned_edge_model = None
    if "learned_edge_cc" in methods_to_run:
        try:
            import torch
            from train_learned_edge import EdgePredictor

            ckpt_dir = Path(args.learned_edge_dir)
            ckpt_path = ckpt_dir / "best.pth"
            if ckpt_path.exists():
                ckpt = torch.load(str(ckpt_path), map_location="cpu",
                                  weights_only=True)
                in_ch = ckpt["config"]["in_channels"]
                learned_edge_model = EdgePredictor(in_channels=in_ch)
                learned_edge_model.load_state_dict(ckpt["model_state_dict"])
                if torch.backends.mps.is_available():
                    learned_edge_model = learned_edge_model.to("mps")
                elif torch.cuda.is_available():
                    learned_edge_model = learned_edge_model.to("cuda")
                learned_edge_model.eval()
                print(f"  Loaded learned edge model from {ckpt_path}")
            else:
                print(f"  [WARN] Learned edge checkpoint not found: {ckpt_path}")
                print(f"         Run train_learned_edge.py first.")
                methods_to_run = [m for m in methods_to_run
                                  if m != "learned_edge_cc"]
        except Exception as e:
            print(f"  [WARN] Failed to load learned edge model: {e}")
            methods_to_run = [m for m in methods_to_run
                              if m != "learned_edge_cc"]

    # Load PICL model if needed
    picl_model = None
    picl_model_cfg = {}
    if "picl" in methods_to_run:
        try:
            import torch
            from train_picl import PICLProjectionHead

            ckpt_path = Path(args.picl_checkpoint)
            if ckpt_path.exists():
                if torch.backends.mps.is_available():
                    _device = torch.device("mps")
                elif torch.cuda.is_available():
                    _device = torch.device("cuda")
                else:
                    _device = torch.device("cpu")
                ckpt = torch.load(str(ckpt_path), map_location=_device,
                                  weights_only=False)
                cfg = ckpt["config"]
                picl_model = PICLProjectionHead(
                    input_dim=cfg.get("input_dim", 771),
                    hidden_dim=cfg.get("hidden_dim", 512),
                    embed_dim=cfg.get("embed_dim", 128),
                )
                picl_model.load_state_dict(ckpt["model_state_dict"])
                picl_model.to(_device)
                picl_model.eval()
                picl_model_cfg = cfg
                print(f"  Loaded PICL model from {ckpt_path} "
                      f"(PQ_things={ckpt.get('PQ_things', 'N/A')})")
            else:
                print(f"  [WARN] PICL checkpoint not found: {ckpt_path}")
                print(f"         Run train_picl.py first (or bash scripts/run_picl_rounds.sh 1)")
                methods_to_run = [m for m in methods_to_run if m != "picl"]
        except Exception as e:
            print(f"  [WARN] Failed to load PICL model: {e}")
            methods_to_run = [m for m in methods_to_run if m != "picl"]

    # Run methods
    all_results = []
    t0_total = time.time()

    for method_name in methods_to_run:
        method_fn = METHODS[method_name]

        if args.sweep:
            grid = SWEEP_GRIDS.get(method_name, {})
            configs = expand_grid(grid) if grid else [DEFAULT_CONFIGS[method_name]]
        else:
            configs = [DEFAULT_CONFIGS[method_name]]

        print(f"\n{'─'*70}")
        print(f"  Method: {method_name} ({len(configs)} config{'s' if len(configs) > 1 else ''})")
        print(f"{'─'*70}")

        best_pq = -1
        best_config = None

        # Pass model for methods that need a trained model
        if method_name == "learned_edge_cc":
            model_for_method = learned_edge_model
        elif method_name == "picl":
            model_for_method = picl_model
        else:
            model_for_method = None

        for i, config in enumerate(configs):
            # Build compact label
            label_parts = [f"{k_}={v}" for k_, v in config.items()
                          if k_ not in ("thing_ids", "dilation_iters",
                                        "depth_blur_sigma")]
            label = f"{method_name} [{', '.join(label_parts)}]"
            if len(configs) > 1:
                label = f"[{i+1}/{len(configs)}] {label}"

            print(f"\n  {label}")
            t0 = time.time()
            metrics = run_single_config(
                method_fn, method_name, files, k_to_trainid, eval_hw,
                config, label=label, model=model_for_method
            )
            dt = time.time() - t0

            print(f"    PQ={metrics['PQ']:5.2f}  PQ_st={metrics['PQ_stuff']:5.2f}  "
                  f"PQ_th={metrics['PQ_things']:5.2f}  "
                  f"inst/img={metrics['avg_instances']:4.1f}  "
                  f"({metrics['seconds_per_image']:.2f}s/img, {dt:.0f}s total)"
                  + (f"  [{metrics['errors']} errors]" if metrics['errors'] else ""))

            result = {
                "method": method_name,
                "config": config,
                "PQ": metrics["PQ"],
                "PQ_stuff": metrics["PQ_stuff"],
                "PQ_things": metrics["PQ_things"],
                "SQ": metrics["SQ"],
                "RQ": metrics["RQ"],
                "avg_instances": metrics["avg_instances"],
                "seconds_per_image": metrics["seconds_per_image"],
                "n_images": metrics["n_images"],
                "errors": metrics["errors"],
                "per_class": metrics["per_class"],
            }
            all_results.append(result)

            if metrics["PQ"] > best_pq:
                best_pq = metrics["PQ"]
                best_config = config

        if best_config is not None:
            print(f"\n  Best {method_name}: PQ={best_pq:.2f}")
            for k_, v in best_config.items():
                if k_ not in ("dilation_iters", "depth_blur_sigma"):
                    print(f"    {k_} = {v}")

    elapsed_total = time.time() - t0_total

    # ─── Summary Table ───
    print(f"\n{'='*70}")
    print(f"SUMMARY: Instance Method Ablation")
    print(f"{'='*70}")

    # Best per method
    best_per_method = {}
    for r in all_results:
        m = r["method"]
        if m not in best_per_method or r["PQ"] > best_per_method[m]["PQ"]:
            best_per_method[m] = r

    print(f"\n  {'Method':<15s} {'PQ':>6s} {'PQ_st':>6s} {'PQ_th':>7s} "
          f"{'inst/img':>8s} {'s/img':>6s}")
    print(f"  {'─'*15} {'─'*6} {'─'*6} {'─'*7} {'─'*8} {'─'*6}")

    for method_name in methods_to_run:
        if method_name in best_per_method:
            r = best_per_method[method_name]
            print(f"  {method_name:<15s} {r['PQ']:6.2f} {r['PQ_stuff']:6.2f} "
                  f"{r['PQ_things']:7.2f} {r['avg_instances']:8.1f} "
                  f"{r['seconds_per_image']:6.2f}")

    print(f"\n  DA3 Baseline: Sobel+CC τ=0.03, PQ_th=20.90")
    print(f"  Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")

    # Per-class breakdown for best overall method
    overall_best = max(all_results, key=lambda r: r["PQ"])
    print(f"\n  Best overall: {overall_best['method']} "
          f"(PQ={overall_best['PQ']:.2f})")
    print(f"\n  Per-class PQ (best config):")
    for name, v in sorted(overall_best["per_class"].items(),
                           key=lambda x: x[1]["PQ"], reverse=True):
        kind = "S" if CLASS_NAMES.index(name) in STUFF_IDS else "T"
        if v["TP"] + v["FP"] + v["FN"] > 0:
            print(f"    [{kind}] {name:15s}: PQ={v['PQ']:5.1f}  SQ={v['SQ']:5.1f}  "
                  f"RQ={v['RQ']:5.1f}  (TP={v['TP']} FP={v['FP']} FN={v['FN']})")

    # ─── Save Results ───
    suffix = "sweep" if args.sweep else "default"
    method_str = args.method
    output_path = output_dir / f"ablation_{method_str}_{suffix}_{args.split}.json"

    save_data = {
        "split": args.split,
        "k": k,
        "eval_resolution": list(eval_hw),
        "semantic_subdir": args.semantic_subdir,
        "depth_subdir": args.depth_subdir,
        "feature_subdir": args.feature_subdir,
        "n_images": len(files),
        "total_seconds": round(elapsed_total, 1),
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
