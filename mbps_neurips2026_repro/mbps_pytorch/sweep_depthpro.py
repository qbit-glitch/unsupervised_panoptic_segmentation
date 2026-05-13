#!/usr/bin/env python3
"""Parameter sweep: DepthPro depth-guided instance splitting on k=80 semantics.

Two-phase ablation of DepthPro for instance pseudo-label generation:
  Phase 1: Coarse (grad_threshold × min_area) grid — 24 configs
  Phase 2: Focused (depth_blur_sigma × dilation_iters) around Phase 1 winner

Uses the same evaluation protocol as reports/depth_model_ablation_study.md
for fair comparison with DA3, DA2-Large, and SPIdepth baselines.

Usage:
    # Full sweep (Phase 1 + Phase 2)
    python mbps_pytorch/sweep_depthpro.py \
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes

    # Phase 1 only
    python mbps_pytorch/sweep_depthpro.py \
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
        --phase 1

    # Phase 2 with known best from Phase 1
    python mbps_pytorch/sweep_depthpro.py \
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
        --phase 2 --phase1_best 0.02 1000
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel
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

# ─── Known baselines (from reports/depth_model_ablation_study.md) ───

BASELINES = {
    "CC-only (no depth)":       {"PQ": 24.80, "PQ_stuff": 32.08, "PQ_things": 14.93,
                                 "tau": "—", "A_min": "—"},
    "SPIdepth (τ=0.20)":        {"PQ": 26.74, "PQ_stuff": 32.08, "PQ_things": 19.41,
                                 "tau": 0.20, "A_min": 1000},
    "DA2-Large (τ=0.03)":       {"PQ": 27.10, "PQ_stuff": 32.08, "PQ_things": 20.20,
                                 "tau": 0.03, "A_min": 1000},
    "DA3 (τ=0.03)":             {"PQ": 27.37, "PQ_stuff": 32.08, "PQ_things": 20.90,
                                 "tau": 0.03, "A_min": 1000},
}


def remap_gt_to_trainids(gt_raw):
    """Remap Cityscapes raw label IDs to trainIDs."""
    out = np.full_like(gt_raw, 255, dtype=np.uint8)
    for cs_id, train_id in CS_ID_TO_TRAIN.items():
        out[gt_raw == cs_id] = train_id
    return out


def resize_nearest(arr, hw):
    """Resize array with nearest-neighbor interpolation."""
    h, w = hw
    return np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST))


# ─── Depth-Guided Instance Generation ───

def depth_guided_instances(semantic, depth, thing_ids=THING_IDS,
                           grad_threshold=0.05, min_area=100,
                           dilation_iters=3, depth_blur_sigma=1.0):
    """Split thing regions using depth gradient edges.

    Args:
        semantic: (H, W) uint8 trainID map.
        depth: (H, W) float32 depth map [0, 1].
        thing_ids: Set of trainIDs for thing classes.
        grad_threshold: Sobel gradient magnitude threshold for edges.
        min_area: Minimum pixel area for a valid instance.
        dilation_iters: Iterations for boundary pixel reclamation.
        depth_blur_sigma: Gaussian blur sigma on depth before Sobel.

    Returns:
        List of (mask, class_id, score) tuples sorted by area descending.
    """
    if depth_blur_sigma > 0:
        depth_smooth = gaussian_filter(depth.astype(np.float64), sigma=depth_blur_sigma)
    else:
        depth_smooth = depth.astype(np.float64)

    gx = sobel(depth_smooth, axis=1)
    gy = sobel(depth_smooth, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    depth_edges = grad_mag > grad_threshold

    assigned = np.zeros(semantic.shape, dtype=bool)
    instances = []

    for cls in sorted(thing_ids):
        cls_mask = semantic == cls
        if cls_mask.sum() < min_area:
            continue
        split_mask = cls_mask & ~depth_edges
        labeled, n_cc = ndimage.label(split_mask)

        cc_list = []
        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = int(cc_mask.sum())
            if area >= min_area:
                cc_list.append((cc_id, cc_mask, area))
        cc_list.sort(key=lambda x: -x[2])

        for cc_id, cc_mask, area in cc_list:
            if dilation_iters > 0:
                dilated = ndimage.binary_dilation(cc_mask, iterations=dilation_iters)
                reclaimed = dilated & cls_mask & ~assigned
                final_mask = cc_mask | reclaimed
            else:
                final_mask = cc_mask
            final_area = float(final_mask.sum())
            if final_area < min_area:
                continue
            assigned |= final_mask
            instances.append((final_mask, cls, final_area))

    instances.sort(key=lambda x: -x[2])
    if instances:
        max_area = instances[0][2]
        instances = [(m, c, s / max_area) for m, c, s in instances]
    return instances


def cc_only_instances(semantic, thing_ids=THING_IDS, min_area=10):
    """Connected component instances without depth splitting."""
    instances = []
    for cls in sorted(thing_ids):
        cls_mask = semantic == cls
        if cls_mask.sum() < min_area:
            continue
        labeled, n_cc = ndimage.label(cls_mask)
        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = float(cc_mask.sum())
            if area >= min_area:
                instances.append((cc_mask, cls, area))

    instances.sort(key=lambda x: -x[2])
    if instances:
        max_area = instances[0][2]
        instances = [(m, c, s / max_area) for m, c, s in instances]
    return instances


# ─── Panoptic Evaluation ───

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
        "RQ": round(float(np.sum(tp) / (np.sum(tp) + 0.5 * np.sum(fp)
                                         + 0.5 * np.sum(fn) + 1e-8)) * 100, 2),
        "per_class": per_class,
    }


# ─── Data Loading ───

def discover_files(cityscapes_root, split, semantic_subdir, depth_subdir):
    """Find matching (semantic, depth, gt_label, gt_instance) file paths."""
    root = Path(cityscapes_root)
    sem_dir = root / semantic_subdir / split
    depth_dir = root / depth_subdir / split
    gt_dir = root / "gtFine" / split

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

        sem_path = sem_dir / city / f"{stem}.png"
        if not sem_path.exists():
            sem_path = sem_dir / city / f"{stem}_leftImg8bit.png"
        if not sem_path.exists():
            continue

        depth_path = depth_dir / city / f"{stem}.npy"
        if not depth_path.exists():
            depth_path = depth_dir / city / f"{stem}_leftImg8bit.npy"
        if not depth_path.exists():
            continue

        files.append((sem_path, depth_path, gt_label_path, gt_inst_path))

    return files


# ─── Sweep Runner ───

def run_single_config(files, cluster_to_trainid, eval_hw, grad_threshold=None,
                       min_area=100, dilation_iters=3, depth_blur_sigma=1.0,
                       label=""):
    """Run evaluation for a single parameter configuration.

    Args:
        files: List of (sem_path, depth_path, gt_label_path, gt_inst_path).
        cluster_to_trainid: LUT mapping cluster IDs to trainIDs (256 entries).
        eval_hw: (H, W) evaluation resolution.
        grad_threshold: Sobel threshold. None = CC-only mode.
        min_area: Minimum instance area in pixels.
        dilation_iters: Boundary reclamation iterations.
        depth_blur_sigma: Gaussian blur on depth before Sobel.
        label: Progress bar label.

    Returns:
        Dict with PQ, PQ_stuff, PQ_things, per_class, avg_instances, etc.
    """
    H, W = eval_hw
    tp_acc = np.zeros(NUM_CLASSES)
    fp_acc = np.zeros(NUM_CLASSES)
    fn_acc = np.zeros(NUM_CLASSES)
    iou_acc = np.zeros(NUM_CLASSES)
    # Semantic IoU accumulators (per-pixel, not per-segment)
    sem_inter = np.zeros(NUM_CLASSES)
    sem_union = np.zeros(NUM_CLASSES)
    total_instances = 0
    n_images = 0

    for sem_path, depth_path, gt_label_path, gt_inst_path in tqdm(
        files, desc=label, leave=False
    ):
        # Load and remap overclustered semantics → trainIDs
        pred_raw = np.array(Image.open(sem_path))
        pred_sem = cluster_to_trainid[pred_raw]
        if pred_sem.shape != (H, W):
            pred_sem = resize_nearest(pred_sem, eval_hw)

        # Generate instances
        if grad_threshold is not None:
            depth = np.load(str(depth_path))
            if depth.shape != (H, W):
                depth = np.array(
                    Image.fromarray(depth).resize((W, H), Image.BILINEAR)
                )
            instances = depth_guided_instances(
                pred_sem, depth, THING_IDS,
                grad_threshold=grad_threshold, min_area=min_area,
                dilation_iters=dilation_iters,
                depth_blur_sigma=depth_blur_sigma,
            )
        else:
            instances = cc_only_instances(pred_sem, THING_IDS, min_area=10)

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

        # Accumulate per-pixel semantic IoU (pred_sem vs gt_sem)
        valid = gt_sem != 255
        for c in range(NUM_CLASSES):
            pred_c = (pred_sem == c) & valid
            gt_c = (gt_sem == c) & valid
            sem_inter[c] += np.sum(pred_c & gt_c)
            sem_union[c] += np.sum(pred_c | gt_c)

    metrics = compute_pq_from_accumulators(tp_acc, fp_acc, fn_acc, iou_acc)
    metrics["avg_instances"] = round(total_instances / max(n_images, 1), 1)
    metrics["n_images"] = n_images

    # Compute mIoU from semantic accumulators
    class_ious = []
    for c in range(NUM_CLASSES):
        if sem_union[c] > 0:
            class_ious.append(sem_inter[c] / sem_union[c])
    metrics["mIoU"] = round(float(np.mean(class_ious)) * 100, 2) if class_ious else 0.0
    return metrics


# ─── Sweep Grids ───

PHASE1_CONFIGS = [
    # (grad_threshold, min_area)
    # Ultra-low τ — DepthPro has 2× better boundary F1 than DA2
    (0.005, 500), (0.005, 1000), (0.005, 1500),
    (0.01, 500),  (0.01, 1000),  (0.01, 1500),
    (0.02, 500),  (0.02, 1000),  (0.02, 1500),
    # DA3-equivalent range
    (0.03, 500),  (0.03, 1000),  (0.03, 1500),
    (0.05, 500),  (0.05, 1000),
    (0.08, 1000), (0.10, 1000),
    # SPIdepth range (for comparison)
    (0.15, 1000), (0.20, 1000),
    # Near CC-only (monotonicity sanity check)
    (0.30, 1000), (0.50, 1000),
    (0.80, 1000), (1.00, 1000),
]


def get_phase2_configs(best_tau, best_amin):
    """Generate Phase 2 configs around Phase 1 winner.

    Sweeps depth_blur_sigma and dilation_iters independently.

    Args:
        best_tau: Best grad_threshold from Phase 1.
        best_amin: Best min_area from Phase 1.

    Returns:
        List of (grad_threshold, min_area, dilation_iters, depth_blur_sigma) tuples.
    """
    configs = []

    # Sweep depth_blur_sigma (dilation_iters=3 default)
    for sigma in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]:
        configs.append((best_tau, best_amin, 3, sigma))

    # Sweep dilation_iters (depth_blur_sigma=1.0 default)
    for dil in [0, 1, 2, 3, 4, 5, 7]:
        configs.append((best_tau, best_amin, dil, 1.0))

    # Remove duplicates (dil=3, sigma=1.0 appears in both)
    seen = set()
    unique = []
    for cfg in configs:
        key = (cfg[0], cfg[1], cfg[2], cfg[3])
        if key not in seen:
            seen.add(key)
            unique.append(cfg)

    return unique


def print_results_table(results, title, show_sigma_dil=False):
    """Print a markdown-style results table.

    Args:
        results: List of tuples. Phase 1: (tau, amin, metrics).
                 Phase 2: (tau, amin, dil, sigma, metrics).
        title: Table title string.
        show_sigma_dil: Whether to show sigma/dilation columns.
    """
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    if show_sigma_dil:
        header = ("| τ | A_min | σ_blur | dilation | inst/img | "
                  "PQ | PQ_stuff | PQ_things | SQ | RQ | mIoU |")
        sep = ("|------|-------|--------|---------|----------|"
               "-----|----------|-----------|------|------|------|")
    else:
        header = ("| τ | A_min | inst/img | "
                  "PQ | PQ_stuff | PQ_things | SQ | RQ | mIoU |")
        sep = ("|------|-------|----------|"
               "-----|----------|-----------|------|------|------|")

    print(f"\n{header}")
    print(sep)

    for entry in results:
        if show_sigma_dil:
            tau, amin, dil, sigma, m = entry
            tau_s = f"{tau:.3f}" if isinstance(tau, float) else str(tau)
            amin_s = str(amin) if isinstance(amin, int) else str(amin)
            print(f"| {tau_s:>6s} | {amin_s:>5s} | {sigma:6.2f} | {dil:>7d} | "
                  f"{m['avg_instances']:8.1f} | "
                  f"{m['PQ']:4.1f} | {m['PQ_stuff']:8.1f} | {m['PQ_things']:9.1f} | "
                  f"{m['SQ']:4.1f} | {m['RQ']:4.1f} | {m.get('mIoU', 0):4.1f} |")
        else:
            tau, amin, m = entry
            tau_s = f"{tau:.3f}" if isinstance(tau, float) else str(tau)
            amin_s = str(amin) if isinstance(amin, int) else str(amin)
            print(f"| {tau_s:>6s} | {amin_s:>5s} | {m['avg_instances']:8.1f} | "
                  f"{m['PQ']:4.1f} | {m['PQ_stuff']:8.1f} | {m['PQ_things']:9.1f} | "
                  f"{m['SQ']:4.1f} | {m['RQ']:4.1f} | {m.get('mIoU', 0):4.1f} |")


def print_comparison_table(best_m, best_tau, best_amin, k):
    """Print comparison with known baselines."""
    print(f"\n  ┌────────────────────────────────────────────────────────────────┐")
    print(f"  │  Depth Model Comparison (k={k} semantics, Cityscapes val)      │")
    print(f"  │  {'Model':<30s} {'τ':>5s}  {'A_min':>5s}  "
          f"{'PQ':>5s}  {'PQ^St':>5s}  {'PQ^Th':>5s}  │")
    print(f"  │{'─'*62}│")
    for name, b in BASELINES.items():
        tau_s = f"{b['tau']:.2f}" if isinstance(b['tau'], float) else str(b['tau'])
        print(f"  │  {name:<30s} {tau_s:>5s}  {str(b['A_min']):>5s}  "
              f"{b['PQ']:5.1f}  {b['PQ_stuff']:5.1f}  {b['PQ_things']:5.1f}  │")
    # DepthPro (this sweep)
    tau_s = f"{best_tau:.3f}" if isinstance(best_tau, float) else str(best_tau)
    amin_s = str(best_amin) if isinstance(best_amin, int) else str(best_amin)
    print(f"  │  {'*** DepthPro (this sweep)':<30s} {tau_s:>5s}  {amin_s:>5s}  "
          f"{best_m['PQ']:5.1f}  {best_m['PQ_stuff']:5.1f}  {best_m['PQ_things']:5.1f}  │")
    print(f"  └────────────────────────────────────────────────────────────────┘")


def print_per_class(best_m, best_label):
    """Print per-class PQ breakdown."""
    print(f"\n  Per-class PQ (best: {best_label}):")
    for name, v in sorted(best_m["per_class"].items(),
                           key=lambda x: x[1]["PQ"], reverse=True):
        kind = "S" if CLASS_NAMES.index(name) in STUFF_IDS else "T"
        if v["TP"] + v["FP"] + v["FN"] > 0:
            print(f"    [{kind}] {name:15s}: PQ={v['PQ']:5.1f}  SQ={v['SQ']:5.1f}  "
                  f"RQ={v['RQ']:5.1f}  (TP={v['TP']} FP={v['FP']} FN={v['FN']})")


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(
        description="DepthPro parameter sweep for depth-guided instance splitting"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val"])
    parser.add_argument("--eval_size", type=int, nargs=2, default=[512, 1024])
    parser.add_argument("--depth_subdir", type=str, default="depth_depthpro")
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_raw_k80")
    parser.add_argument("--centroids_path", type=str, default=None)
    parser.add_argument("--phase", type=str, default="all",
                        choices=["1", "2", "all"],
                        help="Which sweep phase to run")
    parser.add_argument("--phase1_best", type=float, nargs=2, default=None,
                        metavar=("TAU", "AMIN"),
                        help="Skip Phase 1; use this (tau, amin) for Phase 2")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit number of images (for debugging)")
    args = parser.parse_args()

    cs_root = Path(args.cityscapes_root)
    eval_hw = tuple(args.eval_size)

    # ─── Load cluster→trainID mapping ───
    centroids_path = args.centroids_path or str(
        cs_root / args.semantic_subdir / "kmeans_centroids.npz"
    )
    data = np.load(centroids_path)
    cluster_to_class = data["cluster_to_class"]
    k = len(cluster_to_class)

    cluster_to_trainid = np.full(256, 255, dtype=np.uint8)
    for cid, tid in enumerate(cluster_to_class):
        cluster_to_trainid[cid] = int(tid)

    # Print setup
    trainid_to_clusters = defaultdict(list)
    for cid, tid in enumerate(cluster_to_class):
        trainid_to_clusters[int(tid)].append(cid)

    print(f"\n{'='*80}")
    print(f"  DEPTHPRO PARAMETER SWEEP")
    print(f"  Depth-Guided Instance Splitting + k={k} Overclustered Semantics")
    print(f"{'='*80}")
    print(f"  Split: {args.split}")
    print(f"  Eval resolution: {eval_hw[0]}×{eval_hw[1]}")
    print(f"  Depth source: {args.depth_subdir}")
    print(f"  Semantic source: {args.semantic_subdir}")
    print(f"  Phase: {args.phase}")
    print(f"  Thing clusters mapped to trainIDs:")
    for tid in sorted(THING_IDS):
        clusters = trainid_to_clusters.get(tid, [])
        name = CLASS_NAMES[tid]
        if clusters:
            print(f"    {name:15s}: {len(clusters)} clusters → {clusters}")
        else:
            print(f"    {name:15s}: 0 clusters (MISSING)")

    # ─── Discover files ───
    files = discover_files(cs_root, args.split, args.semantic_subdir,
                           args.depth_subdir)
    print(f"\n  Found {len(files)} evaluation images")

    if args.max_images:
        files = files[:args.max_images]
        print(f"  Limited to {args.max_images} images")

    if not files:
        print("  ERROR: No files found. Check paths.")
        return

    # ═══════════════════════════════════════════════════════
    # Phase 1: Coarse τ × A_min sweep
    # ═══════════════════════════════════════════════════════

    phase1_results = []
    best_tau, best_amin = None, None

    if args.phase in ("1", "all"):
        t0_all = time.time()

        # CC-only baseline
        print(f"\n  Running CC-only baseline...")
        t0 = time.time()
        m = run_single_config(files, cluster_to_trainid, eval_hw,
                               grad_threshold=None, label="CC-only")
        dt = time.time() - t0
        phase1_results.append(("CC-only", "—", m))
        print(f"    PQ={m['PQ']:5.1f}  PQ_st={m['PQ_stuff']:5.1f}  "
              f"PQ_th={m['PQ_things']:5.1f}  mIoU={m.get('mIoU', 0):5.1f}  "
              f"inst={m['avg_instances']:4.1f}  ({dt:.0f}s)")

        # Depth-guided configs
        for i, (tau, amin) in enumerate(PHASE1_CONFIGS):
            label = f"τ={tau:.3f} A={amin}"
            print(f"\n  [{i+1}/{len(PHASE1_CONFIGS)}] {label}...")
            t0 = time.time()
            m = run_single_config(files, cluster_to_trainid, eval_hw,
                                   grad_threshold=tau, min_area=amin,
                                   label=label)
            dt = time.time() - t0
            phase1_results.append((tau, amin, m))
            print(f"    PQ={m['PQ']:5.1f}  PQ_st={m['PQ_stuff']:5.1f}  "
                  f"PQ_th={m['PQ_things']:5.1f}  mIoU={m.get('mIoU', 0):5.1f}  "
                  f"inst={m['avg_instances']:4.1f}  ({dt:.0f}s)")

        elapsed = time.time() - t0_all

        # Print Phase 1 results
        print_results_table(phase1_results,
                            f"Phase 1: DepthPro τ × A_min Sweep (k={k})")

        # Find best
        best_entry = max(phase1_results, key=lambda x: x[2]["PQ"])
        best_tau, best_amin, best_m = best_entry
        print(f"\n  ★ Phase 1 Best: τ={best_tau}, A_min={best_amin}")
        print(f"    PQ={best_m['PQ']:.2f}  PQ_stuff={best_m['PQ_stuff']:.2f}  "
              f"PQ_things={best_m['PQ_things']:.2f}")

        print_comparison_table(best_m, best_tau, best_amin, k)
        print_per_class(best_m, f"τ={best_tau}, A_min={best_amin}")
        print(f"\n  Phase 1 time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

        # Save Phase 1 results
        p1_path = str(cs_root / f"sweep_depthpro_phase1_{args.split}.json")
        p1_data = {
            "split": args.split, "k": k, "eval_resolution": list(eval_hw),
            "depth_subdir": args.depth_subdir,
            "semantic_subdir": args.semantic_subdir,
            "phase": 1,
            "best": {"tau": best_tau, "min_area": best_amin,
                     "PQ": best_m["PQ"], "PQ_things": best_m["PQ_things"]},
            "results": [
                {"grad_threshold": tau, "min_area": amin,
                 "PQ": m["PQ"], "PQ_stuff": m["PQ_stuff"],
                 "PQ_things": m["PQ_things"],
                 "SQ": m["SQ"], "RQ": m["RQ"],
                 "mIoU": m.get("mIoU", 0),
                 "avg_instances": m["avg_instances"],
                 "n_images": m["n_images"]}
                for tau, amin, m in phase1_results
            ],
        }
        with open(p1_path, "w") as f:
            json.dump(p1_data, f, indent=2, default=str)
        print(f"  Saved to: {p1_path}")

    # ═══════════════════════════════════════════════════════
    # Phase 2: Focused blur_sigma × dilation_iters sweep
    # ═══════════════════════════════════════════════════════

    if args.phase in ("2", "all"):
        # Use Phase 1 best or CLI override
        if args.phase1_best:
            best_tau, best_amin = args.phase1_best[0], int(args.phase1_best[1])
            print(f"\n  Using Phase 1 best from CLI: τ={best_tau}, "
                  f"A_min={best_amin}")
        elif best_tau is None:
            print("\n  ERROR: Phase 2 requires Phase 1 results or --phase1_best")
            return

        phase2_configs = get_phase2_configs(best_tau, best_amin)
        phase2_results = []
        t0_all = time.time()

        print(f"\n  Phase 2: Sweeping blur_sigma × dilation_iters "
              f"(τ={best_tau}, A_min={best_amin})")
        print(f"  {len(phase2_configs)} configurations")

        for i, (tau, amin, dil, sigma) in enumerate(phase2_configs):
            label = f"σ={sigma:.2f} dil={dil}"
            print(f"\n  [{i+1}/{len(phase2_configs)}] {label}...")
            t0 = time.time()
            m = run_single_config(files, cluster_to_trainid, eval_hw,
                                   grad_threshold=tau, min_area=amin,
                                   dilation_iters=dil,
                                   depth_blur_sigma=sigma,
                                   label=label)
            dt = time.time() - t0
            phase2_results.append((tau, amin, dil, sigma, m))
            print(f"    PQ={m['PQ']:5.1f}  PQ_st={m['PQ_stuff']:5.1f}  "
                  f"PQ_th={m['PQ_things']:5.1f}  mIoU={m.get('mIoU', 0):5.1f}  "
                  f"inst={m['avg_instances']:4.1f}  ({dt:.0f}s)")

        elapsed = time.time() - t0_all

        # Print Phase 2 results
        print_results_table(phase2_results,
                            f"Phase 2: DepthPro σ_blur × dilation Sweep",
                            show_sigma_dil=True)

        # Find best
        best_entry = max(phase2_results, key=lambda x: x[4]["PQ"])
        b_tau, b_amin, b_dil, b_sigma, b_m = best_entry
        print(f"\n  ★ Phase 2 Best: τ={b_tau}, A_min={b_amin}, "
              f"σ={b_sigma}, dilation={b_dil}")
        print(f"    PQ={b_m['PQ']:.2f}  PQ_stuff={b_m['PQ_stuff']:.2f}  "
              f"PQ_things={b_m['PQ_things']:.2f}")

        print_comparison_table(b_m, b_tau, b_amin, k)
        print_per_class(b_m,
                        f"τ={b_tau}, A_min={b_amin}, σ={b_sigma}, dil={b_dil}")
        print(f"\n  Phase 2 time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

        # Save Phase 2 results
        p2_path = str(cs_root / f"sweep_depthpro_phase2_{args.split}.json")
        p2_data = {
            "split": args.split, "k": k, "eval_resolution": list(eval_hw),
            "depth_subdir": args.depth_subdir,
            "semantic_subdir": args.semantic_subdir,
            "phase": 2,
            "best": {"tau": b_tau, "min_area": b_amin,
                     "dilation_iters": b_dil, "depth_blur_sigma": b_sigma,
                     "PQ": b_m["PQ"], "PQ_things": b_m["PQ_things"]},
            "results": [
                {"grad_threshold": tau, "min_area": amin,
                 "dilation_iters": dil, "depth_blur_sigma": sigma,
                 "PQ": m["PQ"], "PQ_stuff": m["PQ_stuff"],
                 "PQ_things": m["PQ_things"],
                 "SQ": m["SQ"], "RQ": m["RQ"],
                 "mIoU": m.get("mIoU", 0),
                 "avg_instances": m["avg_instances"],
                 "n_images": m["n_images"]}
                for tau, amin, dil, sigma, m in phase2_results
            ],
        }
        with open(p2_path, "w") as f:
            json.dump(p2_data, f, indent=2, default=str)
        print(f"  Saved to: {p2_path}")

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
