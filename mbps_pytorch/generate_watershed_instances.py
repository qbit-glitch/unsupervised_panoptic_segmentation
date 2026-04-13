#!/usr/bin/env python3
"""Generate instance pseudo-labels via watershed segmentation on depth maps.

Phase 2.1 of the monocular depth ablation study. Alternative to Sobel-based splitting.

Watershed treats depth as a topographic surface. Instead of binary edge
thresholding, it finds basins (objects at different depths) and ridges
(depth discontinuities) naturally.

Algorithm per image:
  1. Load semantic labels and depth map
  2. For each thing class:
     a. Apply Gaussian blur to depth within class mask
     b. Find seeds via one of:
        - distance transform peaks on class mask
        - depth local minima (nearest objects)
        - combined: both seed types
     c. Run watershed segmentation
     d. Intersect with class mask, filter by min_area
  3. Save instances as NPZ

Usage:
    python mbps_pytorch/generate_watershed_instances.py \
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
        --depth_subdir depth_spidepth \
        --semantic_subdir pseudo_semantic_mapped_k80 \
        --split val --limit 10

    # Evaluate with different seed strategies:
    python mbps_pytorch/generate_watershed_instances.py \
        --cityscapes_root /path/to/cityscapes \
        --seed_mode combined \
        --blur_sigma 2.0 --min_distance 20
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter, binary_dilation, distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tqdm import tqdm

# ─── Cityscapes Constants ───

CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
    22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
    32: 17, 33: 18,
}
STUFF_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
THING_IDS = {11, 12, 13, 14, 15, 16, 17, 18}
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]
WORK_H, WORK_W = 512, 1024


def resize_nearest(arr, hw):
    h, w = hw
    return np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST))


def resize_bilinear(arr, hw):
    h, w = hw
    return np.array(Image.fromarray(arr).resize((w, h), Image.BILINEAR))


# ─── Watershed Instance Generation ───

def watershed_instances(semantic, depth, thing_ids=THING_IDS,
                        seed_mode="combined", blur_sigma=2.0,
                        min_distance=20, min_area=1000):
    """Split thing regions using watershed segmentation on depth.

    Args:
        semantic: (H, W) uint8 trainID map
        depth: (H, W) float32 depth map [0, 1]
        thing_ids: set of trainIDs for thing classes
        seed_mode: "distance" (distance transform peaks), "depth" (depth minima),
                   or "combined" (both)
        blur_sigma: Gaussian blur sigma on depth before watershed
        min_distance: Minimum distance between seed points (pixels)
        min_area: Minimum instance area in pixels

    Returns:
        List of (mask, class_id, score) tuples, sorted by area descending.
    """
    if blur_sigma > 0:
        depth_smooth = gaussian_filter(depth.astype(np.float64), sigma=blur_sigma)
    else:
        depth_smooth = depth.astype(np.float64)

    assigned = np.zeros(semantic.shape, dtype=bool)
    instances = []

    for cls in sorted(thing_ids):
        cls_mask = semantic == cls
        cls_area = cls_mask.sum()
        if cls_area < min_area:
            continue

        # ── Find seed points ──
        markers = np.zeros(semantic.shape, dtype=np.int32)
        marker_id = 1

        if seed_mode in ("distance", "combined"):
            # Distance transform peaks: center of each blob
            dist = distance_transform_edt(cls_mask)
            if dist.max() > 0:
                coords = peak_local_max(
                    dist, min_distance=min_distance,
                    labels=cls_mask.astype(np.uint8),
                    exclude_border=False,
                )
                for y, x in coords:
                    markers[y, x] = marker_id
                    marker_id += 1

        if seed_mode in ("depth", "combined"):
            # Depth local minima within class mask (nearest objects = lowest depth)
            # Invert depth so that nearby objects are peaks
            depth_in_class = np.where(cls_mask, depth_smooth, 1.0)
            inv_depth = 1.0 - depth_in_class
            inv_depth[~cls_mask] = 0

            if inv_depth.max() > 0:
                coords = peak_local_max(
                    inv_depth, min_distance=min_distance,
                    labels=cls_mask.astype(np.uint8),
                    exclude_border=False,
                )
                for y, x in coords:
                    if markers[y, x] == 0:  # Don't overwrite existing seeds
                        markers[y, x] = marker_id
                        marker_id += 1

        if marker_id <= 1:
            # No seeds found — treat entire class as one instance
            if cls_area >= min_area:
                instances.append((cls_mask.copy(), cls, float(cls_area)))
            continue

        # ── Run watershed ──
        # Use depth as the elevation map (watershed finds ridges = depth discontinuities)
        elevation = (depth_smooth * 255).astype(np.uint8)
        # Mask to only segment within this class
        elevation_masked = np.where(cls_mask, elevation, 255)

        labels = watershed(elevation_masked, markers=markers, mask=cls_mask)

        # ── Extract instances from watershed labels ──
        for label_id in range(1, marker_id):
            inst_mask = (labels == label_id) & cls_mask
            area = float(inst_mask.sum())
            if area < min_area:
                continue
            # Don't overlap with already assigned pixels
            inst_mask = inst_mask & ~assigned
            if inst_mask.sum() < min_area:
                continue
            assigned |= inst_mask
            instances.append((inst_mask, cls, area))

    instances.sort(key=lambda x: -x[2])
    if instances:
        max_area = instances[0][2]
        instances = [(m, c, s / max_area) for m, c, s in instances]
    return instances


def multiscale_sobel_instances(semantic, depth, thing_ids=THING_IDS,
                                grad_threshold=0.20, min_area=1000,
                                scales=(0.5, 1.0, 2.0), dilation_iters=3):
    """Split thing regions using multi-scale Sobel gradients on depth.

    Phase 2.2: Computes Sobel at multiple blur scales, takes max response.
    """
    from scipy.ndimage import sobel

    # Multi-scale gradient: compute at each sigma, take max
    grad_mags = []
    for sigma in scales:
        depth_smooth = gaussian_filter(depth.astype(np.float64), sigma=sigma)
        gx = sobel(depth_smooth, axis=1)
        gy = sobel(depth_smooth, axis=0)
        grad_mags.append(np.sqrt(gx ** 2 + gy ** 2))

    grad_mag = np.maximum.reduce(grad_mags)
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
                dilated = binary_dilation(cc_mask, iterations=dilation_iters)
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


def canny_instances(semantic, depth, thing_ids=THING_IDS,
                    low_threshold=20, high_threshold=50,
                    min_area=1000, blur_sigma=1.0, dilation_iters=3):
    """Split thing regions using Canny edge detection on depth.

    Phase 2.3: Canny has built-in NMS, hysteresis, and edge linking.
    """
    if blur_sigma > 0:
        depth_smooth = gaussian_filter(depth.astype(np.float64), sigma=blur_sigma)
    else:
        depth_smooth = depth.astype(np.float64)

    # Scale to uint8 for Canny
    d_min, d_max = depth_smooth.min(), depth_smooth.max()
    if d_max - d_min > 1e-6:
        depth_uint8 = ((depth_smooth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        depth_uint8 = np.zeros_like(depth_smooth, dtype=np.uint8)

    depth_edges = cv2.Canny(depth_uint8, low_threshold, high_threshold).astype(bool)

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
                dilated = binary_dilation(cc_mask, iterations=dilation_iters)
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


# ─── Evaluation (from sweep_depth_model_comparison.py) ───

NUM_CLASSES = 19

def remap_gt_to_trainids(gt_raw):
    out = np.full_like(gt_raw, 255, dtype=np.uint8)
    for cs_id, train_id in CS_ID_TO_TRAIN.items():
        out[gt_raw == cs_id] = train_id
    return out


def evaluate_panoptic_single(pred_sem, pred_instances, gt_sem, gt_inst_map, eval_hw):
    H, W = eval_hw
    pred_pan = np.zeros((H, W), dtype=np.int32)
    pred_segments = {}
    next_id = 1

    for cls in STUFF_IDS:
        mask = pred_sem == cls
        if mask.sum() < 64:
            continue
        pred_pan[mask] = next_id
        pred_segments[next_id] = cls
        next_id += 1

    for mask, cls, score in pred_instances:
        if cls not in THING_IDS:
            continue
        new_pixels = mask & (pred_pan == 0)
        if new_pixels.sum() < 10:
            continue
        pred_pan[new_pixels] = next_id
        pred_segments[next_id] = cls
        next_id += 1

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

    return tp, fp, fn, iou_sum


def compute_pq(tp, fp, fn, iou_sum):
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
        "per_class": per_class,
    }


# ─── Data Discovery ───

def discover_files(cityscapes_root, split, semantic_subdir, depth_subdir):
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


# ─── Main ───

ALGORITHMS = {
    "watershed_distance": lambda sem, dep, ma: watershed_instances(
        sem, dep, seed_mode="distance", min_area=ma),
    "watershed_depth": lambda sem, dep, ma: watershed_instances(
        sem, dep, seed_mode="depth", min_area=ma),
    "watershed_combined": lambda sem, dep, ma: watershed_instances(
        sem, dep, seed_mode="combined", min_area=ma),
    "multiscale_sobel": lambda sem, dep, ma: multiscale_sobel_instances(
        sem, dep, min_area=ma),
    "canny_20_50": lambda sem, dep, ma: canny_instances(
        sem, dep, low_threshold=20, high_threshold=50, min_area=ma),
    "canny_30_80": lambda sem, dep, ma: canny_instances(
        sem, dep, low_threshold=30, high_threshold=80, min_area=ma),
    "canny_10_30": lambda sem, dep, ma: canny_instances(
        sem, dep, low_threshold=10, high_threshold=30, min_area=ma),
}


def run_algorithm(files, eval_hw, algorithm_name, min_area=1000):
    """Run a single splitting algorithm on all files and compute PQ."""
    H, W = eval_hw
    tp_acc = np.zeros(NUM_CLASSES)
    fp_acc = np.zeros(NUM_CLASSES)
    fn_acc = np.zeros(NUM_CLASSES)
    iou_acc = np.zeros(NUM_CLASSES)
    total_inst = 0

    algo_fn = ALGORITHMS[algorithm_name]

    for sem_path, depth_path, gt_label_path, gt_inst_path in tqdm(
        files, desc=algorithm_name, leave=False
    ):
        pred_sem = np.array(Image.open(sem_path))
        if pred_sem.shape != (H, W):
            pred_sem = resize_nearest(pred_sem, eval_hw)

        depth = np.load(str(depth_path))
        if depth.shape != (H, W):
            depth = resize_bilinear(depth, eval_hw)

        instances = algo_fn(pred_sem, depth, min_area)

        gt_raw = np.array(Image.open(gt_label_path))
        gt_sem = remap_gt_to_trainids(gt_raw)
        if gt_sem.shape != (H, W):
            gt_sem = resize_nearest(gt_sem, eval_hw)

        gt_inst_map = np.array(Image.open(gt_inst_path), dtype=np.int32)
        if gt_inst_map.shape != (H, W):
            gt_inst_map = np.array(
                Image.fromarray(gt_inst_map).resize((W, H), Image.NEAREST)
            )

        tp, fp, fn, iou_s = evaluate_panoptic_single(
            pred_sem, instances, gt_sem, gt_inst_map, eval_hw,
        )
        tp_acc += tp
        fp_acc += fp
        fn_acc += fn
        iou_acc += iou_s
        total_inst += len(instances)

    metrics = compute_pq(tp_acc, fp_acc, fn_acc, iou_acc)
    metrics["avg_instances"] = round(total_inst / max(len(files), 1), 1)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Alternative splitting algorithms: watershed, multi-scale Sobel, Canny"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--depth_subdir", type=str, default="depth_spidepth")
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_mapped_k80")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--algorithms", type=str, nargs="+",
                        default=list(ALGORITHMS.keys()),
                        choices=list(ALGORITHMS.keys()),
                        help="Which algorithms to run")
    parser.add_argument("--min_area", type=int, default=1000)
    parser.add_argument("--max_images", type=int, default=None)

    args = parser.parse_args()
    eval_hw = (WORK_H, WORK_W)

    files = discover_files(
        args.cityscapes_root, args.split,
        args.semantic_subdir, args.depth_subdir,
    )
    if args.max_images:
        files = files[:args.max_images]

    print(f"\n{'='*70}")
    print(f"ALTERNATIVE SPLITTING ALGORITHMS")
    print(f"{'='*70}")
    print(f"  Depth:      {args.depth_subdir}")
    print(f"  Semantics:  {args.semantic_subdir}")
    print(f"  Images:     {len(files)}")
    print(f"  Min area:   {args.min_area}")
    print(f"  Algorithms: {args.algorithms}")

    results = {}
    t0_all = time.time()

    for algo_name in args.algorithms:
        print(f"\n  Running {algo_name}...", flush=True)
        t0 = time.time()
        m = run_algorithm(files, eval_hw, algo_name, min_area=args.min_area)
        dt = time.time() - t0
        results[algo_name] = m
        print(f"    PQ={m['PQ']:5.1f}  PQ_st={m['PQ_stuff']:5.1f}  "
              f"PQ_th={m['PQ_things']:5.1f}  inst/img={m['avg_instances']:4.1f}  ({dt:.0f}s)")

    elapsed = time.time() - t0_all

    # ─── Results table ───
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"\n  {'Algorithm':30s} | {'PQ':>6s} | {'PQ_st':>6s} | {'PQ_th':>6s} | {'inst':>5s}")
    print(f"  {'-'*30}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}")

    for algo_name, m in sorted(results.items(), key=lambda x: -x[1]["PQ"]):
        print(f"  {algo_name:30s} | {m['PQ']:6.1f} | {m['PQ_stuff']:6.1f} | "
              f"{m['PQ_things']:6.1f} | {m['avg_instances']:5.1f}")

    # Per-class for best algorithm
    best_algo = max(results.items(), key=lambda x: x[1]["PQ_things"])
    print(f"\n  Best PQ_things: {best_algo[0]} = {best_algo[1]['PQ_things']:.1f}")
    print(f"\n  Per-class (best = {best_algo[0]}):")
    for cls in sorted(THING_IDS):
        name = CLASS_NAMES[cls]
        v = best_algo[1]["per_class"][name]
        if v["TP"] + v["FP"] + v["FN"] > 0:
            print(f"    {name:15s}: PQ={v['PQ']:5.1f}  SQ={v['SQ']:5.1f}  "
                  f"RQ={v['RQ']:5.1f}  (TP={v['TP']} FP={v['FP']} FN={v['FN']})")

    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save
    output_path = Path(args.cityscapes_root) / f"sweep_algorithms_{args.depth_subdir}_{args.split}.json"
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "depth_subdir": args.depth_subdir,
                "semantic_subdir": args.semantic_subdir,
                "split": args.split,
                "min_area": args.min_area,
                "max_images": args.max_images,
            },
            "results": {
                k: {kk: vv for kk, vv in v.items()}
                for k, v in results.items()
            },
        }, f, indent=2)
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
