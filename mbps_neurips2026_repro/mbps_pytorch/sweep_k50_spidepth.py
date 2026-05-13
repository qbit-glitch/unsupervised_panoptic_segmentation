#!/usr/bin/env python3
"""Parameter sweep: depth-guided instance splitting with k=50 overclustered semantics.

Evaluates PQ across (grad_threshold, min_area) configurations on Cityscapes val.
Follows the same procedure as reports/overclustered_spidepth_sweep.md.

Usage:
    python mbps_pytorch/sweep_k50_spidepth.py \
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes
"""

import argparse
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


def remap_gt_to_trainids(gt_raw):
    out = np.full_like(gt_raw, 255, dtype=np.uint8)
    for cs_id, train_id in CS_ID_TO_TRAIN.items():
        out[gt_raw == cs_id] = train_id
    return out


def resize_nearest(arr, hw):
    h, w = hw
    return np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST))


# ─── Depth-Guided Instance Generation ───

def depth_guided_instances(semantic, depth, thing_ids=THING_IDS,
                           grad_threshold=0.05, min_area=100,
                           dilation_iters=3, depth_blur_sigma=1.0):
    """Split thing regions using depth gradient edges.

    Returns list of (mask, class_id, score) tuples sorted by area descending.
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
    """Evaluate one image. Returns per-class (tp, fp, fn, iou) dicts."""
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


# ─── Main Sweep ───

def run_single_config(files, k50_to_trainid, eval_hw, grad_threshold=None,
                       min_area=100, label=""):
    """Run evaluation for a single (grad_threshold, min_area) configuration.

    If grad_threshold is None, uses CC-only mode.
    """
    H, W = eval_hw
    tp_acc = np.zeros(NUM_CLASSES)
    fp_acc = np.zeros(NUM_CLASSES)
    fn_acc = np.zeros(NUM_CLASSES)
    iou_acc = np.zeros(NUM_CLASSES)
    total_instances = 0
    n_images = 0

    for sem_path, depth_path, gt_label_path, gt_inst_path in tqdm(
        files, desc=label, leave=False
    ):
        # Load and remap k=50 semantics → trainIDs
        pred_k50 = np.array(Image.open(sem_path))
        pred_sem = k50_to_trainid[pred_k50]
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

    metrics = compute_pq_from_accumulators(tp_acc, fp_acc, fn_acc, iou_acc)
    metrics["avg_instances"] = round(total_instances / max(n_images, 1), 1)
    metrics["n_images"] = n_images
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep: depth-guided splitting + k=50 overclustered semantics"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--eval_size", type=int, nargs=2, default=[512, 1024])
    parser.add_argument("--semantic_subdir", type=str, default="pseudo_semantic_raw_k50")
    parser.add_argument("--depth_subdir", type=str, default="depth_spidepth")
    parser.add_argument("--centroids_path", type=str, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    cs_root = Path(args.cityscapes_root)
    eval_hw = tuple(args.eval_size)

    # Load cluster→trainID mapping
    centroids_path = args.centroids_path or str(
        cs_root / args.semantic_subdir / "kmeans_centroids.npz"
    )
    data = np.load(centroids_path)
    cluster_to_class = data["cluster_to_class"]
    k = len(cluster_to_class)

    k50_to_trainid = np.full(256, 255, dtype=np.uint8)
    for cid, tid in enumerate(cluster_to_class):
        k50_to_trainid[cid] = int(tid)

    # Print cluster→class mapping
    trainid_to_clusters = defaultdict(list)
    for cid, tid in enumerate(cluster_to_class):
        trainid_to_clusters[int(tid)].append(cid)

    print(f"\n{'='*70}")
    print(f"PARAMETER SWEEP: Depth-Guided Splitting + k={k} Overclustered Semantics")
    print(f"{'='*70}")
    print(f"  Split: {args.split}")
    print(f"  Eval resolution: {eval_hw[0]}x{eval_hw[1]}")
    print(f"  Depth source: {args.depth_subdir}")
    print(f"  Thing clusters mapped to trainIDs:")
    for tid in sorted(THING_IDS):
        if tid in trainid_to_clusters:
            clusters = trainid_to_clusters[tid]
            print(f"    {CLASS_NAMES[tid]:15s}: {len(clusters)} clusters → {clusters}")
        else:
            print(f"    {CLASS_NAMES[tid]:15s}: 0 clusters (MISSING)")

    # Discover files
    files = discover_files(cs_root, args.split, args.semantic_subdir, args.depth_subdir)
    print(f"\n  Found {len(files)} evaluation images")

    if args.max_images:
        files = files[:args.max_images]
        print(f"  Limited to {args.max_images} images")

    # ─── Define sweep configurations ───
    # Same grid as reports/overclustered_spidepth_sweep.md
    configs = [
        # (grad_threshold, min_area)
        # Exact same grid as reports/overclustered_spidepth_sweep.md (k=300)
        (0.05, 100),
        (0.08, 200),
        (0.08, 500),
        (0.12, 200),
        (0.12, 500),
        (0.20, 200),
        (0.20, 500),
        (0.30, 500),
        (0.30, 600),
        (0.30, 700),
        (0.30, 800),
        (0.30, 1000),
        (0.50, 500),
        (0.50, 600),
        (0.50, 700),
        (0.50, 800),
        (0.50, 1000),
        (0.60, 500),
        (0.60, 600),
        (0.60, 700),
        (0.60, 800),
        (0.70, 600),
        (0.70, 700),
        (0.70, 800),
        (0.80, 500),
        (0.80, 1000),
        (0.80, 2000),
        (0.90, 1000),
        (1.00, 500),
        (1.00, 1000),
        (1.00, 2000),
    ]

    # ─── Run sweep ───
    results = []
    t0_all = time.time()

    # CC-only baseline first
    print(f"\n  Running CC-only baseline...")
    t0 = time.time()
    m = run_single_config(files, k50_to_trainid, eval_hw, grad_threshold=None,
                           label="CC-only")
    dt = time.time() - t0
    results.append(("CC-only", "—", m))
    print(f"    PQ={m['PQ']:5.1f}  PQ_st={m['PQ_stuff']:5.1f}  "
          f"PQ_th={m['PQ_things']:5.1f}  inst/img={m['avg_instances']:4.1f}  ({dt:.0f}s)")

    # Depth-guided configs
    for i, (gt, ma) in enumerate(configs):
        label = f"gt={gt:.2f} ma={ma}"
        print(f"\n  [{i+1}/{len(configs)}] {label}...")
        t0 = time.time()
        m = run_single_config(files, k50_to_trainid, eval_hw,
                               grad_threshold=gt, min_area=ma, label=label)
        dt = time.time() - t0
        results.append((gt, ma, m))
        print(f"    PQ={m['PQ']:5.1f}  PQ_st={m['PQ_stuff']:5.1f}  "
              f"PQ_th={m['PQ_things']:5.1f}  inst/img={m['avg_instances']:4.1f}  ({dt:.0f}s)")

    elapsed_all = time.time() - t0_all

    # ─── Print results table ───
    print(f"\n{'='*70}")
    print(f"RESULTS: Parameter Sweep (k={k} overclustered + SPIdepth depth)")
    print(f"{'='*70}")
    print(f"\n| grad_thresh | min_area | inst/img | PQ | PQ_stuff | PQ_things | SQ | RQ |")
    print(f"|------------|----------|----------|-----|----------|-----------|------|------|")

    for gt, ma, m in results:
        gt_str = f"{gt:.2f}" if isinstance(gt, float) else gt
        ma_str = str(ma) if isinstance(ma, int) else ma
        print(f"| {gt_str:>10s} | {ma_str:>8s} | {m['avg_instances']:8.1f} | "
              f"{m['PQ']:4.1f} | {m['PQ_stuff']:8.1f} | {m['PQ_things']:9.1f} | "
              f"{m['SQ']:4.1f} | {m['RQ']:4.1f} |")

    # Find best config
    best_gt, best_ma, best_m = max(results, key=lambda x: x[2]["PQ"])
    print(f"\n  Best config: grad_thresh={best_gt}, min_area={best_ma}")
    print(f"  Best PQ: {best_m['PQ']:.1f}  (PQ_stuff={best_m['PQ_stuff']:.1f}, "
          f"PQ_things={best_m['PQ_things']:.1f})")

    # Comparison table
    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  Comparison with Previous Pipelines                     │")
    print(f"  │  Pipeline                        PQ    PQ^St   PQ^Th   │")
    print(f"  │  CAUSE-27 + SPIdepth (old)       23.1  31.4    11.7    │")
    print(f"  │  k=300 overclustered (CC-only)   25.6  33.1    15.2    │")
    cc_result = results[0][2]
    print(f"  │  k={k} CC-only (this sweep)     "
          f"{cc_result['PQ']:5.1f} {cc_result['PQ_stuff']:5.1f}   {cc_result['PQ_things']:5.1f}    │")
    print(f"  │  k={k} best depth-guided        "
          f"{best_m['PQ']:5.1f} {best_m['PQ_stuff']:5.1f}   {best_m['PQ_things']:5.1f}    │")
    print(f"  │  CUPS (CVPR 2025 SOTA)           27.8  35.1    17.7    │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    print(f"\n  Total sweep time: {elapsed_all:.0f}s ({elapsed_all/60:.1f}min)")

    # Per-class breakdown for best config
    print(f"\n  Per-class PQ (best config: gt={best_gt}, ma={best_ma}):")
    for name, v in sorted(best_m["per_class"].items(),
                           key=lambda x: x[1]["PQ"], reverse=True):
        kind = "S" if CLASS_NAMES.index(name) in STUFF_IDS else "T"
        if v["TP"] + v["FP"] + v["FN"] > 0:
            print(f"    [{kind}] {name:15s}: PQ={v['PQ']:5.1f}  SQ={v['SQ']:5.1f}  "
                  f"RQ={v['RQ']:5.1f}  (TP={v['TP']} FP={v['FP']} FN={v['FN']})")

    # Save results
    import json
    output_path = str(cs_root / f"sweep_k{k}_spidepth_{args.split}.json")
    save_data = {
        "split": args.split, "k": k, "eval_resolution": list(eval_hw),
        "depth_subdir": args.depth_subdir,
        "results": [
            {"grad_threshold": gt, "min_area": ma,
             "PQ": m["PQ"], "PQ_stuff": m["PQ_stuff"], "PQ_things": m["PQ_things"],
             "SQ": m["SQ"], "RQ": m["RQ"],
             "avg_instances": m["avg_instances"], "n_images": m["n_images"]}
            for gt, ma, m in results
        ],
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
