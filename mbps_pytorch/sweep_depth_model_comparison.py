#!/usr/bin/env python3
"""Cross-model depth comparison: sweep grad_threshold × min_area per depth source.

Phase 1.3 of the monocular depth ablation study. Evaluates multiple depth models
on the same semantic labels (k=80 mapped) with per-model threshold optimization.

Usage:
    # Compare SPIdepth vs DA2-Large vs DA3 on val:
    python mbps_pytorch/sweep_depth_model_comparison.py \
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
        --depth_subdirs depth_spidepth depth_dav3 depth_da2_large

    # Quick test (10 images, 3 thresholds):
    python mbps_pytorch/sweep_depth_model_comparison.py \
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
        --depth_subdirs depth_spidepth depth_dav3 \
        --max_images 10 --quick
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel, binary_dilation
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

# ─── Sweep Configurations ───

FULL_SWEEP = [
    # (grad_threshold, min_area)
    (0.03, 500), (0.03, 1000),
    (0.05, 500), (0.05, 1000),
    (0.08, 500), (0.08, 1000),
    (0.10, 500), (0.10, 1000),
    (0.12, 500), (0.12, 1000),
    (0.15, 500), (0.15, 1000),
    (0.20, 500), (0.20, 700), (0.20, 1000), (0.20, 1500),
    (0.25, 500), (0.25, 1000),
    (0.30, 500), (0.30, 700), (0.30, 1000), (0.30, 1500),
    (0.40, 500), (0.40, 1000),
    (0.50, 500), (0.50, 1000),
]

QUICK_SWEEP = [
    (0.10, 1000),
    (0.20, 1000),
    (0.30, 1000),
]


def resize_nearest(arr, hw):
    h, w = hw
    return np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST))


def remap_gt_to_trainids(gt_raw):
    out = np.full_like(gt_raw, 255, dtype=np.uint8)
    for cs_id, train_id in CS_ID_TO_TRAIN.items():
        out[gt_raw == cs_id] = train_id
    return out


# ─── Instance Generation ───

def depth_guided_instances(semantic, depth, thing_ids=THING_IDS,
                           grad_threshold=0.20, min_area=1000,
                           dilation_iters=3, depth_blur_sigma=1.0):
    """Split thing regions using depth Sobel gradient edges."""
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


def cc_only_instances(semantic, thing_ids=THING_IDS, min_area=10):
    """Connected component instances without depth splitting (baseline)."""
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

def evaluate_panoptic_single(pred_sem, pred_instances, gt_sem, gt_inst_map, eval_hw):
    """Evaluate one image. Returns per-class (tp, fp, fn, iou) arrays."""
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
    """Compute PQ/SQ/RQ from accumulators."""
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


# ─��─ Single Config Evaluation ───

def run_config(files, eval_hw, grad_threshold=None, min_area=1000, label=""):
    """Evaluate one (grad_threshold, min_area) configuration.

    If grad_threshold is None, uses CC-only mode.
    Returns metrics dict and edge density.
    """
    H, W = eval_hw
    tp_acc = np.zeros(NUM_CLASSES)
    fp_acc = np.zeros(NUM_CLASSES)
    fn_acc = np.zeros(NUM_CLASSES)
    iou_acc = np.zeros(NUM_CLASSES)
    total_inst = 0
    edge_densities = []

    for sem_path, depth_path, gt_label_path, gt_inst_path in tqdm(
        files, desc=label, leave=False
    ):
        pred_sem = np.array(Image.open(sem_path))
        if pred_sem.shape != (H, W):
            pred_sem = resize_nearest(pred_sem, eval_hw)

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
            # Compute edge density
            depth_smooth = gaussian_filter(depth.astype(np.float64), sigma=1.0)
            gx = sobel(depth_smooth, axis=1)
            gy = sobel(depth_smooth, axis=0)
            grad_mag = np.sqrt(gx**2 + gy**2)
            edge_densities.append(float((grad_mag > grad_threshold).mean()))
        else:
            instances = cc_only_instances(pred_sem, THING_IDS, min_area=10)

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
    metrics["n_images"] = len(files)
    metrics["edge_density"] = round(float(np.mean(edge_densities)), 6) if edge_densities else 0.0
    return metrics


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(
        description="Cross-model depth comparison sweep"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--depth_subdirs", type=str, nargs="+", required=True,
                        help="Depth subdirectories to compare "
                             "(e.g. depth_spidepth depth_dav3 depth_da2_large)")
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_mapped_k80")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--eval_size", type=int, nargs=2, default=[512, 1024])
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Use reduced sweep grid (3 configs instead of 26)")

    args = parser.parse_args()
    eval_hw = tuple(args.eval_size)
    configs = QUICK_SWEEP if args.quick else FULL_SWEEP

    print(f"\n{'='*70}")
    print(f"CROSS-MODEL DEPTH COMPARISON SWEEP")
    print(f"{'='*70}")
    print(f"  Root:      {args.cityscapes_root}")
    print(f"  Semantics: {args.semantic_subdir}")
    print(f"  Depth:     {args.depth_subdirs}")
    print(f"  Configs:   {len(configs)} (grad_threshold × min_area)")

    all_results = {}
    t0_all = time.time()

    for depth_subdir in args.depth_subdirs:
        print(f"\n{'─'*70}")
        print(f"  DEPTH SOURCE: {depth_subdir}")
        print(f"{'─'*70}")

        files = discover_files(
            args.cityscapes_root, args.split,
            args.semantic_subdir, depth_subdir,
        )
        if args.max_images:
            files = files[:args.max_images]
        print(f"  Found {len(files)} images")

        results = []

        # CC-only baseline (shared across all depth sources)
        if depth_subdir == args.depth_subdirs[0]:
            print(f"\n  Running CC-only baseline...")
            t0 = time.time()
            m = run_config(files, eval_hw, grad_threshold=None, label="CC-only")
            dt = time.time() - t0
            results.append(("CC-only", "—", m))
            print(f"    PQ={m['PQ']:5.1f}  PQ_st={m['PQ_stuff']:5.1f}  "
                  f"PQ_th={m['PQ_things']:5.1f}  inst/img={m['avg_instances']:4.1f}  ({dt:.0f}s)")

        # Depth-guided configs
        for i, (gt, ma) in enumerate(configs):
            label = f"{depth_subdir} gt={gt:.2f} ma={ma}"
            print(f"\n  [{i+1}/{len(configs)}] gt={gt:.2f} ma={ma}...", end="", flush=True)
            t0 = time.time()
            m = run_config(files, eval_hw, grad_threshold=gt, min_area=ma, label=label)
            dt = time.time() - t0
            results.append((gt, ma, m))
            print(f"  PQ={m['PQ']:5.1f}  PQ_st={m['PQ_stuff']:5.1f}  "
                  f"PQ_th={m['PQ_things']:5.1f}  edge={m['edge_density']:.4f}  "
                  f"inst/img={m['avg_instances']:4.1f}  ({dt:.0f}s)")

        # Find best
        best_gt, best_ma, best_m = max(results, key=lambda x: x[2]["PQ"])
        best_thing_gt, best_thing_ma, best_thing_m = max(
            results, key=lambda x: x[2]["PQ_things"]
        )

        print(f"\n  Best PQ:        gt={best_gt}, ma={best_ma} → PQ={best_m['PQ']:.1f}")
        print(f"  Best PQ_things: gt={best_thing_gt}, ma={best_thing_ma} → "
              f"PQ_th={best_thing_m['PQ_things']:.1f}")

        # Per-class for best PQ config
        print(f"\n  Per-class (best PQ: gt={best_gt}, ma={best_ma}):")
        for cls in sorted(THING_IDS):
            name = CLASS_NAMES[cls]
            v = best_m["per_class"][name]
            if v["TP"] + v["FP"] + v["FN"] > 0:
                print(f"    {name:15s}: PQ={v['PQ']:5.1f}  SQ={v['SQ']:5.1f}  "
                      f"RQ={v['RQ']:5.1f}  (TP={v['TP']} FP={v['FP']} FN={v['FN']})")

        all_results[depth_subdir] = {
            "n_images": len(files),
            "best_pq": {"gt": best_gt, "ma": best_ma, **{k: v for k, v in best_m.items() if k != "per_class"}},
            "best_pq_things": {"gt": best_thing_gt, "ma": best_thing_ma,
                               **{k: v for k, v in best_thing_m.items() if k != "per_class"}},
            "sweep": [
                {"grad_threshold": gt, "min_area": ma,
                 "PQ": m["PQ"], "PQ_stuff": m["PQ_stuff"], "PQ_things": m["PQ_things"],
                 "edge_density": m.get("edge_density", 0),
                 "avg_instances": m["avg_instances"]}
                for gt, ma, m in results
            ],
            "best_per_class": best_m["per_class"],
        }

    elapsed = time.time() - t0_all

    # ─── Cross-Model Comparison Table ───
    print(f"\n{'='*70}")
    print(f"CROSS-MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"\n  {'Depth Model':25s} | {'Best PQ':>8s} | {'PQ_stuff':>8s} | {'PQ_things':>9s} | {'Opt tau':>8s} | {'Opt MA':>7s}")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}")

    for ds in args.depth_subdirs:
        r = all_results[ds]
        b = r["best_pq"]
        gt_str = str(b['gt']) if not isinstance(b['gt'], float) else f"{b['gt']:.2f}"
        ma_str = str(b['ma'])
        print(f"  {ds:25s} | {b['PQ']:8.1f} | {b['PQ_stuff']:8.1f} | {b['PQ_things']:9.1f} | "
              f"{gt_str:>8s} | {ma_str:>7s}")

    # Per-class comparison at each model's optimal threshold
    print(f"\n  Per-class PQ_things (each at its optimal threshold):")
    print(f"  {'Class':15s}", end="")
    for ds in args.depth_subdirs:
        label = ds.replace("depth_", "")[:10]
        print(f" | {label:>10s}", end="")
    print()
    print(f"  {'-'*15}", end="")
    for _ in args.depth_subdirs:
        print(f"-+-{'-'*10}", end="")
    print()

    for cls in sorted(THING_IDS):
        name = CLASS_NAMES[cls]
        print(f"  {name:15s}", end="")
        for ds in args.depth_subdirs:
            v = all_results[ds]["best_per_class"][name]
            print(f" | {v['PQ']:10.1f}", end="")
        print()

    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save results
    output_path = Path(args.cityscapes_root) / f"sweep_depth_comparison_{args.split}.json"
    save_data = {
        "config": {
            "semantic_subdir": args.semantic_subdir,
            "depth_subdirs": args.depth_subdirs,
            "split": args.split,
            "eval_resolution": list(eval_hw),
            "max_images": args.max_images,
        },
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
