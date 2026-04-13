#!/usr/bin/env python3
"""Cross-model depth comparison on COCO-Stuff-27.

Evaluates multiple depth models for instance pseudo-label quality using
the same pipeline as the Cityscapes sweep but adapted for COCO-Stuff-27
(27 coarse supercategory classes, variable image sizes, COCO panoptic GT).

Usage:
    python mbps_pytorch/sweep_coco_depth_comparison.py \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --depth_subdirs depth_da2_large depth_dav3

    # Quick test:
    python mbps_pytorch/sweep_coco_depth_comparison.py \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --depth_subdirs depth_dav3 --max_images 10 --quick
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

# ─── COCO-Stuff-27 Constants ───

COCOSTUFF27_CLASSNAMES = [
    "electronic", "appliance", "food", "furniture", "indoor",
    "kitchen", "accessory", "animal", "outdoor", "person",
    "sports", "vehicle",
    "ceiling", "floor", "food-stuff", "furniture-stuff", "raw-material",
    "textile", "wall", "window", "building", "ground",
    "plant", "sky", "solid", "structural", "water",
]
NUM_CLASSES = 27
THING_IDS = set(range(12))   # 0-11
STUFF_IDS = set(range(12, 27))  # 12-26

SUPERCATEGORY_TO_COARSE = {
    "electronic": 0, "appliance": 1, "food": 2, "furniture": 3,
    "indoor": 4, "kitchen": 5, "accessory": 6, "animal": 7,
    "outdoor": 8, "person": 9, "sports": 10, "vehicle": 11,
    "ceiling": 12, "floor": 13, "food-stuff": 14, "furniture-stuff": 15,
    "raw-material": 16, "textile": 17, "wall": 18, "window": 19,
    "building": 20, "ground": 21, "plant": 22, "sky": 23,
    "solid": 24, "structural": 25, "water": 26,
}

# ─── Sweep Configurations ───

FULL_SWEEP = [
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

QUICK_SWEEP = [(0.05, 1000), (0.20, 1000), (0.50, 1000)]


# ─── COCO Panoptic GT Loading ───

class COCOPanopticGT:
    """Loads COCO panoptic GT and provides semantic + instance labels."""

    def __init__(self, coco_root):
        panoptic_json = Path(coco_root) / "annotations" / "panoptic_val2017.json"
        self.panoptic_dir = Path(coco_root) / "annotations" / "panoptic_val2017"

        with open(panoptic_json) as f:
            data = json.load(f)

        self.cat_map = {}
        self.cat_isthing = {}
        for cat in data["categories"]:
            self.cat_map[cat["id"]] = cat["supercategory"]
            self.cat_isthing[cat["id"]] = cat.get("isthing", 0) == 1

        self.ann_map = {}
        for ann in data["annotations"]:
            self.ann_map[ann["image_id"]] = ann

    def get_labels(self, image_id):
        """Returns (semantic_label, instance_label) arrays.

        semantic_label: (H, W) uint8, values 0-26 (27 classes), 255=void
        instance_label: (H, W) int32, unique per-segment IDs for things
        """
        if image_id not in self.ann_map:
            return None, None

        ann = self.ann_map[image_id]
        pan_path = self.panoptic_dir / ann["file_name"]
        pan_img = np.array(Image.open(pan_path))
        pan_id = (pan_img[:, :, 0].astype(np.int32) +
                  pan_img[:, :, 1].astype(np.int32) * 256 +
                  pan_img[:, :, 2].astype(np.int32) * 256 * 256)

        H, W = pan_id.shape
        sem_label = np.full((H, W), 255, dtype=np.uint8)
        inst_label = np.zeros((H, W), dtype=np.int32)

        thing_counter = 1
        for seg in ann["segments_info"]:
            mask = pan_id == seg["id"]
            cat_id = seg["category_id"]
            supercat = self.cat_map.get(cat_id)
            if supercat and supercat in SUPERCATEGORY_TO_COARSE:
                coarse_id = SUPERCATEGORY_TO_COARSE[supercat]
                sem_label[mask] = coarse_id
                if coarse_id in THING_IDS:
                    inst_label[mask] = thing_counter
                    thing_counter += 1

        return sem_label, inst_label


# ─── Splitting Algorithms ───

def resize_nearest(arr, target_hw):
    return np.array(Image.fromarray(arr).resize(
        (target_hw[1], target_hw[0]), Image.NEAREST))


def depth_guided_instances(pred_sem, depth, grad_threshold, min_area,
                           depth_blur_sigma=1.0, dilation_iters=3):
    """Standard Sobel splitting — same as Cityscapes version."""
    H, W = pred_sem.shape
    depth_f = depth.astype(np.float64)
    if depth_blur_sigma > 0:
        depth_f = gaussian_filter(depth_f, sigma=depth_blur_sigma)

    gx = sobel(depth_f, axis=1)
    gy = sobel(depth_f, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_density = float((grad_mag > grad_threshold).mean())

    panoptic = np.zeros(pred_sem.shape, dtype=np.int32)
    instance_counter = 1

    for cls in range(NUM_CLASSES):
        cls_mask = pred_sem == cls
        if not cls_mask.any():
            continue

        if cls in STUFF_IDS:
            panoptic[cls_mask] = cls * 1000
            continue

        # Thing class: split by depth edges
        boundary = (grad_mag > grad_threshold) & cls_mask
        interior = cls_mask & ~boundary

        labeled, n_cc = ndimage.label(interior)

        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            if cc_mask.sum() < min_area:
                continue
            # Dilate to reclaim boundary pixels
            dilated = binary_dilation(cc_mask, iterations=dilation_iters)
            reclaimed = dilated & cls_mask & (panoptic == 0)
            final_mask = cc_mask | reclaimed
            panoptic[final_mask] = cls * 1000 + instance_counter
            instance_counter += 1

        # Reclaim remaining unassigned thing pixels
        remaining = cls_mask & (panoptic == 0)
        if remaining.any():
            panoptic[remaining] = cls * 1000

    return panoptic, edge_density


def cc_only_instances(pred_sem, min_area=0):
    """Connected components without depth — baseline."""
    panoptic = np.zeros(pred_sem.shape, dtype=np.int32)
    instance_counter = 1

    for cls in range(NUM_CLASSES):
        cls_mask = pred_sem == cls
        if not cls_mask.any():
            continue

        if cls in STUFF_IDS:
            panoptic[cls_mask] = cls * 1000
            continue

        labeled, n_cc = ndimage.label(cls_mask)
        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            if cc_mask.sum() < min_area:
                continue
            panoptic[cc_mask] = cls * 1000 + instance_counter
            instance_counter += 1

        remaining = cls_mask & (panoptic == 0)
        if remaining.any():
            panoptic[remaining] = cls * 1000

    return panoptic


# ─── PQ Evaluation ───

def compute_pq(pred_pan, gt_sem, gt_inst):
    """Compute PQ between predicted panoptic and GT semantic+instance."""
    H, W = gt_sem.shape
    if pred_pan.shape != (H, W):
        pred_pan = resize_nearest(pred_pan, (H, W))

    tp_acc = np.zeros(NUM_CLASSES)
    fp_acc = np.zeros(NUM_CLASSES)
    fn_acc = np.zeros(NUM_CLASSES)
    iou_acc = np.zeros(NUM_CLASSES)

    for cls in range(NUM_CLASSES):
        gt_cls_mask = gt_sem == cls
        if not gt_cls_mask.any() and not np.any((pred_pan // 1000) == cls):
            continue

        if cls in STUFF_IDS:
            pred_cls_mask = (pred_pan // 1000) == cls
            inter = (gt_cls_mask & pred_cls_mask).sum()
            union = (gt_cls_mask | pred_cls_mask).sum()
            if union > 0:
                iou = inter / union
                if iou > 0.5:
                    tp_acc[cls] += 1
                    iou_acc[cls] += iou
                else:
                    if gt_cls_mask.any():
                        fn_acc[cls] += 1
                    if pred_cls_mask.any():
                        fp_acc[cls] += 1
            else:
                pass
            continue

        # Thing class: match instances
        gt_inst_in_cls = gt_inst.copy()
        gt_inst_in_cls[~gt_cls_mask] = 0
        gt_ids = set(np.unique(gt_inst_in_cls)) - {0}

        pred_cls_mask = (pred_pan // 1000) == cls
        pred_inst = pred_pan.copy()
        pred_inst[~pred_cls_mask] = 0
        pred_ids = set(np.unique(pred_inst)) - {0}

        matched_gt = set()
        matched_pred = set()

        for gt_id in gt_ids:
            gt_mask = gt_inst_in_cls == gt_id
            best_iou = 0
            best_pred = None
            for pred_id in pred_ids:
                if pred_id in matched_pred:
                    continue
                pred_mask = pred_inst == pred_id
                inter = (gt_mask & pred_mask).sum()
                union = (gt_mask | pred_mask).sum()
                if union > 0:
                    iou = inter / union
                    if iou > best_iou:
                        best_iou = iou
                        best_pred = pred_id

            if best_iou > 0.5:
                tp_acc[cls] += 1
                iou_acc[cls] += best_iou
                matched_gt.add(gt_id)
                matched_pred.add(best_pred)

        fn_acc[cls] += len(gt_ids - matched_gt)
        fp_acc[cls] += len(pred_ids - matched_pred)

    return tp_acc, fp_acc, fn_acc, iou_acc


def compute_pq_from_accumulators(tp, fp, fn, iou):
    """Compute PQ, SQ, RQ from accumulated stats."""
    per_class = {}
    all_pq, stuff_pq, thing_pq = [], [], []

    for c in range(NUM_CLASSES):
        t, f_p, f_n, i = tp[c], fp[c], fn[c], iou[c]
        if t == 0:
            pq, sq, rq = 0, 0, 0
        else:
            sq = i / t
            rq = t / (t + 0.5 * f_p + 0.5 * f_n)
            pq = sq * rq

        per_class[COCOSTUFF27_CLASSNAMES[c]] = {
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

def discover_files(coco_root, semantic_subdir, depth_subdir, max_images=None):
    """Find matching (semantic, depth, image_id) tuples."""
    root = Path(coco_root)
    sem_dir = root / semantic_subdir
    depth_dir = root / depth_subdir

    # Find images that have both semantic labels and depth maps
    sem_files = {f.stem: f for f in sem_dir.glob("*.png")}
    depth_files = {f.stem: f for f in depth_dir.glob("*.npy")}

    common_ids = sorted(set(sem_files.keys()) & set(depth_files.keys()))
    if max_images:
        common_ids = common_ids[:max_images]

    files = []
    for img_id_str in common_ids:
        files.append((
            sem_files[img_id_str],
            depth_files[img_id_str],
            int(img_id_str),
        ))

    return files


# ─── Single Config Evaluation ───

def run_config(files, gt_loader, grad_threshold=None, min_area=1000, label=""):
    """Evaluate one (grad_threshold, min_area) configuration."""
    tp_acc = np.zeros(NUM_CLASSES)
    fp_acc = np.zeros(NUM_CLASSES)
    fn_acc = np.zeros(NUM_CLASSES)
    iou_acc = np.zeros(NUM_CLASSES)
    total_inst = 0
    edge_densities = []

    for sem_path, depth_path, image_id in tqdm(files, desc=label, leave=False):
        pred_sem = np.array(Image.open(sem_path))
        depth = np.load(depth_path)

        # Get GT
        gt_sem, gt_inst = gt_loader.get_labels(image_id)
        if gt_sem is None:
            continue

        H, W = gt_sem.shape

        # Resize predictions to GT size
        if pred_sem.shape != (H, W):
            pred_sem = resize_nearest(pred_sem, (H, W))
        if depth.shape != (H, W):
            depth = np.array(Image.fromarray(depth).resize((W, H), Image.BILINEAR))

        if grad_threshold is None:
            pred_pan = cc_only_instances(pred_sem, min_area=min_area)
            ed = 0.0
        else:
            pred_pan, ed = depth_guided_instances(
                pred_sem, depth, grad_threshold, min_area)
            edge_densities.append(ed)

        # Count instances
        pred_things = pred_pan[pred_pan > 0]
        unique_inst = set()
        for v in np.unique(pred_things):
            if v % 1000 != 0:
                unique_inst.add(v)
        total_inst += len(unique_inst)

        # Compute PQ
        tp, fp, fn, iou = compute_pq(pred_pan, gt_sem, gt_inst)
        tp_acc += tp
        fp_acc += fp
        fn_acc += fn
        iou_acc += iou

    metrics = compute_pq_from_accumulators(tp_acc, fp_acc, fn_acc, iou_acc)
    metrics["avg_instances"] = round(total_inst / max(len(files), 1), 1)
    metrics["edge_density"] = round(float(np.mean(edge_densities)), 4) if edge_densities else 0
    return metrics


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="COCO-Stuff-27 depth model sweep")
    parser.add_argument("--coco_root", required=True)
    parser.add_argument("--depth_subdirs", nargs="+", required=True)
    parser.add_argument("--semantic_subdir", default="pseudo_semantic_k80/val2017")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    configs = QUICK_SWEEP if args.quick else FULL_SWEEP

    print(f"{'='*70}")
    print(f"COCO-STUFF-27 DEPTH COMPARISON SWEEP")
    print(f"{'='*70}")
    print(f"  Root:      {args.coco_root}")
    print(f"  Semantics: {args.semantic_subdir}")
    print(f"  Depth:     {args.depth_subdirs}")
    print(f"  Configs:   {len(configs)}")

    gt_loader = COCOPanopticGT(args.coco_root)
    all_results = {}
    t0_all = time.time()

    for depth_subdir in args.depth_subdirs:
        print(f"\n{'─'*70}")
        print(f"  DEPTH SOURCE: {depth_subdir}")
        print(f"{'─'*70}")

        files = discover_files(
            args.coco_root, args.semantic_subdir, depth_subdir, args.max_images)
        print(f"  Found {len(files)} images")

        if len(files) == 0:
            print(f"  SKIPPED (no matching files)")
            continue

        results = []

        # CC-only baseline
        if depth_subdir == args.depth_subdirs[0]:
            print(f"\n  Running CC-only baseline...")
            t0 = time.time()
            m = run_config(files, gt_loader, grad_threshold=None, label="CC-only")
            dt = time.time() - t0
            results.append(("CC-only", "—", m))
            print(f"    PQ={m['PQ']:5.1f}  PQ_st={m['PQ_stuff']:5.1f}  "
                  f"PQ_th={m['PQ_things']:5.1f}  inst/img={m['avg_instances']:4.1f}  ({dt:.0f}s)")

        # Depth-guided configs
        for i, (gt, ma) in enumerate(configs):
            label = f"{depth_subdir} gt={gt:.2f} ma={ma}"
            print(f"\n  [{i+1}/{len(configs)}] gt={gt:.2f} ma={ma}...", end="", flush=True)
            t0 = time.time()
            m = run_config(files, gt_loader, grad_threshold=gt, min_area=ma, label=label)
            dt = time.time() - t0
            results.append((gt, ma, m))
            print(f"  PQ={m['PQ']:5.1f}  PQ_st={m['PQ_stuff']:5.1f}  "
                  f"PQ_th={m['PQ_things']:5.1f}  edge={m['edge_density']:.4f}  "
                  f"inst/img={m['avg_instances']:4.1f}  ({dt:.0f}s)")

        # Find best
        best_gt, best_ma, best_m = max(results, key=lambda x: x[2]["PQ"])
        best_thing_gt, best_thing_ma, best_thing_m = max(
            results, key=lambda x: x[2]["PQ_things"])

        gt_str = str(best_gt) if not isinstance(best_gt, float) else f"{best_gt:.2f}"
        ma_str = str(best_ma)
        print(f"\n  Best PQ:        gt={gt_str}, ma={ma_str} → PQ={best_m['PQ']:.1f}")

        gt_str2 = str(best_thing_gt) if not isinstance(best_thing_gt, float) else f"{best_thing_gt:.2f}"
        ma_str2 = str(best_thing_ma)
        print(f"  Best PQ_things: gt={gt_str2}, ma={ma_str2} → "
              f"PQ_th={best_thing_m['PQ_things']:.1f}")

        # Per-class for best PQ config
        print(f"\n  Per-class things (best PQ: gt={gt_str}, ma={ma_str}):")
        for cls in sorted(THING_IDS):
            name = COCOSTUFF27_CLASSNAMES[cls]
            v = best_m["per_class"][name]
            if v["TP"] + v["FP"] + v["FN"] > 0:
                print(f"    {name:15s}: PQ={v['PQ']:5.1f}  SQ={v['SQ']:5.1f}  "
                      f"RQ={v['RQ']:5.1f}  (TP={v['TP']} FP={v['FP']} FN={v['FN']})")

        all_results[depth_subdir] = {
            "n_images": len(files),
            "best_pq": {"gt": best_gt, "ma": best_ma,
                        **{k: v for k, v in best_m.items() if k != "per_class"}},
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

    # ─── Cross-Model Comparison ───
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"CROSS-MODEL COMPARISON (COCO-Stuff-27)")
        print(f"{'='*70}")
        print(f"\n  {'Depth Model':25s} | {'Best PQ':>8s} | {'PQ_stuff':>8s} | "
              f"{'PQ_things':>9s} | {'Opt tau':>8s} | {'Opt MA':>7s}")
        print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}")

        for ds in args.depth_subdirs:
            if ds not in all_results:
                continue
            r = all_results[ds]
            b = r["best_pq"]
            gt_str = str(b['gt']) if not isinstance(b['gt'], float) else f"{b['gt']:.2f}"
            ma_str = str(b['ma'])
            print(f"  {ds:25s} | {b['PQ']:8.1f} | {b['PQ_stuff']:8.1f} | "
                  f"{b['PQ_things']:9.1f} | {gt_str:>8s} | {ma_str:>7s}")

    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save
    out_path = Path(args.coco_root) / f"sweep_coco_depth_comparison.json"
    save_data = {"config": vars(args), "results": all_results}
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
