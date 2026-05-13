#!/usr/bin/env python3
"""Evaluate k=50 raw overcluster pseudo-labels against Cityscapes GT.

Maps cluster IDs → trainIDs via cluster_to_class from kmeans_centroids.npz,
then evaluates semantic mIoU and panoptic PQ on the train split.

Usage:
    python mbps_pytorch/evaluate_k50_pseudolabels.py \
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
        --split train
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

# ─── Cityscapes Constants ───

CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
    22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
    32: 17, 33: 18,
}
NUM_CLASSES = 19
STUFF_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}  # 11 stuff
THING_IDS = {11, 12, 13, 14, 15, 16, 17, 18}  # 8 things
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]


def remap_gt_to_trainids(gt_raw):
    """Remap Cityscapes labelIds to trainIds (0-18, 255=ignore)."""
    out = np.full_like(gt_raw, 255, dtype=np.uint8)
    for cs_id, train_id in CS_ID_TO_TRAIN.items():
        out[gt_raw == cs_id] = train_id
    return out


def resize_nearest(arr, hw):
    """Resize a label map using nearest interpolation."""
    h, w = hw
    return np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST))


def load_instance_npz(npz_path, target_hw=None):
    """Load SPIdepth instance masks from NPZ format.

    Returns list of (mask_HW, score) tuples at target resolution.
    """
    data = np.load(str(npz_path))
    masks = data["masks"]
    scores = data["scores"]
    num_valid = int(data["num_valid"])
    h = int(data["h_patches"])
    w = int(data["w_patches"])
    masks = masks[:num_valid]
    scores = scores[:num_valid]

    result = []
    for i in range(num_valid):
        m = masks[i].reshape(h, w)
        if target_hw is not None and (h, w) != target_hw:
            m = resize_nearest(m.astype(np.uint8), target_hw).astype(bool)
        result.append((m, float(scores[i])))
    return result


def discover_pairs(cityscapes_root, split, semantic_subdir, instance_subdir=None):
    """Find matching (semantic_pred, gt_label, gt_instance, pred_instance) file paths."""
    root = Path(cityscapes_root)
    sem_dir = root / semantic_subdir / split
    gt_dir = root / "gtFine" / split
    inst_dir = root / instance_subdir / split if instance_subdir else None

    gt_label_files = sorted(gt_dir.rglob("*_gtFine_labelIds.png"))
    print(f"Found {len(gt_label_files)} GT label files in {gt_dir}")

    pairs = []
    inst_found = 0
    for gt_label_path in gt_label_files:
        rel = gt_label_path.relative_to(gt_dir)
        base = str(rel).replace("_gtFine_labelIds.png", "")

        gt_inst_path = gt_dir / (base + "_gtFine_instanceIds.png")
        if not gt_inst_path.exists():
            gt_inst_path = None

        # Predicted semantic
        sem_path = sem_dir / (base + "_leftImg8bit.png")
        if not sem_path.exists():
            sem_path = sem_dir / (base + ".png")
        if not sem_path.exists():
            continue

        # Predicted instance NPZ (optional)
        pred_inst_path = None
        if inst_dir is not None:
            # Try stem_leftImg8bit.npz then stem.npz
            city = base.split("/")[0] if "/" in base else base.rsplit("_", 2)[0]
            stem = base.split("/")[-1] if "/" in base else base
            pred_inst_path = inst_dir / city / f"{stem}_leftImg8bit.npz"
            if not pred_inst_path.exists():
                pred_inst_path = inst_dir / city / f"{stem}.npz"
            if not pred_inst_path.exists():
                pred_inst_path = None
            else:
                inst_found += 1

        pairs.append((sem_path, gt_label_path, gt_inst_path, pred_inst_path))

    print(f"Discovered {len(pairs)} evaluation pairs")
    if inst_dir is not None:
        print(f"  Instance NPZ files: {inst_found}/{len(pairs)}")
    return pairs


def evaluate_semantic(pairs, eval_hw, k50_to_trainid):
    """Compute semantic mIoU by remapping k=50 cluster IDs to trainIDs."""
    H, W = eval_hw
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for sem_path, gt_label_path, *_ in tqdm(pairs, desc="Semantic eval"):
        # Load and remap prediction: k50 → trainID
        pred_k50 = np.array(Image.open(sem_path))
        pred = k50_to_trainid[pred_k50]
        if pred.shape != (H, W):
            pred = resize_nearest(pred, eval_hw)

        # Load and remap GT
        gt_raw = np.array(Image.open(gt_label_path))
        gt = remap_gt_to_trainids(gt_raw)
        if gt.shape != (H, W):
            gt = resize_nearest(gt, eval_hw)

        # Accumulate confusion matrix (ignore 255)
        valid = (gt < NUM_CLASSES) & (pred < NUM_CLASSES)
        if valid.sum() == 0:
            continue
        np.add.at(confusion, (gt[valid], pred[valid]), 1)

    # Compute per-class IoU
    per_class_iou = {}
    ious = []
    for c in range(NUM_CLASSES):
        tp = confusion[c, c]
        fn = confusion[c, :].sum() - tp
        fp = confusion[:, c].sum() - tp
        denom = tp + fn + fp
        iou = tp / denom if denom > 0 else 0.0
        per_class_iou[CLASS_NAMES[c]] = round(iou * 100, 2)
        if confusion[c, :].sum() > 0:  # only count classes present in GT
            ious.append(iou)

    miou = float(np.mean(ious)) * 100 if ious else 0.0
    pixel_acc = float(np.diag(confusion).sum() / (confusion.sum() + 1e-8)) * 100

    print(f"\n{'='*60}")
    print(f"SEMANTIC EVALUATION (k=50 → 19 trainIDs)")
    print(f"{'='*60}")
    print(f"  Per-class IoU:")
    for name in sorted(per_class_iou, key=per_class_iou.get, reverse=True):
        kind = "S" if CLASS_NAMES.index(name) in STUFF_IDS else "T"
        print(f"    [{kind}] {name:15s}: {per_class_iou[name]:5.1f}%")
    print(f"\n  mIoU:           {miou:.2f}%")
    print(f"  Pixel Accuracy: {pixel_acc:.2f}%")
    print(f"  Classes with GT: {len(ious)}/19")

    return {"miou": round(miou, 2), "pixel_accuracy": round(pixel_acc, 2),
            "per_class": per_class_iou}


def evaluate_panoptic(pairs, eval_hw, k50_to_trainid, thing_mode="cc"):
    """Compute PQ/SQ/RQ by remapping k=50 cluster IDs to trainIDs.

    Args:
        thing_mode: "cc" for connected components, "spidepth" for NPZ instance masks
    """
    H, W = eval_hw
    tp = np.zeros(NUM_CLASSES)
    fp = np.zeros(NUM_CLASSES)
    fn = np.zeros(NUM_CLASSES)
    iou_sum = np.zeros(NUM_CLASSES)
    num_evaluated = 0
    inst_used = 0

    mode_label = "CC instances" if thing_mode == "cc" else "SPIdepth instances"

    for sem_path, gt_label_path, gt_inst_path, pred_inst_path in tqdm(pairs, desc="Panoptic eval"):
        if gt_inst_path is None:
            continue

        # Load and remap prediction
        pred_k50 = np.array(Image.open(sem_path))
        pred_sem = k50_to_trainid[pred_k50]
        if pred_sem.shape != (H, W):
            pred_sem = resize_nearest(pred_sem, eval_hw)

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

        # ── Build predicted panoptic map ──
        pred_pan = np.zeros((H, W), dtype=np.int32)
        pred_segments = {}
        next_id = 1

        # Stuff segments from semantic map
        for cls in STUFF_IDS:
            mask = pred_sem == cls
            if mask.sum() < 64:
                continue
            pred_pan[mask] = next_id
            pred_segments[next_id] = cls
            next_id += 1

        # Thing segments — depends on mode
        if thing_mode == "spidepth" and pred_inst_path is not None:
            # Use SPIdepth instance masks; assign class from semantic map
            inst_masks = load_instance_npz(pred_inst_path, target_hw=(H, W))
            # Sort by score descending
            inst_masks.sort(key=lambda x: -x[1])
            for mask, score in inst_masks:
                if mask.sum() < 10:
                    continue
                # Majority semantic class
                sem_vals = pred_sem[mask]
                sem_vals = sem_vals[sem_vals < NUM_CLASSES]
                if len(sem_vals) == 0:
                    continue
                majority_cls = int(np.bincount(sem_vals, minlength=NUM_CLASSES).argmax())
                if majority_cls not in THING_IDS:
                    continue
                # Don't overwrite higher-confidence instances
                new_pixels = mask & (pred_pan == 0)
                if new_pixels.sum() < 10:
                    continue
                pred_pan[new_pixels] = next_id
                pred_segments[next_id] = majority_cls
                next_id += 1
            inst_used += 1
        else:
            # Fallback: CC instances from semantic map
            for cls in THING_IDS:
                cls_mask = pred_sem == cls
                if cls_mask.sum() < 10:
                    continue
                labeled, n_components = ndimage.label(cls_mask)
                for comp_id in range(1, n_components + 1):
                    comp_mask = labeled == comp_id
                    if comp_mask.sum() < 10:
                        continue
                    pred_pan[comp_mask] = next_id
                    pred_segments[next_id] = cls
                    next_id += 1

        # ── Build GT panoptic map ──
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

        # ── Match segments per category ──
        matched_pred = set()
        matched_gt = set()

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
                    matched_gt.add(gt_id)
                else:
                    fn[cat] += 1

            for pred_id in pred_segs:
                if pred_id not in matched_pred:
                    fp[cat] += 1

        num_evaluated += 1

    # ── Compute metrics ──
    per_class = {}
    all_pq, stuff_pq, thing_pq = [], [], []

    print(f"\n{'='*60}")
    print(f"PANOPTIC EVALUATION (k=50 → 19 trainIDs, {mode_label})")
    print(f"{'='*60}")
    if thing_mode == "spidepth":
        print(f"  SPIdepth instances used: {inst_used}/{num_evaluated}")
    print(f"  Per-class PQ:")

    for c in range(NUM_CLASSES):
        t, f_p, f_n = tp[c], fp[c], fn[c]
        iou_s = iou_sum[c]
        if t + f_p + f_n > 0:
            sq = iou_s / (t + 1e-8)
            rq = t / (t + 0.5 * f_p + 0.5 * f_n)
            pq = sq * rq
        else:
            sq = rq = pq = 0.0

        name = CLASS_NAMES[c]
        kind = "S" if c in STUFF_IDS else "T"
        per_class[name] = {
            "PQ": round(pq * 100, 2), "SQ": round(sq * 100, 2),
            "RQ": round(rq * 100, 2), "TP": int(t), "FP": int(f_p), "FN": int(f_n),
        }
        if t + f_p + f_n > 0:
            all_pq.append(pq)
            if c in STUFF_IDS:
                stuff_pq.append(pq)
            else:
                thing_pq.append(pq)

    for name, v in sorted(per_class.items(), key=lambda x: x[1]["PQ"], reverse=True):
        kind = "S" if CLASS_NAMES.index(name) in STUFF_IDS else "T"
        if v["TP"] + v["FP"] + v["FN"] > 0:
            print(f"    [{kind}] {name:15s}: PQ={v['PQ']:5.1f}  SQ={v['SQ']:5.1f}  "
                  f"RQ={v['RQ']:5.1f}  (TP={v['TP']} FP={v['FP']} FN={v['FN']})")

    overall_pq = float(np.mean(all_pq)) * 100 if all_pq else 0.0
    overall_stuff_pq = float(np.mean(stuff_pq)) * 100 if stuff_pq else 0.0
    overall_thing_pq = float(np.mean(thing_pq)) * 100 if thing_pq else 0.0
    overall_sq = float(np.sum(iou_sum) / (np.sum(tp) + 1e-8)) * 100
    overall_rq = float(np.sum(tp) / (np.sum(tp) + 0.5 * np.sum(fp) + 0.5 * np.sum(fn) + 1e-8)) * 100

    print(f"\n  ┌────────────────────────────────────────────┐")
    print(f"  │  PQ (all):    {overall_pq:5.1f}  (19 classes)          │")
    print(f"  │  PQ (stuff):  {overall_stuff_pq:5.1f}  (11 classes)          │")
    print(f"  │  PQ (things): {overall_thing_pq:5.1f}  ( 8 classes)          │")
    print(f"  │  SQ:          {overall_sq:5.1f}                        │")
    print(f"  │  RQ:          {overall_rq:5.1f}                        │")
    print(f"  └────────────────────────────────────────────┘")
    print(f"  Images evaluated: {num_evaluated}")

    return {
        "PQ": round(overall_pq, 2), "PQ_stuff": round(overall_stuff_pq, 2),
        "PQ_things": round(overall_thing_pq, 2), "SQ": round(overall_sq, 2),
        "RQ": round(overall_rq, 2), "per_class": per_class,
        "num_images": num_evaluated,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate k=50 raw overcluster pseudo-labels")
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--eval_size", type=int, nargs=2, default=[512, 1024],
                        help="(H, W) evaluation resolution")
    parser.add_argument("--centroids_path", type=str, default=None,
                        help="Path to kmeans_centroids.npz (default: auto-detect)")
    parser.add_argument("--semantic_subdir", type=str, default="pseudo_semantic_raw_k50")
    parser.add_argument("--instance_subdir", type=str, default=None,
                        help="Instance NPZ subdir (e.g. sweep_instances/gt0.10_ma500). "
                             "If provided, uses SPIdepth instances for things.")
    parser.add_argument("--thing_mode", type=str, default="cc", choices=["cc", "spidepth"],
                        help="How to create thing instances: 'cc' (connected components) "
                             "or 'spidepth' (NPZ instance masks)")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--skip_semantic", action="store_true")
    parser.add_argument("--skip_panoptic", action="store_true")
    args = parser.parse_args()

    if args.thing_mode == "spidepth" and args.instance_subdir is None:
        parser.error("--instance_subdir is required when --thing_mode=spidepth")

    cs_root = Path(args.cityscapes_root)
    eval_hw = tuple(args.eval_size)

    # Load centroids and build cluster→trainID LUT
    centroids_path = args.centroids_path or str(
        cs_root / args.semantic_subdir / "kmeans_centroids.npz"
    )
    data = np.load(centroids_path)
    cluster_to_class = data["cluster_to_class"]  # (k,) array → trainIDs 0-18
    k = len(cluster_to_class)

    # Build LUT: cluster_id → trainID (255 for unmapped)
    k50_to_trainid = np.full(256, 255, dtype=np.uint8)
    for cluster_id, trainid in enumerate(cluster_to_class):
        k50_to_trainid[cluster_id] = int(trainid)

    # Print mapping summary
    trainid_to_clusters = defaultdict(list)
    for cid, tid in enumerate(cluster_to_class):
        trainid_to_clusters[int(tid)].append(cid)

    print(f"\n{'='*60}")
    print(f"k={k} Raw Overcluster Pseudo-Label Evaluation")
    print(f"{'='*60}")
    print(f"  Split: {args.split}")
    print(f"  Eval resolution: {eval_hw[0]}x{eval_hw[1]}")
    print(f"  Centroids: {centroids_path}")
    print(f"\n  Cluster→TrainID mapping ({k} clusters → 19 classes):")
    for tid in sorted(trainid_to_clusters):
        clusters = trainid_to_clusters[tid]
        name = CLASS_NAMES[tid]
        kind = "S" if tid in STUFF_IDS else "T"
        print(f"    [{kind}] {name:15s} (tid={tid:2d}): {len(clusters)} clusters → {clusters}")

    # Discover file pairs
    pairs = discover_pairs(cs_root, args.split, args.semantic_subdir, args.instance_subdir)
    if args.max_images:
        pairs = pairs[:args.max_images]
        print(f"  Limited to {args.max_images} images")

    print(f"  Thing mode: {args.thing_mode}")

    t0 = time.time()
    results = {"split": args.split, "eval_resolution": list(eval_hw), "k": k,
               "thing_mode": args.thing_mode}

    # Semantic evaluation
    if not args.skip_semantic:
        results["semantic"] = evaluate_semantic(pairs, eval_hw, k50_to_trainid)

    # Panoptic evaluation
    if not args.skip_panoptic:
        results["panoptic"] = evaluate_panoptic(pairs, eval_hw, k50_to_trainid,
                                                thing_mode=args.thing_mode)

    elapsed = time.time() - t0

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY — k={k} Pseudo-Label Quality ({args.split})")
    print(f"{'='*60}")
    if "semantic" in results:
        print(f"  Semantic mIoU:     {results['semantic']['miou']:.1f}%")
        print(f"  Pixel Accuracy:    {results['semantic']['pixel_accuracy']:.1f}%")
    if "panoptic" in results:
        r = results["panoptic"]
        print(f"  Panoptic PQ:       {r['PQ']:.1f}%")
        print(f"  PQ Stuff:          {r['PQ_stuff']:.1f}%")
        print(f"  PQ Things:         {r['PQ_things']:.1f}%")
        print(f"  SQ:                {r['SQ']:.1f}%")
        print(f"  RQ:                {r['RQ']:.1f}%")

    # Compare with previous results
    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  Comparison with Previous Pseudo-Labels             │")
    print(f"  │  Metric    CAUSE-27   k=50-raw   Δ                  │")
    if "semantic" in results:
        m = results["semantic"]["miou"]
        print(f"  │  mIoU      40.44      {m:5.1f}     {m-40.44:+5.1f}              │")
    if "panoptic" in results:
        r = results["panoptic"]
        print(f"  │  PQ        23.10      {r['PQ']:5.1f}     {r['PQ']-23.10:+5.1f}              │")
        print(f"  │  PQ^St     31.40      {r['PQ_stuff']:5.1f}     {r['PQ_stuff']-31.40:+5.1f}              │")
        print(f"  │  PQ^Th     11.70      {r['PQ_things']:5.1f}     {r['PQ_things']-11.70:+5.1f}              │")
    print(f"  └─────────────────────────────────────────────────────┘")
    print(f"\n  Total time: {elapsed:.0f}s")

    # Save results
    import json
    output_path = str(cs_root / f"eval_k50_{args.thing_mode}_{args.split}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
