#!/usr/bin/env python3
"""Evaluate semantic pseudo-labels with comprehensive metrics.

Computes:
  - Semantic: mIoU, pixel accuracy, per-class IoU
  - Panoptic: PQ, SQ, RQ (overall, stuff-only, things-only)

For panoptic evaluation, semantic predictions are converted to panoptic format
by treating connected components as separate segments for thing classes, while
stuff classes are treated as single segments per class.

Usage:
    python mbps_pytorch/evaluate_semantic_pseudolabels.py \
        --pred_dir /data/cityscapes/pseudo_semantic_dinov3/train \
        --gt_dir /data/cityscapes/gtFine/train \
        --output /data/cityscapes/pseudo_semantic_dinov3/eval_results.json
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Cityscapes Constants
# --------------------------------------------------------------------------- #
IGNORE_LABEL = 255
NUM_CLASSES = 19

CS_TRAINID_TO_NAME = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}

# Stuff classes: amorphous regions (no instance distinction)
STUFF_TRAINIDS = {0, 1, 2, 3, 4, 8, 9, 10}
# Thing classes: countable objects (instances matter)
THING_TRAINIDS = {5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18}

# Cityscapes labelId -> trainId mapping
CITYSCAPES_ID_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}

# CAUSE 27-class index → 19-class trainId mapping
# CAUSE index = Cityscapes labelID - 7 (first_nonvoid offset)
CAUSE27_TO_TRAINID_19 = {
    0: 0, 1: 1, 2: 255, 3: 255, 4: 2, 5: 3, 6: 4,
    7: 255, 8: 255, 9: 255, 10: 5, 11: 5, 12: 6, 13: 7,
    14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14,
    21: 15, 22: 13, 23: 14, 24: 16, 25: 17, 26: 18,
}


def remap_gt_to_trainid(gt: np.ndarray) -> np.ndarray:
    """Remap Cityscapes labelId GT to trainId format."""
    out = np.full_like(gt, IGNORE_LABEL)
    for label_id, train_id in CITYSCAPES_ID_TO_TRAINID.items():
        out[gt == label_id] = train_id
    return out


def remap_cause27_to_trainid(pred: np.ndarray) -> np.ndarray:
    """Remap CAUSE 27-class predictions to 19-class trainId format."""
    out = np.full_like(pred, IGNORE_LABEL)
    for cause_id, train_id in CAUSE27_TO_TRAINID_19.items():
        out[pred == cause_id] = train_id
    return out


def semantic_to_segments(semantic: np.ndarray):
    """Convert semantic map to list of (class_id, mask) segments.

    Stuff: one segment per class. Things: connected components.
    """
    segments = []
    for class_id in range(NUM_CLASSES):
        class_mask = semantic == class_id
        if not class_mask.any():
            continue

        if class_id in THING_TRAINIDS:
            labeled, n = ndimage.label(class_mask)
            for i in range(1, n + 1):
                segments.append((class_id, labeled == i))
        else:
            segments.append((class_id, class_mask))
    return segments


def compute_pq_for_image(pred: np.ndarray, gt: np.ndarray):
    """Compute per-class TP/FP/FN/IoU for one image pair.

    Returns dict: class_id -> {"tp": int, "fp": int, "fn": int, "iou_sum": float}
    """
    valid = gt != IGNORE_LABEL

    pred_segs = semantic_to_segments(pred)
    gt_segs = semantic_to_segments(gt)

    # Group by class
    pred_by_class = defaultdict(list)
    gt_by_class = defaultdict(list)
    for cid, mask in pred_segs:
        pred_by_class[cid].append(mask)
    for cid, mask in gt_segs:
        gt_by_class[cid].append(mask)

    results = {}
    all_classes = set(pred_by_class.keys()) | set(gt_by_class.keys())

    for cid in all_classes:
        p_masks = pred_by_class.get(cid, [])
        g_masks = gt_by_class.get(cid, [])

        matched_p = set()
        matched_g = set()
        iou_sum = 0.0

        # Match GT to pred greedily
        for gi, g_mask in enumerate(g_masks):
            g_valid = g_mask & valid
            best_iou = 0.0
            best_pi = -1

            for pi, p_mask in enumerate(p_masks):
                if pi in matched_p:
                    continue
                p_valid = p_mask & valid
                inter = (g_valid & p_valid).sum()
                union = (g_valid | p_valid).sum()
                if union == 0:
                    continue
                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
                    best_pi = pi

            if best_iou > 0.5 and best_pi >= 0:
                matched_g.add(gi)
                matched_p.add(best_pi)
                iou_sum += best_iou

        results[cid] = {
            "tp": len(matched_g),
            "fp": len(p_masks) - len(matched_p),
            "fn": len(g_masks) - len(matched_g),
            "iou_sum": iou_sum,
        }

    return results


def evaluate_from_files(pred_dir: str, gt_dir: str, remap_cause27: bool = False):
    """Evaluate pseudo-labels saved as PNGs against GT."""
    pred_path = Path(pred_dir)
    gt_path = Path(gt_dir)

    pred_files = sorted(pred_path.rglob("*.png"))
    if not pred_files:
        logger.error(f"No prediction PNGs found in {pred_dir}")
        return None

    logger.info(f"Found {len(pred_files)} prediction files")

    # Semantic accumulators
    total_intersect = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_union = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_correct = 0
    total_pixels = 0

    # Panoptic accumulators (per-class, summed over images)
    pq_tp = defaultdict(int)
    pq_fp = defaultdict(int)
    pq_fn = defaultdict(int)
    pq_iou = defaultdict(float)

    if remap_cause27:
        logger.info("Will remap CAUSE 27-class predictions → 19-class trainIDs")

    for pred_file in tqdm(pred_files, desc="Evaluating"):
        pred = np.array(Image.open(pred_file), dtype=np.uint8)

        if remap_cause27:
            pred = remap_cause27_to_trainid(pred)

        city = pred_file.parent.name
        stem = pred_file.stem

        gt_trainid_file = gt_path / city / f"{stem.replace('_leftImg8bit', '')}_gtFine_labelTrainIds.png"
        gt_labelid_file = gt_path / city / f"{stem.replace('_leftImg8bit', '')}_gtFine_labelIds.png"

        if gt_trainid_file.exists():
            gt = np.array(Image.open(gt_trainid_file), dtype=np.uint8)
        elif gt_labelid_file.exists():
            gt_raw = np.array(Image.open(gt_labelid_file), dtype=np.uint8)
            gt = remap_gt_to_trainid(gt_raw)
        else:
            logger.warning(f"No GT found for {pred_file.name}")
            continue

        # Resize pred to GT size if needed
        if pred.shape != gt.shape:
            pred = np.array(
                Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.NEAREST),
                dtype=np.uint8,
            )

        valid = gt != IGNORE_LABEL

        # --- Semantic ---
        for c in range(NUM_CLASSES):
            pred_c = pred == c
            gt_c = gt == c
            total_intersect[c] += (pred_c & gt_c & valid).sum()
            total_union[c] += ((pred_c | gt_c) & valid).sum()
        total_correct += ((pred == gt) & valid).sum()
        total_pixels += valid.sum()

        # --- Panoptic ---
        img_pq = compute_pq_for_image(pred, gt)
        for cid, vals in img_pq.items():
            pq_tp[cid] += vals["tp"]
            pq_fp[cid] += vals["fp"]
            pq_fn[cid] += vals["fn"]
            pq_iou[cid] += vals["iou_sum"]

    # ---- Aggregate semantic ----
    iou = total_intersect / (total_union + 1e-10)
    valid_classes = total_union > 0
    miou = iou[valid_classes].mean()
    pixel_acc = total_correct / (total_pixels + 1e-10)

    semantic_results = {
        "mIoU": round(float(miou) * 100, 2),
        "pixel_accuracy": round(float(pixel_acc) * 100, 2),
        "per_class_iou": {
            CS_TRAINID_TO_NAME[c]: round(float(iou[c]) * 100, 2)
            for c in range(NUM_CLASSES)
        },
    }

    # ---- Aggregate panoptic ----
    def _agg(class_ids):
        active = [c for c in class_ids if pq_tp[c] + pq_fn[c] + pq_fp[c] > 0]
        if not active:
            return {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "n_classes": 0}
        pqs, sqs, rqs = [], [], []
        for c in active:
            tp, fp, fn, iou_s = pq_tp[c], pq_fp[c], pq_fn[c], pq_iou[c]
            denom = tp + 0.5 * fp + 0.5 * fn
            pqs.append(iou_s / denom if denom > 0 else 0.0)
            sqs.append(iou_s / tp if tp > 0 else 0.0)
            rqs.append(tp / denom if denom > 0 else 0.0)
        return {
            "PQ": round(float(np.mean(pqs)) * 100, 2),
            "SQ": round(float(np.mean(sqs)) * 100, 2),
            "RQ": round(float(np.mean(rqs)) * 100, 2),
            "n_classes": len(active),
        }

    panoptic_results = {
        "all": _agg(set(range(NUM_CLASSES))),
        "stuff": _agg(STUFF_TRAINIDS),
        "things": _agg(THING_TRAINIDS),
        "per_class": {},
    }

    for c in range(NUM_CLASSES):
        name = CS_TRAINID_TO_NAME[c]
        tp, fp, fn, iou_s = pq_tp[c], pq_fp[c], pq_fn[c], pq_iou[c]
        denom = tp + 0.5 * fp + 0.5 * fn
        panoptic_results["per_class"][name] = {
            "PQ": round(iou_s / denom * 100, 2) if denom > 0 else 0.0,
            "SQ": round(iou_s / tp * 100, 2) if tp > 0 else 0.0,
            "RQ": round(tp / denom * 100, 2) if denom > 0 else 0.0,
            "TP": tp, "FP": fp, "FN": fn,
            "type": "thing" if c in THING_TRAINIDS else "stuff",
        }

    return {
        "semantic": semantic_results,
        "panoptic": panoptic_results,
        "num_images": len(pred_files),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic pseudo-labels")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--remap_cause27", action="store_true",
                        help="Remap CAUSE 27-class predictions to 19-class trainIDs before evaluation")
    args = parser.parse_args()

    results = evaluate_from_files(args.pred_dir, args.gt_dir, remap_cause27=args.remap_cause27)
    if results is None:
        return

    sem = results["semantic"]
    pan = results["panoptic"]

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"\n--- Semantic ({results['num_images']} images) ---")
    print(f"  mIoU:           {sem['mIoU']:.2f}%")
    print(f"  Pixel Accuracy: {sem['pixel_accuracy']:.2f}%")
    print(f"\n  Per-class IoU:")
    for name, val in sem["per_class_iou"].items():
        print(f"    {name:15s}: {val:.2f}%")

    print(f"\n--- Panoptic ---")
    for key in ["all", "stuff", "things"]:
        p = pan[key]
        print(f"  {key.upper():8s}  PQ={p['PQ']:5.2f}%  SQ={p['SQ']:5.2f}%  RQ={p['RQ']:5.2f}%  ({p['n_classes']} classes)")

    print(f"\n  Per-class:")
    for name, pc in pan["per_class"].items():
        print(f"    {name:15s} [{pc['type']:5s}]  PQ={pc['PQ']:6.2f}  SQ={pc['SQ']:6.2f}  RQ={pc['RQ']:6.2f}  TP={pc['TP']:5d}  FP={pc['FP']:5d}  FN={pc['FN']:5d}")

    print("=" * 70)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
