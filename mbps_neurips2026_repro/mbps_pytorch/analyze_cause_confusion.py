#!/usr/bin/env python3
"""
Analyze WHERE the zero-IoU class pixels end up in CAUSE-TR predictions.

Builds a full 19×19 confusion matrix (pred rows × GT cols) and shows:
1. For each GT class, what does CAUSE predict those pixels as?
2. Feature-space analysis: are the missing classes distinguishable in DINOv2 features?
3. Per-class pixel counts and coverage statistics.

Usage:
    python mbps_pytorch/analyze_cause_confusion.py \
        --cityscapes_root /path/to/cityscapes \
        --semantic_subdir pseudo_semantic_cause \
        --split val
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ─── Cityscapes Constants ───

_CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}

_CS_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

NUM_CLASSES = 19
IGNORE_LABEL = 255

# CAUSE 27-class → 19 trainID mapping
_CAUSE27_TO_TRAINID = np.full(256, IGNORE_LABEL, dtype=np.uint8)
for _c27, _t19 in {
    0: 0, 1: 1, 2: 255, 3: 255, 4: 2, 5: 3, 6: 4,
    7: 255, 8: 255, 9: 255, 10: 5, 11: 5, 12: 6, 13: 7,
    14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14,
    21: 15, 22: 13, 23: 14, 24: 16, 25: 17, 26: 18,
}.items():
    _CAUSE27_TO_TRAINID[_c27] = _t19

# Zero-IoU classes we want to analyze
ZERO_IOU_CLASSES = [4, 5, 6, 7, 12, 16, 17]  # fence, pole, t_light, t_sign, rider, train, motorcycle
ZERO_IOU_NAMES = ["fence", "pole", "traffic light", "traffic sign", "rider", "train", "motorcycle"]


def _remap_to_trainids(gt):
    remapped = np.full_like(gt, IGNORE_LABEL, dtype=np.uint8)
    for raw_id, train_id in _CS_ID_TO_TRAIN.items():
        remapped[gt == raw_id] = train_id
    return remapped


def build_confusion_matrix(cityscapes_root, semantic_subdir, split, cause27=True):
    """Build full 19×19 confusion matrix: conf[pred_class, gt_class] = pixel count."""

    img_dir = os.path.join(cityscapes_root, "leftImg8bit", split)
    gt_dir = os.path.join(cityscapes_root, "gtFine", split)
    sem_dir = os.path.join(cityscapes_root, semantic_subdir, split)

    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    gt_total = np.zeros(NUM_CLASSES, dtype=np.int64)

    count = 0
    for city in sorted(os.listdir(img_dir)):
        city_img_dir = os.path.join(img_dir, city)
        if not os.path.isdir(city_img_dir):
            continue
        for fname in sorted(os.listdir(city_img_dir)):
            if not fname.endswith("_leftImg8bit.png"):
                continue
            stem = fname.replace("_leftImg8bit.png", "")

            # Load prediction
            pred_path = os.path.join(sem_dir, city, f"{stem}.png")
            if not os.path.exists(pred_path):
                continue
            pred = np.array(Image.open(pred_path))
            if cause27:
                pred = _CAUSE27_TO_TRAINID[pred]

            # Load GT
            gt_path = os.path.join(gt_dir, city, f"{stem}_gtFine_labelIds.png")
            gt_raw = np.array(Image.open(gt_path))
            gt = _remap_to_trainids(gt_raw)

            # Resize pred to gt size if needed
            if pred.shape != gt.shape:
                pred = np.array(
                    Image.fromarray(pred).resize(
                        (gt.shape[1], gt.shape[0]), Image.NEAREST
                    )
                )

            # Accumulate
            valid = (gt != IGNORE_LABEL) & (pred != IGNORE_LABEL)
            p, g = pred[valid], gt[valid]
            mask = (p < NUM_CLASSES) & (g < NUM_CLASSES)
            np.add.at(conf, (p[mask], g[mask]), 1)

            # GT pixel counts
            for c in range(NUM_CLASSES):
                gt_total[c] += (gt == c).sum()

            count += 1

    return conf, gt_total, count


def analyze_confusion(conf, gt_total):
    """Analyze the confusion matrix, focusing on zero-IoU classes."""

    print("\n" + "=" * 80)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 80)

    # 1. Overall per-class IoU
    print("\n--- Per-Class IoU ---")
    ious = []
    for c in range(NUM_CLASSES):
        tp = conf[c, c]
        fp = conf[c, :].sum() - tp
        fn = conf[:, c].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0.0
        ious.append(iou)
        bar = "█" * int(iou * 40)
        print(f"  {_CS_CLASS_NAMES[c]:15s}: {iou*100:5.1f}%  {bar}  (GT: {gt_total[c]:>10,} px)")
    print(f"\n  mIoU: {np.mean(ious)*100:.2f}%")

    # 2. Focus on zero-IoU classes: WHERE do their GT pixels go?
    print("\n" + "=" * 80)
    print("WHERE DO ZERO-IoU CLASS PIXELS GO?")
    print("(For each GT class, showing what CAUSE predicts those pixels as)")
    print("=" * 80)

    for gt_cls, gt_name in zip(ZERO_IOU_CLASSES, ZERO_IOU_NAMES):
        col = conf[:, gt_cls]  # all predictions for this GT class
        total = col.sum()
        if total == 0:
            print(f"\n  {gt_name}: No valid GT pixels found")
            continue

        print(f"\n  GT class: {gt_name} ({total:,} pixels)")
        # Sort by count descending
        sorted_idx = np.argsort(-col)
        for rank, pred_cls in enumerate(sorted_idx[:5]):
            if col[pred_cls] == 0:
                break
            pct = col[pred_cls] / total * 100
            print(f"    → predicted as {_CS_CLASS_NAMES[pred_cls]:15s}: {col[pred_cls]:>10,} ({pct:5.1f}%)")

    # 3. Full confusion matrix for zero-IoU classes
    print("\n" + "=" * 80)
    print("CONFUSION DETAILS: What CAUSE predicts for each zero-IoU GT class")
    print("=" * 80)

    # Header
    header = "GT class \\ Pred →  "
    for name in _CS_CLASS_NAMES:
        header += f"{name[:6]:>7s}"
    print(header)
    print("-" * len(header))

    for gt_cls, gt_name in zip(ZERO_IOU_CLASSES, ZERO_IOU_NAMES):
        col = conf[:, gt_cls]
        total = col.sum()
        if total == 0:
            continue
        row = f"  {gt_name:15s}  "
        for pred_cls in range(NUM_CLASSES):
            pct = col[pred_cls] / total * 100 if total > 0 else 0
            if pct >= 1.0:
                row += f"{pct:6.1f}%"
            elif pct > 0:
                row += f" {pct:4.1f}%"
            else:
                row += "      ."
        print(row)

    # 4. Adjacency analysis: which classes are spatially near the zero-IoU classes?
    print("\n" + "=" * 80)
    print("MISCLASSIFICATION PATTERNS SUMMARY")
    print("=" * 80)

    patterns = {}
    for gt_cls, gt_name in zip(ZERO_IOU_CLASSES, ZERO_IOU_NAMES):
        col = conf[:, gt_cls]
        total = col.sum()
        if total == 0:
            patterns[gt_name] = "NO GT PIXELS"
            continue
        sorted_idx = np.argsort(-col)
        top_preds = []
        for pred_cls in sorted_idx[:3]:
            if col[pred_cls] > 0:
                pct = col[pred_cls] / total * 100
                top_preds.append(f"{_CS_CLASS_NAMES[pred_cls]}({pct:.0f}%)")
        patterns[gt_name] = " + ".join(top_preds)

    for name, pattern in patterns.items():
        print(f"  {name:15s} → {pattern}")

    # 5. Potential recovery analysis
    print("\n" + "=" * 80)
    print("RECOVERY POTENTIAL")
    print("  (if we could perfectly split the absorbing class)")
    print("=" * 80)

    for gt_cls, gt_name in zip(ZERO_IOU_CLASSES, ZERO_IOU_NAMES):
        col = conf[:, gt_cls]
        total = col.sum()
        if total == 0:
            continue
        # What's the dominant absorber?
        absorber = np.argmax(col)
        absorber_pixels = col[absorber]
        absorber_total = conf[absorber, :].sum()  # total pixels predicted as absorber
        # If we could split absorber into "real absorber" + "recovered class"
        # how much would it help?
        recovery_pct = absorber_pixels / absorber_total * 100 if absorber_total > 0 else 0
        print(f"  {gt_name:15s}: {total:>10,} GT pixels absorbed by {_CS_CLASS_NAMES[absorber]:15s} "
              f"({recovery_pct:.1f}% of {_CS_CLASS_NAMES[absorber]}'s predictions)")

    return ious, patterns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--semantic_subdir", type=str, default="pseudo_semantic_cause")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--cause27", action="store_true", default=True,
                        help="Remap CAUSE 27-class → 19 trainIDs (default: True)")
    parser.add_argument("--no_cause27", dest="cause27", action="store_false")
    args = parser.parse_args()

    print(f"Building confusion matrix for {args.split} split...")
    print(f"  Semantic predictions: {args.semantic_subdir}")
    print(f"  CAUSE 27→19 remap: {args.cause27}")

    conf, gt_total, n_images = build_confusion_matrix(
        args.cityscapes_root, args.semantic_subdir, args.split, args.cause27
    )
    print(f"  Processed {n_images} images")

    ious, patterns = analyze_confusion(conf, gt_total)

    # Save confusion matrix
    out_dir = os.path.join(args.cityscapes_root, args.semantic_subdir)
    np.save(os.path.join(out_dir, f"confusion_matrix_{args.split}.npy"), conf)

    # Save analysis as JSON
    analysis = {
        "num_images": n_images,
        "per_class_iou": {_CS_CLASS_NAMES[c]: float(ious[c] * 100) for c in range(NUM_CLASSES)},
        "miou": float(np.mean(ious) * 100),
        "gt_pixel_counts": {_CS_CLASS_NAMES[c]: int(gt_total[c]) for c in range(NUM_CLASSES)},
        "misclassification_patterns": patterns,
        "confusion_matrix_path": os.path.join(out_dir, f"confusion_matrix_{args.split}.npy"),
    }
    analysis_path = os.path.join(out_dir, f"confusion_analysis_{args.split}.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
