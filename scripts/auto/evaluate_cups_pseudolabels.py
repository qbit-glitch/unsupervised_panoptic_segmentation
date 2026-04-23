#!/usr/bin/env python3
"""Evaluate CUPS-format pseudo-labels (semantic.png + instance.png) against Cityscapes GT.

Handles flat directory structure and raw cluster IDs via Hungarian matching.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}
THING_IDS = set(range(11, 19))
STUFF_IDS = set(range(0, 11))
IGNORE = 255


def remap_gt(gt):
    out = np.full_like(gt, IGNORE)
    for lid, tid in CS_ID_TO_TRAIN.items():
        out[gt == lid] = tid
    return out


def resize_nearest(arr, h, w):
    return np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST))


def hungarian_match(pred, gt, num_classes=19):
    """Global Hungarian matching for cluster IDs to trainIDs."""
    mask = gt != IGNORE
    pred_valid = pred[mask]
    gt_valid = gt[mask]

    max_pred = int(pred_valid.max()) + 1
    cost = np.zeros((max_pred, num_classes))
    for p in range(max_pred):
        for g in range(num_classes):
            cost[p, g] = -((pred_valid == p) & (gt_valid == g)).sum()

    rows, cols = linear_sum_assignment(cost)
    mapping = {int(r): int(c) for r, c in zip(rows, cols)}
    return mapping


def compute_iou(pred, gt, num_classes=19):
    ious = []
    for c in range(num_classes):
        inter = ((pred == c) & (gt == c)).sum()
        union = ((pred == c) | (gt == c)).sum()
        ious.append(inter / (union + 1e-10))
    return np.mean(ious)


def compute_pq(pred_sem, pred_inst, gt_sem, gt_inst):
    """Compute PQ for panoptic labels.
    pred_inst: instance IDs (0 = no instance)
    gt_inst: instance IDs from Cityscapes (1000*class + instance_num)
    """
    # Extract GT instance masks and class IDs
    gt_inst_ids = np.unique(gt_inst)
    gt_masks = []
    gt_classes = []
    for uid in gt_inst_ids:
        if uid < 1000:
            continue
        cls = uid // 1000
        if cls not in CS_ID_TO_TRAIN:
            continue
        tid = CS_ID_TO_TRAIN[cls]
        if tid not in THING_IDS:
            continue
        mask = gt_inst == uid
        if mask.sum() < 10:
            continue
        gt_masks.append(mask)
        gt_classes.append(tid)

    # Extract predicted instance masks
    pred_inst_ids = np.unique(pred_inst)
    pred_masks = []
    pred_classes = []
    for uid in pred_inst_ids:
        if uid == 0:
            continue
        mask = pred_inst == uid
        # Class from semantic label
        cls = int(np.bincount(pred_sem[mask].flatten()).argmax())
        if mask.sum() < 10:
            continue
        pred_masks.append(mask)
        pred_classes.append(cls)

    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return 0.0, 0.0, 0.0

    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            inter = (pm & gm).sum()
            union = (pm | gm).sum()
            iou_matrix[i, j] = inter / (union + 1e-10)

    # Hungarian matching
    rows, cols = linear_sum_assignment(-iou_matrix)

    tp = 0.0
    sq_sum = 0.0
    for i, j in zip(rows, cols):
        if pred_classes[i] == gt_classes[j] and iou_matrix[i, j] > 0.5:
            tp += 1
            sq_sum += iou_matrix[i, j]

    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp

    pq = sq_sum / (tp + 0.5 * fp + 0.5 * fn + 1e-10)
    sq = sq_sum / (tp + 1e-10) if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-10)

    return pq, sq, rq


def evaluate_dir(pseudo_dir: Path, gt_root: Path, max_images: int = None):
    gt_label_dir = gt_root / "gtFine" / "val"
    gt_inst_dir = gt_root / "gtFine" / "val"

    gt_files = sorted(gt_label_dir.rglob("*_gtFine_labelIds.png"))
    if max_images:
        gt_files = gt_files[:max_images]

    all_pred_sem = []
    all_gt_sem = []
    all_pred_inst = []
    all_gt_inst = []

    for gt_path in tqdm(gt_files, desc="Loading"):
        rel = gt_path.relative_to(gt_label_dir)
        base = str(rel).replace("_gtFine_labelIds.png", "")
        stem = Path(base).name  # e.g., aachen_000000_000019

        sem_path = pseudo_dir / (stem + "_leftImg8bit_semantic.png")
        inst_path = pseudo_dir / (stem + "_leftImg8bit_instance.png")

        if not sem_path.exists() or not inst_path.exists():
            continue

        gt_sem = np.array(Image.open(gt_path))
        gt_inst_path = gt_inst_dir / (base + "_gtFine_instanceIds.png")
        gt_inst = np.array(Image.open(gt_inst_path)) if gt_inst_path.exists() else np.zeros_like(gt_sem)

        pred_sem = np.array(Image.open(sem_path))
        pred_inst = np.array(Image.open(inst_path))

        # Resize to common size
        h, w = gt_sem.shape
        if pred_sem.shape != (h, w):
            pred_sem = resize_nearest(pred_sem, h, w)
            pred_inst = resize_nearest(pred_inst, h, w)

        gt_sem = remap_gt(gt_sem)

        all_pred_sem.append(pred_sem)
        all_gt_sem.append(gt_sem)
        all_pred_inst.append(pred_inst)
        all_gt_inst.append(gt_inst)

    if len(all_pred_sem) == 0:
        return {"error": "No valid image pairs found"}

    # Stack for global Hungarian
    all_pred_sem_cat = np.concatenate([p.flatten() for p in all_pred_sem])
    all_gt_sem_cat = np.concatenate([g.flatten() for g in all_gt_sem])

    mapping = hungarian_match(all_pred_sem_cat, all_gt_sem_cat)

    # Apply mapping
    total_iou = 0.0
    total_pq = 0.0
    total_sq = 0.0
    total_rq = 0.0
    n = len(all_pred_sem)

    for pred_sem, gt_sem, pred_inst, gt_inst in zip(all_pred_sem, all_gt_sem, all_pred_inst, all_gt_inst):
        mapped_sem = np.vectorize(lambda x: mapping.get(int(x), IGNORE))(pred_sem).astype(np.uint8)
        total_iou += compute_iou(mapped_sem, gt_sem)
        pq, sq, rq = compute_pq(mapped_sem, pred_inst, gt_sem, gt_inst)
        total_pq += pq
        total_sq += sq
        total_rq += rq

    return {
        "num_images": n,
        "mIoU": total_iou / n,
        "PQ": total_pq / n,
        "SQ": total_sq / n,
        "RQ": total_rq / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo_dir", type=Path, required=True)
    parser.add_argument("--gt_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    results = evaluate_dir(args.pseudo_dir, args.gt_root, args.max_images)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
