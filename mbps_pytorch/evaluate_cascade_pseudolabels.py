#!/usr/bin/env python3
"""Evaluate DINOv3 + MaskCut pseudo-labels against Cityscapes GT.

Computes semantic mIoU, instance AR/AP, and panoptic PQ/SQ/RQ using the
STANDARD Cityscapes stuff/things split for fair comparison with CUPS.

Usage:
    python mbps_pytorch/evaluate_cascade_pseudolabels.py \
        --cityscapes_root /path/to/cityscapes \
        --split val \
        --eval_size 512 1024
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
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

# Standard Cityscapes split: trainIDs 0-10 = stuff, 11-18 = things
_STUFF_IDS = set(range(0, 11))   # 11 classes
_THING_IDS = set(range(11, 19))  # 8 classes
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


def _remap_to_trainids(gt):
    remapped = np.full_like(gt, IGNORE_LABEL, dtype=np.uint8)
    for raw_id, train_id in _CS_ID_TO_TRAIN.items():
        remapped[gt == raw_id] = train_id
    return remapped


def _resize_nearest(arr, target_hw):
    """Resize 2D array using nearest neighbor via PIL."""
    h, w = target_hw
    return np.array(
        Image.fromarray(arr).resize((w, h), Image.NEAREST)
    )


def _load_gt_instances(inst_path, target_hw=None):
    """Load GT thing instances from Cityscapes instanceIds.png."""
    inst_map = np.array(Image.open(inst_path), dtype=np.int32)
    if target_hw and inst_map.shape[:2] != tuple(target_hw):
        inst_map = np.array(
            Image.fromarray(inst_map).resize(
                (target_hw[1], target_hw[0]), Image.NEAREST
            )
        )
    masks, class_ids = [], []
    for uid in np.unique(inst_map):
        if uid < 1000:
            continue
        raw_cls = uid // 1000
        if raw_cls not in _CS_ID_TO_TRAIN:
            continue
        train_id = _CS_ID_TO_TRAIN[raw_cls]
        if train_id not in _THING_IDS:
            continue
        mask = inst_map == uid
        if mask.sum() < 10:
            continue
        masks.append(mask)
        class_ids.append(train_id)
    if masks:
        return np.stack(masks), np.array(class_ids)
    H, W = inst_map.shape[:2]
    return np.zeros((0, H, W), dtype=bool), np.array([], dtype=int)


def _load_pred_instances(npz_path, target_hw):
    """Load predicted instance masks from NPZ, resize to target_hw.

    Supports two formats:
      - Full-resolution: masks shape (M, H, W) — from CutLER detector
      - Patch-space: masks shape (M, N_patches) — from depth-guided pipeline
    """
    data = np.load(str(npz_path))
    masks = data["masks"]
    scores = data["scores"] if "scores" in data else None
    num_valid = int(data["num_valid"]) if "num_valid" in data else masks.shape[0]
    masks = masks[:num_valid]
    if scores is not None:
        scores = scores[:num_valid]

    if masks.shape[0] == 0:
        H, W = target_hw
        return np.zeros((0, H, W), dtype=bool), np.array([], dtype=np.float32)

    H, W = target_hw

    if masks.ndim == 3:
        # Full-resolution masks (M, H_orig, W_orig) — resize if needed
        if masks.shape[1:] == (H, W):
            resized = masks.astype(bool)
        else:
            M = masks.shape[0]
            resized = np.zeros((M, H, W), dtype=bool)
            for i in range(M):
                m_img = Image.fromarray(masks[i].astype(np.uint8) * 255)
                resized[i] = np.array(m_img.resize((W, H), Image.NEAREST)) > 127
    elif masks.ndim == 2:
        # Patch-space masks (M, N_patches)
        M, N = masks.shape
        if "h_patches" in data and "w_patches" in data:
            hp, wp = int(data["h_patches"]), int(data["w_patches"])
        else:
            hp, wp = None, None
            for hp_cand, wp_cand in [(128, 64), (64, 128), (32, 64), (64, 32)]:
                if hp_cand * wp_cand == N:
                    hp, wp = hp_cand, wp_cand
                    break
            if hp is None:
                return None, None

        masks_2d = masks.reshape(M, hp, wp)
        resized = np.zeros((M, H, W), dtype=bool)
        for i in range(M):
            m_img = Image.fromarray(masks_2d[i].astype(np.uint8) * 255)
            resized[i] = np.array(m_img.resize((W, H), Image.NEAREST)) > 127
    else:
        return None, None

    if scores is None:
        scores = resized.sum(axis=(1, 2)).astype(np.float32)
    return resized, scores


def _batch_iou(pred_masks, gt_masks):
    """Compute IoU matrix between pred and GT masks. (M_pred, M_gt)."""
    M_pred = pred_masks.shape[0]
    M_gt = gt_masks.shape[0]
    # Flatten for efficient computation
    pred_flat = pred_masks.reshape(M_pred, -1).astype(np.float32)
    gt_flat = gt_masks.reshape(M_gt, -1).astype(np.float32)
    # intersection = pred @ gt.T (dot product of binary vectors)
    intersection = pred_flat @ gt_flat.T
    # union = |pred| + |gt| - intersection
    pred_area = pred_flat.sum(axis=1, keepdims=True)  # (M_pred, 1)
    gt_area = gt_flat.sum(axis=1, keepdims=True).T     # (1, M_gt)
    union = pred_area + gt_area - intersection
    return intersection / (union + 1e-8)


# ─── Semantic Evaluation ───

def evaluate_semantic(pairs, eval_hw, cause27=False):
    """Compute mIoU and pixel accuracy."""
    print(f"\n{'='*60}")
    print(f"SEMANTIC EVALUATION ({len(pairs)} images)")
    if cause27:
        print(f"  (remapping CAUSE 27-class → 19 trainIDs)")
    print(f"{'='*60}")

    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_correct = 0
    total_valid = 0

    for sem_path, gt_label_path, _, _ in tqdm(pairs, desc="Semantic eval"):
        pred = np.array(Image.open(sem_path))
        if cause27:
            pred = _CAUSE27_TO_TRAINID[pred]
        gt_raw = np.array(Image.open(gt_label_path))
        gt = _remap_to_trainids(gt_raw)

        H, W = eval_hw
        if pred.shape != (H, W):
            pred = _resize_nearest(pred, eval_hw)
        if gt.shape != (H, W):
            gt = _resize_nearest(gt, eval_hw)

        valid = gt != IGNORE_LABEL
        p, g = pred[valid], gt[valid]

        mask = (p < NUM_CLASSES) & (g < NUM_CLASSES)
        np.add.at(conf_matrix, (p[mask], g[mask]), 1)
        total_correct += np.sum(p[mask] == g[mask])
        total_valid += mask.sum()

    # Per-class IoU from confusion matrix
    per_class_iou = {}
    ious = []
    for c in range(NUM_CLASSES):
        tp = conf_matrix[c, c]
        fp = conf_matrix[c, :].sum() - tp
        fn = conf_matrix[:, c].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            iou = tp / denom
            ious.append(iou)
            per_class_iou[_CS_CLASS_NAMES[c]] = float(iou)
        else:
            per_class_iou[_CS_CLASS_NAMES[c]] = None

    miou = float(np.mean(ious)) if ious else 0.0
    pixel_acc = total_correct / (total_valid + 1e-8)

    print(f"\n  Per-class IoU:")
    for name, iou in per_class_iou.items():
        if iou is not None:
            kind = "T" if _CS_CLASS_NAMES.index(name) in _THING_IDS else "S"
            bar = "█" * int(iou * 40)
            print(f"    [{kind}] {name:15s}: {iou*100:5.1f}% {bar}")
        else:
            print(f"    [?] {name:15s}:   N/A")

    print(f"\n  mIoU:           {miou*100:.2f}%")
    print(f"  Pixel Accuracy: {pixel_acc*100:.2f}%")

    return {
        "miou": round(miou * 100, 2),
        "pixel_accuracy": round(pixel_acc * 100, 2),
        "per_class_iou": {k: round(v * 100, 2) if v is not None else None
                          for k, v in per_class_iou.items()},
        "num_images": len(pairs),
    }


# ─── Instance Evaluation ───

def evaluate_instances(pairs, eval_hw):
    """Compute AR@100, AP@50, AP@75 for thing instances."""
    print(f"\n{'='*60}")
    print(f"INSTANCE EVALUATION")
    print(f"{'='*60}")

    all_ap50, all_ap75, all_recall = [], [], []
    pred_counts, gt_counts = [], []
    skipped = 0

    for _, _, inst_path, gt_inst_path in tqdm(pairs, desc="Instance eval"):
        if inst_path is None or gt_inst_path is None:
            skipped += 1
            continue

        pred_masks, pred_scores = _load_pred_instances(inst_path, eval_hw)
        if pred_masks is None:
            skipped += 1
            continue

        gt_masks, gt_classes = _load_gt_instances(gt_inst_path, eval_hw)

        pred_counts.append(pred_masks.shape[0])
        gt_counts.append(gt_masks.shape[0])

        if gt_masks.shape[0] == 0:
            continue
        if pred_masks.shape[0] == 0:
            all_recall.append(0.0)
            all_ap50.append(0.0)
            all_ap75.append(0.0)
            continue

        # Compute IoU matrix
        iou_matrix = _batch_iou(pred_masks, gt_masks)
        M_pred, M_gt = iou_matrix.shape

        # AR@100: fraction of GT matched at IoU >= 0.5
        best_per_gt = iou_matrix.max(axis=0)  # (M_gt,)
        recall = float((best_per_gt >= 0.5).sum() / M_gt)
        all_recall.append(recall)

        # AP computation with greedy matching
        sorted_idx = np.argsort(-pred_scores)

        def compute_ap(iou_thresh):
            gt_matched = set()
            tp = np.zeros(M_pred)
            fp = np.zeros(M_pred)
            for rank, i in enumerate(sorted_idx):
                ious_i = iou_matrix[i].copy()
                # Mask already-matched GT
                for j_matched in gt_matched:
                    ious_i[j_matched] = 0.0
                best_j = int(np.argmax(ious_i))
                if ious_i[best_j] >= iou_thresh:
                    tp[rank] = 1
                    gt_matched.add(best_j)
                else:
                    fp[rank] = 1
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            prec = cum_tp / (cum_tp + cum_fp + 1e-8)
            rec = cum_tp / (M_gt + 1e-8)
            # VOC-style AP
            mrec = np.concatenate(([0.0], rec, [1.0]))
            mpre = np.concatenate(([1.0], prec, [0.0]))
            for k in range(len(mpre) - 1, 0, -1):
                mpre[k - 1] = max(mpre[k - 1], mpre[k])
            idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
            return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))

        all_ap50.append(compute_ap(0.5))
        all_ap75.append(compute_ap(0.75))

    results = {
        "ar_100": round(float(np.mean(all_recall)) * 100, 2) if all_recall else 0.0,
        "ap_50": round(float(np.mean(all_ap50)) * 100, 2) if all_ap50 else 0.0,
        "ap_75": round(float(np.mean(all_ap75)) * 100, 2) if all_ap75 else 0.0,
        "avg_pred_instances": round(float(np.mean(pred_counts)), 1) if pred_counts else 0.0,
        "avg_gt_instances": round(float(np.mean(gt_counts)), 1) if gt_counts else 0.0,
        "num_images": len(all_recall),
        "skipped": skipped,
    }

    print(f"\n  AR@100 (IoU≥0.5): {results['ar_100']:.2f}%")
    print(f"  AP@50:            {results['ap_50']:.2f}%")
    print(f"  AP@75:            {results['ap_75']:.2f}%")
    print(f"  Avg pred instances: {results['avg_pred_instances']:.1f}")
    print(f"  Avg GT instances:   {results['avg_gt_instances']:.1f}")
    print(f"  Images evaluated: {results['num_images']} (skipped: {skipped})")

    return results


# ─── Panoptic Evaluation ───


def _connected_components_things(pred_sem, thing_ids, min_area=10):
    """Create thing instances from connected components of the semantic map."""
    from scipy import ndimage

    segments = []  # list of (mask, class_id)
    for cls in thing_ids:
        cls_mask = pred_sem == cls
        if cls_mask.sum() < min_area:
            continue
        labeled, n_components = ndimage.label(cls_mask)
        for comp_id in range(1, n_components + 1):
            comp_mask = labeled == comp_id
            if comp_mask.sum() < min_area:
                continue
            segments.append((comp_mask, cls))
    return segments


def evaluate_panoptic(pairs, eval_hw, thing_mode="connected_components",
                      pred_stuff_ids=None, pred_thing_ids=None, cause27=False,
                      cc_min_area=10):
    """Compute PQ, SQ, RQ with standard Cityscapes stuff/things split.

    Args:
        thing_mode: "connected_components" (CC of semantic map) or "maskcut"
        pred_stuff_ids: Optional unsupervised stuff IDs for predicted panoptic map.
                        If None, uses standard Cityscapes split.
        pred_thing_ids: Optional unsupervised thing IDs for predicted panoptic map.
                        If None, uses standard Cityscapes split.
        cause27: If True, remap CAUSE 27-class predictions to 19 trainIDs.
    """
    # Predicted side can use unsupervised split; GT always uses standard
    p_stuff = pred_stuff_ids if pred_stuff_ids is not None else _STUFF_IDS
    p_thing = pred_thing_ids if pred_thing_ids is not None else _THING_IDS

    split_info = (f"pred: {len(p_stuff)} stuff/{len(p_thing)} things, "
                  f"GT: 11 stuff/8 things")
    print(f"\n{'='*60}")
    print(f"PANOPTIC EVALUATION ({split_info})")
    print(f"  Thing instance mode: {thing_mode}")
    print(f"{'='*60}")

    tp = np.zeros(NUM_CLASSES)
    fp = np.zeros(NUM_CLASSES)
    fn = np.zeros(NUM_CLASSES)
    iou_sum = np.zeros(NUM_CLASSES)

    num_evaluated = 0
    H, W = eval_hw

    for sem_path, gt_label_path, inst_path, gt_inst_path in tqdm(pairs, desc="Panoptic eval"):
        if gt_label_path is None or gt_inst_path is None:
            continue

        # Load and resize predictions
        pred_sem = np.array(Image.open(sem_path))
        if cause27:
            pred_sem = _CAUSE27_TO_TRAINID[pred_sem]
        if pred_sem.shape != (H, W):
            pred_sem = _resize_nearest(pred_sem, eval_hw)

        # Load GT
        gt_raw = np.array(Image.open(gt_label_path))
        gt_sem = _remap_to_trainids(gt_raw)
        if gt_sem.shape != (H, W):
            gt_sem = _resize_nearest(gt_sem, eval_hw)

        gt_inst_map = np.array(Image.open(gt_inst_path), dtype=np.int32)
        if gt_inst_map.shape != (H, W):
            gt_inst_map = np.array(
                Image.fromarray(gt_inst_map).resize((W, H), Image.NEAREST)
            )

        # ── Build predicted panoptic map ──
        pred_pan = np.zeros((H, W), dtype=np.int32)
        pred_segments = {}  # seg_id -> category_id
        next_id = 1

        # Stuff: one segment per class from semantic map
        for cls in p_stuff:
            mask = pred_sem == cls
            if mask.sum() < 64:
                continue
            pred_pan[mask] = next_id
            pred_segments[next_id] = cls
            next_id += 1

        # Things: depends on mode
        if thing_mode == "connected_components":
            thing_segments = _connected_components_things(pred_sem, p_thing)
            for mask, cls in thing_segments:
                pred_pan[mask] = next_id
                pred_segments[next_id] = cls
                next_id += 1
        elif thing_mode in ("maskcut", "hybrid"):
            pred_inst_masks, pred_inst_scores = None, None
            if inst_path is not None and inst_path.exists():
                pred_inst_masks, pred_inst_scores = _load_pred_instances(inst_path, eval_hw)
            if pred_inst_masks is not None and pred_inst_masks.shape[0] > 0:
                order = np.argsort(-pred_inst_scores) if pred_inst_scores is not None else np.arange(pred_inst_masks.shape[0])
                for idx in order:
                    m = pred_inst_masks[idx]
                    if m.sum() < 10:
                        continue
                    sem_vals = pred_sem[m]
                    sem_vals = sem_vals[sem_vals < NUM_CLASSES]
                    if len(sem_vals) == 0:
                        continue
                    majority_cls = int(np.bincount(sem_vals, minlength=NUM_CLASSES).argmax())
                    if majority_cls not in p_thing:
                        continue
                    pred_pan[m] = next_id
                    pred_segments[next_id] = majority_cls
                    next_id += 1

            # Hybrid: CC fallback for uncovered thing pixels
            if thing_mode == "hybrid":
                from scipy import ndimage
                covered = pred_pan > 0
                for cls in p_thing:
                    cls_mask = (pred_sem == cls) & ~covered
                    if cls_mask.sum() < cc_min_area:
                        continue
                    labeled, n_components = ndimage.label(cls_mask)
                    for comp_id in range(1, n_components + 1):
                        comp_mask = labeled == comp_id
                        if comp_mask.sum() < cc_min_area:
                            continue
                        pred_pan[comp_mask] = next_id
                        pred_segments[next_id] = cls
                        next_id += 1

        # ── Build GT panoptic map ──
        gt_pan = np.zeros((H, W), dtype=np.int32)
        gt_segments = {}
        gt_next_id = 1

        # GT stuff
        for cls in _STUFF_IDS:
            mask = gt_sem == cls
            if mask.sum() < 64:
                continue
            gt_pan[mask] = gt_next_id
            gt_segments[gt_next_id] = cls
            gt_next_id += 1

        # GT things from instanceIds
        for uid in np.unique(gt_inst_map):
            if uid < 1000:
                continue
            raw_cls = uid // 1000
            if raw_cls not in _CS_ID_TO_TRAIN:
                continue
            train_id = _CS_ID_TO_TRAIN[raw_cls]
            if train_id not in _THING_IDS:
                continue
            mask = gt_inst_map == uid
            if mask.sum() < 10:
                continue
            gt_pan[mask] = gt_next_id
            gt_segments[gt_next_id] = train_id
            gt_next_id += 1

        # ── Match GT to pred per category ──
        matched_pred = set()
        matched_gt = set()

        # Group segments by category for efficient matching
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

            # Compute IoU between all GT-pred pairs of this category
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

            # FP: unmatched predictions
            for pred_id in pred_segs:
                if pred_id not in matched_pred:
                    fp[cat] += 1

        num_evaluated += 1

    # ── Compute metrics ──
    per_class = {}
    all_pq, stuff_pq, thing_pq = [], [], []

    for c in range(NUM_CLASSES):
        t, f_p, f_n = tp[c], fp[c], fn[c]
        iou_s = iou_sum[c]

        if t + f_p + f_n > 0:
            sq = iou_s / (t + 1e-8)
            rq = t / (t + 0.5 * f_p + 0.5 * f_n)
            pq = sq * rq
        else:
            sq = rq = pq = 0.0

        name = _CS_CLASS_NAMES[c]
        kind = "stuff" if c in _STUFF_IDS else "thing"
        per_class[name] = {
            "PQ": round(pq * 100, 2), "SQ": round(sq * 100, 2),
            "RQ": round(rq * 100, 2),
            "TP": int(t), "FP": int(f_p), "FN": int(f_n),
            "type": kind,
        }

        if t + f_p + f_n > 0:
            all_pq.append(pq)
            if c in _STUFF_IDS:
                stuff_pq.append(pq)
            else:
                thing_pq.append(pq)

    overall_pq = float(np.mean(all_pq)) * 100 if all_pq else 0.0
    overall_stuff_pq = float(np.mean(stuff_pq)) * 100 if stuff_pq else 0.0
    overall_thing_pq = float(np.mean(thing_pq)) * 100 if thing_pq else 0.0
    overall_sq = float(np.sum(iou_sum) / (np.sum(tp) + 1e-8)) * 100
    overall_rq = float(np.sum(tp) / (np.sum(tp) + 0.5 * np.sum(fp) + 0.5 * np.sum(fn) + 1e-8)) * 100

    # Print results
    print(f"\n  Per-class PQ (sorted):")
    for name, vals in sorted(per_class.items(), key=lambda x: x[1]["PQ"], reverse=True):
        kind = vals["type"][0].upper()
        print(f"    [{kind}] {name:15s}: PQ={vals['PQ']:5.1f}  SQ={vals['SQ']:5.1f}  "
              f"RQ={vals['RQ']:5.1f}  (TP={vals['TP']} FP={vals['FP']} FN={vals['FN']})")

    print(f"\n  ┌────────────────────────────────────────────┐")
    print(f"  │  PQ (all):    {overall_pq:5.1f}  (19 classes)          │")
    print(f"  │  PQ (stuff):  {overall_stuff_pq:5.1f}  (11 classes)          │")
    print(f"  │  PQ (things): {overall_thing_pq:5.1f}  ( 8 classes)          │")
    print(f"  │  SQ:          {overall_sq:5.1f}                        │")
    print(f"  │  RQ:          {overall_rq:5.1f}                        │")
    print(f"  └────────────────────────────────────────────┘")
    print(f"  Images evaluated: {num_evaluated}")

    return {
        "PQ": round(overall_pq, 2),
        "PQ_stuff": round(overall_stuff_pq, 2),
        "PQ_things": round(overall_thing_pq, 2),
        "SQ": round(overall_sq, 2),
        "RQ": round(overall_rq, 2),
        "per_class": per_class,
        "num_images": num_evaluated,
    }


# ─── File Discovery ───

def discover_pairs(cityscapes_root, split, semantic_subdir="pseudo_semantic_dinov3",
                   instance_subdir="pseudo_instance"):
    """Find matching (semantic, gt_label, instance, gt_instance) file paths."""
    root = Path(cityscapes_root)
    sem_dir = root / semantic_subdir / split
    inst_dir = root / instance_subdir / split
    gt_dir = root / "gtFine" / split

    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    # Find all GT files
    gt_label_files = sorted(gt_dir.rglob("*_gtFine_labelIds.png"))
    print(f"Found {len(gt_label_files)} GT label files in {gt_dir}")

    pairs = []
    sem_found = 0
    inst_found = 0

    for gt_label_path in gt_label_files:
        # Extract the base name: city/city_seq_frame
        rel = gt_label_path.relative_to(gt_dir)
        base = str(rel).replace("_gtFine_labelIds.png", "")

        # GT instance
        gt_inst_path = gt_dir / (base + "_gtFine_instanceIds.png")
        if not gt_inst_path.exists():
            gt_inst_path = None

        # Predicted semantic — try _leftImg8bit.png first, then plain .png
        sem_path = sem_dir / (base + "_leftImg8bit.png")
        if not sem_path.exists():
            sem_path = sem_dir / (base + ".png")
        if not sem_path.exists():
            sem_path = None
        else:
            sem_found += 1

        # Predicted instance — try _leftImg8bit.npz first, then plain .npz
        inst_path = inst_dir / (base + "_leftImg8bit.npz")
        if not inst_path.exists():
            inst_path = inst_dir / (base + ".npz")
        if not inst_path.exists():
            inst_path = None
        else:
            inst_found += 1

        if sem_path is not None or inst_path is not None:
            pairs.append((sem_path, gt_label_path, inst_path, gt_inst_path))

    print(f"Discovered {len(pairs)} evaluation pairs")
    print(f"  Semantic pseudo-labels: {sem_found}/{len(gt_label_files)}")
    print(f"  Instance pseudo-labels: {inst_found}/{len(gt_label_files)}")
    return pairs


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="Evaluate cascade pseudo-labels")
    parser.add_argument("--cityscapes_root", type=str, required=True,
                        help="Path to cityscapes dataset root")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"],
                        help="Dataset split to evaluate")
    parser.add_argument("--eval_size", type=int, nargs=2, default=[512, 1024],
                        help="(H, W) evaluation resolution")
    parser.add_argument("--semantic_subdir", type=str, default="pseudo_semantic_dinov3")
    parser.add_argument("--instance_subdir", type=str, default="pseudo_instance")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit number of images (for quick testing)")
    parser.add_argument("--skip_semantic", action="store_true")
    parser.add_argument("--skip_instance", action="store_true")
    parser.add_argument("--skip_panoptic", action="store_true")
    parser.add_argument("--thing_mode", type=str, default="connected_components",
                        choices=["connected_components", "maskcut", "hybrid"],
                        help="How to create thing instances for panoptic eval")
    parser.add_argument("--cc_min_area", type=int, default=10,
                        help="Min area for CC fallback instances in hybrid mode (default: 10)")
    parser.add_argument("--stuff_things", type=str, default=None,
                        help="Path to stuff_things.json for unsupervised stuff/things split. "
                             "If provided, uses these IDs for building predicted panoptic map. "
                             "GT side always uses standard Cityscapes split.")
    parser.add_argument("--cause27", action="store_true",
                        help="Semantic pseudo-labels are in CAUSE 27-class format. "
                             "Remap to standard 19 trainIDs before evaluation.")
    args = parser.parse_args()

    eval_hw = tuple(args.eval_size)

    # Load unsupervised stuff/things split if provided
    pred_stuff_ids = None
    pred_thing_ids = None
    if args.stuff_things:
        import json as _json
        with open(args.stuff_things, "r") as _f:
            _st = _json.load(_f)
        pred_stuff_ids = set(_st.get("stuff_ids", []))
        pred_thing_ids = set(_st.get("thing_ids", []))
        print(f"\nCascade Pseudo-Label Evaluation")
        print(f"  Dataset: Cityscapes {args.split}")
        print(f"  Eval resolution: {eval_hw[0]}x{eval_hw[1]}")
        print(f"  Pred stuff/things: from {args.stuff_things}")
        print(f"    Pred stuff IDs: {sorted(pred_stuff_ids)} ({len(pred_stuff_ids)} classes)")
        print(f"    Pred thing IDs: {sorted(pred_thing_ids)} ({len(pred_thing_ids)} classes)")
        print(f"  GT stuff/things: Standard (11 stuff, 8 things)")
    else:
        print(f"\nCascade Pseudo-Label Evaluation")
        print(f"  Dataset: Cityscapes {args.split}")
        print(f"  Eval resolution: {eval_hw[0]}x{eval_hw[1]}")
        print(f"  Stuff/Things split: Standard (11 stuff, 8 things)")

    # Discover files
    pairs = discover_pairs(args.cityscapes_root, args.split,
                           args.semantic_subdir, args.instance_subdir)

    if args.max_images:
        pairs = pairs[:args.max_images]
        print(f"  Limited to {len(pairs)} images")

    results = {"split": args.split, "eval_resolution": list(eval_hw)}
    t0 = time.time()

    # Check what data is available
    has_semantic = any(p[0] is not None for p in pairs)
    has_instance = any(p[2] is not None for p in pairs)

    # 1. Semantic evaluation
    if has_semantic and not args.skip_semantic:
        sem_pairs = [(s, gl, i, gi) for s, gl, i, gi in pairs if s is not None]
        results["semantic"] = evaluate_semantic(sem_pairs, eval_hw, cause27=args.cause27)
    elif not has_semantic:
        print(f"\n  [SKIP] No semantic pseudo-labels found for {args.split}")

    # 2. Instance evaluation
    if has_instance and not args.skip_instance:
        inst_pairs = [(s, gl, i, gi) for s, gl, i, gi in pairs if i is not None and gi is not None]
        results["instance"] = evaluate_instances(inst_pairs, eval_hw)
    elif not has_instance:
        print(f"\n  [SKIP] No instance pseudo-labels found for {args.split}")

    # 3. Panoptic evaluation (CC mode only needs semantic; maskcut needs both)
    can_panoptic = has_semantic and (args.thing_mode == "connected_components" or has_instance)
    if can_panoptic and not args.skip_panoptic:
        pan_pairs = [(s, gl, i, gi) for s, gl, i, gi in pairs
                     if s is not None and gl is not None]
        results["panoptic"] = evaluate_panoptic(
            pan_pairs, eval_hw, args.thing_mode,
            pred_stuff_ids=pred_stuff_ids, pred_thing_ids=pred_thing_ids,
            cause27=args.cause27,
            cc_min_area=args.cc_min_area,
        )
    elif not can_panoptic:
        print(f"\n  [SKIP] Panoptic eval requires semantic pseudo-labels"
              f" (+ instance for maskcut mode)")

    elapsed = time.time() - t0

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY — Cascade Pseudo-Label Quality ({args.split})")
    print(f"{'='*60}")
    if "semantic" in results:
        print(f"  Semantic mIoU:     {results['semantic']['miou']:.1f}%")
        print(f"  Pixel Accuracy:    {results['semantic']['pixel_accuracy']:.1f}%")
    if "instance" in results:
        print(f"  Instance AR@100:   {results['instance']['ar_100']:.1f}%")
        print(f"  Instance AP@50:    {results['instance']['ap_50']:.1f}%")
        print(f"  Instance AP@75:    {results['instance']['ap_75']:.1f}%")
    if "panoptic" in results:
        r = results["panoptic"]
        print(f"  Panoptic PQ:       {r['PQ']:.1f}%")
        print(f"  PQ Stuff:          {r['PQ_stuff']:.1f}%")
        print(f"  PQ Things:         {r['PQ_things']:.1f}%")
        print(f"  SQ:                {r['SQ']:.1f}%")
        print(f"  RQ:                {r['RQ']:.1f}%")

    # CUPS comparison
    if "panoptic" in results:
        r = results["panoptic"]
        print(f"\n  ┌─────────────────────────────────────────────────┐")
        print(f"  │  Comparison with CUPS (CVPR 2025 SOTA, val)     │")
        print(f"  │  Metric    CUPS     Ours ({args.split:5s})    Δ           │")
        print(f"  │  PQ        27.8     {r['PQ']:5.1f}       {r['PQ']-27.8:+5.1f}        │")
        print(f"  │  PQ^St     35.1     {r['PQ_stuff']:5.1f}       {r['PQ_stuff']-35.1:+5.1f}        │")
        print(f"  │  PQ^Th     17.7     {r['PQ_things']:5.1f}       {r['PQ_things']-17.7:+5.1f}        │")
        print(f"  │  SQ        57.4     {r['SQ']:5.1f}       {r['SQ']-57.4:+5.1f}        │")
        print(f"  │  RQ        35.2     {r['RQ']:5.1f}       {r['RQ']-35.2:+5.1f}        │")
        print(f"  └─────────────────────────────────────────────────┘")

    print(f"\n  Total time: {elapsed:.0f}s")

    # Save
    output_path = args.output or str(
        Path(args.cityscapes_root) / f"eval_cascade_{args.split}.json"
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
