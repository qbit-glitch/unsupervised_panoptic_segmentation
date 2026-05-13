#!/usr/bin/env python3
"""Comprehensive evaluation of pseudo-labels against Cityscapes GT.

Computes:
  1. Semantic: mIoU, pixel accuracy, per-class IoU (Hungarian matching)
  2. Instance: AR@100, AP@50, AP@75, AP_mean, object count stats
  3. Panoptic: PQ, SQ, RQ (overall + things/stuff split)

Usage:
    python mbps_pytorch/evaluate_pseudolabels.py \
        --semantic_dir /data/cityscapes/pseudo_semantic/val \
        --instance_dir /data/cityscapes/pseudo_instance/val \
        --gt_dir /data/cityscapes/gtFine/val \
        --num_classes 19 --image_size 512 1024

    # Semantic-only evaluation:
    python mbps_pytorch/evaluate_pseudolabels.py \
        --semantic_dir /data/cityscapes/pseudo_semantic/val \
        --gt_dir /data/cityscapes/gtFine/val \
        --num_classes 19 --image_size 512 1024
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# Cityscapes raw labelId → trainId mapping
_CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}

# Cityscapes class names (train ID order)
_CS_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

# Things (train IDs 11-18), Stuff (train IDs 0-10)
_THING_TRAIN_IDS = set(range(11, 19))
_STUFF_TRAIN_IDS = set(range(0, 11))

# Cityscapes raw class IDs that are "things" (have instances)
_THING_RAW_IDS = {24, 25, 26, 27, 28, 31, 32, 33}


def _find_gt_file(pred_path, pred_dir, gt_dir, suffix):
    """Find GT file corresponding to a prediction file."""
    rel = pred_path.relative_to(pred_dir)
    base = str(rel).replace("_leftImg8bit.png", "").replace(".png", "")
    base = base.replace("_leftImg8bit.npz", "").replace(".npz", "")
    base = base.replace("_leftImg8bit.npy", "").replace(".npy", "")

    candidate = gt_dir / (base + suffix)
    if candidate.exists():
        return candidate
    return None


def _remap_labelids_to_trainids(gt):
    """Remap Cityscapes raw labelIds to train IDs (0-18)."""
    remapped = np.full_like(gt, 255, dtype=np.uint8)
    for raw_id, train_id in _CS_ID_TO_TRAIN.items():
        remapped[gt == raw_id] = train_id
    return remapped


def _load_gt_instances(instance_path, label_path=None):
    """Load Cityscapes GT instances.

    Cityscapes instanceIds encoding: classId * 1000 + instanceIdx for things.
    Stuff regions have instanceId == classId (no per-instance separation).

    Returns:
        masks: (M, H, W) binary masks for thing instances
        class_ids: (M,) train class ID for each instance
    """
    inst_map = np.array(Image.open(instance_path), dtype=np.int32)
    H, W = inst_map.shape

    masks = []
    class_ids = []

    unique_ids = np.unique(inst_map)
    for uid in unique_ids:
        if uid < 1000:
            continue  # Skip stuff regions (no per-instance GT)
        raw_class = uid // 1000
        if raw_class not in _CS_ID_TO_TRAIN:
            continue
        train_id = _CS_ID_TO_TRAIN[raw_class]
        mask = (inst_map == uid)
        if mask.sum() < 10:  # Skip tiny fragments
            continue
        masks.append(mask)
        class_ids.append(train_id)

    if masks:
        return np.stack(masks), np.array(class_ids)
    return np.zeros((0, H, W), dtype=bool), np.array([], dtype=int)


def _compute_mask_iou(pred_mask, gt_mask):
    """IoU between two binary masks."""
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask | gt_mask)
    return float(intersection / (union + 1e-8))


# ─── Semantic Evaluation ───


def evaluate_semantic(
    semantic_dir: str,
    gt_dir: str,
    num_classes: int = 19,
    ignore_label: int = 255,
) -> dict:
    """Evaluate semantic pseudo-labels against GT using Hungarian matching.

    Returns dict with mIoU, pixel_accuracy, per_class_iou, mapping.
    """
    semantic_dir = Path(semantic_dir)
    gt_dir = Path(gt_dir)

    pred_files = sorted(semantic_dir.rglob("*.png"))
    print(f"\n{'='*60}")
    print(f"SEMANTIC EVALUATION")
    print(f"{'='*60}")
    print(f"Found {len(pred_files)} pseudo-label files")

    all_pred = []
    all_gt = []

    for pred_path in tqdm(pred_files, desc="Loading semantic pairs"):
        pred = np.array(Image.open(pred_path))

        gt_path = _find_gt_file(pred_path, semantic_dir, gt_dir, "_gtFine_labelIds.png")
        if gt_path is None:
            gt_path = _find_gt_file(pred_path, semantic_dir, gt_dir, "_gtFine_labelTrainIds.png")
        if gt_path is None:
            continue

        gt = np.array(Image.open(gt_path))

        # Remap if using raw labelIds
        if "_labelIds.png" in str(gt_path) and "TrainIds" not in str(gt_path):
            gt = _remap_labelids_to_trainids(gt)

        # Resize pred to match GT if needed
        if pred.shape != gt.shape:
            pred = np.array(
                Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
            )

        all_pred.append(pred.flatten())
        all_gt.append(gt.flatten())

    if not all_pred:
        print("ERROR: No matching GT files found!")
        return {"error": "No matching GT files"}

    all_pred = np.concatenate(all_pred)
    all_gt = np.concatenate(all_gt)
    print(f"Total pixels: {len(all_pred):,}")

    # Hungarian matching
    print("Running Hungarian matching...")
    valid = all_gt != ignore_label
    pred_valid = all_pred[valid]
    gt_valid = all_gt[valid]

    cost = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid_pair = (pred_valid < num_classes) & (gt_valid < num_classes)
    p = pred_valid[valid_pair]
    g = gt_valid[valid_pair]
    np.add.at(cost, (p, g), 1)

    row_ind, col_ind = linear_sum_assignment(-cost)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    # Apply mapping
    mapped_pred = np.full_like(all_pred, ignore_label)
    for cluster_id, class_id in mapping.items():
        mapped_pred[all_pred == cluster_id] = class_id

    # Per-class IoU
    per_class_iou = {}
    ious = []
    for c in range(num_classes):
        pred_c = mapped_pred == c
        gt_c = all_gt == c
        v = all_gt != ignore_label

        intersection = np.sum(pred_c & gt_c & v)
        union = np.sum((pred_c | gt_c) & v)

        if union > 0:
            iou = intersection / union
            ious.append(iou)
            per_class_iou[_CS_CLASS_NAMES[c]] = float(iou)
        else:
            per_class_iou[_CS_CLASS_NAMES[c]] = None

    miou = float(np.mean(ious)) if ious else 0.0
    pixel_accuracy = float(np.mean(mapped_pred[valid] == all_gt[valid]))

    # Print results
    print(f"\n  Cluster → Class mapping:")
    for cluster, cls in sorted(mapping.items()):
        print(f"    Cluster {cluster:2d} → {_CS_CLASS_NAMES[cls]} (class {cls})")

    print(f"\n  Per-class IoU:")
    for name, iou in per_class_iou.items():
        if iou is not None:
            bar = "█" * int(iou * 40)
            print(f"    {name:15s}: {iou:.4f} {bar}")
        else:
            print(f"    {name:15s}: N/A")

    print(f"\n  mIoU: {miou:.4f} ({miou * 100:.2f}%)")
    print(f"  Pixel Accuracy: {pixel_accuracy:.4f} ({pixel_accuracy * 100:.2f}%)")
    print(f"  Classes evaluated: {len(ious)}/{num_classes}")

    return {
        "miou": miou,
        "pixel_accuracy": pixel_accuracy,
        "per_class_iou": per_class_iou,
        "mapping": {str(k): v for k, v in mapping.items()},
        "num_classes_evaluated": len(ious),
    }


# ─── Instance Evaluation ───


def evaluate_instances(
    instance_dir: str,
    gt_dir: str,
    image_size: tuple = (512, 1024),
    max_detections: int = 100,
) -> dict:
    """Evaluate instance pseudo-labels against Cityscapes GT.

    Computes AR@100, AP@50, AP@75, AP_mean, and object count stats.
    """
    instance_dir = Path(instance_dir)
    gt_dir = Path(gt_dir)

    inst_files = sorted(instance_dir.rglob("*.npz"))
    if not inst_files:
        inst_files = sorted(instance_dir.rglob("*.npy"))
    print(f"\n{'='*60}")
    print(f"INSTANCE EVALUATION")
    print(f"{'='*60}")
    print(f"Found {len(inst_files)} instance pseudo-label files")

    all_ap50 = []
    all_ap75 = []
    all_recall = []
    pred_counts = []
    gt_counts = []

    for inst_path in tqdm(inst_files, desc="Evaluating instances"):
        # Load predicted instance masks
        pred_scores = None
        if inst_path.suffix == ".npz":
            data = np.load(str(inst_path))
            if "masks" in data:
                pred_masks = data["masks"]  # (M, N_patches) or (M, H, W)
            elif "instance_masks" in data:
                pred_masks = data["instance_masks"]
            else:
                pred_masks = data[list(data.keys())[0]]

            # Get scores and num_valid if available
            if "scores" in data:
                pred_scores = data["scores"]
            if "num_valid" in data:
                num_valid = int(data["num_valid"])
                pred_masks = pred_masks[:num_valid]
                if pred_scores is not None:
                    pred_scores = pred_scores[:num_valid]
        else:
            pred_masks = np.load(str(inst_path))
            if pred_masks.ndim == 2 and pred_masks.shape[0] != pred_masks.shape[1]:
                pass  # Already (M, N) flattened masks
            elif pred_masks.ndim == 2:
                # Instance ID map → convert to binary masks
                unique_ids = np.unique(pred_masks)
                unique_ids = unique_ids[unique_ids > 0]
                pred_masks = np.stack(
                    [pred_masks == uid for uid in unique_ids]
                ) if len(unique_ids) > 0 else np.zeros((0,) + pred_masks.shape, dtype=bool)

        if pred_masks.ndim == 2:
            # Flattened masks at patch resolution (M, N_patches)
            # Reshape to (M, h_patches, w_patches) and upsample
            M, N = pred_masks.shape
            # Infer patch grid from total patches
            h_patches = image_size[0] // 16  # 32
            w_patches = image_size[1] // 16  # 64
            if N == h_patches * w_patches:
                pred_masks = pred_masks.reshape(M, h_patches, w_patches)
                # Upsample each mask to pixel resolution
                resized = []
                for m in pred_masks:
                    m_img = Image.fromarray(m.astype(np.uint8) * 255)
                    m_img = m_img.resize((image_size[1], image_size[0]), Image.NEAREST)
                    resized.append(np.array(m_img) > 127)
                pred_masks = np.stack(resized) if resized else np.zeros((0,) + tuple(image_size), dtype=bool)
            else:
                continue  # Unknown format
        elif pred_masks.ndim == 3:
            # Already (M, H, W), resize if needed
            if pred_masks.shape[1:] != tuple(image_size):
                resized = []
                for m in pred_masks:
                    m_img = Image.fromarray(m.astype(np.uint8) * 255)
                    m_img = m_img.resize((image_size[1], image_size[0]), Image.NEAREST)
                    resized.append(np.array(m_img) > 127)
                pred_masks = np.stack(resized) if resized else np.zeros((0,) + tuple(image_size), dtype=bool)
        else:
            continue

        pred_masks = pred_masks.astype(bool)

        # Find GT instance file
        gt_inst_path = _find_gt_file(inst_path, instance_dir, gt_dir, "_gtFine_instanceIds.png")
        if gt_inst_path is None:
            continue

        gt_masks, gt_classes = _load_gt_instances(gt_inst_path)

        # Resize GT to evaluation size
        if gt_masks.shape[1:] != tuple(image_size) and gt_masks.shape[0] > 0:
            resized = []
            for m in gt_masks:
                m_img = Image.fromarray(m.astype(np.uint8) * 255)
                m_img = m_img.resize((image_size[1], image_size[0]), Image.NEAREST)
                resized.append(np.array(m_img) > 127)
            gt_masks = np.stack(resized)

        pred_counts.append(pred_masks.shape[0])
        gt_counts.append(gt_masks.shape[0])

        if gt_masks.shape[0] == 0:
            continue

        # Use provided scores or fall back to mask area
        if pred_scores is None:
            pred_scores = pred_masks.sum(axis=(1, 2)).astype(np.float32)

        # Limit predictions
        if pred_masks.shape[0] > max_detections:
            top_idx = np.argsort(-pred_scores)[:max_detections]
            pred_masks = pred_masks[top_idx]
            pred_scores = pred_scores[top_idx]

        # Flatten masks for IoU computation
        M_pred = pred_masks.shape[0]
        M_gt = gt_masks.shape[0]

        if M_pred == 0:
            all_ap50.append(0.0)
            all_ap75.append(0.0)
            all_recall.append(0.0)
            continue

        # Compute IoU matrix (M_pred, M_gt)
        pred_flat = pred_masks.reshape(M_pred, -1)
        gt_flat = gt_masks.reshape(M_gt, -1)

        iou_matrix = np.zeros((M_pred, M_gt))
        for i in range(M_pred):
            for j in range(M_gt):
                intersection = np.sum(pred_flat[i] & gt_flat[j])
                union = np.sum(pred_flat[i] | gt_flat[j])
                iou_matrix[i, j] = intersection / (union + 1e-8)

        # AR@100: best recall at IoU=0.5
        matched_gt = set()
        for iou_thresh in [0.5]:
            for j in range(M_gt):
                best_iou = 0.0
                for i in range(M_pred):
                    if iou_matrix[i, j] > best_iou:
                        best_iou = iou_matrix[i, j]
                if best_iou >= iou_thresh:
                    matched_gt.add(j)
        recall = len(matched_gt) / M_gt if M_gt > 0 else 0.0
        all_recall.append(recall)

        # AP@50 (use scores for ranking, greedy matching)
        sorted_idx = np.argsort(-pred_scores)

        def compute_ap_at_thresh(iou_thresh):
            gt_matched = set()
            tp = np.zeros(M_pred)
            fp = np.zeros(M_pred)
            for rank, i in enumerate(sorted_idx):
                best_iou = 0.0
                best_j = -1
                for j in range(M_gt):
                    if j in gt_matched:
                        continue
                    if iou_matrix[i, j] > best_iou:
                        best_iou = iou_matrix[i, j]
                        best_j = j
                if best_iou >= iou_thresh and best_j >= 0:
                    tp[rank] = 1
                    gt_matched.add(best_j)
                else:
                    fp[rank] = 1
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            prec = cum_tp / (cum_tp + cum_fp + 1e-8)
            rec = cum_tp / (M_gt + 1e-8)
            # AP via all-point interpolation
            mrec = np.concatenate(([0.0], rec, [1.0]))
            mpre = np.concatenate(([1.0], prec, [0.0]))
            for k in range(len(mpre) - 1, 0, -1):
                mpre[k - 1] = max(mpre[k - 1], mpre[k])
            idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
            return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))

        all_ap50.append(compute_ap_at_thresh(0.5))
        all_ap75.append(compute_ap_at_thresh(0.75))

    # Aggregate
    mean_ap50 = float(np.mean(all_ap50)) if all_ap50 else 0.0
    mean_ap75 = float(np.mean(all_ap75)) if all_ap75 else 0.0
    mean_recall = float(np.mean(all_recall)) if all_recall else 0.0
    mean_pred_count = float(np.mean(pred_counts)) if pred_counts else 0.0
    mean_gt_count = float(np.mean(gt_counts)) if gt_counts else 0.0

    print(f"\n  AR@100 (IoU=0.5): {mean_recall:.4f} ({mean_recall * 100:.2f}%)")
    print(f"  AP@50:            {mean_ap50:.4f} ({mean_ap50 * 100:.2f}%)")
    print(f"  AP@75:            {mean_ap75:.4f} ({mean_ap75 * 100:.2f}%)")
    print(f"  Avg pred instances/image: {mean_pred_count:.1f}")
    print(f"  Avg GT instances/image:   {mean_gt_count:.1f}")
    print(f"  Images evaluated: {len(all_ap50)}")

    return {
        "ar_100": mean_recall,
        "ap_50": mean_ap50,
        "ap_75": mean_ap75,
        "avg_pred_instances": mean_pred_count,
        "avg_gt_instances": mean_gt_count,
        "num_images": len(all_ap50),
    }


# ─── Panoptic Evaluation ───


def evaluate_panoptic(
    semantic_dir: str,
    instance_dir: str,
    gt_dir: str,
    stuff_things_map: dict = None,
    num_classes: int = 19,
    image_size: tuple = (512, 1024),
    semantic_mapping: dict = None,
) -> dict:
    """Evaluate panoptic quality by merging semantic + instance pseudo-labels.

    Uses standard PQ = SQ * RQ metric.
    """
    semantic_dir = Path(semantic_dir)
    instance_dir = Path(instance_dir)
    gt_dir = Path(gt_dir)

    print(f"\n{'='*60}")
    print(f"PANOPTIC EVALUATION")
    print(f"{'='*60}")

    # Determine thing/stuff classes
    if stuff_things_map:
        thing_ids = set()
        stuff_ids = set()
        for cls_str, cls_type in stuff_things_map.items():
            cls_id = int(cls_str)
            if cls_type == "thing":
                thing_ids.add(cls_id)
            else:
                stuff_ids.add(cls_id)
    else:
        thing_ids = _THING_TRAIN_IDS
        stuff_ids = _STUFF_TRAIN_IDS

    print(f"Thing classes: {sorted(thing_ids)} ({len(thing_ids)} classes)")
    print(f"Stuff classes: {sorted(stuff_ids)} ({len(stuff_ids)} classes)")

    sem_files = sorted(semantic_dir.rglob("*.png"))

    # Accumulators per class
    all_classes = thing_ids | stuff_ids
    max_cls = max(all_classes) + 1
    tp_per_class = np.zeros(max_cls)
    fp_per_class = np.zeros(max_cls)
    fn_per_class = np.zeros(max_cls)
    iou_sum_per_class = np.zeros(max_cls)

    num_evaluated = 0

    for sem_path in tqdm(sem_files, desc="Evaluating panoptic"):
        # Load pseudo semantic
        pred_sem = np.array(Image.open(sem_path))

        # Apply mapping if provided (cluster → GT class)
        if semantic_mapping:
            mapped = np.full_like(pred_sem, 255)
            for cluster_str, cls_id in semantic_mapping.items():
                mapped[pred_sem == int(cluster_str)] = cls_id
            pred_sem = mapped

        # Load pseudo instances
        rel = sem_path.relative_to(semantic_dir)
        inst_path_npz = instance_dir / rel.with_suffix(".npz")
        inst_path_npy = instance_dir / rel.with_suffix(".npy")

        pred_inst_masks = None
        if inst_path_npz.exists():
            data = np.load(str(inst_path_npz))
            key = "masks" if "masks" in data else list(data.keys())[0]
            pred_inst_masks = data[key]
            # Handle num_valid
            if "num_valid" in data:
                nv = int(data["num_valid"])
                pred_inst_masks = pred_inst_masks[:nv]
        elif inst_path_npy.exists():
            pred_inst_masks = np.load(str(inst_path_npy))

        # Reshape flattened patch masks to pixel resolution
        if pred_inst_masks is not None and pred_inst_masks.ndim == 2:
            M, N = pred_inst_masks.shape
            h_p = image_size[0] // 16
            w_p = image_size[1] // 16
            if N == h_p * w_p:
                pred_inst_masks = pred_inst_masks.reshape(M, h_p, w_p)
                resized = []
                for m in pred_inst_masks:
                    m_img = Image.fromarray(m.astype(np.uint8) * 255)
                    m_img = m_img.resize((image_size[1], image_size[0]), Image.NEAREST)
                    resized.append(np.array(m_img) > 127)
                pred_inst_masks = np.stack(resized) if resized else None

        # Load GT
        gt_label_path = _find_gt_file(sem_path, semantic_dir, gt_dir, "_gtFine_labelIds.png")
        gt_inst_path = _find_gt_file(sem_path, semantic_dir, gt_dir, "_gtFine_instanceIds.png")
        if gt_label_path is None or gt_inst_path is None:
            continue

        gt_sem = _remap_labelids_to_trainids(np.array(Image.open(gt_label_path)))
        gt_inst_map = np.array(Image.open(gt_inst_path), dtype=np.int32)

        # Resize everything to evaluation size
        H, W = image_size
        if pred_sem.shape != (H, W):
            pred_sem = np.array(Image.fromarray(pred_sem).resize((W, H), Image.NEAREST))
        if gt_sem.shape != (H, W):
            gt_sem = np.array(Image.fromarray(gt_sem).resize((W, H), Image.NEAREST))
        if gt_inst_map.shape != (H, W):
            gt_inst_map = np.array(
                Image.fromarray(gt_inst_map.astype(np.int32)).resize((W, H), Image.NEAREST)
            )

        # Build predicted panoptic map
        pred_panoptic = np.zeros((H, W), dtype=np.int32)
        pred_segments = []
        next_id = 1

        # First: assign stuff regions (semantic class directly)
        for cls in stuff_ids:
            mask = pred_sem == cls
            if mask.sum() < 64:  # min stuff area
                continue
            seg_id = next_id
            next_id += 1
            pred_panoptic[mask] = seg_id
            pred_segments.append({"id": seg_id, "category_id": cls})

        # Then: overlay thing instances
        if pred_inst_masks is not None and pred_inst_masks.ndim == 3:
            # Resize masks
            for m_idx in range(pred_inst_masks.shape[0]):
                m = pred_inst_masks[m_idx]
                if m.shape != (H, W):
                    m = np.array(
                        Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
                    ) > 127
                else:
                    m = m.astype(bool)

                if m.sum() < 10:
                    continue

                # Determine thing class from majority semantic vote
                sem_in_mask = pred_sem[m]
                if len(sem_in_mask) == 0:
                    continue
                majority_cls = int(np.bincount(sem_in_mask[sem_in_mask < num_classes],
                                               minlength=num_classes).argmax())
                if majority_cls not in thing_ids:
                    continue

                seg_id = next_id
                next_id += 1
                pred_panoptic[m] = seg_id
                pred_segments.append({"id": seg_id, "category_id": majority_cls})

        # Build GT panoptic map
        gt_panoptic = np.zeros((H, W), dtype=np.int32)
        gt_segments = []
        gt_next_id = 1

        # GT stuff: regions where instanceId < 1000 and class is stuff
        for cls in stuff_ids:
            mask = gt_sem == cls
            if mask.sum() < 64:
                continue
            seg_id = gt_next_id
            gt_next_id += 1
            gt_panoptic[mask] = seg_id
            gt_segments.append({"id": seg_id, "category_id": cls})

        # GT things: from instanceIds
        gt_inst_unique = np.unique(gt_inst_map)
        for uid in gt_inst_unique:
            if uid < 1000:
                continue
            raw_cls = uid // 1000
            if raw_cls not in _CS_ID_TO_TRAIN:
                continue
            train_id = _CS_ID_TO_TRAIN[raw_cls]
            if train_id not in thing_ids:
                continue
            mask = gt_inst_map == uid
            if mask.sum() < 10:
                continue
            seg_id = gt_next_id
            gt_next_id += 1
            gt_panoptic[mask] = seg_id
            gt_segments.append({"id": seg_id, "category_id": train_id})

        # Compute PQ per class for this image
        for gt_seg in gt_segments:
            gt_id = gt_seg["id"]
            gt_cat = gt_seg["category_id"]
            if gt_cat not in all_classes:
                continue

            gt_mask = gt_panoptic == gt_id
            best_iou = 0.0
            best_pred_id = None

            for pred_seg in pred_segments:
                if pred_seg["category_id"] != gt_cat:
                    continue
                pred_mask = pred_panoptic == pred_seg["id"]
                intersection = np.sum(pred_mask & gt_mask)
                union = np.sum(pred_mask | gt_mask)
                if union == 0:
                    continue
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_pred_id = pred_seg["id"]

            if best_iou > 0.5:
                tp_per_class[gt_cat] += 1
                iou_sum_per_class[gt_cat] += best_iou
            else:
                fn_per_class[gt_cat] += 1

        # FP: unmatched predictions
        matched_pred_ids = set()
        for gt_seg in gt_segments:
            gt_cat = gt_seg["category_id"]
            gt_mask = gt_panoptic == gt_seg["id"]
            for pred_seg in pred_segments:
                if pred_seg["category_id"] != gt_cat:
                    continue
                pred_mask = pred_panoptic == pred_seg["id"]
                intersection = np.sum(pred_mask & gt_mask)
                union = np.sum(pred_mask | gt_mask)
                if union > 0 and intersection / union > 0.5:
                    matched_pred_ids.add(pred_seg["id"])

        for pred_seg in pred_segments:
            if pred_seg["id"] not in matched_pred_ids:
                cat = pred_seg["category_id"]
                if cat in all_classes:
                    fp_per_class[cat] += 1

        num_evaluated += 1

    # Compute PQ, SQ, RQ per class
    pq_per_class = {}
    for c in all_classes:
        tp = tp_per_class[c]
        fp = fp_per_class[c]
        fn = fn_per_class[c]
        iou_sum = iou_sum_per_class[c]

        if tp + 0.5 * fp + 0.5 * fn > 0:
            sq = iou_sum / (tp + 1e-8)
            rq = tp / (tp + 0.5 * fp + 0.5 * fn)
            pq = sq * rq
        else:
            sq = rq = pq = 0.0

        name = _CS_CLASS_NAMES[c] if c < len(_CS_CLASS_NAMES) else f"class_{c}"
        pq_per_class[name] = {"pq": pq, "sq": sq, "rq": rq, "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    # Aggregate
    thing_pqs = [v["pq"] for c, v in pq_per_class.items()
                 if _CS_CLASS_NAMES.index(c) in thing_ids and v["tp"] + v["fp"] + v["fn"] > 0]
    stuff_pqs = [v["pq"] for c, v in pq_per_class.items()
                 if _CS_CLASS_NAMES.index(c) in stuff_ids and v["tp"] + v["fp"] + v["fn"] > 0]
    all_pqs = thing_pqs + stuff_pqs

    overall_pq = float(np.mean(all_pqs)) if all_pqs else 0.0
    overall_sq = float(np.sum(iou_sum_per_class) / (np.sum(tp_per_class) + 1e-8))
    overall_rq = float(np.sum(tp_per_class) / (
        np.sum(tp_per_class) + 0.5 * np.sum(fp_per_class) + 0.5 * np.sum(fn_per_class) + 1e-8
    ))

    # Print results
    print(f"\n  Per-class PQ:")
    for name, vals in sorted(pq_per_class.items(), key=lambda x: x[1]["pq"], reverse=True):
        cls_idx = _CS_CLASS_NAMES.index(name) if name in _CS_CLASS_NAMES else -1
        kind = "T" if cls_idx in thing_ids else "S"
        print(f"    [{kind}] {name:15s}: PQ={vals['pq']:.4f} SQ={vals['sq']:.4f} "
              f"RQ={vals['rq']:.4f} (TP={vals['tp']} FP={vals['fp']} FN={vals['fn']})")

    pq_things = float(np.mean(thing_pqs)) if thing_pqs else 0.0
    pq_stuff = float(np.mean(stuff_pqs)) if stuff_pqs else 0.0

    print(f"\n  Overall PQ: {overall_pq:.4f} ({overall_pq * 100:.2f}%)")
    print(f"  PQ Things:  {pq_things:.4f} ({pq_things * 100:.2f}%)")
    print(f"  PQ Stuff:   {pq_stuff:.4f} ({pq_stuff * 100:.2f}%)")
    print(f"  Overall SQ: {overall_sq:.4f}")
    print(f"  Overall RQ: {overall_rq:.4f}")
    print(f"  Images evaluated: {num_evaluated}")

    return {
        "pq": overall_pq,
        "pq_things": pq_things,
        "pq_stuff": pq_stuff,
        "sq": overall_sq,
        "rq": overall_rq,
        "per_class_pq": pq_per_class,
        "num_images": num_evaluated,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pseudo-labels against Cityscapes GT"
    )
    parser.add_argument(
        "--semantic_dir", type=str, default=None,
        help="Directory with semantic pseudo-label PNGs"
    )
    parser.add_argument(
        "--instance_dir", type=str, default=None,
        help="Directory with instance pseudo-label NPZ/NPY files"
    )
    parser.add_argument(
        "--gt_dir", type=str, required=True,
        help="Directory with Cityscapes GT (gtFine/val)"
    )
    parser.add_argument(
        "--stuff_things_map", type=str, default=None,
        help="Path to stuff_things_map.json"
    )
    parser.add_argument(
        "--num_classes", type=int, default=19,
        help="Number of classes"
    )
    parser.add_argument(
        "--image_size", type=int, nargs=2, default=[512, 1024],
        help="(H, W) evaluation resolution"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path for results"
    )
    args = parser.parse_args()

    results = {}

    # 1. Semantic evaluation
    if args.semantic_dir:
        sem_results = evaluate_semantic(
            args.semantic_dir, args.gt_dir, args.num_classes
        )
        results["semantic"] = sem_results

    # 2. Instance evaluation
    if args.instance_dir:
        inst_results = evaluate_instances(
            args.instance_dir, args.gt_dir, tuple(args.image_size)
        )
        results["instance"] = inst_results

    # 3. Panoptic evaluation (needs both semantic + instance)
    if args.semantic_dir and args.instance_dir:
        stuff_things = None
        if args.stuff_things_map and os.path.exists(args.stuff_things_map):
            with open(args.stuff_things_map) as f:
                stuff_things = json.load(f)

        # Use semantic mapping from semantic evaluation
        sem_mapping = results.get("semantic", {}).get("mapping", None)

        pan_results = evaluate_panoptic(
            args.semantic_dir, args.instance_dir, args.gt_dir,
            stuff_things, args.num_classes, tuple(args.image_size),
            sem_mapping,
        )
        results["panoptic"] = pan_results

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    if "semantic" in results:
        print(f"  Semantic mIoU:    {results['semantic']['miou']*100:.2f}%")
        print(f"  Pixel Accuracy:   {results['semantic']['pixel_accuracy']*100:.2f}%")
    if "instance" in results:
        print(f"  Instance AR@100:  {results['instance']['ar_100']*100:.2f}%")
        print(f"  Instance AP@50:   {results['instance']['ap_50']*100:.2f}%")
        print(f"  Instance AP@75:   {results['instance']['ap_75']*100:.2f}%")
    if "panoptic" in results:
        print(f"  Panoptic PQ:      {results['panoptic']['pq']*100:.2f}%")
        print(f"  PQ Things:        {results['panoptic']['pq_things']*100:.2f}%")
        print(f"  PQ Stuff:         {results['panoptic']['pq_stuff']*100:.2f}%")

    # Save results
    output_path = args.output
    if output_path is None:
        base = args.semantic_dir or args.instance_dir
        output_path = str(Path(base) / "eval_comprehensive.json")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
