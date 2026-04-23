#!/usr/bin/env python3
"""Evaluate pseudo-label quality against Cityscapes GT before training.

Computes PQ, SQ, RQ, mIoU on training split using the standard 19-class
Cityscapes evaluation. Supports both raw cluster labels (with Hungarian
matching) and pre-mapped trainID labels.

Usage:
    python scripts/evaluate_pseudolabel_quality.py \
        --pseudo_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_depthpro_tau020 \
        --cityscapes_root ~/Desktop/datasets/cityscapes \
        --centroids_path ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80/kmeans_centroids.npz
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_CLASSES = 19
IGNORE_LABEL = 255

_CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}

_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

_STUFF_IDS = set(range(0, 11))
_THING_IDS = set(range(11, 19))


def remap_gt(gt_raw: np.ndarray) -> np.ndarray:
    """Remap Cityscapes labelIds to 19-class trainIDs."""
    out = np.full_like(gt_raw, IGNORE_LABEL, dtype=np.uint8)
    for raw_id, train_id in _CS_ID_TO_TRAIN.items():
        out[gt_raw == raw_id] = train_id
    return out


def compute_cluster_mapping(pseudo_dir: Path, gt_dir: Path,
                            num_clusters: int, split: str) -> np.ndarray:
    """Many-to-one cluster->trainID mapping via majority vote against GT.

    Unlike 1-to-1 Hungarian, this allows multiple clusters to map to the
    same trainID — essential for overclustered labels (e.g. k=80 -> 19 classes).

    Returns:
        LUT of shape (256,) mapping cluster_id -> trainID.
    """
    logger.info(f"Computing cluster->class mapping ({num_clusters} -> {NUM_CLASSES})...")
    counts = np.zeros((num_clusters, NUM_CLASSES), dtype=np.int64)

    gt_split = gt_dir / split
    for city_dir in sorted(gt_split.iterdir()):
        if not city_dir.is_dir():
            continue
        for gt_path in sorted(city_dir.glob("*_gtFine_labelIds.png")):
            stem = gt_path.name.replace("_gtFine_labelIds.png", "")
            sem_path = pseudo_dir / f"{stem}_leftImg8bit_semantic.png"
            if not sem_path.exists():
                sem_path = pseudo_dir / split / city_dir.name / f"{stem}_leftImg8bit_semantic.png"
            if not sem_path.exists():
                continue

            pred = np.array(Image.open(sem_path))
            gt = remap_gt(np.array(Image.open(gt_path)))

            if pred.shape != gt.shape:
                pred = np.array(
                    Image.fromarray(pred).resize(
                        (gt.shape[1], gt.shape[0]), Image.NEAREST
                    )
                )

            valid = (gt != IGNORE_LABEL) & (pred < num_clusters)
            p, g = pred[valid], gt[valid]
            # Vectorized 2D bincount
            idx = p.astype(np.int64) * NUM_CLASSES + g.astype(np.int64)
            flat_counts = np.bincount(idx, minlength=num_clusters * NUM_CLASSES)
            counts += flat_counts.reshape(num_clusters, NUM_CLASSES)

    # Many-to-one: each cluster maps to its most frequent GT class
    lut = np.full(256, IGNORE_LABEL, dtype=np.uint8)
    for cid in range(num_clusters):
        if counts[cid].sum() > 0:
            lut[cid] = counts[cid].argmax()

    matched = int(np.sum(lut[:num_clusters] != IGNORE_LABEL))
    logger.info(f"  Matched {matched}/{num_clusters} clusters")
    return lut


def compute_miou(confusion: np.ndarray) -> tuple:
    """Compute per-class IoU and mIoU from confusion matrix.

    Returns:
        (miou, per_class_iou)
    """
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / denom, 0.0)
    valid = denom > 0
    miou = float(iou[valid].mean()) if valid.any() else 0.0
    return miou, iou


def compute_pq(pred_pan: np.ndarray, gt_pan: np.ndarray,
               pred_cls: np.ndarray, gt_cls: np.ndarray) -> dict:
    """Compute PQ, SQ, RQ per-class.

    Args:
        pred_pan: (H, W) panoptic ID map (cls * 1000 + instance_id).
        gt_pan: (H, W) GT panoptic ID map.
        pred_cls: dict mapping panoptic_id -> class_id for predictions.
        gt_cls: dict mapping panoptic_id -> class_id for GT.

    Returns:
        Dict with per-class and aggregate PQ/SQ/RQ.
    """
    pq_per_class = {}
    for cls in range(NUM_CLASSES):
        pq_per_class[cls] = {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0}

    # Get all unique segment IDs
    pred_ids = set(np.unique(pred_pan)) - {0}
    gt_ids = set(np.unique(gt_pan)) - {0}

    matched_pred = set()
    matched_gt = set()

    # Match: for each GT segment, find best overlapping pred segment
    for gt_id in gt_ids:
        if gt_id not in gt_cls:
            continue
        cls = gt_cls[gt_id]
        gt_mask = gt_pan == gt_id

        # Find overlapping predictions
        overlapping = set(np.unique(pred_pan[gt_mask])) - {0}
        best_iou = 0.0
        best_pred = None

        for pred_id in overlapping:
            if pred_id not in pred_cls or pred_cls[pred_id] != cls:
                continue
            if pred_id in matched_pred:
                continue
            pred_mask = pred_pan == pred_id
            intersection = int((gt_mask & pred_mask).sum())
            union = int((gt_mask | pred_mask).sum())
            iou = intersection / max(union, 1)
            if iou > 0.5 and iou > best_iou:
                best_iou = iou
                best_pred = pred_id

        if best_pred is not None:
            pq_per_class[cls]["tp"] += 1
            pq_per_class[cls]["iou_sum"] += best_iou
            matched_pred.add(best_pred)
            matched_gt.add(gt_id)
        else:
            pq_per_class[cls]["fn"] += 1

    # FP: unmatched predictions
    for pred_id in pred_ids:
        if pred_id in matched_pred or pred_id not in pred_cls:
            continue
        cls = pred_cls[pred_id]
        if cls < NUM_CLASSES:
            pq_per_class[cls]["fp"] += 1

    return pq_per_class


def build_panoptic_map(semantic: np.ndarray, instance: np.ndarray,
                       stuff_ids: set, thing_ids: set) -> tuple:
    """Build panoptic ID map from semantic + instance labels.

    Returns:
        (pan_map, id_to_class) where pan_map has unique IDs per segment.
    """
    H, W = semantic.shape
    pan_map = np.zeros((H, W), dtype=np.int32)
    id_to_class = {}
    next_id = 1

    # Stuff: one segment per class (no instances)
    for cls in sorted(stuff_ids):
        mask = semantic == cls
        if not mask.any():
            continue
        pan_id = next_id
        next_id += 1
        pan_map[mask] = pan_id
        id_to_class[pan_id] = cls

    # Things: each instance is a segment
    for inst_id in np.unique(instance):
        if inst_id == 0:
            continue
        mask = instance == inst_id
        if not mask.any():
            continue
        # Majority class
        cls_vals = semantic[mask]
        valid = cls_vals[cls_vals < NUM_CLASSES]
        if len(valid) == 0:
            continue
        cls = int(np.bincount(valid, minlength=NUM_CLASSES).argmax())
        if cls not in thing_ids:
            continue
        pan_id = next_id
        next_id += 1
        pan_map[mask] = pan_id
        id_to_class[pan_id] = cls

    return pan_map, id_to_class


def build_gt_panoptic(label_path: Path, inst_path: Path) -> tuple:
    """Build GT panoptic map from Cityscapes gtFine files.

    Returns:
        (pan_map, id_to_class)
    """
    gt_label = np.array(Image.open(label_path))
    gt_sem = remap_gt(gt_label)
    gt_inst = np.array(Image.open(inst_path), dtype=np.int32)

    H, W = gt_sem.shape
    pan_map = np.zeros((H, W), dtype=np.int32)
    id_to_class = {}
    next_id = 1

    # Stuff
    for cls in sorted(_STUFF_IDS):
        mask = (gt_sem == cls) & (gt_inst < 1000)
        if not mask.any():
            continue
        pan_id = next_id
        next_id += 1
        pan_map[mask] = pan_id
        id_to_class[pan_id] = cls

    # Things
    for uid in np.unique(gt_inst):
        if uid < 1000:
            continue
        raw_cls = uid // 1000
        if raw_cls not in _CS_ID_TO_TRAIN:
            continue
        train_id = _CS_ID_TO_TRAIN[raw_cls]
        if train_id not in _THING_IDS:
            continue
        mask = gt_inst == uid
        if mask.sum() < 10:
            continue
        pan_id = next_id
        next_id += 1
        pan_map[mask] = pan_id
        id_to_class[pan_id] = train_id

    return pan_map, id_to_class


def main():
    parser = argparse.ArgumentParser(description="Evaluate pseudo-label quality")
    parser.add_argument("--pseudo_dir", type=str, required=True,
                        help="CUPS flat label directory to evaluate")
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--centroids_path", type=str, default=None,
                        help="Path to kmeans_centroids.npz (for cluster->class mapping)")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_clusters", type=int, default=80)
    parser.add_argument("--use_hungarian", action="store_true",
                        help="Force Hungarian matching (auto-detected if >19 unique values)")
    parser.add_argument("--baseline_dir", type=str, default=None,
                        help="Fallback directory for instance labels (e.g. baseline pseudo-labels)")
    args = parser.parse_args()

    pseudo_dir = Path(args.pseudo_dir).expanduser()
    cs_root = Path(args.cityscapes_root).expanduser()
    gt_dir = cs_root / "gtFine"

    # Determine cluster -> class mapping
    lut = None
    if args.centroids_path:
        data = np.load(args.centroids_path)
        lut = np.full(256, IGNORE_LABEL, dtype=np.uint8)
        c2c = data["cluster_to_class"].astype(np.uint8)
        for i, c in enumerate(c2c):
            lut[i] = c
        logger.info(f"Using cluster_to_class from centroids ({len(c2c)} clusters)")
    else:
        # Check if labels need Hungarian matching
        sample = list(pseudo_dir.rglob("*_semantic.png"))[:5]
        if not sample:
            logger.warning("No semantic labels found in %s", pseudo_dir)
            max_val = 0
        else:
            max_val = max(int(np.array(Image.open(p)).max()) for p in sample)
        if max_val >= NUM_CLASSES or args.use_hungarian:
            lut = compute_cluster_mapping(
                pseudo_dir, gt_dir, args.num_clusters, args.split
            )

    # Collect image pairs
    gt_split = gt_dir / args.split
    pairs = []
    for city_dir in sorted(gt_split.iterdir()):
        if not city_dir.is_dir():
            continue
        for gt_label_path in sorted(city_dir.glob("*_gtFine_labelIds.png")):
            stem = gt_label_path.name.replace("_gtFine_labelIds.png", "")
            # Search flat first, then nested
            sem_path = pseudo_dir / f"{stem}_leftImg8bit_semantic.png"
            inst_path = pseudo_dir / f"{stem}_leftImg8bit_instance.png"
            if not sem_path.exists():
                sem_path = pseudo_dir / args.split / city_dir.name / f"{stem}_leftImg8bit_semantic.png"
                inst_path = pseudo_dir / args.split / city_dir.name / f"{stem}_leftImg8bit_instance.png"
            # Fallback to baseline instance labels if variant has no instances
            if not inst_path.exists() and args.baseline_dir:
                baseline_dir = Path(args.baseline_dir)
                inst_path = baseline_dir / f"{stem}_leftImg8bit_instance.png"
                if not inst_path.exists():
                    inst_path = baseline_dir / args.split / city_dir.name / f"{stem}_leftImg8bit_instance.png"
            gt_inst_path = city_dir / f"{stem}_gtFine_instanceIds.png"
            if sem_path.exists() and inst_path.exists() and gt_inst_path.exists():
                pairs.append((sem_path, inst_path, gt_label_path, gt_inst_path))

    logger.info(f"Evaluating {len(pairs)} images from {args.split}")

    # Evaluate
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    pq_accum = {c: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0} for c in range(NUM_CLASSES)}
    total_ignore = 0
    total_pixels = 0

    t0 = time.time()
    for sem_path, inst_path, gt_label_path, gt_inst_path in tqdm(pairs, desc="Evaluating"):
        pred_sem = np.array(Image.open(sem_path))
        pred_inst = np.array(Image.open(inst_path))

        # Map clusters to trainIDs
        if lut is not None:
            pred_mapped = lut[pred_sem]
        else:
            pred_mapped = pred_sem.copy()

        # Resize to GT resolution
        gt_raw = np.array(Image.open(gt_label_path))
        target_h, target_w = gt_raw.shape[:2]
        if pred_mapped.shape != (target_h, target_w):
            pred_mapped = np.array(
                Image.fromarray(pred_mapped).resize((target_w, target_h), Image.NEAREST)
            )
            pred_inst = np.array(
                Image.fromarray(pred_inst).resize((target_w, target_h), Image.NEAREST)
            )

        gt_sem = remap_gt(gt_raw)

        # Ignore pixels
        total_pixels += int(pred_mapped.size)
        total_ignore += int((pred_mapped == IGNORE_LABEL).sum())

        # Confusion matrix for mIoU (vectorized)
        valid = (gt_sem != IGNORE_LABEL) & (pred_mapped != IGNORE_LABEL)
        if valid.any():
            g, p = gt_sem[valid].astype(np.int64), pred_mapped[valid].astype(np.int64)
            idx = g * NUM_CLASSES + p
            flat = np.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES)
            confusion += flat.reshape(NUM_CLASSES, NUM_CLASSES)

        # PQ
        pred_pan, pred_cls = build_panoptic_map(
            pred_mapped, pred_inst, _STUFF_IDS, _THING_IDS
        )
        gt_pan, gt_cls = build_gt_panoptic(gt_label_path, gt_inst_path)

        pq_result = compute_pq(pred_pan, gt_pan, pred_cls, gt_cls)
        for cls in range(NUM_CLASSES):
            for key in ["tp", "fp", "fn"]:
                pq_accum[cls][key] += pq_result[cls][key]
            pq_accum[cls]["iou_sum"] += pq_result[cls]["iou_sum"]

    elapsed = time.time() - t0

    # Compute metrics
    miou, per_iou = compute_miou(confusion)

    # PQ aggregation
    pq_all, sq_all, rq_all = [], [], []
    pq_stuff, pq_things = [], []

    logger.info(f"\n{'='*80}")
    logger.info(f"{'Class':>15s} | {'IoU':>6s} | {'PQ':>6s} | {'SQ':>6s} | {'RQ':>6s} | "
                f"{'TP':>5s} | {'FP':>5s} | {'FN':>5s}")
    logger.info("-" * 80)

    for cls in range(NUM_CLASSES):
        tp = pq_accum[cls]["tp"]
        fp = pq_accum[cls]["fp"]
        fn = pq_accum[cls]["fn"]
        iou_sum = pq_accum[cls]["iou_sum"]

        denom = tp + 0.5 * fp + 0.5 * fn
        pq = iou_sum / max(denom, 1e-8) if denom > 0 else 0.0
        sq = iou_sum / max(tp, 1e-8) if tp > 0 else 0.0
        rq = tp / max(denom, 1e-8) if denom > 0 else 0.0

        pq_all.append(pq)
        sq_all.append(sq)
        rq_all.append(rq)

        if cls in _STUFF_IDS:
            pq_stuff.append(pq)
        else:
            pq_things.append(pq)

        logger.info(f"{_CLASS_NAMES[cls]:>15s} | {per_iou[cls]*100:5.1f}% | "
                     f"{pq*100:5.1f}% | {sq*100:5.1f}% | {rq*100:5.1f}% | "
                     f"{tp:5d} | {fp:5d} | {fn:5d}")

    logger.info("-" * 80)
    mean_pq = 100 * np.mean(pq_all)
    mean_sq = 100 * np.mean(sq_all)
    mean_rq = 100 * np.mean(rq_all)
    mean_pq_stuff = 100 * np.mean(pq_stuff) if pq_stuff else 0
    mean_pq_things = 100 * np.mean(pq_things) if pq_things else 0
    ignore_pct = 100 * total_ignore / max(total_pixels, 1)

    logger.info(f"{'MEAN':>15s} | {miou*100:5.1f}% | {mean_pq:5.1f}% | "
                 f"{mean_sq:5.1f}% | {mean_rq:5.1f}%")
    logger.info(f"\n  PQ       = {mean_pq:.2f}%")
    logger.info(f"  PQ_stuff = {mean_pq_stuff:.2f}%")
    logger.info(f"  PQ_things= {mean_pq_things:.2f}%")
    logger.info(f"  mIoU     = {miou*100:.2f}%")
    logger.info(f"  Ignore   = {ignore_pct:.1f}% of pixels")
    logger.info(f"  Time     = {elapsed:.1f}s ({len(pairs)} images)")

    # Machine-readable summary
    print(f"\nSUMMARY;PQ={mean_pq:.2f};PQ_st={mean_pq_stuff:.2f};"
          f"PQ_th={mean_pq_things:.2f};mIoU={miou*100:.2f};"
          f"ignore={ignore_pct:.1f}")


if __name__ == "__main__":
    main()
