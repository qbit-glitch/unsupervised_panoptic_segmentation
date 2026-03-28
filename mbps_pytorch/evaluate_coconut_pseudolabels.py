#!/usr/bin/env python3
"""Evaluate pseudo-labels against COCONUT GT with Hungarian matching.

Computes:
  1. Semantic mIoU: Hungarian match k=80 clusters → COCO categories
  2. Panoptic Quality (PQ): segment-level matching with IoU > 0.5
  3. Per-class breakdown: PQ_stuff, PQ_things, per-category PQ/SQ/RQ

COCONUT GT format:
  - Panoptic PNGs: RGB encoded as segment_id = R + G*256 + B*65536
  - JSON annotations: segments_info with category_id per segment_id
  - 133 categories (80 things + 53 stuff)

Usage:
    python mbps_pytorch/evaluate_coconut_pseudolabels.py \
        --coconut_root /Users/qbit-glitch/Desktop/datasets/coconut \
        --split val
"""

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def read_label_png(path):
    """Read a label PNG that may be 8-bit (L) or 16-bit (I;16)."""
    img = Image.open(path)
    if img.mode == "I;16":
        return np.array(img, dtype=np.uint16)
    elif img.mode == "I":
        return np.array(img, dtype=np.int32)
    else:
        return np.array(img)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_coconut_annotation(coconut_root, split):
    """Load COCONUT annotation JSON and build lookup tables."""
    if split == "val":
        ann_path = Path(coconut_root) / "relabeled_coco_val.json"
        mask_dir = Path(coconut_root) / "relabeled_coco_val"
    elif split == "train":
        ann_path = Path(coconut_root) / "coconut_s.json"
        mask_dir = Path(coconut_root) / "coconut_s"
    else:
        raise ValueError(f"Unknown split: {split}")

    logger.info(f"Loading COCONUT annotations from {ann_path}...")
    with open(ann_path) as f:
        data = json.load(f)

    # Build category info
    cat_info = {}
    for cat in data["categories"]:
        cat_info[cat["id"]] = {
            "name": cat["name"],
            "isthing": cat.get("isthing", 0) == 1,
            "supercategory": cat.get("supercategory", ""),
        }

    # Build image_id → annotation mapping
    id_to_ann = {}
    for ann in data["annotations"]:
        id_to_ann[ann["image_id"]] = ann

    # Build image_id → image info
    id_to_img = {}
    for img in data["images"]:
        id_to_img[img["id"]] = img

    logger.info(f"Loaded {len(data['images'])} images, {len(cat_info)} categories "
                f"({sum(1 for c in cat_info.values() if c['isthing'])} things, "
                f"{sum(1 for c in cat_info.values() if not c['isthing'])} stuff)")

    return data, mask_dir, cat_info, id_to_ann, id_to_img


def decode_panoptic_png(png_path):
    """Decode COCO panoptic PNG: RGB → segment_id map."""
    img = np.array(Image.open(png_path))
    if img.ndim == 3:
        segment_ids = (img[:, :, 0].astype(np.int32) +
                       img[:, :, 1].astype(np.int32) * 256 +
                       img[:, :, 2].astype(np.int32) * 65536)
    else:
        segment_ids = img.astype(np.int32)
    return segment_ids


def build_gt_category_map(segment_ids, segments_info):
    """Convert segment_id map → per-pixel category_id map."""
    cat_map = np.zeros_like(segment_ids, dtype=np.int32)
    seg_id_to_cat = {}
    seg_id_to_isthing = {}
    for seg in segments_info:
        seg_id_to_cat[seg["id"]] = seg["category_id"]
        seg_id_to_isthing[seg["id"]] = seg.get("isthing", 0)

    for seg_id, cat_id in seg_id_to_cat.items():
        mask = segment_ids == seg_id
        cat_map[mask] = cat_id

    return cat_map, seg_id_to_cat, seg_id_to_isthing


# ═══════════════════════════════════════════════════════════════════
# Semantic Evaluation (mIoU via Hungarian Matching)
# ═══════════════════════════════════════════════════════════════════

def compute_semantic_miou(coconut_root, split, semantic_dir, num_clusters=80):
    """Compute mIoU by Hungarian matching cluster IDs to COCO categories."""
    logger.info("=" * 60)
    logger.info("Semantic Evaluation (mIoU via Hungarian Matching)")
    logger.info("=" * 60)

    data, mask_dir, cat_info, id_to_ann, id_to_img = load_coconut_annotation(
        coconut_root, split)

    semantic_dir = Path(semantic_dir)

    # Collect all unique GT category IDs
    all_gt_cats = sorted(cat_info.keys())
    num_gt_cats = max(all_gt_cats) + 1  # max category ID + 1 for indexing
    logger.info(f"GT categories: {len(all_gt_cats)} (max ID: {max(all_gt_cats)})")

    # Build confusion matrix: (num_clusters, num_gt_cats)
    confusion = np.zeros((num_clusters, num_gt_cats), dtype=np.int64)

    processed = 0
    for img_info in tqdm(data["images"], desc="Building confusion matrix"):
        image_id = img_info["id"]
        if image_id not in id_to_ann:
            continue

        # Load predicted semantic labels
        pred_path = semantic_dir / f"{image_id:012d}.png"
        if not pred_path.exists():
            continue

        pred = read_label_png(pred_path)

        # Load GT panoptic
        ann = id_to_ann[image_id]
        gt_png_path = mask_dir / ann["file_name"]
        if not gt_png_path.exists():
            continue

        segment_ids = decode_panoptic_png(gt_png_path)
        gt_cat_map, _, _ = build_gt_category_map(segment_ids, ann["segments_info"])

        # Resize pred to match GT if needed
        if pred.shape != gt_cat_map.shape:
            from scipy.ndimage import zoom
            scale_h = gt_cat_map.shape[0] / pred.shape[0]
            scale_w = gt_cat_map.shape[1] / pred.shape[1]
            pred = zoom(pred.astype(np.float32), (scale_h, scale_w), order=0).astype(np.int32)

        # Accumulate confusion matrix using vectorized indexing
        valid = gt_cat_map > 0  # skip void/unlabeled
        pred_valid = pred[valid].astype(np.int64)
        gt_valid = gt_cat_map[valid].astype(np.int64)

        # Filter to valid ranges
        mask = (pred_valid < num_clusters) & (gt_valid < num_gt_cats)
        pred_valid = pred_valid[mask]
        gt_valid = gt_valid[mask]

        # Vectorized confusion matrix update
        np.add.at(confusion, (pred_valid, gt_valid), 1)

        processed += 1

    logger.info(f"Processed {processed} images")

    # Hungarian matching: maximize overlap
    # Only consider GT categories that actually appear
    active_gt_cats = [c for c in all_gt_cats if confusion[:, c].sum() > 0]
    logger.info(f"Active GT categories: {len(active_gt_cats)}")

    # Build reduced cost matrix for Hungarian
    cost = np.zeros((num_clusters, len(active_gt_cats)), dtype=np.int64)
    for j, cat_id in enumerate(active_gt_cats):
        cost[:, j] = confusion[:, cat_id]

    row_ind, col_ind = linear_sum_assignment(-cost)  # maximize

    # Build cluster → category mapping
    cluster_to_cat = {}
    for r, c in zip(row_ind, col_ind):
        cat_id = active_gt_cats[c]
        overlap = cost[r, c]
        if overlap > 0:
            cluster_to_cat[r] = cat_id

    logger.info(f"Mapped {len(cluster_to_cat)} clusters to GT categories")

    # Compute per-category IoU
    # Re-iterate to compute IoU with the mapping
    per_cat_tp = defaultdict(int)
    per_cat_fp = defaultdict(int)
    per_cat_fn = defaultdict(int)

    for img_info in tqdm(data["images"], desc="Computing mIoU"):
        image_id = img_info["id"]
        if image_id not in id_to_ann:
            continue

        pred_path = semantic_dir / f"{image_id:012d}.png"
        if not pred_path.exists():
            continue

        pred = read_label_png(pred_path)
        ann = id_to_ann[image_id]
        gt_png_path = mask_dir / ann["file_name"]
        if not gt_png_path.exists():
            continue

        segment_ids = decode_panoptic_png(gt_png_path)
        gt_cat_map, _, _ = build_gt_category_map(segment_ids, ann["segments_info"])

        if pred.shape != gt_cat_map.shape:
            from scipy.ndimage import zoom
            scale_h = gt_cat_map.shape[0] / pred.shape[0]
            scale_w = gt_cat_map.shape[1] / pred.shape[1]
            pred = zoom(pred.astype(np.float32), (scale_h, scale_w), order=0).astype(np.int32)

        # Remap predictions to GT category space
        pred_mapped = np.zeros_like(pred, dtype=np.int32)
        for cluster_id, cat_id in cluster_to_cat.items():
            pred_mapped[pred == cluster_id] = cat_id

        valid = gt_cat_map > 0
        for cat_id in active_gt_cats:
            pred_c = (pred_mapped == cat_id) & valid
            gt_c = (gt_cat_map == cat_id) & valid
            per_cat_tp[cat_id] += int((pred_c & gt_c).sum())
            per_cat_fp[cat_id] += int((pred_c & ~gt_c).sum())
            per_cat_fn[cat_id] += int((~pred_c & gt_c).sum())

    # Compute IoU per category
    ious = {}
    for cat_id in active_gt_cats:
        tp = per_cat_tp[cat_id]
        fp = per_cat_fp[cat_id]
        fn = per_cat_fn[cat_id]
        union = tp + fp + fn
        if union > 0:
            ious[cat_id] = tp / union
        else:
            ious[cat_id] = float('nan')

    # Print results
    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    miou = np.mean(valid_ious) if valid_ious else 0.0

    thing_ious = [ious[c] for c in active_gt_cats
                  if cat_info[c]["isthing"] and not np.isnan(ious.get(c, float('nan')))]
    stuff_ious = [ious[c] for c in active_gt_cats
                  if not cat_info[c]["isthing"] and not np.isnan(ious.get(c, float('nan')))]

    miou_things = np.mean(thing_ious) if thing_ious else 0.0
    miou_stuff = np.mean(stuff_ious) if stuff_ious else 0.0

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Semantic mIoU Results")
    logger.info(f"{'=' * 60}")
    logger.info(f"  mIoU (all):    {miou * 100:.2f}%  ({len(valid_ious)} categories)")
    logger.info(f"  mIoU (things): {miou_things * 100:.2f}%  ({len(thing_ious)} categories)")
    logger.info(f"  mIoU (stuff):  {miou_stuff * 100:.2f}%  ({len(stuff_ious)} categories)")

    # Top and bottom categories
    sorted_cats = sorted(ious.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -1,
                         reverse=True)
    logger.info(f"\nTop 15 categories by IoU:")
    for cat_id, iou in sorted_cats[:15]:
        if np.isnan(iou):
            continue
        name = cat_info[cat_id]["name"]
        kind = "thing" if cat_info[cat_id]["isthing"] else "stuff"
        logger.info(f"  {cat_id:4d} {name:25s} ({kind:5s}): IoU = {iou * 100:.1f}%")

    logger.info(f"\nBottom 15 categories by IoU:")
    valid_sorted = [(c, i) for c, i in sorted_cats if not np.isnan(i)]
    for cat_id, iou in valid_sorted[-15:]:
        name = cat_info[cat_id]["name"]
        kind = "thing" if cat_info[cat_id]["isthing"] else "stuff"
        logger.info(f"  {cat_id:4d} {name:25s} ({kind:5s}): IoU = {iou * 100:.1f}%")

    # Print cluster → category mapping
    logger.info(f"\nCluster → Category mapping:")
    for cluster_id in sorted(cluster_to_cat.keys()):
        cat_id = cluster_to_cat[cluster_id]
        name = cat_info[cat_id]["name"]
        iou = ious.get(cat_id, 0)
        logger.info(f"  Cluster {cluster_id:3d} → {cat_id:4d} {name:25s} (IoU={iou*100:.1f}%)")

    return {
        "miou": miou, "miou_things": miou_things, "miou_stuff": miou_stuff,
        "per_category_iou": {str(k): v for k, v in ious.items()},
        "cluster_to_category": {str(k): v for k, v in cluster_to_cat.items()},
        "num_matched_categories": len(valid_ious),
    }


# ═══════════════════════════════════════════════════════════════════
# Panoptic Quality (PQ) Evaluation
# ═══════════════════════════════════════════════════════════════════

def compute_panoptic_quality(coconut_root, split, panoptic_dir, semantic_dir,
                              cluster_to_cat, num_clusters=80):
    """Compute PQ against COCONUT GT using cluster→category mapping."""
    logger.info("=" * 60)
    logger.info("Panoptic Quality (PQ) Evaluation")
    logger.info("=" * 60)

    data, mask_dir, cat_info, id_to_ann, id_to_img = load_coconut_annotation(
        coconut_root, split)

    panoptic_dir = Path(panoptic_dir)
    semantic_dir = Path(semantic_dir)

    # Per-category PQ accumulators
    per_cat_tp = defaultdict(int)
    per_cat_iou_sum = defaultdict(float)
    per_cat_fp = defaultdict(int)
    per_cat_fn = defaultdict(int)

    processed = 0

    for img_info in tqdm(data["images"], desc="Computing PQ"):
        image_id = img_info["id"]
        if image_id not in id_to_ann:
            continue

        # Load predicted panoptic
        pred_pan_path = panoptic_dir / f"{image_id:012d}.npy"
        if not pred_pan_path.exists():
            continue

        pred_pan = np.load(str(pred_pan_path))  # (H, W) int32: cluster_id * 1000 + inst_id

        # Load GT panoptic
        ann = id_to_ann[image_id]
        gt_png_path = mask_dir / ann["file_name"]
        if not gt_png_path.exists():
            continue

        gt_segment_ids = decode_panoptic_png(gt_png_path)
        gt_H, gt_W = gt_segment_ids.shape

        # Resize pred to GT resolution if needed
        if pred_pan.shape != (gt_H, gt_W):
            pred_pan = np.array(
                Image.fromarray(pred_pan.astype(np.int32)).resize(
                    (gt_W, gt_H), Image.NEAREST
                )
            )

        # Build GT segments: segment_id → (category_id, isthing, mask)
        gt_segments = {}
        for seg in ann["segments_info"]:
            seg_id = seg["id"]
            cat_id = seg["category_id"]
            isthing = seg.get("isthing", 0)
            mask = gt_segment_ids == seg_id
            if mask.sum() > 0:
                gt_segments[seg_id] = {
                    "category_id": cat_id, "isthing": isthing, "mask": mask
                }

        # Build predicted segments: pan_id → (mapped_category_id, mask)
        pred_segments = {}
        unique_pred_ids = np.unique(pred_pan)
        for pan_id in unique_pred_ids:
            if pan_id == 0:
                continue
            cluster_id = int(pan_id) // 1000
            inst_id = int(pan_id) % 1000

            # Map cluster to GT category
            mapped_cat = cluster_to_cat.get(cluster_id, None)
            if mapped_cat is None:
                continue

            mask = pred_pan == pan_id
            if mask.sum() > 0:
                isthing = cat_info.get(mapped_cat, {}).get("isthing", False)
                pred_segments[int(pan_id)] = {
                    "category_id": mapped_cat,
                    "isthing": 1 if isthing else 0,
                    "mask": mask,
                }

        # Match predicted segments to GT segments
        matched_gt = set()
        matched_pred = set()

        # For each predicted segment, find best matching GT segment (same category, IoU > 0.5)
        for pred_id, pred_seg in pred_segments.items():
            best_iou = 0
            best_gt_id = None

            for gt_id, gt_seg in gt_segments.items():
                if gt_id in matched_gt:
                    continue
                if pred_seg["category_id"] != gt_seg["category_id"]:
                    continue

                # Compute IoU
                intersection = (pred_seg["mask"] & gt_seg["mask"]).sum()
                union = (pred_seg["mask"] | gt_seg["mask"]).sum()
                if union == 0:
                    continue
                iou = intersection / union

                if iou > best_iou:
                    best_iou = iou
                    best_gt_id = gt_id

            if best_iou > 0.5 and best_gt_id is not None:
                cat_id = pred_seg["category_id"]
                per_cat_tp[cat_id] += 1
                per_cat_iou_sum[cat_id] += best_iou
                matched_gt.add(best_gt_id)
                matched_pred.add(pred_id)

        # Count FP (unmatched predictions)
        for pred_id, pred_seg in pred_segments.items():
            if pred_id not in matched_pred:
                per_cat_fp[pred_seg["category_id"]] += 1

        # Count FN (unmatched GT)
        for gt_id, gt_seg in gt_segments.items():
            if gt_id not in matched_gt:
                per_cat_fn[gt_seg["category_id"]] += 1

        processed += 1

    logger.info(f"Processed {processed} images")

    # Compute per-category PQ, SQ, RQ
    all_cats = set(list(per_cat_tp.keys()) + list(per_cat_fp.keys()) + list(per_cat_fn.keys()))

    results = {}
    thing_pqs, stuff_pqs = [], []
    thing_sqs, stuff_sqs = [], []
    thing_rqs, stuff_rqs = [], []

    for cat_id in sorted(all_cats):
        tp = per_cat_tp[cat_id]
        fp = per_cat_fp[cat_id]
        fn = per_cat_fn[cat_id]
        iou_sum = per_cat_iou_sum[cat_id]

        sq = iou_sum / tp if tp > 0 else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
        pq = sq * rq

        isthing = cat_info.get(cat_id, {}).get("isthing", False)
        results[cat_id] = {
            "pq": pq, "sq": sq, "rq": rq,
            "tp": tp, "fp": fp, "fn": fn,
            "isthing": isthing,
        }

        if isthing:
            thing_pqs.append(pq)
            thing_sqs.append(sq)
            thing_rqs.append(rq)
        else:
            stuff_pqs.append(pq)
            stuff_sqs.append(sq)
            stuff_rqs.append(rq)

    all_pqs = thing_pqs + stuff_pqs
    pq_all = np.mean(all_pqs) if all_pqs else 0.0
    sq_all = np.mean(thing_sqs + stuff_sqs) if (thing_sqs + stuff_sqs) else 0.0
    rq_all = np.mean(thing_rqs + stuff_rqs) if (thing_rqs + stuff_rqs) else 0.0

    pq_things = np.mean(thing_pqs) if thing_pqs else 0.0
    sq_things = np.mean(thing_sqs) if thing_sqs else 0.0
    rq_things = np.mean(thing_rqs) if thing_rqs else 0.0

    pq_stuff = np.mean(stuff_pqs) if stuff_pqs else 0.0
    sq_stuff = np.mean(stuff_sqs) if stuff_sqs else 0.0
    rq_stuff = np.mean(stuff_rqs) if stuff_rqs else 0.0

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Panoptic Quality Results")
    logger.info(f"{'=' * 60}")
    logger.info(f"           PQ      SQ      RQ     #cats")
    logger.info(f"  All:   {pq_all*100:5.1f}   {sq_all*100:5.1f}   {rq_all*100:5.1f}   {len(all_pqs)}")
    logger.info(f"  Things:{pq_things*100:5.1f}   {sq_things*100:5.1f}   {rq_things*100:5.1f}   {len(thing_pqs)}")
    logger.info(f"  Stuff: {pq_stuff*100:5.1f}   {sq_stuff*100:5.1f}   {rq_stuff*100:5.1f}   {len(stuff_pqs)}")

    # Top categories by PQ
    sorted_results = sorted(results.items(), key=lambda x: x[1]["pq"], reverse=True)
    logger.info(f"\nTop 20 categories by PQ:")
    for cat_id, r in sorted_results[:20]:
        name = cat_info.get(cat_id, {}).get("name", f"cat_{cat_id}")
        kind = "thing" if r["isthing"] else "stuff"
        logger.info(f"  {cat_id:4d} {name:25s} ({kind:5s}): "
                     f"PQ={r['pq']*100:5.1f}  SQ={r['sq']*100:5.1f}  RQ={r['rq']*100:5.1f}  "
                     f"TP={r['tp']:5d}  FP={r['fp']:5d}  FN={r['fn']:5d}")

    # Summary stats
    total_tp = sum(r["tp"] for r in results.values())
    total_fp = sum(r["fp"] for r in results.values())
    total_fn = sum(r["fn"] for r in results.values())
    logger.info(f"\nTotal: TP={total_tp}, FP={total_fp}, FN={total_fn}")

    return {
        "pq": pq_all, "sq": sq_all, "rq": rq_all,
        "pq_things": pq_things, "sq_things": sq_things, "rq_things": rq_things,
        "pq_stuff": pq_stuff, "sq_stuff": sq_stuff, "rq_stuff": rq_stuff,
        "num_thing_cats": len(thing_pqs),
        "num_stuff_cats": len(stuff_pqs),
        "per_category": {str(k): v for k, v in results.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate COCONUT pseudo-labels with Hungarian matching"
    )
    parser.add_argument("--coconut_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--k", type=int, default=80)
    parser.add_argument("--semantic_subdir", type=str, default=None,
                        help="Semantic pseudo-label subdir (default: pseudo_semantic_k{k})")
    parser.add_argument("--panoptic_subdir", type=str, default="pseudo_panoptic",
                        help="Panoptic pseudo-label subdir")
    parser.add_argument("--skip_pq", action="store_true",
                        help="Skip PQ evaluation (only do semantic mIoU)")

    args = parser.parse_args()

    coconut_root = Path(args.coconut_root)
    if args.semantic_subdir is None:
        args.semantic_subdir = f"pseudo_semantic_k{args.k}"

    semantic_dir = coconut_root / args.semantic_subdir / args.split
    panoptic_dir = coconut_root / args.panoptic_subdir / args.split

    t0 = time.time()

    # Step 1: Semantic mIoU
    semantic_results = compute_semantic_miou(
        coconut_root=str(coconut_root),
        split=args.split,
        semantic_dir=str(semantic_dir),
        num_clusters=args.k,
    )

    # Step 2: Panoptic Quality
    if not args.skip_pq:
        cluster_to_cat = {int(k): v for k, v in
                          semantic_results["cluster_to_category"].items()}

        pq_results = compute_panoptic_quality(
            coconut_root=str(coconut_root),
            split=args.split,
            panoptic_dir=str(panoptic_dir),
            semantic_dir=str(semantic_dir),
            cluster_to_cat=cluster_to_cat,
            num_clusters=args.k,
        )
    else:
        pq_results = None

    elapsed = time.time() - t0

    # Save results
    output = {
        "semantic": semantic_results,
        "panoptic": pq_results,
        "elapsed_seconds": round(elapsed, 1),
        "config": {
            "k": args.k,
            "split": args.split,
            "semantic_dir": str(semantic_dir),
            "panoptic_dir": str(panoptic_dir),
        }
    }

    results_path = coconut_root / f"eval_results_{args.split}.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")
    logger.info(f"Total evaluation time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
