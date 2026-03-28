#!/usr/bin/env python3
"""Evaluate trained Cityscapes mobile model on COCONUT (COCO panoptic).

Cross-dataset evaluation: runs RepViT-M0.9 + BiFPN (19-class Cityscapes model)
on COCONUT val images, then evaluates via Hungarian matching against 133 COCO categories.

Pipeline:
  1. Load checkpoint (19-class semantic head)
  2. Run inference on COCO val2017 images (5000 images)
  3. Save semantic predictions (class IDs 0-18) as PNGs
  4. Generate panoptic maps via connected components
  5. Evaluate against COCONUT GT via Hungarian matching (19 → 133)

Usage:
    python mbps_pytorch/evaluate_mobile_on_coconut.py \
        --checkpoint checkpoints/repvit_m0_9_bifpn/best.pth \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --coconut_root /Users/qbit-glitch/Desktop/datasets/coconut \
        --output_dir /Users/qbit-glitch/Desktop/datasets/coconut/mobile_preds
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import model class from train_mobile_panoptic
from mbps_pytorch.train_mobile_panoptic import (
    MobilePanopticModel,
    _STUFF_IDS,
    _THING_IDS,
    _CS_CLASS_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# COCONUT GT Loading (from evaluate_coconut_pseudolabels.py)
# ═══════════════════════════════════════════════════════════════════

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

    cat_info = {}
    for cat in data["categories"]:
        cat_info[cat["id"]] = {
            "name": cat["name"],
            "isthing": cat.get("isthing", 0) == 1,
            "supercategory": cat.get("supercategory", ""),
        }

    id_to_ann = {}
    for ann in data["annotations"]:
        id_to_ann[ann["image_id"]] = ann

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
    for seg in segments_info:
        mask = segment_ids == seg["id"]
        cat_map[mask] = seg["category_id"]
    return cat_map


# ═══════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════

def load_model(checkpoint_path, backbone="repvit_m0_9.dist_450e_in1k",
               num_classes=19, fpn_dim=128, fpn_type="bifpn", device="cpu"):
    """Load trained MobilePanopticModel from checkpoint."""
    model = MobilePanopticModel(
        backbone_name=backbone,
        num_classes=num_classes,
        fpn_dim=fpn_dim,
        pretrained=False,  # we're loading weights
        instance_head="none",
        fpn_type=fpn_type,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    if "model_state_dict" in ckpt:
        epoch = ckpt.get("epoch", "?")
        metrics = ckpt.get("metrics", {})
        logger.info(f"Loaded checkpoint from epoch {epoch}")
        if metrics:
            logger.info(f"  Cityscapes metrics: {metrics}")

    return model


def run_inference(model, coco_root, output_dir, device, split="val",
                  batch_size=8, num_workers=4, input_size=(512, 1024)):
    """Run model inference on COCO val images, save semantic + panoptic predictions."""
    img_dir = Path(coco_root) / f"{split}2017"
    sem_dir = Path(output_dir) / "semantic" / split
    pan_dir = Path(output_dir) / "panoptic" / split
    sem_dir.mkdir(parents=True, exist_ok=True)
    pan_dir.mkdir(parents=True, exist_ok=True)

    # Get all image paths
    img_paths = sorted(img_dir.glob("*.jpg"))
    logger.info(f"Found {len(img_paths)} images in {img_dir}")

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    processed = 0
    t0 = time.time()

    with torch.no_grad():
        for i in tqdm(range(0, len(img_paths), batch_size), desc="Inference"):
            batch_paths = img_paths[i:i + batch_size]
            batch_imgs = []
            batch_sizes = []

            for img_path in batch_paths:
                img = Image.open(img_path).convert("RGB")
                orig_w, orig_h = img.size
                batch_sizes.append((orig_h, orig_w))

                # Resize to model input size
                img_resized = img.resize((input_size[1], input_size[0]), Image.BILINEAR)
                img_t = torch.from_numpy(np.array(img_resized)).float() / 255.0
                img_t = img_t.permute(2, 0, 1)  # HWC → CHW
                batch_imgs.append(img_t)

            batch_tensor = torch.stack(batch_imgs).to(device)
            batch_tensor = (batch_tensor - mean) / std

            out = model(batch_tensor)
            logits = out["logits"]  # (B, 19, H', W')

            for j, img_path in enumerate(batch_paths):
                image_id = int(img_path.stem)
                orig_h, orig_w = batch_sizes[j]

                # Upsample to original resolution
                logit = logits[j:j+1]
                logit_up = F.interpolate(
                    logit, size=(orig_h, orig_w),
                    mode='bilinear', align_corners=False
                )
                pred = logit_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                # Save semantic prediction as PNG
                sem_path = sem_dir / f"{image_id:012d}.png"
                Image.fromarray(pred).save(str(sem_path))

                # Generate panoptic map via connected components
                pan_map, segments = build_panoptic_cc(pred, cc_min_area=50)

                # Save panoptic as NPY (segment_id encoding: class_id * 1000 + inst_id)
                pan_encoded = encode_panoptic(pred, pan_map, segments)
                pan_path = pan_dir / f"{image_id:012d}.npy"
                np.save(str(pan_path), pan_encoded)

                processed += 1

    elapsed = time.time() - t0
    logger.info(f"Inference complete: {processed} images in {elapsed:.1f}s "
                f"({processed/elapsed:.1f} img/s)")

    return str(sem_dir), str(pan_dir)


def build_panoptic_cc(pred_sem, cc_min_area=50):
    """Build panoptic map from semantic prediction via connected components."""
    H, W = pred_sem.shape
    pan_map = np.zeros((H, W), dtype=np.int32)
    segments = {}  # seg_id → class_id
    nxt = 1

    # Stuff: one segment per class
    for cls in _STUFF_IDS:
        mask = pred_sem == cls
        if mask.sum() < 64:
            continue
        pan_map[mask] = nxt
        segments[nxt] = cls
        nxt += 1

    # Things: connected components per class
    for cls in _THING_IDS:
        cls_mask = pred_sem == cls
        if cls_mask.sum() < cc_min_area:
            continue
        labeled, n_cc = ndimage.label(cls_mask)
        for comp in range(1, n_cc + 1):
            cmask = labeled == comp
            if cmask.sum() < cc_min_area:
                continue
            pan_map[cmask] = nxt
            segments[nxt] = cls
            nxt += 1

    return pan_map, segments


def encode_panoptic(pred_sem, pan_map, segments):
    """Encode panoptic map as class_id * 1000 + instance_id for evaluation."""
    H, W = pred_sem.shape
    encoded = np.zeros((H, W), dtype=np.int32)

    for seg_id, cls_id in segments.items():
        mask = pan_map == seg_id
        # For stuff: instance_id = 0. For things: instance_id = seg_id
        if cls_id in _STUFF_IDS:
            encoded[mask] = cls_id * 1000
        else:
            encoded[mask] = cls_id * 1000 + seg_id

    return encoded


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate_semantic(coconut_root, split, sem_dir, num_classes=19):
    """Evaluate semantic predictions via Hungarian matching (19 → 133 COCO categories)."""
    logger.info("=" * 60)
    logger.info("Semantic Evaluation (mIoU via Hungarian Matching)")
    logger.info("=" * 60)

    data, mask_dir, cat_info, id_to_ann, id_to_img = load_coconut_annotation(
        coconut_root, split)

    sem_dir = Path(sem_dir)
    all_gt_cats = sorted(cat_info.keys())
    num_gt_cats = max(all_gt_cats) + 1

    # Build confusion matrix: (num_classes, num_gt_cats)
    confusion = np.zeros((num_classes, num_gt_cats), dtype=np.int64)

    processed = 0
    for img_info in tqdm(data["images"], desc="Building confusion matrix"):
        image_id = img_info["id"]
        if image_id not in id_to_ann:
            continue

        pred_path = sem_dir / f"{image_id:012d}.png"
        if not pred_path.exists():
            continue

        pred = np.array(Image.open(pred_path))

        ann = id_to_ann[image_id]
        gt_png_path = mask_dir / ann["file_name"]
        if not gt_png_path.exists():
            continue

        gt_segment_ids = decode_panoptic_png(gt_png_path)
        gt_cat_map = build_gt_category_map(gt_segment_ids, ann["segments_info"])

        # Resize pred to GT resolution if needed
        if pred.shape != gt_cat_map.shape:
            pred = np.array(
                Image.fromarray(pred).resize(
                    (gt_cat_map.shape[1], gt_cat_map.shape[0]), Image.NEAREST
                )
            )

        valid = gt_cat_map > 0
        pred_valid = pred[valid].astype(np.int64)
        gt_valid = gt_cat_map[valid].astype(np.int64)

        mask = (pred_valid < num_classes) & (gt_valid < num_gt_cats)
        pred_valid = pred_valid[mask]
        gt_valid = gt_valid[mask]

        np.add.at(confusion, (pred_valid, gt_valid), 1)
        processed += 1

    logger.info(f"Processed {processed} images")

    # Hungarian matching
    active_gt_cats = [c for c in all_gt_cats if confusion[:, c].sum() > 0]
    logger.info(f"Active GT categories: {len(active_gt_cats)}")

    cost = np.zeros((num_classes, len(active_gt_cats)), dtype=np.int64)
    for j, cat_id in enumerate(active_gt_cats):
        cost[:, j] = confusion[:, cat_id]

    row_ind, col_ind = linear_sum_assignment(-cost)

    cluster_to_cat = {}
    for r, c in zip(row_ind, col_ind):
        cat_id = active_gt_cats[c]
        overlap = cost[r, c]
        if overlap > 0:
            cluster_to_cat[r] = cat_id

    logger.info(f"Mapped {len(cluster_to_cat)}/{num_classes} classes to COCO categories:")
    for cls_id in sorted(cluster_to_cat.keys()):
        cat_id = cluster_to_cat[cls_id]
        cs_name = _CS_CLASS_NAMES[cls_id]
        coco_name = cat_info[cat_id]["name"]
        coco_kind = "thing" if cat_info[cat_id]["isthing"] else "stuff"
        overlap_pct = cost[cls_id, active_gt_cats.index(cat_id)]
        logger.info(f"  CS:{cls_id:2d} ({cs_name:15s}) → COCO:{cat_id:4d} ({coco_name:25s}, {coco_kind})")

    # Compute per-category IoU
    per_cat_tp = defaultdict(int)
    per_cat_fp = defaultdict(int)
    per_cat_fn = defaultdict(int)

    for img_info in tqdm(data["images"], desc="Computing mIoU"):
        image_id = img_info["id"]
        if image_id not in id_to_ann:
            continue

        pred_path = sem_dir / f"{image_id:012d}.png"
        if not pred_path.exists():
            continue

        pred = np.array(Image.open(pred_path))
        ann = id_to_ann[image_id]
        gt_png_path = mask_dir / ann["file_name"]
        if not gt_png_path.exists():
            continue

        gt_segment_ids = decode_panoptic_png(gt_png_path)
        gt_cat_map = build_gt_category_map(gt_segment_ids, ann["segments_info"])

        if pred.shape != gt_cat_map.shape:
            pred = np.array(
                Image.fromarray(pred).resize(
                    (gt_cat_map.shape[1], gt_cat_map.shape[0]), Image.NEAREST
                )
            )

        # Remap predictions
        pred_mapped = np.zeros_like(pred, dtype=np.int32)
        for cls_id, cat_id in cluster_to_cat.items():
            pred_mapped[pred == cls_id] = cat_id

        valid = gt_cat_map > 0
        for cat_id in active_gt_cats:
            pred_c = (pred_mapped == cat_id) & valid
            gt_c = (gt_cat_map == cat_id) & valid
            per_cat_tp[cat_id] += int((pred_c & gt_c).sum())
            per_cat_fp[cat_id] += int((pred_c & ~gt_c).sum())
            per_cat_fn[cat_id] += int((~pred_c & gt_c).sum())

    ious = {}
    for cat_id in active_gt_cats:
        tp = per_cat_tp[cat_id]
        fp = per_cat_fp[cat_id]
        fn = per_cat_fn[cat_id]
        union = tp + fp + fn
        ious[cat_id] = tp / union if union > 0 else float('nan')

    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    miou = np.mean(valid_ious) if valid_ious else 0.0

    thing_ious = [ious[c] for c in active_gt_cats
                  if cat_info[c]["isthing"] and not np.isnan(ious.get(c, float('nan')))]
    stuff_ious = [ious[c] for c in active_gt_cats
                  if not cat_info[c]["isthing"] and not np.isnan(ious.get(c, float('nan')))]

    miou_things = np.mean(thing_ious) if thing_ious else 0.0
    miou_stuff = np.mean(stuff_ious) if stuff_ious else 0.0

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Semantic mIoU Results (19 Cityscapes → 133 COCO)")
    logger.info(f"{'=' * 60}")
    logger.info(f"  mIoU (all):    {miou * 100:.2f}%  ({len(valid_ious)} categories matched)")
    logger.info(f"  mIoU (things): {miou_things * 100:.2f}%  ({len(thing_ious)} categories)")
    logger.info(f"  mIoU (stuff):  {miou_stuff * 100:.2f}%  ({len(stuff_ious)} categories)")

    # Per-matched-category breakdown
    logger.info(f"\nPer-category IoU (matched categories):")
    sorted_cats = sorted(ious.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -1,
                         reverse=True)
    for cat_id, iou in sorted_cats:
        if np.isnan(iou):
            continue
        name = cat_info[cat_id]["name"]
        kind = "thing" if cat_info[cat_id]["isthing"] else "stuff"
        logger.info(f"  {cat_id:4d} {name:25s} ({kind:5s}): IoU = {iou * 100:.1f}%")

    return {
        "miou": miou, "miou_things": miou_things, "miou_stuff": miou_stuff,
        "per_category_iou": {str(k): v for k, v in ious.items()},
        "cluster_to_category": {str(k): v for k, v in cluster_to_cat.items()},
        "num_matched": len(cluster_to_cat),
    }


def evaluate_panoptic(coconut_root, split, pan_dir, cluster_to_cat, num_classes=19):
    """Evaluate panoptic predictions against COCONUT GT."""
    logger.info("=" * 60)
    logger.info("Panoptic Quality (PQ) Evaluation")
    logger.info("=" * 60)

    data, mask_dir, cat_info, id_to_ann, id_to_img = load_coconut_annotation(
        coconut_root, split)

    pan_dir = Path(pan_dir)

    per_cat_tp = defaultdict(int)
    per_cat_iou_sum = defaultdict(float)
    per_cat_fp = defaultdict(int)
    per_cat_fn = defaultdict(int)

    processed = 0

    for img_info in tqdm(data["images"], desc="Computing PQ"):
        image_id = img_info["id"]
        if image_id not in id_to_ann:
            continue

        pred_pan_path = pan_dir / f"{image_id:012d}.npy"
        if not pred_pan_path.exists():
            continue

        pred_pan = np.load(str(pred_pan_path))

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

        # Build GT segments
        gt_segments = {}
        for seg in ann["segments_info"]:
            seg_id = seg["id"]
            cat_id = seg["category_id"]
            mask = gt_segment_ids == seg_id
            if mask.sum() > 0:
                gt_segments[seg_id] = {
                    "category_id": cat_id,
                    "isthing": seg.get("isthing", 0),
                    "mask": mask,
                }

        # Build predicted segments
        pred_segments = {}
        unique_pred_ids = np.unique(pred_pan)
        for pan_id in unique_pred_ids:
            if pan_id == 0:
                continue
            cls_id = int(pan_id) // 1000

            mapped_cat = cluster_to_cat.get(cls_id, None)
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

        # Match segments
        matched_gt = set()
        matched_pred = set()

        for pred_id, pred_seg in pred_segments.items():
            best_iou = 0
            best_gt_id = None

            for gt_id, gt_seg in gt_segments.items():
                if gt_id in matched_gt:
                    continue
                if pred_seg["category_id"] != gt_seg["category_id"]:
                    continue

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

        for pred_id, pred_seg in pred_segments.items():
            if pred_id not in matched_pred:
                per_cat_fp[pred_seg["category_id"]] += 1

        for gt_id, gt_seg in gt_segments.items():
            if gt_id not in matched_gt:
                per_cat_fn[gt_seg["category_id"]] += 1

        processed += 1

    logger.info(f"Processed {processed} images")

    # Compute PQ
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
    pq_stuff = np.mean(stuff_pqs) if stuff_pqs else 0.0

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Panoptic Quality Results (19 CS → 133 COCO)")
    logger.info(f"{'=' * 60}")
    logger.info(f"           PQ      SQ      RQ     #cats")
    logger.info(f"  All:   {pq_all*100:5.1f}   {sq_all*100:5.1f}   {rq_all*100:5.1f}   {len(all_pqs)}")
    logger.info(f"  Things:{pq_things*100:5.1f}   {np.mean(thing_sqs)*100 if thing_sqs else 0:5.1f}   "
                f"{np.mean(thing_rqs)*100 if thing_rqs else 0:5.1f}   {len(thing_pqs)}")
    logger.info(f"  Stuff: {pq_stuff*100:5.1f}   {np.mean(stuff_sqs)*100 if stuff_sqs else 0:5.1f}   "
                f"{np.mean(stuff_rqs)*100 if stuff_rqs else 0:5.1f}   {len(stuff_pqs)}")

    # Top categories by PQ
    sorted_results = sorted(results.items(), key=lambda x: x[1]["pq"], reverse=True)
    logger.info(f"\nTop 20 categories by PQ:")
    for cat_id, r in sorted_results[:20]:
        name = cat_info.get(cat_id, {}).get("name", f"cat_{cat_id}")
        kind = "thing" if r["isthing"] else "stuff"
        logger.info(f"  {cat_id:4d} {name:25s} ({kind:5s}): "
                     f"PQ={r['pq']*100:5.1f}  SQ={r['sq']*100:5.1f}  RQ={r['rq']*100:5.1f}  "
                     f"TP={r['tp']:5d}  FP={r['fp']:5d}  FN={r['fn']:5d}")

    total_tp = sum(r["tp"] for r in results.values())
    total_fp = sum(r["fp"] for r in results.values())
    total_fn = sum(r["fn"] for r in results.values())
    logger.info(f"\nTotal: TP={total_tp}, FP={total_fp}, FN={total_fn}")

    return {
        "pq": pq_all, "sq": sq_all, "rq": rq_all,
        "pq_things": pq_things, "pq_stuff": pq_stuff,
        "num_thing_cats": len(thing_pqs),
        "num_stuff_cats": len(stuff_pqs),
        "per_category": {str(k): {kk: vv for kk, vv in v.items() if kk != "mask"}
                         for k, v in results.items()},
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Cityscapes mobile model on COCONUT"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best.pth checkpoint")
    parser.add_argument("--coco_root", type=str,
                        default="/Users/qbit-glitch/Desktop/datasets/coco")
    parser.add_argument("--coconut_root", type=str,
                        default="/Users/qbit-glitch/Desktop/datasets/coconut")
    parser.add_argument("--output_dir", type=str,
                        default="/Users/qbit-glitch/Desktop/datasets/coconut/mobile_preds")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--backbone", type=str, default="repvit_m0_9.dist_450e_in1k")
    parser.add_argument("--fpn_dim", type=int, default=128)
    parser.add_argument("--fpn_type", type=str, default="bifpn")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_h", type=int, default=512)
    parser.add_argument("--input_w", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip inference, use existing predictions")
    parser.add_argument("--skip_pq", action="store_true",
                        help="Skip PQ evaluation (only semantic mIoU)")

    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    logger.info(f"Device: {device}")

    t0 = time.time()

    # Step 1: Inference
    sem_dir = os.path.join(args.output_dir, "semantic", args.split)
    pan_dir = os.path.join(args.output_dir, "panoptic", args.split)

    if not args.skip_inference:
        logger.info("Loading model...")
        model = load_model(
            args.checkpoint, backbone=args.backbone,
            num_classes=args.num_classes, fpn_dim=args.fpn_dim,
            fpn_type=args.fpn_type, device=device,
        )

        logger.info("Running inference on COCO val...")
        sem_dir, pan_dir = run_inference(
            model, args.coco_root, args.output_dir, device,
            split=args.split, batch_size=args.batch_size,
            input_size=(args.input_h, args.input_w),
        )
        del model
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    else:
        logger.info(f"Skipping inference, using existing predictions in {args.output_dir}")

    # Step 2: Semantic evaluation
    semantic_results = evaluate_semantic(
        coconut_root=args.coconut_root,
        split=args.split,
        sem_dir=sem_dir,
        num_classes=args.num_classes,
    )

    # Step 3: Panoptic evaluation
    pq_results = None
    if not args.skip_pq:
        cluster_to_cat = {int(k): v for k, v in
                          semantic_results["cluster_to_category"].items()}
        pq_results = evaluate_panoptic(
            coconut_root=args.coconut_root,
            split=args.split,
            pan_dir=pan_dir,
            cluster_to_cat=cluster_to_cat,
            num_classes=args.num_classes,
        )

    elapsed = time.time() - t0

    # Save results
    output = {
        "semantic": {k: v for k, v in semantic_results.items()
                     if k != "per_category_iou"},
        "panoptic": {k: v for k, v in (pq_results or {}).items()
                     if k != "per_category"},
        "elapsed_seconds": round(elapsed, 1),
        "config": {
            "checkpoint": args.checkpoint,
            "backbone": args.backbone,
            "num_classes": args.num_classes,
            "input_size": [args.input_h, args.input_w],
            "device": device,
        }
    }

    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"SUMMARY: Cityscapes Mobile Model → COCONUT")
    logger.info(f"{'=' * 60}")
    logger.info(f"  mIoU: {semantic_results['miou'] * 100:.2f}% "
                f"({semantic_results['num_matched']}/19 classes matched)")
    if pq_results:
        logger.info(f"  PQ:   {pq_results['pq'] * 100:.1f}  "
                     f"(things: {pq_results['pq_things'] * 100:.1f}, "
                     f"stuff: {pq_results['pq_stuff'] * 100:.1f})")
    logger.info(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
