#!/usr/bin/env python3
"""Cross-dataset generalizability evaluation of Cityscapes-trained model.

Evaluates RepViT-M0.9 + BiFPN (19-class Cityscapes) on:
  A) KITTI-STEP: driving scenes, panoptic GT with Cityscapes raw IDs
  B) Mapillary Vistas v2: driving scenes, 124 fine-grained classes

Usage:
    # Both datasets
    python mbps_pytorch/evaluate_cross_dataset.py \
        --checkpoint checkpoints/cups_sem_bifpn_full/best.pth \
        --dataset both

    # KITTI-STEP only
    python mbps_pytorch/evaluate_cross_dataset.py \
        --checkpoint checkpoints/cups_sem_bifpn_full/best.pth \
        --dataset kitti

    # Mapillary only
    python mbps_pytorch/evaluate_cross_dataset.py \
        --checkpoint checkpoints/cups_sem_bifpn_full/best.pth \
        --dataset mapillary
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

from mbps_pytorch.train_mobile_panoptic import (
    MobilePanopticModel,
    _CS_ID_TO_TRAIN,
    _STUFF_IDS,
    _THING_IDS,
    _CS_CLASS_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# KITTI-STEP val sequences
KITTI_VAL_SEQUENCES = ["0002", "0006", "0007", "0008", "0010",
                       "0013", "0014", "0016", "0018"]


# ═══════════════════════════════════════════════════════════════════
# Shared: Model Loading + Inference
# ═══════════════════════════════════════════════════════════════════

def load_model(checkpoint_path, backbone="repvit_m0_9.dist_450e_in1k",
               num_classes=19, fpn_dim=128, fpn_type="bifpn", device="cpu"):
    """Load trained MobilePanopticModel from checkpoint."""
    model = MobilePanopticModel(
        backbone_name=backbone, num_classes=num_classes,
        fpn_dim=fpn_dim, pretrained=False, instance_head="none",
        fpn_type=fpn_type,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    metrics = ckpt.get("metrics", {})
    epoch = ckpt.get("epoch", "?")
    logger.info(f"Loaded checkpoint epoch {epoch}, Cityscapes PQ={metrics.get('PQ', '?')}, "
                f"mIoU={metrics.get('mIoU', '?')}")
    return model


def run_inference_on_images(model, img_paths, device, input_size=(512, 1024),
                             batch_size=4):
    """Run model on a list of image paths, return per-image predictions at original resolution."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    predictions = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(img_paths), batch_size), desc="Inference"):
            batch_paths = img_paths[i:i + batch_size]
            batch_imgs = []
            batch_sizes = []

            for img_path in batch_paths:
                img = Image.open(img_path).convert("RGB")
                orig_w, orig_h = img.size
                batch_sizes.append((orig_h, orig_w))
                img_resized = img.resize((input_size[1], input_size[0]), Image.BILINEAR)
                img_t = torch.from_numpy(np.array(img_resized)).float() / 255.0
                img_t = img_t.permute(2, 0, 1)
                batch_imgs.append(img_t)

            batch_tensor = torch.stack(batch_imgs).to(device)
            batch_tensor = (batch_tensor - mean) / std

            out = model(batch_tensor)
            logits = out["logits"]

            for j, img_path in enumerate(batch_paths):
                orig_h, orig_w = batch_sizes[j]
                logit = logits[j:j+1]
                logit_up = F.interpolate(logit, size=(orig_h, orig_w),
                                         mode='bilinear', align_corners=False)
                pred = logit_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                predictions[str(img_path)] = pred

    return predictions


def build_panoptic_cc(pred_sem, cc_min_area=50):
    """Build panoptic map from semantic prediction via connected components."""
    H, W = pred_sem.shape
    pan_map = np.zeros((H, W), dtype=np.int32)
    segments = {}
    nxt = 1

    for cls in _STUFF_IDS:
        mask = pred_sem == cls
        if mask.sum() < 64:
            continue
        pan_map[mask] = nxt
        segments[nxt] = cls
        nxt += 1

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


def compute_pq(gt_segments, pred_segments, gt_pan, pred_pan):
    """Compute per-category PQ given GT and pred segments.

    Args:
        gt_segments: dict {seg_id: category_id}
        pred_segments: dict {seg_id: category_id}
        gt_pan: (H, W) int32 array with segment IDs
        pred_pan: (H, W) int32 array with segment IDs

    Returns:
        per_cat stats: {cat_id: {tp, fp, fn, iou_sum}}
    """
    per_cat_tp = defaultdict(int)
    per_cat_iou_sum = defaultdict(float)
    per_cat_fp = defaultdict(int)
    per_cat_fn = defaultdict(int)

    # Group by category
    gt_by_cat = defaultdict(list)
    for sid, cat in gt_segments.items():
        gt_by_cat[cat].append(sid)
    pred_by_cat = defaultdict(list)
    for sid, cat in pred_segments.items():
        pred_by_cat[cat].append(sid)

    matched_pred = set()
    all_cats = set(list(gt_by_cat.keys()) + list(pred_by_cat.keys()))

    for cat in all_cats:
        for gt_id in gt_by_cat.get(cat, []):
            gt_mask = gt_pan == gt_id
            best_iou, best_pid = 0.0, None
            for pid in pred_by_cat.get(cat, []):
                if pid in matched_pred:
                    continue
                inter = np.sum(gt_mask & (pred_pan == pid))
                union = np.sum(gt_mask | (pred_pan == pid))
                if union == 0:
                    continue
                iou_val = inter / union
                if iou_val > best_iou:
                    best_iou, best_pid = iou_val, pid
            if best_iou > 0.5 and best_pid is not None:
                per_cat_tp[cat] += 1
                per_cat_iou_sum[cat] += best_iou
                matched_pred.add(best_pid)
            else:
                per_cat_fn[cat] += 1

        for pid in pred_by_cat.get(cat, []):
            if pid not in matched_pred:
                per_cat_fp[cat] += 1

    return per_cat_tp, per_cat_iou_sum, per_cat_fp, per_cat_fn


def summarize_pq(per_cat_tp, per_cat_iou_sum, per_cat_fp, per_cat_fn,
                 cat_names, stuff_cats, thing_cats, title=""):
    """Print PQ summary and return metrics dict."""
    all_cats = set(list(per_cat_tp.keys()) + list(per_cat_fp.keys()) + list(per_cat_fn.keys()))
    thing_pqs, stuff_pqs = [], []
    results = {}

    for cat_id in sorted(all_cats):
        tp = per_cat_tp[cat_id]
        fp = per_cat_fp[cat_id]
        fn = per_cat_fn[cat_id]
        iou_sum = per_cat_iou_sum[cat_id]

        sq = iou_sum / tp if tp > 0 else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
        pq = sq * rq
        results[cat_id] = {"pq": pq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn}

        if cat_id in thing_cats:
            thing_pqs.append(pq)
        elif cat_id in stuff_cats:
            stuff_pqs.append(pq)
        else:
            stuff_pqs.append(pq)  # default to stuff

    all_pqs = thing_pqs + stuff_pqs
    pq_all = np.mean(all_pqs) * 100 if all_pqs else 0.0
    pq_stuff = np.mean(stuff_pqs) * 100 if stuff_pqs else 0.0
    pq_things = np.mean(thing_pqs) * 100 if thing_pqs else 0.0

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Panoptic Quality — {title}")
    logger.info(f"{'=' * 60}")
    logger.info(f"           PQ      #cats")
    logger.info(f"  All:   {pq_all:5.1f}   {len(all_pqs)}")
    logger.info(f"  Things:{pq_things:5.1f}   {len(thing_pqs)}")
    logger.info(f"  Stuff: {pq_stuff:5.1f}   {len(stuff_pqs)}")

    logger.info(f"\nPer-category:")
    for cat_id, r in sorted(results.items(), key=lambda x: x[1]["pq"], reverse=True):
        name = cat_names.get(cat_id, f"cat_{cat_id}")
        kind = "thing" if cat_id in thing_cats else "stuff"
        logger.info(f"  {name:25s} ({kind:5s}): PQ={r['pq']*100:5.1f}  "
                     f"SQ={r['sq']*100:5.1f}  RQ={r['rq']*100:5.1f}  "
                     f"TP={r['tp']:5d}  FP={r['fp']:5d}  FN={r['fn']:5d}")

    total_tp = sum(r["tp"] for r in results.values())
    total_fp = sum(r["fp"] for r in results.values())
    total_fn = sum(r["fn"] for r in results.values())
    logger.info(f"\nTotal: TP={total_tp}, FP={total_fp}, FN={total_fn}")

    return {"pq": pq_all, "pq_stuff": pq_stuff, "pq_things": pq_things,
            "per_category": {str(k): v for k, v in results.items()}}


# ═══════════════════════════════════════════════════════════════════
# Part A: KITTI-STEP Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate_kitti_step(model, device, kitti_root, output_dir, batch_size=4):
    """Evaluate on KITTI-STEP val set.

    KITTI-STEP panoptic encoding (3-channel PNG):
      B = Cityscapes raw semantic class ID
      R = instance ID within class
      G = 0

    Cityscapes raw IDs → train IDs:
      thing classes: 24=person, 25=rider, 26=car, 27=truck, 28=bus, ...
      stuff classes: 7=road, 8=sidewalk, 11=building, 23=sky, etc.
    """
    logger.info("=" * 60)
    logger.info("KITTI-STEP Evaluation")
    logger.info("=" * 60)

    kitti_root = Path(kitti_root)
    img_base = kitti_root / "training" / "image_02"
    gt_base = kitti_root / "kitti-step" / "panoptic_maps" / "val"

    # Collect val image paths paired with GT
    img_gt_pairs = []
    for seq in KITTI_VAL_SEQUENCES:
        img_dir = img_base / seq
        gt_dir = gt_base / seq
        if not img_dir.exists() or not gt_dir.exists():
            logger.warning(f"Skipping sequence {seq}: missing img or GT dir")
            continue
        for gt_file in sorted(gt_dir.glob("*.png")):
            frame = gt_file.stem
            img_file = img_dir / f"{frame}.png"
            if img_file.exists():
                img_gt_pairs.append((str(img_file), str(gt_file)))

    logger.info(f"Found {len(img_gt_pairs)} image-GT pairs across {len(KITTI_VAL_SEQUENCES)} sequences")

    # Run inference
    img_paths = [p[0] for p in img_gt_pairs]
    predictions = run_inference_on_images(
        model, img_paths, device,
        input_size=(375, 1242),  # KITTI native resolution
        batch_size=batch_size,
    )

    # Evaluate
    per_cat_tp = defaultdict(int)
    per_cat_iou_sum = defaultdict(float)
    per_cat_fp = defaultdict(int)
    per_cat_fn = defaultdict(int)

    # Also track semantic confusion matrix for mIoU
    num_classes = 19
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for img_path, gt_path in tqdm(img_gt_pairs, desc="KITTI PQ eval"):
        pred_sem = predictions[img_path]

        # Load GT panoptic map
        gt_pan_rgb = np.array(Image.open(gt_path))
        gt_sem_raw = gt_pan_rgb[:, :, 2]  # B channel = Cityscapes raw ID
        gt_inst_raw = gt_pan_rgb[:, :, 0]  # R channel = instance ID

        # Map GT raw IDs to Cityscapes train IDs
        gt_sem = np.full_like(gt_sem_raw, 255, dtype=np.uint8)
        for raw_id, train_id in _CS_ID_TO_TRAIN.items():
            gt_sem[gt_sem_raw == raw_id] = train_id

        # Resize pred if needed
        if pred_sem.shape != gt_sem.shape:
            pred_sem = np.array(
                Image.fromarray(pred_sem).resize(
                    (gt_sem.shape[1], gt_sem.shape[0]), Image.NEAREST
                )
            )

        # Semantic confusion matrix
        valid = gt_sem < num_classes
        if valid.sum() > 0:
            np.add.at(confusion,
                      (gt_sem[valid].astype(np.int64),
                       pred_sem[valid].astype(np.int64)), 1)

        # Build GT panoptic segments
        gt_pan = np.zeros_like(gt_sem, dtype=np.int32)
        gt_segments = {}
        nxt = 1

        # Stuff: one segment per class
        for cls in _STUFF_IDS:
            mask = gt_sem == cls
            if mask.sum() < 64:
                continue
            gt_pan[mask] = nxt
            gt_segments[nxt] = cls
            nxt += 1

        # Things: use GT instance IDs
        for cls in _THING_IDS:
            raw_id = None
            for rid, tid in _CS_ID_TO_TRAIN.items():
                if tid == cls:
                    raw_id = rid
                    break
            if raw_id is None:
                continue

            cls_mask = gt_sem_raw == raw_id
            if cls_mask.sum() < 10:
                continue

            inst_ids = np.unique(gt_inst_raw[cls_mask])
            for iid in inst_ids:
                if iid == 0 or iid == 255:
                    continue
                mask = cls_mask & (gt_inst_raw == iid)
                if mask.sum() < 10:
                    continue
                gt_pan[mask] = nxt
                gt_segments[nxt] = cls
                nxt += 1

        # Build predicted panoptic via CC
        pred_pan, pred_segments = build_panoptic_cc(pred_sem, cc_min_area=50)

        # Compute PQ for this image
        tp, iou_sum, fp, fn = compute_pq(
            gt_segments, pred_segments, gt_pan, pred_pan)

        for cat in set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())):
            per_cat_tp[cat] += tp[cat]
            per_cat_iou_sum[cat] += iou_sum[cat]
            per_cat_fp[cat] += fp[cat]
            per_cat_fn[cat] += fn[cat]

    # Semantic mIoU
    intersection = np.diag(confusion)
    union = confusion.sum(1) + confusion.sum(0) - intersection
    iou = np.where(union > 0, intersection / union, 0.0)
    active = union > 0
    miou = iou[active].mean() * 100 if active.any() else 0.0

    logger.info(f"\n{'=' * 60}")
    logger.info(f"KITTI-STEP Semantic mIoU")
    logger.info(f"{'=' * 60}")
    logger.info(f"  mIoU: {miou:.2f}% ({active.sum()} active classes)")
    for c in range(num_classes):
        if union[c] > 0:
            logger.info(f"    {_CS_CLASS_NAMES[c]:20s}: IoU = {iou[c]*100:.1f}%")

    # PQ summary
    cat_names = {i: _CS_CLASS_NAMES[i] for i in range(num_classes)}
    pq_results = summarize_pq(
        per_cat_tp, per_cat_iou_sum, per_cat_fp, per_cat_fn,
        cat_names, _STUFF_IDS, _THING_IDS,
        title="KITTI-STEP"
    )
    pq_results["miou"] = miou

    return pq_results


# ═══════════════════════════════════════════════════════════════════
# Part B: Mapillary Vistas v2 Evaluation
# ═══════════════════════════════════════════════════════════════════

def load_mapillary_config(config_path):
    """Load Mapillary Vistas v2 config (class definitions)."""
    with open(config_path) as f:
        cfg = json.load(f)

    cat_info = {}
    for i, lab in enumerate(cfg["labels"]):
        cat_info[i] = {
            "name": lab.get("readable", lab["name"]),
            "isthing": lab.get("instances", False),
        }
    return cat_info


def evaluate_mapillary(model, device, mapillary_root, output_dir, batch_size=2):
    """Evaluate on Mapillary Vistas v2 val set.

    Pipeline:
      1. Run inference → 19-class predictions
      2. Hungarian match 19 → 124 Mapillary categories via semantic labels
      3. PQ evaluation via panoptic GT
    """
    logger.info("=" * 60)
    logger.info("Mapillary Vistas v2 Evaluation")
    logger.info("=" * 60)

    mapillary_root = Path(mapillary_root)
    config_path = mapillary_root / "config_v2.0.json"
    img_dir = mapillary_root / "validation" / "images"
    label_dir = mapillary_root / "validation" / "v2.0" / "labels"
    panoptic_dir = mapillary_root / "validation" / "v2.0" / "panoptic"

    cat_info = load_mapillary_config(config_path)
    num_mapillary_classes = len(cat_info)

    # Get image paths
    img_paths = sorted(img_dir.glob("*.jpg"))
    logger.info(f"Found {len(img_paths)} validation images")

    # Run inference (Mapillary images are large — use smaller input)
    predictions = run_inference_on_images(
        model, [str(p) for p in img_paths], device,
        input_size=(512, 1024),
        batch_size=batch_size,
    )

    # Step 1: Build confusion matrix for Hungarian matching (19 × 124)
    logger.info("Building confusion matrix for Hungarian matching...")
    confusion = np.zeros((19, num_mapillary_classes), dtype=np.int64)

    for img_path in tqdm(img_paths, desc="Confusion matrix"):
        stem = img_path.stem
        label_path = label_dir / f"{stem}.png"
        if not label_path.exists():
            continue

        pred = predictions[str(img_path)]
        gt_label = np.array(Image.open(label_path))

        # Resize pred to GT resolution
        if pred.shape != gt_label.shape:
            pred = np.array(
                Image.fromarray(pred).resize(
                    (gt_label.shape[1], gt_label.shape[0]), Image.NEAREST
                )
            )

        valid = gt_label < num_mapillary_classes
        pred_valid = pred[valid].astype(np.int64)
        gt_valid = gt_label[valid].astype(np.int64)
        mask = pred_valid < 19
        np.add.at(confusion, (pred_valid[mask], gt_valid[mask]), 1)

    # Hungarian matching
    active_gt_cats = [c for c in range(num_mapillary_classes)
                      if confusion[:, c].sum() > 0]
    logger.info(f"Active Mapillary categories: {len(active_gt_cats)}")

    cost = np.zeros((19, len(active_gt_cats)), dtype=np.int64)
    for j, cat_id in enumerate(active_gt_cats):
        cost[:, j] = confusion[:, cat_id]

    row_ind, col_ind = linear_sum_assignment(-cost)

    cluster_to_cat = {}
    for r, c in zip(row_ind, col_ind):
        cat_id = active_gt_cats[c]
        if cost[r, c] > 0:
            cluster_to_cat[r] = cat_id

    logger.info(f"\nClass mapping (19 CS → {len(cluster_to_cat)} Mapillary):")
    for cls_id in sorted(cluster_to_cat.keys()):
        cat_id = cluster_to_cat[cls_id]
        cs_name = _CS_CLASS_NAMES[cls_id]
        mp_name = cat_info[cat_id]["name"]
        mp_kind = "thing" if cat_info[cat_id]["isthing"] else "stuff"
        logger.info(f"  CS:{cls_id:2d} ({cs_name:15s}) → MP:{cat_id:3d} ({mp_name:30s}, {mp_kind})")

    # Step 2: Semantic mIoU with mapping
    logger.info("\nComputing semantic mIoU...")
    per_cat_tp = defaultdict(int)
    per_cat_fp = defaultdict(int)
    per_cat_fn = defaultdict(int)

    for img_path in tqdm(img_paths, desc="Semantic mIoU"):
        stem = img_path.stem
        label_path = label_dir / f"{stem}.png"
        if not label_path.exists():
            continue

        pred = predictions[str(img_path)]
        gt_label = np.array(Image.open(label_path))

        if pred.shape != gt_label.shape:
            pred = np.array(
                Image.fromarray(pred).resize(
                    (gt_label.shape[1], gt_label.shape[0]), Image.NEAREST
                )
            )

        # Remap predictions to Mapillary space
        pred_mapped = np.full_like(pred, 255, dtype=np.int32)
        for cls_id, cat_id in cluster_to_cat.items():
            pred_mapped[pred == cls_id] = cat_id

        valid = gt_label < num_mapillary_classes
        for cat_id in active_gt_cats:
            pred_c = (pred_mapped == cat_id) & valid
            gt_c = (gt_label == cat_id) & valid
            per_cat_tp[cat_id] += int((pred_c & gt_c).sum())
            per_cat_fp[cat_id] += int((pred_c & ~gt_c).sum())
            per_cat_fn[cat_id] += int((~pred_c & gt_c).sum())

    # Compute IoU
    ious = {}
    for cat_id in active_gt_cats:
        tp = per_cat_tp[cat_id]
        fp = per_cat_fp[cat_id]
        fn = per_cat_fn[cat_id]
        union = tp + fp + fn
        ious[cat_id] = tp / union if union > 0 else float('nan')

    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    miou = np.mean(valid_ious) * 100 if valid_ious else 0.0

    matched_ious = [ious[cluster_to_cat[c]] for c in cluster_to_cat
                    if not np.isnan(ious.get(cluster_to_cat[c], float('nan')))]
    miou_matched = np.mean(matched_ious) * 100 if matched_ious else 0.0

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Mapillary Semantic mIoU")
    logger.info(f"{'=' * 60}")
    logger.info(f"  mIoU (all {len(valid_ious)} cats): {miou:.2f}%")
    logger.info(f"  mIoU (matched {len(matched_ious)} cats): {miou_matched:.2f}%")

    # Per matched category
    logger.info(f"\nPer matched category IoU:")
    for cls_id in sorted(cluster_to_cat.keys()):
        cat_id = cluster_to_cat[cls_id]
        iou = ious.get(cat_id, 0)
        if np.isnan(iou):
            continue
        cs_name = _CS_CLASS_NAMES[cls_id]
        mp_name = cat_info[cat_id]["name"]
        logger.info(f"  {cs_name:15s} → {mp_name:25s}: IoU = {iou*100:.1f}%")

    # Step 3: Panoptic Quality
    logger.info("\nComputing Panoptic Quality...")
    pq_tp = defaultdict(int)
    pq_iou = defaultdict(float)
    pq_fp = defaultdict(int)
    pq_fn = defaultdict(int)

    for img_path in tqdm(img_paths, desc="PQ eval"):
        stem = img_path.stem
        pan_path = panoptic_dir / f"{stem}.png"
        label_path = label_dir / f"{stem}.png"
        if not pan_path.exists() or not label_path.exists():
            continue

        pred = predictions[str(img_path)]

        # Load GT
        gt_pan_rgb = np.array(Image.open(pan_path))
        gt_label = np.array(Image.open(label_path))

        # Decode panoptic IDs
        gt_pan_ids = (gt_pan_rgb[:, :, 0].astype(np.int32) +
                      gt_pan_rgb[:, :, 1].astype(np.int32) * 256 +
                      gt_pan_rgb[:, :, 2].astype(np.int32) * 65536)

        gt_H, gt_W = gt_pan_ids.shape

        # Resize pred
        if pred.shape != (gt_H, gt_W):
            pred = np.array(
                Image.fromarray(pred).resize((gt_W, gt_H), Image.NEAREST)
            )

        # Build GT segments: {seg_id: category_id} from panoptic + label
        gt_segments = {}
        for seg_id in np.unique(gt_pan_ids):
            if seg_id == 0:
                continue
            mask = gt_pan_ids == seg_id
            if mask.sum() < 10:
                continue
            # Majority vote from label map
            seg_labels = gt_label[mask]
            cat_id = int(np.bincount(seg_labels).argmax())
            if cat_id >= num_mapillary_classes:
                continue
            gt_segments[int(seg_id)] = cat_id

        # Build pred panoptic
        # Remap predictions to Mapillary category space
        pred_mapped = np.full_like(pred, 255, dtype=np.uint8)
        for cls_id, cat_id in cluster_to_cat.items():
            pred_mapped[pred == cls_id] = cat_id

        pred_pan = np.zeros((gt_H, gt_W), dtype=np.int32)
        pred_segments = {}
        nxt = 1

        # Stuff segments (Mapillary non-instance classes)
        stuff_cat_ids = {cid for cid in cluster_to_cat.values()
                         if not cat_info[cid]["isthing"]}
        thing_cat_ids = {cid for cid in cluster_to_cat.values()
                         if cat_info[cid]["isthing"]}

        for cat_id in stuff_cat_ids:
            mask = pred_mapped == cat_id
            if mask.sum() < 64:
                continue
            pred_pan[mask] = nxt
            pred_segments[nxt] = cat_id
            nxt += 1

        # Thing segments via CC
        for cat_id in thing_cat_ids:
            cls_mask = pred_mapped == cat_id
            if cls_mask.sum() < 50:
                continue
            labeled, n_cc = ndimage.label(cls_mask)
            for comp in range(1, n_cc + 1):
                cmask = labeled == comp
                if cmask.sum() < 50:
                    continue
                pred_pan[cmask] = nxt
                pred_segments[nxt] = cat_id
                nxt += 1

        # Compute PQ
        tp, iou_sum, fp, fn = compute_pq(
            gt_segments, pred_segments, gt_pan_ids, pred_pan)

        for cat in set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())):
            pq_tp[cat] += tp[cat]
            pq_iou[cat] += iou_sum[cat]
            pq_fp[cat] += fp[cat]
            pq_fn[cat] += fn[cat]

    # PQ summary
    mp_stuff = {i for i, c in cat_info.items() if not c["isthing"]}
    mp_things = {i for i, c in cat_info.items() if c["isthing"]}
    cat_names = {i: c["name"] for i, c in cat_info.items()}

    pq_results = summarize_pq(
        pq_tp, pq_iou, pq_fp, pq_fn,
        cat_names, mp_stuff, mp_things,
        title="Mapillary Vistas v2"
    )
    pq_results["miou"] = miou
    pq_results["miou_matched"] = miou_matched

    return pq_results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset evaluation on KITTI-STEP and Mapillary Vistas"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="both",
                        choices=["kitti", "mapillary", "both"])
    parser.add_argument("--kitti_root", type=str,
                        default="/Users/qbit-glitch/Desktop/datasets/kitti-step")
    parser.add_argument("--mapillary_root", type=str,
                        default="/Users/qbit-glitch/Desktop/datasets/mapillary-vistas-v2")
    parser.add_argument("--output_dir", type=str, default="results/cross_dataset")
    parser.add_argument("--backbone", type=str, default="repvit_m0_9.dist_450e_in1k")
    parser.add_argument("--fpn_dim", type=int, default=128)
    parser.add_argument("--fpn_type", type=str, default="bifpn")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")

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

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    # Load model once
    model = load_model(args.checkpoint, backbone=args.backbone,
                       fpn_dim=args.fpn_dim, fpn_type=args.fpn_type,
                       device=device)

    results = {}

    if args.dataset in ("kitti", "both"):
        kitti_results = evaluate_kitti_step(
            model, device, args.kitti_root, args.output_dir,
            batch_size=args.batch_size,
        )
        results["kitti_step"] = kitti_results

    if args.dataset in ("mapillary", "both"):
        mapillary_results = evaluate_mapillary(
            model, device, args.mapillary_root, args.output_dir,
            batch_size=max(1, args.batch_size // 2),  # smaller batch for large images
        )
        results["mapillary"] = mapillary_results

    elapsed = time.time() - t0

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"CROSS-DATASET GENERALIZABILITY SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Cityscapes (training):  PQ = 24.78")

    if "kitti_step" in results:
        r = results["kitti_step"]
        logger.info(f"  KITTI-STEP:            PQ = {r['pq']:.1f}  "
                     f"(mIoU={r['miou']:.1f}%)")
    if "mapillary" in results:
        r = results["mapillary"]
        logger.info(f"  Mapillary Vistas v2:   PQ = {r['pq']:.1f}  "
                     f"(mIoU matched={r.get('miou_matched', 0):.1f}%)")

    logger.info(f"  Total time: {elapsed:.1f}s")

    # Save results
    results_path = os.path.join(args.output_dir, "cross_dataset_results.json")
    # Remove non-serializable items
    clean_results = {}
    for k, v in results.items():
        clean_results[k] = {kk: vv for kk, vv in v.items()}
    with open(results_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
