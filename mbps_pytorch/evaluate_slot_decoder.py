#!/usr/bin/env python3
"""Evaluate Depth-Conditioned Slot Decoder instance masks against Cityscapes GT.

Generates instance masks from trained slot attention model, assigns thing-class
labels via semantic pseudo-labels, and computes PQ/SQ/RQ (things, stuff, all).

Usage:
    python mbps_pytorch/evaluate_slot_decoder.py \
        --checkpoint checkpoints/slot_decoder/best.pth \
        --feature_dir /path/to/dinov3_features_vitl16/val \
        --depth_dir /path/to/depth_depthpro/val \
        --semantic_dir /path/to/pseudo_semantic_raw_k80/val \
        --cityscapes_root /path/to/cityscapes \
        --device mps
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from mbps_pytorch.models.slot_decoder import DepthSlotDecoder, DepthSlotDecoderConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GRID_H, GRID_W = 32, 64
N_PATCHES = GRID_H * GRID_W
EVAL_H, EVAL_W = 512, 1024

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

_STUFF_IDS = set(range(0, 11))
_THING_IDS = set(range(11, 19))
NUM_CLASSES = 19
IGNORE_LABEL = 255


def compute_majority_mapping(semantic_dir: str, cityscapes_root: str, num_clusters: int = 80):
    """Build many-to-one cluster→trainID mapping via majority vote."""
    logger.info(f"Computing majority mapping ({num_clusters} clusters → {NUM_CLASSES} classes)...")
    conf = np.zeros((num_clusters, NUM_CLASSES), dtype=np.int64)

    sem_dir = Path(semantic_dir)
    gt_dir = Path(cityscapes_root) / "gtFine" / "val"

    for sem_path in tqdm(sorted(sem_dir.rglob("*.png")), desc="Majority map"):
        city = sem_path.parent.name
        stem = sem_path.stem
        gt_path = gt_dir / city / f"{stem}_gtFine_labelIds.png"
        if not gt_path.exists():
            continue

        pred = np.array(Image.open(sem_path))
        gt_raw = np.array(Image.open(gt_path))
        gt = np.full_like(gt_raw, IGNORE_LABEL, dtype=np.uint8)
        for raw_id, train_id in _CS_ID_TO_TRAIN.items():
            gt[gt_raw == raw_id] = train_id

        if pred.shape != (EVAL_H, EVAL_W):
            pred = np.array(Image.fromarray(pred).resize((EVAL_W, EVAL_H), Image.NEAREST))
        if gt.shape != (EVAL_H, EVAL_W):
            gt = np.array(Image.fromarray(gt).resize((EVAL_W, EVAL_H), Image.NEAREST))

        valid = (gt != IGNORE_LABEL) & (pred < num_clusters)
        p, g = pred[valid], gt[valid]
        joint = p.astype(np.int64) * NUM_CLASSES + g.astype(np.int64)
        counts = np.bincount(joint, minlength=num_clusters * NUM_CLASSES)
        conf += counts.reshape(num_clusters, NUM_CLASSES)

    lut = np.full(256, IGNORE_LABEL, dtype=np.uint8)
    support = conf.sum(axis=1)
    supported = support > 0
    lut[:num_clusters][supported] = np.argmax(conf[supported], axis=1).astype(np.uint8)
    n_mapped = int(supported.sum())
    logger.info(f"Mapped {n_mapped}/{num_clusters} clusters by majority vote")
    return lut


def generate_slot_instances(
    model: DepthSlotDecoder,
    feature_path: Path,
    depth_path: Path,
    semantic_path: Path,
    cluster_lut: np.ndarray,
    device: torch.device,
    min_area_patches: int = 20,
    min_area_pixels: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate instance masks from slot decoder for one image.

    Returns:
        masks: (N_inst, H, W) binary masks
        class_ids: (N_inst,) trainID per instance
        scores: (N_inst,) confidence scores
    """
    # Load features and depth
    features = np.load(feature_path).astype(np.float32)
    features = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, N, D)
    depth = np.load(depth_path).astype(np.float32)
    depth = torch.from_numpy(depth).unsqueeze(0).to(device)  # (1, H, W)

    # Load semantic labels and map to trainIDs
    sem = np.array(Image.open(semantic_path))
    if sem.shape != (EVAL_H, EVAL_W):
        sem = np.array(Image.fromarray(sem).resize((EVAL_W, EVAL_H), Image.NEAREST))
    sem_trainid = cluster_lut[sem]

    # Forward pass
    with torch.no_grad():
        out = model(features, depth)
        slot_masks = out["masks"]  # (1, K, N)

    K = slot_masks.shape[1]
    slot_masks_np = slot_masks[0].cpu().numpy()  # (K, N)

    # Reshape to spatial grid and upsample
    slot_masks_2d = slot_masks_np.reshape(K, GRID_H, GRID_W)  # (K, 32, 64)

    masks_out = []
    class_ids_out = []
    scores_out = []

    for k in range(K):
        mask_patch = slot_masks_2d[k]  # (32, 64) soft mask

        # Check patch-level area
        patch_area = (mask_patch > 1.0 / K).sum()  # above uniform threshold
        if patch_area < min_area_patches:
            continue

        # Upsample to full resolution
        mask_full = np.array(
            Image.fromarray((mask_patch * 255).astype(np.uint8)).resize(
                (EVAL_W, EVAL_H), Image.BILINEAR
            )
        ).astype(np.float32) / 255.0

        # Hard threshold
        binary_mask = mask_full > (1.0 / K)

        # Check pixel-level area
        if binary_mask.sum() < min_area_pixels:
            continue

        # Assign class via majority vote of semantic labels under mask
        masked_sem = sem_trainid[binary_mask]
        masked_sem = masked_sem[masked_sem != IGNORE_LABEL]
        if len(masked_sem) == 0:
            continue

        cls_counts = np.bincount(masked_sem, minlength=NUM_CLASSES)
        assigned_class = cls_counts.argmax()

        # Only keep thing instances
        if assigned_class not in _THING_IDS:
            continue

        # Confidence = fraction of image covered × softmax concentration
        score = float(mask_patch.max() * patch_area / N_PATCHES)

        masks_out.append(binary_mask)
        class_ids_out.append(assigned_class)
        scores_out.append(score)

    if len(masks_out) == 0:
        return (
            np.zeros((0, EVAL_H, EVAL_W), dtype=bool),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
        )

    return (
        np.stack(masks_out),
        np.array(class_ids_out, dtype=np.int32),
        np.array(scores_out, dtype=np.float32),
    )


def load_gt_instances(inst_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load GT thing instances from Cityscapes instanceIds.png."""
    inst_map = np.array(Image.open(inst_path), dtype=np.int32)
    if inst_map.shape != (EVAL_H, EVAL_W):
        inst_map = np.array(
            Image.fromarray(inst_map).resize((EVAL_W, EVAL_H), Image.NEAREST)
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
    return np.zeros((0, EVAL_H, EVAL_W), dtype=bool), np.array([], dtype=int)


def compute_pq(
    pred_masks: np.ndarray,
    pred_cls: np.ndarray,
    gt_masks: np.ndarray,
    gt_cls: np.ndarray,
) -> dict:
    """Compute per-class PQ between predicted and GT instances."""
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0})

    if gt_masks.shape[0] == 0 and pred_masks.shape[0] == 0:
        return per_class

    # Mark GT as unmatched
    gt_matched = np.zeros(gt_masks.shape[0], dtype=bool)

    if pred_masks.shape[0] > 0 and gt_masks.shape[0] > 0:
        # Compute IoU matrix
        pred_flat = pred_masks.reshape(pred_masks.shape[0], -1).astype(np.float32)
        gt_flat = gt_masks.reshape(gt_masks.shape[0], -1).astype(np.float32)
        intersection = pred_flat @ gt_flat.T
        pred_area = pred_flat.sum(axis=1, keepdims=True)
        gt_area = gt_flat.sum(axis=1, keepdims=True).T
        union = pred_area + gt_area - intersection
        iou_matrix = intersection / (union + 1e-8)

        # Greedy matching per class
        pred_matched = np.zeros(pred_masks.shape[0], dtype=bool)

        for cls_id in _THING_IDS:
            pred_idx = np.where(pred_cls == cls_id)[0]
            gt_idx = np.where(gt_cls == cls_id)[0]

            if len(pred_idx) == 0 and len(gt_idx) == 0:
                continue

            if len(pred_idx) > 0 and len(gt_idx) > 0:
                sub_iou = iou_matrix[np.ix_(pred_idx, gt_idx)]

                # Greedy: match highest IoU first
                while True:
                    if sub_iou.size == 0:
                        break
                    best = np.unravel_index(sub_iou.argmax(), sub_iou.shape)
                    if sub_iou[best] < 0.5:
                        break
                    pi, gi = pred_idx[best[0]], gt_idx[best[1]]
                    per_class[cls_id]["tp"] += 1
                    per_class[cls_id]["iou_sum"] += sub_iou[best]
                    pred_matched[pi] = True
                    gt_matched[gi] = True
                    sub_iou[best[0], :] = 0
                    sub_iou[:, best[1]] = 0

            # Unmatched predictions = FP
            for pi in pred_idx:
                if not pred_matched[pi]:
                    per_class[cls_id]["fp"] += 1

            # Unmatched GT = FN
            for gi in gt_idx:
                if not gt_matched[gi]:
                    per_class[cls_id]["fn"] += 1
    else:
        # All predictions are FP
        for i, cls_id in enumerate(pred_cls):
            per_class[int(cls_id)]["fp"] += 1
        # All GT are FN
        for i, cls_id in enumerate(gt_cls):
            per_class[int(cls_id)]["fn"] += 1

    return per_class


def main():
    parser = argparse.ArgumentParser(description="Evaluate Slot Decoder Instances")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--semantic_dir", type=str, required=True)
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--num_clusters", type=int, default=80)
    parser.add_argument("--min_area_patches", type=int, default=20)
    parser.add_argument("--min_area_pixels", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = DepthSlotDecoderConfig(
        feat_dim=cfg_dict.get("feat_dim", 1024),
        slot_dim=cfg_dict.get("slot_dim", 256),
        num_slots=cfg_dict.get("num_slots", 20),
        slot_iters=cfg_dict.get("slot_iters", 5),
    )
    model = DepthSlotDecoder(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Model loaded: {model.count_parameters():,} params, epoch {ckpt['epoch']}")

    # Compute cluster → trainID mapping
    cluster_lut = compute_majority_mapping(
        args.semantic_dir, args.cityscapes_root, args.num_clusters
    )

    # Collect evaluation pairs
    feature_dir = Path(args.feature_dir)
    depth_dir = Path(args.depth_dir)
    semantic_dir = Path(args.semantic_dir)
    gt_dir = Path(args.cityscapes_root) / "gtFine" / "val"

    feature_files = sorted(feature_dir.rglob("*.npy"))
    logger.info(f"Evaluating {len(feature_files)} images...")

    # Accumulate per-class PQ stats
    global_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0})
    total_pred_instances = 0
    total_gt_instances = 0

    for feat_path in tqdm(feature_files, desc="Evaluating"):
        city = feat_path.parent.name
        stem = feat_path.stem.replace("_leftImg8bit", "")

        depth_path = depth_dir / city / f"{stem}.npy"
        sem_path = semantic_dir / city / f"{stem}.png"
        gt_inst_path = gt_dir / city / f"{stem}_gtFine_instanceIds.png"

        if not all(p.exists() for p in [depth_path, sem_path, gt_inst_path]):
            continue

        # Generate predictions
        pred_masks, pred_cls, pred_scores = generate_slot_instances(
            model, feat_path, depth_path, sem_path, cluster_lut, device,
            min_area_patches=args.min_area_patches,
            min_area_pixels=args.min_area_pixels,
        )

        # Load GT
        gt_masks, gt_cls = load_gt_instances(str(gt_inst_path))

        total_pred_instances += pred_masks.shape[0]
        total_gt_instances += gt_masks.shape[0]

        # Compute PQ
        img_stats = compute_pq(pred_masks, pred_cls, gt_masks, gt_cls)
        for cls_id, stats in img_stats.items():
            for key in ["tp", "fp", "fn"]:
                global_stats[cls_id][key] += stats[key]
            global_stats[cls_id]["iou_sum"] += stats["iou_sum"]

    # Compute final metrics
    logger.info(f"\nTotal predicted instances: {total_pred_instances}")
    logger.info(f"Total GT instances: {total_gt_instances}")
    logger.info(f"\n{'='*70}")
    logger.info(f"{'Class':<18} {'PQ':>6} {'SQ':>6} {'RQ':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    logger.info(f"{'='*70}")

    pq_values = []
    sq_values = []
    rq_values = []

    for cls_id in sorted(_THING_IDS):
        s = global_stats[cls_id]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        iou_sum = s["iou_sum"]

        sq = iou_sum / tp if tp > 0 else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
        pq = sq * rq

        pq_values.append(pq)
        sq_values.append(sq)
        rq_values.append(rq)

        logger.info(
            f"{_CS_CLASS_NAMES[cls_id]:<18} {pq*100:>5.1f}% {sq*100:>5.1f}% {rq*100:>5.1f}% "
            f"{tp:>5} {fp:>5} {fn:>5}"
        )

    mean_pq = np.mean(pq_values) if pq_values else 0.0
    mean_sq = np.mean(sq_values) if sq_values else 0.0
    mean_rq = np.mean(rq_values) if rq_values else 0.0

    logger.info(f"{'='*70}")
    logger.info(f"{'PQ_things':<18} {mean_pq*100:>5.1f}% {mean_sq*100:>5.1f}% {mean_rq*100:>5.1f}%")
    logger.info(f"{'='*70}")

    # Save results
    results = {
        "checkpoint": args.checkpoint,
        "epoch": ckpt["epoch"],
        "PQ_things": round(mean_pq * 100, 2),
        "SQ_things": round(mean_sq * 100, 2),
        "RQ_things": round(mean_rq * 100, 2),
        "total_pred_instances": total_pred_instances,
        "total_gt_instances": total_gt_instances,
        "per_class": {},
    }
    for cls_id in sorted(_THING_IDS):
        s = global_stats[cls_id]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        sq = s["iou_sum"] / tp if tp > 0 else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
        results["per_class"][_CS_CLASS_NAMES[cls_id]] = {
            "PQ": round(sq * rq * 100, 2),
            "SQ": round(sq * 100, 2),
            "RQ": round(rq * 100, 2),
            "TP": tp, "FP": fp, "FN": fn,
        }

    out_path = args.output or f"eval_slot_decoder_ep{ckpt['epoch']}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
