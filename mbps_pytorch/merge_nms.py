#!/usr/bin/env python3
"""Fuse two instance pseudo-label sources via NMS deduplication.

Pools all instances from both sources, sorts by score, and applies
greedy NMS to remove duplicates. No semantic labels needed.

Usage:
    python mbps_pytorch/merge_nms.py \
        --cityscapes_root /path/to/cityscapes \
        --split val \
        --source_a pseudo_instances_depth_layer \
        --source_b pseudo_instances_dinosaur_dinov2_30slots \
        --output_dir pseudo_instances_fused_nms \
        --nms_iou 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def mask_nms(masks: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Greedy NMS on boolean masks. Returns indices to keep."""
    if len(masks) == 0:
        return np.array([], dtype=int)

    order = np.argsort(-scores)
    keep = []
    suppressed = set()

    # Precompute areas
    areas = masks.reshape(len(masks), -1).sum(axis=1).astype(np.float32)

    for idx in order:
        if idx in suppressed:
            continue
        keep.append(idx)
        mask_i = masks[idx]

        # Check remaining candidates
        for jdx in order:
            if jdx in suppressed or jdx == idx:
                continue
            # Fast area check — if no overlap possible, skip
            intersection = np.sum(mask_i & masks[jdx])
            if intersection == 0:
                continue
            union = areas[idx] + areas[jdx] - intersection
            iou = intersection / (union + 1e-8)
            if iou > iou_thresh:
                suppressed.add(jdx)

    return np.array(keep, dtype=int)


def load_instances(npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load instance masks, scores, boxes from NPZ. Returns None if missing."""
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    num_valid = int(data["num_valid"]) if "num_valid" in data else data["masks"].shape[0]
    if num_valid == 0:
        return np.zeros((0, 512, 1024), dtype=bool), np.array([], dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    masks = data["masks"][:num_valid]
    scores = data["scores"][:num_valid] if "scores" in data else np.ones(num_valid, dtype=np.float32)
    boxes = data["boxes"][:num_valid] if "boxes" in data else np.zeros((num_valid, 4), dtype=np.float32)
    return masks, scores, boxes


def find_npz(base_dir: str, city: str, stem: str) -> str | None:
    """Find NPZ file, trying with and without _leftImg8bit suffix."""
    for suffix in ["", "_leftImg8bit"]:
        path = os.path.join(base_dir, city, stem + suffix + ".npz")
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser("NMS-based instance fusion")
    parser.add_argument("--cityscapes_root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--source_a", required=True, help="First instance subdir")
    parser.add_argument("--source_b", required=True, help="Second instance subdir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--nms_iou", type=float, default=0.3,
                        help="IoU threshold for NMS deduplication (default: 0.3)")
    args = parser.parse_args()

    root = args.cityscapes_root
    dir_a = os.path.join(root, args.source_a, args.split)
    dir_b = os.path.join(root, args.source_b, args.split)
    out_dir = os.path.join(root, args.output_dir, args.split)

    # Discover images from both sources
    stems = set()
    for src_dir in [dir_a, dir_b]:
        if not os.path.exists(src_dir):
            continue
        for city in os.listdir(src_dir):
            city_path = os.path.join(src_dir, city)
            if not os.path.isdir(city_path):
                continue
            for fname in os.listdir(city_path):
                if fname.endswith(".npz"):
                    stem = fname.replace(".npz", "").replace("_leftImg8bit", "")
                    stems.add((city, stem))

    image_list = sorted(stems)
    logger.info(f"Found {len(image_list)} images across both sources")
    logger.info(f"  Source A: {args.source_a}")
    logger.info(f"  Source B: {args.source_b}")
    logger.info(f"  NMS IoU threshold: {args.nms_iou}")

    stats = {
        "from_a": 0, "from_b": 0,
        "total_before_nms": 0, "total_after_nms": 0,
        "images": 0,
    }

    t0 = time.time()

    for city, stem in tqdm(image_list, desc="Fusing"):
        all_masks = []
        all_scores = []
        all_boxes = []
        sources = []  # track origin for stats

        # Load source A
        npz_a = find_npz(dir_a, city, stem)
        if npz_a:
            result = load_instances(npz_a)
            if result is not None:
                m, s, b = result
                for i in range(len(m)):
                    all_masks.append(m[i])
                    all_scores.append(s[i])
                    all_boxes.append(b[i])
                    sources.append("a")

        # Load source B
        npz_b = find_npz(dir_b, city, stem)
        if npz_b:
            result = load_instances(npz_b)
            if result is not None:
                m, s, b = result
                for i in range(len(m)):
                    all_masks.append(m[i])
                    all_scores.append(s[i])
                    all_boxes.append(b[i])
                    sources.append("b")

        stats["total_before_nms"] += len(all_masks)

        if len(all_masks) == 0:
            merged_masks = np.zeros((0, 512, 1024), dtype=bool)
            merged_scores = np.array([], dtype=np.float32)
            merged_boxes = np.zeros((0, 4), dtype=np.float32)
        else:
            stacked_masks = np.stack(all_masks, axis=0)
            stacked_scores = np.array(all_scores, dtype=np.float32)
            stacked_boxes = np.stack(all_boxes, axis=0)

            # NMS
            keep = mask_nms(stacked_masks, stacked_scores, args.nms_iou)

            merged_masks = stacked_masks[keep]
            merged_scores = stacked_scores[keep]
            merged_boxes = stacked_boxes[keep]

            for k in keep:
                if sources[k] == "a":
                    stats["from_a"] += 1
                else:
                    stats["from_b"] += 1

        stats["total_after_nms"] += len(merged_masks)
        stats["images"] += 1

        # Save
        city_out = os.path.join(out_dir, city)
        os.makedirs(city_out, exist_ok=True)
        np.savez_compressed(
            os.path.join(city_out, stem + ".npz"),
            masks=merged_masks, scores=merged_scores, boxes=merged_boxes,
            num_valid=len(merged_masks),
        )

    elapsed = time.time() - t0
    avg_before = stats["total_before_nms"] / max(stats["images"], 1)
    avg_after = stats["total_after_nms"] / max(stats["images"], 1)

    logger.info(f"Done in {elapsed:.1f}s")
    logger.info(f"  Images: {stats['images']}")
    logger.info(f"  Before NMS: {stats['total_before_nms']} ({avg_before:.1f}/img)")
    logger.info(f"  After NMS:  {stats['total_after_nms']} ({avg_after:.1f}/img)")
    logger.info(f"  Kept from A: {stats['from_a']}, from B: {stats['from_b']}")
    logger.info(f"  Removed by NMS: {stats['total_before_nms'] - stats['total_after_nms']}")

    stats_path = os.path.join(root, args.output_dir, f"stats_{args.split}.json")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    stats["elapsed_s"] = round(elapsed, 1)
    stats["nms_iou"] = args.nms_iou
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
