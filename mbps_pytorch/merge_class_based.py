#!/usr/bin/env python3
"""Class-based merge of two instance pseudo-label sources.

Uses DINOSAUR instances for car/truck (where it's stronger) and
SPIdepth instances for bus/person/bicycle (where it's stronger).
Class assignment is via majority vote from CAUSE-TR semantic map.

Usage:
    python mbps_pytorch/merge_class_based.py \
        --cityscapes_root /path/to/cityscapes \
        --split val \
        --primary_subdir pseudo_instances_depth_layer \
        --secondary_subdir pseudo_instances_dinosaur_dinov2_30slots \
        --semantic_subdir pseudo_semantic_cause \
        --output_dir pseudo_instances_merged_class_based \
        --cause27 \
        --secondary_classes 13 14
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IGNORE_LABEL = 255
NUM_CLASSES = 19

_CAUSE27_TO_TRAINID = np.full(256, IGNORE_LABEL, dtype=np.uint8)
for _c27, _t19 in {
    0: 0, 1: 1, 2: 255, 3: 255, 4: 2, 5: 3, 6: 4,
    7: 255, 8: 255, 9: 255, 10: 5, 11: 5, 12: 6, 13: 7,
    14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14,
    21: 15, 22: 13, 23: 14, 24: 16, 25: 17, 26: 18,
}.items():
    _CAUSE27_TO_TRAINID[_c27] = _t19

THING_IDS = set(range(11, 19))

CS_NAMES = {
    11: "person", 12: "rider", 13: "car", 14: "truck",
    15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}


def classify_instance(mask: np.ndarray, sem_19: np.ndarray) -> int:
    """Assign class to an instance mask via majority vote from semantic map."""
    sem_vals = sem_19[mask]
    valid = sem_vals[sem_vals != IGNORE_LABEL]
    if len(valid) == 0:
        return IGNORE_LABEL
    classes, counts = np.unique(valid, return_counts=True)
    return int(classes[counts.argmax()])


def load_instances(npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load instance masks, scores, boxes from NPZ."""
    data = np.load(npz_path)
    num_valid = int(data["num_valid"]) if "num_valid" in data else data["masks"].shape[0]
    masks = data["masks"][:num_valid]
    scores = data["scores"][:num_valid] if "scores" in data else np.ones(num_valid, dtype=np.float32)
    boxes = data["boxes"][:num_valid] if "boxes" in data else np.zeros((num_valid, 4), dtype=np.float32)
    return masks, scores, boxes


def main():
    parser = argparse.ArgumentParser("Class-based instance merge")
    parser.add_argument("--cityscapes_root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--primary_subdir", default="pseudo_instances_depth_layer",
                        help="Primary instance source (SPIdepth)")
    parser.add_argument("--secondary_subdir", default="pseudo_instances_dinosaur_dinov2_30slots",
                        help="Secondary instance source (DINOSAUR)")
    parser.add_argument("--semantic_subdir", default="pseudo_semantic_cause")
    parser.add_argument("--output_dir", default="pseudo_instances_merged_class_based")
    parser.add_argument("--cause27", action="store_true")
    parser.add_argument("--secondary_classes", type=int, nargs="+", default=[13, 14],
                        help="TrainIDs to take from secondary source (default: 13=car, 14=truck)")
    args = parser.parse_args()

    root = args.cityscapes_root
    primary_dir = os.path.join(root, args.primary_subdir, args.split)
    secondary_dir = os.path.join(root, args.secondary_subdir, args.split)
    sem_dir = os.path.join(root, args.semantic_subdir, args.split)
    out_dir = os.path.join(root, args.output_dir, args.split)

    secondary_cls = set(args.secondary_classes)
    primary_cls = THING_IDS - secondary_cls

    logger.info(f"Primary ({args.primary_subdir}) classes: "
                f"{sorted(primary_cls)} ({', '.join(CS_NAMES.get(c, '?') for c in sorted(primary_cls))})")
    logger.info(f"Secondary ({args.secondary_subdir}) classes: "
                f"{sorted(secondary_cls)} ({', '.join(CS_NAMES.get(c, '?') for c in sorted(secondary_cls))})")

    # Discover images from semantic dir
    image_list = []
    for city in sorted(os.listdir(sem_dir)):
        city_path = os.path.join(sem_dir, city)
        if not os.path.isdir(city_path):
            continue
        for fname in sorted(os.listdir(city_path)):
            if not fname.endswith(".png"):
                continue
            stem = fname.replace(".png", "")
            image_list.append((city, stem))

    logger.info(f"Found {len(image_list)} images")

    stats = {
        "primary_kept": 0, "primary_dropped": 0,
        "secondary_kept": 0, "secondary_dropped": 0,
        "total_merged": 0, "images": 0,
        "primary_missing": 0, "secondary_missing": 0,
    }

    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    for city, stem in tqdm(image_list, desc="Merging"):
        # Load semantic map
        sem_path = os.path.join(sem_dir, city, stem + ".png")
        sem_np = np.array(Image.open(sem_path).resize((1024, 512), Image.NEAREST), dtype=np.uint8)
        if args.cause27:
            sem_19 = _CAUSE27_TO_TRAINID[sem_np]
        else:
            sem_19 = sem_np

        merged_masks = []
        merged_scores = []
        merged_boxes = []

        # Load primary instances (SPIdepth) — keep only primary_cls
        pri_npz = os.path.join(primary_dir, city, stem + ".npz")
        if not os.path.exists(pri_npz):
            # Try with _leftImg8bit suffix
            pri_npz = os.path.join(primary_dir, city, stem + "_leftImg8bit.npz")
        if os.path.exists(pri_npz):
            masks, scores, boxes = load_instances(pri_npz)
            for i in range(len(masks)):
                cls = classify_instance(masks[i], sem_19)
                if cls in primary_cls:
                    merged_masks.append(masks[i])
                    merged_scores.append(scores[i])
                    merged_boxes.append(boxes[i])
                    stats["primary_kept"] += 1
                else:
                    stats["primary_dropped"] += 1
        else:
            stats["primary_missing"] += 1

        # Load secondary instances (DINOSAUR) — keep only secondary_cls
        sec_npz = os.path.join(secondary_dir, city, stem + ".npz")
        if not os.path.exists(sec_npz):
            sec_npz = os.path.join(secondary_dir, city, stem + "_leftImg8bit.npz")
        if os.path.exists(sec_npz):
            masks, scores, boxes = load_instances(sec_npz)
            for i in range(len(masks)):
                cls = classify_instance(masks[i], sem_19)
                if cls in secondary_cls:
                    merged_masks.append(masks[i])
                    merged_scores.append(scores[i])
                    merged_boxes.append(boxes[i])
                    stats["secondary_kept"] += 1
                else:
                    stats["secondary_dropped"] += 1
        else:
            stats["secondary_missing"] += 1

        # Save merged
        city_out = os.path.join(out_dir, city)
        os.makedirs(city_out, exist_ok=True)

        if merged_masks:
            all_masks = np.stack(merged_masks, axis=0)
            all_scores = np.array(merged_scores, dtype=np.float32)
            all_boxes = np.stack(merged_boxes, axis=0)
        else:
            all_masks = np.zeros((0, 512, 1024), dtype=bool)
            all_scores = np.array([], dtype=np.float32)
            all_boxes = np.zeros((0, 4), dtype=np.float32)

        np.savez_compressed(
            os.path.join(city_out, stem + ".npz"),
            masks=all_masks, scores=all_scores, boxes=all_boxes,
            num_valid=len(all_masks),
        )
        stats["total_merged"] += len(all_masks)
        stats["images"] += 1

    elapsed = time.time() - t0
    avg = stats["total_merged"] / max(stats["images"], 1)

    logger.info(f"Done in {elapsed:.1f}s")
    logger.info(f"  Images: {stats['images']}")
    logger.info(f"  Primary kept/dropped: {stats['primary_kept']}/{stats['primary_dropped']}")
    logger.info(f"  Secondary kept/dropped: {stats['secondary_kept']}/{stats['secondary_dropped']}")
    logger.info(f"  Total merged: {stats['total_merged']} ({avg:.1f} avg/img)")
    logger.info(f"  Primary missing: {stats['primary_missing']}, Secondary missing: {stats['secondary_missing']}")

    # Save stats
    stats_path = os.path.join(root, args.output_dir, f"stats_{args.split}.json")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    stats["elapsed_s"] = round(elapsed, 1)
    stats["avg_per_image"] = round(avg, 2)
    stats["secondary_classes"] = sorted(secondary_cls)
    stats["primary_classes"] = sorted(primary_cls)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
