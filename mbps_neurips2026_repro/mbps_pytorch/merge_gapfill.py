#!/usr/bin/env python3
"""Gap-fill fusion: primary source + secondary fills uncovered regions.

Keeps all primary instances. Adds secondary instances only if their
pixel overlap with any primary instance is below a threshold.

Usage:
    python mbps_pytorch/merge_gapfill.py \
        --cityscapes_root /path/to/cityscapes \
        --split val \
        --primary pseudo_instances_dinosaur_dinov2_30slots \
        --secondary pseudo_instances_depth_layer \
        --output_dir pseudo_instances_gapfill_dino_spi \
        --max_overlap 0.3
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


def load_instances(npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load instance masks, scores, boxes from NPZ."""
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    num_valid = int(data["num_valid"]) if "num_valid" in data else data["masks"].shape[0]
    if num_valid == 0:
        H, W = 512, 1024
        return np.zeros((0, H, W), dtype=bool), np.array([], dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    masks = data["masks"][:num_valid]
    scores = data["scores"][:num_valid] if "scores" in data else np.ones(num_valid, dtype=np.float32)
    boxes = data["boxes"][:num_valid] if "boxes" in data else np.zeros((num_valid, 4), dtype=np.float32)
    return masks, scores, boxes


def find_npz(base_dir: str, city: str, stem: str) -> str | None:
    """Find NPZ file with or without _leftImg8bit suffix."""
    for suffix in ["", "_leftImg8bit"]:
        path = os.path.join(base_dir, city, stem + suffix + ".npz")
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser("Gap-fill instance fusion")
    parser.add_argument("--cityscapes_root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--primary", required=True, help="Primary instance subdir (all kept)")
    parser.add_argument("--secondary", required=True, help="Secondary instance subdir (gap-fill)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_overlap", type=float, default=0.3,
                        help="Max fraction of secondary mask covered by primary to be added (default: 0.3)")
    parser.add_argument("--min_area", type=int, default=500,
                        help="Min area for secondary instances to be added")
    args = parser.parse_args()

    root = args.cityscapes_root
    pri_dir = os.path.join(root, args.primary, args.split)
    sec_dir = os.path.join(root, args.secondary, args.split)
    out_dir = os.path.join(root, args.output_dir, args.split)

    # Discover images
    stems = set()
    for src_dir in [pri_dir, sec_dir]:
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
    logger.info(f"Found {len(image_list)} images")
    logger.info(f"  Primary: {args.primary} (all kept)")
    logger.info(f"  Secondary: {args.secondary} (gap-fill, max_overlap={args.max_overlap})")

    stats = {"primary": 0, "secondary_added": 0, "secondary_rejected": 0, "images": 0}
    t0 = time.time()

    for city, stem in tqdm(image_list, desc="Gap-filling"):
        all_masks = []
        all_scores = []
        all_boxes = []

        # Load all primary instances
        pri_npz = find_npz(pri_dir, city, stem)
        if pri_npz:
            result = load_instances(pri_npz)
            if result is not None:
                m, s, b = result
                for i in range(len(m)):
                    all_masks.append(m[i])
                    all_scores.append(s[i])
                    all_boxes.append(b[i])
                    stats["primary"] += 1

        # Build combined primary coverage mask
        if all_masks:
            coverage = np.zeros_like(all_masks[0], dtype=bool)
            for m in all_masks:
                coverage |= m
        else:
            coverage = np.zeros((512, 1024), dtype=bool)

        # Add secondary instances where overlap with primary is low
        sec_npz = find_npz(sec_dir, city, stem)
        if sec_npz:
            result = load_instances(sec_npz)
            if result is not None:
                m, s, b = result
                for i in range(len(m)):
                    mask = m[i]
                    area = mask.sum()
                    if area < args.min_area:
                        stats["secondary_rejected"] += 1
                        continue
                    # Compute overlap fraction: how much of this mask is already covered
                    overlap = np.sum(mask & coverage)
                    overlap_frac = overlap / (area + 1e-8)
                    if overlap_frac < args.max_overlap:
                        all_masks.append(mask)
                        all_scores.append(s[i])
                        all_boxes.append(b[i])
                        coverage |= mask  # Update coverage
                        stats["secondary_added"] += 1
                    else:
                        stats["secondary_rejected"] += 1

        # Save
        city_out = os.path.join(out_dir, city)
        os.makedirs(city_out, exist_ok=True)

        if all_masks:
            stacked = np.stack(all_masks, axis=0)
            scores = np.array(all_scores, dtype=np.float32)
            boxes = np.stack(all_boxes, axis=0)
        else:
            stacked = np.zeros((0, 512, 1024), dtype=bool)
            scores = np.array([], dtype=np.float32)
            boxes = np.zeros((0, 4), dtype=np.float32)

        np.savez_compressed(
            os.path.join(city_out, stem + ".npz"),
            masks=stacked, scores=scores, boxes=boxes,
            num_valid=len(stacked),
        )
        stats["images"] += 1

    elapsed = time.time() - t0
    total = stats["primary"] + stats["secondary_added"]
    avg = total / max(stats["images"], 1)

    logger.info(f"Done in {elapsed:.1f}s")
    logger.info(f"  Images: {stats['images']}")
    logger.info(f"  Primary kept: {stats['primary']}")
    logger.info(f"  Secondary added: {stats['secondary_added']}")
    logger.info(f"  Secondary rejected: {stats['secondary_rejected']}")
    logger.info(f"  Total: {total} ({avg:.1f}/img)")

    stats_path = os.path.join(root, args.output_dir, f"stats_{args.split}.json")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    stats["elapsed_s"] = round(elapsed, 1)
    stats["avg_per_image"] = round(avg, 2)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
