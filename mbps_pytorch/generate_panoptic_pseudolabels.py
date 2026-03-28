#!/usr/bin/env python3
"""Generate panoptic pseudo-labels from semantic + depth-guided instances.

Combines:
  - Semantic pseudo-labels (DINOv3 linear probe, trainIDs 0-18)
  - Depth-guided instance masks (NPZ from generate_depth_guided_instances.py)
  - Stuff/things classification (from classify_stuff_things.py)

Output per image:
  - panoptic_map: (H, W) int32 — Cityscapes-style encoding:
      stuff pixels: class_id * 1000
      thing pixels: class_id * 1000 + instance_id (1-indexed per class)
  - Saved as .npy (lossless int32) and _panoptic.png (uint16 visualization)

Usage:
    python mbps_pytorch/generate_panoptic_pseudolabels.py \
        --semantic_dir /data/cityscapes/pseudo_semantic_dinov3/train \
        --instance_dir /data/cityscapes/pseudo_instance_depth/train \
        --stuff_things /data/cityscapes/pseudo_semantic_dinov3/stuff_things.json \
        --output_dir /data/cityscapes/pseudo_panoptic/train
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
from scipy import ndimage
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CS_NAMES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic_light", 7: "traffic_sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}


def load_stuff_things(path):
    """Load stuff/things classification from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    stuff_ids = set(data["stuff_ids"])
    thing_ids = set(data["thing_ids"])
    logger.info(f"Loaded stuff/things from {path}")
    logger.info(f"  Stuff ({len(stuff_ids)}): {sorted(stuff_ids)}")
    logger.info(f"  Things ({len(thing_ids)}): {sorted(thing_ids)}")
    return stuff_ids, thing_ids


def generate_panoptic_map(semantic, instance_masks, instance_scores,
                          stuff_ids, thing_ids, min_stuff_area=64):
    """Generate a panoptic map from semantic labels and instance masks.

    Args:
        semantic: (H, W) uint8 trainID map
        instance_masks: (M, H, W) bool instance masks, or None
        instance_scores: (M,) float32 scores, or None
        stuff_ids: set of stuff trainIDs
        thing_ids: set of thing trainIDs
        min_stuff_area: minimum pixel area for a stuff segment

    Returns:
        panoptic: (H, W) int32 — class_id * 1000 + instance_id
        segment_info: list of dicts with segment metadata
    """
    H, W = semantic.shape
    panoptic = np.zeros((H, W), dtype=np.int32)
    segment_info = []
    assigned = np.zeros((H, W), dtype=bool)

    # Step 1: Place thing instances (higher priority — they overlap stuff)
    if instance_masks is not None and instance_masks.shape[0] > 0:
        # Sort by score descending (larger/more confident instances first)
        if instance_scores is not None:
            order = np.argsort(-instance_scores)
        else:
            order = np.arange(instance_masks.shape[0])

        # Track per-class instance counter
        class_inst_counter = defaultdict(int)

        for idx in order:
            mask = instance_masks[idx]
            if mask.sum() < 10:
                continue

            # Determine class from semantic map majority vote
            sem_vals = semantic[mask]
            sem_vals = sem_vals[sem_vals < 19]
            if len(sem_vals) == 0:
                continue
            majority_cls = int(np.bincount(sem_vals, minlength=19).argmax())

            # Only assign if this class is a thing
            if majority_cls not in thing_ids:
                continue

            # Avoid overwriting already-assigned pixels
            valid_mask = mask & ~assigned
            if valid_mask.sum() < 10:
                continue

            class_inst_counter[majority_cls] += 1
            inst_id = class_inst_counter[majority_cls]
            pan_id = majority_cls * 1000 + inst_id

            panoptic[valid_mask] = pan_id
            assigned[valid_mask] = True

            score = float(instance_scores[idx]) if instance_scores is not None else 1.0
            segment_info.append({
                "id": int(pan_id),
                "category_id": int(majority_cls),
                "category_name": CS_NAMES.get(majority_cls, f"class_{majority_cls}"),
                "isthing": True,
                "area": int(valid_mask.sum()),
                "score": round(score, 4),
            })

    # Step 2: Place stuff segments (one per class, from semantic map)
    for cls in sorted(stuff_ids):
        mask = (semantic == cls) & ~assigned
        if mask.sum() < min_stuff_area:
            continue

        pan_id = cls * 1000  # instance_id = 0 for stuff
        panoptic[mask] = pan_id
        assigned[mask] = True

        segment_info.append({
            "id": int(pan_id),
            "category_id": int(cls),
            "category_name": CS_NAMES.get(cls, f"class_{cls}"),
            "isthing": False,
            "area": int(mask.sum()),
        })

    # Step 3: Handle unassigned thing pixels (no instance mask covered them)
    # Use connected components to create instances from remaining thing pixels
    for cls in sorted(thing_ids):
        remaining = (semantic == cls) & ~assigned
        if remaining.sum() < 10:
            continue

        labeled, n_cc = ndimage.label(remaining)
        # Get the current max instance_id for this class
        existing_max = 0
        for seg in segment_info:
            if seg["category_id"] == cls and seg["isthing"]:
                existing_inst = seg["id"] % 1000
                existing_max = max(existing_max, existing_inst)

        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = int(cc_mask.sum())
            if area < 10:
                continue

            existing_max += 1
            pan_id = cls * 1000 + existing_max

            panoptic[cc_mask] = pan_id
            assigned[cc_mask] = True

            segment_info.append({
                "id": int(pan_id),
                "category_id": int(cls),
                "category_name": CS_NAMES.get(cls, f"class_{cls}"),
                "isthing": True,
                "area": area,
                "score": 0.1,  # low confidence for fallback CC instances
            })

    return panoptic, segment_info


def process_dataset(semantic_dir, instance_dir, output_dir, stuff_ids, thing_ids,
                    limit=None):
    """Process all images and generate panoptic pseudo-labels."""
    semantic_dir = Path(semantic_dir)
    instance_dir = Path(instance_dir) if instance_dir else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all semantic label files
    sem_files = sorted(semantic_dir.rglob("*.png"))
    if limit:
        sem_files = sem_files[:limit]
    logger.info(f"Processing {len(sem_files)} images")

    total_segments = 0
    total_stuff = 0
    total_things = 0
    per_class_counts = defaultdict(int)
    t0 = time.time()

    all_segment_info = {}

    for sem_path in tqdm(sem_files, desc="Generating panoptic"):
        # Load semantic
        semantic = np.array(Image.open(sem_path))
        H, W = semantic.shape

        # Find matching instance file
        rel = sem_path.relative_to(semantic_dir)
        instance_masks = None
        instance_scores = None

        if instance_dir is not None:
            inst_path = instance_dir / rel.parent / f"{sem_path.stem}.npz"
            if inst_path.exists():
                data = np.load(str(inst_path))
                masks_flat = data["masks"]
                scores = data["scores"] if "scores" in data else None
                num_valid = int(data["num_valid"]) if "num_valid" in data else masks_flat.shape[0]

                if num_valid > 0:
                    masks_flat = masks_flat[:num_valid]
                    hp = int(data["h_patches"]) if "h_patches" in data else None
                    wp = int(data["w_patches"]) if "w_patches" in data else None

                    if hp and wp:
                        masks_2d = masks_flat.reshape(num_valid, hp, wp)
                        # Resize to semantic resolution if needed
                        if (hp, wp) != (H, W):
                            instance_masks = np.zeros((num_valid, H, W), dtype=bool)
                            for i in range(num_valid):
                                m = Image.fromarray(masks_2d[i].astype(np.uint8) * 255)
                                instance_masks[i] = np.array(
                                    m.resize((W, H), Image.NEAREST)
                                ) > 127
                        else:
                            instance_masks = masks_2d

                    if scores is not None:
                        instance_scores = scores[:num_valid]

        # Generate panoptic map
        panoptic, segment_info = generate_panoptic_map(
            semantic, instance_masks, instance_scores,
            stuff_ids, thing_ids,
        )

        # Save panoptic map as .npy (lossless)
        out_npy = output_dir / rel.parent / f"{sem_path.stem}.npy"
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_npy), panoptic)

        # Save visualization as PNG (uint16 — can hold values up to 65535)
        # Encode as: class_id * 256 + instance_id for visual distinction
        vis = np.zeros((H, W), dtype=np.uint16)
        for seg in segment_info:
            cls = seg["category_id"]
            inst = seg["id"] % 1000
            mask = panoptic == seg["id"]
            # Color encoding for visualization
            vis[mask] = cls * 256 + inst
        out_png = output_dir / rel.parent / f"{sem_path.stem}_panoptic.png"
        Image.fromarray(vis).save(str(out_png))

        # Track stats
        image_id = str(rel).replace(".png", "")
        all_segment_info[image_id] = segment_info
        n_stuff = sum(1 for s in segment_info if not s["isthing"])
        n_things = sum(1 for s in segment_info if s["isthing"])
        total_segments += len(segment_info)
        total_stuff += n_stuff
        total_things += n_things
        for seg in segment_info:
            per_class_counts[seg["category_id"]] += 1

    elapsed = time.time() - t0
    n_images = len(sem_files)

    # Save stats
    stats = {
        "total_images": n_images,
        "total_segments": total_segments,
        "total_stuff_segments": total_stuff,
        "total_thing_segments": total_things,
        "avg_segments_per_image": round(total_segments / max(n_images, 1), 2),
        "avg_stuff_per_image": round(total_stuff / max(n_images, 1), 2),
        "avg_things_per_image": round(total_things / max(n_images, 1), 2),
        "per_class_segment_counts": {
            CS_NAMES.get(k, f"class_{k}"): v
            for k, v in sorted(per_class_counts.items())
        },
        "stuff_ids": sorted(stuff_ids),
        "thing_ids": sorted(thing_ids),
        "elapsed_seconds": round(elapsed, 1),
    }

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Done in {elapsed:.1f}s ({elapsed/max(n_images,1):.3f}s/image)")
    logger.info(f"Total segments: {total_segments} "
                f"({total_stuff} stuff + {total_things} things)")
    logger.info(f"Avg per image: {total_segments/max(n_images,1):.1f} segments "
                f"({total_stuff/max(n_images,1):.1f} stuff + "
                f"{total_things/max(n_images,1):.1f} things)")
    logger.info(f"Stats saved to {stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate panoptic pseudo-labels from semantic + instances"
    )
    parser.add_argument(
        "--semantic_dir", type=str, required=True,
        help="Directory with semantic pseudo-label PNGs (trainIDs 0-18)",
    )
    parser.add_argument(
        "--instance_dir", type=str, default=None,
        help="Directory with depth-guided instance NPZs. "
             "If not provided, thing instances are created via CC from semantic map.",
    )
    parser.add_argument(
        "--stuff_things", type=str, required=True,
        help="Path to stuff_things.json from classify_stuff_things.py",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for panoptic maps (.npy + _panoptic.png)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N images (for testing)",
    )

    args = parser.parse_args()

    stuff_ids, thing_ids = load_stuff_things(args.stuff_things)

    process_dataset(
        semantic_dir=args.semantic_dir,
        instance_dir=args.instance_dir,
        output_dir=args.output_dir,
        stuff_ids=stuff_ids,
        thing_ids=thing_ids,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
