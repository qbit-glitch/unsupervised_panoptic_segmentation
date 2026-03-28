#!/usr/bin/env python3
"""Convert our pseudo-labels to CUPS PseudoLabelDataset flat directory format.

Our data:
  - Semantic: pseudo_semantic_cause_crf/{split}/{city}/{stem}.png (27-class CAUSE)
  - Instance: sweep_instances/gt0.10_ma500/{split}/{city}/{stem}_leftImg8bit.npz (NPZ masks)

CUPS expects (flat dir):
  - {stem}_leftImg8bit_semantic.png (uint8, N-class)
  - {stem}_leftImg8bit_instance.png (uint16, instance IDs)
  - {stem}_leftImg8bit.pt (distribution stats)

Two modes:
  - Default (27-class): loads NPZ instance masks, maps 19-class trainIDs to 27-class CAUSE IDs
  - Raw cluster mode (--cc_instances --num_classes 50): generates CC instances from thing clusters

Usage:
    # Original 27-class with NPZ instances:
    python mbps_pytorch/convert_to_cups_format.py \
        --cityscapes_root /path/to/cityscapes \
        --semantic_subdir pseudo_semantic_cause_crf \
        --instance_subdir sweep_instances/gt0.10_ma500 \
        --output_subdir cups_pseudo_labels \
        --split train

    # Raw k=50 overclusters with CC instances:
    python mbps_pytorch/convert_to_cups_format.py \
        --cityscapes_root /path/to/cityscapes \
        --semantic_subdir pseudo_semantic_raw_k50 \
        --output_subdir cups_pseudo_labels_k50 \
        --split train \
        --num_classes 50 --cc_instances \
        --centroids_path /path/to/pseudo_semantic_raw_k50/kmeans_centroids.npz
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 27-class CAUSE thing class IDs (matching Cityscapes labelID 7+offset for has_instances)
# person=17, rider=18, car=19, truck=20, bus=21, caravan=22, trailer=23, train=24, motorcycle=25, bicycle=26
THING_IDS_27 = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26}

# Mapping from 19-class trainID to 27-class CAUSE ID
TRAINID_TO_CAUSE27 = {
    0: 0, 1: 1, 2: 4, 3: 5, 4: 6, 5: 10, 6: 12, 7: 13,
    8: 14, 9: 15, 10: 16, 11: 17, 12: 18, 13: 19, 14: 20,
    15: 21, 16: 24, 17: 25, 18: 26,
}


def determine_thing_cluster_ids(centroids_path):
    """Determine which raw cluster IDs are 'thing' clusters.

    Loads cluster_to_class mapping from kmeans_centroids.npz and returns the set
    of cluster IDs whose majority GT class is a Cityscapes thing trainID (11-18):
    person(11), rider(12), car(13), truck(14), bus(15), train(16), motorcycle(17), bicycle(18).
    """
    THING_TRAINIDS = {11, 12, 13, 14, 15, 16, 17, 18}
    data = np.load(centroids_path)
    cluster_to_class = data["cluster_to_class"]  # (k,) array, values 0-18 trainIDs
    thing_ids = set()
    for cluster_id, gt_class in enumerate(cluster_to_class):
        if int(gt_class) in THING_TRAINIDS:
            thing_ids.add(cluster_id)
    logger.info(f"Thing cluster IDs ({len(thing_ids)}/{len(cluster_to_class)}): {sorted(thing_ids)}")
    return thing_ids


def build_instance_map_cc(semantic, thing_ids, min_area=100):
    """Build instance map from connected components on thing-cluster regions.

    For each thing cluster ID, finds connected components and assigns unique instance IDs.
    Filters out components smaller than min_area pixels.

    Args:
        semantic: (H, W) uint8 array with cluster IDs (0 to k-1)
        thing_ids: set of cluster IDs considered 'things'
        min_area: minimum pixel area for an instance

    Returns:
        instance_map: (H, W) uint16 array with instance IDs (0=background)
    """
    H, W = semantic.shape
    instance_map = np.zeros((H, W), dtype=np.uint16)
    instance_id = 1

    for cluster_id in sorted(thing_ids):
        mask = semantic == cluster_id
        if mask.sum() == 0:
            continue
        labeled, n_components = ndimage.label(mask)
        for comp_id in range(1, n_components + 1):
            comp_mask = labeled == comp_id
            if comp_mask.sum() >= min_area:
                instance_map[comp_mask] = instance_id
                instance_id += 1

    return instance_map


def load_stuff_things_json(path):
    """Load stuff-things JSON and convert thing IDs from 19-class to 27-class."""
    with open(path) as f:
        data = json.load(f)
    thing_ids_19 = set(data["thing_ids"])
    thing_ids_27 = set()
    for tid in thing_ids_19:
        if tid in TRAINID_TO_CAUSE27:
            thing_ids_27.add(TRAINID_TO_CAUSE27[tid])
    return thing_ids_27


def load_instance_npz(npz_path):
    """Load instance masks from our NPZ format."""
    data = np.load(str(npz_path))
    masks = data["masks"]
    scores = data["scores"]
    num_valid = int(data["num_valid"])
    h = int(data["h_patches"])
    w = int(data["w_patches"])
    masks = masks[:num_valid]
    scores = scores[:num_valid]
    return masks, scores, h, w


def build_instance_map(masks, scores, semantic, thing_ids_27, h, w, num_classes=27):
    """Convert boolean masks to single-channel uint16 instance map.

    Only assigns instance IDs to masks whose majority semantic class is a thing.
    """
    instance_map = np.zeros((h, w), dtype=np.uint16)
    instance_id = 1

    if masks.shape[0] == 0:
        return instance_map

    # Sort by score descending (higher confidence first)
    order = np.argsort(-scores)

    for idx in order:
        mask = masks[idx].reshape(h, w)
        if mask.sum() == 0:
            continue

        # Check majority class
        sem_vals = semantic[mask]
        if len(sem_vals) == 0:
            continue
        majority_class = int(np.bincount(sem_vals, minlength=num_classes).argmax())

        if majority_class in thing_ids_27:
            # Don't overwrite existing instance pixels (higher confidence first)
            new_pixels = mask & (instance_map == 0)
            if new_pixels.sum() > 0:
                instance_map[new_pixels] = instance_id
                instance_id += 1

    return instance_map


def compute_distributions(semantic, instance_map, num_classes=27):
    """Compute per-class pixel distributions for CUPS thing/stuff split."""
    sem_flat = semantic.flatten().astype(np.int64)
    inst_flat = instance_map.flatten()

    # All pixels distribution
    valid_mask = sem_flat < num_classes
    dist_all = torch.zeros(num_classes, dtype=torch.float32)
    if valid_mask.sum() > 0:
        counts = np.bincount(sem_flat[valid_mask], minlength=num_classes)
        dist_all = torch.from_numpy(counts[:num_classes].astype(np.float32))

    # Inside object proposals distribution
    inside_mask = (inst_flat > 0) & valid_mask
    dist_inside = torch.zeros(num_classes, dtype=torch.float32)
    if inside_mask.sum() > 0:
        counts = np.bincount(sem_flat[inside_mask], minlength=num_classes)
        dist_inside = torch.from_numpy(counts[:num_classes].astype(np.float32))

    return {
        "distribution all pixels": dist_all,
        "distribution inside object proposals": dist_inside,
    }


def find_semantic_files(semantic_dir, split):
    """Find all semantic pseudo-label PNGs."""
    split_dir = Path(semantic_dir) / split
    files = []
    for city_dir in sorted(split_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        for png_path in sorted(city_dir.glob("*.png")):
            files.append(png_path)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Convert pseudo-labels to CUPS PseudoLabelDataset format"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--semantic_subdir", type=str, default="pseudo_semantic_cause_crf",
                        help="Subdirectory under cityscapes_root with 27-class semantic labels")
    parser.add_argument("--instance_subdir", type=str, default="sweep_instances/gt0.10_ma500",
                        help="Subdirectory under cityscapes_root with instance NPZ files")
    parser.add_argument("--output_subdir", type=str, default="cups_pseudo_labels",
                        help="Output subdirectory under cityscapes_root")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--stuff_things", type=str, default=None,
                        help="Path to stuff_things JSON (19-class trainID format)")
    parser.add_argument("--trainid_input", action="store_true",
                        help="If set, input semantics are 19-class trainIDs (0-18) and will be remapped to 27-class CAUSE IDs")
    parser.add_argument("--target_h", type=int, default=1024, help="Target height for labels")
    parser.add_argument("--target_w", type=int, default=2048, help="Target width for labels")
    parser.add_argument("--num_classes", type=int, default=27,
                        help="Number of semantic classes (27 for CAUSE, 50 for raw k=50 overclusters)")
    parser.add_argument("--cc_instances", action="store_true",
                        help="Generate instances from connected components on thing-cluster regions "
                             "instead of loading NPZ masks")
    parser.add_argument("--centroids_path", type=str, default=None,
                        help="Path to kmeans_centroids.npz for determining thing cluster IDs "
                             "(required when --cc_instances is set)")
    parser.add_argument("--min_instance_area", type=int, default=100,
                        help="Minimum pixel area for a CC instance (default: 100)")

    args = parser.parse_args()

    if args.cc_instances and args.centroids_path is None:
        parser.error("--centroids_path is required when --cc_instances is set")

    cs_root = Path(args.cityscapes_root)
    semantic_dir = cs_root / args.semantic_subdir
    instance_dir = cs_root / args.instance_subdir if not args.cc_instances else None
    output_dir = cs_root / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = args.num_classes
    use_cc = args.cc_instances

    # Determine thing IDs
    if use_cc:
        thing_ids = determine_thing_cluster_ids(args.centroids_path)
        logger.info(f"Mode: CC instances on {num_classes} raw clusters")
    elif args.stuff_things:
        thing_ids = load_stuff_things_json(args.stuff_things)
        logger.info(f"Thing IDs (27-class): {sorted(thing_ids)}")
    else:
        thing_ids = THING_IDS_27
        logger.info(f"Thing IDs (27-class): {sorted(thing_ids)}")

    # Build trainID→CAUSE27 lookup table if needed
    trainid_lut = None
    if args.trainid_input:
        trainid_lut = np.full(256, 255, dtype=np.uint8)  # 255 = ignore
        for tid, cid in TRAINID_TO_CAUSE27.items():
            trainid_lut[tid] = cid
        logger.info("Input semantics are 19-class trainIDs — will remap to 27-class CAUSE IDs")

    logger.info(f"Num classes: {num_classes}")
    logger.info(f"Semantic dir: {semantic_dir}")
    if instance_dir:
        logger.info(f"Instance dir: {instance_dir}")
    logger.info(f"Output dir: {output_dir}")

    # Find all semantic files
    sem_files = find_semantic_files(semantic_dir, args.split)
    logger.info(f"Found {len(sem_files)} semantic labels for {args.split}")

    t0 = time.time()
    total_instances = 0
    total_thing_instances = 0
    skipped = 0

    for i, sem_path in enumerate(sem_files):
        city = sem_path.parent.name
        stem = sem_path.stem  # e.g., "aachen_000000_000019"

        # Load 27-class semantic label
        semantic = np.array(Image.open(sem_path))

        # Resize semantic to target resolution if needed
        if semantic.shape != (args.target_h, args.target_w):
            semantic = np.array(
                Image.fromarray(semantic).resize(
                    (args.target_w, args.target_h), Image.NEAREST
                )
            )

        # Remap 19-class trainIDs to 27-class CAUSE IDs
        if trainid_lut is not None:
            semantic = trainid_lut[semantic]

        if use_cc:
            # CC instances from raw cluster semantic map
            instance_map = build_instance_map_cc(
                semantic, thing_ids, min_area=args.min_instance_area
            )
            n_instances = int(instance_map.max())
            total_thing_instances += n_instances
        else:
            # Try to find instance NPZ
            npz_path = instance_dir / args.split / city / f"{stem}_leftImg8bit.npz"
            if not npz_path.exists():
                npz_path = instance_dir / args.split / city / f"{stem}.npz"

            if npz_path.exists():
                masks, scores, h, w = load_instance_npz(npz_path)

                # Resize semantic to instance resolution for class assignment
                if (h, w) != (args.target_h, args.target_w):
                    sem_for_inst = np.array(
                        Image.fromarray(semantic).resize((w, h), Image.NEAREST)
                    )
                else:
                    sem_for_inst = semantic

                # Build instance map at instance resolution
                instance_map = build_instance_map(
                    masks, scores, sem_for_inst, thing_ids, h, w,
                    num_classes=num_classes
                )

                # Resize instance map to target resolution
                if (h, w) != (args.target_h, args.target_w):
                    instance_map = np.array(
                        Image.fromarray(instance_map).resize(
                            (args.target_w, args.target_h), Image.NEAREST
                        )
                    )

                n_instances = int(instance_map.max())
                total_instances += masks.shape[0]
                total_thing_instances += n_instances
            else:
                instance_map = np.zeros((args.target_h, args.target_w), dtype=np.uint16)
                skipped += 1

        # Compute distribution stats
        stats = compute_distributions(semantic, instance_map, num_classes=num_classes)

        # Save files with CUPS naming convention
        out_stem = f"{stem}_leftImg8bit"

        # Semantic PNG (uint8)
        sem_out = output_dir / f"{out_stem}_semantic.png"
        Image.fromarray(semantic.astype(np.uint8)).save(str(sem_out))

        # Instance PNG (uint16)
        inst_out = output_dir / f"{out_stem}_instance.png"
        Image.fromarray(instance_map.astype(np.uint16)).save(str(inst_out))

        # Distribution PT
        pt_out = output_dir / f"{out_stem}.pt"
        torch.save(stats, str(pt_out))

        if (i + 1) % 500 == 0 or i == 0:
            logger.info(f"  [{i+1}/{len(sem_files)}] {stem}: {int(instance_map.max())} thing instances")

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s")
    logger.info(f"Total: {len(sem_files)} images, {total_instances} raw instances, "
                f"{total_thing_instances} thing instances")
    logger.info(f"Skipped (no NPZ): {skipped}")
    logger.info(f"Output: {output_dir} ({len(sem_files) * 3} files)")


if __name__ == "__main__":
    main()
