#!/usr/bin/env python3
"""Assemble CUPS-compatible pseudo-label dir: DA3 instances + DINOv3 k=80 semantics.

Input dirs (at 512×1024):
  --semantic_root  pseudo_semantic_raw_dinov3_k80/train/{city}/{stem}.png
  --instance_root  pseudo_instance_dav3/train/{city}/{stem}_instance.png

Output dir (at 1024×2048, CUPS-compatible flat structure):
  {out}/{stem}_leftImg8bit_semantic.png  (uint8, nearest-upsampled ×2)
  {out}/{stem}_leftImg8bit_instance.png  (uint16, nearest-upsampled ×2)
  {out}/{stem}_leftImg8bit.pt            (class distribution tensors)

Usage (run locally, before rsyncing to remote):
  python scripts/assemble_cups_pseudo_da3.py \
    --semantic_root /path/cityscapes/pseudo_semantic_raw_dinov3_k80/train \
    --instance_root /path/cityscapes/pseudo_instance_dav3/train \
    --output_dir    /path/cityscapes/cups_pseudo_labels_da3 \
    --num_classes 80
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Output resolution to match existing CUPS pseudo-labels (1024×2048 Cityscapes full res)
OUT_H, OUT_W = 1024, 2048


def compute_distributions(
    semantic: np.ndarray, instance: np.ndarray, num_classes: int
) -> dict:
    """Compute per-class pixel distributions required by CUPS PseudoLabelDataset.

    Args:
        semantic: (H, W) uint8 array with class IDs 0..num_classes-1
        instance: (H, W) uint16 array with instance IDs (0 = background)
        num_classes: total number of semantic classes (e.g. 80)

    Returns:
        dict with:
          'distribution all pixels': float32 tensor (num_classes,)
          'distribution inside object proposals': float32 tensor (num_classes,)
    """
    in_instance_mask = instance > 0
    dist_all = np.zeros(num_classes, dtype=np.float32)
    dist_in = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        cls_mask = semantic == c
        dist_all[c] = float(cls_mask.sum())
        dist_in[c] = float((cls_mask & in_instance_mask).sum())
    return {
        "distribution all pixels": torch.from_numpy(dist_all),
        "distribution inside object proposals": torch.from_numpy(dist_in),
    }


def process_city(
    city_name: str,
    semantic_city_dir: Path,
    instance_city_dir: Path,
    output_dir: Path,
    num_classes: int,
    stats: dict,
) -> None:
    """Process all images in a single city."""
    sem_files = sorted(semantic_city_dir.glob("*.png"))
    for sem_path in sem_files:
        stem = sem_path.stem  # e.g. aachen_000000_000019

        # Instance PNG: {stem}_instance.png
        inst_path = instance_city_dir / f"{stem}_instance.png"
        if not inst_path.exists():
            logger.warning(f"Missing instance for {stem}, skipping")
            stats["skipped"] += 1
            continue

        # CUPS output names
        out_sem = output_dir / f"{stem}_leftImg8bit_semantic.png"
        out_inst = output_dir / f"{stem}_leftImg8bit_instance.png"
        out_pt = output_dir / f"{stem}_leftImg8bit.pt"

        if out_sem.exists() and out_inst.exists() and out_pt.exists():
            stats["skipped_existing"] += 1
            continue

        # Load at native 512×1024
        sem_arr = np.array(Image.open(sem_path))   # (512, 1024) uint8
        inst_arr = np.array(Image.open(inst_path)) # (512, 1024) uint16

        # Upsample ×2 to 1024×2048 (nearest, preserves class/instance IDs)
        sem_up = np.array(
            Image.fromarray(sem_arr).resize((OUT_W, OUT_H), Image.NEAREST)
        )
        inst_up = np.array(
            Image.fromarray(inst_arr).resize((OUT_W, OUT_H), Image.NEAREST)
        )

        # Save semantic PNG
        if not out_sem.exists():
            Image.fromarray(sem_up.astype(np.uint8)).save(out_sem)

        # Save instance PNG (uint16)
        if not out_inst.exists():
            Image.fromarray(inst_up.astype(np.uint16)).save(out_inst)

        # Compute and save .pt distributions
        if not out_pt.exists():
            dists = compute_distributions(sem_up, inst_up, num_classes)
            torch.save(dists, out_pt)

        stats["processed"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--semantic_root",
        required=True,
        help="Path to pseudo_semantic_raw_dinov3_k80/train/",
    )
    parser.add_argument(
        "--instance_root",
        required=True,
        help="Path to pseudo_instance_dav3/train/",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output flat CUPS-compatible dir (cups_pseudo_labels_da3/)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=80,
        help="Number of semantic classes (default: 80 for k=80 overclustering)",
    )
    args = parser.parse_args()

    sem_root = Path(args.semantic_root)
    inst_root = Path(args.instance_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Semantic root: {sem_root}")
    logger.info(f"Instance root: {inst_root}")
    logger.info(f"Output dir:    {out_dir}")
    logger.info(f"Output resolution: {OUT_H}×{OUT_W} (upsampled ×2 from 512×1024)")
    logger.info(f"Num classes: {args.num_classes}")

    cities = sorted([d.name for d in sem_root.iterdir() if d.is_dir()])
    logger.info(f"Found {len(cities)} cities: {cities[:5]}...")

    stats = {"processed": 0, "skipped": 0, "skipped_existing": 0}

    for city in tqdm(cities, desc="Cities"):
        sem_city_dir = sem_root / city
        inst_city_dir = inst_root / city
        if not inst_city_dir.exists():
            logger.warning(f"No instance dir for city {city}, skipping")
            continue
        process_city(
            city_name=city,
            semantic_city_dir=sem_city_dir,
            instance_city_dir=inst_city_dir,
            output_dir=out_dir,
            num_classes=args.num_classes,
            stats=stats,
        )

    total = stats["processed"] + stats["skipped"] + stats["skipped_existing"]
    logger.info(
        f"Done. Processed: {stats['processed']}, "
        f"Skipped (missing): {stats['skipped']}, "
        f"Skipped (existing): {stats['skipped_existing']}, "
        f"Total: {total}"
    )
    logger.info(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
