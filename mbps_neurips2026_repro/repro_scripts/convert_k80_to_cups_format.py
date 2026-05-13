"""Convert k=80 pseudo-labels to CUPS flat format for training.

Reads:
  - pseudo_semantic_raw_k80/train/{city}/{city}_{seq}_{frame}.png (2048x1024, uint8, 0-79)
  - pseudo_instance_spidepth/train/{city}/{city}_{seq}_{frame}_instance.png (1024x512, uint16)

Writes:
  - cups_pseudo_labels_k80/{city}_{seq}_{frame}_leftImg8bit_semantic.png (2048x1024, uint8)
  - cups_pseudo_labels_k80/{city}_{seq}_{frame}_leftImg8bit_instance.png (2048x1024, uint8)

Usage:
    python scripts/convert_k80_to_cups_format.py --data_root /path/to/cityscapes
"""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def convert(data_root: str) -> None:
    sem_root = os.path.join(data_root, "pseudo_semantic_raw_k80", "train")
    inst_root = os.path.join(data_root, "pseudo_instance_spidepth", "train")
    out_dir = os.path.join(data_root, "cups_pseudo_labels_k80")
    os.makedirs(out_dir, exist_ok=True)

    cities = sorted(
        d for d in os.listdir(sem_root)
        if os.path.isdir(os.path.join(sem_root, d))
    )
    print(f"Found {len(cities)} cities in {sem_root}")

    total = 0
    non_empty_inst = 0
    skipped = 0

    for city in cities:
        sem_city = os.path.join(sem_root, city)
        inst_city = os.path.join(inst_root, city)

        sem_files = sorted(f for f in os.listdir(sem_city) if f.endswith(".png"))

        for sem_file in sem_files:
            # Parse name: {city}_{seq}_{frame}.png
            stem = sem_file.replace(".png", "")
            # Instance file: {city}_{seq}_{frame}_instance.png
            inst_file = f"{stem}_instance.png"

            sem_path = os.path.join(sem_city, sem_file)
            inst_path = os.path.join(inst_city, inst_file)

            if not os.path.exists(inst_path):
                skipped += 1
                continue

            # Load semantic (already 2048x1024, uint8)
            sem_img = Image.open(sem_path)
            sem_arr = np.array(sem_img)

            # Load instance (may be 1024x512, uint16)
            inst_img = Image.open(inst_path)
            inst_arr = np.array(inst_img)

            # Upscale instance to match semantic resolution if needed
            target_h, target_w = sem_arr.shape[:2]
            if inst_arr.shape[0] != target_h or inst_arr.shape[1] != target_w:
                inst_img_resized = Image.fromarray(inst_arr).resize(
                    (target_w, target_h), Image.NEAREST
                )
                inst_arr = np.array(inst_img_resized)

            # Convert to uint8 (CUPS expects uint8 PNGs)
            if inst_arr.max() > 255:
                print(f"  WARNING: {inst_file} has max={inst_arr.max()}, clipping to 255")
            inst_arr = inst_arr.astype(np.uint8)
            sem_arr = sem_arr.astype(np.uint8)

            # Save in CUPS flat format
            cups_stem = f"{stem}_leftImg8bit"
            sem_out = os.path.join(out_dir, f"{cups_stem}_semantic.png")
            inst_out = os.path.join(out_dir, f"{cups_stem}_instance.png")

            Image.fromarray(sem_arr).save(sem_out)
            Image.fromarray(inst_arr).save(inst_out)

            total += 1
            if inst_arr.max() > 0:
                non_empty_inst += 1

        print(f"  {city}: {len(sem_files)} images processed")

    print(f"\nDone. {total} images converted to {out_dir}")
    print(f"Non-empty instances: {non_empty_inst}/{total} ({100*non_empty_inst/max(total,1):.1f}%)")
    print(f"Skipped (no instance match): {skipped}")
    print(f"Total files: {total * 2} (semantic + instance)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default="/Users/qbit-glitch/Desktop/datasets/cityscapes",
        help="Path to cityscapes root",
    )
    args = parser.parse_args()
    convert(args.data_root)
