#!/usr/bin/env python3
"""Organize flat pseudo-label files into Cityscapes subdirectory structure for evaluation.

The evaluate_cascade_pseudolabels.py expects:
    cityscapes_root/pseudo_subdir/val/aachen/aachen_xxx_leftImg8bit.png

But CUPS pseudo-labels are flat:
    cups_pseudo_labels_xxx/aachen_xxx_leftImg8bit_semantic.png
    cups_pseudo_labels_xxx/aachen_xxx_leftImg8bit_instance.png

This script creates symlinks in the expected structure.
"""

import argparse
import re
from pathlib import Path


def organize(flat_dir: Path, output_dir: Path, split: str = "val"):
    """Create city subdirectories and symlinks for evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_dir / split
    split_dir.mkdir(exist_ok=True)

    # Find all PNG files in flat directory
    png_files = list(flat_dir.glob("*_leftImg8bit*.png"))
    print(f"Found {len(png_files)} PNG files in {flat_dir}")

    created = 0
    for png_path in png_files:
        stem = png_path.stem  # e.g., aachen_000000_000019_leftImg8bit_semantic
        # Extract city name
        match = re.match(r"^([a-z]+)_\d+_\d+_leftImg8bit", stem)
        if not match:
            continue
        city = match.group(1)

        city_dir = split_dir / city
        city_dir.mkdir(exist_ok=True)

        # Create symlink: aachen_000000_000019_leftImg8bit.png
        # Remove _semantic or _instance suffix for evaluation script
        base_stem = re.sub(r"_(semantic|instance)$", "", stem)
        link_path = city_dir / (base_stem + ".png")

        if not link_path.exists():
            link_path.symlink_to(png_path.resolve())
            created += 1

    print(f"Created {created} symlinks in {output_dir}/{split}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    organize(args.flat_dir, args.output_dir, args.split)


if __name__ == "__main__":
    main()
