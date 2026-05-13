#!/usr/bin/env python
"""Compute per-class pixel frequencies from Cityscapes training panoptic labels.

Outputs a normalized frequency list (27 floats) that can be pasted into config:
  SELF_TRAINING.CLASS_FREQUENCIES: [...]

Cityscapes 27-class order (index):
  0: road, 1: sidewalk, 2: parking, 3: rail track, 4: building, 5: wall, 6: fence,
  7: guard rail, 8: bridge, 9: tunnel, 10: pole, 11: polegroup, 12: traffic light,
  13: traffic sign, 14: vegetation, 15: terrain, 16: sky, 17: person, 18: rider,
  19: car, 20: truck, 21: bus, 22: caravan, 23: trailer, 24: train, 25: motorcycle,
  26: bicycle
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


CITYSCAPES_27_NAMES = [
    "road", "sidewalk", "parking", "rail track", "building", "wall", "fence",
    "guard rail", "bridge", "tunnel", "pole", "polegroup", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider",
    "car", "truck", "bus", "caravan", "trailer", "train", "motorcycle", "bicycle",
]


def load_panoptic_labels(panoptic_dir: str) -> list:
    """Load all panoptic PNG labels from the given directory."""
    panoptic_dir = Path(panoptic_dir)
    return sorted(panoptic_dir.glob("*.png"))


def compute_frequencies(panoptic_dir: str, num_classes: int = 27) -> np.ndarray:
    """Compute normalized per-class pixel frequencies from panoptic labels.

    Args:
        panoptic_dir: Path to directory containing panoptic PNG labels.
        num_classes: Number of semantic classes (default 27).

    Returns:
        Array of shape (num_classes,) with normalized frequencies.
    """
    label_files = load_panoptic_labels(panoptic_dir)
    if not label_files:
        raise ValueError(f"No panoptic PNG files found in {panoptic_dir}")

    class_counts = Counter()
    total_pixels = 0

    for f in label_files:
        img = np.array(Image.open(f))
        # Panoptic labels encode instance id in lower bits and semantic class in upper bits.
        # Semantic class = img // 1000 for Cityscapes panoptic format.
        semantic = img // 1000
        unique, counts = np.unique(semantic, return_counts=True)
        for u, c in zip(unique.tolist(), counts.tolist()):
            class_counts[u] += c
            total_pixels += c

    frequencies = np.zeros(num_classes, dtype=np.float64)
    for cls_id in range(num_classes):
        frequencies[cls_id] = class_counts.get(cls_id, 0) / max(total_pixels, 1)

    return frequencies


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Cityscapes class pixel frequencies")
    parser.add_argument(
        "--panoptic_dir",
        type=str,
        default="datasets/Cityscapes/gtFine/train",
        help="Directory containing panoptic PNG labels",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=27,
        help="Number of semantic classes",
    )
    args = parser.parse_args()

    # Cityscapes panoptic labels are typically under gtFine/cityscapes_panoptic_train/
    # or similar; allow the user to point directly at the PNG folder.
    panoptic_dir = Path(args.panoptic_dir)
    if not panoptic_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {panoptic_dir}")

    frequencies = compute_frequencies(str(panoptic_dir), num_classes=args.num_classes)

    print("\n# Normalized per-class pixel frequencies (Cityscapes, 27 classes)\n")
    print("_C.SELF_TRAINING.CLASS_FREQUENCIES = [")
    for i, (name, freq) in enumerate(zip(CITYSCAPES_27_NAMES, frequencies)):
        print(f"    {freq:.8e},  # {i:2d}: {name}")
    print("]\n")

    print("# Verification: sum =", frequencies.sum())


if __name__ == "__main__":
    main()
