#!/usr/bin/env python3
"""Remap CAUSE 27-class semantic PNGs to Cityscapes 19-class trainIDs."""
import argparse
import os

import numpy as np
from PIL import Image

CAUSE27_TO_TRAINID = np.full(256, 255, dtype=np.uint8)
for c27, t19 in {
    0: 0, 1: 1, 2: 255, 3: 255, 4: 2, 5: 3, 6: 4,
    7: 255, 8: 255, 9: 255, 10: 5, 11: 5, 12: 6, 13: 7,
    14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14,
    21: 15, 22: 13, 23: 14, 24: 16, 25: 17, 26: 18,
}.items():
    CAUSE27_TO_TRAINID[c27] = t19


def remap_split(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cities = sorted(
        [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    )
    total = 0
    for city in cities:
        city_in = os.path.join(input_dir, city)
        city_out = os.path.join(output_dir, city)
        os.makedirs(city_out, exist_ok=True)
        pngs = sorted([f for f in os.listdir(city_in) if f.endswith(".png")])
        for f in pngs:
            sem = np.array(Image.open(os.path.join(city_in, f)))
            remapped = CAUSE27_TO_TRAINID[sem]
            Image.fromarray(remapped).save(os.path.join(city_out, f))
            total += 1
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remap CAUSE 27-class semantic PNGs to 19-class trainIDs"
    )
    parser.add_argument("--input_dir", required=True, help="CAUSE 27-class output directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for trainID PNGs")
    parser.add_argument(
        "--split", default="both", choices=["train", "val", "both"],
        help="Which split(s) to remap",
    )
    args = parser.parse_args()

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for s in splits:
        in_d = os.path.join(args.input_dir, s)
        out_d = os.path.join(args.output_dir, s)
        if os.path.isdir(in_d):
            n = remap_split(in_d, out_d)
            print(f"Remapped {n} images for {s} split")
        else:
            print(f"Skipping {s} — {in_d} not found")
