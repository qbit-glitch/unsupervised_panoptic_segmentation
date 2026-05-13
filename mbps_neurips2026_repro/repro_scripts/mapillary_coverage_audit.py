"""Compute exact Mapillary v2 → Cityscapes-19 coverage statistics.

Outputs:
- Total Mapillary pixels evaluated
- Fraction mapped to a Cityscapes class vs ignored (void)
- Per-Cityscapes-class pixel coverage in Mapillary val
"""

from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("/Users/qbit-glitch/Desktop/datasets/mapillary-vistas-v2/validation/v2.0/labels")

# Mapillary class index → Cityscapes trainID (from refs/cups/evaluate_mapillary.py)
MAPILLARY_TO_CS = {
    21: 0, 19: 1, 24: 1, 27: 2, 12: 3, 5: 4, 85: 5, 88: 5, 90: 6, 92: 6,
    93: 6, 100: 7, 64: 8, 63: 9, 61: 10, 30: 11, 32: 12, 33: 12, 34: 12,
    108: 13, 114: 14, 107: 15, 111: 16, 110: 17, 105: 18,
}
CS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]
NUM_MAPILLARY_CLASSES = 124  # v2.0
VOID_ID = 255


def main():
    files = sorted(ROOT.glob("*.png"))
    print(f"Found {len(files)} Mapillary v2 val labels")

    # Build mapping LUT
    lut = np.full(256, VOID_ID, dtype=np.int32)
    for mv, cs in MAPILLARY_TO_CS.items():
        lut[mv] = cs

    cs_pixel_counts = np.zeros(20, dtype=np.int64)  # 19 + void
    total = 0
    for i, f in enumerate(files):
        if i % 200 == 0:
            print(f"  {i}/{len(files)}")
        arr = np.array(Image.open(f))
        if arr.ndim == 3:
            arr = arr[..., 0]
        mapped = lut[arr]
        for c in range(19):
            cs_pixel_counts[c] += (mapped == c).sum()
        cs_pixel_counts[19] += (mapped == VOID_ID).sum()
        total += arr.size

    mapped_total = cs_pixel_counts[:19].sum()
    ignored = cs_pixel_counts[19]
    print()
    print("=== Class-level coverage ===")
    print(f"  Mapillary total classes: {NUM_MAPILLARY_CLASSES}")
    print(f"  Mapped to Cityscapes-19: {len(set(MAPILLARY_TO_CS.values()))} unique CS classes (out of 19)")
    print(f"  Mapillary classes that map: {len(MAPILLARY_TO_CS)}/{NUM_MAPILLARY_CLASSES}")
    print(f"  Mapillary classes ignored: {NUM_MAPILLARY_CLASSES - len(MAPILLARY_TO_CS)}/{NUM_MAPILLARY_CLASSES}")
    print()
    print("=== Pixel-level coverage ===")
    print(f"  Total pixels:    {total:>14,}")
    print(f"  Mapped pixels:   {mapped_total:>14,}  ({100*mapped_total/total:.2f}%)")
    print(f"  Ignored pixels:  {ignored:>14,}  ({100*ignored/total:.2f}%)")
    print()
    print("=== Per-Cityscapes-class pixel fraction (of mapped pixels) ===")
    for c in range(19):
        frac = cs_pixel_counts[c] / max(mapped_total, 1) * 100
        print(f"  {c:2d} {CS_NAMES[c]:<14s} {cs_pixel_counts[c]:>13,} ({frac:5.2f}%)")


if __name__ == "__main__":
    main()
