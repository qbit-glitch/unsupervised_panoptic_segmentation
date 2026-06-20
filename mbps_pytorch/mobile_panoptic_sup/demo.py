"""Overlay panoptic masks + category labels on an image (requirement #2).

PIL-only (no cv2 dependency) so it runs in ``.venv``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from auto_annotation import taxonomy_coco as T  # noqa: E402
from mbps_pytorch.mobile_panoptic_sup.coco_eval import coco_contiguous_maps  # noqa: E402

DIV = 1000
_, _IDX2NAME, _, _ = coco_contiguous_maps()


def overlay(image_rgb: np.ndarray, pan: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend per-segment colors over the image and draw a category name per segment."""
    out = image_rgb.copy()
    placed = []
    for seg_id in np.unique(pan):
        cls = int(seg_id) // DIV
        if cls == T.VOID_IDX:
            continue
        mask = pan == seg_id
        color = np.array(T.COCO_CLASSES[cls].color, np.uint8)
        out[mask] = (alpha * color + (1 - alpha) * out[mask]).astype(np.uint8)
        ys, xs = np.where(mask)
        placed.append((_IDX2NAME[cls], int(xs.mean()), int(ys.mean())))
    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    for name, x, y in placed:
        draw.text((x, y), name, fill=(255, 255, 255))
    return np.asarray(img)


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--panoptic", type=Path, required=True, help="class*1000+inst .npy")
    ap.add_argument("--out", type=Path, default=Path("demo_overlay.png"))
    args = ap.parse_args()
    img = np.asarray(Image.open(args.image).convert("RGB"))
    pan = np.load(args.panoptic)
    Image.fromarray(overlay(img, pan)).save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    _main()
