#!/usr/bin/env python3
"""Build a 1-shot exemplar (image + mask) per COCO stuff class from val GT.

INSID3 is in-context: it needs one annotated reference per concept. We pick, for
each stuff class, the val image where that class has the largest segment. These
exemplars are a tiny few-shot REFERENCE set (disclosed) — they are NOT used as
training labels. Run in .venv.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_coco_stuff_exemplars")

from auto_annotation import taxonomy_coco as T  # noqa: E402
from mbps_pytorch.mobile_panoptic_sup import coco_eval as CE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "auto_annotation/data/exemplars_coco")
    ap.add_argument("--min_area", type=int, default=4000)
    args = ap.parse_args()

    catid2idx, *_ = CE.coco_contiguous_maps()
    data = CE._gt_data()
    # largest-area (image, segment) per stuff class, ranked by GT segment area
    best: dict[int, tuple] = {}
    for ann in data["annotations"]:
        for s in ann["segments_info"]:
            idx = catid2idx[s["category_id"]]
            if idx not in T.STUFF_IDXS or s["area"] < args.min_area:
                continue
            if idx not in best or s["area"] > best[idx][0]:
                best[idx] = (s["area"], ann["image_id"], s["id"], ann["file_name"])

    args.out.mkdir(parents=True, exist_ok=True)
    for idx, (area, img_id, sid, png) in sorted(best.items()):
        seg_map = CE._rgb2id(CE._GT_PNG_DIR / png)
        mask = (seg_map == sid).astype(np.uint8) * 255
        d = args.out / f"{idx:03d}_{T.COCO_CLASSES[idx].name}"
        d.mkdir(parents=True, exist_ok=True)
        Image.open(CE.VAL_IMG_DIR / f"{img_id:012d}.jpg").convert("RGB").save(d / "0001.png")
        Image.fromarray(mask).save(d / "0001_mask.png")
    logger.info("wrote %d/%d stuff exemplars to %s",
                len(best), len(T.STUFF_IDXS), args.out)


if __name__ == "__main__":
    main()
