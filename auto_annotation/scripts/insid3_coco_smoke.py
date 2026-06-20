#!/usr/bin/env python3
"""Verify the label-free INSID3 stuff path runs on CPU with a COCO exemplar.

Builds a 1-shot exemplar from one COCO val image's GT (largest stuff segment),
then asks INSID3 to recover that concept on the same image. Proves the
training-free DINOv3 in-context semantic path executes end-to-end on CPU.
Run in .venv_cups_cpu.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external/INSID3"))   # provides `models.insid3`
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("insid3_coco_smoke")

from models.insid3 import INSID3                                       # noqa: E402
from auto_annotation.backends.dinov3_hf import DinoV3HFEncoder         # noqa: E402
from auto_annotation import taxonomy_coco as T                        # noqa: E402
from mbps_pytorch.mobile_panoptic_sup.coco_eval import VAL_IMG_DIR, load_coco_gt  # noqa: E402


def main() -> None:
    enc = DinoV3HFEncoder(model_size="small", device="cpu")
    model = INSID3(encoder=enc, image_size=768, svd_components=500, tau=0.6,
                   merge_threshold=0.2, mask_refiner="bilinear",
                   resize_to_orig_size=True, device="cpu")

    ref_id = 139
    seg_map, seg2cat = load_coco_gt(ref_id)
    stuff = [(int((seg_map == s).sum()), s, c)
             for s, c in seg2cat.items() if c in T.STUFF_IDXS]
    assert stuff, "no stuff segment in reference image"
    area, sid, cat = max(stuff)
    logger.info("exemplar concept=%s (idx %d) area=%d",
                T.COCO_CLASSES[cat].name, cat, area)

    ref_img = Image.open(VAL_IMG_DIR / f"{ref_id:012d}.jpg").convert("RGB")
    ref_mask = Image.fromarray(((seg_map == sid) * 255).astype(np.uint8))

    model.reset_state()
    model.set_reference(ref_img, ref_mask)
    model.set_target(ref_img)
    pred = model.segment().cpu().numpy().astype(bool)
    if pred.shape != seg_map.shape:
        pred = np.array(Image.fromarray(pred).resize(
            (seg_map.shape[1], seg_map.shape[0]), Image.NEAREST))

    gt = seg_map == sid
    iou = float((pred & gt).sum()) / max(1, int((pred | gt).sum()))
    assert pred.sum() > 0, "INSID3 produced an empty mask"
    print(f"INSID3_OK concept={T.COCO_CLASSES[cat].name} "
          f"pred_area={int(pred.sum())} self_iou={iou:.3f}")


if __name__ == "__main__":
    main()
