#!/usr/bin/env python3
"""Auto-label N COCO val images on CPU: SAM3 things -> panoptic (LABEL-FREE arm).

Run in ``.venv_cups_cpu``. Phase-0 plumbing only: things via SAM3 concept prompts;
INSID3 stuff is deferred to Phase 1. Proves the label source runs end-to-end and
emits a sane ``class*1000+inst`` panoptic map per image.

Usage:
    HF_HUB_OFFLINE=1 SAM3_OFFLINE=1 PYTHONPATH=. \\
      .venv_cups_cpu/bin/python auto_annotation/scripts/autolabel_coco_slice.py --limit 5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("autolabel_coco_slice")

from auto_annotation import taxonomy_coco as T  # noqa: E402

COCO_VAL = Path("/Volumes/code_files/datasets/coco/val2017")
THING_PROMPTS = ["person", "car", "truck", "bus", "bicycle", "motorcycle", "dog", "cat"]
SCORE_THR = 0.5
DIV = 1000


def _load_sam3():
    import torch  # noqa: F401  (import triggers the cpu shim's torch patches in order)
    from auto_annotation.backends.sam3_compat import enable_cpu_sam3

    enable_cpu_sam3()  # MUST precede `import sam3`
    sys.path.insert(0, str(ROOT / "external/sam3"))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(device="cpu")
    return Sam3Processor(model, resolution=1008, device="cpu")


def _masks_for(proc, state, prompt: str) -> list[tuple[float, np.ndarray]]:
    out = proc.set_text_prompt(state=state, prompt=prompt)
    masks, scores = out.get("masks"), out.get("scores")
    if masks is None:
        return []
    m = masks.detach().cpu().numpy()
    if m.ndim == 4:
        m = m[:, 0]
    res = []
    for i in range(m.shape[0]):
        sc = float(scores[i]) if scores is not None else 1.0
        if sc >= SCORE_THR:
            res.append((sc, m[i] > 0.5))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--out", type=Path, default=ROOT / "auto_annotation/outputs/coco_slice")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    proc = _load_sam3()

    for jpg in sorted(COCO_VAL.glob("*.jpg"))[: args.limit]:
        img = np.asarray(Image.open(jpg).convert("RGB"))
        h, w = img.shape[:2]
        pan = np.full((h, w), T.VOID_IDX * DIV, dtype=np.int32)
        state = proc.set_image(Image.fromarray(img))
        dets: list[tuple[float, int, np.ndarray]] = []
        for prompt in THING_PROMPTS:
            cls = T.name_to_idx(prompt)
            for sc, mask in _masks_for(proc, state, prompt):
                dets.append((sc, cls, mask))
        counts: dict[int, int] = {}
        for _, cls, mask in sorted(dets, key=lambda d: d[0]):  # ascending score: best wins
            counts[cls] = counts.get(cls, 0) + 1
            pan[mask] = cls * DIV + counts[cls]
        np.save(args.out / f"{jpg.stem}_panoptic.npy", pan)
        logger.info("%s: %d dets, %d classes", jpg.stem, len(dets), len(counts))
    logger.info("done -> %s", args.out)


if __name__ == "__main__":
    main()
