#!/usr/bin/env python3
"""Production label-free COCO auto-labeler: SAM3 things + INSID3 stuff -> panoptic.

Combines the two CPU-verified label-free sources into a full things+stuff COCO
panoptic, written in COCO format (rgb2id PNG + json) so EoMT's loader and the PQ
evaluator consume it unchanged. Run in .venv_cups_cpu.

  # bounded CPU smoke (1 image, 15 things, 8 stuff):
  HF_HUB_OFFLINE=1 SAM3_OFFLINE=1 PYTHONPATH=. .venv_cups_cpu/bin/python \\
    auto_annotation/scripts/autolabel_coco_full.py --limit 1 --n_things 15 --n_stuff 8
  # full GPU run: --device cuda --n_things 80 --n_stuff 53 --limit 0 (all)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external/INSID3"))     # `models.insid3` first
import importlib
_INSID3 = importlib.import_module("models.insid3").INSID3
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("autolabel_coco_full")

from auto_annotation import taxonomy_coco as T                      # noqa: E402
from auto_annotation.backends.dinov3_hf import DinoV3HFEncoder      # noqa: E402
from mbps_pytorch.mobile_panoptic_sup.coco_eval import VAL_IMG_DIR  # noqa: E402
from mbps_pytorch.mobile_panoptic_sup.to_coco_panoptic import pan_to_coco  # noqa: E402

DIV = 1000
SCORE_THR = 0.5
_THING_NAMES = [T.COCO_CLASSES[i].name for i in sorted(T.THING_IDXS)]


def _load_sam3(device: str):
    import torch  # noqa: F401
    from auto_annotation.backends.sam3_compat import enable_cpu_sam3
    if device != "cuda":
        enable_cpu_sam3()
    sys.path.insert(0, str(ROOT / "external/sam3"))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(device=device)
    return Sam3Processor(model, resolution=1008, device=device)


def _load_insid3(device: str):
    enc = DinoV3HFEncoder(model_size="small", device=device)
    return _INSID3(encoder=enc, image_size=768, svd_components=500, tau=0.6,
                   merge_threshold=0.2, mask_refiner="bilinear",
                   resize_to_orig_size=True, device=device)


def _discover_stuff_exemplars(root: Path, n: int) -> list[tuple[int, list[Path]]]:
    out = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        idx = int(sub.name[:3])
        imgs = sorted(p for p in sub.glob("*.png") if "_mask" not in p.stem)
        if imgs:
            out.append((idx, imgs))
    return out[:n] if n else out


def _sam3_things(proc, state, prompts) -> list[tuple[float, int, np.ndarray]]:
    dets = []
    for name in prompts:
        out = proc.set_text_prompt(state=state, prompt=name)
        masks, scores = out.get("masks"), out.get("scores")
        if masks is None:
            continue
        m = masks.detach().cpu().numpy()
        if m.ndim == 4:
            m = m[:, 0]
        for i in range(m.shape[0]):
            sc = float(scores[i]) if scores is not None else 1.0
            if sc >= SCORE_THR:
                dets.append((sc, T.name_to_idx(name), m[i] > 0.5))
    return dets


def _insid3_stuff(model, exemplars, pil_img, hw) -> dict[int, np.ndarray]:
    h, w = hw
    out = {}
    for idx, paths in exemplars:
        model.reset_state()
        for p in paths:
            mp = p.with_name(f"{p.stem}_mask.png")
            if mp.exists():
                model.set_reference(Image.open(p).convert("RGB"), Image.open(mp))
        if getattr(model, "_ref_images", None) is None:
            continue
        model.set_target(pil_img)
        pred = model.segment().cpu().numpy().astype(bool)
        if pred.shape != (h, w):
            pred = np.array(Image.fromarray(pred).resize((w, h), Image.NEAREST))
        if pred.any():
            out[idx] = pred
    return out


def _merge(things, stuff, hw) -> np.ndarray:
    h, w = hw
    pan = np.full((h, w), T.VOID_IDX * DIV, dtype=np.int32)
    for idx, mask in stuff.items():           # stuff base, inst 0
        pan[mask] = idx * DIV
    counts: dict[int, int] = {}
    for _, cls, mask in sorted(things, key=lambda d: d[0]):   # things on top, best wins
        counts[cls] = counts.get(cls, 0) + 1
        pan[mask] = cls * DIV + counts[cls]
    return pan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--n_things", type=int, default=15)
    ap.add_argument("--n_stuff", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=ROOT / "auto_annotation/outputs/coco_full")
    ap.add_argument("--exemplars", type=Path,
                    default=ROOT / "auto_annotation/data/exemplars_coco")
    ap.add_argument("--img_dir", type=Path, default=VAL_IMG_DIR,
                    help="image folder to label (point at train2017 for Phase 1)")
    ap.add_argument("--shard", default="0/1", help="k/n: label only shard k of n")
    args = ap.parse_args()
    k, n = (int(v) for v in args.shard.split("/"))
    args.out.mkdir(parents=True, exist_ok=True)

    proc = _load_sam3(args.device)
    insid3 = _load_insid3(args.device)
    exemplars = _discover_stuff_exemplars(args.exemplars, args.n_stuff)
    prompts = _THING_NAMES[: args.n_things] if args.n_things else _THING_NAMES
    logger.info("things=%d stuff=%d device=%s", len(prompts), len(exemplars), args.device)

    jpgs = sorted(args.img_dir.glob("*.jpg"))[k::n]   # shard for parallel GPU workers
    jpgs = jpgs[: args.limit] if args.limit else jpgs
    for jpg in jpgs:
        if (args.out / f"{jpg.stem}_panoptic.npy").exists():
            continue
        pil = Image.open(jpg).convert("RGB")
        arr = np.asarray(pil)
        hw = arr.shape[:2]
        things = _sam3_things(proc, proc.set_image(pil), prompts)
        stuff = _insid3_stuff(insid3, exemplars, pil, hw)
        pan = _merge(things, stuff, hw)
        rgb, ann = pan_to_coco(pan, jpg.stem)
        np.save(args.out / f"{jpg.stem}_panoptic.npy", pan)
        Image.fromarray(rgb).save(args.out / f"{jpg.stem}.png")
        (args.out / f"{jpg.stem}.json").write_text(json.dumps(ann))
        logger.info("%s: %d things, %d stuff, %d segments",
                    jpg.stem, len(things), len(stuff), len(ann["segments_info"]))
    logger.info("done -> %s", args.out)


if __name__ == "__main__":
    main()
