#!/usr/bin/env python3
"""Production COCO auto-labeler: Mask2Former + INSID3 (semantics) + SAM3 (instances).

Per the design: a COCO-panoptic Mask2Former gives the DENSE common-class stuff
semantic; INSID3 (in-context, frozen DINOv3) fills remaining void / rare classes;
SAM3 gives the thing instances. Output = COCO-panoptic format (<stem>.png rgb2id +
<stem>.json segments_info) so EoMT's loader and the PQ evaluator read it unchanged.

NOTE on label-free: Mask2Former-COCO is COCO-GT-trained, so on COCO these labels are
distilled from a supervised teacher (disclosed) — genuinely label-free only on
out-of-domain data. Pass --no_m2f for the pure INSID3-only label-free arm.

  # full run (GPU):
  python autolabel_coco_full.py --img_dir /path/coco/train2017 --device cuda \
    --n_things 80 --n_stuff 53 --limit 3000 --out /path/coco/autolabels_train
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external/INSID3"))     # `models.insid3` first
_INSID3 = importlib.import_module("models.insid3").INSID3
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("autolabel_coco_full")

from auto_annotation import taxonomy_coco as T                      # noqa: E402
from auto_annotation.backends.dinov3_hf import DinoV3HFEncoder      # noqa: E402
from mbps_pytorch.mobile_panoptic_sup.coco_eval import VAL_IMG_DIR  # noqa: E402
from mbps_pytorch.mobile_panoptic_sup.to_coco_panoptic import pan_to_coco  # noqa: E402

DIV = 1000
VOID = T.VOID_IDX * DIV
SCORE_THR = 0.5
_THING_NAMES = [T.COCO_CLASSES[i].name for i in sorted(T.THING_IDXS)]


# ── model loaders ──────────────────────────────────────────────────────────
def _load_sam3(device: str):
    import torch  # noqa: F401
    from auto_annotation.backends.sam3_compat import enable_cpu_sam3
    if device != "cuda":
        enable_cpu_sam3()
    sys.path.insert(0, str(ROOT / "external/sam3"))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    return Sam3Processor(build_sam3_image_model(device=device), resolution=1008, device=device)


def _load_insid3(device: str):
    enc = DinoV3HFEncoder(model_size="small", device=device)
    return _INSID3(encoder=enc, image_size=768, svd_components=500, tau=0.6,
                   merge_threshold=0.2, mask_refiner="bilinear",
                   resize_to_orig_size=True, device=device)


def _load_m2f(device: str, repo: str):
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    proc = AutoImageProcessor.from_pretrained(repo)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(repo).eval().to(device)
    return model, proc


# ── per-image inference ────────────────────────────────────────────────────
def _discover_stuff_exemplars(root: Path, n: int) -> list[tuple[int, list[Path]]]:
    out = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = sorted(p for p in sub.glob("*.png") if "_mask" not in p.stem)
        if imgs:
            out.append((int(sub.name[:3]), imgs))
    return out[:n] if n else out


def _m2f_stuff(model, proc, pil, hw) -> dict[int, np.ndarray]:
    """Dense Mask2Former-COCO stuff masks, keyed by our contiguous class idx."""
    h, w = hw
    inputs = proc(images=pil, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_panoptic_segmentation(out, target_sizes=[(h, w)])[0]
    seg = res["segmentation"].cpu().numpy()
    id2label = model.config.id2label
    stuff: dict[int, np.ndarray] = {}
    for s in res["segments_info"]:
        name = id2label.get(s["label_id"]) or id2label.get(str(s["label_id"]))
        if name is None:
            continue
        try:
            idx = T.name_to_idx(name)
        except KeyError:
            continue
        if idx not in T.STUFF_IDXS:           # things come from SAM3
            continue
        m = seg == s["id"]
        stuff[idx] = m if idx not in stuff else (stuff[idx] | m)
    return stuff


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


def _merge(things, m2f_stuff, insid3_stuff, hw) -> np.ndarray:
    """M2F dense stuff base -> INSID3 fills remaining void -> SAM3 things on top."""
    h, w = hw
    pan = np.full((h, w), VOID, dtype=np.int32)
    for idx, mask in m2f_stuff.items():          # 1. Mask2Former dense stuff
        pan[mask] = idx * DIV
    for idx, mask in insid3_stuff.items():        # 2. INSID3 fills only what M2F left void
        fill = mask & (pan == VOID)
        pan[fill] = idx * DIV
    counts: dict[int, int] = {}
    for _, cls, mask in sorted(things, key=lambda d: d[0]):   # 3. SAM3 things on top
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
    ap.add_argument("--img_dir", type=Path, default=VAL_IMG_DIR)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--no_m2f", action="store_true", help="drop Mask2Former (pure label-free arm)")
    ap.add_argument("--m2f_repo", default="facebook/mask2former-swin-large-coco-panoptic")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    k, n = (int(v) for v in args.shard.split("/"))

    proc = _load_sam3(args.device)
    insid3 = _load_insid3(args.device)
    m2f = None if args.no_m2f else _load_m2f(args.device, args.m2f_repo)
    exemplars = _discover_stuff_exemplars(args.exemplars, args.n_stuff)
    prompts = _THING_NAMES[: args.n_things] if args.n_things else _THING_NAMES
    logger.info("things=%d stuff=%d m2f=%s device=%s",
                len(prompts), len(exemplars), m2f is not None, args.device)

    jpgs = sorted(args.img_dir.glob("*.jpg"))[k::n]
    jpgs = jpgs[: args.limit] if args.limit else jpgs
    for jpg in jpgs:
        if (args.out / f"{jpg.stem}.json").exists():
            continue
        pil = Image.open(jpg).convert("RGB")
        hw = np.asarray(pil).shape[:2]
        things = _sam3_things(proc, proc.set_image(pil), prompts)
        m2f_stuff = _m2f_stuff(m2f[0], m2f[1], pil, hw) if m2f else {}
        insid3_stuff = _insid3_stuff(insid3, exemplars, pil, hw)
        pan = _merge(things, m2f_stuff, insid3_stuff, hw)
        rgb, ann = pan_to_coco(pan, jpg.stem)
        np.save(args.out / f"{jpg.stem}_panoptic.npy", pan)
        Image.fromarray(rgb).save(args.out / f"{jpg.stem}.png")
        (args.out / f"{jpg.stem}.json").write_text(json.dumps(ann))
        logger.info("%s: %d things, %d m2f-stuff, %d insid3-fill, %d segments",
                    jpg.stem, len(things), len(m2f_stuff), len(insid3_stuff),
                    len(ann["segments_info"]))
    logger.info("done -> %s", args.out)


if __name__ == "__main__":
    main()
