#!/usr/bin/env python3
"""Auto-label a folder of frames -> trainable panoptic labels (resumable).

Best stack: ensemble stuff {M2F-Mapillary, M2F-Cityscapes, EoMT} + SAM3 things + CRF.
Per frame writes to <out>/:  {stem}_pan.png  (uint16, id = class_idx*1000 + inst)
                             {stem}_seg.json (segments_info: id, category, isthing, area)
Skips frames already labelled, so it can be killed/resumed or moved to a GPU box.

ponytail: SAM3 tiling dropped here (2x slower for ~+36% instances) — fine for bulk
          pseudo-labels feeding self-training; add --tiles 2 if you want it.

Run: PYTHONPATH=<repo> .venv_cups_cpu/bin/python auto_annotation/scripts/autolabel_folder.py \
        --frames <dir> --out <dir> [--tiles 2] [--limit N]
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from auto_annotation import demo_panoptic as D       # noqa: E402
from auto_annotation import quality_ablation as Q     # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("autolabel")

PROMPTS = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "rider",
           "auto rickshaw", "cow", "dog", "cart"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiles", type=int, default=2, help="SAM3 tiling (0=off, 2=on)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--shard", default="0/1", help="k/n split: this worker takes every n-th frame at offset k")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = D.DemoConfig(model_size="small", device=args.device)
    k, n = (int(x) for x in args.shard.split("/"))

    from auto_annotation import aux_signals as AX
    import zipfile
    zf = None                                            # --frames may be a folder OR a .zip
    if args.frames.endswith(".zip"):                     # read images in-place, no 22GB extract
        zf = zipfile.ZipFile(args.frames)
        items = [(nm, Path(nm).stem) for nm in zf.namelist()
                 if nm.lower().endswith((".jpg", ".png")) and not nm.endswith("/")]
        items.sort()
    else:
        fs = sorted(Path(args.frames).glob("*.jpg")) + sorted(Path(args.frames).glob("*.png"))
        items = [(str(f), f.stem) for f in fs]
    items = [it for j, it in enumerate(items) if j % n == k]          # this GPU's shard
    todo = [it for it in items if not (out / f"{it[1]}_pan.png").exists()]
    if args.limit:
        todo = todo[:args.limit]
    log.info("shard %d/%d: %d frames, %d to label (device=%s, tiles=%d)", k, n,
             len(items), len(todo), args.device, args.tiles)
    if not todo:
        return

    m2f_map = D.Mask2FormerStuffRunner(cfg)            # not gated -> downloads online
    m2f_cs = Q.M2FCityscapesRunner(device=args.device)
    eomt = Q.EoMTCityscapesRunner(device=args.device)
    depth = AX.DepthRunner(device=args.device)
    import os as _os                                   # SAM3 gated: SAM3_OFFLINE=1 uses local
    if _os.environ.get("SAM3_OFFLINE"):                #   cache (no token); else download w/ token
        _os.environ["HF_HUB_OFFLINE"] = "1"
    sam3 = D.Sam3Runner(cfg)                           # LAST

    for i, (src, stem) in enumerate(todo):
        t0 = time.time()
        pil = Image.open(zf.open(src) if zf else src).convert("RGB"); rgb = np.array(pil)
        sem = Q.crf_refine(rgb, Q.ensemble_stuff(
            [m2f_map.predict(pil), m2f_cs.predict(pil), eomt.predict(pil)]))
        insts = (Q.sam3_tiled(sam3, pil, PROMPTS, cfg.instance_score_thr, tiles=2)
                 if args.tiles >= 2 else D.run_instances(sam3.predict(pil, PROMPTS)))
        pan, segs = D.merge_panoptic(sem, insts, cfg)   # id = class*1000+inst (<23000)
        dep = depth.predict(pil)                          # depth-agreement QA -> review flag
        agree = AX.agreement_rgb(AX.edges(pan), AX.depth_edges(dep), rgb.shape[:2])
        missed = float((agree == (255, 0, 0)).all(-1).mean())
        pan16 = np.where(pan // 1000 == D.VOID, 65535, pan).astype(np.uint16)  # void->sentinel
        Image.fromarray(pan16, mode="I;16").save(out / f"{stem}_pan.png")
        (out / f"{stem}_seg.json").write_text(json.dumps(
            {"segments": segs, "missed_split_frac": round(missed, 4)}))
        log.info("[%d/%d] %s | %d segs | missed %.1f%% | %.1fs", i + 1, len(todo),
                 stem[:26], len(segs), 100 * missed, time.time() - t0)
    log.info("done -> %s", out)


if __name__ == "__main__":
    main()


def _selfcheck():  # ponytail: uint16 round-trips panoptic id + void sentinel
    import tempfile
    pan = np.array([[17 * 1000 + 3, 65535]], np.uint16)  # car inst3, void-sentinel
    p = Path(tempfile.mktemp(suffix=".png"))
    Image.fromarray(pan, mode="I;16").save(p)
    back = np.array(Image.open(p)).astype(np.uint16)
    assert (back == pan).all(), back
    assert back[0, 0] // 1000 == 17 and back[0, 0] % 1000 == 3 and back[0, 1] == 65535
