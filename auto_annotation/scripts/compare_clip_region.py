#!/usr/bin/env python3
"""Source A: CLIP region-labeler vs Mask2Former labels on the same crisp masks.

Per image: Mask2Former gives crisp dense stuff regions; CLIP re-labels those regions
(+ SAM3 instance regions) over the open 19-class vocab. We compare stuff-mIoU and the
CLIP-vs-M2F agreement (the QA signal). Load order: M2F + CLIP before SAM3 (shim gotcha).

Run: PYTHONPATH=. .venv_cups_cpu/bin/python auto_annotation/scripts/compare_clip_region.py --n 4
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from auto_annotation import demo_panoptic as D  # noqa: E402
from auto_annotation import demo_viz as V  # noqa: E402
from auto_annotation import clip_region as CR  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("clip"); log.setLevel(logging.INFO)

VOCAB = [n for n, *_ in D._CS]   # all 19 Cityscapes class names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "auto_annotation/outputs/panoptic_demo")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = D.DemoConfig(model_size="small")
    val = cfg.cityscapes_root / "leftImg8bit" / cfg.split_city
    targets = sorted(val.glob("*_leftImg8bit.png"))[:args.n]

    m2f = D.Mask2FormerStuffRunner(cfg)               # 1st (clean torch)
    clip = CR.ClipRegionLabeler(VOCAB, device=cfg.device)  # 2nd (clean torch)
    sam3 = D.Sam3Runner(cfg)                          # LAST (applies shims)

    rows = []
    for t in targets:
        pil = Image.open(t).convert("RGB"); rgb = np.array(pil)
        t0 = time.time()
        sem_m2f = m2f.predict(pil)
        insts = D.run_instances(sam3.predict(pil, cfg.thing_prompts))
        regions = CR.regions_from_semantic(sem_m2f) + CR.regions_from_instances(insts)
        sem_clip, conf = clip.dense_map(rgb, regions)
        mi_m2f = D.semantic_miou(sem_m2f, cfg, t)
        mi_clip = D.semantic_miou(sem_clip, cfg, t)
        both = (sem_m2f != D.VOID) & (sem_clip != D.VOID)
        agree = float((sem_m2f[both] == sem_clip[both]).mean()) if both.any() else 0.0
        log.info("%s | mIoU M2F=%.2f CLIP=%.2f | CLIP-vs-M2F agree=%.0f%% | regions=%d | %.0fs",
                 t.name[:24], mi_m2f, mi_clip, 100 * agree, len(regions), time.time() - t0)
        rows.append(dict(name=t.name, rgb=rgb, sem_m2f=sem_m2f, sem_clip=sem_clip,
                        conf=conf, mi_m2f=mi_m2f, mi_clip=mi_clip, agree=agree))

    _grid(rows, args.out / "clip_region_compare.png")
    log.info("MEAN | M2F-mIoU=%.3f CLIP-mIoU=%.3f agree=%.0f%% | -> %s",
             np.mean([r["mi_m2f"] for r in rows]), np.mean([r["mi_clip"] for r in rows]),
             100 * np.mean([r["agree"] for r in rows]), args.out)


def _grid(rows, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    def gtcs(name):
        gt = D.gt_labelids(D.DemoConfig(), Path(
            "/Volumes/code_files/datasets/cityscapes/leftImg8bit/val/frankfurt") / name)
        out = np.full(gt.shape, D.VOID, np.int32)
        for idx in range(len(D.IDX2LABELID)):
            out[gt == D.IDX2LABELID[idx]] = idx
        return out

    n = len(rows)
    fig, axes = plt.subplots(n, 5, figsize=(27, 5.2 * n))
    if n == 1:
        axes = axes[None, :]
    for r, row in zip(rows, axes):
        # disagreement heat: red where CLIP != M2F (both labelled)
        dis = np.zeros((*r["sem_m2f"].shape, 3), np.uint8)
        both = (r["sem_m2f"] != D.VOID) & (r["sem_clip"] != D.VOID)
        dis[both & (r["sem_m2f"] != r["sem_clip"])] = (255, 0, 0)
        cols = [r["rgb"],
                V.overlay(r["rgb"], V.colorize_semantic(r["sem_m2f"])),
                V.overlay(r["rgb"], V.colorize_semantic(r["sem_clip"])),
                V.overlay(r["rgb"], dis, 0.6),
                V.overlay(r["rgb"], V.colorize_semantic(gtcs(r["name"])))]
        subt = [r["name"][:22], f"Mask2Former {r['mi_m2f']:.2f}",
                f"CLIP-on-masks {r['mi_clip']:.2f}",
                f"disagreement (QA) {100*r['agree']:.0f}% agree", "gtFine"]
        for ax, im, ti in zip(row, cols, subt):
            ax.imshow(im); ax.axis("off"); ax.set_title(ti, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=80, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
