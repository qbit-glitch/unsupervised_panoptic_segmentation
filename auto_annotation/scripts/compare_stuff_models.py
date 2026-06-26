#!/usr/bin/env python3
"""Compare STUFF semantic sources on the same Cityscapes images:
   INSID3 (in-context) vs SAM 3 (text) vs Mask2Former-Mapillary (closed-set).

Decides the stuff stage of the pipeline empirically. Saves a comparison grid +
per-model stuff-mIoU vs gtFine. Run:
   PYTHONPATH=. .venv_cups_cpu/bin/python auto_annotation/scripts/compare_stuff_models.py --n 4
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

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("cmp"); log.setLevel(logging.INFO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "auto_annotation/outputs/panoptic_demo")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = D.DemoConfig(model_size="small")
    val = cfg.cityscapes_root / "leftImg8bit" / cfg.split_city
    imgs = sorted(val.glob("*_leftImg8bit.png"))
    targets, pool = imgs[:args.n], imgs[args.n:args.n + 10]

    insid3 = D.load_insid3(cfg)
    sam3 = D.Sam3Runner(cfg)
    m2f = D.Mask2FormerStuffRunner(cfg)
    refs = D.build_stuff_refs(cfg, pool)

    rows = []
    for t in targets:
        pil = Image.open(t).convert("RGB"); rgb = np.array(pil)
        t0 = time.time()
        sem_i = D.run_semantic(insid3, refs, pil, cfg)
        sem_s = D.run_semantic_sam3(sam3, pil, cfg)
        sem_m = m2f.predict(pil)
        mi = D.semantic_miou(sem_i, cfg, t)
        ms = D.semantic_miou(sem_s, cfg, t)
        mm = D.semantic_miou(sem_m, cfg, t)
        # void fraction = coverage gap
        void_s = float((sem_s == D.VOID).mean())
        void_m = float((sem_m == D.VOID).mean())
        log.info("%s | mIoU INSID3=%.2f SAM3=%.2f(void %.0f%%) M2F=%.2f(void %.0f%%) | %.0fs",
                 t.name[:24], mi, ms, 100 * void_s, mm, 100 * void_m, time.time() - t0)
        rows.append(dict(name=t.name, rgb=rgb, sem_i=sem_i, sem_s=sem_s, sem_m=sem_m,
                        mi=mi, ms=ms, mm=mm))

    _grid(rows, args.out / "stuff_compare.png")
    log.info("MEAN stuff-mIoU | INSID3=%.3f  SAM3=%.3f  Mask2Former=%.3f | -> %s",
             np.mean([r["mi"] for r in rows]), np.mean([r["ms"] for r in rows]),
             np.mean([r["mm"] for r in rows]), args.out)


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
    titles = ["original", "INSID3 (in-context)", "SAM3 (text)",
              "Mask2Former (Mapillary)", "gtFine (truth)"]
    for r, row in zip(rows, axes):
        cols = [r["rgb"],
                V.overlay(r["rgb"], V.colorize_semantic(r["sem_i"])),
                V.overlay(r["rgb"], V.colorize_semantic(r["sem_s"])),
                V.overlay(r["rgb"], V.colorize_semantic(r["sem_m"])),
                V.overlay(r["rgb"], V.colorize_semantic(gtcs(r["name"])))]
        subt = [titles[0], f"{titles[1]} {r['mi']:.2f}", f"{titles[2]} {r['ms']:.2f}",
                f"{titles[3]} {r['mm']:.2f}", titles[4]]
        for ax, im, ti in zip(row, cols, subt):
            ax.imshow(im); ax.axis("off"); ax.set_title(ti, fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=80, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
