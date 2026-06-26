#!/usr/bin/env python3
"""Cumulative quality ablation on 4 Cityscapes images, visualized as a progression:
   original | baseline | +ensemble-stuff | +tiling | +CRF-refine.

Levers: ensemble {M2F-Mapillary, M2F-Cityscapes, EoMT-Cityscapes} stuff (#3,#4) +
SAM3-tiled things (#1) + dense-CRF boundaries (#2). Build all transformers models
BEFORE SAM3 (shim gotcha).

Run: PYTHONPATH=<repo> .venv_cups_cpu/bin/python auto_annotation/scripts/run_quality_ablation.py
"""

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
from auto_annotation import quality_ablation as Q  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("ablate")
N = 4


def main():
    cfg = D.DemoConfig(model_size="small")
    val = cfg.cityscapes_root / "leftImg8bit/val/frankfurt"
    targets = sorted(val.glob("*_leftImg8bit.png"))[:N]

    m2f_map = D.Mask2FormerStuffRunner(cfg)        # baseline stuff
    m2f_cs = Q.M2FCityscapesRunner()               # domain-matched
    eomt = Q.EoMTCityscapesRunner()                # modern SOTA (#4)
    sam3 = D.Sam3Runner(cfg)                        # LAST

    rows = []
    for t in targets:
        pil = Image.open(t).convert("RGB"); rgb = np.array(pil)
        t0 = time.time()
        sem_map = m2f_map.predict(pil)
        sem_cs = m2f_cs.predict(pil)
        sem_eo = eomt.predict(pil)
        sem_ens = Q.ensemble_stuff([sem_map, sem_cs, sem_eo])
        sem_crf = Q.crf_refine(rgb, sem_ens)

        insts_full = D.run_instances(sam3.predict(pil, cfg.thing_prompts))
        insts_tile = Q.sam3_tiled(sam3, pil, cfg.thing_prompts, cfg.instance_score_thr)

        pan_base, _ = D.merge_panoptic(sem_map, insts_full, cfg)
        pan_ens, _ = D.merge_panoptic(sem_ens, insts_full, cfg)
        pan_tile, _ = D.merge_panoptic(sem_ens, insts_tile, cfg)
        pan_crf, _ = D.merge_panoptic(sem_crf, insts_tile, cfg)

        mi_base = D.semantic_miou(sem_map, cfg, t)
        mi_ens = D.semantic_miou(sem_ens, cfg, t)
        mi_crf = D.semantic_miou(sem_crf, cfg, t)
        log.info("%s | stuff-mIoU base=%.3f ens=%.3f crf=%.3f | inst %d->%d (tiled) | %.0fs",
                 t.name[:24], mi_base, mi_ens, mi_crf, len(insts_full), len(insts_tile),
                 time.time() - t0)
        rows.append(dict(name=t.name, rgb=rgb, pan_base=pan_base, pan_ens=pan_ens,
                        pan_tile=pan_tile, pan_crf=pan_crf,
                        mi_base=mi_base, mi_ens=mi_ens, mi_crf=mi_crf,
                        n_full=len(insts_full), n_tile=len(insts_tile)))

    _grid(rows, cfg, ROOT / "auto_annotation/outputs/panoptic_demo/quality_ablation.png")
    log.info("MEAN stuff-mIoU base=%.3f ens=%.3f crf=%.3f | inst %.1f->%.1f",
             np.mean([r["mi_base"] for r in rows]), np.mean([r["mi_ens"] for r in rows]),
             np.mean([r["mi_crf"] for r in rows]),
             np.mean([r["n_full"] for r in rows]), np.mean([r["n_tile"] for r in rows]))


def _grid(rows, cfg, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(rows)
    fig, axes = plt.subplots(n, 5, figsize=(27, 5.2 * n))
    if n == 1:
        axes = axes[None, :]
    def pc(p): return V.overlay(rows[0]["rgb"] * 0 + 0, p, 1.0)  # unused placeholder
    for r, row in zip(rows, axes):
        cols = [r["rgb"],
                V.overlay(r["rgb"], V.colorize_panoptic(r["pan_base"], cfg.label_divisor)),
                V.overlay(r["rgb"], V.colorize_panoptic(r["pan_ens"], cfg.label_divisor)),
                V.overlay(r["rgb"], V.colorize_panoptic(r["pan_tile"], cfg.label_divisor)),
                V.overlay(r["rgb"], V.colorize_panoptic(r["pan_crf"], cfg.label_divisor))]
        titles = [r["name"][:22],
                  f"baseline M2F-Map\nstuff-mIoU {r['mi_base']:.2f}, inst {r['n_full']}",
                  f"+ensemble stuff\nmIoU {r['mi_ens']:.2f}",
                  f"+tiling\ninst {r['n_full']}->{r['n_tile']}",
                  f"+CRF refine\nmIoU {r['mi_crf']:.2f}"]
        for ax, im, ti in zip(row, cols, titles):
            ax.imshow(im); ax.axis("off"); ax.set_title(ti, fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=80, bbox_inches="tight"); plt.close(fig)
    print("grid ->", out, flush=True)


if __name__ == "__main__":
    main()
