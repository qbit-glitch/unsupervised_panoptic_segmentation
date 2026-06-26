#!/usr/bin/env python3
"""Run the full INSID3+SAM3 panoptic demo on N Cityscapes images and save a grid.

Verifies the pipeline end-to-end and renders [original | semantic | instances |
panoptic] for each image, plus semantic mIoU vs gtFine. Used to validate quality
before/independently of the notebook.

Run: PYTHONPATH=. .venv_cups_cpu/bin/python auto_annotation/scripts/run_panoptic_demo.py \
        --n 4 --model-size small
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("demo")


def semantic_miou(sem: np.ndarray, cfg, img_path) -> float:
    gt = D.gt_labelids(cfg, img_path)
    per = []
    for name in cfg.stuff_classes:
        idx = D.NAME2IDX[name]
        p, g = (sem == idx), (gt == D.IDX2LABELID[idx])
        u = (p | g).sum()
        if u:
            per.append((p & g).sum() / u)
    return float(np.mean(per)) if per else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--model-size", default="small", choices=["small", "base", "large"])
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--out", type=Path, default=ROOT / "auto_annotation/outputs/panoptic_demo")
    args = ap.parse_args()

    cfg = D.DemoConfig(model_size=args.model_size, image_size=args.image_size)
    val = cfg.cityscapes_root / "leftImg8bit" / cfg.split_city
    imgs = sorted(val.glob("*_leftImg8bit.png"))
    targets, pool = imgs[:args.n], imgs[args.n:args.n + 10]
    args.out.mkdir(parents=True, exist_ok=True)
    log.info("targets=%d  pool=%d  model=%s", len(targets), len(pool), args.model_size)

    insid3 = D.load_insid3(cfg)
    sam3 = D.Sam3Runner(cfg)
    refs = D.build_stuff_refs(cfg, pool)

    results = []
    for ti, tgt in enumerate(targets):
        t0 = time.time()
        pil = Image.open(tgt).convert("RGB")
        rgb = np.array(pil)
        sem_insid3 = D.run_semantic(insid3, refs, pil, cfg)
        sem_sam3 = D.run_semantic_sam3(sam3, pil, cfg)
        dets = sam3.predict(pil, cfg.thing_prompts)
        insts = D.run_instances(dets)
        # panoptic built from the crisper SAM3 stuff + SAM3 things
        pan, segs = D.merge_panoptic(sem_sam3, insts, cfg)
        miou_i = semantic_miou(sem_insid3, cfg, tgt)
        miou_s = semantic_miou(sem_sam3, cfg, tgt)
        n_things = sum(s["isthing"] for s in segs)
        log.info("[%d/%d] %s | mIoU INSID3=%.3f SAM3=%.3f | inst=%d things=%d | %.0fs",
                 ti + 1, len(targets), tgt.name[:28], miou_i, miou_s, len(insts),
                 n_things, time.time() - t0)
        results.append(dict(name=tgt.name, rgb=rgb, sem_insid3=sem_insid3,
                            sem_sam3=sem_sam3, insts=insts, pan=pan, segs=segs,
                            miou_i=miou_i, miou_s=miou_s))

    _save_grid(results, cfg, args.out / f"grid_{args.model_size}.png")
    log.info("mean mIoU: INSID3=%.3f  SAM3=%.3f | grid -> %s",
             float(np.mean([r["miou_i"] for r in results])),
             float(np.mean([r["miou_s"] for r in results])), args.out)


def _save_grid(results, cfg, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(n, 5, figsize=(27, 5.2 * n))
    if n == 1:
        axes = axes[None, :]
    cols = ["original", "semantic: INSID3", "semantic: SAM3", "instances: SAM3",
            "panoptic (SAM3 stuff+things)"]
    present = []
    for r, row in zip(results, axes):
        semi_c = V.colorize_semantic(r["sem_insid3"])
        sems_c = V.colorize_semantic(r["sem_sam3"])
        inst_c = V.colorize_instances(r["insts"], r["rgb"].shape[:2])
        pan_c = V.colorize_panoptic(r["pan"], cfg.label_divisor)
        panels = [r["rgb"], V.overlay(r["rgb"], semi_c), V.overlay(r["rgb"], sems_c),
                  V.overlay(r["rgb"], inst_c, 0.6), V.overlay(r["rgb"], pan_c)]
        titles = [cols[0], f"{cols[1]} (mIoU {r['miou_i']:.2f})",
                  f"{cols[2]} (mIoU {r['miou_s']:.2f})", cols[3], cols[4]]
        for ax, im, title in zip(row, panels, titles):
            ax.imshow(im); ax.axis("off"); ax.set_title(title, fontsize=11)
        present += list(np.unique(r["sem_sam3"])) + [s["class_idx"] for s in r["segs"]]
    fig.legend(handles=V.legend_handles(present), loc="lower center", ncol=10,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=85, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
