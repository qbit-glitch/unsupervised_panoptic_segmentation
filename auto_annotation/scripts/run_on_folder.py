#!/usr/bin/env python3
"""Run the BEST stack on an arbitrary image folder (no GT) and save per-step viz.

Winning stack: ensemble stuff {M2F-Mapillary, M2F-Cityscapes, EoMT} + SAM3-tiled things
(Indian prompts incl. auto-rickshaw/cow) + dense-CRF + depth-agreement QA. CPU.

Run: PYTHONPATH=<repo> .venv_cups_cpu/bin/python auto_annotation/scripts/run_on_folder.py \
        --folder "/Volumes/code_files_2/.../frames" --n 6
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
from auto_annotation import quality_ablation as Q  # noqa: E402
from auto_annotation import aux_signals as AX  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("folder")

# Indian/unstructured-traffic concept prompts (all must be in D.NAME2IDX)
INDIAN_PROMPTS = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "rider",
                  "auto rickshaw", "cow", "dog", "cart"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", type=Path, default=ROOT / "auto_annotation/outputs/youtube_jaipur")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = D.DemoConfig(model_size="small")
    allimg = sorted(Path(args.folder).glob("*.jpg")) + sorted(Path(args.folder).glob("*.png"))
    # spread across the folder
    idxs = np.linspace(0, len(allimg) - 1, min(args.n, len(allimg))).astype(int)
    targets = [allimg[i] for i in idxs]
    log.info("folder has %d imgs; running %d: %s", len(allimg), len(targets),
             [t.name[:18] for t in targets])

    m2f_map = D.Mask2FormerStuffRunner(cfg)
    m2f_cs = Q.M2FCityscapesRunner()
    eomt = Q.EoMTCityscapesRunner()
    depth = AX.DepthRunner(device=cfg.device)
    sam3 = D.Sam3Runner(cfg)                       # LAST

    rows = []
    for t in targets:
        pil = Image.open(t).convert("RGB"); rgb = np.array(pil); t0 = time.time()
        sem_ens = Q.ensemble_stuff([m2f_map.predict(pil), m2f_cs.predict(pil), eomt.predict(pil)])
        sem_fin = Q.crf_refine(rgb, sem_ens)
        insts = Q.sam3_tiled(sam3, pil, INDIAN_PROMPTS, cfg.instance_score_thr, tiles=2)
        dep = depth.predict(pil)
        pan, segs = D.merge_panoptic(sem_fin, insts, cfg)
        agree = AX.agreement_rgb(AX.edges(pan), AX.depth_edges(dep), rgb.shape[:2])
        # which Indian things were found
        from collections import Counter
        cnt = Counter(D.IDX2NAME[i["class_idx"]] for i in insts)
        log.info("%s | inst=%d %s | segs=%d | %.0fs", t.name[:26], len(insts),
                 dict(cnt), len(segs), time.time() - t0)
        rows.append(dict(name=t.name, rgb=rgb, sem=sem_fin, insts=insts, pan=pan,
                        dep=dep, agree=agree, segs=segs))
        _save_one(rows[-1], cfg, args.out, len(rows) - 1)  # index prevents name collisions

    _grid(rows, cfg, args.out / "ALL_steps.png")
    log.info("done -> %s", args.out)


def _save_one(r, cfg, out, idx=0):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    panels = [r["rgb"], (cm.magma(r["dep"])[:, :, :3] * 255).astype(np.uint8),
              V.colorize_semantic(r["sem"]),
              V.overlay(r["rgb"], V.colorize_instances(r["insts"], r["rgb"].shape[:2]), 0.6),
              V.overlay(r["rgb"], V.colorize_panoptic(r["pan"], cfg.label_divisor)),
              V.overlay(r["rgb"], r["agree"], 0.7)]
    titles = ["original", "depth", "semantic (ensemble)", f"instances ({len(r['insts'])})",
              "FINAL panoptic", "depth-agreement QA"]
    fig, ax = plt.subplots(2, 3, figsize=(24, 13)); ax = ax.ravel()
    for a, im, ti in zip(ax, panels, titles):
        a.imshow(im); a.axis("off"); a.set_title(ti, fontsize=12)
    fig.suptitle(r["name"][:70], fontsize=11)
    fig.tight_layout(); fig.savefig(out / f"frame{idx:02d}_{Path(r['name']).stem[:34]}_steps.png",
                                    dpi=72, bbox_inches="tight"); plt.close(fig)


def _grid(rows, cfg, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(20, 4.6 * n))
    if n == 1:
        axes = axes[None, :]
    for r, row in zip(rows, axes):
        cols = [r["rgb"], V.colorize_semantic(r["sem"]),
                V.overlay(r["rgb"], V.colorize_panoptic(r["pan"], cfg.label_divisor))]
        for a, im, ti in zip(row, cols, [r["name"][:30], "semantic", "FINAL panoptic"]):
            a.imshow(im); a.axis("off"); a.set_title(ti, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=72, bbox_inches="tight"); plt.close(fig)
    print("grid ->", path, flush=True)


if __name__ == "__main__":
    main()
