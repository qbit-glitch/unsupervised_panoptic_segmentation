#!/usr/bin/env python3
"""Track B: depth + cross-cue AGREEMENT as the high-ROI auxiliary signal.

For each Cityscapes image: panoptic (Mask2Former stuff + SAM3 things) + monocular
depth (Depth-Anything-V2) + an EDGE-AGREEMENT QA map:
  green  = panoptic boundary supported by a depth discontinuity  (confident)
  yellow = panoptic boundary with NO depth support               (possibly spurious)
  red    = depth discontinuity with NO panoptic boundary         (MISSED split -> review;
           e.g. two adjacent cars/people merged into one segment)
Red is the actionable signal: it localizes exactly where the auto-labels likely err.
Load order: Mask2Former + depth (transformers) BEFORE SAM3 (shim gotcha).

Run: PYTHONPATH=<repo> .venv_cups_cpu/bin/python auto_annotation/scripts/run_aux_signals.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from auto_annotation import demo_panoptic as D  # noqa: E402
from auto_annotation import demo_viz as V  # noqa: E402

N = 4


def depth_map(pipe, pil):
    out = pipe(pil)["depth"]
    d = np.array(out, dtype=np.float32)
    return (d - d.min()) / (np.ptp(d) + 1e-6)   # 0..1  (np.ptp: ndarray.ptp gone in numpy 2)


def edges(label_map, k=3):
    """Boundary pixels: where a max-filtered label differs from the original."""
    import cv2
    lm = label_map.astype(np.int32)
    mx = cv2.dilate(lm.astype(np.float32), np.ones((k, k), np.uint8))
    mn = -cv2.dilate(-lm.astype(np.float32), np.ones((k, k), np.uint8))
    return (mx != mn)


def depth_edges(depth, thr=0.06):
    import cv2
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return mag > thr


def agreement_rgb(pan_edge, dep_edge, shape):
    import cv2
    de = cv2.dilate(dep_edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    pe = cv2.dilate(pan_edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out = np.zeros((*shape, 3), np.uint8)
    out[pe & de] = (0, 220, 0)        # confident boundary
    out[pe & ~de] = (230, 230, 0)     # boundary w/o depth support
    out[de & ~pe] = (255, 0, 0)       # MISSED split -> review
    return out


def main():
    from transformers import pipeline
    cfg = D.DemoConfig(model_size="small")
    val = cfg.cityscapes_root / "leftImg8bit/val/frankfurt"
    targets = sorted(val.glob("*_leftImg8bit.png"))[:N]

    m2f = D.Mask2FormerStuffRunner(cfg)                                  # 1st
    depth = pipeline("depth-estimation",                                # 2nd (transformers)
                     model="depth-anything/Depth-Anything-V2-Small-hf", device="cpu")
    sam3 = D.Sam3Runner(cfg)                                            # LAST

    rows = []
    for t in targets:
        pil = Image.open(t).convert("RGB"); rgb = np.array(pil)
        t0 = time.time()
        sem = m2f.predict(pil)
        insts = D.run_instances(sam3.predict(pil, cfg.thing_prompts))
        pan, _ = D.merge_panoptic(sem, insts, cfg)
        dep = depth_map(depth, pil)
        agree = agreement_rgb(edges(pan), depth_edges(dep), rgb.shape[:2])
        missed = float((agree == (255, 0, 0)).all(-1).mean())
        print(f"{t.name[:26]} | missed-split px={100*missed:.1f}% | {time.time()-t0:.0f}s",
              flush=True)
        rows.append(dict(name=t.name, rgb=rgb, pan=pan, dep=dep, agree=agree))

    _grid(rows, cfg, ROOT / "auto_annotation/outputs/panoptic_demo/aux_signals.png")
    print("done", flush=True)


def _grid(rows, cfg, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(22, 5.2 * n))
    if n == 1:
        axes = axes[None, :]
    titles = ["original", "depth (Depth-Anything-V2)", "panoptic (M2F+SAM3)",
              "depth-agreement QA  (green=ok yellow=weak RED=missed-split)"]
    for r, row in zip(rows, axes):
        import matplotlib.cm as cm
        dcol = (cm.magma(r["dep"])[:, :, :3] * 255).astype(np.uint8)
        cols = [r["rgb"], dcol, V.overlay(r["rgb"], V.colorize_panoptic(r["pan"], cfg.label_divisor)),
                V.overlay(r["rgb"], r["agree"], 0.7)]
        for ax, im, ti in zip(row, cols, titles):
            ax.imshow(im); ax.axis("off"); ax.set_title(ti, fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=80, bbox_inches="tight"); plt.close(fig)
    print("grid ->", out, flush=True)


if __name__ == "__main__":
    main()
