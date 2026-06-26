#!/usr/bin/env python3
"""Improve Indian-domain instances: cheap tweaks + INSID3 recall, BEFORE vs AFTER.

BEFORE = current best (base prompts, TILES=2, thr 0.50).
AFTER  = synonym-rich prompts (scooter/three-wheeler/pedestrian...) + TILES=3 + thr 0.40
         + INSID3 in-context recall for auto-rickshaw & cart (bootstrap exemplar from the
         highest-confidence SAM3 detection across frames, propagate to recover misses).
Stuff (ensemble + CRF) is identical in both, so the diff isolates the instance gains.

Run: PYTHONPATH=<repo> .venv_cups_cpu/bin/python auto_annotation/scripts/run_improved.py
"""

import logging
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "INSID3"))
sys.path.insert(0, str(ROOT))
from auto_annotation import demo_panoptic as D  # noqa: E402
from auto_annotation import demo_viz as V  # noqa: E402
from auto_annotation import quality_ablation as Q  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("improve")

FOLDER = "/Volumes/code_files_2/self_made_dataset/youtube_videos/frames"
N = 3
BASE_PROMPTS = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "rider",
                "auto rickshaw", "cow", "dog", "cart"]
# synonym prompts that catch GENUINELY DIFFERENT objects (trimmed for CPU cost) -> idx
SYN = {"car": "car", "truck": "truck", "bus": "bus", "motorcycle": "motorcycle",
       "scooter": "motorcycle", "bicycle": "bicycle", "person": "person",
       "pedestrian": "person", "rider": "rider", "auto rickshaw": "auto rickshaw",
       "three wheeler": "auto rickshaw", "cow": "cow", "dog": "dog", "cart": "cart"}
RICH_PROMPTS = list(SYN.keys())
P2IDX = {p: D.NAME2IDX[c] for p, c in SYN.items()}


def load_insid3(device="cpu"):
    from models.insid3 import INSID3
    from auto_annotation.backends.dinov3_hf import DinoV3HFEncoder
    enc = DinoV3HFEncoder(model_size="small", device=device)
    m = INSID3(encoder=enc, image_size=768, svd_components=500, tau=0.6,
               merge_threshold=0.2, mask_refiner="bilinear", resize_to_orig_size=True,
               device=device)
    for p in m.parameters():
        p.requires_grad = False
    return m


def bootstrap_exemplar(sam3, frames, concept):
    """Return (PIL image, PIL mask) of the highest-confidence SAM3 detection."""
    best = None
    for pil in frames:
        for d in sam3.predict(pil, [concept]):
            if best is None or d["score"] > best[0]:
                best = (d["score"], pil, d["mask"])
    if best is None:
        return None
    _, pil, mask = best
    return pil, Image.fromarray((mask * 255).astype(np.uint8))


def insid3_recall(insid3, exemplar, pil, existing, class_idx, max_frac=0.12):
    """Propagate a concept exemplar; add fresh connected components as instances."""
    if exemplar is None:
        return []
    insid3.set_reference(exemplar[0], exemplar[1])
    insid3.set_target(pil)
    region = insid3.segment().cpu().numpy().astype(bool)
    W, H = pil.size
    if region.shape != (H, W):
        region = np.array(Image.fromarray(region).resize((W, H), Image.NEAREST))
    add = []
    n, lab = cv2.connectedComponents(region.astype(np.uint8))
    for k in range(1, n):
        comp = lab == k
        a = comp.sum()
        if a < 800 or a > max_frac * H * W:
            continue
        if all(Q._mask_iou(comp, e["mask"]) < 0.3 for e in existing):
            add.append({"mask": comp, "class_idx": class_idx, "score": 0.5})
    return add


def main():
    cfg = D.DemoConfig(model_size="small")
    allimg = sorted(Path(FOLDER).glob("*.jpg"))
    idxs = np.linspace(0, len(allimg) - 1, N).astype(int)
    targets = [allimg[i] for i in idxs]
    pils = [Image.open(t).convert("RGB") for t in targets]

    m2f_map = D.Mask2FormerStuffRunner(cfg)
    m2f_cs = Q.M2FCityscapesRunner()
    eomt = Q.EoMTCityscapesRunner()
    insid3 = load_insid3(cfg.device)
    sam3 = D.Sam3Runner(cfg)                  # LAST

    # bootstrap exemplars from the frames themselves (need a small pool incl. busy frames)
    pool = [Image.open(p).convert("RGB") for p in allimg[:6]]
    ex_auto = bootstrap_exemplar(sam3, pool, "auto rickshaw")
    ex_cart = bootstrap_exemplar(sam3, pool, "cart")
    log.info("exemplars: auto=%s cart=%s", ex_auto is not None, ex_cart is not None)

    rows = []
    for t, pil in zip(targets, pils):
        rgb = np.array(pil); t0 = time.time()
        sem = Q.crf_refine(rgb, Q.ensemble_stuff(
            [m2f_map.predict(pil), m2f_cs.predict(pil), eomt.predict(pil)]))
        # BEFORE
        before = Q.sam3_tiled(sam3, pil, BASE_PROMPTS, 0.50, tiles=2)
        # AFTER: synonym-rich prompts + lower thr (TILES=2 kept for CPU tractability)
        after = Q.sam3_tiled(sam3, pil, RICH_PROMPTS, 0.40, tiles=2, name_to_idx=P2IDX)
        # AFTER + INSID3 recall (auto rickshaw, cart)
        rec = insid3_recall(insid3, ex_auto, pil, after, D.NAME2IDX["auto rickshaw"]) \
            + insid3_recall(insid3, ex_cart, pil, after, D.NAME2IDX["cart"])
        after_r = after + rec
        pan_b, _ = D.merge_panoptic(sem, before, cfg)
        pan_a, _ = D.merge_panoptic(sem, after_r, cfg)
        cb = Counter(D.IDX2NAME[i["class_idx"]] for i in before)
        ca = Counter(D.IDX2NAME[i["class_idx"]] for i in after_r)
        log.info("%s | BEFORE %d %s | AFTER %d (+%d insid3) %s | %.0fs",
                 t.name[:22], len(before), dict(cb), len(after_r), len(rec), dict(ca),
                 time.time() - t0)
        rows.append(dict(name=t.name, rgb=rgb, before=before, after=after_r,
                        pan_b=pan_b, pan_a=pan_a, n_rec=len(rec)))

    _grid(rows, cfg)


def _grid(rows, cfg):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = ROOT / "auto_annotation/outputs/youtube_jaipur/improved_before_after.png"
    n = len(rows)
    fig, ax = plt.subplots(n, 5, figsize=(28, 5.2 * n))
    if n == 1:
        ax = ax[None, :]
    for r, row in zip(rows, ax):
        sh = r["rgb"].shape[:2]
        cols = [r["rgb"],
                V.overlay(r["rgb"], V.colorize_instances(r["before"], sh), 0.6),
                V.overlay(r["rgb"], V.colorize_instances(r["after"], sh), 0.6),
                V.overlay(r["rgb"], V.colorize_panoptic(r["pan_b"], cfg.label_divisor)),
                V.overlay(r["rgb"], V.colorize_panoptic(r["pan_a"], cfg.label_divisor))]
        tit = [r["name"][:22], f"BEFORE inst ({len(r['before'])})",
               f"AFTER inst ({len(r['after'])}, +{r['n_rec']} INSID3)",
               "BEFORE panoptic", "AFTER panoptic"]
        for a, im, ti in zip(row, cols, tit):
            a.imshow(im); a.axis("off"); a.set_title(ti, fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=78, bbox_inches="tight"); plt.close(fig)
    print("grid ->", out, flush=True)


if __name__ == "__main__":
    main()
