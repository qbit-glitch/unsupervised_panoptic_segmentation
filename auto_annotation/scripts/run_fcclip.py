#!/usr/bin/env python3
"""Run FC-CLIP (open-vocab panoptic, CLIP-based) on Cityscapes images — the proper
'CLIP signal' (vs the failed naive CLIP-on-masks). CPU via MSDeformAttn pytorch
fallback. Must run with CWD = external/fc-clip (relative data paths).

Outputs panoptic overlay + stuff-mIoU vs gtFine (COCO-panoptic classes mapped to
Cityscapes-19 by name). Run:
   PYTHONPATH=<repo> .venv_cups_cpu/bin/python auto_annotation/scripts/run_fcclip.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path("/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation")
FCCLIP = REPO / "external" / "fc-clip"
os.chdir(FCCLIP)                      # relative data paths in fc-clip need this
sys.path.insert(0, str(FCCLIP))
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.engine import DefaultPredictor  # noqa: E402
from detectron2.data import MetadataCatalog  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from fcclip import add_maskformer2_config, add_fcclip_config  # noqa: E402

from auto_annotation import demo_panoptic as D  # noqa: E402
from auto_annotation import demo_viz as V  # noqa: E402

CFG = "configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_coco.yaml"
N = 4


def coco_name_to_cs(name: str):
    n = name.lower()
    table = [
        (["road", "highway"], "road"), (["pavement", "sidewalk"], "sidewalk"),
        (["building", "house", "skyscraper", "hovel"], "building"),
        (["wall"], "wall"), (["fence", "railing", "guard rail"], "fence"),
        (["pole"], "pole"), (["traffic light"], "traffic light"),
        (["traffic sign", "street sign", "signboard", "billboard"], "traffic sign"),
        (["tree", "bush", "plant", "vegetation", "leaves"], "vegetation"),
        (["grass", "dirt", "sand", "gravel", "earth", "hill", "mountain"], "terrain"),
        (["sky"], "sky"), (["person", "man", "woman", "rider"], "person"),
        (["car", "van"], "car"), (["truck"], "truck"), (["bus"], "bus"),
        (["train", "railroad"], "train"), (["motorcycle"], "motorcycle"),
        (["bicycle", "bike"], "bicycle"),
    ]
    for keys, cs in table:
        if any(k in n for k in keys):
            return D.NAME2IDX[cs]
    return D.VOID


def main():
    cfg = get_cfg()
    add_deeplab_config(cfg); add_maskformer2_config(cfg); add_fcclip_config(cfg)
    cfg.merge_from_file(CFG)
    cfg.MODEL.WEIGHTS = str(FCCLIP / "fcclip_cocopan.pth")
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    print("building FC-CLIP predictor (downloads open_clip ConvNeXt-L backbone)...", flush=True)
    predictor = DefaultPredictor(cfg)
    meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    names = list(getattr(meta, "stuff_classes", []) or meta.thing_classes)
    cs_lut = np.array([coco_name_to_cs(n) for n in names], dtype=np.int32)
    print(f"predictor ready | {len(names)} classes", flush=True)

    val = D.DemoConfig().cityscapes_root / "leftImg8bit/val/frankfurt"
    targets = sorted(val.glob("*_leftImg8bit.png"))[:N]
    rows = []
    for t in targets:
        bgr = cv2.imread(str(t)); rgb = bgr[:, :, ::-1].copy()
        t0 = time.time()
        out = predictor(bgr)
        pan_t, seg_info = out["panoptic_seg"]
        pan = pan_t.cpu().numpy()
        sem = np.full(pan.shape, D.VOID, np.int32)
        for s in seg_info:
            cid = s["category_id"]
            if 0 <= cid < len(cs_lut):
                sem[pan == s["id"]] = cs_lut[cid]
        miou = D.semantic_miou(sem, D.DemoConfig(), t)
        print(f"{t.name[:26]} | FC-CLIP stuff-mIoU={miou:.3f} | segs={len(seg_info)} "
              f"| {time.time()-t0:.0f}s", flush=True)
        rows.append(dict(name=t.name, rgb=rgb, sem=sem, miou=miou))

    _grid(rows)
    print(f"MEAN FC-CLIP stuff-mIoU={np.mean([r['miou'] for r in rows]):.3f}", flush=True)


def _grid(rows):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = REPO / "auto_annotation/outputs/panoptic_demo/fcclip_result.png"

    def gtcs(name):
        gt = D.gt_labelids(D.DemoConfig(), Path(
            "/Volumes/code_files/datasets/cityscapes/leftImg8bit/val/frankfurt") / name)
        o = np.full(gt.shape, D.VOID, np.int32)
        for idx in range(len(D.IDX2LABELID)):
            o[gt == D.IDX2LABELID[idx]] = idx
        return o
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5.2 * n))
    if n == 1:
        axes = axes[None, :]
    for r, row in zip(rows, axes):
        cols = [r["rgb"], V.overlay(r["rgb"], V.colorize_semantic(r["sem"])),
                V.overlay(r["rgb"], V.colorize_semantic(gtcs(r["name"])))]
        subt = [r["name"][:24], f"FC-CLIP (CLIP open-vocab) {r['miou']:.2f}", "gtFine"]
        for ax, im, ti in zip(row, cols, subt):
            ax.imshow(im); ax.axis("off"); ax.set_title(ti, fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=80, bbox_inches="tight"); plt.close(fig)
    print("grid ->", out, flush=True)


if __name__ == "__main__":
    main()
