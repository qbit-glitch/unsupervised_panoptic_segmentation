#!/usr/bin/env python3
"""End-to-end auto-annotation demo on 5 Cityscapes val images (CPU).

Semantic stage  : INSID3 (real, frozen DINOv3-HF) — 1-shot in-context per concept,
                  reference masks taken from gtFine of OTHER images (not the targets).
Instance stage  : connected components on INSID3 thing-concept masks (CPU stand-in
                  for SAM 3, which is CUDA-only on this machine — see README).
Merge + QA      : the real auto_annotation.stages.{panoptic_merge, quality}.

Outputs per image: original | colorized panoptic, plus a semantic-mIoU sanity number
vs gtFine. Writes auto_annotation/outputs/cityscapes_demo/.

Run: .venv_cups_cpu/bin/python auto_annotation/scripts/run_cityscapes_demo.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "INSID3"))
sys.path.insert(0, str(ROOT))

from models.insid3 import INSID3  # noqa: E402
from auto_annotation.backends.dinov3_hf import DinoV3HFEncoder  # noqa: E402
from auto_annotation.config import PipelineConfig  # noqa: E402
from auto_annotation.io_utils import colorize_panoptic  # noqa: E402
from auto_annotation.schemas import InstanceMask, InstanceResult, SemanticResult  # noqa: E402
from auto_annotation.stages.panoptic_merge import merge_panoptic  # noqa: E402
from auto_annotation.stages.quality import score_frame  # noqa: E402
from auto_annotation.taxonomy import name_to_id  # noqa: E402

CS = Path("/Volumes/code_files/datasets/cityscapes")
VAL = CS / "leftImg8bit/val/frankfurt"
GT = CS / "gtFine/val/frankfurt"
IMAGE_SIZE = 768
DEVICE = "cpu"
MIN_INSTANCE_PX = 400

# concept -> (cityscapes labelId, is_thing). Painted in list order (things last win).
CONCEPTS = [
    ("sky", 23, False), ("building", 11, False), ("vegetation", 21, False),
    ("sidewalk", 8, False), ("road", 7, False),
    ("car", 26, True), ("person", 24, True),
]


def gt_labelids(img_path: Path) -> np.ndarray:
    stem = img_path.name.replace("_leftImg8bit.png", "")
    return np.array(Image.open(GT / f"{stem}_gtFine_labelIds.png"))


def pick_reference(pool, label_id):
    """Return (image_path, bool_mask) for the pool image richest in this class."""
    best, best_px = None, 0
    for p in pool:
        m = gt_labelids(p) == label_id
        if m.sum() > best_px:
            best, best_px = (p, m), m.sum()
    return best if best_px > 2000 else None


def main() -> None:
    imgs = sorted(VAL.glob("*_leftImg8bit.png"))
    targets, pool = imgs[:5], imgs[5:14]
    print(f"targets: {[t.name for t in targets]}")

    encoder = DinoV3HFEncoder(model_size="small", device=DEVICE)
    model = INSID3(encoder=encoder, image_size=IMAGE_SIZE, svd_components=500,
                   tau=0.6, merge_threshold=0.2, mask_refiner="bilinear",
                   resize_to_orig_size=True, device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    # one reference (image, mask) per concept, from the pool
    refs = {}
    for name, lid, _ in CONCEPTS:
        r = pick_reference(pool, lid)
        if r:
            refs[name] = (Image.open(r[0]).convert("RGB"),
                          Image.fromarray((r[1] * 255).astype(np.uint8)))
            print(f"  ref[{name}] <- {r[0].name}")
        else:
            print(f"  ref[{name}] SKIPPED (too few px in pool)")

    cfg = PipelineConfig()
    out_dir = ROOT / "auto_annotation/outputs/cityscapes_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    ious = []

    for ti, tgt in enumerate(targets):
        rgb = Image.open(tgt).convert("RGB")
        H, W = rgb.height, rgb.width
        sem = np.full((H, W), 255, dtype=np.int32)  # 255 = void/unassigned
        instances = []

        for name, lid, is_thing in CONCEPTS:
            if name not in refs:
                continue
            model.set_reference(*refs[name])
            model.set_target(rgb)
            pred = model.segment().cpu().numpy().astype(bool)
            if pred.shape != (H, W):
                pred = np.array(Image.fromarray(pred).resize((W, H), Image.NEAREST))
            cid = name_to_id(name)
            sem[pred] = cid
            if is_thing:
                n, lab = cv2.connectedComponents(pred.astype(np.uint8))
                for k in range(1, n):
                    comp = lab == k
                    if comp.sum() >= MIN_INSTANCE_PX:
                        instances.append(InstanceMask(comp, cid, 0.9, track_id=len(instances)))

        # semantic mIoU sanity vs gtFine (mapped classes only)
        gt = gt_labelids(tgt)
        per = []
        for name, lid, _ in CONCEPTS:
            if name not in refs:
                continue
            p, g = (sem == name_to_id(name)), (gt == lid)
            u = (p | g).sum()
            if u:
                per.append((p & g).sum() / u)
        miou = float(np.mean(per)) if per else 0.0
        ious.append(miou)

        sem_res = SemanticResult(label_map=sem,
                                 confidence=np.full((H, W), 0.85, np.float32))
        pan = merge_panoptic(sem_res, InstanceResult(instances), cfg)
        q = score_frame(sem_res, cfg)

        color = colorize_panoptic(pan)
        orig = np.array(rgb)
        montage = np.concatenate([orig, color], axis=1)
        Image.fromarray(montage).save(out_dir / f"{tgt.stem}_panoptic.png")
        n_things = sum(s["isthing"] for s in pan.segments_info)
        print(f"[{ti+1}/5] {tgt.name[:28]} | segs={len(pan.segments_info)} "
              f"things={n_things} | sem-mIoU={miou:.3f} | route={q.reason}")

    print(f"\nmean semantic mIoU over 5 imgs (7 classes, 1-shot INSID3): "
          f"{np.mean(ious):.3f}")
    print(f"montages -> {out_dir}")


if __name__ == "__main__":
    main()
