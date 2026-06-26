#!/usr/bin/env python3
"""Sanity-check: DINOv3-HF adapter + INSID3 segment() on one Cityscapes pair.

Reference = 'road' mask (gtFine labelId 7) from image A; target = image B.
Reports the IoU of the predicted mask vs image B's gtFine road region — a quick
"does the in-context segmentation actually work" check before the full demo.

Run: .venv_cups_cpu/bin/python auto_annotation/scripts/validate_insid3.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
INSID3_ROOT = ROOT / "external" / "INSID3"
sys.path.insert(0, str(INSID3_ROOT))

from models.insid3 import INSID3  # noqa: E402
from auto_annotation.backends.dinov3_hf import DinoV3HFEncoder  # noqa: E402

CS = Path("/Volumes/code_files/datasets/cityscapes")
VAL = CS / "leftImg8bit/val/frankfurt"
GT = CS / "gtFine/val/frankfurt"
ROAD_LABEL_ID = 7
IMAGE_SIZE = 768
DEVICE = "cpu"


def _gt_path(img_path: Path) -> Path:
    stem = img_path.name.replace("_leftImg8bit.png", "")
    return GT / f"{stem}_gtFine_labelIds.png"


def _road_mask(img_path: Path) -> np.ndarray:
    lab = np.array(Image.open(_gt_path(img_path)))
    return (lab == ROAD_LABEL_ID)


def main() -> None:
    imgs = sorted(VAL.glob("*_leftImg8bit.png"))[:2]
    ref_img, tgt_img = imgs[0], imgs[1]
    print(f"reference: {ref_img.name}\ntarget:    {tgt_img.name}")

    # reference road mask as a PIL image (uint8 0/255)
    ref_mask = Image.fromarray((_road_mask(ref_img) * 255).astype(np.uint8))

    print("building DINOv3-HF encoder (downloads ViT-S/16 on first run)...")
    encoder = DinoV3HFEncoder(model_size="small", device=DEVICE)

    # feature-shape sanity
    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    fmap = encoder.get_intermediate_layers(dummy, n=1, reshape=True)[0]
    print(f"feature map shape: {tuple(fmap.shape)} (expect [1,{encoder.hidden_size},"
          f"{IMAGE_SIZE//encoder.patch_size},{IMAGE_SIZE//encoder.patch_size}])")

    model = INSID3(encoder=encoder, image_size=IMAGE_SIZE, svd_components=500,
                   tau=0.6, merge_threshold=0.2, mask_refiner="bilinear",
                   resize_to_orig_size=True, device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    print("running in-context segmentation...")
    model.set_reference(Image.open(ref_img).convert("RGB"), ref_mask)
    model.set_target(Image.open(tgt_img).convert("RGB"))
    pred = model.segment().cpu().numpy().astype(bool)

    gt = _road_mask(tgt_img)
    if pred.shape != gt.shape:
        gt = np.array(Image.fromarray(gt).resize(pred.shape[::-1], Image.NEAREST))
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    iou = inter / union if union else 0.0
    print(f"pred road px: {int(pred.sum())} | gt road px: {int(gt.sum())} | "
          f"IoU={iou:.3f}")

    out = ROOT / "auto_annotation/outputs/validate_road_pred.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((pred * 255).astype(np.uint8)).save(out)
    print(f"saved predicted mask -> {out}")


if __name__ == "__main__":
    main()
