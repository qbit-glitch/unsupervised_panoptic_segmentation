"""Build EoMT with a mobile plain-ViT encoder and overfit a few COCO images (CPU).

Uses COCO GT (not auto-labels) so this isolates "does the mobile student train"
from "are the auto-labels good". Run in ``.venv``.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[2]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("smoke_eomt")

# Import refs/eomt modules FIRST, while ONLY refs/eomt is on the path. Both
# mbps_pytorch/ and refs/eomt/ ship `models/` and `training/` packages, so caching
# refs/eomt's versions here stops the later mbps_pytorch import from shadowing them.
sys.path.insert(0, str(ROOT / "refs/eomt"))
from models.vit import ViT                                            # noqa: E402
from models.eomt import EoMT                                          # noqa: E402
from training.mask_classification_loss import MaskClassificationLoss  # noqa: E402

sys.path.insert(0, str(ROOT))
from mbps_pytorch.mobile_panoptic_sup.coco_eval import VAL_IMG_DIR, load_coco_gt  # noqa: E402


def build(img: int = 512, num_classes: int = 133, num_q: int = 100,
          backbone: str = "facebook/dinov2-small", masked_attn: bool = True) -> EoMT:
    enc = ViT(img_size=(img, img), backbone_name=backbone)
    return EoMT(encoder=enc, num_classes=num_classes, num_q=num_q,
                num_blocks=4, masked_attn_enabled=masked_attn)


def _nearest_resize(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape
    ys = (np.arange(size) * h / size).astype(int)
    xs = (np.arange(size) * w / size).astype(int)
    return arr[ys][:, xs]


def _make_batch(image_ids: list[int], img: int) -> dict:
    imgs, targets = [], []
    for iid in image_ids:
        seg_map, seg2cat = load_coco_gt(iid)
        seg_small = _nearest_resize(seg_map, img)
        seg_ids = [s for s in seg2cat if bool((seg_small == s).any())]
        if not seg_ids:
            continue
        pil = Image.open(VAL_IMG_DIR / f"{iid:012d}.jpg").convert("RGB").resize((img, img))
        imgs.append(torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 255.0)
        masks = torch.stack([torch.from_numpy(seg_small == s) for s in seg_ids]).bool()
        labels = torch.tensor([seg2cat[s] for s in seg_ids], dtype=torch.long)
        targets.append({"masks": masks, "labels": labels,
                        "is_crowd": torch.zeros(len(seg_ids), dtype=torch.bool)})
    return {"images": torch.stack(imgs), "targets": targets}


def overfit(steps: int = 40, img: int = 320,
            image_ids: tuple = (139, 285, 632, 724)) -> tuple[float, float]:
    torch.manual_seed(0)
    model = build(img=img, masked_attn=False).train()   # 1 layer => faster on CPU
    crit = MaskClassificationLoss(num_points=2048, oversample_ratio=3.0,
                                  importance_sample_ratio=0.75, mask_coefficient=5.0,
                                  dice_coefficient=5.0, class_coefficient=2.0,
                                  num_labels=133, no_object_coefficient=0.1)
    batch = _make_batch(list(image_ids), img)
    logger.info("batch: %d images with targets", len(batch["targets"]))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    first, last = None, 0.0
    for i in range(steps):
        opt.zero_grad()
        mask_l, cls_l = model(batch["images"])
        loss = sum(sum(crit(m, batch["targets"], c).values())
                   for m, c in zip(mask_l, cls_l))
        loss.backward()
        opt.step()
        last = float(loss)
        first = last if first is None else first
        if i % 10 == 0:
            logger.info("step %d loss %.3f", i, last)
    logger.info("first %.3f -> last %.3f", first, last)
    return first, last


if __name__ == "__main__":
    f, l = overfit()
    assert l < 0.6 * f, f"loss did not drop enough: {f:.3f} -> {l:.3f}"
    print(f"OVERFIT_OK {f:.3f} -> {l:.3f}")
