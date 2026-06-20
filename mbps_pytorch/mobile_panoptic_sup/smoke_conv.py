"""Overfit the conv student on a few COCO GT images (semantic CE) on CPU."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("smoke_conv")

from mbps_pytorch.mobile_panoptic_sup.coco_eval import VAL_IMG_DIR, load_coco_gt  # noqa: E402
from mbps_pytorch.mobile_panoptic_sup.student_conv import ConvStudent  # noqa: E402


def _sem_map(seg_map: np.ndarray, seg2cat: dict, size: int) -> np.ndarray:
    sem = np.full(seg_map.shape, 255, np.int64)
    for sid, cat in seg2cat.items():
        sem[seg_map == sid] = cat
    h, w = sem.shape
    ys = (np.arange(size) * h / size).astype(int)
    xs = (np.arange(size) * w / size).astype(int)
    return sem[ys][:, xs]


def overfit(steps: int = 40, img: int = 512,
            ids: tuple = (139, 285, 632, 724)) -> tuple[float, float]:
    torch.manual_seed(0)
    model = ConvStudent(num_classes=133, pretrained=False).train()
    imgs, sems = [], []
    for iid in ids:
        seg_map, seg2cat = load_coco_gt(iid)
        pil = Image.open(VAL_IMG_DIR / f"{iid:012d}.jpg").convert("RGB").resize((img, img))
        imgs.append(torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 255.0)
        sems.append(torch.from_numpy(_sem_map(seg_map, seg2cat, img // 4)).long())
    x, y = torch.stack(imgs), torch.stack(sems)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first, last = None, 0.0
    for i in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y, ignore_index=255)
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
    print(f"CONV_OVERFIT_OK {f:.3f} -> {l:.3f}")
