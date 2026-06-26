"""Train EoMT-mobile on label-free auto-labels with gradient accumulation.

Effective batch = micro_batch * accum (default 4*4 = 16, the standard segmentation
protocol). Reads the auto-label COCO-panoptic format (per-image <stem>.png rgb2id +
<stem>.json segments_info); a --gt_smoke N mode trains on N COCO val GT images to
verify the loop on hardware before the real labels are ready.

  # box GPU smoke (verify grad-accum eff-16 trains):
  python train_eomt_mobile.py --gt_smoke 32 --device cuda --eff_batch 16 --micro_batch 4 --steps 30
  # real run on auto-labels:
  python train_eomt_mobile.py --data_dir /mnt/HDD_16TB/coco/autolabels_train \\
    --img_dir /mnt/HDD_16TB/coco/train2017 --device cuda --eff_batch 16 --micro_batch 4 --epochs 40
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "refs/eomt"))     # import refs/eomt's models+training FIRST
from models.eomt import EoMT                                              # noqa: E402
from models.vit import ViT                                                # noqa: E402
from training.mask_classification_loss import MaskClassificationLoss      # noqa: E402

sys.path.insert(0, str(ROOT))
from mbps_pytorch.mobile_panoptic_sup.coco_eval import VAL_IMG_DIR, load_coco_gt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_eomt_mobile")
DIV = 1000


def _nearest_resize(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape
    ys = (np.arange(size) * h / size).astype(int)
    xs = (np.arange(size) * w / size).astype(int)
    return arr[ys][:, xs]


def _build_target(seg_map: np.ndarray, seg2cat: dict, size: int) -> dict | None:
    seg_small = _nearest_resize(seg_map, size)
    seg_ids = [s for s in seg2cat if bool((seg_small == s).any())]
    if not seg_ids:
        return None
    masks = torch.stack([torch.from_numpy(seg_small == s) for s in seg_ids]).bool()
    labels = torch.tensor([seg2cat[s] for s in seg_ids], dtype=torch.long)
    return {"masks": masks, "labels": labels,
            "is_crowd": torch.zeros(len(seg_ids), dtype=torch.bool)}


def _load_img(path: Path, size: int) -> torch.Tensor:
    pil = Image.open(path).convert("RGB").resize((size, size))
    return torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 255.0


def _rgb2id(png: Path) -> np.ndarray:
    a = np.asarray(Image.open(png).convert("RGB")).astype(np.int64)
    return a[..., 0] + a[..., 1] * 256 + a[..., 2] * 65536


class AutoLabelDataset(Dataset):
    """Auto-label COCO-panoptic dir (<stem>.png rgb2id + <stem>.json segments_info)."""

    def __init__(self, data_dir: Path, img_dir: Path, size: int) -> None:
        self.size = size
        self.img_dir = img_dir
        self.items = sorted(p for p in data_dir.glob("*.json"))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        jf = self.items[i]
        ann = json.loads(jf.read_text())
        seg_map = _rgb2id(jf.with_suffix(".png"))
        seg2cat = {s["id"]: s["category_id"] for s in ann["segments_info"]}
        tgt = _build_target(seg_map, seg2cat, self.size)
        if tgt is None:
            return self[(i + 1) % len(self)]
        return _load_img(self.img_dir / f"{jf.stem}.jpg", self.size), tgt


class GtSmokeDataset(Dataset):
    """N COCO val GT images — to verify the training loop on hardware."""

    def __init__(self, n: int, size: int) -> None:
        self.size = size
        self.ids = [int(p.stem) for p in sorted(VAL_IMG_DIR.glob("*.jpg"))[:n]]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        seg_map, seg2cat = load_coco_gt(self.ids[i])
        tgt = _build_target(seg_map, seg2cat, self.size)
        if tgt is None:
            return self[(i + 1) % len(self)]
        return _load_img(VAL_IMG_DIR / f"{self.ids[i]:012d}.jpg", self.size), tgt


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    return imgs, [b[1] for b in batch]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=None)
    ap.add_argument("--img_dir", type=Path, default=None)
    ap.add_argument("--gt_smoke", type=int, default=0)
    ap.add_argument("--backbone", default="facebook/dinov2-small")
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--eff_batch", type=int, default=16)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=0, help="cap optimizer steps (smoke)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_masked_attn", action="store_true",
                    help="disable EoMT masked attention (debug only; cripples mask quality)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "checkpoints/eomt_mobile")
    args = ap.parse_args()

    assert args.eff_batch % args.micro_batch == 0, "eff_batch must be divisible by micro_batch"
    accum = args.eff_batch // args.micro_batch
    args.out.mkdir(parents=True, exist_ok=True)

    if args.gt_smoke:
        ds = GtSmokeDataset(args.gt_smoke, args.img)
    else:
        ds = AutoLabelDataset(args.data_dir, args.img_dir, args.img)
    loader = DataLoader(ds, batch_size=args.micro_batch, shuffle=True,
                        num_workers=args.workers, collate_fn=_collate, drop_last=True)
    logger.info("dataset=%d micro_batch=%d accum=%d eff_batch=%d steps/epoch=%d",
                len(ds), args.micro_batch, accum, args.eff_batch, len(loader) // accum)

    dev = torch.device(args.device)
    use_masked = not args.no_masked_attn
    model = EoMT(encoder=ViT(img_size=(args.img, args.img), backbone_name=args.backbone),
                 num_classes=133, num_q=100, num_blocks=4,
                 masked_attn_enabled=use_masked).to(dev).train()
    crit = MaskClassificationLoss(num_points=12544, oversample_ratio=3.0,
                                  importance_sample_ratio=0.75, mask_coefficient=5.0,
                                  dice_coefficient=5.0, class_coefficient=2.0,
                                  num_labels=133, no_object_coefficient=0.1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    loss_w = {"loss_mask": 5.0, "loss_dice": 5.0, "loss_cross_entropy": 2.0}  # Mask2Former weights
    total_steps = args.steps or args.epochs * max(1, len(loader) // accum)
    opt_step, micro = 0, 0
    opt.zero_grad()
    for epoch in range(args.epochs):
        for imgs, targets in loader:
            imgs = imgs.to(dev)
            targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]
            if use_masked:  # anneal masked-attn prob 1->0 so train converges to unmasked eval
                model.attn_mask_probs.fill_(max(0.0, 1.0 - opt_step / total_steps))
            mask_l, cls_l = model(imgs)
            loss = sum(sum(loss_w.get(k, 1.0) * v for k, v in crit(m, targets, c).items())
                       for m, c in zip(mask_l, cls_l)) / accum
            loss.backward()
            micro += 1
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                opt_step += 1
                if opt_step % 5 == 0:
                    logger.info("epoch %d opt_step %d loss %.3f",
                                epoch, opt_step, float(loss) * accum)
                if args.steps and opt_step >= args.steps:
                    torch.save(model.state_dict(), args.out / "smoke.pt")
                    logger.info("SMOKE_DONE opt_step=%d", opt_step)
                    return
        torch.save(model.state_dict(), args.out / f"epoch_{epoch:03d}.pt")
        logger.info("saved epoch %d", epoch)


if __name__ == "__main__":
    main()
