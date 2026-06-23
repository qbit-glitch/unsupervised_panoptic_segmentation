"""Eval refs/eomt EoMT-ViT-S distill checkpoint on COCO val2017.

Greedy-NMS Mask2Former post-processing → panoptic seg map → eval_pq.

Usage:
    python eval_distill.py --ckpt /mnt/HDD_16TB/coco/eomt_distill_vits/smoke.pt
    python eval_distill.py --ckpt ...epoch_003.pt --limit 200  # quick sanity
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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "refs/eomt"))
from models.eomt import EoMT        # noqa: E402
from models.vit import ViT          # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_distill")

COCO = Path(os.environ.get("COCO_ROOT", "/mnt/HDD_16TB/coco"))
_VAL_JSON = COCO / "annotations/panoptic_val2017.json"
_VAL_PNG  = COCO / "annotations/panoptic_val2017"
_VAL_IMGS = COCO / "val2017"

# category map: sorted-by-id → contiguous 0..132
_CATS  = sorted(json.loads(_VAL_JSON.read_text())["categories"], key=lambda c: c["id"])
CATID2IDX = {c["id"]: i for i, c in enumerate(_CATS)}
IDX2NAME  = {i: c["name"] for i, c in enumerate(_CATS)}
THING_SET = {i for i, c in enumerate(_CATS) if c["isthing"] == 1}
STUFF_SET = {i for i, c in enumerate(_CATS) if c["isthing"] == 0}

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ── dataset ───────────────────────────────────────────────────────────────────

def _rgb2id(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.int64)
    return a[..., 0] + a[..., 1] * 256 + a[..., 2] * 256 * 256


class ValDS(Dataset):
    def __init__(self, size: int = 640, limit: int = 0) -> None:
        data = json.loads(_VAL_JSON.read_text())
        imgid2file = {img["id"]: img["file_name"] for img in data["images"]}
        self._anns = data["annotations"][:limit] if limit else data["annotations"]
        self._imgid2file = imgid2file
        self._size = size

    def __len__(self) -> int:
        return len(self._anns)

    def __getitem__(self, i: int) -> dict:
        ann = self._anns[i]
        img_file = self._imgid2file[ann["image_id"]]
        img = Image.open(_VAL_IMGS / img_file).convert("RGB").resize(
            (self._size, self._size), Image.BILINEAR)
        pv = (torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
              - _MEAN) / _STD

        pan_rgb = np.array(Image.open(_VAL_PNG / ann["file_name"]).convert("RGB"))
        pan_id  = _rgb2id(pan_rgb)  # original resolution

        gt_seg  = {}   # seg_id → cat_idx
        for s in ann["segments_info"]:
            if not s.get("iscrowd", 0):
                gt_seg[s["id"]] = CATID2IDX[s["category_id"]]

        return {"pv": pv, "pan_id": pan_id, "gt_seg": gt_seg,
                "image_id": ann["image_id"]}


def _collate(batch):
    return batch   # keep as list of dicts (variable GT sizes)


# ── panoptic post-processing ──────────────────────────────────────────────────

def _greedy_panoptic(
    mask_logits: torch.Tensor,   # [Q, H, W]
    cls_logits: torch.Tensor,    # [Q, C+1]   last slot = no-object
    orig_h: int,
    orig_w: int,
    conf_thresh: float = 0.5,
    mask_thresh: float = 0.5,
    overlap_thresh: float = 0.8,
) -> tuple[np.ndarray, dict]:
    """Greedy Mask2Former panoptic post-processor."""
    Q, H, W = mask_logits.shape
    scores = cls_logits[:, :-1].softmax(-1)         # [Q, C]  no-obj excluded
    cls_ids = scores.argmax(-1)                     # [Q]
    conf    = scores.max(-1).values                 # [Q]

    # filter no-object and low-confidence
    keep = (cls_ids < scores.shape[-1]) & (conf > conf_thresh)
    order = conf[keep].argsort(descending=True)

    masks_bin = (mask_logits[keep][order].sigmoid() > mask_thresh).cpu().numpy()  # [K, H, W]
    cls_kept  = cls_ids[keep][order].cpu().numpy()
    conf_kept = conf[keep][order].cpu().numpy()

    seg_map = np.zeros((H, W), dtype=np.int32)
    assigned = np.zeros((H, W), dtype=bool)
    seg2cat: dict[int, int] = {}
    sid = 1

    for k in range(masks_bin.shape[0]):
        m = masks_bin[k]
        area = m.sum()
        if area == 0:
            continue
        overlap = assigned[m].sum() / area
        if overlap > overlap_thresh:
            continue
        m_new = m & ~assigned
        if m_new.sum() == 0:
            continue
        seg_map[m_new] = sid
        assigned[m_new] = True
        seg2cat[sid] = int(cls_kept[k])
        sid += 1

    # resize to original resolution
    seg_map_big = np.array(
        Image.fromarray(seg_map.astype(np.int32)).resize((orig_w, orig_h), Image.NEAREST))
    return seg_map_big, seg2cat


# ── greedy IoU>0.5 PQ ─────────────────────────────────────────────────────────

def _compute_pq(
    gt_pan: np.ndarray, gt_seg: dict,
    pred_pan: np.ndarray, pred_seg: dict,
) -> tuple[dict, dict, dict, dict]:
    """Returns per-cat TP/IoU/FP/FN counts."""
    tp:  dict[int, int]   = {}
    iou: dict[int, float] = {}
    fp:  dict[int, int]   = {}
    fn:  dict[int, int]   = {}

    gt_by_cat:   dict[int, list] = {}
    pred_by_cat: dict[int, list] = {}
    for sid, c in gt_seg.items():
        gt_by_cat.setdefault(c, []).append(sid)
    for sid, c in pred_seg.items():
        pred_by_cat.setdefault(c, []).append(sid)

    all_cats = set(list(gt_by_cat) + list(pred_by_cat))
    matched: set = set()

    for c in all_cats:
        tp[c] = tp.get(c, 0); iou[c] = iou.get(c, 0.0)
        fp[c] = fp.get(c, 0); fn[c] = fn.get(c, 0)
        for g in gt_by_cat.get(c, []):
            gm = gt_pan == g
            best_i, best_p = 0.0, None
            for p in pred_by_cat.get(c, []):
                if p in matched:
                    continue
                pm = pred_pan == p
                inter = float(np.logical_and(gm, pm).sum())
                union = float(np.logical_or(gm, pm).sum())
                i = inter / union if union > 0 else 0.0
                if i > best_i:
                    best_i, best_p = i, p
            if best_i > 0.5:
                matched.add(best_p)
                tp[c]  += 1
                iou[c] += best_i
            else:
                fn[c] += 1
        fp[c] += max(0, len(pred_by_cat.get(c, [])) - tp.get(c, 0))

    return tp, iou, fp, fn


def _summarize(
    tp: dict, iou: dict, fp: dict, fn: dict,
) -> dict:
    cats = sorted(set(list(tp) + list(fp) + list(fn)))
    pq_per, sq_per, rq_per = {}, {}, {}
    for c in cats:
        t = tp.get(c, 0); i = iou.get(c, 0.0)
        f = fp.get(c, 0); n = fn.get(c, 0)
        rq = t / (t + 0.5 * f + 0.5 * n) if (t + f + n) > 0 else 0.0
        sq = i / t if t > 0 else 0.0
        pq_per[c] = sq * rq
        sq_per[c] = sq
        rq_per[c] = rq

    def _mean(d, idxs):
        v = [d[c] for c in idxs if c in d]
        return float(np.mean(v)) * 100 if v else 0.0

    all_cats  = set(cats)
    thing_cats = all_cats & THING_SET
    stuff_cats = all_cats & STUFF_SET

    return {
        "PQ":        _mean(pq_per, all_cats),
        "PQ_things": _mean(pq_per, thing_cats),
        "PQ_stuff":  _mean(pq_per, stuff_cats),
        "SQ":        _mean(sq_per, all_cats),
        "RQ":        _mean(rq_per, all_cats),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--backbone", default="vit_small_patch16_224")
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--conf_thresh", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = torch.device(a.device)

    model = EoMT(
        encoder=ViT(img_size=(a.img, a.img), backbone_name=a.backbone),
        num_classes=133, num_q=100, num_blocks=4,
        masked_attn_enabled=False,   # eval: no mask annealing
    ).to(dev).eval()
    sd = torch.load(a.ckpt, map_location=dev)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    log.info("loaded %s", a.ckpt)

    ds = ValDS(size=a.img, limit=a.limit)
    loader = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers,
                        collate_fn=_collate)
    log.info("val images: %d", len(ds))

    tp_acc: dict = {}; iou_acc: dict = {}
    fp_acc: dict = {}; fn_acc: dict = {}

    with torch.no_grad():
        for batch in loader:
            pvs = torch.stack([b["pv"] for b in batch]).to(dev)
            mask_l, cls_l = model(pvs)
            m_final = mask_l[-1]   # [B, Q, H, W]
            c_final = cls_l[-1]    # [B, Q, C+1]

            for j, item in enumerate(batch):
                orig_h, orig_w = item["pan_id"].shape
                pred_pan, pred_seg = _greedy_panoptic(
                    m_final[j], c_final[j], orig_h, orig_w, a.conf_thresh)

                tp, iou, fp, fn = _compute_pq(
                    item["pan_id"], item["gt_seg"], pred_pan, pred_seg)
                for c in set(list(tp) + list(fp) + list(fn)):
                    tp_acc[c]  = tp_acc.get(c, 0) + tp.get(c, 0)
                    iou_acc[c] = iou_acc.get(c, 0.0) + iou.get(c, 0.0)
                    fp_acc[c]  = fp_acc.get(c, 0) + fp.get(c, 0)
                    fn_acc[c]  = fn_acc.get(c, 0) + fn.get(c, 0)

    res = _summarize(tp_acc, iou_acc, fp_acc, fn_acc)
    print(f"\n=== EoMT-ViT-S distill eval ({len(ds)} images) ===")
    print(f"PQ:        {res['PQ']:.2f}%")
    print(f"PQ_things: {res['PQ_things']:.2f}%")
    print(f"PQ_stuff:  {res['PQ_stuff']:.2f}%")
    print(f"SQ:        {res['SQ']:.2f}%")
    print(f"RQ:        {res['RQ']:.2f}%")


if __name__ == "__main__":
    main()
