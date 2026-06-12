#!/usr/bin/env python3
"""Phase 0b complementarity audit (KILL GATE) for the DepthG x CAUSE-TR fusion.

Inputs: the mapped val prediction dumps written by eval_fusion_adapter.py
(--dump_preds) for CAUSE vanilla, DepthG mono, DepthG official, plus gtFine.

Outputs (reports/):
    fusion_audit_results.json   per-class IoU, agreement stats, oracle headroom
    fusion_audit_perclass.csv   27-class IoU table (cause / mono / official)

Analyses (spec Phase 0b):
  1. Per-class IoU per model (27-class, each at its own protocol crop).
  2. Pixel agreement: agree-right / agree-wrong / cause-only-right /
     depthg-only-right (DepthG preds nearest-resized 320->322; both protocols
     crop the same normalized region x in [0.25, 0.75], y full).
  3. GT-oracle per-pixel-best upper bound -> headroom (GT analysis-only).
  4. Concat sanity: whiten+L2 per branch, concat [z; g_aligned], MiniBatch
     k-means k=27 on train-cache tokens, Hungarian mIoU vs z-only baseline.

Gate (spec): headroom >= ~1.5 mIoU over the stronger vanilla model AND
disagreements not dominated by both-wrong.

Run:
    .venv_cups_cpu/bin/python mbps_pytorch/audit_fusion_complementarity.py
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DATA = Path("/Volumes/code_files/datasets/cityscapes")
AUDIT = DATA / "fusion_audit"
CACHE = DATA / "fusion_feature_cache"
N = 27
# Cityscapes labelIds 7..33 -> 27-class trainids (the CityscapesSeg "-7" rule)
CLASS_NAMES = [
    "road", "sidewalk", "parking", "rail track", "building", "wall", "fence",
    "guard rail", "bridge", "tunnel", "pole", "polegroup", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "caravan", "trailer", "train", "motorcycle", "bicycle",
]
FOCUS = ["traffic light", "motorcycle", "pole", "person"]


def gt_crop(stem: str, res: int) -> np.ndarray:
    """gtFine labelIds -> 27-class ids at the protocol's center crop, -1 ignore."""
    city = stem.split("_")[0]
    p = DATA / "gtFine" / "val" / city / f"{stem}_gtFine_labelIds.png"
    img = Image.open(p)
    w, h = img.size
    scale = res / min(h, w)
    img = img.resize((round(w * scale), round(h * scale)), Image.NEAREST)
    w2, h2 = img.size
    left, top = (w2 - res) // 2, (h2 - res) // 2
    img = img.crop((left, top, left + res, top + res))
    arr = np.array(img).astype(np.int64) - 7
    arr[(arr < 0) | (arr >= N)] = -1
    return arr


def load_preds(dump_dir: Path) -> Dict[str, np.ndarray]:
    out = {}
    for f in sorted(dump_dir.glob("*_pred.png")):
        out[f.name.replace("_pred.png", "")] = np.array(Image.open(f)).astype(np.int64)
    if not out:
        raise RuntimeError(f"no dumps in {dump_dir}")
    return out


def iou_per_class(conf: np.ndarray) -> np.ndarray:
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    return tp / np.maximum(tp + fp + fn, 1e-9)


def confusion(preds: Dict[str, np.ndarray], res: int) -> np.ndarray:
    conf = np.zeros((N, N), dtype=np.int64)
    for stem, pr in preds.items():
        gt = gt_crop(stem, res)
        m = gt >= 0
        conf += np.bincount(gt[m] * N + pr[m], minlength=N * N).reshape(N, N)
    return conf


def resize_nearest(arr: np.ndarray, size: int) -> np.ndarray:
    return np.array(Image.fromarray(arr.astype(np.uint8)).resize((size, size), Image.NEAREST)).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cause_dir", default=str(AUDIT / "cause_vanilla"))
    ap.add_argument("--mono_dir", default=str(AUDIT / "depthg_mono"))
    ap.add_argument("--official_dir", default=str(AUDIT / "depthg_official"))
    ap.add_argument("--concat_images", type=int, default=400)
    ap.add_argument("--skip_concat", action="store_true")
    args = ap.parse_args()

    cause = load_preds(Path(args.cause_dir))      # 322x322
    mono = load_preds(Path(args.mono_dir))        # 320x320
    official = load_preds(Path(args.official_dir))
    stems = sorted(set(cause) & set(mono) & set(official))
    logger.info("common val stems: %d", len(stems))

    results: Dict = {"n_images": len(stems)}

    # 1) per-class IoU, each at its own protocol resolution
    per_class: Dict[str, List[float]] = {}
    for name, preds, res in (("cause", cause, 322), ("depthg_mono", mono, 320),
                             ("depthg_official", official, 320)):
        conf = confusion({s: preds[s] for s in stems}, res)
        iou = iou_per_class(conf)
        per_class[name] = (iou * 100).round(2).tolist()
        results[f"miou_{name}"] = float(np.mean(iou) * 100)
        logger.info("%s mIoU=%.2f", name, results[f"miou_{name}"])

    # 2+3) agreement + oracle at 322 (cause grid), for both depthg ckpts
    for dg_name, dg_preds in (("mono", mono), ("official", official)):
        agree = np.zeros(4, dtype=np.int64)  # both-right, both-wrong-or-half: see keys
        conf_oracle = np.zeros((N, N), dtype=np.int64)
        keys = ["agree_right", "agree_wrong", "cause_only_right", "depthg_only_right",
                "both_wrong_disagree"]
        counts = {k: 0 for k in keys}
        for s in stems:
            gt = gt_crop(s, 322)
            c = cause[s]
            d = resize_nearest(dg_preds[s], 322)
            m = gt >= 0
            cr, dr = (c == gt), (d == gt)
            counts["agree_right"] += int(np.sum(m & cr & dr & (c == d)))
            counts["agree_wrong"] += int(np.sum(m & ~cr & ~dr & (c == d)))
            counts["cause_only_right"] += int(np.sum(m & cr & ~dr))
            counts["depthg_only_right"] += int(np.sum(m & ~cr & dr))
            counts["both_wrong_disagree"] += int(np.sum(m & ~cr & ~dr & (c != d)))
            oracle = np.where(cr, c, np.where(dr, d, c))
            conf_oracle += np.bincount(gt[m] * N + oracle[m], minlength=N * N).reshape(N, N)
        total = sum(counts.values())
        results[f"agreement_{dg_name}"] = {k: round(v / total * 100, 2) for k, v in counts.items()}
        oracle_miou = float(np.mean(iou_per_class(conf_oracle)) * 100)
        stronger = max(results["miou_cause"], results[f"miou_depthg_{dg_name}"])
        results[f"oracle_miou_{dg_name}"] = oracle_miou
        results[f"headroom_{dg_name}"] = round(oracle_miou - stronger, 2)
        logger.info("[%s] oracle=%.2f stronger-vanilla=%.2f headroom=%.2f | %s",
                    dg_name, oracle_miou, stronger, oracle_miou - stronger,
                    results[f"agreement_{dg_name}"])

    # 4) concat sanity on train cache tokens
    if not args.skip_concat:
        import torch
        import torch.nn.functional as Fn
        from sklearn.cluster import MiniBatchKMeans
        from scipy.optimize import linear_sum_assignment

        zs, gs, gts = [], [], []
        z_files = sorted((CACHE / "cause_z" / "train").glob("*/*_codes.npy"))[: args.concat_images]
        for zf in z_files:
            stem = zf.name.replace("_codes.npy", "")
            city = zf.parent.name
            gf = CACHE / "depthg_g_mono" / "train" / city / f"{stem}_g.npy"
            lp = DATA / "gtFine" / "train" / city / f"{stem}_gtFine_labelIds.png"
            if not (gf.is_file() and lp.is_file()):
                continue
            z = torch.from_numpy(np.load(zf).astype(np.float32))          # (23,46,90)
            g = torch.from_numpy(np.load(gf).astype(np.float32))          # (40,80,100)
            g = Fn.interpolate(g.permute(2, 0, 1)[None], z.shape[:2],
                               mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
            gt_full = np.array(Image.open(lp).resize((46, 23), Image.NEAREST)).astype(np.int64) - 7
            gt_full[(gt_full < 0) | (gt_full >= N)] = -1
            zs.append(z.reshape(-1, 90).numpy())
            gs.append(g.reshape(-1, 100).numpy())
            gts.append(gt_full.reshape(-1))
        Z = np.concatenate(zs); G = np.concatenate(gs); Y = np.concatenate(gts)
        logger.info("concat sanity tokens: %d (from %d images)", len(Y), len(zs))

        def whiten_l2(x: np.ndarray) -> np.ndarray:
            x = (x - x.mean(0)) / (x.std(0) + 1e-8)
            return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

        def km_miou(feats: np.ndarray) -> float:
            km = MiniBatchKMeans(n_clusters=N, n_init=10, random_state=42,
                                 batch_size=4096).fit(feats)
            pred = km.labels_
            m = Y >= 0
            conf = np.bincount(Y[m] * N + pred[m], minlength=N * N).reshape(N, N)
            r, c = linear_sum_assignment(conf, maximize=True)
            remap = np.zeros(N, dtype=np.int64)
            remap[c] = r
            conf2 = np.bincount(Y[m] * N + remap[pred[m]], minlength=N * N).reshape(N, N)
            return float(np.mean(iou_per_class(conf2)) * 100)

        results["concat_kmeans_miou"] = km_miou(np.concatenate(
            [whiten_l2(Z), whiten_l2(G)], axis=1))
        results["zonly_kmeans_miou"] = km_miou(whiten_l2(Z))
        logger.info("concat kmeans mIoU=%.2f vs z-only=%.2f",
                    results["concat_kmeans_miou"], results["zonly_kmeans_miou"])

    # focus classes table
    idx = [CLASS_NAMES.index(n) for n in FOCUS]
    results["focus_classes"] = {
        CLASS_NAMES[i]: {m: per_class[m][i] for m in per_class} for i in idx}

    out_json = Path("reports/fusion_audit_results.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    with open("reports/fusion_audit_perclass.csv", "w") as fh:
        fh.write("class," + ",".join(per_class) + "\n")
        for i, cname in enumerate(CLASS_NAMES):
            fh.write(cname + "," + ",".join(str(per_class[m][i]) for m in per_class) + "\n")
    logger.info("audit written -> %s", out_json)

    # gate verdict
    hr = results.get("headroom_mono")
    both_wrong = results.get("agreement_mono", {}).get("agree_wrong", 0) + \
        results.get("agreement_mono", {}).get("both_wrong_disagree", 0)
    informative = results.get("agreement_mono", {}).get("depthg_only_right", 0)
    logger.info("GATE(mono): headroom=%.2f (need >=~1.5), depthg-only-right=%.2f%%, "
                "both-wrong=%.2f%%", hr, informative, both_wrong)


if __name__ == "__main__":
    main()
