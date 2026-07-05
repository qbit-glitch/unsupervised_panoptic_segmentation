#!/usr/bin/env python3
"""Class-agnostic thing-instance mask eval: AnyUp vs bilinear instance pseudo-labels.

Question this settles: did AnyUp-upsampled semantics make the *instance masks* better or
worse, isolated from the cluster->class assignment that whole-pipeline PQ_things confounds?

Protocol (COCO-panoptic, class-agnostic):
  - pred instances = unique nonzero ids in {dir}/{stem}_leftImg8bit_instance.png (thing masks)
  - gt   instances = unique ids >=1000 in gtFine/train/{city}/{stem}_gtFine_instanceIds.png
  - match at IoU>0.5 (unique by construction). TP/FP/FN accumulated with NO class term.
  - SQ = mean IoU over TP ; RQ = TP/(TP+.5FP+.5FN) ; PQ = SQ*RQ
  - precision=TP/(TP+FP), recall=TP/(TP+FN)  (recall is the recognition bottleneck)
  - boundary-F over TP matches (2px tol) = does AnyUp's sharper edge help matched masks

The GT convention is fixed and identical for both conditions, so the ANY-vs-BIL delta is
valid even where absolute numbers differ from official 19-class PQ (crowd/ignore cancel).

ponytail: greedy match == optimal because IoU>0.5 admits at most one partner per segment.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parents[1]
ANY_DIR = ROOT / "cups_pseudo_labels_causetr_anyup_train"
BIL_DIR = ROOT / "cups_pseudo_labels_causetr_depthedge_train_full"
GT_DIR  = Path("/Volumes/code_files/datasets/cityscapes/gtFine/train")


def load(p): return np.array(Image.open(p))


def boundary_f(pm, gm, tol=2):
    """Boundary F-score of two boolean masks, cropped to their union bbox for speed."""
    ys, xs = np.where(pm | gm)
    if ys.size == 0:
        return 0.0
    y0, y1 = max(ys.min() - tol - 1, 0), ys.max() + tol + 2
    x0, x1 = max(xs.min() - tol - 1, 0), xs.max() + tol + 2
    pm, gm = pm[y0:y1, x0:x1], gm[y0:y1, x0:x1]
    pb = pm ^ binary_erosion(pm)
    gb = gm ^ binary_erosion(gm)
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0
    dt_g = distance_transform_edt(~gb)
    dt_p = distance_transform_edt(~pb)
    prec = float((dt_g[pb] <= tol).mean())
    rec = float((dt_p[gb] <= tol).mean())
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def frame_stats(pred, gt, do_bf=True, bf_tol=2):
    """Return dict: tp, fp, fn, iou_sum, bf_sum for one frame (class-agnostic)."""
    pred_ids = np.unique(pred); pred_ids = pred_ids[pred_ids != 0]
    gt_mask = gt >= 1000
    gt_ids = np.unique(gt[gt_mask])
    P, G = pred_ids.size, gt_ids.size
    if P == 0 or G == 0:
        return dict(tp=0, fp=P, fn=G, iou_sum=0.0, bf_sum=0.0)
    pa = np.bincount(np.searchsorted(pred_ids, pred[pred != 0]), minlength=P).astype(np.int64)
    ga = np.bincount(np.searchsorted(gt_ids, gt[gt_mask]), minlength=G).astype(np.int64)
    sel = (pred != 0) & gt_mask
    if sel.any():
        pl = np.searchsorted(pred_ids, pred[sel])
        gl = np.searchsorted(gt_ids, gt[sel])
        inter = np.bincount(pl * G + gl, minlength=P * G).reshape(P, G).astype(np.int64)
    else:
        inter = np.zeros((P, G), np.int64)
    union = pa[:, None] + ga[None, :] - inter
    iou = inter / np.maximum(union, 1)
    tp = 0; iou_sum = 0.0; bf_sum = 0.0
    while True:
        i, j = np.unravel_index(np.argmax(iou), iou.shape)
        if iou[i, j] <= 0.5:
            break
        tp += 1; iou_sum += float(iou[i, j])
        if do_bf:
            bf_sum += boundary_f(pred == pred_ids[i], gt == gt_ids[j], bf_tol)
        iou[i, :] = 0; iou[:, j] = 0
    return dict(tp=tp, fp=P - tp, fn=G - tp, iou_sum=iou_sum, bf_sum=bf_sum)


def agg_metrics(tp, fp, fn, iou_sum, bf_sum):
    tp = max(tp, 0)
    sq = iou_sum / tp if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 0.0
    return dict(
        PQ=100 * sq * rq, SQ=100 * sq, RQ=100 * rq,
        precision=100 * tp / (tp + fp) if (tp + fp) else 0.0,
        recall=100 * tp / (tp + fn) if (tp + fn) else 0.0,
        boundaryF=100 * bf_sum / tp if tp else 0.0,
        TP=tp, FP=fp, FN=fn)


def per_image_pq(s):
    d = s["tp"] + 0.5 * s["fp"] + 0.5 * s["fn"]
    pq = s["iou_sum"] / d if d else np.nan
    rq = s["tp"] / d if d else np.nan
    return pq, rq


def selftest():
    # two 40x40 frames. GT: one square [5:15,5:15] id 26000, one [20:35,20:35] id 24000.
    gt = np.zeros((40, 40), np.uint16)
    gt[5:15, 5:15] = 26000
    gt[20:35, 20:35] = 24000
    # pred: perfect match on square1 (id1), a shifted/partial on square2 (id2, IoU<0.5), one spurious (id3)
    pred = np.zeros((40, 40), np.uint16)
    pred[5:15, 5:15] = 1                 # IoU 1.0 with 26000  -> TP
    pred[20:28, 20:28] = 2               # 8x8 vs 15x15 -> IoU=64/225=0.28 -> not matched
    pred[0:3, 0:3] = 3                   # spurious -> FP
    s = frame_stats(pred, gt, do_bf=True)
    assert s["tp"] == 1, s
    assert s["fp"] == 2, s               # id2 (unmatched) + id3 (spurious)
    assert s["fn"] == 1, s               # 24000 unmatched
    assert abs(s["iou_sum"] - 1.0) < 1e-6, s
    m = agg_metrics(**{k: s[k] for k in ("tp", "fp", "fn", "iou_sum", "bf_sum")})
    assert abs(m["RQ"] - 100 * (1 / (1 + 0.5 * 2 + 0.5 * 1))) < 1e-6, m
    assert abs(m["SQ"] - 100.0) < 1e-6, m
    assert 99.0 <= m["boundaryF"] <= 100.0, m   # perfect-overlap boundary
    print("selftest OK:", m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-bf", action="store_true")
    ap.add_argument("--bf-tol", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "analysis-output/anyup_classagnostic"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    do_bf = not a.no_bf
    out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)

    stems = sorted(p.name.replace("_leftImg8bit_instance.png", "")
                   for p in ANY_DIR.glob("*_leftImg8bit_instance.png"))
    frames = []
    for st in stems:
        city = st.split("_")[0]
        gtp = GT_DIR / city / f"{st}_gtFine_instanceIds.png"
        bil = BIL_DIR / f"{st}_leftImg8bit_instance.png"
        if gtp.exists() and bil.exists():
            frames.append((st, city, gtp, bil))
    if a.limit:
        frames = frames[:a.limit]
    print(f"{len(frames)} frames (both conditions + GT present) | bf={do_bf}", flush=True)

    acc = {c: dict(tp=0, fp=0, fn=0, iou_sum=0.0, bf_sum=0.0) for c in ("any", "bil")}
    pi = {c: {"pq": [], "rq": []} for c in ("any", "bil")}
    t0 = time.time(); skipped = 0
    for k, (st, city, gtp, bil) in enumerate(frames):
        try:
            g = load(gtp)
            preds = {"any": load(ANY_DIR / f"{st}_leftImg8bit_instance.png"), "bil": load(bil)}
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  skip {st}: {e}", flush=True)
            continue
        for c in ("any", "bil"):
            s = frame_stats(preds[c], g, do_bf=do_bf, bf_tol=a.bf_tol)
            for key in acc[c]:
                acc[c][key] += s[key]
            pq_i, rq_i = per_image_pq(s)
            pi[c]["pq"].append(pq_i); pi[c]["rq"].append(rq_i)
        if (k + 1) % 100 == 0:
            el = time.time() - t0
            mb = agg_metrics(**acc["bil"]); ma = agg_metrics(**acc["any"])
            print(f"  [{k+1}/{len(frames)}] {el/(k+1):.2f}s/f ETA {el/(k+1)*(len(frames)-k-1)/60:.1f}m "
                  f"| BIL PQ={mb['PQ']:.2f} RQ={mb['RQ']:.2f} rec={mb['recall']:.2f} "
                  f"| ANY PQ={ma['PQ']:.2f} RQ={ma['RQ']:.2f} rec={ma['recall']:.2f}", flush=True)
            json.dump({"partial_k": k + 1, "acc": acc}, open(out / "partial.json", "w"))

    res = {c: agg_metrics(**acc[c]) for c in ("any", "bil")}
    # paired stats on per-image PQ and RQ (frames valid in both)
    stats_out = {}
    for m in ("pq", "rq"):
        va = np.array(pi["any"][m]); vb = np.array(pi["bil"][m])
        ok = ~(np.isnan(va) | np.isnan(vb))
        va, vb = va[ok], vb[ok]
        diff = va - vb
        w = sstats.wilcoxon(va, vb) if len(diff) and np.any(diff != 0) else None
        sem = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
        stats_out[m] = dict(
            n=int(len(diff)), mean_diff=float(diff.mean()), median_diff=float(np.median(diff)),
            ci95=[float(diff.mean() - 1.96 * sem), float(diff.mean() + 1.96 * sem)],
            frac_anyup_better=float((diff > 0).mean()), frac_tie=float((diff == 0).mean()),
            wilcoxon_p=float(w.pvalue) if w else None)

    summary = dict(n_frames=len(frames) - skipped, skipped=skipped, do_bf=do_bf,
                   bf_tol=a.bf_tol, results=res, paired=stats_out)
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print("\n=== CLASS-AGNOSTIC THING-INSTANCE MASK EVAL (any vs bilinear) ===", flush=True)
    hdr = f"{'metric':<11}" + "".join(f"{c:>12}" for c in ("bilinear", "anyup", "Δ(any-bil)"))
    print(hdr, flush=True)
    for key in ("PQ", "SQ", "RQ", "precision", "recall", "boundaryF", "TP", "FP", "FN"):
        b, an = res["bil"][key], res["any"][key]
        print(f"{key:<11}{b:>12.2f}{an:>12.2f}{an-b:>12.2f}", flush=True)
    print("\npaired per-image (n, mean Δ, median Δ, 95% CI, frac anyup better, Wilcoxon p):", flush=True)
    for m in ("pq", "rq"):
        s = stats_out[m]
        print(f"  {m.upper()}: n={s['n']} meanΔ={s['mean_diff']:.4f} medΔ={s['median_diff']:.4f} "
              f"CI95=[{s['ci95'][0]:.4f},{s['ci95'][1]:.4f}] better={s['frac_anyup_better']:.3f} "
              f"p={s['wilcoxon_p']}", flush=True)
    print(f"\nwrote {out/'summary.json'}", flush=True)
    make_figures(res, stats_out, pi, out)


def make_figures(res, stats_out, pi, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    keys = ["PQ", "SQ", "RQ", "precision", "recall", "boundaryF"]
    x = np.arange(len(keys)); w = 0.38
    ax[0].bar(x - w / 2, [res["bil"][k] for k in keys], w, label="bilinear", color="#4C72B0")
    ax[0].bar(x + w / 2, [res["any"][k] for k in keys], w, label="anyup", color="#C44E52")
    ax[0].set_xticks(x); ax[0].set_xticklabels(keys, rotation=20)
    ax[0].set_ylabel("score (%)"); ax[0].legend()
    ax[0].set_title("Class-agnostic thing-instance masks: anyup vs bilinear")
    va = np.array(pi["any"]["pq"]); vb = np.array(pi["bil"]["pq"])
    ok = ~(np.isnan(va) | np.isnan(vb)); d = 100 * (va[ok] - vb[ok])
    ax[1].hist(d, bins=60, color="#55A868", edgecolor="k", alpha=0.8)
    ax[1].axvline(0, color="k", lw=1)
    ax[1].axvline(d.mean(), color="red", ls="--", label=f"mean Δ={d.mean():.2f}")
    ax[1].set_xlabel("per-image PQ Δ (anyup − bilinear), %"); ax[1].set_ylabel("frames")
    ax[1].legend(); ax[1].set_title("Per-image PQ delta distribution")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figures/figure-01-anyup-vs-bilinear.{ext}", dpi=140)
    print(f"wrote {out/'figures/figure-01-anyup-vs-bilinear.pdf'}", flush=True)


if __name__ == "__main__":
    main()
