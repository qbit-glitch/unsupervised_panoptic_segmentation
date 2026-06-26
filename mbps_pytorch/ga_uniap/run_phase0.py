import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mbps_pytorch.sweep_depthpro import compute_pq_from_accumulators, NUM_CLASSES
from mbps_pytorch.ga_uniap.config import Phase0Config, VARIANTS, weight_sweep
from mbps_pytorch.ga_uniap.preflight import list_val_stems
from mbps_pytorch.ga_uniap.features import extract_grid_features
from mbps_pytorch.ga_uniap.geometry import grid_geometry
from mbps_pytorch.ga_uniap.pooling import ga_aggo_merge
from mbps_pytorch.ga_uniap.evaluate import load_gt_pair, score_image


def apply_spec(spec, feats, normal, height, cfg):
    """Run one affinity spec (VARIANTS-style dict) -> (K,gh,gw) masks."""
    wf, wn, wh = spec["weights"]
    if spec["mode"] == "single":
        return ga_aggo_merge(feats, normal, height, cfg.thresholds, cfg.min_size, wf, wn, wh)
    # split: feature-only segments + geometry-heavy segments; oracle labeling
    # downstream keeps stuff-from-A / things-from-B implicitly.
    masks_a = ga_aggo_merge(feats, None, None, cfg.thresholds, cfg.min_size, 1.0, 0.0, 0.0)
    wtf, wtn, wth = spec["weights_things"]
    masks_b = ga_aggo_merge(feats, normal, height, cfg.thresholds, cfg.min_size, wtf, wtn, wth)
    if len(masks_a) or len(masks_b):
        return np.concatenate([masks_a, masks_b], axis=0)
    return np.zeros((0, cfg.grid_h, cfg.grid_w), bool)


def run_all(specs, stems, cfg, progress=False):
    """Image-major: features+geometry computed ONCE per image; all specs share them."""
    acc = {n: [np.zeros(NUM_CLASSES) for _ in range(4)] for n in specs}
    used = {n: 0 for n in specs}
    for idx, (stem, city) in enumerate(stems):
        geo = grid_geometry(stem, city, cfg)
        if geo is None:
            continue
        normal, height = geo
        feats = extract_grid_features(stem, city, cfg)
        gt_sem, gt_inst = load_gt_pair(stem, city, cfg)
        for name, spec in specs.items():
            masks = apply_spec(spec, feats, normal, height, cfg)
            t, f, fnn, i = score_image(masks, gt_sem, gt_inst, cfg)
            a = acc[name]
            a[0] += t; a[1] += f; a[2] += fnn; a[3] += i
            used[name] += 1
        if progress and (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(stems)} images", flush=True)
    results = {}
    for name in specs:
        r = compute_pq_from_accumulators(*acc[name])
        r["_n_used"] = used[name]
        results[name] = r
    return results


def write_report(results, sweep_results, cfg, path):
    lines = ["# GA-UniAP Phase 0 — Kill-Gate Results", "",
             f"Grid {cfg.grid_h}x{cfg.grid_w}, thresholds {cfg.thresholds}, "
             f"min_size {cfg.min_size}, n_images {results['V0_vanilla']['_n_used']}.",
             "Backbone DINOv3 ViT-B/16. Oracle GT-majority labeling "
             "(diagnostic upper bound; isolates grouping, not labeling).", "",
             "| Variant | PQ | PQ_things | PQ_stuff | SQ | RQ |",
             "|---|---|---|---|---|---|"]
    for name in ["V0_vanilla", "V1_augment", "V2_split", "V3_geom_only"]:
        r = results[name]
        lines.append(f"| {name} | {r['PQ']:.2f} | {r['PQ_things']:.2f} | "
                     f"{r['PQ_stuff']:.2f} | {r['SQ']:.2f} | {r['RQ']:.2f} |")
    v0 = results["V0_vanilla"]["PQ_things"]
    best_pq, best_name = max((results[k]["PQ_things"], k)
                             for k in ["V1_augment", "V2_split", "V3_geom_only"])
    delta = best_pq - v0
    verdict = ("PASS — a geometric variant beats vanilla; proceed to Phase 1"
               if delta >= 1.0 else
               "FAIL — geometry does not beat appearance inside the pooling; stop")
    lines += ["", f"**Best geometric Δ PQ_things vs V0 = {delta:+.2f} "
              f"({best_name} {best_pq:.2f} vs V0 {v0:.2f}). {verdict}.**"]
    if sweep_results:
        lines += ["", "## V1 weight sweep (w_f=1)", "",
                  "| w_n | w_h | PQ | PQ_things |", "|---|---|---|---|"]
        for (wf, wn, wh), r in sweep_results.items():
            lines.append(f"| {wn} | {wh} | {r['PQ']:.2f} | {r['PQ_things']:.2f} |")
        best_sweep = max(r["PQ_things"] for r in sweep_results.values())
        lines += ["", f"Best sweep PQ_things = {best_sweep:.2f} (vs V0 {v0:.2f})."]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))
    print(f"wrote {path} — {verdict}")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", type=str, default="reports/ga_uniap_phase0.json")
    ap.add_argument("--report", type=str, default="reports/ga_uniap_phase0.md")
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args()
    cfg = Phase0Config(n_images=a.n)
    stems = list_val_stems(cfg)

    specs = dict(VARIANTS)
    sweep_names = {}
    if a.sweep:
        for (wf, wn, wh) in weight_sweep():
            name = f"sweep_n{wn}_h{wh}"
            specs[name] = {"mode": "single", "weights": (wf, wn, wh), "weights_things": None}
            sweep_names[name] = (wf, wn, wh)

    print(f"running {len(specs)} specs on {len(stems)} stems", flush=True)
    t0 = time.time()
    results = run_all(specs, stems, cfg, progress=a.progress)
    for name in ["V0_vanilla", "V1_augment", "V2_split", "V3_geom_only"]:
        r = results[name]
        print(f"{name:14s} PQ={r['PQ']:.2f} PQ_th={r['PQ_things']:.2f} "
              f"PQ_st={r['PQ_stuff']:.2f}  (n={r['_n_used']})")
    print(f"total {time.time()-t0:.0f}s")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    sweep_results = {sweep_names[n]: results[n] for n in sweep_names}
    write_report(results, sweep_results, cfg, a.report)


if __name__ == "__main__":
    main()
