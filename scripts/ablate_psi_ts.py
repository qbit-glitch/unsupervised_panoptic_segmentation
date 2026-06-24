"""Sweep ψ_ts (thing-stuff threshold) over our fragmentation-frequency ratio.

Computes per-cluster ratio ONCE, then evaluates every threshold without re-scanning.
Two modes:
  - instance (default): ratio = images where cluster has ≥2 distinct instance IDs / images where cluster appears
  - cc (--cc): ratio = images where cluster's mask is fragmented (no single blob > dominance%) / images where cluster appears

Usage (on fics-lab):
    CS=/mnt/HDD_16TB/datasets/Cityscapes
    SEM=pseudo_semantic_raw_dinov3_k27_spherical_kmeans_vitl16

    # instance mode (uses gated unMORE instances):
    python3 scripts/ablate_psi_ts.py \
        --semantic_dir $CS/$SEM/train \
        --instance_dir $CS/pseudo_instance_spidepth/train \
        --output reports/ablate_psi_ts_instance.json

    # cc mode (semantic only, no instances needed):
    python3 scripts/ablate_psi_ts.py \
        --semantic_dir $CS/$SEM/train \
        --cc \
        --output reports/ablate_psi_ts_cc.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

PSI_VALUES = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14]
K27_CITYSCAPES_NAMES = {
    # From CUPS paper / Niu et al. Cityscapes-27 mapping (informational only)
    # Cluster IDs 0-26; these names come from CAUSE-TR's concept alignment.
    # We print raw cluster IDs + ratios; users map to concept names separately.
}


def compute_ratios_instance(semantic_dir: str, instance_dir: str, num_clusters: int = 27):
    """Per-cluster: fraction of images where cluster has ≥2 distinct instance IDs."""
    appear = np.zeros(num_clusters, dtype=np.int64)
    multi = np.zeros(num_clusters, dtype=np.int64)

    sem_paths = sorted(glob.glob(os.path.join(semantic_dir, "**", "*.png"), recursive=True))
    if not sem_paths:
        sys.exit(f"No PNGs found in {semantic_dir}")

    matched = 0
    for sp in sem_paths:
        stem = os.path.basename(sp).replace("_leftImg8bit.png", "").replace(".png", "")
        city = os.path.basename(os.path.dirname(sp))
        cands = [
            os.path.join(instance_dir, city, f"{stem}.png"),
            os.path.join(instance_dir, city, f"{stem}_leftImg8bit.png"),
        ]
        ip = next((p for p in cands if os.path.exists(p)), None)
        if ip is None:
            continue
        matched += 1
        sem = np.array(Image.open(sp), dtype=np.int32)
        inst = np.array(Image.open(ip), dtype=np.int32)
        for c in range(num_clusters):
            mask = sem == c
            if not mask.any():
                continue
            appear[c] += 1
            ids = np.unique(inst[mask])
            ids = ids[ids > 0]
            if len(ids) > 1:
                multi[c] += 1

    print(f"Matched {matched}/{len(sem_paths)} semantic↔instance pairs", flush=True)
    ratio = multi / (appear + 1e-9)
    return ratio, appear, multi


def compute_ratios_cc(semantic_dir: str, num_clusters: int = 27,
                      min_area: int = 64, dominance: float = 0.75):
    """Per-cluster: fraction of images where cluster mask is fragmented (no dominant blob)."""
    from scipy import ndimage
    appear = np.zeros(num_clusters, dtype=np.int64)
    fragmented = np.zeros(num_clusters, dtype=np.int64)

    sem_paths = sorted(glob.glob(os.path.join(semantic_dir, "**", "*.png"), recursive=True))
    if not sem_paths:
        sys.exit(f"No PNGs found in {semantic_dir}")

    for sp in sem_paths:
        sem = np.array(Image.open(sp), dtype=np.int32)
        for c in range(num_clusters):
            mask = sem == c
            tot = int(mask.sum())
            if tot < min_area:
                continue
            appear[c] += 1
            labeled, n = ndimage.label(mask)
            if n == 0:
                continue
            sizes = np.bincount(labeled.ravel())[1:]
            largest = int(sizes.max()) if len(sizes) else 0
            if largest / (tot + 1e-9) < dominance:
                fragmented[c] += 1

    ratio = fragmented / (appear + 1e-9)
    return ratio, appear, fragmented


def sweep_thresholds(ratio: np.ndarray, appear: np.ndarray, counts: np.ndarray,
                     num_clusters: int = 27):
    """Print threshold sensitivity table and return results dict."""
    print("\n" + "=" * 70)
    print("Per-cluster fragmentation ratios (sorted descending):")
    print("=" * 70)
    order = np.argsort(ratio)[::-1]
    print(f"{'Cluster':>8}  {'Ratio':>7}  {'Appear':>8}  {'Multi/Frag':>10}")
    print("-" * 40)
    for c in order:
        print(f"  c{int(c):02d}      {ratio[c]:.4f}   {appear[c]:8d}   {counts[c]:10d}")

    print("\n" + "=" * 70)
    print(f"{'ψ':>6}  {'#things':>8}  thing_clusters")
    print("=" * 70)

    results = {}
    for psi in PSI_VALUES:
        things = sorted(int(c) for c in np.where(ratio > psi)[0])
        stuff = sorted(set(range(num_clusters)) - set(things))
        print(f"  {psi:.2f}    {len(things):>5}    {things}")
        results[str(psi)] = {"things": things, "stuff": stuff, "n_things": len(things)}

    # Stability analysis: which clusters flip between thresholds?
    print("\n" + "=" * 70)
    print("Cluster stability — ratio distance to boundary:")
    boundaries = np.array(PSI_VALUES)
    for c in order:
        r = ratio[c]
        # Find which ψ values straddle this ratio
        flips = [(psi, "thing→stuff") for psi in PSI_VALUES if r > psi - 0.005 and r < psi + 0.02]
        if flips:
            print(f"  c{int(c):02d}  ratio={r:.4f}  UNSTABLE near ψ={[p for p,_ in flips]}")
    print("=" * 70)

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--semantic_dir", required=True)
    p.add_argument("--instance_dir", default=None)
    p.add_argument("--cc", action="store_true", help="Use CC mode (no instance dir needed)")
    p.add_argument("--num_clusters", type=int, default=27)
    p.add_argument("--min_area", type=int, default=64, help="CC mode: min pixels per cluster")
    p.add_argument("--dominance", type=float, default=0.75,
                   help="CC mode: largest blob fraction below which cluster is 'fragmented'")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    if a.cc or a.instance_dir is None:
        print(f"CC mode — scanning {a.semantic_dir}", flush=True)
        ratio, appear, counts = compute_ratios_cc(
            a.semantic_dir, a.num_clusters, a.min_area, a.dominance)
        mode = "cc"
    else:
        print(f"Instance mode — {a.semantic_dir} + {a.instance_dir}", flush=True)
        ratio, appear, counts = compute_ratios_instance(
            a.semantic_dir, a.instance_dir, a.num_clusters)
        mode = "instance"

    results = sweep_thresholds(ratio, appear, counts, a.num_clusters)

    out = {
        "mode": mode,
        "num_clusters": a.num_clusters,
        "per_cluster_ratio": {f"c{c:02d}": float(ratio[c]) for c in range(a.num_clusters)},
        "per_cluster_appear": {f"c{c:02d}": int(appear[c]) for c in range(a.num_clusters)},
        "thresholds": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n→ saved {a.output}")


if __name__ == "__main__":
    main()
