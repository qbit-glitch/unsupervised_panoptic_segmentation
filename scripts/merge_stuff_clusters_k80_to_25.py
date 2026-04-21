"""Merge k=80 stuff clusters into 10 Cityscapes-style super-classes.

M1 fix for Stage-2 M2F PQ_things=0. The k=80 pseudo-label head gives
M2F an 80-way classification target; with 100 queries and 20k steps the
rare thing classes get <100 gradient events in the whole budget (~36x
undertrained vs the M2F paper). Shrinking the output space 80 → 25
keeps the semantic granularity the K-means overclustering was designed
for on the *instance* side while letting stuff queries pool gradients
across semantically-similar clusters.

Mapping strategy (derived from ``weights/kmeans_centroids_k80_santosh.npz``):

* 15 thing clusters (identified by ``PseudoLabelDataset`` with
  ``THING_STUFF_THRESHOLD=0.05`` on DepthPro tau=0.20 proposals):
      {3, 11, 14, 15, 16, 29, 32, 37, 38, 45, 46, 62, 65, 73, 75}
  Each stays standalone → new IDs 10-24.

* 65 stuff clusters, grouped by their ``cluster_to_class`` CAUSE label
  into 10 coarse stuff buckets → new IDs 0-9:
      CAUSE class 0  (16 clusters)  → new id 0
      CAUSE class 1  ( 7 clusters)  → new id 1
      CAUSE class 2  (15 clusters)  → new id 2
      CAUSE class 3  ( 3 clusters)  → new id 3
      CAUSE class 4  ( 3 clusters)  → new id 4
      CAUSE class 5  ( 1 cluster)   → new id 5
      CAUSE class 7  ( 1 cluster)   → new id 6
      CAUSE class 8  (14 clusters)  → new id 7
      CAUSE class 9  ( 2 clusters)  → new id 8
      CAUSE class 10 ( 3 clusters)  → new id 9

Writes a new pseudo-label directory alongside the original with three
kinds of file per sample:
  * ``*_leftImg8bit_semantic.png``  — uint8 PNG, values 0-24 (was 0-79)
  * ``*_leftImg8bit_instance.png``  — copied verbatim from the input
  * ``*_leftImg8bit.pt``            — length-25 distribution tensors

The instance PNG is untouched because the DepthPro tau=0.20 proposals
don't change under semantic remapping; only the pixel-wise class IDs
do. ``PseudoLabelDataset`` re-runs its own thing/stuff split on the
emitted ``.pt`` distributions so the 15 things (new IDs 10-24) end up
in ``self.things_classes`` and the 10 stuff ids in ``self.stuff_classes``.

Usage::

    python scripts/merge_stuff_clusters_k80_to_25.py \\
        --input_dir /path/to/cups_pseudo_labels_depthpro_tau020 \\
        --output_dir /path/to/cups_pseudo_labels_depthpro_tau020_merge25 \\
        --centroids  weights/kmeans_centroids_k80_santosh.npz

Flags:
  --dry_run            only report the mapping + coverage, don't write files
  --max_files N        process only N samples (smoke test)
"""
from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image


NUM_OLD = 80
# Things identified by the probe + PseudoLabelDataset split logic.
# Keep the original old-id ordering so the thing sub-space is deterministic.
THINGS_OLD = (3, 11, 14, 15, 16, 29, 32, 37, 38, 45, 46, 62, 65, 73, 75)
NUM_THINGS = len(THINGS_OLD)
# Stuff CAUSE classes in sorted order → new stuff IDs 0..9
STUFF_CAUSE_CLASSES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10)
NUM_STUFFS = len(STUFF_CAUSE_CLASSES)
NUM_NEW = NUM_STUFFS + NUM_THINGS  # 10 + 15 = 25


def build_lookup(centroids_npz: str) -> np.ndarray:
    """Return a length-80 ``old_id -> new_id`` lookup.

    Raises if the k=80 space contains a stuff cluster whose CAUSE class
    is not in ``STUFF_CAUSE_CLASSES`` (that would silently drop pixels
    into the unmapped 0 bucket).
    """
    data = np.load(centroids_npz)
    if "cluster_to_class" not in data.files:
        raise KeyError(
            f"{centroids_npz} is missing 'cluster_to_class'; "
            f"use the santosh canonical centroids."
        )
    c2c = data["cluster_to_class"]
    if c2c.shape != (NUM_OLD,):
        raise ValueError(f"cluster_to_class shape {c2c.shape} != ({NUM_OLD},)")
    stuff_cause_to_new = {c: i for i, c in enumerate(STUFF_CAUSE_CLASSES)}
    things_old_to_new = {old: NUM_STUFFS + i for i, old in enumerate(THINGS_OLD)}
    lut = np.full(NUM_OLD, -1, dtype=np.int64)
    for old_id in range(NUM_OLD):
        if old_id in things_old_to_new:
            lut[old_id] = things_old_to_new[old_id]
        else:
            cause = int(c2c[old_id])
            if cause not in stuff_cause_to_new:
                raise ValueError(
                    f"stuff cluster {old_id} maps to CAUSE class {cause} "
                    f"which is not in STUFF_CAUSE_CLASSES={STUFF_CAUSE_CLASSES}. "
                    f"Either add it to the mapping or re-derive STUFF_CAUSE_CLASSES."
                )
            lut[old_id] = stuff_cause_to_new[cause]
    if (lut < 0).any() or (lut >= NUM_NEW).any():
        raise RuntimeError("LUT has out-of-range entries — check mapping logic")
    return lut


def remap_png(src: Path, dst: Path, lut: np.ndarray) -> Counter:
    """Rewrite an 8-bit semantic PNG through ``lut``. Returns pixel histogram."""
    arr = np.array(Image.open(src))
    if arr.dtype != np.uint8:
        raise ValueError(f"{src} has dtype {arr.dtype}, expected uint8")
    if arr.max() >= NUM_OLD:
        raise ValueError(f"{src} has max pixel {arr.max()} >= {NUM_OLD}")
    remapped = lut[arr].astype(np.uint8)
    Image.fromarray(remapped).save(dst)
    return Counter(remapped.flatten().tolist())


def remap_pt(src: Path, dst: Path, lut: np.ndarray) -> None:
    """Project length-80 distribution tensors to length-25 via ``lut``.

    Uses ``index_add`` so clusters that share a new ID accumulate their
    per-sample histogram weights correctly.
    """
    blob = torch.load(src, weights_only=False)
    out = {}
    index = torch.from_numpy(lut).long()
    for key in ("distribution all pixels", "distribution inside object proposals"):
        if key not in blob:
            raise KeyError(f"{src} missing required key {key!r}")
        old_dist = blob[key]
        if old_dist.shape != (NUM_OLD,):
            raise ValueError(
                f"{src}[{key!r}] shape {tuple(old_dist.shape)} != ({NUM_OLD},)"
            )
        new_dist = torch.zeros(NUM_NEW, dtype=old_dist.dtype)
        new_dist.index_add_(0, index, old_dist)
        out[key] = new_dist
    torch.save(out, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--centroids",
        default="weights/kmeans_centroids_k80_santosh.npz",
        help="Path to k=80 centroids with 'cluster_to_class' (santosh canonical).",
    )
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--max_files", type=int, default=None)
    args = ap.parse_args()

    lut = build_lookup(args.centroids)
    print("=" * 70)
    print(f"LUT (old_id → new_id):")
    for old in range(NUM_OLD):
        role = "thing" if lut[old] >= NUM_STUFFS else "stuff"
        print(f"  old={old:2d} → new={int(lut[old]):2d} ({role})")
    print(f"\nTotal new classes: {NUM_NEW} "
          f"({NUM_STUFFS} stuff + {NUM_THINGS} thing)")

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"--input_dir {in_dir} is not a directory")
    if args.dry_run:
        print("\n[dry_run] exiting before any writes")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sample-level files: each sample is a stem (e.g., aachen_000000_000019_leftImg8bit)
    # with _semantic.png, _instance.png, and .pt. Pair them by stem.
    sem_files = sorted(p.name for p in in_dir.glob("*_leftImg8bit_semantic.png"))
    if args.max_files:
        sem_files = sem_files[: args.max_files]
    print(f"\nProcessing {len(sem_files)} samples…")

    total_hist: Counter = Counter()
    missing = 0
    for i, sem_name in enumerate(sem_files):
        stem = sem_name.replace("_semantic.png", "")
        sem_src = in_dir / sem_name
        inst_src = in_dir / f"{stem}_instance.png"
        pt_src = in_dir / f"{stem}.pt"
        if not (inst_src.exists() and pt_src.exists()):
            missing += 1
            continue

        sem_dst = out_dir / sem_name
        inst_dst = out_dir / inst_src.name
        pt_dst = out_dir / pt_src.name

        total_hist.update(remap_png(sem_src, sem_dst, lut))
        # Instance PNG unchanged; copy verbatim (hardlink where possible).
        if inst_dst.exists():
            inst_dst.unlink()
        try:
            os.link(inst_src, inst_dst)
        except OSError:
            shutil.copy2(inst_src, inst_dst)
        remap_pt(pt_src, pt_dst, lut)

        if (i + 1) % 500 == 0:
            print(f"  [{i + 1}/{len(sem_files)}]")

    print(f"\nDone. Missing pair count: {missing}")
    print("Pixel histogram over new IDs:")
    total = sum(total_hist.values())
    for new_id in range(NUM_NEW):
        n = total_hist.get(new_id, 0)
        pct = 100.0 * n / max(total, 1)
        role = "thing" if new_id >= NUM_STUFFS else "stuff"
        print(f"  new={new_id:2d} ({role}): {n:12d} px ({pct:5.2f}%)")


if __name__ == "__main__":
    main()
