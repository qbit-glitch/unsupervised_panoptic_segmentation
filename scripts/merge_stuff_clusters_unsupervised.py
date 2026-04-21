"""Merge k=80 stuff clusters into 10 super-classes using UNSUPERVISED centroid
clustering (AgglomerativeClustering on 90-D CAUSE feature space).

This is a fully unsupervised replacement for
``scripts/merge_stuff_clusters_k80_to_25.py`` which uses
``cluster_to_class`` derived from Cityscapes ground-truth labels.

Thing clusters (identified by DepthPro instance overlap ratio, no GT used)
stay standalone. Stuff clusters are grouped by feature-space similarity.

Usage::

    python scripts/merge_stuff_clusters_unsupervised.py \
        --input_dir  /path/to/cups_pseudo_labels_depthpro_tau020 \
        --output_dir /path/to/cups_pseudo_labels_depthpro_tau020_merge25_unsupervised \
        --centroids  weights/kmeans_centroids_k80_santosh.npz
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
from sklearn.cluster import AgglomerativeClustering

NUM_OLD = 80
# Thing clusters identified by PseudoLabelDataset with THING_STUFF_THRESHOLD=0.05
# on DepthPro tau=0.20 proposals. This is fully unsupervised (instance overlap).
THINGS_OLD = (3, 11, 14, 15, 16, 29, 32, 37, 38, 45, 46, 62, 65, 73, 75)
NUM_THINGS = len(THINGS_OLD)
NUM_STUFFS_TARGET = 10
NUM_NEW = NUM_STUFFS_TARGET + NUM_THINGS  # 10 + 15 = 25


def build_lookup_unsupervised(centroids_npz: str) -> np.ndarray:
    """Return a length-80 ``old_id -> new_id`` lookup using unsupervised clustering.

    Steps:
      1. Load centroids (80, 90).
      2. Extract stuff centroids (65 clusters not in THINGS_OLD).
      3. Run AgglomerativeClustering(n_clusters=10) on stuff centroids.
      4. Map each stuff cluster to its agglomerative cluster id (0..9).
      5. Map thing clusters to new ids 10..24.
    """
    data = np.load(centroids_npz)
    if "centroids" not in data.files:
        raise KeyError(f"{centroids_npz} missing 'centroids'")
    centroids = data["centroids"]
    if centroids.shape != (NUM_OLD, 90):
        raise ValueError(f"Expected centroids shape ({NUM_OLD}, 90), got {centroids.shape}")

    stuff_mask = np.ones(NUM_OLD, dtype=bool)
    stuff_mask[list(THINGS_OLD)] = False
    stuff_old_ids = np.where(stuff_mask)[0]
    stuff_centroids = centroids[stuff_mask]

    # Unsupervised agglomerative clustering on stuff centroids
    clustering = AgglomerativeClustering(n_clusters=NUM_STUFFS_TARGET, metric="euclidean", linkage="ward")
    stuff_new_ids = clustering.fit_predict(stuff_centroids)

    # Build LUT
    lut = np.full(NUM_OLD, -1, dtype=np.int64)
    for i, old_id in enumerate(stuff_old_ids):
        lut[old_id] = int(stuff_new_ids[i])
    for i, old_id in enumerate(THINGS_OLD):
        lut[old_id] = NUM_STUFFS_TARGET + i

    if (lut < 0).any() or (lut >= NUM_NEW).any():
        raise RuntimeError("LUT has out-of-range entries")

    # Print merge statistics
    print("=" * 60)
    print("Unsupervised stuff cluster merge (AgglomerativeClustering)")
    print("=" * 60)
    for new_id in range(NUM_STUFFS_TARGET):
        members = np.where(lut == new_id)[0].tolist()
        print(f"  New stuff id {new_id}: {len(members)} old clusters -> {members}")
    print(f"  Thing ids {NUM_STUFFS_TARGET}..{NUM_NEW - 1}: {THINGS_OLD}")
    print("=" * 60)

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
    """Project length-80 distribution tensors to length-25 via ``lut``."""
    blob = torch.load(src, weights_only=False)
    out = {}
    index = torch.from_numpy(lut).long()
    for key in ("distribution all pixels", "distribution inside object proposals"):
        if key not in blob:
            raise KeyError(f"{src} missing required key {key!r}")
        old_dist = blob[key]
        if old_dist.shape != (NUM_OLD,):
            raise ValueError(f"{src}[{key!r}] shape {tuple(old_dist.shape)} != ({NUM_OLD},)")
        new_dist = torch.zeros(NUM_NEW, dtype=old_dist.dtype)
        new_dist.index_add_(0, index, old_dist)
        out[key] = new_dist
    torch.save(out, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Source k=80 pseudo-label directory")
    ap.add_argument("--output_dir", required=True, help="Destination 25-class directory")
    ap.add_argument("--centroids", default="weights/kmeans_centroids_k80_santosh.npz")
    ap.add_argument("--dry_run", action="store_true", help="Only report mapping, don't write files")
    ap.add_argument("--max_files", type=int, default=0, help="Process only N samples (smoke test)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    lut = build_lookup_unsupervised(args.centroids)

    if args.dry_run:
        print("Dry run complete. No files written.")
        return

    splits = ["train", "val"]
    total_samples = 0
    for split in splits:
        split_in = input_dir / split
        split_out = output_dir / split
        if not split_in.exists():
            print(f"Skipping {split}: {split_in} not found")
            continue
        split_out.mkdir(parents=True, exist_ok=True)

        # Copy city subdirectories
        for city_dir in sorted(split_in.iterdir()):
            if not city_dir.is_dir():
                continue
            city_out = split_out / city_dir.name
            city_out.mkdir(parents=True, exist_ok=True)

            for src_file in sorted(city_dir.glob("*_leftImg8bit_semantic.png")):
                stem = src_file.name.replace("_leftImg8bit_semantic.png", "")
                dst_sem = city_out / f"{stem}_leftImg8bit_semantic.png"
                remap_png(src_file, dst_sem, lut)

                # Copy instance PNG verbatim
                src_inst = city_dir / f"{stem}_leftImg8bit_instance.png"
                dst_inst = city_out / f"{stem}_leftImg8bit_instance.png"
                if src_inst.exists():
                    shutil.copy2(src_inst, dst_inst)

                # Remap .pt distribution
                src_pt = city_dir / f"{stem}_leftImg8bit.pt"
                dst_pt = city_out / f"{stem}_leftImg8bit.pt"
                if src_pt.exists():
                    remap_pt(src_pt, dst_pt, lut)

                total_samples += 1
                if args.max_files > 0 and total_samples >= args.max_files:
                    print(f"Reached --max_files={args.max_files}, stopping.")
                    return

    print(f"Done. Wrote {total_samples} samples to {output_dir}")


if __name__ == "__main__":
    main()
