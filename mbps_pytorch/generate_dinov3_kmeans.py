#!/usr/bin/env python3
"""Generate k=80 semantic pseudo-labels from pre-extracted DINOv3 features.

Fits MiniBatchKMeans on DINOv3 train features (2975 × 2048 patches × 768 dim),
then assigns clusters to all train+val images.

Output: raw cluster IDs 0-79 as PNG files (uint8) in pseudo_semantic_raw_dinov3_k80/

Usage (fit on train, assign train+val):
    python mbps_pytorch/generate_dinov3_kmeans.py \
        --cityscapes_root /path/to/cityscapes \
        --k 80 --n_init 10 --batch_size 4096

Usage (assign using pre-fitted centroids):
    python mbps_pytorch/generate_dinov3_kmeans.py \
        --cityscapes_root /path/to/cityscapes \
        --split val --k 80 \
        --load_centroids /path/to/pseudo_semantic_raw_dinov3_k80/kmeans_centroids.npz
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEAT_H, FEAT_W = 32, 64   # DINOv3 ViT-B/16 patch grid at 512×1024 input
OUT_H, OUT_W = 512, 1024   # Output label resolution


def find_feature_files(cityscapes_root: Path, split: str, feat_subdir: str) -> list:
    """Find all .npy feature files for a given split."""
    feat_dir = cityscapes_root / feat_subdir / split
    files = []
    for city_dir in sorted(feat_dir.iterdir()):
        if city_dir.is_dir():
            for npy in sorted(city_dir.glob("*.npy")):
                stem = npy.stem.replace("_leftImg8bit", "")
                files.append({"feat": npy, "stem": stem, "city": city_dir.name})
    return files


def fit_kmeans(
    train_files: list,
    k: int = 80,
    n_init: int = 5,
    batch_size: int = 4096,
    sample_frac: float = 0.5,
    max_iter: int = 150,
) -> MiniBatchKMeans:
    """Fit MiniBatchKMeans on subsampled DINOv3 features from all training images."""
    logger.info(f"Fitting k={k} MiniBatchKMeans on {len(train_files)} train images "
                f"(sample_frac={sample_frac}, batch_size={batch_size})")

    # Collect features: subsample patches from each image
    t0 = time.time()
    rng = np.random.default_rng(42)
    sampled = []
    n_patches_per_image = int(FEAT_H * FEAT_W * sample_frac)

    for entry in tqdm(train_files, desc="Loading train features"):
        feat = np.load(str(entry["feat"])).astype(np.float32)  # (2048, 768)
        # L2-normalize features (cosine similarity used in affinity)
        norms = np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-8
        feat = feat / norms
        idx = rng.choice(len(feat), n_patches_per_image, replace=False)
        sampled.append(feat[idx])

    X = np.concatenate(sampled, axis=0)  # (N_train * n_patches_per_image, 768)
    logger.info(f"Feature matrix: {X.shape}, loaded in {time.time()-t0:.1f}s")

    # Fit MiniBatchKMeans
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        n_init=n_init,
        max_iter=max_iter,
        random_state=42,
        verbose=1,
    )
    logger.info("Fitting k-means...")
    t1 = time.time()
    kmeans.fit(X)
    logger.info(f"Fitting done in {time.time()-t1:.1f}s, inertia={kmeans.inertia_:.3f}")
    return kmeans


def assign_clusters(
    files: list,
    kmeans: MiniBatchKMeans,
    output_dir: Path,
    split: str,
) -> None:
    """Assign cluster IDs to all images and save as PNG (uint8, raw IDs 0 to k-1)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for entry in tqdm(files, desc=f"Assigning {split}"):
        feat = np.load(str(entry["feat"])).astype(np.float32)  # (2048, 768)
        norms = np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-8
        feat = feat / norms

        cluster_ids = kmeans.predict(feat).astype(np.uint8)  # (2048,)
        cluster_2d = cluster_ids.reshape(FEAT_H, FEAT_W)

        # Upsample to OUT_H × OUT_W via nearest-neighbor
        label_full = np.array(
            Image.fromarray(cluster_2d).resize((OUT_W, OUT_H), Image.NEAREST)
        )

        city_dir = output_dir / split / entry["city"]
        city_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(label_full).save(str(city_dir / f"{entry['stem']}.png"))


def main():
    parser = argparse.ArgumentParser(description="DINOv3 k-means k=80 pseudo-labels")
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--feat_subdir", type=str, default="dinov3_features",
                        help="Feature directory under cityscapes_root")
    parser.add_argument("--output_subdir", type=str, default="pseudo_semantic_raw_dinov3_k80")
    parser.add_argument("--k", type=int, default=80)
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--sample_frac", type=float, default=0.3,
                        help="Fraction of patches per image used for fitting (default 0.3)")
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--load_centroids", type=str, default=None,
                        help="Path to pre-fitted centroids npz to skip fitting")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    root = Path(args.cityscapes_root)
    out_dir = root / args.output_subdir

    train_files = find_feature_files(root, "train", args.feat_subdir)
    logger.info(f"Found {len(train_files)} train images")

    if args.load_centroids:
        logger.info(f"Loading centroids from {args.load_centroids}")
        data = np.load(args.load_centroids)
        kmeans = MiniBatchKMeans(n_clusters=args.k, random_state=42)
        kmeans.cluster_centers_ = data["centers"].astype(np.float32)
        kmeans.n_features_in_ = kmeans.cluster_centers_.shape[1]
    else:
        kmeans = fit_kmeans(
            train_files,
            k=args.k,
            n_init=args.n_init,
            batch_size=args.batch_size,
            sample_frac=args.sample_frac,
            max_iter=args.max_iter,
        )
        # Save centroids
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(str(out_dir / "kmeans_centroids.npz"), centers=kmeans.cluster_centers_)
        logger.info(f"Saved centroids to {out_dir / 'kmeans_centroids.npz'}")

    # Assign for each split
    for split in args.splits:
        files = find_feature_files(root, split, args.feat_subdir)
        logger.info(f"Assigning {split}: {len(files)} images")
        assign_clusters(files, kmeans, out_dir, split)
        logger.info(f"Done {split}")

    logger.info(f"All done → {out_dir}")


if __name__ == "__main__":
    main()
