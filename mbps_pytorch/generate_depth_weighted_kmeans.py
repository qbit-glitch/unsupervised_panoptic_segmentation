#!/usr/bin/env python3
"""Generate depth-weighted k=80 semantic pseudo-labels.

Concatenates scaled depth features (sinusoidal encoding +/- Sobel gradients)
with L2-normalized DINOv3 features, then runs MiniBatchKMeans(k=80).

Output: raw cluster IDs 0-79 as PNG files (uint8) in pseudo_semantic_depth_weighted_k80/

Usage (fit on train, assign train+val):
    python mbps_pytorch/generate_depth_weighted_kmeans.py \
        --cityscapes_root /path/to/cityscapes \
        --depth_weight 0.3 --depth_encoding sinusoidal

Usage (with Sobel gradients):
    python mbps_pytorch/generate_depth_weighted_kmeans.py \
        --cityscapes_root /path/to/cityscapes \
        --depth_weight 0.3 --depth_encoding sobel_sinusoidal

Usage (assign using pre-fitted centroids):
    python mbps_pytorch/generate_depth_weighted_kmeans.py \
        --cityscapes_root /path/to/cityscapes \
        --split val --depth_weight 0.3 \
        --load_centroids /path/to/pseudo_semantic_depth_weighted_k80/kmeans_centroids.npz
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import sobel
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

FEAT_H, FEAT_W = 32, 64
DEPTH_H, DEPTH_W = 512, 1024
OUT_H, OUT_W = 512, 1024
DEFAULT_FREQ_BANDS: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def sinusoidal_depth_encoding(
    depth_flat: np.ndarray,
    freq_bands: Sequence[float] = DEFAULT_FREQ_BANDS,
) -> np.ndarray:
    """Encode depth values with sinusoidal positional encoding.

    Args:
        depth_flat: (N,) array of depth values in [0, 1].
        freq_bands: Frequency multipliers.

    Returns:
        (N, 2*len(freq_bands)) encoded features.
    """
    encodings = []
    for freq in freq_bands:
        encodings.append(np.sin(freq * np.pi * depth_flat))
        encodings.append(np.cos(freq * np.pi * depth_flat))
    return np.stack(encodings, axis=-1)


def compute_sobel_gradients(depth_2d: np.ndarray) -> np.ndarray:
    """Compute Sobel gradients on a 2D depth map.

    Args:
        depth_2d: (H, W) depth map.

    Returns:
        (H*W, 2) array of (gx, gy) gradient values.
    """
    gx = sobel(depth_2d, axis=1)
    gy = sobel(depth_2d, axis=0)
    return np.stack([gx.ravel(), gy.ravel()], axis=-1)


def downsample_depth(
    depth: np.ndarray, target_h: int = FEAT_H, target_w: int = FEAT_W
) -> np.ndarray:
    """Downsample depth map via block averaging to match feature resolution.

    Args:
        depth: (H, W) depth map, float32.
        target_h: Target height.
        target_w: Target width.

    Returns:
        (target_h, target_w) downsampled depth map.
    """
    src_h, src_w = depth.shape
    block_h = src_h // target_h
    block_w = src_w // target_w
    # Reshape into blocks and average
    trimmed = depth[: target_h * block_h, : target_w * block_w]
    blocks = trimmed.reshape(target_h, block_h, target_w, block_w)
    return blocks.mean(axis=(1, 3))


def encode_depth(
    depth_2d: np.ndarray, encoding: str
) -> np.ndarray:
    """Encode a downsampled depth map into feature vectors.

    Args:
        depth_2d: (FEAT_H, FEAT_W) depth map in [0, 1].
        encoding: One of 'sinusoidal', 'sobel_sinusoidal', 'raw'.

    Returns:
        (FEAT_H*FEAT_W, D) depth feature array where D depends on encoding:
            sinusoidal: 12, sobel_sinusoidal: 14, raw: 15.
    """
    depth_flat = depth_2d.ravel()  # (2048,)

    if encoding == "raw":
        sin_feat = sinusoidal_depth_encoding(depth_flat)  # (2048, 12)
        sobel_feat = compute_sobel_gradients(depth_2d)  # (2048, 2)
        raw_feat = depth_flat[:, np.newaxis]  # (2048, 1)
        return np.concatenate([sin_feat, sobel_feat, raw_feat], axis=-1)

    if encoding == "sobel_sinusoidal":
        sin_feat = sinusoidal_depth_encoding(depth_flat)  # (2048, 12)
        sobel_feat = compute_sobel_gradients(depth_2d)  # (2048, 2)
        return np.concatenate([sin_feat, sobel_feat], axis=-1)

    # Default: sinusoidal only
    return sinusoidal_depth_encoding(depth_flat)


def find_feature_files(
    cityscapes_root: Path, split: str, feat_subdir: str, depth_subdir: str
) -> List[dict]:
    """Find paired DINOv3 feature and depth files for a given split.

    Args:
        cityscapes_root: Root directory of the Cityscapes dataset.
        split: Dataset split ('train' or 'val').
        feat_subdir: Subdirectory for DINOv3 features.
        depth_subdir: Subdirectory for depth maps.

    Returns:
        List of dicts with keys 'feat', 'depth', 'stem', 'city'.
    """
    feat_dir = cityscapes_root / feat_subdir / split
    depth_dir = cityscapes_root / depth_subdir / split
    files = []
    missing_depth = 0

    for city_dir in sorted(feat_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        for npy in sorted(city_dir.glob("*.npy")):
            stem = npy.stem.replace("_leftImg8bit", "")
            depth_path = depth_dir / city_dir.name / f"{stem}.npy"
            if not depth_path.exists():
                missing_depth += 1
                continue
            files.append({
                "feat": npy,
                "depth": depth_path,
                "stem": stem,
                "city": city_dir.name,
            })

    if missing_depth > 0:
        logger.warning(f"Skipped {missing_depth} images with missing depth in {split}")
    return files


def load_combined_features(
    entry: dict,
    encoding: str,
    depth_weight: float,
) -> np.ndarray:
    """Load and combine DINOv3 + depth features for a single image.

    Args:
        entry: Dict with 'feat' and 'depth' paths.
        encoding: Depth encoding type.
        depth_weight: Scaling factor (lambda) for depth features.

    Returns:
        (2048, 768+D) combined feature array.
    """
    # Load and L2-normalize DINOv3 features
    dino_feat = np.load(str(entry["feat"])).astype(np.float32)  # (2048, 768)
    norms = np.linalg.norm(dino_feat, axis=-1, keepdims=True) + 1e-8
    dino_feat = dino_feat / norms

    # Load depth, downsample, encode
    depth = np.load(str(entry["depth"])).astype(np.float32)  # (512, 1024)
    depth_2d = downsample_depth(depth, FEAT_H, FEAT_W)  # (32, 64)
    depth_feat = encode_depth(depth_2d, encoding)  # (2048, D)
    depth_feat_scaled = depth_weight * depth_feat

    return np.concatenate([dino_feat, depth_feat_scaled], axis=-1)


def fit_kmeans(
    train_files: List[dict],
    encoding: str,
    depth_weight: float,
    k: int = 80,
    n_init: int = 5,
    batch_size: int = 4096,
    sample_frac: float = 0.5,
    max_iter: int = 150,
) -> MiniBatchKMeans:
    """Fit MiniBatchKMeans on combined DINOv3+depth features from training images.

    Args:
        train_files: List of file entry dicts.
        encoding: Depth encoding type.
        depth_weight: Scaling factor for depth features.
        k: Number of clusters.
        n_init: Number of random initializations.
        batch_size: Mini-batch size for k-means.
        sample_frac: Fraction of patches to subsample per image.
        max_iter: Maximum iterations.

    Returns:
        Fitted MiniBatchKMeans model.
    """
    logger.info(
        f"Fitting k={k} MiniBatchKMeans on {len(train_files)} train images "
        f"(encoding={encoding}, depth_weight={depth_weight}, "
        f"sample_frac={sample_frac}, batch_size={batch_size})"
    )

    t0 = time.time()
    rng = np.random.default_rng(42)
    sampled = []
    n_patches_per_image = int(FEAT_H * FEAT_W * sample_frac)

    for entry in tqdm(train_files, desc="Loading train features"):
        combined = load_combined_features(entry, encoding, depth_weight)
        idx = rng.choice(len(combined), n_patches_per_image, replace=False)
        sampled.append(combined[idx])

    X = np.concatenate(sampled, axis=0)
    logger.info(f"Feature matrix: {X.shape}, loaded in {time.time() - t0:.1f}s")

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
    logger.info(f"Fitting done in {time.time() - t1:.1f}s, inertia={kmeans.inertia_:.3f}")
    return kmeans


def assign_clusters(
    files: List[dict],
    kmeans: MiniBatchKMeans,
    output_dir: Path,
    split: str,
    encoding: str,
    depth_weight: float,
) -> None:
    """Assign cluster IDs to all images and save as PNG (uint8, raw IDs 0 to k-1).

    Args:
        files: List of file entry dicts.
        kmeans: Fitted MiniBatchKMeans model.
        output_dir: Root output directory.
        split: Dataset split name.
        encoding: Depth encoding type.
        depth_weight: Scaling factor for depth features.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for entry in tqdm(files, desc=f"Assigning {split}"):
        combined = load_combined_features(entry, encoding, depth_weight)
        cluster_ids = kmeans.predict(combined).astype(np.uint8)  # (2048,)
        cluster_2d = cluster_ids.reshape(FEAT_H, FEAT_W)

        # Upsample to OUT_H x OUT_W via nearest-neighbor
        label_full = np.array(
            Image.fromarray(cluster_2d).resize((OUT_W, OUT_H), Image.NEAREST)
        )

        city_dir = output_dir / split / entry["city"]
        city_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(label_full).save(str(city_dir / f"{entry['stem']}.png"))


def main() -> None:
    """Entry point: parse args, fit or load k-means, assign clusters."""
    parser = argparse.ArgumentParser(
        description="Depth-weighted DINOv3 k-means k=80 pseudo-labels"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument(
        "--feat_subdir", type=str, default="dinov3_features",
        help="DINOv3 feature directory under cityscapes_root",
    )
    parser.add_argument(
        "--depth_subdir", type=str, default="depth_depthpro",
        help="Depth .npy directory under cityscapes_root",
    )
    parser.add_argument(
        "--output_subdir", type=str, default="pseudo_semantic_depth_weighted_k80",
    )
    parser.add_argument("--k", type=int, default=80)
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument(
        "--sample_frac", type=float, default=0.3,
        help="Fraction of patches per image used for fitting",
    )
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument(
        "--depth_weight", type=float, default=0.3,
        help="Scaling factor (lambda) for depth features before concatenation",
    )
    parser.add_argument(
        "--depth_encoding", type=str, default="sinusoidal",
        choices=["sinusoidal", "sobel_sinusoidal", "raw"],
        help="Depth encoding type: sinusoidal (12d), sobel_sinusoidal (14d), raw (15d)",
    )
    parser.add_argument(
        "--load_centroids", type=str, default=None,
        help="Path to pre-fitted centroids npz to skip fitting",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    root = Path(args.cityscapes_root)
    out_dir = root / args.output_subdir

    train_files = find_feature_files(root, "train", args.feat_subdir, args.depth_subdir)
    logger.info(f"Found {len(train_files)} train images with paired depth")

    if args.load_centroids:
        logger.info(f"Loading centroids from {args.load_centroids}")
        data = np.load(args.load_centroids)
        kmeans = MiniBatchKMeans(n_clusters=args.k, random_state=42)
        kmeans.cluster_centers_ = data["centers"].astype(np.float32)
        kmeans.n_features_in_ = kmeans.cluster_centers_.shape[1]
    else:
        kmeans = fit_kmeans(
            train_files,
            encoding=args.depth_encoding,
            depth_weight=args.depth_weight,
            k=args.k,
            n_init=args.n_init,
            batch_size=args.batch_size,
            sample_frac=args.sample_frac,
            max_iter=args.max_iter,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(str(out_dir / "kmeans_centroids.npz"), centers=kmeans.cluster_centers_)
        logger.info(f"Saved centroids to {out_dir / 'kmeans_centroids.npz'}")

    for split in args.splits:
        files = find_feature_files(root, split, args.feat_subdir, args.depth_subdir)
        logger.info(f"Assigning {split}: {len(files)} images")
        assign_clusters(
            files, kmeans, out_dir, split,
            encoding=args.depth_encoding,
            depth_weight=args.depth_weight,
        )
        logger.info(f"Done {split}")

    logger.info(f"All done -> {out_dir}")


if __name__ == "__main__":
    main()
