#!/usr/bin/env python3
"""Multi-Seed Consensus: exploit k-means stochasticity to filter unreliable pixels.

Runs k-means with multiple seeds on pre-extracted DINOv3 features, maps each
run's clusters to 19-class trainIDs via majority vote against GT, then masks
pixels where fewer than a threshold of seeds agree on the class.

Usage:
    python scripts/generate_consensus_labels.py \
        --cityscapes_root ~/Desktop/datasets/cityscapes \
        --baseline_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_depthpro_tau020 \
        --output_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_consensus5 \
        --seeds 42,123,456,789,1024 \
        --min_agreement 3
"""

import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEAT_H, FEAT_W = 32, 64
NUM_CLASSES = 19

_CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}


def _extract_city(cups_stem: str) -> str:
    match = re.match(r"^(.+?)_\d{6}_\d{6}_leftImg8bit$", cups_stem)
    return match.group(1) if match else cups_stem.split("_")[0]


def _base_stem(cups_stem: str) -> str:
    return cups_stem.replace("_leftImg8bit", "")


def find_feature_files(feat_dir: Path, split: str) -> list:
    """Find all .npy feature files for a split, returns list of dicts."""
    files = []
    split_dir = feat_dir / split
    for city_dir in sorted(split_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        for npy in sorted(city_dir.glob("*.npy")):
            stem = npy.stem.replace("_leftImg8bit", "")
            files.append({"feat": npy, "stem": stem, "city": city_dir.name})
    return files


def fit_kmeans_seed(train_files: list, k: int, seed: int,
                    batch_size: int = 4096, sample_frac: float = 0.3,
                    max_iter: int = 100) -> MiniBatchKMeans:
    """Fit MiniBatchKMeans with a specific random seed."""
    logger.info(f"  Fitting k={k} with seed={seed}...")
    rng = np.random.default_rng(seed)
    n_patches = int(FEAT_H * FEAT_W * sample_frac)
    sampled = []

    for entry in tqdm(train_files, desc=f"  Loading (seed={seed})", leave=False):
        feat = np.load(str(entry["feat"])).astype(np.float32)
        norms = np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-8
        feat = feat / norms
        idx = rng.choice(len(feat), n_patches, replace=False)
        sampled.append(feat[idx])

    X = np.concatenate(sampled, axis=0)

    kmeans = MiniBatchKMeans(
        n_clusters=k, batch_size=batch_size, n_init=5,
        max_iter=max_iter, random_state=seed, verbose=0,
    )
    t0 = time.time()
    kmeans.fit(X)
    logger.info(f"  seed={seed} done in {time.time()-t0:.1f}s, inertia={kmeans.inertia_:.1f}")
    return kmeans


def compute_cluster_to_class(centers: np.ndarray, train_files: list,
                             gt_dir: Path, k: int) -> np.ndarray:
    """Compute cluster->trainID mapping via majority vote against GT labels.

    Returns:
        cluster_to_class: (k,) array, values 0-18 or 255.
    """
    # Accumulate per-cluster class counts
    counts = np.zeros((k, NUM_CLASSES), dtype=np.int64)

    for entry in tqdm(train_files, desc="  Cluster->class", leave=False):
        feat = np.load(str(entry["feat"])).astype(np.float32)
        norms = np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-8
        feat = feat / norms
        cluster_ids = predict_clusters(centers, feat)  # (2048,) values 0 to k-1

        # Load GT
        gt_path = gt_dir / "train" / entry["city"] / f"{entry['stem']}_gtFine_labelIds.png"
        if not gt_path.exists():
            continue
        gt_raw = np.array(Image.open(gt_path))
        gt_small = np.array(Image.fromarray(gt_raw).resize((FEAT_W, FEAT_H), Image.NEAREST))

        gt_train = np.full_like(gt_small, 255, dtype=np.uint8)
        for raw_id, train_id in _CS_ID_TO_TRAIN.items():
            gt_train[gt_small == raw_id] = train_id

        gt_flat = gt_train.ravel()
        for patch_idx in range(len(cluster_ids)):
            cid = cluster_ids[patch_idx]
            gid = gt_flat[patch_idx]
            if gid < NUM_CLASSES:
                counts[cid, gid] += 1

    # Majority vote
    cluster_to_class = np.full(k, 255, dtype=np.uint8)
    for cid in range(k):
        if counts[cid].sum() > 0:
            cluster_to_class[cid] = counts[cid].argmax()

    matched = np.sum(cluster_to_class < NUM_CLASSES)
    logger.info(f"  Matched {matched}/{k} clusters")
    return cluster_to_class


def predict_clusters(centers: np.ndarray, feat: np.ndarray) -> np.ndarray:
    """Manual nearest-centroid prediction (sklearn-version-independent).

    Args:
        centers: (K, D) cluster centroids.
        feat: (N, D) feature vectors.

    Returns:
        (N,) uint8 cluster IDs.
    """
    dists = cdist(feat, centers, metric="euclidean")  # (N, K)
    return dists.argmin(axis=1).astype(np.uint8)


def predict_trainids(centers: np.ndarray, cluster_to_class: np.ndarray,
                     feat: np.ndarray, target_hw: tuple) -> np.ndarray:
    """Predict trainIDs for a single image at target resolution.

    Args:
        centers: (K, D) cluster centroids.
        cluster_to_class: (K,) cluster -> trainID mapping.
        feat: (N_patches, D) L2-normalized features.
        target_hw: (H, W) target resolution for output.

    Returns:
        (H, W) uint8 trainID map.
    """
    cluster_ids = predict_clusters(centers, feat)
    cluster_2d = cluster_ids.reshape(FEAT_H, FEAT_W)
    target_h, target_w = target_hw
    upsampled = np.array(
        Image.fromarray(cluster_2d).resize((target_w, target_h), Image.NEAREST)
    )
    return cluster_to_class[upsampled]


def compute_distributions(semantic: np.ndarray, instance: np.ndarray,
                          num_classes: int) -> dict:
    """Compute per-class pixel distributions for CUPS."""
    sem_flat = semantic.flatten().astype(np.int64)
    inst_flat = instance.flatten()
    valid = sem_flat < num_classes

    dist_all = torch.zeros(num_classes, dtype=torch.float32)
    if valid.sum() > 0:
        c = np.bincount(sem_flat[valid], minlength=num_classes)
        dist_all = torch.from_numpy(c[:num_classes].astype(np.float32))

    inside = (inst_flat > 0) & valid
    dist_inside = torch.zeros(num_classes, dtype=torch.float32)
    if inside.sum() > 0:
        c = np.bincount(sem_flat[inside], minlength=num_classes)
        dist_inside = torch.from_numpy(c[:num_classes].astype(np.float32))

    return {
        "distribution all pixels": dist_all,
        "distribution inside object proposals": dist_inside,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-seed consensus pseudo-labels")
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--baseline_dir", type=str, required=True,
                        help="Baseline CUPS label directory (provides original cluster IDs + instances)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--centroids_path", type=str, required=True,
                        help="Path to baseline kmeans_centroids.npz")
    parser.add_argument("--features_subdir", type=str, default="dinov3_features")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1024",
                        help="Comma-separated seeds (first must be baseline seed)")
    parser.add_argument("--min_agreement", type=int, default=3,
                        help="Minimum seed agreement to keep a pixel (default: 3/5)")
    parser.add_argument("--k", type=int, default=80)
    parser.add_argument("--num_clusters", type=int, default=80,
                        help="Number of clusters in baseline labels")
    args = parser.parse_args()

    t0_total = time.time()
    seeds = [int(s) for s in args.seeds.split(",")]
    n_seeds = len(seeds)
    logger.info(f"Seeds: {seeds}, min_agreement: {args.min_agreement}/{n_seeds}")

    cs_root = Path(args.cityscapes_root).expanduser()
    baseline_dir = Path(args.baseline_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    feat_dir = cs_root / args.features_subdir
    gt_dir = cs_root / "gtFine"
    train_files = find_feature_files(feat_dir, "train")
    logger.info(f"Found {len(train_files)} training images")

    # Load baseline cluster_to_class
    baseline_data = np.load(args.centroids_path)
    baseline_c2c = baseline_data["cluster_to_class"].astype(np.uint8)

    # Step 1: Fit k-means for each seed and compute cluster_to_class
    # Store (centers, c2c) — NOT full per-image maps (saves ~30 GB RAM)
    seed_models = []  # list of (centers, cluster_to_class)
    for i, seed in enumerate(seeds):
        logger.info(f"\n--- Seed {seed} ({i+1}/{n_seeds}) ---")
        if seed == 42 and "centers" in baseline_data:
            logger.info("  Reusing baseline centroids (seed=42)")
            centers = baseline_data["centers"].astype(np.float32)
            c2c = baseline_c2c
        else:
            kmeans = fit_kmeans_seed(train_files, args.k, seed)
            centers = kmeans.cluster_centers_.astype(np.float32)
            c2c = compute_cluster_to_class(centers, train_files, gt_dir, args.k)
        seed_models.append((centers, c2c))

    # Build stem -> feature path lookup
    stem_to_feat = {}
    for entry in train_files:
        stem_to_feat[entry["stem"]] = entry["feat"]

    # Step 2: Per-image consensus (O(1) memory per image)
    logger.info(f"\n--- Applying consensus (min_agreement={args.min_agreement}) ---")

    baseline_stems = []
    for p in sorted(baseline_dir.glob("*_semantic.png")):
        baseline_stems.append(p.name.replace("_semantic.png", ""))

    total_kept = 0
    total_masked = 0
    total_pixels = 0
    agreement_hist = np.zeros(n_seeds + 1, dtype=np.int64)

    for cups_stem in tqdm(baseline_stems, desc="Consensus"):
        base = _base_stem(cups_stem)

        # Load baseline semantic + instance
        semantic = np.array(Image.open(baseline_dir / f"{cups_stem}_semantic.png"))
        instance = np.array(Image.open(baseline_dir / f"{cups_stem}_instance.png"))
        sem_h, sem_w = semantic.shape

        # Baseline trainID map
        baseline_tids = baseline_c2c[semantic]

        # Load features for this image
        feat_path = stem_to_feat.get(base)
        if feat_path is None:
            # No features — copy unchanged
            Image.fromarray(semantic.astype(np.uint8)).save(
                str(output_dir / f"{cups_stem}_semantic.png")
            )
            Image.fromarray(instance.astype(np.uint16)).save(
                str(output_dir / f"{cups_stem}_instance.png")
            )
            stats = compute_distributions(semantic, instance, args.num_clusters)
            torch.save(stats, str(output_dir / f"{cups_stem}.pt"))
            total_pixels += sem_h * sem_w
            continue

        feat = np.load(str(feat_path)).astype(np.float32)
        norms = np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-8
        feat = feat / norms

        # Predict trainIDs for each seed at baseline resolution
        seed_tids = np.stack([
            predict_trainids(centers, c2c, feat, (sem_h, sem_w))
            for centers, c2c in seed_models
        ], axis=0)  # (n_seeds, sem_h, sem_w)

        n_pixels = sem_h * sem_w
        total_pixels += n_pixels

        # Count agreement with baseline per pixel
        agree_count = np.sum(seed_tids == baseline_tids[None, :, :], axis=0)

        for a in range(n_seeds + 1):
            agreement_hist[a] += int(np.sum(agree_count == a))

        # Mask unreliable pixels
        reliable = agree_count >= args.min_agreement
        masked = ~reliable & (baseline_tids < NUM_CLASSES)

        out_semantic = semantic.copy()
        out_semantic[masked] = 255

        total_kept += int(reliable.sum())
        total_masked += int(masked.sum())

        # Regenerate .pt
        stats = compute_distributions(out_semantic, instance, args.num_clusters)

        # Save
        Image.fromarray(out_semantic.astype(np.uint8)).save(
            str(output_dir / f"{cups_stem}_semantic.png")
        )
        Image.fromarray(instance.astype(np.uint16)).save(
            str(output_dir / f"{cups_stem}_instance.png")
        )
        torch.save(stats, str(output_dir / f"{cups_stem}.pt"))

    elapsed_total = time.time() - t0_total

    logger.info(f"\n{'='*60}")
    logger.info(f"Consensus complete in {elapsed_total:.1f}s ({len(baseline_stems)} images)")
    logger.info(f"  Kept: {total_kept} pixels ({100*total_kept/max(total_pixels,1):.1f}%)")
    logger.info(f"  Masked: {total_masked} pixels ({100*total_masked/max(total_pixels,1):.1f}%)")
    logger.info(f"\nAgreement histogram:")
    for a in range(n_seeds + 1):
        pct = 100 * agreement_hist[a] / max(total_pixels, 1)
        logger.info(f"  {a}/{n_seeds} seeds agree: {agreement_hist[a]:>12d} pixels ({pct:.1f}%)")
    logger.info(f"Output: {output_dir} ({len(baseline_stems) * 3} files)")


if __name__ == "__main__":
    main()
