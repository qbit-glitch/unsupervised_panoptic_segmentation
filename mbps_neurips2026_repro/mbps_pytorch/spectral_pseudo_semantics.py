#!/usr/bin/env python3
"""Approach 1: Spectral Enrichment + Hierarchical NCut Merge for COCO-Stuff-27.

Enrich DINOv3 features with per-image spectral eigenvectors from the graph
Laplacian (EAGLE/DeepCut++ inspired), then k-means + hierarchical merge.

Usage:
    python mbps_pytorch/spectral_pseudo_semantics.py \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --alpha 0.7 --n_eig 20 --k 300 --merge_method ncut
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ─── COCO-Stuff-27 class definitions (reused from generate_coco_pseudo_semantics.py) ───

COCOSTUFF27_CLASSNAMES = [
    "electronic", "appliance", "food", "furniture", "indoor",
    "kitchen", "accessory", "animal", "outdoor", "person",
    "sports", "vehicle",
    "ceiling", "floor", "food-stuff", "furniture-stuff", "raw-material",
    "textile", "wall", "window", "building", "ground",
    "plant", "sky", "solid", "structural", "water",
]
NUM_CLASSES = 27
THING_IDS = set(range(12))
STUFF_IDS = set(range(12, 27))

SUPERCATEGORY_TO_COARSE = {
    "electronic": 0, "appliance": 1, "food": 2, "furniture": 3,
    "indoor": 4, "kitchen": 5, "accessory": 6, "animal": 7,
    "outdoor": 8, "person": 9, "sports": 10, "vehicle": 11,
    "ceiling": 12, "floor": 13, "food-stuff": 14, "furniture-stuff": 15,
    "raw-material": 16, "textile": 17, "wall": 18, "window": 19,
    "building": 20, "ground": 21, "plant": 22, "sky": 23,
    "solid": 24, "structural": 25, "water": 26,
}


def load_coco_panoptic_gt(coco_root: str, image_id: int) -> Optional[np.ndarray]:
    """Load COCO panoptic GT and convert to 27-class semantic label map."""
    panoptic_json = Path(coco_root) / "annotations" / "panoptic_val2017.json"
    panoptic_dir = Path(coco_root) / "annotations" / "panoptic_val2017"

    if not hasattr(load_coco_panoptic_gt, "_cache"):
        with open(panoptic_json) as f:
            data = json.load(f)
        cat_map = {cat["id"]: cat["supercategory"] for cat in data["categories"]}
        ann_map = {ann["image_id"]: ann for ann in data["annotations"]}
        load_coco_panoptic_gt._cache = (cat_map, ann_map, str(panoptic_dir))

    cat_map, ann_map, pdir = load_coco_panoptic_gt._cache
    if image_id not in ann_map:
        return None

    ann = ann_map[image_id]
    pan_img = np.array(Image.open(Path(pdir) / ann["file_name"]))
    pan_id = (pan_img[:, :, 0].astype(np.int32) +
              pan_img[:, :, 1].astype(np.int32) * 256 +
              pan_img[:, :, 2].astype(np.int32) * 256 * 256)

    sem_label = np.full(pan_id.shape, 255, dtype=np.uint8)
    for seg in ann["segments_info"]:
        mask = pan_id == seg["id"]
        supercat = cat_map.get(seg["category_id"])
        if supercat and supercat in SUPERCATEGORY_TO_COARSE:
            sem_label[mask] = SUPERCATEGORY_TO_COARSE[supercat]
    return sem_label


# ─── Spectral Feature Enrichment ───

def build_feature_affinity(
    features: np.ndarray,
    k_neighbors: int = 10,
) -> np.ndarray:
    """Build sparse cosine affinity matrix from patch features.

    Args:
        features: Patch features, shape (N, C). Already L2-normalized.
        k_neighbors: Keep only top-k neighbors per patch.

    Returns:
        Sparse affinity matrix A_feat, shape (N, N).
    """
    n = features.shape[0]
    sim = features @ features.T  # (N, N) cosine similarity

    # Sparsify: keep top-k per row
    if k_neighbors < n:
        topk_idx = np.argpartition(-sim, k_neighbors, axis=1)[:, :k_neighbors]
        mask = np.zeros_like(sim, dtype=bool)
        rows = np.arange(n)[:, None]
        mask[rows, topk_idx] = True
        # Symmetrize
        mask = mask | mask.T
        sim = sim * mask

    # Shift to [0, 1] and zero out negative
    sim = np.clip(sim, 0.0, None)
    np.fill_diagonal(sim, 0.0)
    return sim


def build_color_spatial_affinity(
    image: np.ndarray,
    patch_grid: int,
    sigma_color: float = 0.3,
    sigma_spatial: float = 5.0,
) -> np.ndarray:
    """Build color+spatial RBF affinity matrix from downsampled LAB image.

    Args:
        image: RGB image, any size.
        patch_grid: Downsample to this grid size (e.g., 32).
        sigma_color: RBF bandwidth for LAB color distance.
        sigma_spatial: RBF bandwidth for spatial distance.

    Returns:
        Affinity matrix A_color, shape (N, N) where N=patch_grid².
    """
    from skimage.color import rgb2lab

    # Downsample to patch grid
    img_small = np.array(Image.fromarray(image).resize(
        (patch_grid, patch_grid), Image.BILINEAR
    ))
    lab = rgb2lab(img_small).reshape(-1, 3)  # (N, 3)
    lab = lab / np.array([100.0, 128.0, 128.0])  # Normalize to ~[0,1]

    # Spatial coordinates
    ys, xs = np.mgrid[:patch_grid, :patch_grid]
    coords = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float32)
    coords = coords / patch_grid  # Normalize to [0, 1]

    # RBF kernels
    color_dist = np.sum((lab[:, None, :] - lab[None, :, :]) ** 2, axis=-1)
    spatial_dist = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)

    affinity = np.exp(-color_dist / (2.0 * sigma_color ** 2)
                      - spatial_dist / (2.0 * sigma_spatial ** 2))
    np.fill_diagonal(affinity, 0.0)
    return affinity


def compute_spectral_features(
    affinity: np.ndarray,
    n_eigenvectors: int = 20,
) -> np.ndarray:
    """Compute spectral features (eigenvectors of symmetric normalized Laplacian).

    Args:
        affinity: Affinity matrix A, shape (N, N).
        n_eigenvectors: Number of smallest non-trivial eigenvectors to return.

    Returns:
        Eigenvector matrix V, shape (N, n_eigenvectors).
    """
    n = affinity.shape[0]
    d = np.sum(affinity, axis=1)
    d_inv_sqrt = 1.0 / (np.sqrt(d) + 1e-8)

    # Symmetric normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
    l_sym = (np.eye(n) - (d_inv_sqrt[:, None] * affinity) * d_inv_sqrt[None, :])

    # Eigendecompose — take smallest eigenvectors (skip index 0 = trivial)
    end_idx = min(1 + n_eigenvectors, n)
    eigenvalues, eigenvectors = eigh(l_sym, subset_by_index=[1, end_idx - 1])

    return eigenvectors  # (N, n_eigenvectors)


def enrich_features_spectral(
    features: np.ndarray,
    image: np.ndarray,
    patch_grid: int,
    alpha: float = 0.7,
    k_neighbors: int = 10,
    n_eigenvectors: int = 20,
    sigma_color: float = 0.3,
    sigma_spatial: float = 5.0,
) -> np.ndarray:
    """Enrich DINOv3 features with spectral eigenvectors.

    Args:
        features: DINOv3 patch features (N, C), L2-normalized.
        image: RGB image (H, W, 3).
        patch_grid: Patch grid size (e.g., 32).
        alpha: Weight for feature affinity vs color affinity.
        k_neighbors: Sparsity for feature affinity graph.
        n_eigenvectors: Number of spectral dimensions to add.
        sigma_color: Color RBF bandwidth.
        sigma_spatial: Spatial RBF bandwidth.

    Returns:
        Enriched features (N, C + n_eigenvectors).
    """
    a_feat = build_feature_affinity(features, k_neighbors)

    if alpha < 1.0:
        a_color = build_color_spatial_affinity(
            image, patch_grid, sigma_color, sigma_spatial
        )
        affinity = alpha * a_feat + (1.0 - alpha) * a_color
    else:
        affinity = a_feat

    if n_eigenvectors == 0:
        return features

    spectral = compute_spectral_features(affinity, n_eigenvectors)

    # Scale spectral features to match DINOv3 feature magnitude
    feat_scale = np.std(features)
    spectral_scaled = spectral * feat_scale

    return np.concatenate([features, spectral_scaled], axis=1)


# ─── Hierarchical Merge ───

def merge_clusters_ncut(
    centroids: np.ndarray,
    n_target: int = 27,
) -> np.ndarray:
    """Merge overclusters to target K via recursive NCut on centroid graph.

    Args:
        centroids: Cluster centroids (K, C).
        n_target: Target number of final clusters.

    Returns:
        Merge mapping: array of length K, maps old cluster ID to new ID [0, n_target).
    """
    k = centroids.shape[0]
    if k <= n_target:
        return np.arange(k)

    # Build centroid affinity
    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8
    centroids_norm = centroids / norms
    sim = centroids_norm @ centroids_norm.T
    sim = np.clip(sim, 0.0, None)
    np.fill_diagonal(sim, 0.0)

    # Agglomerative clustering on centroid similarity
    clustering = AgglomerativeClustering(
        n_clusters=n_target,
        metric="cosine",
        linkage="average",
    )
    merge_labels = clustering.fit_predict(centroids_norm)
    return merge_labels


def merge_clusters_ward(
    centroids: np.ndarray,
    n_target: int = 27,
) -> np.ndarray:
    """Merge overclusters via Ward hierarchical clustering."""
    clustering = AgglomerativeClustering(
        n_clusters=n_target,
        linkage="ward",
    )
    merge_labels = clustering.fit_predict(centroids)
    return merge_labels


# ─── Main Pipeline ───

def main():
    parser = argparse.ArgumentParser(description="Spectral Enrichment + NCut Merge")
    parser.add_argument("--coco_root", required=True)
    parser.add_argument("--features_subdir", default="dinov3_features/val2017")
    parser.add_argument("--train_features_subdir", default=None,
                        help="Include train features for clustering (e.g., dinov3_features/train2017)")
    parser.add_argument("--patch_grid", type=int, default=32)
    parser.add_argument("--output_subdir", default=None,
                        help="Auto-generated if not specified")

    # Spectral parameters
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Feature vs color affinity weight (1.0=feature only)")
    parser.add_argument("--k_neighbors", type=int, default=10,
                        help="Top-k neighbors for sparse feature affinity")
    parser.add_argument("--n_eig", type=int, default=20,
                        help="Number of spectral eigenvector dimensions (0=skip)")
    parser.add_argument("--sigma_color", type=float, default=0.3)
    parser.add_argument("--sigma_spatial", type=float, default=5.0)

    # Clustering parameters
    parser.add_argument("--k", type=int, default=300,
                        help="Overclustering K for k-means")
    parser.add_argument("--merge_method", default="ncut",
                        choices=["ncut", "ward", "none"],
                        help="How to merge overclusters to 27")
    parser.add_argument("--n_target", type=int, default=NUM_CLASSES,
                        help="Target number of classes after merge")

    # Eval
    parser.add_argument("--eval_images", type=int, default=None,
                        help="Number of images for mIoU eval (None=all)")
    args = parser.parse_args()

    root = Path(args.coco_root)
    feat_dir = root / args.features_subdir
    pg = args.patch_grid
    n_patches = pg * pg

    # Auto-generate output subdir
    if args.output_subdir is None:
        args.output_subdir = (
            f"pseudo_semantic_spectral_a{args.alpha}_e{args.n_eig}"
            f"_k{args.k}_{args.merge_method}"
        )
    out_dir = root / args.output_subdir / "val2017"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ─── Step 1: Load features + compute spectral enrichment ───
    feat_files = sorted(feat_dir.glob("*.npy"))
    logger.info(f"Found {len(feat_files)} val feature files")

    # Optionally include train features
    train_feat_files = []
    if args.train_features_subdir:
        train_dir = root / args.train_features_subdir
        train_feat_files = sorted(train_dir.glob("*.npy"))
        logger.info(f"Found {len(train_feat_files)} train feature files")

    all_enriched = []
    val_image_ids = []
    val_count = 0

    print(f"\n{'='*60}")
    print(f"SPECTRAL PSEUDO-SEMANTIC LABELS")
    print(f"{'='*60}")
    print(f"  alpha={args.alpha}, n_eig={args.n_eig}, k_neighbors={args.k_neighbors}")
    print(f"  K={args.k}, merge={args.merge_method}, target={args.n_target}")
    print(f"  Features: {feat_dir}")
    print(f"  Output: {out_dir}")

    # Process val features
    print(f"\nEnriching val features with spectral eigenvectors...")
    for fp in tqdm(feat_files, desc="Spectral enrichment (val)"):
        feat = np.load(fp)  # (1024, 1024)
        image_id = int(fp.stem)
        val_image_ids.append(image_id)

        # L2-normalize features
        norms = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
        feat_norm = feat / norms

        if args.n_eig > 0:
            # Load RGB image for color affinity
            img_path = root / "val2017" / f"{image_id:012d}.jpg"
            if img_path.exists() and args.alpha < 1.0:
                image = np.array(Image.open(img_path).convert("RGB"))
            else:
                image = None

            enriched = enrich_features_spectral(
                feat_norm, image, pg,
                alpha=args.alpha, k_neighbors=args.k_neighbors,
                n_eigenvectors=args.n_eig,
                sigma_color=args.sigma_color, sigma_spatial=args.sigma_spatial,
            )
        else:
            enriched = feat_norm

        all_enriched.append(enriched)
        val_count += 1

    # Process train features (if provided, for clustering only)
    if train_feat_files:
        print(f"Loading train features (no spectral, for clustering mass)...")
        for fp in tqdm(train_feat_files, desc="Loading train features"):
            feat = np.load(fp)
            norms = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
            feat_norm = feat / norms

            if args.n_eig > 0:
                # Pad with zeros for spectral dims (train doesn't get spectral enrichment)
                pad = np.zeros((feat_norm.shape[0], args.n_eig), dtype=np.float32)
                enriched = np.concatenate([feat_norm, pad], axis=1)
            else:
                enriched = feat_norm
            all_enriched.append(enriched)

    all_features = np.concatenate(all_enriched, axis=0)
    print(f"Feature matrix: {all_features.shape} "
          f"({val_count} val + {len(train_feat_files)} train images)")

    # ─── Step 2: K-means overclustering ───
    print(f"\nRunning MiniBatchKMeans with K={args.k}...")
    kmeans = MiniBatchKMeans(
        n_clusters=args.k, batch_size=10000, max_iter=300,
        random_state=42, verbose=1,
    )
    labels = kmeans.fit_predict(all_features)
    centroids = kmeans.cluster_centers_

    # Extract val-only labels
    val_labels = labels[:val_count * n_patches].reshape(val_count, n_patches)
    print(f"Clustering done. Val labels shape: {val_labels.shape}")

    # ─── Step 3: Hierarchical merge ───
    if args.merge_method != "none" and args.k > args.n_target:
        print(f"\nMerging {args.k} clusters → {args.n_target} via {args.merge_method}...")
        if args.merge_method == "ncut":
            merge_map = merge_clusters_ncut(centroids, args.n_target)
        elif args.merge_method == "ward":
            merge_map = merge_clusters_ward(centroids, args.n_target)
        else:
            raise ValueError(f"Unknown merge method: {args.merge_method}")

        # Remap labels
        val_labels_merged = merge_map[val_labels]
        effective_k = args.n_target
        # Recompute centroids for merged clusters
        merged_centroids = np.zeros((args.n_target, centroids.shape[1]))
        for i in range(args.n_target):
            mask = merge_map == i
            if mask.any():
                merged_centroids[i] = centroids[mask].mean(axis=0)
        centroids = merged_centroids
    else:
        val_labels_merged = val_labels
        effective_k = args.k

    print(f"Effective clusters after merge: {effective_k}")

    # ─── Step 4: Hungarian matching against GT ───
    print(f"\nComputing Hungarian matching...")
    cost_matrix = np.zeros((effective_k, NUM_CLASSES), dtype=np.float64)

    for idx, img_id in tqdm(enumerate(val_image_ids), desc="Building cost matrix",
                            total=len(val_image_ids)):
        gt_sem = load_coco_panoptic_gt(args.coco_root, img_id)
        if gt_sem is None:
            continue

        gt_resized = np.array(Image.fromarray(gt_sem).resize(
            (pg, pg), Image.NEAREST
        ))
        gt_flat = gt_resized.flatten()
        pred_flat = val_labels_merged[idx]

        for p, g in zip(pred_flat, gt_flat):
            if g < NUM_CLASSES:
                cost_matrix[int(p), int(g)] += 1

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    cluster_to_class: Dict[int, int] = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > 0:
            cluster_to_class[r] = c

    for k_id in range(effective_k):
        if k_id not in cluster_to_class:
            best_class = int(np.argmax(cost_matrix[k_id]))
            cluster_to_class[k_id] = best_class

    # ─── Step 5: Save pseudo-labels ───
    print(f"\nSaving pseudo-semantic labels to {out_dir}...")
    for idx, img_id in tqdm(enumerate(val_image_ids), desc="Saving labels",
                            total=len(val_image_ids)):
        img_path = root / "val2017" / f"{img_id:012d}.jpg"
        if not img_path.exists():
            continue
        img = Image.open(img_path)
        w, h = img.size

        cluster_map = val_labels_merged[idx].reshape(pg, pg)
        class_map = np.vectorize(cluster_to_class.get)(cluster_map).astype(np.uint8)
        class_map_full = np.array(
            Image.fromarray(class_map).resize((w, h), Image.NEAREST)
        )

        out_path = out_dir / f"{img_id:012d}.png"
        Image.fromarray(class_map_full).save(out_path)

    # ─── Step 6: mIoU evaluation ───
    n_eval = args.eval_images or len(val_image_ids)
    print(f"\nComputing mIoU on {n_eval} images...")
    iou_per_class = np.zeros(NUM_CLASSES)
    count_per_class = np.zeros(NUM_CLASSES)

    for idx, img_id in enumerate(val_image_ids[:n_eval]):
        gt_sem = load_coco_panoptic_gt(args.coco_root, img_id)
        if gt_sem is None:
            continue
        pred_path = out_dir / f"{img_id:012d}.png"
        if not pred_path.exists():
            continue
        pred = np.array(Image.open(pred_path))

        if pred.shape != gt_sem.shape:
            pred = np.array(Image.fromarray(pred).resize(
                (gt_sem.shape[1], gt_sem.shape[0]), Image.NEAREST))

        for c in range(NUM_CLASSES):
            gt_mask = gt_sem == c
            pred_mask = pred == c
            inter = (gt_mask & pred_mask).sum()
            union = (gt_mask | pred_mask).sum()
            if union > 0:
                iou_per_class[c] += inter / union
                count_per_class[c] += 1

    valid = count_per_class > 0
    miou = (iou_per_class[valid] / count_per_class[valid]).mean() * 100

    elapsed = time.time() - t0

    # Per-class results
    print(f"\n{'='*60}")
    print(f"RESULTS: mIoU = {miou:.1f}%")
    print(f"{'='*60}")
    print(f"  Config: alpha={args.alpha}, n_eig={args.n_eig}, k={args.k}, "
          f"merge={args.merge_method}")
    print(f"  Time: {elapsed:.0f}s")

    things_iou, stuff_iou = [], []
    for c in range(NUM_CLASSES):
        if count_per_class[c] > 0:
            iou = iou_per_class[c] / count_per_class[c] * 100
            label = "T" if c in THING_IDS else "S"
            print(f"  [{label}] {COCOSTUFF27_CLASSNAMES[c]:20s}: IoU={iou:.1f}%")
            if c in THING_IDS:
                things_iou.append(iou)
            else:
                stuff_iou.append(iou)

    things_miou = np.mean(things_iou) if things_iou else 0.0
    stuff_miou = np.mean(stuff_iou) if stuff_iou else 0.0
    print(f"\n  Things mIoU: {things_miou:.1f}%  |  Stuff mIoU: {stuff_miou:.1f}%")

    # Save metadata
    meta = {
        "method": "spectral_enrichment",
        "alpha": args.alpha,
        "n_eig": args.n_eig,
        "k_neighbors": args.k_neighbors,
        "k": args.k,
        "merge_method": args.merge_method,
        "n_target": args.n_target,
        "sigma_color": args.sigma_color,
        "sigma_spatial": args.sigma_spatial,
        "miou": round(miou, 2),
        "things_miou": round(things_miou, 2),
        "stuff_miou": round(stuff_miou, 2),
        "n_val_images": val_count,
        "n_train_images": len(train_feat_files),
        "n_eval_images": n_eval,
        "elapsed_seconds": round(elapsed, 1),
        "cluster_to_class": {str(k): int(v) for k, v in cluster_to_class.items()},
    }
    meta_path = out_dir.parent / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved metadata to {meta_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
