#!/usr/bin/env python3
"""Generate semantic pseudo-labels via K-means on DINOv3 features.

Pipeline:
  1. Load DINOv3 patch features (from extract_dinov3_features.py)
  2. Subsample features for K-means fitting (memory-efficient)
  3. Run K-means clustering (K=19 for Cityscapes, K=27 for COCO)
  4. Assign cluster labels to all images
  5. Upsample from patch resolution to pixel resolution
  6. (Optional) CRF post-processing for spatial coherence
  7. (Optional) Hungarian matching against GT for evaluation

Usage:
    python mbps_pytorch/generate_semantic_pseudolabels.py \
        --feature_dir /data/cityscapes/dinov3_features/train \
        --output_dir /data/cityscapes/pseudo_semantic/train \
        --num_classes 19 \
        --image_size 512 1024

    # Evaluate on val set:
    python mbps_pytorch/generate_semantic_pseudolabels.py \
        --feature_dir /data/cityscapes/dinov3_features/val \
        --output_dir /data/cityscapes/pseudo_semantic/val \
        --num_classes 19 \
        --image_size 512 1024 \
        --kmeans_model /data/cityscapes/pseudo_semantic/train/kmeans_model.pkl \
        --gt_dir /data/cityscapes/gtFine/val \
        --evaluate
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def load_features_for_kmeans(
    feature_dir: str,
    subsample_ratio: float = 0.1,
    max_samples: int = 500_000,
) -> np.ndarray:
    """Load and subsample features for K-means fitting.

    Args:
        feature_dir: Directory with per-image .npy feature files.
        subsample_ratio: Fraction of patches to sample per image.
        max_samples: Maximum total samples for K-means.

    Returns:
        (N_samples, 768) float32 array.
    """
    feature_dir = Path(feature_dir)
    feature_files = sorted(feature_dir.rglob("*.npy"))
    print(f"Found {len(feature_files)} feature files")

    all_features = []
    total_patches = 0

    for fpath in tqdm(feature_files, desc="Loading features"):
        feats = np.load(str(fpath))  # (N_patches, 768)
        if feats.dtype == np.float16:
            feats = feats.astype(np.float32)
        n = feats.shape[0]
        total_patches += n

        # Subsample
        k = max(1, int(n * subsample_ratio))
        indices = np.random.choice(n, k, replace=False)
        all_features.append(feats[indices])

    features = np.concatenate(all_features, axis=0)
    print(f"Total patches: {total_patches}, sampled: {features.shape[0]}")

    # Further subsample if too many
    if features.shape[0] > max_samples:
        indices = np.random.choice(features.shape[0], max_samples, replace=False)
        features = features[indices]
        print(f"Further subsampled to {max_samples}")

    # L2-normalize features before K-Means
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / (norms + 1e-8)
    print(f"L2-normalized features (mean norm before: {norms.mean():.2f})")

    return features


def fit_kmeans(
    features: np.ndarray,
    num_classes: int,
    n_init: int = 10,
    max_iter: int = 300,
) -> MiniBatchKMeans:
    """Fit MiniBatchKMeans on subsampled features.

    Args:
        features: (N, 768) float32 array.
        num_classes: Number of clusters (K).
        n_init: Number of initializations.
        max_iter: Max iterations per init.

    Returns:
        Fitted MiniBatchKMeans model.
    """
    print(f"Fitting K-means with K={num_classes} on {features.shape[0]} samples...")
    kmeans = MiniBatchKMeans(
        n_clusters=num_classes,
        n_init=n_init,
        max_iter=max_iter,
        batch_size=min(10000, features.shape[0]),
        random_state=42,
        verbose=1,
    )
    kmeans.fit(features)
    print(f"K-means inertia: {kmeans.inertia_:.2f}")
    return kmeans


def assign_labels(
    feature_dir: str,
    output_dir: str,
    kmeans: MiniBatchKMeans,
    h_patches: int,
    w_patches: int,
    image_size: tuple,
    use_crf: bool = False,
    image_dir: str = None,
):
    """Assign cluster labels to all images and save as PNG.

    Args:
        feature_dir: Directory with per-image .npy feature files.
        output_dir: Where to save label PNGs.
        kmeans: Fitted K-means model.
        h_patches: Number of patches in height.
        w_patches: Number of patches in width.
        image_size: (H, W) target pixel resolution.
        use_crf: Whether to apply CRF post-processing.
        image_dir: Path to original images (needed for CRF).
    """
    feature_dir = Path(feature_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    feature_files = sorted(feature_dir.rglob("*.npy"))

    for fpath in tqdm(feature_files, desc="Assigning labels"):
        # Output path mirrors input structure
        rel_path = fpath.relative_to(feature_dir)
        out_path = output_dir / rel_path.with_suffix(".png")
        if out_path.exists():
            continue

        # Load features, L2-normalize, and predict
        feats = np.load(str(fpath))  # (N_patches, 768)
        if feats.dtype == np.float16:
            feats = feats.astype(np.float32)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        labels = kmeans.predict(feats)  # (N_patches,)

        # Reshape to spatial grid
        label_map = labels.reshape(h_patches, w_patches)  # (32, 64)

        # Upsample to pixel resolution using nearest neighbor
        label_img = Image.fromarray(label_map.astype(np.uint8), mode="L")
        label_img = label_img.resize(
            (image_size[1], image_size[0]),
            Image.NEAREST,
        )

        if use_crf and image_dir is not None:
            label_img = _apply_crf(label_img, fpath, feature_dir, image_dir,
                                   image_size, kmeans.n_clusters)

        # Save
        out_path.parent.mkdir(parents=True, exist_ok=True)
        label_img.save(str(out_path))


def _apply_crf(
    label_img: Image.Image,
    feature_path: Path,
    feature_dir: Path,
    image_dir: str,
    image_size: tuple,
    num_classes: int,
) -> Image.Image:
    """Apply dense CRF post-processing for spatial coherence.

    Supports SimpleCRF (pip install simplecrf) or pydensecrf as fallback.
    """
    # Find corresponding original image
    rel_path = feature_path.relative_to(feature_dir)
    # Feature files already have _leftImg8bit in name, just swap extension
    img_path = Path(image_dir) / rel_path.with_suffix(".png")
    if not img_path.exists():
        # Try adding _leftImg8bit suffix
        img_name = str(rel_path).replace(".npy", "_leftImg8bit.png")
        img_path = Path(image_dir) / img_name
    if not img_path.exists():
        return label_img

    orig_img = np.array(
        Image.open(img_path).convert("RGB").resize(
            (image_size[1], image_size[0]), Image.BILINEAR
        )
    )

    labels = np.array(label_img).astype(np.int32)
    H, W = labels.shape

    # Build unary potentials from labels
    prob = np.full((num_classes, H, W), 0.3 / (num_classes - 1), dtype=np.float32)
    for c in range(num_classes):
        prob[c][labels == c] = 0.7
    unary = -np.log(np.clip(prob, 1e-10, 1.0))  # (C, H, W)

    # Try SimpleCRF first (correct API: densecrf(image, prob_map, params_tuple))
    try:
        from denseCRF import densecrf
        img_uint8 = orig_img.astype(np.uint8).copy()
        # param = (w1_bilateral, alpha_spatial, beta_rgb, w2_gaussian, gamma_spatial, iterations)
        param = (10.0, 80.0, 13.0, 3.0, 3.0, 10.0)
        refined = densecrf(img_uint8, prob.transpose(1, 2, 0).copy(), param)
        refined = refined.astype(np.uint8)
        return Image.fromarray(refined, mode="L")
    except ImportError:
        pass
    except Exception as e:
        print(f"WARNING: SimpleCRF failed: {e}")

    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_labels
        d = dcrf.DenseCRF2D(W, H, num_classes)
        u = unary_from_labels(labels, num_classes, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(u)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=orig_img, compat=10)
        d.addPairwiseGaussian(sxy=3, compat=3)
        Q = d.inference(10)
        refined = np.argmax(Q, axis=0).reshape(H, W).astype(np.uint8)
        return Image.fromarray(refined, mode="L")
    except ImportError:
        print("WARNING: No CRF library found, skipping.")
        return label_img


def hungarian_match(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    num_pred_classes: int,
    num_gt_classes: int,
    ignore_label: int = 255,
) -> dict:
    """Compute optimal cluster-to-class mapping via Hungarian matching.

    Args:
        pred_labels: (N,) predicted cluster IDs.
        gt_labels: (N,) ground truth class IDs.
        num_pred_classes: Number of predicted clusters.
        num_gt_classes: Number of GT classes.
        ignore_label: GT label to ignore.

    Returns:
        Dict mapping cluster_id -> gt_class_id.
    """
    # Build cost matrix
    valid = gt_labels != ignore_label
    pred_valid = pred_labels[valid]
    gt_valid = gt_labels[valid]

    # Cost matrix: (num_pred, num_gt) = -count(pred==i AND gt==j)
    cost = np.zeros((num_pred_classes, num_gt_classes), dtype=np.int64)
    for p, g in zip(pred_valid, gt_valid):
        if p < num_pred_classes and g < num_gt_classes:
            cost[p, g] += 1

    # Hungarian algorithm (maximize overlap = minimize negative overlap)
    row_ind, col_ind = linear_sum_assignment(-cost)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[int(r)] = int(c)

    return mapping


def evaluate_pseudolabels(
    output_dir: str,
    gt_dir: str,
    num_classes: int,
    ignore_label: int = 255,
):
    """Evaluate pseudo-labels against ground truth using mIoU.

    Args:
        output_dir: Directory with pseudo-label PNGs.
        gt_dir: Directory with GT label PNGs (Cityscapes gtFine).
        num_classes: Number of classes.
        ignore_label: GT label to ignore.
    """
    output_dir = Path(output_dir)
    gt_dir = Path(gt_dir)

    # Collect all predictions and GT for Hungarian matching
    all_pred = []
    all_gt = []

    pred_files = sorted(output_dir.rglob("*.png"))
    print(f"Evaluating {len(pred_files)} pseudo-label images...")

    for pred_path in tqdm(pred_files, desc="Loading for evaluation"):
        pred = np.array(Image.open(pred_path))

        # Find corresponding GT — handle Cityscapes naming conventions
        rel = pred_path.relative_to(output_dir)
        base = str(rel).replace("_leftImg8bit.png", "").replace(".png", "")
        gt_path = None
        for suffix in [
            "_gtFine_labelTrainIds.png",
            "_gtFine_labelIds.png",
            "_labelTrainIds.png",
            "_labelIds.png",
        ]:
            candidate = gt_dir / (base + suffix)
            if candidate.exists():
                gt_path = candidate
                break
        if gt_path is None:
            continue

        gt = np.array(Image.open(gt_path))

        # Remap raw Cityscapes labelIds to train IDs (0-18) if needed
        if "_labelIds.png" in str(gt_path) and "TrainIds" not in str(gt_path):
            _CS_ID_TO_TRAIN = {
                7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
                21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                28: 15, 31: 16, 32: 17, 33: 18,
            }
            remapped = np.full_like(gt, ignore_label)
            for raw_id, train_id in _CS_ID_TO_TRAIN.items():
                remapped[gt == raw_id] = train_id
            gt = remapped

        # Resize pred to match GT if needed
        if pred.shape != gt.shape:
            pred = np.array(
                Image.fromarray(pred).resize(
                    (gt.shape[1], gt.shape[0]), Image.NEAREST
                )
            )

        all_pred.append(pred.flatten())
        all_gt.append(gt.flatten())

    if not all_pred:
        print("ERROR: No matching GT files found!")
        return

    all_pred = np.concatenate(all_pred)
    all_gt = np.concatenate(all_gt)

    # Hungarian matching
    print("Running Hungarian matching...")
    mapping = hungarian_match(all_pred, all_gt, num_classes, num_classes, ignore_label)
    print(f"Cluster-to-class mapping: {mapping}")

    # Apply mapping and compute mIoU
    mapped_pred = np.full_like(all_pred, ignore_label)
    for cluster_id, class_id in mapping.items():
        mapped_pred[all_pred == cluster_id] = class_id

    # Per-class IoU
    ious = []
    for c in range(num_classes):
        pred_c = mapped_pred == c
        gt_c = all_gt == c
        valid = all_gt != ignore_label

        intersection = np.sum(pred_c & gt_c & valid)
        union = np.sum((pred_c | gt_c) & valid)

        if union > 0:
            iou = intersection / union
            ious.append(iou)
            print(f"  Class {c:2d}: IoU = {iou:.4f}")
        else:
            print(f"  Class {c:2d}: IoU = N/A (no pixels)")

    miou = np.mean(ious) if ious else 0.0
    print(f"\nmIoU: {miou:.4f} ({miou * 100:.2f}%)")
    print(f"Evaluated on {len(ious)}/{num_classes} classes")

    # Save results
    results = {
        "miou": float(miou),
        "per_class_iou": {str(i): float(iou) for i, iou in enumerate(ious)},
        "cluster_to_class_mapping": {str(k): v for k, v in mapping.items()},
    }
    results_path = Path(output_dir) / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate semantic pseudo-labels via K-means on DINOv3 features"
    )
    parser.add_argument(
        "--feature_dir", type=str, required=True,
        help="Directory with DINOv3 .npy feature files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Where to save pseudo-label PNGs"
    )
    parser.add_argument(
        "--num_classes", type=int, default=19,
        help="Number of clusters (19 for Cityscapes, 27 for COCO)"
    )
    parser.add_argument(
        "--image_size", type=int, nargs=2, default=[512, 1024],
        help="(H, W) pixel resolution for output labels"
    )
    parser.add_argument(
        "--subsample_ratio", type=float, default=0.1,
        help="Fraction of patches to sample for K-means fitting"
    )
    parser.add_argument(
        "--max_samples", type=int, default=500000,
        help="Max samples for K-means"
    )
    parser.add_argument(
        "--kmeans_model", type=str, default=None,
        help="Path to pre-fitted K-means model (.pkl). If not provided, fits a new one."
    )
    parser.add_argument(
        "--use_crf", action="store_true",
        help="Apply CRF post-processing (requires pydensecrf)"
    )
    parser.add_argument(
        "--image_dir", type=str, default=None,
        help="Path to original images (needed for CRF)"
    )
    parser.add_argument(
        "--gt_dir", type=str, default=None,
        help="Path to GT labels (for evaluation)"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate pseudo-labels against GT"
    )
    args = parser.parse_args()

    # Load metadata
    metadata_path = os.path.join(args.feature_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        h_patches = metadata["h_patches"]
        w_patches = metadata["w_patches"]
        print(f"Loaded metadata: {h_patches}x{w_patches} patches, "
              f"dim={metadata['hidden_size']}")
    else:
        # Infer from image_size and patch_size=16
        h_patches = args.image_size[0] // 16
        w_patches = args.image_size[1] // 16
        print(f"No metadata found, inferring: {h_patches}x{w_patches} patches")

    # Fit or load K-means
    if args.kmeans_model and os.path.exists(args.kmeans_model):
        print(f"Loading K-means model from {args.kmeans_model}")
        with open(args.kmeans_model, "rb") as f:
            kmeans = pickle.load(f)
    else:
        features = load_features_for_kmeans(
            args.feature_dir,
            subsample_ratio=args.subsample_ratio,
            max_samples=args.max_samples,
        )
        kmeans = fit_kmeans(features, args.num_classes)

        # Save K-means model
        model_path = os.path.join(args.output_dir, "kmeans_model.pkl")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(kmeans, f)
        print(f"Saved K-means model to {model_path}")

    # Assign labels to all images
    assign_labels(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        kmeans=kmeans,
        h_patches=h_patches,
        w_patches=w_patches,
        image_size=tuple(args.image_size),
        use_crf=args.use_crf,
        image_dir=args.image_dir,
    )

    # Evaluate if requested
    if args.evaluate and args.gt_dir:
        evaluate_pseudolabels(
            output_dir=args.output_dir,
            gt_dir=args.gt_dir,
            num_classes=args.num_classes,
        )


if __name__ == "__main__":
    main()
