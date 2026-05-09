"""MSDA evaluation: extract adapted features → k-means → PQ/mIoU.

Usage:
    python -m mbps_pytorch.msda.evaluate \
        --checkpoint checkpoints/msda/conv_hybrid/best.pt \
        --feature_dir /path/to/dinov3_features_vitl16 \
        --depth_dir /path/to/depth_depthpro \
        --gt_dir /path/to/cityscapes/gtFine
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader

from .architectures import AdapterConfig, create_adapter
from .dataset import CachedFeatureDataset

logger = logging.getLogger(__name__)

CITYSCAPES_19_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

CITYSCAPES_19_IDS = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

DEAD_CLASSES = {"wall", "fence", "traffic light", "rider", "truck", "train", "motorcycle"}


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, List[str]]:
    """Extract adapted features for all images."""
    model.eval()
    all_features = []
    all_stems = []

    for batch in loader:
        features = batch["features"].to(device)
        depth = batch["depth"].to(device)
        stems = batch["stem"]

        adapted = model(features, depth)  # (B, N, D)
        adapted_np = adapted.cpu().numpy()

        for i in range(adapted_np.shape[0]):
            all_features.append(adapted_np[i])
            all_stems.append(stems[i])

    return all_features, all_stems


def run_spherical_kmeans(
    features_list: List[np.ndarray],
    k: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Run spherical k-means on L2-normalized features."""
    stacked = np.concatenate(features_list, axis=0)  # (total_patches, D)
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    stacked = stacked / (norms + 1e-8)

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=4096,
        max_iter=300,
        n_init=3,
    )
    kmeans.fit(stacked)

    centroids = kmeans.cluster_centers_
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

    assignments = []
    offset = 0
    for feat in features_list:
        n = feat.shape[0]
        feat_norm = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
        sims = feat_norm @ centroids.T
        labels = sims.argmax(axis=1)
        assignments.append(labels)
        offset += n

    return centroids, assignments


def load_gt_semantic(gt_dir: Path, stem: str) -> Optional[np.ndarray]:
    """Load Cityscapes GT semantic label."""
    parts = stem.replace("_leftImg8bit", "").split("_")
    city = parts[0]
    gt_stem = "_".join(parts[:3])
    gt_path = gt_dir / "val" / city / f"{gt_stem}_gtFine_labelIds.png"
    if not gt_path.exists():
        return None
    from PIL import Image
    gt = np.array(Image.open(gt_path))
    return gt


def hungarian_mapping(
    assignments: List[np.ndarray],
    gt_labels: List[np.ndarray],
    num_clusters: int,
    num_classes: int = 19,
    spatial_h: int = 32,
    spatial_w: int = 64,
) -> np.ndarray:
    """Compute Hungarian mapping from clusters to GT classes."""
    cost_matrix = np.zeros((num_clusters, num_classes), dtype=np.int64)

    for pred, gt in zip(assignments, gt_labels):
        pred_2d = pred.reshape(spatial_h, spatial_w)
        from PIL import Image
        gt_resized = np.array(
            Image.fromarray(gt).resize((spatial_w, spatial_h), Image.NEAREST)
        )

        for c in range(num_clusters):
            mask = pred_2d == c
            if not mask.any():
                continue
            gt_in_cluster = gt_resized[mask]
            for cls_idx, label_id in enumerate(CITYSCAPES_19_IDS):
                cost_matrix[c, cls_idx] += (gt_in_cluster == label_id).sum()

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    mapping = np.full(num_clusters, -1, dtype=np.int32)
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > 0:
            mapping[r] = c
    return mapping


def compute_metrics(
    assignments: List[np.ndarray],
    gt_labels: List[np.ndarray],
    mapping: np.ndarray,
    spatial_h: int = 32,
    spatial_w: int = 64,
    num_classes: int = 19,
) -> Dict[str, float]:
    """Compute mIoU and per-class IoU."""
    intersection = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)

    for pred, gt in zip(assignments, gt_labels):
        pred_2d = pred.reshape(spatial_h, spatial_w)
        pred_mapped = mapping[pred_2d]

        from PIL import Image
        gt_resized = np.array(
            Image.fromarray(gt).resize((spatial_w, spatial_h), Image.NEAREST)
        )

        for cls_idx, label_id in enumerate(CITYSCAPES_19_IDS):
            gt_mask = gt_resized == label_id
            pred_mask = pred_mapped == cls_idx
            intersection[cls_idx] += (gt_mask & pred_mask).sum()
            union[cls_idx] += (gt_mask | pred_mask).sum()

    per_class_iou = np.zeros(num_classes)
    for c in range(num_classes):
        if union[c] > 0:
            per_class_iou[c] = intersection[c] / union[c]

    valid = union > 0
    miou = per_class_iou[valid].mean() if valid.any() else 0.0

    results = {
        "mIoU": float(miou * 100),
        "per_class_iou": {},
        "dead_class_recovery": {},
    }

    for idx, name in enumerate(CITYSCAPES_19_CLASSES):
        iou_val = float(per_class_iou[idx] * 100)
        results["per_class_iou"][name] = iou_val
        if name in DEAD_CLASSES:
            results["dead_class_recovery"][name] = iou_val

    recovered = sum(
        1 for v in results["dead_class_recovery"].values() if v > 5.0
    )
    results["dead_classes_recovered"] = recovered
    results["dead_classes_total"] = len(DEAD_CLASSES)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="MSDA Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    arch_name = ckpt["arch"]
    cfg_dict = ckpt["config"]
    cfg = AdapterConfig(**cfg_dict)

    model = create_adapter(arch_name, cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info(f"Loaded {arch_name} from {args.checkpoint}")

    val_ds = CachedFeatureDataset(
        args.feature_dir, args.depth_dir, split="val", augment=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    train_ds = CachedFeatureDataset(
        args.feature_dir, args.depth_dir, split="train", augment=False
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    logger.info("Extracting train features for k-means fitting...")
    train_features, _ = extract_features(model, train_loader, device)
    logger.info(f"Extracted {len(train_features)} train feature maps")

    logger.info("Extracting val features...")
    val_features, val_stems = extract_features(model, val_loader, device)
    logger.info(f"Extracted {len(val_features)} val feature maps")

    spatial_h = 32
    spatial_w = 64
    if val_features[0].shape[0] != 2048:
        side = int(np.sqrt(val_features[0].shape[0] * 2))
        spatial_w = side
        spatial_h = val_features[0].shape[0] // spatial_w
        logger.info(f"Non-standard resolution: {spatial_h}×{spatial_w}")

    logger.info(f"Running spherical k-means with k={args.k}...")
    centroids, train_assignments = run_spherical_kmeans(
        train_features, k=args.k
    )
    _, val_assignments = run_spherical_kmeans(
        val_features, k=args.k
    )
    val_centroids_sims = []
    for feat in val_features:
        feat_norm = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
        sims = feat_norm @ centroids.T
        val_centroids_sims.append(sims.argmax(axis=1))
    val_assignments = val_centroids_sims

    gt_dir = Path(args.gt_dir)
    gt_labels = []
    valid_val_assignments = []
    for stem, assignment in zip(val_stems, val_assignments):
        gt = load_gt_semantic(gt_dir, stem)
        if gt is not None:
            gt_labels.append(gt)
            valid_val_assignments.append(assignment)

    logger.info(f"Loaded {len(gt_labels)} GT labels for Hungarian mapping")

    logger.info("Computing Hungarian mapping...")
    mapping = hungarian_mapping(
        valid_val_assignments, gt_labels, args.k,
        spatial_h=spatial_h, spatial_w=spatial_w,
    )

    logger.info("Computing metrics...")
    results = compute_metrics(
        valid_val_assignments, gt_labels, mapping,
        spatial_h=spatial_h, spatial_w=spatial_w,
    )

    results["arch"] = arch_name
    results["checkpoint"] = args.checkpoint
    results["k"] = args.k

    logger.info(f"mIoU: {results['mIoU']:.2f}%")
    logger.info("Per-class IoU:")
    for name, iou in results["per_class_iou"].items():
        marker = " [DEAD]" if name in DEAD_CLASSES else ""
        recovered = " ✓ RECOVERED" if name in DEAD_CLASSES and iou > 5.0 else ""
        logger.info(f"  {name:20s}: {iou:6.2f}%{marker}{recovered}")
    logger.info(
        f"Dead classes recovered: {results['dead_classes_recovered']}/{results['dead_classes_total']}"
    )

    output_path = args.output or str(
        Path(args.checkpoint).parent / "eval_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
