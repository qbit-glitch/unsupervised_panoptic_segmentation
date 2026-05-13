#!/usr/bin/env python3
"""Approach 2: SAM Superpixel Consensus Clustering for COCO-Stuff-27.

Replace coarse 32×32 patches with SAM's boundary-aware segments, pool DINOv3
features per segment, then cluster segment embeddings globally.

Usage:
    # Phase A: Pre-compute SAM segments
    python mbps_pytorch/sam_consensus_pseudo_semantics.py \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --phase sam --sam_checkpoint weights/sam_vit_b_01ec64.pth \
        --sam_model_type vit_b --device mps

    # Phase B+C: Cluster and evaluate
    python mbps_pytorch/sam_consensus_pseudo_semantics.py \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --phase cluster --k 300
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ─── COCO-Stuff-27 class definitions ───

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


# ─── Phase A: SAM Segment Extraction ───

def extract_sam_segments(
    coco_root: str,
    sam_checkpoint: str,
    sam_model_type: str = "vit_b",
    device: str = "mps",
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.86,
    stability_score_thresh: float = 0.92,
    min_mask_area: int = 100,
) -> None:
    """Pre-compute SAM segments for all val images."""
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    root = Path(coco_root)
    img_dir = root / "val2017"
    feat_dir = root / "dinov3_features" / "val2017"
    out_dir = root / "sam_segments" / "val2017"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only process images that have DINOv3 features
    feat_files = sorted(feat_dir.glob("*.npy"))
    image_ids = [int(f.stem) for f in feat_files]
    print(f"Processing {len(image_ids)} images with SAM {sam_model_type}")

    # Load SAM
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam = sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_area,
    )

    for img_id in tqdm(image_ids, desc="SAM segmentation"):
        out_path = out_dir / f"{img_id:012d}.npz"
        if out_path.exists():
            continue

        img_path = img_dir / f"{img_id:012d}.jpg"
        if not img_path.exists():
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        masks = mask_generator.generate(image)

        # Sort by area (largest first) and extract masks
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)

        # Store as compressed array of masks + metadata
        mask_arrays = []
        areas = []
        iou_scores = []
        stability_scores = []
        for m in masks:
            mask_arrays.append(m["segmentation"].astype(np.bool_))
            areas.append(m["area"])
            iou_scores.append(m["predicted_iou"])
            stability_scores.append(m["stability_score"])

        if mask_arrays:
            np.savez_compressed(
                str(out_path),
                masks=np.stack(mask_arrays),  # (N, H, W) bool
                areas=np.array(areas),
                iou_scores=np.array(iou_scores),
                stability_scores=np.array(stability_scores),
            )
        else:
            # Save empty
            np.savez_compressed(str(out_path), masks=np.zeros((0,) + image.shape[:2], dtype=bool))

    print(f"SAM segments saved to {out_dir}")


# ─── Phase B+C: Feature Pooling + Clustering ───

def pool_features_per_segment(
    features: np.ndarray,
    masks: np.ndarray,
    patch_grid: int = 32,
) -> np.ndarray:
    """Pool DINOv3 features per SAM segment.

    Args:
        features: DINOv3 features (N_patches, C) for one image.
        masks: SAM masks (N_segments, H, W) bool.
        patch_grid: Feature grid size.

    Returns:
        Segment embeddings (N_segments, C).
    """
    c = features.shape[1]
    feat_grid = features.reshape(patch_grid, patch_grid, c)

    segment_embeddings = []
    for mask in masks:
        # Downsample mask to patch grid
        mask_small = np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (patch_grid, patch_grid), Image.NEAREST
            )
        ) > 127

        if mask_small.sum() == 0:
            # Mask is too small at patch resolution — use center pixel
            h, w = mask.shape
            cy, cx = np.where(mask)
            if len(cy) > 0:
                py = int(cy.mean() * patch_grid / h)
                px = int(cx.mean() * patch_grid / w)
                py = min(py, patch_grid - 1)
                px = min(px, patch_grid - 1)
                segment_embeddings.append(feat_grid[py, px])
            else:
                segment_embeddings.append(np.zeros(c, dtype=np.float32))
            continue

        # Mean pool features within mask
        pooled = feat_grid[mask_small].mean(axis=0)
        segment_embeddings.append(pooled)

    return np.stack(segment_embeddings) if segment_embeddings else np.zeros((0, c))


def assign_segments_to_pixels(
    masks: np.ndarray,
    segment_labels: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Map segment cluster labels back to pixels.

    Overlapping segments: largest (first) wins. Uncovered pixels: nearest neighbor.

    Args:
        masks: SAM masks (N_segments, H, W) bool.
        segment_labels: Cluster label per segment (N_segments,).
        image_shape: (H, W) of the output.

    Returns:
        Pixel-level label map (H, W).
    """
    from scipy.ndimage import distance_transform_edt

    h, w = image_shape
    label_map = np.full((h, w), -1, dtype=np.int32)

    # Assign from largest to smallest (masks already sorted by area desc)
    for i, (mask, label) in enumerate(zip(masks, segment_labels)):
        if mask.shape != (h, w):
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
            ) > 127
        uncovered = label_map == -1
        label_map[uncovered & mask] = int(label)

    # Handle uncovered pixels via nearest-neighbor
    uncovered = label_map == -1
    if uncovered.any() and (~uncovered).any():
        covered_mask = ~uncovered
        _, nearest_idx = distance_transform_edt(uncovered, return_indices=True)
        label_map[uncovered] = label_map[nearest_idx[0][uncovered], nearest_idx[1][uncovered]]

    # Any still uncovered (shouldn't happen): assign 0
    label_map[label_map == -1] = 0

    return label_map


def run_clustering_phase(args: argparse.Namespace) -> None:
    """Run feature pooling, clustering, and evaluation."""
    root = Path(args.coco_root)
    feat_dir = root / "dinov3_features" / "val2017"
    sam_dir = root / "sam_segments" / "val2017"
    pg = args.patch_grid

    if args.output_subdir is None:
        args.output_subdir = f"pseudo_semantic_sam_k{args.k}"
    out_dir = root / args.output_subdir / "val2017"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ─── Load features and SAM segments ───
    feat_files = sorted(feat_dir.glob("*.npy"))
    image_ids = []
    all_segment_embeddings = []
    segment_counts = []  # How many segments per image

    print(f"\n{'='*60}")
    print(f"SAM CONSENSUS CLUSTERING")
    print(f"{'='*60}")
    print(f"  K={args.k}, patch_grid={pg}")
    print(f"  Features: {feat_dir}")
    print(f"  SAM segments: {sam_dir}")
    print(f"  Output: {out_dir}")

    print(f"\nPooling DINOv3 features per SAM segment...")
    for fp in tqdm(feat_files, desc="Feature pooling"):
        image_id = int(fp.stem)
        sam_path = sam_dir / f"{image_id:012d}.npz"
        if not sam_path.exists():
            continue

        feat = np.load(fp)  # (1024, 1024)
        # L2-normalize
        norms = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
        feat_norm = feat / norms

        data = np.load(sam_path)
        masks = data["masks"]  # (N, H, W) bool

        if len(masks) == 0:
            continue

        seg_embed = pool_features_per_segment(feat_norm, masks, pg)
        if seg_embed.shape[0] == 0:
            continue

        # L2-normalize segment embeddings
        seg_norms = np.linalg.norm(seg_embed, axis=1, keepdims=True) + 1e-8
        seg_embed = seg_embed / seg_norms

        all_segment_embeddings.append(seg_embed)
        segment_counts.append(seg_embed.shape[0])
        image_ids.append(image_id)

    all_embeddings = np.concatenate(all_segment_embeddings, axis=0)
    total_segments = all_embeddings.shape[0]
    print(f"Total segments: {total_segments} across {len(image_ids)} images "
          f"(avg {total_segments/len(image_ids):.0f}/image)")

    # ─── K-means clustering ───
    print(f"\nRunning MiniBatchKMeans with K={args.k}...")
    kmeans = MiniBatchKMeans(
        n_clusters=args.k, batch_size=min(10000, total_segments),
        max_iter=300, random_state=42, verbose=1,
    )
    segment_cluster_labels = kmeans.fit_predict(all_embeddings)

    # ─── Hungarian matching against GT ───
    print(f"\nComputing Hungarian matching...")
    cost_matrix = np.zeros((args.k, NUM_CLASSES), dtype=np.float64)

    seg_offset = 0
    for idx, img_id in tqdm(enumerate(image_ids), desc="Building cost matrix",
                            total=len(image_ids)):
        gt_sem = load_coco_panoptic_gt(args.coco_root, img_id)
        if gt_sem is None:
            seg_offset += segment_counts[idx]
            continue

        n_segs = segment_counts[idx]
        seg_labels = segment_cluster_labels[seg_offset:seg_offset + n_segs]
        seg_offset += n_segs

        # Load SAM masks
        sam_path = sam_dir / f"{img_id:012d}.npz"
        masks = np.load(sam_path)["masks"]

        # For each segment, find majority GT class
        for seg_idx, (mask, cluster_id) in enumerate(zip(masks[:n_segs], seg_labels)):
            if mask.shape != gt_sem.shape:
                mask_resized = np.array(
                    Image.fromarray(mask.astype(np.uint8) * 255).resize(
                        (gt_sem.shape[1], gt_sem.shape[0]), Image.NEAREST
                    )
                ) > 127
            else:
                mask_resized = mask

            gt_in_seg = gt_sem[mask_resized]
            gt_valid = gt_in_seg[gt_in_seg < NUM_CLASSES]
            if len(gt_valid) > 0:
                # Weight by pixel count
                for g in gt_valid:
                    cost_matrix[int(cluster_id), int(g)] += 1

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    cluster_to_class: Dict[int, int] = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > 0:
            cluster_to_class[r] = c
    for k_id in range(args.k):
        if k_id not in cluster_to_class:
            cluster_to_class[k_id] = int(np.argmax(cost_matrix[k_id]))

    # ─── Save pseudo-labels ───
    print(f"\nSaving pseudo-semantic labels...")
    seg_offset = 0
    for idx, img_id in tqdm(enumerate(image_ids), desc="Saving labels",
                            total=len(image_ids)):
        img_path = root / "val2017" / f"{img_id:012d}.jpg"
        if not img_path.exists():
            seg_offset += segment_counts[idx]
            continue
        img = Image.open(img_path)
        w, h = img.size

        n_segs = segment_counts[idx]
        seg_labels = segment_cluster_labels[seg_offset:seg_offset + n_segs]
        seg_offset += n_segs

        # Map cluster → class
        class_labels = np.array([cluster_to_class.get(int(l), 0) for l in seg_labels])

        # Load SAM masks and assign to pixels
        sam_path = sam_dir / f"{img_id:012d}.npz"
        masks = np.load(sam_path)["masks"][:n_segs]

        pixel_labels = assign_segments_to_pixels(masks, class_labels, (h, w))
        pixel_labels = pixel_labels.astype(np.uint8)

        out_path = out_dir / f"{img_id:012d}.png"
        Image.fromarray(pixel_labels).save(out_path)

    # ─── mIoU evaluation ───
    n_eval = args.eval_images or len(image_ids)
    print(f"\nComputing mIoU on {n_eval} images...")
    iou_per_class = np.zeros(NUM_CLASSES)
    count_per_class = np.zeros(NUM_CLASSES)

    for idx, img_id in enumerate(image_ids[:n_eval]):
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

    print(f"\n{'='*60}")
    print(f"RESULTS: mIoU = {miou:.1f}%")
    print(f"{'='*60}")
    print(f"  Config: K={args.k}, SAM segments")
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
        "method": "sam_consensus",
        "k": args.k,
        "miou": round(miou, 2),
        "things_miou": round(things_miou, 2),
        "stuff_miou": round(stuff_miou, 2),
        "total_segments": int(total_segments),
        "n_images": len(image_ids),
        "avg_segments_per_image": round(total_segments / max(len(image_ids), 1), 1),
        "n_eval_images": n_eval,
        "elapsed_seconds": round(elapsed, 1),
        "cluster_to_class": {str(k): int(v) for k, v in cluster_to_class.items()},
    }
    meta_path = out_dir.parent / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved metadata to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="SAM Consensus Pseudo-Semantics")
    parser.add_argument("--coco_root", required=True)
    parser.add_argument("--phase", required=True, choices=["sam", "cluster", "all"],
                        help="Phase to run: sam (extract segments), cluster (pool+cluster+eval), all")

    # SAM parameters
    parser.add_argument("--sam_checkpoint", default=None,
                        help="Path to SAM checkpoint (required for phase=sam)")
    parser.add_argument("--sam_model_type", default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.86)
    parser.add_argument("--stability_score_thresh", type=float, default=0.92)
    parser.add_argument("--min_mask_area", type=int, default=100)

    # Clustering parameters
    parser.add_argument("--k", type=int, default=300)
    parser.add_argument("--patch_grid", type=int, default=32)
    parser.add_argument("--output_subdir", default=None)
    parser.add_argument("--eval_images", type=int, default=None)

    args = parser.parse_args()

    if args.phase in ("sam", "all"):
        if args.sam_checkpoint is None:
            parser.error("--sam_checkpoint required for phase=sam")
        extract_sam_segments(
            args.coco_root, args.sam_checkpoint, args.sam_model_type,
            args.device, args.points_per_side, args.pred_iou_thresh,
            args.stability_score_thresh, args.min_mask_area,
        )

    if args.phase in ("cluster", "all"):
        run_clustering_phase(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
