#!/usr/bin/env python3
"""Generate pseudo-semantic labels for COCO val from DINOv3 features via k-means.

Fully unsupervised clustering; Hungarian matching against GT only for evaluation
(maps cluster IDs to 27 COCO-Stuff supercategory classes).

Usage:
    python mbps_pytorch/generate_coco_pseudo_semantics.py \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --k 80 --output_subdir pseudo_semantic_k80
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


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

# Mapping from COCO panoptic supercategory name -> coarse class index
SUPERCATEGORY_TO_COARSE = {
    "electronic": 0, "appliance": 1, "food": 2, "furniture": 3,
    "indoor": 4, "kitchen": 5, "accessory": 6, "animal": 7,
    "outdoor": 8, "person": 9, "sports": 10, "vehicle": 11,
    "ceiling": 12, "floor": 13, "food-stuff": 14, "furniture-stuff": 15,
    "raw-material": 16, "textile": 17, "wall": 18, "window": 19,
    "building": 20, "ground": 21, "plant": 22, "sky": 23,
    "solid": 24, "structural": 25, "water": 26,
}


def load_coco_panoptic_gt(coco_root, image_id):
    """Load COCO panoptic GT and convert to 27-class semantic label map."""
    panoptic_json = os.path.join(coco_root, "annotations", "panoptic_val2017.json")
    panoptic_dir = os.path.join(coco_root, "annotations", "panoptic_val2017")

    if not hasattr(load_coco_panoptic_gt, "_cache"):
        with open(panoptic_json) as f:
            data = json.load(f)

        # Build category lookup
        cat_map = {}
        for cat in data["categories"]:
            cat_map[cat["id"]] = cat["supercategory"]

        # Build per-image annotation lookup
        ann_map = {}
        for ann in data["annotations"]:
            ann_map[ann["image_id"]] = ann

        load_coco_panoptic_gt._cache = (cat_map, ann_map, panoptic_dir)

    cat_map, ann_map, pdir = load_coco_panoptic_gt._cache

    if image_id not in ann_map:
        return None

    ann = ann_map[image_id]
    pan_path = os.path.join(pdir, ann["file_name"])
    pan_img = np.array(Image.open(pan_path))
    pan_id = pan_img[:, :, 0].astype(np.int32) + \
             pan_img[:, :, 1].astype(np.int32) * 256 + \
             pan_img[:, :, 2].astype(np.int32) * 256 * 256

    sem_label = np.full(pan_id.shape, 255, dtype=np.uint8)  # void

    for seg in ann["segments_info"]:
        mask = pan_id == seg["id"]
        cat_id = seg["category_id"]
        supercat = cat_map.get(cat_id, None)
        if supercat and supercat in SUPERCATEGORY_TO_COARSE:
            sem_label[mask] = SUPERCATEGORY_TO_COARSE[supercat]

    return sem_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", required=True)
    parser.add_argument("--features_subdir", default="dinov3_features/val2017")
    parser.add_argument("--k", type=int, default=80, help="Number of clusters")
    parser.add_argument("--output_subdir", default="pseudo_semantic_k80/val2017")
    parser.add_argument("--patch_grid", type=int, default=32,
                        help="Patch grid size (sqrt of num patches)")
    parser.add_argument("--l2_normalize", action="store_true",
                        help="L2-normalize features before clustering")
    args = parser.parse_args()

    root = Path(args.coco_root)
    feat_dir = root / args.features_subdir
    out_dir = root / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Load all features ───
    feat_files = sorted(feat_dir.glob("*.npy"))
    print(f"Found {len(feat_files)} feature files in {feat_dir}")

    all_features = []
    image_ids = []
    for fp in tqdm(feat_files, desc="Loading features"):
        feat = np.load(fp)  # (1024, 1024) = (num_patches, feat_dim)
        all_features.append(feat)
        image_ids.append(int(fp.stem))

    all_features = np.concatenate(all_features, axis=0)  # (N*1024, 1024)
    print(f"Feature matrix: {all_features.shape}")

    if args.l2_normalize:
        print("L2-normalizing features...")
        norms = np.linalg.norm(all_features, axis=-1, keepdims=True) + 1e-8
        all_features = all_features / norms
        print(f"  Norm range after: [{np.linalg.norm(all_features[0]):.3f}]")

    # ─── Step 2: K-means clustering ───
    print(f"Running MiniBatchKMeans with k={args.k}...")
    kmeans = MiniBatchKMeans(
        n_clusters=args.k,
        batch_size=10000,
        max_iter=300,
        random_state=42,
        verbose=1,
    )
    labels = kmeans.fit_predict(all_features)
    print(f"Clustering done. Labels shape: {labels.shape}")

    # Reshape to per-image
    n_patches = args.patch_grid * args.patch_grid
    labels_per_image = labels.reshape(len(image_ids), n_patches)

    # ─── Step 3: Hungarian matching against GT ───
    print("Computing cost matrix for Hungarian matching...")
    cost_matrix = np.zeros((args.k, NUM_CLASSES), dtype=np.float64)

    for idx, img_id in tqdm(enumerate(image_ids), desc="Building cost matrix",
                            total=len(image_ids)):
        gt_sem = load_coco_panoptic_gt(args.coco_root, img_id)
        if gt_sem is None:
            continue

        # Resize GT to patch grid
        gt_resized = np.array(Image.fromarray(gt_sem).resize(
            (args.patch_grid, args.patch_grid), Image.NEAREST
        ))
        gt_flat = gt_resized.flatten()
        pred_flat = labels_per_image[idx]

        for p, g in zip(pred_flat, gt_flat):
            if g < NUM_CLASSES:
                cost_matrix[p, g] += 1

    # Hungarian matching: maximize overlap → minimize negative
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    cluster_to_class = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > 0:
            cluster_to_class[r] = c

    # For unmatched clusters, assign to most common class
    for k_id in range(args.k):
        if k_id not in cluster_to_class:
            best_class = np.argmax(cost_matrix[k_id])
            cluster_to_class[k_id] = int(best_class)

    print(f"Cluster→class mapping ({len(cluster_to_class)} clusters):")
    class_counts = {}
    for c in cluster_to_class.values():
        class_counts[COCOSTUFF27_CLASSNAMES[c]] = class_counts.get(
            COCOSTUFF27_CLASSNAMES[c], 0) + 1
    for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count} clusters")

    # ─── Step 4: Generate and save pseudo-semantic labels ───
    print(f"Saving pseudo-semantic labels to {out_dir}...")
    for idx, img_id in tqdm(enumerate(image_ids), desc="Saving labels",
                            total=len(image_ids)):
        # Get original image size
        img_path = root / "val2017" / f"{img_id:012d}.jpg"
        if not img_path.exists():
            continue
        img = Image.open(img_path)
        W, H = img.size

        # Map cluster labels to classes
        cluster_map = labels_per_image[idx].reshape(args.patch_grid, args.patch_grid)
        class_map = np.vectorize(cluster_to_class.get)(cluster_map).astype(np.uint8)

        # Upscale to original resolution
        class_map_full = np.array(Image.fromarray(class_map).resize((W, H), Image.NEAREST))

        # Save as PNG
        out_path = out_dir / f"{img_id:012d}.png"
        Image.fromarray(class_map_full).save(out_path)

    # ─── Step 5: Quick quality check ───
    print("\nComputing mIoU on the pseudo-semantic labels...")
    iou_per_class = np.zeros(NUM_CLASSES)
    count_per_class = np.zeros(NUM_CLASSES)

    for idx, img_id in enumerate(image_ids[:100]):  # quick check on 100 images
        gt_sem = load_coco_panoptic_gt(args.coco_root, img_id)
        if gt_sem is None:
            continue
        pred_path = out_dir / f"{img_id:012d}.png"
        if not pred_path.exists():
            continue
        pred = np.array(Image.open(pred_path))

        # Resize to match
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
    print(f"Quick mIoU (100 images): {miou:.1f}%")
    for c in range(NUM_CLASSES):
        if count_per_class[c] > 0:
            iou = iou_per_class[c] / count_per_class[c] * 100
            print(f"  {COCOSTUFF27_CLASSNAMES[c]:20s}: IoU={iou:.1f}%")

    # Save metadata
    meta = {
        "k": args.k,
        "n_images": len(image_ids),
        "cluster_to_class": {str(k): int(v) for k, v in cluster_to_class.items()},
        "quick_miou_100img": round(miou, 2),
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    # Save centroids
    centroids_path = out_dir / "kmeans_centroids.npz"
    np.savez(centroids_path,
             centroids=kmeans.cluster_centers_,
             cluster_to_class=np.array(
                 [cluster_to_class.get(i, 255) for i in range(args.k)]))
    print(f"Saved centroids to {centroids_path}")


if __name__ == "__main__":
    main()
