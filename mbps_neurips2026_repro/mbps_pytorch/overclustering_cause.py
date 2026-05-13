#!/usr/bin/env python3
"""
Overclustering on CAUSE features to recover zero-IoU classes.

Instead of using CAUSE's 27 cluster probe (14 of which are dead), this script:
1. Extracts 90-dim CAUSE features AND 768-dim raw DINOv2 features at patch level
2. Runs k-means with k=27, 50, 100 on both feature spaces
3. Uses many-to-one majority-vote matching to 19 Cityscapes classes
4. Evaluates mIoU and per-class IoU for each configuration

Usage:
    python mbps_pytorch/overclustering_cause.py \
        --cityscapes_root /path/to/cityscapes \
        --split val
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# Add CAUSE repo to path
CAUSE_DIR = str(Path(__file__).resolve().parent.parent / "refs" / "cause")
if CAUSE_DIR not in sys.path:
    sys.path.insert(0, CAUSE_DIR)

from models.dinov2vit import dinov2_vit_base_14
from modules.segment import Segment_TR
from modules.segment_module import Cluster, transform, untransform

# ─── Cityscapes Constants ───

_CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}

_CS_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

NUM_CLASSES = 19
IGNORE_LABEL = 255

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _remap_to_trainids(gt):
    remapped = np.full_like(gt, IGNORE_LABEL, dtype=np.uint8)
    for raw_id, train_id in _CS_ID_TO_TRAIN.items():
        remapped[gt == raw_id] = train_id
    return remapped


def load_cause_models(checkpoint_dir, device):
    """Load CAUSE DINOv2 ViT-B/14 backbone + Segment_TR + Cluster."""
    cause_args = SimpleNamespace(
        dim=768, reduced_dim=90, projection_dim=2048,
        num_codebook=2048, n_classes=27,
        num_queries=23 * 23, crop_size=322, patch_size=14,
    )

    # Backbone
    backbone_path = os.path.join(checkpoint_dir, "checkpoint", "dinov2_vit_base_14.pth")
    net = dinov2_vit_base_14()
    state = torch.load(backbone_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=False)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad = False

    # Segment_TR
    seg_path = os.path.join(checkpoint_dir, "CAUSE", "cityscapes", "dinov2_vit_base_14", "2048", "segment_tr.pth")
    segment = Segment_TR(cause_args).to(device)
    seg_state = torch.load(seg_path, map_location="cpu", weights_only=True)
    segment.load_state_dict(seg_state, strict=False)
    segment.eval()

    # Cluster (for comparison with original)
    cluster_path = os.path.join(checkpoint_dir, "CAUSE", "cityscapes", "dinov2_vit_base_14", "2048", "cluster_tr.pth")
    cluster = Cluster(cause_args).to(device)
    cluster_state = torch.load(cluster_path, map_location="cpu", weights_only=True)
    cluster.load_state_dict(cluster_state, strict=False)

    # Codebook
    mod_path = os.path.join(checkpoint_dir, "CAUSE", "cityscapes", "modularity",
                            "dinov2_vit_base_14", "2048", "modular.npy")
    cb = torch.from_numpy(np.load(mod_path)).to(device)
    cluster.codebook.data = cb
    segment.head.codebook = cb
    segment.head_ema.codebook = cb

    return net, segment, cluster, cause_args


def extract_features_single_crop(net, segment, img_tensor):
    """
    Extract both raw DINOv2 features and CAUSE 90-dim features for a single crop.
    Returns:
        raw_feat: (1, P, 768) raw DINOv2 patch features
        cause_feat: (1, P, 90) CAUSE Segment_TR features
    """
    with torch.no_grad():
        feat = net(img_tensor)[:, 1:, :]  # (1, P, 768) remove CLS
        feat_flip = net(img_tensor.flip(dims=[3]))[:, 1:, :]

        seg_feat = transform(segment.head_ema(feat))        # (1, 90, h, w)
        seg_feat_flip = transform(segment.head_ema(feat_flip))
        seg_feat = (seg_feat + seg_feat_flip.flip(dims=[3])) / 2  # (1, 90, h, w)

        # CAUSE features in patch space (untransform to get (1, P, 90))
        cause_feat = untransform(seg_feat)  # (1, P, 90)

    return feat, cause_feat


def extract_features_sliding_window(net, segment, img_resized, crop_size=322):
    """
    Extract features using sliding window, returning stitched feature maps.
    Returns:
        raw_map: (768, H, W) raw DINOv2 feature map at resized resolution
        cause_map: (90, H, W) CAUSE feature map at resized resolution
    """
    _, _, H, W = img_resized.shape
    patch_h = int(math.sqrt(23 * 23))  # 23
    stride = crop_size // 2

    if H <= crop_size:
        y_positions = [0]
    else:
        y_positions = list(range(0, H - crop_size, stride))
        if y_positions[-1] + crop_size < H:
            y_positions.append(H - crop_size)

    if W <= crop_size:
        x_positions = [0]
    else:
        x_positions = list(range(0, W - crop_size, stride))
        if x_positions[-1] + crop_size < W:
            x_positions.append(W - crop_size)

    y_positions = sorted(set(y_positions))
    x_positions = sorted(set(x_positions))

    raw_sum = torch.zeros(768, H, W, device=img_resized.device)
    cause_sum = torch.zeros(90, H, W, device=img_resized.device)
    count = torch.zeros(1, H, W, device=img_resized.device)

    for y_pos in y_positions:
        for x_pos in x_positions:
            y_end = min(y_pos + crop_size, H)
            x_end = min(x_pos + crop_size, W)
            crop = img_resized[:, :, y_pos:y_end, x_pos:x_end]

            ch, cw = crop.shape[2], crop.shape[3]
            if ch < crop_size or cw < crop_size:
                crop = F.pad(crop, (0, crop_size - cw, 0, crop_size - ch), mode='reflect')

            raw_feat, cause_feat = extract_features_single_crop(net, segment, crop)

            # Upsample patch features to pixel level
            # raw_feat: (1, 529, 768) → reshape to (1, 768, 23, 23) → upsample to (1, 768, crop_size, crop_size)
            raw_2d = raw_feat.permute(0, 2, 1).view(1, 768, patch_h, patch_h)
            raw_up = F.interpolate(raw_2d, size=(crop_size, crop_size), mode='bilinear', align_corners=False)[0]

            cause_2d = cause_feat.permute(0, 2, 1).view(1, 90, patch_h, patch_h)
            cause_up = F.interpolate(cause_2d, size=(crop_size, crop_size), mode='bilinear', align_corners=False)[0]

            raw_sum[:, y_pos:y_end, x_pos:x_end] += raw_up[:, :ch, :cw]
            cause_sum[:, y_pos:y_end, x_pos:x_end] += cause_up[:, :ch, :cw]
            count[:, y_pos:y_end, x_pos:x_end] += 1

    raw_map = raw_sum / count.clamp(min=1)
    cause_map = cause_sum / count.clamp(min=1)
    return raw_map, cause_map


def extract_all_features(net, segment, cityscapes_root, split, device, crop_size=322, patch_size=14, limit=0):
    """
    Extract features from all images in a split.
    Returns per-image feature maps at a downsampled resolution for memory efficiency.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    img_dir = os.path.join(cityscapes_root, "leftImg8bit", split)

    images = []
    for city in sorted(os.listdir(img_dir)):
        city_dir = os.path.join(img_dir, city)
        if not os.path.isdir(city_dir):
            continue
        for fname in sorted(os.listdir(city_dir)):
            if fname.endswith("_leftImg8bit.png"):
                stem = fname.replace("_leftImg8bit.png", "")
                images.append({"city": city, "stem": stem, "img_path": os.path.join(city_dir, fname)})

    if limit > 0:
        images = images[:limit]

    # Store downsampled features for memory efficiency
    # At patch resolution: ~23×46 per image (for 322×644)
    all_cause_features = []  # list of (90, ph, pw) arrays
    all_raw_features = []    # list of (768, ph, pw) arrays
    all_gt_maps = []         # list of (ph, pw) trainID arrays
    image_infos = []

    print(f"Extracting features from {len(images)} images...")
    for entry in tqdm(images, desc="Feature extraction"):
        img_pil = Image.open(entry["img_path"]).convert("RGB")
        orig_w, orig_h = img_pil.size

        # Resize to CAUSE scale
        scale = crop_size / min(orig_h, orig_w)
        new_h = (int(round(orig_h * scale)) // patch_size) * patch_size
        new_w = (int(round(orig_w * scale)) // patch_size) * patch_size

        img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
        img_tensor = normalize(transforms.ToTensor()(img_resized)).unsqueeze(0).to(device)

        raw_map, cause_map = extract_features_sliding_window(net, segment, img_tensor, crop_size)
        # raw_map: (768, new_h, new_w), cause_map: (90, new_h, new_w)

        # Downsample to patch resolution for memory
        ph = new_h // patch_size
        pw = new_w // patch_size
        raw_ds = F.adaptive_avg_pool2d(raw_map.unsqueeze(0), (ph, pw)).squeeze(0)   # (768, ph, pw)
        cause_ds = F.adaptive_avg_pool2d(cause_map.unsqueeze(0), (ph, pw)).squeeze(0)  # (90, ph, pw)

        # Load and downsample GT
        gt_path = os.path.join(cityscapes_root, "gtFine", split, entry["city"],
                               f"{entry['stem']}_gtFine_labelIds.png")
        gt_raw = np.array(Image.open(gt_path))
        gt = _remap_to_trainids(gt_raw)
        gt_ds = np.array(Image.fromarray(gt).resize((pw, ph), Image.NEAREST))

        all_cause_features.append(cause_ds.cpu().float().numpy().astype(np.float64))
        all_raw_features.append(raw_ds.cpu().float().numpy().astype(np.float64))
        all_gt_maps.append(gt_ds)
        image_infos.append({"city": entry["city"], "stem": entry["stem"],
                            "orig_hw": (orig_h, orig_w), "resized_hw": (new_h, new_w),
                            "patch_hw": (ph, pw)})

    return all_cause_features, all_raw_features, all_gt_maps, image_infos


def collect_features_for_kmeans(feature_maps, gt_maps, max_per_image=2000, seed=42):
    """Collect random feature vectors paired with GT labels for k-means fitting."""
    rng = np.random.RandomState(seed)
    all_feats = []
    all_labels = []

    for feat_map, gt_map in zip(feature_maps, gt_maps):
        D, H, W = feat_map.shape
        # Flatten: (D, H*W) → (H*W, D)
        feats_flat = feat_map.reshape(D, -1).T  # (H*W, D)
        gt_flat = gt_map.flatten()  # (H*W,)

        # Only keep valid pixels
        valid = gt_flat != IGNORE_LABEL
        feats_valid = feats_flat[valid]
        gt_valid = gt_flat[valid]

        if len(feats_valid) == 0:
            continue

        # Subsample
        n = min(max_per_image, len(feats_valid))
        idx = rng.choice(len(feats_valid), n, replace=False)
        all_feats.append(feats_valid[idx])
        all_labels.append(gt_valid[idx])

    return np.concatenate(all_feats), np.concatenate(all_labels)


def run_kmeans_and_evaluate(feature_maps, gt_maps, k_values, feat_name, max_fit_samples=500000):
    """Run k-means with different k values and evaluate each."""
    # Collect features for fitting
    print(f"\n{'='*60}")
    print(f"K-MEANS OVERCLUSTERING on {feat_name} features")
    print(f"{'='*60}")

    feats_fit, labels_fit = collect_features_for_kmeans(feature_maps, gt_maps, max_per_image=2000)
    print(f"Collected {len(feats_fit)} feature vectors for k-means fitting")
    print(f"Feature dim: {feats_fit.shape[1]}")
    print(f"GT label distribution:")
    for c in range(NUM_CLASSES):
        count = (labels_fit == c).sum()
        if count > 0:
            print(f"  {_CS_CLASS_NAMES[c]:15s}: {count:>8,} ({count/len(labels_fit)*100:.1f}%)")

    # Replace NaN/inf values with 0
    feats_fit = np.nan_to_num(feats_fit, nan=0.0, posinf=0.0, neginf=0.0)

    # L2-normalize features (since CAUSE uses cosine similarity)
    norms = np.linalg.norm(feats_fit, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    feats_norm = feats_fit / norms

    results = {}
    for k in k_values:
        print(f"\n--- K-means with k={k} ---")
        t0 = time.time()

        # Fit k-means on normalized features
        if len(feats_norm) > max_fit_samples:
            idx = np.random.choice(len(feats_norm), max_fit_samples, replace=False)
            fit_data = feats_norm[idx]
        else:
            fit_data = feats_norm

        kmeans = MiniBatchKMeans(
            n_clusters=k, batch_size=10000, max_iter=300,
            random_state=42, n_init=3, verbose=0
        )
        kmeans.fit(fit_data)
        print(f"  K-means fit: {time.time()-t0:.1f}s")

        # Predict on ALL fitting samples to build confusion matrix
        cluster_labels = kmeans.predict(feats_norm)

        # Build confusion matrix: (k, 19) — count pixels per (cluster, GT class)
        conf = np.zeros((k, NUM_CLASSES), dtype=np.int64)
        for cl, gt in zip(cluster_labels, labels_fit):
            if gt < NUM_CLASSES:
                conf[cl, gt] += 1

        # Many-to-one mapping: each cluster → its majority GT class
        cluster_to_class = np.argmax(conf, axis=1)

        # Show cluster distribution
        active_per_class = {}
        for c in range(NUM_CLASSES):
            active = int((cluster_to_class == c).sum())
            active_per_class[_CS_CLASS_NAMES[c]] = active
        print(f"  Clusters per class: {json.dumps(active_per_class)}")

        # Which of the zero-IoU classes got clusters?
        zero_iou_ids = [4, 5, 6, 7, 12, 16, 17]  # fence, pole, t_light, t_sign, rider, train, motorcycle
        for zid in zero_iou_ids:
            n_clusters = (cluster_to_class == zid).sum()
            name = _CS_CLASS_NAMES[zid]
            if n_clusters > 0:
                # How many pixels assigned to this class?
                assigned_pixels = conf[cluster_to_class == zid, :].sum()
                correct_pixels = conf[cluster_to_class == zid, zid].sum()
                print(f"  ★ {name}: {n_clusters} clusters, {correct_pixels:,}/{assigned_pixels:,} correct pixels")
            else:
                print(f"  ✗ {name}: NO clusters assigned")

        # Evaluate: predict on all images and compute IoU
        print("  Evaluating on full feature maps...")
        full_conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        centroids = kmeans.cluster_centers_  # (k, D)
        centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

        for feat_map, gt_map in zip(feature_maps, gt_maps):
            D, H, W = feat_map.shape
            feats = feat_map.reshape(D, -1).T  # (H*W, D)
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            feats_n = feats / np.maximum(norms, 1e-8)

            # Cosine similarity → nearest centroid
            sim = feats_n @ centroids_norm.T  # (H*W, k)
            cluster_ids = np.argmax(sim, axis=1)  # (H*W,)

            # Map to GT classes
            pred_classes = cluster_to_class[cluster_ids]  # (H*W,)
            gt_flat = gt_map.flatten()

            # Accumulate confusion matrix
            valid = gt_flat != IGNORE_LABEL
            p, g = pred_classes[valid], gt_flat[valid]
            mask = (p < NUM_CLASSES) & (g < NUM_CLASSES)
            np.add.at(full_conf, (p[mask], g[mask]), 1)

        # Compute per-class IoU
        per_class_iou = {}
        ious = []
        for c in range(NUM_CLASSES):
            tp = full_conf[c, c]
            fp = full_conf[c, :].sum() - tp
            fn = full_conf[:, c].sum() - tp
            denom = tp + fp + fn
            iou = tp / denom if denom > 0 else 0.0
            ious.append(iou)
            per_class_iou[_CS_CLASS_NAMES[c]] = iou
        miou = np.mean(ious)

        print(f"\n  Per-class IoU (k={k}):")
        for name, iou in per_class_iou.items():
            bar = "█" * int(iou * 30)
            marker = " ★" if name in ["fence", "pole", "traffic light", "traffic sign",
                                       "rider", "train", "motorcycle"] and iou > 0 else ""
            print(f"    {name:15s}: {iou*100:5.1f}% {bar}{marker}")
        print(f"\n  mIoU: {miou*100:.2f}%")

        results[k] = {
            "miou": float(miou * 100),
            "per_class_iou": {name: float(iou * 100) for name, iou in per_class_iou.items()},
            "clusters_per_class": active_per_class,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--k_values", type=str, default="27,50,100",
                        help="Comma-separated k values for k-means")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(Path(__file__).resolve().parent.parent / "refs" / "cause")

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    k_values = [int(k) for k in args.k_values.split(",")]

    # Load models
    print("Loading CAUSE models...")
    net, segment, cluster, cause_args = load_cause_models(args.checkpoint_dir, device)

    # Extract features
    cause_features, raw_features, gt_maps, image_infos = extract_all_features(
        net, segment, args.cityscapes_root, args.split, device,
        crop_size=cause_args.crop_size, patch_size=cause_args.patch_size,
        limit=args.limit
    )

    print(f"\nExtracted features for {len(cause_features)} images")
    total_patches = sum(f.shape[1] * f.shape[2] for f in cause_features)
    print(f"Total patches: {total_patches:,}")
    cause_mb = sum(f.nbytes for f in cause_features) / 1e6
    raw_mb = sum(f.nbytes for f in raw_features) / 1e6
    print(f"Memory: CAUSE features {cause_mb:.1f} MB, DINOv2 features {raw_mb:.1f} MB")

    # Run k-means on CAUSE 90-dim features
    cause_results = run_kmeans_and_evaluate(cause_features, gt_maps, k_values, "CAUSE-90dim")

    # Run k-means on raw DINOv2 768-dim features
    raw_results = run_kmeans_and_evaluate(raw_features, gt_maps, k_values, "DINOv2-768dim")

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY: Overclustering Results")
    print("=" * 80)
    print(f"\n{'Config':<30s} {'mIoU':>8s}  {'fence':>8s} {'pole':>8s} {'t_light':>8s} {'t_sign':>8s} {'rider':>8s} {'train':>8s} {'moto':>8s}")
    print("-" * 110)

    # CAUSE baseline
    print(f"{'CAUSE-27 (baseline)':<30s} {'40.44':>8s}  {'0.0':>8s} {'0.0':>8s} {'0.0':>8s} {'0.0':>8s} {'0.0':>8s} {'0.0':>8s} {'0.0':>8s}")

    zero_names = ["fence", "pole", "traffic light", "traffic sign", "rider", "train", "motorcycle"]
    for feat_name, results in [("CAUSE-90dim", cause_results), ("DINOv2-768dim", raw_results)]:
        for k, res in results.items():
            label = f"{feat_name} k={k}"
            vals = [f"{res['per_class_iou'].get(n, 0):.1f}" for n in zero_names]
            print(f"{label:<30s} {res['miou']:>7.1f}%  {'  '.join(f'{v:>6s}%' for v in vals)}")

    # Save results
    out_path = os.path.join(args.cityscapes_root, "pseudo_semantic_cause",
                            f"overclustering_results_{args.split}.json")
    all_results = {
        "cause_features": {str(k): v for k, v in cause_results.items()},
        "dinov2_features": {str(k): v for k, v in raw_results.items()},
    }
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
