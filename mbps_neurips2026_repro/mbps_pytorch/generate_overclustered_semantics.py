#!/usr/bin/env python3
"""
Generate pixel-level semantic pseudo-labels using CAUSE features + k-means overclustering.

Replaces CAUSE's 27 dead-cluster probe with k-means on 90-dim Segment_TR features,
recovering all 7 zero-IoU classes (fence, pole, traffic light/sign, rider, train, motorcycle).

Two output modes:
  - Default: map k clusters → 19 trainID classes via majority vote (for evaluation)
  - --raw_clusters: save raw cluster IDs (0 to k-1) directly (for CUPS Stage-2 training)

Pipeline:
1. Extract CAUSE 90-dim features at patch level → fit k-means
2. Build majority-vote cluster→class mapping using GT
3. For each image: sliding window → 90-dim features → k-means assignment → [mapping] → CRF → PNG

Usage:
    # Mapped 19-class output (original behavior):
    python mbps_pytorch/generate_overclustered_semantics.py \
        --cityscapes_root /path/to/cityscapes \
        --split val --k 300

    # Raw cluster output for CUPS training:
    python mbps_pytorch/generate_overclustered_semantics.py \
        --cityscapes_root /path/to/cityscapes \
        --split val --k 50 --raw_clusters \
        --output_subdir pseudo_semantic_raw_k50

    # Reuse pre-fitted centroids for train split:
    python mbps_pytorch/generate_overclustered_semantics.py \
        --cityscapes_root /path/to/cityscapes \
        --split train --k 50 --raw_clusters \
        --output_subdir pseudo_semantic_raw_k50 \
        --load_centroids /path/to/pseudo_semantic_raw_k50/kmeans_centroids.npz
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

try:
    import pydensecrf.densecrf as dcrf
    import pydensecrf.utils as crf_utils
    HAS_CRF = True
except ImportError:
    HAS_CRF = False

# ─── Constants ───

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
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _remap_to_trainids(gt):
    remapped = np.full_like(gt, IGNORE_LABEL, dtype=np.uint8)
    for raw_id, train_id in _CS_ID_TO_TRAIN.items():
        remapped[gt == raw_id] = train_id
    return remapped


def get_cityscapes_images(cityscapes_root, split):
    img_dir = os.path.join(cityscapes_root, "leftImg8bit", split)
    images = []
    for city in sorted(os.listdir(img_dir)):
        city_dir = os.path.join(img_dir, city)
        if not os.path.isdir(city_dir):
            continue
        for fname in sorted(os.listdir(city_dir)):
            if fname.endswith("_leftImg8bit.png"):
                stem = fname.replace("_leftImg8bit.png", "")
                images.append({"city": city, "stem": stem,
                               "img_path": os.path.join(city_dir, fname)})
    return images


def load_cause_models(checkpoint_dir, device):
    cause_args = SimpleNamespace(
        dim=768, reduced_dim=90, projection_dim=2048,
        num_codebook=2048, n_classes=27,
        num_queries=23 * 23, crop_size=322, patch_size=14,
    )
    backbone_path = os.path.join(checkpoint_dir, "checkpoint", "dinov2_vit_base_14.pth")
    net = dinov2_vit_base_14()
    state = torch.load(backbone_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=False)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad = False

    seg_path = os.path.join(checkpoint_dir, "CAUSE", "cityscapes", "dinov2_vit_base_14", "2048", "segment_tr.pth")
    segment = Segment_TR(cause_args).to(device)
    seg_state = torch.load(seg_path, map_location="cpu", weights_only=True)
    segment.load_state_dict(seg_state, strict=False)
    segment.eval()

    # Load codebook for Segment_TR
    mod_path = os.path.join(checkpoint_dir, "CAUSE", "cityscapes", "modularity",
                            "dinov2_vit_base_14", "2048", "modular.npy")
    cb = torch.from_numpy(np.load(mod_path)).to(device)
    segment.head.codebook = cb
    segment.head_ema.codebook = cb

    return net, segment, cause_args


def extract_cause_features_crop(net, segment, img_tensor):
    """Extract 90-dim CAUSE features for a single 322x322 crop.
    Returns: (1, 90, 23, 23) feature map."""
    with torch.no_grad():
        feat = net(img_tensor)[:, 1:, :]
        feat_flip = net(img_tensor.flip(dims=[3]))[:, 1:, :]
        seg_feat = transform(segment.head_ema(feat))
        seg_feat_flip = transform(segment.head_ema(feat_flip))
        seg_feat = (seg_feat + seg_feat_flip.flip(dims=[3])) / 2  # (1, 90, 23, 23)
    return seg_feat


def sliding_window_features(net, segment, img_resized, crop_size=322):
    """Extract 90-dim CAUSE features via sliding window.
    Returns: (90, H, W) feature map at resized image resolution."""
    _, _, H, W = img_resized.shape
    stride = crop_size // 2

    y_positions = [0] if H <= crop_size else sorted(set(
        list(range(0, H - crop_size, stride)) + [H - crop_size]))
    x_positions = [0] if W <= crop_size else sorted(set(
        list(range(0, W - crop_size, stride)) + [W - crop_size]))

    feat_sum = torch.zeros(90, H, W, device=img_resized.device)
    count = torch.zeros(1, H, W, device=img_resized.device)

    for y_pos in y_positions:
        for x_pos in x_positions:
            y_end = min(y_pos + crop_size, H)
            x_end = min(x_pos + crop_size, W)
            crop = img_resized[:, :, y_pos:y_end, x_pos:x_end]
            ch, cw = crop.shape[2], crop.shape[3]
            if ch < crop_size or cw < crop_size:
                crop = F.pad(crop, (0, crop_size - cw, 0, crop_size - ch), mode='reflect')

            feat_2d = extract_cause_features_crop(net, segment, crop)  # (1, 90, 23, 23)
            # Upsample to pixel resolution
            feat_up = F.interpolate(feat_2d, size=(crop_size, crop_size),
                                    mode='bilinear', align_corners=False)[0]  # (90, crop, crop)
            feat_sum[:, y_pos:y_end, x_pos:x_end] += feat_up[:, :ch, :cw]
            count[:, y_pos:y_end, x_pos:x_end] += 1

    return feat_sum / count.clamp(min=1)  # (90, H, W)


def dense_crf(image_np, logits_np, max_iter=10):
    """Apply DenseCRF. image_np: (H,W,3) uint8, logits_np: (C,H,W) float."""
    if not HAS_CRF:
        return logits_np
    C, H, W = logits_np.shape
    image = np.ascontiguousarray(image_np[:, :, ::-1])
    U = crf_utils.unary_from_softmax(logits_np)
    U = np.ascontiguousarray(U)
    d = dcrf.DenseCRF2D(W, H, C)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=1, compat=3)
    d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=image, compat=4)
    Q = d.inference(max_iter)
    return np.array(Q).reshape((C, H, W))


def fit_kmeans(net, segment, cityscapes_root, device, crop_size, patch_size,
               k=300, max_per_image=2000, seed=42):
    """Fit k-means on CAUSE 90-dim features from val set."""
    print(f"\n=== Fitting k-means (k={k}) on val set features ===")
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    images = get_cityscapes_images(cityscapes_root, "val")
    rng = np.random.RandomState(seed)

    all_feats = []
    all_labels = []

    for entry in tqdm(images, desc="Extracting features for k-means"):
        img_pil = Image.open(entry["img_path"]).convert("RGB")
        orig_w, orig_h = img_pil.size
        scale = crop_size / min(orig_h, orig_w)
        new_h = (int(round(orig_h * scale)) // patch_size) * patch_size
        new_w = (int(round(orig_w * scale)) // patch_size) * patch_size

        img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
        img_tensor = normalize(transforms.ToTensor()(img_resized)).unsqueeze(0).to(device)

        feat_map = sliding_window_features(net, segment, img_tensor, crop_size)
        # Downsample to patch level for k-means fitting
        ph, pw = new_h // patch_size, new_w // patch_size
        feat_ds = F.adaptive_avg_pool2d(feat_map.unsqueeze(0), (ph, pw)).squeeze(0)
        feats = feat_ds.cpu().numpy().reshape(90, -1).T.astype(np.float64)  # (ph*pw, 90)

        # Load GT
        gt_path = os.path.join(cityscapes_root, "gtFine", "val", entry["city"],
                               f"{entry['stem']}_gtFine_labelIds.png")
        gt_raw = np.array(Image.open(gt_path))
        gt = _remap_to_trainids(gt_raw)
        gt_ds = np.array(Image.fromarray(gt).resize((pw, ph), Image.NEAREST)).flatten()

        # Keep only valid pixels
        valid = gt_ds != IGNORE_LABEL
        feats_valid = feats[valid]
        gt_valid = gt_ds[valid]

        if len(feats_valid) == 0:
            continue

        n = min(max_per_image, len(feats_valid))
        idx = rng.choice(len(feats_valid), n, replace=False)
        all_feats.append(feats_valid[idx])
        all_labels.append(gt_valid[idx])

    all_feats = np.concatenate(all_feats)
    all_labels = np.concatenate(all_labels)
    print(f"Collected {len(all_feats)} feature vectors")

    # L2 normalize
    norms = np.linalg.norm(all_feats, axis=1, keepdims=True)
    feats_norm = all_feats / np.maximum(norms, 1e-8)

    # Fit k-means
    print(f"Fitting MiniBatchKMeans (k={k})...")
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10000, max_iter=300,
                             random_state=42, n_init=3)
    kmeans.fit(feats_norm)

    # Build majority-vote mapping: cluster → GT class
    cluster_labels = kmeans.predict(feats_norm)
    conf = np.zeros((k, NUM_CLASSES), dtype=np.int64)
    for cl, gt in zip(cluster_labels, all_labels):
        if gt < NUM_CLASSES:
            conf[cl, gt] += 1
    cluster_to_class = np.argmax(conf, axis=1).astype(np.uint8)

    # Normalize centroids for cosine similarity
    centroids = kmeans.cluster_centers_
    centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

    # Print mapping summary
    for c in range(NUM_CLASSES):
        n_clusters = int((cluster_to_class == c).sum())
        print(f"  {_CS_CLASS_NAMES[c]:15s}: {n_clusters} clusters")

    return centroids_norm, cluster_to_class


def predict_with_kmeans(feat_map, centroids_norm_torch, cluster_to_class_torch):
    """Predict class labels from 90-dim feature map using k-means centroids.
    feat_map: (90, H, W) torch tensor
    Returns: (NUM_CLASSES, H, W) soft logits for CRF, or (H, W) hard labels
    """
    C, H, W = feat_map.shape
    # Normalize features
    feat_norm = F.normalize(feat_map, dim=0)  # (90, H, W)

    # Cosine similarity with centroids: (k, 90) @ (90, H*W) → (k, H*W)
    sim = centroids_norm_torch @ feat_norm.reshape(C, -1)  # (k, H*W)

    # Soft assignment: map k clusters → 19 classes by summing similarities
    # For each of 19 classes, sum the similarities from all clusters mapped to that class
    k = centroids_norm_torch.shape[0]
    class_logits = torch.zeros(NUM_CLASSES, H * W, device=feat_map.device)
    for cls_id in range(NUM_CLASSES):
        cluster_mask = cluster_to_class_torch == cls_id
        if cluster_mask.any():
            class_logits[cls_id] = sim[cluster_mask].max(dim=0).values  # max pooling across clusters

    return class_logits.reshape(NUM_CLASSES, H, W)


def predict_raw_clusters(feat_map, centroids_norm_torch):
    """Predict raw cluster logits from 90-dim feature map using k-means centroids.
    feat_map: (90, H, W) torch tensor
    centroids_norm_torch: (k, 90) torch tensor
    Returns: (k, H, W) cosine similarity logits (raw cluster space, no class mapping)
    """
    C, H, W = feat_map.shape
    feat_norm = F.normalize(feat_map, dim=0)  # (90, H, W)
    sim = centroids_norm_torch @ feat_norm.reshape(C, -1)  # (k, H*W)
    return sim.reshape(-1, H, W)  # (k, H, W)


def generate_pseudolabels(net, segment, cityscapes_root, split, device,
                          centroids_norm, cluster_to_class, output_dir,
                          crop_size=322, patch_size=14, skip_crf=False, limit=0,
                          raw_clusters=False):
    """Generate pixel-level pseudo-labels using k-means overclustering.

    Args:
        raw_clusters: If True, save raw cluster IDs (0 to k-1) instead of mapped trainIDs.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    images = get_cityscapes_images(cityscapes_root, split)
    if limit > 0:
        images = images[:limit]

    # Convert to torch tensors
    centroids_torch = torch.from_numpy(centroids_norm).float().to(device)
    c2c_torch = torch.from_numpy(cluster_to_class).long().to(device)

    k = centroids_norm.shape[0]
    num_output_classes = k if raw_clusters else NUM_CLASSES
    stats = {c: 0 for c in range(num_output_classes)}

    mode_str = f"raw k={k} clusters" if raw_clusters else f"{NUM_CLASSES}-class mapped"
    print(f"\n=== Generating pseudo-labels for {split} ({len(images)} images, {mode_str}) ===")
    print(f"  CRF: {'enabled' if (not skip_crf and HAS_CRF) else 'disabled'}")

    for entry in tqdm(images, desc=f"Generating {split}"):
        img_pil = Image.open(entry["img_path"]).convert("RGB")
        orig_w, orig_h = img_pil.size

        scale = crop_size / min(orig_h, orig_w)
        new_h = (int(round(orig_h * scale)) // patch_size) * patch_size
        new_w = (int(round(orig_w * scale)) // patch_size) * patch_size

        img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
        img_tensor = normalize(transforms.ToTensor()(img_resized)).unsqueeze(0).to(device)

        # Extract pixel-level CAUSE features
        feat_map = sliding_window_features(net, segment, img_tensor, crop_size)

        # Predict using k-means centroids
        if raw_clusters:
            logits = predict_raw_clusters(feat_map, centroids_torch)  # (k, H, W)
        else:
            logits = predict_with_kmeans(feat_map, centroids_torch, c2c_torch)  # (19, H, W)

        # CRF or hard argmax
        if not skip_crf and HAS_CRF:
            img_np = np.array(img_resized)
            softmax_logits = F.softmax(logits * 5, dim=0).cpu().numpy()
            crf_result = dense_crf(img_np, softmax_logits)
            pred = np.argmax(crf_result, axis=0).astype(np.uint8)
        else:
            pred = logits.argmax(dim=0).cpu().numpy().astype(np.uint8)

        # Upsample to original resolution
        pred_pil = Image.fromarray(pred)
        pred_full = np.array(pred_pil.resize((orig_w, orig_h), Image.NEAREST))

        # Save
        city_dir = os.path.join(output_dir, split, entry["city"])
        os.makedirs(city_dir, exist_ok=True)
        out_path = os.path.join(city_dir, f"{entry['stem']}.png")
        Image.fromarray(pred_full.astype(np.uint8)).save(out_path)

        for c in range(num_output_classes):
            stats[c] += int((pred_full == c).sum())

    total = sum(stats.values())
    print(f"\nGenerated {len(images)} pseudo-labels")
    print("Class distribution:")
    for c in range(num_output_classes):
        if raw_clusters:
            gt_cls = int(cluster_to_class[c])
            class_name = f"cluster_{c:03d} ({_CS_CLASS_NAMES[gt_cls]})"
        else:
            class_name = _CS_CLASS_NAMES[c]
        pct = stats[c] / total * 100 if total > 0 else 0
        print(f"  {class_name:35s}: {pct:.1f}%  ({stats[c]:>12,} px)")

    # Save stats
    stats_path = os.path.join(output_dir, split, "stats.json")
    with open(stats_path, "w") as f:
        json.dump({"total_images": len(images), "per_class_pixels": stats,
                    "num_output_classes": num_output_classes,
                    "raw_clusters": raw_clusters}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=300)
    parser.add_argument("--output_subdir", type=str, default=None)
    parser.add_argument("--skip_crf", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--raw_clusters", action="store_true",
                        help="Save raw cluster IDs (0 to k-1) instead of mapped 19-class trainIDs")
    parser.add_argument("--load_centroids", type=str, default=None,
                        help="Path to kmeans_centroids.npz to reuse pre-fitted centroids")
    args = parser.parse_args()

    # Set default output subdir based on mode
    if args.output_subdir is None:
        if args.raw_clusters:
            args.output_subdir = f"pseudo_semantic_raw_k{args.k}"
        else:
            args.output_subdir = f"pseudo_semantic_overclustered_k{args.k}"

    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(Path(__file__).resolve().parent.parent / "refs" / "cause")

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    output_dir = os.path.join(args.cityscapes_root, args.output_subdir)

    # Load models
    net, segment, cause_args = load_cause_models(args.checkpoint_dir, device)

    # Get k-means centroids: either fit new or load pre-fitted
    if args.load_centroids:
        print(f"\nLoading pre-fitted centroids from {args.load_centroids}")
        data = np.load(args.load_centroids)
        centroids_norm = data["centroids"]
        cluster_to_class = data["cluster_to_class"]
        print(f"  Loaded: k={centroids_norm.shape[0]} centroids, "
              f"cluster_to_class shape={cluster_to_class.shape}")
    else:
        # Fit k-means on val features
        centroids_norm, cluster_to_class = fit_kmeans(
            net, segment, args.cityscapes_root, device,
            cause_args.crop_size, cause_args.patch_size, k=args.k
        )
        # Save centroids for reuse
        centroids_path = os.path.join(output_dir, "kmeans_centroids.npz")
        os.makedirs(output_dir, exist_ok=True)
        np.savez(centroids_path, centroids=centroids_norm, cluster_to_class=cluster_to_class)
        print(f"Saved k-means centroids to {centroids_path}")

    # Generate pseudo-labels
    generate_pseudolabels(
        net, segment, args.cityscapes_root, args.split, device,
        centroids_norm, cluster_to_class, output_dir,
        cause_args.crop_size, cause_args.patch_size,
        skip_crf=args.skip_crf, limit=args.limit,
        raw_clusters=args.raw_clusters,
    )

    print(f"\nDone! Pseudo-labels saved to {output_dir}/{args.split}/")
    if args.raw_clusters:
        print(f"\nConvert to CUPS format with:")
        print(f"  python mbps_pytorch/convert_to_cups_format.py \\")
        print(f"    --cityscapes_root {args.cityscapes_root} \\")
        print(f"    --semantic_subdir {args.output_subdir} --split {args.split} \\")
        print(f"    --num_classes {args.k} --cc_instances \\")
        print(f"    --centroids_path {output_dir}/kmeans_centroids.npz")
    else:
        print(f"\nEvaluate with:")
        print(f"  python mbps_pytorch/evaluate_cascade_pseudolabels.py \\")
        print(f"    --cityscapes_root {args.cityscapes_root} \\")
        print(f"    --semantic_subdir {args.output_subdir} --split {args.split} \\")
        print(f"    --instance_subdir pseudo_instance_spidepth --thing_mode hybrid")


if __name__ == "__main__":
    main()
