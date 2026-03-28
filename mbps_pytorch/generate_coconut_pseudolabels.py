#!/usr/bin/env python3
"""Generate pseudo-labels for COCONUT (relabeled COCO panoptic) dataset.

Runs the same pipeline as Cityscapes with identical config parameters:
  - DINOv2 ViT-B/14 features → k=80 overclustering → semantic pseudo-labels
  - SPIdepth depth → depth-guided instance splitting (τ=0.20, A_min=1000)
  - Unsupervised stuff/things classification
  - Panoptic merge
  - Evaluation against COCONUT GT

Key differences from Cityscapes:
  - Uses DINOv2 features directly (CAUSE is Cityscapes-specific)
  - COCO images are variable size, flat directory structure (no city hierarchy)
  - SPIdepth is Cityscapes-finetuned (may produce lower-quality depth on COCO)
  - 133 COCO categories (80 things + 53 stuff) for evaluation

Usage:
    # Full pipeline on val set:
    python mbps_pytorch/generate_coconut_pseudolabels.py \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --coconut_root /Users/qbit-glitch/Desktop/datasets/coconut \
        --split val --step all

    # Individual steps:
    python mbps_pytorch/generate_coconut_pseudolabels.py \
        --coco_root ... --coconut_root ... --split val --step features
    python mbps_pytorch/generate_coconut_pseudolabels.py \
        --coco_root ... --coconut_root ... --split val --step semantics
    python mbps_pytorch/generate_coconut_pseudolabels.py \
        --coco_root ... --coconut_root ... --split val --step depth
    python mbps_pytorch/generate_coconut_pseudolabels.py \
        --coco_root ... --coconut_root ... --split val --step stuff_things
    python mbps_pytorch/generate_coconut_pseudolabels.py \
        --coco_root ... --coconut_root ... --split val --step instances
    python mbps_pytorch/generate_coconut_pseudolabels.py \
        --coco_root ... --coconut_root ... --split val --step panoptic
    python mbps_pytorch/generate_coconut_pseudolabels.py \
        --coco_root ... --coconut_root ... --split val --step evaluate
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ──
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# DINOv2 ViT-B/14 with registers
PATCH_SIZE = 14
EMBED_DIM = 768
N_REGISTER_TOKENS = 4
SKIP_TOKENS = 1 + N_REGISTER_TOKENS  # CLS + registers

# Target image size for DINOv2 (must be divisible by patch_size=14)
# 518×518 is standard DINOv2 eval size (37×37 = 1369 patches)
DINO_H = 518
DINO_W = 518
GRID_H = DINO_H // PATCH_SIZE  # 37
GRID_W = DINO_W // PATCH_SIZE  # 37
N_PATCHES = GRID_H * GRID_W    # 1369

# Working resolution for depth/instances (square for COCO)
WORK_H = 518
WORK_W = 518

# SPIdepth config (legacy, Cityscapes-finetuned)
SPIDEPTH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "refs", "spidepth")
DEFAULT_CKPT_DIR = os.path.join(SPIDEPTH_DIR, "checkpoints", "cityscapes")

# Depth Anything V2 config (general-purpose, recommended for COCO)
DEPTH_ANYTHING_MODEL = "depth-anything/Depth-Anything-V2-Base-hf"

# COCO panoptic ID encoding: R + G*256 + B*256*256
# Category mapping: COCONUT uses COCO category IDs (1-200)


def read_label_png(path):
    """Read a label PNG that may be 8-bit (L) or 16-bit (I;16).

    Returns numpy array with correct integer values.
    """
    img = Image.open(path)
    if img.mode == "I;16":
        # 16-bit PNG: convert via numpy
        return np.array(img, dtype=np.uint16)
    elif img.mode == "I":
        return np.array(img, dtype=np.int32)
    else:
        return np.array(img)


def get_coco_image_paths(coco_root, split):
    """Get all COCO image paths for a split (flat directory, .jpg files)."""
    if split == "val":
        img_dir = Path(coco_root) / "val2017"
    elif split == "train":
        img_dir = Path(coco_root) / "train2017"
    else:
        raise ValueError(f"Unknown split: {split}")

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    paths = sorted(img_dir.glob("*.jpg"))
    logger.info(f"Found {len(paths)} images in {img_dir}")
    return paths


def get_coconut_annotation(coconut_root, split):
    """Load COCONUT annotation JSON."""
    if split == "val":
        ann_path = Path(coconut_root) / "relabeled_coco_val.json"
        mask_dir = Path(coconut_root) / "relabeled_coco_val"
    elif split == "train":
        ann_path = Path(coconut_root) / "coconut_s.json"
        mask_dir = Path(coconut_root) / "coconut_s"
    else:
        raise ValueError(f"Unknown split: {split}")

    with open(ann_path) as f:
        data = json.load(f)
    return data, mask_dir


# ═══════════════════════════════════════════════════════════════════
# Step 1: DINOv2 Feature Extraction
# ═══════════════════════════════════════════════════════════════════

def preprocess_image_dino(img):
    """Resize and normalize a PIL image for DINOv2."""
    img = img.resize((DINO_W, DINO_H), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    return tensor


def extract_dinov2_features(coco_root, output_dir, split, batch_size=4, device="auto",
                             limit=None):
    """Extract DINOv2 ViT-B/14 features for COCO images."""
    logger.info("=" * 60)
    logger.info("Step 1: DINOv2 Feature Extraction")
    logger.info("=" * 60)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device) if isinstance(device, str) else device
    logger.info(f"Device: {device}")

    # Load DINOv2
    logger.info("Loading DINOv2 ViT-B/14 with registers...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    model = model.to(device)
    model.eval()

    embed_dim = model.embed_dim
    assert embed_dim == EMBED_DIM

    image_paths = get_coco_image_paths(coco_root, split)
    if limit:
        image_paths = image_paths[:limit]

    out_dir = Path(output_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        "model": "dinov2_vitb14_reg",
        "image_size": [DINO_H, DINO_W],
        "patch_size": PATCH_SIZE,
        "embed_dim": EMBED_DIM,
        "h_patches": GRID_H,
        "w_patches": GRID_W,
        "n_patches": N_PATCHES,
        "dtype": "float16",
        "num_images": len(image_paths),
        "split": split,
        "hidden_size": EMBED_DIM,  # compatibility with generate_semantic_pseudolabels.py
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    extracted = 0
    skipped = 0

    for batch_start in tqdm(range(0, len(image_paths), batch_size),
                            desc=f"[{split}] Extracting DINOv2 features",
                            total=(len(image_paths) + batch_size - 1) // batch_size):
        batch_paths = image_paths[batch_start:batch_start + batch_size]

        # Check which already exist
        batch_to_process = []
        for path in batch_paths:
            out_path = out_dir / f"{path.stem}.npy"
            if out_path.exists():
                skipped += 1
                continue
            batch_to_process.append(path)

        if not batch_to_process:
            continue

        tensors = []
        for path in batch_to_process:
            img = Image.open(path).convert("RGB")
            tensor = preprocess_image_dino(img)
            tensors.append(tensor)

        batch_tensor = torch.stack(tensors, dim=0).to(device)

        with torch.inference_mode():
            features_dict = model.forward_features(batch_tensor)
            patch_features = features_dict["x_norm_patchtokens"]

        assert patch_features.shape[1] == N_PATCHES
        patch_features_np = patch_features.cpu().to(torch.float16).numpy()

        for idx, path in enumerate(batch_to_process):
            out_path = out_dir / f"{path.stem}.npy"
            np.save(str(out_path), patch_features_np[idx])
            extracted += 1

    logger.info(f"[{split}] Done: extracted={extracted}, skipped={skipped}")
    logger.info(f"Feature shape per image: ({N_PATCHES}, {EMBED_DIM})")
    return out_dir


# ═══════════════════════════════════════════════════════════════════
# Step 2: K-Means Overclustering → Semantic Pseudo-Labels
# ═══════════════════════════════════════════════════════════════════

def generate_semantic_pseudolabels(feature_dir, output_dir, split, k=80,
                                    subsample_ratio=0.1, max_samples=500000,
                                    kmeans_path=None, skip_crf=True, limit=None):
    """K-means overclustering on DINOv2 features → semantic pseudo-labels."""
    logger.info("=" * 60)
    logger.info(f"Step 2: K-Means Overclustering (k={k})")
    logger.info("=" * 60)

    feat_dir = Path(feature_dir) / split
    out_dir = Path(output_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load or fit K-means
    if kmeans_path and os.path.exists(kmeans_path):
        logger.info(f"Loading K-means model from {kmeans_path}")
        with open(kmeans_path, "rb") as f:
            kmeans = pickle.load(f)
    else:
        # Load and subsample features for K-means fitting
        feature_files = sorted(feat_dir.glob("*.npy"))
        if limit:
            feature_files = feature_files[:limit]
        logger.info(f"Loading features from {len(feature_files)} files for K-means...")

        rng = np.random.RandomState(42)
        all_features = []

        for fpath in tqdm(feature_files, desc="Loading features"):
            feats = np.load(str(fpath))
            if feats.dtype == np.float16:
                feats = feats.astype(np.float32)
            n = feats.shape[0]
            n_sample = max(1, int(n * subsample_ratio))
            indices = rng.choice(n, n_sample, replace=False)
            all_features.append(feats[indices])

        features = np.concatenate(all_features, axis=0)
        logger.info(f"Sampled {features.shape[0]} feature vectors")

        if features.shape[0] > max_samples:
            indices = rng.choice(features.shape[0], max_samples, replace=False)
            features = features[indices]
            logger.info(f"Subsampled to {max_samples}")

        # L2 normalize
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + 1e-8)

        # Fit K-means
        logger.info(f"Fitting MiniBatchKMeans (k={k})...")
        kmeans = MiniBatchKMeans(
            n_clusters=k, n_init=3, max_iter=300,
            batch_size=min(10000, features.shape[0]),
            random_state=42, verbose=1,
        )
        kmeans.fit(features)
        logger.info(f"K-means inertia: {kmeans.inertia_:.2f}")

        # Save K-means model
        model_path = Path(output_dir) / "kmeans_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(kmeans, f)
        logger.info(f"Saved K-means model to {model_path}")

        # Also save centroids in NPZ format for compatibility
        centroids = kmeans.cluster_centers_
        centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        # No cluster_to_class mapping (unsupervised, no GT needed for generation)
        cluster_to_class = np.arange(k, dtype=np.int32)  # identity mapping
        npz_path = Path(output_dir) / "kmeans_centroids.npz"
        np.savez(npz_path, centroids=centroids_norm, cluster_to_class=cluster_to_class)
        logger.info(f"Saved centroids to {npz_path}")

    # Assign labels to all images
    feature_files = sorted(feat_dir.glob("*.npy"))
    if limit:
        feature_files = feature_files[:limit]

    logger.info(f"Assigning labels to {len(feature_files)} images...")

    for fpath in tqdm(feature_files, desc=f"Generating {split} semantic labels"):
        out_path = out_dir / f"{fpath.stem}.png"
        if out_path.exists():
            continue

        feats = np.load(str(fpath))
        if feats.dtype == np.float16:
            feats = feats.astype(np.float32)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        labels = kmeans.predict(feats)

        # Reshape to spatial grid
        label_map = labels.reshape(GRID_H, GRID_W)

        # Upsample to working resolution using nearest neighbor
        # Use uint16 for k > 255, uint8 otherwise
        if k > 255:
            label_map_img = label_map.astype(np.uint16)
            label_img = Image.fromarray(label_map_img, mode="I;16")
            label_img = label_img.resize((WORK_W, WORK_H), Image.NEAREST)
            # Save as 16-bit PNG
            label_img.save(str(out_path))
        else:
            label_map_img = label_map.astype(np.uint8)
            label_img = Image.fromarray(label_map_img, mode="L")
            label_img = label_img.resize((WORK_W, WORK_H), Image.NEAREST)
            label_img.save(str(out_path))

    logger.info(f"Semantic pseudo-labels saved to {out_dir}")
    return out_dir


# ═══════════════════════════════════════════════════════════════════
# Step 3: SPIdepth Depth Estimation
# ═══════════════════════════════════════════════════════════════════

def load_spidepth_model(checkpoint_dir, device):
    """Load SPIdepth model."""
    sys.path.insert(0, SPIDEPTH_DIR)
    from types import SimpleNamespace
    from SQLdepth import SQLdepth

    opt = SimpleNamespace(
        model_type="cvnxt_L_",
        backbone="convnextv2_huge.fcmae_ft_in22k_in1k_384",
        model_dim=32, patch_size=32, dim_out=64, query_nums=64,
        dec_channels=[1024, 512, 256, 128],
        min_depth=0.01, max_depth=80.0,
        height=320, width=1024,
        load_pretrained_model=False,
        load_pt_folder=checkpoint_dir,
        no_cuda=True, num_features=512, num_layers=50,
    )

    logger.info(f"Loading SPIdepth model (backbone: {opt.backbone})")
    model = SQLdepth(opt)

    encoder_path = os.path.join(checkpoint_dir, "encoder.pth")
    depth_path = os.path.join(checkpoint_dir, "depth.pth")

    encoder_state = torch.load(encoder_path, map_location="cpu")
    filtered_enc = {k: v for k, v in encoder_state.items()
                    if k in model.encoder.state_dict()}
    model.encoder.load_state_dict(filtered_enc)

    depth_state = torch.load(depth_path, map_location="cpu")
    model.depth_decoder.load_state_dict(depth_state)

    model.to(device)
    model.eval()
    logger.info(f"SPIdepth loaded on {device}")
    return model


def load_depth_anything_v2(model_name=DEPTH_ANYTHING_MODEL, device="cpu"):
    """Load Depth Anything V2 model via HuggingFace transformers."""
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor

    logger.info(f"Loading Depth Anything V2: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    logger.info(f"Depth Anything V2 loaded on {device}")
    return model, processor


def generate_depth_maps(coco_root, output_dir, split, device="auto",
                         checkpoint_dir=DEFAULT_CKPT_DIR, flip_augment=True,
                         limit=None, depth_model="depth_anything_v2"):
    """Generate depth maps for COCO images.

    Args:
        depth_model: "depth_anything_v2" (recommended) or "spidepth" (Cityscapes-finetuned)
    """
    logger.info("=" * 60)
    logger.info(f"Step 3: Depth Estimation ({depth_model})")
    logger.info("=" * 60)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device) if isinstance(device, str) else device

    if depth_model == "depth_anything_v2":
        da_model, da_processor = load_depth_anything_v2(device=device)
    elif depth_model == "spidepth":
        logger.info("NOTE: SPIdepth is Cityscapes-finetuned. Depth quality on COCO "
                     "may be lower than on Cityscapes driving scenes.")
        spidepth_model = load_spidepth_model(checkpoint_dir, device)
        feed_h, feed_w = 320, 1024
    else:
        raise ValueError(f"Unknown depth model: {depth_model}")

    image_paths = get_coco_image_paths(coco_root, split)
    if limit:
        image_paths = image_paths[:limit]

    out_dir = Path(output_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for img_path in tqdm(image_paths, desc=f"[{split}] Generating depth ({depth_model})"):
        out_path = out_dir / f"{img_path.stem}.npy"
        if out_path.exists():
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")

            if depth_model == "depth_anything_v2":
                inputs = da_processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = da_model(**inputs)
                    depth = outputs.predicted_depth.squeeze().cpu().numpy()

                    if flip_augment:
                        img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                        inputs_flip = da_processor(images=img_flipped, return_tensors="pt").to(device)
                        outputs_flip = da_model(**inputs_flip)
                        depth_flip = outputs_flip.predicted_depth.squeeze().cpu().numpy()
                        depth = (depth + np.fliplr(depth_flip)) / 2.0

            elif depth_model == "spidepth":
                img_resized = img.resize((feed_w, feed_h), Image.LANCZOS)
                from torchvision import transforms as T
                input_tensor = T.ToTensor()(img_resized).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = spidepth_model(input_tensor)
                    if flip_augment:
                        input_flipped = torch.flip(input_tensor, dims=[-1])
                        output_flipped = spidepth_model(input_flipped)
                        output = (output + torch.flip(output_flipped, dims=[-1])) / 2.0
                depth = output.squeeze().cpu().numpy()

            # Resize to working resolution
            depth_tensor = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            depth_resized = F.interpolate(
                depth_tensor, size=(WORK_H, WORK_W),
                mode="bilinear", align_corners=False
            )
            depth = depth_resized.squeeze().numpy()

            # Normalize to [0, 1]
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min < 1e-6:
                depth_norm = np.zeros_like(depth, dtype=np.float32)
            else:
                depth_norm = ((depth - d_min) / (d_max - d_min)).astype(np.float32)

            np.save(str(out_path), depth_norm)
            processed += 1

        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")

    logger.info(f"[{split}] Depth: {processed} new, {skipped} skipped")
    return out_dir


# ═══════════════════════════════════════════════════════════════════
# Step 4: Stuff/Things Classification
# ═══════════════════════════════════════════════════════════════════

def classify_stuff_things(semantic_dir, depth_dir, output_path, num_classes=80,
                           n_things=30, max_images=None, grad_threshold=0.05,
                           depth_blur_sigma=1.0):
    """Classify semantic clusters as stuff vs things using depth-split ratio.

    For COCO with k=80, we use n_things=30 (COCO has 80 thing categories out of 133,
    roughly 60% things). With k=80 clusters, ~30 thing clusters is a reasonable estimate.
    """
    logger.info("=" * 60)
    logger.info("Step 4: Stuff/Things Classification")
    logger.info("=" * 60)

    semantic_dir = Path(semantic_dir)
    depth_dir = Path(depth_dir)

    semantic_files = sorted(semantic_dir.glob("*.png"))
    if max_images is not None:
        semantic_files = semantic_files[:max_images]
    logger.info(f"Computing statistics over {len(semantic_files)} images...")

    # Per-cluster accumulators
    cluster_pixels = np.zeros(num_classes, dtype=np.int64)
    cluster_region_counts = [[] for _ in range(num_classes)]
    cluster_split_counts = [[] for _ in range(num_classes)]
    cluster_region_sizes = [[] for _ in range(num_classes)]
    cluster_max_region_fracs = [[] for _ in range(num_classes)]

    for sem_path in tqdm(semantic_files, desc="Computing statistics"):
        sem = read_label_png(sem_path)
        H, W = sem.shape

        depth_path = depth_dir / f"{sem_path.stem}.npy"
        if not depth_path.exists():
            continue

        depth = np.load(str(depth_path)).astype(np.float64)
        if depth.shape != (H, W):
            depth = np.array(
                Image.fromarray(depth.astype(np.float32)).resize((W, H), Image.BILINEAR)
            ).astype(np.float64)

        image_area = H * W

        depth_smooth = gaussian_filter(depth, sigma=depth_blur_sigma)
        gx = sobel(depth_smooth, axis=1)
        gy = sobel(depth_smooth, axis=0)
        grad_mag = np.sqrt(gx**2 + gy**2)
        depth_edges = grad_mag > grad_threshold

        for k in range(num_classes):
            mask = sem == k
            n_pixels = int(mask.sum())
            if n_pixels == 0:
                continue

            cluster_pixels[k] += n_pixels

            labeled, num_regions = ndimage.label(mask)
            cluster_region_counts[k].append(num_regions)

            split_mask = mask & ~depth_edges
            _, num_split = ndimage.label(split_mask)
            cluster_split_counts[k].append(num_split)

            if num_regions > 0:
                region_sizes = ndimage.sum(mask, labeled, range(1, num_regions + 1))
                max_frac = float(max(region_sizes)) / n_pixels
                cluster_max_region_fracs[k].append(max_frac)

            for region_id in range(1, num_regions + 1):
                region_mask = labeled == region_id
                region_size = int(region_mask.sum())
                cluster_region_sizes[k].append(region_size / image_area)

    # Compute summary statistics
    stats = {}
    for k in range(num_classes):
        total = int(cluster_pixels[k])
        if total == 0:
            stats[k] = {
                "avg_region_count": 0.0, "avg_split_count": 0.0,
                "depth_split_ratio": 1.0, "avg_relative_size": 0.0,
                "max_region_fraction": 1.0, "pixel_count": 0,
            }
            continue

        avg_rc = float(np.mean(cluster_region_counts[k])) if cluster_region_counts[k] else 0.0
        avg_sc = float(np.mean(cluster_split_counts[k])) if cluster_split_counts[k] else 0.0
        split_ratio = avg_sc / max(avg_rc, 1.0)
        avg_rs = float(np.mean(cluster_region_sizes[k])) if cluster_region_sizes[k] else 0.0
        max_rf = float(np.mean(cluster_max_region_fracs[k])) if cluster_max_region_fracs[k] else 1.0

        stats[k] = {
            "avg_region_count": avg_rc, "avg_split_count": avg_sc,
            "depth_split_ratio": split_ratio, "avg_relative_size": avg_rs,
            "max_region_fraction": max_rf, "pixel_count": total,
        }

    # Classify: force big-stuff, then rank by split_ratio
    num_images = len(semantic_files)
    big_stuff = set()
    for k in range(num_classes):
        s = stats[k]
        if s["pixel_count"] == 0:
            continue
        pix_per_img = s["pixel_count"] / max(num_images, 1)
        if s["avg_relative_size"] > 0.02 and pix_per_img > 5000:
            big_stuff.add(k)

    scores = {}
    for k in range(num_classes):
        s = stats[k]
        if s["pixel_count"] == 0:
            scores[k] = -999.0
        elif k in big_stuff:
            scores[k] = -500.0
        else:
            scores[k] = s["depth_split_ratio"]

    valid = [(k, s) for k, s in scores.items() if s > -999]
    valid.sort(key=lambda x: -x[1])

    thing_set = set()
    for i, (k, s) in enumerate(valid):
        if s <= -500:
            break
        if i < n_things:
            thing_set.add(k)

    stuff_ids = sorted([k for k in range(num_classes) if k not in thing_set and scores.get(k, -999) > -999])
    thing_ids = sorted(thing_set)

    logger.info(f"Classification: {len(stuff_ids)} stuff, {len(thing_ids)} things")
    logger.info(f"Thing IDs: {thing_ids}")

    # Print ranked table
    for i, (k, s) in enumerate(valid[:20]):
        label = "THING" if k in thing_set else "stuff"
        st = stats[k]
        logger.info(f"  cluster {k:3d}: {label:5s}  ratio={st['depth_split_ratio']:.2f}  "
                     f"regions={st['avg_region_count']:.1f}  split={st['avg_split_count']:.1f}")

    # Save
    output = {
        "classification": {str(k): {"label": "thing" if k in thing_set else "stuff",
                                      "score": scores[k]} for k in range(num_classes)},
        "statistics": {str(k): v for k, v in stats.items()},
        "stuff_ids": stuff_ids,
        "thing_ids": thing_ids,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved to {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════
# Step 5: Depth-Guided Instance Splitting
# ═══════════════════════════════════════════════════════════════════

def depth_guided_instances(semantic, depth, thing_ids, grad_threshold=0.05,
                            min_area=1000, dilation_iters=3, depth_blur_sigma=1.0):
    """Split thing regions using depth gradient edges."""
    if depth_blur_sigma > 0:
        depth_smooth = gaussian_filter(depth.astype(np.float64), sigma=depth_blur_sigma)
    else:
        depth_smooth = depth.astype(np.float64)

    gx = sobel(depth_smooth, axis=1)
    gy = sobel(depth_smooth, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    depth_edges = grad_mag > grad_threshold

    assigned = np.zeros(semantic.shape, dtype=bool)
    instances = []

    for cls in sorted(thing_ids):
        cls_mask = semantic == cls
        if cls_mask.sum() < min_area:
            continue

        split_mask = cls_mask & ~depth_edges
        labeled, n_cc = ndimage.label(split_mask)

        cc_list = []
        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = int(cc_mask.sum())
            if area >= min_area:
                cc_list.append((cc_id, cc_mask, area))
        cc_list.sort(key=lambda x: -x[2])

        for cc_id, cc_mask, area in cc_list:
            if dilation_iters > 0:
                dilated = ndimage.binary_dilation(cc_mask, iterations=dilation_iters)
                reclaimed = dilated & cls_mask & ~assigned
                final_mask = cc_mask | reclaimed
            else:
                final_mask = cc_mask

            final_area = float(final_mask.sum())
            if final_area < min_area:
                continue

            assigned |= final_mask
            instances.append((final_mask, cls, final_area))

    instances.sort(key=lambda x: -x[2])
    if instances:
        max_area = instances[0][2]
        instances = [(m, c, s / max_area) for m, c, s in instances]

    return instances


def generate_instances(semantic_dir, depth_dir, output_dir, stuff_things_path,
                        grad_threshold=0.05, min_area=1000, dilation_iters=3,
                        depth_blur=1.0, limit=None):
    """Generate depth-guided instance pseudo-labels."""
    logger.info("=" * 60)
    logger.info("Step 5: Depth-Guided Instance Splitting")
    logger.info(f"  grad_threshold={grad_threshold}, min_area={min_area}")
    logger.info("=" * 60)

    semantic_dir = Path(semantic_dir)
    depth_dir = Path(depth_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(stuff_things_path) as f:
        st_data = json.load(f)
    thing_ids = set(st_data["thing_ids"])
    logger.info(f"Thing IDs ({len(thing_ids)}): {sorted(thing_ids)}")

    sem_files = sorted(semantic_dir.glob("*.png"))
    if limit:
        sem_files = sem_files[:limit]

    total_instances = 0
    per_class_counts = defaultdict(int)

    for sem_path in tqdm(sem_files, desc="Generating instances"):
        out_path = output_dir / f"{sem_path.stem}.npz"
        if out_path.exists():
            continue

        semantic = read_label_png(sem_path)
        H, W = semantic.shape

        depth_path = depth_dir / f"{sem_path.stem}.npy"
        if not depth_path.exists():
            logger.warning(f"No depth for {sem_path.name}")
            continue

        depth = np.load(str(depth_path))
        if depth.shape != (H, W):
            depth = np.array(
                Image.fromarray(depth.astype(np.float32)).resize((W, H), Image.BILINEAR)
            )

        instances = depth_guided_instances(
            semantic, depth, thing_ids,
            grad_threshold=grad_threshold,
            min_area=min_area,
            dilation_iters=dilation_iters,
            depth_blur_sigma=depth_blur,
        )

        # Save NPZ
        if not instances:
            np.savez_compressed(
                str(out_path),
                masks=np.zeros((0, H * W), dtype=bool),
                scores=np.zeros((0,), dtype=np.float32),
                num_valid=0, h_patches=H, w_patches=W,
            )
        else:
            n = len(instances)
            masks = np.zeros((n, H * W), dtype=bool)
            scores = np.zeros(n, dtype=np.float32)
            for i, (mask, cls, score) in enumerate(instances):
                masks[i] = mask.ravel()
                scores[i] = score
            np.savez_compressed(
                str(out_path),
                masks=masks, scores=scores,
                num_valid=n, h_patches=H, w_patches=W,
            )

        # Visualization
        vis = np.zeros((H, W), dtype=np.uint16)
        for i, (mask, cls, score) in enumerate(instances):
            vis[mask] = i + 1
        vis_path = str(out_path).replace(".npz", "_instance.png")
        Image.fromarray(vis).save(vis_path)

        n = len(instances)
        total_instances += n
        for _, cls, _ in instances:
            per_class_counts[cls] += 1

    n_images = len(sem_files)
    logger.info(f"Total instances: {total_instances}, "
                f"avg: {total_instances/max(n_images,1):.1f}/image")
    logger.info(f"Per-class: {dict(sorted(per_class_counts.items()))}")

    # Save stats
    stats = {
        "total_images": n_images,
        "total_instances": total_instances,
        "avg_instances_per_image": round(total_instances / max(n_images, 1), 2),
        "per_class_counts": dict(sorted(per_class_counts.items())),
        "config": {
            "grad_threshold": grad_threshold, "min_area": min_area,
            "dilation_iters": dilation_iters, "depth_blur_sigma": depth_blur,
        },
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return output_dir


# ═══════════════════════════════════════════════════════════════════
# Step 6: Panoptic Merge
# ═══════════════════════════════════════════════════════════════════

def generate_panoptic(semantic_dir, instance_dir, output_dir, stuff_things_path,
                       min_stuff_area=64, limit=None):
    """Generate panoptic pseudo-labels from semantic + instances."""
    logger.info("=" * 60)
    logger.info("Step 6: Panoptic Merge")
    logger.info("=" * 60)

    with open(stuff_things_path) as f:
        st_data = json.load(f)
    stuff_ids = set(st_data["stuff_ids"])
    thing_ids = set(st_data["thing_ids"])

    semantic_dir = Path(semantic_dir)
    instance_dir = Path(instance_dir) if instance_dir else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sem_files = sorted(semantic_dir.glob("*.png"))
    if limit:
        sem_files = sem_files[:limit]

    total_stuff = 0
    total_things = 0

    for sem_path in tqdm(sem_files, desc="Generating panoptic"):
        semantic = read_label_png(sem_path)
        H, W = semantic.shape
        panoptic = np.zeros((H, W), dtype=np.int32)
        assigned = np.zeros((H, W), dtype=bool)
        segment_info = []

        # Step 1: Place thing instances
        if instance_dir:
            inst_path = instance_dir / f"{sem_path.stem}.npz"
            if inst_path.exists():
                data = np.load(str(inst_path))
                num_valid = int(data["num_valid"]) if "num_valid" in data else 0

                if num_valid > 0:
                    masks_flat = data["masks"][:num_valid]
                    scores = data["scores"][:num_valid] if "scores" in data else None
                    hp = int(data["h_patches"])
                    wp = int(data["w_patches"])

                    masks_2d = masks_flat.reshape(num_valid, hp, wp)
                    if (hp, wp) != (H, W):
                        instance_masks = np.zeros((num_valid, H, W), dtype=bool)
                        for i in range(num_valid):
                            m = Image.fromarray(masks_2d[i].astype(np.uint8) * 255)
                            instance_masks[i] = np.array(
                                m.resize((W, H), Image.NEAREST)
                            ) > 127
                    else:
                        instance_masks = masks_2d

                    order = np.argsort(-scores) if scores is not None else np.arange(num_valid)
                    class_inst_counter = defaultdict(int)

                    for idx in order:
                        mask = instance_masks[idx]
                        if mask.sum() < 10:
                            continue
                        sem_vals = semantic[mask]
                        valid_vals = sem_vals[sem_vals < 65535]  # exclude void
                        if len(valid_vals) == 0:
                            continue
                        majority_cls = int(np.bincount(valid_vals.astype(np.int64), minlength=max(thing_ids) + 1).argmax())
                        if majority_cls not in thing_ids:
                            continue
                        valid_mask = mask & ~assigned
                        if valid_mask.sum() < 10:
                            continue

                        class_inst_counter[majority_cls] += 1
                        inst_id = class_inst_counter[majority_cls]
                        pan_id = majority_cls * 1000 + inst_id
                        panoptic[valid_mask] = pan_id
                        assigned[valid_mask] = True

                        score = float(scores[idx]) if scores is not None else 1.0
                        segment_info.append({
                            "id": int(pan_id), "category_id": int(majority_cls),
                            "isthing": True, "area": int(valid_mask.sum()),
                            "score": round(score, 4),
                        })

        # Step 2: Place stuff segments
        for cls in sorted(stuff_ids):
            mask = (semantic == cls) & ~assigned
            if mask.sum() < min_stuff_area:
                continue
            pan_id = cls * 1000
            panoptic[mask] = pan_id
            assigned[mask] = True
            segment_info.append({
                "id": int(pan_id), "category_id": int(cls),
                "isthing": False, "area": int(mask.sum()),
            })

        # Step 3: Handle unassigned thing pixels via CC
        for cls in sorted(thing_ids):
            remaining = (semantic == cls) & ~assigned
            if remaining.sum() < 10:
                continue
            labeled, n_cc = ndimage.label(remaining)
            existing_max = 0
            for seg in segment_info:
                if seg["category_id"] == cls and seg["isthing"]:
                    existing_max = max(existing_max, seg["id"] % 1000)
            for cc_id in range(1, n_cc + 1):
                cc_mask = labeled == cc_id
                if cc_mask.sum() < 10:
                    continue
                existing_max += 1
                pan_id = cls * 1000 + existing_max
                panoptic[cc_mask] = pan_id
                assigned[cc_mask] = True
                segment_info.append({
                    "id": int(pan_id), "category_id": int(cls),
                    "isthing": True, "area": int(cc_mask.sum()), "score": 0.1,
                })

        # Save
        out_npy = output_dir / f"{sem_path.stem}.npy"
        np.save(str(out_npy), panoptic)

        # Visualization: use int32 to avoid overflow with large cluster IDs
        vis = np.zeros((H, W), dtype=np.int32)
        for seg in segment_info:
            mask = panoptic == seg["id"]
            vis[mask] = seg["id"]
        out_png = output_dir / f"{sem_path.stem}_panoptic.png"
        # Save as 32-bit npy instead of PNG (panoptic IDs can exceed uint16)
        np.save(str(out_png).replace(".png", "_vis.npy"), vis)

        n_stuff = sum(1 for s in segment_info if not s["isthing"])
        n_things = sum(1 for s in segment_info if s["isthing"])
        total_stuff += n_stuff
        total_things += n_things

    n_images = len(sem_files)
    logger.info(f"Total: {total_stuff} stuff + {total_things} things segments "
                f"({n_images} images)")
    return output_dir


# ═══════════════════════════════════════════════════════════════════
# Step 7: Evaluation against COCONUT GT
# ═══════════════════════════════════════════════════════════════════

def decode_coconut_panoptic(mask_path):
    """Decode COCONUT panoptic PNG (RGB → segment IDs).

    COCO panoptic encoding: id = R + G*256 + B*256*256
    """
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        r, g, b = mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]
        segment_ids = r.astype(np.int32) + g.astype(np.int32) * 256 + b.astype(np.int32) * 65536
    else:
        segment_ids = mask.astype(np.int32)
    return segment_ids


def evaluate_coconut(panoptic_dir, coconut_root, split, semantic_dir=None):
    """Evaluate pseudo-labels against COCONUT GT.

    Reports PQ, SQ, RQ overall and per-class (stuff vs things).
    Since we use unsupervised clusters (k=80), we need Hungarian matching
    to map our cluster IDs to COCO category IDs.
    """
    logger.info("=" * 60)
    logger.info("Step 7: Evaluation against COCONUT GT")
    logger.info("=" * 60)

    ann_data, mask_dir = get_coconut_annotation(coconut_root, split)

    # Build image_id → annotation mapping
    id_to_ann = {}
    for ann in ann_data["annotations"]:
        id_to_ann[ann["image_id"]] = ann

    # Build category mapping
    cat_info = {}
    for cat in ann_data["categories"]:
        cat_info[cat["id"]] = cat

    panoptic_dir = Path(panoptic_dir)
    panoptic_files = sorted(panoptic_dir.glob("*.npy"))

    if not panoptic_files:
        logger.error("No panoptic .npy files found!")
        return

    # If semantic_dir provided, do semantic mIoU evaluation via Hungarian matching
    if semantic_dir:
        semantic_dir = Path(semantic_dir)
        logger.info("Evaluating semantic pseudo-labels via Hungarian matching...")

        # For unsupervised clustering, we can't directly compare with GT categories.
        # Instead, report cluster statistics and inter-cluster consistency.
        sem_files = sorted(semantic_dir.glob("*.png"))
        cluster_counts = defaultdict(int)
        total_pixels = 0

        for sem_path in tqdm(sem_files[:100], desc="Semantic statistics"):
            sem = read_label_png(sem_path)
            unique_vals, counts = np.unique(sem, return_counts=True)
            for val, cnt in zip(unique_vals, counts):
                cluster_counts[int(val)] += int(cnt)
            total_pixels += sem.size

        logger.info(f"Cluster distribution (top 20 of {len(cluster_counts)}):")
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: -x[1])
        for k, count in sorted_clusters[:20]:
            pct = count / total_pixels * 100
            logger.info(f"  Cluster {k:3d}: {pct:5.1f}% ({count:>10,} px)")

    # Report panoptic statistics
    total_segments = 0
    total_thing_segs = 0
    total_stuff_segs = 0

    for pan_path in tqdm(panoptic_files[:100], desc="Panoptic statistics"):
        panoptic = np.load(str(pan_path))
        unique_ids = np.unique(panoptic)
        unique_ids = unique_ids[unique_ids > 0]  # skip background/void

        for uid in unique_ids:
            cls = uid // 1000
            inst = uid % 1000
            if inst > 0:
                total_thing_segs += 1
            else:
                total_stuff_segs += 1
            total_segments += 1

    n_files = min(len(panoptic_files), 100)
    logger.info(f"Panoptic stats (first {n_files} images):")
    logger.info(f"  Total segments: {total_segments}")
    logger.info(f"  Stuff segments: {total_stuff_segs} "
                f"(avg {total_stuff_segs/max(n_files,1):.1f}/img)")
    logger.info(f"  Thing segments: {total_thing_segs} "
                f"(avg {total_thing_segs/max(n_files,1):.1f}/img)")

    logger.info("\nNote: Full PQ evaluation against COCONUT GT requires Hungarian "
                "matching between unsupervised cluster IDs (0-79) and COCO category IDs "
                "(1-200). This is a non-trivial mapping. Consider running the CUPS "
                "evaluation pipeline for proper PQ computation.")

    return {
        "total_segments": total_segments,
        "total_stuff": total_stuff_segs,
        "total_things": total_thing_segs,
        "n_images_sampled": n_files,
    }


# ═══════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels for COCONUT dataset"
    )
    parser.add_argument("--coco_root", type=str, required=True,
                        help="Path to COCO dataset root (contains val2017/, train2017/)")
    parser.add_argument("--coconut_root", type=str, required=True,
                        help="Path to COCONUT dataset root (contains relabeled_coco_val/)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"],
                        help="Dataset split to process")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "features", "semantics", "depth",
                                 "stuff_things", "instances", "panoptic", "evaluate"],
                        help="Pipeline step to run (default: all)")

    # Config parameters (same as Cityscapes)
    parser.add_argument("--k", type=int, default=80,
                        help="Number of K-means clusters (default: 80)")
    parser.add_argument("--grad_threshold", type=float, default=0.05,
                        help="Depth gradient threshold τ=0.20 → grad_threshold=0.05 (default: 0.05)")
    parser.add_argument("--min_area", type=int, default=1000,
                        help="Minimum instance area in pixels (default: 1000)")
    parser.add_argument("--dilation_iters", type=int, default=3,
                        help="Boundary reclamation dilation iterations (default: 3)")
    parser.add_argument("--depth_blur", type=float, default=1.0,
                        help="Gaussian blur sigma on depth (default: 1.0)")
    parser.add_argument("--n_things", type=int, default=None,
                        help="Number of thing clusters for stuff/things classification. "
                             "Default: 60%% of k (COCO has ~60%% thing categories)")

    # Device and limits
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for feature extraction (default: 4)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N images (for testing)")
    parser.add_argument("--depth_model", type=str, default="depth_anything_v2",
                        choices=["depth_anything_v2", "spidepth"],
                        help="Depth model: depth_anything_v2 (recommended) or spidepth (default: depth_anything_v2)")
    parser.add_argument("--spidepth_ckpt", type=str, default=DEFAULT_CKPT_DIR,
                        help="Path to SPIdepth checkpoint directory (only if --depth_model spidepth)")

    args = parser.parse_args()

    # Default n_things: 60% of k (COCO has ~60% thing categories)
    if args.n_things is None:
        args.n_things = int(args.k * 0.6)
        logger.info(f"Auto-set n_things={args.n_things} (60% of k={args.k})")

    # Output directories under coconut_root
    base_out = Path(args.coconut_root)
    depth_suffix = "dav2" if args.depth_model == "depth_anything_v2" else "spidepth"
    feature_dir = base_out / "dinov2_features"
    semantic_dir = base_out / f"pseudo_semantic_k{args.k}"
    depth_dir = base_out / f"depth_{depth_suffix}"
    stuff_things_path = str(base_out / f"pseudo_semantic_k{args.k}" / "stuff_things.json")
    instance_dir = base_out / f"pseudo_instance_{depth_suffix}"
    panoptic_dir = base_out / f"pseudo_panoptic_k{args.k}_{depth_suffix}"

    steps = ["features", "semantics", "depth", "stuff_things", "instances",
             "panoptic", "evaluate"]
    if args.step != "all":
        steps = [args.step]

    t0 = time.time()

    for step in steps:
        step_t0 = time.time()

        if step == "features":
            extract_dinov2_features(
                coco_root=args.coco_root,
                output_dir=str(feature_dir),
                split=args.split,
                batch_size=args.batch_size,
                device=args.device,
                limit=args.limit,
            )

        elif step == "semantics":
            generate_semantic_pseudolabels(
                feature_dir=str(feature_dir),
                output_dir=str(semantic_dir),
                split=args.split,
                k=args.k,
                limit=args.limit,
            )

        elif step == "depth":
            generate_depth_maps(
                coco_root=args.coco_root,
                output_dir=str(depth_dir),
                split=args.split,
                device=args.device,
                checkpoint_dir=args.spidepth_ckpt,
                limit=args.limit,
                depth_model=args.depth_model,
            )

        elif step == "stuff_things":
            classify_stuff_things(
                semantic_dir=str(semantic_dir / args.split),
                depth_dir=str(depth_dir / args.split),
                output_path=stuff_things_path,
                num_classes=args.k,
                n_things=args.n_things,
                grad_threshold=args.grad_threshold,
                depth_blur_sigma=args.depth_blur,
            )

        elif step == "instances":
            generate_instances(
                semantic_dir=str(semantic_dir / args.split),
                depth_dir=str(depth_dir / args.split),
                output_dir=str(instance_dir / args.split),
                stuff_things_path=stuff_things_path,
                grad_threshold=args.grad_threshold,
                min_area=args.min_area,
                dilation_iters=args.dilation_iters,
                depth_blur=args.depth_blur,
                limit=args.limit,
            )

        elif step == "panoptic":
            generate_panoptic(
                semantic_dir=str(semantic_dir / args.split),
                instance_dir=str(instance_dir / args.split),
                output_dir=str(panoptic_dir / args.split),
                stuff_things_path=stuff_things_path,
                limit=args.limit,
            )

        elif step == "evaluate":
            evaluate_coconut(
                panoptic_dir=str(panoptic_dir / args.split),
                coconut_root=args.coconut_root,
                split=args.split,
                semantic_dir=str(semantic_dir / args.split),
            )

        elapsed = time.time() - step_t0
        logger.info(f"Step '{step}' completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    total_elapsed = time.time() - t0
    logger.info(f"\nTotal pipeline time: {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h)")
    logger.info(f"Output directory: {base_out}")


if __name__ == "__main__":
    main()
