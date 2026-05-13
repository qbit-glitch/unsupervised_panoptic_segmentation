#!/usr/bin/env python3
"""Generate unsupervised semantic pseudo-labels via K-means on DINOv3 features.

Truly unsupervised: no GT labels used. Uses DINOv3 ViT-L/16 features with
MiniBatchKMeans clustering (K=54 overclustering) + optional CRF refinement.

Pipeline:
  1. Extract DINOv3 ViT-L/16 patch features for all train images
  2. Fit MiniBatchKMeans (K=54) on subsampled features
  3. Predict cluster assignments for all patches
  4. Upsample to pixel resolution + CRF/bilateral refinement
  5. Save as PNG pseudo-labels (uint8, values 0 to K-1)
  6. Optionally evaluate via Hungarian alignment against GT

CUPS ablation (Table 7b): K=54 gives PQ=30.6 (+2.8 over K=27).

Usage:
    python mbps_pytorch/generate_semantic_pseudolabels_unsupervised.py \
        --data_dir /data/cityscapes/leftImg8bit/train \
        --output_dir /data/cityscapes/pseudo_semantic_k54/train \
        --num_clusters 54 \
        --device auto

    # Evaluate with Hungarian alignment:
    python mbps_pytorch/generate_semantic_pseudolabels_unsupervised.py \
        --data_dir /data/cityscapes/leftImg8bit/train \
        --output_dir /data/cityscapes/pseudo_semantic_k54/train \
        --num_clusters 54 \
        --evaluate \
        --gt_dir /data/cityscapes/gtFine/train
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Cityscapes label mappings (for evaluation only)
CITYSCAPES_ID_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}
CS_TRAINID_TO_NAME = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}
IGNORE_LABEL = 255
NUM_GT_CLASSES = 19


# --------------------------------------------------------------------------- #
# Feature Extraction
# --------------------------------------------------------------------------- #
def create_feature_extractor(model_name: str, device: str):
    """Create DINOv3 feature extractor.

    Uses the last hidden state (patch tokens only, CLS + registers removed).
    For K-means clustering, the final hidden state works better than Key
    features because it captures global semantic information rather than
    local patch-to-patch similarity.
    """
    from transformers import AutoModel

    logger.info(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    model.to(device)

    config = model.config
    embed_dim = getattr(config, "hidden_size", 768)
    patch_size = getattr(config, "patch_size", 16)
    num_registers = getattr(config, "num_register_tokens", 4)
    skip_tokens = 1 + num_registers  # CLS + registers

    logger.info(f"  embed_dim={embed_dim}, patch_size={patch_size}, "
                f"registers={num_registers}, skip_tokens={skip_tokens}")

    return model, embed_dim, patch_size, skip_tokens


def preprocess_image(image_path: str, device: str):
    """Load and preprocess an image for DINOv3.

    Returns:
        pixel_values: (1, 3, H, W) tensor, ImageNet-normalized
        orig_size: (H, W) original image size
    """
    img = Image.open(image_path).convert("RGB")
    orig_size = (img.height, img.width)

    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD

    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    # Ensure dims divisible by patch_size=16
    _, _, H, W = tensor.shape
    new_H = (H // 16) * 16
    new_W = (W // 16) * 16
    if new_H != H or new_W != W:
        tensor = F.interpolate(tensor, size=(new_H, new_W), mode="bilinear",
                               align_corners=False)

    return tensor.to(device), orig_size


@torch.inference_mode()
def extract_features_single(model, pixel_values, skip_tokens: int):
    """Extract patch features from DINOv3 for a single image.

    Args:
        model: DINOv3 model
        pixel_values: (1, 3, H, W) tensor
        skip_tokens: number of prefix tokens to skip (CLS + registers)

    Returns:
        features: (N, D) numpy array, N = h_patches * w_patches
        h_patches, w_patches: spatial dimensions
    """
    H, W = pixel_values.shape[-2:]
    h_patches = H // 16
    w_patches = W // 16

    outputs = model(pixel_values, output_hidden_states=True)
    # Use last hidden state for semantic clustering
    last_hidden = outputs.last_hidden_state  # (1, num_tokens, D)
    patch_tokens = last_hidden[:, skip_tokens:, :]  # (1, N, D)

    return patch_tokens.squeeze(0).float().cpu().numpy(), h_patches, w_patches


def extract_all_features(
    image_files: list,
    model,
    skip_tokens: int,
    device: str,
    save_dir: str = None,
):
    """Extract features for all images.

    If save_dir is provided, saves per-image .npy files and returns file paths.
    Otherwise returns features in memory (requires ~23 GB for 2975 images).

    Returns:
        all_features: (total_patches, D) array or None if save_dir is used
        per_image_info: list of (h_patches, w_patches, n_patches) per image
        feature_files: list of .npy paths if save_dir is used, else None
    """
    per_image_info = []
    feature_files = [] if save_dir else None
    all_features_list = [] if save_dir is None else None

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i, img_path in enumerate(tqdm(image_files, desc="Extracting features")):
        pixel_values, orig_size = preprocess_image(str(img_path), device)
        features, h_p, w_p = extract_features_single(model, pixel_values, skip_tokens)
        per_image_info.append((h_p, w_p, features.shape[0]))

        if save_dir:
            feat_path = os.path.join(save_dir, f"features_{i:05d}.npy")
            np.save(feat_path, features)
            feature_files.append(feat_path)
        else:
            all_features_list.append(features)

        # Periodic memory cleanup
        if i % 100 == 0 and device != "cpu":
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

    if all_features_list is not None:
        all_features = np.concatenate(all_features_list, axis=0)
        logger.info(f"All features shape: {all_features.shape} "
                    f"({all_features.nbytes / 1e9:.1f} GB)")
        return all_features, per_image_info, None
    else:
        total_patches = sum(info[2] for info in per_image_info)
        logger.info(f"Saved {len(feature_files)} feature files, "
                    f"total patches: {total_patches}")
        return None, per_image_info, feature_files


# --------------------------------------------------------------------------- #
# K-Means Clustering
# --------------------------------------------------------------------------- #
def fit_kmeans(
    all_features=None,
    feature_files=None,
    per_image_info=None,
    num_clusters: int = 54,
    subsample: int = 500_000,
    seed: int = 42,
):
    """Fit MiniBatchKMeans on features.

    If all_features is None, loads from feature_files with subsampling.

    Returns:
        kmeans: fitted MiniBatchKMeans model
    """
    if all_features is not None:
        # Subsample if needed
        n_total = all_features.shape[0]
        if subsample > 0 and n_total > subsample:
            rng = np.random.RandomState(seed)
            idx = rng.choice(n_total, size=subsample, replace=False)
            fit_features = all_features[idx]
            logger.info(f"Subsampled {subsample}/{n_total} features for K-means fit")
        else:
            fit_features = all_features
    else:
        # Load and subsample from files
        logger.info(f"Loading features from {len(feature_files)} files for subsampling...")
        total_patches = sum(info[2] for info in per_image_info)
        rng = np.random.RandomState(seed)

        # Determine how many patches to sample per image
        samples_per_image = max(1, subsample // len(feature_files))

        sampled = []
        for feat_path, (h_p, w_p, n_patches) in zip(
            tqdm(feature_files, desc="Loading for K-means"),
            per_image_info,
        ):
            feat = np.load(feat_path)
            k = min(samples_per_image, n_patches)
            idx = rng.choice(n_patches, size=k, replace=False)
            sampled.append(feat[idx])

        fit_features = np.concatenate(sampled, axis=0)
        logger.info(f"Collected {fit_features.shape[0]} features for K-means")

    logger.info(f"Fitting MiniBatchKMeans with K={num_clusters}...")
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=10000,
        max_iter=300,
        random_state=seed,
        n_init=3,
        verbose=1,
    )
    kmeans.fit(fit_features)
    logger.info(f"K-means inertia: {kmeans.inertia_:.2f}")

    return kmeans


def predict_all_labels(
    kmeans,
    all_features=None,
    feature_files=None,
    per_image_info=None,
):
    """Predict cluster labels for all images.

    Returns:
        per_image_labels: list of (N,) int arrays, one per image
    """
    per_image_labels = []

    if all_features is not None:
        offset = 0
        for h_p, w_p, n_patches in tqdm(per_image_info, desc="Predicting labels"):
            feat = all_features[offset:offset + n_patches]
            labels = kmeans.predict(feat)
            per_image_labels.append(labels)
            offset += n_patches
    else:
        for feat_path in tqdm(feature_files, desc="Predicting labels"):
            feat = np.load(feat_path)
            labels = kmeans.predict(feat)
            per_image_labels.append(labels)

    return per_image_labels


# --------------------------------------------------------------------------- #
# Refinement: CRF or Bilateral Filter Fallback
# --------------------------------------------------------------------------- #
def _try_import_crf():
    """Try to import pydensecrf."""
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_labels
        return dcrf, unary_from_labels
    except ImportError:
        return None, None


def refine_with_crf(label_map_up, image_rgb, num_classes, dcrf, unary_from_labels):
    """Refine upsampled label map using Dense CRF.

    Args:
        label_map_up: (H, W) int32, nearest-neighbor upsampled labels
        image_rgb: (H, W, 3) uint8 RGB image
        num_classes: K
        dcrf: pydensecrf.densecrf module
        unary_from_labels: pydensecrf utility function

    Returns:
        refined: (H, W) uint8 refined labels
    """
    H, W = label_map_up.shape

    d = dcrf.DenseCRF2D(W, H, num_classes)

    # Unary potentials
    U = unary_from_labels(
        label_map_up.flatten().astype(np.int32),
        num_classes,
        gt_prob=0.9,
        zero_unsure=False,
    )
    d.setUnaryEnergy(U)

    # Bilateral pairwise (appearance + position)
    d.addPairwiseBilateral(
        sxy=67,
        srgb=3,
        rgbim=image_rgb.astype(np.uint8),
        compat=4,
    )

    # Gaussian pairwise (position only)
    d.addPairwiseGaussian(sxy=3, compat=3)

    Q = d.inference(10)
    refined = np.argmax(np.array(Q).reshape(num_classes, H * W), axis=0)
    return refined.reshape(H, W).astype(np.uint8)


def refine_with_bilateral(label_map_up, image_rgb, num_classes):
    """Fallback refinement using guided bilateral-style label smoothing.

    When pydensecrf is not available, this provides boundary refinement
    using appearance-guided majority voting in local neighborhoods.

    Args:
        label_map_up: (H, W) int32, nearest-neighbor upsampled labels
        image_rgb: (H, W, 3) uint8 RGB image
        num_classes: K

    Returns:
        refined: (H, W) uint8 refined labels
    """
    from scipy.ndimage import uniform_filter

    H, W = label_map_up.shape
    refined = label_map_up.copy()

    # Convert image to LAB-like features for appearance comparison
    img_float = image_rgb.astype(np.float32) / 255.0

    # For each pixel, look at a local window and vote based on
    # color similarity — approximates CRF bilateral term
    window = 7
    half = window // 2

    # Compute per-pixel color features (smoothed)
    img_smooth = np.stack([
        uniform_filter(img_float[:, :, c], size=3)
        for c in range(3)
    ], axis=-1)

    # Edge map from image gradients
    gray = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
    gy, gx = np.gradient(gray)
    edge_magnitude = np.sqrt(gx**2 + gy**2)

    # Only refine near patch boundaries (where label changes occur)
    # Detect label boundaries
    label_gy = np.abs(np.diff(label_map_up.astype(np.float32), axis=0))
    label_gx = np.abs(np.diff(label_map_up.astype(np.float32), axis=1))

    boundary_mask = np.zeros((H, W), dtype=bool)
    boundary_mask[:-1, :] |= label_gy > 0
    boundary_mask[1:, :] |= label_gy > 0
    boundary_mask[:, :-1] |= label_gx > 0
    boundary_mask[:, 1:] |= label_gx > 0

    # Dilate boundary by a few pixels
    boundary_mask = ndimage.binary_dilation(boundary_mask, iterations=4)

    # For boundary pixels, do color-guided majority vote
    boundary_coords = np.argwhere(boundary_mask)
    for y, x in boundary_coords:
        y_lo, y_hi = max(0, y - half), min(H, y + half + 1)
        x_lo, x_hi = max(0, x - half), min(W, x + half + 1)

        patch_labels = label_map_up[y_lo:y_hi, x_lo:x_hi]
        patch_colors = img_smooth[y_lo:y_hi, x_lo:x_hi]
        center_color = img_smooth[y, x]

        # Color similarity weights
        color_diff = np.sum((patch_colors - center_color) ** 2, axis=-1)
        weights = np.exp(-color_diff / (2 * 0.05**2))

        # Weighted vote
        unique_labels = np.unique(patch_labels)
        best_label = label_map_up[y, x]
        best_weight = 0
        for lbl in unique_labels:
            w = weights[patch_labels == lbl].sum()
            if w > best_weight:
                best_weight = w
                best_label = lbl

        refined[y, x] = best_label

    return refined.astype(np.uint8)


def upsample_and_refine(
    patch_labels,
    image_path,
    h_patches,
    w_patches,
    num_classes,
    use_crf,
    dcrf_mod,
    unary_fn,
):
    """Upsample patch labels to pixel resolution and optionally refine.

    Args:
        patch_labels: (N,) int, cluster assignments
        image_path: path to RGB image
        h_patches, w_patches: patch grid dims
        num_classes: K
        use_crf: whether to apply refinement
        dcrf_mod: pydensecrf module (or None)
        unary_fn: unary_from_labels function (or None)

    Returns:
        label_map: (H, W) uint8 semantic map
    """
    img = Image.open(image_path).convert("RGB")
    H, W = img.height, img.width
    image_rgb = np.array(img, dtype=np.uint8)

    # Reshape to spatial grid
    label_map = patch_labels.reshape(h_patches, w_patches).astype(np.uint8)

    # Nearest-neighbor upsample to pixel resolution
    label_map_up = np.array(
        Image.fromarray(label_map).resize((W, H), Image.NEAREST),
        dtype=np.int32,
    )

    if not use_crf:
        return label_map_up.astype(np.uint8)

    # Try CRF first, fall back to bilateral
    if dcrf_mod is not None:
        return refine_with_crf(label_map_up, image_rgb, num_classes, dcrf_mod, unary_fn)
    else:
        return refine_with_bilateral(label_map_up, image_rgb, num_classes)


# --------------------------------------------------------------------------- #
# Hungarian Alignment (for evaluation only — uses GT)
# --------------------------------------------------------------------------- #
def compute_hungarian_mapping(
    pred_dir: str,
    gt_dir: str,
    num_pred_classes: int = 54,
    num_gt_classes: int = 19,
):
    """Compute many-to-one mapping from pseudo-classes to GT classes.

    For overclustering (K > num_gt), multiple pseudo-classes map to the same GT.
    Uses greedy argmax on confusion matrix (standard for overclustering eval).

    Returns:
        mapping: dict {pseudo_class_id -> gt_train_id}
    """
    pred_path = Path(pred_dir)
    gt_path = Path(gt_dir)

    pred_files = sorted(pred_path.rglob("*.png"))
    logger.info(f"Computing Hungarian mapping on {len(pred_files)} images...")

    confusion = np.zeros((num_pred_classes, num_gt_classes), dtype=np.int64)

    for pred_file in tqdm(pred_files, desc="Building confusion matrix"):
        pred = np.array(Image.open(pred_file), dtype=np.int32)

        # Find GT file
        city = pred_file.parent.name
        stem = pred_file.stem.replace("_leftImg8bit", "")

        gt_trainid_file = gt_path / city / f"{stem}_gtFine_labelTrainIds.png"
        gt_labelid_file = gt_path / city / f"{stem}_gtFine_labelIds.png"

        if gt_trainid_file.exists():
            gt = np.array(Image.open(gt_trainid_file), dtype=np.int32)
        elif gt_labelid_file.exists():
            gt_raw = np.array(Image.open(gt_labelid_file), dtype=np.int32)
            gt = np.full_like(gt_raw, IGNORE_LABEL)
            for label_id, train_id in CITYSCAPES_ID_TO_TRAINID.items():
                gt[gt_raw == label_id] = train_id
        else:
            continue

        # Resize pred to GT if needed
        if pred.shape != gt.shape:
            pred = np.array(
                Image.fromarray(pred.astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), Image.NEAREST
                ),
                dtype=np.int32,
            )

        valid = gt != IGNORE_LABEL
        for pc in range(num_pred_classes):
            pred_mask = (pred == pc) & valid
            if not pred_mask.any():
                continue
            gt_vals = gt[pred_mask]
            for gc in range(num_gt_classes):
                confusion[pc, gc] += (gt_vals == gc).sum()

    # Many-to-one: each pseudo-class maps to GT class with most overlap
    mapping = {}
    for pc in range(num_pred_classes):
        if confusion[pc].sum() > 0:
            mapping[pc] = int(np.argmax(confusion[pc]))
        else:
            mapping[pc] = 0

    # Log mapping
    for pc, gc in sorted(mapping.items()):
        name = CS_TRAINID_TO_NAME.get(gc, f"class_{gc}")
        count = confusion[pc, gc]
        total = confusion[pc].sum()
        pct = 100 * count / (total + 1e-10)
        logger.info(f"  Cluster {pc:2d} -> {name:15s} ({pct:.1f}% of pixels)")

    return mapping, confusion


def evaluate_with_hungarian(pred_dir, gt_dir, mapping, num_gt_classes=19):
    """Evaluate pseudo-labels after remapping via Hungarian alignment.

    Computes mIoU, pixel accuracy, per-class IoU.

    Returns:
        results: dict with metrics
    """
    pred_path = Path(pred_dir)
    gt_path = Path(gt_dir)
    pred_files = sorted(pred_path.rglob("*.png"))

    total_intersect = np.zeros(num_gt_classes, dtype=np.int64)
    total_union = np.zeros(num_gt_classes, dtype=np.int64)
    total_correct = 0
    total_pixels = 0

    for pred_file in tqdm(pred_files, desc="Evaluating"):
        pred = np.array(Image.open(pred_file), dtype=np.int32)

        city = pred_file.parent.name
        stem = pred_file.stem.replace("_leftImg8bit", "")
        gt_trainid_file = gt_path / city / f"{stem}_gtFine_labelTrainIds.png"
        gt_labelid_file = gt_path / city / f"{stem}_gtFine_labelIds.png"

        if gt_trainid_file.exists():
            gt = np.array(Image.open(gt_trainid_file), dtype=np.int32)
        elif gt_labelid_file.exists():
            gt_raw = np.array(Image.open(gt_labelid_file), dtype=np.int32)
            gt = np.full_like(gt_raw, IGNORE_LABEL)
            for label_id, train_id in CITYSCAPES_ID_TO_TRAINID.items():
                gt[gt_raw == label_id] = train_id
        else:
            continue

        if pred.shape != gt.shape:
            pred = np.array(
                Image.fromarray(pred.astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), Image.NEAREST
                ),
                dtype=np.int32,
            )

        # Remap predictions
        remapped = np.zeros_like(pred)
        for pc, gc in mapping.items():
            remapped[pred == pc] = gc

        valid = gt != IGNORE_LABEL
        for c in range(num_gt_classes):
            pred_c = remapped == c
            gt_c = gt == c
            total_intersect[c] += ((pred_c & gt_c) & valid).sum()
            total_union[c] += (((pred_c | gt_c)) & valid).sum()

        total_correct += ((remapped == gt) & valid).sum()
        total_pixels += valid.sum()

    iou = total_intersect / (total_union + 1e-10)
    valid_classes = total_union > 0
    miou = iou[valid_classes].mean()
    pixel_acc = total_correct / (total_pixels + 1e-10)

    results = {
        "mIoU": round(float(miou) * 100, 2),
        "pixel_accuracy": round(float(pixel_acc) * 100, 2),
        "per_class_iou": {
            CS_TRAINID_TO_NAME[c]: round(float(iou[c]) * 100, 2)
            for c in range(num_gt_classes)
        },
    }

    return results


# --------------------------------------------------------------------------- #
# Main Pipeline
# --------------------------------------------------------------------------- #
def generate_pseudolabels(
    data_dir: str,
    output_dir: str,
    model_name: str,
    device: str,
    num_clusters: int = 54,
    subsample: int = 500_000,
    use_crf: bool = True,
    save_features_dir: str = None,
    load_features_dir: str = None,
    seed: int = 42,
):
    """Full pipeline: extract -> cluster -> upsample -> save."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Collect images
    image_files = sorted(
        list(data_path.rglob("*.png")) + list(data_path.rglob("*.jpg"))
    )
    if not image_files:
        logger.error(f"No images found in {data_dir}")
        return

    logger.info(f"Found {len(image_files)} images")

    # --- Step 1: Feature extraction ---
    if load_features_dir and os.path.isdir(load_features_dir):
        logger.info(f"Loading pre-extracted features from {load_features_dir}")
        feature_files = sorted(Path(load_features_dir).glob("features_*.npy"))

        if len(feature_files) != len(image_files):
            logger.error(f"Feature count ({len(feature_files)}) != image count "
                        f"({len(image_files)}). Re-extract features.")
            return

        # Load info from first file to get dimensions
        sample = np.load(str(feature_files[0]))
        n_patches = sample.shape[0]
        # Infer patch grid from Cityscapes resolution
        # Default: 512x1024 -> 32x64 patches
        per_image_info = []
        for ff in feature_files:
            feat = np.load(str(ff))
            n = feat.shape[0]
            # Try to infer h,w from standard resolutions
            for hp, wp in [(32, 64), (64, 128), (16, 32)]:
                if hp * wp == n:
                    per_image_info.append((hp, wp, n))
                    break
            else:
                # Fallback: assume square-ish
                side = int(np.sqrt(n))
                per_image_info.append((side, n // side, n))

        all_features = None
        feature_files_str = [str(f) for f in feature_files]
    else:
        model, embed_dim, patch_size, skip_tokens = create_feature_extractor(
            model_name, device
        )

        # Decide whether to save to disk or keep in memory
        # ~23 GB for 2975 images with ViT-L (1024-dim)
        use_disk = save_features_dir is not None
        if not use_disk:
            # Estimate memory: 2975 * 2048 * 1024 * 4 bytes = ~23 GB
            est_gb = len(image_files) * 2048 * embed_dim * 4 / 1e9
            if est_gb > 30:
                logger.warning(f"Estimated feature memory: {est_gb:.1f} GB — "
                              f"consider using --save_features")
                save_features_dir = str(output_path.parent / "features_cache")
                use_disk = True
                logger.info(f"Auto-saving features to {save_features_dir}")

        all_features, per_image_info, feature_files_str = extract_all_features(
            image_files, model, skip_tokens, device,
            save_dir=save_features_dir if use_disk else None,
        )

        # Free model memory
        del model
        if device != "cpu":
            torch.mps.empty_cache() if "mps" in device else torch.cuda.empty_cache()

    # --- Step 2: K-means clustering ---
    kmeans = fit_kmeans(
        all_features=all_features,
        feature_files=feature_files_str if all_features is None else None,
        per_image_info=per_image_info,
        num_clusters=num_clusters,
        subsample=subsample,
        seed=seed,
    )

    # Save K-means model
    kmeans_path = output_path / "kmeans_model.npz"
    os.makedirs(output_path, exist_ok=True)
    np.savez(
        str(kmeans_path),
        cluster_centers=kmeans.cluster_centers_,
        n_clusters=num_clusters,
        inertia=kmeans.inertia_,
    )
    logger.info(f"K-means model saved to {kmeans_path}")

    # --- Step 3: Predict labels ---
    per_image_labels = predict_all_labels(
        kmeans,
        all_features=all_features,
        feature_files=feature_files_str if all_features is None else None,
        per_image_info=per_image_info,
    )

    # Free features memory
    del all_features

    # --- Step 4: Upsample + refine + save ---
    dcrf_mod, unary_fn = _try_import_crf()
    if use_crf:
        if dcrf_mod is not None:
            logger.info("Using Dense CRF for refinement")
        else:
            logger.info("pydensecrf not available — using bilateral filter fallback")

    cluster_pixel_counts = np.zeros(num_clusters, dtype=np.int64)

    for i, (img_path, labels) in enumerate(
        tqdm(
            zip(image_files, per_image_labels),
            total=len(image_files),
            desc="Upsampling + saving",
        )
    ):
        h_p, w_p, _ = per_image_info[i]

        label_map = upsample_and_refine(
            labels, str(img_path), h_p, w_p, num_clusters,
            use_crf, dcrf_mod, unary_fn,
        )

        # Save with Cityscapes directory structure
        rel_path = img_path.relative_to(data_path)
        out_path = output_path / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(label_map).save(str(out_path))

        # Accumulate stats
        for c in range(num_clusters):
            cluster_pixel_counts[c] += (label_map == c).sum()

    # --- Step 5: Save statistics ---
    stats = {
        "num_images": len(image_files),
        "num_clusters": num_clusters,
        "model_name": model_name,
        "subsample": subsample,
        "use_crf": use_crf,
        "crf_available": dcrf_mod is not None,
        "cluster_pixel_counts": {
            int(c): int(cluster_pixel_counts[c]) for c in range(num_clusters)
        },
        "empty_clusters": int((cluster_pixel_counts == 0).sum()),
    }
    stats_path = output_path / "generation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Pseudo-labels saved to {output_path}")
    logger.info(f"  Total images: {len(image_files)}")
    logger.info(f"  Empty clusters: {stats['empty_clusters']}/{num_clusters}")

    # Distribution of cluster sizes
    total_pixels = cluster_pixel_counts.sum()
    if total_pixels > 0:
        pcts = cluster_pixel_counts / total_pixels * 100
        logger.info(f"  Largest cluster: {pcts.max():.1f}% of pixels")
        logger.info(f"  Smallest non-empty: {pcts[pcts > 0].min():.2f}% of pixels")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Generate unsupervised semantic pseudo-labels via K-means on DINOv3"
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to images (e.g., leftImg8bit/train)"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for pseudo-label PNGs"
    )
    parser.add_argument(
        "--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="HuggingFace DINOv3 model name"
    )
    parser.add_argument(
        "--num_clusters", type=int, default=54,
        help="Number of K-means clusters (default: 54, CUPS optimal)"
    )
    parser.add_argument(
        "--subsample", type=int, default=500_000,
        help="Number of features to subsample for K-means fit"
    )
    parser.add_argument(
        "--use_crf", action="store_true", default=False,
        help="Apply CRF/bilateral refinement"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: auto, mps, cuda, cpu"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--save_features", default=None,
        help="Directory to save extracted features (for re-use)"
    )
    parser.add_argument(
        "--load_features", default=None,
        help="Directory to load pre-extracted features from"
    )

    # Evaluation arguments
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate pseudo-labels via Hungarian alignment against GT"
    )
    parser.add_argument(
        "--gt_dir", default=None,
        help="Path to GT labels (e.g., gtFine/train) for evaluation"
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Skip generation, only evaluate existing pseudo-labels"
    )

    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    logger.info(f"Device: {device}")
    logger.info(f"K={args.num_clusters}, model={args.model}")

    # --- Generate ---
    if not args.eval_only:
        generate_pseudolabels(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_name=args.model,
            device=device,
            num_clusters=args.num_clusters,
            subsample=args.subsample,
            use_crf=args.use_crf,
            save_features_dir=args.save_features,
            load_features_dir=args.load_features,
            seed=args.seed,
        )

    # --- Evaluate ---
    if args.evaluate or args.eval_only:
        if not args.gt_dir:
            logger.error("--gt_dir required for evaluation")
            sys.exit(1)

        mapping, confusion = compute_hungarian_mapping(
            pred_dir=args.output_dir,
            gt_dir=args.gt_dir,
            num_pred_classes=args.num_clusters,
            num_gt_classes=NUM_GT_CLASSES,
        )

        results = evaluate_with_hungarian(
            pred_dir=args.output_dir,
            gt_dir=args.gt_dir,
            mapping=mapping,
            num_gt_classes=NUM_GT_CLASSES,
        )

        # Print results
        print("\n" + "=" * 60)
        print("UNSUPERVISED SEMANTIC EVALUATION (Hungarian aligned)")
        print("=" * 60)
        print(f"  K = {args.num_clusters}")
        print(f"  mIoU:           {results['mIoU']:.2f}%")
        print(f"  Pixel Accuracy: {results['pixel_accuracy']:.2f}%")
        print(f"\n  Per-class IoU:")
        for name, val in results["per_class_iou"].items():
            print(f"    {name:15s}: {val:.2f}%")
        print("=" * 60)

        # Save results
        eval_results = {
            "method": "unsupervised_kmeans",
            "num_clusters": args.num_clusters,
            "model": args.model,
            "hungarian_mapping": {str(k): v for k, v in mapping.items()},
            "semantic": results,
        }
        eval_path = Path(args.output_dir) / "eval_results.json"
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f"Evaluation results saved to {eval_path}")


if __name__ == "__main__":
    main()
