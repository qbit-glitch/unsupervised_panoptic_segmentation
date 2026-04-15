#!/usr/bin/env python3
"""Generate depth-conditioned overclustered semantic pseudo-labels.

Extends CAUSE k-means overclustering with depth-derived features concatenated
to the 90D Segment_TR features before clustering. Four depth variants:
  - raw:        1D depth scalar (total 91D)
  - sobel:      3D Sobel gradients dx, dy, magnitude (total 93D)
  - sinusoidal: 16D sinusoidal positional encoding (total 106D)
  - none:       baseline without depth (total 90D, reproduces existing pipeline)

Alpha parameter controls depth feature scaling before concatenation.

Usage:
    # Baseline (no depth, reproduces existing k=300):
    python mbps_pytorch/generate_depth_overclustered_semantics.py \
        --cityscapes_root /path/to/cityscapes --variant none --k 300

    # Depth-conditioned with Sobel gradients:
    python mbps_pytorch/generate_depth_overclustered_semantics.py \
        --cityscapes_root /path/to/cityscapes --variant sobel --alpha 1.0

    # Sweep alpha for sinusoidal encoding:
    for a in 0.1 0.5 1.0 2.0; do
        python mbps_pytorch/generate_depth_overclustered_semantics.py \
            --cityscapes_root /path/to/cityscapes --variant sinusoidal --alpha $a
    done
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import sobel
from sklearn.cluster import MiniBatchKMeans
from torchvision import transforms
from tqdm import tqdm

# Add CAUSE repo to path
CAUSE_DIR = str(Path(__file__).resolve().parent.parent / "refs" / "cause")
if CAUSE_DIR not in sys.path:
    sys.path.insert(0, CAUSE_DIR)

from models.dinov2vit import dinov2_vit_base_14
from modules.segment import Segment_TR
from modules.segment_module import Cluster, transform

try:
    import pydensecrf.densecrf as dcrf
    import pydensecrf.utils as crf_utils
    HAS_CRF = True
except ImportError:
    HAS_CRF = False

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO,
)
logger = logging.getLogger(__name__)

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
DEFAULT_FREQ_BANDS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)  # 8 bands → 16D


# ─── Depth Encoding ───

def sinusoidal_depth_encoding(
    depth_flat: np.ndarray,
    freq_bands: Sequence[float] = DEFAULT_FREQ_BANDS,
) -> np.ndarray:
    """Encode depth values with sinusoidal positional encoding.

    Args:
        depth_flat: (N,) array of depth values in [0, 1].

    Returns:
        (N, 2*len(freq_bands)) encoded features.
    """
    encodings = []
    for freq in freq_bands:
        encodings.append(np.sin(freq * np.pi * depth_flat))
        encodings.append(np.cos(freq * np.pi * depth_flat))
    return np.stack(encodings, axis=-1)


def compute_sobel_gradients(depth_2d: np.ndarray) -> np.ndarray:
    """Compute Sobel gradients + magnitude on a 2D depth map.

    Returns:
        (H*W, 3) array of (gx, gy, magnitude).
    """
    gx = sobel(depth_2d, axis=1)
    gy = sobel(depth_2d, axis=0)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.stack([gx.ravel(), gy.ravel(), mag.ravel()], axis=-1)


def downsample_depth(
    depth: np.ndarray, target_h: int, target_w: int,
) -> np.ndarray:
    """Downsample depth map via block averaging.

    Args:
        depth: (H, W) float32.

    Returns:
        (target_h, target_w) downsampled depth.
    """
    src_h, src_w = depth.shape
    block_h = src_h // target_h
    block_w = src_w // target_w
    trimmed = depth[:target_h * block_h, :target_w * block_w]
    blocks = trimmed.reshape(target_h, block_h, target_w, block_w)
    return blocks.mean(axis=(1, 3))


def encode_depth_features(
    depth_2d: np.ndarray, variant: str,
) -> np.ndarray:
    """Encode a downsampled depth map into feature vectors.

    Args:
        depth_2d: (H, W) depth map in [0, 1].
        variant: 'raw' (1D), 'sobel' (3D), 'sinusoidal' (16D).

    Returns:
        (H*W, D) depth features.
    """
    depth_flat = depth_2d.ravel()

    if variant == "raw":
        return depth_flat[:, np.newaxis]  # (N, 1)
    elif variant == "sobel":
        return compute_sobel_gradients(depth_2d)  # (N, 3)
    elif variant == "sinusoidal":
        return sinusoidal_depth_encoding(depth_flat)  # (N, 16)
    else:
        raise ValueError(f"Unknown depth variant: {variant}")


def get_depth_dim(variant: str) -> int:
    """Return feature dimension for a depth encoding variant."""
    dims = {"none": 0, "raw": 1, "sobel": 3, "sinusoidal": 16}
    return dims[variant]


# ─── CAUSE Model Loading (reused from generate_overclustered_semantics.py) ───

def _remap_to_trainids(gt: np.ndarray) -> np.ndarray:
    remapped = np.full_like(gt, IGNORE_LABEL, dtype=np.uint8)
    for raw_id, train_id in _CS_ID_TO_TRAIN.items():
        remapped[gt == raw_id] = train_id
    return remapped


def get_cityscapes_images(cityscapes_root: str, split: str):
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


def load_cause_models(checkpoint_dir: str, device: torch.device):
    from types import SimpleNamespace
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

    seg_path = os.path.join(
        checkpoint_dir, "CAUSE", "cityscapes",
        "dinov2_vit_base_14", "2048", "segment_tr.pth",
    )
    segment = Segment_TR(cause_args).to(device)
    seg_state = torch.load(seg_path, map_location="cpu", weights_only=True)
    segment.load_state_dict(seg_state, strict=False)
    segment.eval()

    mod_path = os.path.join(
        checkpoint_dir, "CAUSE", "cityscapes", "modularity",
        "dinov2_vit_base_14", "2048", "modular.npy",
    )
    cb = torch.from_numpy(np.load(mod_path)).to(device)
    segment.head.codebook = cb
    segment.head_ema.codebook = cb

    return net, segment, cause_args


def extract_cause_features_crop(net, segment, img_tensor: torch.Tensor):
    with torch.no_grad():
        feat = net(img_tensor)[:, 1:, :]
        feat_flip = net(img_tensor.flip(dims=[3]))[:, 1:, :]
        seg_feat = transform(segment.head_ema(feat))
        seg_feat_flip = transform(segment.head_ema(feat_flip))
        seg_feat = (seg_feat + seg_feat_flip.flip(dims=[3])) / 2
    return seg_feat  # (1, 90, 23, 23)


def sliding_window_features(net, segment, img_resized: torch.Tensor, crop_size: int = 322):
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

            feat_2d = extract_cause_features_crop(net, segment, crop)
            feat_up = F.interpolate(feat_2d, size=(crop_size, crop_size),
                                    mode='bilinear', align_corners=False)[0]
            feat_sum[:, y_pos:y_end, x_pos:x_end] += feat_up[:, :ch, :cw]
            count[:, y_pos:y_end, x_pos:x_end] += 1

    return feat_sum / count.clamp(min=1)


def dense_crf(image_np: np.ndarray, logits_np: np.ndarray, max_iter: int = 10):
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


# ─── Depth-Conditioned K-means ───

def load_depth_map(
    cityscapes_root: str, depth_subdir: str, split: str,
    city: str, stem: str,
) -> Optional[np.ndarray]:
    """Load a DepthPro .npy depth map. Returns (512, 1024) or None."""
    npy_path = os.path.join(cityscapes_root, depth_subdir, split, city, f"{stem}.npy")
    if not os.path.isfile(npy_path):
        logger.warning("Depth not found: %s", npy_path)
        return None
    return np.load(npy_path).astype(np.float32)


def fit_kmeans_with_depth(
    net, segment, cityscapes_root: str, depth_subdir: str,
    device: torch.device, crop_size: int, patch_size: int,
    k: int = 300, variant: str = "none", alpha: float = 1.0,
    max_per_image: int = 2000, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit k-means on concatenated [90D CAUSE | alpha*depth_D] features.

    Returns:
        (centroids_norm, cluster_to_class) where centroids_norm is (k, total_D).
    """
    depth_d = get_depth_dim(variant)
    total_d = 90 + depth_d
    logger.info("Fitting k-means: k=%d, variant=%s, alpha=%.2f, total_dim=%d",
                k, variant, alpha, total_d)

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    images = get_cityscapes_images(cityscapes_root, "val")
    rng = np.random.RandomState(seed)

    all_feats = []
    all_labels = []

    for entry in tqdm(images, desc="Extracting features"):
        img_pil = Image.open(entry["img_path"]).convert("RGB")
        orig_w, orig_h = img_pil.size
        scale = crop_size / min(orig_h, orig_w)
        new_h = (int(round(orig_h * scale)) // patch_size) * patch_size
        new_w = (int(round(orig_w * scale)) // patch_size) * patch_size

        img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
        img_tensor = normalize(transforms.ToTensor()(img_resized)).unsqueeze(0).to(device)

        feat_map = sliding_window_features(net, segment, img_tensor, crop_size)
        ph, pw = new_h // patch_size, new_w // patch_size
        feat_ds = F.adaptive_avg_pool2d(
            feat_map.unsqueeze(0), (ph, pw),
        ).squeeze(0)
        feats_90 = feat_ds.cpu().numpy().reshape(90, -1).T.astype(np.float64)

        # Concatenate depth features if variant != none
        if variant != "none":
            depth_map = load_depth_map(
                cityscapes_root, depth_subdir, "val", entry["city"], entry["stem"],
            )
            if depth_map is None:
                depth_feats = np.zeros((feats_90.shape[0], depth_d), dtype=np.float64)
            else:
                depth_ds = downsample_depth(depth_map, ph, pw)
                depth_feats = encode_depth_features(depth_ds, variant).astype(np.float64)
                depth_feats *= alpha
            feats = np.concatenate([feats_90, depth_feats], axis=1)
        else:
            feats = feats_90

        # Load GT for majority-vote mapping
        gt_path = os.path.join(
            cityscapes_root, "gtFine", "val", entry["city"],
            f"{entry['stem']}_gtFine_labelIds.png",
        )
        gt_raw = np.array(Image.open(gt_path))
        gt = _remap_to_trainids(gt_raw)
        gt_ds = np.array(Image.fromarray(gt).resize((pw, ph), Image.NEAREST)).flatten()

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
    logger.info("Collected %d feature vectors (dim=%d)", len(all_feats), all_feats.shape[1])

    # L2 normalize
    norms = np.linalg.norm(all_feats, axis=1, keepdims=True)
    feats_norm = all_feats / np.maximum(norms, 1e-8)

    # Fit k-means
    logger.info("Fitting MiniBatchKMeans (k=%d)...", k)
    kmeans = MiniBatchKMeans(
        n_clusters=k, batch_size=10000, max_iter=300,
        random_state=42, n_init=3,
    )
    kmeans.fit(feats_norm)

    # Majority-vote mapping
    cluster_labels = kmeans.predict(feats_norm)
    conf = np.zeros((k, NUM_CLASSES), dtype=np.int64)
    for cl, gt in zip(cluster_labels, all_labels):
        if gt < NUM_CLASSES:
            conf[cl, gt] += 1
    cluster_to_class = np.argmax(conf, axis=1).astype(np.uint8)

    centroids = kmeans.cluster_centers_
    centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

    for c in range(NUM_CLASSES):
        n_clusters = int((cluster_to_class == c).sum())
        logger.info("  %15s: %d clusters", _CS_CLASS_NAMES[c], n_clusters)

    return centroids_norm, cluster_to_class


def predict_with_depth_kmeans(
    feat_map: torch.Tensor,
    depth_features: Optional[torch.Tensor],
    centroids_norm_torch: torch.Tensor,
    cluster_to_class_torch: torch.Tensor,
    raw_clusters: bool = False,
) -> torch.Tensor:
    """Predict using concatenated [90D | depth_D] centroids.

    Args:
        feat_map: (90, H, W) CAUSE features.
        depth_features: (D, H, W) depth features or None.
        centroids_norm_torch: (k, total_D) L2-normalized centroids.
        cluster_to_class_torch: (k,) mapping.
        raw_clusters: If True, return (k, H, W) raw similarities.

    Returns:
        (NUM_CLASSES, H, W) or (k, H, W) logits.
    """
    if depth_features is not None:
        combined = torch.cat([feat_map, depth_features], dim=0)  # (total_D, H, W)
    else:
        combined = feat_map

    C, H, W = combined.shape
    feat_norm = F.normalize(combined, dim=0)
    sim = centroids_norm_torch @ feat_norm.reshape(C, -1)  # (k, H*W)

    if raw_clusters:
        return sim.reshape(-1, H, W)

    k = centroids_norm_torch.shape[0]
    class_logits = torch.zeros(NUM_CLASSES, H * W, device=feat_map.device)
    for cls_id in range(NUM_CLASSES):
        cluster_mask = cluster_to_class_torch == cls_id
        if cluster_mask.any():
            class_logits[cls_id] = sim[cluster_mask].max(dim=0).values

    return class_logits.reshape(NUM_CLASSES, H, W)


def generate_pseudolabels_with_depth(
    net, segment, cityscapes_root: str, depth_subdir: str,
    split: str, device: torch.device,
    centroids_norm: np.ndarray, cluster_to_class: np.ndarray,
    output_dir: str, variant: str, alpha: float,
    crop_size: int = 322, patch_size: int = 14,
    skip_crf: bool = False, limit: int = 0,
    raw_clusters: bool = False,
) -> None:
    """Generate pseudo-labels with depth-augmented features."""
    normalize_tf = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    images = get_cityscapes_images(cityscapes_root, split)
    if limit > 0:
        images = images[:limit]

    centroids_torch = torch.from_numpy(centroids_norm).float().to(device)
    c2c_torch = torch.from_numpy(cluster_to_class).long().to(device)

    k = centroids_norm.shape[0]
    num_out = k if raw_clusters else NUM_CLASSES
    mode_str = f"raw k={k}" if raw_clusters else f"{NUM_CLASSES}-class mapped"
    logger.info("Generating %s pseudo-labels: %d images, variant=%s, alpha=%.2f",
                mode_str, len(images), variant, alpha)

    for entry in tqdm(images, desc=f"Generating {split}"):
        img_pil = Image.open(entry["img_path"]).convert("RGB")
        orig_w, orig_h = img_pil.size

        scale = crop_size / min(orig_h, orig_w)
        new_h = (int(round(orig_h * scale)) // patch_size) * patch_size
        new_w = (int(round(orig_w * scale)) // patch_size) * patch_size

        img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
        img_tensor = normalize_tf(transforms.ToTensor()(img_resized)).unsqueeze(0).to(device)

        feat_map = sliding_window_features(net, segment, img_tensor, crop_size)

        # Prepare depth features at pixel resolution
        depth_features = None
        if variant != "none":
            depth_map = load_depth_map(
                cityscapes_root, depth_subdir, split, entry["city"], entry["stem"],
            )
            if depth_map is not None:
                # Interpolate depth to pixel resolution matching feat_map
                depth_resized = np.array(
                    Image.fromarray(depth_map).resize((new_w, new_h), Image.BILINEAR)
                )
                depth_enc = encode_depth_features(depth_resized, variant)  # (H*W, D)
                depth_enc = depth_enc.reshape(new_h, new_w, -1).transpose(2, 0, 1)  # (D, H, W)
                depth_features = torch.from_numpy(depth_enc * alpha).float().to(device)

        logits = predict_with_depth_kmeans(
            feat_map, depth_features, centroids_torch, c2c_torch,
            raw_clusters=raw_clusters,
        )

        if not skip_crf and HAS_CRF:
            img_np = np.array(img_resized)
            softmax_logits = F.softmax(logits * 5, dim=0).cpu().numpy()
            crf_result = dense_crf(img_np, softmax_logits)
            pred = np.argmax(crf_result, axis=0).astype(np.uint8)
        else:
            pred = logits.argmax(dim=0).cpu().numpy().astype(np.uint8)

        pred_pil = Image.fromarray(pred)
        pred_full = np.array(pred_pil.resize((orig_w, orig_h), Image.NEAREST))

        city_dir = os.path.join(output_dir, split, entry["city"])
        os.makedirs(city_dir, exist_ok=True)
        out_path = os.path.join(city_dir, f"{entry['stem']}.png")
        Image.fromarray(pred_full.astype(np.uint8)).save(out_path)

    logger.info("Generated %d pseudo-labels to %s/%s/", len(images), output_dir, split)


# ─── Main ───

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Depth-conditioned overclustered semantic pseudo-labels",
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=300)
    parser.add_argument("--depth_subdir", type=str, default="depth_depthpro")
    parser.add_argument(
        "--variant", type=str, default="none",
        choices=["none", "raw", "sobel", "sinusoidal"],
    )
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Scaling for depth features before concatenation")
    parser.add_argument("--output_subdir", type=str, default=None)
    parser.add_argument("--skip_crf", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--raw_clusters", action="store_true")
    parser.add_argument("--load_centroids", type=str, default=None)
    args = parser.parse_args()

    if args.output_subdir is None:
        if args.variant == "none":
            args.output_subdir = f"pseudo_semantic_overclustered_k{args.k}"
        else:
            args.output_subdir = (
                f"pseudo_semantic_depth_{args.variant}_a{args.alpha}_k{args.k}"
            )

    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(
            Path(__file__).resolve().parent.parent / "refs" / "cause"
        )

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    output_dir = os.path.join(args.cityscapes_root, args.output_subdir)
    logger.info("Device: %s", device)
    logger.info("Variant: %s, alpha: %.2f, k: %d", args.variant, args.alpha, args.k)
    logger.info("Output: %s", output_dir)

    net, segment, cause_args = load_cause_models(args.checkpoint_dir, device)

    if args.load_centroids:
        logger.info("Loading centroids from %s", args.load_centroids)
        data = np.load(args.load_centroids)
        centroids_norm = data["centroids"]
        cluster_to_class = data["cluster_to_class"]
    else:
        centroids_norm, cluster_to_class = fit_kmeans_with_depth(
            net, segment, args.cityscapes_root, args.depth_subdir,
            device, cause_args.crop_size, cause_args.patch_size,
            k=args.k, variant=args.variant, alpha=args.alpha,
        )
        os.makedirs(output_dir, exist_ok=True)
        centroids_path = os.path.join(output_dir, "kmeans_centroids.npz")
        np.savez(centroids_path, centroids=centroids_norm,
                 cluster_to_class=cluster_to_class,
                 variant=args.variant, alpha=args.alpha)
        logger.info("Saved centroids to %s", centroids_path)

    generate_pseudolabels_with_depth(
        net, segment, args.cityscapes_root, args.depth_subdir,
        args.split, device, centroids_norm, cluster_to_class,
        output_dir, args.variant, args.alpha,
        cause_args.crop_size, cause_args.patch_size,
        skip_crf=args.skip_crf, limit=args.limit,
        raw_clusters=args.raw_clusters,
    )

    logger.info("Done! Output: %s/%s/", output_dir, args.split)
    logger.info(
        "\nEvaluate:\n"
        "  python mbps_pytorch/evaluate_cascade_pseudolabels.py \\\n"
        "    --cityscapes_root %s --split %s \\\n"
        "    --semantic_subdir %s \\\n"
        "    --num_clusters %d --skip_instance",
        args.cityscapes_root, args.split, args.output_subdir, args.k,
    )


if __name__ == "__main__":
    main()
