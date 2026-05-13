#!/usr/bin/env python3
"""Classify semantic clusters into stuff vs things using DINOv3 CLS attention.

Uses CUPS-style overlap ratio: for each semantic class k, compute what fraction
of its pixels fall inside the DINOv3 CLS attention saliency mask. DINO's
self-attention naturally highlights foreground/thing objects.

  overlap_ratio_k = pixels_of_class_k_inside_saliency / total_pixels_of_class_k

High ratio -> thing (salient foreground), low ratio -> stuff (background).

This replaces the depth-gradient-based classify_stuff_things.py which achieved
only 53% accuracy because building/vegetation/sky have high depth variation.

Usage:
    python mbps_pytorch/classify_stuff_things_attention.py \
        --image_dir /data/cityscapes/leftImg8bit/train \
        --semantic_dir /data/cityscapes/pseudo_semantic_cause_trainid/train \
        --output_path /data/cityscapes/pseudo_semantic_cause_trainid/stuff_things_attention.json
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]


# --------------------------------------------------------------------------- #
# DINOv3 CLS Attention Extractor
# --------------------------------------------------------------------------- #
class CLSAttentionExtractor(torch.nn.Module):
    """Extract CLS-to-patch attention from DINOv3's last transformer layer.

    Uses Q/K projection hooks because DINOv3's HuggingFace implementation
    does not expose output_attentions. Computes per-head CLS attention and
    averages over heads.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__()
        logger.info(f"Loading DINOv3 model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.model.to(device)

        config = self.model.config
        self.embed_dim = getattr(config, "hidden_size", 768)
        self.patch_size = getattr(config, "patch_size", 16)
        self.num_registers = getattr(config, "num_register_tokens", 4)
        self.skip_tokens = 1 + self.num_registers  # CLS + registers
        self.num_heads = getattr(config, "num_attention_heads", 12)
        self.head_dim = self.embed_dim // self.num_heads

        self._q_output = None
        self._k_output = None
        self._hooks = self._register_qk_hooks()

        logger.info(
            f"  embed_dim={self.embed_dim}, patch_size={self.patch_size}, "
            f"heads={self.num_heads}, registers={self.num_registers}"
        )

    def _find_module(self, path: str):
        module = self.model
        for part in path.split("."):
            module = getattr(module, part)
        return module

    def _register_qk_hooks(self):
        num_layers = getattr(self.model.config, "num_hidden_layers", 12)
        last = num_layers - 1

        patterns = [
            ("layer.{}.attention.q_proj", "layer.{}.attention.k_proj"),
            ("encoder.layer.{}.attention.attention.query",
             "encoder.layer.{}.attention.attention.key"),
        ]

        for q_pat, k_pat in patterns:
            try:
                q_mod = self._find_module(q_pat.format(last))
                k_mod = self._find_module(k_pat.format(last))
                logger.info(f"  Hooked Q at: model.{q_pat.format(last)}")
                logger.info(f"  Hooked K at: model.{k_pat.format(last)}")

                def hook_q(mod, inp, out):
                    self._q_output = out

                def hook_k(mod, inp, out):
                    self._k_output = out

                hq = q_mod.register_forward_hook(hook_q)
                hk = k_mod.register_forward_hook(hook_k)
                return [hq, hk]
            except AttributeError:
                continue

        raise RuntimeError("Could not hook Q/K projections in DINOv3 model")

    @torch.inference_mode()
    def extract(self, pixel_values: torch.Tensor) -> np.ndarray:
        """Extract CLS attention map as (H_patches, W_patches) numpy array.

        Args:
            pixel_values: (1, 3, H, W) normalized image tensor.

        Returns:
            attention_map: (H_patches, W_patches) float32 array, values in [0,1].
        """
        H, W = pixel_values.shape[-2:]
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        self._q_output = None
        self._k_output = None
        _ = self.model(pixel_values)

        Q = self._q_output.float()  # (1, seq_len, embed_dim)
        K = self._k_output.float()

        # Sanitize
        Q = torch.nan_to_num(Q, nan=0.0, posinf=1e4, neginf=-1e4)
        K = torch.nan_to_num(K, nan=0.0, posinf=1e4, neginf=-1e4)

        # Reshape to multi-head: (1, seq_len, D) -> (1, num_heads, seq_len, head_dim)
        seq_len = Q.shape[1]
        Q_h = Q.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K_h = K.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # CLS query (token 0) attending to patch tokens (skip CLS + registers)
        Q_cls = Q_h[:, :, 0:1, :]             # (1, H, 1, d)
        K_patches = K_h[:, :, self.skip_tokens:, :]  # (1, H, N_patches, d)

        # Per-head attention: (1, num_heads, 1, N_patches)
        logits = torch.matmul(Q_cls, K_patches.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = torch.softmax(logits, dim=-1)

        # Average over heads: (N_patches,)
        attn = attn.mean(dim=1).squeeze(0).squeeze(0)  # (N_patches,)

        # Reshape to spatial grid
        attn_map = attn.cpu().numpy().reshape(h_patches, w_patches)
        return attn_map


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def preprocess_image(image_path: str, device: str = "cpu", input_size: int = None):
    """Load and preprocess an image for DINOv3.

    Returns:
        pixel_values: (1, 3, H, W) tensor, ImageNet-normalized.
        orig_size: (H, W) original image size.
    """
    img = Image.open(image_path).convert("RGB")
    orig_size = (img.height, img.width)

    if input_size is not None:
        h, w = img.height, img.width
        if h < w:
            new_h = input_size
            new_w = int(w * input_size / h)
        else:
            new_w = input_size
            new_h = int(h * input_size / w)
        img = img.resize((new_w, new_h), Image.BILINEAR)

    img_np = np.array(img, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img_np = (img_np - mean) / std

    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    # Ensure dimensions divisible by patch_size=16
    _, _, H, W = tensor.shape
    new_H = (H // 16) * 16
    new_W = (W // 16) * 16
    if new_H != H or new_W != W:
        tensor = F.interpolate(tensor, size=(new_H, new_W), mode="bilinear", align_corners=False)

    return tensor.to(device), orig_size


# --------------------------------------------------------------------------- #
# Saliency binarization
# --------------------------------------------------------------------------- #
def otsu_threshold(arr: np.ndarray) -> float:
    """Compute Otsu's threshold on a float array."""
    arr_flat = arr.ravel()
    hist, bin_edges = np.histogram(arr_flat, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return 0.0

    sum_total = (hist * bin_centers).sum()
    sum_bg, w_bg = 0.0, 0
    best_var, threshold = 0.0, bin_centers[0]

    for i in range(256):
        w_bg += hist[i]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += hist[i] * bin_centers[i]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_total - sum_bg) / w_fg
        between_var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if between_var > best_var:
            best_var = between_var
            threshold = bin_centers[i]

    return float(threshold)


def binarize_saliency(attn_map: np.ndarray, method: str = "otsu",
                      percentile: float = 70.0) -> np.ndarray:
    """Binarize attention map into saliency mask.

    Args:
        attn_map: (H, W) float attention map.
        method: "otsu", "percentile", or "mean_std".
        percentile: For percentile method, top X% is salient.

    Returns:
        mask: (H, W) bool array.
    """
    if method == "otsu":
        thresh = otsu_threshold(attn_map)
    elif method == "percentile":
        thresh = np.percentile(attn_map, 100 - percentile)
    elif method == "mean_std":
        thresh = attn_map.mean() + 0.5 * attn_map.std()
    else:
        raise ValueError(f"Unknown saliency method: {method}")

    return attn_map > thresh


# --------------------------------------------------------------------------- #
# Overlap statistics
# --------------------------------------------------------------------------- #
def discover_pairs(image_dir: Path, semantic_dir: Path):
    """Match Cityscapes images to CAUSE-TR semantic pseudo-labels."""
    pairs = []
    for img_path in sorted(image_dir.rglob("*_leftImg8bit.png")):
        rel = img_path.relative_to(image_dir)
        stem = img_path.stem.replace("_leftImg8bit", "")
        sem_path = semantic_dir / rel.parent / f"{stem}.png"
        if sem_path.exists():
            pairs.append((img_path, sem_path))
    return pairs


def compute_overlap_statistics(
    image_dir: str,
    semantic_dir: str,
    extractor: CLSAttentionExtractor,
    num_classes: int,
    max_images: int,
    input_size: int,
    saliency_method: str,
    percentile: float,
    device: str,
) -> dict:
    """Compute per-class overlap ratio between saliency mask and semantic labels."""
    image_dir = Path(image_dir)
    semantic_dir = Path(semantic_dir)

    pairs = discover_pairs(image_dir, semantic_dir)
    if max_images is not None:
        pairs = pairs[:max_images]

    logger.info(f"Computing saliency overlap over {len(pairs)} images...")

    # Accumulators (CUPS-style)
    class_saliency_pixels = np.zeros(num_classes, dtype=np.int64)
    class_total_pixels = np.zeros(num_classes, dtype=np.int64)
    # Extra diagnostics
    class_attention_sum = np.zeros(num_classes, dtype=np.float64)
    num_images_per_class = np.zeros(num_classes, dtype=np.int64)

    for img_path, sem_path in tqdm(pairs, desc="Computing saliency overlap"):
        # Extract CLS attention
        pixel_values, orig_size = preprocess_image(str(img_path), device, input_size)
        attn_map = extractor.extract(pixel_values)  # (H_p, W_p)

        # Load semantic label at original resolution
        semantic = np.array(Image.open(sem_path))  # (H, W) uint8 trainIDs
        sem_H, sem_W = semantic.shape

        # Upsample attention to semantic label resolution
        attn_tensor = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0).float()
        attn_upsampled = F.interpolate(
            attn_tensor, size=(sem_H, sem_W), mode="bilinear", align_corners=False
        ).squeeze().numpy()

        # Binarize
        saliency_mask = binarize_saliency(attn_upsampled, saliency_method, percentile)

        # Accumulate per-class overlap
        for k in range(num_classes):
            class_mask = semantic == k
            n_total = int(class_mask.sum())
            if n_total == 0:
                continue
            class_total_pixels[k] += n_total
            class_saliency_pixels[k] += int((class_mask & saliency_mask).sum())
            class_attention_sum[k] += float(attn_upsampled[class_mask].sum())
            num_images_per_class[k] += 1

    # Compute per-class statistics
    num_processed = len(pairs)
    # Use first semantic label to get image area
    if pairs:
        first_sem = np.array(Image.open(pairs[0][1]))
        image_area = first_sem.shape[0] * first_sem.shape[1]
    else:
        image_area = 1024 * 2048  # Cityscapes default

    stats = {}
    for k in range(num_classes):
        total = int(class_total_pixels[k])
        salient = int(class_saliency_pixels[k])
        n_imgs = int(num_images_per_class[k])
        if total == 0:
            stats[k] = {
                "overlap_ratio": 0.0,
                "total_pixels": 0,
                "saliency_pixels": 0,
                "mean_attention": 0.0,
                "num_images": 0,
                "avg_coverage": 0.0,
            }
        else:
            # avg_coverage: fraction of image this class covers on average
            avg_coverage = total / (max(n_imgs, 1) * image_area)
            stats[k] = {
                "overlap_ratio": salient / total,
                "total_pixels": total,
                "saliency_pixels": salient,
                "mean_attention": float(class_attention_sum[k] / total),
                "num_images": n_imgs,
                "avg_coverage": avg_coverage,
            }

    return stats


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def classify_stuff_things(stats: dict, num_classes: int, n_things: int,
                          coverage_threshold: float = 0.0) -> dict:
    """Classify stuff vs things by ranking overlap_ratio.

    Two-stage approach when coverage_threshold > 0:
      1. Force classes with avg_coverage > threshold to stuff (large background)
      2. Rank remaining by overlap_ratio, top n_things -> things
    """
    forced_stuff = set()
    scores = {}
    for k in range(num_classes):
        s = stats[k]
        if s["total_pixels"] == 0:
            scores[k] = -999.0
        elif coverage_threshold > 0 and s["avg_coverage"] > coverage_threshold:
            scores[k] = -500.0  # forced stuff due to large coverage
            forced_stuff.add(k)
        else:
            scores[k] = s["overlap_ratio"]

    if forced_stuff:
        forced_names = [CS_NAMES[k] if k < len(CS_NAMES) else str(k)
                        for k in sorted(forced_stuff)]
        logger.info(f"Forced to stuff (coverage > {coverage_threshold:.0%}): "
                    f"{forced_names}")

    # Sort by score, top n_things -> things (from non-forced classes)
    valid = [(k, sc) for k, sc in scores.items() if sc > -999]
    valid.sort(key=lambda x: -x[1])

    thing_set = set()
    for i, (k, sc) in enumerate(valid):
        if sc <= -500:
            break  # don't promote forced stuff
        if i < n_things:
            thing_set.add(k)

    classification = {}
    for k in range(num_classes):
        label = "thing" if k in thing_set else "stuff"
        classification[k] = {"label": label, "score": scores[k]}

    return classification


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Classify stuff vs things using DINOv3 CLS attention saliency"
    )
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to Cityscapes leftImg8bit/train/")
    parser.add_argument("--semantic_dir", type=str, required=True,
                        help="Path to CAUSE-TR semantic pseudo-labels (trainID PNGs)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output JSON path")
    parser.add_argument("--model_name", type=str,
                        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                        help="HuggingFace model name")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--n_things", type=int, default=8,
                        help="Number of thing classes to select")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit images processed (recommend 500 for testing)")
    parser.add_argument("--input_size", type=int, default=512,
                        help="Resize shorter side for DINOv3 inference")
    parser.add_argument("--coverage_threshold", type=float, default=0.10,
                        help="Force classes covering > X%% of image to stuff "
                             "(0 to disable, default: 0.10 = 10%%)")
    parser.add_argument("--saliency_method", type=str, default="otsu",
                        choices=["otsu", "percentile", "mean_std"])
    parser.add_argument("--percentile", type=float, default=70.0,
                        help="For percentile method: top X%% is salient")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda/mps/cpu/auto")

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
    logger.info(f"Using device: {device}")

    # Load model
    extractor = CLSAttentionExtractor(args.model_name, device)

    # Compute overlap statistics
    stats = compute_overlap_statistics(
        image_dir=args.image_dir,
        semantic_dir=args.semantic_dir,
        extractor=extractor,
        num_classes=args.num_classes,
        max_images=args.max_images,
        input_size=args.input_size,
        saliency_method=args.saliency_method,
        percentile=args.percentile,
        device=device,
    )

    # Classify
    classification = classify_stuff_things(
        stats, args.num_classes, args.n_things, args.coverage_threshold
    )

    # Print results
    gt_stuff = set(range(0, 11))
    gt_things = set(range(11, 19))
    correct = 0

    print(f"\nDINOv3 CLS Attention Stuff-Things Classification "
          f"({args.num_classes} classes, top {args.n_things} by overlap_ratio -> things, "
          f"coverage_thresh={args.coverage_threshold:.0%}):")
    print("-" * 115)
    print(f"  {'ID':>3}  {'Name':>15}  {'Label':>5}  {'Ratio':>7}  "
          f"{'Cov%':>6}  {'MeanAttn':>8}  {'TotalPx':>12}  {'SalPx':>12}  "
          f"{'#Imgs':>6}  {'Match':>5}")
    print("-" * 115)

    sorted_classes = sorted(
        range(args.num_classes),
        key=lambda k: classification[k]["score"],
        reverse=True,
    )

    n_stuff = 0
    n_things_count = 0
    for k in sorted_classes:
        c = classification[k]
        s = stats[k]
        name = CS_NAMES[k] if k < len(CS_NAMES) else f"cluster_{k}"

        if c["label"] == "stuff":
            n_stuff += 1
        else:
            n_things_count += 1

        gt_label = "thing" if k in gt_things else "stuff"
        match = "Y" if c["label"] == gt_label else "N"
        if c["label"] == gt_label:
            correct += 1

        cov_pct = s['avg_coverage'] * 100
        forced = "*" if c['score'] == -500.0 else " "
        print(f"  {k:3d}  {name:>15}  {c['label']:>5}  {s['overlap_ratio']:7.4f}  "
              f"{cov_pct:5.1f}%  {s['mean_attention']:8.5f}  {s['total_pixels']:12d}  "
              f"{s['saliency_pixels']:12d}  {s['num_images']:6d}  "
              f"{match} (GT: {gt_label}){forced}")

    print(f"\nResult: {n_stuff} stuff, {n_things_count} things")
    print(f"Accuracy vs GT split: {correct}/{args.num_classes} "
          f"({100 * correct / args.num_classes:.0f}%)")

    stuff_ids = sorted([k for k in range(args.num_classes)
                        if classification[k]["label"] == "stuff"])
    thing_ids = sorted([k for k in range(args.num_classes)
                        if classification[k]["label"] == "thing"])
    print(f"Stuff IDs: {stuff_ids}")
    print(f"Thing IDs: {thing_ids}")

    # Save JSON
    output = {
        "classification": {str(k): v for k, v in classification.items()},
        "statistics": {str(k): v for k, v in stats.items()},
        "stuff_ids": stuff_ids,
        "thing_ids": thing_ids,
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
