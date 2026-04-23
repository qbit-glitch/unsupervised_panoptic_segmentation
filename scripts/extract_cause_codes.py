#!/usr/bin/env python3
"""Pre-extract frozen CAUSE 90D codes + depth patches for adapter training.

Runs DINOv2 + Segment_TR once per image and saves the 90D codes as .npy files.
This avoids running the backbone during adapter training (100x faster).

With --save_dino768, also saves raw DINOv2 768D patch features (float16) for
DCFA-X cross-attention input.

Output structure:
    cityscapes/cause_codes_90d/
        train/{city}/{stem}_codes.npy    # (ph, pw, 90) float32
        train/{city}/{stem}_depth.npy    # (ph, pw) float32
        train/{city}/{stem}_dino768.npy  # (ph, pw, 768) float16 [optional]
        val/{city}/{stem}_codes.npy
        val/{city}/{stem}_depth.npy
        val/{city}/{stem}_dino768.npy    # [optional]

Usage:
    python scripts/extract_cause_codes.py \
        --cityscapes_root /path/to/cityscapes

    # Also extract 768D DINOv2 features:
    python scripts/extract_cause_codes.py \
        --cityscapes_root /path/to/cityscapes --save_dino768
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CAUSE_DIR = str(PROJECT_ROOT / "refs" / "cause")
if CAUSE_DIR not in sys.path:
    sys.path.insert(0, CAUSE_DIR)

from mbps_pytorch.generate_depth_overclustered_semantics import (
    extract_cause_features_crop,
    get_cityscapes_images,
    load_cause_models,
    sliding_window_features,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO,
)
logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _reshape_tokens(x: torch.Tensor) -> torch.Tensor:
    """Reshape (B, P, D) → (B, D, sqrt(P), sqrt(P)).

    Same as CAUSE's `transform()` in segment_module.py.
    """
    b, p, d = x.shape
    s = int(math.sqrt(p))
    return x.permute(0, 2, 1).view(b, d, s, s)


def extract_both_crop(
    net: torch.nn.Module,
    segment: torch.nn.Module,
    img_tensor: torch.Tensor,
) -> tuple:
    """Extract both 90D CAUSE codes and 768D DINOv2 features from a crop.

    Args:
        net: DINOv2 backbone.
        segment: Segment_TR module.
        img_tensor: (1, 3, crop_h, crop_w) normalized crop.

    Returns:
        codes_2d: (1, 90, ph, pw) CAUSE codes.
        dino_2d: (1, 768, ph, pw) raw DINOv2 features.
    """
    with torch.no_grad():
        feat = net(img_tensor)[:, 1:, :]  # (1, P, 768)
        feat_flip = net(img_tensor.flip(dims=[3]))[:, 1:, :]

        # 90D CAUSE codes (same as extract_cause_features_crop)
        seg_feat = _reshape_tokens(segment.head_ema(feat))
        seg_feat_flip = _reshape_tokens(segment.head_ema(feat_flip))
        codes_2d = (seg_feat + seg_feat_flip.flip(dims=[3])) / 2

        # 768D DINOv2 features with flip averaging
        dino_2d = _reshape_tokens(feat)
        dino_flip_2d = _reshape_tokens(feat_flip)
        dino_2d = (dino_2d + dino_flip_2d.flip(dims=[3])) / 2

    return codes_2d, dino_2d


def sliding_window_both(
    net: torch.nn.Module,
    segment: torch.nn.Module,
    img_resized: torch.Tensor,
    crop_size: int = 322,
) -> tuple:
    """Sliding-window extraction of both 90D codes and 768D DINOv2 features.

    Args:
        net: DINOv2 backbone.
        segment: Segment_TR module.
        img_resized: (1, 3, H, W) full image tensor.
        crop_size: Crop size for sliding window.

    Returns:
        codes_map: (90, H, W) pixel-resolution CAUSE codes.
        dino_map: (768, H, W) pixel-resolution DINOv2 features.
    """
    _, _, h, w = img_resized.shape
    device = img_resized.device
    stride = crop_size // 2

    y_positions = [0] if h <= crop_size else sorted(set(
        list(range(0, h - crop_size, stride)) + [h - crop_size]))
    x_positions = [0] if w <= crop_size else sorted(set(
        list(range(0, w - crop_size, stride)) + [w - crop_size]))

    codes_sum = torch.zeros(90, h, w, device=device)
    dino_sum = torch.zeros(768, h, w, device=device)
    count = torch.zeros(1, h, w, device=device)

    for y_pos in y_positions:
        for x_pos in x_positions:
            y_end = min(y_pos + crop_size, h)
            x_end = min(x_pos + crop_size, w)
            crop = img_resized[:, :, y_pos:y_end, x_pos:x_end]
            ch, cw = crop.shape[2], crop.shape[3]
            if ch < crop_size or cw < crop_size:
                crop = F.pad(
                    crop, (0, crop_size - cw, 0, crop_size - ch), mode="reflect",
                )

            codes_2d, dino_2d = extract_both_crop(net, segment, crop)

            codes_up = F.interpolate(
                codes_2d, size=(crop_size, crop_size),
                mode="bilinear", align_corners=False,
            )[0]  # (90, crop_size, crop_size)
            dino_up = F.interpolate(
                dino_2d, size=(crop_size, crop_size),
                mode="bilinear", align_corners=False,
            )[0]  # (768, crop_size, crop_size)

            codes_sum[:, y_pos:y_end, x_pos:x_end] += codes_up[:, :ch, :cw]
            dino_sum[:, y_pos:y_end, x_pos:x_end] += dino_up[:, :ch, :cw]
            count[:, y_pos:y_end, x_pos:x_end] += 1

    denom = count.clamp(min=1)
    return codes_sum / denom, dino_sum / denom


def extract_and_save(
    net: torch.nn.Module,
    segment: torch.nn.Module,
    cityscapes_root: str,
    depth_subdir: str,
    output_subdir: str,
    split: str,
    device: torch.device,
    crop_size: int = 322,
    patch_size: int = 14,
    save_dino768: bool = False,
) -> None:
    """Extract 90D codes + depth patches (+ optional 768D) for a split."""
    normalize_tf = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    images = get_cityscapes_images(cityscapes_root, split)
    logger.info("Extracting %s: %d images (save_dino768=%s)", split, len(images), save_dino768)

    out_dir = os.path.join(cityscapes_root, output_subdir, split)

    for entry in tqdm(images, desc=f"Extracting {split}"):
        city = entry["city"]
        stem = entry["stem"]
        city_dir = os.path.join(out_dir, city)
        os.makedirs(city_dir, exist_ok=True)

        codes_path = os.path.join(city_dir, f"{stem}_codes.npy")
        depth_path = os.path.join(city_dir, f"{stem}_depth.npy")
        dino_path = os.path.join(city_dir, f"{stem}_dino768.npy")

        # Skip if all required files exist
        codes_done = os.path.isfile(codes_path) and os.path.isfile(depth_path)
        dino_done = (not save_dino768) or os.path.isfile(dino_path)
        if codes_done and dino_done:
            continue

        # Load image
        img_pil = Image.open(entry["img_path"]).convert("RGB")
        orig_w, orig_h = img_pil.size

        # Resize to patch-aligned dimensions
        scale = crop_size / min(orig_h, orig_w)
        new_h = (int(round(orig_h * scale)) // patch_size) * patch_size
        new_w = (int(round(orig_w * scale)) // patch_size) * patch_size
        img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)

        img_tensor = normalize_tf(transforms.ToTensor()(img_resized)).unsqueeze(0).to(device)
        ph = new_h // patch_size
        pw = new_w // patch_size

        if save_dino768:
            # Extract both 90D and 768D in one pass
            codes_map, dino_map = sliding_window_both(
                net, segment, img_tensor, crop_size,
            )
            # Downsample to patch grid
            codes_ds = F.adaptive_avg_pool2d(
                codes_map.unsqueeze(0), (ph, pw),
            ).squeeze(0)
            dino_ds = F.adaptive_avg_pool2d(
                dino_map.unsqueeze(0), (ph, pw),
            ).squeeze(0)

            codes_np = codes_ds.cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            dino_np = dino_ds.cpu().numpy().transpose(1, 2, 0).astype(np.float16)
            np.save(codes_path, codes_np)
            np.save(dino_path, dino_np)

            del codes_map, dino_map, codes_ds, dino_ds
        else:
            # Original path: 90D only
            with torch.no_grad():
                feat_map = sliding_window_features(
                    net, segment, img_tensor, crop_size,
                )
            feat_ds = F.adaptive_avg_pool2d(
                feat_map.unsqueeze(0), (ph, pw),
            ).squeeze(0)
            codes_np = feat_ds.cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            np.save(codes_path, codes_np)

        # Load and downsample depth
        depth_npy_path = os.path.join(
            cityscapes_root, depth_subdir, split, city, f"{stem}.npy",
        )
        if os.path.isfile(depth_npy_path):
            depth_full = np.load(depth_npy_path).astype(np.float32)
            depth_t = torch.from_numpy(depth_full).unsqueeze(0).unsqueeze(0)
            depth_ds = F.adaptive_avg_pool2d(depth_t, (ph, pw)).squeeze().numpy()
        else:
            depth_ds = np.zeros((ph, pw), dtype=np.float32)

        np.save(depth_path, depth_ds)

    logger.info("Saved to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-extract frozen CAUSE codes")
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--depth_subdir", type=str, default="depth_depthpro")
    parser.add_argument("--output_subdir", type=str, default="cause_codes_90d")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument(
        "--save_dino768", action="store_true",
        help="Also save raw DINOv2 768D features as float16 for DCFA-X.",
    )
    args = parser.parse_args()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(PROJECT_ROOT / "refs" / "cause")

    device = get_device()
    logger.info("Device: %s", device)

    net, segment, cause_args = load_cause_models(args.checkpoint_dir, device)

    t0 = time.time()
    for split in args.splits:
        extract_and_save(
            net, segment, args.cityscapes_root, args.depth_subdir,
            args.output_subdir, split, device,
            crop_size=cause_args.crop_size, patch_size=cause_args.patch_size,
            save_dino768=args.save_dino768,
        )

    logger.info("Total extraction time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
