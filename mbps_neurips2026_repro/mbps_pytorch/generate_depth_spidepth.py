#!/usr/bin/env python3
"""Generate depth maps using SPIdepth (CVPR 2025) on Cityscapes.

SPIdepth is a self-supervised monocular depth estimation model using
ConvNeXtv2-Huge backbone + Query Transformer decoder. This script runs
inference on Cityscapes images and saves depth maps as .npy files
compatible with the MBPS pipeline.

Output format: float32 .npy files, shape (H, W), values in [0, 1] normalized.

Usage:
    python mbps_pytorch/generate_depth_spidepth.py \
        --cityscapes_root /path/to/cityscapes \
        --output_dir /path/to/cityscapes/depth_spidepth \
        --split both \
        --device mps
"""

import argparse
import logging
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

# Add SPIdepth repo to path
SPIDEPTH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "refs", "spidepth")
sys.path.insert(0, SPIDEPTH_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Default checkpoint path
DEFAULT_CKPT_DIR = os.path.join(SPIDEPTH_DIR, "checkpoints", "cityscapes")

# SPIdepth Cityscapes config (from args_files/args_cvnXt_H_cityscapes_finetune_eval.txt)
SPIDEPTH_CONFIG = {
    "model_type": "cvnxt_L_",
    "backbone": "convnextv2_huge.fcmae_ft_in22k_in1k_384",
    "model_dim": 32,
    "patch_size": 32,
    "dim_out": 64,
    "query_nums": 64,
    "dec_channels": [1024, 512, 256, 128],
    "min_depth": 0.01,
    "max_depth": 80.0,
    "height": 320,
    "width": 1024,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate depth maps using SPIdepth (CVPR 2025)"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True,
                        help="Path to Cityscapes dataset root")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: {cityscapes_root}/depth_spidepth)")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CKPT_DIR,
                        help="Path to SPIdepth checkpoint directory")
    parser.add_argument("--split", type=str, default="both",
                        choices=["train", "val", "both"],
                        help="Which split(s) to process")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--target_size", type=int, nargs=2, default=[512, 1024],
                        help="Output depth map size (H W), default: 512 1024")
    parser.add_argument("--flip_augment", action="store_true", default=True,
                        help="Use flip augmentation for better depth (default: True)")
    parser.add_argument("--no_flip_augment", action="store_true",
                        help="Disable flip augmentation")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Log file path")
    return parser.parse_args()


def get_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def load_spidepth_model(checkpoint_dir, device):
    """Load SPIdepth model with Cityscapes checkpoint."""
    from types import SimpleNamespace
    from SQLdepth import SQLdepth

    # Build opt namespace — load_pretrained_model=False to avoid CUDA map_location issues
    opt = SimpleNamespace(
        model_type=SPIDEPTH_CONFIG["model_type"],
        backbone=SPIDEPTH_CONFIG["backbone"],
        model_dim=SPIDEPTH_CONFIG["model_dim"],
        patch_size=SPIDEPTH_CONFIG["patch_size"],
        dim_out=SPIDEPTH_CONFIG["dim_out"],
        query_nums=SPIDEPTH_CONFIG["query_nums"],
        dec_channels=SPIDEPTH_CONFIG["dec_channels"],
        min_depth=SPIDEPTH_CONFIG["min_depth"],
        max_depth=SPIDEPTH_CONFIG["max_depth"],
        height=SPIDEPTH_CONFIG["height"],
        width=SPIDEPTH_CONFIG["width"],
        load_pretrained_model=False,  # We load manually below
        load_pt_folder=checkpoint_dir,
        no_cuda=True,
        num_features=512,
        num_layers=50,
    )

    logger.info(f"Loading SPIdepth model (backbone: {opt.backbone})")
    model = SQLdepth(opt)

    # Manually load weights with map_location='cpu' to avoid CUDA issues on MPS
    encoder_path = os.path.join(checkpoint_dir, "encoder.pth")
    depth_path = os.path.join(checkpoint_dir, "depth.pth")

    logger.info(f"Loading encoder weights from {encoder_path}")
    encoder_state = torch.load(encoder_path, map_location="cpu")
    filtered_enc = {k: v for k, v in encoder_state.items() if k in model.encoder.state_dict()}
    model.encoder.load_state_dict(filtered_enc)

    logger.info(f"Loading depth decoder weights from {depth_path}")
    depth_state = torch.load(depth_path, map_location="cpu")
    model.depth_decoder.load_state_dict(depth_state)

    model.to(device)
    model.eval()
    logger.info(f"Model loaded on {device}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params / 1e6:.1f}M")

    return model


def get_image_paths(cityscapes_root, split):
    """Get all leftImg8bit image paths for a split."""
    img_dir = Path(cityscapes_root) / "leftImg8bit" / split
    if not img_dir.exists():
        logger.error(f"Image directory not found: {img_dir}")
        return []

    paths = sorted(img_dir.rglob("*_leftImg8bit.png"))
    logger.info(f"Found {len(paths)} images in {split} split")
    return paths


def predict_depth(model, image_path, device, feed_height, feed_width, flip_augment=True):
    """Run SPIdepth inference on a single image.

    Returns: depth map as numpy array (H_feed, W_feed), values in metric depth range.
    """
    # Load and preprocess
    input_image = Image.open(image_path).convert("RGB")
    original_width, original_height = input_image.size

    # Resize to network input size
    input_image = input_image.resize((feed_width, feed_height), Image.LANCZOS)
    input_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Forward pass
        output = model(input_tensor)

        if flip_augment:
            # Flip augmentation: average normal and flipped predictions
            input_flipped = torch.flip(input_tensor, dims=[-1])
            output_flipped = model(input_flipped)
            output = (output + torch.flip(output_flipped, dims=[-1])) / 2.0

    # Output is metric depth (B, 1, H, W)
    depth = output.squeeze().cpu().numpy()  # (H, W)
    return depth


def normalize_depth(depth):
    """Normalize depth to [0, 1] range."""
    d_min = depth.min()
    d_max = depth.max()
    if d_max - d_min < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - d_min) / (d_max - d_min)).astype(np.float32)


def generate_depth_maps(model, image_paths, output_dir, device, target_size,
                        flip_augment=True):
    """Generate depth maps for all images."""
    feed_height = SPIDEPTH_CONFIG["height"]
    feed_width = SPIDEPTH_CONFIG["width"]
    target_h, target_w = target_size

    processed = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    for img_path in tqdm(image_paths, desc="Generating depth"):
        try:
            # Build output path preserving city/filename structure
            city = img_path.parent.name
            stem = img_path.stem.replace("_leftImg8bit", "")
            out_dir = Path(output_dir) / city
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}.npy"

            if out_path.exists():
                skipped += 1
                continue

            # Predict depth
            depth = predict_depth(model, img_path, device, feed_height, feed_width,
                                  flip_augment=flip_augment)

            # Resize to target size if different from feed size
            if (depth.shape[0] != target_h) or (depth.shape[1] != target_w):
                depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
                depth_resized = F.interpolate(
                    depth_tensor, size=(target_h, target_w),
                    mode="bilinear", align_corners=False
                )
                depth = depth_resized.squeeze().numpy()

            # Normalize to [0, 1]
            depth_normalized = normalize_depth(depth)

            # Save
            np.save(out_path, depth_normalized)
            processed += 1

        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")
            errors += 1

    elapsed = time.time() - start_time
    total = processed + skipped
    logger.info(f"Depth generation complete: {processed} new, {skipped} skipped, "
                f"{errors} errors, {elapsed:.1f}s ({total / max(elapsed, 1):.1f} img/s)")
    return processed, skipped, errors


def main():
    args = parse_args()

    # Setup logging to file
    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("SPIdepth Depth Map Generation")
    logger.info("=" * 70)
    logger.info(f"Cityscapes root: {args.cityscapes_root}")
    logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
    logger.info(f"Target size: {args.target_size}")
    logger.info(f"Flip augmentation: {not args.no_flip_augment}")

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.cityscapes_root, "depth_spidepth")
    logger.info(f"Output dir: {args.output_dir}")

    # Device
    device = get_device(args.device)
    logger.info(f"Device: {device}")

    # Load model
    model = load_spidepth_model(args.checkpoint_dir, device)

    # Process splits
    flip_aug = not args.no_flip_augment
    splits = ["train", "val"] if args.split == "both" else [args.split]

    total_processed = 0
    total_skipped = 0
    total_errors = 0

    for split in splits:
        logger.info(f"\n--- Processing {split} split ---")
        image_paths = get_image_paths(args.cityscapes_root, split)
        if not image_paths:
            continue

        split_out = os.path.join(args.output_dir, split)
        p, s, e = generate_depth_maps(
            model, image_paths, split_out, device, args.target_size,
            flip_augment=flip_aug
        )
        total_processed += p
        total_skipped += s
        total_errors += e

    logger.info(f"\n{'=' * 70}")
    logger.info(f"TOTAL: {total_processed} processed, {total_skipped} skipped, {total_errors} errors")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
