#!/usr/bin/env python3
"""Extract DINOv3 ViT-B/16 patch features for all Cityscapes images.

Usage:
    # Extract features for train split
    python mbps_pytorch/extract_dinov3_features.py \
        --data_dir /data/cityscapes/leftImg8bit/train \
        --output_dir /data/cityscapes/dinov3_features/train \
        --batch_size 8

    # Extract features for val split
    python mbps_pytorch/extract_dinov3_features.py \
        --data_dir /data/cityscapes/leftImg8bit/val \
        --output_dir /data/cityscapes/dinov3_features/val \
        --batch_size 8

    # Upload to GCS afterwards:
    # gsutil -m rsync -r /data/cityscapes/dinov3_features/ \
    #     gs://mbps-panoptic/datasets/cityscapes/dinov3_features/

Output:
    Per-image .npy files of shape (2048, 768):
      - 2048 = 32 x 64 patch tokens (for 512x1024 input with patch_size=16)
      - 768 = DINOv3 ViT-B embedding dimension
    Also saves spatial dimensions as metadata: (H_patches=32, W_patches=64)

Model: facebook/dinov3-vitb16-pretrain-lvd1689m
  - Patch size: 16
  - Embedding dim: 768
  - 12 heads, 12 blocks
  - 4 register tokens (excluded from output)
  - 1 CLS token (excluded from output)
  - Total tokens for 512x1024: 1 + 4 + 32*64 = 2053
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


def get_image_paths(data_dir: str) -> list:
    """Get all Cityscapes image paths sorted by name.

    Cityscapes structure: data_dir/{city}/{city}_XXXXXX_XXXXXX_leftImg8bit.png
    """
    data_path = Path(data_dir)
    image_paths = sorted(data_path.rglob("*_leftImg8bit.png"))
    if not image_paths:
        # Fallback: try any image format
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            image_paths = sorted(data_path.rglob(ext))
            if image_paths:
                break
    return image_paths


def extract_features(
    data_dir: str,
    output_dir: str,
    model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    batch_size: int = 8,
    image_size: tuple = (512, 1024),
    device: str = "auto",
):
    """Extract DINOv3 patch features for all images.

    Args:
        data_dir: Path to Cityscapes images (leftImg8bit/{split}/).
        output_dir: Where to save .npy feature files.
        model_name: HuggingFace model identifier.
        batch_size: Images per forward pass.
        image_size: (H, W) to resize images to. Must be multiples of 16.
        device: "auto", "cuda", "cpu".
    """
    assert image_size[0] % 16 == 0 and image_size[1] % 16 == 0, \
        f"Image size {image_size} must be multiples of patch_size=16"

    h_patches = image_size[0] // 16  # 32
    w_patches = image_size[1] // 16  # 64
    n_patches = h_patches * w_patches  # 2048

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    # Override processor to use our target size
    processor.size = {"height": image_size[0], "width": image_size[1]}
    processor.crop_size = {"height": image_size[0], "width": image_size[1]}
    processor.do_center_crop = False  # Don't crop, we resize to exact dims

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # Get number of register tokens from model config
    n_register_tokens = getattr(model.config, "num_register_tokens", 4)
    skip_tokens = 1 + n_register_tokens  # CLS + register tokens
    print(f"Model config: patch_size={model.config.patch_size}, "
          f"hidden_size={model.config.hidden_size}, "
          f"register_tokens={n_register_tokens}")
    print(f"Expected output: {n_patches} patch tokens of dim {model.config.hidden_size}")
    print(f"Spatial grid: {h_patches} x {w_patches}")

    # Get image paths
    image_paths = get_image_paths(data_dir)
    print(f"Found {len(image_paths)} images in {data_dir}")
    if not image_paths:
        print("ERROR: No images found!")
        return

    # Save metadata
    metadata = {
        "model_name": model_name,
        "image_size": list(image_size),
        "patch_size": model.config.patch_size,
        "hidden_size": model.config.hidden_size,
        "h_patches": h_patches,
        "w_patches": w_patches,
        "n_patches": n_patches,
        "n_register_tokens": n_register_tokens,
        "num_images": len(image_paths),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Process in batches
    skipped = 0
    for batch_start in tqdm(range(0, len(image_paths), batch_size),
                            desc="Extracting features"):
        batch_paths = image_paths[batch_start:batch_start + batch_size]

        # Check which ones already exist
        batch_to_process = []
        batch_indices = []
        for i, path in enumerate(batch_paths):
            # Output path mirrors input structure: {city}/{image_id}.npy
            rel_path = path.relative_to(data_dir)
            out_path = Path(output_dir) / rel_path.with_suffix(".npy")
            if out_path.exists():
                skipped += 1
                continue
            batch_to_process.append(path)
            batch_indices.append(i)

        if not batch_to_process:
            continue

        # Load and preprocess images
        images = []
        for path in batch_to_process:
            img = Image.open(path).convert("RGB")
            img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
            images.append(img)

        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.inference_mode():
            outputs = model(**inputs)

        # Extract patch tokens (skip CLS + register tokens)
        # last_hidden_state: (B, 1 + n_register + n_patches, hidden_size)
        hidden = outputs.last_hidden_state
        patch_features = hidden[:, skip_tokens:, :]  # (B, n_patches, 768)

        assert patch_features.shape[1] == n_patches, \
            f"Expected {n_patches} patches, got {patch_features.shape[1]}"

        # Save per-image
        patch_features_np = patch_features.cpu().numpy().astype(np.float16)

        for idx, path in enumerate(batch_to_process):
            rel_path = path.relative_to(data_dir)
            out_path = Path(output_dir) / rel_path.with_suffix(".npy")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), patch_features_np[idx])

    if skipped > 0:
        print(f"Skipped {skipped} already-extracted images")
    print(f"Done! Features saved to {output_dir}")
    print(f"Shape per image: ({n_patches}, {model.config.hidden_size}) = "
          f"({h_patches}x{w_patches}, {model.config.hidden_size})")


def main():
    parser = argparse.ArgumentParser(
        description="Extract DINOv3 ViT-B/16 features for Cityscapes images"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to Cityscapes images (e.g., /data/cityscapes/leftImg8bit/train)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Where to save .npy feature files"
    )
    parser.add_argument(
        "--model_name", type=str,
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--image_height", type=int, default=512,
        help="Image height (must be multiple of 16)"
    )
    parser.add_argument(
        "--image_width", type=int, default=1024,
        help="Image width (must be multiple of 16)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cuda, cpu"
    )
    args = parser.parse_args()

    extract_features(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        image_size=(args.image_height, args.image_width),
        device=args.device,
    )


if __name__ == "__main__":
    main()
