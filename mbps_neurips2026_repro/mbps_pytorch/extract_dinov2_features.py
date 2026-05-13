#!/usr/bin/env python3
"""Extract DINOv2 ViT-B/14 (with registers) patch features for Cityscapes images.

Usage:
    # Extract features for both train and val (default)
    python mbps_pytorch/extract_dinov2_features.py \
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes

    # Extract features for a specific split
    python mbps_pytorch/extract_dinov2_features.py \
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
        --split val

    # Custom output subdirectory and device
    python mbps_pytorch/extract_dinov2_features.py \
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
        --output_subdir dinov2_features \
        --device mps \
        --batch_size 4

Output:
    Per-image .npy files of shape (2048, 768) saved as float16:
      - 2048 = 32 x 64 patch tokens (for 448x896 input with patch_size=14)
      - 768 = DINOv2 ViT-B embedding dimension
    Also saves metadata.json per split with spatial dimensions.

Model: dinov2_vitb14_reg (facebook/dinov2, with registers)
  - Patch size: 14
  - Embedding dim: 768
  - 12 heads, 12 blocks
  - 4 register tokens (excluded from output)
  - 1 CLS token (excluded from output)
  - Input: 448x896 -> 32x64 = 2048 patch tokens
  - Total tokens: 1 (CLS) + 4 (registers) + 2048 (patches) = 2053
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# DINOv2 normalization constants (ImageNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Target spatial grid (must match DINOSAUR code: GRID_H=32, GRID_W=64)
GRID_H = 32
GRID_W = 64
PATCH_SIZE = 14
IMAGE_H = GRID_H * PATCH_SIZE  # 448
IMAGE_W = GRID_W * PATCH_SIZE  # 896
N_PATCHES = GRID_H * GRID_W   # 2048

# DINOv2 with registers has 4 register tokens + 1 CLS token
N_REGISTER_TOKENS = 4
SKIP_TOKENS = 1 + N_REGISTER_TOKENS  # 5


def get_image_paths(data_dir: str) -> list:
    """Get all Cityscapes image paths sorted by name.

    Cityscapes structure: data_dir/{city}/{city}_XXXXXX_XXXXXX_leftImg8bit.png
    """
    data_path = Path(data_dir)
    image_paths = sorted(data_path.rglob("*_leftImg8bit.png"))
    if not image_paths:
        # Fallback: try any PNG
        image_paths = sorted(data_path.rglob("*.png"))
    return image_paths


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Resize and normalize a PIL image for DINOv2.

    Args:
        img: PIL Image in RGB format, any size.

    Returns:
        Tensor of shape (3, IMAGE_H, IMAGE_W), normalized with ImageNet stats.
    """
    img = img.resize((IMAGE_W, IMAGE_H), Image.BILINEAR)
    # Convert to float tensor [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    # Normalize with ImageNet stats
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    # HWC -> CHW
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    return tensor


def extract_features(
    cityscapes_root: str,
    output_subdir: str = "dinov2_features",
    splits: list = None,
    batch_size: int = 4,
    device: str = "auto",
):
    """Extract DINOv2 ViT-B/14 (with registers) patch features for Cityscapes.

    Args:
        cityscapes_root: Root of Cityscapes dataset (contains leftImg8bit/).
        output_subdir: Subdirectory name under cityscapes_root for output.
        splits: List of splits to process (e.g., ["train", "val"]).
        batch_size: Images per forward pass.
        device: "auto", "mps", "cuda", or "cpu".
    """
    if splits is None:
        splits = ["train", "val"]

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    # Load DINOv2 ViT-B/14 with registers via torch.hub
    print("Loading DINOv2 ViT-B/14 with registers (dinov2_vitb14_reg)...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    model = model.to(device)
    model.eval()

    # Verify model config
    embed_dim = model.embed_dim
    patch_size = model.patch_size
    num_register_tokens = getattr(model, "num_register_tokens", N_REGISTER_TOKENS)
    print(f"Model config: patch_size={patch_size}, embed_dim={embed_dim}, "
          f"register_tokens={num_register_tokens}")
    assert patch_size == PATCH_SIZE, f"Expected patch_size={PATCH_SIZE}, got {patch_size}"
    assert embed_dim == 768, f"Expected embed_dim=768, got {embed_dim}"
    print(f"Input size: {IMAGE_H}x{IMAGE_W} -> {GRID_H}x{GRID_W} = {N_PATCHES} patches")
    print(f"Output shape per image: ({N_PATCHES}, {embed_dim})")

    # Process each split
    for split in splits:
        data_dir = os.path.join(cityscapes_root, "leftImg8bit", split)
        output_dir = os.path.join(cityscapes_root, output_subdir, split)

        if not os.path.isdir(data_dir):
            print(f"WARNING: {data_dir} does not exist, skipping split '{split}'")
            continue

        os.makedirs(output_dir, exist_ok=True)

        # Get image paths
        image_paths = get_image_paths(data_dir)
        print(f"\n[{split}] Found {len(image_paths)} images in {data_dir}")
        if not image_paths:
            print(f"[{split}] ERROR: No images found, skipping.")
            continue

        # Save metadata
        metadata = {
            "model": "dinov2_vitb14_reg",
            "source": "facebookresearch/dinov2 (torch.hub)",
            "image_size": [IMAGE_H, IMAGE_W],
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "h_patches": GRID_H,
            "w_patches": GRID_W,
            "n_patches": N_PATCHES,
            "n_register_tokens": num_register_tokens,
            "skip_tokens": SKIP_TOKENS,
            "dtype": "float16",
            "num_images": len(image_paths),
            "split": split,
        }
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Process in batches
        skipped = 0
        extracted = 0

        for batch_start in tqdm(range(0, len(image_paths), batch_size),
                                desc=f"[{split}] Extracting DINOv2 features",
                                total=(len(image_paths) + batch_size - 1) // batch_size):
            batch_paths = image_paths[batch_start:batch_start + batch_size]

            # Check which images already have extracted features (resume support)
            batch_to_process = []
            for path in batch_paths:
                rel_path = path.relative_to(data_dir)
                out_path = Path(output_dir) / rel_path.with_suffix(".npy")
                if out_path.exists():
                    skipped += 1
                    continue
                batch_to_process.append(path)

            if not batch_to_process:
                continue

            # Load and preprocess images
            tensors = []
            for path in batch_to_process:
                img = Image.open(path).convert("RGB")
                tensor = preprocess_image(img)
                tensors.append(tensor)

            # Stack into batch: (B, 3, H, W)
            batch_tensor = torch.stack(tensors, dim=0).to(device)

            # Forward pass - get intermediate features with patch tokens
            with torch.inference_mode():
                # DINOv2 forward_features returns dict with:
                #   "x_norm_clstoken": (B, D)
                #   "x_norm_regtokens": (B, n_reg, D)
                #   "x_norm_patchtokens": (B, n_patches, D)
                features_dict = model.forward_features(batch_tensor)
                patch_features = features_dict["x_norm_patchtokens"]  # (B, N_PATCHES, 768)

            assert patch_features.shape[1] == N_PATCHES, (
                f"Expected {N_PATCHES} patch tokens, got {patch_features.shape[1]}. "
                f"Input shape was {batch_tensor.shape}."
            )
            assert patch_features.shape[2] == embed_dim, (
                f"Expected embed_dim={embed_dim}, got {patch_features.shape[2]}"
            )

            # Convert to float16 numpy and save per-image
            patch_features_np = patch_features.cpu().to(torch.float16).numpy()

            for idx, path in enumerate(batch_to_process):
                rel_path = path.relative_to(data_dir)
                out_path = Path(output_dir) / rel_path.with_suffix(".npy")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(out_path), patch_features_np[idx])
                extracted += 1

        print(f"[{split}] Done: extracted={extracted}, skipped={skipped} (already existed)")
        print(f"[{split}] Features saved to {output_dir}")

    print("\nAll done!")
    print(f"Feature shape per image: ({N_PATCHES}, {embed_dim}) = "
          f"({GRID_H}x{GRID_W}, {embed_dim}) as float16")


def main():
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 ViT-B/14 (with registers) features for Cityscapes images"
    )
    parser.add_argument(
        "--cityscapes_root", type=str,
        default="/Users/qbit-glitch/Desktop/datasets/cityscapes",
        help="Root of Cityscapes dataset (contains leftImg8bit/)"
    )
    parser.add_argument(
        "--output_subdir", type=str, default="dinov2_features",
        help="Subdirectory name under cityscapes_root for output (default: dinov2_features)"
    )
    parser.add_argument(
        "--split", type=str, default=None,
        help="Single split to process (train or val). Default: both train and val."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for feature extraction (default: 4)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, mps, cuda, cpu (default: auto)"
    )
    args = parser.parse_args()

    # Determine splits
    if args.split is not None:
        splits = [args.split]
    else:
        splits = ["train", "val"]

    extract_features(
        cityscapes_root=args.cityscapes_root,
        output_subdir=args.output_subdir,
        splits=splits,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
