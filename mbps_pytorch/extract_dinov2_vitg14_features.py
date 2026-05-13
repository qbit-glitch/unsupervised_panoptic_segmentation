#!/usr/bin/env python3
"""Extract DINOv2 ViT-g/14 (with registers) patch features for Cityscapes.

Uses torch.hub (facebookresearch/dinov2) since ViT-g/14 isn't on HuggingFace
transformers. Patch size = 14, so input is resized to 518x1036 (multiples of 14)
giving a 37x74 = 2738 patch grid.

Output: Per-image .npy files of shape (2738, 1536) in float16.

Usage:
    python mbps_pytorch/extract_dinov2_vitg14_features.py \
        --data_dir /data/cityscapes/leftImg8bit/train \
        --output_dir /data/cityscapes/dinov2g14_features/train \
        --batch_size 2
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PATCH_SIZE = 14
IMAGE_H, IMAGE_W = 518, 1036
H_PATCHES = IMAGE_H // PATCH_SIZE  # 37
W_PATCHES = IMAGE_W // PATCH_SIZE  # 74
N_PATCHES = H_PATCHES * W_PATCHES  # 2738


def get_image_paths(data_dir: str) -> list:
    data_path = Path(data_dir)
    image_paths = sorted(data_path.rglob("*_leftImg8bit.png"))
    if not image_paths:
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            image_paths = sorted(data_path.rglob(ext))
            if image_paths:
                break
    return image_paths


def extract_features(
    data_dir: str,
    output_dir: str,
    batch_size: int = 2,
    device: str = "auto",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Device: {device}")

    logger.info("Loading dinov2_vitg14_reg via torch.hub...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14_reg")
    model = model.to(device)
    model.eval()

    n_register = getattr(model, "num_register_tokens", 4)
    skip_tokens = 1 + n_register
    logger.info(
        f"embed_dim={model.embed_dim}, patch_size={PATCH_SIZE}, "
        f"register_tokens={n_register}, grid={H_PATCHES}x{W_PATCHES}={N_PATCHES}"
    )

    transform = transforms.Compose([
        transforms.Resize((IMAGE_H, IMAGE_W), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_paths = get_image_paths(data_dir)
    logger.info(f"Found {len(image_paths)} images in {data_dir}")
    if not image_paths:
        logger.error("No images found!")
        return

    metadata = {
        "model": "dinov2_vitg14_reg",
        "image_size": [IMAGE_H, IMAGE_W],
        "patch_size": PATCH_SIZE,
        "hidden_size": model.embed_dim,
        "h_patches": H_PATCHES,
        "w_patches": W_PATCHES,
        "n_patches": N_PATCHES,
        "n_register_tokens": n_register,
        "num_images": len(image_paths),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    skipped = 0
    for batch_start in tqdm(range(0, len(image_paths), batch_size), desc="Extracting"):
        batch_paths = image_paths[batch_start:batch_start + batch_size]

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

        tensors = []
        for path in batch_to_process:
            img = Image.open(path).convert("RGB")
            tensors.append(transform(img))
        batch_tensor = torch.stack(tensors).to(device)

        with torch.inference_mode():
            out = model.forward_features(batch_tensor)
            tokens = out["x_norm_patchtokens"]

        assert tokens.shape[1] == N_PATCHES, (
            f"Expected {N_PATCHES} patches, got {tokens.shape[1]}"
        )

        features_np = tokens.cpu().numpy().astype(np.float16)

        for idx, path in enumerate(batch_to_process):
            rel_path = path.relative_to(data_dir)
            out_path = Path(output_dir) / rel_path.with_suffix(".npy")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), features_np[idx])

    if skipped > 0:
        logger.info(f"Skipped {skipped} already-extracted images")
    logger.info(f"Done! Shape per image: ({N_PATCHES}, {model.embed_dim})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 ViT-g/14 (register) features for Cityscapes"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    extract_features(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
