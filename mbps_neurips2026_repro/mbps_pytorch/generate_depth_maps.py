#!/usr/bin/env python3
"""Generate monocular depth maps using Depth Anything V3.

Uses DA3MONO-LARGE for relative monocular depth estimation.
Outputs per-image depth maps as .npy files (float32, H x W).

Usage:
    # Install DA3 first:
    # pip install depth_anything_3  (or from source: pip install -e .)

    python mbps_pytorch/generate_depth_maps.py \
        --data_dir /data/cityscapes/leftImg8bit/train \
        --output_dir /data/cityscapes/depth_dav3/train \
        --model_name depth-anything/DA3MONO-LARGE

    # Fallback if DA3 not available — use Depth Anything V2:
    python mbps_pytorch/generate_depth_maps.py \
        --data_dir /data/cityscapes/leftImg8bit/train \
        --output_dir /data/cityscapes/depth_dav3/train \
        --use_dav2 --dav2_model_name depth-anything/Depth-Anything-V2-Large-hf
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def get_image_paths(data_dir: str) -> list:
    """Get all Cityscapes image paths."""
    data_path = Path(data_dir)
    image_paths = sorted(data_path.rglob("*_leftImg8bit.png"))
    if not image_paths:
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            image_paths = sorted(data_path.rglob(ext))
            if image_paths:
                break
    return image_paths


def generate_depth_da3(
    data_dir: str,
    output_dir: str,
    model_name: str = "depth-anything/DA3MONO-LARGE",
    image_size: tuple = (512, 1024),
    device: str = "auto",
):
    """Generate depth maps using Depth Anything V3.

    Args:
        data_dir: Path to images.
        output_dir: Where to save .npy depth files.
        model_name: DA3 model name on HuggingFace.
        image_size: (H, W) target resolution.
        device: Device string.
    """
    import torch
    from depth_anything_3.api import DepthAnything3

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading DA3 model: {model_name}")
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=torch.device(device))

    image_paths = get_image_paths(data_dir)
    print(f"Found {len(image_paths)} images")
    os.makedirs(output_dir, exist_ok=True)

    skipped = 0
    for path in tqdm(image_paths, desc="Generating depth maps"):
        rel_path = path.relative_to(data_dir)
        out_path = Path(output_dir) / rel_path.with_suffix(".npy")

        if out_path.exists():
            skipped += 1
            continue

        # DA3 handles image loading and preprocessing internally
        prediction = model.inference([str(path)])

        # prediction.depth: (1, H_orig, W_orig) relative depth
        depth = prediction.depth[0]  # (H, W)

        # Resize to target resolution
        depth_img = Image.fromarray(depth.astype(np.float32), mode="F")
        depth_img = depth_img.resize(
            (image_size[1], image_size[0]), Image.BILINEAR
        )
        depth = np.array(depth_img)

        # Normalize to [0, 1] range
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), depth.astype(np.float32))

    if skipped > 0:
        print(f"Skipped {skipped} already-processed images")
    print(f"Done! Depth maps saved to {output_dir}")


def generate_depth_da2(
    data_dir: str,
    output_dir: str,
    model_name: str = "depth-anything/Depth-Anything-V2-Large-hf",
    image_size: tuple = (512, 1024),
    device: str = "auto",
    batch_size: int = 4,
):
    """Fallback: Generate depth maps using Depth Anything V2 via HF Transformers.

    Args:
        data_dir: Path to images.
        output_dir: Where to save .npy depth files.
        model_name: DA2 model name on HuggingFace.
        image_size: (H, W) target resolution.
        device: Device string.
        batch_size: Batch size for inference.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading DA2 model: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    image_paths = get_image_paths(data_dir)
    print(f"Found {len(image_paths)} images")
    os.makedirs(output_dir, exist_ok=True)

    skipped = 0
    for i in tqdm(range(0, len(image_paths), batch_size), desc="DA2 depth"):
        batch_paths = image_paths[i:i + batch_size]

        # Filter already processed
        to_process = []
        for path in batch_paths:
            rel_path = path.relative_to(data_dir)
            out_path = Path(output_dir) / rel_path.with_suffix(".npy")
            if out_path.exists():
                skipped += 1
            else:
                to_process.append(path)

        if not to_process:
            continue

        images = [
            Image.open(p).convert("RGB").resize(
                (image_size[1], image_size[0]), Image.BILINEAR
            )
            for p in to_process
        ]

        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        # predicted_depth: (B, H_model, W_model)
        depths = outputs.predicted_depth

        for idx, path in enumerate(to_process):
            depth = depths[idx].cpu().numpy()

            # Resize to target resolution
            depth_img = Image.fromarray(depth.astype(np.float32), mode="F")
            depth_img = depth_img.resize(
                (image_size[1], image_size[0]), Image.BILINEAR
            )
            depth = np.array(depth_img)

            # Normalize to [0, 1]
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth = (depth - d_min) / (d_max - d_min)
            else:
                depth = np.zeros_like(depth)

            rel_path = path.relative_to(data_dir)
            out_path = Path(output_dir) / rel_path.with_suffix(".npy")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), depth.astype(np.float32))

    if skipped > 0:
        print(f"Skipped {skipped} already-processed images")
    print(f"Done! Depth maps saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate monocular depth maps using Depth Anything V3 (or V2 fallback)"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to images"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Where to save .npy depth files"
    )
    parser.add_argument(
        "--model_name", type=str, default="depth-anything/DA3MONO-LARGE",
        help="DA3 model name"
    )
    parser.add_argument(
        "--image_size", type=int, nargs=2, default=[512, 1024],
        help="(H, W) target resolution"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cuda, cpu"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size (DA2 only)"
    )
    parser.add_argument(
        "--use_dav2", action="store_true",
        help="Use Depth Anything V2 instead of V3"
    )
    parser.add_argument(
        "--dav2_model_name", type=str,
        default="depth-anything/Depth-Anything-V2-Large-hf",
        help="DA2 model name"
    )
    args = parser.parse_args()

    if args.use_dav2:
        generate_depth_da2(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_name=args.dav2_model_name,
            image_size=tuple(args.image_size),
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        try:
            generate_depth_da3(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                model_name=args.model_name,
                image_size=tuple(args.image_size),
                device=args.device,
            )
        except ImportError:
            print("WARNING: depth_anything_3 not installed. "
                  "Falling back to Depth Anything V2.")
            print("To use DA3: pip install depth_anything_3")
            generate_depth_da2(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                model_name=args.dav2_model_name,
                image_size=tuple(args.image_size),
                device=args.device,
                batch_size=args.batch_size,
            )


if __name__ == "__main__":
    main()
