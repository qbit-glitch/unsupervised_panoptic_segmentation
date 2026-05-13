#!/usr/bin/env python3
"""Generate depth maps for COCO val images using DA2-Large or DA3.

Outputs: float32 .npy files at original image resolution, [0,1] normalized.
Unlike Cityscapes (fixed 512x1024), COCO images have variable sizes, so depth
maps are saved at the original resolution to avoid artifacts.

Usage:
    python mbps_pytorch/generate_coco_depth.py \
        --model da2_large \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --device mps

    python mbps_pytorch/generate_coco_depth.py \
        --model da2_large \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --image_ids_file pseudo_semantic_k80/val2017  # Only process images with pseudo-semantics
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def normalize_depth(depth):
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        return (depth - d_min) / (d_max - d_min)
    return np.zeros_like(depth)


def auto_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Generate depth maps for COCO")
    parser.add_argument("--model", required=True,
                        choices=["da2_large", "dav3"],
                        help="Depth model")
    parser.add_argument("--coco_root", required=True, help="Path to COCO dataset root")
    parser.add_argument("--split", default="val2017", help="Image directory name")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--image_ids_from", default=None,
                        help="Directory with .png files — only process matching image IDs")
    args = parser.parse_args()

    import torch
    device = auto_device() if args.device == "auto" else args.device

    root = Path(args.coco_root)
    img_dir = root / args.split

    # Map model names
    MODEL_MAP = {
        "da2_large": {"hf_name": "depth-anything/Depth-Anything-V2-Large-hf",
                      "out_subdir": "depth_da2_large", "backend": "hf"},
        "dav3": {"hf_name": "depth-anything/DA3MONO-LARGE",
                 "out_subdir": "depth_dav3", "backend": "da3"},
    }
    info = MODEL_MAP[args.model]
    out_subdir = info["out_subdir"]
    out_dir = root / out_subdir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover images
    if args.image_ids_from:
        # Only process images matching IDs in this directory
        ref_dir = root / args.image_ids_from
        ref_ids = {f.stem for f in ref_dir.glob("*.png")}
        image_paths = sorted([
            img_dir / f"{img_id}.jpg"
            for img_id in ref_ids
            if (img_dir / f"{img_id}.jpg").exists()
        ])
    else:
        image_paths = sorted(img_dir.glob("*.jpg"))

    if args.max_images:
        image_paths = image_paths[:args.max_images]

    print(f"\n{'='*60}")
    print(f"COCO DEPTH GENERATION: {args.model}")
    print(f"{'='*60}")
    print(f"  Model:   {info['hf_name']}")
    print(f"  Device:  {device}")
    print(f"  Images:  {len(image_paths)} in {img_dir}")
    print(f"  Output:  {out_dir}")

    # Load model
    skipped = 0
    processed = 0
    t0 = time.time()

    if info["backend"] == "hf":
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        processor = AutoImageProcessor.from_pretrained(info["hf_name"])
        model = AutoModelForDepthEstimation.from_pretrained(info["hf_name"])
        model = model.to(device).eval()

        for i in tqdm(range(0, len(image_paths), args.batch_size),
                      desc=f"{args.model} depth"):
            batch_paths = image_paths[i:i + args.batch_size]
            to_process = []
            for path in batch_paths:
                out_path = out_dir / f"{path.stem}.npy"
                if out_path.exists():
                    skipped += 1
                else:
                    to_process.append((path, out_path))
            if not to_process:
                continue

            # Process one at a time (COCO has variable sizes, can't batch)
            for path, out_path in to_process:
                image = Image.open(path).convert("RGB")
                W, H = image.size
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.inference_mode():
                    depth = model(**inputs).predicted_depth.squeeze().cpu().numpy()

                # Flip augmentation
                image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
                inputs_flip = processor(images=image_flip, return_tensors="pt")
                inputs_flip = {k: v.to(device) for k, v in inputs_flip.items()}
                with torch.inference_mode():
                    depth_flip = model(**inputs_flip).predicted_depth.squeeze().cpu().numpy()
                depth = (depth + depth_flip[:, ::-1]) / 2.0

                if depth.shape != (H, W):
                    depth_img = Image.fromarray(depth.astype(np.float32), mode="F")
                    depth_img = depth_img.resize((W, H), Image.BILINEAR)
                    depth = np.array(depth_img)
                depth = normalize_depth(depth)
                np.save(str(out_path), depth.astype(np.float32))
                processed += 1

    elif info["backend"] == "da3":
        from depth_anything_3.api import DepthAnything3
        model = DepthAnything3.from_pretrained(info["hf_name"])
        model = model.to(device=torch.device(device))

        for path in tqdm(image_paths, desc=f"{args.model} depth"):
            out_path = out_dir / f"{path.stem}.npy"
            if out_path.exists():
                skipped += 1
                continue
            image = Image.open(path).convert("RGB")
            W, H = image.size
            prediction = model.inference([str(path)])
            depth = prediction.depth[0]  # (H_proc, W_proc)
            if depth.shape != (H, W):
                depth_img = Image.fromarray(depth.astype(np.float32), mode="F")
                depth_img = depth_img.resize((W, H), Image.BILINEAR)
                depth = np.array(depth_img)
            depth = normalize_depth(depth)
            np.save(str(out_path), depth.astype(np.float32))
            processed += 1

    elapsed = time.time() - t0
    total = len(image_paths)
    print(f"\n{'='*60}")
    print(f"COMPLETE: {processed} generated, {skipped} skipped")
    print(f"Time: {elapsed:.1f}s ({elapsed/max(processed,1):.2f}s/image)")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
