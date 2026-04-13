#!/usr/bin/env python3
"""Extract 64x64 DINOv3 ViT-L/16 features for COCO images at 1024x1024 resolution.

Current features: 32x32 (518px input, 1024 patches × 1024-dim)
This script: 64x64 (1024px input, 4096 patches × 1024-dim)

Usage:
    python mbps_pytorch/extract_dinov3_features_coco_hires.py \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --split val2017 --device mps --batch_size 1

Output: {coco_root}/dinov3_features_64x64/{split}/{image_id}.npy
        Shape per image: (4096, 1024) = 64×64 patches × 1024 dim
        ~16 MB per image, ~8 GB total for 501 val images
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", required=True)
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--model_name", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--image_size", type=int, default=1024,
                        help="Input resolution (square). 1024 → 64×64 patches")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (1 recommended for 1024px on MPS)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--image_ids_from", default=None,
                        help="Directory with files to match image IDs against")
    args = parser.parse_args()

    root = Path(args.coco_root)
    img_dir = root / args.split
    patch_size = 16
    h_patches = args.image_size // patch_size
    w_patches = args.image_size // patch_size
    n_patches = h_patches * w_patches

    out_dir = root / f"dinov3_features_{h_patches}x{w_patches}" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover images
    if args.image_ids_from:
        ref_dir = root / args.image_ids_from
        ref_ids = {f.stem for f in ref_dir.iterdir() if f.suffix in ('.png', '.npy', '.jpg')}
        image_paths = sorted([
            img_dir / f"{img_id}.jpg"
            for img_id in ref_ids
            if (img_dir / f"{img_id}.jpg").exists()
        ])
    else:
        image_paths = sorted(img_dir.glob("*.jpg"))

    if args.max_images:
        image_paths = image_paths[:args.max_images]

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"\n{'='*60}")
    print(f"COCO HIRES FEATURE EXTRACTION")
    print(f"{'='*60}")
    print(f"  Model:    {args.model_name}")
    print(f"  Input:    {args.image_size}×{args.image_size}")
    print(f"  Patches:  {h_patches}×{w_patches} = {n_patches}")
    print(f"  Images:   {len(image_paths)} in {img_dir}")
    print(f"  Output:   {out_dir}")
    print(f"  Device:   {device}")
    print(f"  Est size: ~{n_patches * 1024 * 4 / 1e6:.0f} MB/image, "
          f"~{n_patches * 1024 * 4 * len(image_paths) / 1e9:.1f} GB total")

    # Load model
    print(f"\nLoading model...")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    processor.size = {"height": args.image_size, "width": args.image_size}
    processor.crop_size = {"height": args.image_size, "width": args.image_size}
    processor.do_center_crop = False

    model = AutoModel.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()

    n_register = getattr(model.config, "num_register_tokens", 4)
    skip_tokens = 1 + n_register
    hidden_dim = model.config.hidden_size
    print(f"  Hidden dim: {hidden_dim}, register tokens: {n_register}")
    print(f"  Expected output: ({n_patches}, {hidden_dim}) per image")

    # Extract
    skipped = 0
    processed = 0
    t0 = time.time()

    for path in tqdm(image_paths, desc="Extracting features"):
        out_path = out_dir / f"{path.stem}.npy"
        if out_path.exists():
            skipped += 1
            continue

        img = Image.open(path).convert("RGB")
        img_resized = img.resize((args.image_size, args.image_size), Image.BILINEAR)

        inputs = processor(images=img_resized, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state  # (1, 1+reg+patches, dim)
        patch_features = hidden[:, skip_tokens:, :]  # (1, n_patches, dim)

        if patch_features.shape[1] != n_patches:
            print(f"  WARNING: {path.stem} got {patch_features.shape[1]} patches, "
                  f"expected {n_patches}. Skipping.")
            continue

        feat_np = patch_features[0].cpu().numpy().astype(np.float32)
        np.save(str(out_path), feat_np)
        processed += 1

    elapsed = time.time() - t0

    # Save metadata
    meta = {
        "model_name": args.model_name,
        "image_size": args.image_size,
        "h_patches": h_patches,
        "w_patches": w_patches,
        "n_patches": n_patches,
        "hidden_dim": hidden_dim,
        "n_register_tokens": n_register,
        "resize": args.image_size,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"COMPLETE: {processed} extracted, {skipped} skipped")
    print(f"Time: {elapsed:.1f}s ({elapsed/max(processed,1):.2f}s/image)")
    print(f"Output: {out_dir}")
    print(f"Shape: ({n_patches}, {hidden_dim})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
