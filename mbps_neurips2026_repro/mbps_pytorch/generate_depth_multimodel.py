#!/usr/bin/env python3
"""Unified multi-model monocular depth generation for ablation study.

Generates depth maps from 7 monocular depth estimation models for fair
comparison in the instance pseudo-label ablation study (Phase 1).

All outputs: float32 .npy files, (H, W), [0,1] normalized per-image.
This ensures Sobel gradient thresholds are comparable across models.

Supported models:
  - da2_large:    Depth Anything V2 Large (HF Transformers)
  - da2_giant:    Depth Anything V2 Giant (HF Transformers)
  - depthpro:     Apple DepthPro (HF Transformers)
  - dinov3_vitl:  DINOv3 ViT-L/16 DPT depth decoder (official repo)
  - marigold:     Marigold v1.0 diffusion-based depth (diffusers)
  - zoedepth:     ZoeDepth (Intel ISL)
  - spidepth:     SPIdepth (CVPR 2025) — for reference/re-generation

Usage:
    # Generate DA2-Large depth on Cityscapes val:
    python mbps_pytorch/generate_depth_multimodel.py \
        --model da2_large \
        --cityscapes_root /path/to/cityscapes \
        --split val \
        --device auto

    # Generate DepthPro on specific GPU:
    python mbps_pytorch/generate_depth_multimodel.py \
        --model depthpro \
        --cityscapes_root /path/to/cityscapes \
        --split val \
        --device cuda:1

    # Generate Marigold (slow, diffusion-based):
    python mbps_pytorch/generate_depth_multimodel.py \
        --model marigold \
        --cityscapes_root /path/to/cityscapes \
        --split val \
        --device cuda:0 \
        --marigold_steps 10
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_H, TARGET_W = 512, 1024

MODEL_INFO = {
    "da2_large": {
        "hf_name": "depth-anything/Depth-Anything-V2-Large-hf",
        "type": "relative",
        "output_subdir": "depth_da2_large",
    },
    "da2_giant": {
        "hf_name": "depth-anything/Depth-Anything-V2-Giant-hf",
        "type": "relative",
        "output_subdir": "depth_da2_giant",
    },
    "depthpro": {
        "hf_name": "apple/DepthPro-hf",
        "type": "metric",
        "output_subdir": "depth_depthpro",
    },
    "dinov3_vitl": {
        "hf_name": None,  # Uses official repo, not HF
        "type": "zero-shot",
        "output_subdir": "depth_dinov3_vitl",
    },
    "marigold": {
        "hf_name": "prs-eth/marigold-v1-0",
        "type": "diffusion",
        "output_subdir": "depth_marigold",
    },
    "zoedepth": {
        "hf_name": None,  # Uses local repo
        "type": "metric",
        "output_subdir": "depth_zoedepth",
    },
    "spidepth": {
        "hf_name": None,  # Uses local repo
        "type": "self-supervised",
        "output_subdir": "depth_spidepth",
    },
}


def get_image_paths(data_dir):
    """Get all Cityscapes image paths from a split directory."""
    data_path = Path(data_dir)
    paths = sorted(data_path.rglob("*_leftImg8bit.png"))
    if not paths:
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            paths = sorted(data_path.rglob(ext))
            if paths:
                break
    return paths


def normalize_depth(depth):
    """Normalize depth to [0, 1] per image."""
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        return (depth - d_min) / (d_max - d_min)
    return np.zeros_like(depth)


def resize_depth(depth, target_h, target_w):
    """Resize depth map to target resolution with bilinear interpolation."""
    if depth.shape == (target_h, target_w):
        return depth
    depth_img = Image.fromarray(depth.astype(np.float32), mode="F")
    depth_img = depth_img.resize((target_w, target_h), Image.BILINEAR)
    return np.array(depth_img)


def auto_device():
    """Detect best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ═══════════════════════════════════════════════════════
# Model-specific generators
# ═══════════════════════════════════════════════════════

def generate_da2(image_paths, output_dir, data_dir, model_name, device,
                 batch_size=4, flip_augment=True):
    """Depth Anything V2 (Large or Giant) via HF Transformers."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    logger.info(f"Loading DA2: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model = model.to(device).eval()

    skipped = 0
    for i in tqdm(range(0, len(image_paths), batch_size), desc="DA2 depth"):
        batch_paths = image_paths[i:i + batch_size]
        to_process = []
        for path in batch_paths:
            stem = path.stem.replace("_leftImg8bit", "")
            city = path.parent.name
            out_path = Path(output_dir) / city / f"{stem}.npy"
            if out_path.exists():
                skipped += 1
            else:
                to_process.append((path, out_path))

        if not to_process:
            continue

        images = [Image.open(p).convert("RGB") for p, _ in to_process]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
        depths = outputs.predicted_depth.cpu().numpy()

        if flip_augment:
            images_flip = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
            inputs_flip = processor(images=images_flip, return_tensors="pt")
            inputs_flip = {k: v.to(device) for k, v in inputs_flip.items()}
            with torch.inference_mode():
                outputs_flip = model(**inputs_flip)
            depths_flip = outputs_flip.predicted_depth.cpu().numpy()
            depths_flip = depths_flip[:, :, ::-1]  # Flip back
            depths = (depths + depths_flip) / 2.0

        for idx, (path, out_path) in enumerate(to_process):
            depth = resize_depth(depths[idx], TARGET_H, TARGET_W)
            depth = normalize_depth(depth)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), depth.astype(np.float32))

    if skipped:
        logger.info(f"Skipped {skipped} existing files")


def generate_depthpro(image_paths, output_dir, data_dir, device):
    """Apple DepthPro via HF Transformers."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_name = "apple/DepthPro-hf"
    logger.info(f"Loading DepthPro: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model = model.to(device).eval()

    skipped = 0
    for path in tqdm(image_paths, desc="DepthPro"):
        stem = path.stem.replace("_leftImg8bit", "")
        city = path.parent.name
        out_path = Path(output_dir) / city / f"{stem}.npy"
        if out_path.exists():
            skipped += 1
            continue

        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        depth = outputs.predicted_depth.squeeze().cpu().numpy()
        depth = resize_depth(depth, TARGET_H, TARGET_W)
        depth = normalize_depth(depth)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), depth.astype(np.float32))

    if skipped:
        logger.info(f"Skipped {skipped} existing files")


def generate_dinov3_vitl(image_paths, output_dir, data_dir, device):
    """DINOv3 ViT-L/16 DPT depth decoder (official Meta repo)."""
    import torch

    # Add DINOv3 repo to path
    dinov3_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "refs", "dinov3")
    if dinov3_dir not in sys.path:
        sys.path.insert(0, dinov3_dir)

    from dinov3.hub.depthers import _make_dinov3_dpt_depther, DepthWeights
    from torchvision import transforms

    logger.info("Loading DINOv3 ViT-L/16 depth decoder (SynthMix weights)")
    depther = _make_dinov3_dpt_depther(
        backbone_name="dinov3_vitl16",
        pretrained=True,
        depther_weights=DepthWeights.SYNTHMIX,
        autocast_dtype=torch.float32,
    )
    depther = depther.to(device).eval()

    # DINOv3 preprocessing: ImageNet normalization, resize to multiple of 16
    transform = transforms.Compose([
        transforms.Resize((518, 518)),  # DINOv3 standard size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    skipped = 0
    for path in tqdm(image_paths, desc="DINOv3 ViT-L depth"):
        stem = path.stem.replace("_leftImg8bit", "")
        city = path.parent.name
        out_path = Path(output_dir) / city / f"{stem}.npy"
        if out_path.exists():
            skipped += 1
            continue

        image = Image.open(path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            output = depther(img_tensor)
            # Output is typically a dict or tensor with depth predictions
            if isinstance(output, dict):
                depth = output.get("depth", output.get("predicted_depth"))
            elif isinstance(output, torch.Tensor):
                depth = output
            else:
                depth = output

            if isinstance(depth, torch.Tensor):
                depth = depth.squeeze().cpu().numpy()

        depth = resize_depth(depth, TARGET_H, TARGET_W)
        depth = normalize_depth(depth)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), depth.astype(np.float32))

    if skipped:
        logger.info(f"Skipped {skipped} existing files")


def generate_marigold(image_paths, output_dir, data_dir, device,
                      num_inference_steps=10, ensemble_size=5):
    """Marigold v1.0 diffusion-based depth (from diffusers)."""
    import torch
    from diffusers import MarigoldDepthPipeline

    logger.info(f"Loading Marigold (steps={num_inference_steps}, ensemble={ensemble_size})")
    pipe = MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-v1-0",
        torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
    )
    pipe = pipe.to(device)

    skipped = 0
    for path in tqdm(image_paths, desc="Marigold depth"):
        stem = path.stem.replace("_leftImg8bit", "")
        city = path.parent.name
        out_path = Path(output_dir) / city / f"{stem}.npy"
        if out_path.exists():
            skipped += 1
            continue

        image = Image.open(path).convert("RGB")

        output = pipe(
            image,
            num_inference_steps=num_inference_steps,
            ensemble_size=ensemble_size,
        )
        depth = output.prediction.squeeze().cpu().numpy()

        depth = resize_depth(depth, TARGET_H, TARGET_W)
        depth = normalize_depth(depth)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), depth.astype(np.float32))

    if skipped:
        logger.info(f"Skipped {skipped} existing files")


def generate_zoedepth(image_paths, output_dir, data_dir, device):
    """ZoeDepth (Intel ISL) zero-shot metric depth."""
    import torch

    logger.info("Loading ZoeDepth NK model")
    # Try torch hub first
    try:
        model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
    except Exception:
        # Fallback: try MiDaS DPT-Large
        logger.warning("ZoeDepth hub load failed, trying MiDaS DPT-Large")
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")

    model = model.to(device).eval()

    # MiDaS transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform

    skipped = 0
    for path in tqdm(image_paths, desc="ZoeDepth"):
        stem = path.stem.replace("_leftImg8bit", "")
        city = path.parent.name
        out_path = Path(output_dir) / city / f"{stem}.npy"
        if out_path.exists():
            skipped += 1
            continue

        image = Image.open(path).convert("RGB")
        img_np = np.array(image)

        input_batch = transform(img_np).to(device)

        with torch.inference_mode():
            prediction = model(input_batch)
            depth = prediction.squeeze().cpu().numpy()

        depth = resize_depth(depth, TARGET_H, TARGET_W)
        depth = normalize_depth(depth)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), depth.astype(np.float32))

    if skipped:
        logger.info(f"Skipped {skipped} existing files")


def generate_spidepth(image_paths, output_dir, data_dir, device):
    """SPIdepth (CVPR 2025) self-supervised depth. Delegates to generate_depth_spidepth.py."""
    logger.info("For SPIdepth, use the dedicated script: generate_depth_spidepth.py")
    logger.info("This model requires the SPIdepth repo and checkpoint at refs/spidepth/")
    raise NotImplementedError(
        "SPIdepth depth maps should be generated with generate_depth_spidepth.py. "
        "They already exist at depth_spidepth/ in the Cityscapes root."
    )


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

GENERATORS = {
    "da2_large": lambda paths, out, data, dev, args: generate_da2(
        paths, out, data, MODEL_INFO["da2_large"]["hf_name"], dev,
        batch_size=args.batch_size, flip_augment=args.flip_augment,
    ),
    "da2_giant": lambda paths, out, data, dev, args: generate_da2(
        paths, out, data, MODEL_INFO["da2_giant"]["hf_name"], dev,
        batch_size=args.batch_size, flip_augment=args.flip_augment,
    ),
    "depthpro": lambda paths, out, data, dev, args: generate_depthpro(
        paths, out, data, dev,
    ),
    "dinov3_vitl": lambda paths, out, data, dev, args: generate_dinov3_vitl(
        paths, out, data, dev,
    ),
    "marigold": lambda paths, out, data, dev, args: generate_marigold(
        paths, out, data, dev,
        num_inference_steps=args.marigold_steps,
        ensemble_size=args.marigold_ensemble,
    ),
    "zoedepth": lambda paths, out, data, dev, args: generate_zoedepth(
        paths, out, data, dev,
    ),
    "spidepth": lambda paths, out, data, dev, args: generate_spidepth(
        paths, out, data, dev,
    ),
}


def main():
    parser = argparse.ArgumentParser(
        description="Unified multi-model monocular depth generation"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_INFO.keys()),
                        help="Which depth model to use")
    parser.add_argument("--cityscapes_root", type=str, required=True,
                        help="Path to Cityscapes dataset root")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "both"],
                        help="Which split to process")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cuda:0, cuda:1, mps, cpu")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (DA2 models only)")
    parser.add_argument("--flip_augment", action="store_true", default=True,
                        help="Use flip augmentation (DA2 models)")
    parser.add_argument("--no_flip_augment", action="store_false", dest="flip_augment",
                        help="Disable flip augmentation")
    parser.add_argument("--marigold_steps", type=int, default=10,
                        help="Marigold diffusion steps (default: 10, more=slower+better)")
    parser.add_argument("--marigold_ensemble", type=int, default=5,
                        help="Marigold ensemble size (default: 5)")
    parser.add_argument("--output_subdir", type=str, default=None,
                        help="Override output subdirectory name")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit to first N images (for testing)")
    parser.add_argument("--target_size", type=int, nargs=2, default=[512, 1024],
                        help="Output depth map size (H W)")

    args = parser.parse_args()

    global TARGET_H, TARGET_W
    TARGET_H, TARGET_W = args.target_size

    import torch
    if args.device == "auto":
        device = auto_device()
    else:
        device = args.device

    info = MODEL_INFO[args.model]
    output_subdir = args.output_subdir or info["output_subdir"]
    root = Path(args.cityscapes_root)

    splits = ["train", "val"] if args.split == "both" else [args.split]

    print(f"\n{'='*60}")
    print(f"DEPTH GENERATION: {args.model}")
    print(f"{'='*60}")
    print(f"  Model:    {args.model} ({info['type']})")
    print(f"  HF name:  {info['hf_name'] or 'local repo'}")
    print(f"  Device:   {device}")
    print(f"  Output:   {output_subdir}/")
    print(f"  Target:   {TARGET_H}x{TARGET_W}")
    print(f"  Splits:   {splits}")

    for split in splits:
        data_dir = root / "leftImg8bit" / split
        output_dir = root / output_subdir / split

        image_paths = get_image_paths(str(data_dir))
        if args.max_images:
            image_paths = image_paths[:args.max_images]

        print(f"\n  Processing {split}: {len(image_paths)} images → {output_dir}")
        t0 = time.time()

        GENERATORS[args.model](image_paths, str(output_dir), str(data_dir), device, args)

        elapsed = time.time() - t0
        n = len(image_paths)
        print(f"  Done in {elapsed:.1f}s ({elapsed/max(n,1):.2f}s/image)")

    # Verify output
    for split in splits:
        output_dir = root / output_subdir / split
        n_files = len(list(output_dir.rglob("*.npy")))
        print(f"\n  {split}: {n_files} depth maps in {output_dir}")

    print(f"\n{'='*60}")
    print(f"COMPLETE: {args.model} depth maps ready at {root / output_subdir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
