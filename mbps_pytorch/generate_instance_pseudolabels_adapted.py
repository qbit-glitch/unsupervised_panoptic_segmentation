#!/usr/bin/env python3
"""Generate instance pseudo-labels using adapted depth models."""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel
from tqdm import tqdm

from mbps_pytorch.models.adapters import (
    inject_lora_into_depth_model,
    inject_lora_into_depthpro,
    freeze_non_adapter_params,
)
from mbps_pytorch.train_depth_adapter_lora import (
    load_dav3_model, load_da2_model, load_depthpro_model,
)

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

THING_IDS = {11, 12, 13, 14, 15, 16, 17, 18}


def depth_guided_instances(semantic, depth, tau=0.03, min_area=1000,
                           dilation_iters=3, depth_blur_sigma=1.0):
    if depth_blur_sigma > 0:
        depth_smooth = gaussian_filter(depth.astype(np.float64), sigma=depth_blur_sigma)
    else:
        depth_smooth = depth.astype(np.float64)
    gx = sobel(depth_smooth, axis=1)
    gy = sobel(depth_smooth, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    depth_edges = grad_mag > tau

    assigned = np.zeros(semantic.shape, dtype=bool)
    instances = []

    for cls in sorted(THING_IDS):
        cls_mask = semantic == cls
        if not cls_mask.any():
            continue
        edges = depth_edges & cls_mask
        edges = ndimage.binary_dilation(edges, iterations=dilation_iters)
        fg = cls_mask & (~edges)
        labeled, num = ndimage.label(fg)
        for inst_id in range(1, num + 1):
            mask = labeled == inst_id
            if mask.sum() < min_area:
                continue
            mask = mask & (~assigned)
            if mask.sum() < min_area:
                continue
            instances.append((mask, cls, 1.0))
            assigned |= mask
    return instances


def generate_adapted_instances(
    checkpoint_path, model_type, image_dir, output_dir, device,
    semantic_dir=None, tau=0.20, min_area=1000, dilation_iters=3,
    depth_blur_sigma=1.0, image_size=(512, 1024),
):
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(list(image_dir.rglob("*.png")) + list(image_dir.rglob("*.jpg")))
    if not image_files:
        logger.error("No images found in %s", image_dir)
        return
    logger.info("Found %d images", len(image_files))

    logger.info("Loading %s model...", model_type)
    if model_type == "dav3":
        model = load_dav3_model(device=str(device))
    elif model_type == "da2":
        model = load_da2_model(device=str(device))
    elif model_type == "depthpro":
        model = load_depthpro_model(device=str(device))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if model_type == "depthpro":
        inject_lora_into_depthpro(model, variant="dora", rank=4, alpha=4.0)
    else:
        inject_lora_into_depth_model(model, variant="dora", rank=4, alpha=4.0)
    freeze_non_adapter_params(model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device).eval()

    stats = {"total_instances": 0, "total_images": 0, "instances_per_image": []}

    for img_path in tqdm(image_files, desc="Generating instances"):
        rel_path = img_path.relative_to(image_dir)
        out_path = output_dir / rel_path.parent / (rel_path.stem + "_instance.png")
        out_npz = output_dir / rel_path.parent / (rel_path.stem + ".npz")
        if out_path.exists() and out_npz.exists():
            data = np.load(str(out_npz))
            nv = int(data.get("num_valid", 0))
            stats["total_instances"] += nv
            stats["total_images"] += 1
            stats["instances_per_image"].append(nv)
            continue

        img = Image.open(img_path).convert("RGB").resize((image_size[1], image_size[0]), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.inference_mode():
            if model_type == "dav3":
                pred = model.inference_batch(img_tensor)
                depth = pred.cpu().numpy().squeeze()
            else:
                inputs = model.processor(images=[img], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                pred = model(**inputs).predicted_depth
                depth = pred.cpu().numpy().squeeze()

        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        semantic = np.zeros(image_size, dtype=np.int32)
        if semantic_dir:
            sem_path = Path(semantic_dir) / rel_path
            if sem_path.exists():
                semantic = np.array(Image.open(sem_path).resize((image_size[1], image_size[0]), Image.NEAREST))

        instances = depth_guided_instances(
            semantic, depth, tau=tau, min_area=min_area,
            dilation_iters=dilation_iters, depth_blur_sigma=depth_blur_sigma,
        )

        instance_map = np.zeros(image_size, dtype=np.int32)
        for i, (mask, cls, score) in enumerate(instances):
            instance_map[mask] = i + 1

        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(instance_map.astype(np.uint16)).save(str(out_path))

        masks = np.stack([m for m, c, s in instances]) if instances else np.zeros((0, image_size[0], image_size[1]), dtype=bool)
        scores = np.array([s for m, c, s in instances], dtype=np.float32) if instances else np.zeros(0, dtype=np.float32)
        classes = np.array([c for m, c, s in instances], dtype=np.int32) if instances else np.zeros(0, dtype=np.int32)
        np.savez_compressed(
            str(out_npz),
            masks=masks, scores=scores, classes=classes,
            num_valid=len(instances), tau=tau, min_area=min_area,
        )

        stats["total_instances"] += len(instances)
        stats["total_images"] += 1
        stats["instances_per_image"].append(len(instances))

    if stats["total_images"] > 0:
        avg = stats["total_instances"] / stats["total_images"]
        logger.info("Images: %d, Instances: %d, Avg: %.1f",
                    stats["total_images"], stats["total_instances"], avg)


def main():
    parser = argparse.ArgumentParser(description="Generate adapted instance pseudo-labels")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["dav3", "da2", "depthpro"])
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--semantic_dir", type=str, default=None)
    parser.add_argument("--tau", type=float, default=0.20)
    parser.add_argument("--min_area", type=int, default=1000)
    parser.add_argument("--dilation_iters", type=int, default=3)
    parser.add_argument("--depth_blur_sigma", type=float, default=1.0)
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 1024])
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    generate_adapted_instances(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        device=device,
        semantic_dir=args.semantic_dir,
        tau=args.tau,
        min_area=args.min_area,
        dilation_iters=args.dilation_iters,
        depth_blur_sigma=args.depth_blur_sigma,
        image_size=tuple(args.image_size),
    )


if __name__ == "__main__":
    main()
