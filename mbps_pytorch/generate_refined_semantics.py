#!/usr/bin/env python3
"""Generate refined semantic pseudo-labels using a trained CSCMRefineNet.

Takes a trained checkpoint and generates refined 27-class semantic labels
for all images in the specified split(s). Outputs both:
  1. Argmax PNGs (uint8, 27-class, at original 1024×2048 resolution)
  2. Optionally, soft logits as .pt files (27, 32, 64) float16

Usage:
    python mbps_pytorch/generate_refined_semantics.py \
        --cityscapes_root /path/to/cityscapes \
        --checkpoint checkpoints/refine_net/best.pth \
        --output_dir /path/to/cityscapes/pseudo_semantic_refined \
        --split both --device auto
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from mbps_pytorch.refine_net import CSCMRefineNet

PATCH_H, PATCH_W = 32, 64
NUM_CLASSES = 27


def generate(args):
    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    if "metrics" in ckpt:
        print(f"  Checkpoint metrics: {ckpt['metrics']}")

    # Build model from saved config
    model = CSCMRefineNet(
        num_classes=config.get("num_classes", NUM_CLASSES),
        feature_dim=config.get("feature_dim", 768),
        bridge_dim=config.get("bridge_dim", 192),
        num_blocks=config.get("num_blocks", 4),
        block_type=config.get("block_type", "conv"),
        layer_type=config.get("layer_type", "gated_delta_net"),
        scan_mode=config.get("scan_mode", "bidirectional"),
        coupling_strength=config.get("coupling_strength", 0.1),
        d_state=config.get("d_state", 64),
        chunk_size=config.get("chunk_size", 64),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    splits = ["train", "val"] if args.split == "both" else [args.split]

    for split in splits:
        print(f"\n=== Generating refined semantics for {split} ===")

        # Find all images
        img_dir = os.path.join(args.cityscapes_root, "leftImg8bit", split)
        entries = []
        for city in sorted(os.listdir(img_dir)):
            city_path = os.path.join(img_dir, city)
            if not os.path.isdir(city_path):
                continue
            for fname in sorted(os.listdir(city_path)):
                if not fname.endswith("_leftImg8bit.png"):
                    continue
                stem = fname.replace("_leftImg8bit.png", "")
                entries.append({"stem": stem, "city": city})

        print(f"Processing {len(entries)} images")
        stats = {"total": 0, "per_class": {i: 0 for i in range(NUM_CLASSES)}}
        t0 = time.time()

        with torch.no_grad():
            for entry in tqdm(entries, desc=f"Generating {split}"):
                stem, city = entry["stem"], entry["city"]

                # Load inputs
                cause_logits = _load_cause_logits(
                    args.cityscapes_root, args.logits_subdir,
                    args.semantic_subdir, split, city, stem, device,
                )
                dinov2_features = _load_dinov2_features(
                    args.cityscapes_root, split, city, stem, device,
                )
                depth, depth_grads = _load_depth(
                    args.cityscapes_root, split, city, stem, device,
                )

                # Forward pass (model takes DINOv2 features + depth only)
                refined_logits = model(
                    dinov2_features, depth, depth_grads,
                )  # (1, 27, 32, 64)

                # Argmax at patch resolution
                pred_patch = refined_logits.argmax(dim=1).squeeze(0).cpu().numpy()
                # (32, 64) uint8

                # Upsample to original resolution via nearest neighbor
                pred_full = np.array(
                    Image.fromarray(pred_patch.astype(np.uint8)).resize(
                        (2048, 1024), Image.NEAREST
                    )
                )

                # Save PNG
                city_dir = os.path.join(args.output_dir, split, city)
                os.makedirs(city_dir, exist_ok=True)
                out_path = os.path.join(city_dir, f"{stem}.png")
                Image.fromarray(pred_full.astype(np.uint8)).save(out_path)

                # Optionally save soft logits
                if args.save_logits:
                    probs = F.softmax(refined_logits, dim=1).squeeze(0)
                    logits_path = os.path.join(city_dir, f"{stem}_logits.pt")
                    torch.save(probs.half().cpu(), logits_path)

                # Stats
                stats["total"] += 1
                for c in range(NUM_CLASSES):
                    stats["per_class"][c] += int((pred_full == c).sum())

        dt = time.time() - t0
        print(f"\n{split}: Generated {stats['total']} refined labels in {dt:.1f}s")

        # Save stats
        stats_path = os.path.join(args.output_dir, split, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    print("\nDone!")


def _load_cause_logits(root, logits_subdir, semantic_subdir, split, city, stem, device):
    """Load CAUSE logits: prefer soft .pt, fallback to one-hot from PNG."""
    if logits_subdir:
        pt_path = os.path.join(root, logits_subdir, split, city, f"{stem}_logits.pt")
        if os.path.exists(pt_path):
            logits = torch.load(pt_path, weights_only=True).float()
            if logits.shape[1:] != (PATCH_H, PATCH_W):
                logits = F.interpolate(
                    logits.unsqueeze(0), size=(PATCH_H, PATCH_W),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
            return logits.unsqueeze(0).to(device)

    # Fallback: one-hot from argmax PNG
    sem_path = os.path.join(root, semantic_subdir, split, city, f"{stem}.png")
    sem = np.array(Image.open(sem_path))
    sem_patch = np.array(
        Image.fromarray(sem).resize((PATCH_W, PATCH_H), Image.NEAREST)
    )
    onehot = np.zeros((NUM_CLASSES, PATCH_H, PATCH_W), dtype=np.float32)
    smooth = 0.1
    onehot[:] = smooth / NUM_CLASSES
    for c in range(NUM_CLASSES):
        mask = sem_patch == c
        onehot[c][mask] = 1.0 - smooth + smooth / NUM_CLASSES
    return torch.from_numpy(onehot).unsqueeze(0).to(device)


def _load_dinov2_features(root, split, city, stem, device):
    """Load DINOv2 features: (2048, 768) → (1, 768, 32, 64)."""
    feat = np.load(
        os.path.join(root, "dinov2_features", split, city,
                     f"{stem}_leftImg8bit.npy")
    ).astype(np.float32)
    feat = feat.reshape(PATCH_H, PATCH_W, -1).transpose(2, 0, 1)
    return torch.from_numpy(feat).unsqueeze(0).to(device)


def _load_depth(root, split, city, stem, device):
    """Load depth + compute Sobel gradients: → (1,1,32,64), (1,2,32,64)."""
    depth_full = np.load(
        os.path.join(root, "depth_spidepth", split, city, f"{stem}.npy")
    )
    depth_t = torch.from_numpy(depth_full).unsqueeze(0).unsqueeze(0).float()
    depth_patch = F.interpolate(
        depth_t, size=(PATCH_H, PATCH_W), mode="bilinear", align_corners=False,
    )  # (1, 1, 32, 64)

    # Sobel gradients
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=torch.float32).reshape(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=torch.float32).reshape(1, 1, 3, 3)
    grad_x = F.conv2d(depth_patch, kx, padding=1)
    grad_y = F.conv2d(depth_patch, ky, padding=1)
    depth_grads = torch.cat([grad_x, grad_y], dim=1)  # (1, 2, 32, 64)

    return depth_patch.to(device), depth_grads.to(device)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate refined semantic pseudo-labels")
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained CSCMRefineNet checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for refined labels")
    parser.add_argument("--split", type=str, default="both",
                        choices=["train", "val", "both"])
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_cause_crf")
    parser.add_argument("--logits_subdir", type=str, default=None,
                        help="Subdirectory for soft logits input")
    parser.add_argument("--save_logits", action="store_true",
                        help="Also save refined soft logits as .pt")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args)
