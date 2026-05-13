#!/usr/bin/env python3
"""Extract 2x-densified DINOv3 ViT-L/16 features via sub-patch shift-and-interleave.

Implements the training-free feature upsampling from "Upsampling DINOv2 Features"
(arxiv 2410.19836). For 2x densification with patch_size=16, we use 4 shifts of
8px each: (0,0), (0,8), (8,0), (8,8). Each shift produces a 32x64 feature grid;
the 4 grids interleave into a single 64x128 grid.

Output: Per-image .npy files of shape (8192, 1024) in float32.

Usage:
    python mbps_pytorch/extract_shift_avg_features.py \
        --cityscapes_root /data/cityscapes \
        --split train --device mps --batch_size 1 \
        --output_subdir dinov3_features_shiftavg_vitl16
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
PATCH_SIZE = 16
IMAGE_H, IMAGE_W = 512, 1024
H_PATCHES = IMAGE_H // PATCH_SIZE  # 32
W_PATCHES = IMAGE_W // PATCH_SIZE  # 64


def get_image_paths(data_dir: Path) -> List[Path]:
    paths = sorted(data_dir.rglob("*_leftImg8bit.png"))
    if not paths:
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            paths = sorted(data_dir.rglob(ext))
            if paths:
                break
    return paths


def compute_shifts(n_shifts_per_axis: int) -> List[Tuple[int, int]]:
    """Compute sub-patch pixel offsets for shift-and-interleave.

    For n=2 with patch_size=16: offsets are 0, 8 → shifts (0,0),(0,8),(8,0),(8,8).
    """
    step = PATCH_SIZE // n_shifts_per_axis
    shifts = []
    for dy in range(n_shifts_per_axis):
        for dx in range(n_shifts_per_axis):
            shifts.append((dy * step, dx * step))
    return shifts


def shift_image(img_tensor: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Shift image by (dy, dx) pixels via circular padding + crop.

    Args:
        img_tensor: (C, H, W) float tensor.
        dy: vertical shift in pixels.
        dx: horizontal shift in pixels.

    Returns:
        Shifted (C, H, W) tensor.
    """
    if dy == 0 and dx == 0:
        return img_tensor
    # Pad and crop to simulate shift without losing information
    # Shift right by dx means we want features that "see" content offset by dx
    # Use roll for clean wraparound (image edges are less important than interior)
    return torch.roll(img_tensor, shifts=(-dy, -dx), dims=(1, 2))


def interleave_grids(
    grids: List[np.ndarray],
    n_shifts_per_axis: int,
    h_patches: int,
    w_patches: int,
) -> np.ndarray:
    """Interleave shifted feature grids into a dense grid.

    For n_shifts_per_axis=2: 4 grids of (32, 64, D) → one (64, 128, D) grid.
    Shift (dy_idx, dx_idx) goes to positions [dy_idx::n, dx_idx::n].
    """
    D = grids[0].shape[-1]
    out_h = h_patches * n_shifts_per_axis
    out_w = w_patches * n_shifts_per_axis
    dense = np.zeros((out_h, out_w, D), dtype=grids[0].dtype)

    idx = 0
    for dy_idx in range(n_shifts_per_axis):
        for dx_idx in range(n_shifts_per_axis):
            dense[dy_idx::n_shifts_per_axis, dx_idx::n_shifts_per_axis, :] = grids[idx]
            idx += 1

    return dense.reshape(-1, D)


def extract_shifted_features(
    cityscapes_root: Path,
    split: str,
    output_subdir: str,
    n_shifts_per_axis: int = 2,
    device: str = "auto",
    batch_size: int = 1,
) -> None:
    data_dir = cityscapes_root / "leftImg8bit" / split
    output_dir = cityscapes_root / output_subdir / split
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Device: {device}")

    logger.info(f"Loading {MODEL_NAME}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    processor.size = {"height": IMAGE_H, "width": IMAGE_W}
    processor.crop_size = {"height": IMAGE_H, "width": IMAGE_W}
    processor.do_center_crop = False

    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    n_register = getattr(model.config, "num_register_tokens", 4)
    skip_tokens = 1 + n_register
    hidden_dim = model.config.hidden_size
    logger.info(
        f"hidden_dim={hidden_dim}, patch_size={PATCH_SIZE}, "
        f"register_tokens={n_register}, base_grid={H_PATCHES}x{W_PATCHES}"
    )

    shifts = compute_shifts(n_shifts_per_axis)
    out_h = H_PATCHES * n_shifts_per_axis
    out_w = W_PATCHES * n_shifts_per_axis
    out_patches = out_h * out_w
    logger.info(
        f"Shifts: {shifts} ({len(shifts)} passes per image), "
        f"output grid: {out_h}x{out_w}={out_patches}"
    )

    image_paths = get_image_paths(data_dir)
    logger.info(f"Found {len(image_paths)} images in {data_dir}")
    if not image_paths:
        logger.error("No images found!")
        return

    metadata = {
        "model_name": MODEL_NAME,
        "image_size": [IMAGE_H, IMAGE_W],
        "patch_size": PATCH_SIZE,
        "hidden_size": hidden_dim,
        "h_patches": out_h,
        "w_patches": out_w,
        "n_patches": out_patches,
        "n_shifts_per_axis": n_shifts_per_axis,
        "shifts_px": shifts,
        "base_grid": [H_PATCHES, W_PATCHES],
        "num_images": len(image_paths),
    }
    with open(str(output_dir / "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    skipped = 0
    for path in tqdm(image_paths, desc=f"Extracting {split}"):
        rel_path = path.relative_to(data_dir)
        out_path = output_dir / rel_path.with_suffix(".npy")
        if out_path.exists():
            skipped += 1
            continue

        img = Image.open(path).convert("RGB")
        img_resized = img.resize((IMAGE_W, IMAGE_H), Image.BILINEAR)

        inputs_base = processor(images=[img_resized], return_tensors="pt")
        img_tensor = inputs_base["pixel_values"][0]  # (C, H, W)

        shift_grids = []
        for dy, dx in shifts:
            shifted = shift_image(img_tensor, dy, dx)
            pixel_values = shifted.unsqueeze(0).to(device)

            with torch.inference_mode():
                outputs = model(pixel_values=pixel_values)

            patches = outputs.last_hidden_state[:, skip_tokens:, :]
            assert patches.shape[1] == H_PATCHES * W_PATCHES, (
                f"Expected {H_PATCHES * W_PATCHES} patches, got {patches.shape[1]}"
            )
            feat_2d = patches[0].cpu().numpy().reshape(H_PATCHES, W_PATCHES, hidden_dim)
            shift_grids.append(feat_2d)

        dense_feat = interleave_grids(shift_grids, n_shifts_per_axis, H_PATCHES, W_PATCHES)

        norms = np.linalg.norm(dense_feat, axis=-1, keepdims=True) + 1e-8
        dense_feat = dense_feat / norms

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), dense_feat.astype(np.float32))

    if skipped > 0:
        logger.info(f"Skipped {skipped} already-extracted images")
    logger.info(
        f"Done! Shape per image: ({out_patches}, {hidden_dim}), "
        f"grid: {out_h}x{out_w}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract shift-and-average DINOv3 ViT-L/16 features"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_subdir", type=str, default="dinov3_features_shiftavg_vitl16")
    parser.add_argument("--n_shifts_per_axis", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    extract_shifted_features(
        cityscapes_root=Path(args.cityscapes_root),
        split=args.split,
        output_subdir=args.output_subdir,
        n_shifts_per_axis=args.n_shifts_per_axis,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
