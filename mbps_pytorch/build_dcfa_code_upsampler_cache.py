#!/usr/bin/env python3
"""Cache paired low/high DCFA 90D code maps for code-space upsampler training.

For each Cityscapes image this script computes:
  - low_code:     adapted DCFA 90D code grid, e.g. 23x46
  - teacher_code: denser crop-teacher DCFA 90D grid, e.g. 64x128
  - rgb/depth guidance at teacher resolution

The teacher grid is produced from CAUSE's sliding-window crop features before
pooling, then passed through the frozen DCFA adapter at the higher grid. This
keeps the target in the same 90D DCFA semantic space while giving the upsampler
a denser spatial signal.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mbps_pytorch.generate_depth_overclustered_semantics import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    _apply_depth_adapter,
    _load_depth_adapter,
    downsample_depth,
    get_cityscapes_images,
    load_cause_models,
    load_depth_map,
    sliding_window_features,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dcfa_code_cache")


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp_path, path)


def adapt_grid(
    adapter: torch.nn.Module,
    code_chw: torch.Tensor,
    depth_hw: np.ndarray,
    cityscapes_root: str,
    codes_subdir: str,
    split: str,
    city: str,
    stem: str,
    device: torch.device,
) -> np.ndarray:
    code_np = code_chw.detach().cpu().numpy().astype(np.float32)
    channels, height, width = code_np.shape
    if channels != 90:
        raise ValueError(f"Expected 90D code map, got {code_np.shape}")
    feats_90 = code_np.reshape(90, -1).T
    adjusted = _apply_depth_adapter(
        adapter,
        feats_90,
        depth_hw.astype(np.float32),
        cityscapes_root,
        codes_subdir,
        split,
        city,
        stem,
        height,
        width,
        device,
    )
    return adjusted.T.reshape(90, height, width).astype(np.float32)


def pool_code_grid(code_map: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Average-pool a 90D map, with an MPS fallback for non-divisible sizes."""
    x = code_map.unsqueeze(0)
    if code_map.device.type == "mps":
        pooled = F.adaptive_avg_pool2d(x.cpu(), target_hw).to(code_map.device)
    else:
        pooled = F.adaptive_avg_pool2d(x, target_hw)
    return pooled.squeeze(0)


@torch.inference_mode()
def build_cache(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    logger.info("Using device: %s", device)
    net, segment, _ = load_cause_models(args.checkpoint_dir, device)
    adapter = _load_depth_adapter(args.adapter_checkpoint, device)
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    images = get_cityscapes_images(args.cityscapes_root, args.split)
    if args.limit_images > 0:
        images = images[: args.limit_images]
    logger.info("Caching %d %s images to %s", len(images), args.split, args.output_dir)

    out_root = Path(args.output_dir) / args.split
    low_hw = (args.low_h, args.low_w)
    high_hw = (args.high_h, args.high_w)

    for entry in tqdm(images, desc=f"cache {args.split}"):
        out_path = out_root / entry["city"] / f"{entry['stem']}.npz"
        if out_path.exists() and not args.overwrite:
            continue

        img_pil = Image.open(entry["img_path"]).convert("RGB")
        orig_w, orig_h = img_pil.size
        scale = args.crop_size / min(orig_h, orig_w)
        new_h = (int(round(orig_h * scale)) // args.patch_size) * args.patch_size
        new_w = (int(round(orig_w * scale)) // args.patch_size) * args.patch_size
        img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
        img_tensor = normalize(transforms.ToTensor()(img_resized)).unsqueeze(0).to(device)

        feat_map = sliding_window_features(net, segment, img_tensor, args.crop_size)
        low_cause = pool_code_grid(feat_map, low_hw)
        high_cause = pool_code_grid(feat_map, high_hw)

        depth_map = load_depth_map(
            args.cityscapes_root,
            args.depth_subdir,
            args.split,
            entry["city"],
            entry["stem"],
        )
        if depth_map is None:
            depth_low = np.zeros(low_hw, dtype=np.float32)
            depth_high = np.zeros(high_hw, dtype=np.float32)
        else:
            depth_low = downsample_depth(depth_map, *low_hw)
            depth_high = downsample_depth(depth_map, *high_hw)

        low_code = adapt_grid(
            adapter,
            low_cause,
            depth_low,
            args.cityscapes_root,
            args.codes_subdir,
            args.split,
            entry["city"],
            entry["stem"],
            device,
        )
        teacher_code = adapt_grid(
            adapter,
            high_cause,
            depth_high,
            args.cityscapes_root,
            args.codes_subdir,
            args.split,
            entry["city"],
            entry["stem"],
            device,
        )

        rgb = np.asarray(
            img_pil.resize((args.high_w, args.high_h), Image.BILINEAR),
            dtype=np.uint8,
        )
        save_npz_atomic(
            out_path,
            low_code=low_code.astype(np.float16),
            teacher_code=teacher_code.astype(np.float16),
            rgb=rgb,
            depth=depth_high.astype(np.float16),
            low_hw=np.asarray(low_hw, dtype=np.int16),
            high_hw=np.asarray(high_hw, dtype=np.int16),
        )

    logger.info("Done. Cache root: %s", args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cityscapes_root", default="/Users/qbit-glitch/Desktop/datasets/cityscapes")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--output_dir", default="outputs/code_upsampler/cache_dcfa_v3_90d_64x128")
    parser.add_argument("--checkpoint_dir", default="refs/cause")
    parser.add_argument("--adapter_checkpoint", default="results/depth_adapter/V3_dd16_h384_l2/best.pt")
    parser.add_argument("--depth_subdir", default="depth_depthpro")
    parser.add_argument("--codes_subdir", default="cause_codes_90d")
    parser.add_argument("--crop_size", type=int, default=322)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--low_h", type=int, default=23)
    parser.add_argument("--low_w", type=int, default=46)
    parser.add_argument("--high_h", type=int, default=64)
    parser.add_argument("--high_w", type=int, default=128)
    parser.add_argument("--limit_images", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_cache(args)


if __name__ == "__main__":
    main()
