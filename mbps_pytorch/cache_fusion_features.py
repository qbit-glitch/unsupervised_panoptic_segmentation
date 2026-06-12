#!/usr/bin/env python3
"""Build the fusion feature cache: frozen CAUSE-TR codes + frozen DepthG codes.

Two independent passes (separate processes — CAUSE's `modules` package and
DepthG's `modules.py` collide in sys.path, so never import both stacks in one
process):

  --model cause   CAUSE z via the canonical full-image forward reused from
                  generate_depth_overclustered_semantics (shortest side -> 322,
                  patch-multiple resize 322x644, sliding 322-crops, flip-avg),
                  pooled to the (23, 46) token grid -> (23, 46, 90) fp16.
                  Also writes DepthPro depth pooled to (23, 46), per-image
                  min-max normalized to [0, 1], fp16.
  --model depthg  DepthG g via the model's own half-res inference path
                  (640x1280 -> 320x640, flip-averaged net code) ->
                  (40, 80, 100) fp16.

Layout:
    {out_root}/cause_z/{split}/{city}/{stem}_codes.npy
    {out_root}/cause_z/{split}/{city}/{stem}_depth.npy
    {out_root}/{g_subdir}/{split}/{city}/{stem}_g.npy

Smoke:
    .venv_cups_cpu/bin/python mbps_pytorch/cache_fusion_features.py \
        --model cause --cities aachen --limit 3 --device cpu
    .venv_cups_cpu/bin/python mbps_pytorch/cache_fusion_features.py \
        --model depthg --cities aachen --limit 3 --device cpu

Full (background, per user convention):
    nohup .venv_cups_cpu/bin/python mbps_pytorch/cache_fusion_features.py \
        --model cause --device mps > logs/cache_fusion_cause_TS.log 2>&1 &
    nohup .venv_cups_cpu/bin/python mbps_pytorch/cache_fusion_features.py \
        --model depthg --device mps > logs/cache_fusion_depthg_TS.log 2>&1 &
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DATA_ROOT = Path("/Volumes/code_files/datasets/cityscapes")
CACHE_ROOT = DATA_ROOT / "fusion_feature_cache"
MONO_CKPT = PROJECT_ROOT / "checkpoints" / "depthg_depthpro_monocular" / "epoch6_step1680.ckpt"
PATCH = 14
CROP = 322


def load_image(path: Path, size_hw: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size_hw[1], size_hw[0]), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def list_frames(split: str, cities: List[str] | None, limit: int | None):
    img_root = DATA_ROOT / "leftImg8bit" / split
    cities = cities or sorted(d.name for d in img_root.iterdir() if d.is_dir())
    out = []
    for city in cities:
        files = sorted((img_root / city).glob("*_leftImg8bit.png"))
        if limit:
            files = files[:limit]
        out.extend((city, f) for f in files)
    return out


def cause_target_size(orig_hw: Tuple[int, int]) -> Tuple[int, int]:
    """Shortest side -> CROP, both sides floored to PATCH multiples
    (mirrors generate_depth_overclustered_semantics image prep)."""
    h, w = orig_hw
    scale = CROP / min(h, w)
    return (int(round(h * scale)) // PATCH) * PATCH, (int(round(w * scale)) // PATCH) * PATCH


def find_depth(split: str, city: str, stem: str) -> Path | None:
    root = DATA_ROOT / "depth_depthpro" / split / city
    for cand in (root / f"{stem}_leftImg8bit.npy", root / f"{stem}.npy"):
        if cand.is_file():
            return cand
    return None


def run_cause(args: argparse.Namespace, device: torch.device) -> None:
    from mbps_pytorch.generate_depth_overclustered_semantics import (
        load_cause_models,
        sliding_window_features,
    )

    net, segment, _ = load_cause_models(str(PROJECT_ROOT / "refs" / "cause"), device)
    frames = list_frames(args.split, args.cities, args.limit)
    logger.info("CAUSE pass: %d frames -> %s", len(frames), CACHE_ROOT / "cause_z" / args.split)

    t0, done = time.time(), 0
    for city, f in frames:
        stem = f.name.replace("_leftImg8bit.png", "")
        out_dir = CACHE_ROOT / "cause_z" / args.split / city
        out_dir.mkdir(parents=True, exist_ok=True)
        z_out, d_out = out_dir / f"{stem}_codes.npy", out_dir / f"{stem}_depth.npy"
        if z_out.exists() and d_out.exists():
            continue

        img = Image.open(f).convert("RGB")
        th, tw = cause_target_size((img.size[1], img.size[0]))
        x = load_image(f, (th, tw)).to(device)
        ph, pw = th // PATCH, tw // PATCH

        if not z_out.exists():
            with torch.no_grad():
                feat_map = sliding_window_features(net, segment, x, CROP)  # (90, th, tw)
                z = F.adaptive_avg_pool2d(feat_map.unsqueeze(0), (ph, pw)).squeeze(0)
            np.save(z_out, z.permute(1, 2, 0).cpu().numpy().astype(np.float16))

        if not d_out.exists():
            dp = find_depth(args.split, city, stem)
            if dp is None:
                logger.warning("missing DepthPro depth for %s/%s", city, stem)
            else:
                d = torch.from_numpy(np.load(dp).astype(np.float32))[None, None]
                d = F.adaptive_avg_pool2d(d, (ph, pw)).squeeze()
                d = (d - d.min()) / (d.max() - d.min() + 1e-8)
                np.save(d_out, d.numpy().astype(np.float16))

        done += 1
        if done % 25 == 0:
            logger.info("cause %d/%d (%.1fs/img)", done, len(frames), (time.time() - t0) / done)
    logger.info("CAUSE pass complete: %d new", done)


def run_depthg(args: argparse.Namespace, device: torch.device) -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "refs" / "cups"))
    sys.path.insert(0, str(PROJECT_ROOT / "refs" / "cups" / "external" / "depthg"))
    sys.path.insert(0, str(PROJECT_ROOT / "refs" / "cups" / "external" / "depthg" / "src"))
    from cups.semantics.model import DepthG  # noqa: E402

    model = DepthG(device=device, checkpoint_root=str(args.depthg_ckpt),
                   img_shape=(640, 1280), stride=(160, 160), crop=(320, 320))
    frames = list_frames(args.split, args.cities, args.limit)
    out_root = CACHE_ROOT / args.g_subdir / args.split
    logger.info("DepthG pass: %d frames -> %s (ckpt=%s)", len(frames), out_root, args.depthg_ckpt)

    t0, done = time.time(), 0
    for city, f in frames:
        stem = f.name.replace("_leftImg8bit.png", "")
        out_dir = out_root / city
        out_dir.mkdir(parents=True, exist_ok=True)
        g_out = out_dir / f"{stem}_g.npy"
        if g_out.exists():
            continue

        x = load_image(f, (640, 1280)).to(device)
        with torch.no_grad():
            small = F.interpolate(x, (320, 640), mode="bilinear", align_corners=False)
            code = model(small)                       # (1, d_g, 40, 80)
            code2 = model(small.flip(dims=[3]))
            code = (code + code2.flip(dims=[3])) / 2  # flip-avg, mirrors model.py:102-104
        np.save(g_out, code.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float16))

        done += 1
        if done % 25 == 0:
            logger.info("depthg %d/%d (%.1fs/img)", done, len(frames), (time.time() - t0) / done)
    logger.info("DepthG pass complete: %d new", done)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=("cause", "depthg"))
    p.add_argument("--split", default="train")
    p.add_argument("--cities", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None, choices=("cpu", "mps"))
    p.add_argument("--depthg_ckpt", type=Path, default=MONO_CKPT)
    p.add_argument("--g_subdir", default="depthg_g_mono",
                   help="use depthg_g_official when caching the release ckpt")
    args = p.parse_args()
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    logger.info("model=%s split=%s device=%s", args.model, args.split, device)

    if args.model == "cause":
        run_cause(args, device)
    else:
        run_depthg(args, device)


if __name__ == "__main__":
    main()
