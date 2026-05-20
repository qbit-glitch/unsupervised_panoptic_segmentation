#!/usr/bin/env python3
"""Generate fixed-K cluster PNGs from trained DCFA 90D code upsamplers."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mbps_pytorch.train_code_upsampler import make_model, pick_device


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("code_upsampler_kmeans")

IMAGE_H, IMAGE_W = 512, 1024


def discover_cache_files(cache_dir: Path, split: str) -> List[Path]:
    files = sorted((cache_dir / split).rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No cache .npz files found under {cache_dir / split}")
    return files


def iter_batches(items: List[Path], batch_size: int) -> Iterable[List[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_batch(paths: List[Path], device: torch.device):
    lows = []
    guides = []
    for path in paths:
        data = np.load(path)
        lows.append(torch.from_numpy(data["low_code"].astype(np.float32)))
        rgb = torch.from_numpy(data["rgb"].astype(np.float32) / 255.0).permute(2, 0, 1)
        depth = torch.from_numpy(data["depth"].astype(np.float32)).unsqueeze(0)
        guides.append(torch.cat([rgb, depth], dim=0))
    low = torch.stack(lows, dim=0).to(device)
    guide = torch.stack(guides, dim=0).to(device)
    return low, guide


@torch.inference_mode()
def predict_features(model: torch.nn.Module, paths: List[Path], device: torch.device) -> np.ndarray:
    low, guide = load_batch(paths, device)
    pred = model(low, guide, output_size=guide.shape[-2:])
    pred = F.normalize(pred.float(), dim=1, eps=1e-6)
    return pred.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)


def load_trained_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    raw = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = dict(raw.get("args", {}))
    if "variant" not in ckpt_args:
        raise ValueError(f"Checkpoint missing training args: {checkpoint_path}")
    args = SimpleNamespace(**ckpt_args)
    model = make_model(args)
    model.load_state_dict(raw["model"])
    model.to(device).eval()
    return model


def fit_kmeans(
    model: torch.nn.Module,
    files: List[Path],
    device: torch.device,
    k: int,
    sample_frac: float,
    inference_batch_size: int,
    kmeans_batch_size: int,
    n_init: int,
    max_iter: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sampled = []
    t0 = time.time()
    total_batches = (len(files) + inference_batch_size - 1) // inference_batch_size
    for batch in tqdm(
        iter_batches(files, inference_batch_size),
        total=total_batches,
        desc="Upsampler train features",
    ):
        feats = predict_features(model, batch, device)
        flat = feats.reshape(feats.shape[0], -1, feats.shape[-1])
        n_tokens = flat.shape[1]
        n_pick = max(1, int(n_tokens * sample_frac))
        for image_feat in flat:
            idx = rng.choice(n_tokens, min(n_pick, n_tokens), replace=False)
            sampled.append(image_feat[idx])

    matrix = np.concatenate(sampled, axis=0).astype(np.float32)
    logger.info("Sample matrix %s loaded in %.1fs", matrix.shape, time.time() - t0)
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=kmeans_batch_size,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
        verbose=1,
    )
    t1 = time.time()
    kmeans.fit(matrix)
    logger.info("KMeans finished in %.1fs, inertia=%.3f", time.time() - t1, kmeans.inertia_)
    centers = kmeans.cluster_centers_.astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
    return centers


def assign_split(
    model: torch.nn.Module,
    files: List[Path],
    cache_dir: Path,
    split: str,
    out_dir: Path,
    centers: np.ndarray,
    device: torch.device,
    inference_batch_size: int,
) -> None:
    split_out = out_dir / split
    split_out.mkdir(parents=True, exist_ok=True)
    centers = centers.astype(np.float32)
    total_batches = (len(files) + inference_batch_size - 1) // inference_batch_size
    for batch in tqdm(iter_batches(files, inference_batch_size), total=total_batches, desc=f"Assign {split}"):
        feats = predict_features(model, batch, device)
        bsz, height, width, _ = feats.shape
        flat = feats.reshape(bsz, height * width, -1)
        labels = np.argmax(flat @ centers.T, axis=2).astype(np.uint8)
        for path, label in zip(batch, labels):
            rel = path.relative_to(cache_dir / split)
            city = rel.parent.name
            stem = path.stem
            label_grid = label.reshape(height, width)
            label_full = np.array(
                Image.fromarray(label_grid).resize((IMAGE_W, IMAGE_H), Image.NEAREST)
            )
            city_dir = split_out / city
            city_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(label_full).save(city_dir / f"{stem}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", default="outputs/code_upsampler/cache_dcfa_v3_90d_64x128")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k", type=int, default=80)
    parser.add_argument("--sample_frac", type=float, default=0.025)
    parser.add_argument("--inference_batch_size", type=int, default=8)
    parser.add_argument("--kmeans_batch_size", type=int, default=4096)
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_assign", type=int, default=0)
    parser.add_argument("--load_centers", default=None)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    logger.info("Using device: %s", device)

    model = load_trained_model(Path(args.checkpoint), device)
    if args.load_centers:
        data = np.load(args.load_centers)
        centers = data["centers"].astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
        logger.info("Loaded centers from %s", args.load_centers)
    else:
        train_files = discover_cache_files(cache_dir, "train")
        if args.limit_train > 0:
            train_files = train_files[: args.limit_train]
        centers = fit_kmeans(
            model,
            train_files,
            device,
            args.k,
            args.sample_frac,
            args.inference_batch_size,
            args.kmeans_batch_size,
            args.n_init,
            args.max_iter,
            args.seed,
        )
        np.savez_compressed(
            out_dir / "kmeans_centers.npz",
            centers=centers,
            checkpoint=str(args.checkpoint),
            k=np.array(args.k, dtype=np.int32),
            sample_frac=np.array(args.sample_frac, dtype=np.float32),
            seed=np.array(args.seed, dtype=np.int32),
        )
        logger.info("Saved centers to %s", out_dir / "kmeans_centers.npz")

    val_files = discover_cache_files(cache_dir, "val")
    if args.limit_assign > 0:
        val_files = val_files[: args.limit_assign]
    assign_split(
        model,
        val_files,
        cache_dir,
        "val",
        out_dir,
        centers,
        device,
        args.inference_batch_size,
    )
    logger.info("Done. Output: %s", out_dir / "val")


if __name__ == "__main__":
    main()
