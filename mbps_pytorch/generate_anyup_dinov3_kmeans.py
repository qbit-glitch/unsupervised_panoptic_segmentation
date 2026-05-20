#!/usr/bin/env python3
"""Generate K-means pseudo-labels from DINOv3 features upsampled by AnyUp.

This runner streams pre-extracted Cityscapes DINOv3 features through the
official AnyUp upsampler, optionally caches the dense upsampled tensors, fits
K-means on a patch subsample from train images, then assigns raw cluster IDs
for requested splits.

Default baseline:
    DINOv3 ViT-L/16 features: 32x64x1024
    AnyUp output grid:        64x128x1024
    Clusters:                 K=80

Usage:
    python3 mbps_pytorch/generate_anyup_dinov3_kmeans.py \
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
        --feat_subdir dinov3_features_vitl16 \
        --k 80 --target_h 64 --target_w 128 --splits val
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMAGE_H, IMAGE_W = 512, 1024


def find_feature_files(cityscapes_root: Path, split: str, feat_subdir: str) -> List[Dict]:
    feat_dir = cityscapes_root / feat_subdir / split
    files = []
    for city_dir in sorted(feat_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        for npy in sorted(city_dir.glob("*.npy")):
            stem = npy.stem.replace("_leftImg8bit", "")
            files.append({"feat": npy, "stem": stem, "city": city_dir.name})
    return files


def iter_batches(items: List[Dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def resolve_grid(n_patches: int) -> tuple[int, int]:
    known = {
        2048: (32, 64),
        8192: (64, 128),
        32768: (128, 256),
    }
    if n_patches in known:
        return known[n_patches]
    h = int(np.sqrt(n_patches * IMAGE_H / IMAGE_W))
    w = n_patches // max(h, 1)
    if h * w != n_patches:
        raise ValueError(f"Cannot infer grid for {n_patches} patches")
    return h, w


def load_anyup(project_root: Path, device: torch.device):
    anyup_root = project_root / "refs" / "anyup"
    if not anyup_root.exists():
        raise FileNotFoundError(
            f"{anyup_root} does not exist. Clone https://github.com/wimmerth/anyup "
            "to refs/anyup first."
        )
    sys.path.insert(0, str(anyup_root))
    os.environ.setdefault("TORCH_HOME", str(project_root / "weights" / "torchhub"))
    from hubconf import anyup_multi_backbone

    model = anyup_multi_backbone(use_natten=False, pretrained=True, device=str(device))
    model.eval()
    return model


def load_image_batch(
    cityscapes_root: Path, split: str, entries: List[Dict], device: torch.device
) -> torch.Tensor:
    tensors = []
    for entry in entries:
        img_path = (
            cityscapes_root
            / "leftImg8bit"
            / split
            / entry["city"]
            / f"{entry['stem']}_leftImg8bit.png"
        )
        img = Image.open(img_path).convert("RGB").resize((IMAGE_W, IMAGE_H), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    tensor = torch.stack(tensors, dim=0)
    mean = IMAGENET_MEAN.to(tensor.dtype)
    std = IMAGENET_STD.to(tensor.dtype)
    return ((tensor - mean) / std).to(device)


def load_feature_batch(entries: List[Dict], device: torch.device) -> torch.Tensor:
    tensors = []
    for entry in entries:
        feat = np.load(str(entry["feat"])).astype(np.float32)
        h, w = resolve_grid(feat.shape[0])
        tensors.append(torch.from_numpy(feat).reshape(h, w, -1).permute(2, 0, 1))
    return torch.stack(tensors, dim=0).to(device)


def cache_path(cache_dir: Path, split: str, entry: Dict) -> Path:
    return cache_dir / split / entry["city"] / f"{entry['stem']}.npy"


def flatten_and_normalize(raw: np.ndarray) -> np.ndarray:
    if raw.ndim == 3:
        feat = raw.reshape(-1, raw.shape[-1])
    elif raw.ndim == 2:
        feat = raw
    else:
        raise ValueError(f"Expected cached AnyUp feature with 2 or 3 dims, got {raw.shape}")
    feat = feat.astype(np.float32, copy=False)
    norms = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
    return feat / norms


def load_cached_anyup(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    raw = np.load(str(path))
    if raw.ndim == 3 and raw.shape[:2] != target_hw:
        raise ValueError(f"Cached feature {path} has grid {raw.shape[:2]}, expected {target_hw}")
    if raw.ndim == 2 and raw.shape[0] != target_hw[0] * target_hw[1]:
        raise ValueError(f"Cached feature {path} has {raw.shape[0]} tokens, expected {target_hw}")
    return flatten_and_normalize(raw)


def save_cached_anyup(path: Path, raw: np.ndarray, cache_dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if cache_dtype == "float16" else np.float32
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "wb") as f:
        np.save(f, raw.astype(dtype, copy=False))
    os.replace(tmp_path, path)


@torch.inference_mode()
def anyup_feature_batch(
    model,
    cityscapes_root: Path,
    split: str,
    entries: List[Dict],
    device: torch.device,
    target_hw: tuple[int, int],
    q_chunk_size: int,
    cache_dir: Path | None,
    cache_dtype: str,
    overwrite_cache: bool,
) -> List[np.ndarray]:
    outputs: List[np.ndarray | None] = [None] * len(entries)
    missing_entries = []
    missing_indices = []

    if cache_dir is not None:
        for i, entry in enumerate(entries):
            path = cache_path(cache_dir, split, entry)
            if path.exists() and not overwrite_cache:
                outputs[i] = load_cached_anyup(path, target_hw)
            else:
                missing_entries.append(entry)
                missing_indices.append(i)
    else:
        missing_entries = entries
        missing_indices = list(range(len(entries)))

    if missing_entries:
        img = load_image_batch(cityscapes_root, split, missing_entries, device)
        feat = load_feature_batch(missing_entries, device)
        out = model(img, feat, output_size=target_hw, q_chunk_size=q_chunk_size)
        out = out.permute(0, 2, 3, 1).float().cpu().numpy()
        for entry, idx, raw in zip(missing_entries, missing_indices, out):
            if cache_dir is not None:
                save_cached_anyup(cache_path(cache_dir, split, entry), raw, cache_dtype)
            outputs[idx] = flatten_and_normalize(raw)

    return [feat for feat in outputs if feat is not None]


def fit_kmeans(
    model,
    cityscapes_root: Path,
    train_files: List[Dict],
    device: torch.device,
    target_hw: tuple[int, int],
    k: int,
    sample_frac: float,
    batch_size: int,
    n_init: int,
    max_iter: int,
    q_chunk_size: int,
    upsample_batch_size: int,
    cache_dir: Path | None,
    cache_dtype: str,
    overwrite_cache: bool,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sampled = []
    t0 = time.time()
    n_per_image = max(1, int(target_hw[0] * target_hw[1] * sample_frac))
    logger.info(
        "Fitting K=%d on %d train images with AnyUp grid %dx%d, %d samples/image",
        k,
        len(train_files),
        target_hw[0],
        target_hw[1],
        n_per_image,
    )

    total_batches = (len(train_files) + upsample_batch_size - 1) // upsample_batch_size
    for batch in tqdm(
        iter_batches(train_files, upsample_batch_size),
        total=total_batches,
        desc="AnyUp train batches",
    ):
        feats = anyup_feature_batch(
            model,
            cityscapes_root,
            "train",
            batch,
            device,
            target_hw,
            q_chunk_size,
            cache_dir,
            cache_dtype,
            overwrite_cache,
        )
        for feat in feats:
            idx = rng.choice(feat.shape[0], min(n_per_image, feat.shape[0]), replace=False)
            sampled.append(feat[idx].astype(np.float32, copy=False))

    X = np.concatenate(sampled, axis=0)
    logger.info("Sample matrix %s loaded in %.1fs", X.shape, time.time() - t0)

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
        verbose=1,
    )
    t1 = time.time()
    kmeans.fit(X)
    logger.info("K-means finished in %.1fs, inertia=%.3f", time.time() - t1, kmeans.inertia_)
    centers = kmeans.cluster_centers_.astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
    return centers


def assign_split(
    model,
    cityscapes_root: Path,
    files: List[Dict],
    split: str,
    device: torch.device,
    centers: np.ndarray,
    out_dir: Path,
    target_hw: tuple[int, int],
    q_chunk_size: int,
    upsample_batch_size: int,
    cache_dir: Path | None,
    cache_dtype: str,
    overwrite_cache: bool,
) -> None:
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    centers = centers.astype(np.float32)
    h, w = target_hw

    total_batches = (len(files) + upsample_batch_size - 1) // upsample_batch_size
    for batch in tqdm(
        iter_batches(files, upsample_batch_size),
        total=total_batches,
        desc=f"Assigning {split}",
    ):
        feats = anyup_feature_batch(
            model,
            cityscapes_root,
            split,
            batch,
            device,
            target_hw,
            q_chunk_size,
            cache_dir,
            cache_dtype,
            overwrite_cache,
        )
        for entry, feat in zip(batch, feats):
            labels = (feat @ centers.T).argmax(axis=1).astype(np.uint8)
            label_grid = labels.reshape(h, w)
            label_full = np.array(
                Image.fromarray(label_grid).resize((IMAGE_W, IMAGE_H), Image.NEAREST)
            )
            city_dir = split_dir / entry["city"]
            city_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(label_full).save(str(city_dir / f"{entry['stem']}.png"))


def main():
    parser = argparse.ArgumentParser(description="DINOv3 + AnyUp K-means pseudo-labels")
    parser.add_argument("--cityscapes_root", required=True)
    parser.add_argument("--feat_subdir", default="dinov3_features_vitl16")
    parser.add_argument("--output_subdir", default=None)
    parser.add_argument(
        "--output_root",
        default=None,
        help="Directory that will contain output_subdir. Defaults to cityscapes_root; "
        "use a repo-local path when the dataset root is read-only.",
    )
    parser.add_argument("--k", type=int, default=80)
    parser.add_argument("--target_h", type=int, default=64)
    parser.add_argument("--target_w", type=int, default=128)
    parser.add_argument("--sample_frac", type=float, default=0.025)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q_chunk_size", type=int, default=512)
    parser.add_argument("--upsample_batch_size", type=int, default=1)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--load_centroids", default=None)
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_assign", type=int, default=0)
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional directory for reusable raw AnyUp tensors, e.g. on /Volumes/code_files.",
    )
    parser.add_argument("--cache_dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--overwrite_cache", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cityscapes_root = Path(args.cityscapes_root)
    device = torch.device(args.device)
    target_hw = (args.target_h, args.target_w)
    out_name = args.output_subdir or f"pseudo_semantic_raw_dinov3_anyup_{args.target_h}x{args.target_w}_k{args.k}"
    output_root = Path(args.output_root) if args.output_root else cityscapes_root
    out_dir = output_root / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Using AnyUp feature cache: %s (%s)", cache_dir, args.cache_dtype)

    model = load_anyup(project_root, device)

    if args.load_centroids:
        data = np.load(args.load_centroids)
        centers = data["centers"].astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
        logger.info("Loaded centers from %s", args.load_centroids)
    else:
        train_files = find_feature_files(cityscapes_root, "train", args.feat_subdir)
        if args.limit_train:
            train_files = train_files[: args.limit_train]
        centers = fit_kmeans(
            model,
            cityscapes_root,
            train_files,
            device,
            target_hw,
            args.k,
            args.sample_frac,
            args.batch_size,
            args.n_init,
            args.max_iter,
            args.q_chunk_size,
            args.upsample_batch_size,
            cache_dir,
            args.cache_dtype,
            args.overwrite_cache,
            args.seed,
        )
        np.savez(str(out_dir / "centroids.npz"), centers=centers)
        logger.info("Saved centers to %s", out_dir / "centroids.npz")

    for split in args.splits:
        files = find_feature_files(cityscapes_root, split, args.feat_subdir)
        if args.limit_assign:
            files = files[: args.limit_assign]
        assign_split(
            model,
            cityscapes_root,
            files,
            split,
            device,
            centers,
            out_dir,
            target_hw,
            args.q_chunk_size,
            args.upsample_batch_size,
            cache_dir,
            args.cache_dtype,
            args.overwrite_cache,
        )

    meta = {
        "feat_subdir": args.feat_subdir,
        "output_subdir": out_name,
        "k": args.k,
        "target_hw": list(target_hw),
        "sample_frac": args.sample_frac,
        "q_chunk_size": args.q_chunk_size,
        "upsample_batch_size": args.upsample_batch_size,
        "splits": args.splits,
        "device": str(device),
        "cache_dir": str(cache_dir) if cache_dir else None,
        "cache_dtype": args.cache_dtype,
        "overwrite_cache": args.overwrite_cache,
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("All done -> %s", out_dir)


if __name__ == "__main__":
    main()
