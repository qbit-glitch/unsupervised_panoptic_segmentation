#!/usr/bin/env python3
"""Build a CAUSE-code cache whose depth channel comes from DA3.

The DCFA trainer reads cached files from:
    {codes_subdir}/{split}/{city}/{stem}_codes.npy
    {codes_subdir}/{split}/{city}/{stem}_depth.npy

The original cache was built from DepthPro by default. This script reuses the
frozen CAUSE code grids and replaces only the cached depth grid with DA3,
downsampled to the same patch grid as each code file.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO,
)
logger = logging.getLogger(__name__)


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link cached feature files when possible, otherwise copy them."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def find_depth(depth_root: Path, split: str, city: str, stem: str) -> Path | None:
    """Find DA3 depth saved either as stem.npy or stem_leftImg8bit.npy."""
    candidates = [
        depth_root / split / city / f"{stem}.npy",
        depth_root / split / city / f"{stem}_leftImg8bit.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def downsample_depth(depth: np.ndarray, ph: int, pw: int) -> np.ndarray:
    """Downsample full-resolution depth to the CAUSE patch grid."""
    depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return F.adaptive_avg_pool2d(depth_t, (ph, pw)).squeeze().numpy().astype(np.float32)


def build_split(
    cityscapes_root: Path,
    source_subdir: str,
    depth_subdir: str,
    output_subdir: str,
    split: str,
    overwrite_depth: bool,
) -> tuple[int, int]:
    """Build one split. Returns (processed, missing_depth)."""
    source_root = cityscapes_root / source_subdir / split
    depth_root = cityscapes_root / depth_subdir
    output_root = cityscapes_root / output_subdir / split

    code_files = sorted(source_root.glob("*/*_codes.npy"))
    logger.info("Split %s: found %d source code files", split, len(code_files))

    processed = 0
    missing_depth = 0
    for code_path in tqdm(code_files, desc=f"DA3 cache {split}"):
        city = code_path.parent.name
        stem = code_path.name.replace("_codes.npy", "")
        out_city = output_root / city
        out_city.mkdir(parents=True, exist_ok=True)

        out_code = out_city / code_path.name
        out_depth = out_city / f"{stem}_depth.npy"
        link_or_copy(code_path, out_code)

        # DINO features are image-only and reusable for future DCFA-X variants.
        dino_path = code_path.with_name(f"{stem}_dino768.npy")
        if dino_path.exists():
            link_or_copy(dino_path, out_city / dino_path.name)

        if out_depth.exists() and not overwrite_depth:
            processed += 1
            continue

        depth_path = find_depth(depth_root, split, city, stem)
        if depth_path is None:
            missing_depth += 1
            logger.warning("Missing DA3 depth for %s/%s", city, stem)
            continue

        codes = np.load(code_path, mmap_mode="r")
        ph, pw = codes.shape[:2]
        depth = np.load(depth_path).astype(np.float32)
        depth_ds = downsample_depth(depth, ph, pw)
        np.save(out_depth, depth_ds)
        processed += 1

    logger.info(
        "Split %s complete: processed=%d missing_depth=%d output=%s",
        split, processed, missing_depth, output_root,
    )
    return processed, missing_depth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cityscapes_root", type=Path, required=True)
    parser.add_argument("--source_subdir", default="cause_codes_90d")
    parser.add_argument("--depth_subdir", default="depth_dav3")
    parser.add_argument("--output_subdir", default="cause_codes_90d_da3")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--overwrite_depth", action="store_true")
    args = parser.parse_args()

    total_processed = 0
    total_missing = 0
    for split in args.splits:
        processed, missing = build_split(
            args.cityscapes_root,
            args.source_subdir,
            args.depth_subdir,
            args.output_subdir,
            split,
            args.overwrite_depth,
        )
        total_processed += processed
        total_missing += missing

    logger.info(
        "DA3 CAUSE-code cache complete: processed=%d missing_depth=%d",
        total_processed, total_missing,
    )
    if total_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
