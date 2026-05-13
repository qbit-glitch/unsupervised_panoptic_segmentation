#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import importlib.util
import logging
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

_TYPE_PATH = Path(__file__).resolve().parents[1] / "refs" / "cups" / "cups" / "rare_instance_pool_types.py"
_TYPE_SPEC = importlib.util.spec_from_file_location("cups.rare_instance_pool_types", _TYPE_PATH)
_TYPE_MODULE = importlib.util.module_from_spec(_TYPE_SPEC)
sys.modules["cups.rare_instance_pool_types"] = _TYPE_MODULE
_TYPE_SPEC.loader.exec_module(_TYPE_MODULE)
InstanceCrop = _TYPE_MODULE.InstanceCrop

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RARE_TRAINIDS = {3, 4, 5, 6, 8, 9, 11, 12, 14, 15, 16, 18}
DROP_TRAINIDS = {8}
WARN_EMPTY_TRAINIDS = sorted((RARE_TRAINIDS | {17}) - DROP_TRAINIDS)


def _load_cluster_to_class(path: Path) -> np.ndarray:
    data = np.load(path)
    if "cluster_to_class" not in data:
        raise KeyError(f"{path} does not contain cluster_to_class")
    raw = data["cluster_to_class"].astype(np.uint8)
    lut = np.full(256, 255, dtype=np.uint8)
    lut[: len(raw)] = raw
    # Defensive fallback: if a centroids file omits the first 19 indices
    # (e.g. partial/legacy artifacts), default unmapped cluster i in [0,19)
    # to trainID i so the script does not silently drop those clusters.
    # For our k=80 production centroids this branch never triggers because
    # all 80 cluster slots are populated.
    for cls_id in range(19):
        if lut[cls_id] == 255:
            lut[cls_id] = cls_id
    return lut


def _city_from_stem(stem: str) -> str:
    return stem.split("_")[0]


def _stem_from_semantic(path: Path) -> str:
    return path.name.replace("_semantic.png", "")


def _resolve_image(image_dir: Path, stem: str) -> Path | None:
    city = _city_from_stem(stem)
    variants = [f"{stem}.png"]
    if stem.endswith("_leftImg8bit"):
        variants.append(f"{stem.replace('_leftImg8bit', '')}_leftImg8bit.png")
    else:
        variants.append(f"{stem}_leftImg8bit.png")
    roots = [image_dir, image_dir / city]
    for root in roots:
        for name in variants:
            p = root / name
            if p.exists():
                return p
    return None


def _resolve_depth(depth_dir: Path, stem: str) -> Path | None:
    city = _city_from_stem(stem)
    base = stem.replace("_leftImg8bit", "")
    variants = [f"{base}.npy", f"{stem}.npy"]
    roots = [depth_dir, depth_dir / city]
    for root in roots:
        for name in variants:
            p = root / name
            if p.exists():
                return p
    return None


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _encode_jpeg(arr: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def _encode_mask_png(mask: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def _normalise_depth(depth: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if depth.shape != shape:
        depth = np.array(Image.fromarray(depth.astype(np.float32)).resize((shape[1], shape[0]), Image.BILINEAR))
    depth = depth.astype(np.float32)
    d_min, d_max = float(np.nanmin(depth)), float(np.nanmax(depth))
    if d_max > d_min:
        depth = (depth - d_min) / (d_max - d_min)
    return np.nan_to_num(depth, nan=0.5, posinf=1.0, neginf=0.0)


def _extract_for_file(
    sem_path: Path,
    image_dir: Path,
    depth_dir: Path,
    cluster_to_class: np.ndarray,
    min_area: int,
    min_side: int,
) -> Dict[int, List[InstanceCrop]]:
    inst_path = sem_path.with_name(sem_path.name.replace("_semantic.png", "_instance.png"))
    if not inst_path.exists():
        logger.warning("Missing instance PNG for %s", sem_path.name)
        return {}
    stem = _stem_from_semantic(sem_path)
    image_path = _resolve_image(image_dir, stem)
    depth_path = _resolve_depth(depth_dir, stem)
    if image_path is None:
        logger.warning("Missing image for %s", stem)
        return {}

    semantic = np.array(Image.open(sem_path))
    if semantic.ndim == 3:
        semantic = semantic[..., 0]
    instance = np.array(Image.open(inst_path))
    if instance.ndim == 3:
        instance = instance[..., 0]
    image = np.array(Image.open(image_path).convert("RGB"))
    if image.shape[:2] != semantic.shape:
        image = np.array(Image.fromarray(image).resize((semantic.shape[1], semantic.shape[0]), Image.BILINEAR))
    if depth_path is not None:
        depth = _normalise_depth(np.load(depth_path), semantic.shape)
    else:
        depth = np.full(semantic.shape, 0.5, dtype=np.float32)

    out: Dict[int, List[InstanceCrop]] = {}
    for instance_id in np.unique(instance):
        if int(instance_id) == 0:
            continue
        mask = instance == instance_id
        area = int(mask.sum())
        if area < min_area:
            continue
        bbox = _bbox_from_mask(mask)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        if min(x2 - x1, y2 - y1) < min_side:
            continue
        clusters = semantic[mask]
        valid_clusters = clusters[clusters < len(cluster_to_class)]
        if valid_clusters.size == 0:
            continue
        majority_cluster = int(np.bincount(valid_clusters.astype(np.int64)).argmax())
        train_id = int(cluster_to_class[majority_cluster])
        if train_id in DROP_TRAINIDS or train_id not in RARE_TRAINIDS:
            continue

        crop_img = image[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]
        depth_values = depth[mask]
        src_depth_quantile = float(np.median(depth_values)) if depth_values.size else 0.5
        out.setdefault(train_id, []).append(
            InstanceCrop(
                train_id=train_id,
                image_jpeg=_encode_jpeg(crop_img),
                mask_png=_encode_mask_png(crop_mask),
                src_depth_quantile=src_depth_quantile,
                bbox=(x1, y1, x2, y2),
                area=area,
            )
        )
    return out


def build_pool(
    pseudo_dir: Path,
    image_dir: Path,
    depth_dir: Path,
    centroids: Path,
    max_per_class: int = 3000,
    workers: int = 8,
    min_area: int = 256,
    min_side: int = 32,
) -> Dict[int, List[InstanceCrop]]:
    cluster_to_class = _load_cluster_to_class(centroids)
    sem_paths = sorted(pseudo_dir.rglob("*_semantic.png"))
    logger.info("Found %d semantic pseudo-labels", len(sem_paths))
    pool: Dict[int, List[InstanceCrop]] = {c: [] for c in sorted(RARE_TRAINIDS - DROP_TRAINIDS)}

    def process(path: Path) -> Dict[int, List[InstanceCrop]]:
        return _extract_for_file(
            sem_path=path,
            image_dir=image_dir,
            depth_dir=depth_dir,
            cluster_to_class=cluster_to_class,
            min_area=min_area,
            min_side=min_side,
        )

    # Use ProcessPool by default for true CPU parallelism (PIL JPEG encode +
    # numpy ops dominate; the worker function is module-level so it pickles
    # cleanly). Fall back to ThreadPool if processes can't be spawned (e.g.
    # nested executor inside an async context).
    executor_cls = ProcessPoolExecutor if workers > 1 else ThreadPoolExecutor
    try:
        with executor_cls(max_workers=max(1, workers)) as executor:
            for partial in tqdm(executor.map(process, sem_paths), total=len(sem_paths), desc="rare-pool"):
                for train_id, crops in partial.items():
                    remaining = max_per_class - len(pool.setdefault(train_id, []))
                    if remaining > 0:
                        pool[train_id].extend(crops[:remaining])
    except (OSError, RuntimeError) as exc:
        logger.warning("ProcessPool failed (%s); retrying with ThreadPool", exc)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for partial in tqdm(executor.map(process, sem_paths), total=len(sem_paths), desc="rare-pool"):
                for train_id, crops in partial.items():
                    remaining = max_per_class - len(pool.setdefault(train_id, []))
                    if remaining > 0:
                        pool[train_id].extend(crops[:remaining])

    for train_id in WARN_EMPTY_TRAINIDS:
        count = len(pool.get(train_id, []))
        if count == 0:
            logger.warning("Rare pool bucket trainID=%d is empty", train_id)
        else:
            logger.info("Rare pool bucket trainID=%d: %d crops", train_id, count)
    return pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a persistent rare-instance copy-paste pool.")
    parser.add_argument("--pseudo-dir", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--depth-dir", required=True, type=Path)
    parser.add_argument("--centroids", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-per-class", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool = build_pool(
        pseudo_dir=args.pseudo_dir.expanduser(),
        image_dir=args.image_dir.expanduser(),
        depth_dir=args.depth_dir.expanduser(),
        centroids=args.centroids.expanduser(),
        max_per_class=args.max_per_class,
        workers=args.workers,
    )
    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with args.out.expanduser().open("wb") as f:
        pickle.dump(pool, f, protocol=4)
    total = sum(len(v) for v in pool.values())
    logger.info("Wrote %d crops across %d classes to %s", total, len(pool), args.out.expanduser())


if __name__ == "__main__":
    main()
