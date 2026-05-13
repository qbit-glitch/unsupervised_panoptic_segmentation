#!/usr/bin/env python3
"""Convert Waymo Open Dataset v2 camera panoptic parquet files to CUPS layout.

The CUPS Waymo validation loader expects:

    <out>/validation/<segment>/<frame>_<camera>_image.jpg
    <out>/validation/<segment>/<frame>_<camera>_semantic.png
    <out>/validation/<segment>/<frame>_<camera>_instance.png

Waymo v2 stores RGB frames and camera panoptic labels in separate parquet
components. This script materializes only frames that have camera segmentation
labels, matching rows by (segment, timestamp, camera_name).
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


IMAGE_COL = "[CameraImageComponent].image"
PANOPTIC_COL = "[CameraSegmentationLabelComponent].panoptic_label"
DIVISOR_COL = "[CameraSegmentationLabelComponent].panoptic_label_divisor"


def _iter_parquets(path: Path) -> Iterable[Path]:
    return sorted(path.glob("*.parquet"))


def _row_key(row: dict) -> Tuple[str, int, int]:
    return (
        str(row["key.segment_context_name"]),
        int(row["key.frame_timestamp_micros"]),
        int(row["key.camera_name"]),
    )


def _safe_segment_dir(name: str) -> str:
    return name.replace("/", "_")


def build_image_index(camera_image_dir: Path, cache_path: Path | None = None) -> Dict[Tuple[str, int, int], Tuple[str, int]]:
    if cache_path is not None and cache_path.exists():
        payload = json.loads(cache_path.read_text())
        return {
            (item["segment"], int(item["timestamp"]), int(item["camera"])): (item["file"], int(item["row"]))
            for item in payload
        }

    index: Dict[Tuple[str, int, int], Tuple[str, int]] = {}
    files = list(_iter_parquets(camera_image_dir))
    for parquet_path in tqdm(files, desc="Indexing camera_image parquet"):
        table = pq.read_table(
            parquet_path,
            columns=["key.segment_context_name", "key.frame_timestamp_micros", "key.camera_name"],
        )
        rows = table.to_pylist()
        for row_idx, row in enumerate(rows):
            index[_row_key(row)] = (parquet_path.name, row_idx)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = [
            {
                "segment": key[0],
                "timestamp": key[1],
                "camera": key[2],
                "file": value[0],
                "row": value[1],
            }
            for key, value in index.items()
        ]
        cache_path.write_text(json.dumps(serializable))

    return index


def _load_image_row(parquet_path: Path, row_idx: int) -> bytes:
    row_group = None
    offset = row_idx
    pf = pq.ParquetFile(parquet_path)
    running = 0
    for group_idx in range(pf.num_row_groups):
        nrows = pf.metadata.row_group(group_idx).num_rows
        if running <= row_idx < running + nrows:
            row_group = group_idx
            offset = row_idx - running
            break
        running += nrows
    if row_group is None:
        raise IndexError(f"row {row_idx} out of range for {parquet_path}")
    table = pf.read_row_group(row_group, columns=[IMAGE_COL])
    return table.slice(offset, 1).to_pylist()[0][IMAGE_COL]


def convert(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    out_root = Path(args.output_root)
    image_dir = raw_root / "validation" / "camera_image"
    seg_dir = raw_root / "validation" / "camera_segmentation"
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    if not seg_dir.is_dir():
        raise FileNotFoundError(seg_dir)

    index_cache = Path(args.index_cache) if args.index_cache else out_root / "waymo_v2_image_index.json"
    image_index = build_image_index(image_dir, index_cache)

    max_images = int(args.max_images)
    written = 0
    missing = 0

    seg_files = [p for p in _iter_parquets(seg_dir) if pq.ParquetFile(p).metadata.num_rows > 0]
    for seg_path in tqdm(seg_files, desc="Converting labelled frames"):
        table = pq.read_table(
            seg_path,
            columns=[
                "key.segment_context_name",
                "key.frame_timestamp_micros",
                "key.camera_name",
                DIVISOR_COL,
                PANOPTIC_COL,
            ],
        )
        for row in table.to_pylist():
            key = _row_key(row)
            match = image_index.get(key)
            if match is None:
                missing += 1
                continue

            segment, timestamp, camera_name = key
            scene_dir = out_root / "validation" / _safe_segment_dir(segment)
            scene_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{timestamp}_cam{camera_name}"
            image_path = scene_dir / f"{stem}_image.jpg"
            semantic_path = scene_dir / f"{stem}_semantic.png"
            instance_path = scene_dir / f"{stem}_instance.png"

            if not (image_path.exists() and semantic_path.exists() and instance_path.exists()):
                image_file, image_row_idx = match
                image_bytes = _load_image_row(image_dir / image_file, image_row_idx)
                image_path.write_bytes(image_bytes)

                panoptic = np.array(Image.open(io.BytesIO(row[PANOPTIC_COL])), dtype=np.uint16)
                divisor = int(row[DIVISOR_COL])
                semantic = (panoptic // divisor).astype(np.uint8)
                instance = (panoptic % divisor).astype(np.uint16)
                Image.fromarray(semantic).save(semantic_path)
                Image.fromarray(instance).save(instance_path)

            written += 1
            if max_images > 0 and written >= max_images:
                print(f"Converted {written} labelled camera frames ({missing} missing image matches).")
                return

    print(f"Converted {written} labelled camera frames ({missing} missing image matches).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_root",
        required=True,
        help="Directory containing validation/camera_image and validation/camera_segmentation.",
    )
    parser.add_argument("--output_root", required=True, help="CUPS-style Waymo preprocessed output root.")
    parser.add_argument("--max_images", type=int, default=0, help="Optional conversion limit for smoke tests.")
    parser.add_argument("--index_cache", default=None, help="Optional JSON cache for the camera image row index.")
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
