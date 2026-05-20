#!/usr/bin/env python3
"""Stream VoteCut ImageNet annotations into unMORE top-1 mask files.

The official unMORE preprocessing script json.load()s a 6.5 GB COCO-style
annotation file. This version streams the images/annotations arrays and stores
only the best-weight annotation per image in SQLite before writing masks.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pycocotools.mask as mask_util


CHUNK_SIZE = 4 * 1024 * 1024


def find_array_start(path: Path, key: str) -> int:
    marker = f'"{key}": ['.encode("utf-8")
    carry = b""
    offset = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                raise ValueError(f"Could not find JSON array key {key!r} in {path}")
            haystack = carry + chunk
            index = haystack.find(marker)
            if index >= 0:
                return offset - len(carry) + index + len(marker)
            offset += len(chunk)
            carry = haystack[-(len(marker) + 32) :]


def iter_json_array(path: Path, key: str) -> Iterator[dict]:
    start = find_array_start(path, key)
    decoder = json.JSONDecoder()
    buffer = ""
    with path.open("rb") as handle:
        handle.seek(start)
        while True:
            stripped = buffer.lstrip()
            leading = len(buffer) - len(stripped)
            if leading:
                buffer = stripped

            if not buffer:
                chunk = handle.read(CHUNK_SIZE)
                if not chunk:
                    raise ValueError(f"Unexpected EOF while reading {key!r}")
                buffer += chunk.decode("utf-8")
                continue

            if buffer[0] == "]":
                return
            if buffer[0] == ",":
                buffer = buffer[1:]
                continue

            try:
                item, consumed = decoder.raw_decode(buffer)
            except JSONDecodeError:
                chunk = handle.read(CHUNK_SIZE)
                if not chunk:
                    raise
                buffer += chunk.decode("utf-8")
                continue

            yield item
            buffer = buffer[consumed:]


def connect_db(path: Path, rebuild: bool) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rebuild and path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS images ("
        "image_id INTEGER PRIMARY KEY, file_name TEXT NOT NULL, height INTEGER, width INTEGER)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS best_annotations ("
        "image_id INTEGER PRIMARY KEY, weight REAL NOT NULL, segmentation TEXT NOT NULL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_file_name ON images(file_name)")
    return conn


def build_selection_db(
    annotation_path: Path,
    db_path: Path,
    rebuild: bool,
    image_limit: int | None,
    annotation_limit: int | None,
) -> None:
    conn = connect_db(db_path, rebuild=rebuild)
    image_count = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    best_count = conn.execute("SELECT COUNT(*) FROM best_annotations").fetchone()[0]
    if image_count and best_count and not rebuild:
        print(f"Selection DB already has images={image_count} best_annotations={best_count}", flush=True)
        conn.close()
        return

    print("Streaming images into SQLite", flush=True)
    cur = conn.cursor()
    image_scanned = 0
    for idx, image in enumerate(iter_json_array(annotation_path, "images"), 1):
        image_scanned = idx
        cur.execute(
            "INSERT OR REPLACE INTO images(image_id, file_name, height, width) VALUES (?, ?, ?, ?)",
            (int(image["id"]), image["file_name"], image.get("height"), image.get("width")),
        )
        if idx % 10000 == 0:
            conn.commit()
            print(f"  images {idx}", flush=True)
        if image_limit is not None and idx >= image_limit:
            break
    conn.commit()
    print(f"Finished images: {image_scanned}", flush=True)

    allowed_ids: set[int] | None = None
    if image_limit is not None:
        allowed_ids = {row[0] for row in conn.execute("SELECT image_id FROM images")}

    print("Streaming annotations and selecting highest-weight mask per image", flush=True)
    annotation_scanned = 0
    for idx, ann in enumerate(iter_json_array(annotation_path, "annotations"), 1):
        annotation_scanned = idx
        image_id = int(ann["image_id"])
        if allowed_ids is not None and image_id not in allowed_ids:
            if annotation_limit is not None and idx >= annotation_limit:
                break
            continue
        weight = float(ann.get("weight", ann.get("score", 0.0)))
        segmentation = json.dumps(ann["segmentation"], separators=(",", ":"))
        cur.execute(
            """
            INSERT INTO best_annotations(image_id, weight, segmentation)
            VALUES (?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                weight = excluded.weight,
                segmentation = excluded.segmentation
            WHERE excluded.weight > best_annotations.weight
            """,
            (image_id, weight, segmentation),
        )
        if idx % 10000 == 0:
            conn.commit()
            count = conn.execute("SELECT COUNT(*) FROM best_annotations").fetchone()[0]
            print(f"  annotations {idx} selected={count}", flush=True)
        if annotation_limit is not None and idx >= annotation_limit:
            break
    conn.commit()
    best_count = conn.execute("SELECT COUNT(*) FROM best_annotations").fetchone()[0]
    print(f"Finished annotations scanned={annotation_scanned} selected={best_count}", flush=True)
    conn.close()


def decode_largest_component(segmentation_json: str) -> np.ndarray:
    segmentation = json.loads(segmentation_json)
    if isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), str):
        segmentation["counts"] = segmentation["counts"].encode("utf-8")
    mask = mask_util.decode(segmentation)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = np.asarray(mask > 0, dtype=np.uint8)
    if mask.sum() == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    if num_labels <= 1:
        return mask
    largest_cc_index = int(np.argmax(stats[1:, -1]) + 1)
    return np.asarray(labels == largest_cc_index, dtype=np.uint8)


def write_masks(
    db_path: Path,
    image_root: Path,
    output_root: Path,
    max_output: int | None,
    require_image: bool,
    skip_existing: bool,
    status_path: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    query = """
        SELECT images.file_name, best_annotations.weight, best_annotations.segmentation
        FROM best_annotations
        JOIN images ON images.image_id = best_annotations.image_id
        ORDER BY images.file_name
    """
    written = skipped_existing = missing_images = decode_failed = considered = 0
    for file_name, weight, segmentation_json in conn.execute(query):
        if max_output is not None and written >= max_output:
            break
        considered += 1
        image_path = image_root / file_name
        output_path = (output_root / file_name).with_suffix(".png")
        if require_image and not image_path.exists():
            missing_images += 1
            continue
        if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
            skipped_existing += 1
            continue
        try:
            mask = decode_largest_component(segmentation_json)
        except Exception as exc:  # noqa: BLE001
            decode_failed += 1
            if decode_failed <= 20:
                print(f"DECODE_FAILED {file_name}: {exc}", flush=True)
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), mask * 255, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        written += 1
        if considered % 1000 == 0:
            save_status(status_path, considered, written, skipped_existing, missing_images, decode_failed)
            print(
                f"  masks considered={considered} written={written} "
                f"existing={skipped_existing} missing_images={missing_images} decode_failed={decode_failed}",
                flush=True,
            )

    save_status(status_path, considered, written, skipped_existing, missing_images, decode_failed)
    print(
        f"Finished masks considered={considered} written={written} existing={skipped_existing} "
        f"missing_images={missing_images} decode_failed={decode_failed}",
        flush=True,
    )
    conn.close()


def save_status(
    status_path: Path,
    considered: int,
    written: int,
    skipped_existing: int,
    missing_images: int,
    decode_failed: int,
) -> None:
    if not status_path.parent.exists():
        status_path.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "considered": considered,
        "written": written,
        "skipped_existing": skipped_existing,
        "missing_images": missing_images,
        "decode_failed": decode_failed,
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation",
        default="datasets/unmore_imagenet/annotations/imagenet_train_votecut_kmax_3_tuam_0.2.json",
    )
    parser.add_argument("--image-root", default="datasets/unmore_imagenet/train")
    parser.add_argument("--output-root", default="datasets/unmore_imagenet/masks_top1_single_component")
    parser.add_argument("--sqlite", default="datasets/unmore_imagenet/votecut_top1_selection.sqlite")
    parser.add_argument("--status-path", default="datasets/unmore_imagenet/votecut_top1_mask_conversion_status.json")
    parser.add_argument("--phase", choices=["all", "select", "write"], default="all")
    parser.add_argument("--rebuild-db", action="store_true")
    parser.add_argument("--image-limit", type=int, default=None)
    parser.add_argument("--annotation-limit", type=int, default=None)
    parser.add_argument("--max-output", type=int, default=None)
    parser.add_argument("--no-require-image", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    annotation_path = Path(args.annotation)
    image_root = Path(args.image_root)
    output_root = Path(args.output_root)
    db_path = Path(args.sqlite)
    status_path = Path(args.status_path)

    if args.phase in {"all", "select"}:
        build_selection_db(
            annotation_path=annotation_path,
            db_path=db_path,
            rebuild=args.rebuild_db,
            image_limit=args.image_limit,
            annotation_limit=args.annotation_limit,
        )
    if args.phase in {"all", "write"}:
        write_masks(
            db_path=db_path,
            image_root=image_root,
            output_root=output_root,
            max_output=args.max_output,
            require_image=not args.no_require_image,
            skip_existing=not args.overwrite,
            status_path=status_path,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
