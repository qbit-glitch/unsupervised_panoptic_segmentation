#!/usr/bin/env python3
"""Build a sharded ImageNet/VoteCut training cache for unMORE student models.

The official unMORE pipeline trains its objectness stage from ImageNet-1K plus
VoteCut masks. The VoteCut annotation file is large enough that json.load() is
unfriendly on local machines, so this script streams the COCO-style JSON into a
small SQLite staging database and then writes the same sharded cache format used
by ``mbps_pytorch.unmore_distill.UnmoreTeacherCacheDataset``.

Each cache row is:

    image -> VoteCut boxes, scores/weights, compressed RLE masks, metadata

By default this keeps all VoteCut annotations with ``weight >= 0.5``. That
matches the official unMORE ImageNet merge script used for class-agnostic
detector training, while still allowing smoke tests through ``--image-limit``
and ``--annotation-limit``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sqlite3
import time
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Iterator, List


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


def iter_json_array(path: Path, key: str) -> Iterator[Dict[str, Any]]:
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


def _progress(message: str, start_time: float, count: int) -> None:
    elapsed = max(time.time() - start_time, 1e-6)
    rate = count / elapsed
    print(f"{message}: {count:,} ({rate:,.1f}/s)", flush=True)


def connect_db(path: Path, rebuild: bool) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rebuild and path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS images ("
        "image_id INTEGER PRIMARY KEY, "
        "file_name TEXT NOT NULL, "
        "height INTEGER NOT NULL, "
        "width INTEGER NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS annotations ("
        "row_id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "image_id INTEGER NOT NULL, "
        "score REAL NOT NULL, "
        "bbox TEXT NOT NULL, "
        "bbox_area REAL NOT NULL, "
        "category_id INTEGER NOT NULL, "
        "segmentation TEXT NOT NULL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_file_name ON images(file_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_annotations_image_score ON annotations(image_id, score DESC)")
    return conn


def db_counts(conn: sqlite3.Connection) -> Dict[str, int]:
    return {
        "images": int(conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]),
        "annotations": int(conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]),
    }


def build_staging_db(
    annotation_json: Path,
    db_path: Path,
    score_min: float,
    rebuild: bool,
    image_limit: int | None,
    annotation_limit: int | None,
    progress_every: int,
) -> Dict[str, Any]:
    conn = connect_db(db_path, rebuild=rebuild)
    existing = db_counts(conn)
    if existing["images"] and existing["annotations"] and not rebuild:
        print(
            f"Reusing staging DB: images={existing['images']:,} annotations={existing['annotations']:,}",
            flush=True,
        )
        conn.close()
        return {"reused": True, **existing}

    conn.execute("DELETE FROM images")
    conn.execute("DELETE FROM annotations")
    conn.commit()

    print(f"Streaming images from {annotation_json}", flush=True)
    start_time = time.time()
    inserted_images = 0
    cur = conn.cursor()
    for idx, image in enumerate(iter_json_array(annotation_json, "images"), 1):
        cur.execute(
            "INSERT OR REPLACE INTO images(image_id, file_name, height, width) VALUES (?, ?, ?, ?)",
            (
                int(image["id"]),
                str(image["file_name"]),
                int(image.get("height", 0)),
                int(image.get("width", 0)),
            ),
        )
        inserted_images = idx
        if idx % progress_every == 0:
            conn.commit()
            _progress("  images", start_time, idx)
        if image_limit is not None and idx >= image_limit:
            break
    conn.commit()
    _progress("Finished images", start_time, inserted_images)

    allowed_ids: set[int] | None = None
    if image_limit is not None:
        allowed_ids = {int(row[0]) for row in conn.execute("SELECT image_id FROM images")}

    print(f"Streaming annotations with weight >= {score_min}", flush=True)
    start_time = time.time()
    scanned_annotations = 0
    kept_annotations = 0
    dropped_score = 0
    dropped_image = 0
    for idx, ann in enumerate(iter_json_array(annotation_json, "annotations"), 1):
        scanned_annotations = idx
        image_id = int(ann["image_id"])
        if allowed_ids is not None and image_id not in allowed_ids:
            dropped_image += 1
            if annotation_limit is not None and idx >= annotation_limit:
                break
            continue

        score = float(ann.get("weight", ann.get("score", 0.0)))
        if score < score_min:
            dropped_score += 1
            if annotation_limit is not None and idx >= annotation_limit:
                break
            continue

        bbox = [float(x) for x in ann["bbox"]]
        bbox_area = float(ann.get("area", max(bbox[2], 0.0) * max(bbox[3], 0.0)))
        category_id = int(ann.get("category_id", 1))
        cur.execute(
            """
            INSERT INTO annotations(image_id, score, bbox, bbox_area, category_id, segmentation)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                score,
                json.dumps(bbox, separators=(",", ":")),
                bbox_area,
                category_id,
                json.dumps(ann["segmentation"], separators=(",", ":")),
            ),
        )
        kept_annotations += 1

        if idx % progress_every == 0:
            conn.commit()
            _progress("  annotations scanned", start_time, idx)
            print(f"    kept={kept_annotations:,}", flush=True)
        if annotation_limit is not None and idx >= annotation_limit:
            break
    conn.commit()
    _progress("Finished annotations scanned", start_time, scanned_annotations)

    counts = db_counts(conn)
    conn.close()
    return {
        "reused": False,
        "images": counts["images"],
        "annotations": counts["annotations"],
        "scanned_annotations": scanned_annotations,
        "kept_annotations": kept_annotations,
        "dropped_by_score": dropped_score,
        "dropped_by_image_filter": dropped_image,
    }


def _load_annotations(
    conn: sqlite3.Connection,
    image_id: int,
    top_k: int | None,
) -> List[Dict[str, Any]]:
    query = """
        SELECT score, bbox, bbox_area, category_id, segmentation
        FROM annotations
        WHERE image_id = ?
        ORDER BY score DESC
    """
    if top_k is not None:
        query += " LIMIT ?"
        rows = conn.execute(query, (image_id, top_k))
    else:
        rows = conn.execute(query, (image_id,))

    anns = []
    for score, bbox, bbox_area, category_id, segmentation in rows:
        anns.append(
            {
                "score": float(score),
                "bbox_xywh": json.loads(bbox),
                "bbox_area": float(bbox_area),
                "category_id": int(category_id),
                "segmentation": json.loads(segmentation),
            }
        )
    return anns


def write_cache(
    db_path: Path,
    image_root: Path,
    out_dir: Path,
    dataset_name: str,
    shard_size: int,
    top_k: int | None,
    skip_empty: bool,
    require_images: bool,
    max_records: int | None,
) -> Dict[str, Any]:
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"

    conn = sqlite3.connect(db_path)
    query = "SELECT image_id, file_name, height, width FROM images ORDER BY file_name"

    total_images = int(conn.execute("SELECT COUNT(*) FROM images").fetchone()[0])
    total_records = 0
    total_predictions = 0
    missing_images = 0
    empty_images = 0
    score_sum = 0.0
    score_min_seen = math.inf
    score_max_seen = 0.0

    print(f"Writing cache shards to {out_dir}", flush=True)
    start_time = time.time()
    shard_f = None
    shard_idx = -1
    line_idx = 0
    try:
        with index_path.open("w") as index_f:
            for scanned, (image_id, file_name, height, width) in enumerate(conn.execute(query), 1):
                annotations = _load_annotations(conn, int(image_id), top_k)
                if skip_empty and not annotations:
                    empty_images += 1
                    continue

                image_path = image_root / str(file_name)
                if require_images and not image_path.exists():
                    missing_images += 1
                    continue

                if total_records % shard_size == 0:
                    if shard_f is not None:
                        shard_f.close()
                    shard_idx += 1
                    shard_name = f"shard_{shard_idx:05d}.jsonl.gz"
                    shard_f = gzip.open(shards_dir / shard_name, "wt", encoding="utf-8")
                    line_idx = 0

                assert shard_f is not None
                shard_name = f"shard_{shard_idx:05d}.jsonl.gz"
                for ann in annotations:
                    score = float(ann["score"])
                    score_sum += score
                    score_min_seen = min(score_min_seen, score)
                    score_max_seen = max(score_max_seen, score)

                sample = {
                    "dataset": dataset_name,
                    "image_id": int(image_id),
                    "file_name": str(file_name),
                    "image_path": str(image_path),
                    "height": int(height),
                    "width": int(width),
                    "num_predictions": len(annotations),
                    "teacher": {
                        "format": "imagenet_votecut_coco_annotations",
                        "boxes": [ann["bbox_xywh"] for ann in annotations],
                        "scores": [ann["score"] for ann in annotations],
                        "category_ids": [ann["category_id"] for ann in annotations],
                        "bbox_areas": [ann["bbox_area"] for ann in annotations],
                        "segmentations": [ann["segmentation"] for ann in annotations],
                    },
                }
                shard_f.write(json.dumps(sample, separators=(",", ":")) + "\n")
                index_f.write(
                    json.dumps(
                        {
                            "dataset": dataset_name,
                            "image_id": int(image_id),
                            "file_name": str(file_name),
                            "image_path": str(image_path),
                            "height": int(height),
                            "width": int(width),
                            "num_predictions": len(annotations),
                            "shard": f"shards/{shard_name}",
                            "line_index": line_idx,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                line_idx += 1
                total_records += 1
                total_predictions += len(annotations)

                if total_records == 1 or total_records % 10000 == 0:
                    _progress("  records written", start_time, total_records)
                    print(f"    scanned_images={scanned:,}/{total_images:,}", flush=True)

                if max_records is not None and total_records >= max_records:
                    break
    finally:
        if shard_f is not None:
            shard_f.close()
        conn.close()

    n_shards = shard_idx + 1 if total_records else 0
    return {
        "images_in_staging_db": total_images,
        "records": total_records,
        "shards": n_shards,
        "shard_size": shard_size,
        "total_cached_predictions": total_predictions,
        "mean_cached_score": score_sum / total_predictions if total_predictions else 0.0,
        "min_cached_score": score_min_seen if total_predictions else None,
        "max_cached_score": score_max_seen if total_predictions else None,
        "missing_images": missing_images,
        "empty_images_skipped": empty_images,
        "index": "index.jsonl",
        "shards_dir": "shards",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="imagenet_votecut_w050")
    parser.add_argument(
        "--annotation-json",
        type=Path,
        default=Path("datasets/unmore_imagenet/annotations/imagenet_train_votecut_kmax_3_tuam_0.2.json"),
    )
    parser.add_argument("--image-root", type=Path, default=Path("datasets/unmore_imagenet/train"))
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/Volumes/code_files/mbps_datasets/unmore_teacher_cache"),
    )
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--score-min", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--skip-empty", action="store_true", default=True)
    parser.add_argument("--include-empty", action="store_false", dest="skip_empty")
    parser.add_argument("--require-images", action="store_true")
    parser.add_argument("--image-limit", type=int)
    parser.add_argument("--annotation-limit", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--rebuild-db", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_root / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.db_path or (out_dir / "staging.sqlite")

    staging_stats = build_staging_db(
        annotation_json=args.annotation_json,
        db_path=db_path,
        score_min=args.score_min,
        rebuild=args.rebuild_db,
        image_limit=args.image_limit,
        annotation_limit=args.annotation_limit,
        progress_every=args.progress_every,
    )
    cache_stats = write_cache(
        db_path=db_path,
        image_root=args.image_root,
        out_dir=out_dir,
        dataset_name=args.dataset_name,
        shard_size=args.shard_size,
        top_k=args.top_k,
        skip_empty=args.skip_empty,
        require_images=args.require_images,
        max_records=args.max_records,
    )

    manifest = {
        "dataset_name": args.dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format_version": 1,
        "annotation_json": str(args.annotation_json),
        "image_root": str(args.image_root),
        "out_dir": str(out_dir),
        "db_path": str(db_path),
        "score_min": args.score_min,
        "top_k": args.top_k,
        "skip_empty": args.skip_empty,
        "image_limit": args.image_limit,
        "annotation_limit": args.annotation_limit,
        "max_records": args.max_records,
        "staging_stats": staging_stats,
        "cache_stats": cache_stats,
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
