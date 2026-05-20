#!/usr/bin/env python3
"""Build a compact sharded teacher-output cache from official unMORE results.

The official Detectron2 evaluator writes one large COCO results JSON with one
record per prediction. This script groups those predictions by image and writes
compressed JSONL shards:

    image -> teacher boxes, scores, compressed RLE masks, metadata

The cache intentionally keeps segmentation masks as COCO RLE strings. Decoding
them during cache construction would make the cache far larger and slower to
produce.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _iter_progress(items: Iterable[Any], total: int, desc: str) -> Iterable[Any]:
    try:
        from tqdm import tqdm

        yield from tqdm(items, total=total, desc=desc)
    except Exception:
        for idx, item in enumerate(items, 1):
            if idx == 1 or idx % 10000 == 0 or idx == total:
                print(f"{desc}: {idx}/{total}", flush=True)
            yield item


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_coco_images(annotation_json: Path, image_root: Path) -> Dict[int, Dict[str, Any]]:
    with annotation_json.open("r") as f:
        data = json.load(f)

    images: Dict[int, Dict[str, Any]] = {}
    for img in data.get("images", []):
        image_id = int(img["id"])
        file_name = img["file_name"]
        images[image_id] = {
            "image_id": image_id,
            "file_name": file_name,
            "height": int(img.get("height", 0)),
            "width": int(img.get("width", 0)),
            "image_path": str(image_root / file_name),
        }
    return images


def _load_and_group_predictions(
    predictions_json: Path,
    score_min: float,
    top_k: int | None,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, Any]]:
    with predictions_json.open("r") as f:
        predictions = json.load(f)

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    score_sum = 0.0
    kept = 0
    dropped_score = 0

    for pred in _iter_progress(predictions, len(predictions), "group predictions"):
        score = float(pred.get("score", 0.0))
        if score < score_min:
            dropped_score += 1
            continue

        image_id = int(pred["image_id"])
        bbox = [float(x) for x in pred["bbox"]]
        seg = pred.get("segmentation")
        if isinstance(seg, dict) and isinstance(seg.get("counts"), bytes):
            seg = {"size": seg["size"], "counts": seg["counts"].decode("ascii")}

        grouped[image_id].append(
            {
                "bbox_xywh": bbox,
                "bbox_area": float(bbox[2] * bbox[3]),
                "score": score,
                "category_id": int(pred.get("category_id", 1)),
                "segmentation": seg,
            }
        )
        score_sum += score
        kept += 1

    dropped_topk = 0
    for image_id, preds in grouped.items():
        preds.sort(key=lambda x: x["score"], reverse=True)
        if top_k is not None and len(preds) > top_k:
            dropped_topk += len(preds) - top_k
            grouped[image_id] = preds[:top_k]

    stats = {
        "raw_predictions": len(predictions),
        "kept_after_score_filter": kept,
        "dropped_by_score": dropped_score,
        "dropped_by_top_k": dropped_topk,
        "mean_score_before_top_k": score_sum / kept if kept else 0.0,
        "images_with_predictions": len(grouped),
    }
    return dict(grouped), stats


def _write_shards(
    dataset_name: str,
    images: Dict[int, Dict[str, Any]],
    grouped: Dict[int, List[Dict[str, Any]]],
    out_dir: Path,
    shard_size: int,
    verify_images: bool,
) -> Dict[str, Any]:
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "index.jsonl"
    missing_images = 0
    total_predictions = 0
    score_sum = 0.0
    score_min_seen = math.inf
    score_max_seen = 0.0

    image_ids = sorted(images)
    n_shards = math.ceil(len(image_ids) / shard_size)

    with index_path.open("w") as index_f:
        for shard_idx in _iter_progress(range(n_shards), n_shards, "write shards"):
            start = shard_idx * shard_size
            shard_image_ids = image_ids[start : start + shard_size]
            shard_name = f"shard_{shard_idx:05d}.jsonl.gz"
            shard_path = shards_dir / shard_name

            with gzip.open(shard_path, "wt", encoding="utf-8") as shard_f:
                for line_idx, image_id in enumerate(shard_image_ids):
                    meta = images[image_id]
                    preds = grouped.get(image_id, [])
                    if verify_images and not Path(meta["image_path"]).exists():
                        missing_images += 1

                    total_predictions += len(preds)
                    for pred in preds:
                        score = float(pred["score"])
                        score_sum += score
                        score_min_seen = min(score_min_seen, score)
                        score_max_seen = max(score_max_seen, score)

                    sample = {
                        "dataset": dataset_name,
                        "image_id": image_id,
                        "file_name": meta["file_name"],
                        "image_path": meta["image_path"],
                        "height": meta["height"],
                        "width": meta["width"],
                        "num_predictions": len(preds),
                        "teacher": {
                            "format": "detectron2_coco_instances_results",
                            "boxes": [p["bbox_xywh"] for p in preds],
                            "scores": [p["score"] for p in preds],
                            "category_ids": [p["category_id"] for p in preds],
                            "bbox_areas": [p["bbox_area"] for p in preds],
                            "segmentations": [p["segmentation"] for p in preds],
                        },
                    }
                    shard_f.write(json.dumps(sample, separators=(",", ":")) + "\n")

                    index_f.write(
                        json.dumps(
                            {
                                "dataset": dataset_name,
                                "image_id": image_id,
                                "file_name": meta["file_name"],
                                "image_path": meta["image_path"],
                                "height": meta["height"],
                                "width": meta["width"],
                                "num_predictions": len(preds),
                                "shard": f"shards/{shard_name}",
                                "line_index": line_idx,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )

    return {
        "images": len(images),
        "shards": n_shards,
        "shard_size": shard_size,
        "total_cached_predictions": total_predictions,
        "mean_cached_score": score_sum / total_predictions if total_predictions else 0.0,
        "min_cached_score": score_min_seen if total_predictions else None,
        "max_cached_score": score_max_seen if total_predictions else None,
        "missing_images": missing_images,
        "index": "index.jsonl",
        "shards_dir": "shards",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--annotation-json", type=Path, required=True)
    parser.add_argument("--predictions-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--score-min", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--no-top-k", action="store_true")
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--hash-inputs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_k = None if args.no_top_k else args.top_k

    out_dir = args.out_root / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading annotations: {args.annotation_json}", flush=True)
    images = _load_coco_images(args.annotation_json, args.image_root)

    print(f"Loading predictions: {args.predictions_json}", flush=True)
    grouped, pred_stats = _load_and_group_predictions(
        predictions_json=args.predictions_json,
        score_min=args.score_min,
        top_k=top_k,
    )

    print(f"Writing cache: {out_dir}", flush=True)
    cache_stats = _write_shards(
        dataset_name=args.dataset_name,
        images=images,
        grouped=grouped,
        out_dir=out_dir,
        shard_size=args.shard_size,
        verify_images=args.verify_images,
    )

    input_hashes = {}
    if args.hash_inputs:
        input_hashes = {
            "annotation_json_sha256": _sha256(args.annotation_json),
            "predictions_json_sha256": _sha256(args.predictions_json),
        }

    manifest = {
        "dataset_name": args.dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "annotation_json": str(args.annotation_json),
        "predictions_json": str(args.predictions_json),
        "image_root": str(args.image_root),
        "out_dir": str(out_dir),
        "score_min": args.score_min,
        "top_k": top_k,
        "format_version": 1,
        "prediction_stats": pred_stats,
        "cache_stats": cache_stats,
        "input_hashes": input_hashes,
    }

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
