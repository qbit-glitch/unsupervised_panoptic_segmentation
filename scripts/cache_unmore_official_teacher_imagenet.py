#!/usr/bin/env python3
"""Cache official unMORE Cascade Mask R-CNN predictions on ImageNet images.

This builds the same grouped teacher-cache format consumed by
``UnmoreTeacherCacheDataset``, but the labels come from the official unMORE
Cascade Mask R-CNN checkpoint instead of VoteCut annotations.

The intended use is a large unlabeled ImageNet teacher cache on the external
drive:

    image -> official teacher boxes, masks, scores, metadata

The script reads image metadata from an existing sharded cache index, typically
``imagenet_votecut_w050/index.jsonl``, so it reuses stable ImageNet ids and
absolute/symlinked image paths.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNMORE_ROOT = PROJECT_ROOT / "test-instance-labels" / "unMORE"
UNMORE_CAD_ROOT = UNMORE_ROOT / "cad"
sys.path.insert(0, str(UNMORE_ROOT))
sys.path.insert(0, str(UNMORE_CAD_ROOT))

from detectron2.config import get_cfg  # noqa: E402
from cad.config import add_cuvler_config  # noqa: E402
from cad.engine.defaults import DefaultPredictor  # noqa: E402

import cad.modeling  # noqa: E402,F401  # registers custom unMORE model components


def iter_source_records(index_path: Path, offset: int, max_images: int | None) -> Iterable[Dict[str, Any]]:
    yielded = 0
    with index_path.open("r") as f:
        for idx, line in enumerate(f):
            if idx < offset:
                continue
            yield json.loads(line)
            yielded += 1
            if max_images is not None and yielded >= max_images:
                return


def encode_mask(binary_mask: np.ndarray) -> Dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        rle["counts"] = counts.decode("ascii")
    return rle


def xyxy_to_xywh(box: List[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = box
    x1 = min(max(float(x1), 0.0), float(width))
    y1 = min(max(float(y1), 0.0), float(height))
    x2 = min(max(float(x2), x1), float(width))
    y2 = min(max(float(y2), y1), float(height))
    return [x1, y1, x2 - x1, y2 - y1]


def build_predictor(args: argparse.Namespace) -> DefaultPredictor:
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(max(1, min(2, args.torch_threads)))
    cfg = get_cfg()
    add_cuvler_config(cfg)
    cfg.merge_from_file(str(args.config_file))
    cfg.MODEL.WEIGHTS = str(args.weights)
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    cfg.TEST.DETECTIONS_PER_IMAGE = args.detections_per_image
    cfg.INPUT.MIN_SIZE_TEST = args.min_size_test
    cfg.INPUT.MAX_SIZE_TEST = args.max_size_test
    cfg.freeze()
    return DefaultPredictor(cfg)


def instances_to_teacher(instances, width: int, height: int, score_min: float, top_k: int | None) -> Dict[str, List[Any]]:
    if len(instances) == 0:
        return {"boxes": [], "scores": [], "category_ids": [], "bbox_areas": [], "segmentations": []}

    instances = instances.to("cpu")
    scores = instances.scores.numpy()
    order = np.argsort(-scores)
    if top_k is not None:
        order = order[:top_k]

    boxes = instances.pred_boxes.tensor.numpy()
    masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else np.zeros(len(instances), dtype=np.int64)

    out = {"boxes": [], "scores": [], "category_ids": [], "bbox_areas": [], "segmentations": []}
    for det_idx in order.tolist():
        score = float(scores[det_idx])
        if score < score_min:
            continue
        box = xyxy_to_xywh(boxes[det_idx].tolist(), width, height)
        if box[2] <= 0.0 or box[3] <= 0.0:
            continue
        if masks is None:
            continue
        binary_mask = masks[det_idx].astype(np.uint8)
        if binary_mask.sum() == 0:
            continue
        out["boxes"].append(box)
        out["scores"].append(score)
        out["category_ids"].append(int(classes[det_idx]) + 1)
        out["bbox_areas"].append(float(box[2] * box[3]))
        out["segmentations"].append(encode_mask(binary_mask))
    return out


def write_manifest(out_dir: Path, args: argparse.Namespace, stats: Dict[str, Any]) -> None:
    manifest = {
        "dataset_name": args.dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format_version": 1,
        "source_cache": str(args.source_cache),
        "source_index": str(args.source_cache / "index.jsonl"),
        "config_file": str(args.config_file),
        "weights": str(args.weights),
        "out_dir": str(out_dir),
        "offset": args.offset,
        "max_images": args.max_images,
        "score_min": args.score_min,
        "top_k": args.top_k,
        "detections_per_image": args.detections_per_image,
        "min_size_test": args.min_size_test,
        "max_size_test": args.max_size_test,
        "device": args.device,
        "cache_stats": stats,
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
        f.write("\n")


def cache_predictions(args: argparse.Namespace) -> None:
    out_dir = args.out_root / args.dataset_name
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_existing:
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    predictor = build_predictor(args)
    index_path = args.source_cache / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing source index: {index_path}")

    output_index_path = out_dir / "index.jsonl"
    total_records = 0
    total_predictions = 0
    missing_images = 0
    unreadable_images = 0
    score_sum = 0.0
    score_min_seen = math.inf
    score_max_seen = 0.0
    start_time = time.time()

    shard_f = None
    shard_idx = -1
    line_idx = 0
    try:
        with output_index_path.open("w") as index_f:
            source_iter = iter_source_records(index_path, args.offset, args.max_images)
            for source_idx, rec in enumerate(source_iter, 1):
                image_path = Path(rec["image_path"])
                if not image_path.exists():
                    missing_images += 1
                    continue
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    unreadable_images += 1
                    continue
                height, width = image.shape[:2]

                with torch.inference_mode():
                    outputs = predictor(image)
                teacher = instances_to_teacher(
                    outputs["instances"],
                    width=width,
                    height=height,
                    score_min=args.score_min,
                    top_k=args.top_k,
                )

                if total_records % args.shard_size == 0:
                    if shard_f is not None:
                        shard_f.close()
                    shard_idx += 1
                    shard_name = f"shard_{shard_idx:05d}.jsonl.gz"
                    shard_f = gzip.open(shards_dir / shard_name, "wt", encoding="utf-8")
                    line_idx = 0

                assert shard_f is not None
                shard_name = f"shard_{shard_idx:05d}.jsonl.gz"
                num_predictions = len(teacher["scores"])
                for score in teacher["scores"]:
                    score_sum += float(score)
                    score_min_seen = min(score_min_seen, float(score))
                    score_max_seen = max(score_max_seen, float(score))

                sample = {
                    "dataset": args.dataset_name,
                    "image_id": int(rec["image_id"]),
                    "file_name": rec["file_name"],
                    "image_path": str(image_path),
                    "height": int(height),
                    "width": int(width),
                    "num_predictions": num_predictions,
                    "teacher": {
                        "format": "official_unmore_cascade_mask_rcnn_predictions",
                        **teacher,
                    },
                }
                shard_f.write(json.dumps(sample, separators=(",", ":")) + "\n")
                index_f.write(
                    json.dumps(
                        {
                            "dataset": args.dataset_name,
                            "image_id": int(rec["image_id"]),
                            "file_name": rec["file_name"],
                            "image_path": str(image_path),
                            "height": int(height),
                            "width": int(width),
                            "num_predictions": num_predictions,
                            "shard": f"shards/{shard_name}",
                            "line_index": line_idx,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                total_records += 1
                total_predictions += num_predictions
                line_idx += 1

                if total_records == 1 or total_records % args.log_every == 0:
                    index_f.flush()
                    shard_f.flush()
                    elapsed = max(time.time() - start_time, 1e-6)
                    ips = total_records / elapsed
                    print(
                        f"records={total_records:,} predictions={total_predictions:,} "
                        f"source_seen={source_idx:,} ips={ips:.3f} missing={missing_images} unreadable={unreadable_images}",
                        flush=True,
                    )
    finally:
        if shard_f is not None:
            shard_f.close()

    stats = {
        "records": total_records,
        "shards": shard_idx + 1 if total_records else 0,
        "shard_size": args.shard_size,
        "total_cached_predictions": total_predictions,
        "mean_cached_score": score_sum / total_predictions if total_predictions else 0.0,
        "min_cached_score": score_min_seen if total_predictions else None,
        "max_cached_score": score_max_seen if total_predictions else None,
        "missing_images": missing_images,
        "unreadable_images": unreadable_images,
        "index": "index.jsonl",
        "shards_dir": "shards",
    }
    write_manifest(out_dir, args, stats)
    print(json.dumps(stats, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="imagenet_official_unmore_teacher_s005_10k")
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=Path("/Volumes/code_files/mbps_datasets/unmore_teacher_cache/imagenet_votecut_w050"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/Volumes/code_files/mbps_datasets/unmore_teacher_cache"),
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=UNMORE_CAD_ROOT / "model_zoo/configs/unMORE-IN+COCO/cascade_mask_rcnn_R_50_FPN.yaml",
    )
    parser.add_argument("--weights", type=Path, default=UNMORE_ROOT / "checkpoints/unMORE_model.pth")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--max-images", type=int, default=10000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--score-min", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--detections-per-image", type=int, default=100)
    parser.add_argument("--min-size-test", type=int, default=800)
    parser.add_argument("--max-size-test", type=int, default=1333)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--allow-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_predictions(args)


if __name__ == "__main__":
    main()
