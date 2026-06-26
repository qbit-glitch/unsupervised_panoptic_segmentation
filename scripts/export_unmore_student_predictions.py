#!/usr/bin/env python3
"""Export compact unMORE-student predictions and optionally run COCO AP eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image, ImageFile
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mbps_pytorch.unmore_distill import build_unmore_dinov3s_maskrcnn  # noqa: E402


ImageFile.LOAD_TRUNCATED_IMAGES = True


def resolve_data_path(path: str | Path, dataset_roots: List[Path]) -> Path:
    """Resolve cached dataset paths after moving datasets out of the repo."""

    raw = Path(path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        for root in dataset_roots:
            candidates.append(root / raw)
            parts = raw.parts
            if parts and parts[0] == "datasets":
                candidates.append(root / Path(*parts[1:]))
                if len(parts) > 2:
                    candidates.append(root / Path(*parts[2:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not resolve data path {path!s}. Tried:\n  {searched}")


def parse_dataset_roots(values: List[str] | None) -> List[Path]:
    roots: List[Path] = []
    env_roots = []
    import os

    if os.environ.get("UNMORE_DATASETS_ROOTS"):
        env_roots.extend(os.environ["UNMORE_DATASETS_ROOTS"].split(":"))
    if os.environ.get("UNMORE_DATASETS_ROOT"):
        env_roots.append(os.environ["UNMORE_DATASETS_ROOT"])

    for item in [*(values or []), *env_roots]:
        for part in str(item).split(":"):
            part = part.strip()
            if part:
                roots.append(Path(part).expanduser())
    return roots


class CacheImageDataset(Dataset):
    """Image-only view of an unMORE teacher cache index."""

    def __init__(
        self,
        cache_dir: Path,
        max_images: int | None = None,
        dataset_roots: List[Path] | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.dataset_roots = dataset_roots or []
        manifest_path = cache_dir / "manifest.json"
        index_path = cache_dir / "index.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Missing cache index: {index_path}")

        with manifest_path.open("r") as f:
            self.manifest = json.load(f)

        self.records: List[Dict[str, Any]] = []
        with index_path.open("r") as f:
            for line in f:
                self.records.append(json.loads(line))
                if max_images is not None and len(self.records) >= max_images:
                    break

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image_path = resolve_data_path(rec["image_path"], self.dataset_roots)
        image = Image.open(image_path).convert("RGB")
        return F.to_tensor(image), {
            "image_id": int(rec["image_id"]),
            "file_name": rec["file_name"],
            "image_path": str(image_path),
            "height": int(rec["height"]),
            "width": int(rec["width"]),
        }


def collate_images(batch):
    images, metas = zip(*batch)
    return list(images), list(metas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/Volumes/code_files/mbps_datasets/unmore_teacher_cache/coco20k_official_unmore"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    parser.add_argument("--detections-per-image", type=int, default=100)
    parser.add_argument("--annotation-json", type=Path, default=None)
    parser.add_argument(
        "--datasets-root",
        action="append",
        default=None,
        help=(
            "Dataset root used to resolve cached relative paths. Can be passed "
            "multiple times or as a colon-separated list. A leading 'datasets/' "
            "component is tried both with and without that prefix."
        ),
    )
    parser.add_argument("--iou-types", nargs="+", default=["bbox", "segm"], choices=["bbox", "segm"])
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--fpn-dim", type=int, default=None)
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "mps":
        try:
            torch.empty(1, device="mps")
        except Exception as exc:
            raise RuntimeError("MPS was requested but is not usable in this environment") from exc
    return torch.device(name)


def load_student(args: argparse.Namespace, device: torch.device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    fpn_dim = args.fpn_dim if args.fpn_dim is not None else int(ckpt_args.get("fpn_dim", 192))
    min_size = args.min_size if args.min_size is not None else int(ckpt_args.get("min_size", 512))
    max_size = args.max_size if args.max_size is not None else int(ckpt_args.get("max_size", 896))

    model = build_unmore_dinov3s_maskrcnn(
        pretrained_backbone=False,
        train_backbone=True,
        fpn_dim=fpn_dim,
        min_size=min_size,
        max_size=max_size,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    model.roi_heads.detections_per_img = args.detections_per_image
    return model, ckpt_args


def encode_mask(binary_mask: np.ndarray) -> Dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        rle["counts"] = counts.decode("ascii")
    return rle


def xyxy_to_xywh(box: torch.Tensor, width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, x1), float(width))
    y2 = min(max(y2, y1), float(height))
    return [x1, y1, x2 - x1, y2 - y1]


def export_predictions(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dataset_roots = parse_dataset_roots(args.datasets_root)
    dataset = CacheImageDataset(args.cache_dir, max_images=args.max_images, dataset_roots=dataset_roots)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_images,
    )
    model, ckpt_args = load_student(args, device)

    results: List[Dict[str, Any]] = []
    image_ids: List[int] = []
    with torch.no_grad():
        for images, metas in tqdm(loader, desc="student export"):
            outputs = model([image.to(device) for image in images])
            for output, meta in zip(outputs, metas):
                image_id = int(meta["image_id"])
                image_ids.append(image_id)
                scores = output["scores"].detach().cpu()
                keep = torch.nonzero(scores >= args.score_thresh, as_tuple=False).flatten()
                if args.detections_per_image > 0:
                    keep = keep[: args.detections_per_image]

                boxes = output["boxes"].detach().cpu()
                masks = output["masks"].detach().cpu()
                for det_idx in keep.tolist():
                    mask_prob = masks[det_idx, 0].numpy()
                    binary = mask_prob >= args.mask_thresh
                    if not binary.any():
                        continue
                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": xyxy_to_xywh(boxes[det_idx], meta["width"], meta["height"]),
                            "score": float(scores[det_idx]),
                            "segmentation": encode_mask(binary),
                        }
                    )

    predictions_path = args.output_dir / "coco_instances_results.json"
    with predictions_path.open("w") as f:
        json.dump(results, f)

    metadata = {
        "checkpoint": str(args.checkpoint),
        "cache_dir": str(args.cache_dir),
        "num_images": len(dataset),
        "num_predictions": len(results),
        "score_thresh": args.score_thresh,
        "mask_thresh": args.mask_thresh,
        "detections_per_image": args.detections_per_image,
        "checkpoint_args": ckpt_args,
        "teacher_cache_manifest": dataset.manifest,
        "dataset_roots": [str(root) for root in dataset_roots],
        "predictions_json": str(predictions_path),
    }
    with (args.output_dir / "export_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, default=str)
        f.write("\n")
    return {"metadata": metadata, "results": results, "image_ids": image_ids}


def evaluate_coco(
    annotation_json: Path,
    results: List[Dict[str, Any]],
    image_ids: List[int],
    iou_types: List[str],
) -> Dict[str, Dict[str, Any]]:
    metrics: Dict[str, Dict[str, Any]] = {}
    if not results:
        return metrics

    coco_gt = COCO(str(annotation_json))
    unique_image_ids = sorted(set(image_ids))
    names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    for iou_type in iou_types:
        try:
            eval_results = [dict(row) for row in results]
            if iou_type == "segm":
                for row in eval_results:
                    row.pop("bbox", None)
            coco_dt = coco_gt.loadRes(eval_results)
            coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
            coco_eval.params.imgIds = unique_image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            metrics[iou_type] = {name: float(value) for name, value in zip(names, coco_eval.stats.tolist())}
        except Exception as exc:
            metrics[iou_type] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"COCOeval failed for iou_type={iou_type}: {metrics[iou_type]['error']}", flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    exported = export_predictions(args)
    annotation_json = args.annotation_json
    if annotation_json is None:
        annotation_json = Path(exported["metadata"]["teacher_cache_manifest"]["annotation_json"])
    annotation_json = resolve_data_path(annotation_json, parse_dataset_roots(args.datasets_root))

    if not args.skip_eval:
        metrics = evaluate_coco(annotation_json, exported["results"], exported["image_ids"], args.iou_types)
        with (args.output_dir / "coco_eval_metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)
            f.write("\n")
        print(json.dumps(metrics, indent=2), flush=True)

    print(f"Done. Outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
