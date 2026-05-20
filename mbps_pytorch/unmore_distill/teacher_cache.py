"""Dataset loader for sharded unMORE teacher-output caches."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


def _rle_to_mask(rle: Dict[str, Any]) -> torch.Tensor:
    rle = dict(rle)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("ascii")
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return torch.from_numpy(decoded).to(torch.uint8)


def _xywh_to_xyxy(box: List[float]) -> List[float]:
    x, y, w, h = box
    return [x, y, x + max(w, 0.0), y + max(h, 0.0)]


class UnmoreTeacherCacheDataset(Dataset):
    """Load grouped teacher predictions from external-drive JSONL shards.

    The cache format is produced by ``scripts/build_unmore_teacher_cache.py``.
    Shards are loaded lazily and retained in memory per Dataset worker, which
    avoids repeatedly scanning compressed JSONL files during training.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        max_images: Optional[int] = None,
        score_min: float = 0.1,
        max_instances: int = 30,
        min_box_size: float = 2.0,
        skip_empty: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.score_min = score_min
        self.max_instances = max_instances
        self.min_box_size = min_box_size

        index_path = self.cache_dir / "index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing cache index: {index_path}")

        with (self.cache_dir / "manifest.json").open("r") as f:
            self.manifest = json.load(f)

        self.records: List[Dict[str, Any]] = []
        with index_path.open("r") as f:
            for line in f:
                rec = json.loads(line)
                if skip_empty and rec.get("num_predictions", 0) == 0:
                    continue
                self.records.append(rec)
                if max_images is not None and len(self.records) >= max_images:
                    break

        self._shard_cache: Dict[str, List[Dict[str, Any]]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _load_shard(self, rel_path: str) -> List[Dict[str, Any]]:
        if rel_path in self._shard_cache:
            return self._shard_cache[rel_path]
        shard_path = self.cache_dir / rel_path
        with gzip.open(shard_path, "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        self._shard_cache[rel_path] = rows
        return rows

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        rows = self._load_shard(rec["shard"])
        return rows[int(rec["line_index"])]

    def __getitem__(self, idx: int):
        sample = self._load_sample(idx)
        image_path = Path(sample["image_path"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image)

        teacher = sample["teacher"]
        selected = []
        for rank, score in enumerate(teacher["scores"]):
            if float(score) < self.score_min:
                continue
            box = teacher["boxes"][rank]
            if box[2] < self.min_box_size or box[3] < self.min_box_size:
                continue
            seg = teacher["segmentations"][rank]
            if not isinstance(seg, dict):
                continue
            selected.append(rank)
            if len(selected) >= self.max_instances:
                break

        boxes: List[List[float]] = []
        masks: List[torch.Tensor] = []
        scores: List[float] = []
        areas: List[float] = []
        for rank in selected:
            box_xyxy = _xywh_to_xyxy([float(x) for x in teacher["boxes"][rank]])
            if box_xyxy[2] <= box_xyxy[0] or box_xyxy[3] <= box_xyxy[1]:
                continue
            mask = _rle_to_mask(teacher["segmentations"][rank])
            boxes.append(box_xyxy)
            masks.append(mask)
            scores.append(float(teacher["scores"][rank]))
            areas.append(float(teacher["bbox_areas"][rank]))

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            masks_tensor = torch.stack(masks, dim=0)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            area = torch.tensor(areas, dtype=torch.float32)
            teacher_scores = torch.tensor(scores, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            masks_tensor = torch.zeros((0, sample["height"], sample["width"]), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            teacher_scores = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels,
            "masks": masks_tensor,
            "image_id": torch.tensor([int(sample["image_id"])]),
            "area": area,
            "iscrowd": torch.zeros((len(boxes_tensor),), dtype=torch.int64),
            "teacher_scores": teacher_scores,
        }
        meta = {
            "dataset": sample["dataset"],
            "file_name": sample["file_name"],
            "image_path": str(image_path),
            "raw_num_predictions": sample["num_predictions"],
            "selected_predictions": len(boxes_tensor),
        }
        return image_tensor, target, meta


def collate_unmore_teacher_batch(batch):
    images, targets, metas = zip(*batch)
    return list(images), list(targets), list(metas)
