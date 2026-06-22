"""COCO-train-118k panoptic GT dataloader for distillation.

Reads official panoptic PNG+JSON GT (no auto-labels), returns
{pixel_values, mask_labels, class_labels} in the EoMT processor
convention (same as coco_eomt_finetune.CocoAutoDS).

Category ordering: sorted by COCO category_id → contiguous 0..132.
Crowd segments (iscrowd=1) are treated as void / skipped.

COCO_ROOT env var (default /mnt/HDD_16TB/coco) must contain:
  train2017/*.jpg
  annotations/panoptic_train2017.json
  annotations/panoptic_train2017/*.png
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = "tue-mps/eomt-dinov3-coco-panoptic-large-640"
_COCO = Path(os.environ.get("COCO_ROOT", "/mnt/HDD_16TB/coco"))

_TRAIN_JSON = _COCO / "annotations/panoptic_train2017.json"
_TRAIN_PNG_DIR = _COCO / "annotations/panoptic_train2017"
_TRAIN_IMG_DIR = _COCO / "train2017"

_PAN_URL = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"


def ensure_panoptic_gt() -> None:
    """Download + unzip panoptic GT if missing (idempotent)."""
    if _TRAIN_JSON.exists() and _TRAIN_PNG_DIR.is_dir():
        return
    zip_path = _COCO / "panoptic_annotations_trainval2017.zip"
    if not zip_path.exists():
        print(f"Downloading panoptic GT from {_PAN_URL} ...", flush=True)
        subprocess.run(["wget", "-q", "-O", str(zip_path), _PAN_URL], check=True)
    print(f"Unzipping {zip_path} ...", flush=True)
    subprocess.run(["unzip", "-q", "-n", str(zip_path), "-d", str(_COCO)], check=True)
    print("Panoptic GT ready.", flush=True)


def _rgb2id(arr: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 → HxW int32 (panopticapi encoding R+G*256+B*256²)."""
    a = arr.astype(np.int32)
    return a[..., 0] + a[..., 1] * 256 + a[..., 2] * 256 * 256


def _load_meta(json_path: Path) -> tuple[dict[int, int], list[dict], dict[int, str]]:
    """Returns (catid→contiguous_idx, ann_list, imgid→img_filename)."""
    data = json.loads(json_path.read_text())
    cats = sorted(data["categories"], key=lambda c: c["id"])
    catid2idx: dict[int, int] = {c["id"]: i for i, c in enumerate(cats)}
    imgid2file: dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}
    return catid2idx, data["annotations"], imgid2file


class CocoPanopticGTDS(Dataset):
    """COCO train2017 panoptic GT → EoMT-processor-compatible batch items."""

    def __init__(self, proc: Any, limit: int = 0) -> None:
        catid2idx, anns, imgid2file = _load_meta(_TRAIN_JSON)
        self._catid2idx = catid2idx
        self._imgid2file = imgid2file
        self._anns = anns[:limit] if limit else anns
        self._proc = proc

    def __len__(self) -> int:
        return len(self._anns)

    def __getitem__(self, i: int) -> dict:
        ann = self._anns[i]
        img_file = self._imgid2file[ann["image_id"]]
        img = Image.open(_TRAIN_IMG_DIR / img_file).convert("RGB")

        pan_rgb = np.array(Image.open(_TRAIN_PNG_DIR / ann["file_name"]).convert("RGB"))
        pan_id = _rgb2id(pan_rgb)  # HxW int32 segment ids

        # Build 1-based seg map + id→1-based class mapping for processor
        seg = np.zeros_like(pan_id, dtype=np.int64)
        id2cls: dict[int, int] = {}
        k = 1
        for seg_info in ann["segments_info"]:
            if seg_info.get("iscrowd", 0):
                continue  # void
            idx = self._catid2idx[seg_info["category_id"]]  # 0..132
            seg[pan_id == seg_info["id"]] = k
            id2cls[k] = idx + 1  # processor expects 1-based class
            k += 1

        enc = self._proc.preprocess(
            images=[img],
            segmentation_maps=[seg],
            instance_id_to_semantic_id=id2cls,
            return_tensors="pt",
        )
        return {
            "pixel_values": enc["pixel_values"][0],
            "mask_labels": enc["mask_labels"][0],     # [Q, H, W] float32 {0,1}
            "class_labels": enc["class_labels"][0],   # [Q] int64 0..132
        }


def collate(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "mask_labels": [b["mask_labels"] for b in batch],
        "class_labels": [b["class_labels"] for b in batch],
    }


def make_loader(proc: Any, bs: int = 4, workers: int = 4, limit: int = 0) -> DataLoader:
    ds = CocoPanopticGTDS(proc, limit=limit)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
    )


if __name__ == "__main__":
    import time
    from transformers import AutoImageProcessor

    print("=== coco_distill_data smoke ===", flush=True)
    ensure_panoptic_gt()

    proc = AutoImageProcessor.from_pretrained(REPO)
    proc.ignore_index = 255

    print("Loading dataset (limit=4)...", flush=True)
    ds = CocoPanopticGTDS(proc, limit=4)
    assert len(ds) == 4
    batch = collate([ds[i] for i in range(2)])

    # ── GT assertions ──────────────────────────────────────────────────────────
    pv = batch["pixel_values"]
    assert pv.dtype == torch.float32, f"pixel_values dtype {pv.dtype}"
    print(f"pixel_values: {pv.shape}", flush=True)

    for j, ml in enumerate(batch["mask_labels"]):
        assert ml.dtype == torch.float32, f"mask_labels[{j}] dtype {ml.dtype}"
        vals = set(ml.unique().tolist())
        assert vals.issubset({0.0, 1.0}), f"mask_labels[{j}] has non-binary values: {vals}"
        assert ml.sum() > 0, f"mask_labels[{j}] is all zeros (empty masks)"
        print(f"mask_labels[{j}]: {ml.shape}  unique={sorted(vals)}", flush=True)

    for j, cl in enumerate(batch["class_labels"]):
        assert cl.dtype == torch.int64, f"class_labels[{j}] dtype {cl.dtype}"
        lo, hi = int(cl.min()), int(cl.max())
        assert 0 <= lo and hi <= 132, f"class_labels[{j}] out of range [{lo},{hi}]"
        print(f"class_labels[{j}]: {cl.shape}  range=[{lo},{hi}]", flush=True)

    # ── Teacher assertions ─────────────────────────────────────────────────────
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from teacher_wrapper import TeacherWrapper

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nBuilding teacher on {dev}...", flush=True)
    teacher = TeacherWrapper(device=dev)

    t0 = time.time()
    targets = teacher(pv)
    ms_per_img = (time.time() - t0) / pv.shape[0] * 1000

    assert torch.isfinite(targets.class_logits).all(), "teacher class_logits has inf/nan"
    assert torch.isfinite(targets.mask_logits).all(), "teacher mask_logits has inf/nan"
    assert targets.class_logits.shape[0] == 2
    assert targets.class_logits.shape[2] == 134  # 133 classes + 1 no-object

    print(f"class_logits:  {targets.class_logits.shape}", flush=True)
    print(f"mask_logits:   {targets.mask_logits.shape}", flush=True)
    print(f"teacher speed: {ms_per_img:.1f} ms/img", flush=True)
    print("\nSMOKE PASS", flush=True)
