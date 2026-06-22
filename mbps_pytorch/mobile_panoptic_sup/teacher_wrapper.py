"""Frozen fp16 EoMT teacher for knowledge distillation.

Loads tue-mps/eomt-dinov3-coco-panoptic-large-640, applies the
empty_weight NaN fix, casts to fp16, freezes all params.

Usage:
    teacher = TeacherWrapper("cuda")
    targets = teacher(pixel_values)  # TeacherTargets namedtuple
    # targets.class_logits: [B, Q, 134]   fp32
    # targets.mask_logits:  [B, Q, h, w]  fp32
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

REPO = "tue-mps/eomt-dinov3-coco-panoptic-large-640"
_COCO = Path(os.environ.get("COCO_ROOT", "/mnt/HDD_16TB/coco"))


def _cat_id2label() -> dict[int, str]:
    for jpath in [
        _COCO / "annotations/panoptic_train2017.json",
        _COCO / "annotations/panoptic_val2017.json",
    ]:
        if jpath.exists():
            cats = sorted(json.loads(jpath.read_text())["categories"], key=lambda c: c["id"])
            return {i: c["name"] for i, c in enumerate(cats)}
    raise FileNotFoundError(
        f"No panoptic JSON found under {_COCO}/annotations/. "
        "Set COCO_ROOT or download panoptic_annotations_trainval2017.zip."
    )


@dataclass
class TeacherTargets:
    class_logits: torch.Tensor  # [B, Q, 134]  fp32
    mask_logits: torch.Tensor   # [B, Q, h, w] fp32


class TeacherWrapper(nn.Module):
    """Frozen fp16 EoMT teacher; forward returns TeacherTargets (fp32 on cpu)."""

    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        from transformers import AutoModelForUniversalSegmentation

        id2label = _cat_id2label()
        model = AutoModelForUniversalSegmentation.from_pretrained(
            REPO,
            num_labels=133,
            id2label=id2label,
            label2id={v: k for k, v in id2label.items()},
            ignore_mismatched_sizes=True,
        )
        # Fix shipped empty_weight all-zeros → CE NaN (safe even in inference)
        with torch.no_grad():
            model.criterion.empty_weight.fill_(1.0)
            model.criterion.empty_weight[-1] = model.config.no_object_weight

        model = model.eval().half().to(device)
        for p in model.parameters():
            p.requires_grad_(False)

        self._model = model
        self._dev = device
        print(
            f"TeacherWrapper: {REPO} | fp16 | {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params",
            flush=True,
        )

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> TeacherTargets:
        out = self._model(pixel_values=pixel_values.half().to(self._dev))
        return TeacherTargets(
            class_logits=out.class_queries_logits.float(),
            mask_logits=out.masks_queries_logits.float(),
        )
