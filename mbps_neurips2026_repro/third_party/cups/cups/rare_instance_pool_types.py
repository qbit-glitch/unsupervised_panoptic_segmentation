from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

__all__ = ("InstanceCrop",)


@dataclass(frozen=True)
class InstanceCrop:
    train_id: int
    image_jpeg: bytes
    mask_png: bytes
    src_depth_quantile: float
    bbox: Tuple[int, int, int, int]
    area: int
