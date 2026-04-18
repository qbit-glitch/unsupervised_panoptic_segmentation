"""G3 LSJ + G4 ColorJitter modules."""
from __future__ import annotations

import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ColorJitter

__all__ = ["LargeScaleJitter", "ColorJitterModule"]


class LargeScaleJitter(nn.Module):
    def __init__(self, min_scale: float = 0.1, max_scale: float = 2.0, target_size: Tuple[int, int] = (640, 1280)) -> None:
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.target_size = target_size

    def forward(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sample scale.
        scale = random.uniform(self.min_scale, self.max_scale)
        H, W = image.shape[-2:]
        new_H = max(1, int(H * scale))
        new_W = max(1, int(W * scale))
        img = F.interpolate(image.unsqueeze(0).float(), size=(new_H, new_W), mode="bilinear", align_corners=False).squeeze(0)
        lbl = F.interpolate(label.unsqueeze(0).float(), size=(new_H, new_W), mode="nearest").squeeze(0).long()
        # Pad or crop to target size.
        tH, tW = self.target_size
        if new_H < tH or new_W < tW:
            pad_h = max(0, tH - new_H)
            pad_w = max(0, tW - new_W)
            img = F.pad(img, (0, pad_w, 0, pad_h), value=0.0)
            lbl = F.pad(lbl, (0, pad_w, 0, pad_h), value=255)  # ignore in pad
            new_H, new_W = img.shape[-2:]
        if new_H > tH or new_W > tW:
            y = random.randint(0, new_H - tH)
            x = random.randint(0, new_W - tW)
            img = img[:, y : y + tH, x : x + tW]
            lbl = lbl[:, y : y + tH, x : x + tW]
        return img, lbl


class ColorJitterModule(nn.Module):
    def __init__(self, brightness: float = 0.4, contrast: float = 0.4, saturation: float = 0.4, hue: float = 0.1) -> None:
        super().__init__()
        self.jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self._identity = all(v == 0.0 for v in (brightness, contrast, saturation, hue))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self._identity:
            return image
        return self.jitter(image)
