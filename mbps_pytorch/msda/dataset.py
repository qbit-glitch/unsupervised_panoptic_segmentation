"""Cached feature dataset for MSDA training."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CachedFeatureDataset(Dataset):
    """Dataset loading pre-cached DINOv3 features + DepthPro depth maps.

    Features: (2048, 1024) = 32×64 patches × 1024-dim
    Depth: (512, 1024) full-resolution depth map
    """

    def __init__(
        self,
        feature_dir: str,
        depth_dir: str,
        split: str = "train",
        spatial_h: int = 32,
        spatial_w: int = 64,
        augment: bool = False,
        feature_noise_std: float = 0.01,
        feature_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dir = Path(feature_dir) / split
        self.depth_dir = Path(depth_dir) / split
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.augment = augment
        self.feature_noise_std = feature_noise_std
        self.feature_dropout = feature_dropout

        self.samples = self._discover_samples()
        logger.info(
            f"CachedFeatureDataset: {len(self.samples)} samples from {split}"
        )

    def _discover_samples(self) -> List[Tuple[Path, Path]]:
        samples = []
        for city_dir in sorted(self.feature_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            city_name = city_dir.name
            depth_city = self.depth_dir / city_name
            for feat_file in sorted(city_dir.glob("*.npy")):
                stem = feat_file.stem
                depth_stem = stem.replace("_leftImg8bit", "")
                depth_file = depth_city / f"{depth_stem}.npy"
                if not depth_file.exists():
                    depth_file = depth_city / f"{stem}.npy"
                if depth_file.exists():
                    samples.append((feat_file, depth_file))
                else:
                    logger.warning(f"No depth for {feat_file.name}, skipping")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        feat_path, depth_path = self.samples[idx]

        features = np.load(feat_path).astype(np.float32)  # (2048, 1024)
        depth = np.load(depth_path).astype(np.float32)  # (512, 1024)

        features = torch.from_numpy(features)  # (N, D)
        depth = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)

        if self.augment:
            features = self._augment_features(features)
            depth = self._augment_depth(depth)

        return {
            "features": features,
            "depth": depth,
            "stem": feat_path.stem,
        }

    def _augment_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.feature_noise_std > 0:
            noise = torch.randn_like(features) * self.feature_noise_std
            features = features + noise
        if self.feature_dropout > 0:
            mask = torch.rand(features.shape[0], 1) > self.feature_dropout
            features = features * mask.float()
        return features

    def _augment_depth(self, depth: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > 0.5:
            depth = torch.flip(depth, dims=[-1])
        return depth
