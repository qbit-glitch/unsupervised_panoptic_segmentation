"""Dataset for M2PR (Mamba2 Panoptic Refiner) training.

Loads precomputed features from disk:
  - DINOv2 features: .npy (2048, 768) float16
  - CAUSE soft logits: .pt (27, 32, 64) float16  [or one-hot from .png]
  - SPIdepth depth: .npy (512, 1024) float32
  - Instance masks: .png uint16

Computes geometric features and instance descriptors on-the-fly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from mbps_pytorch.models.refiner.geometric_features import compute_geometric_features
from mbps_pytorch.models.refiner.instance_encoder import compute_instance_descriptor_fast


class RefinerDataset(Dataset):
    """Dataset for M2PR refiner training.

    Args:
        cityscapes_root: Path to Cityscapes dataset root.
        split: "train" or "val".
        dinov2_dir: Subdir for DINOv2 features (default: "dinov2_features").
        semantic_dir: Subdir for CAUSE semantics (default: "pseudo_semantic_cause").
        depth_dir: Subdir for depth maps (default: "depth_spidepth").
        instance_dir: Subdir for instance masks (default: "pseudo_instance_spidepth").
        use_soft_logits: If True, load CAUSE soft logits from .pt files.
        num_classes: Number of semantic classes (27).
        target_h: Patch grid height (32).
        target_w: Patch grid width (64).
        cache_features: Cache computed features in memory.
    """

    def __init__(
        self,
        cityscapes_root: str,
        split: str = "train",
        dinov2_dir: str = "dinov2_features",
        semantic_dir: str = "pseudo_semantic_cause",
        depth_dir: str = "depth_spidepth",
        instance_dir: str = "pseudo_instance_spidepth",
        use_soft_logits: bool = True,
        num_classes: int = 27,
        target_h: int = 32,
        target_w: int = 64,
        cache_features: bool = False,
    ):
        super().__init__()
        self.cityscapes_root = cityscapes_root
        self.split = split
        self.num_classes = num_classes
        self.target_h = target_h
        self.target_w = target_w
        self.use_soft_logits = use_soft_logits
        self.cache_features = cache_features

        self.dinov2_root = os.path.join(cityscapes_root, dinov2_dir, split)
        self.semantic_root = os.path.join(cityscapes_root, semantic_dir, split)
        self.depth_root = os.path.join(cityscapes_root, depth_dir, split)
        self.instance_root = os.path.join(cityscapes_root, instance_dir, split)

        # Enumerate samples
        self.samples = self._find_samples()
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def _find_samples(self) -> list:
        """Find all samples by scanning DINOv2 features directory."""
        samples = []
        if not os.path.isdir(self.dinov2_root):
            raise FileNotFoundError(f"DINOv2 directory not found: {self.dinov2_root}")

        for city in sorted(os.listdir(self.dinov2_root)):
            city_dir = os.path.join(self.dinov2_root, city)
            if not os.path.isdir(city_dir):
                continue
            for fname in sorted(os.listdir(city_dir)):
                if not fname.endswith(".npy"):
                    continue
                dinov2_fname = fname  # Keep original for DINOv2 path
                # Strip _leftImg8bit suffix (DINOv2 files have it, others don't)
                stem = fname.replace(".npy", "").replace("_leftImg8bit", "")
                # Verify all required files exist
                paths = self._get_paths(city, stem, dinov2_fname=dinov2_fname)
                if all(os.path.exists(p) for p in [
                    paths["dinov2"], paths["semantic"], paths["depth"],
                    paths["instance"],
                ]):
                    samples.append({"city": city, "stem": stem, "dinov2_fname": dinov2_fname})

        return samples

    def _get_paths(self, city: str, stem: str, dinov2_fname: Optional[str] = None) -> Dict[str, str]:
        """Get file paths for a sample."""
        # DINOv2 files may have _leftImg8bit suffix; use original filename if provided
        dinov2_name = dinov2_fname if dinov2_fname else f"{stem}.npy"
        return {
            "dinov2": os.path.join(self.dinov2_root, city, dinov2_name),
            "semantic": os.path.join(self.semantic_root, city, f"{stem}.png"),
            "logits": os.path.join(self.semantic_root, city, f"{stem}_logits.pt"),
            "depth": os.path.join(self.depth_root, city, f"{stem}.npy"),
            "instance": os.path.join(self.instance_root, city, f"{stem}_instance.png"),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_features and idx in self._cache:
            return self._cache[idx]

        sample = self.samples[idx]
        paths = self._get_paths(sample["city"], sample["stem"], dinov2_fname=sample.get("dinov2_fname"))

        # 1. DINOv2 features: (2048, 768) float16 → float32
        dino_feat = torch.from_numpy(
            np.load(paths["dinov2"])
        ).float()  # (N, 768)

        # 2. CAUSE semantic logits: soft or one-hot
        if self.use_soft_logits and os.path.exists(paths["logits"]):
            # Soft logits: (27, 32, 64) float16 → (N, 27) float32
            cause_logits = torch.load(
                paths["logits"], map_location="cpu", weights_only=True
            ).float()  # (27, H, W)
            cause_logits = cause_logits.permute(1, 2, 0).reshape(-1, self.num_classes)
        else:
            # Fallback: one-hot from argmax PNG
            from PIL import Image
            sem_img = np.array(Image.open(paths["semantic"]))  # (H, W) uint8
            # Downsample to patch level
            sem_patch = np.array(
                Image.fromarray(sem_img).resize(
                    (self.target_w, self.target_h), Image.NEAREST
                )
            )  # (32, 64)
            sem_flat = sem_patch.ravel()  # (N,)
            cause_logits = torch.zeros(len(sem_flat), self.num_classes)
            for i, c in enumerate(sem_flat):
                if c < self.num_classes:
                    cause_logits[i, c] = 1.0

        # 3. Depth map: (H_d, W_d) → geometric features (N, 18)
        depth_map = np.load(paths["depth"]).astype(np.float32)
        geo_features = compute_geometric_features(
            torch.from_numpy(depth_map), self.target_h, self.target_w
        )  # (N, 18)

        # Depth at patch level for FiLM conditioning: (N,)
        depth_patch = F.interpolate(
            torch.from_numpy(depth_map).float().unsqueeze(0).unsqueeze(0),
            size=(self.target_h, self.target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze().reshape(-1)  # (N,)

        # 4. Instance descriptor: (N, 8) + instance labels: (N,)
        from PIL import Image
        inst_img = np.array(Image.open(paths["instance"]))  # (H, W) uint16
        sem_img_np = np.array(Image.open(paths["semantic"]))  # (H, W) uint8

        inst_descriptor = compute_instance_descriptor_fast(
            inst_img, sem_img_np, depth_map, self.target_h, self.target_w
        )  # (N, 8)

        # Instance labels at patch level (for discriminative loss)
        inst_patch = np.array(
            Image.fromarray(inst_img.astype(np.uint16)).resize(
                (self.target_w, self.target_h), Image.NEAREST
            )
        )
        inst_labels = torch.from_numpy(inst_patch.ravel().astype(np.int64))  # (N,)

        result = {
            "dino_features": dino_feat,       # (N, 768)
            "cause_logits": cause_logits,     # (N, 27)
            "geo_features": geo_features,     # (N, 18)
            "inst_descriptor": inst_descriptor,  # (N, 8)
            "depth": depth_patch,             # (N,)
            "instance_labels": inst_labels,   # (N,)
            "stem": sample["stem"],
            "city": sample["city"],
        }

        if self.cache_features:
            self._cache[idx] = result

        return result


def refiner_collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """Custom collate function for RefinerDataset.

    Stacks tensor fields and collects metadata.
    """
    result = {}
    tensor_keys = [
        "dino_features", "cause_logits", "geo_features",
        "inst_descriptor", "depth", "instance_labels",
    ]
    for key in tensor_keys:
        result[key] = torch.stack([b[key] for b in batch])

    result["stems"] = [b["stem"] for b in batch]
    result["cities"] = [b["city"] for b in batch]
    return result
