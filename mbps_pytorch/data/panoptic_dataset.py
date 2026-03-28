"""Cityscapes Panoptic Dataset with Pseudo-Labels for Mask2Former.

Loads images + pseudo-semantic labels + pseudo-instance labels and converts
them into per-segment mask+class targets for Mask2Former training.

For each image:
- Thing instances: each connected instance mask gets its majority semantic class.
- Stuff regions: one mask per stuff class present in the image.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)

# Cityscapes 19-class split
CITYSCAPES_STUFF_CLASSES = set(range(11))    # 0-10: road, sidewalk, building, ...
CITYSCAPES_THING_CLASSES = set(range(11, 19))  # 11-18: person, rider, car, ...


class CityscapesPanopticPseudoDataset(Dataset):
    """Cityscapes with pseudo-label panoptic annotations for Mask2Former.

    Args:
        data_dir: Cityscapes root (with leftImg8bit/).
        pseudo_semantic_dir: Dir with semantic pseudo-label PNGs (trainID 0-18).
        pseudo_instance_dir: Dir with instance pseudo-label PNGs (0=bg, 1+=instance).
        stuff_things_path: Path to stuff_things.json (maps class_id → "stuff"/"thing").
            If None, uses Cityscapes default split.
        split: "train" or "val".
        crop_size: Random crop size for training augmentation.
        min_scale: Min scale for random resize augmentation.
        max_scale: Max scale for random resize augmentation.
        mask_stride: Downsample factor for target masks (4 = 1/4 resolution).
    """

    def __init__(
        self,
        data_dir: str,
        pseudo_semantic_dir: str,
        pseudo_instance_dir: str,
        stuff_things_path: str | None = None,
        split: str = "train",
        crop_size: int = 512,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
        mask_stride: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.pseudo_semantic_dir = pseudo_semantic_dir
        self.pseudo_instance_dir = pseudo_instance_dir
        self.split = split
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.mask_stride = mask_stride

        # Load stuff/things classification
        if stuff_things_path and os.path.exists(stuff_things_path):
            with open(stuff_things_path) as f:
                mapping = json.load(f)
            # Support multiple JSON formats
            if "stuff_ids" in mapping:
                # Format: {stuff_ids: [...], thing_ids: [...]}
                self.stuff_classes = set(mapping["stuff_ids"])
                self.thing_classes = set(mapping["thing_ids"])
            elif "classification" in mapping:
                # Format: {classification: {id: {label: "stuff"/"thing"}}}
                self.stuff_classes = {int(k) for k, v in mapping["classification"].items()
                                      if v.get("label") == "stuff"}
                self.thing_classes = {int(k) for k, v in mapping["classification"].items()
                                      if v.get("label") == "thing"}
            else:
                # Format: {id: "stuff"/"thing"}
                self.stuff_classes = {int(k) for k, v in mapping.items() if v == "stuff"}
                self.thing_classes = {int(k) for k, v in mapping.items() if v == "thing"}
        else:
            self.stuff_classes = CITYSCAPES_STUFF_CLASSES
            self.thing_classes = CITYSCAPES_THING_CLASSES

        # ImageNet normalization
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} {split} samples")

    def _load_samples(self) -> list[dict[str, str]]:
        """Find image-pseudolabel triplets."""
        samples = []
        img_dir = os.path.join(self.data_dir, "leftImg8bit", self.split)

        if not os.path.isdir(img_dir):
            logger.warning(f"Image dir not found: {img_dir}")
            return samples

        for city in sorted(os.listdir(img_dir)):
            city_dir = os.path.join(img_dir, city)
            if not os.path.isdir(city_dir):
                continue

            for fname in sorted(os.listdir(city_dir)):
                if not fname.endswith("_leftImg8bit.png"):
                    continue

                base = fname.replace("_leftImg8bit.png", "")
                base_with_suffix = f"{base}_leftImg8bit"

                # Find matching pseudo-labels
                # Naming: {base}_leftImg8bit.png for semantic, {base}_leftImg8bit_instance.png for instance
                sem_path = os.path.join(self.pseudo_semantic_dir, self.split, city, f"{base_with_suffix}.png")
                inst_path = os.path.join(self.pseudo_instance_dir, self.split, city, f"{base_with_suffix}_instance.png")

                # Fallback naming: without _leftImg8bit suffix
                if not os.path.exists(sem_path):
                    sem_path = os.path.join(self.pseudo_semantic_dir, self.split, city, f"{base}.png")
                if not os.path.exists(inst_path):
                    inst_path = os.path.join(self.pseudo_instance_dir, self.split, city, f"{base}_instance.png")

                samples.append({
                    "image": os.path.join(city_dir, fname),
                    "semantic": sem_path,
                    "instance": inst_path,
                    "image_id": base,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load image + pseudo-labels and convert to Mask2Former targets.

        Returns:
            image: (3, crop_H, crop_W) normalized float32.
            targets: {
                labels: (M,) int64 class indices,
                masks: (M, crop_H/stride, crop_W/stride) float32 binary masks,
            }
            metadata: {image_id: str}
        """
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["image"]).convert("RGB")

        # Load pseudo-labels
        sem_label = self._load_label(sample["semantic"])
        inst_label = self._load_label(sample["instance"])

        # Convert to numpy
        image_np = np.array(image, dtype=np.float32) / 255.0  # (H, W, 3)
        sem_np = np.array(sem_label, dtype=np.int64)  # (H, W)
        inst_np = np.array(inst_label, dtype=np.int64)  # (H, W)

        # Data augmentation (train only)
        if self.split == "train":
            image_np, sem_np, inst_np = self._augment(image_np, sem_np, inst_np)

        # Convert to Mask2Former format: per-segment masks + class labels
        labels, masks = self._create_panoptic_targets(sem_np, inst_np)

        # To tensors
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # (3, H, W)
        image_tensor = self.normalize(image_tensor)

        # Downsample masks to mask_stride resolution
        H, W = image_np.shape[:2]
        mask_H, mask_W = H // self.mask_stride, W // self.mask_stride
        if len(masks) > 0:
            masks_tensor = torch.from_numpy(np.stack(masks)).float()  # (M, H, W)
            masks_tensor = torch.nn.functional.interpolate(
                masks_tensor.unsqueeze(0),
                size=(mask_H, mask_W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # (M, mask_H, mask_W)
            # Re-binarize after interpolation
            masks_tensor = (masks_tensor > 0.5).float()
        else:
            masks_tensor = torch.zeros(0, mask_H, mask_W)

        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        return {
            "image": image_tensor,
            "targets": {
                "labels": labels_tensor,
                "masks": masks_tensor,
            },
            "metadata": {"image_id": sample["image_id"]},
        }

    def _load_label(self, path: str) -> Image.Image:
        """Load a label image, returning zeros if not found."""
        if os.path.exists(path):
            return Image.open(path)
        else:
            logger.debug(f"Label not found, using zeros: {path}")
            return Image.fromarray(np.zeros((512, 1024), dtype=np.uint8))

    def _augment(
        self, image: np.ndarray, sem: np.ndarray, inst: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random resize + crop + flip augmentation."""
        H, W = image.shape[:2]

        # Random scale
        scale = random.uniform(self.min_scale, self.max_scale)
        new_H = int(H * scale)
        new_W = int(W * scale)

        # Resize
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        image_pil = image_pil.resize((new_W, new_H), Image.BILINEAR)
        sem_pil = Image.fromarray(sem.astype(np.uint8))
        sem_pil = sem_pil.resize((new_W, new_H), Image.NEAREST)
        inst_pil = Image.fromarray(inst.astype(np.uint16))
        inst_pil = inst_pil.resize((new_W, new_H), Image.NEAREST)

        image = np.array(image_pil, dtype=np.float32) / 255.0
        sem = np.array(sem_pil, dtype=np.int64)
        inst = np.array(inst_pil, dtype=np.int64)

        H, W = image.shape[:2]

        # Random crop
        crop_H = min(self.crop_size, H)
        crop_W = min(self.crop_size, W)

        # Pad if image is smaller than crop
        if H < self.crop_size or W < self.crop_size:
            pad_H = max(0, self.crop_size - H)
            pad_W = max(0, self.crop_size - W)
            image = np.pad(image, ((0, pad_H), (0, pad_W), (0, 0)), mode="constant")
            sem = np.pad(sem, ((0, pad_H), (0, pad_W)), mode="constant", constant_values=255)
            inst = np.pad(inst, ((0, pad_H), (0, pad_W)), mode="constant")
            H, W = image.shape[:2]
            crop_H = self.crop_size
            crop_W = self.crop_size

        top = random.randint(0, H - crop_H)
        left = random.randint(0, W - crop_W)
        image = image[top:top + crop_H, left:left + crop_W]
        sem = sem[top:top + crop_H, left:left + crop_W]
        inst = inst[top:top + crop_H, left:left + crop_W]

        # Random horizontal flip
        if random.random() > 0.5:
            image = image[:, ::-1].copy()
            sem = sem[:, ::-1].copy()
            inst = inst[:, ::-1].copy()

        return image, sem, inst

    def _create_panoptic_targets(
        self, sem: np.ndarray, inst: np.ndarray,
    ) -> tuple[list[int], list[np.ndarray]]:
        """Convert semantic + instance maps to per-segment targets.

        For thing classes: each unique instance ID gets its own mask + majority class.
        For stuff classes: one mask per class covering all pixels of that class.

        Returns:
            labels: list of class IDs (int).
            masks: list of binary masks (H, W) float32.
        """
        labels = []
        masks = []

        # Handle stuff classes: one mask per stuff class present
        for cls_id in self.stuff_classes:
            cls_mask = (sem == cls_id)
            if cls_mask.sum() < 64:  # Skip tiny regions
                continue
            labels.append(cls_id)
            masks.append(cls_mask.astype(np.float32))

        # Handle thing classes: per-instance masks
        thing_mask = np.isin(sem, list(self.thing_classes))
        instance_ids = np.unique(inst[thing_mask])
        instance_ids = instance_ids[instance_ids > 0]  # Skip background

        for inst_id in instance_ids:
            inst_mask = (inst == inst_id)
            if inst_mask.sum() < 16:  # Skip tiny instances
                continue

            # Majority vote for semantic class
            sem_vals = sem[inst_mask]
            # Filter to thing classes only
            thing_vals = sem_vals[np.isin(sem_vals, list(self.thing_classes))]
            if len(thing_vals) == 0:
                continue
            cls_id = np.bincount(thing_vals).argmax()

            labels.append(int(cls_id))
            masks.append(inst_mask.astype(np.float32))

        return labels, masks
