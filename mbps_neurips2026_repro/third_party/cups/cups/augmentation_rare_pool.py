from __future__ import annotations

import io
import logging
import pickle
import random
from copy import deepcopy
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.structures import BitMasks, Boxes, Instances
from PIL import Image
from torch import Tensor

from cups.augmentation import _paste_one
from cups.rare_instance_pool_types import InstanceCrop

logger = logging.getLogger(__name__)

__all__: Tuple[str, ...] = ("InstanceCrop", "RareInstancePoolCopyPaste")


class RareInstancePoolCopyPaste(nn.Module):
    """Copy-paste rare instances from a persistent class-balanced pool."""

    def __init__(
        self,
        thing_class: int,
        pool_path: str,
        scale_range: Tuple[float, float] = (0.25, 1.5),
        pastes_per_image: Tuple[int, int] = (1, 3),
        use_depth_placement: bool = True,
        class_repeat_overrides: Sequence[Sequence[int | float]] = (),
        thing_id_to_trainid: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.thing_class = int(thing_class)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.pastes_per_image = (int(pastes_per_image[0]), int(pastes_per_image[1]))
        self.use_depth_placement = bool(use_depth_placement)
        self.thing_id_to_trainid = tuple(int(v) for v in thing_id_to_trainid) if thing_id_to_trainid else None
        self.trainid_to_thing_id = (
            {train_id: thing_id for thing_id, train_id in enumerate(self.thing_id_to_trainid)}
            if self.thing_id_to_trainid is not None
            else None
        )
        overrides = {int(k): float(v) for k, v in class_repeat_overrides}

        with open(pool_path, "rb") as f:
            raw_pool = pickle.load(f)
        self.pool: Dict[int, List[InstanceCrop]] = {
            int(k): list(v) for k, v in raw_pool.items() if len(v) > 0
        }
        if self.trainid_to_thing_id is not None:
            self.pool = {k: v for k, v in self.pool.items() if k in self.trainid_to_thing_id}

        self.n_class = {k: len(v) for k, v in self.pool.items()}
        self.class_ids = sorted(self.pool)
        self.weights = [
            (1.0 / np.sqrt(max(1, self.n_class[c]))) * overrides.get(c, 1.0)
            for c in self.class_ids
        ]

    @torch.no_grad()
    def forward(
        self,
        batch_source: List[Dict[str, Tensor | Instances]],
        batch_target: List[Dict[str, Tensor | Instances]],
    ) -> List[Dict[str, Tensor | Instances]]:
        if not self.class_ids:
            return batch_target

        batch_target_copy = deepcopy(batch_target)
        output: List[Dict[str, Tensor | Instances]] = []
        for sample in batch_target:
            image_original: Tensor = sample["image"]
            semantic_segmentation_original: Tensor = sample["sem_seg"]
            instance_masks_original: Tensor = sample["instances"].gt_masks.tensor  # type: ignore
            bounding_boxes_original: Tensor = sample["instances"].gt_boxes.tensor  # type: ignore
            classes_original: Tensor = sample["instances"].gt_classes  # type: ignore
            if instance_masks_original.shape[0] == 0:
                instance_masks_original = instance_masks_original.to(image_original.device)
                bounding_boxes_original = bounding_boxes_original.to(image_original.device)
                classes_original = classes_original.to(image_original.device)

            low, high = self.pastes_per_image
            num_pastes = int(torch.randint(low=low, high=high + 1, size=(1,)).item())
            for _ in range(num_pastes):
                train_id = random.choices(self.class_ids, weights=self.weights, k=1)[0]
                crop = random.choice(self.pool[train_id])
                image_crop, instance_mask = self._decode_crop(crop, image_original.device, image_original.dtype)
                image_crop, instance_mask = self._augment_crop(image_crop, instance_mask)
                if not instance_mask.any():
                    continue

                top, left = self._choose_position(sample, image_crop, instance_mask, crop.src_depth_quantile)
                class_id = self._class_id_for_trainid(train_id, classes_original.device, classes_original.dtype)
                (
                    image_original,
                    semantic_segmentation_original,
                    instance_masks_original,
                    bounding_boxes_original,
                    classes_original,
                ) = _paste_one(
                    image_original=image_original,
                    semantic_segmentation_original=semantic_segmentation_original,
                    instance_masks_original=instance_masks_original,
                    bounding_boxes_original=bounding_boxes_original,
                    classes_original=classes_original,
                    image_crop=image_crop,
                    instance_mask=instance_mask,
                    class_id=class_id,
                    top=top,
                    left=left,
                )

            valid_objects = instance_masks_original.any(dim=-1).any(dim=-1)
            if valid_objects.shape[0] != bounding_boxes_original.shape[0]:
                logger.warning(
                    "rare-pool: shape mismatch valid=%d boxes=%d, dropping pasted instances for this sample",
                    int(valid_objects.shape[0]),
                    int(bounding_boxes_original.shape[0]),
                )
                return batch_target_copy
            instance_masks_original = instance_masks_original[valid_objects]
            bounding_boxes_original = bounding_boxes_original[valid_objects]
            classes_original = classes_original[valid_objects]

            out_sample = dict(sample)
            out_sample.update(
                {
                    "image": image_original,
                    "sem_seg": semantic_segmentation_original,
                    "instances": Instances(
                        image_size=tuple(image_original.shape[1:]),
                        gt_masks=BitMasks(instance_masks_original),
                        gt_boxes=Boxes(bounding_boxes_original),
                        gt_classes=classes_original,
                    ),
                }
            )
            output.append(out_sample)
        return output

    @staticmethod
    def _decode_crop(crop: InstanceCrop, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        image = Image.open(io.BytesIO(crop.image_jpeg)).convert("RGB")
        mask = Image.open(io.BytesIO(crop.mask_png)).convert("L")
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = np.asarray(mask, dtype=np.uint8) > 0
        image_t = torch.from_numpy(image_arr).permute(2, 0, 1).to(device=device, dtype=dtype)
        mask_t = torch.from_numpy(mask_arr).to(device=device)
        return image_t, mask_t

    def _augment_crop(self, image_crop: Tensor, instance_mask: Tensor) -> Tuple[Tensor, Tensor]:
        scale_factor = float(np.random.uniform(self.scale_range[0], self.scale_range[1]))
        instance_mask = F.interpolate(
            instance_mask[None, None].float(),
            scale_factor=scale_factor,
            mode="nearest",
        )[0, 0].bool()
        image_crop = F.interpolate(
            image_crop[None],
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )[0]
        if torch.rand(1).item() > 0.5:
            instance_mask = instance_mask.flip(dims=(-1,))
            image_crop = image_crop.flip(dims=(-1,))
        return image_crop, instance_mask

    def _class_id_for_trainid(self, train_id: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.trainid_to_thing_id is not None:
            class_id = self.trainid_to_thing_id[train_id]
        else:
            class_id = train_id
        return torch.tensor(class_id, device=device, dtype=dtype)

    def _choose_position(
        self,
        sample: Dict[str, Tensor | Instances],
        image_crop: Tensor,
        instance_mask: Tensor,
        src_depth_quantile: float,
    ) -> Tuple[int | None, int | None]:
        target_shape = sample["image"].shape[1:]  # type: ignore
        h, w = instance_mask.shape
        h = min(h, target_shape[0])
        w = min(w, target_shape[1])
        max_top = target_shape[0] - h
        max_left = target_shape[1] - w
        if max_top < 0 or max_left < 0:
            return None, None
        if not self.use_depth_placement or "depth" not in sample:
            return None, None

        depth = sample["depth"]  # type: ignore
        if depth.ndim == 3:
            depth = depth[0]
        mask = instance_mask[:h, :w]
        for _ in range(3):
            top = int(torch.randint(0, max_top + 1, size=(1,)).item()) if max_top > 0 else 0
            left = int(torch.randint(0, max_left + 1, size=(1,)).item()) if max_left > 0 else 0
            target_patch = depth[top : top + h, left : left + w]
            values = target_patch[mask.to(target_patch.device)]
            if values.numel() == 0:
                continue
            target_median = float(values.median().item())
            if abs(target_median - float(src_depth_quantile)) <= 0.15:
                return top, left
        return None, None
