"""CUPS Stage-3 augmentations ported to EoMT (image, target-dict) format.

Faithful port of refs/cups/cups/augmentation.py for the EoMT self-training
pipeline. CUPS applies these INSIDE training_step, AFTER the teacher has
generated pseudo-labels on the clean image, in this exact order:

    1. CopyPasteAugmentation (max 3 thing objects, scale 0.25-1.5, h-flip)
    2. PhotometricAugmentations (gauss blur p=1.0, color jitter p=0.5,
       grayscale p=0.2)
    3. RandomCrop (resolution r in [512, 1024], crop (r, 2r))
    4. ResolutionJitter (random choice of fixed resolutions)

EoMT target format per image: dict with
    masks:    BoolTensor [N, H, W]
    labels:   LongTensor [N]
    is_crowd: BoolTensor [N]

Images are float tensors in [0, 255] (the EoMT LightningModule divides by
255 inside forward).
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torchvision import tv_tensors

# CUPS drops masks with <= 4 pixels after geometric augmentation.
_MIN_PIXELS_AFTER_AUG = 4


def _rebuild_target(masks: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "masks": tv_tensors.Mask(masks.bool()),
        "labels": labels.long(),
        "is_crowd": torch.zeros_like(labels, dtype=torch.bool),
    }


def _filter_small(masks: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if masks.shape[0] == 0:
        return masks, labels
    valid = masks.flatten(1).sum(dim=1) > _MIN_PIXELS_AFTER_AUG
    return masks[valid], labels[valid]


def _mask_bbox(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """Tight (y1, x1, y2, x2) box around a bool mask (assumed nonempty)."""
    ys, xs = torch.where(mask)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


@torch.no_grad()
def copy_paste_batch(
    imgs: torch.Tensor,
    targets: List[Dict[str, torch.Tensor]],
    thing_classes: Set[int],
    max_num_pasted_objects: int = 3,
    scale_range: Tuple[float, float] = (0.25, 1.5),
    use_random_horizontal_flipping: bool = True,
    min_bounding_box_size: Tuple[int, int] = (32, 32),
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """Port of CUPS CopyPasteAugmentation (source batch == target batch).

    Collects thing-object crops from the whole batch, then pastes a random
    number (1..max) of randomly scaled/flipped objects into each image.
    The pasted region erases the overlapped pixels from every existing mask
    (CUPS sets sem_seg to the thing slot and zeroes instance masks there);
    fully occluded masks are dropped.
    """
    B, _, H, W = imgs.shape

    # -- collect paste candidates from the batch (things only) --------------
    image_crops: List[torch.Tensor] = []
    mask_crops: List[torch.Tensor] = []
    crop_labels: List[int] = []
    for b in range(B):
        masks = targets[b]["masks"]
        labels = targets[b]["labels"]
        for i in range(masks.shape[0]):
            if int(labels[i]) not in thing_classes:
                continue
            mask = masks[i]
            if not bool(mask.any()):
                continue
            y1, x1, y2, x2 = _mask_bbox(mask)
            if (y2 - y1) <= min_bounding_box_size[0] or (x2 - x1) <= min_bounding_box_size[1]:
                continue
            image_crops.append(imgs[b, :, y1:y2, x1:x2])
            mask_crops.append(mask[y1:y2, x1:x2])
            crop_labels.append(int(labels[i]))

    if len(mask_crops) == 0:
        return imgs, targets

    out_imgs = imgs.clone()
    out_targets: List[Dict[str, torch.Tensor]] = []

    for b in range(B):
        image = out_imgs[b]
        masks = targets[b]["masks"].clone().bool()
        labels = targets[b]["labels"].clone()

        num_paste = int(torch.randint(1, max_num_pasted_objects + 1, (1,)).item())
        for _ in range(num_paste):
            idx = int(torch.randint(0, len(mask_crops), (1,)).item())
            obj_mask = mask_crops[idx]
            obj_img = image_crops[idx]
            scale = float(torch.empty(1).uniform_(*scale_range).item())
            obj_mask_s = F.interpolate(
                obj_mask[None, None].float(), scale_factor=scale, mode="nearest"
            )[0, 0].bool()
            obj_img_s = F.interpolate(obj_img[None], scale_factor=scale, mode="bilinear")[0]
            if use_random_horizontal_flipping and torch.rand(1).item() > 0.5:
                obj_mask_s = obj_mask_s.flip(dims=(-1,))
                obj_img_s = obj_img_s.flip(dims=(-1,))

            # Clip oversized objects to image extent (CUPS truncates).
            obj_img_s = obj_img_s[..., :H, :W]
            obj_mask_s = obj_mask_s[:H, :W]
            if obj_mask_s.numel() == 0 or not bool(obj_mask_s.any()):
                continue
            ch, cw = obj_mask_s.shape
            top = int(torch.randint(0, max(1, H - ch + 1), (1,)).item())
            left = int(torch.randint(0, max(1, W - cw + 1), (1,)).item())

            pasted = torch.zeros(H, W, dtype=torch.bool, device=image.device)
            pasted[top : top + ch, left : left + cw] = obj_mask_s

            region = image[:, top : top + ch, left : left + cw]
            region[:, obj_mask_s] = obj_img_s.to(region.dtype)[:, obj_mask_s]

            # Erase the pasted region from every existing mask.
            if masks.shape[0] > 0:
                masks = masks & ~pasted[None]
            masks = torch.cat((masks, pasted[None]), dim=0)
            labels = torch.cat(
                (labels, torch.tensor([crop_labels[idx]], dtype=labels.dtype, device=labels.device))
            )

        masks, labels = _filter_small(masks, labels)
        out_targets.append(_rebuild_target(masks, labels))

    return out_imgs, out_targets


class PhotometricAugmentations:
    """Port of CUPS PhotometricAugmentations (kornia, image-only).

    GaussianBlur(7x7, sigma 0.1-2.0, p=1.0) -> ColorJitter(0.4/0.4/0.4/0.1,
    p=0.5) -> RandomGrayscale(p=0.2). Operates on float images in [0, 255].
    """

    def __init__(self) -> None:
        from kornia.augmentation import (
            AugmentationSequential,
            ColorJitter,
            RandomGaussianBlur,
            RandomGrayscale,
        )

        self.augmentations = AugmentationSequential(
            RandomGaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0), p=1.0, keepdim=True),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5, keepdim=True),
            RandomGrayscale(p=0.2, keepdim=True),
            keepdim=True,
        )

    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        out = []
        for b in range(imgs.shape[0]):
            x = imgs[b].float() / 255.0
            x = self.augmentations(x).squeeze(0).clamp_(0.0, 1.0)
            out.append(x * 255.0)
        return torch.stack(out)


@torch.no_grad()
def random_crop_batch(
    imgs: torch.Tensor,
    targets: List[Dict[str, torch.Tensor]],
    resolution_min: int = 512,
    resolution_max: int = 1024,
    long_side_scale: float = 2.0,
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """Port of CUPS RandomCrop: one random (r, long_side_scale*r) crop shared
    across the batch, clamped to the image extent."""
    B, _, H, W = imgs.shape
    resolution = random.randint(resolution_min, resolution_max)
    crop_h = min(resolution, H)
    crop_w = min(round(long_side_scale * resolution), W)

    top = int(torch.randint(0, H - crop_h + 1, (1,)).item())
    left = int(torch.randint(0, W - crop_w + 1, (1,)).item())

    out_imgs = imgs[:, :, top : top + crop_h, left : left + crop_w]
    out_targets: List[Dict[str, torch.Tensor]] = []
    for b in range(B):
        masks = targets[b]["masks"][:, top : top + crop_h, left : left + crop_w]
        masks, labels = _filter_small(masks.bool(), targets[b]["labels"])
        out_targets.append(_rebuild_target(masks, labels))
    return out_imgs, out_targets


@torch.no_grad()
def resolution_jitter_batch(
    imgs: torch.Tensor,
    targets: List[Dict[str, torch.Tensor]],
    resolutions: Sequence[Tuple[int, int]] = ((384, 768), (416, 832), (448, 896)),
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """Port of CUPS ResolutionJitter: resize the whole batch to one randomly
    chosen resolution (bilinear image, nearest masks)."""
    resolution = random.choice(list(resolutions))
    out_imgs = F.interpolate(imgs.float(), size=tuple(resolution), mode="bilinear")
    out_targets: List[Dict[str, torch.Tensor]] = []
    for b in range(imgs.shape[0]):
        masks = targets[b]["masks"]
        if masks.shape[0] > 0:
            masks = F.interpolate(
                masks[None].float(), size=tuple(resolution), mode="nearest"
            )[0].bool()
        else:
            masks = torch.zeros(
                (0, resolution[0], resolution[1]), dtype=torch.bool, device=imgs.device
            )
        masks, labels = _filter_small(masks, targets[b]["labels"])
        out_targets.append(_rebuild_target(masks, labels))
    return out_imgs, out_targets
