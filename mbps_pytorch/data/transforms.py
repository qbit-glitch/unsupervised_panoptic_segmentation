"""PyTorch-native data augmentation transforms for MBPS.

All transforms operate on torch tensors and use ``torch`` / standard
Python random for stochasticity.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def normalize(
    image: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Normalize image with ImageNet statistics.

    Args:
        image: Input image of shape (H, W, 3), values in [0, 1].
        mean: Per-channel mean.
        std: Per-channel std.

    Returns:
        Normalized image.
    """
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device)
    return (image - mean_t) / std_t


def denormalize(
    image: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Reverse ImageNet normalization.

    Args:
        image: Normalized image of shape (H, W, 3).
        mean: Per-channel mean.
        std: Per-channel std.

    Returns:
        Denormalized image with values in [0, 1].
    """
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device)
    return image * std_t + mean_t


def random_crop(
    image: torch.Tensor,
    crop_size: Tuple[int, int],
    top: int,
    left: int,
) -> torch.Tensor:
    """Deterministic crop at (top, left) with prior padding if needed.

    The caller is responsible for sampling ``top`` and ``left`` so that
    all arrays in a sample share the same crop coordinates.

    Args:
        image: Input image of shape (H, W, C) or (H, W).
        crop_size: Target crop size (crop_h, crop_w).
        top: Top-left row offset after padding.
        left: Top-left column offset after padding.

    Returns:
        Cropped image of shape (crop_h, crop_w, C) or (crop_h, crop_w).
    """
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size

    # If image is smaller than crop, pad with reflect
    pad_h = max(crop_h - h, 0)
    pad_w = max(crop_w - w, 0)

    if pad_h > 0 or pad_w > 0:
        if image.ndim == 3:
            # F.pad expects (N, C, H, W) or (C, H, W); we use (H, W, C) here
            # so we pad manually via numpy-style padding
            image = torch.nn.functional.pad(
                image.permute(2, 0, 1).unsqueeze(0),
                (0, pad_w, 0, pad_h),
                mode="reflect",
            ).squeeze(0).permute(1, 2, 0)
        else:
            image = torch.nn.functional.pad(
                image.unsqueeze(0).unsqueeze(0),
                (0, pad_w, 0, pad_h),
                mode="reflect",
            ).squeeze(0).squeeze(0)

    return image[top: top + crop_h, left: left + crop_w]


def random_horizontal_flip(
    image: torch.Tensor,
    do_flip: bool,
) -> torch.Tensor:
    """Deterministic horizontal flip.

    Args:
        image: Input image of shape (H, W, C) or (H, W).
        do_flip: Whether to flip.

    Returns:
        Possibly flipped image.
    """
    if do_flip:
        return torch.flip(image, dims=[1])
    return image


def color_jitter(
    image: torch.Tensor,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
) -> torch.Tensor:
    """Random color jittering.

    Args:
        image: Input image of shape (H, W, 3), values in [0, 1].
        brightness: Brightness jitter range.
        contrast: Contrast jitter range.
        saturation: Saturation jitter range.
        hue: Hue jitter range (unused for simplicity, kept for API compat).

    Returns:
        Color-jittered image, clipped to [0, 1].
    """
    # Brightness
    b_factor = 1.0 + random.uniform(-brightness, brightness)
    image = image * b_factor

    # Contrast
    c_factor = 1.0 + random.uniform(-contrast, contrast)
    mean_val = image.mean(dim=(0, 1), keepdim=True)
    image = c_factor * (image - mean_val) + mean_val

    # Saturation
    s_factor = 1.0 + random.uniform(-saturation, saturation)
    gray = image.mean(dim=-1, keepdim=True)
    image = s_factor * (image - gray) + gray

    return image.clamp(0.0, 1.0)


def resize_image(
    image: torch.Tensor,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """Resize image using bilinear interpolation.

    Args:
        image: Input image of shape (H, W, C).
        target_size: Target size (H, W).

    Returns:
        Resized image of shape (target_H, target_W, C).
    """
    # F.interpolate expects (N, C, H, W)
    img = image.permute(2, 0, 1).unsqueeze(0)
    img = F.interpolate(img, size=target_size, mode="bilinear", align_corners=False)
    return img.squeeze(0).permute(1, 2, 0)


def resize_label(
    label: torch.Tensor,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """Resize label using nearest neighbor interpolation.

    Args:
        label: Input label of shape (H, W).
        target_size: Target size (H, W).

    Returns:
        Resized label of shape (target_H, target_W).
    """
    lbl = label.unsqueeze(0).unsqueeze(0).float()
    lbl = F.interpolate(lbl, size=target_size, mode="nearest")
    return lbl.squeeze(0).squeeze(0).to(torch.int32)


class TrainTransform:
    """Composite training transform.

    Applies random crop, horizontal flip, color jitter, and normalization.

    Args:
        crop_size: Crop dimensions (H, W).
        mean: ImageNet mean.
        std: ImageNet std.
    """

    def __init__(
        self,
        crop_size: Tuple[int, int] = (512, 512),
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.crop_size = crop_size
        self.mean = mean
        self.std = std

    def __call__(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply training transforms to a sample.

        Operates on numpy arrays (as loaded by the dataset) and returns
        numpy arrays.  The dataset ``__getitem__`` converts to torch
        tensors afterwards.

        Args:
            sample: Dictionary with 'image', 'depth', and optional labels.

        Returns:
            Transformed sample with numpy arrays.
        """
        result: Dict[str, Any] = {}

        image = torch.tensor(sample["image"], dtype=torch.float32)
        depth = torch.tensor(sample["depth"], dtype=torch.float32)

        # Determine crop coordinates (shared for image, depth, and labels)
        h, w = image.shape[:2]
        crop_h, crop_w = self.crop_size
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        top = random.randint(0, max(h + pad_h - crop_h, 0))
        left = random.randint(0, max(w + pad_w - crop_w, 0))

        # Random crop
        image = random_crop(image, self.crop_size, top, left)
        depth = random_crop(depth, self.crop_size, top, left)

        # Random horizontal flip (same decision for all)
        do_flip = random.random() < 0.5
        image = random_horizontal_flip(image, do_flip)
        depth = random_horizontal_flip(depth, do_flip)

        # Color jitter (image only)
        image = color_jitter(image)

        # Normalize image
        image = normalize(image, self.mean, self.std)

        result["image"] = image.numpy()
        result["depth"] = depth.numpy()
        result["image_id"] = sample.get("image_id", "")

        # Apply same spatial transforms to labels
        for label_key in ("semantic_label", "instance_label"):
            if label_key in sample:
                label = torch.tensor(sample[label_key], dtype=torch.int32)
                label = random_crop(label, self.crop_size, top, left)
                label = random_horizontal_flip(label, do_flip)
                result[label_key] = label.numpy()

        return result


class EvalTransform:
    """Evaluation transform (normalize only, no augmentation).

    Args:
        mean: ImageNet mean.
        std: ImageNet std.
    """

    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.mean = mean
        self.std = std

    def __call__(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply evaluation transforms.

        Args:
            sample: Dictionary with 'image', 'depth', and optional labels.

        Returns:
            Normalized sample with numpy arrays.
        """
        result: Dict[str, Any] = {}

        image = torch.tensor(sample["image"], dtype=torch.float32)
        result["image"] = normalize(image, self.mean, self.std).numpy()
        result["depth"] = np.array(sample["depth"], dtype=np.float32)
        result["image_id"] = sample.get("image_id", "")

        for label_key in ("semantic_label", "instance_label"):
            if label_key in sample:
                result[label_key] = np.array(sample[label_key], dtype=np.int32)

        return result
