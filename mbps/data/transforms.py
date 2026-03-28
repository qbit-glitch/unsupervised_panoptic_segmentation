"""JAX-native data augmentation transforms for MBPS.

All transforms operate on JAX arrays and use jax.random for stochasticity.
"""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def normalize(
    image: jnp.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> jnp.ndarray:
    """Normalize image with ImageNet statistics.

    Args:
        image: Input image of shape (H, W, 3), values in [0, 1].
        mean: Per-channel mean.
        std: Per-channel std.

    Returns:
        Normalized image.
    """
    mean = jnp.array(mean, dtype=image.dtype)
    std = jnp.array(std, dtype=image.dtype)
    return (image - mean) / std


def denormalize(
    image: jnp.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> jnp.ndarray:
    """Reverse ImageNet normalization.

    Args:
        image: Normalized image of shape (H, W, 3).
        mean: Per-channel mean.
        std: Per-channel std.

    Returns:
        Denormalized image with values in [0, 1].
    """
    mean = jnp.array(mean, dtype=image.dtype)
    std = jnp.array(std, dtype=image.dtype)
    return image * std + mean


def random_crop(
    key: jax.Array,
    image: jnp.ndarray,
    crop_size: Tuple[int, int],
) -> jnp.ndarray:
    """Random crop of an image.

    Args:
        key: PRNG key.
        image: Input image of shape (H, W, C).
        crop_size: Target crop size (crop_h, crop_w).

    Returns:
        Cropped image of shape (crop_h, crop_w, C).
    """
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size

    # If image is smaller than crop, pad
    pad_h = jnp.maximum(crop_h - h, 0)
    pad_w = jnp.maximum(crop_w - w, 0)

    if image.ndim == 3:
        image = jnp.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    else:
        image = jnp.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")

    h, w = image.shape[:2]
    key1, key2 = jax.random.split(key)
    top = jax.random.randint(key1, (), 0, h - crop_h + 1)
    left = jax.random.randint(key2, (), 0, w - crop_w + 1)

    return jax.lax.dynamic_slice(
        image,
        (top, left, 0) if image.ndim == 3 else (top, left),
        (crop_h, crop_w, image.shape[2]) if image.ndim == 3 else (crop_h, crop_w),
    )


def random_horizontal_flip(
    key: jax.Array,
    image: jnp.ndarray,
    prob: float = 0.5,
) -> jnp.ndarray:
    """Random horizontal flip.

    Args:
        key: PRNG key.
        image: Input image of shape (H, W, C) or (H, W).
        prob: Flip probability.

    Returns:
        Possibly flipped image.
    """
    do_flip = jax.random.uniform(key) < prob
    return jnp.where(do_flip, jnp.flip(image, axis=1), image)


def color_jitter(
    key: jax.Array,
    image: jnp.ndarray,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
) -> jnp.ndarray:
    """Random color jittering.

    Args:
        key: PRNG key.
        image: Input image of shape (H, W, 3), values in [0, 1].
        brightness: Brightness jitter range.
        contrast: Contrast jitter range.
        saturation: Saturation jitter range.
        hue: Hue jitter range.

    Returns:
        Color-jittered image, clipped to [0, 1].
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Brightness
    b_factor = 1.0 + jax.random.uniform(k1, (), minval=-brightness, maxval=brightness)
    image = image * b_factor

    # Contrast
    c_factor = 1.0 + jax.random.uniform(k2, (), minval=-contrast, maxval=contrast)
    mean_val = jnp.mean(image, axis=(0, 1), keepdims=True)
    image = c_factor * (image - mean_val) + mean_val

    # Saturation
    s_factor = 1.0 + jax.random.uniform(k3, (), minval=-saturation, maxval=saturation)
    gray = jnp.mean(image, axis=-1, keepdims=True)
    image = s_factor * (image - gray) + gray

    return jnp.clip(image, 0.0, 1.0)


def resize_image(
    image: jnp.ndarray,
    target_size: Tuple[int, int],
) -> jnp.ndarray:
    """Resize image using bilinear interpolation.

    Args:
        image: Input image of shape (H, W, C).
        target_size: Target size (H, W).

    Returns:
        Resized image.
    """
    return jax.image.resize(
        image,
        (*target_size, image.shape[-1]) if image.ndim == 3 else target_size,
        method="bilinear",
    )


def resize_label(
    label: jnp.ndarray,
    target_size: Tuple[int, int],
) -> jnp.ndarray:
    """Resize label using nearest neighbor interpolation.

    Args:
        label: Input label of shape (H, W).
        target_size: Target size (H, W).

    Returns:
        Resized label.
    """
    return jax.image.resize(label, target_size, method="nearest").astype(jnp.int32)


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
        self, sample: Dict[str, np.ndarray], key: jax.Array | None = None
    ) -> Dict[str, jnp.ndarray]:
        """Apply training transforms to a sample.

        Args:
            sample: Dictionary with 'image', 'depth', and optional labels.
            key: PRNG key. If None, uses a default seed.

        Returns:
            Transformed sample with JAX arrays.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        k1, k2, k3 = jax.random.split(key, 3)
        result = {}

        image = jnp.array(sample["image"], dtype=jnp.float32)
        depth = jnp.array(sample["depth"], dtype=jnp.float32)

        # Random crop — apply same crop to image, depth, and labels
        h, w = image.shape[:2]
        crop_h, crop_w = self.crop_size
        pad_h = jnp.maximum(crop_h - h, 0)
        pad_w = jnp.maximum(crop_w - w, 0)

        top = jax.random.randint(k1, (), 0, jnp.maximum(h + pad_h - crop_h + 1, 1))
        left = jax.random.randint(
            jax.random.split(k1)[1], (), 0, jnp.maximum(w + pad_w - crop_w + 1, 1)
        )

        image = random_crop(k1, image, self.crop_size)
        depth = random_crop(k1, depth, self.crop_size)

        # Random horizontal flip
        image = random_horizontal_flip(k2, image)
        depth = random_horizontal_flip(k2, depth)

        # Color jitter (image only)
        image = color_jitter(k3, image)

        # Normalize image
        image = normalize(image, self.mean, self.std)

        result["image"] = image
        result["depth"] = depth
        result["image_id"] = sample.get("image_id", "")

        # Apply same spatial transforms to labels
        for label_key in ("semantic_label", "instance_label"):
            if label_key in sample:
                label = jnp.array(sample[label_key], dtype=jnp.int32)
                label = random_crop(k1, label, self.crop_size)
                label = random_horizontal_flip(k2, label)
                result[label_key] = label

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
        self, sample: Dict[str, np.ndarray], key: jax.Array | None = None
    ) -> Dict[str, jnp.ndarray]:
        """Apply evaluation transforms.

        Args:
            sample: Dictionary with 'image', 'depth', and optional labels.
            key: PRNG key (unused for eval).

        Returns:
            Normalized sample with JAX arrays.
        """
        result = {}
        image = jnp.array(sample["image"], dtype=jnp.float32)
        result["image"] = normalize(image, self.mean, self.std)
        result["depth"] = jnp.array(sample["depth"], dtype=jnp.float32)
        result["image_id"] = sample.get("image_id", "")

        for label_key in ("semantic_label", "instance_label"):
            if label_key in sample:
                result[label_key] = jnp.array(sample[label_key], dtype=jnp.int32)

        return result
