"""Tests for Mask2FormerPanoptic._stack_images normalization.

DINOv3 (official repo, refs/dinov3/README.md) expects input pre-processed with
ImageNet mean/std. The backbone wrapper (refs/cups/cups/model/backbone_dinov3_vit.py)
does NOT apply normalization internally, so the meta-arch must do it.

Without normalization, raw [0, 1] inputs sit ~(0.5, 0.2) per channel vs the
distribution DINOv3 was pretrained on (mean subtracted, divided by std).
That shifts every patch token by a large constant, pushing the model into
a region of feature space where the decoder has never seen gradients — the
observed loss plateau at ~70 for 5000 steps.
"""
from __future__ import annotations

import torch

from cups.model.modeling.meta_arch.mask2former_panoptic import Mask2FormerPanoptic

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class _StubMetaArch(Mask2FormerPanoptic):
    def __init__(self) -> None:  # noqa: D401
        torch.nn.Module.__init__(self)
        self.num_stuff_classes = 12
        self.num_thing_classes = 8
        self.num_classes = 20
        self._device = torch.device("cpu")
        # Mirror the buffers the real __init__ registers so _stack_images works.
        self.register_buffer(
            "_pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    @property
    def device(self) -> torch.device:
        return self._device


def test_uint8_is_imagenet_normalized() -> None:
    """uint8 image in [0, 255] should be /255 then (x - mean) / std."""
    model = _StubMetaArch()
    img = torch.full((3, 8, 16), 128, dtype=torch.uint8)
    batch = [{"image": img}]
    stacked = model._stack_images(batch)

    expected = (img.float() / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    assert torch.allclose(stacked[0], expected, atol=1e-5), \
        f"expected normalized={expected[:, 0, 0]}, got={stacked[0, :, 0, 0]}"


def test_float_input_is_normalized_assuming_0_1_range() -> None:
    """Float images should already be in [0, 1] then normalized."""
    model = _StubMetaArch()
    img = torch.full((3, 8, 16), 0.5, dtype=torch.float32)
    batch = [{"image": img}]
    stacked = model._stack_images(batch)

    expected = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    assert torch.allclose(stacked[0], expected, atol=1e-5), \
        f"expected normalized={expected[:, 0, 0]}, got={stacked[0, :, 0, 0]}"


def test_padding_region_stays_zero() -> None:
    """Padded region (beyond smallest image) must remain 0 after normalization.

    Otherwise the pad bleeds into patch tokens and corrupts the whole feature map
    for the smaller sample.
    """
    model = _StubMetaArch()
    small = torch.full((3, 4, 8), 128, dtype=torch.uint8)
    big = torch.full((3, 8, 16), 128, dtype=torch.uint8)
    batch = [{"image": small}, {"image": big}]
    stacked = model._stack_images(batch)

    # Rows 4-7, cols 0-15 of sample 0 are padding.
    assert torch.all(stacked[0, :, 4:, :] == 0.0), \
        "padding region of small sample must remain zero"


def test_output_distribution_roughly_centred() -> None:
    """A uniform image of all-127 (near 0.5) should normalize to near-zero mean per channel."""
    model = _StubMetaArch()
    img = torch.full((3, 16, 16), 127, dtype=torch.uint8)
    batch = [{"image": img}]
    stacked = model._stack_images(batch)

    per_channel_mean = stacked[0].mean(dim=(-1, -2))
    # 127/255 ~= 0.4980, so (0.4980 - mean_i) / std_i
    expected_mean = torch.tensor([0.0569, 0.1877, 0.4091])
    assert torch.allclose(per_channel_mean, expected_mean, atol=1e-3), \
        f"expected per-channel means {expected_mean}, got {per_channel_mean}"
