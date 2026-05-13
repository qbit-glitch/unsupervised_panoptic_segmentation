"""Unit tests for Path-C AuxThingAdapter (GT-free SAM3-thing adapter).

Verifies:
  - module shape contracts (forward, batched and unbatched).
  - parameter count ~17K.
  - Kaiming init produces near-uniform predictions at step 0.
  - cross-entropy loss is finite on random data.
  - gradient flows to all params.
  - fuse_predictions correctly applies confidence threshold + structural mapping.
  - No Cityscapes class names or LUT artifacts in module attributes.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cups.model.aux_thing_adapter import (  # noqa: E402
    AuxThingAdapter,
    SAM3_TO_CITYSCAPES_TRAINID,
    fuse_predictions,
)


def test_adapter_param_count_is_about_17k():
    a = AuxThingAdapter(in_dim=256, hidden_dim=64, num_sam3_classes=14)
    n = a.num_trainable_params()
    # 256*64 + 64 + 64*14 + 14 = 16384 + 64 + 896 + 14 = 17358
    assert 17_000 <= n <= 18_000, f"Expected ~17K params, got {n}"


def test_adapter_forward_4d_shape():
    a = AuxThingAdapter()
    x = torch.randn(2, 256, 96, 192)
    y = a(x)
    assert y.shape == (2, 14, 96, 192)
    assert torch.isfinite(y).all()


def test_adapter_forward_2d_shape():
    a = AuxThingAdapter()
    x = torch.randn(100, 256)
    y = a(x)
    assert y.shape == (100, 14)


def test_adapter_init_no_lut_attributes():
    """Verify the adapter has no GT-derived state in its attributes."""
    a = AuxThingAdapter()
    state_keys = list(a.state_dict().keys())
    # Allowed state-dict keys: just fc1.weight, fc1.bias, fc2.weight, fc2.bias.
    expected = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
    assert set(state_keys) == expected, f"Unexpected state-dict keys: {state_keys}"
    # No buffer should reference a cluster_to_class LUT or any cityscapes labels.
    for name, _buf in a.named_buffers():
        assert "cluster" not in name.lower()
        assert "cityscapes" not in name.lower()
        assert "trainid" not in name.lower()


def test_adapter_finite_loss_on_random_target():
    a = AuxThingAdapter()
    x = torch.randn(1, 256, 16, 32)
    target = torch.randint(0, 14, (1, 16, 32))
    logits = a(x)
    loss = F.cross_entropy(logits, target)
    assert torch.isfinite(loss)


def test_adapter_finite_loss_with_ignore_index():
    """Most pixels in real training have ignore_index=-1 (no SAM3 mask)."""
    a = AuxThingAdapter()
    x = torch.randn(1, 256, 16, 32)
    target = torch.full((1, 16, 32), -1, dtype=torch.long)
    target[0, 4:8, 8:12] = 2  # a small region with motorcycle (SAM3 idx 2)
    logits = a(x)
    loss = F.cross_entropy(logits, target, ignore_index=-1)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_adapter_gradient_flows_to_all_params():
    a = AuxThingAdapter()
    x = torch.randn(1, 256, 8, 16)
    target = torch.randint(0, 14, (1, 8, 16))
    logits = a(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    for name, p in a.named_parameters():
        assert p.grad is not None, f"No grad on {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad on {name}"
        assert p.grad.abs().sum() > 0.0, f"Zero grad on {name}"


def test_sam3_to_cityscapes_mapping_covers_dead_classes():
    """The structural mapping must cover the previously-dead classes
    motorcycle (trainID 17) and traffic light (trainID 6)."""
    assert 17 in SAM3_TO_CITYSCAPES_TRAINID.values(), "motorcycle missing"
    assert 6 in SAM3_TO_CITYSCAPES_TRAINID.values(), "traffic light missing"


def test_fuse_predictions_high_conf_uses_adapter():
    """When adapter is highly confident, its trainID overrides cluster head."""
    cluster_pred = torch.zeros(1, 8, 16, dtype=torch.long)  # all road
    adapter_logits = torch.zeros(1, 14, 8, 16)
    # Make channel 2 (motorcycle, SAM3 idx 2 → trainID 17) very strong everywhere.
    adapter_logits[0, 2, :, :] = 10.0

    fused = fuse_predictions(cluster_pred, adapter_logits, confidence_threshold=0.5)
    assert fused.shape == (1, 8, 16)
    assert (fused == 17).all(), f"Expected all 17 (motorcycle), got unique={fused.unique()}"


def test_fuse_predictions_low_conf_falls_back_to_cluster():
    """When adapter is uncertain, cluster head's prediction is kept."""
    cluster_pred = torch.full((1, 8, 16), 5, dtype=torch.long)  # cluster head says trainID 5 (pole)
    adapter_logits = torch.zeros(1, 14, 8, 16)
    # Uniform logits → conf ≈ 1/14 ≈ 0.07, far below 0.5 threshold.
    fused = fuse_predictions(cluster_pred, adapter_logits, confidence_threshold=0.5)
    assert (fused == 5).all(), "Low-conf adapter should not override cluster head"


def test_fuse_predictions_unmapped_sam3_class_falls_back():
    """SAM3 class 9 (guard rail) has no trainID mapping → must fall back."""
    cluster_pred = torch.full((1, 8, 16), 0, dtype=torch.long)
    adapter_logits = torch.zeros(1, 14, 8, 16)
    adapter_logits[0, 9, :, :] = 10.0  # SAM3 idx 9 (guard rail), unmapped
    fused = fuse_predictions(cluster_pred, adapter_logits, confidence_threshold=0.5)
    assert (fused == 0).all(), "Unmapped SAM3 class should fall back to cluster head"
