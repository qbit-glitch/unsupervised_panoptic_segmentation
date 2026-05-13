"""Tests for FRACAL post-hoc logit calibration.

FRACAL (Alexandridis et al., CVPR 2025) calibrates per-class logits using the
fractal dimension of per-class spatial occupancy. Tail classes typically occupy
spatially compact regions (low D), so they receive a positive logit shift.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch


class TestBoxCountingFractalDimension:
    """Box-counting fractal dimension on known synthetic shapes."""

    def test_returns_two_for_solid_square(self) -> None:
        """A solid filled square has fractal dimension ~ 2.0 (within tolerance)."""
        from mbps_pytorch.stage4.fracal_calibration import box_counting_dimension

        h = w = 128
        mask = np.zeros((h, w), dtype=bool)
        mask[16:112, 16:112] = True

        d = box_counting_dimension(mask)
        assert 1.85 < d < 2.05, f"solid square should give D ~ 2.0, got {d}"

    def test_returns_about_one_for_horizontal_line(self) -> None:
        """A 1-pixel-thick horizontal line has fractal dimension ~ 1.0."""
        from mbps_pytorch.stage4.fracal_calibration import box_counting_dimension

        mask = np.zeros((128, 128), dtype=bool)
        mask[64, :] = True

        d = box_counting_dimension(mask)
        assert 0.85 < d < 1.20, f"line should give D ~ 1.0, got {d}"

    def test_returns_zero_for_empty_mask(self) -> None:
        """An empty mask returns 0.0 (handled gracefully, no crash)."""
        from mbps_pytorch.stage4.fracal_calibration import box_counting_dimension

        mask = np.zeros((64, 64), dtype=bool)
        d = box_counting_dimension(mask)
        assert d == 0.0

    def test_returns_zero_for_single_pixel(self) -> None:
        """A single-pixel mask is degenerate; return 0.0."""
        from mbps_pytorch.stage4.fracal_calibration import box_counting_dimension

        mask = np.zeros((64, 64), dtype=bool)
        mask[32, 32] = True
        d = box_counting_dimension(mask)
        assert math.isfinite(d)
        # Single pixel ~ point ~ dim 0; allow up to small positive
        assert 0.0 <= d < 0.5


class TestPerClassFractalDim:
    """Computing per-class fractal dimensions from a stack of label maps."""

    def test_returns_one_value_per_class(self) -> None:
        """Output shape is (num_classes,)."""
        from mbps_pytorch.stage4.fracal_calibration import per_class_fractal_dim

        # 3 images, 4 classes (0..3)
        torch.manual_seed(0)
        labels = torch.randint(0, 4, (3, 64, 64))
        d = per_class_fractal_dim(labels, num_classes=4)

        assert d.shape == (4,)
        assert torch.all(torch.isfinite(d))

    def test_class_with_no_pixels_gets_zero(self) -> None:
        """Classes absent from all images get D = 0.0."""
        from mbps_pytorch.stage4.fracal_calibration import per_class_fractal_dim

        # Class 3 never appears
        labels = torch.zeros((2, 32, 32), dtype=torch.long)
        labels[0, 0:16, :] = 1
        labels[1, :, 0:16] = 2

        d = per_class_fractal_dim(labels, num_classes=4)
        assert d[3].item() == 0.0


class TestFracalCalibrate:
    """Apply FRACAL calibration to logits."""

    def test_zero_lambda_returns_input_unchanged(self) -> None:
        """λ = 0 means no calibration; logits pass through."""
        from mbps_pytorch.stage4.fracal_calibration import fracal_calibrate

        torch.manual_seed(0)
        logits = torch.randn(2, 5, 32, 32)
        per_class_d = torch.tensor([1.5, 2.0, 1.0, 1.8, 0.5])

        out = fracal_calibrate(logits, per_class_d, lam=0.0)
        assert torch.allclose(out, logits)

    def test_calibration_shape_matches_input(self) -> None:
        """Output has same shape as input logits."""
        from mbps_pytorch.stage4.fracal_calibration import fracal_calibrate

        logits = torch.randn(2, 5, 32, 32)
        per_class_d = torch.tensor([1.5, 2.0, 1.0, 1.8, 0.5])
        out = fracal_calibrate(logits, per_class_d, lam=1.0)
        assert out.shape == logits.shape

    def test_low_fractal_dim_class_is_boosted(self) -> None:
        """Class with the smallest fractal dim gets the largest logit shift."""
        from mbps_pytorch.stage4.fracal_calibration import fracal_calibrate

        # 5 classes; class 4 has lowest D
        per_class_d = torch.tensor([2.0, 2.0, 2.0, 2.0, 0.5])

        logits = torch.zeros(1, 5, 4, 4)
        out = fracal_calibrate(logits, per_class_d, lam=1.0)

        # Per-class shift = lam * (mean_D - D_c)
        # mean_D = (2.0*4 + 0.5)/5 = 1.7
        # shift_c4 = 1.0 * (1.7 - 0.5) = 1.2 (largest)
        # shift_c0..3 = 1.0 * (1.7 - 2.0) = -0.3
        expected_shift_4 = 1.7 - 0.5
        expected_shift_0 = 1.7 - 2.0

        assert math.isclose(out[0, 4, 0, 0].item(), expected_shift_4, abs_tol=1e-6)
        assert math.isclose(out[0, 0, 0, 0].item(), expected_shift_0, abs_tol=1e-6)

    def test_calibration_preserves_argmax_when_classes_have_same_D(self) -> None:
        """If all classes have identical D, calibration applies zero shift everywhere."""
        from mbps_pytorch.stage4.fracal_calibration import fracal_calibrate

        torch.manual_seed(0)
        logits = torch.randn(2, 5, 16, 16)
        per_class_d = torch.full((5,), 1.5)
        out = fracal_calibrate(logits, per_class_d, lam=1.0)
        assert torch.allclose(out, logits)
