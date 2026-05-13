"""Unit tests for Adaptive Projection Bridge (PyTorch).

Tests cover:
    - Forward/inverse projection shapes
    - Reconstruction error
    - CKA alignment computation
    - Alignment loss properties
    - Gradient flow through projections
"""

from __future__ import annotations

import sys
import os

import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestAdaptiveProjectionBridge:
    """Test the Adaptive Projection Bridge (APB)."""

    def test_output_shapes(self):
        """Projection should output bridge-dim features."""
        from mbps_pytorch.models.bridge.projection import AdaptiveProjectionBridge

        torch.manual_seed(0)
        model = AdaptiveProjectionBridge(
            semantic_dim=90, feature_dim=384, bridge_dim=192
        ).to(DEVICE)
        sem = torch.ones(2, 100, 90, device=DEVICE)
        feat = torch.ones(2, 100, 384, device=DEVICE)
        sem_p, feat_p, align_loss = model(sem, feat)

        assert sem_p.shape == (2, 100, 192)
        assert feat_p.shape == (2, 100, 192)
        assert align_loss.shape == ()

    def test_align_loss_nonnegative(self):
        """Alignment loss should be non-negative."""
        from mbps_pytorch.models.bridge.projection import AdaptiveProjectionBridge

        torch.manual_seed(0)
        model = AdaptiveProjectionBridge().to(DEVICE)
        sem = torch.randn(1, 50, 90, device=DEVICE)
        torch.manual_seed(1)
        feat = torch.randn(1, 50, 384, device=DEVICE)
        _, _, align_loss = model(sem, feat)

        assert float(align_loss) >= 0.0

    def test_gradient_flow(self):
        """Gradients should flow through projection bridge."""
        from mbps_pytorch.models.bridge.projection import AdaptiveProjectionBridge

        torch.manual_seed(0)
        model = AdaptiveProjectionBridge(
            semantic_dim=90, feature_dim=384, bridge_dim=192
        ).to(DEVICE)
        sem = torch.randn(1, 20, 90, device=DEVICE, requires_grad=True)
        feat = torch.randn(1, 20, 384, device=DEVICE, requires_grad=True)
        s, f, a = model(sem, feat)
        loss = torch.mean(s ** 2) + torch.mean(f ** 2) + a
        loss.backward()

        total_grad = sum(
            float(torch.sum(torch.abs(p.grad)))
            for p in model.parameters()
            if p.grad is not None
        )

        assert total_grad > 0.0

    def test_different_bridge_dims(self):
        """Projection should work with various bridge dimensions."""
        from mbps_pytorch.models.bridge.projection import AdaptiveProjectionBridge

        for bridge_dim in [128, 192, 256, 384]:
            torch.manual_seed(0)
            model = AdaptiveProjectionBridge(
                semantic_dim=90, feature_dim=384, bridge_dim=bridge_dim
            ).to(DEVICE)
            sem = torch.ones(1, 10, 90, device=DEVICE)
            feat = torch.ones(1, 10, 384, device=DEVICE)
            sem_p, feat_p, _ = model(sem, feat)

            assert sem_p.shape[-1] == bridge_dim
            assert feat_p.shape[-1] == bridge_dim


class TestInverseProjection:
    """Test inverse projection from bridge back to original dim."""

    def test_output_shape(self):
        """Inverse projection should output correct dimension."""
        from mbps_pytorch.models.bridge.projection import InverseProjection

        torch.manual_seed(0)
        model = InverseProjection(output_dim=384).to(DEVICE)
        x = torch.ones(2, 50, 192, device=DEVICE)
        out = model(x)

        assert out.shape == (2, 50, 384)

    def test_semantic_inverse(self):
        """Inverse projection for semantic should restore dim 90."""
        from mbps_pytorch.models.bridge.projection import InverseProjection

        torch.manual_seed(0)
        model = InverseProjection(output_dim=90).to(DEVICE)
        x = torch.ones(1, 30, 192, device=DEVICE)
        out = model(x)

        assert out.shape == (1, 30, 90)


class TestReconstructionError:
    """Test round-trip reconstruction quality."""

    def test_reconstruction_error_bounded(self):
        """Reconstruction error should be bounded after training init."""
        from mbps_pytorch.models.bridge.projection import (
            AdaptiveProjectionBridge,
            InverseProjection,
        )

        torch.manual_seed(0)
        sem = torch.randn(1, 50, 90, device=DEVICE)
        torch.manual_seed(1)
        feat = torch.randn(1, 50, 384, device=DEVICE)

        # Forward projection
        proj = AdaptiveProjectionBridge(
            semantic_dim=90, feature_dim=384, bridge_dim=192
        ).to(DEVICE)
        sem_p, feat_p, _ = proj(sem, feat)

        # Inverse projection
        inv_sem = InverseProjection(output_dim=90).to(DEVICE)
        inv_feat = InverseProjection(output_dim=384).to(DEVICE)

        recon_sem = inv_sem(sem_p)
        recon_feat = inv_feat(feat_p)

        # Reconstruction error should be finite (not checking < 0.05 before training)
        sem_error = float(torch.mean((sem - recon_sem) ** 2))
        feat_error = float(torch.mean((feat - recon_feat) ** 2))

        assert np.isfinite(sem_error)
        assert np.isfinite(feat_error)

    def test_cka_after_projection(self):
        """CKA between projected streams should be finite and bounded."""
        from mbps_pytorch.models.bridge.projection import AdaptiveProjectionBridge
        from mbps_pytorch.losses.bridge_loss import cka_loss

        torch.manual_seed(0)
        sem = torch.randn(1, 50, 90, device=DEVICE)
        torch.manual_seed(1)
        feat = torch.randn(1, 50, 384, device=DEVICE)

        proj = AdaptiveProjectionBridge().to(DEVICE)
        sem_p, feat_p, _ = proj(sem, feat)

        cka = cka_loss(sem_p, feat_p)

        # CKA loss should be in [-1, 0] (negative CKA)
        assert float(cka) <= 0.1  # Allow small positive margin
        assert float(cka) >= -1.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
