"""Unit tests for MBPS core modules (PyTorch).

Tests cover:
    - Model components (shape checks, forward pass)
    - Loss functions (value ranges, gradients)
    - Data transforms
    - Bridge modules
    - Training curriculum
"""

from __future__ import annotations

import sys
import os

import pytest
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestBackbone:
    """Test DINO ViT-S/8 backbone."""

    def test_output_shape(self):
        """Backbone should output (B, N+1, 384) tokens."""
        from mbps_pytorch.models.backbone.dino_vits8 import DINOViTS8

        model = DINOViTS8().to(DEVICE)
        x = torch.ones((1, 3, 224, 224), device=DEVICE)
        out = model(x)

        # ViT-S/8 on 224x224: (224/8)^2 = 784 patches + 1 cls = 785
        assert out.shape == (1, 785, 384)

    def test_spatial_features(self):
        """Spatial features should have (B, N, 384) without CLS."""
        from mbps_pytorch.models.backbone.dino_vits8 import DINOViTS8

        model = DINOViTS8().to(DEVICE)
        x = torch.ones((1, 3, 224, 224), device=DEVICE)
        out = model.get_spatial_features(x)

        assert out.shape == (1, 784, 384)

    def test_frozen_gradients(self):
        """Backbone should produce zero gradients (frozen)."""
        from mbps_pytorch.models.backbone.dino_vits8 import DINOViTS8

        model = DINOViTS8(freeze=True).to(DEVICE)
        x = torch.ones((1, 3, 64, 64), device=DEVICE)

        for p in model.parameters():
            assert not p.requires_grad


class TestSemanticHead:
    """Test DepthG semantic head."""

    def test_output_shape(self):
        """DepthG should output (B, N, 90) semantic codes."""
        from mbps_pytorch.models.semantic.depthg_head import DepthGHead

        model = DepthGHead(input_dim=384, code_dim=90).to(DEVICE)
        x = torch.ones((2, 100, 384), device=DEVICE)
        out = model(x)

        assert out.shape == (2, 100, 90)


class TestInstanceHead:
    """Test instance segmentation head."""

    def test_output_shapes(self):
        """Instance head should output masks and scores."""
        from mbps_pytorch.models.instance.cascade_mask_rcnn import InstanceHead

        model = InstanceHead(
            max_instances=50, input_dim=384, hidden_dim=128
        ).to(DEVICE)
        x = torch.ones((2, 100, 384), device=DEVICE)
        masks, scores = model(x)

        assert masks.shape == (2, 50, 100)
        assert scores.shape == (2, 50)

    def test_scores_range(self):
        """Instance scores should be in [0, 1]."""
        from mbps_pytorch.models.instance.cascade_mask_rcnn import InstanceHead

        torch.manual_seed(42)
        model = InstanceHead(
            max_instances=20, input_dim=384, hidden_dim=64
        ).to(DEVICE)
        x = torch.randn((1, 50, 384), device=DEVICE)
        _, scores = model(x)

        assert torch.all(scores >= 0)
        assert torch.all(scores <= 1)


class TestProjectionBridge:
    """Test Adaptive Projection Bridge."""

    def test_output_shapes(self):
        """Projection should output bridge-dim features."""
        from mbps_pytorch.models.bridge.projection import AdaptiveProjectionBridge

        model = AdaptiveProjectionBridge(
            semantic_dim=90, feature_dim=384, bridge_dim=192
        ).to(DEVICE)
        sem = torch.ones((2, 100, 90), device=DEVICE)
        feat = torch.ones((2, 100, 384), device=DEVICE)
        sem_p, feat_p, align_loss = model(sem, feat)

        assert sem_p.shape == (2, 100, 192)
        assert feat_p.shape == (2, 100, 192)
        assert align_loss.shape == ()

    def test_align_loss_nonnegative(self):
        """Alignment loss should be non-negative."""
        from mbps_pytorch.models.bridge.projection import AdaptiveProjectionBridge

        torch.manual_seed(0)
        model = AdaptiveProjectionBridge().to(DEVICE)
        sem = torch.randn((1, 50, 90), device=DEVICE)
        feat = torch.randn((1, 50, 384), device=DEVICE)
        _, _, align_loss = model(sem, feat)

        assert float(align_loss) >= 0.0


class TestMamba2:
    """Test Mamba2 SSD module."""

    def test_output_shape(self):
        """Mamba2 block should preserve sequence shape."""
        from mbps_pytorch.models.bridge.mamba2_ssd import Mamba2Block

        model = Mamba2Block(dim=64, state_dim=16, chunk_size=32).to(DEVICE)
        x = torch.ones((2, 128, 64), device=DEVICE)
        out = model(x)

        assert out.shape == (2, 128, 64)

    def test_stack_output_shape(self):
        """Mamba2 stack should preserve shape."""
        from mbps_pytorch.models.bridge.mamba2_ssd import Mamba2Stack

        model = Mamba2Stack(
            num_layers=2, dim=32, state_dim=8, chunk_size=16
        ).to(DEVICE)
        x = torch.ones((1, 64, 32), device=DEVICE)
        out = model(x)

        assert out.shape == (1, 64, 32)


class TestBiCMS:
    """Test Bidirectional Cross-Modal Scan."""

    def test_output_shapes(self):
        """BiCMS should produce two output streams of same shape."""
        from mbps_pytorch.models.bridge.bicms import BidirectionalCrossModalScan

        model = BidirectionalCrossModalScan(
            dim=32, num_layers=1, state_dim=8, chunk_size=16
        ).to(DEVICE)
        sem = torch.ones((1, 32, 32), device=DEVICE)
        feat = torch.ones((1, 32, 32), device=DEVICE)
        out_sem, out_feat = model(sem, feat)

        assert out_sem.shape == (1, 32, 32)
        assert out_feat.shape == (1, 32, 32)


class TestDepthConditioning:
    """Test Unified Depth Conditioning Module."""

    def test_output_shapes(self):
        """UDCM should output conditioned features of same shape."""
        from mbps_pytorch.models.bridge.depth_conditioning import (
            UnifiedDepthConditioning,
        )

        model = UnifiedDepthConditioning(bridge_dim=64).to(DEVICE)
        depth = torch.ones((2, 50), device=DEVICE)
        sem = torch.ones((2, 50, 64), device=DEVICE)
        feat = torch.ones((2, 50, 64), device=DEVICE)
        out_sem, out_feat, d_loss = model(depth, sem, feat)

        assert out_sem.shape == (2, 50, 64)
        assert out_feat.shape == (2, 50, 64)


class TestLossFunctions:
    """Test loss function computations."""

    def test_dice_loss_range(self):
        """Dice loss should be in [0, 1]."""
        from mbps_pytorch.losses.instance_loss import dice_loss

        pred = torch.zeros((1, 5, 100), device=DEVICE)
        target = torch.ones((1, 5, 100), device=DEVICE)
        loss = dice_loss(pred, target)

        assert float(loss) >= 0.0
        assert float(loss) <= 1.0

    def test_dice_loss_perfect(self):
        """Dice loss should be near 0 for perfect prediction."""
        from mbps_pytorch.losses.instance_loss import dice_loss

        pred = torch.ones((1, 5, 100), device=DEVICE) * 10.0
        target = torch.ones((1, 5, 100), device=DEVICE)
        loss = dice_loss(pred, target)

        assert float(loss) < 0.1

    def test_cka_loss_self_alignment(self):
        """CKA of feature with itself should give -1 (perfect alignment)."""
        from mbps_pytorch.losses.bridge_loss import cka_loss

        torch.manual_seed(0)
        features = torch.randn((1, 50, 32), device=DEVICE)
        loss = cka_loss(features, features)

        assert abs(float(loss) - (-1.0)) < 0.01

    def test_pq_proxy_range(self):
        """PQ proxy should be in [0, 1]."""
        from mbps_pytorch.losses.pq_proxy_loss import PQProxyLoss

        torch.manual_seed(0)
        pq_loss = PQProxyLoss().to(DEVICE)
        pred_masks = torch.randn((1, 10, 100), device=DEVICE)
        pred_scores = torch.ones((1, 10), device=DEVICE) * 0.8
        teacher_masks = torch.randn((1, 10, 100), device=DEVICE)
        teacher_scores = torch.ones((1, 10), device=DEVICE) * 0.9

        result = pq_loss(pred_masks, pred_scores, teacher_masks, teacher_scores)

        assert float(result["total"]) >= 0.0
        assert float(result["total"]) <= 2.0


class TestCurriculum:
    """Test training curriculum."""

    def test_phase_transitions(self):
        """Phases should transition at correct epochs."""
        from mbps_pytorch.training.curriculum import TrainingCurriculum

        curriculum = TrainingCurriculum(
            phase_a_end=20, phase_b_end=40, total_epochs=60
        )

        assert curriculum.get_phase(1) == "A"
        assert curriculum.get_phase(20) == "A"
        assert curriculum.get_phase(21) == "B"
        assert curriculum.get_phase(40) == "B"
        assert curriculum.get_phase(41) == "C"

    def test_beta_rampup(self):
        """Beta should ramp from 0->1 during Phase B."""
        from mbps_pytorch.training.curriculum import TrainingCurriculum

        curriculum = TrainingCurriculum(phase_a_end=20, phase_b_end=40)

        config_start = curriculum.get_config(21)
        config_mid = curriculum.get_config(30)
        config_end = curriculum.get_config(40)

        assert abs(config_start.beta - 0.05) < 0.15
        assert abs(config_mid.beta - 0.5) < 0.15
        assert abs(config_end.beta - 1.0) < 0.15

    def test_phase_a_no_instance(self):
        """Phase A should have zero instance weight."""
        from mbps_pytorch.training.curriculum import TrainingCurriculum

        curriculum = TrainingCurriculum()
        config = curriculum.get_config(10)

        assert config.beta == 0.0
        assert not config.use_bridge


class TestTransforms:
    """Test data transforms."""

    def test_normalize_denormalize(self):
        """Normalize -> denormalize should be identity."""
        from mbps_pytorch.data.transforms import denormalize, normalize

        np.random.seed(0)
        image = np.random.rand(224, 224, 3).astype(np.float32)

        normalized = normalize(image)
        recovered = denormalize(normalized)

        np.testing.assert_allclose(image, recovered, atol=1e-5)

    def test_random_crop_shape(self):
        """Random crop should produce correct output shape."""
        from mbps_pytorch.data.transforms import random_crop

        image = np.ones((512, 512, 3), dtype=np.float32)
        cropped = random_crop(image, (320, 320), top=0, left=0)

        assert cropped.shape == (320, 320, 3)


class TestStuffThingsClassifier:
    """Test stuff-things classifier."""

    def test_output_shape(self):
        """Should output one score per cluster."""
        from mbps_pytorch.models.classifier.stuff_things_mlp import (
            StuffThingsClassifier,
        )

        model = StuffThingsClassifier(input_dim=3).to(DEVICE)
        cues = torch.ones((2, 27, 3), device=DEVICE)
        scores = model(cues)

        assert scores.shape == (2, 27)

    def test_scores_in_range(self):
        """Scores should be in [0, 1] (sigmoid output)."""
        from mbps_pytorch.models.classifier.stuff_things_mlp import (
            StuffThingsClassifier,
        )

        torch.manual_seed(42)
        model = StuffThingsClassifier(input_dim=3).to(DEVICE)
        cues = torch.randn((1, 10, 3), device=DEVICE)
        scores = model(cues)

        assert torch.all(scores >= 0)
        assert torch.all(scores <= 1)


class TestHungarianMatching:
    """Test Hungarian matching."""

    def test_perfect_match(self):
        """Perfect clustering should give accuracy = 1.0."""
        from mbps_pytorch.evaluation.hungarian_matching import hungarian_match

        pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        gt = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])

        mapping, acc = hungarian_match(pred, gt, 3, 3)
        assert abs(acc - 1.0) < 1e-5


class TestEMA:
    """Test EMA updates."""

    def test_ema_update(self):
        """EMA should smoothly average parameters."""
        from mbps_pytorch.training.ema import EMAState

        # Create a simple model with known weights
        model = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        ema = EMAState(model, momentum=0.5)

        # Update with new weights = 0
        with torch.no_grad():
            model.weight.fill_(0.0)
        ema.update(model)

        # EMA = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        ema_sd = ema.get_state_dict()
        expected = 0.5
        np.testing.assert_allclose(
            ema_sd["weight"].cpu().numpy(),
            np.full_like(ema_sd["weight"].cpu().numpy(), expected),
            atol=1e-6,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
