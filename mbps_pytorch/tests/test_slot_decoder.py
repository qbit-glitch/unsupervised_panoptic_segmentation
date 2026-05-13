"""Unit tests for Depth-Conditioned Slot Attention Decoder."""

import pytest
import torch
import numpy as np

from mbps_pytorch.models.slot_decoder import DepthSlotDecoder, DepthSlotDecoderConfig
from mbps_pytorch.models.slot_decoder.depth_film import DepthFiLM, GRID_H, GRID_W, N_PATCHES
from mbps_pytorch.models.slot_decoder.slot_attention import SlotAttention
from mbps_pytorch.models.slot_decoder.decoder import SpatialBroadcastDecoder


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def config():
    return DepthSlotDecoderConfig(
        feat_dim=1024,
        slot_dim=256,
        num_slots=20,
        slot_iters=3,
        decoder_hidden=512,
        decoder_layers=2,
    )


class TestDepthFiLM:
    def test_output_shape(self, device):
        film = DepthFiLM(feat_dim=256, n_freq=8, hidden_dim=128).to(device)
        features = torch.randn(2, N_PATCHES, 256, device=device)
        depth = torch.rand(2, 512, 1024, device=device)

        out = film(features, depth)
        assert out.shape == (2, N_PATCHES, 256)

    def test_identity_init(self, device):
        """FiLM should initialize to approximate identity (gamma=1, beta=0)."""
        film = DepthFiLM(feat_dim=256, n_freq=8, hidden_dim=128).to(device)
        features = torch.randn(1, N_PATCHES, 256, device=device)
        depth = torch.rand(1, 512, 1024, device=device)

        out = film(features, depth)
        # Should be close to input at initialization
        diff = (out - features).abs().mean().item()
        assert diff < 0.1, f"Initial FiLM output too far from identity: {diff:.4f}"

    def test_depth_encoding_shape(self, device):
        film = DepthFiLM(feat_dim=256, n_freq=16, hidden_dim=128).to(device)
        depth = torch.rand(2, 512, 1024, device=device)

        enc = film.encode_depth(depth)
        expected_dim = 2 * 16 + 3  # sin + cos + raw + grad_x + grad_y
        assert enc.shape == (2, N_PATCHES, expected_dim)


class TestSlotAttention:
    def test_output_shapes(self, device):
        sa = SlotAttention(num_slots=10, dim=128, iters=3).to(device)
        inputs = torch.randn(2, N_PATCHES, 128, device=device)

        slots, attn = sa(inputs)
        assert slots.shape == (2, 10, 128)
        assert attn.shape == (2, 10, N_PATCHES)

    def test_attention_sums_to_one(self, device):
        """Attention should sum to 1 over slots for each patch."""
        sa = SlotAttention(num_slots=10, dim=128, iters=3).to(device)
        inputs = torch.randn(2, N_PATCHES, 128, device=device)

        _, attn = sa(inputs)
        attn_sum = attn.sum(dim=1)  # Sum over slots: (B, N)
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)

    def test_gradient_flow(self, device):
        sa = SlotAttention(num_slots=5, dim=64, iters=2).to(device)
        inputs = torch.randn(1, 100, 64, device=device, requires_grad=True)

        slots, _ = sa(inputs)
        loss = slots.sum()
        loss.backward()
        assert inputs.grad is not None
        assert inputs.grad.abs().sum() > 0


class TestSpatialBroadcastDecoder:
    def test_output_shapes(self, device):
        decoder = SpatialBroadcastDecoder(
            slot_dim=128, target_dim=1024, hidden_dim=256, n_layers=2
        ).to(device)
        slots = torch.randn(2, 10, 128, device=device)

        recon, masks = decoder(slots)
        assert recon.shape == (2, N_PATCHES, 1024)
        assert masks.shape == (2, 10, N_PATCHES)

    def test_masks_sum_to_one(self, device):
        """Decoder masks should sum to 1 over slots."""
        decoder = SpatialBroadcastDecoder(
            slot_dim=128, target_dim=512, hidden_dim=256, n_layers=2
        ).to(device)
        slots = torch.randn(2, 5, 128, device=device)

        _, masks = decoder(slots)
        mask_sum = masks.sum(dim=1)
        assert torch.allclose(mask_sum, torch.ones_like(mask_sum), atol=1e-5)


class TestDepthSlotDecoder:
    def test_forward_shapes(self, config, device):
        model = DepthSlotDecoder(config).to(device)
        features = torch.randn(2, N_PATCHES, 1024, device=device)
        depth = torch.rand(2, 512, 1024, device=device)

        out = model(features, depth)
        assert out["recon"].shape == (2, N_PATCHES, 1024)
        assert out["masks"].shape == (2, 20, N_PATCHES)
        assert out["slots"].shape == (2, 20, 256)
        assert out["attn"].shape == (2, 20, N_PATCHES)

    def test_instance_mask_extraction(self, config, device):
        model = DepthSlotDecoder(config).to(device)
        features = torch.randn(1, N_PATCHES, 1024, device=device)
        depth = torch.rand(1, 512, 1024, device=device)

        instance_map, scores = model.get_instance_masks(features, depth)
        assert instance_map.shape == (1, 512, 1024)
        assert scores.shape == (1, 20)
        assert instance_map.dtype == torch.int64

    def test_parameter_count(self, config, device):
        model = DepthSlotDecoder(config).to(device)
        n_params = model.count_parameters()
        # Should be in 5-15M range for this config
        assert 1_000_000 < n_params < 50_000_000, f"Unexpected param count: {n_params:,}"

    def test_gradient_flow_full(self, device):
        cfg = DepthSlotDecoderConfig(
            feat_dim=64, slot_dim=32, num_slots=5, slot_iters=2,
            decoder_hidden=64, decoder_layers=1,
        )
        model = DepthSlotDecoder(cfg).to(device)
        features = torch.randn(1, N_PATCHES, 64, device=device)
        depth = torch.rand(1, 512, 1024, device=device)

        out = model(features, depth)
        loss = out["recon"].sum() + out["masks"].sum()
        loss.backward()

        # Check all parameters received gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_reconstruction_loss_decreases(self, device):
        """Verify model can overfit on a single sample."""
        cfg = DepthSlotDecoderConfig(
            feat_dim=64, slot_dim=32, num_slots=5, slot_iters=2,
            decoder_hidden=64, decoder_layers=1,
        )
        model = DepthSlotDecoder(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        features = torch.randn(1, N_PATCHES, 64, device=device)
        depth = torch.rand(1, 512, 1024, device=device)

        losses = []
        for _ in range(20):
            out = model(features, depth)
            loss = torch.nn.functional.mse_loss(out["recon"], features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


class TestLossFunctions:
    def test_pseudo_label_loss(self, device):
        from mbps_pytorch.train_slot_decoder import pseudo_label_loss

        masks_raw = torch.rand(2, 10, N_PATCHES, device=device, requires_grad=True)
        masks = masks_raw / masks_raw.sum(dim=1, keepdim=True)  # normalize (keeps grad)

        instance_masks = torch.zeros(2, 3, N_PATCHES, device=device)
        # Create 3 non-overlapping instances
        instance_masks[0, 0, :500] = 1.0
        instance_masks[0, 1, 500:1000] = 1.0
        instance_masks[0, 2, 1000:1500] = 1.0
        instance_masks[1, 0, :700] = 1.0
        instance_masks[1, 1, 700:1400] = 1.0
        instance_masks[1, 2, 1400:2048] = 1.0

        num_instances = torch.tensor([3, 3], device=device)
        loss = pseudo_label_loss(masks, instance_masks, num_instances)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0
        assert loss.requires_grad

    def test_depth_consistency_loss(self, device):
        from mbps_pytorch.train_slot_decoder import depth_consistency_loss

        masks = torch.rand(2, 5, N_PATCHES, device=device)
        masks = masks / masks.sum(dim=1, keepdim=True)
        depth = torch.rand(2, 512, 1024, device=device)

        loss = depth_consistency_loss(masks, depth)
        assert loss.ndim == 0
        assert loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
