"""Tests for Mac-compatible Mamba2 (pure PyTorch).

Run with:
    python -m pytest mbps_pytorch/mamba2/test_mamba2.py -v
"""

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from .ssd import segsum, ssd_minimal_discrete, ssd_chunk_scan_combined
from .norm import RMSNormGated, rms_norm_fn
from .mamba2 import Mamba2
from .gated_delta_net import GatedDeltaNet, chunk_gated_delta_rule, recurrent_gated_delta_rule
from .scan import raster_scan, raster_unscan, cross_scan, cross_unscan, interleave_tokens, deinterleave_tokens
from .vision import VisionMamba2, CrossModalMamba2


# ---------------------------------------------------------------------------
# Test 1: SSD numerical correctness
# ---------------------------------------------------------------------------

class TestSSDCore:
    """Test the core SSD algorithm against the minimal discrete version."""

    def test_ssd_minimal_discrete_shapes(self):
        """ssd_minimal_discrete should produce correct output shapes."""
        torch.manual_seed(42)
        batch, seqlen, chunk_size, nheads, headdim, dstate = 2, 256, 64, 4, 32, 16

        X = torch.randn(batch, seqlen, nheads, headdim)
        A = torch.randn(batch, seqlen, nheads)
        B = torch.randn(batch, seqlen, nheads, dstate)
        C = torch.randn(batch, seqlen, nheads, dstate)

        Y, final_state = ssd_minimal_discrete(X, A, B, C, chunk_size)
        assert Y.shape == (batch, seqlen, nheads, headdim)
        assert final_state.shape == (batch, nheads, headdim, dstate)

    def test_ssd_combined_shapes(self):
        """ssd_chunk_scan_combined should produce correct output shapes."""
        torch.manual_seed(42)
        batch, seqlen, chunk_size = 2, 256, 64
        nheads, headdim, ngroups, dstate = 4, 32, 1, 16

        x = torch.randn(batch, seqlen, nheads, headdim)
        dt = torch.randn(batch, seqlen, nheads)
        A = -torch.exp(torch.rand(nheads))
        B = torch.randn(batch, seqlen, ngroups, dstate)
        C = torch.randn(batch, seqlen, ngroups, dstate)
        D = torch.randn(nheads)

        out = ssd_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D,
                                      dt_bias=torch.zeros(nheads), dt_softplus=True)
        assert out.shape == (batch, seqlen, nheads, headdim)

    def test_ssd_minimal_vs_combined_consistency(self):
        """Both SSD paths should produce similar results.

        The minimal version takes pre-discretized (X=x*dt, A=A*dt) inputs,
        while the combined version handles dt processing internally.
        """
        torch.manual_seed(42)
        batch, seqlen, chunk_size = 1, 256, 64
        nheads, headdim, dstate = 4, 32, 64
        ngroups = 1

        x = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float32)
        dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32) - 4)
        A = -torch.exp(torch.rand(nheads, dtype=torch.float32))
        B = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.float32)
        C = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.float32)

        # Combined path
        y_combined = ssd_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None)

        # Minimal path (needs pre-discretization and expanded B/C from ngroups to nheads)
        B_expanded = B.repeat(1, 1, nheads // ngroups, 1)
        C_expanded = C.repeat(1, 1, nheads // ngroups, 1)
        y_minimal, _ = ssd_minimal_discrete(
            x * dt.unsqueeze(-1), A * dt, B_expanded, C_expanded, chunk_size
        )

        # They won't be exactly equal due to different numerical paths,
        # but should be close
        assert torch.allclose(y_combined, y_minimal, atol=1e-3, rtol=1e-2), \
            f"Max diff: {(y_combined - y_minimal).abs().max().item():.6f}"

    def test_ssd_combined_with_z_gating(self):
        """Test SSD with z (SiLU gating)."""
        torch.manual_seed(42)
        batch, seqlen, chunk_size = 2, 128, 64
        nheads, headdim, ngroups, dstate = 4, 32, 1, 16

        x = torch.randn(batch, seqlen, nheads, headdim)
        dt = torch.randn(batch, seqlen, nheads)
        A = -torch.exp(torch.rand(nheads))
        B = torch.randn(batch, seqlen, ngroups, dstate)
        C = torch.randn(batch, seqlen, ngroups, dstate)
        z = torch.randn(batch, seqlen, nheads, headdim)

        out = ssd_chunk_scan_combined(x, dt, A, B, C, chunk_size, z=z,
                                      dt_bias=torch.zeros(nheads), dt_softplus=True)
        assert out.shape == (batch, seqlen, nheads, headdim)
        assert not torch.isnan(out).any()

    def test_ssd_combined_return_final_states(self):
        """Test return_final_states flag."""
        torch.manual_seed(42)
        batch, seqlen, chunk_size = 2, 128, 64
        nheads, headdim, ngroups, dstate = 4, 32, 1, 16

        x = torch.randn(batch, seqlen, nheads, headdim)
        dt = torch.randn(batch, seqlen, nheads)
        A = -torch.exp(torch.rand(nheads))
        B = torch.randn(batch, seqlen, ngroups, dstate)
        C = torch.randn(batch, seqlen, ngroups, dstate)

        out, final_states = ssd_chunk_scan_combined(
            x, dt, A, B, C, chunk_size,
            dt_bias=torch.zeros(nheads), dt_softplus=True,
            return_final_states=True,
        )
        assert out.shape == (batch, seqlen, nheads, headdim)
        assert final_states.shape == (batch, nheads, headdim, dstate)


# ---------------------------------------------------------------------------
# Test 2: RMSNormGated
# ---------------------------------------------------------------------------

class TestRMSNormGated:
    def test_basic(self):
        norm = RMSNormGated(64, eps=1e-5)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_with_gating(self):
        norm = RMSNormGated(64, eps=1e-5, norm_before_gate=False)
        x = torch.randn(2, 10, 64)
        z = torch.randn(2, 10, 64)
        out = norm(x, z)
        assert out.shape == x.shape

    def test_with_group_size(self):
        norm = RMSNormGated(64, eps=1e-5, group_size=32)
        x = torch.randn(2, 10, 64)
        z = torch.randn(2, 10, 64)
        out = norm(x, z)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Test 3: Full Mamba2 module — forward pass
# ---------------------------------------------------------------------------

class TestMamba2Module:
    def test_forward_shape(self):
        """Basic forward pass produces correct output shape."""
        torch.manual_seed(42)
        model = Mamba2(d_model=128, d_state=64, headdim=32, chunk_size=64)
        x = torch.randn(2, 256, 128)
        y = model(x)
        assert y.shape == x.shape

    def test_forward_small(self):
        """Forward with smaller dimensions."""
        model = Mamba2(d_model=64, d_state=16, headdim=16, chunk_size=32, ngroups=1)
        x = torch.randn(1, 64, 64)
        y = model(x)
        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_forward_no_rmsnorm(self):
        """Forward without RMSNorm."""
        model = Mamba2(d_model=64, d_state=16, headdim=16, chunk_size=32, rmsnorm=False)
        x = torch.randn(1, 64, 64)
        y = model(x)
        assert y.shape == x.shape

    def test_forward_d_ssm_less_than_d_inner(self):
        """Forward with d_ssm < d_inner (gated MLP branch active)."""
        model = Mamba2(d_model=64, d_state=16, headdim=16, expand=2,
                       d_ssm=64, chunk_size=32)  # d_inner=128, d_ssm=64 → d_mlp=32
        x = torch.randn(1, 64, 64)
        y = model(x)
        assert y.shape == x.shape

    def test_forward_D_has_hdim(self):
        """Forward with D having per-headdim values."""
        model = Mamba2(d_model=64, d_state=16, headdim=16, chunk_size=32, D_has_hdim=True)
        x = torch.randn(1, 64, 64)
        y = model(x)
        assert y.shape == x.shape

    def test_forward_with_dt_limit(self):
        """Forward with dt_limit clamping."""
        model = Mamba2(d_model=64, d_state=16, headdim=16, chunk_size=32,
                       dt_limit=(0.001, 0.1))
        x = torch.randn(1, 64, 64)
        y = model(x)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Test 4: Gradient flow
# ---------------------------------------------------------------------------

class TestGradients:
    def test_backward(self):
        """All parameters should receive gradients."""
        torch.manual_seed(42)
        model = Mamba2(d_model=64, d_state=16, headdim=16, chunk_size=32)
        x = torch.randn(1, 64, 64)
        y = model(x)
        loss = y.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_backward_no_nan(self):
        """Multiple forward-backward passes should not produce NaN."""
        model = Mamba2(d_model=64, d_state=16, headdim=16, chunk_size=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(3):
            x = torch.randn(1, 64, 64)
            y = model(x)
            loss = y.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check parameters for NaN
        for name, param in model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN parameter after training: {name}"


# ---------------------------------------------------------------------------
# Test 5: Sequence length edge cases
# ---------------------------------------------------------------------------

class TestSequenceLengthEdgeCases:
    def test_seqlen_not_divisible_by_chunk_size(self):
        """Should handle sequence lengths not divisible by chunk_size."""
        model = Mamba2(d_model=64, d_state=16, headdim=16, chunk_size=64)
        # 100 is not divisible by 64
        x = torch.randn(1, 100, 64)
        y = model(x)
        assert y.shape == (1, 100, 64)

    def test_seqlen_shorter_than_chunk_size(self):
        """Should handle sequence shorter than chunk_size."""
        model = Mamba2(d_model=64, d_state=16, headdim=16, chunk_size=64)
        x = torch.randn(1, 32, 64)
        y = model(x)
        assert y.shape == (1, 32, 64)

    def test_seqlen_equals_chunk_size(self):
        """Should handle seqlen == chunk_size."""
        model = Mamba2(d_model=64, d_state=16, headdim=16, chunk_size=64)
        x = torch.randn(1, 64, 64)
        y = model(x)
        assert y.shape == (1, 64, 64)


# ---------------------------------------------------------------------------
# Test 6: MPS device (if available)
# ---------------------------------------------------------------------------

class TestMPS:
    @pytest.mark.skipif(
        not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
        reason="MPS not available"
    )
    def test_forward_on_mps(self):
        """Forward pass should work on MPS."""
        device = torch.device("mps")
        model = Mamba2(d_model=128, d_state=64, headdim=32, chunk_size=64).to(device)
        x = torch.randn(1, 128, 128, device=device)
        y = model(x)
        assert y.shape == x.shape
        assert y.device.type == "mps"
        assert not torch.isnan(y).any()

    @pytest.mark.skipif(
        not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
        reason="MPS not available"
    )
    def test_backward_on_mps(self):
        """Backward pass should work on MPS."""
        device = torch.device("mps")
        model = Mamba2(d_model=64, d_state=16, headdim=16, chunk_size=32).to(device)
        x = torch.randn(1, 64, 64, device=device)
        y = model(x)
        loss = y.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# Test 7: Step (autoregressive decoding)
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_basic(self):
        """Single-step inference should produce correct shapes."""
        model = Mamba2(d_model=64, d_state=16, headdim=16, chunk_size=32)
        conv_state, ssm_state = model.allocate_inference_cache(batch_size=1, max_seqlen=128)
        hidden = torch.randn(1, 1, 64)

        out, conv_state, ssm_state = model.step(hidden, conv_state, ssm_state)
        assert out.shape == (1, 1, 64)

    def test_step_multiple(self):
        """Multiple steps should update states."""
        model = Mamba2(d_model=64, d_state=16, headdim=16, chunk_size=32)
        conv_state, ssm_state = model.allocate_inference_cache(batch_size=1, max_seqlen=128)

        for _ in range(10):
            hidden = torch.randn(1, 1, 64)
            out, conv_state, ssm_state = model.step(hidden, conv_state, ssm_state)
            assert out.shape == (1, 1, 64)
            assert not torch.isnan(out).any()


# ===========================================================================
# Vision Mamba2 Tests
# ===========================================================================


# ---------------------------------------------------------------------------
# Test 8: Scan functions
# ---------------------------------------------------------------------------

class TestScanFunctions:
    def test_raster_roundtrip(self):
        """raster_scan -> raster_unscan should be identity."""
        x = torch.randn(2, 64, 8, 16)
        tokens = raster_scan(x)
        assert tokens.shape == (2, 128, 64)
        recovered = raster_unscan(tokens, 8, 16)
        assert torch.allclose(x, recovered)

    def test_cross_scan_shapes(self):
        """cross_scan should produce (4*B, H*W, C)."""
        x = torch.randn(2, 64, 8, 16)
        scanned = cross_scan(x)
        assert scanned.shape == (8, 128, 64)  # 4*2=8, 8*16=128

    def test_cross_scan_unscan_roundtrip(self):
        """cross_scan -> identity Mamba2 -> cross_unscan should recover directions."""
        x = torch.randn(2, 32, 8, 16)
        B, C, H, W = x.shape
        scanned = cross_scan(x)  # (8, 128, 32)
        # Simulate identity (no processing)
        dirs = cross_unscan(scanned, B, H, W)  # (2, 4, 128, 32)
        assert dirs.shape == (2, 4, 128, 32)
        # Dir 0 (raster forward) should match raster_scan
        raster = raster_scan(x)
        assert torch.allclose(dirs[:, 0], raster)

    def test_cross_scan_directions_differ(self):
        """Different scan directions should produce different orderings."""
        x = torch.arange(24).float().reshape(1, 1, 4, 6)  # Non-symmetric
        scanned = cross_scan(x)  # (4, 24, 1)
        dir0, dir1, dir2, dir3 = scanned.chunk(4, dim=0)
        # Dir 0 and dir 1 should be reverses of each other
        assert torch.allclose(dir0, dir1.flip(1))
        # Dir 2 and dir 3 should be reverses of each other
        assert torch.allclose(dir2, dir3.flip(1))
        # Dir 0 and dir 2 should differ (row-major vs col-major)
        assert not torch.allclose(dir0, dir2)

    def test_interleave_deinterleave_roundtrip(self):
        """interleave -> deinterleave should recover original."""
        s = torch.randn(2, 100, 64)
        f = torch.randn(2, 100, 64)
        interleaved = interleave_tokens(s, f)
        assert interleaved.shape == (2, 200, 64)
        s2, f2 = deinterleave_tokens(interleaved)
        assert torch.allclose(s, s2)
        assert torch.allclose(f, f2)

    def test_interleave_ordering(self):
        """Interleaving should produce [s1, f1, s2, f2, ...]."""
        s = torch.tensor([[[1., 2.], [3., 4.]]])  # (1, 2, 2)
        f = torch.tensor([[[5., 6.], [7., 8.]]])
        interleaved = interleave_tokens(s, f)
        expected = torch.tensor([[[1., 2.], [5., 6.], [3., 4.], [7., 8.]]])
        assert torch.allclose(interleaved, expected)


# ---------------------------------------------------------------------------
# Test 9: VisionMamba2 — all 3 scan modes with image input
# ---------------------------------------------------------------------------

VISION_MAMBA_KWARGS = dict(d_state=16, headdim=16, chunk_size=32)


class TestVisionMamba2:
    @pytest.mark.parametrize("scan_mode", ["raster", "bidirectional", "cross_scan"])
    def test_image_forward_shape(self, scan_mode):
        """4D image input → same shape output."""
        model = VisionMamba2(d_model=64, scan_mode=scan_mode, **VISION_MAMBA_KWARGS)
        x = torch.randn(1, 64, 8, 16)
        y = model(x)
        assert y.shape == x.shape, f"{scan_mode}: expected {x.shape}, got {y.shape}"

    @pytest.mark.parametrize("scan_mode", ["raster", "bidirectional", "cross_scan"])
    def test_sequence_forward_shape(self, scan_mode):
        """3D sequence input → same shape output."""
        model = VisionMamba2(d_model=64, scan_mode=scan_mode, **VISION_MAMBA_KWARGS)
        x = torch.randn(1, 128, 64)
        y = model(x)
        assert y.shape == x.shape, f"{scan_mode}: expected {x.shape}, got {y.shape}"

    @pytest.mark.parametrize("scan_mode", ["raster", "bidirectional", "cross_scan"])
    def test_image_backward(self, scan_mode):
        """All scan modes should support backward pass on images."""
        torch.manual_seed(42)
        model = VisionMamba2(d_model=64, scan_mode=scan_mode, **VISION_MAMBA_KWARGS)
        x = torch.randn(1, 64, 8, 16)
        y = model(x)
        loss = y.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{scan_mode}: no grad for {name}"
            assert not torch.isnan(param.grad).any(), f"{scan_mode}: NaN grad for {name}"

    @pytest.mark.parametrize("scan_mode", ["raster", "bidirectional", "cross_scan"])
    def test_no_nan(self, scan_mode):
        """Output should not contain NaN."""
        torch.manual_seed(42)
        model = VisionMamba2(d_model=64, scan_mode=scan_mode, **VISION_MAMBA_KWARGS)
        x = torch.randn(2, 64, 8, 16)
        y = model(x)
        assert not torch.isnan(y).any(), f"{scan_mode}: NaN in output"

    def test_different_scan_modes_differ(self):
        """Different scan modes should produce different outputs (not identical)."""
        torch.manual_seed(42)
        x = torch.randn(1, 64, 8, 16)
        outputs = {}
        for mode in ["raster", "bidirectional", "cross_scan"]:
            torch.manual_seed(0)  # Same init for all
            model = VisionMamba2(d_model=64, scan_mode=mode, **VISION_MAMBA_KWARGS)
            with torch.no_grad():
                outputs[mode] = model(x)
        # At least raster and cross_scan should differ (different architectures)
        assert not torch.allclose(outputs["raster"], outputs["cross_scan"], atol=1e-3)

    def test_batch_size_gt_1(self):
        """Should work with batch_size > 1."""
        model = VisionMamba2(d_model=64, scan_mode="cross_scan", **VISION_MAMBA_KWARGS)
        x = torch.randn(4, 64, 8, 16)
        y = model(x)
        assert y.shape == (4, 64, 8, 16)


# ---------------------------------------------------------------------------
# Test 10: CrossModalMamba2 — all 3 scan modes
# ---------------------------------------------------------------------------

class TestCrossModalMamba2:
    @pytest.mark.parametrize("scan_mode", ["raster", "bidirectional", "cross_scan"])
    def test_image_forward_shape(self, scan_mode):
        """4D image inputs → same shape outputs."""
        model = CrossModalMamba2(d_model=64, scan_mode=scan_mode, **VISION_MAMBA_KWARGS)
        s = torch.randn(1, 64, 8, 16)
        f = torch.randn(1, 64, 8, 16)
        fs, ff = model(s, f)
        assert fs.shape == s.shape, f"{scan_mode}: semantic shape mismatch"
        assert ff.shape == f.shape, f"{scan_mode}: feature shape mismatch"

    @pytest.mark.parametrize("scan_mode", ["raster", "bidirectional", "cross_scan"])
    def test_sequence_forward_shape(self, scan_mode):
        """3D sequence inputs → same shape outputs."""
        model = CrossModalMamba2(d_model=64, scan_mode=scan_mode, **VISION_MAMBA_KWARGS)
        s = torch.randn(1, 128, 64)
        f = torch.randn(1, 128, 64)
        fs, ff = model(s, f)
        assert fs.shape == s.shape
        assert ff.shape == f.shape

    @pytest.mark.parametrize("scan_mode", ["raster", "bidirectional", "cross_scan"])
    def test_backward(self, scan_mode):
        """Cross-modal should support backward pass."""
        model = CrossModalMamba2(d_model=64, scan_mode=scan_mode, **VISION_MAMBA_KWARGS)
        s = torch.randn(1, 64, 8, 16)
        f = torch.randn(1, 64, 8, 16)
        fs, ff = model(s, f)
        loss = fs.sum() + ff.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{scan_mode}: no grad for {name}"

    @pytest.mark.parametrize("scan_mode", ["raster", "bidirectional", "cross_scan"])
    def test_no_nan(self, scan_mode):
        """Output should not contain NaN."""
        torch.manual_seed(42)
        model = CrossModalMamba2(d_model=64, scan_mode=scan_mode, **VISION_MAMBA_KWARGS)
        s = torch.randn(2, 64, 8, 16)
        f = torch.randn(2, 64, 8, 16)
        fs, ff = model(s, f)
        assert not torch.isnan(fs).any(), f"{scan_mode}: NaN in fused_semantic"
        assert not torch.isnan(ff).any(), f"{scan_mode}: NaN in fused_features"

    def test_cross_modal_mixes_streams(self):
        """Output should depend on both inputs (not just passthrough)."""
        torch.manual_seed(42)
        model = CrossModalMamba2(d_model=64, scan_mode="bidirectional", **VISION_MAMBA_KWARGS)
        s = torch.randn(1, 64, 8, 16)
        f1 = torch.randn(1, 64, 8, 16)
        f2 = torch.randn(1, 64, 8, 16)
        with torch.no_grad():
            fs1, _ = model(s, f1)
            fs2, _ = model(s, f2)
        # Same semantic, different features → different fused semantic
        assert not torch.allclose(fs1, fs2, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 11: Vision Mamba2 on MPS
# ---------------------------------------------------------------------------

class TestVisionMPS:
    @pytest.mark.skipif(
        not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
        reason="MPS not available"
    )
    @pytest.mark.parametrize("scan_mode", ["raster", "bidirectional", "cross_scan"])
    def test_vision_forward_mps(self, scan_mode):
        """VisionMamba2 forward on MPS for all scan modes."""
        device = torch.device("mps")
        model = VisionMamba2(d_model=64, scan_mode=scan_mode, **VISION_MAMBA_KWARGS).to(device)
        x = torch.randn(1, 64, 8, 16, device=device)
        y = model(x)
        assert y.shape == x.shape
        assert y.device.type == "mps"
        assert not torch.isnan(y).any()

    @pytest.mark.skipif(
        not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
        reason="MPS not available"
    )
    @pytest.mark.parametrize("scan_mode", ["raster", "bidirectional", "cross_scan"])
    def test_vision_backward_mps(self, scan_mode):
        """VisionMamba2 backward on MPS."""
        device = torch.device("mps")
        model = VisionMamba2(d_model=64, scan_mode=scan_mode, **VISION_MAMBA_KWARGS).to(device)
        x = torch.randn(1, 64, 8, 16, device=device)
        y = model(x)
        y.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{scan_mode}: no grad for {name}"

    @pytest.mark.skipif(
        not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
        reason="MPS not available"
    )
    def test_cross_modal_mps(self):
        """CrossModalMamba2 on MPS."""
        device = torch.device("mps")
        model = CrossModalMamba2(d_model=64, scan_mode="bidirectional", **VISION_MAMBA_KWARGS).to(device)
        s = torch.randn(1, 64, 8, 16, device=device)
        f = torch.randn(1, 64, 8, 16, device=device)
        fs, ff = model(s, f)
        assert fs.shape == s.shape
        assert fs.device.type == "mps"


# ===========================================================================
# GatedDeltaNet Tests
# ===========================================================================


# ---------------------------------------------------------------------------
# Test 12: GatedDeltaNet core kernels
# ---------------------------------------------------------------------------

class TestGatedDeltaRuleKernels:
    def test_chunk_output_shape(self):
        """chunk_gated_delta_rule should produce correct output shapes."""
        torch.manual_seed(42)
        B, H, L, D_k, D_v = 2, 4, 128, 32, 64
        q = torch.randn(B, H, L, D_k)
        k = torch.randn(B, H, L, D_k)
        v = torch.randn(B, H, L, D_v)
        beta = torch.sigmoid(torch.randn(B, H, L))
        g = -torch.abs(torch.randn(B, H, L))
        o = chunk_gated_delta_rule(q, k, v, beta, g, chunk_size=64)
        assert o.shape == (B, H, L, D_v)
        assert not torch.isnan(o).any()

    def test_recurrent_output_shape(self):
        """recurrent_gated_delta_rule should produce correct shapes."""
        torch.manual_seed(42)
        B, H, L, D_k, D_v = 1, 2, 32, 16, 32
        q = torch.randn(B, H, L, D_k)
        k = torch.randn(B, H, L, D_k)
        v = torch.randn(B, H, L, D_v)
        beta = torch.sigmoid(torch.randn(B, H, L))
        g = -torch.abs(torch.randn(B, H, L))
        o = recurrent_gated_delta_rule(q, k, v, beta, g)
        assert o.shape == (B, H, L, D_v)
        assert not torch.isnan(o).any()

    def test_chunk_vs_recurrent_consistency(self):
        """Chunked and recurrent should produce similar results."""
        torch.manual_seed(42)
        B, H, L, D_k, D_v = 1, 2, 64, 16, 32
        q = F.normalize(torch.randn(B, H, L, D_k), p=2, dim=-1)
        k = F.normalize(torch.randn(B, H, L, D_k), p=2, dim=-1)
        v = torch.randn(B, H, L, D_v) * 0.1
        beta = torch.sigmoid(torch.randn(B, H, L))
        g = -torch.abs(torch.randn(B, H, L)) * 0.1

        o_chunk = chunk_gated_delta_rule(q, k, v, beta, g, chunk_size=32)
        o_recur = recurrent_gated_delta_rule(q, k, v, beta, g)

        assert torch.allclose(o_chunk, o_recur, atol=1e-3, rtol=1e-2), \
            f"Max diff: {(o_chunk - o_recur).abs().max().item():.6f}"

    def test_chunk_seqlen_not_divisible(self):
        """Should handle seqlen not divisible by chunk_size."""
        torch.manual_seed(42)
        B, H, L, D_k, D_v = 1, 2, 100, 16, 32  # 100 not divisible by 64
        q = torch.randn(B, H, L, D_k)
        k = torch.randn(B, H, L, D_k)
        v = torch.randn(B, H, L, D_v)
        beta = torch.sigmoid(torch.randn(B, H, L))
        g = -torch.abs(torch.randn(B, H, L))
        o = chunk_gated_delta_rule(q, k, v, beta, g, chunk_size=64)
        assert o.shape == (B, H, L, D_v)


# ---------------------------------------------------------------------------
# Test 13: GatedDeltaNet module
# ---------------------------------------------------------------------------

class TestGatedDeltaNetModule:
    def test_forward_shape(self):
        """Basic forward pass produces correct output shape."""
        torch.manual_seed(42)
        model = GatedDeltaNet(d_model=128)
        x = torch.randn(2, 64, 128)
        y = model(x)
        assert y.shape == x.shape

    def test_forward_small(self):
        """Forward with small dimensions."""
        torch.manual_seed(42)
        model = GatedDeltaNet(d_model=64, num_heads=4, chunk_size=32)
        x = torch.randn(1, 64, 64)
        y = model(x)
        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_forward_no_mamba_gate(self):
        """Forward without Mamba-style gate (use logsigmoid gate)."""
        torch.manual_seed(42)
        model = GatedDeltaNet(d_model=64, num_heads=4, use_mamba_gate=False)
        x = torch.randn(1, 64, 64)
        y = model(x)
        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_backward(self):
        """All parameters should receive gradients."""
        torch.manual_seed(42)
        model = GatedDeltaNet(d_model=64, num_heads=4, chunk_size=32)
        x = torch.randn(1, 64, 64)
        y = model(x)
        loss = y.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_api_compatibility_with_mamba2_kwargs(self):
        """GatedDeltaNet should accept and ignore Mamba2-specific kwargs."""
        model = GatedDeltaNet(
            d_model=64, d_state=16, expand=2, headdim=16, ngroups=1,
        )
        x = torch.randn(1, 32, 64)
        y = model(x)
        assert y.shape == x.shape

    def test_seqlen_not_divisible(self):
        """Should handle seqlen not divisible by chunk_size."""
        model = GatedDeltaNet(d_model=64, num_heads=4, chunk_size=64)
        x = torch.randn(1, 100, 64)
        y = model(x)
        assert y.shape == (1, 100, 64)


# ---------------------------------------------------------------------------
# Test 14: GatedDeltaNet on MPS
# ---------------------------------------------------------------------------

class TestGatedDeltaNetMPS:
    @pytest.mark.skipif(
        not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
        reason="MPS not available"
    )
    def test_forward_on_mps(self):
        """GatedDeltaNet forward pass on MPS."""
        torch.manual_seed(42)
        device = torch.device("mps")
        model = GatedDeltaNet(d_model=64, num_heads=4, chunk_size=32).to(device)
        x = torch.randn(1, 64, 64, device=device)
        y = model(x)
        assert y.shape == x.shape
        assert y.device.type == "mps"
        assert not torch.isnan(y).any()

    @pytest.mark.skipif(
        not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
        reason="MPS not available"
    )
    def test_backward_on_mps(self):
        """GatedDeltaNet backward pass on MPS."""
        torch.manual_seed(42)
        device = torch.device("mps")
        model = GatedDeltaNet(d_model=64, num_heads=4, chunk_size=32).to(device)
        x = torch.randn(1, 64, 64, device=device)
        y = model(x)
        loss = y.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# Test 15: VisionMamba2 with layer_type="gated_delta_net"
# ---------------------------------------------------------------------------

GDN_KWARGS = dict(d_state=16, headdim=16, chunk_size=32)


class TestVisionGatedDeltaNet:
    @pytest.mark.parametrize("scan_mode", ["raster", "bidirectional", "cross_scan"])
    def test_image_forward_shape(self, scan_mode):
        """4D image input with GDN layer → same shape output."""
        torch.manual_seed(42)
        model = VisionMamba2(d_model=64, scan_mode=scan_mode,
                             layer_type="gated_delta_net", **GDN_KWARGS)
        x = torch.randn(1, 64, 8, 16)
        y = model(x)
        assert y.shape == x.shape, f"{scan_mode}: expected {x.shape}, got {y.shape}"

    @pytest.mark.parametrize("scan_mode", ["raster", "bidirectional", "cross_scan"])
    def test_image_backward(self, scan_mode):
        """All scan modes with GDN should support backward pass."""
        torch.manual_seed(42)
        model = VisionMamba2(d_model=64, scan_mode=scan_mode,
                             layer_type="gated_delta_net", **GDN_KWARGS)
        x = torch.randn(1, 64, 8, 16)
        y = model(x)
        loss = y.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{scan_mode}: no grad for {name}"
            assert not torch.isnan(param.grad).any(), f"{scan_mode}: NaN grad for {name}"

    @pytest.mark.parametrize("scan_mode", ["raster", "bidirectional", "cross_scan"])
    def test_no_nan(self, scan_mode):
        """GDN output should not contain NaN."""
        torch.manual_seed(42)
        model = VisionMamba2(d_model=64, scan_mode=scan_mode,
                             layer_type="gated_delta_net", **GDN_KWARGS)
        x = torch.randn(2, 64, 8, 16)
        y = model(x)
        assert not torch.isnan(y).any(), f"{scan_mode}: NaN in output"

    def test_gdn_vs_mamba2_differ(self):
        """GDN and Mamba2 should produce different outputs."""
        torch.manual_seed(42)
        x = torch.randn(1, 64, 8, 16)
        outputs = {}
        for lt in ["mamba2", "gated_delta_net"]:
            torch.manual_seed(0)
            model = VisionMamba2(d_model=64, scan_mode="raster",
                                 layer_type=lt, **GDN_KWARGS)
            with torch.no_grad():
                outputs[lt] = model(x)
        assert not torch.allclose(outputs["mamba2"], outputs["gated_delta_net"], atol=1e-3)


# ---------------------------------------------------------------------------
# Test 16: CrossModalMamba2 with layer_type="gated_delta_net"
# ---------------------------------------------------------------------------

class TestCrossModalGatedDeltaNet:
    @pytest.mark.parametrize("scan_mode", ["raster", "bidirectional", "cross_scan"])
    def test_image_forward_shape(self, scan_mode):
        """4D image inputs with GDN → same shape outputs."""
        torch.manual_seed(42)
        model = CrossModalMamba2(d_model=64, scan_mode=scan_mode,
                                 layer_type="gated_delta_net", **GDN_KWARGS)
        s = torch.randn(1, 64, 8, 16)
        f = torch.randn(1, 64, 8, 16)
        fs, ff = model(s, f)
        assert fs.shape == s.shape, f"{scan_mode}: semantic shape mismatch"
        assert ff.shape == f.shape, f"{scan_mode}: feature shape mismatch"

    @pytest.mark.parametrize("scan_mode", ["raster", "bidirectional", "cross_scan"])
    def test_backward(self, scan_mode):
        """Cross-modal with GDN should support backward pass."""
        torch.manual_seed(42)
        model = CrossModalMamba2(d_model=64, scan_mode=scan_mode,
                                 layer_type="gated_delta_net", **GDN_KWARGS)
        s = torch.randn(1, 64, 8, 16)
        f = torch.randn(1, 64, 8, 16)
        fs, ff = model(s, f)
        loss = fs.sum() + ff.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{scan_mode}: no grad for {name}"

    @pytest.mark.parametrize("scan_mode", ["raster", "bidirectional", "cross_scan"])
    def test_no_nan(self, scan_mode):
        """GDN cross-modal output should not contain NaN."""
        torch.manual_seed(42)
        model = CrossModalMamba2(d_model=64, scan_mode=scan_mode,
                                 layer_type="gated_delta_net", **GDN_KWARGS)
        s = torch.randn(2, 64, 8, 16)
        f = torch.randn(2, 64, 8, 16)
        fs, ff = model(s, f)
        assert not torch.isnan(fs).any(), f"{scan_mode}: NaN in fused_semantic"
        assert not torch.isnan(ff).any(), f"{scan_mode}: NaN in fused_features"


# ---------------------------------------------------------------------------
# Test 17: GatedDeltaNet Vision on MPS
# ---------------------------------------------------------------------------

class TestVisionGatedDeltaNetMPS:
    @pytest.mark.skipif(
        not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
        reason="MPS not available"
    )
    @pytest.mark.parametrize("scan_mode", ["raster", "bidirectional", "cross_scan"])
    def test_gdn_vision_forward_mps(self, scan_mode):
        """VisionMamba2 + GDN forward on MPS."""
        torch.manual_seed(42)
        device = torch.device("mps")
        model = VisionMamba2(d_model=64, scan_mode=scan_mode,
                             layer_type="gated_delta_net", **GDN_KWARGS).to(device)
        x = torch.randn(1, 64, 8, 16, device=device)
        y = model(x)
        assert y.shape == x.shape
        assert y.device.type == "mps"
        assert not torch.isnan(y).any()

    @pytest.mark.skipif(
        not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
        reason="MPS not available"
    )
    @pytest.mark.parametrize("scan_mode", ["raster", "bidirectional", "cross_scan"])
    def test_gdn_vision_backward_mps(self, scan_mode):
        """VisionMamba2 + GDN backward on MPS."""
        torch.manual_seed(42)
        device = torch.device("mps")
        model = VisionMamba2(d_model=64, scan_mode=scan_mode,
                             layer_type="gated_delta_net", **GDN_KWARGS).to(device)
        x = torch.randn(1, 64, 8, 16, device=device)
        y = model(x)
        y.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{scan_mode}: no grad for {name}"

    @pytest.mark.skipif(
        not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
        reason="MPS not available"
    )
    def test_gdn_cross_modal_mps(self):
        """CrossModalMamba2 + GDN on MPS."""
        torch.manual_seed(42)
        device = torch.device("mps")
        model = CrossModalMamba2(d_model=64, scan_mode="bidirectional",
                                 layer_type="gated_delta_net", **GDN_KWARGS).to(device)
        s = torch.randn(1, 64, 8, 16, device=device)
        f = torch.randn(1, 64, 8, 16, device=device)
        fs, ff = model(s, f)
        assert fs.shape == s.shape
        assert fs.device.type == "mps"
