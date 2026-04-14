"""Unit tests for noise-robustness mitigations (Exps 15-21).

Tests:
1. test_spectral_norm_project — modify m beyond ball, verify projection clips
2. test_swa_accumulator — accumulate 3 snapshots, verify average
3. test_cosine_warmup_lambda — verify 0→0.5→1.0 progression
4. test_magnitude_freeze_toggle — verify requires_grad switches
5. test_confidence_map_range — softmax confidence in [0, 1]
6. test_m_init_norm_buffer — verify buffer exists after construction
"""

import math
import torch
import torch.nn as nn

import sys
import os
import importlib.util

# Direct import of lora module to avoid cups/__init__.py dependency chain
_LORA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "cups", "model", "lora.py"
)
spec = importlib.util.spec_from_file_location("cups.model.lora", _LORA_PATH)
lora_module = importlib.util.module_from_spec(spec)
sys.modules["cups.model.lora"] = lora_module
spec.loader.exec_module(lora_module)

DoRALinear = lora_module.DoRALinear
LoRALinear = lora_module.LoRALinear
MitigationConfig = lora_module.MitigationConfig
SWAAccumulator = lora_module.SWAAccumulator
spectral_norm_project = lora_module.spectral_norm_project
cosine_warmup_lambda = lora_module.cosine_warmup_lambda


def test_spectral_norm_project():
    """M3: Modify m beyond norm ball, verify projection clips it back."""
    linear = nn.Linear(64, 32)
    dora = DoRALinear(linear, rank=4, alpha=4.0, dropout=0.0)

    init_norm = dora._m_init_norm.item()
    delta = 0.1
    max_allowed = init_norm * (1.0 + delta)

    # Artificially inflate magnitude to 2x initial norm
    dora.lora_magnitude.data.mul_(2.0)
    inflated_norm = dora.lora_magnitude.data.norm().item()
    assert inflated_norm > max_allowed, (
        f"Setup error: inflated norm {inflated_norm:.4f} should exceed "
        f"max allowed {max_allowed:.4f}"
    )

    # Wrap in a dummy model and project
    model = nn.Sequential(dora)
    count = spectral_norm_project(model, delta)

    projected_norm = dora.lora_magnitude.data.norm().item()
    assert count == 1, f"Expected 1 projection, got {count}"
    assert abs(projected_norm - max_allowed) < 1e-4, (
        f"Projected norm {projected_norm:.4f} != max allowed {max_allowed:.4f}"
    )
    print("PASS: spectral_norm_project clips magnitude to norm ball")


def test_spectral_norm_project_no_clip():
    """M3: When m is within ball, no projection should happen."""
    linear = nn.Linear(64, 32)
    dora = DoRALinear(linear, rank=4, alpha=4.0, dropout=0.0)

    original_norm = dora.lora_magnitude.data.norm().item()

    model = nn.Sequential(dora)
    count = spectral_norm_project(model, delta=0.1)

    after_norm = dora.lora_magnitude.data.norm().item()
    assert count == 0, f"Expected 0 projections, got {count}"
    assert abs(after_norm - original_norm) < 1e-6, "Norm should not change"
    print("PASS: spectral_norm_project skips adapters within ball")


def test_swa_accumulator():
    """M4: Accumulate 3 snapshots, verify averaged params."""
    linear = nn.Linear(64, 32)
    dora = DoRALinear(linear, rank=4, alpha=4.0, dropout=0.0)
    model = nn.Sequential(dora)

    swa = SWAAccumulator()

    # Snapshot 1: record initial state
    swa.update(model)
    snap1_mag = dora.lora_magnitude.data.clone()
    snap1_A = dora.lora_A.data.clone()

    # Snapshot 2: perturb and record
    dora.lora_magnitude.data.add_(1.0)
    dora.lora_A.data.add_(0.5)
    swa.update(model)
    snap2_mag = dora.lora_magnitude.data.clone()
    snap2_A = dora.lora_A.data.clone()

    # Snapshot 3: perturb again and record
    dora.lora_magnitude.data.add_(2.0)
    dora.lora_A.data.add_(1.0)
    swa.update(model)
    snap3_mag = dora.lora_magnitude.data.clone()
    snap3_A = dora.lora_A.data.clone()

    assert swa.count == 3, f"Expected 3 snapshots, got {swa.count}"

    # Apply averaged params
    n_updated = swa.apply(model)
    assert n_updated > 0, "Expected some params to be updated"

    # Verify magnitude is average of 3 snapshots
    expected_mag = (snap1_mag + snap2_mag + snap3_mag) / 3.0
    assert torch.allclose(dora.lora_magnitude.data, expected_mag, atol=1e-5), (
        "Magnitude not correctly averaged"
    )

    # Verify lora_A is average of 3 snapshots
    expected_A = (snap1_A + snap2_A + snap3_A) / 3.0
    assert torch.allclose(dora.lora_A.data, expected_A, atol=1e-5), (
        "lora_A not correctly averaged"
    )

    # Reset and verify clean state
    swa.reset()
    assert swa.count == 0, "Count should be 0 after reset"
    print("PASS: SWAAccumulator correctly averages 3 snapshots")


def test_cosine_warmup_lambda():
    """M1: Verify cosine warmup shape: 0 at start, ~0.5 at midpoint, 1.0 at end."""
    warmup_steps = 500

    # Step 0: should be 0
    val_0 = cosine_warmup_lambda(0, warmup_steps, is_lora_group=True)
    assert abs(val_0) < 1e-6, f"Step 0 should be ~0, got {val_0}"

    # Midpoint: should be ~0.5
    val_mid = cosine_warmup_lambda(250, warmup_steps, is_lora_group=True)
    assert abs(val_mid - 0.5) < 0.01, f"Midpoint should be ~0.5, got {val_mid}"

    # End: should be 1.0
    val_end = cosine_warmup_lambda(500, warmup_steps, is_lora_group=True)
    assert abs(val_end - 1.0) < 1e-6, f"End should be 1.0, got {val_end}"

    # Past end: should stay 1.0
    val_past = cosine_warmup_lambda(1000, warmup_steps, is_lora_group=True)
    assert abs(val_past - 1.0) < 1e-6, f"Past end should be 1.0, got {val_past}"

    # Non-LoRA group: always 1.0
    val_head = cosine_warmup_lambda(0, warmup_steps, is_lora_group=False)
    assert abs(val_head - 1.0) < 1e-6, f"Head group should be 1.0, got {val_head}"

    # Monotonic increase check
    prev = 0.0
    for s in range(0, warmup_steps + 1, 10):
        val = cosine_warmup_lambda(s, warmup_steps, is_lora_group=True)
        assert val >= prev - 1e-8, (
            f"Not monotonic: step {s} val {val} < prev {prev}"
        )
        prev = val

    print("PASS: cosine_warmup_lambda has correct shape")


def test_magnitude_freeze_toggle():
    """M2: Verify requires_grad toggles correctly on lora_magnitude."""
    linear = nn.Linear(64, 32)
    dora = DoRALinear(linear, rank=4, alpha=4.0, dropout=0.0)

    # Initially trainable
    assert dora.lora_magnitude.requires_grad is True, (
        "lora_magnitude should start trainable"
    )

    # Freeze
    dora.lora_magnitude.requires_grad_(False)
    assert dora.lora_magnitude.requires_grad is False, (
        "lora_magnitude should be frozen after toggle"
    )

    # Verify other params still trainable
    assert dora.lora_A.requires_grad is True, "lora_A should remain trainable"
    assert dora.lora_B.requires_grad is True, "lora_B should remain trainable"

    # Forward still works with frozen magnitude
    dora.eval()
    x = torch.randn(1, 10, 64)
    y = dora(x)
    assert y.shape == (1, 10, 32), f"Shape mismatch after freeze: {y.shape}"

    # Unfreeze
    dora.lora_magnitude.requires_grad_(True)
    assert dora.lora_magnitude.requires_grad is True, (
        "lora_magnitude should be unfrozen after second toggle"
    )

    # Gradient flows after unfreeze
    dora.train()
    x = torch.randn(1, 10, 64)
    y = dora(x)
    loss = y.sum()
    loss.backward()
    assert dora.lora_magnitude.grad is not None, (
        "lora_magnitude should have gradients after unfreeze"
    )

    print("PASS: magnitude freeze/unfreeze toggle works correctly")


def test_confidence_map_range():
    """M5: Verify softmax confidence map is in [0, 1] and clamp works."""
    # Simulate teacher logits (C=19 classes, H=8, W=16)
    logits = torch.randn(19, 8, 16)
    temperature = 1.0
    min_weight = 0.1

    # Compute confidence map (same logic as make_pseudo_labels)
    probs = (logits / temperature).softmax(dim=0)  # (C, H, W)
    confidence = probs.max(dim=0).values  # (H, W)
    confidence_weights = confidence.clamp(min=min_weight)

    # All values in [min_weight, 1.0]
    assert confidence_weights.min().item() >= min_weight - 1e-6, (
        f"Min weight {confidence_weights.min().item()} < min_weight {min_weight}"
    )
    assert confidence_weights.max().item() <= 1.0 + 1e-6, (
        f"Max weight {confidence_weights.max().item()} > 1.0"
    )

    # Shape check
    assert confidence_weights.shape == (8, 16), (
        f"Expected (8, 16), got {confidence_weights.shape}"
    )

    # High temperature -> more uniform -> lower confidence
    conf_high_t = (logits / 10.0).softmax(dim=0).max(dim=0).values
    conf_low_t = (logits / 0.1).softmax(dim=0).max(dim=0).values
    assert conf_high_t.mean() < conf_low_t.mean(), (
        "Higher temperature should produce lower confidence"
    )

    print("PASS: confidence map in valid range with correct temperature behavior")


def test_m_init_norm_buffer():
    """Verify _m_init_norm buffer exists and has correct value after construction."""
    linear = nn.Linear(128, 64)
    dora = DoRALinear(linear, rank=4, alpha=4.0, dropout=0.0)

    # Buffer should exist
    assert hasattr(dora, "_m_init_norm"), "Missing _m_init_norm buffer"

    # Should be a tensor (not a Parameter)
    assert isinstance(dora._m_init_norm, torch.Tensor), (
        f"Expected Tensor, got {type(dora._m_init_norm)}"
    )
    assert not dora._m_init_norm.requires_grad, (
        "_m_init_norm should not require gradients (it's a buffer)"
    )

    # Value should match initial magnitude norm
    expected_norm = dora.lora_magnitude.data.norm().item()
    actual_norm = dora._m_init_norm.item()
    assert abs(actual_norm - expected_norm) < 1e-5, (
        f"Buffer norm {actual_norm:.6f} != magnitude norm {expected_norm:.6f}"
    )

    # Should be positive (weights are non-degenerate)
    assert actual_norm > 0, f"Buffer norm should be positive, got {actual_norm}"

    # Should be a scalar
    assert dora._m_init_norm.dim() == 0, (
        f"Expected scalar, got dim={dora._m_init_norm.dim()}"
    )

    # Should be listed in state_dict (buffers are)
    sd = dora.state_dict()
    assert "_m_init_norm" in sd, "_m_init_norm should appear in state_dict"

    print("PASS: _m_init_norm buffer exists with correct initial value")


def test_mitigation_config_defaults():
    """MitigationConfig defaults: all mitigations OFF."""
    cfg = MitigationConfig()

    assert cfg.cosine_warmup_enabled is False
    assert cfg.magnitude_warmup_enabled is False
    assert cfg.spectral_norm_ball_enabled is False
    assert cfg.swa_enabled is False
    assert cfg.confidence_weighted_loss_enabled is False
    assert cfg.adaptive_delayed_start_enabled is False

    # Verify key default values
    assert cfg.cosine_warmup_steps == 500
    assert cfg.magnitude_warmup_freeze_steps == 200
    assert cfg.spectral_norm_ball_delta == 0.1
    assert cfg.swa_fraction == 0.3
    assert cfg.confidence_weighted_loss_temperature == 1.0
    assert cfg.confidence_weighted_loss_min_weight == 0.1
    assert cfg.adaptive_delayed_start_tau == 0.7
    assert cfg.adaptive_delayed_start_max_wait == 1000

    print("PASS: MitigationConfig defaults all OFF with correct hyperparameters")


if __name__ == "__main__":
    test_spectral_norm_project()
    test_spectral_norm_project_no_clip()
    test_swa_accumulator()
    test_cosine_warmup_lambda()
    test_magnitude_freeze_toggle()
    test_confidence_map_range()
    test_m_init_norm_buffer()
    test_mitigation_config_defaults()
    print("\n=== All 8 mitigation tests passed ===")
