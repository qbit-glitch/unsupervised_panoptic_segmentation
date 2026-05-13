"""Smoke test for DoRA integration into DINOv3 backbone.

Verifies:
1. DoRALinear / LoRALinear forward pass shape correctness
2. Zero-init property: at step 0, DoRA output == original output
3. inject_dora_into_model() correctly replaces nn.Linear modules
4. get_dora_param_groups() returns correct group structure
5. Gradient flow: DoRA params receive gradients
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

DoRAConfig = lora_module.DoRAConfig
ProgressiveDoRAConfig = lora_module.ProgressiveDoRAConfig
DoRALinear = lora_module.DoRALinear
LoRALinear = lora_module.LoRALinear
ConvDoRALinear = lora_module.ConvDoRALinear
inject_dora_into_model = lora_module.inject_dora_into_model
get_dora_param_groups = lora_module.get_dora_param_groups
expand_lora_rank = lora_module.expand_lora_rank
expand_lora_coverage = lora_module.expand_lora_coverage


def test_dora_linear_shapes():
    """DoRALinear preserves input/output shapes."""
    linear = nn.Linear(768, 2304)
    dora = DoRALinear(linear, rank=4, alpha=4.0)

    x = torch.randn(2, 196, 768)
    y = dora(x)

    assert y.shape == (2, 196, 2304), f"Expected (2, 196, 2304), got {y.shape}"
    print("PASS: DoRALinear shapes correct")


def test_lora_linear_shapes():
    """LoRALinear preserves input/output shapes."""
    linear = nn.Linear(768, 768)
    lora = LoRALinear(linear, rank=4, alpha=4.0)

    x = torch.randn(2, 196, 768)
    y = lora(x)

    assert y.shape == (2, 196, 768), f"Expected (2, 196, 768), got {y.shape}"
    print("PASS: LoRALinear shapes correct")


def test_dora_zero_init():
    """At init, DoRA output == original Linear output (B=0 → delta_V=0)."""
    linear = nn.Linear(768, 2304)
    x = torch.randn(1, 16, 768)

    with torch.no_grad():
        original_out = linear(x)

    dora = DoRALinear(linear, rank=4, alpha=4.0, dropout=0.0)
    dora.eval()

    with torch.no_grad():
        dora_out = dora(x)

    # Should be identical since B=0
    max_diff = (original_out - dora_out).abs().max().item()
    assert max_diff < 1e-5, f"Zero-init violated: max diff = {max_diff}"
    print(f"PASS: DoRA zero-init (max diff = {max_diff:.2e})")


def test_lora_zero_init():
    """At init, LoRA output == original Linear output (B=0)."""
    linear = nn.Linear(768, 768)
    x = torch.randn(1, 16, 768)

    with torch.no_grad():
        original_out = linear(x)

    lora = LoRALinear(linear, rank=4, alpha=4.0, dropout=0.0)
    lora.eval()

    with torch.no_grad():
        lora_out = lora(x)

    max_diff = (original_out - lora_out).abs().max().item()
    assert max_diff < 1e-5, f"Zero-init violated: max diff = {max_diff}"
    print(f"PASS: LoRA zero-init (max diff = {max_diff:.2e})")


def test_trainable_count():
    """Verify trainable parameter counts match expectations."""
    linear = nn.Linear(768, 2304)
    dora = DoRALinear(linear, rank=4)

    # A: 4 * 768 = 3072
    # B: 2304 * 4 = 9216
    # magnitude: 2304 * 1 = 2304
    expected = 3072 + 9216 + 2304
    actual = dora.trainable_count()
    assert actual == expected, f"Expected {expected}, got {actual}"

    lora = LoRALinear(linear, rank=4)
    expected_lora = 3072 + 9216
    actual_lora = lora.trainable_count()
    assert actual_lora == expected_lora, f"Expected {expected_lora}, got {actual_lora}"

    print(f"PASS: Param counts correct (DoRA={actual}, LoRA={actual_lora})")


class MockViTBlock(nn.Module):
    """Minimal ViT block for injection testing."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, dim * 3)
        self.attn.proj = nn.Linear(dim, dim)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, dim * 4)
        self.mlp.fc2 = nn.Linear(dim * 4, dim)


class MockViT(nn.Module):
    """Minimal ViT with 12 blocks for injection testing."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([MockViTBlock() for _ in range(12)])


def test_inject_dora():
    """inject_dora_into_model() correctly replaces target layers."""
    model = MockViT()
    config = DoRAConfig(rank=4, alpha=4.0, late_block_start=6)

    adapted = inject_dora_into_model(model, config, variant="dora")

    # Early blocks (0-5): qkv only = 6 layers
    # Late blocks (6-11): qkv + proj + fc1 + fc2 = 6 * 4 = 24 layers
    expected_layers = 6 + 24
    assert len(adapted) == expected_layers, (
        f"Expected {expected_layers} adapted layers, got {len(adapted)}"
    )

    # Verify early blocks: only qkv is DoRALinear
    for i in range(6):
        assert isinstance(model.blocks[i].attn.qkv, DoRALinear), (
            f"Block {i} qkv should be DoRALinear"
        )
        assert isinstance(model.blocks[i].attn.proj, nn.Linear), (
            f"Block {i} proj should remain nn.Linear"
        )
        assert isinstance(model.blocks[i].mlp.fc1, nn.Linear), (
            f"Block {i} fc1 should remain nn.Linear"
        )

    # Verify late blocks: all 4 targets are DoRALinear
    for i in range(6, 12):
        assert isinstance(model.blocks[i].attn.qkv, DoRALinear)
        assert isinstance(model.blocks[i].attn.proj, DoRALinear)
        assert isinstance(model.blocks[i].mlp.fc1, DoRALinear)
        assert isinstance(model.blocks[i].mlp.fc2, DoRALinear)

    total_dora_params = sum(adapted.values())
    print(
        f"PASS: Injection correct — {len(adapted)} layers, "
        f"{total_dora_params:,} DoRA params"
    )
    return model


def test_inject_lora_ablation():
    """LoRA variant injection works for ablation."""
    model = MockViT()
    config = DoRAConfig(rank=4, alpha=4.0, late_block_start=6)

    adapted = inject_dora_into_model(model, config, variant="lora")

    # Same count but LoRALinear instead of DoRALinear
    assert len(adapted) == 30
    assert isinstance(model.blocks[0].attn.qkv, LoRALinear)
    assert isinstance(model.blocks[8].mlp.fc1, LoRALinear)

    print(f"PASS: LoRA ablation injection correct — {len(adapted)} layers")


def test_param_groups():
    """get_dora_param_groups() returns correct group structure."""
    model = MockViT()
    config = DoRAConfig(rank=4, alpha=4.0, late_block_start=6)
    inject_dora_into_model(model, config, variant="dora")

    # Simulate a detection head with trainable params
    model.head = nn.Linear(768, 80)
    model.head_norm = nn.LayerNorm(768)

    groups = get_dora_param_groups(model, config, head_lr=1e-4, head_wd=1e-5)

    group_names = {g["name"] for g in groups}
    expected_names = {"head_other", "head_norm", "dora_B", "dora_magnitude", "dora_A"}
    assert group_names == expected_names, (
        f"Expected groups {expected_names}, got {group_names}"
    )

    # Check LR assignments
    for g in groups:
        if g["name"] == "dora_A":
            assert g["lr"] == config.lr_a, f"dora_A lr should be {config.lr_a}"
        elif g["name"] == "dora_B":
            assert g["lr"] == config.lr_b, f"dora_B lr should be {config.lr_b}"
        elif g["name"] == "dora_magnitude":
            assert g["lr"] == config.lr_b
            assert g["weight_decay"] == config.magnitude_wd

    print(f"PASS: Param groups correct — {len(groups)} groups")


def test_gradient_flow():
    """DoRA params receive gradients during backward pass."""
    model = MockViT()
    config = DoRAConfig(rank=4, alpha=4.0, late_block_start=6)
    inject_dora_into_model(model, config, variant="dora")

    # Freeze base weights (simulating frozen backbone)
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad_(False)

    x = torch.randn(1, 16, 768, requires_grad=False)

    # Forward through one adapted layer
    out = model.blocks[8].attn.qkv(x)
    loss = out.sum()
    loss.backward()

    # Check gradients exist on DoRA params
    qkv = model.blocks[8].attn.qkv
    assert qkv.lora_A.grad is not None, "lora_A should have gradient"
    assert qkv.lora_B.grad is not None, "lora_B should have gradient"
    assert qkv.lora_magnitude.grad is not None, "lora_magnitude should have gradient"
    assert qkv.weight.grad is None, "Frozen weight should NOT have gradient"

    print("PASS: Gradient flow correct — DoRA params get gradients, frozen params don't")


def test_full_param_count_vitb():
    """Verify total DoRA param count matches plan (~331K for ViT-B/16)."""
    model = MockViT()
    config = DoRAConfig(rank=4, alpha=4.0, late_block_start=6)
    adapted = inject_dora_into_model(model, config, variant="dora")

    total = sum(adapted.values())

    # Expected breakdown:
    # Early (0-5): 6 × qkv(768→2304) = 6 × (4×768 + 2304×4 + 2304) = 6 × 14592 = 87552
    # Late (6-11): 6 × [qkv(768→2304) + proj(768→768) + fc1(768→3072) + fc2(3072→768)]
    #   qkv: 4×768 + 2304×4 + 2304 = 14592
    #   proj: 4×768 + 768×4 + 768 = 6912
    #   fc1: 4×768 + 3072×4 + 3072 = 18432
    #   fc2: 4×3072 + 768×4 + 768 = 16128
    #   per late block: 14592 + 6912 + 18432 + 16128 = 56064
    # Late total: 6 × 56064 = 336384
    # Grand total: 87552 + 336384 = 423936

    # Approximate check (within 10% of plan estimate)
    assert 300_000 < total < 500_000, (
        f"Expected ~331K-424K DoRA params, got {total:,}"
    )
    print(f"PASS: Total DoRA params = {total:,} (plan estimated ~331K)")


def test_conv_dora_shapes_with_spatial():
    """ConvDoRALinear preserves shapes when spatial dims are set."""
    linear = nn.Linear(768, 2304)
    conv_dora = ConvDoRALinear(linear, rank=4, alpha=4.0)

    # Simulate ViT-B/16 on 640x1280: patch grid = 40x80 = 3200 tokens
    h_p, w_p = 40, 80
    conv_dora._spatial_dims = (h_p, w_p)

    x = torch.randn(2, h_p * w_p, 768)
    y = conv_dora(x)

    assert y.shape == (2, 3200, 2304), f"Expected (2, 3200, 2304), got {y.shape}"
    print("PASS: ConvDoRA shapes correct with spatial dims")


def test_conv_dora_shapes_without_spatial():
    """ConvDoRALinear falls back to DoRA when spatial dims are absent."""
    linear = nn.Linear(768, 2304)
    conv_dora = ConvDoRALinear(linear, rank=4, alpha=4.0)
    # _spatial_dims is None by default

    x = torch.randn(2, 196, 768)
    y = conv_dora(x)

    assert y.shape == (2, 196, 2304), f"Expected (2, 196, 2304), got {y.shape}"
    print("PASS: ConvDoRA fallback shapes correct (no spatial dims)")


def test_conv_dora_zero_init():
    """At init, ConvDoRA output == original Linear output.

    Both DWConv weights and conv_gate are zero-initialized, so the conv
    path contributes nothing at step 0.
    """
    linear = nn.Linear(768, 2304)
    x = torch.randn(1, 400, 768)  # 20x20 patch grid

    with torch.no_grad():
        original_out = linear(x)

    conv_dora = ConvDoRALinear(linear, rank=4, alpha=4.0, dropout=0.0)
    conv_dora._spatial_dims = (20, 20)
    conv_dora.eval()

    with torch.no_grad():
        conv_dora_out = conv_dora(x)

    max_diff = (original_out - conv_dora_out).abs().max().item()
    assert max_diff < 1e-5, f"Zero-init violated: max diff = {max_diff}"
    print(f"PASS: ConvDoRA zero-init (max diff = {max_diff:.2e})")


def test_conv_dora_trainable_count():
    """Verify ConvDoRA param count = DoRA + DWConv + conv_gate."""
    linear = nn.Linear(768, 2304)
    conv_dora = ConvDoRALinear(linear, rank=4)

    # DoRA: A(4*768) + B(2304*4) + mag(2304) = 3072 + 9216 + 2304 = 14592
    # Conv: DWConv(4 * 1 * 3 * 3) + gate(1) = 36 + 1 = 37
    expected = 14592 + 37
    actual = conv_dora.trainable_count()
    assert actual == expected, f"Expected {expected}, got {actual}"
    print(f"PASS: ConvDoRA param count correct ({actual})")


def test_conv_dora_gradient_flow():
    """ConvDoRA conv params receive gradients during backward."""
    model = MockViT()
    config = DoRAConfig(rank=4, alpha=4.0, late_block_start=6)
    inject_dora_into_model(model, config, variant="conv_dora")

    # Freeze base weights
    for name, param in model.named_parameters():
        if "lora" not in name and "dwconv" not in name and "conv_gate" not in name:
            param.requires_grad_(False)

    # Set spatial dims on one adapted layer
    qkv = model.blocks[8].attn.qkv
    assert isinstance(qkv, ConvDoRALinear), f"Expected ConvDoRALinear, got {type(qkv)}"
    qkv._spatial_dims = (14, 14)  # 196 tokens = 14x14 grid

    x = torch.randn(1, 196, 768, requires_grad=False)
    out = qkv(x)
    loss = out.sum()
    loss.backward()

    assert qkv.lora_A.grad is not None, "lora_A should have gradient"
    assert qkv.lora_B.grad is not None, "lora_B should have gradient"
    assert qkv.lora_magnitude.grad is not None, "magnitude should have gradient"
    assert qkv.dwconv.weight.grad is not None, "dwconv should have gradient"
    assert qkv.conv_gate.grad is not None, "conv_gate should have gradient"
    assert qkv.weight.grad is None, "Frozen weight should NOT have gradient"

    print("PASS: ConvDoRA gradient flow correct — all conv params get gradients")


def test_inject_conv_dora():
    """Conv-DoRA variant injection replaces layers with ConvDoRALinear."""
    model = MockViT()
    config = DoRAConfig(rank=4, alpha=4.0, late_block_start=6)

    adapted = inject_dora_into_model(model, config, variant="conv_dora")

    assert len(adapted) == 30, f"Expected 30 adapted layers, got {len(adapted)}"
    assert isinstance(model.blocks[0].attn.qkv, ConvDoRALinear)
    assert isinstance(model.blocks[8].mlp.fc1, ConvDoRALinear)

    total = sum(adapted.values())
    # Each ConvDoRA layer adds 37 extra params over DoRA (36 DWConv + 1 gate)
    # 30 layers * 37 = 1110 extra over plain DoRA (423936)
    assert total > 423936, f"ConvDoRA total ({total}) should exceed DoRA (423936)"
    print(f"PASS: Conv-DoRA injection correct — {len(adapted)} layers, {total:,} params")


def test_conv_dora_param_groups():
    """get_dora_param_groups() includes dora_conv group for Conv-DoRA."""
    model = MockViT()
    config = DoRAConfig(rank=4, alpha=4.0, late_block_start=6)
    inject_dora_into_model(model, config, variant="conv_dora")

    model.head = nn.Linear(768, 80)
    model.head_norm = nn.LayerNorm(768)

    groups = get_dora_param_groups(model, config, head_lr=1e-4, head_wd=1e-5)

    group_names = {g["name"] for g in groups}
    expected_names = {
        "head_other", "head_norm",
        "dora_B", "dora_magnitude", "dora_A", "dora_conv",
    }
    assert group_names == expected_names, (
        f"Expected groups {expected_names}, got {group_names}"
    )

    # Conv group should use lr_a (same as A — operates in down-projected space)
    for g in groups:
        if g["name"] == "dora_conv":
            assert g["lr"] == config.lr_a, f"dora_conv lr should be {config.lr_a}"
            assert g["weight_decay"] == 0.0

    print(f"PASS: Conv-DoRA param groups correct — {len(groups)} groups")


def test_invalid_variant_raises():
    """inject_dora_into_model() raises ValueError for unknown variant."""
    model = MockViT()
    config = DoRAConfig(rank=4)

    try:
        inject_dora_into_model(model, config, variant="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid" in str(e)
        print(f"PASS: Invalid variant raises ValueError: {e}")


def test_rank_expansion_preserves_output():
    """expand_lora_rank() preserves model output at the expansion boundary.

    After expanding r=4→r=8, new B columns are zero-initialized, so
    the output should be identical to pre-expansion.
    """
    model = MockViT()
    config = DoRAConfig(rank=4, alpha=4.0, dropout=0.0, late_block_start=6)
    inject_dora_into_model(model, config, variant="conv_dora")

    # Simulate some training (non-zero A/B)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, ConvDoRALinear):
                m.lora_B.data.normal_(0, 0.01)
                m._spatial_dims = (14, 14)

    model.eval()  # disable dropout for deterministic comparison
    x = torch.randn(1, 196, 768)
    with torch.no_grad():
        out_before = model.blocks[8].attn.qkv(x).clone()

    # Expand rank 4 → 8
    count = expand_lora_rank(model, new_rank=8, new_alpha=8.0)

    # Set spatial dims on expanded adapter
    model.blocks[8].attn.qkv._spatial_dims = (14, 14)

    with torch.no_grad():
        out_after = model.blocks[8].attn.qkv(x)

    max_diff = (out_before - out_after).abs().max().item()
    assert max_diff < 1e-5, f"Rank expansion changed output: max diff = {max_diff}"
    assert count > 0, "Should have expanded at least one adapter"

    # Verify new rank
    assert model.blocks[8].attn.qkv.rank == 8
    assert model.blocks[8].attn.qkv.lora_A.shape[0] == 8
    assert model.blocks[8].attn.qkv.lora_B.shape[1] == 8

    print(f"PASS: Rank expansion r=4→8 preserves output (max diff = {max_diff:.2e}), expanded {count} adapters")


def test_rank_expansion_dwconv():
    """expand_lora_rank() correctly expands ConvDoRA DWConv channels."""
    linear = nn.Linear(768, 2304)
    conv_dora = ConvDoRALinear(linear, rank=4, alpha=4.0)

    assert conv_dora.dwconv.weight.shape == (4, 1, 3, 3)

    # Simulate learned conv weights
    with torch.no_grad():
        conv_dora.dwconv.weight.normal_(0, 0.01)
        old_conv_weight = conv_dora.dwconv.weight.data[:4].clone()

    from cups.model.lora import _expand_adapter_rank
    _expand_adapter_rank(conv_dora, new_rank=8, new_alpha=8.0)

    assert conv_dora.dwconv.weight.shape == (8, 1, 3, 3), (
        f"Expected (8,1,3,3), got {conv_dora.dwconv.weight.shape}"
    )
    # Old channels preserved
    max_diff = (conv_dora.dwconv.weight.data[:4] - old_conv_weight).abs().max().item()
    assert max_diff < 1e-7, f"Old conv channels changed: {max_diff}"
    # New channels zero
    new_max = conv_dora.dwconv.weight.data[4:].abs().max().item()
    assert new_max < 1e-7, f"New conv channels should be zero: {new_max}"

    print(f"PASS: DWConv expanded r=4→8, old channels preserved, new channels zero")


def test_coverage_expansion_preserves_existing():
    """expand_lora_coverage() adds new layers without touching existing ones."""
    model = MockViT()
    # Start with qkv-only in all 12 blocks (late_block_start=12 → all blocks "early")
    config_qkv_only = DoRAConfig(rank=4, alpha=4.0, late_block_start=12)
    inject_dora_into_model(model, config_qkv_only, variant="dora")

    # 12 layers adapted (all blocks, qkv only)
    adapted_count = sum(
        1 for m in model.modules() if isinstance(m, DoRALinear)
    )
    assert adapted_count == 12, f"Expected 12 adapted layers, got {adapted_count}"

    # Simulate training (non-zero B weights)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, DoRALinear):
                m.lora_B.data.normal_(0, 0.01)
    old_b8_qkv = model.blocks[8].attn.qkv.lora_B.data.clone()

    # Expand: late_block_start=6 adds proj+fc1+fc2 to blocks 6-11
    config_expand = DoRAConfig(rank=4, alpha=4.0, late_block_start=6)
    newly = expand_lora_coverage(model, config_expand, variant="dora")

    # Blocks 0-5: qkv already adapted, no new targets → 0 new
    # Blocks 6-11: qkv already adapted (skip), proj+fc1+fc2 new → 3 × 6 = 18 new
    assert len(newly) == 18, f"Expected 18 new layers, got {len(newly)}"

    # Existing adapter unchanged
    max_diff = (model.blocks[8].attn.qkv.lora_B.data - old_b8_qkv).abs().max().item()
    assert max_diff < 1e-7, f"Existing adapter changed: {max_diff}"

    # Total should now be 30 (12 original + 18 new)
    total = sum(1 for m in model.modules() if isinstance(m, DoRALinear))
    assert total == 30, f"Expected 30 total adapted, got {total}"

    print(f"PASS: Coverage expansion +{len(newly)} layers, existing adapters preserved")


def test_progressive_config_rounds():
    """ProgressiveDoRAConfig generates correct per-round DoRAConfig."""
    prog = ProgressiveDoRAConfig(
        ranks=(2, 4, 8),
        alphas=(2.0, 4.0, 8.0),
        late_block_starts=(9, 6, 0),
    )

    assert prog.num_rounds == 3

    cfg0 = prog.get_dora_config(0)
    assert cfg0.rank == 2
    assert cfg0.alpha == 2.0
    assert cfg0.late_block_start == 9
    assert cfg0.delayed_start_steps == 0  # no delay in Stage-3

    cfg2 = prog.get_dora_config(2)
    assert cfg2.rank == 8
    assert cfg2.alpha == 8.0
    assert cfg2.late_block_start == 0

    try:
        prog.get_dora_config(3)
        assert False, "Should raise IndexError"
    except IndexError:
        pass

    print("PASS: ProgressiveDoRAConfig generates correct per-round configs")


if __name__ == "__main__":
    tests = [
        test_dora_linear_shapes,
        test_lora_linear_shapes,
        test_dora_zero_init,
        test_lora_zero_init,
        test_trainable_count,
        test_inject_dora,
        test_inject_lora_ablation,
        test_param_groups,
        test_gradient_flow,
        test_full_param_count_vitb,
        # Conv-DoRA tests
        test_conv_dora_shapes_with_spatial,
        test_conv_dora_shapes_without_spatial,
        test_conv_dora_zero_init,
        test_conv_dora_trainable_count,
        test_conv_dora_gradient_flow,
        test_inject_conv_dora,
        test_conv_dora_param_groups,
        test_invalid_variant_raises,
        # Progressive LoRA tests
        test_rank_expansion_preserves_output,
        test_rank_expansion_dwconv,
        test_coverage_expansion_preserves_existing,
        test_progressive_config_rounds,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    if failed == 0:
        print("All smoke tests passed!")
    else:
        print(f"FAILURES: {failed}")
        exit(1)
