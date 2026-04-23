"""Comprehensive test suite for depth adapter architectures.

Verifies three depth adapter systems match their architecture specs:
  - DepthPro: Architecture-DepthPro-Adapters.md
  - DA2:      Architecture-DA2-Large-Adapters.md
  - DA3:      Architecture-DA3-Adapters.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import copy
import tempfile
import torch
import torch.nn as nn

from mbps_pytorch.models.adapters import (
    inject_lora_into_depth_model,
    inject_lora_into_depthpro,
    freeze_non_adapter_params,
    count_adapter_params,
)
from mbps_pytorch.models.adapters.lora_layers import DoRALinear

# --------------------------------------------------------------------------- #
# Mock HuggingFace-style Dinov2Layer (for DepthPro + DA2)
# --------------------------------------------------------------------------- #


class MockDinov2Attention(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)


class MockDinov2Output(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.dense = nn.Linear(dim, dim)


class MockDinov2MLP(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)


class MockDinov2Layer(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.attention = nn.Module()
        self.attention.attention = MockDinov2Attention(dim)
        self.attention.output = MockDinov2Output(dim)
        self.mlp = MockDinov2MLP(dim)


class MockHFDinov2Model(nn.Module):
    def __init__(self, num_layers=24, dim=1024):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([MockDinov2Layer(dim) for _ in range(num_layers)])


# --------------------------------------------------------------------------- #
# Mock DepthPro
# --------------------------------------------------------------------------- #


class MockDepthPro(nn.Module):
    def __init__(self, num_layers=24, dim=1024):
        super().__init__()
        self.depth_pro = nn.Module()
        self.depth_pro.encoder = nn.Module()
        self.depth_pro.encoder.patch_encoder = nn.Module()
        self.depth_pro.encoder.patch_encoder.model = MockHFDinov2Model(num_layers, dim)
        self.depth_pro.encoder.image_encoder = nn.Module()
        self.depth_pro.encoder.image_encoder.model = MockHFDinov2Model(num_layers, dim)
        self.fov_model = nn.Module()
        self.fov_model.fov_encoder = nn.Module()
        self.fov_model.fov_encoder.model = MockHFDinov2Model(num_layers, dim)


# --------------------------------------------------------------------------- #
# Mock DA2 (HF Depth Anything V2)
# --------------------------------------------------------------------------- #


class MockDA2Model(nn.Module):
    def __init__(self, num_layers=24, dim=1024):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.encoder = nn.Module()
        self.backbone.encoder.layer = nn.ModuleList([MockDinov2Layer(dim) for _ in range(num_layers)])


# --------------------------------------------------------------------------- #
# Mock DA3 (Custom ViT)
# --------------------------------------------------------------------------- #


class MockCustomBlock(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.attn_q = nn.Linear(dim, dim)
        self.attn_k = nn.Linear(dim, dim)
        self.attn_v = nn.Linear(dim, dim)
        self.attn_proj = nn.Linear(dim, dim)
        self.mlp_fc1 = nn.Linear(dim, dim * 4)
        self.mlp_fc2 = nn.Linear(dim * 4, dim)


class MockDA3Model(nn.Module):
    """Mock DA3 with a non-standard path so structured block discovery fails.

    Uses ``custom_blocks`` instead of ``blocks`` so
    ``_find_encoder_blocks`` returns empty and generic fallback is triggered.
    """

    def __init__(self, num_blocks=24, dim=1024):
        super().__init__()
        # Intentionally NOT named "blocks" so structured finder skips it
        self.custom_blocks = nn.ModuleList([MockCustomBlock(dim) for _ in range(num_blocks)])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _is_dora(module):
    return isinstance(module, DoRALinear)


def _is_plain_linear(module):
    return type(module) is nn.Linear


# --------------------------------------------------------------------------- #
# Test 1: DepthPro Adapter Injection
# --------------------------------------------------------------------------- #


def test_depthpro_adapter_injection():
    print("\n[Test 1] DepthPro Adapter Injection")
    model = MockDepthPro(num_layers=24, dim=1024)
    adapted = inject_lora_into_depthpro(
        model,
        variant="dora",
        rank=4,
        alpha=4.0,
        dropout=0.05,
        late_block_start=18,
        adapt_patch_encoder=True,
        adapt_image_encoder=True,
        adapt_fov_encoder=False,
    )

    patch_dino = model.depth_pro.encoder.patch_encoder.model
    image_dino = model.depth_pro.encoder.image_encoder.model
    fov_dino = model.fov_model.fov_encoder.model

    # Verify patch encoder has adapters
    assert _is_dora(patch_dino.encoder.layer[0].attention.attention.query), "Patch encoder Q not adapted"
    assert _is_dora(patch_dino.encoder.layer[0].attention.attention.value), "Patch encoder V not adapted"

    # Verify image encoder has adapters
    assert _is_dora(image_dino.encoder.layer[0].attention.attention.query), "Image encoder Q not adapted"
    assert _is_dora(image_dino.encoder.layer[0].attention.attention.value), "Image encoder V not adapted"

    # Verify FOV encoder has 0 adapters
    assert _is_plain_linear(fov_dino.encoder.layer[0].attention.attention.query), "FOV encoder Q should be plain"
    assert _is_plain_linear(fov_dino.encoder.layer[0].attention.attention.value), "FOV encoder V should be plain"

    # Early blocks (0-17): only Q+V adapted
    for block_idx in range(18):
        for dino in (patch_dino, image_dino):
            layer = dino.encoder.layer[block_idx]
            assert _is_dora(layer.attention.attention.query), f"Block {block_idx} Q not adapted"
            assert _is_dora(layer.attention.attention.value), f"Block {block_idx} V not adapted"
            assert _is_plain_linear(layer.attention.attention.key), f"Block {block_idx} K should be plain"
            assert _is_plain_linear(layer.attention.output.dense), f"Block {block_idx} proj should be plain"
            assert _is_plain_linear(layer.mlp.fc1), f"Block {block_idx} fc1 should be plain"
            assert _is_plain_linear(layer.mlp.fc2), f"Block {block_idx} fc2 should be plain"

    # Late blocks (18-23): Q+K+V+proj+fc1+fc2 adapted
    for block_idx in range(18, 24):
        for dino in (patch_dino, image_dino):
            layer = dino.encoder.layer[block_idx]
            assert _is_dora(layer.attention.attention.query), f"Late block {block_idx} Q not adapted"
            assert _is_dora(layer.attention.attention.key), f"Late block {block_idx} K not adapted"
            assert _is_dora(layer.attention.attention.value), f"Late block {block_idx} V not adapted"
            assert _is_dora(layer.attention.output.dense), f"Late block {block_idx} proj not adapted"
            assert _is_dora(layer.mlp.fc1), f"Late block {block_idx} fc1 not adapted"
            assert _is_dora(layer.mlp.fc2), f"Late block {block_idx} fc2 not adapted"

    # Total params ~1,658,880 (1.66M)
    total_adapted = sum(adapted.values())
    expected = 1_658_880
    assert total_adapted == expected, f"Expected {expected} adapter params, got {total_adapted}"

    print(f"  ✓ Patch + Image encoders adapted")
    print(f"  ✓ FOV encoder frozen (no adapters)")
    print(f"  ✓ Early blocks (0-17): Q+V only")
    print(f"  ✓ Late blocks (18-23): Q+K+V+proj+fc1+fc2")
    print(f"  ✓ Total adapter params: {total_adapted:,} (expected {expected:,})")


# --------------------------------------------------------------------------- #
# Test 2: DA2 Adapter Injection (Structured)
# --------------------------------------------------------------------------- #


def test_da2_adapter_injection():
    print("\n[Test 2] DA2 Adapter Injection (Structured)")
    model = MockDA2Model(num_layers=24, dim=1024)
    adapted = inject_lora_into_depth_model(
        model,
        variant="dora",
        rank=4,
        alpha=4.0,
        dropout=0.05,
        late_block_start=18,
        adapt_decoder=False,
    )

    blocks = model.backbone.encoder.layer

    # Verify finds backbone.encoder.layer
    assert len(blocks) == 24, "Should discover 24 encoder blocks"

    # Early blocks: Q+V adapted
    for block_idx in range(18):
        layer = blocks[block_idx]
        assert _is_dora(layer.attention.attention.query), f"Block {block_idx} Q not adapted"
        assert _is_dora(layer.attention.attention.value), f"Block {block_idx} V not adapted"
        assert _is_plain_linear(layer.attention.attention.key), f"Block {block_idx} K should be plain"
        assert _is_plain_linear(layer.attention.output.dense), f"Block {block_idx} proj should be plain"
        assert _is_plain_linear(layer.mlp.fc1), f"Block {block_idx} fc1 should be plain"
        assert _is_plain_linear(layer.mlp.fc2), f"Block {block_idx} fc2 should be plain"

    # Late blocks: Q+K+V+proj+fc1+fc2 adapted
    for block_idx in range(18, 24):
        layer = blocks[block_idx]
        assert _is_dora(layer.attention.attention.query), f"Late block {block_idx} Q not adapted"
        assert _is_dora(layer.attention.attention.key), f"Late block {block_idx} K not adapted"
        assert _is_dora(layer.attention.attention.value), f"Late block {block_idx} V not adapted"
        assert _is_dora(layer.attention.output.dense), f"Late block {block_idx} proj not adapted"
        assert _is_dora(layer.mlp.fc1), f"Late block {block_idx} fc1 not adapted"
        assert _is_dora(layer.mlp.fc2), f"Late block {block_idx} fc2 not adapted"

    # Total params ~829,440
    total_adapted = sum(adapted.values())
    expected = 829_440
    assert total_adapted == expected, f"Expected {expected} adapter params, got {total_adapted}"

    print(f"  ✓ Structured injection found backbone.encoder.layer")
    print(f"  ✓ Early blocks (0-17): Q+V only")
    print(f"  ✓ Late blocks (18-23): Q+K+V+proj+fc1+fc2")
    print(f"  ✓ Total adapter params: {total_adapted:,} (expected {expected:,})")


# --------------------------------------------------------------------------- #
# Test 3: DA3 Adapter Injection (Generic Fallback)
# --------------------------------------------------------------------------- #


def test_da3_adapter_injection():
    print("\n[Test 3] DA3 Adapter Injection (Generic Fallback)")
    model = MockDA3Model(num_blocks=24, dim=1024)
    adapted = inject_lora_into_depth_model(
        model,
        variant="dora",
        rank=4,
        alpha=4.0,
        dropout=0.05,
        late_block_start=6,
        adapt_decoder=False,
    )

    # Verify total params > 0 (generic fallback found something)
    total_adapted = sum(adapted.values())
    assert total_adapted > 0, f"Expected >0 adapter params, got {total_adapted}"

    # Verify early blocks (0-5): only Q+V adapted
    for block_idx in range(6):
        block = model.custom_blocks[block_idx]
        assert _is_dora(block.attn_q), f"Early block {block_idx} attn_q not adapted"
        assert _is_plain_linear(block.attn_k), f"Early block {block_idx} attn_k should be plain"
        assert _is_dora(block.attn_v), f"Early block {block_idx} attn_v not adapted"
        assert _is_plain_linear(block.attn_proj), f"Early block {block_idx} attn_proj should be plain"
        assert _is_plain_linear(block.mlp_fc1), f"Early block {block_idx} mlp_fc1 should be plain"
        assert _is_plain_linear(block.mlp_fc2), f"Early block {block_idx} mlp_fc2 should be plain"

    # Verify late blocks (6-23): more layers adapted
    for block_idx in range(6, 24):
        block = model.custom_blocks[block_idx]
        assert _is_dora(block.attn_q), f"Late block {block_idx} attn_q not adapted"
        assert _is_dora(block.attn_k), f"Late block {block_idx} attn_k not adapted"
        assert _is_dora(block.attn_v), f"Late block {block_idx} attn_v not adapted"
        assert _is_dora(block.attn_proj), f"Late block {block_idx} attn_proj not adapted"
        assert _is_dora(block.mlp_fc1), f"Late block {block_idx} mlp_fc1 not adapted"
        assert _is_dora(block.mlp_fc2), f"Late block {block_idx} mlp_fc2 not adapted"

    print(f"  ✓ Generic fallback triggered (no structured blocks found)")
    print(f"  ✓ Early blocks (0-5): Q+V only")
    print(f"  ✓ Late blocks (6-23): Q+K+V+proj+fc1+fc2")
    print(f"  ✓ Total adapter params: {total_adapted:,}")


# --------------------------------------------------------------------------- #
# Test 4: Teacher-Student Separation
# --------------------------------------------------------------------------- #


def test_teacher_student_separation():
    print("\n[Test 4] Teacher-Student Separation")
    model = MockDA2Model(num_layers=24, dim=1024)
    teacher = copy.deepcopy(model)

    # Inject adapters into student only
    inject_lora_into_depth_model(
        model,
        variant="dora",
        rank=4,
        alpha=4.0,
        dropout=0.05,
        late_block_start=18,
    )
    freeze_non_adapter_params(model)

    # Freeze teacher (no adapters, so everything frozen)
    freeze_non_adapter_params(teacher)

    teacher_trainable = count_adapter_params(teacher)
    student_trainable = count_adapter_params(model)

    assert teacher_trainable == 0, f"Teacher should have 0 trainable params, got {teacher_trainable}"
    assert student_trainable > 0, f"Student should have >0 trainable params, got {student_trainable}"

    print(f"  ✓ Teacher trainable params: {teacher_trainable}")
    print(f"  ✓ Student trainable params: {student_trainable:,}")


# --------------------------------------------------------------------------- #
# Test 5: Checkpoint Roundtrip
# --------------------------------------------------------------------------- #


def test_checkpoint_roundtrip():
    print("\n[Test 5] Checkpoint Roundtrip")
    model = MockDA2Model(num_layers=24, dim=1024)
    inject_lora_into_depth_model(
        model,
        variant="dora",
        rank=4,
        alpha=4.0,
        dropout=0.05,
        late_block_start=18,
    )

    # Manually set some adapter weights to non-zero values so we can verify preservation
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.fill_(0.42)

    adapter_config = {
        "variant": "dora",
        "rank": 4,
        "alpha": 4.0,
        "dropout": 0.05,
        "late_block_start": 18,
    }

    state_dict = model.state_dict()

    # Save
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
        torch.save({"model": state_dict, "adapter_config": adapter_config}, ckpt_path)

    # Fresh model + same injection
    fresh_model = MockDA2Model(num_layers=24, dim=1024)
    inject_lora_into_depth_model(
        fresh_model,
        variant="dora",
        rank=4,
        alpha=4.0,
        dropout=0.05,
        late_block_start=18,
    )

    # Load
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = fresh_model.load_state_dict(ckpt["model"], strict=False)

    # Verify adapter weights preserved
    mismatches = []
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), fresh_model.named_parameters()):
        assert n1 == n2, f"Parameter name mismatch: {n1} vs {n2}"
        if "lora_" in n1:
            if not torch.allclose(p1, p2):
                mismatches.append(n1)

    assert len(mismatches) == 0, f"Adapter weights not preserved for: {mismatches}"

    # Verify config round-tripped
    assert ckpt["adapter_config"] == adapter_config, "Adapter config mismatch"

    print(f"  ✓ Checkpoint saved and loaded")
    print(f"  ✓ All adapter weights preserved")
    print(f"  ✓ Adapter config preserved")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=" * 70)
    print("Running Depth Adapter Architecture Tests")
    print("=" * 70)

    test_depthpro_adapter_injection()
    test_da2_adapter_injection()
    test_da3_adapter_injection()
    test_teacher_student_separation()
    test_checkpoint_roundtrip()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
