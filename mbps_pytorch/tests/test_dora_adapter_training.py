"""Comprehensive tests for DINOv2+CAUSE-TR DoRA adapter architecture.

Verifies architecture matches the spec in:
Research/mbps-panoptic-segmentation/Knowledge/Architecture-DINOv2-CAUSE-TR-Adapters.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import tempfile
import torch
import torch.nn as nn

from mbps_pytorch.models.adapters import (
    inject_lora_into_dinov2,
    inject_lora_into_cause_tr,
    freeze_non_adapter_params,
    count_adapter_params,
    count_total_params,
    DoRALinear,
)


# --------------------------------------------------------------------------- #
# Mock Models
# --------------------------------------------------------------------------- #

class MockDINOv2Block(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, dim * 3)
        self.attn.proj = nn.Linear(dim, dim)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, dim * 4)
        self.mlp.fc2 = nn.Linear(dim * 4, dim)


class MockDINOv2(nn.Module):
    def __init__(self, num_blocks=12, dim=768):
        super().__init__()
        self.blocks = nn.ModuleList([MockDINOv2Block(dim) for _ in range(num_blocks)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk.attn.proj(blk.attn.qkv(x))
            x = blk.mlp.fc2(blk.mlp.fc1(x))
        return x


class MockSelfAttention(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.out_proj = nn.Linear(dim, dim)


class MockMultiheadAttention(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.out_proj = nn.Linear(dim, dim)


class MockTRDecoder(nn.Module):
    """TRDecoder-like module with dims matching the spec's stated param counts."""

    def __init__(self, dim=768):
        super().__init__()
        self.tr = nn.Module()
        self.tr.self_attn = MockSelfAttention(dim)
        self.tr.multihead_attn = MockMultiheadAttention(dim)
        # Using 3072 hidden dim so DoRA param counts match spec:
        # linear1: 768->3072 => 18,432 params  (spec table value)
        # linear2: 3072->768 => 16,128 params  (spec table value)
        self.tr.linear1 = nn.Linear(dim, dim * 4)
        self.tr.linear2 = nn.Linear(dim * 4, dim)


class MockSegmentTR(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.head = MockTRDecoder(dim)

    def forward(self, x):
        return self.head.tr.linear2(self.head.tr.linear1(x))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _dora_params(in_features, out_features, rank=4):
    """Compute expected DoRA trainable param count for a linear layer."""
    return rank * in_features + out_features * rank + out_features


def ema_update(student_head, teacher_head, lamb=0.99):
    """EMA update matching parameters by name to handle adapter count mismatch."""
    student_state = dict(student_head.named_parameters())
    with torch.no_grad():
        for name, p_t in teacher_head.named_parameters():
            if name in student_state:
                p_s = student_state[name]
                if p_s.shape == p_t.shape:
                    p_t.data = lamb * p_t.data + (1 - lamb) * p_s.data


# --------------------------------------------------------------------------- #
# Test 1: Parameter Count Verification
# --------------------------------------------------------------------------- #

def test_parameter_counts():
    """Verify adapter parameter counts match the spec table."""
    rank = 4
    alpha = 4.0
    dropout = 0.05
    late_block_start = 6

    # Expected per-layer DoRA param counts (r=4)
    qkv_params = _dora_params(768, 2304, rank)      # 14,592
    proj_params = _dora_params(768, 768, rank)      # 6,912
    fc1_params = _dora_params(768, 3072, rank)      # 18,432
    fc2_params = _dora_params(3072, 768, rank)      # 16,128

    # DINOv2 expected totals
    early_total = 6 * qkv_params                     # 87,552
    late_total = 6 * (qkv_params + proj_params + fc1_params + fc2_params)  # 336,384
    dinov2_subtotal = early_total + late_total       # 423,936

    # CAUSE-TR expected totals (using dim*4=3072 to match spec counts)
    cause_self_attn = _dora_params(768, 768, rank)   # 6,912
    cause_multihead = _dora_params(768, 768, rank)   # 6,912
    cause_linear1 = _dora_params(768, 3072, rank)    # 18,432
    cause_linear2 = _dora_params(3072, 768, rank)    # 16,128
    cause_total = cause_self_attn + cause_multihead + cause_linear1 + cause_linear2  # 48,384

    grand_total = dinov2_subtotal + cause_total      # ~472,320

    # Build and inject
    model = MockDINOv2(num_blocks=12, dim=768)
    adapted_dinov2 = inject_lora_into_dinov2(
        model,
        variant="dora",
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        late_block_start=late_block_start,
    )

    # Verify individual layer counts
    early_qkv_sum = sum(
        v for k, v in adapted_dinov2.items()
        if k.startswith("blocks.") and int(k.split(".")[1]) < late_block_start
    )
    late_all_sum = sum(
        v for k, v in adapted_dinov2.items()
        if k.startswith("blocks.") and int(k.split(".")[1]) >= late_block_start
    )

    assert early_qkv_sum == early_total, f"Early total mismatch: {early_qkv_sum} != {early_total}"
    assert late_all_sum == late_total, f"Late total mismatch: {late_all_sum} != {late_total}"

    # Count via helper
    freeze_non_adapter_params(model)
    actual_dinov2 = count_adapter_params(model)
    assert actual_dinov2 == dinov2_subtotal, f"DINOv2 subtotal mismatch: {actual_dinov2} != {dinov2_subtotal}"

    # Inject CAUSE-TR
    segment = MockSegmentTR(dim=768)
    adapted_cause = inject_lora_into_cause_tr(
        segment,
        variant="dora",
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        adapt_head=True,
        adapt_projection=False,
        adapt_ema=False,
    )

    freeze_non_adapter_params(segment)
    actual_cause = count_adapter_params(segment)
    assert actual_cause == cause_total, f"CAUSE-TR total mismatch: {actual_cause} != {cause_total}"

    # Verify adapted dict keys match expected
    assert len(adapted_cause) == 4
    assert "head.tr.self_attn.out_proj" in adapted_cause
    assert "head.tr.multihead_attn.out_proj" in adapted_cause
    assert "head.tr.linear1" in adapted_cause
    assert "head.tr.linear2" in adapted_cause

    # Grand total
    actual_grand = actual_dinov2 + actual_cause
    assert actual_grand == grand_total, f"Grand total mismatch: {actual_grand} != {grand_total}"

    print(f"  DINOv2 Early (0-5):   {early_total:,} params  OK")
    print(f"  DINOv2 Late (6-11):   {late_total:,} params  OK")
    print(f"  DINOv2 Subtotal:      {dinov2_subtotal:,} params  OK")
    print(f"  CAUSE-TR Head:        {cause_total:,} params  OK")
    print(f"  Grand Total:          {grand_total:,} params  OK")
    print("Test 1 PASSED: Parameter counts match spec.")


# --------------------------------------------------------------------------- #
# Test 2: Teacher-Student Separation
# --------------------------------------------------------------------------- #

def test_teacher_student_separation():
    """Verify teacher has 0 trainable params; student has adapters."""
    rank = 4
    alpha = 4.0
    dropout = 0.05

    # Student with adapters
    student_backbone = MockDINOv2(num_blocks=12, dim=768)
    inject_lora_into_dinov2(
        student_backbone,
        variant="dora",
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        late_block_start=6,
    )
    freeze_non_adapter_params(student_backbone)

    student_segment = MockSegmentTR(dim=768)
    inject_lora_into_cause_tr(
        student_segment,
        variant="dora",
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        adapt_head=True,
    )
    freeze_non_adapter_params(student_segment)

    # Teacher WITHOUT adapters
    teacher_backbone = MockDINOv2(num_blocks=12, dim=768)
    teacher_segment = MockSegmentTR(dim=768)
    freeze_non_adapter_params(teacher_backbone)
    freeze_non_adapter_params(teacher_segment)

    student_trainable = count_adapter_params(student_backbone) + count_adapter_params(student_segment)
    teacher_trainable = count_adapter_params(teacher_backbone) + count_adapter_params(teacher_segment)

    assert student_trainable > 0, "Student should have trainable adapter params"
    assert teacher_trainable == 0, f"Teacher should have 0 trainable params, got {teacher_trainable}"

    # Verify adapter state dict keys exist in student but NOT in teacher
    student_keys = set(student_backbone.state_dict().keys())
    teacher_keys = set(teacher_backbone.state_dict().keys())
    adapter_keys = {k for k in student_keys if "lora_" in k or "dwconv" in k or "conv_gate" in k}

    assert len(adapter_keys) > 0, "Student should have adapter keys"
    assert not any(k in teacher_keys for k in adapter_keys), (
        "Teacher should not contain any adapter keys"
    )

    print(f"  Student trainable params: {student_trainable:,}")
    print(f"  Teacher trainable params: {teacher_trainable}")
    print(f"  Adapter keys in student:  {len(adapter_keys)}")
    print("Test 2 PASSED: Teacher-student separation correct.")


# --------------------------------------------------------------------------- #
# Test 3: EMA Update Correctness
# --------------------------------------------------------------------------- #

def test_ema_update_correctness():
    """Verify EMA update only touches matching params; adapter params don't corrupt EMA."""
    rank = 4

    # Student head WITH adapters
    student_head = MockSegmentTR(dim=768)
    inject_lora_into_cause_tr(student_head, variant="dora", rank=rank, adapt_head=True)
    freeze_non_adapter_params(student_head)

    # EMA head WITHOUT adapters (frozen teacher)
    ema_head = MockSegmentTR(dim=768)
    freeze_non_adapter_params(ema_head)

    # Save original EMA values for all parameters
    ema_orig = {name: p.data.clone() for name, p in ema_head.named_parameters()}
    student_state = dict(student_head.named_parameters())

    # Perturb student base weights so EMA change is detectable
    with torch.no_grad():
        for name, p in student_head.named_parameters():
            if "lora_" not in name and "dwconv" not in name and "conv_gate" not in name:
                p.data += 1.0  # shift base weights

    # Run EMA update
    lamb = 0.99
    ema_update(student_head, ema_head, lamb=lamb)

    # Verify only matching params are updated
    for name, p_ema in ema_head.named_parameters():
        orig = ema_orig[name]
        if name in student_state and student_state[name].shape == p_ema.shape:
            # Should have been updated (base param, same shape)
            p_s = student_state[name]
            expected = lamb * orig + (1 - lamb) * p_s
            assert torch.allclose(p_ema, expected), f"EMA mismatch for {name}"
        else:
            # Should remain unchanged
            assert torch.allclose(p_ema, orig), f"EMA param {name} was incorrectly modified"

    # Verify no adapter keys leaked into EMA head
    ema_keys = set(ema_head.state_dict().keys())
    assert not any("lora_" in k for k in ema_keys), "EMA head should not have lora keys"

    print("Test 3 PASSED: EMA update correctness verified.")


# --------------------------------------------------------------------------- #
# Test 4: Checkpoint Save/Load Roundtrip
# --------------------------------------------------------------------------- #

def test_checkpoint_roundtrip():
    """Save and load adapter checkpoint with adapter_config metadata."""
    rank = 4
    alpha = 4.0
    dropout = 0.05
    late_block_start = 6

    model = MockDINOv2(num_blocks=12, dim=768)
    inject_lora_into_dinov2(
        model,
        variant="dora",
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        late_block_start=late_block_start,
    )
    freeze_non_adapter_params(model)

    segment = MockSegmentTR(dim=768)
    inject_lora_into_cause_tr(segment, variant="dora", rank=rank, alpha=alpha, dropout=dropout, adapt_head=True)
    freeze_non_adapter_params(segment)

    adapter_config = {
        "variant": "dora",
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "late_block_start": late_block_start,
        "adapt_cause": True,
    }

    # Save checkpoint
    ckpt = {
        "backbone": model.state_dict(),
        "segment": segment.state_dict(),
        "adapter_config": adapter_config,
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name

    torch.save(ckpt, ckpt_path)

    # Load checkpoint
    loaded = torch.load(ckpt_path, weights_only=False)
    loaded_config = loaded["adapter_config"]

    # Verify adapter_config contains required fields
    required_keys = {"variant", "rank", "alpha", "dropout", "late_block_start", "adapt_cause"}
    assert required_keys.issubset(set(loaded_config.keys())), (
        f"Missing keys in adapter_config: {required_keys - set(loaded_config.keys())}"
    )
    assert loaded_config["variant"] == "dora"
    assert loaded_config["rank"] == rank
    assert loaded_config["alpha"] == alpha
    assert loaded_config["dropout"] == dropout
    assert loaded_config["late_block_start"] == late_block_start
    assert loaded_config["adapt_cause"] is True

    # Verify adapter weights are preserved by loading into fresh model
    model2 = MockDINOv2(num_blocks=12, dim=768)
    inject_lora_into_dinov2(
        model2,
        variant=loaded_config["variant"],
        rank=loaded_config["rank"],
        alpha=loaded_config["alpha"],
        dropout=loaded_config["dropout"],
        late_block_start=loaded_config["late_block_start"],
    )
    freeze_non_adapter_params(model2)
    model2.load_state_dict(loaded["backbone"], strict=True)

    segment2 = MockSegmentTR(dim=768)
    inject_lora_into_cause_tr(
        segment2,
        variant=loaded_config["variant"],
        rank=loaded_config["rank"],
        alpha=loaded_config["alpha"],
        dropout=loaded_config["dropout"],
        adapt_head=loaded_config["adapt_cause"],
    )
    freeze_non_adapter_params(segment2)
    segment2.load_state_dict(loaded["segment"], strict=True)

    # Verify adapter weights match after roundtrip
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        if "lora_" in n1 or "dwconv" in n1 or "conv_gate" in n1:
            assert torch.allclose(p1, p2), f"Backbone adapter weight mismatch: {n1}"

    for (n1, p1), (n2, p2) in zip(segment.named_parameters(), segment2.named_parameters()):
        if "lora_" in n1 or "dwconv" in n1 or "conv_gate" in n1:
            assert torch.allclose(p1, p2), f"Segment adapter weight mismatch: {n1}"

    # Cleanup
    Path(ckpt_path).unlink(missing_ok=True)

    print("Test 4 PASSED: Checkpoint save/load roundtrip verified.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=" * 70)
    print("DINOv2+CAUSE-TR DoRA Adapter Training Tests")
    print("=" * 70)

    print("\n[TEST 1] Parameter Count Verification")
    test_parameter_counts()

    print("\n[TEST 2] Teacher-Student Separation")
    test_teacher_student_separation()

    print("\n[TEST 3] EMA Update Correctness")
    test_ema_update_correctness()

    print("\n[TEST 4] Checkpoint Save/Load Roundtrip")
    test_checkpoint_roundtrip()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
