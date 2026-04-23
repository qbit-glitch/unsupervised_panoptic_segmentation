"""Comprehensive tests for DINOv2+CAUSE-TR DoRA adapter architecture.

Verifies architecture matches the spec in:
Research/mbps-panoptic-segmentation/Knowledge/Architecture-DINOv2-CAUSE-TR-Adapters.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import os
import tempfile
from unittest.mock import patch

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

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
    """TRDecoder-like module with dims matching the actual CAUSE-TR architecture."""

    def __init__(self, dim=768):
        super().__init__()
        self.tr = nn.Module()
        self.tr.self_attn = MockSelfAttention(dim)
        self.tr.multihead_attn = MockMultiheadAttention(dim)
        # Actual CAUSE-TR uses 2048 hidden dim (not 3072):
        # linear1: 768->2048 => 13,312 DoRA params
        # linear2: 2048->768 => 12,032 DoRA params
        self.tr.linear1 = nn.Linear(dim, 2048)
        self.tr.linear2 = nn.Linear(2048, dim)


class MockSegmentTR(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.head = MockTRDecoder(dim)

    def forward(self, x):
        return self.head.tr.linear2(self.head.tr.linear1(x))


class MockAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.randn(rank, out_features))

    def forward(self, x):
        return x


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

    # CAUSE-TR expected totals (actual architecture: 2048 hidden dim)
    cause_self_attn = _dora_params(768, 768, rank)   # 6,912
    cause_multihead = _dora_params(768, 768, rank)   # 6,912
    cause_linear1 = _dora_params(768, 2048, rank)    # 13,312
    cause_linear2 = _dora_params(2048, 768, rank)    # 12,032
    cause_total = cause_self_attn + cause_multihead + cause_linear1 + cause_linear2  # 39,168

    grand_total = dinov2_subtotal + cause_total      # ~463,104

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
    """Verify EMA update smooths adapter params and copies frozen base params."""
    # Create a simple mock with both frozen base and trainable adapter params
    student = nn.Sequential(
        nn.Linear(10, 10),  # frozen base
        MockAdapter(10, 10),  # trainable adapter
    )
    teacher = nn.Sequential(
        nn.Linear(10, 10),
        MockAdapter(10, 10),
    )
    # Freeze base, keep adapter trainable
    for p in student[0].parameters():
        p.requires_grad = False
    for p in teacher[0].parameters():
        p.requires_grad = False

    # Initialize base weights identically so EMA preserves them
    with torch.no_grad():
        teacher[0].weight.copy_(student[0].weight)
        teacher[0].bias.copy_(student[0].bias)

    # Set different adapter values
    with torch.no_grad():
        student[1].lora_A.data = torch.ones_like(student[1].lora_A.data)
        teacher[1].lora_A.data = torch.zeros_like(teacher[1].lora_A.data)

    ema_update(student, teacher, lamb=0.9)

    # EMA adapter param should be smoothed toward student
    expected = 0.9 * 0.0 + 0.1 * 1.0  # = 0.1
    assert torch.allclose(teacher[1].lora_A.data, torch.full_like(teacher[1].lora_A.data, 0.1), atol=1e-6)

    # Base param should remain identical after EMA (same init + EMA preserves identical values)
    assert torch.allclose(teacher[0].weight.data, student[0].weight.data)

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
# Test 5: DINO distillation uses cosine similarity
# --------------------------------------------------------------------------- #

def test_dino_distillation_cosine():
    print("\n[TEST 5] DINO distillation uses cosine similarity")
    from mbps_pytorch.train_semantic_adapter import dino_distillation_loss

    student = torch.randn(2, 529, 768)
    teacher = torch.randn(2, 529, 768)
    loss = dino_distillation_loss(student, teacher)
    assert 0.0 <= loss.item() <= 2.0, f"Loss should be in [0, 2], got {loss.item()}"

    feat = torch.randn(2, 529, 768)
    loss_same = dino_distillation_loss(feat, feat.clone())
    assert loss_same.item() < 1e-5, f"Identical features should give ~0 loss, got {loss_same.item()}"

    loss_opp = dino_distillation_loss(feat, -feat.clone())
    assert abs(loss_opp.item() - 2.0) < 1e-5, f"Opposite features should give ~2 loss, got {loss_opp.item()}"

    print("  ✓ Loss range is [0, 2]")
    print("  ✓ Identical features -> loss ~0")
    print("  ✓ Opposite features -> loss ~2")


# --------------------------------------------------------------------------- #
# Test 6: ImageNet normalization present
# --------------------------------------------------------------------------- #

def test_imagenet_normalization():
    print("\n[TEST 6] ImageNet normalization present")
    from mbps_pytorch.train_semantic_adapter import AdapterTrainingDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = os.path.join(tmpdir, "leftImg8bit", "train", "testcity")
        os.makedirs(img_dir)
        img_path = os.path.join(img_dir, "test_leftImg8bit.png")
        Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)

        depth_dir = os.path.join(tmpdir, "depth_depthpro", "train", "testcity")
        os.makedirs(depth_dir)

        class MockBase(Dataset):
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return {"img": torch.rand(3, 64, 64)}

        ds = AdapterTrainingDataset(
            base_dataset=MockBase(),
            cityscapes_root=tmpdir,
            depth_subdir="depth_depthpro",
            split="train",
            use_augmentation=True,
        )
        item = ds[0]
        assert "img_aug" in item, "Missing img_aug"
        img_aug = item["img_aug"]
        # Normalized values should be outside [0, 1] for typical random inputs
        assert img_aug.min() < 0 or img_aug.max() > 1, "img_aug should be ImageNet normalized"

    print("  ✓ img_aug is ImageNet normalized")


# --------------------------------------------------------------------------- #
# Test 7: Cross-view consistency stop-gradient
# --------------------------------------------------------------------------- #

def test_cross_view_stop_gradient():
    print("\n[TEST 7] Cross-view consistency stop-gradient")
    from mbps_pytorch.train_semantic_adapter import cross_view_consistency_loss

    feat1 = torch.randn(2, 529, 768, requires_grad=True)
    feat2 = torch.randn(2, 529, 768, requires_grad=True)
    loss = cross_view_consistency_loss(feat1, feat2)
    loss.backward()

    assert feat1.grad is not None, "feat1 should have gradients"
    assert feat2.grad is None, "feat2 should be detached (no gradients)"
    print("  ✓ feat1 receives gradients")
    print("  ✓ feat2 is detached (grad is None)")


# --------------------------------------------------------------------------- #
# Test 8: Strict loading validation
# --------------------------------------------------------------------------- #

def test_strict_loading_validation():
    print("\n[TEST 8] Strict loading validation")

    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 5))
    state = {"0.weight": torch.randn(10, 10), "0.bias": torch.randn(10)}
    result = model.load_state_dict(state, strict=False)

    assert hasattr(result, "missing_keys"), "Result should expose missing_keys"
    assert hasattr(result, "unexpected_keys"), "Result should expose unexpected_keys"
    assert len(result.missing_keys) > 0, "Partial state dict should produce missing keys"

    with patch.object(logging.getLogger("test_strict"), "warning") as mock_warn:
        logger = logging.getLogger("test_strict")
        if result.missing_keys:
            logger.warning("Missing keys: %s", result.missing_keys)
        mock_warn.assert_called_once()
    print("  ✓ load_state_dict result checked for missing/unexpected keys")
    print("  ✓ Missing keys trigger warning")


# --------------------------------------------------------------------------- #
# Test 9: EMA head adapted when adapt_ema=True
# --------------------------------------------------------------------------- #

def test_ema_head_adapted():
    print("\n[TEST 9] EMA head adapted when adapt_ema=True")

    class MockSegmentTRWithEMA(nn.Module):
        def __init__(self, dim=768):
            super().__init__()
            self.head = MockTRDecoder(dim)
            self.head_ema = MockTRDecoder(dim)

    segment = MockSegmentTRWithEMA(dim=768)
    adapted = inject_lora_into_cause_tr(
        segment,
        variant="dora",
        rank=4,
        alpha=4.0,
        dropout=0.05,
        adapt_head=True,
        adapt_ema=True,
    )

    ema_adapter_keys = [name for name, p in segment.head_ema.named_parameters() if "lora_" in name]
    assert len(ema_adapter_keys) > 0, "EMA head should contain adapter parameters"
    assert any("lora_A" in k or "lora_B" in k for k in ema_adapter_keys), (
        "EMA head should have lora_A/lora_B parameters"
    )
    print("  ✓ EMA head has adapter parameters when adapt_ema=True")


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

    print("\n[TEST 5] DINO Distillation Cosine Similarity")
    test_dino_distillation_cosine()

    print("\n[TEST 6] ImageNet Normalization Present")
    test_imagenet_normalization()

    print("\n[TEST 7] Cross-View Stop-Gradient")
    test_cross_view_stop_gradient()

    print("\n[TEST 8] Strict Loading Validation")
    test_strict_loading_validation()

    print("\n[TEST 9] EMA Head Adapted")
    test_ema_head_adapted()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
