"""Sanity tests for CAUSE-TR Mode B retraining.

These 10 tests are the gate before any 40-epoch run. They verify the freeze
contract (codebook + cluster_probe immutable, backbone untrained), the adapter
identity-init invariant, gradient flow through the custom non-detached centroid
loss, and the dim-flow pipeline.

Run:
    cd <project_root>
    python -m pytest mbps_pytorch/tests/test_cause_modeb_freeze.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mbps_pytorch.models.adapters.dinov3_to_dinov2_adapter import DINOv3ToDINOv2Adapter
from mbps_pytorch.training.cause_modeb_freeze import (
    install_frozen_into_cluster,
    load_frozen_codebook_and_probe,
    verify_freeze,
    wire_codebook_into_segment,
)
from mbps_pytorch.training.cause_modeb_trainer import (
    CAUSEModeBConfig,
    CAUSEModeBTrainer,
)
from mbps_pytorch.train_cause_dinov3 import DeviceAwareCluster

# CAUSE module sys.path is prepared by train_cause_dinov3 import.
from modules.segment import Segment_TR
from modules.segment_module import ema_init


CENTROIDS_NPZ = _PROJECT_ROOT / "refs/cause/CAUSE/cityscapes/extracted_centroids/cause_tr_centroids_cityscapes.npz"


# ---- Fixtures ---------------------------------------------------------- #

@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")  # tests must work on CPU


@pytest.fixture(scope="module")
def cause_args():
    # 64 patches (8x8 grid) keeps tests cheap while preserving tensor shapes.
    return SimpleNamespace(
        dim=768, reduced_dim=90, projection_dim=2048,
        num_codebook=2048, n_classes=27, num_queries=64,
    )


@pytest.fixture
def frozen_artifacts(device):
    cb, cp = load_frozen_codebook_and_probe(CENTROIDS_NPZ, device=device)
    return cb, cp


@pytest.fixture
def cluster_with_frozen(cause_args, frozen_artifacts, device):
    cb, cp = frozen_artifacts
    cluster = DeviceAwareCluster(cause_args, device).to(device)
    install_frozen_into_cluster(cluster, cb, cp)
    cluster.bank_init()
    return cluster, cb, cp


@pytest.fixture
def model_assembly(cluster_with_frozen, cause_args, device):
    cluster, cb, cp = cluster_with_frozen
    segment = Segment_TR(cause_args).to(device)
    wire_codebook_into_segment(segment, cluster)
    ema_init(segment.head, segment.head_ema)
    ema_init(segment.projection_head, segment.projection_head_ema)
    for p in segment.head_ema.parameters():
        p.requires_grad_(False)
    for p in segment.projection_head_ema.parameters():
        p.requires_grad_(False)
    adapter = DINOv3ToDINOv2Adapter(dim=768, init="identity").to(device)
    return segment, cluster, adapter, cb, cp


@pytest.fixture
def trainer(model_assembly, device):
    segment, cluster, adapter, _, _ = model_assembly
    backbone = nn.Identity()
    cfg = CAUSEModeBConfig(
        epochs=1, batch_size=2, grad_accum_steps=1,
        head_lr=1e-3, adapter_lr=1e-3,  # bigger lr for the overfit test
        log_every=10**9,
    )
    return CAUSEModeBTrainer(
        backbone=backbone, adapter=adapter, segment=segment,
        cluster=cluster, device=device, cfg=cfg,
    )


# ---- 1. Frozen artifact shape & dtype --------------------------------- #

def test_01_frozen_shapes_and_dtype(frozen_artifacts):
    cb, cp = frozen_artifacts
    assert cb.shape == (2048, 768)
    assert cp.shape == (27, 90)
    assert cb.dtype == torch.float32
    assert cp.dtype == torch.float32
    assert cb.requires_grad is False
    assert cp.requires_grad is False


# ---- 2. Backbone + cluster freeze flags ------------------------------- #

def test_02_freeze_flags(model_assembly):
    segment, cluster, adapter, _, _ = model_assembly
    assert cluster.codebook.requires_grad is False
    assert cluster.cluster_probe.requires_grad is False
    # head_ema and projection_head_ema must not be trainable.
    for p in segment.head_ema.parameters():
        assert p.requires_grad is False
    for p in segment.projection_head_ema.parameters():
        assert p.requires_grad is False
    # Student head + projection_head must be trainable.
    student_trainable = sum(p.requires_grad for p in segment.head.parameters())
    assert student_trainable > 0
    student_trainable_proj = sum(p.requires_grad for p in segment.projection_head.parameters())
    assert student_trainable_proj > 0
    # Adapter must be trainable.
    assert all(p.requires_grad for p in adapter.parameters())


# ---- 3. Frozen tensors immutable after one optim step ------------------ #

def test_03_codebook_probe_immutable_after_step(trainer, model_assembly):
    segment, cluster, adapter, cb, cp = model_assembly
    cb_before = cluster.codebook.detach().clone()
    cp_before = cluster.cluster_probe.detach().clone()

    img = torch.randn(2, 64, 768, device=trainer.device)  # bypass backbone (Identity)
    out = trainer._step(img)
    out["loss"].backward()
    trainer.optimizer.step()

    # Allow float32 ULP (~3e-8) drift; reject any real training change.
    assert torch.allclose(cluster.codebook.detach(), cb_before, atol=1e-7, rtol=0)
    assert torch.allclose(cluster.cluster_probe.detach(), cp_before, atol=1e-7, rtol=0)


# ---- 4. Adapter weight has changed after one step --------------------- #

def test_04_adapter_changes_after_step(trainer, model_assembly):
    segment, cluster, adapter, _, _ = model_assembly
    weight_before = adapter.fc.weight.detach().clone()

    img = torch.randn(2, 64, 768, device=trainer.device)
    out = trainer._step(img)
    out["loss"].backward()
    trainer.optimizer.step()

    assert not torch.equal(adapter.fc.weight.detach(), weight_before), (
        "Adapter weight should have moved away from identity after one step"
    )


# ---- 5. Adapter identity at init -------------------------------------- #

def test_05_adapter_identity_init():
    adapter = DINOv3ToDINOv2Adapter(dim=768, init="identity")
    x = torch.randn(2, 16, 768)
    y = adapter(x)
    assert torch.allclose(y, x, atol=1e-6)
    assert torch.allclose(adapter.fc.weight.detach(), torch.eye(768), atol=1e-6)


# ---- 6. Single-batch overfit check (loss decreases) -------------------- #

def test_06_single_batch_overfit_loss_decreases(trainer):
    img = torch.randn(2, 64, 768, device=trainer.device)
    losses: list[float] = []
    for _ in range(8):
        out = trainer._step(img)
        trainer.optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        trainer.optimizer.step()
        losses.append(float(out["loss"].detach()))
    first, last = losses[0], losses[-1]
    assert last < first, (
        f"Loss should decrease over 8 single-batch iters; "
        f"got first={first:.4f} last={last:.4f} (history={losses})"
    )


# ---- 7. Shape pipeline check ------------------------------------------ #

def test_07_shape_pipeline(model_assembly, trainer):
    segment, cluster, adapter, _, _ = model_assembly
    x = torch.randn(2, 64, 768, device=trainer.device)
    a = adapter(x)
    assert a.shape == (2, 64, 768)
    seg_feat = segment.head(a, drop=segment.dropout)
    assert seg_feat.shape == (2, 64, 90)
    proj_feat = segment.projection_head(seg_feat)
    assert proj_feat.shape == (2, 64, 2048)


# ---- 8. Save / load roundtrip preserves frozen tensors ---------------- #

def test_08_save_load_roundtrip(trainer, model_assembly, tmp_path):
    segment, cluster, adapter, cb, cp = model_assembly
    ckpt_dir = tmp_path / "best"
    trainer.cfg.output_dir = tmp_path
    trainer._save_checkpoint(epoch=1, metrics={"mIoU": 0.0, "pAcc": 0.0}, tag="best")

    loaded = torch.load(ckpt_dir / "cluster_tr.pth", map_location="cpu", weights_only=False)
    assert torch.equal(loaded["codebook"], cb.cpu())
    assert torch.equal(loaded["cluster_probe"], cp.cpu())


# ---- 9. Frozen probe rows unchanged across many steps ----------------- #

def test_09_probe_unchanged_across_many_steps(trainer, model_assembly):
    segment, cluster, adapter, cb, cp = model_assembly
    cp_before = cluster.cluster_probe.detach().clone()
    img = torch.randn(2, 64, 768, device=trainer.device)
    for _ in range(20):
        out = trainer._step(img)
        trainer.optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        trainer.optimizer.step()
    assert torch.allclose(cluster.cluster_probe.detach(), cp_before, atol=1e-7, rtol=0)


# ---- 10. verify_freeze raises when corrupted -------------------------- #

def test_10_verify_freeze_catches_corruption(model_assembly):
    segment, cluster, adapter, cb, cp = model_assembly
    # Currently clean — should pass.
    verify_freeze(cluster, expected_codebook=cb, expected_probe=cp)
    # Corrupt the codebook on purpose.
    with torch.no_grad():
        cluster.codebook.add_(0.5)
    with pytest.raises(AssertionError):
        verify_freeze(cluster, expected_codebook=cb, expected_probe=cp)
    # Restore.
    with torch.no_grad():
        cluster.codebook.copy_(cb)
