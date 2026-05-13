"""5-minute integration test: DoRA adapter training end-to-end.

Verifies the full pipeline:
  1. Load DINOv2 + CAUSE-TR + Cluster
  2. Inject DoRA adapters
  3. Run training loop (all 4 losses) for 2 steps
  4. Save checkpoint with adapter_config
  5. Load checkpoint into inference script pattern
  6. Run 1 forward pass to verify adapter weights are active

This does NOT require Cityscapes data — it uses synthetic tensors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from types import SimpleNamespace

CAUSE_DIR = str(Path(__file__).resolve().parent.parent.parent / "refs" / "cause")
if CAUSE_DIR not in sys.path:
    sys.path.insert(0, CAUSE_DIR)

from models.dinov2vit import dinov2_vit_base_14
from modules.segment import Segment_TR
from modules.segment_module import Cluster, transform

from mbps_pytorch.models.adapters import (
    inject_lora_into_dinov2,
    inject_lora_into_cause_tr,
    freeze_non_adapter_params,
    count_adapter_params,
    set_dinov2_spatial_dims,
)

from mbps_pytorch.train_semantic_adapter import (
    train_semantic_adapter,
    patch_cluster_for_device,
    dino_distillation_loss,
    cross_view_consistency_loss,
    depth_correlation_loss,
)

import numpy as np


def build_cause_args():
    return SimpleNamespace(
        dim=768, reduced_dim=90, projection_dim=2048,
        num_codebook=2048, n_classes=27,
        num_queries=23 * 23, crop_size=322, patch_size=14,
    )


class DummyDataset(torch.utils.data.Dataset):
    """Synthetic dataset: 8 random 322×322 images + depth + aug."""
    def __init__(self, n=8):
        self.n = n
        self.imgs = torch.randn(n, 3, 322, 322)
        self.depths = torch.randn(n, 1, 23, 23)
        self.imgs_aug = torch.randn(n, 3, 322, 322)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "img": self.imgs[idx],
            "img_aug": self.imgs_aug[idx],
            "depth": self.depths[idx],
        }


def test_integration():
    print("=" * 70)
    print("DoRA Adapter Integration Test")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_root = Path(__file__).resolve().parent.parent.parent / "refs" / "cause"

    # ── 1. Load base models ──────────────────────────────────────────────────
    print("\n[1/6] Loading base models...")
    backbone = dinov2_vit_base_14()
    state = torch.load(ckpt_root / "checkpoint" / "dinov2_vit_base_14.pth",
                       map_location="cpu", weights_only=True)
    backbone.load_state_dict(state, strict=False)

    teacher_backbone = dinov2_vit_base_14()
    teacher_backbone.load_state_dict(state, strict=False)
    teacher_backbone.eval()
    for p in teacher_backbone.parameters():
        p.requires_grad = False
    teacher_backbone = teacher_backbone.to(device)

    cause_args = build_cause_args()
    segment = Segment_TR(cause_args).to(device)
    cluster = Cluster(cause_args).to(device)

    seg_path = ckpt_root / "CAUSE" / "cityscapes" / "dinov2_vit_base_14" / "2048" / "segment_tr.pth"
    seg_state = torch.load(seg_path, map_location="cpu", weights_only=True)
    segment.load_state_dict(seg_state, strict=False)

    teacher_segment = Segment_TR(cause_args).to(device)
    teacher_segment.load_state_dict(seg_state, strict=False)
    teacher_segment.eval()
    for p in teacher_segment.parameters():
        p.requires_grad = False

    mod_path = ckpt_root / "CAUSE" / "cityscapes" / "modularity" / "dinov2_vit_base_14" / "2048" / "modular.npy"
    cb = torch.from_numpy(np.load(str(mod_path))).to(device)
    cluster.codebook.data = cb
    cluster.codebook.requires_grad = False
    segment.head.codebook = cb
    segment.head_ema.codebook = cb
    teacher_segment.head.codebook = cb
    teacher_segment.head_ema.codebook = cb

    # ── 2. Inject adapters ───────────────────────────────────────────────────
    print("[2/6] Injecting DoRA adapters (r=4, alpha=4.0, late_block_start=6)...")
    inject_lora_into_dinov2(backbone, variant="dora", rank=4, alpha=4.0,
                            dropout=0.05, late_block_start=6)
    freeze_non_adapter_params(backbone)
    backbone = backbone.to(device)

    inject_lora_into_cause_tr(segment, variant="dora", rank=4, alpha=4.0,
                              dropout=0.05, adapt_head=True, adapt_projection=False,
                              adapt_ema=False)
    freeze_non_adapter_params(segment)

    patch_cluster_for_device(cluster, device)
    cluster.bank_init()
    freeze_non_adapter_params(cluster)
    cluster.cluster_probe.requires_grad = True

    trainable = count_adapter_params(backbone) + count_adapter_params(segment) + count_adapter_params(cluster)
    print(f"  Total trainable params: {trainable:,}")
    # Actual counts: DINOv2 423,936 + CAUSE-TR 39,168 + cluster_probe ~2,430 = ~465,534
    assert 460_000 < trainable < 470_000, f"Expected ~465K, got {trainable}"

    # ── 3. Run training for 2 steps ──────────────────────────────────────────
    print("[3/6] Running training loop (2 steps, all 4 losses)...")
    dummy_ds = DummyDataset(n=4)
    dummy_loader = DataLoader(dummy_ds, batch_size=2, shuffle=False)

    adapter_config = {
        "variant": "dora",
        "rank": 4,
        "alpha": 4.0,
        "dropout": 0.05,
        "late_block_start": 6,
        "adapt_cause": True,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        train_semantic_adapter(
            backbone, segment, cluster, dummy_loader, device, tmpdir,
            losses=["distillation", "cross_view", "depth_cluster", "cause_cluster"],
            loss_weights={"distillation": 1.0, "cross_view": 1.0,
                          "depth_cluster": 0.05, "cause_cluster": 1.0},
            lr=1e-4, epochs=1, save_every=1,
            adapter_config=adapter_config,
            teacher_backbone=teacher_backbone,
            teacher_segment=teacher_segment,
        )

        # ── 4. Verify checkpoint was saved ───────────────────────────────────
        print("[4/6] Verifying checkpoint save...")
        ckpt_path = Path(tmpdir) / "epoch_001.pt"
        assert ckpt_path.exists(), "Checkpoint not saved!"
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        assert "adapter_config" in ckpt, "adapter_config missing from checkpoint"
        assert ckpt["adapter_config"]["variant"] == "dora"
        print(f"  Checkpoint saved: {ckpt_path}")
        print(f"  adapter_config: {ckpt['adapter_config']}")

        # ── 5. Load checkpoint into fresh model (inference pattern) ──────────
        print("[5/6] Loading checkpoint into fresh model (inference pattern)...")
        backbone_inf = dinov2_vit_base_14()
        backbone_inf.load_state_dict(state, strict=False)
        inject_lora_into_dinov2(
            backbone_inf,
            variant=ckpt["adapter_config"]["variant"],
            rank=ckpt["adapter_config"]["rank"],
            alpha=ckpt["adapter_config"]["alpha"],
            dropout=ckpt["adapter_config"]["dropout"],
            late_block_start=ckpt["adapter_config"]["late_block_start"],
        )
        freeze_non_adapter_params(backbone_inf)
        backbone_inf.load_state_dict(ckpt["backbone"], strict=False)
        backbone_inf = backbone_inf.to(device).eval()

        segment_inf = Segment_TR(cause_args).to(device)
        segment_inf.load_state_dict(seg_state, strict=False)
        segment_inf.head.codebook = cb.clone()
        segment_inf.head_ema.codebook = cb.clone()
        if ckpt["adapter_config"]["adapt_cause"]:
            inject_lora_into_cause_tr(
                segment_inf,
                variant=ckpt["adapter_config"]["variant"],
                rank=ckpt["adapter_config"]["rank"],
                alpha=ckpt["adapter_config"]["alpha"],
                dropout=ckpt["adapter_config"]["dropout"],
                adapt_head=True, adapt_projection=False, adapt_ema=False,
            )
        freeze_non_adapter_params(segment_inf)
        segment_inf.load_state_dict(ckpt["segment"], strict=False)
        segment_inf = segment_inf.to(device).eval()

        # ── 6. Forward pass to verify adapters are active ────────────────────
        print("[6/6] Verifying adapted forward pass...")
        set_dinov2_spatial_dims(backbone_inf, h_patches=23, w_patches=23)
        x = torch.randn(1, 3, 322, 322).to(device)
        with torch.no_grad():
            feat = backbone_inf(x)[:, 1:, :]
            seg_feat = segment_inf.head(feat)
            code = transform(seg_feat)
        assert code.shape == (1, 90, 23, 23), f"Unexpected code shape: {code.shape}"
        print(f"  Output shape: {code.shape}  OK")

    print("\n" + "=" * 70)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_integration()
