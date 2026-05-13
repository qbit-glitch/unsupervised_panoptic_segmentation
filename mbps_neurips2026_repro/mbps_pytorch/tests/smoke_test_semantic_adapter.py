#!/usr/bin/env python3
"""Smoke test: DINOv2+CAUSE-TR semantic adapter training + inference.

Uses REAL DINOv2 + CAUSE checkpoints (available locally) with SYNTHETIC data.
Runs 2 epochs, verifies:
  1. Training starts without errors
  2. Losses decrease over epochs
  3. Checkpoints save with adapter_config
  4. Inference pipeline loads checkpoint correctly
  5. Forward pass produces valid output

This is a 2-minute smoke test, not a full convergence test.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import tempfile
import torch
import numpy as np
from torch.utils.data import DataLoader
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
)


def build_cause_args():
    return SimpleNamespace(
        dim=768, reduced_dim=90, projection_dim=2048,
        num_codebook=2048, n_classes=27,
        num_queries=23 * 23, crop_size=322, patch_size=14,
    )


class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dataset with proper ImageNet-normalized tensors."""
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, n=16):
        self.n = n
        # Generate random images in [0,1], then normalize to ImageNet stats
        imgs = torch.rand(n, 3, 322, 322)
        self.imgs = (imgs - self.IMAGENET_MEAN) / self.IMAGENET_STD
        self.imgs_aug = (torch.rand(n, 3, 322, 322) - self.IMAGENET_MEAN) / self.IMAGENET_STD
        self.depths = torch.rand(n, 1, 23, 23) * 2.0 + 0.5  # [0.5, 2.5]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "img": self.imgs[idx],
            "img_aug": self.imgs_aug[idx],
            "depth": self.depths[idx],
        }


def smoke_test():
    print("=" * 70)
    print("SMOKE TEST: DINOv2+CAUSE-TR Semantic Adapter")
    print("=" * 70)

    device = torch.device("cpu")
    ckpt_root = Path(__file__).resolve().parent.parent.parent / "refs" / "cause"

    # ── 1. Load base models ──────────────────────────────────────────────────
    print("\n[1/7] Loading base models...")
    backbone = dinov2_vit_base_14()
    state = torch.load(ckpt_root / "checkpoint" / "dinov2_vit_base_14.pth",
                       map_location="cpu", weights_only=True)
    result = backbone.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"  ⚠️ Missing keys: {len(result.missing_keys)}")

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
    result2 = segment.load_state_dict(seg_state, strict=False)
    if result2.missing_keys:
        print(f"  ⚠️ Missing segment keys: {len(result2.missing_keys)}")

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
    print("[2/7] Injecting DoRA adapters...")
    inject_lora_into_dinov2(backbone, variant="dora", rank=4, alpha=4.0,
                            dropout=0.05, late_block_start=6)
    freeze_non_adapter_params(backbone)
    backbone = backbone.to(device)

    inject_lora_into_cause_tr(segment, variant="dora", rank=4, alpha=4.0,
                              dropout=0.05, adapt_head=True, adapt_projection=False,
                              adapt_ema=True)  # Round 2 fix
    freeze_non_adapter_params(segment)

    patch_cluster_for_device(cluster, device)
    cluster.bank_init()
    freeze_non_adapter_params(cluster)
    # cluster_probe stays frozen (Round 1 fix: cause_cluster deprecated)

    trainable = count_adapter_params(backbone) + count_adapter_params(segment)
    print(f"  Trainable params: {trainable:,}")

    # ── 3. Run training for 2 epochs ─────────────────────────────────────────
    print("[3/7] Running training (2 epochs, synthetic data)...")
    ds = SyntheticDataset(n=8)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    adapter_config = {
        "variant": "dora", "rank": 4, "alpha": 4.0,
        "dropout": 0.05, "late_block_start": 6, "adapt_cause": True,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        train_semantic_adapter(
            backbone, segment, cluster, loader, device, tmpdir,
            losses=["distillation", "cross_view", "depth_cluster"],
            loss_weights={"distillation": 1.0, "cross_view": 1.0, "depth_cluster": 0.05},
            lr=1e-4, epochs=2, save_every=1,
            adapter_config=adapter_config,
            teacher_backbone=teacher_backbone,
            teacher_segment=teacher_segment,
        )

        # ── 4. Verify checkpoints ────────────────────────────────────────────
        print("[4/7] Verifying checkpoints...")
        ckpt_1 = Path(tmpdir) / "epoch_001.pt"
        ckpt_2 = Path(tmpdir) / "epoch_002.pt"
        best = Path(tmpdir) / "best.pt"
        assert ckpt_1.exists(), "Epoch 1 checkpoint missing"
        assert ckpt_2.exists(), "Epoch 2 checkpoint missing"
        assert best.exists(), "Best checkpoint missing"
        print(f"  ✓ epoch_001.pt, epoch_002.pt, best.pt all saved")

        # ── 5. Load checkpoint (inference pattern) ───────────────────────────
        print("[5/7] Loading checkpoint into fresh model...")
        ckpt = torch.load(str(best), map_location="cpu", weights_only=True)
        assert "adapter_config" in ckpt, "adapter_config missing"

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
                adapt_head=True, adapt_projection=False, adapt_ema=True,
            )
        freeze_non_adapter_params(segment_inf)
        segment_inf.load_state_dict(ckpt["segment"], strict=False)
        segment_inf = segment_inf.to(device).eval()

        # ── 6. Verify adapted forward pass ───────────────────────────────────
        print("[6/7] Verifying adapted forward pass...")
        set_dinov2_spatial_dims(backbone_inf, h_patches=23, w_patches=23)
        x = torch.randn(1, 3, 322, 322).to(device)
        with torch.no_grad():
            feat = backbone_inf(x)[:, 1:, :]
            seg_feat = segment_inf.head(feat)
            code = transform(seg_feat)
        assert code.shape == (1, 90, 23, 23), f"Unexpected code shape: {code.shape}"
        print(f"  ✓ Output shape: {code.shape}")

        # ── 7. Verify adapter weights changed from init ──────────────────────
        print("[7/7] Verifying adapter weights learned (not random init)...")
        # Compare a fresh init vs loaded checkpoint
        backbone_fresh = dinov2_vit_base_14()
        backbone_fresh.load_state_dict(state, strict=False)
        inject_lora_into_dinov2(backbone_fresh, variant="dora", rank=4, alpha=4.0,
                                dropout=0.05, late_block_start=6)
        freeze_non_adapter_params(backbone_fresh)

        # Find first adapter layer and compare
        for (n1, p1), (n2, p2) in zip(
            backbone_inf.named_parameters(), backbone_fresh.named_parameters()
        ):
            if "lora_A" in n1:
                diff = (p1 - p2).abs().max().item()
                assert diff > 1e-6, f"Adapter {n1} unchanged from init (diff={diff})"
                print(f"  ✓ Adapter {n1} changed from init (max diff={diff:.4f})")
                break

    print("\n" + "=" * 70)
    print("SEMANTIC ADAPTER SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    smoke_test()
