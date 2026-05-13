#!/usr/bin/env python3
"""Smoke test: load CAUSE + depth finetune pipeline on MPS, run 1 batch."""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project and CAUSE to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
CAUSE_DIR = str(PROJECT_ROOT / "refs" / "cause")
sys.path.insert(0, CAUSE_DIR)

# Mock pydensecrf
import types
mock_crf = types.ModuleType("pydensecrf")
mock_crf.densecrf = types.ModuleType("pydensecrf.densecrf")
mock_crf.utils = types.ModuleType("pydensecrf.utils")
sys.modules["pydensecrf"] = mock_crf
sys.modules["pydensecrf.densecrf"] = mock_crf.densecrf
sys.modules["pydensecrf.utils"] = mock_crf.utils

from models.dinov2vit import dinov2_vit_base_14
from modules.segment import Segment_TR
from modules.segment_module import Cluster, transform
from mbps_pytorch.train_cause_depth_finetune import (
    build_cause_args,
    depth_correlation_loss,
    ema_update,
    patch_cluster_for_device,
)

DATA_DIR = "/Users/qbit-glitch/Desktop/datasets"
CAUSE_CKPT = str(PROJECT_ROOT / "refs" / "cause")
DEVICE = torch.device("mps")

print(f"PyTorch {torch.__version__}, device={DEVICE}")

# 1. Load backbone
print("Loading DINOv2 backbone...")
t0 = time.time()
backbone_path = os.path.join(CAUSE_CKPT, "checkpoint", "dinov2_vit_base_14.pth")
net = dinov2_vit_base_14()
state = torch.load(backbone_path, map_location="cpu", weights_only=True)
net.load_state_dict(state, strict=False)
net = net.to(DEVICE).eval()
for p in net.parameters():
    p.requires_grad = False
print(f"  Backbone loaded in {time.time()-t0:.1f}s")

# 2. Load Segment_TR + Cluster
print("Loading Segment_TR + Cluster...")
cause_args = build_cause_args()
segment = Segment_TR(cause_args).to(DEVICE)
cluster = Cluster(cause_args).to(DEVICE)

seg_path = os.path.join(
    CAUSE_CKPT, "CAUSE", "cityscapes", "dinov2_vit_base_14", "2048", "segment_tr.pth",
)
segment.load_state_dict(
    torch.load(seg_path, map_location="cpu", weights_only=True), strict=False,
)

mod_path = os.path.join(
    CAUSE_CKPT, "CAUSE", "cityscapes", "modularity",
    "dinov2_vit_base_14", "2048", "modular.npy",
)
cb = torch.from_numpy(np.load(mod_path)).to(DEVICE)
cluster.codebook.data = cb
cluster.codebook.requires_grad = False
segment.head.codebook = cb
segment.head_ema.codebook = cb

patch_cluster_for_device(cluster, DEVICE)
cluster.bank_init()
print("  Models loaded.")

# 3. Fake 1-batch forward pass
print("Running 1-batch forward pass...")
batch_size = 2
img = torch.randn(batch_size, 3, 322, 322, device=DEVICE)
depth = torch.rand(batch_size, 1, 23, 23, device=DEVICE)

t0 = time.time()

# Backbone
with torch.no_grad():
    feat = net(img)[:, 1:, :]  # (B, 529, 768)
print(f"  Backbone forward: {time.time()-t0:.2f}s, feat={feat.shape}")

# Student + Teacher
seg_feat = segment.head(feat, drop=segment.dropout)
with torch.no_grad():
    seg_feat_ema = segment.head_ema(feat)

# Cluster loss
loss_cluster, _ = cluster.forward_centroid(seg_feat_ema)
print(f"  Cluster loss: {loss_cluster.item():.4f}")

# Depth correlation loss
code_spatial = transform(seg_feat_ema)
loss_depth = depth_correlation_loss(code_spatial, depth, feature_samples=11, shift=0.0)
print(f"  Depth loss: {loss_depth.item():.4f}")

# Total loss + backward
loss = loss_cluster + 0.05 * loss_depth
loss.backward()
print(f"  Backward pass: OK")

# EMA update
ema_update(segment.head, segment.head_ema, lamb=0.99)
print(f"  EMA update: OK")

elapsed = time.time() - t0
print(f"\nSmoke test PASSED in {elapsed:.1f}s")
print(f"  Total loss = {loss.item():.4f} (cluster={loss_cluster.item():.4f}, depth={loss_depth.item():.4f})")

# Memory estimate
if hasattr(torch.mps, "current_allocated_memory"):
    mem = torch.mps.current_allocated_memory() / 1024**3
    print(f"  MPS memory: {mem:.2f} GB")
