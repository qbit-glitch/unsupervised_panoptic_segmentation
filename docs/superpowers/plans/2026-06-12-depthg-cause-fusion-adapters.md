# DepthG × CAUSE-TR Fusion Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train two decoupled cross-model residual adapters (A: DepthG→CAUSE-TR, B: CAUSE-TR→DepthG) with scale-separated teacher losses, and show each adapted model beats its baseline paper's published Cityscapes number under the paper's own protocol.

**Architecture:** Frozen CAUSE-TR (DINOv2 ViT-B/14, 90-d codes, 14-px grid) and frozen DepthG (DINO ViT-B/8, 100-d codes, 8-px grid) each get a tiny preservation-anchored residual adapter conditioned on a 16-d projection of the *other* model's codes. Stratified pair loss: short-range pairs (≤4 patches) taught by DepthG similarities, long-range pairs (≥8 patches) taught by CAUSE similarities; each adapter self-teaches at its own strong scale. Readout = the frozen original probes (primary) per spec §4.6.

**Tech Stack:** PyTorch (CPU/MPS, `.venv_cups_cpu/bin/python`), vendored repos `refs/cause` (official CAUSE eval) and `refs/depthg` (official DepthG eval, already patched to dump `metrics.json`), existing modules `mbps_pytorch/models/semantic/{depth_adapter.py,stego_loss.py}`, trainer template `mbps_pytorch/train_depth_adapter.py`.

**Spec:** `docs/superpowers/specs/2026-06-12-depthg-cause-fusion-adapter-design.md` (rev 2 + §4.6 frozen-probe readout amendment).

**Known baseline anchors (already measured locally, in-repo):**
- CAUSE-TR official eval reproduction: `refs/cause/eval_cause_tr_dinov2.py` docstring — "Reproduces published results: mIoU=29.9%, pAcc=89.8%" (CRF protocol, NiceTool Hungarian, 27 classes).
- DepthG official ckpt reproduction: `refs/depthg/metrics.json` — cluster mIoU **20.94**, linear mIoU **29.13** (CRF on, ckpt `saved_models/cityscapes_vit_base_1.ckpt`, 267 val images per their loader).
- Mono-retrained DepthG (`checkpoints/depthg_depthpro_monocular/epoch6_step1680.ckpt`): cluster mIoU **14.8** (documented in `mbps_pytorch/probe_depthg_depthpro_monocular.py` docstring).
- Exact *published* paper numbers are pinned in Task 2 — never quoted from memory.

**Conventions for every task:** Python = `.venv_cups_cpu/bin/python` from the project root `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation`. Long runs = `nohup … > logs/<name>_$(date +%Y%m%d_%H%M%S).log 2>&1 &`. Commit only the files named in the task (the working tree has many unrelated modifications — never `git add -A`). Data root = `/Volumes/code_files/datasets/cityscapes`.

---

### Task 1: Environment and artifact verification

**Files:** none created — read-only smoke checks.

- [ ] **Step 1: Verify checkpoints, eval scripts, and data exist**

```bash
cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
ls -la checkpoints/depthg_depthpro_monocular/epoch6_step1680.ckpt
ls -la refs/depthg/saved_models/cityscapes_vit_base_1.ckpt
ls refs/cause/CAUSE/cityscapes/
ls /Volumes/code_files/datasets/cityscapes/leftImg8bit/train | head -3
ls /Volumes/code_files/datasets/cityscapes/depth_depthpro/train | head -3
ls /Volumes/code_files/datasets/cityscapes/gtFine/val | head -3
```

Expected: every path exists. `refs/cause/CAUSE/cityscapes/` contains a `dinov2_vit_base_14/2048/` subtree with `segment_tr.pth` and `cluster_tr.pth` (per `load_segment_tr`/`load_cluster_tr` in `refs/cause/eval_cause_tr_dinov2.py:198-248`). If `saved_models/cityscapes_vit_base_1.ckpt` is missing, STOP and report — Phase 0a needs it.

- [ ] **Step 2: Confirm both eval entry points import**

```bash
.venv_cups_cpu/bin/python -c "
import sys
sys.path.insert(0, 'refs/cause')
from modules.segment import Segment_TR
from modules.segment_module import Cluster
print('cause imports OK')
sys.path.insert(0, 'refs/cups'); sys.path.insert(0, 'refs/cups/external/depthg'); sys.path.insert(0, 'refs/cups/external/depthg/src')
from cups.semantics.model import DepthG
print('depthg wrapper import OK')
"
```

Expected: both `OK` lines, no tracebacks.

- [ ] **Step 3: Locate the DepthG eval branch that wrote metrics.json**

```bash
grep -n "metrics.json\|model_path\|num_images" refs/depthg/src/train_segmentation.py | head -20
```

Expected: a block (added by this project) that loads a checkpoint, runs validation with CRF, and dumps `metrics.json`. Record the line numbers and the variable/override that selects the checkpoint path — Task 3 and Task 11 reuse this exact path. If the checkpoint is hardcoded, note the variable name for Task 3 Step 2.

---

### Task 2: Phase 0a — pin published numbers (verified-numbers ledger)

**Files:**
- Create: `reports/fusion_adapter_numbers_ledger.md`

- [ ] **Step 1: Verify the published numbers from the actual papers**

Use WebFetch/WebSearch on the two papers (NOT memory, NOT this plan):
- CAUSE: Kim et al., "Causal Unsupervised Semantic Segmentation" (arXiv 2310.07379) — Cityscapes 27-class table, CAUSE-TR row with DINOv2 ViT-B/14 if reported (else the headline CAUSE-TR row): unsupervised/cluster mIoU and pAcc. Cross-check against the `eval_cause_tr_dinov2.py` docstring claim (mIoU=29.9, pAcc=89.8).
- DepthG: Sick et al., "Unsupervised Semantic Segmentation Through Depth-Guided Feature Correlation and Sampling" (CVPR 2024) — Cityscapes ViT-B row: unsupervised cluster mIoU (+ accuracy, linear if reported). Cross-check against the local reproduction 20.94.

- [ ] **Step 2: Write the ledger**

Create `reports/fusion_adapter_numbers_ledger.md` with one row per number: value, metric, protocol notes (CRF? resolution? class count?), source (paper table number + arXiv id / local file), verification date. Include the three local anchors (29.9 docstring, 20.94 metrics.json, 14.8 probe docstring) marked "local reproduction".

- [ ] **Step 3: Define the concrete gates and record them in the ledger**

- Gate 1A: adapted CAUSE-TR cluster mIoU ≥ (published CAUSE-TR mIoU + 1.0) under the CAUSE protocol (CRF row).
- Gate 1B: adapted mono DepthG cluster mIoU > published DepthG cluster mIoU under the DepthG protocol (CRF row). Secondary: delta vs 14.8 mono baseline.

- [ ] **Step 4: Commit**

```bash
git add reports/fusion_adapter_numbers_ledger.md
git commit -m "docs(fusion): verified-numbers ledger for DepthG/CAUSE published baselines"
```

---

### Task 3: Phase 0a — re-run both official evals (protocol fidelity gate)

**Files:** none created — runs existing scripts; results appended to the ledger.

- [ ] **Step 1: Re-run the CAUSE-TR official eval**

```bash
cd refs/cause
nohup ../../.venv_cups_cpu/bin/python eval_cause_tr_dinov2.py \
    --data_dir /Volumes/code_files/datasets \
    --device mps \
    > ../../logs/phase0a_cause_eval_$(date +%Y%m%d_%H%M%S).log 2>&1 &
cd ../..
```

Expected (in log, after the CRF pass): mIoU within ~1.0 of the published number pinned in Task 2 (docstring anchor: 29.9). If the dataloader path fails, check how `ContrastiveSegDataset` joins `--data_dir` with the dataset name and adjust `--data_dir` accordingly (it expects the parent directory that contains `cityscapes/`).

- [ ] **Step 2: Re-run the DepthG official eval (official ckpt) and the mono ckpt**

From the eval branch located in Task 1 Step 3: run it once with `saved_models/cityscapes_vit_base_1.ckpt` (expected: cluster mIoU ≈ 20.94, matching `metrics.json`), then once with `../../checkpoints/depthg_depthpro_monocular/epoch6_step1680.ckpt` (expected: cluster mIoU ≈ 14.8). Use the checkpoint variable/override identified in Task 1; run with `nohup`, logs to `logs/phase0a_depthg_{official,mono}_*.log`. Copy each run's `metrics.json` content into the ledger before the next run overwrites it.

- [ ] **Step 3: Apply the 0a gate**

All three reproductions within ~1.0 mIoU of their anchors → record PASS in the ledger and proceed. Any larger gap → reconcile (CRF flag, resolution, split) and document the cause in the ledger before proceeding. Commit the ledger update:

```bash
git add reports/fusion_adapter_numbers_ledger.md
git commit -m "docs(fusion): phase 0a protocol-fidelity reproductions recorded"
```

---

### Task 4: Feature cache builder

**Files:**
- Create: `mbps_pytorch/cache_fusion_features.py`

Caches per train-split image: CAUSE codes `z` (32, 64, 90) at 448×896 (14-px grid), DepthG codes `g` (40, 80, 100) at 320×640 (the half-res flip-averaged path DepthG itself uses for inference), DepthPro depth pooled to (32, 64). All fp16. Output layout:

```
/Volumes/code_files/datasets/cityscapes/fusion_feature_cache/
    cause_z/train/{city}/{stem}_codes.npy   (32, 64, 90)
    cause_z/train/{city}/{stem}_depth.npy   (32, 64)
    depthg_g_mono/train/{city}/{stem}_g.npy (40, 80, 100)
```

- [ ] **Step 1: Write the cache builder**

```python
#!/usr/bin/env python3
"""Build the fusion feature cache: frozen CAUSE-TR codes + frozen DepthG codes.

CAUSE side: image resized to 448x896, ImageNet-normalized, DINOv2 ViT-B/14
forward -> tokens[:, 1:, :] -> segment.head_ema -> (32, 64, 90) fp16.
DepthG side: image resized to 640x1280 then halved to 320x640 (the model's own
half-res inference path, flip-averaged) -> net code -> (40, 80, 100) fp16.
Depth: DepthPro .npy pooled to the CAUSE grid, per-image min-max normalized.

Usage (smoke):
    .venv_cups_cpu/bin/python mbps_pytorch/cache_fusion_features.py \
        --cities aachen --limit 3 --device cpu
Full run (nohup, ~2975 images):
    nohup .venv_cups_cpu/bin/python mbps_pytorch/cache_fusion_features.py \
        > logs/cache_fusion_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAUSE_ROOT = PROJECT_ROOT / "refs" / "cause"
sys.path.insert(0, str(CAUSE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "refs" / "cups"))
sys.path.insert(0, str(PROJECT_ROOT / "refs" / "cups" / "external" / "depthg"))
sys.path.insert(0, str(PROJECT_ROOT / "refs" / "cups" / "external" / "depthg" / "src"))

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DATA_ROOT = Path("/Volumes/code_files/datasets/cityscapes")
CACHE_ROOT = DATA_ROOT / "fusion_feature_cache"
MONO_CKPT = PROJECT_ROOT / "checkpoints" / "depthg_depthpro_monocular" / "epoch6_step1680.ckpt"


def load_image(path: Path, size: tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size[1], size[0]), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def build_cause(device: torch.device):
    """Load frozen DINOv2 backbone + Segment_TR head, mirroring
    refs/cause/eval_cause_tr_dinov2.py (load_backbone/load_segment_tr).
    The SimpleNamespace must carry every attribute Segment_TR reads; copy ALL
    parser defaults from eval_cause_tr_dinov2.py (lines ~355-380)."""
    import models.dinov2vit as model_module
    from modules.segment import Segment_TR
    from utils.utils import ckpt_to_arch, freeze

    ckpt_rel = "checkpoint/dinov2_vit_base_14.pth"
    args = SimpleNamespace(
        dataset="cityscapes", ckpt=ckpt_rel, num_codebook=2048,
        reduced_dim=90, projection_dim=2048, dim=768, grid=True,
        num_queries=32 * 64, device=str(device),
    )
    net = getattr(model_module, ckpt_to_arch(ckpt_rel))()
    state = torch.load(str(CAUSE_ROOT / ckpt_rel), map_location=device)
    net.load_state_dict(state, strict=False)
    net = net.to(device); freeze(net); net.eval()

    segment = Segment_TR(args).to(device)
    seg_ckpt = CAUSE_ROOT / "CAUSE" / "cityscapes" / "dinov2_vit_base_14" / "2048" / "segment_tr.pth"
    segment.load_state_dict(torch.load(str(seg_ckpt), map_location=device), strict=False)
    segment.eval()
    return net, segment


def build_depthg(device: torch.device, ckpt: Path):
    from cups.semantics.model import DepthG
    return DepthG(device=device, checkpoint_root=str(ckpt),
                  img_shape=(640, 1280), stride=(160, 160), crop=(320, 320))


@torch.no_grad()
def cause_codes(net, segment, img448: torch.Tensor) -> np.ndarray:
    feat = net(img448)[:, 1:, :]              # (1, 32*64, 768)
    code = segment.head_ema(feat)             # (1, 32*64, 90)
    return code.reshape(32, 64, 90).cpu().numpy().astype(np.float16)


@torch.no_grad()
def depthg_codes(model, img640: torch.Tensor) -> np.ndarray:
    small = F.interpolate(img640, (320, 640), mode="bilinear", align_corners=False)
    code = model(small)                        # (1, 100, 40, 80) — net last output
    code2 = model(small.flip(dims=[3]))
    code = (code + code2.flip(dims=[3])) / 2   # flip-average, mirrors model.py:102-104
    return code.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float16)


def pooled_depth(depth_path: Path) -> np.ndarray:
    d = torch.from_numpy(np.load(depth_path).astype(np.float32))[None, None]
    d = F.adaptive_avg_pool2d(d, (32, 64)).squeeze()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)   # per-image [0,1]
    return d.numpy().astype(np.float16)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train")
    p.add_argument("--cities", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None, choices=("cpu", "mps"))
    p.add_argument("--depthg_ckpt", type=Path, default=MONO_CKPT)
    p.add_argument("--g_subdir", default="depthg_g_mono",
                   help="use depthg_g_official when caching the CUPS-release ckpt")
    args = p.parse_args()
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))

    net, segment = build_cause(device)
    dg = build_depthg(device, args.depthg_ckpt)
    img_root = DATA_ROOT / "leftImg8bit" / args.split
    depth_root = DATA_ROOT / "depth_depthpro" / args.split
    cities = args.cities or sorted(d.name for d in img_root.iterdir() if d.is_dir())

    n_done = 0
    for city in cities:
        files = sorted((img_root / city).glob("*_leftImg8bit.png"))
        if args.limit:
            files = files[: args.limit]
        for f in files:
            stem = f.name.replace("_leftImg8bit.png", "")
            z_dir = CACHE_ROOT / "cause_z" / args.split / city
            g_dir = CACHE_ROOT / args.g_subdir / args.split / city
            z_dir.mkdir(parents=True, exist_ok=True); g_dir.mkdir(parents=True, exist_ok=True)
            z_out, d_out, g_out = z_dir / f"{stem}_codes.npy", z_dir / f"{stem}_depth.npy", g_dir / f"{stem}_g.npy"
            if z_out.exists() and d_out.exists() and g_out.exists():
                continue
            if not z_out.exists():
                np.save(z_out, cause_codes(net, segment, load_image(f, (448, 896)).to(device)))
            if not d_out.exists():
                dp = depth_root / city / f"{stem}_leftImg8bit.npy"
                if not dp.exists():
                    dp = depth_root / city / f"{stem}.npy"
                np.save(d_out, pooled_depth(dp))
            if not g_out.exists():
                np.save(g_out, depthg_codes(dg, load_image(f, (640, 1280)).to(device)))
            n_done += 1
            if n_done % 50 == 0:
                print(f"[cache] {n_done} images done ({city}/{stem})", flush=True)
    print(f"[cache] complete: {n_done} new images")


if __name__ == "__main__":
    main()
```

Implementation notes for this step: (a) the `SimpleNamespace` attribute list above is a starting set — open `refs/cause/eval_cause_tr_dinov2.py` lines 355-380 and `refs/cause/modules/segment.py::Segment_TR.__init__` and copy EVERY attribute the constructor reads (add missing ones; mismatches fail loudly at construction, not silently). (b) Confirm the DepthG wrapper's `__call__` no-call_type branch returns the code tensor shaped `(1, 100, 40, 80)` for a 320×640 input; if it returns `(feats, code)` tuple from `model.net`, take `[-1]` (see `refs/cups/cups/semantics/model.py:135`).

- [ ] **Step 2: Smoke run (3 images, CPU)**

```bash
.venv_cups_cpu/bin/python mbps_pytorch/cache_fusion_features.py --cities aachen --limit 3 --device cpu
.venv_cups_cpu/bin/python -c "
import numpy as np, glob
z = np.load(sorted(glob.glob('/Volumes/code_files/datasets/cityscapes/fusion_feature_cache/cause_z/train/aachen/*_codes.npy'))[0])
g = np.load(sorted(glob.glob('/Volumes/code_files/datasets/cityscapes/fusion_feature_cache/depthg_g_mono/train/aachen/*_g.npy'))[0])
d = np.load(sorted(glob.glob('/Volumes/code_files/datasets/cityscapes/fusion_feature_cache/cause_z/train/aachen/*_depth.npy'))[0])
assert z.shape == (32, 64, 90) and z.dtype == np.float16, z.shape
assert g.shape == (40, 80, 100) and g.dtype == np.float16, g.shape
assert d.shape == (32, 64) and 0.0 <= d.min() and d.max() <= 1.0
print('cache shapes OK')
"
```

Expected: `cache shapes OK`.

- [ ] **Step 3: Commit, then launch the full train-split build in the background**

```bash
git add mbps_pytorch/cache_fusion_features.py
git commit -m "feat(fusion): fusion feature cache builder (CAUSE z + DepthG g + depth)"
nohup .venv_cups_cpu/bin/python mbps_pytorch/cache_fusion_features.py \
    > logs/cache_fusion_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Expected: runs for several hours (2975 images, two model forwards each). Continue with Tasks 5-7 while it runs; Task 8 (training) needs it complete. Verify completion later with: `find /Volumes/code_files/datasets/cityscapes/fusion_feature_cache/cause_z/train -name "*_codes.npy" | wc -l` → 2975.

---

### Task 5: Adapter module + stratified pair sampler + teacher loss (TDD)

**Files:**
- Create: `mbps_pytorch/models/semantic/cross_model_adapter.py`
- Create: `mbps_pytorch/tests/test_fusion_adapter.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the cross-model fusion adapter, pair sampler, and teacher loss."""
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mbps_pytorch.models.semantic.cross_model_adapter import (
    CrossModelAdapter,
    sample_stratified_pairs,
    teacher_guided_correlation_loss,
)


def test_adapter_identity_at_init() -> None:
    """Zero-init output layer -> adapter starts as exact identity (both sides)."""
    for code_dim, cond_dim in ((90, 100), (100, 90)):
        adapter = CrossModelAdapter(code_dim=code_dim, cond_dim=cond_dim,
                                    proj_width=16, hidden_dim=384, num_layers=2)
        codes = torch.randn(2, 50, code_dim)
        cond = torch.randn(2, 50, cond_dim)
        out = adapter(codes, cond)
        assert out.shape == codes.shape
        assert torch.allclose(out, codes), "adapter must start as identity"


def test_sampler_offset_constraints() -> None:
    h, w = 32, 64
    gen = torch.Generator().manual_seed(0)
    (i_s, j_s), (i_l, j_l) = sample_stratified_pairs(
        h, w, n_short=512, n_long=512, r_short=4, r_long=8,
        device=torch.device("cpu"), generator=gen,
    )
    assert i_s.shape == (512,) and i_l.shape == (512,)
    ri, ci = i_s // w, i_s % w
    rj, cj = j_s // w, j_s % w
    linf_s = torch.max((ri - rj).abs(), (ci - cj).abs())
    assert (linf_s >= 1).all() and (linf_s <= 4).all(), "short pairs must have L_inf in [1, 4]"
    ri, ci = i_l // w, i_l % w
    rj, cj = j_l // w, j_l % w
    linf_l = torch.max((ri - rj).abs(), (ci - cj).abs())
    assert (linf_l >= 8).all(), "long pairs must have L_inf >= 8"


def test_teacher_loss_zero_cases() -> None:
    n, ds, dt = 64, 90, 100
    gen = torch.Generator().manual_seed(1)
    idx_i = torch.randint(0, n, (32,), generator=gen)
    idx_j = torch.randint(0, n, (32,), generator=gen)
    # Case 1: zero teacher -> cosine weights are 0/eps-stable -> finite loss
    teacher = torch.randn(n, dt, generator=gen)
    student = torch.randn(n, ds, generator=gen)
    loss = teacher_guided_correlation_loss(student, torch.zeros(n, dt), idx_i, idx_j)
    assert torch.isfinite(loss)
    # Case 2: student pairs identical (cos=1) -> (1-cos)^2 = 0 regardless of teacher
    same_student = torch.ones(n, ds)
    loss2 = teacher_guided_correlation_loss(same_student, teacher, idx_i, idx_j)
    assert loss2.abs().item() < 1e-6


def test_teacher_loss_pulls_weighted_pairs() -> None:
    """High-teacher-similarity pairs with dissimilar student codes -> positive loss."""
    n = 16
    teacher = torch.ones(n, 100)                      # all pairs: teacher cos = 1
    student = torch.randn(n, 90)
    idx_i = torch.arange(8)
    idx_j = torch.arange(8, 16)
    loss = teacher_guided_correlation_loss(student, teacher, idx_i, idx_j)
    assert loss.item() > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv_cups_cpu/bin/python -m pytest mbps_pytorch/tests/test_fusion_adapter.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mbps_pytorch.models.semantic.cross_model_adapter'`.

- [ ] **Step 3: Write the implementation**

```python
"""Cross-model fusion adapter: condition one frozen model's codes on another's.

Wraps the proven DCFA `DepthAdapter` (concat-conditioning, zero-init residual)
with a learned projection of the cross-model code into the 16-d conditioning
slot that DCFA used for sinusoidal depth. Also provides the stratified pair
sampler and the teacher-guided correlation loss (DCFA's depth kernel replaced
by clamped teacher-code cosine similarity — spec section 4.3).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mbps_pytorch.models.semantic.depth_adapter import DepthAdapter


class CrossModelAdapter(nn.Module):
    """codes' = codes + r([codes; W_proj @ cond]) with zero-init residual head.

    Side A: code_dim=90 (CAUSE), cond_dim=100 (DepthG).
    Side B: code_dim=100 (DepthG), cond_dim=90 (CAUSE).
    """

    def __init__(
        self,
        code_dim: int,
        cond_dim: int,
        proj_width: int = 16,
        hidden_dim: int = 384,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.cond_dim = cond_dim
        self.proj_width = proj_width
        self.proj = nn.Linear(cond_dim, proj_width)
        self.core = DepthAdapter(
            code_dim=code_dim, depth_dim=proj_width,
            hidden_dim=hidden_dim, num_layers=num_layers,
        )

    def forward(self, codes: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """codes: (B, N, code_dim); cond: (B, N, cond_dim), grid-aligned."""
        return self.core(codes, self.proj(cond))


def sample_stratified_pairs(
    h: int,
    w: int,
    n_short: int = 512,
    n_long: int = 512,
    r_short: int = 4,
    r_long: int = 8,
    device: torch.device = torch.device("cpu"),
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Sample flat-index pixel pairs on an (h, w) grid.

    Short pool: anchor + offset with L_inf in [1, r_short] (clamped to grid).
    Long pool: uniform pairs rejection-filtered to L_inf >= r_long.
    Returns ((idx_i_short, idx_j_short), (idx_i_long, idx_j_long)).
    """
    ai = torch.randint(0, h, (n_short,), device=device, generator=generator)
    aj = torch.randint(0, w, (n_short,), device=device, generator=generator)
    dr = torch.randint(-r_short, r_short + 1, (n_short,), device=device, generator=generator)
    dc = torch.randint(-r_short, r_short + 1, (n_short,), device=device, generator=generator)
    zero = (dr == 0) & (dc == 0)
    dr[zero] = 1  # nudge zero offsets to a valid neighbour
    bi = (ai + dr).clamp(0, h - 1)
    bj = (aj + dc).clamp(0, w - 1)
    # clamping can re-create zero offsets at borders; nudge the row index inward
    still_zero = (bi == ai) & (bj == aj)
    bi[still_zero] = (ai[still_zero] + torch.where(ai[still_zero] < h - 1, 1, -1)).clamp(0, h - 1)
    short = (ai * w + aj, bi * w + bj)

    n = h * w
    oversample = n_long * 4
    pi = torch.randint(0, n, (oversample,), device=device, generator=generator)
    pj = torch.randint(0, n, (oversample,), device=device, generator=generator)
    linf = torch.max((pi // w - pj // w).abs(), (pi % w - pj % w).abs())
    keep = (linf >= r_long).nonzero(as_tuple=True)[0]
    if keep.numel() < n_long:  # tiny grids: pad by repeating accepted pairs
        reps = (n_long + keep.numel() - 1) // max(keep.numel(), 1)
        keep = keep.repeat(reps)
    keep = keep[:n_long]
    return short, (pi[keep], pj[keep])


def teacher_guided_correlation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
) -> torch.Tensor:
    """w_ij * (1 - cos(student_i, student_j))^2 with w_ij = max(cos(teacher_i, teacher_j), 0).

    Same functional form as stego_loss.depth_guided_correlation_loss, with the
    depth kernel replaced by clamped teacher cosine similarity (spec 4.3).
    student: (N, Ds); teacher: (N, Dt); idx_*: (P,) flat indices.
    """
    with torch.no_grad():
        w = F.cosine_similarity(teacher[idx_i], teacher[idx_j], dim=-1).clamp_min(0.0)
    cos = F.cosine_similarity(student[idx_i], student[idx_j], dim=-1)
    return (w * (1.0 - cos) ** 2).mean()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv_cups_cpu/bin/python -m pytest mbps_pytorch/tests/test_fusion_adapter.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add mbps_pytorch/models/semantic/cross_model_adapter.py mbps_pytorch/tests/test_fusion_adapter.py
git commit -m "feat(fusion): CrossModelAdapter + stratified sampler + teacher loss (TDD)"
```

---

### Task 6: Fusion adapter trainer

**Files:**
- Create: `mbps_pytorch/train_fusion_adapter.py`
- Modify: `mbps_pytorch/tests/test_fusion_adapter.py` (append dataset test)

- [ ] **Step 1: Append the failing dataset-alignment test**

```python
# append to mbps_pytorch/tests/test_fusion_adapter.py

def test_fusion_dataset_alignment(tmp_path) -> None:
    """Side A: cond g aligned to z grid (32x64). Side B: cond z aligned to g grid (40x80)."""
    import numpy as np
    from mbps_pytorch.train_fusion_adapter import FusionPairDataset

    city = tmp_path / "cause_z" / "train" / "x"
    gcity = tmp_path / "depthg_g_mono" / "train" / "x"
    city.mkdir(parents=True); gcity.mkdir(parents=True)
    np.save(city / "im0_codes.npy", np.random.randn(32, 64, 90).astype(np.float16))
    np.save(city / "im0_depth.npy", np.random.rand(32, 64).astype(np.float16))
    np.save(gcity / "im0_g.npy", np.random.randn(40, 80, 100).astype(np.float16))

    ds_a = FusionPairDataset(str(tmp_path), "train", side="A", g_subdir="depthg_g_mono")
    item = ds_a[0]
    assert item["codes"].shape == (32 * 64, 90)
    assert item["cond"].shape == (32 * 64, 100)
    assert item["depth"].shape == (32 * 64,)
    assert tuple(item["spatial_shape"].tolist()) == (32, 64)

    ds_b = FusionPairDataset(str(tmp_path), "train", side="B", g_subdir="depthg_g_mono")
    item = ds_b[0]
    assert item["codes"].shape == (40 * 80, 100)
    assert item["cond"].shape == (40 * 80, 90)
    assert tuple(item["spatial_shape"].tolist()) == (40, 80)
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv_cups_cpu/bin/python -m pytest mbps_pytorch/tests/test_fusion_adapter.py::test_fusion_dataset_alignment -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` on `train_fusion_adapter`.

- [ ] **Step 3: Write the trainer**

Clone the structure of `mbps_pytorch/train_depth_adapter.py` (AdamW lr=1e-3 wd=1e-4, CosineAnnealingLR eta_min=1e-5, grad clip 1.0, best-by-val-loss checkpoint dict with full config). New content:

```python
#!/usr/bin/env python3
"""Train the DepthG x CAUSE-TR cross-model fusion adapters (spec 2026-06-12 rev 2).

Side A: codes = CAUSE z (32x64x90), cond = DepthG g bilinear-aligned to 32x64.
Side B: codes = DepthG g (40x80x100), cond = CAUSE z bilinear-aligned to 40x80.

teacher_mode:
    plain  A1: all 1024 pairs uniform, cross-teacher only (no stratification)
    strat  A2/B1: 512 short pairs cross/self-taught per spec 4.4/4.5
    dual   A3: strat, short-range weight = mean(cross-teacher cosine, depth kernel)

Loss (strat, side A):
    L = corr(short; teacher=g) + corr(long; teacher=frozen z)
      + lambda_preserve * MSE(z', z)
Side B swaps the teacher roles (long=cross z, short=self g).

Smoke:
    .venv_cups_cpu/bin/python mbps_pytorch/train_fusion_adapter.py \
        --side A --teacher_mode strat --epochs 1 --limit_train_images 8 \
        --output_dir results/fusion_adapter/smoke
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mbps_pytorch.models.semantic.cross_model_adapter import (
    CrossModelAdapter,
    sample_stratified_pairs,
    teacher_guided_correlation_loss,
)

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CACHE = "/Volumes/code_files/datasets/cityscapes/fusion_feature_cache"


def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class FusionPairDataset(Dataset):
    """Loads paired (z, g, depth) grids and aligns the conditioning grid."""

    def __init__(self, cache_root: str, split: str, side: str,
                 g_subdir: str = "depthg_g_mono", limit_images: int = 0) -> None:
        if side not in ("A", "B"):
            raise ValueError(f"side must be A or B, got {side}")
        self.side = side
        self.files: List[Dict[str, str]] = []
        z_root = Path(cache_root) / "cause_z" / split
        g_root = Path(cache_root) / g_subdir / split
        for city in sorted(p.name for p in z_root.iterdir() if p.is_dir()):
            for zf in sorted((z_root / city).glob("*_codes.npy")):
                stem = zf.name.replace("_codes.npy", "")
                gf = g_root / city / f"{stem}_g.npy"
                df = z_root / city / f"{stem}_depth.npy"
                if gf.is_file() and df.is_file():
                    self.files.append({"z": str(zf), "g": str(gf), "d": str(df)})
        if limit_images > 0:
            self.files = self.files[:limit_images]

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _align(grid: torch.Tensor, hw: tuple) -> torch.Tensor:
        """(h, w, C) -> bilinear -> (hw[0]*hw[1], C)."""
        x = grid.permute(2, 0, 1).unsqueeze(0)
        x = F.interpolate(x, hw, mode="bilinear", align_corners=False)
        return x.squeeze(0).permute(1, 2, 0).reshape(-1, x.shape[1])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        e = self.files[idx]
        z = torch.from_numpy(np.load(e["z"]).astype(np.float32))   # (32, 64, 90)
        g = torch.from_numpy(np.load(e["g"]).astype(np.float32))   # (40, 80, 100)
        d = torch.from_numpy(np.load(e["d"]).astype(np.float32))   # (32, 64)
        if self.side == "A":
            h, w = z.shape[:2]
            return {"codes": z.reshape(-1, 90), "cond": self._align(g, (h, w)),
                    "depth": d.reshape(-1),
                    "spatial_shape": torch.tensor([h, w], dtype=torch.long)}
        h, w = g.shape[:2]
        d_up = F.interpolate(d[None, None], (h, w), mode="bilinear", align_corners=False)
        return {"codes": g.reshape(-1, 100), "cond": self._align(z, (h, w)),
                "depth": d_up.reshape(-1),
                "spatial_shape": torch.tensor([h, w], dtype=torch.long)}


def fusion_loss(adjusted: torch.Tensor, codes: torch.Tensor, cond: torch.Tensor,
                depth: torch.Tensor, hw: tuple, side: str, teacher_mode: str,
                lambda_preserve: float, sigma_d: float = 0.5) -> Dict[str, torch.Tensor]:
    """Per-batch fusion loss. adjusted/codes: (B,N,Dc); cond: (B,N,Dx); depth: (B,N)."""
    b = adjusted.shape[0]
    device = adjusted.device
    l_corr = torch.tensor(0.0, device=device)
    for i in range(b):
        if teacher_mode == "plain":
            n = adjusted.shape[1]
            idx_i = torch.randint(0, n, (1024,), device=device)
            idx_j = torch.randint(0, n, (1024,), device=device)
            l_corr = l_corr + teacher_guided_correlation_loss(adjusted[i], cond[i], idx_i, idx_j)
            continue
        (si, sj), (li, lj) = sample_stratified_pairs(hw[0], hw[1], device=device)
        # side A: short <- cross-teacher (DepthG cond), long <- self-teacher (frozen z codes)
        # side B: short <- self-teacher (frozen g codes), long <- cross-teacher (CAUSE cond)
        if side == "A":
            l_short = teacher_guided_correlation_loss(adjusted[i], cond[i], si, sj)
            l_long = teacher_guided_correlation_loss(adjusted[i], codes[i].detach(), li, lj)
        else:
            l_short = teacher_guided_correlation_loss(adjusted[i], codes[i].detach(), si, sj)
            l_long = teacher_guided_correlation_loss(adjusted[i], cond[i], li, lj)
        if teacher_mode == "dual" and side == "A":
            with torch.no_grad():
                w_d = torch.exp(-(depth[i][si] - depth[i][sj]) ** 2 / (2 * sigma_d ** 2))
                w_t = F.cosine_similarity(cross[si], cross[sj], dim=-1).clamp_min(0.0)
                w = 0.5 * (w_d + w_t)
            cos = F.cosine_similarity(adjusted[i][si], adjusted[i][sj], dim=-1)
            l_short = (w * (1.0 - cos) ** 2).mean()
        l_corr = l_corr + l_short + l_long
    l_corr = l_corr / b
    l_pres = F.mse_loss(adjusted, codes)
    return {"loss": l_corr + lambda_preserve * l_pres, "corr": l_corr, "preserve": l_pres}


def run_epoch(adapter, loader, device, args, optimizer=None, epoch=0) -> Dict[str, float]:
    is_train = optimizer is not None
    adapter.train() if is_train else adapter.eval()
    totals = {"loss": 0.0, "corr": 0.0, "preserve": 0.0, "drift": 0.0}
    count = 0
    with torch.enable_grad() if is_train else torch.no_grad():
        for step, batch in enumerate(loader):
            codes = batch["codes"].to(device)
            cond = batch["cond"].to(device)
            depth = batch["depth"].to(device)
            hw = tuple(batch["spatial_shape"][0].tolist())
            if is_train:
                optimizer.zero_grad()
            adjusted = adapter(codes, cond)
            parts = fusion_loss(adjusted, codes, cond, depth, hw, args.side,
                                args.teacher_mode, args.lambda_preserve)
            if is_train:
                parts["loss"].backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
                optimizer.step()
            totals["loss"] += parts["loss"].item()
            totals["corr"] += parts["corr"].item()
            totals["preserve"] += parts["preserve"].item()
            totals["drift"] += (adjusted - codes).norm(dim=-1).mean().item()
            count += 1
            if is_train and step % 20 == 0:
                logger.info("ep%d step %d/%d loss=%.4f corr=%.4f pres=%.6f",
                            epoch, step, len(loader), parts["loss"].item(),
                            parts["corr"].item(), parts["preserve"].item())
    return {k: v / max(count, 1) for k, v in totals.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_root", default=DEFAULT_CACHE)
    p.add_argument("--side", required=True, choices=("A", "B"))
    p.add_argument("--teacher_mode", default="strat", choices=("plain", "strat", "dual"))
    p.add_argument("--g_subdir", default="depthg_g_mono")
    p.add_argument("--proj_width", type=int, default=16)
    p.add_argument("--hidden_dim", type=int, default=384)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--lambda_preserve", type=float, default=20.0)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--limit_train_images", type=int, default=0)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.teacher_mode == "dual" and args.side != "A":
        p.error("--teacher_mode dual is defined for side A only (spec 4.4 run A3)")

    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto"
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    os.makedirs(args.output_dir, exist_ok=True)

    full = FusionPairDataset(args.cache_root, "train", args.side,
                             g_subdir=args.g_subdir, limit_images=args.limit_train_images)
    n_val = max(1, int(len(full) * args.val_fraction))
    train_ds = torch.utils.data.Subset(full, range(len(full) - n_val))
    val_ds = torch.utils.data.Subset(full, range(len(full) - n_val, len(full)))
    logger.info("side=%s mode=%s train=%d val=%d", args.side, args.teacher_mode,
                len(train_ds), len(val_ds))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    code_dim, cond_dim = (90, 100) if args.side == "A" else (100, 90)
    adapter = CrossModelAdapter(code_dim, cond_dim, args.proj_width,
                                args.hidden_dim, args.num_layers).to(device)
    adapter_config = {"side": args.side, "code_dim": code_dim, "cond_dim": cond_dim,
                      "proj_width": args.proj_width, "hidden_dim": args.hidden_dim,
                      "num_layers": args.num_layers, "teacher_mode": args.teacher_mode,
                      "g_subdir": args.g_subdir,
                      "pair_offsets": {"r_short": 4, "r_long": 8}}
    logger.info("adapter params=%d", sum(q.numel() for q in adapter.parameters()))

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr)

    best_val = float("inf")
    best_ckpt = os.path.join(args.output_dir, "best.pt")
    t0 = time.time()
    for epoch in range(args.epochs):
        tr = run_epoch(adapter, train_loader, device, args, optimizer, epoch)
        va = run_epoch(adapter, val_loader, device, args)
        logger.info("ep %d/%d | train loss=%.4f corr=%.4f pres=%.6f drift=%.4f | "
                    "val loss=%.4f drift=%.4f | %.0fs",
                    epoch, args.epochs - 1, tr["loss"], tr["corr"], tr["preserve"],
                    tr["drift"], va["loss"], va["drift"], time.time() - t0)
        scheduler.step()
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save({"state_dict": adapter.state_dict(),
                        "adapter_config": adapter_config,
                        "epoch": epoch, "val_loss": best_val,
                        "train_args": vars(args)}, best_ckpt)
            logger.info("  -> new best val=%.4f saved %s", best_val, best_ckpt)
    logger.info("done. best val=%.4f ckpt=%s", best_val, best_ckpt)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the dataset test, then the full test file**

```bash
.venv_cups_cpu/bin/python -m pytest mbps_pytorch/tests/test_fusion_adapter.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Smoke-train one epoch on 8 cached images (needs Task 4 Step 2 output)**

```bash
.venv_cups_cpu/bin/python mbps_pytorch/train_fusion_adapter.py \
    --side A --teacher_mode strat --epochs 1 --limit_train_images 8 --batch_size 2 \
    --output_dir results/fusion_adapter/smoke_A
.venv_cups_cpu/bin/python mbps_pytorch/train_fusion_adapter.py \
    --side B --teacher_mode strat --epochs 1 --limit_train_images 8 --batch_size 2 \
    --output_dir results/fusion_adapter/smoke_B
```

Expected: both finish without error; `best.pt` exists in each output dir; logged drift starts near 0 (identity init).

- [ ] **Step 6: Commit**

```bash
git add mbps_pytorch/train_fusion_adapter.py mbps_pytorch/tests/test_fusion_adapter.py
git commit -m "feat(fusion): fusion adapter trainer with stratified teacher loss"
```

---

### Task 7: Eval glue — adapted codes through the official protocols

**Files:**
- Create: `mbps_pytorch/eval_fusion_adapter.py`

This script has two modes per side: `--vanilla` (no adapter — must reproduce Task 3's numbers, proving the glue is protocol-faithful) and adapter mode (`--adapter_ckpt`). It can also `--dump_preds <dir>` (argmax PNGs at GT resolution, val split) for the Task 8 audit.

- [ ] **Step 1: Write the eval glue**

Structure (full file ~300 lines; the load/CRF/NiceTool code is copied from the named sources, not reinvented):

```python
#!/usr/bin/env python3
"""Evaluate vanilla or fusion-adapted codes under each baseline paper's protocol.

Side A (CAUSE): mirrors refs/cause/eval_cause_tr_dinov2.py exactly — DINOv2
backbone -> head_ema -> [optional CrossModelAdapter] -> interpolate ->
cluster.forward_centroid (FROZEN cluster_tr probe) -> NiceTool Hungarian,
with the same CRF + flip protocol. The ONLY insertion is the adapter applied
to the (B, N, 90) head_ema tokens, conditioned on DepthG codes computed live
for the same image (640x1280 -> half-res flip-averaged code -> bilinear to
the CAUSE token grid).

Side B (DepthG): mirrors the metrics.json eval branch of
refs/depthg/src/train_segmentation.py — LitUnsupervisedSegmenter from ckpt,
their val dataset/transforms, code -> [optional adapter] -> frozen
cluster_probe + linear_probe -> UnsupervisedMetrics, same CRF setting.
Conditioning z is computed live on the cropped val tensor via the CAUSE
backbone + head_ema (no spatial bookkeeping).

Usage:
    # protocol-fidelity check (must match Task 3 numbers):
    .venv_cups_cpu/bin/python mbps_pytorch/eval_fusion_adapter.py --side A --vanilla
    # adapted run:
    .venv_cups_cpu/bin/python mbps_pytorch/eval_fusion_adapter.py --side A \
        --adapter_ckpt results/fusion_adapter/A2_strat_w16/best.pt
    # audit dumps:
    .venv_cups_cpu/bin/python mbps_pytorch/eval_fusion_adapter.py --side A --vanilla \
        --dump_preds /Volumes/code_files/datasets/cityscapes/fusion_audit/cause_vanilla
"""
```

Implementation requirements (each is a concrete copy-from-source instruction):
1. Side A: copy `NiceTool`, `dense_crf`/`do_crf`, `load_backbone`, `load_segment_tr`, `load_cluster_tr`, and the `test_with_crf` loop from `refs/cause/eval_cause_tr_dinov2.py` (import the script's helpers via `sys.path.insert(0, 'refs/cause')` + `importlib.util.spec_from_file_location` so nothing is duplicated; fall back to copying functions only if import-by-path fails). Insert after `seg_feat_ema = segment.head_ema(feat)`:

```python
if adapter is not None:
    b, n, _ = seg_feat_ema.shape
    th, tw = token_grid_hw(img)            # img_h//14, img_w//14
    g = depthg_code_for(batch_paths)       # (b, 100, 40, 80) live, fp32
    g = F.interpolate(g, (th, tw), mode="bilinear", align_corners=False)
    cond = g.permute(0, 2, 3, 1).reshape(b, n, 100)
    seg_feat_ema = adapter(seg_feat_ema, cond)
```

`depthg_code_for` loads the original image at 640×1280 and runs the same flip-averaged half-res forward as `cache_fusion_features.depthg_codes` (import that function). The CAUSE val dataloader must expose image paths or indices; if `ContrastiveSegDataset` returns only tensors, recover the path list from the dataset object's internal file list (it is constructed sorted) and index by batch order with `shuffle=False`.
2. Side B: load `LitUnsupervisedSegmenter` from the checkpoint exactly as the metrics.json branch does (Task 1 Step 3 located it), build their val dataset, iterate with `shuffle=False`; per batch: `code = model(img)` per their validation_step (`refs/depthg/src/train_segmentation.py:471-487`), reshape code `(B, 100, h, w)` to tokens, compute `z_live = segment.head_ema(dinov2(img))` on the same (cropped) tensor — pad image H/W to multiples of 14 with `F.interpolate` to the nearest multiple — align to `(h, w)`, apply adapter, reshape back, then run the FROZEN `model.cluster_probe` and `model.linear_probe` and `UnsupervisedMetrics` with the same CRF setting as the reproduction. Print `final/cluster/mIoU` and `final/linear/mIoU`.
3. `--dump_preds`: after computing per-image argmax at label resolution, save `{stem}_pred.png` (uint8 class ids) into the given dir. Works in both vanilla and adapted modes, both sides.
4. Load the adapter with its saved `adapter_config` (recurring pattern P086): construct `CrossModelAdapter(**{k: cfg[k] for k in ("code_dim","cond_dim","proj_width","hidden_dim","num_layers")})`, then `load_state_dict`.
5. Secondary readout (spec §4.6): flag `--refit_kmeans` fits 27-way spherical k-means (L2-normalized codes, `sklearn.cluster.KMeans(n_clusters=27, n_init=10, random_state=42)` on ≤500k subsampled train tokens loaded from the fusion cache, save centroids `.npz` next to the adapter ckpt, reuse if present) and reports its Hungarian mIoU alongside the frozen-probe number.

- [ ] **Step 2: Protocol-fidelity check (the glue's own gate)**

```bash
nohup .venv_cups_cpu/bin/python mbps_pytorch/eval_fusion_adapter.py --side A --vanilla \
    > logs/glue_check_A_$(date +%Y%m%d_%H%M%S).log 2>&1 &
nohup .venv_cups_cpu/bin/python mbps_pytorch/eval_fusion_adapter.py --side B --vanilla \
    > logs/glue_check_B_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Expected: side A reproduces Task 3 Step 1's mIoU within ±0.3 (same protocol, same code path); side B reproduces ≈14.8 (mono ckpt default) and, with `--depthg_ckpt refs/depthg/saved_models/cityscapes_vit_base_1.ckpt`, ≈20.94. Any mismatch is a glue bug — fix before proceeding (this is the cheapest place to catch protocol drift).

- [ ] **Step 3: Commit**

```bash
git add mbps_pytorch/eval_fusion_adapter.py
git commit -m "feat(fusion): official-protocol eval glue with adapter injection + pred dumps"
```

---

### Task 8: Phase 0b — complementarity audit (KILL GATE)

**Files:**
- Modify: `notebooks/compare_retrained_vs_cups_baseline.ipynb` (append audit section)

- [ ] **Step 1: Dump val predictions for both vanilla models**

```bash
nohup .venv_cups_cpu/bin/python mbps_pytorch/eval_fusion_adapter.py --side A --vanilla \
    --dump_preds /Volumes/code_files/datasets/cityscapes/fusion_audit/cause_vanilla \
    > logs/audit_dump_A_$(date +%Y%m%d_%H%M%S).log 2>&1 &
nohup .venv_cups_cpu/bin/python mbps_pytorch/eval_fusion_adapter.py --side B --vanilla \
    --dump_preds /Volumes/code_files/datasets/cityscapes/fusion_audit/depthg_mono \
    > logs/audit_dump_B_mono_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# after those finish, the official-ckpt DepthG dump (audit reference):
nohup .venv_cups_cpu/bin/python mbps_pytorch/eval_fusion_adapter.py --side B --vanilla \
    --depthg_ckpt refs/depthg/saved_models/cityscapes_vit_base_1.ckpt \
    --dump_preds /Volumes/code_files/datasets/cityscapes/fusion_audit/depthg_official \
    > logs/audit_dump_B_off_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

- [ ] **Step 2: Append audit cells to the notebook**

Cells (each model's preds are Hungarian-mapped to GT classes by the dump's own eval — the dumped PNGs store mapped class ids):
1. Per-class IoU table: CAUSE vanilla vs DepthG (mono and official), 27 classes, highlighting traffic light, motorcycle, pole, person.
2. Agreement maps: for 6 sample images, render agree-right / agree-wrong / A-only-right / B-only-right masks.
3. GT-oracle: `oracle = where(cause_correct, cause_pred, where(depthg_correct, depthg_pred, cause_pred))`; report oracle mIoU and headroom = oracle − max(vanilla mIoUs). GT is analysis-only.
4. Concat sanity: load 200 val images' live features is too slow in a notebook — instead subsample 300k tokens from the *train* fusion cache (z aligned + g), per-branch whiten (subtract mean, divide std) + L2-normalize, concat to 190-d, `KMeans(27)`, Hungarian-match against GT pooled to the token grid, report mIoU vs the same protocol on z alone.

- [ ] **Step 3: Apply the kill gate and record the verdict**

PASS = oracle headroom ≥ ~1.5 mIoU over the stronger vanilla model AND disagreements not dominated by both-wrong. Record PASS/KILL + the headroom number in `reports/fusion_adapter_numbers_ledger.md`. On KILL: write the one-page negative note `reports/depthg_cause_fusion_negative_note.md`, commit, and STOP the plan (per spec Phase 0). If only the official DepthG ckpt shows complementarity, record that as a mono-claim kill signal and decide with the user before training.

```bash
git add notebooks/compare_retrained_vs_cups_baseline.ipynb reports/fusion_adapter_numbers_ledger.md
git commit -m "feat(fusion): phase 0b complementarity audit + gate verdict"
```

---

### Task 9: Phase 1A — train the A run matrix

**Files:** none created — four training runs (needs Task 4 full cache + Task 8 PASS).

- [ ] **Step 1: Launch A1, A2 (primary), A3, A4 sequentially or two-at-a-time**

```bash
for RUN in "A1 plain 16" "A2 strat 16" "A3 dual 16" "A4 strat 32"; do
  set -- $RUN
  nohup .venv_cups_cpu/bin/python mbps_pytorch/train_fusion_adapter.py \
      --side A --teacher_mode $2 --proj_width $3 \
      --epochs 50 --batch_size 32 --lambda_preserve 20.0 \
      --output_dir results/fusion_adapter/$1_$2_w$3 \
      > logs/fusion_$1_$(date +%Y%m%d_%H%M%S).log 2>&1
done &
```

Expected: each run logs decreasing val loss; drift grows from 0 and stabilizes (if drift explodes past ~the code norm, the preservation anchor failed — investigate before evaluating). Roughly hours per run on MPS (2048 tokens/image, 32-image batches, ~84 steps/epoch).

- [ ] **Step 2: Evaluate all four under the official CAUSE protocol**

```bash
for RUN in A1_plain_w16 A2_strat_w16 A3_dual_w16 A4_strat_w32; do
  nohup .venv_cups_cpu/bin/python mbps_pytorch/eval_fusion_adapter.py --side A \
      --adapter_ckpt "results/fusion_adapter/${RUN}/best.pt" \
      > "logs/eval_${RUN}_$(date +%Y%m%d_%H%M%S).log" 2>&1
done &
```

- [ ] **Step 3: Record results + apply Gate 1A**

Append every run's cluster mIoU (frozen-probe primary, `--refit_kmeans` secondary for the best run) to the ledger. Gate 1A: best adapted ≥ published CAUSE-TR + 1.0. Record pass/fail either way. Commit ledger.

---

### Task 10: Phase 1A attribution row — DCFA under the official protocol

**Files:** none created (one extra eval run).

The benchmark table needs vanilla → +DCFA → +Adapter A attribution under the SAME protocol. The DCFA V3 checkpoint loads with `mbps_pytorch/models/semantic/depth_adapter.py::DepthAdapter`.

- [ ] **Step 1: Locate the trained DCFA V3 checkpoint**

```bash
ls results/depth_adapter*/best.pt 2>/dev/null; ls checkpoints/ | grep -i adapter
grep -rn "adapter" reports/depth_semantic_ablation_complete.md | grep -i "ckpt\|path" | head -5
```

If no checkpoint is found, retrain it (cheap, the original command from `train_depth_adapter.py`'s docstring with `--depth_dim 16 --hidden_dim 384 --num_layers 2 --lambda_preserve 20.0` on the Task 4 cache — the cache layout matches `PreextractedCodesDataset` expectations with `--codes_subdir fusion_feature_cache/cause_z`).

- [ ] **Step 2: Add `--dcfa_ckpt` support to the side-A eval glue**

In `eval_fusion_adapter.py`, when `--dcfa_ckpt` is given instead of `--adapter_ckpt`: load `DepthAdapter` from the checkpoint's saved kwargs, condition on `sinusoidal_depth_encode` of the live DepthPro depth pooled to the token grid (reuse `pooled_depth` from `cache_fusion_features.py`), same insertion point. Run it; record the DCFA row in the ledger. Commit the glue change:

```bash
git add mbps_pytorch/eval_fusion_adapter.py
git commit -m "feat(fusion): DCFA attribution row support in side-A eval glue"
```

---

### Task 11: Phase 1B — train and evaluate the B run matrix

**Files:** none created — two training runs + evals (needs Task 4 cache + Task 8 PASS).

- [ ] **Step 1: Launch B1 (primary) and B2**

```bash
for RUN in "B1 strat 16" "B2 strat 64"; do
  set -- $RUN
  nohup .venv_cups_cpu/bin/python mbps_pytorch/train_fusion_adapter.py \
      --side B --teacher_mode $2 --proj_width $3 \
      --epochs 50 --batch_size 32 --lambda_preserve 20.0 \
      --output_dir results/fusion_adapter/$1_$2_w$3 \
      > logs/fusion_$1_$(date +%Y%m%d_%H%M%S).log 2>&1
done &
```

- [ ] **Step 2: Evaluate both under the official DepthG protocol (mono substrate)**

```bash
for RUN in B1_strat_w16 B2_strat_w64; do
  nohup .venv_cups_cpu/bin/python mbps_pytorch/eval_fusion_adapter.py --side B \
      --adapter_ckpt "results/fusion_adapter/${RUN}/best.pt" \
      > "logs/eval_${RUN}_$(date +%Y%m%d_%H%M%S).log" 2>&1
done &
```

- [ ] **Step 3: Record results + apply Gate 1B**

Ledger rows: cluster + linear mIoU per run (frozen probes). Gate 1B primary: best adapted mono DepthG cluster mIoU > published DepthG cluster mIoU. Secondary: delta vs 14.8. Commit ledger.

---

### Task 12: Phase 2 — benchmark table, report, memory

**Files:**
- Create: `reports/depthg_cause_fusion_adapter_report.md`

- [ ] **Step 1: Write the report**

Sections: (1) verified-numbers ledger (inlined); (2) benchmark table — published rows (STEGO if available from the DepthG paper's table, DepthG, CAUSE-TR; every number cited to the ledger), reproduction rows, attribution rows (CAUSE-TR → +DCFA → +Adapter A; DepthG-official / DepthG-mono → +Adapter B), with protocol column (CRF, probe type); (3) full run matrix A1-A4/B1-B2 with both readouts; (4) gates pass/fail; (5) negative results and anomalies; (6) reproducibility index (every command + log path + ckpt path); (7) next actions — Phase 3 entry decision per spec, and the novelty literature check (spec risk 8) that must pass before any paper claim is drafted from these results.

- [ ] **Step 2: Self-check the report numbers**

Every number in the report must trace to a log file or the ledger — grep each one: `grep -rn "<number>" logs/ reports/fusion_adapter_numbers_ledger.md`. No number appears that is not in a log.

- [ ] **Step 3: Commit + update memory**

```bash
git add reports/depthg_cause_fusion_adapter_report.md
git commit -m "docs(fusion): DepthG x CAUSE-TR fusion adapter benchmark report"
```

Then update the auto-memory: write `memory/depthg_cause_fusion_results.md` (one-fact file with the headline numbers + gate verdicts + report path) and add its one-line pointer to `MEMORY.md`. Call `gcc_commit` with the experiment record (id `fusion-adapters-v1`, metrics from the ledger, gate conclusions).

---

## Execution-order notes

- Task 4 Step 3's full cache build (~hours) runs in the background; Tasks 5-7 don't need the full cache (smoke uses the 3-image cache from Task 4 Step 2). Task 8 needs Task 7; Tasks 9/11 need the full cache AND Task 8 PASS.
- Tasks 9 and 11 are independent of each other (decoupled adapters, shared cache) — run them in parallel if MPS memory allows, else sequentially.
- The KILL GATE (Task 8 Step 3) is the only sanctioned early exit; everything after it assumes PASS.
- Phase 3 (coupled bridge) is intentionally absent — it gets its own spec + plan only if Gates 1A AND 1B both pass (spec §5).
