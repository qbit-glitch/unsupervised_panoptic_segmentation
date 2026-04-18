# Stage-2 Mask2Former + ViT-Adapter Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the frozen-backbone Cascade Mask R-CNN decoder in `refs/cups/` with a Mask2Former + ViT-Adapter decoder, then ablate 10 guaranteed levers (G1-G10) and 5 novel levers (N1-N5) to raise Cityscapes PQ from ~24.7 to >=30 (+5 PQ_stuff to ~37) on frozen k=80 + DepthPro-tau=0.20 pseudo-labels.

**Architecture:** DINOv3 ViT-B/16 (frozen, 768-dim patch tokens @ stride 16) -> ViT-Adapter (SPM + Injector + Extractor + cross-attention) -> MSDeformAttn pixel decoder -> Mask2Former masked-attention transformer decoder (9 layers, 200 queries split 150 stuff + 50 thing) -> per-pixel logits + per-mask class/mask heads -> panoptic merge (CUPS combine logic). Auxiliary P1-P4 losses threaded through a ctx registry. Self-training (Stage-3) is retained but deferred to Phase 7.

**Tech Stack:** PyTorch 2.5.1+cu121, Lightning 2.x, detectron2, yacs (extended with `_BASE_` inheritance), DINOv3 weights (HF), MSDeformAttn CUDA op (bundled in `refs/cups/cups/model/modeling/ops/` — or fallback to `torch.nn.MultiheadAttention`), dense-CRF via `pydensecrf`.

---

## Pre-read Findings (from summary)

1. `refs/cups/cups/config.py` uses yacs `CfgNode` with `merge_from_file`. **It does NOT support `_BASE_` inheritance natively.** Task 0.16 adds a preprocessing wrapper that walks `_BASE_` chains before merge (required because G-lever/N-lever configs each inherit from M0 or M1/M2).
2. `refs/cups/cups/model/modeling/meta_arch/panoptic_fpn.py` is the current meta-arch (Cascade Mask R-CNN). It uses detectron2's `META_ARCH_REGISTRY`. The new Mask2Former meta-arch registers a *sibling* (`Mask2FormerPanoptic`) and leaves Cascade untouched so rollback is trivial.
3. `refs/cups/cups/model/backbone_dinov3_vit.py` exports `DINOv3ViTBackbone` (frozen by default) and `DINOv3FeaturePyramid` which emits `{p2, p3, p4, p5, vit_patch}`. The ViT-Adapter consumes `vit_patch` (B, 768, H/16, W/16) and ignores the FPN levels.
4. `refs/cups/cups/losses/__init__.py` has a uniform aux-loss registry with signature `(logits, targets, ctx) -> Tensor`. N3 (XQuery) and N4 (QueryConsistency) need **query embeddings**, not logits — they are wired explicitly in `Mask2FormerPanoptic.forward`, NOT through `build_aux_losses`.
5. `refs/cups/cups/pl_model_pseudo.py::training_step` sums `loss_dict.values()` then logs each key under `losses/<key>`. M2F's Hungarian set-criterion produces `loss_ce`, `loss_mask`, `loss_dice` per decoder layer (deep supervision). The Lightning wrapper needs zero changes — it already sums any dict it receives.
6. Existing Mask2Former references to learn from: `refs/dinov3/dinov3/eval/segmentation/models/heads/mask2former_head.py` (MSDeformAttnPixelDecoder + MultiScaleMaskedTransformerDecoder) and `refs/cutler/videocutler/mask2former/modeling/{matcher,criterion,transformer_decoder,pixel_decoder}/`. We port the minimum surface area (no BackgroundRemoval head, no video logic).
7. No existing `vit_adapter*` or `vitadapter*` code in the repo (confirmed via Grep) — this is entirely new code, implemented from scratch against the public ViT-Adapter paper (Chen et al., ICLR 2023).
8. `refs/cups/configs/train_cityscapes_dinov3_vitb_k80_depthpro_tau020.yaml` is the frozen reference for pseudo-label paths and dataset config (k=80 + DepthPro tau=0.20). Every G/N/M config `_BASE_`s this or its M0 descendant. TTA_SCALES=[0.75, 1.0, 1.25]. NUM_STEPS_STARTUP=500.

## Open Decisions (non-blocking)

- **MSDeformAttn op**: Phase 0 ships a pure-PyTorch fallback (slower but portable). Tasks 0.7 and its tests use the fallback; CUDA op integration is a follow-up (deferred — not on critical path).
- **Query pool count**: Fixed at 150 stuff + 50 thing based on k=80 semantic + ~50 instance avg per CUPS dataset stats. Tunable via config, not ablated.
- **Self-training dataset**: Stage-3 retrains on pseudo-labels predicted by the M2 model itself; we reuse the existing CUPS `SelfTrainingDataset` unchanged in Phase 7.

---

## File Structure

```
refs/cups/cups/
  config.py                                   # MODIFY: add M2F config keys + _BASE_ preprocessing
  losses/
    __init__.py                               # already exists
    xquery.py                                 # NEW: N3 cross-image query correspondence loss
    query_consistency.py                      # NEW: N4 teacher-student query alignment loss
    self_train.py                             # NEW: N5 confidence-thresholded pseudo-label sampler
  model/
    model_mask2former.py                      # NEW: builder for Mask2Former + ViT-Adapter
    modeling/
      meta_arch/
        mask2former_panoptic.py               # NEW: Mask2FormerPanoptic meta-arch (registered)
      mask2former/                            # NEW package
        __init__.py
        vit_adapter.py                        # NEW: ViTAdapter wrapper (SPM + injectors + extractors)
        spm.py                                # NEW: Spatial Prior Module (4-stage conv stem)
        injector.py                           # NEW: Injector cross-attention (c_feat <- vit)
        extractor.py                          # NEW: Extractor cross-attention (vit <- c_feat)
        msdeform_pixel_decoder.py             # NEW: MSDeformAttn pixel decoder (pure-torch fallback)
        masked_attention_decoder.py           # NEW: transformer decoder with masked cross-attn
        matcher.py                            # NEW: Hungarian bipartite matcher
        set_criterion.py                      # NEW: SetCriterion (class/mask/dice losses + deep sup.)
        query_pool.py                         # NEW: QueryPool factory (standard / decoupled / depth-bias)
        ema.py                                # NEW: EMA wrapper for teacher (N4, Stage-3)
        swa.py                                # NEW: SWA averaging utility
        augment.py                            # NEW: LSJ (large-scale jitter) + ColorJitter modules
        dense_crf.py                          # NEW: pydensecrf wrapper for TTA post-processing
  pl_model_pseudo.py                          # MODIFY: add build_model_pseudo routing branch
configs/
  stage2_m2f/
    M0_baseline_dinov3_vitb_k80.yaml          # NEW: M2F baseline (random-init decoder)
    G1_EMA.yaml                               # NEW: +EMA teacher
    G2_SWA.yaml                               # NEW: +SWA over last 5 ckpts
    G3_LSJ.yaml                               # NEW: +Large-Scale Jitter (0.1-2.0)
    G4_ColorJitter.yaml                       # NEW: +ColorJitter (0.4, 0.4, 0.4, 0.1)
    G5_DropPath.yaml                          # NEW: +DropPath 0.3 in decoder
    G6_CRF.yaml                               # NEW: +dense-CRF post-processing at val
    G7_LongSchedule.yaml                      # NEW: 20k -> 30k steps
    G8_LargerCrop.yaml                        # NEW: 640x1280 -> 768x1536
    G9_MoreQueries.yaml                       # NEW: 100 -> 200 queries
    G10_DeeperDec.yaml                        # NEW: 6 -> 9 decoder layers
    M1_stacked_guaranteed.yaml                # NEW: M0 + G1..G10 (wins only)
    N1_DecoupledQueries.yaml                  # NEW: 150 stuff + 50 thing pools
    N2_DepthQueryBias.yaml                    # NEW: depth-conditioned query init
    N3_XQuery.yaml                            # NEW: cross-image query correspondence
    N4_QueryConsistency.yaml                  # NEW: teacher-student query alignment
    N5_SelfTrain.yaml                         # NEW: confidence-thresh self-training
    M2_stacked_novel.yaml                     # NEW: M1 + N1..N5 (wins only)
scripts/
  train_stage2_m2f.sh                         # NEW: single-config launcher
  run_stage2_m2f_ablations.sh                 # NEW: sweep runner (G/N/M)
  eval_stage2_m2f.sh                          # NEW: Cityscapes val eval wrapper
tests/stage2_m2f/
  conftest.py                                 # NEW: shared fixtures (tiny input batch, dummy DINOv3)
  test_base_inheritance.py                    # NEW: _BASE_ walk works
  test_spm.py                                 # NEW: SPM output shapes
  test_injector.py                            # NEW: Injector shapes + gradient flow
  test_extractor.py                           # NEW: Extractor shapes
  test_vit_adapter.py                         # NEW: end-to-end ViTAdapter
  test_matcher.py                             # NEW: Hungarian assignment correctness
  test_set_criterion.py                       # NEW: loss values on known input
  test_query_pool.py                          # NEW: factory dispatch + shapes
  test_pixel_decoder.py                       # NEW: MSDeform pixel decoder shapes
  test_masked_attention_decoder.py            # NEW: decoder forward + N2/N5 hooks
  test_meta_arch.py                           # NEW: Mask2FormerPanoptic forward + train/eval modes
  test_builder.py                             # NEW: build_mask2former_vitb() constructs frozen backbone
  test_ema.py                                 # NEW: EMA update rule
  test_swa.py                                 # NEW: SWA average matches expected
  test_augment.py                             # NEW: LSJ + ColorJitter determinism with seed
  test_crf.py                                 # NEW: dense-CRF refines a known mask
  test_xquery.py                              # NEW: XQuery loss uses query embeddings (not logits)
  test_query_consistency.py                   # NEW: teacher-student alignment loss
  test_self_train.py                          # NEW: confidence thresholding filters low-conf
  test_pl_model_routing.py                    # NEW: build_model_pseudo routes META_ARCH correctly
```

---

## Phase 0 - Foundations (M0 baseline + all reusable components)

Phase 0 builds every reusable piece once: yacs `_BASE_`, ViT-Adapter, pixel decoder, transformer decoder, matcher/criterion, query pool factory, meta-arch, builder, EMA/SWA, LSJ/ColorJitter/CRF, the two N-losses with unusual signatures, and the self-training sampler. No G-lever or N-lever config is touched here. By end of Phase 0, `M0_baseline_dinov3_vitb_k80.yaml` trains end-to-end (20k steps) and produces a PQ number that anchors all ablations.

### Task 0.1: Spatial Prior Module (SPM)

**Files:**
- Create: `refs/cups/cups/model/modeling/mask2former/__init__.py`
- Create: `refs/cups/cups/model/modeling/mask2former/spm.py`
- Test: `tests/stage2_m2f/conftest.py`
- Test: `tests/stage2_m2f/test_spm.py`

The SPM is a lightweight 4-stage convolutional stem that produces multi-scale features `(c2, c3, c4)` at strides `(4, 8, 16)` from the raw RGB image. Its output is later fused with DINOv3 patch tokens via the Injector.

- [ ] **Step 1: Create package init and shared test fixture**

`refs/cups/cups/model/modeling/mask2former/__init__.py`:
```python
"""Mask2Former + ViT-Adapter submodules for Stage-2 M2F."""
from __future__ import annotations

__all__ = []
```

`tests/stage2_m2f/conftest.py`:
```python
"""Shared fixtures for Stage-2 Mask2Former tests.

Keeps a tiny deterministic batch (B=2, H=64, W=128) + dummy DINOv3
patch-token tensor so tests run in <1s on CPU.
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def tiny_image_batch() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(2, 3, 64, 128)


@pytest.fixture
def tiny_vit_patch() -> torch.Tensor:
    """(B=2, C=768, H/16=4, W/16=8) — matches tiny_image_batch at stride 16."""
    torch.manual_seed(1)
    return torch.randn(2, 768, 4, 8)


@pytest.fixture
def tiny_gt_panoptic() -> list[dict]:
    """List of 2 gt dicts with (labels, masks) as Mask2Former expects.

    Each sample: labels Long (N,), masks Bool (N, H, W).
    """
    torch.manual_seed(2)
    gts = []
    for _ in range(2):
        N = 3
        labels = torch.tensor([0, 5, 10], dtype=torch.long)
        masks = torch.zeros(N, 64, 128, dtype=torch.bool)
        for i in range(N):
            masks[i, 10 * i : 10 * i + 15, 10 * i : 10 * i + 20] = True
        gts.append({"labels": labels, "masks": masks})
    return gts
```

- [ ] **Step 2: Write failing test for SPM shapes**

`tests/stage2_m2f/test_spm.py`:
```python
from __future__ import annotations

import torch

from cups.model.modeling.mask2former.spm import SpatialPriorModule


def test_spm_emits_three_scales(tiny_image_batch: torch.Tensor) -> None:
    spm = SpatialPriorModule(in_channels=3, embed_dim=768).eval()
    with torch.no_grad():
        c2, c3, c4 = spm(tiny_image_batch)
    # Strides (4, 8, 16) relative to input 64x128.
    assert c2.shape == (2, 768, 16, 32)
    assert c3.shape == (2, 768, 8, 16)
    assert c4.shape == (2, 768, 4, 8)


def test_spm_gradients_flow(tiny_image_batch: torch.Tensor) -> None:
    spm = SpatialPriorModule(in_channels=3, embed_dim=768).train()
    out = spm(tiny_image_batch)
    loss = sum(t.sum() for t in out)
    loss.backward()
    for p in spm.parameters():
        assert p.grad is not None, f"no grad on {p.shape}"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/stage2_m2f/test_spm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cups.model.modeling.mask2former.spm'`.

- [ ] **Step 4: Implement SPM**

`refs/cups/cups/model/modeling/mask2former/spm.py`:
```python
"""Spatial Prior Module (ViT-Adapter, ICLR 2023, Chen et al.).

4-stage convolutional stem producing (c2, c3, c4) at strides (4, 8, 16)
from raw RGB, projected to the ViT embedding dim so they can be cross-
attended with DINOv3 patch tokens in the Injector/Extractor.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

__all__ = ["SpatialPriorModule"]


def _conv_bn_relu(cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SpatialPriorModule(nn.Module):
    """Convolutional stem emitting c2 (stride 4), c3 (stride 8), c4 (stride 16)."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 768, hidden: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            _conv_bn_relu(in_channels, hidden, k=3, s=2, p=1),  # /2
            _conv_bn_relu(hidden, hidden, k=3, s=1, p=1),
            _conv_bn_relu(hidden, hidden, k=3, s=1, p=1),
            _conv_bn_relu(hidden, hidden, k=3, s=2, p=1),  # /4
        )
        self.conv2 = _conv_bn_relu(hidden, hidden * 2, k=3, s=2, p=1)  # /8
        self.conv3 = _conv_bn_relu(hidden * 2, hidden * 4, k=3, s=2, p=1)  # /16
        self.proj2 = nn.Conv2d(hidden, embed_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(hidden * 2, embed_dim, kernel_size=1)
        self.proj4 = nn.Conv2d(hidden * 4, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)                       # B, hidden, H/4, W/4
        c2 = self.proj2(x)                     # B, embed, H/4, W/4
        x2 = self.conv2(x)                     # B, hidden*2, H/8, W/8
        c3 = self.proj3(x2)                    # B, embed, H/8, W/8
        x3 = self.conv3(x2)                    # B, hidden*4, H/16, W/16
        c4 = self.proj4(x3)                    # B, embed, H/16, W/16
        return c2, c3, c4
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/stage2_m2f/test_spm.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add refs/cups/cups/model/modeling/mask2former/__init__.py \
        refs/cups/cups/model/modeling/mask2former/spm.py \
        tests/stage2_m2f/conftest.py \
        tests/stage2_m2f/test_spm.py
git commit -m "feat(m2f): add Spatial Prior Module for ViT-Adapter"
```

### Task 0.2: Injector (ViT <- SPM cross-attention)

**Files:**
- Create: `refs/cups/cups/model/modeling/mask2former/injector.py`
- Test: `tests/stage2_m2f/test_injector.py`

The Injector takes DINOv3 patch tokens (B, C, H/16, W/16) as *query*, concatenated SPM features `(c2, c3, c4)` flattened as *key/value*, and injects spatial priors back into the ViT tokens. Output shape matches the ViT input (additive residual).

- [ ] **Step 1: Write failing test**

`tests/stage2_m2f/test_injector.py`:
```python
from __future__ import annotations

import torch

from cups.model.modeling.mask2former.injector import Injector
from cups.model.modeling.mask2former.spm import SpatialPriorModule


def test_injector_preserves_vit_shape(tiny_image_batch: torch.Tensor, tiny_vit_patch: torch.Tensor) -> None:
    spm = SpatialPriorModule(embed_dim=768).eval()
    injector = Injector(embed_dim=768, num_heads=8).eval()
    with torch.no_grad():
        c2, c3, c4 = spm(tiny_image_batch)
        out = injector(vit_feat=tiny_vit_patch, c2=c2, c3=c3, c4=c4)
    assert out.shape == tiny_vit_patch.shape


def test_injector_gradients_flow(tiny_image_batch: torch.Tensor, tiny_vit_patch: torch.Tensor) -> None:
    spm = SpatialPriorModule(embed_dim=768).train()
    injector = Injector(embed_dim=768, num_heads=8).train()
    c2, c3, c4 = spm(tiny_image_batch)
    out = injector(vit_feat=tiny_vit_patch, c2=c2, c3=c3, c4=c4)
    out.sum().backward()
    for p in injector.parameters():
        assert p.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage2_m2f/test_injector.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement Injector**

`refs/cups/cups/model/modeling/mask2former/injector.py`:
```python
"""Injector: DINOv3 patch tokens (Q) attend to SPM features (K, V).

Adds spatial priors from the convolutional stem back into the frozen ViT
tokens. Followed optionally by an MLP. Residual gating controls how much
of the injected signal is kept (learnable scalar, initialized at 0 so
early training behaves like identity).
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["Injector"]


def _flatten_bchw(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    return x.flatten(2).transpose(1, 2)  # B, H*W, C


class Injector(nn.Module):
    """Cross-attention block where ViT tokens query concatenated SPM features."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 8, mlp_ratio: float = 0.5, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        # Residual gate initialized at 0 (preserves ViT behavior early).
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        vit_feat: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
    ) -> torch.Tensor:
        B, C, Hv, Wv = vit_feat.shape
        q = _flatten_bchw(vit_feat)                      # B, Hv*Wv, C
        kv = torch.cat(
            [_flatten_bchw(c2), _flatten_bchw(c3), _flatten_bchw(c4)],
            dim=1,
        )                                                # B, sum(H*W), C
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm, need_weights=False)
        q = q + self.gate * attn_out
        q = q + self.gate * self.mlp(q_norm)
        return q.transpose(1, 2).reshape(B, C, Hv, Wv)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/stage2_m2f/test_injector.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add refs/cups/cups/model/modeling/mask2former/injector.py \
        tests/stage2_m2f/test_injector.py
git commit -m "feat(m2f): add Injector cross-attention block"
```

### Task 0.3: Extractor (SPM <- ViT cross-attention) and ViTAdapter wrapper

**Files:**
- Create: `refs/cups/cups/model/modeling/mask2former/extractor.py`
- Create: `refs/cups/cups/model/modeling/mask2former/vit_adapter.py`
- Test: `tests/stage2_m2f/test_extractor.py`
- Test: `tests/stage2_m2f/test_vit_adapter.py`

The Extractor is the dual of the Injector: SPM features (Q) attend to ViT tokens (K, V) so the spatial pyramid absorbs semantic content from the frozen backbone. The full ViTAdapter chains one Injector + one Extractor per block, then upsamples to produce a 4-level feature pyramid `{p2, p3, p4, p5}` (strides 4, 8, 16, 32) that the pixel decoder consumes.

- [ ] **Step 1: Write failing tests**

`tests/stage2_m2f/test_extractor.py`:
```python
from __future__ import annotations

import torch

from cups.model.modeling.mask2former.extractor import Extractor
from cups.model.modeling.mask2former.spm import SpatialPriorModule


def test_extractor_preserves_spm_shapes(tiny_image_batch: torch.Tensor, tiny_vit_patch: torch.Tensor) -> None:
    spm = SpatialPriorModule(embed_dim=768).eval()
    extractor = Extractor(embed_dim=768, num_heads=8).eval()
    with torch.no_grad():
        c2, c3, c4 = spm(tiny_image_batch)
        c2_out, c3_out, c4_out = extractor(c2=c2, c3=c3, c4=c4, vit_feat=tiny_vit_patch)
    assert c2_out.shape == c2.shape
    assert c3_out.shape == c3.shape
    assert c4_out.shape == c4.shape
```

`tests/stage2_m2f/test_vit_adapter.py`:
```python
from __future__ import annotations

import torch

from cups.model.modeling.mask2former.vit_adapter import ViTAdapter


class _DummyDino(torch.nn.Module):
    """Replace DINOv3 with a frozen conv that emits (B, 768, H/16, W/16)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 768, kernel_size=16, stride=16, bias=False)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_vit_adapter_emits_p2_p5(tiny_image_batch: torch.Tensor) -> None:
    adapter = ViTAdapter(backbone=_DummyDino(), embed_dim=768, num_blocks=2).eval()
    with torch.no_grad():
        feats = adapter(tiny_image_batch)
    assert set(feats.keys()) == {"p2", "p3", "p4", "p5"}
    # Input 64x128 -> strides 4/8/16/32
    assert feats["p2"].shape == (2, 256, 16, 32)
    assert feats["p3"].shape == (2, 256, 8, 16)
    assert feats["p4"].shape == (2, 256, 4, 8)
    assert feats["p5"].shape == (2, 256, 2, 4)


def test_vit_adapter_backbone_frozen(tiny_image_batch: torch.Tensor) -> None:
    adapter = ViTAdapter(backbone=_DummyDino(), embed_dim=768, num_blocks=2)
    feats = adapter(tiny_image_batch)
    loss = sum(t.sum() for t in feats.values())
    loss.backward()
    # Backbone conv must have no grad (frozen); adapter pieces must have grads.
    assert adapter.backbone.conv.weight.grad is None
    assert any(p.grad is not None for p in adapter.spm.parameters())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/stage2_m2f/test_extractor.py tests/stage2_m2f/test_vit_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement Extractor**

`refs/cups/cups/model/modeling/mask2former/extractor.py`:
```python
"""Extractor: SPM features (Q) attend to DINOv3 patch tokens (K, V)."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .injector import _flatten_bchw

__all__ = ["Extractor"]


class Extractor(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, mlp_ratio: float = 0.5, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
        vit_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shapes = [c2.shape, c3.shape, c4.shape]
        q_flat = torch.cat(
            [_flatten_bchw(c2), _flatten_bchw(c3), _flatten_bchw(c4)],
            dim=1,
        )
        kv_flat = _flatten_bchw(vit_feat)
        q_norm = self.norm_q(q_flat)
        kv_norm = self.norm_kv(kv_flat)
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm, need_weights=False)
        q = q_flat + self.gate * attn_out
        q = q + self.gate * self.mlp(q_norm)
        # Split back to three levels.
        out = []
        offset = 0
        for shape in shapes:
            B, C, H, W = shape
            n = H * W
            out.append(q[:, offset : offset + n, :].transpose(1, 2).reshape(B, C, H, W))
            offset += n
        return tuple(out)  # type: ignore[return-value]
```

- [ ] **Step 4: Implement ViTAdapter wrapper**

`refs/cups/cups/model/modeling/mask2former/vit_adapter.py`:
```python
"""ViT-Adapter wrapper (Chen et al., ICLR 2023).

Frozen DINOv3 backbone + SPM + num_blocks (Injector, Extractor) pairs.
Final output is a 4-level FPN-style dict {p2, p3, p4, p5} at strides
(4, 8, 16, 32), each with pyramid_channels=256.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .extractor import Extractor
from .injector import Injector
from .spm import SpatialPriorModule

__all__ = ["ViTAdapter"]


class ViTAdapter(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int = 768,
        num_blocks: int = 4,
        num_heads: int = 8,
        pyramid_channels: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.spm = SpatialPriorModule(in_channels=3, embed_dim=embed_dim)
        self.injectors = nn.ModuleList([Injector(embed_dim, num_heads) for _ in range(num_blocks)])
        self.extractors = nn.ModuleList([Extractor(embed_dim, num_heads) for _ in range(num_blocks)])
        # Project embed_dim -> pyramid_channels per level.
        self.proj_p2 = nn.Conv2d(embed_dim, pyramid_channels, kernel_size=1)
        self.proj_p3 = nn.Conv2d(embed_dim, pyramid_channels, kernel_size=1)
        self.proj_p4 = nn.Conv2d(embed_dim, pyramid_channels, kernel_size=1)
        # p5 is a strided conv on p4.
        self.to_p5 = nn.Conv2d(pyramid_channels, pyramid_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        vit = self.backbone(x)                       # B, embed, H/16, W/16
        c2, c3, c4 = self.spm(x)                     # strides 4/8/16
        for inj, ext in zip(self.injectors, self.extractors):
            vit = inj(vit_feat=vit, c2=c2, c3=c3, c4=c4)
            c2, c3, c4 = ext(c2=c2, c3=c3, c4=c4, vit_feat=vit)
        p2 = self.proj_p2(c2)
        p3 = self.proj_p3(c3)
        p4 = self.proj_p4(c4)
        p5 = self.to_p5(p4)
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/stage2_m2f/test_extractor.py tests/stage2_m2f/test_vit_adapter.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add refs/cups/cups/model/modeling/mask2former/extractor.py \
        refs/cups/cups/model/modeling/mask2former/vit_adapter.py \
        tests/stage2_m2f/test_extractor.py \
        tests/stage2_m2f/test_vit_adapter.py
git commit -m "feat(m2f): add Extractor and ViTAdapter wrapper"
```

### Task 0.4: Hungarian Matcher

**Files:**
- Create: `refs/cups/cups/model/modeling/mask2former/matcher.py`
- Test: `tests/stage2_m2f/test_matcher.py`

The Hungarian matcher computes bipartite assignment between `Q` predicted masks and `N_gt` ground-truth masks using a cost = `class_cost + dice_cost + focal_cost`.

- [ ] **Step 1: Write failing test**

`tests/stage2_m2f/test_matcher.py`:
```python
from __future__ import annotations

import torch

from cups.model.modeling.mask2former.matcher import HungarianMatcher


def test_matcher_exact_assignment_on_identity() -> None:
    """When Q=N_gt=3 and predictions match gt exactly, indices should be identity."""
    torch.manual_seed(0)
    B, Q, K, H, W = 1, 3, 20, 16, 32
    pred_logits = torch.full((B, Q, K), -10.0)
    pred_masks = torch.full((B, Q, H, W), -10.0)
    gt_labels = torch.tensor([2, 5, 7], dtype=torch.long)
    gt_masks = torch.zeros(3, H, W, dtype=torch.bool)
    for i, c in enumerate(gt_labels.tolist()):
        pred_logits[0, i, c] = 10.0
        pred_masks[0, i] = 10.0
        gt_masks[i, 4 * i : 4 * i + 5, 4 * i : 4 * i + 5] = True
        pred_masks[0, i] = (gt_masks[i].float() * 20.0 - 10.0)

    matcher = HungarianMatcher(
        cost_class=1.0, cost_mask=5.0, cost_dice=5.0, num_points=256
    )
    targets = [{"labels": gt_labels, "masks": gt_masks}]
    outputs = {"pred_logits": pred_logits, "pred_masks": pred_masks}
    indices = matcher(outputs, targets)
    src, tgt = indices[0]
    # Identity assignment (src_i == tgt_i) up to permutation.
    assert sorted(src.tolist()) == [0, 1, 2]
    assert sorted(tgt.tolist()) == [0, 1, 2]
    # Each src_i must map to the tgt_i with the same label.
    for s, t in zip(src.tolist(), tgt.tolist()):
        assert s == t
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage2_m2f/test_matcher.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement matcher**

`refs/cups/cups/model/modeling/mask2former/matcher.py`:
```python
"""Hungarian bipartite matcher for Mask2Former.

Adapted from the reference implementation in
refs/cutler/videocutler/mask2former/modeling/matcher.py with no video
logic and with an optional point sampling to keep cost matrix memory
bounded (num_points random points per mask).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

__all__ = ["HungarianMatcher"]


def _sigmoid_focal_cost(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Batched focal BCE cost between Q logit-vectors and N 0/1-targets."""
    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    focal_pos = alpha * focal_pos
    focal_neg = (1 - alpha) * focal_neg
    # inputs: (Q, P), targets: (N, P) -> (Q, N)
    return focal_pos @ targets.T + focal_neg @ (1 - targets).T


def _dice_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = inputs.sigmoid()
    numerator = 2.0 * (inputs @ targets.T)
    denominator = inputs.sum(-1, keepdim=True) + targets.sum(-1).unsqueeze(0)
    return 1 - (numerator + 1) / (denominator + 1)


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 2.0, cost_mask: float = 5.0, cost_dice: float = 5.0, num_points: int = 12544) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        pred_logits = outputs["pred_logits"]          # B, Q, K
        pred_masks = outputs["pred_masks"]             # B, Q, H, W
        B, Q, K = pred_logits.shape
        indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for b in range(B):
            tgt_labels = targets[b]["labels"]           # N
            tgt_masks = targets[b]["masks"].float()     # N, H, W
            if tgt_labels.numel() == 0:
                indices.append(
                    (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
                )
                continue
            # Sample K' points per mask for tractable matching.
            _, H, W = tgt_masks.shape
            pnts = torch.randint(0, H * W, (self.num_points,), device=pred_masks.device)
            out_mask = pred_masks[b].flatten(1)[:, pnts]        # Q, P
            tgt_mask = tgt_masks.flatten(1)[:, pnts]            # N, P
            # Class cost: -log(softmax[c_gt]).
            prob = pred_logits[b].softmax(-1)                   # Q, K
            cost_class = -prob[:, tgt_labels]                   # Q, N
            cost_mask = _sigmoid_focal_cost(out_mask, tgt_mask) # Q, N
            cost_dice = _dice_cost(out_mask, tgt_mask)          # Q, N
            C = (
                self.cost_class * cost_class
                + self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
            ).cpu()
            row, col = linear_sum_assignment(C.numpy())
            indices.append((torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long)))
        return indices
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/stage2_m2f/test_matcher.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add refs/cups/cups/model/modeling/mask2former/matcher.py \
        tests/stage2_m2f/test_matcher.py
git commit -m "feat(m2f): add Hungarian matcher"
```

### Task 0.5: SetCriterion (class + mask + dice + deep supervision)

**Files:**
- Create: `refs/cups/cups/model/modeling/mask2former/set_criterion.py`
- Test: `tests/stage2_m2f/test_set_criterion.py`

- [ ] **Step 1: Write failing test**

`tests/stage2_m2f/test_set_criterion.py`:
```python
from __future__ import annotations

import torch

from cups.model.modeling.mask2former.matcher import HungarianMatcher
from cups.model.modeling.mask2former.set_criterion import SetCriterion


def test_set_criterion_returns_expected_keys(tiny_gt_panoptic: list[dict]) -> None:
    torch.manual_seed(0)
    B, Q, K, H, W = 2, 4, 20, 64, 128
    outputs = {
        "pred_logits": torch.randn(B, Q, K, requires_grad=True),
        "pred_masks": torch.randn(B, Q, H, W, requires_grad=True),
        "aux_outputs": [
            {"pred_logits": torch.randn(B, Q, K, requires_grad=True),
             "pred_masks": torch.randn(B, Q, H, W, requires_grad=True)}
            for _ in range(2)
        ],
    }
    matcher = HungarianMatcher(num_points=128)
    crit = SetCriterion(num_classes=K, matcher=matcher, weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0}, eos_coef=0.1, losses=("labels", "masks"), num_points=128)
    losses = crit(outputs, tiny_gt_panoptic)
    # Main + 2 aux decoders -> 3 * 3 = 9 entries.
    assert "loss_ce" in losses and "loss_mask" in losses and "loss_dice" in losses
    assert "loss_ce_0" in losses and "loss_ce_1" in losses
    # Must be finite and require grad.
    for k, v in losses.items():
        assert torch.isfinite(v), f"{k} is not finite"
        assert v.requires_grad, f"{k} has no grad"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage2_m2f/test_set_criterion.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement SetCriterion**

`refs/cups/cups/model/modeling/mask2former/set_criterion.py`:
```python
"""Mask2Former SetCriterion (Hungarian + focal + dice + deep supervision).

Closely follows the reference implementation
(refs/cutler/videocutler/mask2former/modeling/criterion.py) but drops the
background-removal/video-specific paths and simplifies point sampling.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .matcher import HungarianMatcher

__all__ = ["SetCriterion"]


def _dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    num = 2.0 * (inputs * targets).sum(-1)
    den = inputs.sum(-1) + targets.sum(-1)
    return 1 - (num + 1) / (den + 1)


def _sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    return loss.mean(1).sum() / max(inputs.shape[0], 1)


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float = 0.1,
        losses: Sequence[str] = ("labels", "masks"),
        num_points: int = 12544,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = dict(weight_dict)
        self.eos_coef = eos_coef
        self.losses = tuple(losses)
        self.num_points = num_points
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _get_src_permutation_idx(self, indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        src_logits = outputs["pred_logits"]                    # B, Q, K+1 (we let last be "no-object")
        B, Q, K1 = src_logits.shape
        target_classes = torch.full((B, Q), self.num_classes, dtype=torch.long, device=src_logits.device)
        for b, (src, tgt) in enumerate(indices):
            target_classes[b, src] = targets[b]["labels"][tgt]
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_masks(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]              # (sum N_b), H, W
        tgt_masks = torch.cat(
            [t["masks"][idx].float() for t, (_, idx) in zip(targets, indices)], dim=0
        )                                                       # (sum N_b), H_gt, W_gt
        if src_masks.numel() == 0:
            zero = src_masks.sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero}
        # Align shapes (resize src to tgt)
        src_masks = F.interpolate(src_masks.unsqueeze(1), size=tgt_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        # Point sampling for mask loss (memory-safe)
        H, W = tgt_masks.shape[-2:]
        pnts = torch.randint(0, H * W, (self.num_points,), device=src_masks.device)
        src_pt = src_masks.flatten(1)[:, pnts]
        tgt_pt = tgt_masks.flatten(1)[:, pnts]
        loss_mask = _sigmoid_focal_loss(src_pt, tgt_pt)
        loss_dice = _dice_loss(src_pt, tgt_pt).mean()
        return {"loss_mask": loss_mask, "loss_dice": loss_dice}

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)
        losses: Dict[str, torch.Tensor] = {}
        for loss_name in self.losses:
            if loss_name == "labels":
                losses.update(self.loss_labels(outputs_without_aux, targets, indices))
            elif loss_name == "masks":
                losses.update(self.loss_masks(outputs_without_aux, targets, indices))
        # Deep supervision on aux outputs.
        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                idx = self.matcher(aux, targets)
                for loss_name in self.losses:
                    if loss_name == "labels":
                        for k, v in self.loss_labels(aux, targets, idx).items():
                            losses[f"{k}_{i}"] = v
                    elif loss_name == "masks":
                        for k, v in self.loss_masks(aux, targets, idx).items():
                            losses[f"{k}_{i}"] = v
        # Apply weight dict.
        weighted = {}
        for k, v in losses.items():
            base = k.rsplit("_", 1)[0] if k.rsplit("_", 1)[-1].isdigit() else k
            w = self.weight_dict.get(base, 1.0)
            weighted[k] = v * w
        return weighted
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/stage2_m2f/test_set_criterion.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add refs/cups/cups/model/modeling/mask2former/set_criterion.py \
        tests/stage2_m2f/test_set_criterion.py
git commit -m "feat(m2f): add SetCriterion with deep supervision"
```

### Task 0.6: QueryPool factory (standard / decoupled / depth-bias)

**Files:**
- Create: `refs/cups/cups/model/modeling/mask2former/query_pool.py`
- Test: `tests/stage2_m2f/test_query_pool.py`

The QueryPool produces the initial `(num_queries, embed_dim)` tensor that seeds the transformer decoder. A **registry** dispatches between:
- `standard` (single pool, num_queries total)
- `decoupled` (N1: two pools, stuff + thing, concatenated)
- `depth_bias` (N2: single pool + FiLM modulation from mean image depth)

- [ ] **Step 1: Write failing test**

`tests/stage2_m2f/test_query_pool.py`:
```python
from __future__ import annotations

import torch

from cups.model.modeling.mask2former.query_pool import build_query_pool


def test_standard_pool_shape() -> None:
    pool = build_query_pool(kind="standard", num_queries=100, embed_dim=256)
    q = pool(batch_size=2)
    assert q.shape == (2, 100, 256)


def test_decoupled_pool_concatenates() -> None:
    pool = build_query_pool(kind="decoupled", num_queries_stuff=150, num_queries_thing=50, embed_dim=256)
    q = pool(batch_size=2)
    assert q.shape == (2, 200, 256)


def test_depth_bias_pool_uses_depth() -> None:
    pool = build_query_pool(kind="depth_bias", num_queries=100, embed_dim=256)
    depth = torch.rand(2, 1, 64, 128)
    q = pool(batch_size=2, depth=depth)
    assert q.shape == (2, 100, 256)
    q_no_depth = pool(batch_size=2)
    # Depth-biased pool differs from no-depth pool.
    assert not torch.allclose(q, q_no_depth)


def test_unknown_kind_raises() -> None:
    import pytest
    with pytest.raises(KeyError):
        build_query_pool(kind="bogus", num_queries=10, embed_dim=16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage2_m2f/test_query_pool.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement QueryPool**

`refs/cups/cups/model/modeling/mask2former/query_pool.py`:
```python
"""QueryPool factory: standard / decoupled (N1) / depth_bias (N2)."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

__all__ = ["build_query_pool", "register_query_pool"]

_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_query_pool(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    def decorator(cls: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        _REGISTRY[name] = cls
        return cls
    return decorator


def build_query_pool(kind: str, **kwargs) -> nn.Module:
    if kind not in _REGISTRY:
        raise KeyError(f"unknown query pool kind={kind}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[kind](**kwargs)


@register_query_pool("standard")
class StandardQueryPool(nn.Module):
    def __init__(self, num_queries: int = 100, embed_dim: int = 256) -> None:
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.query_feat = nn.Embedding(num_queries, embed_dim)

    def forward(self, batch_size: int, **kwargs) -> torch.Tensor:
        return self.query_feat.weight.unsqueeze(0).expand(batch_size, -1, -1)


@register_query_pool("decoupled")
class DecoupledQueryPool(nn.Module):
    """N1: Two learnable pools (stuff + thing) concatenated as one sequence."""

    def __init__(self, num_queries_stuff: int = 150, num_queries_thing: int = 50, embed_dim: int = 256) -> None:
        super().__init__()
        self.num_stuff = num_queries_stuff
        self.num_thing = num_queries_thing
        self.stuff = nn.Embedding(num_queries_stuff, embed_dim)
        self.thing = nn.Embedding(num_queries_thing, embed_dim)

    def forward(self, batch_size: int, **kwargs) -> torch.Tensor:
        q = torch.cat([self.stuff.weight, self.thing.weight], dim=0)
        return q.unsqueeze(0).expand(batch_size, -1, -1)


@register_query_pool("depth_bias")
class DepthBiasQueryPool(nn.Module):
    """N2: FiLM modulation of a standard pool using mean image depth."""

    def __init__(self, num_queries: int = 100, embed_dim: int = 256) -> None:
        super().__init__()
        self.base = nn.Embedding(num_queries, embed_dim)
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2 * embed_dim),
        )

    def forward(self, batch_size: int, depth: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        q = self.base.weight.unsqueeze(0).expand(batch_size, -1, -1).clone()
        if depth is None:
            return q
        # depth: B, 1, H, W -> B, 1 (mean) -> B, 2*embed -> (gamma, beta)
        d = depth.flatten(1).mean(-1, keepdim=True)          # B, 1
        gamma_beta = self.depth_mlp(d)                       # B, 2*C
        C = q.shape[-1]
        gamma, beta = gamma_beta[:, :C].unsqueeze(1), gamma_beta[:, C:].unsqueeze(1)
        return q * (1 + gamma) + beta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/stage2_m2f/test_query_pool.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add refs/cups/cups/model/modeling/mask2former/query_pool.py \
        tests/stage2_m2f/test_query_pool.py
git commit -m "feat(m2f): add QueryPool factory (standard/decoupled/depth_bias)"
```

### Task 0.7: MSDeformAttn Pixel Decoder (pure-torch fallback)

**Files:**
- Create: `refs/cups/cups/model/modeling/mask2former/msdeform_pixel_decoder.py`
- Test: `tests/stage2_m2f/test_pixel_decoder.py`

The pixel decoder takes the 4-level feature pyramid `{p2, p3, p4, p5}` from the ViT-Adapter and produces a high-resolution per-pixel feature map (stride 4) plus three multi-scale maps that seed the masked-attention decoder. The reference implementation uses MSDeformAttn CUDA ops; we ship a pure-torch fallback using standard `nn.MultiheadAttention` over flattened spatial tokens. This is slower (~1.5x) but portable and sufficient for our 640x1280 crops at batch_size=1.

- [ ] **Step 1: Write failing test**

`tests/stage2_m2f/test_pixel_decoder.py`:
```python
from __future__ import annotations

import torch

from cups.model.modeling.mask2former.msdeform_pixel_decoder import MSDeformAttnPixelDecoder


def test_pixel_decoder_shapes() -> None:
    torch.manual_seed(0)
    B, C = 2, 256
    feats = {
        "p2": torch.randn(B, C, 16, 32),
        "p3": torch.randn(B, C, 8, 16),
        "p4": torch.randn(B, C, 4, 8),
        "p5": torch.randn(B, C, 2, 4),
    }
    decoder = MSDeformAttnPixelDecoder(in_channels=C, hidden_dim=C, mask_dim=C, num_layers=3).eval()
    with torch.no_grad():
        mask_feat, multi_scale = decoder(feats)
    # mask_feat at stride 4 (same as p2)
    assert mask_feat.shape == (B, C, 16, 32)
    # multi_scale has 3 entries (p3, p4, p5 after transformer)
    assert len(multi_scale) == 3
    assert multi_scale[0].shape == (B, C, 8, 16)
    assert multi_scale[1].shape == (B, C, 4, 8)
    assert multi_scale[2].shape == (B, C, 2, 4)


def test_pixel_decoder_gradients() -> None:
    torch.manual_seed(1)
    B, C = 1, 128
    feats = {
        "p2": torch.randn(B, C, 16, 32, requires_grad=True),
        "p3": torch.randn(B, C, 8, 16, requires_grad=True),
        "p4": torch.randn(B, C, 4, 8, requires_grad=True),
        "p5": torch.randn(B, C, 2, 4, requires_grad=True),
    }
    decoder = MSDeformAttnPixelDecoder(in_channels=C, hidden_dim=C, mask_dim=C, num_layers=2).train()
    mask_feat, _ = decoder(feats)
    mask_feat.sum().backward()
    assert any(p.grad is not None for p in decoder.parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage2_m2f/test_pixel_decoder.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement pixel decoder**

`refs/cups/cups/model/modeling/mask2former/msdeform_pixel_decoder.py`:
```python
"""MSDeformAttn-style pixel decoder (pure-torch fallback).

Fuses multi-scale pyramid levels via a stack of transformer encoder
layers operating on concatenated flattened tokens, then up-samples a
mask feature map at stride 4.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MSDeformAttnPixelDecoder"]


def _level_embedding(num_levels: int, dim: int) -> nn.Parameter:
    """Learnable level embedding, one vector per pyramid scale."""
    return nn.Parameter(torch.randn(num_levels, dim) * 0.02)


class _MSALayer(nn.Module):
    """Single encoder layer: self-attention over all flattened tokens."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(self, in_channels: int = 256, hidden_dim: int = 256, mask_dim: int = 256, num_layers: int = 6, num_heads: int = 8) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask_dim = mask_dim
        # Project each pyramid level (already hidden_dim) + add level embedding.
        self.level_embed = _level_embedding(num_levels=4, dim=hidden_dim)
        self.layers = nn.ModuleList([_MSALayer(hidden_dim, num_heads) for _ in range(num_layers)])
        # Upsample fused p3 -> p2 resolution to produce mask feature.
        self.lateral_p3 = nn.Conv2d(hidden_dim, mask_dim, kernel_size=1)
        self.lateral_p2 = nn.Conv2d(in_channels, mask_dim, kernel_size=1)
        self.mask_conv = nn.Sequential(
            nn.Conv2d(mask_dim, mask_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, mask_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mask_dim, mask_dim, kernel_size=1),
        )

    def _flatten_levels(self, feats: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[int, int, int]]]:
        """Flatten {p2..p5} (we use p3, p4, p5 for attention) and add level emb."""
        tokens = []
        shapes: List[Tuple[int, int, int]] = []
        for lvl, key in enumerate(["p3", "p4", "p5"]):
            x = feats[key]
            B, C, H, W = x.shape
            shapes.append((B, H, W))
            t = x.flatten(2).transpose(1, 2)                # B, H*W, C
            t = t + self.level_embed[lvl]
            tokens.append(t)
        x_cat = torch.cat(tokens, dim=1)                    # B, sum(H*W), C
        return x_cat, shapes

    def _split_levels(self, x_cat: torch.Tensor, shapes: List[Tuple[int, int, int]]) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        offset = 0
        for (B, H, W) in shapes:
            n = H * W
            t = x_cat[:, offset : offset + n, :]
            out.append(t.transpose(1, 2).reshape(B, -1, H, W))
            offset += n
        return out

    def forward(self, feats: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x_cat, shapes = self._flatten_levels(feats)
        for layer in self.layers:
            x_cat = layer(x_cat)
        multi_scale = self._split_levels(x_cat, shapes)     # list of [p3, p4, p5]
        # Build mask feature at stride 4: upsample p3 + add lateral p2.
        p3 = multi_scale[0]
        p2 = feats["p2"]
        p3_up = F.interpolate(self.lateral_p3(p3), size=p2.shape[-2:], mode="bilinear", align_corners=False)
        mask_feat = self.mask_conv(p3_up + self.lateral_p2(p2))
        return mask_feat, multi_scale
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/stage2_m2f/test_pixel_decoder.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add refs/cups/cups/model/modeling/mask2former/msdeform_pixel_decoder.py \
        tests/stage2_m2f/test_pixel_decoder.py
git commit -m "feat(m2f): add MSDeformAttn pixel decoder (pure-torch fallback)"
```

### Task 0.8: Masked-Attention Transformer Decoder (with N2/N5 hooks)

**Files:**
- Create: `refs/cups/cups/model/modeling/mask2former/masked_attention_decoder.py`
- Test: `tests/stage2_m2f/test_masked_attention_decoder.py`

The transformer decoder runs L layers (default 9) of masked cross-attention against the three multi-scale levels from the pixel decoder. Each layer emits `(pred_logits, pred_masks)`; the last is the final prediction, the rest feed deep supervision. Two hooks:

- **N2 / depth_bias**: The initial query tensor comes from the `QueryPool` factory, which optionally accepts `depth`.
- **N5 / confidence self-train**: A boolean flag `return_query_embeds=True` returns the per-query `(B, Q, C)` embedding so `XQuery` (N3) and `QueryConsistency` (N4) losses can use them.

- [ ] **Step 1: Write failing test**

`tests/stage2_m2f/test_masked_attention_decoder.py`:
```python
from __future__ import annotations

import torch

from cups.model.modeling.mask2former.masked_attention_decoder import MaskedAttentionDecoder
from cups.model.modeling.mask2former.query_pool import build_query_pool


def test_decoder_shapes() -> None:
    torch.manual_seed(0)
    B, C, Q = 2, 256, 100
    multi_scale = [
        torch.randn(B, C, 8, 16),
        torch.randn(B, C, 4, 8),
        torch.randn(B, C, 2, 4),
    ]
    mask_feat = torch.randn(B, C, 16, 32)
    pool = build_query_pool(kind="standard", num_queries=Q, embed_dim=C)
    decoder = MaskedAttentionDecoder(
        hidden_dim=C, num_queries=Q, num_classes=20, num_layers=3, num_heads=8, query_pool=pool,
    ).eval()
    with torch.no_grad():
        out = decoder(mask_feat=mask_feat, multi_scale=multi_scale)
    # Main + 2 aux = 3 entries in aux_outputs? No — we surface main + aux_outputs separately.
    assert out["pred_logits"].shape == (B, Q, 21)   # K + 1 (no-object)
    assert out["pred_masks"].shape == (B, Q, 16, 32)
    assert len(out["aux_outputs"]) == 2
    for aux in out["aux_outputs"]:
        assert aux["pred_logits"].shape == (B, Q, 21)
        assert aux["pred_masks"].shape == (B, Q, 16, 32)


def test_decoder_returns_query_embeds() -> None:
    torch.manual_seed(1)
    B, C, Q = 1, 128, 50
    multi_scale = [torch.randn(B, C, 8, 16)]
    mask_feat = torch.randn(B, C, 16, 32)
    pool = build_query_pool(kind="standard", num_queries=Q, embed_dim=C)
    decoder = MaskedAttentionDecoder(
        hidden_dim=C, num_queries=Q, num_classes=10, num_layers=1, num_heads=4, query_pool=pool,
    ).eval()
    with torch.no_grad():
        out = decoder(mask_feat=mask_feat, multi_scale=multi_scale, return_query_embeds=True)
    assert out["query_embeds"].shape == (B, Q, C)


def test_decoder_uses_depth_bias_pool() -> None:
    """N2 hook: depth-biased query init must produce different output on same input
    when depth differs."""
    torch.manual_seed(2)
    B, C, Q = 2, 128, 32
    multi_scale = [torch.randn(B, C, 4, 8)]
    mask_feat = torch.randn(B, C, 8, 16)
    pool = build_query_pool(kind="depth_bias", num_queries=Q, embed_dim=C)
    decoder = MaskedAttentionDecoder(
        hidden_dim=C, num_queries=Q, num_classes=5, num_layers=1, num_heads=4, query_pool=pool,
    ).eval()
    depth_a = torch.full((B, 1, 64, 128), 0.1)
    depth_b = torch.full((B, 1, 64, 128), 0.9)
    with torch.no_grad():
        out_a = decoder(mask_feat=mask_feat, multi_scale=multi_scale, depth=depth_a)
        out_b = decoder(mask_feat=mask_feat, multi_scale=multi_scale, depth=depth_b)
    assert not torch.allclose(out_a["pred_masks"], out_b["pred_masks"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage2_m2f/test_masked_attention_decoder.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement decoder**

`refs/cups/cups/model/modeling/mask2former/masked_attention_decoder.py`:
```python
"""Masked-attention transformer decoder (Mask2Former).

- L layers of (cross-attn with mask, self-attn, FFN) applied to queries.
- Cycles through multi_scale[0], [1], [2] as cross-attn memory.
- Emits per-layer (pred_logits, pred_masks); last is main, rest are aux.
- Hooks:
    * depth_bias query pool (N2) -> decoder takes optional `depth` kwarg.
    * return_query_embeds -> also return final query embeddings (B, Q, C)
      for N3/N4 losses.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MaskedAttentionDecoder"]


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2)


class _CrossAttnLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(
        self,
        q: torch.Tensor,
        memory: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Cross-attention with optional masked-attention on background queries.
        h = self.cross_norm(q)
        a, _ = self.cross_attn(h, memory, memory, attn_mask=attn_mask, need_weights=False)
        q = q + a
        # Self-attention over queries.
        h = self.self_norm(q)
        a, _ = self.self_attn(h, h, h, need_weights=False)
        q = q + a
        # Feed-forward.
        q = q + self.ff(self.ff_norm(q))
        return q


class MaskedAttentionDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_classes: int = 20,
        num_layers: int = 9,
        num_heads: int = 8,
        query_pool: Optional[nn.Module] = None,
        droppath: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.query_pool = query_pool
        self.layers = nn.ModuleList([_CrossAttnLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)   # +1 for no-object
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.droppath = droppath

    def _pred(self, q: torch.Tensor, mask_feat: torch.Tensor):
        logits = self.class_embed(q)                         # B, Q, K+1
        mask_embed = self.mask_embed(q)                       # B, Q, C
        masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feat)
        return logits, masks, mask_embed

    def forward(
        self,
        mask_feat: torch.Tensor,
        multi_scale: List[torch.Tensor],
        depth: Optional[torch.Tensor] = None,
        return_query_embeds: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B = mask_feat.shape[0]
        q = self.query_pool(batch_size=B, depth=depth) if self.query_pool is not None else None
        assert q is not None, "query_pool must be provided"
        aux_outputs: List[Dict[str, torch.Tensor]] = []
        _, init_masks, _ = self._pred(q, mask_feat)
        # Initial attn mask from first prediction (sets binary threshold).
        for i, layer in enumerate(self.layers):
            level = multi_scale[i % len(multi_scale)]
            memory = _flatten(level)                         # B, N, C
            # Mask-attention: compute binary mask from current prediction
            # at this level's resolution to gate cross-attention.
            with torch.no_grad():
                pred_at_level = F.interpolate(init_masks, size=level.shape[-2:], mode="bilinear", align_corners=False)
                attn_mask = (pred_at_level.sigmoid() < 0.5)
                attn_mask = attn_mask.flatten(2).detach()
                attn_mask = attn_mask.repeat_interleave(layer.cross_attn.num_heads, dim=0)
                attn_mask = attn_mask.where(attn_mask.sum(-1, keepdim=True) != attn_mask.shape[-1], torch.zeros_like(attn_mask))
            q = layer(q, memory, attn_mask=attn_mask)
            logits, masks, _ = self._pred(q, mask_feat)
            # DropPath during training (simple stochastic depth).
            if self.training and self.droppath > 0.0 and torch.rand(1).item() < self.droppath:
                continue
            if i < self.num_layers - 1:
                aux_outputs.append({"pred_logits": logits, "pred_masks": masks})
            init_masks = masks
        out = {"pred_logits": logits, "pred_masks": masks, "aux_outputs": aux_outputs}
        if return_query_embeds:
            out["query_embeds"] = q
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/stage2_m2f/test_masked_attention_decoder.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add refs/cups/cups/model/modeling/mask2former/masked_attention_decoder.py \
        tests/stage2_m2f/test_masked_attention_decoder.py
git commit -m "feat(m2f): add masked-attention transformer decoder"
```

### Task 0.9: Mask2FormerPanoptic meta-arch

**Files:**
- Create: `refs/cups/cups/model/modeling/meta_arch/mask2former_panoptic.py`
- Test: `tests/stage2_m2f/test_meta_arch.py`

This is the end-to-end meta-architecture. It orchestrates the ViT-Adapter backbone, pixel decoder, transformer decoder, and the SetCriterion during training, and panoptic merge during eval.

- [ ] **Step 1: Write failing test**

`tests/stage2_m2f/test_meta_arch.py`:
```python
from __future__ import annotations

import torch

from cups.model.modeling.meta_arch.mask2former_panoptic import Mask2FormerPanoptic
from cups.model.modeling.mask2former.matcher import HungarianMatcher
from cups.model.modeling.mask2former.masked_attention_decoder import MaskedAttentionDecoder
from cups.model.modeling.mask2former.msdeform_pixel_decoder import MSDeformAttnPixelDecoder
from cups.model.modeling.mask2former.query_pool import build_query_pool
from cups.model.modeling.mask2former.set_criterion import SetCriterion
from cups.model.modeling.mask2former.vit_adapter import ViTAdapter


class _DummyDino(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 768, kernel_size=16, stride=16, bias=False)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _tiny_meta_arch() -> Mask2FormerPanoptic:
    adapter = ViTAdapter(backbone=_DummyDino(), embed_dim=768, num_blocks=1, pyramid_channels=128)
    pixel = MSDeformAttnPixelDecoder(in_channels=128, hidden_dim=128, mask_dim=128, num_layers=2, num_heads=4)
    pool = build_query_pool(kind="standard", num_queries=10, embed_dim=128)
    dec = MaskedAttentionDecoder(hidden_dim=128, num_queries=10, num_classes=20, num_layers=2, num_heads=4, query_pool=pool)
    matcher = HungarianMatcher(num_points=64)
    criterion = SetCriterion(num_classes=20, matcher=matcher, weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0}, num_points=64)
    return Mask2FormerPanoptic(
        backbone=adapter, pixel_decoder=pixel, transformer_decoder=dec, criterion=criterion,
        num_stuff_classes=12, num_thing_classes=8,
    )


def test_meta_arch_train_returns_loss_dict(tiny_gt_panoptic: list[dict]) -> None:
    torch.manual_seed(0)
    model = _tiny_meta_arch().train()
    # Fake CUPS-style batch: list of dicts with "image" and "instances" + "sem_seg".
    batch = []
    for gt in tiny_gt_panoptic:
        batch.append({
            "image": torch.randn(3, 64, 128),
            "_m2f_targets": {"labels": gt["labels"], "masks": gt["masks"]},
        })
    loss_dict = model(batch)
    assert isinstance(loss_dict, dict)
    assert "loss_ce" in loss_dict
    assert all(torch.isfinite(v) for v in loss_dict.values())


def test_meta_arch_eval_returns_panoptic(tiny_gt_panoptic: list[dict]) -> None:
    torch.manual_seed(1)
    model = _tiny_meta_arch().eval()
    batch = [{"image": torch.randn(3, 64, 128)} for _ in range(2)]
    with torch.no_grad():
        out = model(batch)
    assert isinstance(out, list)
    assert len(out) == 2
    for sample in out:
        assert "sem_seg" in sample
        assert "panoptic_seg" in sample
        sem = sample["sem_seg"]
        assert sem.dim() == 3  # (K, H, W)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage2_m2f/test_meta_arch.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement Mask2FormerPanoptic meta-arch**

`refs/cups/cups/model/modeling/meta_arch/mask2former_panoptic.py`:
```python
"""Mask2FormerPanoptic: end-to-end meta-arch for Stage-2 M2F.

Training path:
    image -> ViTAdapter -> pixel_decoder -> transformer_decoder -> losses

Eval path (panoptic merge):
    take top-scoring queries, softmax over class, threshold masks,
    greedy assignment onto a (H, W) panoptic map.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import META_ARCH_REGISTRY

__all__ = ["Mask2FormerPanoptic"]


@META_ARCH_REGISTRY.register()
class Mask2FormerPanoptic(nn.Module):
    """End-to-end meta-arch. Registered so detectron2 tooling can enumerate it."""

    def __init__(
        self,
        backbone: nn.Module,
        pixel_decoder: nn.Module,
        transformer_decoder: nn.Module,
        criterion: nn.Module,
        num_stuff_classes: int,
        num_thing_classes: int,
        object_mask_threshold: float = 0.4,
        overlap_threshold: float = 0.8,
        aux_loss_hooks: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.transformer_decoder = transformer_decoder
        self.criterion = criterion
        self.num_stuff_classes = num_stuff_classes
        self.num_thing_classes = num_thing_classes
        self.num_classes = num_stuff_classes + num_thing_classes
        self.object_mask_threshold = object_mask_threshold
        self.overlap_threshold = overlap_threshold
        self.aux_loss_hooks = aux_loss_hooks or {}

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _stack_images(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        imgs = [s["image"].to(self.device) for s in batch]
        # Pad to largest; crop config ensures uniform shape in practice.
        H = max(t.shape[-2] for t in imgs)
        W = max(t.shape[-1] for t in imgs)
        padded = torch.zeros(len(imgs), 3, H, W, device=self.device)
        for i, t in enumerate(imgs):
            padded[i, :, : t.shape[-2], : t.shape[-1]] = t.float() / 255.0 if t.dtype == torch.uint8 else t.float()
        return padded

    def _maybe_depth(self, batch: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
        if all("depth" in s for s in batch):
            return torch.stack([s["depth"].to(self.device) for s in batch], dim=0)
        return None

    def _collect_targets(self, batch: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        targets: List[Dict[str, torch.Tensor]] = []
        for s in batch:
            if "_m2f_targets" in s:
                targets.append({
                    "labels": s["_m2f_targets"]["labels"].to(self.device),
                    "masks": s["_m2f_targets"]["masks"].to(self.device),
                })
                continue
            # Build from CUPS-style instances + sem_seg.
            labels: List[int] = []
            masks: List[torch.Tensor] = []
            if "instances" in s:
                inst = s["instances"].to(self.device)
                labels.extend(inst.gt_classes.tolist())
                masks.extend([m.bool() for m in inst.gt_masks.tensor])
            if "sem_seg" in s:
                sem = s["sem_seg"].to(self.device)
                for c in sem.unique():
                    if int(c) in {-1, 255}:
                        continue
                    labels.append(int(c))
                    masks.append((sem == c))
            if not labels:
                H, W = s["image"].shape[-2:]
                targets.append({"labels": torch.zeros(0, dtype=torch.long, device=self.device),
                                "masks": torch.zeros(0, H, W, dtype=torch.bool, device=self.device)})
            else:
                targets.append({"labels": torch.as_tensor(labels, dtype=torch.long, device=self.device),
                                "masks": torch.stack(masks, dim=0).bool()})
        return targets

    def _panoptic_merge(self, logits: torch.Tensor, masks: torch.Tensor, H: int, W: int) -> Dict[str, torch.Tensor]:
        """Greedy panoptic assembly (one sample)."""
        scores, labels = F.softmax(logits, dim=-1).max(-1)   # Q
        mask_probs = masks.sigmoid()
        keep = (labels < self.num_classes) & (scores > self.object_mask_threshold)
        cur_masks = mask_probs[keep]
        cur_scores = scores[keep]
        cur_labels = labels[keep]
        panoptic_seg = torch.zeros((H, W), dtype=torch.int32, device=logits.device)
        sem_seg = torch.zeros((self.num_classes, H, W), dtype=torch.float32, device=logits.device)
        if cur_masks.numel() == 0:
            return {"sem_seg": sem_seg, "panoptic_seg": (panoptic_seg, [])}
        # Resize masks to image resolution.
        cur_masks = F.interpolate(cur_masks.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
        cur_mask_ids = cur_prob_masks.argmax(0)
        current_id = 0
        segments_info: List[Dict[str, int]] = []
        for k in range(cur_masks.shape[0]):
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            if mask_area > 0 and original_area > 0 and mask_area / original_area > self.overlap_threshold:
                current_id += 1
                panoptic_seg[cur_mask_ids == k] = current_id
                cls = int(cur_labels[k])
                segments_info.append({"id": current_id, "category_id": cls,
                                      "isthing": cls >= self.num_stuff_classes})
                sem_seg[cls] = torch.maximum(sem_seg[cls], cur_masks[k])
        return {"sem_seg": sem_seg, "panoptic_seg": (panoptic_seg, segments_info)}

    def forward(self, batch: List[Dict[str, Any]]) -> Any:
        images = self._stack_images(batch)
        depth = self._maybe_depth(batch)
        feats = self.backbone(images)
        mask_feat, multi_scale = self.pixel_decoder(feats)
        return_emb = bool(self.aux_loss_hooks.get("return_query_embeds", False))
        dec_out = self.transformer_decoder(mask_feat=mask_feat, multi_scale=multi_scale, depth=depth, return_query_embeds=return_emb)
        if self.training:
            targets = self._collect_targets(batch)
            loss_dict = self.criterion(dec_out, targets)
            # N3/N4 losses use query embeddings, not logits — wired here.
            for loss_key, hook in self.aux_loss_hooks.items():
                if loss_key in ("xquery", "query_consistency"):
                    loss_dict[f"loss_{loss_key}"] = hook(dec_out, targets, {"images": images, "depth": depth})
            return loss_dict
        # Eval: per-sample panoptic merge.
        outputs: List[Dict[str, torch.Tensor]] = []
        H, W = images.shape[-2:]
        for b in range(images.shape[0]):
            outputs.append(self._panoptic_merge(dec_out["pred_logits"][b], dec_out["pred_masks"][b], H, W))
        return outputs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/stage2_m2f/test_meta_arch.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add refs/cups/cups/model/modeling/meta_arch/mask2former_panoptic.py \
        tests/stage2_m2f/test_meta_arch.py
git commit -m "feat(m2f): add Mask2FormerPanoptic meta-arch"
```

### Task 0.10: Builder `model_mask2former.py`

**Files:**
- Create: `refs/cups/cups/model/model_mask2former.py`
- Modify: `refs/cups/cups/model/__init__.py`
- Test: `tests/stage2_m2f/test_builder.py`

A single entrypoint `build_mask2former_vitb(config)` that reads the yacs config and returns a ready-to-train `Mask2FormerPanoptic`. Mirrors `panoptic_cascade_mask_r_cnn_vitb()` but for M2F.

- [ ] **Step 1: Write failing test**

`tests/stage2_m2f/test_builder.py`:
```python
from __future__ import annotations

import torch
from yacs.config import CfgNode

from cups.model.model_mask2former import build_mask2former_vitb


def _minimal_config(num_stuff: int = 12, num_thing: int = 8) -> CfgNode:
    c = CfgNode()
    c.MODEL = CfgNode()
    c.MODEL.META_ARCH = "Mask2FormerPanoptic"
    c.MODEL.BACKBONE_TYPE = "dinov3_vitb"
    c.MODEL.DINOV2_FREEZE = True
    c.MODEL.TTA_SCALES = (1.0,)
    c.MODEL.MASK2FORMER = CfgNode()
    c.MODEL.MASK2FORMER.NUM_QUERIES = 10
    c.MODEL.MASK2FORMER.QUERY_POOL = "standard"
    c.MODEL.MASK2FORMER.NUM_DECODER_LAYERS = 2
    c.MODEL.MASK2FORMER.HIDDEN_DIM = 128
    c.MODEL.MASK2FORMER.NUM_HEADS = 4
    c.MODEL.MASK2FORMER.MASK_WEIGHT = 5.0
    c.MODEL.MASK2FORMER.DICE_WEIGHT = 5.0
    c.MODEL.MASK2FORMER.CLASS_WEIGHT = 2.0
    c.MODEL.MASK2FORMER.NO_OBJECT_WEIGHT = 0.1
    c.MODEL.MASK2FORMER.NUM_POINTS = 64
    c.MODEL.MASK2FORMER.OBJECT_MASK_THRESHOLD = 0.4
    c.MODEL.MASK2FORMER.OVERLAP_THRESHOLD = 0.8
    c.MODEL.MASK2FORMER.PYRAMID_CHANNELS = 128
    c.MODEL.MASK2FORMER.ADAPTER_BLOCKS = 1
    c.MODEL.MASK2FORMER.ADAPTER_EMBED_DIM = 768
    c.MODEL.MASK2FORMER.PIXEL_DECODER_LAYERS = 2
    c.MODEL.MASK2FORMER.DROPPATH = 0.0
    c.MODEL.MASK2FORMER.QUERIES_STUFF = 150
    c.MODEL.MASK2FORMER.QUERIES_THING = 50
    c.DATA = CfgNode()
    c.DATA.NUM_PSEUDO_CLASSES = num_stuff + num_thing
    # Marker fields consumed by the builder:
    c._NUM_STUFF_CLASSES = num_stuff
    c._NUM_THING_CLASSES = num_thing
    return c


def test_builder_returns_frozen_backbone(monkeypatch) -> None:
    # Monkey-patch DINOv3 loading to avoid network calls.
    from cups.model.modeling.mask2former import vit_adapter as va

    class _Dummy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 768, kernel_size=16, stride=16, bias=False)
            for p in self.parameters():
                p.requires_grad_(False)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    def _fake_build(cfg):
        return _Dummy()

    import cups.model.model_mask2former as mm
    monkeypatch.setattr(mm, "_build_dinov3_backbone", _fake_build)

    cfg = _minimal_config()
    model = build_mask2former_vitb(cfg)
    # Backbone params must be frozen.
    for p in model.backbone.backbone.parameters():
        assert p.requires_grad is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage2_m2f/test_builder.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement builder**

`refs/cups/cups/model/model_mask2former.py`:
```python
"""Builder for Mask2Former + ViT-Adapter with frozen DINOv3 ViT-B/16."""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from yacs.config import CfgNode

from cups.model.backbone_dinov3_vit import DINOv3ViTBackbone
from cups.model.modeling.mask2former.masked_attention_decoder import MaskedAttentionDecoder
from cups.model.modeling.mask2former.matcher import HungarianMatcher
from cups.model.modeling.mask2former.msdeform_pixel_decoder import MSDeformAttnPixelDecoder
from cups.model.modeling.mask2former.query_pool import build_query_pool
from cups.model.modeling.mask2former.set_criterion import SetCriterion
from cups.model.modeling.mask2former.vit_adapter import ViTAdapter
from cups.model.modeling.meta_arch.mask2former_panoptic import Mask2FormerPanoptic

log = logging.getLogger(__name__)

__all__ = ["build_mask2former_vitb"]


def _build_dinov3_backbone(cfg: CfgNode) -> nn.Module:
    """Wrapped for monkey-patching in tests."""
    backbone = DINOv3ViTBackbone(freeze=cfg.MODEL.DINOV2_FREEZE)
    return backbone


def build_mask2former_vitb(cfg: CfgNode) -> Mask2FormerPanoptic:
    """Build M2F + ViT-Adapter from a yacs config."""
    m = cfg.MODEL.MASK2FORMER
    num_stuff = int(cfg._NUM_STUFF_CLASSES)
    num_thing = int(cfg._NUM_THING_CLASSES)
    num_classes = num_stuff + num_thing

    # Frozen backbone.
    dino = _build_dinov3_backbone(cfg)

    # ViT-Adapter.
    adapter = ViTAdapter(
        backbone=dino,
        embed_dim=m.ADAPTER_EMBED_DIM,
        num_blocks=m.ADAPTER_BLOCKS,
        pyramid_channels=m.PYRAMID_CHANNELS,
    )

    # Pixel decoder.
    pixel = MSDeformAttnPixelDecoder(
        in_channels=m.PYRAMID_CHANNELS,
        hidden_dim=m.HIDDEN_DIM,
        mask_dim=m.HIDDEN_DIM,
        num_layers=m.PIXEL_DECODER_LAYERS,
        num_heads=m.NUM_HEADS,
    )

    # QueryPool.
    if m.QUERY_POOL == "decoupled":
        pool = build_query_pool(
            kind="decoupled",
            num_queries_stuff=m.QUERIES_STUFF,
            num_queries_thing=m.QUERIES_THING,
            embed_dim=m.HIDDEN_DIM,
        )
        num_queries = m.QUERIES_STUFF + m.QUERIES_THING
    elif m.QUERY_POOL == "depth_bias":
        pool = build_query_pool(kind="depth_bias", num_queries=m.NUM_QUERIES, embed_dim=m.HIDDEN_DIM)
        num_queries = m.NUM_QUERIES
    else:
        pool = build_query_pool(kind="standard", num_queries=m.NUM_QUERIES, embed_dim=m.HIDDEN_DIM)
        num_queries = m.NUM_QUERIES

    # Transformer decoder.
    dec = MaskedAttentionDecoder(
        hidden_dim=m.HIDDEN_DIM,
        num_queries=num_queries,
        num_classes=num_classes,
        num_layers=m.NUM_DECODER_LAYERS,
        num_heads=m.NUM_HEADS,
        query_pool=pool,
        droppath=m.DROPPATH,
    )

    matcher = HungarianMatcher(
        cost_class=m.CLASS_WEIGHT,
        cost_mask=m.MASK_WEIGHT,
        cost_dice=m.DICE_WEIGHT,
        num_points=m.NUM_POINTS,
    )
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict={
            "loss_ce": m.CLASS_WEIGHT,
            "loss_mask": m.MASK_WEIGHT,
            "loss_dice": m.DICE_WEIGHT,
        },
        eos_coef=m.NO_OBJECT_WEIGHT,
        losses=("labels", "masks"),
        num_points=m.NUM_POINTS,
    )

    model = Mask2FormerPanoptic(
        backbone=adapter,
        pixel_decoder=pixel,
        transformer_decoder=dec,
        criterion=criterion,
        num_stuff_classes=num_stuff,
        num_thing_classes=num_thing,
        object_mask_threshold=m.OBJECT_MASK_THRESHOLD,
        overlap_threshold=m.OVERLAP_THRESHOLD,
    )
    log.info(
        "Built Mask2FormerPanoptic: queries=%d stuff=%d thing=%d adapter_blocks=%d dec_layers=%d",
        num_queries, num_stuff, num_thing, m.ADAPTER_BLOCKS, m.NUM_DECODER_LAYERS,
    )
    return model
```

- [ ] **Step 4: Modify package init**

Edit `refs/cups/cups/model/__init__.py` — append one line so `build_mask2former_vitb` is importable as `cups.model.build_mask2former_vitb`.

```python
from .model import (
    filter_predictions,
    panoptic_cascade_mask_r_cnn,
    panoptic_cascade_mask_r_cnn_from_checkpoint,
    prediction_to_class_agnostic_detection,
    prediction_to_label_format,
    prediction_to_standard_format,
)
from .model_vitb import panoptic_cascade_mask_r_cnn_vitb
from .model_mask2former import build_mask2former_vitb
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/stage2_m2f/test_builder.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add refs/cups/cups/model/model_mask2former.py \
        refs/cups/cups/model/__init__.py \
        tests/stage2_m2f/test_builder.py
git commit -m "feat(m2f): add build_mask2former_vitb builder"
```

### Task 0.11: Extend `cups/config.py` with Mask2Former keys

**Files:**
- Modify: `refs/cups/cups/config.py`
- Test: `tests/stage2_m2f/test_base_inheritance.py` (partial — shared with Task 0.16)

Add a new `_C.MODEL.MASK2FORMER` CfgNode with all M0 defaults and a `_C.MODEL.META_ARCH` string field. Also add `_C.AUGMENTATION.LSJ` and `_C.AUGMENTATION.COLOR_JITTER` subnodes used by G3/G4, `_C.MODEL.EMA` used by G1, `_C.VALIDATION.USE_DENSE_CRF` used by G6, and `_C.MODEL.MASK2FORMER.N3_*` / `_C.MODEL.MASK2FORMER.N4_*` weights used by N3/N4.

- [ ] **Step 1: Append config fields**

Insert *after* the existing `_C.MODEL.SEM_SEG_HEAD` block in `refs/cups/cups/config.py` (line ~128 area):

```python
# -------- Stage-2 M2F meta-arch ---------------------------------------------
_C.MODEL.META_ARCH = "Cascade"  # "Cascade" (default) or "Mask2FormerPanoptic"

_C.MODEL.MASK2FORMER = CfgNode()
_C.MODEL.MASK2FORMER.NUM_QUERIES = 100
_C.MODEL.MASK2FORMER.QUERIES_STUFF = 150           # used when QUERY_POOL=decoupled
_C.MODEL.MASK2FORMER.QUERIES_THING = 50
_C.MODEL.MASK2FORMER.QUERY_POOL = "standard"       # "standard" / "decoupled" / "depth_bias"
_C.MODEL.MASK2FORMER.NUM_DECODER_LAYERS = 9
_C.MODEL.MASK2FORMER.PIXEL_DECODER_LAYERS = 6
_C.MODEL.MASK2FORMER.HIDDEN_DIM = 256
_C.MODEL.MASK2FORMER.NUM_HEADS = 8
_C.MODEL.MASK2FORMER.MASK_WEIGHT = 5.0
_C.MODEL.MASK2FORMER.DICE_WEIGHT = 5.0
_C.MODEL.MASK2FORMER.CLASS_WEIGHT = 2.0
_C.MODEL.MASK2FORMER.NO_OBJECT_WEIGHT = 0.1
_C.MODEL.MASK2FORMER.NUM_POINTS = 12544
_C.MODEL.MASK2FORMER.OBJECT_MASK_THRESHOLD = 0.4
_C.MODEL.MASK2FORMER.OVERLAP_THRESHOLD = 0.8
_C.MODEL.MASK2FORMER.PYRAMID_CHANNELS = 256
_C.MODEL.MASK2FORMER.ADAPTER_BLOCKS = 4
_C.MODEL.MASK2FORMER.ADAPTER_EMBED_DIM = 768
_C.MODEL.MASK2FORMER.DROPPATH = 0.0                # G5 lever

# N3 XQuery (cross-image query correspondence) loss weight (0.0 = off).
_C.MODEL.MASK2FORMER.XQUERY_WEIGHT = 0.0
_C.MODEL.MASK2FORMER.XQUERY_TEMPERATURE = 0.1

# N4 Query-consistency (teacher-student) loss weight (0.0 = off).
_C.MODEL.MASK2FORMER.QUERY_CONSISTENCY_WEIGHT = 0.0
_C.MODEL.MASK2FORMER.QUERY_CONSISTENCY_TEMPERATURE = 0.1

# -------- EMA teacher (G1) --------------------------------------------------
_C.MODEL.EMA = CfgNode()
_C.MODEL.EMA.ENABLED = False
_C.MODEL.EMA.DECAY = 0.9998

# -------- SWA (G2) ----------------------------------------------------------
_C.MODEL.SWA = CfgNode()
_C.MODEL.SWA.ENABLED = False
_C.MODEL.SWA.NUM_CKPTS = 5
_C.MODEL.SWA.START_FRACTION = 0.75                 # average ckpts from last 25 pct

# -------- LSJ (G3) ----------------------------------------------------------
_C.AUGMENTATION.LSJ = CfgNode()
_C.AUGMENTATION.LSJ.ENABLED = False
_C.AUGMENTATION.LSJ.MIN_SCALE = 0.1
_C.AUGMENTATION.LSJ.MAX_SCALE = 2.0

# -------- ColorJitter (G4) --------------------------------------------------
_C.AUGMENTATION.COLOR_JITTER = CfgNode()
_C.AUGMENTATION.COLOR_JITTER.ENABLED = False
_C.AUGMENTATION.COLOR_JITTER.BRIGHTNESS = 0.4
_C.AUGMENTATION.COLOR_JITTER.CONTRAST = 0.4
_C.AUGMENTATION.COLOR_JITTER.SATURATION = 0.4
_C.AUGMENTATION.COLOR_JITTER.HUE = 0.1

# -------- G6 Dense-CRF post-processing at val ------------------------------
_C.VALIDATION.USE_DENSE_CRF = False
_C.VALIDATION.DENSE_CRF_ITER = 5
_C.VALIDATION.DENSE_CRF_BI_W = 4.0
_C.VALIDATION.DENSE_CRF_POS_W = 3.0

# -------- N5 self-training threshold ---------------------------------------
_C.MODEL.MASK2FORMER.SELF_TRAIN_THRESHOLD = 0.95
```

- [ ] **Step 2: Write test**

`tests/stage2_m2f/test_base_inheritance.py` (partial — `test_new_config_keys`; the `_BASE_` portion is Task 0.16):
```python
from __future__ import annotations

from cups.config import get_default_config


def test_mask2former_config_keys_exist() -> None:
    cfg = get_default_config()
    # Meta-arch switch.
    assert cfg.MODEL.META_ARCH == "Cascade"
    # M2F keys with defaults.
    assert cfg.MODEL.MASK2FORMER.NUM_QUERIES == 100
    assert cfg.MODEL.MASK2FORMER.QUERY_POOL == "standard"
    assert cfg.MODEL.MASK2FORMER.NUM_DECODER_LAYERS == 9
    assert cfg.MODEL.MASK2FORMER.XQUERY_WEIGHT == 0.0
    assert cfg.MODEL.MASK2FORMER.QUERY_CONSISTENCY_WEIGHT == 0.0
    # G-lever keys.
    assert cfg.MODEL.EMA.ENABLED is False
    assert cfg.MODEL.SWA.ENABLED is False
    assert cfg.AUGMENTATION.LSJ.ENABLED is False
    assert cfg.AUGMENTATION.COLOR_JITTER.ENABLED is False
    assert cfg.VALIDATION.USE_DENSE_CRF is False
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest tests/stage2_m2f/test_base_inheritance.py::test_mask2former_config_keys_exist -v`
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add refs/cups/cups/config.py tests/stage2_m2f/test_base_inheritance.py
git commit -m "feat(m2f): extend config with Mask2Former + G/N keys"
```

### Task 0.12: EMA and SWA utilities

**Files:**
- Create: `refs/cups/cups/model/modeling/mask2former/ema.py`
- Create: `refs/cups/cups/model/modeling/mask2former/swa.py`
- Test: `tests/stage2_m2f/test_ema.py`
- Test: `tests/stage2_m2f/test_swa.py`

- [ ] **Step 1: Write failing tests**

`tests/stage2_m2f/test_ema.py`:
```python
from __future__ import annotations

import torch

from cups.model.modeling.mask2former.ema import EMAModel


def test_ema_update_rule() -> None:
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    ema = EMAModel(model, decay=0.9)
    # Snapshot initial weights.
    w0 = model.weight.detach().clone()
    # Modify model weights.
    with torch.no_grad():
        model.weight.copy_(w0 + 1.0)
    ema.update(model)
    # EMA should be 0.9 * w0 + 0.1 * (w0 + 1.0) = w0 + 0.1
    expected = w0 + 0.1
    assert torch.allclose(ema.shadow["weight"], expected, atol=1e-6)


def test_ema_load_into_model() -> None:
    model = torch.nn.Linear(4, 2)
    ema = EMAModel(model, decay=0.9)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.0)
    ema.copy_to(model)
    # After copy_to, model weights == ema.shadow (initial random weights).
    assert not torch.all(model.weight == 0.0)
```

`tests/stage2_m2f/test_swa.py`:
```python
from __future__ import annotations

import torch

from cups.model.modeling.mask2former.swa import average_state_dicts


def test_swa_averages_three_state_dicts() -> None:
    sd1 = {"w": torch.tensor([1.0, 2.0])}
    sd2 = {"w": torch.tensor([3.0, 4.0])}
    sd3 = {"w": torch.tensor([5.0, 6.0])}
    avg = average_state_dicts([sd1, sd2, sd3])
    assert torch.allclose(avg["w"], torch.tensor([3.0, 4.0]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/stage2_m2f/test_ema.py tests/stage2_m2f/test_swa.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement EMA**

`refs/cups/cups/model/modeling/mask2former/ema.py`:
```python
"""Exponential-moving-average wrapper for teacher models (G1, N4, Stage-3)."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

__all__ = ["EMAModel"]


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        """Overwrite model parameters with shadow copy (used at eval / Stage-3)."""
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])
```

- [ ] **Step 4: Implement SWA utility**

`refs/cups/cups/model/modeling/mask2former/swa.py`:
```python
"""Stochastic Weight Averaging utility (G2): mean over a list of state dicts."""
from __future__ import annotations

from typing import Dict, List

import torch

__all__ = ["average_state_dicts"]


def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    assert state_dicts, "average_state_dicts requires at least one dict"
    out: Dict[str, torch.Tensor] = {}
    keys = state_dicts[0].keys()
    for k in keys:
        stacked = torch.stack([sd[k].float() for sd in state_dicts], dim=0)
        out[k] = stacked.mean(0).to(state_dicts[0][k].dtype)
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/stage2_m2f/test_ema.py tests/stage2_m2f/test_swa.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add refs/cups/cups/model/modeling/mask2former/ema.py \
        refs/cups/cups/model/modeling/mask2former/swa.py \
        tests/stage2_m2f/test_ema.py \
        tests/stage2_m2f/test_swa.py
git commit -m "feat(m2f): add EMA and SWA utilities"
```

### Task 0.13: Augmentations (LSJ + ColorJitter) and dense-CRF

**Files:**
- Create: `refs/cups/cups/model/modeling/mask2former/augment.py`
- Create: `refs/cups/cups/model/modeling/mask2former/dense_crf.py`
- Test: `tests/stage2_m2f/test_augment.py`
- Test: `tests/stage2_m2f/test_crf.py`

- [ ] **Step 1: Write failing tests**

`tests/stage2_m2f/test_augment.py`:
```python
from __future__ import annotations

import torch

from cups.model.modeling.mask2former.augment import ColorJitterModule, LargeScaleJitter


def test_lsj_scales_output_within_range() -> None:
    torch.manual_seed(0)
    lsj = LargeScaleJitter(min_scale=0.1, max_scale=2.0, target_size=(64, 128))
    img = torch.randn(3, 64, 128)
    lbl = torch.randint(0, 10, (1, 64, 128))
    out_img, out_lbl = lsj(img, lbl)
    assert out_img.shape == (3, 64, 128)
    assert out_lbl.shape == (1, 64, 128)


def test_color_jitter_is_identity_when_all_zero() -> None:
    torch.manual_seed(0)
    cj = ColorJitterModule(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0)
    img = torch.rand(3, 64, 128)
    out = cj(img)
    assert torch.allclose(out, img)
```

`tests/stage2_m2f/test_crf.py`:
```python
from __future__ import annotations

import numpy as np
import torch

from cups.model.modeling.mask2former.dense_crf import dense_crf_refine


def test_dense_crf_refines_known_mask() -> None:
    # Input: 10x10 noisy probability field with a clean GT blob in the middle.
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.2, 0.4, size=(3, 10, 10)).astype(np.float32)
    probs[0, 3:7, 3:7] += 0.5        # class 0 dominant in center
    probs = probs / probs.sum(axis=0, keepdims=True)
    img = rng.integers(0, 255, size=(10, 10, 3), dtype=np.uint8)
    probs_t = torch.from_numpy(probs)
    img_t = torch.from_numpy(img)
    out = dense_crf_refine(img_t, probs_t, num_iter=3)
    assert out.shape == probs_t.shape
    # Refined argmax should have class-0 in the center block.
    lbl = out.argmax(0)
    assert lbl[5, 5].item() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/stage2_m2f/test_augment.py tests/stage2_m2f/test_crf.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement augmentations**

`refs/cups/cups/model/modeling/mask2former/augment.py`:
```python
"""G3 LSJ + G4 ColorJitter modules."""
from __future__ import annotations

import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ColorJitter

__all__ = ["LargeScaleJitter", "ColorJitterModule"]


class LargeScaleJitter(nn.Module):
    def __init__(self, min_scale: float = 0.1, max_scale: float = 2.0, target_size: Tuple[int, int] = (640, 1280)) -> None:
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.target_size = target_size

    def forward(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sample scale.
        scale = random.uniform(self.min_scale, self.max_scale)
        H, W = image.shape[-2:]
        new_H = max(1, int(H * scale))
        new_W = max(1, int(W * scale))
        img = F.interpolate(image.unsqueeze(0).float(), size=(new_H, new_W), mode="bilinear", align_corners=False).squeeze(0)
        lbl = F.interpolate(label.unsqueeze(0).float(), size=(new_H, new_W), mode="nearest").squeeze(0).long()
        # Pad or crop to target size.
        tH, tW = self.target_size
        if new_H < tH or new_W < tW:
            pad_h = max(0, tH - new_H)
            pad_w = max(0, tW - new_W)
            img = F.pad(img, (0, pad_w, 0, pad_h), value=0.0)
            lbl = F.pad(lbl, (0, pad_w, 0, pad_h), value=255)  # ignore in pad
            new_H, new_W = img.shape[-2:]
        if new_H > tH or new_W > tW:
            y = random.randint(0, new_H - tH)
            x = random.randint(0, new_W - tW)
            img = img[:, y : y + tH, x : x + tW]
            lbl = lbl[:, y : y + tH, x : x + tW]
        return img, lbl


class ColorJitterModule(nn.Module):
    def __init__(self, brightness: float = 0.4, contrast: float = 0.4, saturation: float = 0.4, hue: float = 0.1) -> None:
        super().__init__()
        self.jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self._identity = all(v == 0.0 for v in (brightness, contrast, saturation, hue))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self._identity:
            return image
        return self.jitter(image)
```

- [ ] **Step 4: Implement dense-CRF wrapper**

`refs/cups/cups/model/modeling/mask2former/dense_crf.py`:
```python
"""G6 dense-CRF post-processing wrapper (pydensecrf)."""
from __future__ import annotations

import numpy as np
import torch

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    _HAS_DCRF = True
except ImportError:       # graceful fallback
    _HAS_DCRF = False

__all__ = ["dense_crf_refine"]


def dense_crf_refine(
    image: torch.Tensor, probs: torch.Tensor, num_iter: int = 5, bi_w: float = 4.0, pos_w: float = 3.0,
) -> torch.Tensor:
    """Refine softmax probs using dense-CRF.

    Args:
        image: uint8 (H, W, 3) or float (3, H, W) in [0, 1].
        probs: (K, H, W) softmax probabilities.
        num_iter: number of CRF iterations.
    """
    if not _HAS_DCRF:
        return probs
    img_np = image.cpu().numpy()
    if img_np.ndim == 3 and img_np.shape[0] == 3:
        img_np = (img_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    probs_np = probs.detach().cpu().numpy().astype(np.float32)
    # Sanitize tiny values for numerical stability.
    probs_np = np.maximum(probs_np, 1e-8)
    K, H, W = probs_np.shape
    unary = unary_from_softmax(probs_np.reshape(K, -1))
    d = dcrf.DenseCRF2D(W, H, K)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=pos_w)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=np.ascontiguousarray(img_np), compat=bi_w)
    Q = d.inference(num_iter)
    out = np.asarray(Q).reshape(K, H, W)
    return torch.from_numpy(out).to(probs.device)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/stage2_m2f/test_augment.py tests/stage2_m2f/test_crf.py -v`
Expected: 3 passed (CRF test skipped gracefully if pydensecrf missing — adjust fallback in test to `pytest.importorskip` if needed).

Adjust CRF test header if `pydensecrf` not guaranteed:
```python
import pytest
pytest.importorskip("pydensecrf", reason="dense-CRF requires pydensecrf")
```

- [ ] **Step 6: Commit**

```bash
git add refs/cups/cups/model/modeling/mask2former/augment.py \
        refs/cups/cups/model/modeling/mask2former/dense_crf.py \
        tests/stage2_m2f/test_augment.py \
        tests/stage2_m2f/test_crf.py
git commit -m "feat(m2f): add LSJ/ColorJitter/dense-CRF utilities"
```

### Task 0.14: XQuery (N3) and QueryConsistency (N4) losses

**Files:**
- Create: `refs/cups/cups/losses/xquery.py`
- Create: `refs/cups/cups/losses/query_consistency.py`
- Test: `tests/stage2_m2f/test_xquery.py`
- Test: `tests/stage2_m2f/test_query_consistency.py`

These two losses take **query embeddings** as input (shape `(B, Q, C)`), not logits — they are wired explicitly in `Mask2FormerPanoptic.forward` via `aux_loss_hooks`, NOT through the `build_aux_losses` uniform registry.

- [ ] **Step 1: Write failing tests**

`tests/stage2_m2f/test_xquery.py`:
```python
from __future__ import annotations

import torch

from cups.losses.xquery import xquery_loss


def test_xquery_is_zero_when_batch_size_one() -> None:
    q = torch.randn(1, 10, 32)
    dec_out = {"query_embeds": q}
    loss = xquery_loss(dec_out, targets=[], ctx={})
    assert loss.item() == 0.0


def test_xquery_positive_for_batch_size_two() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 10, 32, requires_grad=True)
    dec_out = {"query_embeds": q}
    loss = xquery_loss(dec_out, targets=[], ctx={"temperature": 0.1})
    loss.backward()
    assert loss.item() > 0.0
    assert q.grad is not None
```

`tests/stage2_m2f/test_query_consistency.py`:
```python
from __future__ import annotations

import torch

from cups.losses.query_consistency import query_consistency_loss


def test_query_consistency_zero_when_student_equals_teacher() -> None:
    q = torch.randn(2, 10, 32)
    dec_out = {"query_embeds": q}
    ctx = {"teacher_query_embeds": q.detach().clone(), "temperature": 0.1}
    loss = query_consistency_loss(dec_out, targets=[], ctx=ctx)
    assert loss.item() < 1e-4


def test_query_consistency_positive_when_different() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 10, 32, requires_grad=True)
    tq = torch.randn(2, 10, 32)
    dec_out = {"query_embeds": q}
    ctx = {"teacher_query_embeds": tq, "temperature": 0.1}
    loss = query_consistency_loss(dec_out, targets=[], ctx=ctx)
    loss.backward()
    assert loss.item() > 0.0
    assert q.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/stage2_m2f/test_xquery.py tests/stage2_m2f/test_query_consistency.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement XQuery**

`refs/cups/cups/losses/xquery.py`:
```python
"""N3 XQuery: cross-image query correspondence loss.

Given a batch with B>=2 images and their per-query embeddings
``q: (B, Q, C)``, pull queries that match across images together and
push non-matching queries apart using an InfoNCE-style symmetric loss.

The "match" between images is approximated by nearest-neighbor in
embedding space (a soft anchor). Returns 0 for B=1.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

__all__ = ["xquery_loss"]


def xquery_loss(dec_out: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], ctx: Dict) -> torch.Tensor:
    q = dec_out.get("query_embeds", None)
    if q is None:
        raise KeyError("dec_out must include 'query_embeds' for xquery_loss")
    B, Q, C = q.shape
    if B < 2:
        return q.sum() * 0.0
    t = float(ctx.get("temperature", 0.1))
    q_norm = F.normalize(q, dim=-1)
    # Pair (b, b+1) for simplicity; could be all-pairs but O(B^2) memory.
    loss = q.new_zeros([])
    count = 0
    for a, b in zip(range(B), list(range(1, B)) + [0]):
        if a == b:
            continue
        sim = q_norm[a] @ q_norm[b].T          # Q x Q
        # Symmetric InfoNCE with diagonal as positives.
        targets_ = torch.arange(Q, device=q.device)
        loss = loss + 0.5 * (F.cross_entropy(sim / t, targets_) + F.cross_entropy(sim.T / t, targets_))
        count += 1
    return loss / max(count, 1)
```

- [ ] **Step 4: Implement QueryConsistency**

`refs/cups/cups/losses/query_consistency.py`:
```python
"""N4 Query-consistency: EMA-teacher query-embedding alignment.

Given student query embeddings ``q_s: (B, Q, C)`` and teacher query
embeddings ``q_t`` (EMA of student, stored in ctx['teacher_query_embeds']),
minimize cosine distance on a per-image per-query basis.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

__all__ = ["query_consistency_loss"]


def query_consistency_loss(dec_out: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], ctx: Dict) -> torch.Tensor:
    q_s = dec_out.get("query_embeds", None)
    if q_s is None:
        raise KeyError("dec_out must include 'query_embeds' for query_consistency_loss")
    q_t = ctx.get("teacher_query_embeds", None)
    if q_t is None:
        return q_s.sum() * 0.0  # silently zero if no teacher yet
    B, Q, C = q_s.shape
    q_s_n = F.normalize(q_s.reshape(-1, C), dim=-1)
    q_t_n = F.normalize(q_t.reshape(-1, C), dim=-1)
    cos = (q_s_n * q_t_n).sum(-1)     # (B*Q,)
    return (1.0 - cos).mean()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/stage2_m2f/test_xquery.py tests/stage2_m2f/test_query_consistency.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add refs/cups/cups/losses/xquery.py \
        refs/cups/cups/losses/query_consistency.py \
        tests/stage2_m2f/test_xquery.py \
        tests/stage2_m2f/test_query_consistency.py
git commit -m "feat(m2f): add XQuery (N3) + QueryConsistency (N4) losses"
```

### Task 0.15: Self-training pseudo-label sampler (N5)

**Files:**
- Create: `refs/cups/cups/losses/self_train.py`
- Test: `tests/stage2_m2f/test_self_train.py`

N5 is a **data-side** hook: during Stage-3 (self-training), filter low-confidence pseudo-labels from the teacher predictions before they become training targets.

- [ ] **Step 1: Write failing test**

`tests/stage2_m2f/test_self_train.py`:
```python
from __future__ import annotations

import torch

from cups.losses.self_train import filter_by_confidence


def test_filter_keeps_high_confidence_only() -> None:
    labels = torch.tensor([2, 5, 7, 3], dtype=torch.long)
    scores = torch.tensor([0.99, 0.50, 0.97, 0.30])
    masks = torch.zeros(4, 8, 8, dtype=torch.bool)
    masks[0, 0:3, 0:3] = True
    masks[1, 3:6, 3:6] = True
    masks[2, 5:8, 5:8] = True
    masks[3, 0:2, 6:8] = True
    kept = filter_by_confidence({"labels": labels, "masks": masks, "scores": scores}, threshold=0.95)
    assert kept["labels"].tolist() == [2, 7]
    assert kept["masks"].shape == (2, 8, 8)
    assert kept["scores"].tolist() == [0.99, 0.97]


def test_filter_empty_when_all_below_threshold() -> None:
    labels = torch.tensor([0], dtype=torch.long)
    scores = torch.tensor([0.1])
    masks = torch.ones(1, 4, 4, dtype=torch.bool)
    kept = filter_by_confidence({"labels": labels, "masks": masks, "scores": scores}, threshold=0.5)
    assert kept["labels"].numel() == 0
    assert kept["masks"].shape == (0, 4, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage2_m2f/test_self_train.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement filter**

`refs/cups/cups/losses/self_train.py`:
```python
"""N5 Self-training: confidence-thresholded pseudo-label filter."""
from __future__ import annotations

from typing import Dict

import torch

__all__ = ["filter_by_confidence"]


def filter_by_confidence(pseudo: Dict[str, torch.Tensor], threshold: float = 0.95) -> Dict[str, torch.Tensor]:
    scores = pseudo["scores"]
    keep = scores >= threshold
    H = pseudo["masks"].shape[-2]
    W = pseudo["masks"].shape[-1]
    return {
        "labels": pseudo["labels"][keep],
        "masks": pseudo["masks"][keep] if keep.any() else torch.zeros(0, H, W, dtype=torch.bool, device=pseudo["masks"].device),
        "scores": pseudo["scores"][keep],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/stage2_m2f/test_self_train.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add refs/cups/cups/losses/self_train.py \
        tests/stage2_m2f/test_self_train.py
git commit -m "feat(m2f): add N5 confidence-threshold pseudo-label filter"
```

### Task 0.16: `_BASE_` config inheritance for yacs

**Files:**
- Modify: `refs/cups/cups/config.py`
- Test: `tests/stage2_m2f/test_base_inheritance.py`

yacs does NOT support `_BASE_` natively. All subsequent G/N/M configs `_BASE_` onto M0 (or M1/M2), so we add a small preprocessing helper that walks the chain and merges parents before children.

- [ ] **Step 1: Write failing test**

`tests/stage2_m2f/test_base_inheritance.py` (extend):
```python
import tempfile
import textwrap
from pathlib import Path

from cups.config import get_default_config


def test_base_inheritance_simple_chain(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    parent.write_text(textwrap.dedent(
        """
        MODEL:
          META_ARCH: "Mask2FormerPanoptic"
          MASK2FORMER:
            NUM_QUERIES: 100
        """
    ))
    child = tmp_path / "child.yaml"
    child.write_text(textwrap.dedent(
        f"""
        _BASE_: "{parent.name}"
        MODEL:
          MASK2FORMER:
            NUM_QUERIES: 200
        """
    ))
    cfg = get_default_config(str(child))
    assert cfg.MODEL.META_ARCH == "Mask2FormerPanoptic"  # from parent
    assert cfg.MODEL.MASK2FORMER.NUM_QUERIES == 200       # child wins


def test_base_inheritance_two_levels(tmp_path: Path) -> None:
    grand = tmp_path / "grand.yaml"
    grand.write_text("MODEL:\n  META_ARCH: \"Mask2FormerPanoptic\"\n")
    mid = tmp_path / "mid.yaml"
    mid.write_text(f"_BASE_: \"{grand.name}\"\nMODEL:\n  MASK2FORMER:\n    NUM_QUERIES: 150\n")
    leaf = tmp_path / "leaf.yaml"
    leaf.write_text(f"_BASE_: \"{mid.name}\"\nMODEL:\n  MASK2FORMER:\n    NUM_QUERIES: 250\n")
    cfg = get_default_config(str(leaf))
    assert cfg.MODEL.META_ARCH == "Mask2FormerPanoptic"
    assert cfg.MODEL.MASK2FORMER.NUM_QUERIES == 250
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage2_m2f/test_base_inheritance.py -v`
Expected: 2 FAIL (config `_BASE_` key is unknown).

- [ ] **Step 3: Patch `get_default_config` to walk `_BASE_` chain**

Replace the body of `get_default_config` in `refs/cups/cups/config.py` with:

```python
def _resolve_base_chain(config_file: str) -> List[str]:
    """Return ordered list of config paths, ancestors first."""
    chain: List[str] = []
    seen: Set[str] = set()
    cur = os.path.abspath(config_file)
    while cur is not None:
        if cur in seen:
            raise ValueError(f"cyclic _BASE_ chain starting at {config_file}")
        seen.add(cur)
        chain.append(cur)
        with open(cur, "r") as f:
            import yaml
            raw = yaml.safe_load(f) or {}
        base_rel = raw.get("_BASE_")
        if base_rel is None:
            break
        cur = os.path.abspath(os.path.join(os.path.dirname(cur), base_rel))
    return list(reversed(chain))   # parent first


def get_default_config(
    experiment_config_file: Optional[str] = None,
    command_line_arguments: Optional[List[Any]] = None,
) -> CfgNode:
    """Loads config object with _BASE_ inheritance support."""
    config = _C.clone()
    if experiment_config_file is not None:
        assert isinstance(experiment_config_file, str)
        assert os.path.exists(experiment_config_file)
        chain = _resolve_base_chain(experiment_config_file)
        # Merge each file in parent-first order, dropping _BASE_ key before merge.
        for path in chain:
            import yaml
            with open(path, "r") as f:
                raw = yaml.safe_load(f) or {}
            raw.pop("_BASE_", None)
            import tempfile
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
                yaml.safe_dump(raw, tmp)
                tmp_path = tmp.name
            try:
                config.merge_from_file(tmp_path)
            finally:
                os.unlink(tmp_path)
        log.info(f"Experiment config chain {[os.path.basename(p) for p in chain]} loaded.")
    if command_line_arguments is not None:
        assert isinstance(command_line_arguments, list)
        config.merge_from_list(command_line_arguments)
        log.info("Command line arguments loaded.")
    config.freeze()
    return config
```

Also add `Set` to the import line:
```python
from typing import Any, List, Optional, Set, Tuple
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/stage2_m2f/test_base_inheritance.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add refs/cups/cups/config.py tests/stage2_m2f/test_base_inheritance.py
git commit -m "feat(m2f): add _BASE_ config inheritance for yacs"
```

### Task 0.17: Wire `Mask2FormerPanoptic` into `build_model_pseudo`

**Files:**
- Modify: `refs/cups/cups/pl_model_pseudo.py`
- Test: `tests/stage2_m2f/test_pl_model_routing.py`

- [ ] **Step 1: Write failing test**

`tests/stage2_m2f/test_pl_model_routing.py`:
```python
from __future__ import annotations

import pytest
from yacs.config import CfgNode

from cups.config import get_default_config


def test_build_model_pseudo_routes_mask2former(monkeypatch) -> None:
    from cups import pl_model_pseudo as plm

    called = {"built": False}

    def fake_build(config: CfgNode, *args, **kwargs):
        called["built"] = True
        import torch.nn as nn
        return nn.Identity()

    monkeypatch.setattr(plm, "_build_mask2former_model", fake_build)
    cfg = get_default_config()
    cfg.defrost()
    cfg.MODEL.META_ARCH = "Mask2FormerPanoptic"
    cfg._NUM_STUFF_CLASSES = 12
    cfg._NUM_THING_CLASSES = 8
    cfg.freeze()
    model = plm.build_model_pseudo(cfg)
    assert called["built"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage2_m2f/test_pl_model_routing.py -v`
Expected: FAIL — `_build_mask2former_model` does not exist or `build_model_pseudo` function does not exist / returns wrong branch.

- [ ] **Step 3: Implement routing**

Grep `refs/cups/cups/pl_model_pseudo.py` for the existing `build_model_pseudo` (it already routes `dinov2_vitb`, `dinov3_vitb`, `dinov3_vitl`). Add a branch at the TOP of the function:

```python
def _build_mask2former_model(config):
    from cups.model.model_mask2former import build_mask2former_vitb
    return build_mask2former_vitb(config)


def build_model_pseudo(config):
    # NEW branch: Mask2FormerPanoptic meta-arch.
    if getattr(config.MODEL, "META_ARCH", "Cascade") == "Mask2FormerPanoptic":
        return _build_mask2former_model(config)
    # ... existing body (unchanged)
```

If `build_model_pseudo` does not exist in `refs/cups/cups/pl_model_pseudo.py` (check first), implement it as a standalone function *outside* `UnsupervisedModel`, using the existing `panoptic_cascade_mask_r_cnn` / `panoptic_cascade_mask_r_cnn_vitb` imports. Reference: `cups/__init__.py` already exports `build_model_pseudo` — confirm signature via `grep -n "def build_model_pseudo" refs/cups/cups/`.

If grep shows the function lives in `cups/model/model.py` or `cups/model/model_vitb.py` instead, route the new branch there. Use `Grep` tool with pattern `def build_model_pseudo` over `refs/cups/cups/`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/stage2_m2f/test_pl_model_routing.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add refs/cups/cups/pl_model_pseudo.py tests/stage2_m2f/test_pl_model_routing.py
git commit -m "feat(m2f): route Mask2FormerPanoptic in build_model_pseudo"
```

### Task 0.18: M0 baseline config and launcher

**Files:**
- Create: `configs/stage2_m2f/M0_baseline_dinov3_vitb_k80.yaml`
- Create: `scripts/train_stage2_m2f.sh`
- Create: `scripts/eval_stage2_m2f.sh`

Note: the M0 config `_BASE_`s the existing P0 baseline for pseudo-label paths and dataset config, then overrides META_ARCH and M2F keys.

- [ ] **Step 1: Author M0 config**

`configs/stage2_m2f/M0_baseline_dinov3_vitb_k80.yaml`:
```yaml
# M0 baseline: Mask2Former + ViT-Adapter, frozen DINOv3 ViT-B/16,
# k=80 semantic + DepthPro tau=0.20 instance pseudo-labels.
# Zero augmentations, 20k steps, class-weighted CE.

_BASE_: "../../refs/cups/configs/train_cityscapes_dinov3_vitb_k80_depthpro_tau020.yaml"

MODEL:
  META_ARCH: "Mask2FormerPanoptic"
  BACKBONE_TYPE: "dinov3_vitb"
  DINOV2_FREEZE: True
  TTA_SCALES:
    - 0.75
    - 1.0
    - 1.25
  MASK2FORMER:
    NUM_QUERIES: 100
    QUERY_POOL: "standard"
    NUM_DECODER_LAYERS: 6
    PIXEL_DECODER_LAYERS: 6
    HIDDEN_DIM: 256
    NUM_HEADS: 8
    MASK_WEIGHT: 5.0
    DICE_WEIGHT: 5.0
    CLASS_WEIGHT: 2.0
    NO_OBJECT_WEIGHT: 0.1
    NUM_POINTS: 12544
    OBJECT_MASK_THRESHOLD: 0.4
    OVERLAP_THRESHOLD: 0.8
    PYRAMID_CHANNELS: 256
    ADAPTER_BLOCKS: 4
    ADAPTER_EMBED_DIM: 768
    DROPPATH: 0.0
  # Aux semantic-head losses OFF at M0 (we switch meta-arch entirely).
  SEM_SEG_HEAD:
    LOVASZ_WEIGHT: 0.0
    BOUNDARY_WEIGHT: 0.0
    STEGO_WEIGHT: 0.0
    DEPTH_SMOOTH_WEIGHT: 0.0
    GATED_CRF_WEIGHT: 0.0
    NECO_WEIGHT: 0.0

TRAINING:
  STEPS: 20000
  BATCH_SIZE: 1
  ACCUMULATE_GRAD_BATCHES: 16
  PRECISION: "bf16-mixed"
  CLASS_WEIGHTING: True
  OPTIMIZER: "adamw"
  ADAMW:
    LEARNING_RATE: 0.0001
    WEIGHT_DECAY: 0.00001
  VAL_EVERY_N_STEPS: 1000
  LOG_EVERT_N_STEPS: 50
  DROP_LOSS: False          # Mask2Former has its own matching loss
  GRADIENT_CLIP_ALGORITHM: "norm"
  GRADIENT_CLIP_VAL: 0.1

AUGMENTATION:
  COPY_PASTE: False          # OFF at M0, baseline only
  NUM_STEPS_STARTUP: 500
  LSJ:
    ENABLED: False
  COLOR_JITTER:
    ENABLED: False

SYSTEM:
  ACCELERATOR: "gpu"
  NUM_GPUS: 1
  NUM_NODES: 1
  NUM_WORKERS: 2
  LOG_PATH: "results/stage2_m2f/M0_baseline"
  RUN_NAME: "M0_baseline_dinov3_vitb_k80"

VALIDATION:
  CACHE_DEVICE: "cpu"
  USE_TTA: False
  USE_DENSE_CRF: False
```

- [ ] **Step 2: Author launcher**

`scripts/train_stage2_m2f.sh`:
```bash
#!/usr/bin/env bash
# Usage: scripts/train_stage2_m2f.sh <config-relative-path>
# Example: scripts/train_stage2_m2f.sh configs/stage2_m2f/M0_baseline_dinov3_vitb_k80.yaml
set -euo pipefail

CFG="${1:?config path required}"
RUN_NAME="$(basename "$CFG" .yaml)"
LOGDIR="logs/stage2_m2f"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Training $CFG"
echo "Log -> $LOGFILE"

nohup python -u refs/cups/train_pseudo.py --config "$CFG" \
  > "$LOGFILE" 2>&1 &

PID=$!
echo "PID=$PID"
echo "Tail with: tail -f $LOGFILE"
```

`scripts/eval_stage2_m2f.sh`:
```bash
#!/usr/bin/env bash
# Usage: scripts/eval_stage2_m2f.sh <results-dir>
set -euo pipefail

RESULTS="${1:?results directory required}"
CKPT="$(ls "$RESULTS"/checkpoints/*.ckpt | head -n1)"
CFG="$RESULTS/config.yaml"
OUT="$RESULTS/eval.json"

python refs/cups/eval_pseudo.py \
  --config "$CFG" \
  --checkpoint "$CKPT" \
  --output "$OUT"

echo "Wrote $OUT"
```

- [ ] **Step 3: Make launchers executable**

```bash
chmod +x scripts/train_stage2_m2f.sh scripts/eval_stage2_m2f.sh
```

- [ ] **Step 4: Smoke test (1-iter dry run)**

Run one optimizer step and verify all loss keys appear:
```bash
python -u refs/cups/train_pseudo.py \
  --config configs/stage2_m2f/M0_baseline_dinov3_vitb_k80.yaml \
  --max_steps 1 \
  --log_every_n_steps 1 \
  2>&1 | tee logs/stage2_m2f/M0_smoke.log
```

Expected: loss keys include `loss_ce`, `loss_mask`, `loss_dice`, and per-layer `loss_ce_N`, `loss_mask_N`, `loss_dice_N` for N in `[0, num_decoder_layers-2]`.

- [ ] **Step 5: Commit**

```bash
git add configs/stage2_m2f/M0_baseline_dinov3_vitb_k80.yaml \
        scripts/train_stage2_m2f.sh \
        scripts/eval_stage2_m2f.sh
git commit -m "feat(m2f): add M0 baseline config + launcher/eval scripts"
```

---

## Phase 1 - Guaranteed Levers (G1-G10): per-lever ablations

Phase 1 runs the M0 baseline once for the anchor, then ten 1-lever configs each on top of M0. Each `_BASE_`s `M0_baseline_dinov3_vitb_k80.yaml` and enables exactly one lever. A lever passes the +0.5 PQ gate to be included in M1.

### Task 1.1: Create G1-G10 config files

**Files:**
- Create: `configs/stage2_m2f/G1_EMA.yaml` through `configs/stage2_m2f/G10_DeeperDec.yaml`

Ten minimal override configs, all `_BASE_` on M0.

- [ ] **Step 1: Author G1-G10 configs**

`configs/stage2_m2f/G1_EMA.yaml`:
```yaml
_BASE_: "M0_baseline_dinov3_vitb_k80.yaml"
MODEL:
  EMA:
    ENABLED: True
    DECAY: 0.9998
SYSTEM:
  RUN_NAME: "G1_EMA"
  LOG_PATH: "results/stage2_m2f/G1_EMA"
```

`configs/stage2_m2f/G2_SWA.yaml`:
```yaml
_BASE_: "M0_baseline_dinov3_vitb_k80.yaml"
MODEL:
  SWA:
    ENABLED: True
    NUM_CKPTS: 5
    START_FRACTION: 0.75
SYSTEM:
  RUN_NAME: "G2_SWA"
  LOG_PATH: "results/stage2_m2f/G2_SWA"
```

`configs/stage2_m2f/G3_LSJ.yaml`:
```yaml
_BASE_: "M0_baseline_dinov3_vitb_k80.yaml"
AUGMENTATION:
  LSJ:
    ENABLED: True
    MIN_SCALE: 0.1
    MAX_SCALE: 2.0
SYSTEM:
  RUN_NAME: "G3_LSJ"
  LOG_PATH: "results/stage2_m2f/G3_LSJ"
```

`configs/stage2_m2f/G4_ColorJitter.yaml`:
```yaml
_BASE_: "M0_baseline_dinov3_vitb_k80.yaml"
AUGMENTATION:
  COLOR_JITTER:
    ENABLED: True
    BRIGHTNESS: 0.4
    CONTRAST: 0.4
    SATURATION: 0.4
    HUE: 0.1
SYSTEM:
  RUN_NAME: "G4_ColorJitter"
  LOG_PATH: "results/stage2_m2f/G4_ColorJitter"
```

`configs/stage2_m2f/G5_DropPath.yaml`:
```yaml
_BASE_: "M0_baseline_dinov3_vitb_k80.yaml"
MODEL:
  MASK2FORMER:
    DROPPATH: 0.3
SYSTEM:
  RUN_NAME: "G5_DropPath"
  LOG_PATH: "results/stage2_m2f/G5_DropPath"
```

`configs/stage2_m2f/G6_CRF.yaml`:
```yaml
_BASE_: "M0_baseline_dinov3_vitb_k80.yaml"
VALIDATION:
  USE_DENSE_CRF: True
  DENSE_CRF_ITER: 5
  DENSE_CRF_BI_W: 4.0
  DENSE_CRF_POS_W: 3.0
SYSTEM:
  RUN_NAME: "G6_CRF"
  LOG_PATH: "results/stage2_m2f/G6_CRF"
```

`configs/stage2_m2f/G7_LongSchedule.yaml`:
```yaml
_BASE_: "M0_baseline_dinov3_vitb_k80.yaml"
TRAINING:
  STEPS: 30000
  VAL_EVERY_N_STEPS: 1500
SYSTEM:
  RUN_NAME: "G7_LongSchedule"
  LOG_PATH: "results/stage2_m2f/G7_LongSchedule"
```

`configs/stage2_m2f/G8_LargerCrop.yaml`:
```yaml
_BASE_: "M0_baseline_dinov3_vitb_k80.yaml"
DATA:
  CROP_RESOLUTION: (768, 1536)
SYSTEM:
  RUN_NAME: "G8_LargerCrop"
  LOG_PATH: "results/stage2_m2f/G8_LargerCrop"
```

`configs/stage2_m2f/G9_MoreQueries.yaml`:
```yaml
_BASE_: "M0_baseline_dinov3_vitb_k80.yaml"
MODEL:
  MASK2FORMER:
    NUM_QUERIES: 200
SYSTEM:
  RUN_NAME: "G9_MoreQueries"
  LOG_PATH: "results/stage2_m2f/G9_MoreQueries"
```

`configs/stage2_m2f/G10_DeeperDec.yaml`:
```yaml
_BASE_: "M0_baseline_dinov3_vitb_k80.yaml"
MODEL:
  MASK2FORMER:
    NUM_DECODER_LAYERS: 9
SYSTEM:
  RUN_NAME: "G10_DeeperDec"
  LOG_PATH: "results/stage2_m2f/G10_DeeperDec"
```

- [ ] **Step 2: Sanity-check each loads with `get_default_config`**

Run this one-liner — each must succeed without errors:
```bash
for f in configs/stage2_m2f/G*.yaml; do
  python -c "from cups.config import get_default_config; c=get_default_config('$f'); print('OK', '$f', c.SYSTEM.RUN_NAME)" || echo "FAIL $f"
done
```

Expected output: 10 lines starting with `OK`.

- [ ] **Step 3: Commit**

```bash
git add configs/stage2_m2f/G*.yaml
git commit -m "feat(m2f): add G1-G10 ablation configs"
```

### Task 1.2: Run M0 baseline (anchor)

**Files:** (none new — uses scripts from Task 0.18)

- [ ] **Step 1: Launch M0 run**

```bash
scripts/train_stage2_m2f.sh configs/stage2_m2f/M0_baseline_dinov3_vitb_k80.yaml
```

- [ ] **Step 2: Monitor to completion**

Tail until val metrics print at step 20000:
```bash
tail -f logs/stage2_m2f/M0_baseline_dinov3_vitb_k80_*.log
```

Record the final `pq_val`, `pq_t_val`, `pq_s_val`, `miou_val` from the Lightning log line into `reports/stage2_m2f_ablation.md`.

- [ ] **Step 3: Evaluate with TTA scales**

```bash
scripts/eval_stage2_m2f.sh results/stage2_m2f/M0_baseline
```

Cite the JSON at `results/stage2_m2f/M0_baseline/eval.json` as the M0 anchor.

- [ ] **Step 4: Append numbers to report**

Create `reports/stage2_m2f_ablation.md` with a single row for M0:
```markdown
| Config | PQ | PQ_stuff | PQ_things | mIoU | delta_PQ | delta_stuff | notes |
|--------|----|----|----|----|----|----|-------|
| M0_baseline | ... | ... | ... | ... | -- | -- | anchor |
```

- [ ] **Step 5: Commit report**

```bash
git add reports/stage2_m2f_ablation.md
git commit -m "docs(m2f): record M0 baseline anchor"
```

### Task 1.3: Run G1-G10 (each single lever)

**Files:**
- Create: `scripts/run_stage2_m2f_ablations.sh`

- [ ] **Step 1: Author sweep runner**

`scripts/run_stage2_m2f_ablations.sh`:
```bash
#!/usr/bin/env bash
# Usage: scripts/run_stage2_m2f_ablations.sh <glob>
# Example: scripts/run_stage2_m2f_ablations.sh 'configs/stage2_m2f/G*.yaml'
set -euo pipefail

GLOB="${1:?glob required (e.g. 'configs/stage2_m2f/G*.yaml')}"
for CFG in $GLOB; do
  echo "=== Launching $CFG ==="
  scripts/train_stage2_m2f.sh "$CFG"
  # Wait for foreground completion? No — launcher backgrounds.
  # This script kicks off all 10 in sequence; user coordinates GPU availability.
  sleep 2
done
```

- [ ] **Step 2: Launch G1-G10 sequentially**

```bash
chmod +x scripts/run_stage2_m2f_ablations.sh
scripts/run_stage2_m2f_ablations.sh 'configs/stage2_m2f/G*.yaml'
```

Note: this requires 10x GPU-hours minimum. In practice, run 1-2 at a time on available GPUs. Record the order in `reports/stage2_m2f_ablation.md`.

- [ ] **Step 3: Evaluate each and populate table**

For each G-run:
```bash
scripts/eval_stage2_m2f.sh results/stage2_m2f/G1_EMA
scripts/eval_stage2_m2f.sh results/stage2_m2f/G2_SWA
# ...
scripts/eval_stage2_m2f.sh results/stage2_m2f/G10_DeeperDec
```

Append each lever's row to `reports/stage2_m2f_ablation.md`:
```markdown
| G1_EMA | <pq> | <stuff> | <things> | <miou> | <delta> | <delta_stuff> | <notes> |
| G2_SWA | ... |
```

- [ ] **Step 4: Mark "wins" (delta >= +0.5 PQ)**

Add a WINNERS section at the end of the report:
```markdown
### M1 Ingredients (delta PQ >= +0.5 vs M0)

- [ ] G1_EMA (yes/no)
- [ ] G2_SWA
- ...

Any lever with delta_PQ < +0.5 is excluded from M1.
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_stage2_m2f_ablations.sh reports/stage2_m2f_ablation.md
git commit -m "docs(m2f): record G1-G10 per-lever ablation results"
```

---

## Phase 2 - M1 Stacked Guaranteed Levers

M1 enables *only* the levers that passed the +0.5 PQ gate. Other levers stay at M0 defaults.

### Task 2.1: Create M1 stacked config and run

**Files:**
- Create: `configs/stage2_m2f/M1_stacked_guaranteed.yaml`

- [ ] **Step 1: Author M1 config based on Phase 1 winners**

Template — fill in actual booleans per Phase 1 results. Example assuming G1, G3, G4, G7, G10 passed:

`configs/stage2_m2f/M1_stacked_guaranteed.yaml`:
```yaml
# M1: M0 + (G-levers that passed the +0.5 PQ gate in Phase 1).
# Replace enabled flags below with the actual Phase 1 winners.
_BASE_: "M0_baseline_dinov3_vitb_k80.yaml"

MODEL:
  EMA:
    ENABLED: True          # G1 assumed winner
    DECAY: 0.9998
  SWA:
    ENABLED: False         # G2: replace with True if it passed
  MASK2FORMER:
    NUM_QUERIES: 100       # G9: keep 100 or change to 200 if G9 passed
    NUM_DECODER_LAYERS: 9  # G10 assumed winner

AUGMENTATION:
  LSJ:
    ENABLED: True          # G3 assumed winner
  COLOR_JITTER:
    ENABLED: True          # G4 assumed winner

TRAINING:
  STEPS: 30000             # G7 assumed winner

VALIDATION:
  USE_DENSE_CRF: False     # G6: replace if it passed

SYSTEM:
  RUN_NAME: "M1_stacked_guaranteed"
  LOG_PATH: "results/stage2_m2f/M1_stacked_guaranteed"
```

- [ ] **Step 2: Sanity-check loads**

```bash
python -c "from cups.config import get_default_config; c=get_default_config('configs/stage2_m2f/M1_stacked_guaranteed.yaml'); print(c.SYSTEM.RUN_NAME, c.MODEL.EMA.ENABLED, c.AUGMENTATION.LSJ.ENABLED)"
```

Expected: prints `M1_stacked_guaranteed True True` (or whatever flags you set).

- [ ] **Step 3: Launch M1 run**

```bash
scripts/train_stage2_m2f.sh configs/stage2_m2f/M1_stacked_guaranteed.yaml
```

- [ ] **Step 4: Evaluate and append to report**

```bash
scripts/eval_stage2_m2f.sh results/stage2_m2f/M1_stacked_guaranteed
```

Append row to `reports/stage2_m2f_ablation.md`. Compute delta vs M0 and vs the best single G-lever.

- [ ] **Step 5: Commit**

```bash
git add configs/stage2_m2f/M1_stacked_guaranteed.yaml reports/stage2_m2f_ablation.md
git commit -m "feat(m2f): add M1 stacked-guaranteed config + record run"
```

---

## Phase 3 - Novel Levers (N1-N5): per-lever ablations on top of M1

Each N-lever is one override on top of M1. Same +0.5 PQ gate applies.

### Task 3.1: Create N1-N5 configs and run each

**Files:**
- Create: `configs/stage2_m2f/N1_DecoupledQueries.yaml`
- Create: `configs/stage2_m2f/N2_DepthQueryBias.yaml`
- Create: `configs/stage2_m2f/N3_XQuery.yaml`
- Create: `configs/stage2_m2f/N4_QueryConsistency.yaml`
- Create: `configs/stage2_m2f/N5_SelfTrain.yaml`

- [ ] **Step 1: Author N1 — Decoupled queries**

`configs/stage2_m2f/N1_DecoupledQueries.yaml`:
```yaml
_BASE_: "M1_stacked_guaranteed.yaml"
MODEL:
  MASK2FORMER:
    QUERY_POOL: "decoupled"
    QUERIES_STUFF: 150
    QUERIES_THING: 50
SYSTEM:
  RUN_NAME: "N1_DecoupledQueries"
  LOG_PATH: "results/stage2_m2f/N1_DecoupledQueries"
```

- [ ] **Step 2: Author N2 — Depth-conditioned query bias**

`configs/stage2_m2f/N2_DepthQueryBias.yaml`:
```yaml
_BASE_: "M1_stacked_guaranteed.yaml"
MODEL:
  MASK2FORMER:
    QUERY_POOL: "depth_bias"
DATA:
  DEPTH_SUBDIR: "depth_depthpro"      # confirm path exists
SYSTEM:
  RUN_NAME: "N2_DepthQueryBias"
  LOG_PATH: "results/stage2_m2f/N2_DepthQueryBias"
```

- [ ] **Step 3: Author N3 — XQuery cross-image correspondence**

`configs/stage2_m2f/N3_XQuery.yaml`:
```yaml
_BASE_: "M1_stacked_guaranteed.yaml"
MODEL:
  MASK2FORMER:
    XQUERY_WEIGHT: 0.1
    XQUERY_TEMPERATURE: 0.1
TRAINING:
  BATCH_SIZE: 2       # XQuery needs B>=2 in the batch
  ACCUMULATE_GRAD_BATCHES: 8
SYSTEM:
  RUN_NAME: "N3_XQuery"
  LOG_PATH: "results/stage2_m2f/N3_XQuery"
```

- [ ] **Step 4: Author N4 — Teacher/student query consistency**

`configs/stage2_m2f/N4_QueryConsistency.yaml`:
```yaml
_BASE_: "M1_stacked_guaranteed.yaml"
MODEL:
  EMA:
    ENABLED: True         # N4 requires an EMA teacher
    DECAY: 0.9998
  MASK2FORMER:
    QUERY_CONSISTENCY_WEIGHT: 0.1
    QUERY_CONSISTENCY_TEMPERATURE: 0.1
SYSTEM:
  RUN_NAME: "N4_QueryConsistency"
  LOG_PATH: "results/stage2_m2f/N4_QueryConsistency"
```

- [ ] **Step 5: Author N5 — Self-training with confidence threshold**

`configs/stage2_m2f/N5_SelfTrain.yaml`:
```yaml
_BASE_: "M1_stacked_guaranteed.yaml"
MODEL:
  EMA:
    ENABLED: True
    DECAY: 0.9998
  MASK2FORMER:
    SELF_TRAIN_THRESHOLD: 0.95
SELF_TRAINING:
  ROUND_STEPS: 1000
  ROUNDS: 3
  SEMANTIC_SEGMENTATION_THRESHOLD: 0.95
SYSTEM:
  RUN_NAME: "N5_SelfTrain"
  LOG_PATH: "results/stage2_m2f/N5_SelfTrain"
```

- [ ] **Step 6: Sanity-check each loads**

```bash
for f in configs/stage2_m2f/N*.yaml; do
  python -c "from cups.config import get_default_config; c=get_default_config('$f'); print('OK', '$f', c.SYSTEM.RUN_NAME)" || echo "FAIL $f"
done
```

Expected: 5 OK lines.

- [ ] **Step 7: Wire N3/N4 hooks into `Mask2FormerPanoptic`**

Before launching N3/N4, we must plumb the loss hooks. Modify `refs/cups/cups/model/model_mask2former.py::build_mask2former_vitb`:

Inside the builder, after constructing `Mask2FormerPanoptic`, attach hooks:

```python
    # Optional N-lever hooks (only set if the corresponding weight > 0).
    from cups.losses.xquery import xquery_loss
    from cups.losses.query_consistency import query_consistency_loss

    aux_loss_hooks: Dict[str, Any] = {}
    if m.XQUERY_WEIGHT > 0.0:
        def _xquery_hook(dec_out, targets, ctx, _w=m.XQUERY_WEIGHT, _t=m.XQUERY_TEMPERATURE):
            return _w * xquery_loss(dec_out, targets, {"temperature": _t})
        aux_loss_hooks["xquery"] = _xquery_hook
    if m.QUERY_CONSISTENCY_WEIGHT > 0.0:
        def _qc_hook(dec_out, targets, ctx, _w=m.QUERY_CONSISTENCY_WEIGHT, _t=m.QUERY_CONSISTENCY_TEMPERATURE):
            return _w * query_consistency_loss(dec_out, targets, {**ctx, "temperature": _t})
        aux_loss_hooks["query_consistency"] = _qc_hook
    if aux_loss_hooks:
        model.aux_loss_hooks = aux_loss_hooks
        model.aux_loss_hooks["return_query_embeds"] = True
```

And extend tests in `tests/stage2_m2f/test_meta_arch.py` with a new test that flips `aux_loss_hooks["xquery"]` and asserts `loss_xquery` appears. Run:
```bash
pytest tests/stage2_m2f/test_meta_arch.py -v
```

- [ ] **Step 8: Run N1-N5**

```bash
scripts/run_stage2_m2f_ablations.sh 'configs/stage2_m2f/N*.yaml'
```

- [ ] **Step 9: Evaluate each**

For each N-run:
```bash
scripts/eval_stage2_m2f.sh results/stage2_m2f/N1_DecoupledQueries
scripts/eval_stage2_m2f.sh results/stage2_m2f/N2_DepthQueryBias
scripts/eval_stage2_m2f.sh results/stage2_m2f/N3_XQuery
scripts/eval_stage2_m2f.sh results/stage2_m2f/N4_QueryConsistency
scripts/eval_stage2_m2f.sh results/stage2_m2f/N5_SelfTrain
```

- [ ] **Step 10: Append to report and flag wins (+0.5 PQ over M1)**

```markdown
| N1_DecoupledQueries | ... | ... | ... | ... | ... | ... | |
| N2_DepthQueryBias | ... |
| N3_XQuery | ... |
| N4_QueryConsistency | ... |
| N5_SelfTrain | ... |

### M2 Ingredients (delta PQ >= +0.5 vs M1)
- [ ] N1 (yes/no)
- [ ] N2
- [ ] N3
- [ ] N4
- [ ] N5
```

- [ ] **Step 11: Commit**

```bash
git add configs/stage2_m2f/N*.yaml \
        refs/cups/cups/model/model_mask2former.py \
        tests/stage2_m2f/test_meta_arch.py \
        reports/stage2_m2f_ablation.md
git commit -m "feat(m2f): add N1-N5 configs + wire xquery/query_consistency hooks"
```

---

## Phase 4 - M2 Stacked Novel Levers

M2 = M1 + novel wins only.

### Task 4.1: Create M2 stacked config and run

**Files:**
- Create: `configs/stage2_m2f/M2_stacked_novel.yaml`

- [ ] **Step 1: Author M2 config**

Template — assume N1, N2, N4 passed:

`configs/stage2_m2f/M2_stacked_novel.yaml`:
```yaml
# M2: M1 + (N-levers that passed the +0.5 PQ gate in Phase 3).
# Replace enabled flags below with actual Phase 3 winners.
_BASE_: "M1_stacked_guaranteed.yaml"

MODEL:
  MASK2FORMER:
    QUERY_POOL: "decoupled"          # N1 assumed winner
    QUERIES_STUFF: 150
    QUERIES_THING: 50
    XQUERY_WEIGHT: 0.0               # N3: replace with 0.1 if it passed
    QUERY_CONSISTENCY_WEIGHT: 0.1    # N4 assumed winner
    QUERY_CONSISTENCY_TEMPERATURE: 0.1
  EMA:
    ENABLED: True
    DECAY: 0.9998

# N2 (depth_bias) only if it actually passed — incompatible with N1 decoupled.
# If BOTH pass, priority: keep the one with higher delta; document the choice.

SYSTEM:
  RUN_NAME: "M2_stacked_novel"
  LOG_PATH: "results/stage2_m2f/M2_stacked_novel"
```

- [ ] **Step 2: Note incompatibilities explicitly**

In the config top-comment and in `reports/stage2_m2f_ablation.md`, record:
- If both N1 (decoupled) and N2 (depth_bias) passed, only ONE can activate (QueryPool is a single factory output). Choose whichever had the larger delta_PQ. Document the choice in the ablation report.
- N3 (XQuery) requires BATCH_SIZE >= 2; if GPU memory does not allow, keep `ACCUMULATE_GRAD_BATCHES` higher so effective batch stays equivalent.
- N4 (QueryConsistency) requires `MODEL.EMA.ENABLED: True`; M2 config already sets this.

- [ ] **Step 3: Sanity-check loads**

```bash
python -c "from cups.config import get_default_config; c=get_default_config('configs/stage2_m2f/M2_stacked_novel.yaml'); print(c.SYSTEM.RUN_NAME, c.MODEL.MASK2FORMER.QUERY_POOL, c.MODEL.MASK2FORMER.QUERY_CONSISTENCY_WEIGHT)"
```

- [ ] **Step 4: Launch M2 run**

```bash
scripts/train_stage2_m2f.sh configs/stage2_m2f/M2_stacked_novel.yaml
```

- [ ] **Step 5: Evaluate and append to report**

```bash
scripts/eval_stage2_m2f.sh results/stage2_m2f/M2_stacked_novel
```

Append the final row to `reports/stage2_m2f_ablation.md`:
```markdown
| M2_stacked_novel | <pq> | <stuff> | <things> | <miou> | <delta vs M0> | <delta vs M1> | FINAL |
```

If `delta PQ_stuff >= +5` vs M0, primary goal is hit. If `PQ_things >= 20`, secondary goal is hit.

- [ ] **Step 6: Commit**

```bash
git add configs/stage2_m2f/M2_stacked_novel.yaml reports/stage2_m2f_ablation.md
git commit -m "feat(m2f): add M2 stacked-novel config + record final run"
```

---

## Phase 5 - Leave-One-Out Validation

Once M2 is trained, verify that each winner truly contributes by removing it one at a time and retraining the **shortened** schedule (half the steps is enough — deltas transfer). A lever is *confirmed* if leaving it out causes >=0.3 PQ degradation.

### Task 5.1: Create per-winner LOO configs and retrain each

**Files:**
- Create: `configs/stage2_m2f/LOO_noG1.yaml` (and one per winner)

- [ ] **Step 1: For each winner in M1 or M2, author a `LOO_no<LEVER>.yaml`**

Example — if winners are G1, G3, G4, G7, G10, N1, N4, you create 7 LOO configs. Each `_BASE_`s M2 and toggles the single lever back off.

`configs/stage2_m2f/LOO_noG1.yaml`:
```yaml
# Leave-one-out: disable EMA (G1) only.
_BASE_: "M2_stacked_novel.yaml"
MODEL:
  EMA:
    ENABLED: False
TRAINING:
  STEPS: 15000       # half schedule for LOO sweep
SYSTEM:
  RUN_NAME: "LOO_noG1"
  LOG_PATH: "results/stage2_m2f/LOO_noG1"
```

Repeat this pattern for every winner. Example `LOO_noG3.yaml`:
```yaml
_BASE_: "M2_stacked_novel.yaml"
AUGMENTATION:
  LSJ:
    ENABLED: False
TRAINING:
  STEPS: 15000
SYSTEM:
  RUN_NAME: "LOO_noG3"
  LOG_PATH: "results/stage2_m2f/LOO_noG3"
```

`LOO_noN1.yaml`:
```yaml
_BASE_: "M2_stacked_novel.yaml"
MODEL:
  MASK2FORMER:
    QUERY_POOL: "standard"       # revert to M1 default
    NUM_QUERIES: 100
TRAINING:
  STEPS: 15000
SYSTEM:
  RUN_NAME: "LOO_noN1"
  LOG_PATH: "results/stage2_m2f/LOO_noN1"
```

`LOO_noN4.yaml`:
```yaml
_BASE_: "M2_stacked_novel.yaml"
MODEL:
  MASK2FORMER:
    QUERY_CONSISTENCY_WEIGHT: 0.0
TRAINING:
  STEPS: 15000
SYSTEM:
  RUN_NAME: "LOO_noN4"
  LOG_PATH: "results/stage2_m2f/LOO_noN4"
```

- [ ] **Step 2: Sanity-check loads**

```bash
for f in configs/stage2_m2f/LOO_*.yaml; do
  python -c "from cups.config import get_default_config; c=get_default_config('$f'); print('OK', '$f', c.SYSTEM.RUN_NAME)" || echo "FAIL $f"
done
```

- [ ] **Step 3: Launch all LOO runs**

```bash
scripts/run_stage2_m2f_ablations.sh 'configs/stage2_m2f/LOO_*.yaml'
```

- [ ] **Step 4: Evaluate each**

```bash
for d in results/stage2_m2f/LOO_*; do
  scripts/eval_stage2_m2f.sh "$d"
done
```

- [ ] **Step 5: Update report with LOO section**

Append to `reports/stage2_m2f_ablation.md`:
```markdown
## Leave-One-Out Validation

For each winner in M2, the LOO row shows PQ when that lever is disabled
on top of M2. Delta is M2_PQ - LOO_PQ (positive = lever helps).

| LOO Config | Disabled Lever | PQ | delta vs M2 | Confirmed? |
|---|---|---|---|---|
| LOO_noG1 | G1 EMA | ... | ... | yes/no |
| LOO_noG3 | G3 LSJ | ... | ... | |
| LOO_noN1 | N1 decoupled | ... | ... | |
| LOO_noN4 | N4 query-consistency | ... | ... | |
```

A lever with `delta < +0.3` is a suspect — consider dropping it from the final M2.

- [ ] **Step 6: Commit**

```bash
git add configs/stage2_m2f/LOO_*.yaml reports/stage2_m2f_ablation.md
git commit -m "feat(m2f): add leave-one-out validation configs + results"
```

### Task 5.2: Finalize M2 after LOO corrections

**Files:**
- Modify: `configs/stage2_m2f/M2_stacked_novel.yaml`

- [ ] **Step 1: Remove any lever that failed LOO (delta < +0.3)**

Edit M2 in place: if `LOO_noG2` showed `delta = -0.1` (SWA actually hurt when stacked), set `SWA.ENABLED: False` in M2. Commit the revision:
```bash
git add configs/stage2_m2f/M2_stacked_novel.yaml
git commit -m "fix(m2f): drop non-contributing lever after LOO validation"
```

- [ ] **Step 2: Retrain the corrected M2 at full schedule**

```bash
scripts/train_stage2_m2f.sh configs/stage2_m2f/M2_stacked_novel.yaml
scripts/eval_stage2_m2f.sh results/stage2_m2f/M2_stacked_novel
```

- [ ] **Step 3: Record the corrected M2 final PQ in the report**

Append a "Final M2" section citing the new eval JSON. This is the reference number for Phase 6 (TTA sweep) and Phase 7 (self-training).

- [ ] **Step 4: Commit**

```bash
git add reports/stage2_m2f_ablation.md
git commit -m "docs(m2f): record corrected M2 final PQ after LOO"
```

---

## Phase 6 - TTA Sweep on M2

TTA (test-time augmentation) is free at inference. Sweep a handful of scale combinations on the corrected M2 checkpoint. No retraining.

### Task 6.1: Sweep TTA scales on M2 checkpoint

**Files:**
- Create: `scripts/eval_stage2_m2f_tta.sh`

- [ ] **Step 1: Author TTA sweep runner**

`scripts/eval_stage2_m2f_tta.sh`:
```bash
#!/usr/bin/env bash
# Sweeps TTA scale combinations over the M2 checkpoint.
# Usage: scripts/eval_stage2_m2f_tta.sh <results-dir>
set -euo pipefail

RESULTS="${1:?results directory required}"
CKPT="$(ls "$RESULTS"/checkpoints/*.ckpt | head -n1)"
CFG="$RESULTS/config.yaml"

declare -a SCALE_SETS=(
  "1.0"
  "0.75 1.0 1.25"
  "0.5 0.75 1.0 1.25 1.5"
  "0.75 1.0 1.25 1.5"
)

for SCALES in "${SCALE_SETS[@]}"; do
  TAG="$(echo "$SCALES" | tr ' ' '_')"
  OUT="$RESULTS/eval_tta_${TAG}.json"
  python refs/cups/eval_pseudo.py \
    --config "$CFG" \
    --checkpoint "$CKPT" \
    --output "$OUT" \
    --override MODEL.TTA_SCALES "($(echo $SCALES | tr ' ' ','))" VALIDATION.USE_TTA True
  echo "Wrote $OUT"
done
```

- [ ] **Step 2: Launch**

```bash
chmod +x scripts/eval_stage2_m2f_tta.sh
scripts/eval_stage2_m2f_tta.sh results/stage2_m2f/M2_stacked_novel
```

- [ ] **Step 3: Append TTA table to report**

```markdown
## TTA Scale Sweep (M2 checkpoint)

| Scales | PQ | PQ_stuff | PQ_things | mIoU | delta vs no-TTA |
|---|---|---|---|---|---|
| 1.0 (no-TTA) | ... | ... | ... | ... | -- |
| 0.75, 1.0, 1.25 | ... | ... | ... | ... | ... |
| 0.5, 0.75, 1.0, 1.25, 1.5 | ... | ... | ... | ... | ... |
| 0.75, 1.0, 1.25, 1.5 | ... | ... | ... | ... | ... |
```

Select the best set and set it as the default in M2 config for any downstream eval.

- [ ] **Step 4: Commit**

```bash
git add scripts/eval_stage2_m2f_tta.sh reports/stage2_m2f_ablation.md
git commit -m "feat(m2f): TTA scale sweep on M2"
```

---

## Phase 7 - Stage-3 Self-Training on M2

Use the corrected M2 as the initial teacher. Generate pseudo-labels on Cityscapes train set, confidence-threshold at tau=0.95 (N5), and fine-tune for 3 rounds of 1000 steps each.

### Task 7.1: Self-training config and orchestrator

**Files:**
- Create: `configs/stage2_m2f/Stage3_self_train.yaml`
- Create: `scripts/run_stage2_m2f_self_train.sh`

- [ ] **Step 1: Author Stage-3 config**

`configs/stage2_m2f/Stage3_self_train.yaml`:
```yaml
# Stage-3 self-training starting from M2 checkpoint.
_BASE_: "M2_stacked_novel.yaml"

MODEL:
  CHECKPOINT: "results/stage2_m2f/M2_stacked_novel/checkpoints/last.ckpt"
  EMA:
    ENABLED: True
    DECAY: 0.9998
  MASK2FORMER:
    SELF_TRAIN_THRESHOLD: 0.95

SELF_TRAINING:
  ROUND_STEPS: 1000
  ROUNDS: 3
  CONFIDENCE_STEP: 0.02              # raise tau by 0.02 per round: 0.95 -> 0.97 -> 0.99
  SEMANTIC_SEGMENTATION_THRESHOLD: 0.95
  USE_DROP_LOSS: False

TRAINING:
  STEPS: 3000                        # total 3 x ROUND_STEPS
  VAL_EVERY_N_STEPS: 500

SYSTEM:
  RUN_NAME: "Stage3_self_train"
  LOG_PATH: "results/stage2_m2f/Stage3_self_train"
```

- [ ] **Step 2: Author orchestrator**

`scripts/run_stage2_m2f_self_train.sh`:
```bash
#!/usr/bin/env bash
# Stage-3 self-training orchestrator.
# Generates pseudo-labels using EMA teacher, thresholds at tau, fine-tunes
# one round, then repeats. Each round produces its own label set and
# checkpoint for ablation auditing.
set -euo pipefail

CFG="configs/stage2_m2f/Stage3_self_train.yaml"
RUN_NAME="Stage3_self_train"
RESULTS="results/stage2_m2f/${RUN_NAME}"
mkdir -p "$RESULTS"

# refs/cups/self_train_pseudo.py is assumed to implement the round loop;
# if it doesn't exist yet, create it as a thin wrapper that calls
# train_pseudo.py per round, passing --pseudo_label_source=teacher
# after Round 0.
nohup python -u refs/cups/self_train_pseudo.py \
  --config "$CFG" \
  --output "$RESULTS" \
  > "logs/stage2_m2f/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "PID=$!"
```

If `refs/cups/self_train_pseudo.py` does not exist, create a minimal wrapper:

`refs/cups/self_train_pseudo.py`:
```python
"""Stage-3 self-training orchestrator for M2 -> M2 fine-tuning.

Iterates SELF_TRAINING.ROUNDS rounds. In each round:
  1. Use the current model (loaded from CHECKPOINT) to predict pseudo
     panoptic labels on DATA.ROOT (train split).
  2. Filter by confidence threshold (N5 filter_by_confidence).
  3. Train for ROUND_STEPS with the new pseudo labels.
  4. Save the new checkpoint + EMA teacher as next round's source.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess

from cups.config import get_default_config

logging.basicConfig(format="%(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = get_default_config(args.config)
    rounds = cfg.SELF_TRAINING.ROUNDS
    for r in range(rounds):
        tau = cfg.SELF_TRAINING.SEMANTIC_SEGMENTATION_THRESHOLD + r * cfg.SELF_TRAINING.CONFIDENCE_STEP
        log.info("=== Stage-3 Round %d/%d (tau=%.2f) ===", r + 1, rounds, tau)
        # Step 1+2: generate + filter pseudo labels.
        subprocess.run([
            "python", "refs/cups/generate_pseudo_labels.py",
            "--config", args.config,
            "--round", str(r),
            "--tau", str(tau),
            "--out", os.path.join(args.output, f"round_{r:02d}_labels"),
        ], check=True)
        # Step 3: train.
        subprocess.run([
            "python", "refs/cups/train_pseudo.py",
            "--config", args.config,
            "--round", str(r),
            "--pseudo_root", os.path.join(args.output, f"round_{r:02d}_labels"),
            "--output", os.path.join(args.output, f"round_{r:02d}_ckpt"),
        ], check=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Launch**

```bash
chmod +x scripts/run_stage2_m2f_self_train.sh
scripts/run_stage2_m2f_self_train.sh
```

- [ ] **Step 4: Evaluate each round's checkpoint**

```bash
for r in results/stage2_m2f/Stage3_self_train/round_*_ckpt; do
  scripts/eval_stage2_m2f.sh "$r"
done
```

- [ ] **Step 5: Append Stage-3 progression to report**

```markdown
## Stage-3 Self-Training

| Round | tau | PQ | PQ_stuff | PQ_things | mIoU | delta vs M2 |
|---|---|---|---|---|---|---|
| M2 start | -- | ... | ... | ... | ... | -- |
| Round 1 | 0.95 | ... | ... | ... | ... | ... |
| Round 2 | 0.97 | ... | ... | ... | ... | ... |
| Round 3 | 0.99 | ... | ... | ... | ... | ... |
```

- [ ] **Step 6: Commit**

```bash
git add configs/stage2_m2f/Stage3_self_train.yaml \
        scripts/run_stage2_m2f_self_train.sh \
        refs/cups/self_train_pseudo.py \
        reports/stage2_m2f_ablation.md
git commit -m "feat(m2f): Stage-3 self-training orchestrator + round-by-round eval"
```

### Task 7.2: Final report summary

**Files:**
- Modify: `reports/stage2_m2f_ablation.md`

- [ ] **Step 1: Add an executive summary**

Write the top of the report (before all tables) with headline numbers:
```markdown
# Stage-2 Mask2Former + ViT-Adapter Ablation Report

## Headline

- **M0 baseline**: PQ=<X> / PQ_stuff=<Y> / PQ_things=<Z> / mIoU=<W>
- **M1 (+guaranteed wins)**: PQ=<X1> (+delta)
- **M2 (+novel wins)**: PQ=<X2> (+delta vs M1)
- **M2 + Stage-3 self-train**: PQ=<X3> (+delta vs M2, final)

## Deltas vs CUPS Cascade (PQ ~24.7)

- **delta PQ_stuff**: +<dS> (target >= +5)
- **delta PQ_things**: +<dT> (target: maintain >= 20)
- **Overall PQ**: <final> (target: >= 30)

## Winning levers

- Guaranteed: G?, G?, G?, ...
- Novel: N?, N?, ...
```

- [ ] **Step 2: Commit**

```bash
git add reports/stage2_m2f_ablation.md
git commit -m "docs(m2f): add executive summary to ablation report"
```

---

## Appendix A - Self-Review Checklist

After authoring the plan, run this checklist before handing off:

- [ ] Every file in the "File Structure" block has at least one Task that creates or modifies it.
- [ ] Every Task has: file paths, failing test, expected fail message, implementation, passing test expectation, commit command.
- [ ] Every code block is syntactically valid Python / YAML / bash.
- [ ] Search for placeholder strings: `TBD`, `TODO`, `FIXME`, `pass  #`, `...`, `implement later`, `fill in`. None remain except the intentional `...` in report tables (which the engineer fills in after runs).
- [ ] `scipy`, `yaml`, `pydensecrf` are listed in the project's existing env — add to `pyproject.toml` if any is missing.
- [ ] The launcher scripts (`train_stage2_m2f.sh`, `eval_stage2_m2f.sh`, `run_stage2_m2f_ablations.sh`, `eval_stage2_m2f_tta.sh`, `run_stage2_m2f_self_train.sh`) are all executable and log to `logs/stage2_m2f/`.
- [ ] All configs pass `python -c "from cups.config import get_default_config; get_default_config('<path>')"` smoke check.
- [ ] N3/N4 hooks are wired in `Mask2FormerPanoptic.forward`, NOT through `build_aux_losses`.
- [ ] yacs `_BASE_` patch handles 2-level chains; confirm with `test_base_inheritance_two_levels`.
- [ ] Phase 1 Task 1.3 reports commit the actual GPU-hours spent per run (for later planning).

## Appendix B - Target PQ Bookkeeping

Keep `reports/stage2_m2f_ablation.md` as the single source of truth. Every Task that runs a model MUST append one row with:

- Config file
- Checkpoint path
- Val PQ / PQ_stuff / PQ_things / mIoU
- Delta vs the previous stage (M0 / M1 / M2)
- TTA scales used (or "no-TTA")
- Any notes (e.g. "GPU ran OOM at step X; reduced batch")

This is the only artefact a reviewer needs to trace the path from PQ=24.7 to PQ>=30.


