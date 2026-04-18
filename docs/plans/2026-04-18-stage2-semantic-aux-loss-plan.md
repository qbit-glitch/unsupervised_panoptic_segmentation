# Stage-2 Semantic Loss Augmentation Plan (4 Passes)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Incrementally add four auxiliary semantic losses to CUPS Stage-2 training so the Cascade Mask R-CNN + DINOv3 ViT-B/16 panoptic model converges to higher PQ_stuff and semantic quality without regressing PQ_things.

**Architecture:** Each pass extends `CustomSemSegFPNHead.losses()` in `refs/cups/cups/model/modeling/roi_heads/semantic_seg.py` with an optional aux term, gated by a new `cfg.MODEL.SEM_SEG_HEAD.*_WEIGHT` key (default 0.0 = off). Losses live in a new `refs/cups/cups/losses/` package. Each pass ships with a test and its own derived config; promotion to the next pass is gated on **ΔPQ_stuff ≥ +0.5** over the prior checkpoint on Cityscapes val.

**Tech Stack:** PyTorch, Detectron2 (CUPS fork), DINOv3 ViT-B/16 frozen, k=80 semantic pseudo-labels, DepthPro τ=0.20 instance pseudo-labels, Cityscapes 500-img val, PQ / SQ / RQ / mIoU via existing `mbps_pytorch/evaluate_semantic_pseudolabels.py`.

---

## Context

Why this change: Current Stage-2 best checkpoint sits at **PQ ≈ 24.7 / PQ_stuff ≈ 31.9** vs CUPS target PQ_stuff 35.1. Per project memory, **instance quality is already resolved** (DepthPro τ=0.20 pseudo-labels hit PQ_things=23.35). The remaining gap is entirely semantic quality on stuff regions — a loss-function problem, not a backbone or pseudo-label problem. The semantic head today computes only **pixel-wise CE with class_weight**; no IoU-surrogate, no boundary term, no structural prior, no depth regularizer. This plan adds those signals one at a time so we know exactly which helps.

Four named passes:

| Pass | Name | Rationale | Gate |
|------|------|-----------|------|
| **P1** | **LoCE** — Lovász + Boundary-weighted CE | Direct IoU surrogate + sharper class boundaries. Cheap, proven. | Keep if ΔPQ_stuff ≥ +0.5 vs P0 |
| **P2** | **FeatMirror** — STEGO DINOv3 correspondence distillation | Wires the already-written `stego_loss.py` as self-supervised structural prior. | Keep if ΔPQ_stuff ≥ +0.5 vs best(P0,P1) |
| **P3** | **DGLR** — Depth-Guided Logit Regularizer | Penalises ‖∇logits‖ where ‖∇depth‖ is small. Compounds with DCFA story. | Keep if ΔPQ_stuff ≥ +0.5 vs best(P0,P1,P2) |
| **P4** | **DAff** — Dense-Affinity (Gated-CRF + NeCo) | Pairwise smoothness + patch-neighbourhood consistency. Heavier. | Only run after P1–P3 each ablated individually |

---

## Files to Modify / Create

### Modify (CUPS in-place, per user direction)
- `refs/cups/cups/config.py` — add `MODEL.SEM_SEG_HEAD` CfgNode sub-tree with new weight keys
- `refs/cups/cups/model/modeling/roi_heads/semantic_seg.py` — extend `CustomSemSegFPNHead.from_config`, `forward`, `losses`
- `refs/cups/cups/data/pseudo_label_dataset.py` — add `batch["boundary"]` precomputed from `sem_seg` (Pass 1 only if `BOUNDARY_WEIGHT>0`)

### Create (new code)
- `refs/cups/configs/train_cityscapes_dinov3_vitb_k80_depthpro_tau020.yaml` — **P0 baseline** (frozen ViT-B + k=80 semantic + DepthPro τ=0.20 instances)
- `refs/cups/cups/losses/__init__.py` — package init, re-exports
- `refs/cups/cups/losses/lovasz.py` — Lovász-Softmax (port of Berman 2018 reference impl)
- `refs/cups/cups/losses/boundary.py` — boundary-weighted CE + aux Sobel-BCE head
- `refs/cups/cups/losses/stego_adapter.py` — thin shim calling `mbps_pytorch.models.semantic.stego_loss`
- `refs/cups/cups/losses/depth_smoothness.py` — Sobel-based logit smoothness gated by depth edges
- `refs/cups/cups/losses/dense_affinity.py` — Gated-CRF + NeCo
- `refs/cups/configs/train_cityscapes_*_P1_loce.yaml` — derived from P0, enables LoCE
- `refs/cups/configs/train_cityscapes_*_P2_featmirror.yaml` — derived from best P0/P1
- `refs/cups/configs/train_cityscapes_*_P3_dglr.yaml` — derived from best P0/P1/P2
- `refs/cups/configs/train_cityscapes_*_P4_daff.yaml` — derived from best P0/P1/P2/P3
- `tests/losses/test_lovasz.py`, `test_boundary.py`, `test_stego_adapter.py`, `test_depth_smoothness.py`, `test_dense_affinity.py`
- `scripts/eval_stage2_passes.sh` — per-pass val eval, dumps `results/stage2_P{n}_{tag}/pq.json`

### Reuse (no edits)
- `mbps_pytorch/models/semantic/stego_loss.py` — `stego_loss(semantic_codes, dino_features, ...)` at line 56
- `mbps_pytorch/evaluate_semantic_pseudolabels.py` — PQ/SQ/RQ + per-class breakdown
- `refs/cups/cups/pl_model_pseudo.py:164` — `loss = sum(loss_dict.values())` auto-aggregates any new keys

---

## Pass 0 — Baseline Config + Aux-Loss Scaffold

**Why:** Need a reproducible frozen-backbone baseline on k=80 + DepthPro τ=0.20, and a loss-registry scaffold so P1–P4 add one line each instead of editing the head every time.

**Files:**
- Create: `refs/cups/configs/train_cityscapes_dinov3_vitb_k80_depthpro_tau020.yaml`
- Create: `refs/cups/cups/losses/__init__.py`
- Modify: `refs/cups/cups/config.py`
- Modify: `refs/cups/cups/model/modeling/roi_heads/semantic_seg.py` (lines 111–175 region)

### Task 0.1: Add config keys

**Files:** Modify `refs/cups/cups/config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/losses/test_config_keys.py
from refs.cups.cups.config import add_panoptic_cups_config
from detectron2.config import get_cfg

def test_sem_seg_head_aux_weights_exist():
    cfg = get_cfg()
    add_panoptic_cups_config(cfg)
    head = cfg.MODEL.SEM_SEG_HEAD
    assert head.LOSS_WEIGHT == 1.0
    assert head.LOVASZ_WEIGHT == 0.0
    assert head.BOUNDARY_WEIGHT == 0.0
    assert head.BOUNDARY_DILATE_PX == 3
    assert head.STEGO_WEIGHT == 0.0
    assert head.STEGO_TEMPERATURE == 0.1
    assert head.STEGO_KNN_K == 7
    assert head.DEPTH_SMOOTH_WEIGHT == 0.0
    assert head.DEPTH_SMOOTH_ALPHA == 10.0
    assert head.GATED_CRF_WEIGHT == 0.0
    assert head.NECO_WEIGHT == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/losses/test_config_keys.py -v`
Expected: FAIL — `AttributeError: 'CfgNode' object has no attribute 'LOVASZ_WEIGHT'`.

- [ ] **Step 3: Extend config**

Append to `refs/cups/cups/config.py` (inside `add_panoptic_cups_config` / `_C` setup):

```python
# Stage-2 semantic head aux losses ------------------------------------
_C.MODEL.SEM_SEG_HEAD = CN()  # may already exist from D2 default
_C.MODEL.SEM_SEG_HEAD.NAME = "CustomSemSegFPNHead"
_C.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0
_C.MODEL.SEM_SEG_HEAD.CLASS_WEIGHT = []           # optional per-class CE weight

# P1 — LoCE
_C.MODEL.SEM_SEG_HEAD.LOVASZ_WEIGHT = 0.0
_C.MODEL.SEM_SEG_HEAD.BOUNDARY_WEIGHT = 0.0       # aux BCE on Sobel edges
_C.MODEL.SEM_SEG_HEAD.BOUNDARY_DILATE_PX = 3      # CE up-weight band (px)
_C.MODEL.SEM_SEG_HEAD.BOUNDARY_CE_MULT = 2.0      # CE multiplier inside band

# P2 — FeatMirror
_C.MODEL.SEM_SEG_HEAD.STEGO_WEIGHT = 0.0
_C.MODEL.SEM_SEG_HEAD.STEGO_TEMPERATURE = 0.1
_C.MODEL.SEM_SEG_HEAD.STEGO_KNN_K = 7
_C.MODEL.SEM_SEG_HEAD.STEGO_FEATURE_SOURCE = "fpn_p2"  # {"fpn_p2", "vit_patch"}

# P3 — DGLR
_C.MODEL.SEM_SEG_HEAD.DEPTH_SMOOTH_WEIGHT = 0.0
_C.MODEL.SEM_SEG_HEAD.DEPTH_SMOOTH_ALPHA = 10.0    # exp(-alpha*|grad depth|)

# P4 — DAff
_C.MODEL.SEM_SEG_HEAD.GATED_CRF_WEIGHT = 0.0
_C.MODEL.SEM_SEG_HEAD.GATED_CRF_KERNEL = 5
_C.MODEL.SEM_SEG_HEAD.GATED_CRF_RGB_SIGMA = 0.1
_C.MODEL.SEM_SEG_HEAD.NECO_WEIGHT = 0.0
_C.MODEL.SEM_SEG_HEAD.NECO_K = 5
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/losses/test_config_keys.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add refs/cups/cups/config.py tests/losses/test_config_keys.py
git commit -m "feat(cups-config): add SEM_SEG_HEAD aux-loss weights scaffold"
```

### Task 0.2: Scaffold loss registry

**Files:** Create `refs/cups/cups/losses/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/losses/test_losses_registry.py
from refs.cups.cups.losses import build_aux_losses

def test_registry_returns_callables():
    reg = build_aux_losses()
    assert "lovasz_softmax" in reg
    assert "boundary_ce" in reg
    assert "stego_corr" in reg
    assert "depth_smoothness" in reg
    assert "gated_crf" in reg
    assert "neco" in reg
    for name, fn in reg.items():
        assert callable(fn), f"{name} is not callable"
```

- [ ] **Step 2: Run — expect import failure**

Run: `pytest tests/losses/test_losses_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'refs.cups.cups.losses'`.

- [ ] **Step 3: Create the package**

`refs/cups/cups/losses/__init__.py`:

```python
"""Stage-2 auxiliary semantic-head losses (P1-P4).

Each loss is a free function with a uniform signature:
    loss(logits, targets, ctx: dict) -> Tensor
where ctx may contain optional tensors (depth, dino_features, boundary,
rgb_image) used by specific losses.
"""
from __future__ import annotations
from typing import Callable, Dict

def build_aux_losses() -> Dict[str, Callable]:
    from .lovasz import lovasz_softmax
    from .boundary import boundary_ce
    from .stego_adapter import stego_corr
    from .depth_smoothness import depth_smoothness
    from .dense_affinity import gated_crf, neco
    return {
        "lovasz_softmax": lovasz_softmax,
        "boundary_ce": boundary_ce,
        "stego_corr": stego_corr,
        "depth_smoothness": depth_smoothness,
        "gated_crf": gated_crf,
        "neco": neco,
    }

__all__ = ["build_aux_losses"]
```

Write empty stub modules for each loss so imports succeed. Each stub contains:

```python
# refs/cups/cups/losses/lovasz.py   (and similar for the other 5)
import torch

def lovasz_softmax(logits: torch.Tensor, targets: torch.Tensor, ctx: dict) -> torch.Tensor:
    raise NotImplementedError("Implemented in Pass 1")
```

- [ ] **Step 4: Run — should pass**

Run: `pytest tests/losses/test_losses_registry.py -v`
Expected: PASS (stubs are callable even if they raise inside).

- [ ] **Step 5: Commit**

```bash
git add refs/cups/cups/losses/__init__.py refs/cups/cups/losses/*.py tests/losses/test_losses_registry.py
git commit -m "feat(cups-losses): add aux-loss package skeleton + registry"
```

### Task 0.3: Wire aggregation hook in `CustomSemSegFPNHead`

**Files:** Modify `refs/cups/cups/model/modeling/roi_heads/semantic_seg.py` at lines 111–175

- [ ] **Step 1: Write the failing test**

```python
# tests/losses/test_head_aggregation.py
import torch
from refs.cups.cups.model.modeling.roi_heads.semantic_seg import CustomSemSegFPNHead
from tests.losses.head_fixture import make_head_with_aux

def test_aggregates_all_aux_weights_zero():
    head = make_head_with_aux(num_classes=80, weights={})  # all zero
    logits = torch.randn(2, 80, 32, 64)
    targets = torch.randint(0, 80, (2, 128, 256))
    losses = head.losses(logits, targets, ctx={})
    assert set(losses.keys()) == {"loss_sem_seg"}

def test_aggregates_when_lovasz_enabled():
    head = make_head_with_aux(num_classes=80, weights={"LOVASZ_WEIGHT": 0.5})
    logits = torch.randn(2, 80, 32, 64, requires_grad=True)
    targets = torch.randint(0, 80, (2, 128, 256))
    losses = head.losses(logits, targets, ctx={})
    assert "loss_sem_seg" in losses
    assert "loss_lovasz" in losses
    assert losses["loss_lovasz"].requires_grad
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/losses/test_head_aggregation.py -v`
Expected: FAIL — `losses()` currently returns only `loss_sem_seg`; `ctx` kwarg not accepted.

- [ ] **Step 3: Extend the head**

Patch `refs/cups/cups/model/modeling/roi_heads/semantic_seg.py`:

```python
# In from_config (after reading LOSS_WEIGHT, CLASS_WEIGHT):
ret["aux_weights"] = {
    "lovasz":        cfg.MODEL.SEM_SEG_HEAD.LOVASZ_WEIGHT,
    "boundary":      cfg.MODEL.SEM_SEG_HEAD.BOUNDARY_WEIGHT,
    "stego":         cfg.MODEL.SEM_SEG_HEAD.STEGO_WEIGHT,
    "depth_smooth":  cfg.MODEL.SEM_SEG_HEAD.DEPTH_SMOOTH_WEIGHT,
    "gated_crf":     cfg.MODEL.SEM_SEG_HEAD.GATED_CRF_WEIGHT,
    "neco":          cfg.MODEL.SEM_SEG_HEAD.NECO_WEIGHT,
}
ret["aux_params"] = {
    "boundary_dilate_px":  cfg.MODEL.SEM_SEG_HEAD.BOUNDARY_DILATE_PX,
    "boundary_ce_mult":    cfg.MODEL.SEM_SEG_HEAD.BOUNDARY_CE_MULT,
    "stego_temperature":   cfg.MODEL.SEM_SEG_HEAD.STEGO_TEMPERATURE,
    "stego_knn_k":         cfg.MODEL.SEM_SEG_HEAD.STEGO_KNN_K,
    "stego_feature_source":cfg.MODEL.SEM_SEG_HEAD.STEGO_FEATURE_SOURCE,
    "depth_smooth_alpha":  cfg.MODEL.SEM_SEG_HEAD.DEPTH_SMOOTH_ALPHA,
    "gated_crf_kernel":    cfg.MODEL.SEM_SEG_HEAD.GATED_CRF_KERNEL,
    "gated_crf_rgb_sigma": cfg.MODEL.SEM_SEG_HEAD.GATED_CRF_RGB_SIGMA,
    "neco_k":              cfg.MODEL.SEM_SEG_HEAD.NECO_K,
}

# In __init__:
self.aux_weights = aux_weights
self.aux_params  = aux_params
from cups.losses import build_aux_losses
self._aux_fns = build_aux_losses()

# In forward — accept and store ctx:
def forward(self, features, targets=None, *, ctx=None):
    x = self.layers(features)
    if self.training:
        return None, self.losses(x, targets, ctx=ctx or {})
    x = F.interpolate(x, scale_factor=self.common_stride, mode="bilinear",
                      align_corners=False)
    return x, {}

# New losses() body:
def losses(self, predictions, targets, ctx=None):
    ctx = ctx or {}
    predictions_up = F.interpolate(predictions.float(),
                                    scale_factor=self.common_stride,
                                    mode="bilinear", align_corners=False)
    weight = (torch.tensor(self.class_weight, device=predictions.device,
                            dtype=predictions.dtype)
                if self.class_weight else None)
    ce = F.cross_entropy(predictions_up, targets,
                          reduction="mean",
                          ignore_index=self.ignore_value,
                          weight=weight)
    out = {"loss_sem_seg": ce * self.loss_weight}

    # Optional P1-P4 terms
    aux_ctx = {
        **ctx,
        "logits_up":  predictions_up,
        "logits_low": predictions,          # at common_stride resolution
        "targets":    targets,
        "class_weight": weight,
        "ignore_index": self.ignore_value,
        "params":     self.aux_params,
    }
    name_map = [
        ("lovasz",        "loss_lovasz"),
        ("boundary",      "loss_boundary"),
        ("stego",         "loss_stego"),
        ("depth_smooth",  "loss_depth_smooth"),
        ("gated_crf",     "loss_gated_crf"),
        ("neco",          "loss_neco"),
    ]
    fn_map = {
        "lovasz": "lovasz_softmax", "boundary": "boundary_ce",
        "stego": "stego_corr", "depth_smooth": "depth_smoothness",
        "gated_crf": "gated_crf", "neco": "neco",
    }
    for key, out_key in name_map:
        w = self.aux_weights.get(key, 0.0)
        if w <= 0.0:
            continue
        fn = self._aux_fns[fn_map[key]]
        out[out_key] = fn(predictions_up, targets, aux_ctx) * w
    return out
```

- [ ] **Step 4: Create `tests/losses/head_fixture.py`** — constructs a minimal `CustomSemSegFPNHead` using monkey-patched cfg object with override weights. Fixture must bypass FPN and drive `losses()` directly with supplied logits/targets.

- [ ] **Step 5: Run the test — Lovász sub-test will still fail because stub raises**

Run: `pytest tests/losses/test_head_aggregation.py::test_aggregates_all_aux_weights_zero -v`
Expected: PASS (only CE returned).

Run: `pytest tests/losses/test_head_aggregation.py::test_aggregates_when_lovasz_enabled -v`
Expected: FAIL — stub `NotImplementedError`. That unblocks in Pass 1.

- [ ] **Step 6: Thread `ctx` through caller**

Modify `refs/cups/cups/model/modeling/panoptic_fpn.py` (the call-site around line 125) so `sem_seg_head(features, gt_sem_seg)` becomes `sem_seg_head(features, gt_sem_seg, ctx=dict(depth=batched_inputs_depth, rgb=images.tensor, dino_features=backbone_fpn_p2))`. Each ctx item is optional — semantic head ignores missing keys.

- [ ] **Step 7: Commit**

```bash
git add refs/cups/cups/model/modeling/roi_heads/semantic_seg.py \
        refs/cups/cups/model/modeling/panoptic_fpn.py \
        tests/losses/test_head_aggregation.py tests/losses/head_fixture.py
git commit -m "feat(cups-head): aux-loss aggregation hook in CustomSemSegFPNHead"
```

### Task 0.4: Create P0 baseline config

**Files:** Create `refs/cups/configs/train_cityscapes_dinov3_vitb_k80_depthpro_tau020.yaml`

- [ ] **Step 1: Derive from existing frozen-backbone k=80 config**

Start from `refs/cups/configs/train_cityscapes_dinov3_vitb_k80_anydesk.yaml` (frozen — user feedback str-00022 says NEVER fine-tune with DoRA). Modify:

```yaml
_BASE_: train_cityscapes_dinov3_vitb_k80_anydesk.yaml
INPUT:
  DATA_DIR: /path/to/cityscapes
  PSEUDO_SEMANTIC_SUBDIR: pseudo_semantic_raw_k80
  PSEUDO_INSTANCE_SUBDIR: cups_pseudo_labels_depthpro   # DepthPro tau=0.20
  DEPTH_SUBDIR: depth_depthpro                          # keep depth in batch
MODEL:
  SEM_SEG_HEAD:
    NAME: CustomSemSegFPNHead
    LOSS_WEIGHT: 1.0
    # all aux weights default to 0 -- this is pure baseline
OUTPUT_DIR: results/stage2_P0_baseline
```

- [ ] **Step 2: Smoke-test one training step**

Run: `python refs/cups/train.py --config-file refs/cups/configs/train_cityscapes_dinov3_vitb_k80_depthpro_tau020.yaml SOLVER.MAX_ITER 1`
Expected: one iteration completes, loss dict contains only `loss_sem_seg`, `loss_cls`, `loss_box_reg`, `loss_mask` (no aux keys).

- [ ] **Step 3: Launch full P0 baseline training**

Run via `nohup` per project convention (user preference: `nohup python -u … > logs/stage2_P0.log 2>&1 &`). 8k iters, bs=16 A6000.

- [ ] **Step 4: Eval P0 checkpoint**

```bash
bash scripts/eval_stage2_passes.sh results/stage2_P0_baseline/model_final.pth > results/stage2_P0_baseline/eval.txt
```

Record: PQ / PQ_stuff / PQ_things / mIoU. This is the **reference number** all passes are compared to.

- [ ] **Step 5: Commit**

```bash
git add refs/cups/configs/train_cityscapes_dinov3_vitb_k80_depthpro_tau020.yaml \
        scripts/eval_stage2_passes.sh results/stage2_P0_baseline/eval.txt
git commit -m "feat(stage2): P0 baseline config + eval (k=80 + DepthPro tau=0.20)"
```

---

## Pass 1 — LoCE (Lovász-Softmax + Boundary-Weighted CE)

**Why:** CE optimises a surrogate of pixel accuracy, not IoU. Lovász-Softmax (Berman 2018, CVPR) is the differentiable convex surrogate of Jaccard — directly improves per-class IoU, which *is* the SQ/RQ component of PQ_stuff. Stacking with a boundary-weighted CE term (up-weighting CE near class boundaries + auxiliary BCE on Sobel edges) additionally sharpens the interfaces that dominate stuff PQ regressions (road/sidewalk, vegetation/building).

**Files:**
- Create: `refs/cups/cups/losses/lovasz.py`
- Create: `refs/cups/cups/losses/boundary.py`
- Modify: `refs/cups/cups/data/pseudo_label_dataset.py` (precompute `boundary` mask)
- Create: `refs/cups/configs/train_cityscapes_*_P1_loce.yaml`
- Create: `tests/losses/test_lovasz.py`, `tests/losses/test_boundary.py`

### Task 1.1: Lovász-Softmax port

- [ ] **Step 1: Failing test** — `tests/losses/test_lovasz.py`

```python
import torch
from refs.cups.cups.losses.lovasz import lovasz_softmax

def test_perfect_pred_zero_loss():
    C, H, W = 5, 16, 32
    targets = torch.randint(0, C, (2, H, W))
    logits = torch.full((2, C, H, W), -10.0)
    for b in range(2):
        for i, v in enumerate(targets[b].flatten()):
            logits[b, v, i // W, i % W] = 10.0
    ctx = {"ignore_index": 255}
    loss = lovasz_softmax(logits, targets, ctx)
    assert loss.item() < 1e-3

def test_gradient_flows():
    logits = torch.randn(2, 5, 16, 32, requires_grad=True)
    targets = torch.randint(0, 5, (2, 16, 32))
    loss = lovasz_softmax(logits, targets, {"ignore_index": 255})
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
```

- [ ] **Step 2: Run — expect FAIL** (`NotImplementedError`)

- [ ] **Step 3: Port Berman 2018 reference implementation** into `refs/cups/cups/losses/lovasz.py`. Signature:

```python
import torch
from torch.nn import functional as F

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = gt_sorted.sum()
    intersection = p - gt_sorted.cumsum(0)
    union = p + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def _lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor,
                         classes: str = "present") -> torch.Tensor:
    C = probas.size(1)
    losses = []
    class_iter = range(C) if classes == "all" else torch.unique(labels).tolist()
    for c in class_iter:
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        err = (fg - probas[:, c]).abs()
        errs_sorted, perm = torch.sort(err, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errs_sorted, _lovasz_grad(fg_sorted)))
    return torch.stack(losses).mean() if losses else probas.sum() * 0.0

def lovasz_softmax(logits: torch.Tensor, targets: torch.Tensor, ctx: dict) -> torch.Tensor:
    ignore = ctx.get("ignore_index", 255)
    probas = F.softmax(logits, dim=1)
    # (B, C, H, W) -> (B*H*W, C)
    B, C, H, W = probas.shape
    probas_flat = probas.permute(0, 2, 3, 1).reshape(-1, C)
    labels_flat = targets.reshape(-1)
    valid = labels_flat != ignore
    return _lovasz_softmax_flat(probas_flat[valid], labels_flat[valid])
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/losses/test_lovasz.py -v`

- [ ] **Step 5: Commit**

```bash
git add refs/cups/cups/losses/lovasz.py tests/losses/test_lovasz.py
git commit -m "feat(loss): Lovász-Softmax (P1)"
```

### Task 1.2: Boundary-weighted CE + Sobel BCE

- [ ] **Step 1: Failing test** — `tests/losses/test_boundary.py`

```python
import torch
from refs.cups.cups.losses.boundary import boundary_ce, compute_boundary_mask

def test_boundary_mask_binary_and_thin():
    targets = torch.zeros(1, 32, 32, dtype=torch.long)
    targets[0, :, 16:] = 1  # vertical split
    mask = compute_boundary_mask(targets, dilate_px=2)
    assert mask.dtype == torch.bool
    assert mask.sum() > 0
    assert mask.sum() < 32 * 32  # not the whole image

def test_boundary_ce_up_weights_boundary():
    logits = torch.randn(1, 2, 32, 32, requires_grad=True)
    targets = torch.zeros(1, 32, 32, dtype=torch.long); targets[0, :, 16:] = 1
    ctx = {"params": {"boundary_dilate_px": 2, "boundary_ce_mult": 4.0},
           "ignore_index": 255}
    loss = boundary_ce(logits, targets, ctx)
    loss.backward()
    assert loss.item() > 0
    assert logits.grad is not None
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `refs/cups/cups/losses/boundary.py`**

```python
import torch
from torch.nn import functional as F

_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1,1,3,3)
_SOBEL_Y = _SOBEL_X.transpose(-1, -2)

def compute_boundary_mask(targets: torch.Tensor, dilate_px: int = 3) -> torch.Tensor:
    # Label-change detection via max-pooling trick over one-hot.
    t = targets.unsqueeze(1).float()
    # Max-pool then compare — if any neighbour differs, flag as boundary.
    k = 2 * dilate_px + 1
    eq = F.max_pool2d(t, kernel_size=k, stride=1, padding=dilate_px) != \
         -F.max_pool2d(-t, kernel_size=k, stride=1, padding=dilate_px)
    return eq.squeeze(1)

def boundary_ce(logits: torch.Tensor, targets: torch.Tensor, ctx: dict) -> torch.Tensor:
    params = ctx["params"]
    ignore = ctx.get("ignore_index", 255)
    dilate = int(params["boundary_dilate_px"])
    mult   = float(params["boundary_ce_mult"])
    cw     = ctx.get("class_weight", None)

    # Pixelwise CE, then up-weight boundary band
    ce_px = F.cross_entropy(logits, targets, reduction="none",
                              ignore_index=ignore, weight=cw)  # (B,H,W)
    mask = compute_boundary_mask(targets, dilate).float()
    weighted = ce_px * (1.0 + (mult - 1.0) * mask)
    valid = (targets != ignore).float()
    return (weighted * valid).sum() / valid.sum().clamp_min(1.0)
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/losses/test_boundary.py -v`

- [ ] **Step 5: Commit**

```bash
git add refs/cups/cups/losses/boundary.py tests/losses/test_boundary.py
git commit -m "feat(loss): boundary-weighted CE (P1)"
```

### Task 1.3: Pass-1 config + smoke + train + eval

- [ ] **Step 1: Create `refs/cups/configs/train_cityscapes_dinov3_vitb_k80_depthpro_tau020_P1_loce.yaml`**

```yaml
_BASE_: train_cityscapes_dinov3_vitb_k80_depthpro_tau020.yaml
MODEL:
  SEM_SEG_HEAD:
    LOVASZ_WEIGHT: 0.5          # matched to CE scale; tune
    BOUNDARY_WEIGHT: 0.3
    BOUNDARY_DILATE_PX: 3
    BOUNDARY_CE_MULT: 2.0
OUTPUT_DIR: results/stage2_P1_loce
```

- [ ] **Step 2: Smoke test — 1 iter**

Run: `python refs/cups/train.py --config-file <P1.yaml> SOLVER.MAX_ITER 1`
Expected: `loss_dict` now has `loss_sem_seg`, `loss_lovasz`, `loss_boundary` (plus instance losses).

- [ ] **Step 3: Full training run** (nohup, 8k iters). Log to `logs/stage2_P1_loce.log`.

- [ ] **Step 4: Eval vs P0**

```bash
bash scripts/eval_stage2_passes.sh results/stage2_P1_loce/model_final.pth > results/stage2_P1_loce/eval.txt
python - <<'PY'
import json; a=json.load(open("results/stage2_P0_baseline/pq.json")); b=json.load(open("results/stage2_P1_loce/pq.json"))
print("delta PQ_stuff:", b["pq_stuff"] - a["pq_stuff"])
PY
```

Gate: if ΔPQ_stuff < +0.5 → rollback by setting P1 weights back to 0 and note failure in `reports/stage2_loss_ablation.md`. If ≥ +0.5 → proceed to P2.

- [ ] **Step 5: Commit**

```bash
git add refs/cups/configs/train_cityscapes_*P1_loce.yaml results/stage2_P1_loce/
git commit -m "feat(stage2): P1 LoCE (Lovász + boundary CE) config + eval"
```

---

## Pass 2 — FeatMirror (STEGO DINOv3 Correspondence)

**Why:** Frozen DINOv3 patch features carry strong unsupervised semantic structure (~45–55% mIoU via KNN clustering). STEGO distils that structure into the predicted semantic logits: anchor logit should be close (in cosine) to the logit of its top-k DINOv3 neighbours. This is complementary to CE (which follows the noisy pseudo-labels) — the student learns to respect DINOv3 correspondences even where pseudo-labels err. The loss function is already implemented in `mbps_pytorch/models/semantic/stego_loss.py` (286 lines, tested implicitly in `train_depth_adapter.py`).

**Files:**
- Create: `refs/cups/cups/losses/stego_adapter.py`
- Modify: `refs/cups/cups/model/modeling/panoptic_fpn.py` — pass FPN-P2 (or pre-FPN ViT features) into `ctx["dino_features"]`
- Create: `refs/cups/configs/train_cityscapes_*_P2_featmirror.yaml`
- Create: `tests/losses/test_stego_adapter.py`

### Task 2.1: STEGO shim

- [ ] **Step 1: Failing test** — `tests/losses/test_stego_adapter.py`

```python
import torch
from refs.cups.cups.losses.stego_adapter import stego_corr

def test_requires_dino_features_in_ctx():
    logits = torch.randn(2, 5, 16, 32)
    targets = torch.randint(0, 5, (2, 16, 32))
    try:
        stego_corr(logits, targets, {"params": {"stego_temperature": 0.1, "stego_knn_k": 7}})
        assert False, "should raise KeyError"
    except KeyError as e:
        assert "dino_features" in str(e)

def test_returns_scalar_with_grad():
    B, N, D = 2, 64, 128
    logits = torch.randn(B, 5, 8, 8, requires_grad=True)
    targets = torch.randint(0, 5, (B, 8, 8))
    dino = torch.randn(B, N, D)
    ctx = {"dino_features": dino,
           "params": {"stego_temperature": 0.1, "stego_knn_k": 4,
                      "stego_feature_source": "fpn_p2"}}
    loss = stego_corr(logits, targets, ctx)
    loss.backward()
    assert loss.dim() == 0
    assert logits.grad is not None
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement shim** in `refs/cups/cups/losses/stego_adapter.py`

```python
import torch
import torch.nn.functional as F
from mbps_pytorch.models.semantic.stego_loss import stego_loss as _stego

def stego_corr(logits: torch.Tensor, targets: torch.Tensor, ctx: dict) -> torch.Tensor:
    if "dino_features" not in ctx:
        raise KeyError("stego_corr requires ctx['dino_features']")
    dino = ctx["dino_features"]          # (B, Ns, Ds) or (B, Ds, Hs, Ws)
    if dino.dim() == 4:
        B, D, Hs, Ws = dino.shape
        dino = dino.permute(0, 2, 3, 1).reshape(B, Hs*Ws, D)

    # Downsample logits to DINO spatial resolution to pair patches
    B, C, H, W = logits.shape
    Hs = Ws = int(dino.shape[1] ** 0.5) if dino.shape[1] == int(dino.shape[1] ** 0.5) ** 2 \
              else None
    if Hs is None:
        # general aspect ratio: recover from ctx
        Hs, Ws = ctx["params"].get("stego_patch_grid", (H, W))
    codes = F.adaptive_avg_pool2d(logits, output_size=(Hs, Ws))
    codes = codes.permute(0, 2, 3, 1).reshape(B, Hs*Ws, C)

    return _stego(
        semantic_codes=codes,
        dino_features=dino,
        temperature=ctx["params"]["stego_temperature"],
        knn_k=ctx["params"]["stego_knn_k"],
    )
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add refs/cups/cups/losses/stego_adapter.py tests/losses/test_stego_adapter.py
git commit -m "feat(loss): STEGO correspondence shim (P2)"
```

### Task 2.2: Expose DINOv3 features to the semantic head

**Context:** `DINOv3Backbone` currently returns only FPN levels. We need patch features at stride-16 (or FPN-P2 at stride-4, configurable via `STEGO_FEATURE_SOURCE`).

- [ ] **Step 1: Extend backbone wrapper**

Modify `refs/cups/cups/model/modeling/backbone/dinov3_backbone.py` (or the current CUPS FPN wrapper) so `forward(images)` returns a dict with an extra `"vit_patch"` key holding `(B, D, H/16, W/16)` pre-FPN tokens. This is additional output — no other callers affected.

- [ ] **Step 2: Thread through `PanopticFPN.forward`**

```python
# refs/cups/cups/model/modeling/panoptic_fpn.py
features = self.backbone(images.tensor)              # dict with FPN + vit_patch
vit_patch = features.pop("vit_patch", None)
ctx = {"depth": batched_depth, "rgb": images.tensor,
       "dino_features": vit_patch if self.sem_seg_head.aux_params
                          ["stego_feature_source"] == "vit_patch"
                          else features["p2"]}
sem_seg_results, sem_seg_losses = self.sem_seg_head(features, gt_sem_seg, ctx=ctx)
```

- [ ] **Step 3: Smoke — 1 iter with P2 config (Lovász+Boundary still on, STEGO weight=0.1)**

- [ ] **Step 4: Commit**

```bash
git add refs/cups/cups/model/modeling/backbone/dinov3_backbone.py \
        refs/cups/cups/model/modeling/panoptic_fpn.py
git commit -m "feat(cups-bb): expose vit_patch features for STEGO aux loss"
```

### Task 2.3: P2 config + train + eval + gate

- [ ] **Step 1: Config `train_cityscapes_*_P2_featmirror.yaml`**

```yaml
_BASE_: train_cityscapes_dinov3_vitb_k80_depthpro_tau020_P1_loce.yaml
MODEL:
  SEM_SEG_HEAD:
    STEGO_WEIGHT: 0.1
    STEGO_TEMPERATURE: 0.1
    STEGO_KNN_K: 7
    STEGO_FEATURE_SOURCE: "fpn_p2"     # start conservative; try vit_patch later
OUTPUT_DIR: results/stage2_P2_featmirror
```

- [ ] **Step 2: 1-iter smoke → full train → eval**

- [ ] **Step 3: Gate** ΔPQ_stuff ≥ +0.5 vs best(P0,P1). Rollback weight to 0 if fails.

- [ ] **Step 4: Commit**

```bash
git add refs/cups/configs/train_cityscapes_*P2_featmirror.yaml results/stage2_P2_featmirror/
git commit -m "feat(stage2): P2 FeatMirror (STEGO) config + eval"
```

---

## Pass 3 — DGLR (Depth-Guided Logit Regularizer)

**Why:** Semantic label changes should correlate with depth discontinuities — flat walls, roads, and sky are smooth in depth *and* smooth in semantics. Penalising `‖∇logits‖ · exp(−α‖∇depth‖)` encodes this as a soft prior. Depth is **already in the Stage-2 batch** (`pseudo_label_dataset.py:220`, `batch["depth"]` shape `(1,H,W)`), and the semantic head `forward` already accepts a `depth` kwarg (line 137). Compounds with the DCFA adapter paper story (memory: depth-semantic ablation COMPLETE, DCFA lp=20 +6.24 mIoU).

**Files:**
- Create: `refs/cups/cups/losses/depth_smoothness.py`
- Modify: `refs/cups/cups/model/modeling/panoptic_fpn.py` — ensure `depth` forwarded into `ctx["depth"]`
- Create: `refs/cups/configs/train_cityscapes_*_P3_dglr.yaml`
- Create: `tests/losses/test_depth_smoothness.py`

### Task 3.1: Depth-smoothness loss

- [ ] **Step 1: Failing test** — `tests/losses/test_depth_smoothness.py`

```python
import torch
from refs.cups.cups.losses.depth_smoothness import depth_smoothness

def test_zero_when_depth_has_uniform_gradient_matching_logits():
    logits = torch.zeros(1, 5, 16, 32, requires_grad=True)
    depth = torch.linspace(0, 1, 32).view(1,1,1,32).expand(1,1,16,32).clone()
    ctx = {"depth": depth, "params": {"depth_smooth_alpha": 10.0}}
    loss = depth_smoothness(logits, None, ctx)
    assert loss.item() < 1e-5

def test_penalises_logit_jumps_on_smooth_depth():
    depth = torch.zeros(1, 1, 16, 32)
    logits = torch.zeros(1, 5, 16, 32, requires_grad=True)
    logits.data[0, 0, :, :16] = 10.0  # big jump in logits, flat depth
    ctx = {"depth": depth, "params": {"depth_smooth_alpha": 10.0}}
    loss = depth_smoothness(logits, None, ctx)
    loss.backward()
    assert loss.item() > 0
    assert logits.grad is not None
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `refs/cups/cups/losses/depth_smoothness.py`**

```python
import torch
from torch.nn import functional as F

def _sobel(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                       dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    C = x.shape[1]
    kx = kx.expand(C, 1, 3, 3); ky = ky.expand(C, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return gx, gy

def depth_smoothness(logits: torch.Tensor, targets, ctx: dict) -> torch.Tensor:
    depth = ctx["depth"]                       # (B, 1, H, W) or (B, H, W)
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    alpha = float(ctx["params"]["depth_smooth_alpha"])
    B, _, H, W = logits.shape
    # Align depth to logit resolution
    if depth.shape[-2:] != (H, W):
        depth = F.interpolate(depth.float(), size=(H, W),
                               mode="bilinear", align_corners=False)

    lg_x, lg_y = _sobel(logits)               # (B, C, H, W)
    d_x,  d_y  = _sobel(depth)                # (B, 1, H, W)

    # depth-edge gating: small where depth is smooth
    w = torch.exp(-alpha * (d_x.abs() + d_y.abs()))   # (B, 1, H, W)
    lg_mag = (lg_x.abs() + lg_y.abs()).mean(dim=1, keepdim=True)  # (B,1,H,W)
    return (w * lg_mag).mean()
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/losses/test_depth_smoothness.py -v`

- [ ] **Step 5: Commit**

```bash
git add refs/cups/cups/losses/depth_smoothness.py tests/losses/test_depth_smoothness.py
git commit -m "feat(loss): depth-guided logit regularizer (P3)"
```

### Task 3.2: P3 config + train + eval + gate

- [ ] **Step 1: Config `train_cityscapes_*_P3_dglr.yaml`**

```yaml
_BASE_: <best-of-P0_or_P1_or_P2>.yaml
MODEL:
  SEM_SEG_HEAD:
    DEPTH_SMOOTH_WEIGHT: 0.1       # tune in {0.05, 0.1, 0.2}
    DEPTH_SMOOTH_ALPHA: 10.0
OUTPUT_DIR: results/stage2_P3_dglr
```

- [ ] **Step 2: 1-iter smoke (must see `loss_depth_smooth` in log)**

- [ ] **Step 3: Full train + eval + gate** ΔPQ_stuff ≥ +0.5 vs best(P0–P2).

- [ ] **Step 4: Commit**

```bash
git add refs/cups/configs/train_cityscapes_*P3_dglr.yaml results/stage2_P3_dglr/
git commit -m "feat(stage2): P3 DGLR config + eval"
```

---

## Pass 4 — DAff (Dense-Affinity: Gated-CRF + NeCo)

**Why:** Gated-CRF (Obukhov 2019) adds dense pairwise compatibility between pixels similar in RGB + position — a differentiable smoothness prior that fires strongly inside uniform stuff regions without requiring test-time CRF. NeCo (Neighbourhood-Consistency, ICCV 2023) adds a k-NN consistency term over DINOv3 patch features, which targets the **same** structural inductive bias as STEGO but at the neighbourhood level rather than pairwise anchor–positive. Memory item `str-00014` explicitly flags NAMR+NeCo as the closure path vs Falcon. Deferred to P4 because both are heavier than P1–P3 and their marginal gains are best measured on top of whatever P1–P3 leave on the table.

**Files:**
- Create: `refs/cups/cups/losses/dense_affinity.py`
- Create: `refs/cups/configs/train_cityscapes_*_P4_daff.yaml`
- Create: `tests/losses/test_dense_affinity.py`

### Task 4.1: Gated-CRF loss

- [ ] **Step 1: Failing test** — `tests/losses/test_dense_affinity.py`

```python
import torch
from refs.cups.cups.losses.dense_affinity import gated_crf, neco

def test_gated_crf_scalar():
    logits = torch.randn(2, 5, 32, 64, requires_grad=True)
    rgb = torch.randn(2, 3, 32, 64)
    ctx = {"rgb": rgb, "params": {"gated_crf_kernel": 5,
                                   "gated_crf_rgb_sigma": 0.1}}
    loss = gated_crf(logits, None, ctx)
    loss.backward()
    assert loss.dim() == 0
    assert logits.grad is not None

def test_neco_scalar():
    logits = torch.randn(2, 5, 16, 32, requires_grad=True)
    dino = torch.randn(2, 16*32, 128)
    ctx = {"dino_features": dino, "params": {"neco_k": 5}}
    loss = neco(logits, None, ctx)
    loss.backward()
    assert loss.dim() == 0
    assert logits.grad is not None
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `refs/cups/cups/losses/dense_affinity.py`**

```python
import torch
from torch.nn import functional as F

def gated_crf(logits: torch.Tensor, targets, ctx: dict) -> torch.Tensor:
    """Gated-CRF pairwise loss (Obukhov 2019, simplified).

    Pairwise affinity w_ij = exp(-||rgb_i - rgb_j||^2 / 2 sigma_rgb^2) *
                             exp(-||pos_i - pos_j||^2 / 2 sigma_pos^2)
    Loss = sum_ij w_ij * (1 - <p_i, p_j>)
    Approximated via local window K (default 5) for tractability.
    """
    rgb = ctx["rgb"]                                  # (B, 3, H, W)
    K   = int(ctx["params"]["gated_crf_kernel"])
    srgb = float(ctx["params"]["gated_crf_rgb_sigma"])
    B, C, H, W = logits.shape
    if rgb.shape[-2:] != (H, W):
        rgb = F.interpolate(rgb, size=(H, W), mode="bilinear", align_corners=False)

    probs = F.softmax(logits, dim=1)                  # (B, C, H, W)
    # unfold local patches
    unfold = lambda x: F.unfold(x, kernel_size=K, padding=K // 2)  # (B, C*K*K, H*W)
    p_u = unfold(probs).view(B, C, K*K, H*W)          # (B, C, K^2, HW)
    r_u = unfold(rgb).view(B, 3, K*K, H*W)            # (B, 3, K^2, HW)

    p_center = probs.view(B, C, 1, H*W)
    r_center = rgb.view(B, 3, 1, H*W)
    rdiff = (r_u - r_center).pow(2).sum(dim=1)        # (B, K^2, HW)
    w = torch.exp(-rdiff / (2 * srgb ** 2))            # (B, K^2, HW)
    dot = (p_u * p_center).sum(dim=1)                  # (B, K^2, HW)
    # exclude centre itself
    mid = K * K // 2
    w[:, mid, :] = 0.0
    return (w * (1.0 - dot)).mean()

def neco(logits: torch.Tensor, targets, ctx: dict) -> torch.Tensor:
    """Neighbourhood-Consistency on DINO patch features.

    For each patch, find top-k DINO neighbours and pull semantic logits
    closer in cosine space.  Similar to STEGO but neighbourhood-averaged.
    """
    dino = ctx["dino_features"]                       # (B, N, D) or (B, D, H, W)
    k = int(ctx["params"]["neco_k"])
    B, C, H, W = logits.shape
    if dino.dim() == 4:
        Bd, Dd, Hd, Wd = dino.shape
        dino = dino.permute(0, 2, 3, 1).reshape(Bd, Hd*Wd, Dd)
    codes = F.adaptive_avg_pool2d(logits, (int(dino.shape[1] ** 0.5),
                                             int(dino.shape[1] ** 0.5)))
    codes = codes.flatten(2).transpose(1, 2)           # (B, N, C)
    codes = F.normalize(codes, dim=-1)
    dn = F.normalize(dino, dim=-1)

    losses = []
    for b in range(B):
        sim = dn[b] @ dn[b].T
        sim.fill_diagonal_(-1e9)
        _, idx = sim.topk(k, dim=-1)                   # (N, k)
        neigh = codes[b][idx]                          # (N, k, C)
        anchor = codes[b].unsqueeze(1)                  # (N, 1, C)
        losses.append((1.0 - (anchor * neigh).sum(-1)).mean())
    return torch.stack(losses).mean()
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/losses/test_dense_affinity.py -v`

- [ ] **Step 5: Commit**

```bash
git add refs/cups/cups/losses/dense_affinity.py tests/losses/test_dense_affinity.py
git commit -m "feat(loss): Gated-CRF + NeCo (P4)"
```

### Task 4.2: P4 config + train + eval + gate

- [ ] **Step 1: Config `train_cityscapes_*_P4_daff.yaml`**

```yaml
_BASE_: <best-of-P0..P3>.yaml
MODEL:
  SEM_SEG_HEAD:
    GATED_CRF_WEIGHT: 0.1
    GATED_CRF_KERNEL: 5
    GATED_CRF_RGB_SIGMA: 0.1
    NECO_WEIGHT: 0.1
    NECO_K: 5
OUTPUT_DIR: results/stage2_P4_daff
```

- [ ] **Step 2: 1-iter smoke (must see `loss_gated_crf`, `loss_neco`)**

- [ ] **Step 3: Full train + eval** (P4 heavy, monitor step time; if >2× P0, reduce `gated_crf_kernel` from 5 → 3).

- [ ] **Step 4: Gate** ΔPQ_stuff ≥ +0.5 vs best(P0–P3). Also run ablation `GATED_CRF_WEIGHT=0.1, NECO_WEIGHT=0.0` and vice-versa (two extra checkpoints) to attribute gain.

- [ ] **Step 5: Commit**

```bash
git add refs/cups/configs/train_cityscapes_*P4_daff.yaml results/stage2_P4_daff/
git commit -m "feat(stage2): P4 DAff (Gated-CRF + NeCo) config + eval"
```

---

## Verification (end-to-end)

1. **Unit tests**

```bash
pytest tests/losses/ -v
```

Expected: 12+ tests pass (config keys, registry, head aggregation, lovasz×2, boundary×2, stego×2, depth_smoothness×2, dense_affinity×2).

2. **Smoke tests per pass**

```bash
for P in P0 P1 P2 P3 P4; do
  python refs/cups/train.py --config-file refs/cups/configs/train_cityscapes_*${P}*.yaml SOLVER.MAX_ITER 1
done
```

Expected: each prints its extra `loss_*` keys in the iter-0 log line.

3. **Full ablation table** (written to `reports/stage2_loss_ablation.md`)

| Pass | Added loss(es) | PQ | PQ_stuff | PQ_things | mIoU | Δ vs prior-best |
|------|---------------|----|----|----|----|----|
| P0 | CE only | … | … | … | … | — |
| P1 | +Lovász +Boundary-CE | … | … | … | … | … |
| P2 | +STEGO | … | … | … | … | … |
| P3 | +Depth-smoothness | … | … | … | … | … |
| P4 | +Gated-CRF +NeCo | … | … | … | … | … |

4. **Memory / CCR updates after each pass**

After each pass that passes its gate, commit to CCR with the `experiment=` field populated (id=`stage2-P{n}`, metrics, conclusion). If a pass fails its gate, still commit — record the failure as a pattern.

5. **Release artefacts**

Per project memory (`a6000_pseudolabel_divergence.md`): release the P0 pseudo-labels + centroids on HF Hub alongside the best final checkpoint so the full pipeline is reproducible.

---

## Reused Existing Code

| Code | Path | Role |
|------|------|------|
| `stego_loss(semantic_codes, dino_features, ...)` | `mbps_pytorch/models/semantic/stego_loss.py:56` | Core STEGO (Pass 2) |
| `depth_guided_correlation_loss` | `mbps_pytorch/models/semantic/stego_loss.py:116` | Reference pattern for DGLR |
| `CustomSemSegFPNHead.forward/losses` | `refs/cups/cups/model/modeling/roi_heads/semantic_seg.py:132–175` | Host for all aux terms |
| `loss = sum(loss_dict.values())` | `refs/cups/cups/pl_model_pseudo.py:164` | Auto-aggregates new keys |
| `batch["depth"]` (1, H, W) normalised | `refs/cups/cups/data/pseudo_label_dataset.py:220` | P3 depth input |
| `evaluate_from_files(pred_dir, gt_dir)` | `mbps_pytorch/evaluate_semantic_pseudolabels.py` | PQ / mIoU aggregation |
| `DepthEncoder` Sobel code | `refs/cups/cups/model/modeling/roi_heads/depth_film_semantic_head.py:60–91` | Sobel template for DGLR |

## Failure Modes to Watch For

- **Loss balancing drift.** If any aux loss grows >5× `loss_sem_seg` during training, the model is optimising the aux signal at the expense of the pseudo-labels. Halve that weight and restart.
- **Gate false-positive at early steps.** PQ on val should be measured only at the final checkpoint, not intermediate. Pseudo-label noise makes earlier checkpoints unreliable.
- **STEGO on FPN-P2 giving ≈0 signal.** FPN-P2 is already semantically specialised — the correspondence is too weak. Switch `STEGO_FEATURE_SOURCE` to `vit_patch` (stride-16) before concluding P2 fails.
- **DGLR hurting small things.** Depth is noisy on thin structures (poles, pedestrians). If `PQ_things` drops ≥ 0.5 at P3 keep-gate, try `DEPTH_SMOOTH_ALPHA` ∈ {5, 15, 20} to re-gate the edge mask.
- **Gated-CRF OOM on bs=16.** Fall back to local 3×3 kernel or reduce bs to 8 for P4 only.
