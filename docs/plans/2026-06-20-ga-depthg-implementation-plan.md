# GA-DepthG (Track 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DepthG's scalar depth-correlation loss term with a gravity-aligned (height + surface-normal) affinity in `refs/depthg`, and run the scalar→height→normal→both ablation on DepthG's own DINO ViT-B/8 footing.

**Architecture:** DepthG's loss term `dd = depth_correlation(norm(d1),norm(d2))` (`src/modules.py:1265`) is swapped for `A = wₙ(n·n) + w_h(ĥ·ĥ)` computed by the same einsum on a 4-channel `[ĥ, nx, ny, nz]` geometry tensor. Geometry is precomputed/cached from DepthPro depth + Cityscapes intrinsics (RANSAC ground plane, per-scene-standardized height, unit normals). Depth/geometry are train-only; inference and eval are unchanged from DepthG.

**Tech Stack:** PyTorch + PyTorch-Lightning (STEGO/DepthG era, pinned in `refs/depthg/requirements.txt`), Hydra configs, frozen DINO ViT-B/8, NumPy/SciPy/sklearn for geometry.

## Global Constraints
- All four ablation runs use the SAME depth source: **DepthPro** (`/Volumes/code_files/datasets/cityscapes/depth_depthpro`). DepthG's published 23.1 (ZoeDepth) is an EXTERNAL reference only.
- Backbone frozen **DINO ViT-B/8**, `model_type=vit_base`, `dim=100`, `dataset_name=cityscapes`, `depth_sampling=none` (random sampling — not a confound).
- Geometry tensor channels: `[0]=ĥ` per-scene-standardized `(h-median_ground)/scale_90`; `[1:4]=unit normal`. Re-normalize normal channels after any bilinear interpolation.
- Ground plane fit on the **FULL image (pre-crop)**, then apply DepthG's crop to the height/normal maps.
- Train on remote GPU (santosh/A6000) in `.venv_depthg`; geometry precompute + eval run **local** (Mac).
- Eval = k-means cluster → CRF → Hungarian, **27-class** Cityscapes (identical to DepthG/CUPS).
- Novelty claim is role+setting only (label-free affinity, monocular, frozen DINO, multi-class) — never the geometric quantities (HHA 2014 owns them). See `memory/novelty_geometric_affinity.md`.

---

### Task 0: Env + baseline protocol lock

**Files:** none (setup + verification).

- [ ] **Step 1: Create the DepthG env**

Run: `cd refs/depthg && python3 -m venv ../../.venv_depthg && ../../.venv_depthg/bin/pip install -r requirements.txt`
Expected: install completes (pinned torch/lightning resolve).

- [ ] **Step 2: Download DINO ViT-B/8 backbone**

Run: `../../.venv_depthg/bin/python src/download_models.py`
Expected: DINO ViT-B/8 weights present under `saved_models/`.

- [ ] **Step 3: Eval the reproduced baseline checkpoint to lock the protocol**

Run: `../../.venv_depthg/bin/python src/eval_segmentation.py model_paths=["saved_models/cityscapes_vitb.ckpt"] dataset_name=cityscapes run_crf=True`
Expected: cluster+CRF+Hungarian mIoU ≈ **23.09** (±0.3). Record the exact number — this is the protocol anchor.

- [ ] **Step 4: Commit the recorded baseline**

```bash
git add docs/plans/2026-06-20-ga-depthg-implementation-plan.md
git commit -m "chore(ga-depthg): lock DepthG baseline protocol (mIoU=<recorded>)"
```

---

### Task 1: Geometry feature module

**Files:**
- Create: `refs/depthg/src/geometry_features.py`
- Test: `refs/depthg/tests/test_geometry_features.py`

**Interfaces:**
- Produces: `compute_geometry(depth_inv: np.ndarray, fx,fy,u0,v0,cam_h: float) -> np.ndarray` returning `(4,H,W)` float32 `[ĥ, nx, ny, nz]`; `load_geometry_for(stem,city,split) -> (4,H,W)`.

- [ ] **Step 1: Copy the validated geometry primitives**

Copy verbatim into `geometry_features.py` these functions from the already-validated `mbps_pytorch/premise_check_geometry_affinity.py`: `back_project`, `fit_ground_plane`, `surface_normals` (they passed the premise self-check and the 150-image run). Keep their exact bodies.

- [ ] **Step 2: Write the failing test**

```python
# refs/depthg/tests/test_geometry_features.py
import numpy as np
from src.geometry_features import compute_geometry

def test_flat_ground_height_and_normal():
    H, W = 64, 128
    # synthetic: depth increases with row (ground receding); inverse-depth in (0,1]
    inv = np.linspace(0.9, 0.05, H)[:, None].repeat(W, 1).astype(np.float32)
    g = compute_geometry(inv, fx=500, fy=500, u0=W/2, v0=H/2, cam_h=1.22)
    assert g.shape == (4, H, W)
    n = g[1:4]
    assert abs(np.linalg.norm(n, axis=0).mean() - 1.0) < 1e-2      # unit normals
    assert abs(np.median(g[0])) < 0.5                              # ĥ standardized ~0 median
```

- [ ] **Step 3: Run test to verify it fails**

Run: `../../.venv_depthg/bin/python -m pytest tests/test_geometry_features.py -v`
Expected: FAIL ("cannot import name 'compute_geometry'").

- [ ] **Step 4: Implement `compute_geometry`**

```python
def compute_geometry(depth_inv, fx, fy, u0, v0, cam_h):
    pts, valid = back_project(depth_inv, fx, fy, u0, v0)
    H, W, _ = pts.shape
    region = np.zeros((H, W), bool)
    region[int(H*0.62):, int(W*0.2):int(W*0.8)] = True
    region &= valid
    plane = fit_ground_plane(pts, region)
    if plane is None:
        n_g = np.array([0, 1, 0.0]); d = 0.0; scale = 1.0
    else:
        n_g, d = plane; scale = cam_h / (abs(d) + 1e-9)
    height = (pts @ n_g + d) * scale
    if height[:int(H*0.4)][valid[:int(H*0.4)]].mean() < 0:
        height = -height; n_g = -n_g
    normal = surface_normals(pts * scale, valid)                  # (H,W,3) unit
    med = np.median(height[valid]) if valid.any() else 0.0
    s90 = np.percentile(np.abs(height[valid] - med), 90) + 1e-6 if valid.any() else 1.0
    h_hat = (height - med) / s90                                  # per-scene standardized
    g = np.concatenate([h_hat[None], normal.transpose(2, 0, 1)], 0).astype(np.float32)
    g[:, ~valid] = 0.0
    return g
```

- [ ] **Step 5: Run test to verify it passes**

Run: `../../.venv_depthg/bin/python -m pytest tests/test_geometry_features.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add refs/depthg/src/geometry_features.py refs/depthg/tests/test_geometry_features.py
git commit -m "feat(ga-depthg): geometry feature module (height+normal from DepthPro)"
```

---

### Task 2: Geometric affinity op

**Files:**
- Modify: `refs/depthg/src/modules.py` (add function after `depth_correlation`, ~line 815)
- Test: `refs/depthg/tests/test_geometric_affinity.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `geometric_affinity(g1, g2, mode: str, w_h: float, w_n: float) -> Tensor` of shape `(n,h,w,h,w)`. `g1,g2` are `(n,4,h,w)` with channel 0 = ĥ, channels 1:4 = unit normal. `mode ∈ {'height','normal','both'}`.

- [ ] **Step 1: Write the failing test**

```python
# refs/depthg/tests/test_geometric_affinity.py
import torch
from src.modules import geometric_affinity

def test_normal_cosine_extremes():
    n = 1
    up = torch.tensor([0., 0., 1.]).reshape(1, 3, 1, 1)
    side = torch.tensor([1., 0., 0.]).reshape(1, 3, 1, 1)
    g_up = torch.cat([torch.zeros(1, 1, 1, 1), up], 1)
    g_side = torch.cat([torch.zeros(1, 1, 1, 1), side], 1)
    assert torch.isclose(geometric_affinity(g_up, g_up, 'normal', .4, .6)[0, 0, 0, 0, 0], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(geometric_affinity(g_up, g_side, 'normal', .4, .6)[0, 0, 0, 0, 0], torch.tensor(0.0), atol=1e-5)

def test_both_is_weighted_sum():
    g = torch.randn(1, 4, 2, 2); g[:, 1:] = torch.nn.functional.normalize(g[:, 1:], dim=1)
    h = geometric_affinity(g, g, 'height', .4, .6)
    nrm = geometric_affinity(g, g, 'normal', .4, .6)
    both = geometric_affinity(g, g, 'both', .4, .6)
    assert torch.allclose(both, .6 * nrm + .4 * h, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `../../.venv_depthg/bin/python -m pytest tests/test_geometric_affinity.py -v`
Expected: FAIL ("cannot import name 'geometric_affinity'").

- [ ] **Step 3: Implement (insert after `depth_correlation` at modules.py:815)**

```python
def geometric_affinity(g1, g2, mode, w_h, w_n):
    # g: (n,4,h,w) -> channel 0 = standardized height, 1:4 = unit normal
    h1, n1 = g1[:, :1], g1[:, 1:]
    h2, n2 = g2[:, :1], g2[:, 1:]
    height_corr = torch.einsum("nchw,ncij->nhwij", h1, h2)   # ĥ_i * ĥ_j
    normal_corr = torch.einsum("nchw,ncij->nhwij", n1, n2)   # n_i . n_j (cosine, unit normals)
    if mode == "height":
        return height_corr
    if mode == "normal":
        return normal_corr
    return w_n * normal_corr + w_h * height_corr             # "both"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `../../.venv_depthg/bin/python -m pytest tests/test_geometric_affinity.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add refs/depthg/src/modules.py refs/depthg/tests/test_geometric_affinity.py
git commit -m "feat(ga-depthg): geometric affinity op (normal-cosine + height term)"
```

---

### Task 3: Wire affinity into the loss

**Files:**
- Modify: `refs/depthg/src/modules.py:1256` (`depth_feature_correlation`) and `:1280` (`forward`)
- Test: `refs/depthg/tests/test_depth_feature_correlation.py`

**Interfaces:**
- Consumes: `geometric_affinity` (Task 2); `cfg.affinity_mode, cfg.geom_w_h, cfg.geom_w_n`.
- Produces: `depth_feature_correlation(c1,c2,d1,d2,shift)` that, when `cfg.affinity_mode != 'scalar'`, treats `d1,d2` as `(n,4,h,w)` geometry tensors and uses `geometric_affinity`; otherwise unchanged.

- [ ] **Step 1: Write the failing test**

```python
# refs/depthg/tests/test_depth_feature_correlation.py
import torch
from types import SimpleNamespace
from src.modules import ContrastiveCorrelationLoss

def _cfg(mode):
    return SimpleNamespace(affinity_mode=mode, geom_w_h=.4, geom_w_n=.6, zero_clamp=True, stabalize=False)

def test_geometry_mode_runs_finite():
    loss = ContrastiveCorrelationLoss(_cfg('both'))
    c = torch.randn(2, 100, 8, 8)
    g = torch.randn(2, 4, 8, 8); g[:, 1:] = torch.nn.functional.normalize(g[:, 1:], dim=1)
    out, dd = loss.depth_feature_correlation(c, c, g, g, shift=0.03)
    assert torch.isfinite(out).all() and dd.shape == (2, 8, 8, 8, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `../../.venv_depthg/bin/python -m pytest tests/test_depth_feature_correlation.py -v`
Expected: FAIL (AttributeError on `affinity_mode`, or wrong branch).

- [ ] **Step 3: Edit `depth_feature_correlation` (modules.py:1256-1278)**

Replace the `dd = depth_correlation(norm(d1), norm(d2))` block with:

```python
        d1 = torch.nn.functional.interpolate(d1, size=c1.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.nn.functional.interpolate(d2, size=c2.shape[2:], mode='bilinear', align_corners=True)

        mode = getattr(self.cfg, "affinity_mode", "scalar")
        if mode == "scalar":
            dd = depth_correlation(norm(d1), norm(d2))
        else:
            d1 = torch.cat([d1[:, :1], norm(d1[:, 1:])], 1)   # re-normalize normals after interp
            d2 = torch.cat([d2[:, :1], norm(d2[:, 1:])], 1)
            dd = geometric_affinity(d1, d2, mode, self.cfg.geom_w_h, self.cfg.geom_w_n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `../../.venv_depthg/bin/python -m pytest tests/test_depth_feature_correlation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add refs/depthg/src/modules.py refs/depthg/tests/test_depth_feature_correlation.py
git commit -m "feat(ga-depthg): use geometric affinity in depth_feature_correlation"
```

---

### Task 4: Geometry caching script

**Files:**
- Create: `refs/depthg/scripts/cache_geometry.py`
- Test: `refs/depthg/tests/test_cache_geometry.py` (runs on 2 images)

**Interfaces:**
- Consumes: `compute_geometry`, `load_intrinsics` (Task 1).
- Produces: cached `(4,H,W)` `.npy` per image under `<data_dir>/geometry_depthpro/<split>/<city>/<stem>.npy`, aligned to DepthG's crop pipeline (plane fit on full image, then identical crop applied).

- [ ] **Step 1: Write the test (2-image smoke)**

```python
# refs/depthg/tests/test_cache_geometry.py
import numpy as np, subprocess, glob, os
def test_cache_two_images(tmp_path):
    subprocess.run(["../../.venv_depthg/bin/python", "scripts/cache_geometry.py",
                    "--split", "val", "--limit", "2", "--out", str(tmp_path)], check=True)
    fs = glob.glob(str(tmp_path) + "/val/*/*.npy")
    assert len(fs) == 2
    g = np.load(fs[0]); assert g.shape[0] == 4
```

- [ ] **Step 2: Run to verify it fails**

Run: `../../.venv_depthg/bin/python -m pytest tests/test_cache_geometry.py -v`
Expected: FAIL (script missing).

- [ ] **Step 3: Implement `cache_geometry.py`**

```python
import argparse, glob, os
from pathlib import Path
import numpy as np
from src.geometry_features import compute_geometry, load_intrinsics  # load_intrinsics copied in Task 1

DATA = Path("/Volumes/code_files/datasets/cityscapes")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(DATA / "geometry_depthpro"))
    a = ap.parse_args()
    files = sorted(glob.glob(str(DATA / f"depth_depthpro/{a.split}/*/*.npy")))
    if a.limit: files = files[:a.limit]
    for f in files:
        stem, city = Path(f).stem, Path(f).parent.name
        fx, fy, u0, v0, cam_h = load_intrinsics(stem, city, a.split)   # full-image intrinsics
        g = compute_geometry(np.load(f).astype(np.float32), fx, fy, u0, v0, cam_h)
        dst = Path(a.out) / a.split / city; dst.mkdir(parents=True, exist_ok=True)
        np.save(dst / f"{stem}.npy", g)
    print(f"cached {len(files)} geometry tensors -> {a.out}")

if __name__ == "__main__":
    main()
```

> **Crop-alignment note:** geometry is computed on the FULL image here. If DepthG's loader crops images at train time (`src/data.py` five-crop), apply the *same* crop transform to the loaded `(4,H,W)` geometry inside the dataset (Task 5), so geometry and features stay pixel-aligned. Re-normalize normals after any resize.

- [ ] **Step 4: Run to verify it passes**

Run: `../../.venv_depthg/bin/python -m pytest tests/test_cache_geometry.py -v`
Expected: PASS.

- [ ] **Step 5: Cache the full train+val set**

Run: `../../.venv_depthg/bin/python scripts/cache_geometry.py --split train && ../../.venv_depthg/bin/python scripts/cache_geometry.py --split val`
Expected: ~2975 train + 500 val `.npy` files written.

- [ ] **Step 6: Commit**

```bash
git add refs/depthg/scripts/cache_geometry.py refs/depthg/tests/test_cache_geometry.py
git commit -m "feat(ga-depthg): geometry caching script + full-set precompute"
```

---

### Task 5: Dataset loads geometry; config keys

**Files:**
- Modify: `refs/depthg/src/data.py` (Cityscapes dataset `__getitem__`, apply same transform/crop as the image)
- Modify: `refs/depthg/src/configs/train_config.yml`
- Modify: `refs/depthg/src/train_segmentation.py` (pass geometry where `depth` is passed to the loss)
- Test: `refs/depthg/tests/test_dataset_geometry.py`

**Interfaces:**
- Consumes: cached geometry from Task 4; `cfg.return_geometry`.
- Produces: dataset batch key `geometry` of shape `(B,4,H,W)`, routed into `ContrastiveCorrelationLoss.forward(..., depth=geometry, depth_pos=geometry_pos)` when `affinity_mode != 'scalar'`.

- [ ] **Step 1: Add config keys to `train_config.yml`**

```yaml
affinity_mode: scalar        # scalar | height | normal | both
geom_w_h: 0.4
geom_w_n: 0.6
return_geometry: false
geometry_dir: geometry_depthpro
```

- [ ] **Step 2: Write the failing test**

```python
# refs/depthg/tests/test_dataset_geometry.py
from src.data import ContrastiveSegDataset   # adjust to the actual Cityscapes dataset class
def test_batch_has_geometry(cityscapes_cfg):
    ds = ContrastiveSegDataset(**cityscapes_cfg, return_geometry=True)
    sample = ds[0]
    assert sample["geometry"].shape[0] == 4
```

- [ ] **Step 3: Implement geometry load in `data.py`**

In the Cityscapes dataset `__getitem__`, mirror the depth block: when `self.return_geometry`, load `np.load(<geometry_dir>/<split>/<city>/<stem>.npy)`, apply the SAME crop/resize transform used for the image, re-normalize channels 1:4, add to the returned dict as `geometry` (and `geometry_pos` for the second crop, same as depth).

- [ ] **Step 4: Route geometry into the loss in `train_segmentation.py`**

Where the training step calls `self.contrastive_corr_loss_fn(..., depth=batch["depth"], depth_pos=...)`, pass `batch["geometry"]` / `batch["geometry_pos"]` instead when `cfg.affinity_mode != "scalar"`.

- [ ] **Step 5: Run the test**

Run: `../../.venv_depthg/bin/python -m pytest tests/test_dataset_geometry.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add refs/depthg/src/data.py refs/depthg/src/configs/train_config.yml refs/depthg/src/train_segmentation.py refs/depthg/tests/test_dataset_geometry.py
git commit -m "feat(ga-depthg): dataset geometry loading + config + loss routing"
```

---

### Task 6: Smoke train (no-NaN gate)

**Files:** none (verification).

- [ ] **Step 1: 50-step smoke with `affinity_mode=both`**

Run (remote GPU): `../../.venv_depthg/bin/python src/train_segmentation.py dataset_name=cityscapes model_type=vit_base dim=100 depth_feat_correlation_loss=True depth_sampling=none return_geometry=True affinity_mode=both geom_w_h=0.4 geom_w_n=0.6 max_steps=50 batch_size=8 data_dir=<DATA>`
Expected: 50 steps complete, loss finite (no NaN/Inf), geometry term non-zero.

- [ ] **Step 2: Commit a note if any config tweak was needed**

```bash
git commit -am "chore(ga-depthg): smoke-train passes (both mode, no NaN)" --allow-empty
```

---

### Task 7: Run the ablation + eval table

**Files:** `reports/ga_depthg_ablation.md` (results).

- [ ] **Step 1: Train the four runs** (remote GPU, full schedule, the Cityscapes ViT-B hyperparameters from `paper_reproduction.sh` plus `return_geometry`/`affinity_mode`):

```
affinity_mode=scalar   # matched internal baseline (DepthPro)
affinity_mode=height
affinity_mode=normal
affinity_mode=both     # GA-DepthG
```

- [ ] **Step 2: Small sweep on `both`**: `depth_feat_weight ∈ {0.09, 0.2, 0.4}`, `depth_feat_shift ∈ {0.0, 0.03}`, `geom_w_n/geom_w_h ∈ {0.6/0.4, 0.8/0.2}` (the scalar-tuned 0.09 likely under-weights a non-degenerate affinity).

- [ ] **Step 3: Eval each** (local): `eval_segmentation.py model_paths=[...] dataset_name=cityscapes run_crf=True` → cluster+CRF+Hungarian mIoU (27-class).

- [ ] **Step 4: Write `reports/ga_depthg_ablation.md`** with the table: external DepthG (ZoeDepth) 23.1 · scalar(matched) · height · normal · both. **Kill-gate:** if `both ≤ scalar(matched)` after the sweep, stop and reassess.

- [ ] **Step 5: Commit results**

```bash
git add reports/ga_depthg_ablation.md
git commit -m "exp(ga-depthg): ablation table (scalar/height/normal/both)"
```

---

## Self-Review (done)
- **Spec coverage:** every spec section maps to a task — geometry module (T1), affinity op (T2), loss wiring (T3), data prep/cache (T4), dataset+config (T5), env/baseline (T0), ablation+eval (T6-T7). ✓
- **Placeholder scan:** code provided for all novel functions/tests; experiment steps give exact commands + the kill-gate. Class/attr names (`ContrastiveSegDataset`, exact depth-routing call site) to confirm against `data.py`/`train_segmentation.py` at execution — flagged inline in T5. 
- **Type consistency:** geometry tensor is `(n,4,h,w)` everywhere; `geometric_affinity` signature identical in T2/T3; `compute_geometry` returns `(4,H,W)` consumed by T4. ✓
