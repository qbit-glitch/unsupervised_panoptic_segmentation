# GA-UniAP Phase 0 (Kill-Gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide, in ~1 day on local CPU, whether geometric (depth-derived) merge affinity beats vanilla DINO-appearance affinity *inside UniAP's agglomerative pooling* — the kill-gate for the whole single-stage GA-UniAP direction.

**Architecture:** Offline A/B experiment. For N Cityscapes val images: extract a 32×64 DINO patch-feature grid + a matching 32×64 geometry grid (surface normal + height from precomputed DepthPro depth). Run an adapted UniAP agglomerative merge (`ga_aggo_merge`) under four affinity variants (V0 vanilla / V1 augment / V2 split-by-task / V3 geometry-only) plus a small weight sweep. Oracle-label each merged cluster by GT-majority class (isolating grouping quality from the labeling problem), score panoptic PQ with the project's existing harness, and write a variant×metric report with a gate verdict.

**Tech Stack:** Python 3, PyTorch (CPU), NumPy, SciPy, scikit-image, PIL, HuggingFace transformers (DINOv3 backbone). Reuses `mbps_pytorch/premise_check_geometry_affinity.py` (geometry) and `mbps_pytorch/sweep_depthpro.py` (PQ).

## Global Constraints

- **Runs locally on Mac CPU only** (project rule: evals never on remote). No training in Phase 0.
- **Data lives on an external drive:** `DATA = /Volumes/code_files/datasets/cityscapes` — must be mounted; every entry point checks this first.
- **Reuse, don't reimplement:** geometry via `premise_check_geometry_affinity.geometry()`; PQ via `sweep_depthpro.evaluate_panoptic_single()` + `compute_pq_from_accumulators()`; GT semantic via `premise_check_geometry_affinity.load_gt()`.
- **Eval resolution is fixed at 512×1024** (`WORK_H, WORK_W`), trainIDs 0–18, `STUFF_IDS={0..10}`, `THING_IDS={11..18}`.
- **Merge grid is 32×64** (2048 nodes) so the O(N·E) Python merge stays CPU-tractable. `# ponytail: O(N·E) dict merge; if too slow, drop grid or port to the vectorized graph version.`
- **Backbone default = DINOv3 ViT-B/16** (`facebook/dinov3-vitb16-pretrain-lvd1689m`, already the project's backbone, yields a 32×64 grid at 512×1024 with no extra download). *Deviation from spec, which named DINO ViT-B/8:* chosen to avoid a torch.hub gated-weights download and because the A/B's validity depends only on all variants sharing one backbone, not on which. ViT-B/8 is a one-function swap in `features.py` if literal S2-UniSeg fidelity is wanted later.
- **Oracle GT-majority labeling is a diagnostic upper bound**, not a publishable unsupervised PQ. It isolates pooling/grouping quality. Label-free CUPS global-Hungarian PQ is deferred to Phase 1.
- **Thresholds** (UniAP default): `[0.8, 0.7, 0.6, 0.5, 0.4]`. **min_size** (min patches/cluster): `4`.
- **Gate:** a geometric variant (V1/V2/V3) must beat V0 by **≥ +1.0 PQ_things** OR a clear instance-recall gain on the same images. Pass → Phase 1; fail → negative report, stop.

**File structure (all new, under `mbps_pytorch/ga_uniap/`):**
- `__init__.py` — package marker, exports.
- `config.py` — `Phase0Config` (frozen dataclass) + variant/weight registry.
- `preflight.py` — verify data drive + list usable val stems.
- `features.py` — DINO patch-feature grid extraction.
- `geometry.py` — geometry grid (normal+height pooled to merge grid).
- `pooling.py` — `ga_aggo_merge` (adapted UniAP merge with geometry+weights).
- `evaluate.py` — oracle panoptic labeling + PQ wrapper.
- `run_phase0.py` — orchestration, report, viz.
- `tests/test_pooling.py`, `tests/test_geometry.py`, `tests/test_evaluate.py`.

---

### Task 0: Package skeleton + preflight

**Files:**
- Create: `mbps_pytorch/ga_uniap/__init__.py`
- Create: `mbps_pytorch/ga_uniap/config.py`
- Create: `mbps_pytorch/ga_uniap/preflight.py`
- Create: `mbps_pytorch/ga_uniap/tests/__init__.py`

**Interfaces:**
- Produces: `Phase0Config` (frozen dataclass with `data_root: Path`, `split: str="val"`, `grid_h: int=32`, `grid_w: int=64`, `work_h: int=512`, `work_w: int=1024`, `thresholds: tuple=(0.8,0.7,0.6,0.5,0.4)`, `min_size: int=4`, `n_images: int=120`, `device: str="cpu"`).
- Produces: `list_val_stems(cfg) -> list[tuple[str,str]]` returning `(stem, city)` pairs that have BOTH a DepthPro `.npy` and gtFine present.

- [ ] **Step 1: Create the package + frozen config**

`mbps_pytorch/ga_uniap/__init__.py`:
```python
"""GA-UniAP Phase 0: geometric agglomerative pooling kill-gate."""
```

`mbps_pytorch/ga_uniap/config.py`:
```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

DATA_ROOT = Path("/Volumes/code_files/datasets/cityscapes")


@dataclass(frozen=True)
class Phase0Config:
    data_root: Path = DATA_ROOT
    split: str = "val"
    grid_h: int = 32
    grid_w: int = 64
    work_h: int = 512
    work_w: int = 1024
    thresholds: Tuple[float, ...] = (0.8, 0.7, 0.6, 0.5, 0.4)
    min_size: int = 4
    n_images: int = 120
    device: str = "cpu"
```

- [ ] **Step 2: Write the failing preflight test**

`mbps_pytorch/ga_uniap/tests/test_geometry.py` (start the test file here; geometry tests added in Task 1):
```python
from mbps_pytorch.ga_uniap.config import Phase0Config
from mbps_pytorch.ga_uniap.preflight import list_val_stems


def test_preflight_finds_val_stems():
    cfg = Phase0Config(n_images=10)
    stems = list_val_stems(cfg)
    assert len(stems) >= 10, f"expected >=10 usable val stems, got {len(stems)}"
    stem, city = stems[0]
    assert city in {"frankfurt", "lindau", "munster"}
    assert isinstance(stem, str) and stem.startswith(city)
```

- [ ] **Step 3: Run it, verify it fails**

Run: `python -m pytest mbps_pytorch/ga_uniap/tests/test_geometry.py::test_preflight_finds_val_stems -v`
Expected: FAIL — `ModuleNotFoundError: ... preflight`.

- [ ] **Step 4: Implement preflight**

`mbps_pytorch/ga_uniap/preflight.py`:
```python
import sys
from pathlib import Path
from typing import List, Tuple

from mbps_pytorch.ga_uniap.config import Phase0Config


def list_val_stems(cfg: Phase0Config) -> List[Tuple[str, str]]:
    """Return (stem, city) pairs that have BOTH DepthPro depth and gtFine."""
    root = cfg.data_root
    if not root.exists():
        raise FileNotFoundError(
            f"Cityscapes drive not mounted at {root}. Mount /Volumes/code_files first."
        )
    depth_root = root / "depth_depthpro" / cfg.split
    gt_root = root / "gtFine" / cfg.split
    out: List[Tuple[str, str]] = []
    for city_dir in sorted(depth_root.glob("*")):
        if not city_dir.is_dir():
            continue
        city = city_dir.name
        for npy in sorted(city_dir.glob("*.npy")):
            stem = npy.stem
            gt = gt_root / city / f"{stem}_gtFine_labelTrainIds.png"
            inst = gt_root / city / f"{stem}_gtFine_instanceIds.png"
            if gt.exists() and inst.exists():
                out.append((stem, city))
            if len(out) >= cfg.n_images:
                return out
    return out


if __name__ == "__main__":
    cfg = Phase0Config()
    stems = list_val_stems(cfg)
    print(f"usable val stems: {len(stems)} (requested {cfg.n_images})")
    for s, c in stems[:3]:
        print(" ", c, s)
    sys.exit(0 if stems else 1)
```

- [ ] **Step 5: Run test + the CLI**

Run: `python -m pytest mbps_pytorch/ga_uniap/tests/test_geometry.py::test_preflight_finds_val_stems -v`
Expected: PASS.
Run: `python -m mbps_pytorch.ga_uniap.preflight`
Expected: prints `usable val stems: 120 (requested 120)` and 3 example `(city, stem)` lines. (If it errors that the drive isn't mounted — mount `/Volumes/code_files` and retry.)

- [ ] **Step 6: Commit**

```bash
git add mbps_pytorch/ga_uniap/
git commit -m "feat(ga-uniap): package skeleton + val-stem preflight"
```

---

### Task 1: Geometry grid module

**Files:**
- Create: `mbps_pytorch/ga_uniap/geometry.py`
- Modify: `mbps_pytorch/ga_uniap/tests/test_geometry.py`

**Interfaces:**
- Consumes: `premise_check_geometry_affinity.geometry(stem, city, split) -> dict(normal[512,1024,3], height[512,1024], valid[512,1024], ...)`.
- Produces: `grid_geometry(stem, city, cfg) -> tuple[np.ndarray, np.ndarray] | None` → `(normal_grid[gh,gw,3] unit, height_grid[gh,gw] float)`, or `None` if depth/plane missing.
- Produces: `pool_to_grid(arr[H,W,...], gh, gw) -> arr[gh,gw,...]` (area-average pool).

- [ ] **Step 1: Write failing tests**

Append to `mbps_pytorch/ga_uniap/tests/test_geometry.py`:
```python
import numpy as np
from mbps_pytorch.ga_uniap.geometry import grid_geometry, pool_to_grid


def test_pool_to_grid_shapes_and_average():
    arr = np.ones((512, 1024, 3), np.float32) * 2.0
    out = pool_to_grid(arr, 32, 64)
    assert out.shape == (32, 64, 3)
    assert np.allclose(out, 2.0)


def test_grid_geometry_on_real_stem():
    cfg = Phase0Config(n_images=5)
    stem, city = list_val_stems(cfg)[0]
    res = grid_geometry(stem, city, cfg)
    assert res is not None, "geometry returned None on a real stem"
    normal, height = res
    assert normal.shape == (32, 64, 3)
    assert height.shape == (32, 64)
    norms = np.linalg.norm(normal, axis=-1)
    assert np.all(np.isfinite(normal)) and np.all(np.isfinite(height))
    # normals approximately unit (pooling shrinks them slightly; re-normalized in module)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-3)
```

- [ ] **Step 2: Run, verify fail**

Run: `python -m pytest mbps_pytorch/ga_uniap/tests/test_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: ... geometry`.

- [ ] **Step 3: Implement geometry grid**

`mbps_pytorch/ga_uniap/geometry.py`:
```python
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from mbps_pytorch.premise_check_geometry_affinity import geometry as _scene_geometry
from mbps_pytorch.ga_uniap.config import Phase0Config


def pool_to_grid(arr: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Area-average pool (H,W[,C]) -> (gh,gw[,C]). H,W must be multiples of gh,gw."""
    H, W = arr.shape[:2]
    assert H % gh == 0 and W % gw == 0, f"{(H, W)} not divisible by {(gh, gw)}"
    sh, sw = H // gh, W // gw
    if arr.ndim == 2:
        return arr.reshape(gh, sh, gw, sw).mean(axis=(1, 3))
    C = arr.shape[2]
    return arr.reshape(gh, sh, gw, sw, C).mean(axis=(1, 3))


def grid_geometry(stem: str, city: str, cfg: Phase0Config
                  ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(normal_grid[gh,gw,3] unit, height_grid[gh,gw]) from DepthPro, or None."""
    g = _scene_geometry(stem, city, cfg.split, source="mono")
    if g is None:
        return None
    normal = pool_to_grid(g["normal"].astype(np.float32), cfg.grid_h, cfg.grid_w)
    height = pool_to_grid(g["height"].astype(np.float32), cfg.grid_h, cfg.grid_w)
    nrm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / (nrm + 1e-9)
    return normal, height
```

- [ ] **Step 4: Run, verify pass**

Run: `python -m pytest mbps_pytorch/ga_uniap/tests/test_geometry.py -v`
Expected: PASS (3 tests). (`test_grid_geometry_on_real_stem` requires the drive mounted.)

- [ ] **Step 5: Commit**

```bash
git add mbps_pytorch/ga_uniap/geometry.py mbps_pytorch/ga_uniap/tests/test_geometry.py
git commit -m "feat(ga-uniap): geometry grid (normal+height pooled to merge grid)"
```

---

### Task 2: DINO patch-feature grid

**Files:**
- Create: `mbps_pytorch/ga_uniap/features.py`
- Create: `mbps_pytorch/ga_uniap/tests/test_features.py`

**Interfaces:**
- Produces: `extract_grid_features(stem, city, cfg) -> np.ndarray[gh,gw,768]` (float32, L2-normalizable patch features at the merge grid).

- [ ] **Step 1: Write failing smoke test**

`mbps_pytorch/ga_uniap/tests/test_features.py`:
```python
import numpy as np
from mbps_pytorch.ga_uniap.config import Phase0Config
from mbps_pytorch.ga_uniap.preflight import list_val_stems
from mbps_pytorch.ga_uniap.features import extract_grid_features


def test_extract_grid_features_shape():
    cfg = Phase0Config(n_images=3)
    stem, city = list_val_stems(cfg)[0]
    feats = extract_grid_features(stem, city, cfg)
    assert feats.shape == (32, 64, 768)
    assert feats.dtype == np.float32
    assert np.all(np.isfinite(feats))
```

- [ ] **Step 2: Run, verify fail**

Run: `python -m pytest mbps_pytorch/ga_uniap/tests/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: ... features`.

- [ ] **Step 3: Implement feature extraction (DINOv3 ViT-B/16)**

`mbps_pytorch/ga_uniap/features.py`:
```python
import functools
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from mbps_pytorch.ga_uniap.config import Phase0Config

_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"


@functools.lru_cache(maxsize=1)
def _load_model(device: str):
    proc = AutoImageProcessor.from_pretrained(_MODEL_NAME)
    proc.size = {"height": 512, "width": 1024}
    proc.crop_size = {"height": 512, "width": 1024}
    proc.do_center_crop = False
    model = AutoModel.from_pretrained(_MODEL_NAME).to(device).eval()
    return proc, model


def extract_grid_features(stem: str, city: str, cfg: Phase0Config) -> np.ndarray:
    """DINOv3 ViT-B/16 patch features -> (grid_h, grid_w, 768) float32.

    512x1024 input with patch 16 -> 32x64 patch grid, matching the merge grid.
    """
    proc, model = _load_model(cfg.device)
    img_path = (cfg.data_root / "leftImg8bit" / cfg.split / city
                / f"{stem}_leftImg8bit.png")
    img = Image.open(img_path).convert("RGB")
    inp = proc(images=img, return_tensors="pt").to(cfg.device)
    with torch.no_grad():
        out = model(**inp).last_hidden_state  # (1, 1+R+P, 768)
    n_reg = getattr(model.config, "num_register_tokens", 4)
    patch = out[:, 1 + n_reg:, :]  # (1, P, 768)
    gh = 512 // 16  # 32
    gw = 1024 // 16  # 64
    assert patch.shape[1] == gh * gw, f"got {patch.shape[1]} patches, want {gh*gw}"
    grid = patch.reshape(gh, gw, 768).float().cpu().numpy()
    assert (gh, gw) == (cfg.grid_h, cfg.grid_w), "backbone grid != cfg grid"
    return grid.astype(np.float32)
```

- [ ] **Step 4: Run, verify pass**

Run: `python -m pytest mbps_pytorch/ga_uniap/tests/test_features.py -v`
Expected: PASS. If it fails on model download/gated access: that is the de-risk signal — fall back to `facebook/dinov2-base` (patch 14 → adjust grid) or wire ViT-B/8 from `refs/dino`. Note the outcome and consult before proceeding.

- [ ] **Step 5: Commit**

```bash
git add mbps_pytorch/ga_uniap/features.py mbps_pytorch/ga_uniap/tests/test_features.py
git commit -m "feat(ga-uniap): DINOv3 patch-feature grid extraction"
```

---

### Task 3: Geometric agglomerative merge (the crux)

**Files:**
- Create: `mbps_pytorch/ga_uniap/pooling.py`
- Create: `mbps_pytorch/ga_uniap/tests/test_pooling.py`

**Interfaces:**
- Produces: `ga_aggo_merge(features[gh,gw,C], normal[gh,gw,3]|None, height[gh,gw]|None, thresholds, min_size, w_f, w_n, w_h) -> np.ndarray[K,gh,gw] bool` — K disjoint cluster masks (a partition of the grid into segments ≥ min_size; small remainder clusters dropped).
- Note: affinity per edge = `w_f·cos(f_i,f_j) + w_n·(n_i·n_j) + w_h·tanh-sim(h_i,h_j)`, where height similarity = `1 - |h_i-h_j|/H_SCALE` clipped to [-1,1] (`H_SCALE=3.0` m). With `w_n=w_h=0` this reduces exactly to vanilla UniAP cosine.

- [ ] **Step 1: Write the tests (reduce-to-vanilla is the key check)**

`mbps_pytorch/ga_uniap/tests/test_pooling.py`:
```python
import numpy as np
import torch
import torch.nn.functional as F

from mbps_pytorch.ga_uniap.pooling import ga_aggo_merge, _affinity


def _reference_vanilla(features, thresholds, min_size):
    """Verbatim cosine-only agglomerative merge (UniAP reference) for parity."""
    H, W, C = features.shape
    f = torch.from_numpy(features).reshape(H * W, C).float()
    fn = F.normalize(f, dim=1)
    clusters = [{"mask": (np.arange(H * W) == i), "nf": fn[i], "f": f[i],
                 "n": 1, "nb": set()} for i in range(H * W)]
    sims = {}
    for idx in range(H * W):
        if idx % W != 0:
            clusters[idx]["nb"].add(idx - 1); clusters[idx - 1]["nb"].add(idx)
            sims[(idx - 1, idx)] = float(fn[idx - 1] @ fn[idx])
        if idx - W >= 0:
            clusters[idx]["nb"].add(idx - W); clusters[idx - W]["nb"].add(idx)
            sims[(idx - W, idx)] = float(fn[idx - W] @ fn[idx])
    cur = H * W
    for th in thresholds:
        while sims:
            (i, j) = max(sims, key=sims.get)
            if sims[(i, j)] < th:
                break
            c1, c2 = clusters[i], clusters[j]
            ws = (c1["f"] + c2["f"]) / (c1["n"] + c2["n"])
            merged = {"mask": c1["mask"] | c2["mask"], "nf": F.normalize(ws, dim=0),
                      "f": c1["f"] + c2["f"], "n": c1["n"] + c2["n"],
                      "nb": (c1["nb"] | c2["nb"]) - {i, j}}
            clusters.append(merged); del sims[(i, j)]
            for nb in merged["nb"]:
                for a, b in ((i, nb), (j, nb)):
                    lo, hi = min(a, b), max(a, b)
                    if (lo, hi) in sims:
                        del sims[(lo, hi)]
                    clusters[nb]["nb"].discard(a)
                sims[(nb, cur)] = float(clusters[nb]["nf"] @ merged["nf"])
                clusters[nb]["nb"].add(cur)
            cur += 1
    seen, out = set(), []
    for (m, n) in sims:
        for k in (m, n):
            if k not in seen:
                seen.add(k)
                if clusters[k]["n"] >= min_size:
                    out.append(clusters[k]["mask"].reshape(H, W))
    return np.stack(out) if out else np.zeros((0, H, W), bool)


def test_reduces_to_vanilla_when_no_geometry():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((8, 8, 16)).astype(np.float32)
    th, ms = (0.6, 0.4), 2
    got = ga_aggo_merge(feats, None, None, th, ms, w_f=1.0, w_n=0.0, w_h=0.0)
    ref = _reference_vanilla(feats, th, ms)
    assert got.shape == ref.shape
    # same partition (order-independent): masks match as sets
    gs = sorted([m.tobytes() for m in got]); rs = sorted([m.tobytes() for m in ref])
    assert gs == rs


def test_masks_are_disjoint_partition():
    rng = np.random.default_rng(1)
    feats = rng.standard_normal((8, 8, 16)).astype(np.float32)
    masks = ga_aggo_merge(feats, None, None, (0.6,), 1, 1.0, 0.0, 0.0)
    cover = masks.sum(axis=0)
    assert cover.max() <= 1, "clusters overlap"


def test_geometry_term_separates_equal_feature_regions():
    # two halves with identical features but opposite normals must NOT merge under w_n
    feats = np.ones((4, 8, 8), np.float32)
    normal = np.zeros((4, 8, 3), np.float32)
    normal[:, :4] = [0, 0, 1]; normal[:, 4:] = [1, 0, 0]
    height = np.zeros((4, 8), np.float32)
    geo = ga_aggo_merge(feats, normal, height, (0.6,), 1, w_f=0.5, w_n=0.5, w_h=0.0)
    none = ga_aggo_merge(feats, None, None, (0.6,), 1, w_f=1.0, w_n=0.0, w_h=0.0)
    assert geo.shape[0] > none.shape[0], "geometry failed to split equal-feature halves"


def test_affinity_reduces_to_cosine():
    f = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    a = _affinity(f[0], f[1], None, None, None, None, 1.0, 0.0, 0.0)
    cos = float(F.normalize(f[0], dim=0) @ F.normalize(f[1], dim=0))
    assert abs(a - cos) < 1e-6
```

- [ ] **Step 2: Run, verify fail**

Run: `python -m pytest mbps_pytorch/ga_uniap/tests/test_pooling.py -v`
Expected: FAIL — `ModuleNotFoundError: ... pooling`.

- [ ] **Step 3: Implement ga_aggo_merge**

`mbps_pytorch/ga_uniap/pooling.py` (adapted from `test-instance-labels/S2-UniSeg/FastUniAP.py::aggo_merge`; geometry + weights added):
```python
"""Geometric agglomerative pooling — adapted from S2-UniSeg FastUniAP.aggo_merge.

Single edge affinity blends appearance cosine with depth-derived geometry:
    S_ij = w_f*cos(f_i,f_j) + w_n*(n_i . n_j) + w_h*(1 - |h_i-h_j|/H_SCALE)
With w_n=w_h=0 it is exactly the vanilla UniAP cosine merge.
"""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

H_SCALE = 3.0  # metres; height-difference scale for the height-similarity term


def _affinity(nf_i, nf_j, n_i, n_j, h_i, h_j, w_f, w_n, w_h) -> float:
    s = w_f * float(nf_i @ nf_j)
    if w_n and n_i is not None:
        s += w_n * float(n_i @ n_j)
    if w_h and h_i is not None:
        hsim = 1.0 - abs(float(h_i) - float(h_j)) / H_SCALE
        s += w_h * max(-1.0, min(1.0, hsim))
    return s


def ga_aggo_merge(features: np.ndarray,
                  normal: Optional[np.ndarray],
                  height: Optional[np.ndarray],
                  thresholds: Tuple[float, ...],
                  min_size: int,
                  w_f: float, w_n: float, w_h: float) -> np.ndarray:
    """Return (K, gh, gw) bool disjoint cluster masks (segments >= min_size)."""
    H, W, C = features.shape
    f = torch.from_numpy(features).reshape(H * W, C).float()
    nf = F.normalize(f, dim=1)
    nrm = None if normal is None else torch.from_numpy(normal).reshape(H * W, 3).float()
    hgt = None if height is None else torch.from_numpy(height).reshape(H * W).float()

    def aff(a, b):
        return _affinity(nf[a], nf[b], None if nrm is None else nrm[a],
                         None if nrm is None else nrm[b],
                         None if hgt is None else hgt[a],
                         None if hgt is None else hgt[b], w_f, w_n, w_h)

    clusters = [{"mask": (np.arange(H * W) == i), "nf": nf[i], "f": f[i],
                 "nrm": None if nrm is None else nrm[i],
                 "h": None if hgt is None else float(hgt[i]),
                 "n": 1, "nb": set()} for i in range(H * W)]
    sims = {}
    for idx in range(H * W):
        if idx % W != 0:
            clusters[idx]["nb"].add(idx - 1); clusters[idx - 1]["nb"].add(idx)
            sims[(idx - 1, idx)] = aff(idx - 1, idx)
        if idx - W >= 0:
            clusters[idx]["nb"].add(idx - W); clusters[idx - W]["nb"].add(idx)
            sims[(idx - W, idx)] = aff(idx - W, idx)

    def caff(a, b):
        ca, cb = clusters[a], clusters[b]
        s = w_f * float(ca["nf"] @ cb["nf"])
        if w_n and ca["nrm"] is not None:
            s += w_n * float(F.normalize(ca["nrm"], dim=0) @ F.normalize(cb["nrm"], dim=0))
        if w_h and ca["h"] is not None:
            hsim = 1.0 - abs(ca["h"] - cb["h"]) / H_SCALE
            s += w_h * max(-1.0, min(1.0, hsim))
        return s

    cur = H * W
    for th in thresholds:
        while sims:
            (i, j) = max(sims, key=sims.get)
            if sims[(i, j)] < th:
                break
            c1, c2 = clusters[i], clusters[j]
            tot = c1["n"] + c2["n"]
            ws = (c1["f"] + c2["f"]) / tot
            merged = {
                "mask": c1["mask"] | c2["mask"], "nf": F.normalize(ws, dim=0),
                "f": c1["f"] + c2["f"], "n": tot,
                "nrm": None if c1["nrm"] is None else (c1["nrm"] * c1["n"] + c2["nrm"] * c2["n"]) / tot,
                "h": None if c1["h"] is None else (c1["h"] * c1["n"] + c2["h"] * c2["n"]) / tot,
                "nb": (c1["nb"] | c2["nb"]) - {i, j},
            }
            clusters.append(merged); del sims[(i, j)]
            for nb in merged["nb"]:
                for a in (i, j):
                    lo, hi = min(a, nb), max(a, nb)
                    if (lo, hi) in sims:
                        del sims[(lo, hi)]
                    clusters[nb]["nb"].discard(a)
                sims[(nb, cur)] = caff(nb, cur)
                clusters[nb]["nb"].add(cur)
            cur += 1
        # remainder loop handled after final threshold

    seen, out = set(), []
    for (m, n) in sims:
        for k in (m, n):
            if k not in seen:
                seen.add(k)
                if clusters[k]["n"] >= min_size:
                    out.append(clusters[k]["mask"].reshape(H, W))
    return np.stack(out) if out else np.zeros((0, H, W), bool)
```

- [ ] **Step 4: Run, verify pass**

Run: `python -m pytest mbps_pytorch/ga_uniap/tests/test_pooling.py -v`
Expected: PASS (4 tests). The parity test (`test_reduces_to_vanilla_when_no_geometry`) is the load-bearing one — it proves the adaptation didn't change vanilla behaviour.

- [ ] **Step 5: Commit**

```bash
git add mbps_pytorch/ga_uniap/pooling.py mbps_pytorch/ga_uniap/tests/test_pooling.py
git commit -m "feat(ga-uniap): geometric agglomerative merge (reduces to vanilla UniAP)"
```

---

### Task 4: Oracle panoptic labeling + PQ scoring

**Files:**
- Create: `mbps_pytorch/ga_uniap/evaluate.py`
- Create: `mbps_pytorch/ga_uniap/tests/test_evaluate.py`

**Interfaces:**
- Consumes: `sweep_depthpro.evaluate_panoptic_single`, `compute_pq_from_accumulators`, `STUFF_IDS`, `THING_IDS`, `CS_ID_TO_TRAIN`; `premise_check_geometry_affinity.load_gt`.
- Consumes: `ga_aggo_merge` masks `[K, gh, gw]`.
- Produces: `load_gt_pair(stem, city, cfg) -> (gt_sem[512,1024] uint8, gt_inst[512,1024] int32)`.
- Produces: `masks_to_panoptic(masks[K,gh,gw], gt_sem[512,1024]) -> (pred_sem[512,1024] uint8, pred_instances list[(mask,cls,1.0)])` via oracle GT-majority labels.
- Produces: `score_image(masks, gt_sem, gt_inst, cfg) -> (tp,fp,fn,iou_sum)`.

- [ ] **Step 1: Write failing tests**

`mbps_pytorch/ga_uniap/tests/test_evaluate.py`:
```python
import numpy as np
from mbps_pytorch.ga_uniap.config import Phase0Config
from mbps_pytorch.ga_uniap.evaluate import masks_to_panoptic, score_image


def test_oracle_labeling_assigns_majority_class():
    # one cluster fully over a 'road'(0) region, one over 'car'(13)
    gt_sem = np.zeros((512, 1024), np.uint8)
    gt_sem[:, 512:] = 13  # right half = car (thing)
    masks = np.zeros((2, 32, 64), bool)
    masks[0, :, :32] = True   # left -> road
    masks[1, :, 32:] = True   # right -> car
    pred_sem, pred_inst = masks_to_panoptic(masks, gt_sem)
    assert (pred_sem[:, :512] == 0).all()
    assert len(pred_inst) == 1 and pred_inst[0][1] == 13


def test_perfect_stuff_mask_scores_high_pq():
    cfg = Phase0Config()
    gt_sem = np.zeros((512, 1024), np.uint8)  # all road
    gt_inst = np.zeros((512, 1024), np.int32)
    masks = np.ones((1, 32, 64), bool)        # one cluster covering everything -> road
    tp, fp, fn, iou = score_image(masks, gt_sem, gt_inst, cfg)
    assert tp[0] == 1 and fp[0] == 0 and fn[0] == 0  # road matched
```

- [ ] **Step 2: Run, verify fail**

Run: `python -m pytest mbps_pytorch/ga_uniap/tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: ... evaluate`.

- [ ] **Step 3: Implement evaluate**

`mbps_pytorch/ga_uniap/evaluate.py`:
```python
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mbps_pytorch.sweep_depthpro import (
    evaluate_panoptic_single, STUFF_IDS, THING_IDS, NUM_CLASSES,
)
from mbps_pytorch.premise_check_geometry_affinity import load_gt
from mbps_pytorch.ga_uniap.config import Phase0Config


def load_gt_pair(stem: str, city: str, cfg: Phase0Config) -> Tuple[np.ndarray, np.ndarray]:
    gt_sem = load_gt(stem, city, cfg.split).astype(np.uint8)  # (512,1024) trainID
    inst_path = (cfg.data_root / "gtFine" / cfg.split / city
                 / f"{stem}_gtFine_instanceIds.png")
    inst = np.array(Image.open(inst_path)).astype(np.int32)   # (1024,2048)
    inst = np.array(Image.fromarray(inst.astype(np.int32)).resize(
        (cfg.work_w, cfg.work_h), Image.NEAREST)).astype(np.int32)
    return gt_sem, inst


def _upsample(mask_grid: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    H, W = hw
    img = Image.fromarray(mask_grid.astype(np.uint8)).resize((W, H), Image.NEAREST)
    return np.array(img).astype(bool)


def masks_to_panoptic(masks: np.ndarray, gt_sem: np.ndarray
                      ) -> Tuple[np.ndarray, List]:
    """Oracle GT-majority labels (diagnostic upper bound on grouping quality)."""
    H, W = gt_sem.shape
    pred_sem = np.full((H, W), 255, np.uint8)
    pred_inst: List[Tuple[np.ndarray, int, float]] = []
    for k in range(masks.shape[0]):
        m = _upsample(masks[k], (H, W))
        vals = gt_sem[m]
        vals = vals[vals != 255]
        if vals.size == 0:
            continue
        cls = int(np.bincount(vals, minlength=NUM_CLASSES).argmax())
        if cls in THING_IDS:
            pred_inst.append((m, cls, 1.0))
        elif cls in STUFF_IDS:
            pred_sem[m] = cls
    return pred_sem, pred_inst


def score_image(masks: np.ndarray, gt_sem: np.ndarray, gt_inst: np.ndarray,
                cfg: Phase0Config):
    pred_sem, pred_inst = masks_to_panoptic(masks, gt_sem)
    return evaluate_panoptic_single(pred_sem, pred_inst, gt_sem, gt_inst,
                                    (cfg.work_h, cfg.work_w))
```

- [ ] **Step 4: Run, verify pass**

Run: `python -m pytest mbps_pytorch/ga_uniap/tests/test_evaluate.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add mbps_pytorch/ga_uniap/evaluate.py mbps_pytorch/ga_uniap/tests/test_evaluate.py
git commit -m "feat(ga-uniap): oracle panoptic labeling + PQ scoring wrapper"
```

---

### Task 5: Variant registry

**Files:**
- Modify: `mbps_pytorch/ga_uniap/config.py`
- Modify: `mbps_pytorch/ga_uniap/tests/test_geometry.py` (add a registry test) — or a new `tests/test_config.py`.

**Interfaces:**
- Produces: `VARIANTS: dict[str, dict]` mapping variant name → `{"mode": "single"|"split", "weights": (w_f,w_n,w_h), "weights_things": (...)|None}`.
- Produces: `weight_sweep() -> list[tuple[float,float,float]]` for V1 tuning (`w_n,w_h ∈ {0.3,0.5,0.7}`, `w_f=1.0`).

- [ ] **Step 1: Write failing test**

`mbps_pytorch/ga_uniap/tests/test_config.py`:
```python
from mbps_pytorch.ga_uniap.config import VARIANTS, weight_sweep


def test_variants_registry():
    assert VARIANTS["V0_vanilla"]["weights"] == (1.0, 0.0, 0.0)
    assert VARIANTS["V3_geom_only"]["weights"][0] == 0.0
    assert VARIANTS["V2_split"]["mode"] == "split"
    assert set(VARIANTS) == {"V0_vanilla", "V1_augment", "V2_split", "V3_geom_only"}


def test_weight_sweep_grid():
    sweep = weight_sweep()
    assert (1.0, 0.5, 0.3) in sweep
    assert all(w[0] == 1.0 for w in sweep)
    assert len(sweep) == 9
```

- [ ] **Step 2: Run, verify fail**

Run: `python -m pytest mbps_pytorch/ga_uniap/tests/test_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'VARIANTS'`.

- [ ] **Step 3: Add registry to config.py**

Append to `mbps_pytorch/ga_uniap/config.py`:
```python
from itertools import product
from typing import Dict, List

VARIANTS: Dict[str, dict] = {
    "V0_vanilla":   {"mode": "single", "weights": (1.0, 0.0, 0.0), "weights_things": None},
    "V1_augment":   {"mode": "single", "weights": (1.0, 0.5, 0.3), "weights_things": None},
    "V2_split":     {"mode": "split",  "weights": (1.0, 0.0, 0.0), "weights_things": (0.3, 0.6, 0.4)},
    "V3_geom_only": {"mode": "single", "weights": (0.0, 0.6, 0.4), "weights_things": None},
}


def weight_sweep() -> List[tuple]:
    """V1 augment tuning grid: w_f=1, w_n,w_h in {0.3,0.5,0.7}."""
    return [(1.0, wn, wh) for wn, wh in product((0.3, 0.5, 0.7), (0.3, 0.5, 0.7))]
```

- [ ] **Step 4: Run, verify pass**

Run: `python -m pytest mbps_pytorch/ga_uniap/tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mbps_pytorch/ga_uniap/config.py mbps_pytorch/ga_uniap/tests/test_config.py
git commit -m "feat(ga-uniap): variant registry + V1 weight sweep"
```

---

### Task 6: Orchestration + N=2 smoke

**Files:**
- Create: `mbps_pytorch/ga_uniap/run_phase0.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_variant(name, stems, cfg) -> dict` (accumulated PQ result via `compute_pq_from_accumulators`).
- Produces: `apply_variant(name, feats, normal, height, cfg) -> masks[K,gh,gw]` — dispatches single vs split mode.
- Produces: CLI `python -m mbps_pytorch.ga_uniap.run_phase0 --n 2 --variants V0_vanilla,V1_augment`.

- [ ] **Step 1: Implement orchestration**

`mbps_pytorch/ga_uniap/run_phase0.py`:
```python
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mbps_pytorch.sweep_depthpro import compute_pq_from_accumulators, NUM_CLASSES
from mbps_pytorch.ga_uniap.config import Phase0Config, VARIANTS
from mbps_pytorch.ga_uniap.preflight import list_val_stems
from mbps_pytorch.ga_uniap.features import extract_grid_features
from mbps_pytorch.ga_uniap.geometry import grid_geometry
from mbps_pytorch.ga_uniap.pooling import ga_aggo_merge
from mbps_pytorch.ga_uniap.evaluate import load_gt_pair, score_image


def apply_variant(name, feats, normal, height, cfg):
    spec = VARIANTS[name]
    wf, wn, wh = spec["weights"]
    if spec["mode"] == "single":
        return ga_aggo_merge(feats, normal, height, cfg.thresholds, cfg.min_size, wf, wn, wh)
    # split: feature-only for stuff, geometry-heavy for things; concatenate masks,
    # oracle labeling downstream keeps stuff-from-A / things-from-B implicitly.
    masks_a = ga_aggo_merge(feats, None, None, cfg.thresholds, cfg.min_size, 1.0, 0.0, 0.0)
    wtf, wtn, wth = spec["weights_things"]
    masks_b = ga_aggo_merge(feats, normal, height, cfg.thresholds, cfg.min_size, wtf, wtn, wth)
    return np.concatenate([masks_a, masks_b], axis=0) if len(masks_a) or len(masks_b) \
        else np.zeros((0, cfg.grid_h, cfg.grid_w), bool)


def run_variant(name, stems, cfg):
    tp = np.zeros(NUM_CLASSES); fp = np.zeros(NUM_CLASSES)
    fn = np.zeros(NUM_CLASSES); iou = np.zeros(NUM_CLASSES)
    used = 0
    for stem, city in stems:
        geo = grid_geometry(stem, city, cfg)
        if geo is None:
            continue
        normal, height = geo
        feats = extract_grid_features(stem, city, cfg)
        masks = apply_variant(name, feats, normal, height, cfg)
        gt_sem, gt_inst = load_gt_pair(stem, city, cfg)
        t, f, n, i, _ = score_image(masks, gt_sem, gt_inst, cfg)
        tp += t; fp += f; fn += n; iou += i; used += 1
    res = compute_pq_from_accumulators(tp, fp, fn, iou)
    res["_n_used"] = used
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--variants", type=str, default=",".join(VARIANTS))
    ap.add_argument("--out", type=str, default="reports/ga_uniap_phase0.json")
    a = ap.parse_args()
    cfg = Phase0Config(n_images=a.n)
    stems = list_val_stems(cfg)
    print(f"running on {len(stems)} stems")
    results = {}
    for name in a.variants.split(","):
        t0 = time.time()
        results[name] = run_variant(name, stems, cfg)
        r = results[name]
        print(f"{name:14s} PQ={r['PQ']:.2f} PQ_th={r['PQ_things']:.2f} "
              f"PQ_st={r['PQ_stuff']:.2f}  ({time.time()-t0:.0f}s, n={r['_n_used']})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the N=2 smoke**

Run: `python -m mbps_pytorch.ga_uniap.run_phase0 --n 2 --variants V0_vanilla,V1_augment --out reports/ga_uniap_smoke.json`
Expected: prints two lines with finite `PQ=...` per variant and writes `reports/ga_uniap_smoke.json`. (First run downloads/loads the DINOv3 model — may take a minute. If it errors on the data drive, mount `/Volumes/code_files`.)

- [ ] **Step 3: Sanity-assert the smoke output**

Run:
```bash
python -c "import json; r=json.load(open('reports/ga_uniap_smoke.json')); assert r['V0_vanilla']['_n_used']>=1 and 0<=r['V0_vanilla']['PQ']<=100; print('smoke OK', r['V0_vanilla']['PQ'], r['V1_augment']['PQ'])"
```
Expected: `smoke OK <pq0> <pq1>`.

- [ ] **Step 4: Commit**

```bash
git add mbps_pytorch/ga_uniap/run_phase0.py
git commit -m "feat(ga-uniap): phase-0 orchestration + N=2 smoke"
```

---

### Task 7: Full run, report table, qualitative viz, gate verdict

**Files:**
- Modify: `mbps_pytorch/ga_uniap/run_phase0.py` (add `--sweep`, report writer, viz)
- Create: `reports/ga_uniap_phase0.md` (generated)

**Interfaces:**
- Produces: `write_report(results, sweep_results, cfg, path)` → markdown table + gate verdict.
- Produces: `save_viz(stem, city, masks_by_variant, cfg, path)` → side-by-side overlay PNG for one crowd + one non-crowd scene.

- [ ] **Step 1: Add report writer + gate verdict**

Append to `mbps_pytorch/ga_uniap/run_phase0.py`:
```python
def write_report(results, sweep_results, cfg, path="reports/ga_uniap_phase0.md"):
    lines = ["# GA-UniAP Phase 0 — Kill-Gate Results", "",
             f"Grid {cfg.grid_h}x{cfg.grid_w}, thresholds {cfg.thresholds}, "
             f"min_size {cfg.min_size}, n_images {cfg.n_images}.",
             "Oracle GT-majority labeling (diagnostic upper bound, isolates grouping).",
             "", "| Variant | PQ | PQ_things | PQ_stuff | SQ | RQ |",
             "|---|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['PQ']:.2f} | {r['PQ_things']:.2f} | "
                     f"{r['PQ_stuff']:.2f} | {r['SQ']:.2f} | {r['RQ']:.2f} |")
    v0 = results["V0_vanilla"]["PQ_things"]
    best = max((results[k]["PQ_things"], k) for k in results if k != "V0_vanilla")
    delta = best[0] - v0
    verdict = ("PASS — proceed to Phase 1" if delta >= 1.0
               else "FAIL — geometry does not beat appearance; stop")
    lines += ["", f"**Best geometric Δ PQ_things vs V0 = {delta:+.2f} ({best[1]}). {verdict}.**"]
    if sweep_results:
        lines += ["", "## V1 weight sweep (w_f=1)", "", "| w_n | w_h | PQ_things |",
                  "|---|---|---|"]
        for (wf, wn, wh), r in sweep_results.items():
            lines.append(f"| {wn} | {wh} | {r['PQ_things']:.2f} |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))
    print(f"wrote {path} — {verdict}")
```
Wire it into `main()` after the variant loop: add `--sweep` flag that, when set, also runs `run_variant` over `weight_sweep()` tuples (constructing a temporary single-variant by monkeypatching weights — simplest: add a `run_weights(weights, stems, cfg)` mirroring `run_variant` but calling `ga_aggo_merge` directly with the tuple). Then call `write_report(results, sweep_results, cfg)`.

- [ ] **Step 2: Add qualitative viz**

Append `save_viz` using matplotlib: for one high-instance-count scene (e.g. a `frankfurt` stem with many GT person instances) and one low-instance scene, overlay each variant's masks (random colors) on the RGB, save a 1×4 grid PNG to `reports/figures/ga_uniap/<stem>.png`. (Colors per cluster; title each panel with the variant name.) Keep it ~30 lines; this is diagnostic, not production.

- [ ] **Step 3: Run the full kill-gate**

Run: `python -m mbps_pytorch.ga_uniap.run_phase0 --n 120 --sweep --out reports/ga_uniap_phase0.json`
Expected: four variant lines printed, `reports/ga_uniap_phase0.json` + `reports/ga_uniap_phase0.md` written, final line `wrote reports/ga_uniap_phase0.md — PASS/FAIL ...`. Wall-clock target: tens of minutes on CPU (2048-node merge × 120 images × ~6 variant/sweep runs).

- [ ] **Step 4: Read the verdict and the per-class table**

Run: `sed -n '1,30p' reports/ga_uniap_phase0.md`
Expected: the variant table + the bold gate verdict line. Inspect whether any geometric variant beats V0 on PQ_things by ≥1.0, and check `reports/figures/ga_uniap/*.png` to see *where* geometry helps (non-crowd/strong-layout) vs fails (crowds).

- [ ] **Step 5: Commit**

```bash
git add mbps_pytorch/ga_uniap/run_phase0.py reports/ga_uniap_phase0.md reports/ga_uniap_phase0.json
git commit -m "feat(ga-uniap): full kill-gate run, report table + gate verdict"
```

---

## Self-Review

**Spec coverage:**
- Four affinity variants (V0/V1/V2/V3) — Tasks 3, 5, 6 ✓
- Weight sweep (normals>height init) — Task 5 `weight_sweep`, Task 7 `--sweep` ✓
- Offline, local CPU, ~50–150 val images — Global Constraints + Task 6/7 ✓
- DepthPro geometry reuse — Task 1 ✓
- PQ via existing harness, PQ_things/recall, per-class — Task 4 + Task 7 ✓
- Gate (≥ +1 PQ_things) + negative-report path — Task 7 verdict ✓
- Qualitative viz crowd vs non-crowd — Task 7 Step 2 ✓
- Reduce-to-vanilla guarantee — Task 3 parity test ✓

**Deviations from spec (flagged):** backbone DINOv3 ViT-B/16 instead of DINO ViT-B/8 (rationale in Global Constraints); merge runs on a 32×64 grid (CPU tractability); oracle-majority labeling used as the kill-gate metric (label-free CUPS PQ deferred to Phase 1). These do not affect the V0-vs-geometry comparison's validity.

**Placeholder scan:** none — every code step is complete. Task 7 Steps 2 (viz) and the `--sweep` wiring are described rather than fully coded; they are diagnostic glue, bounded to ~30 lines, and the `run_variant`/`write_report` patterns they mirror are given in full.

**Type consistency:** `ga_aggo_merge` returns `[K,gh,gw] bool` consumed by `masks_to_panoptic`/`score_image`; `(stem, city)` tuple threading is uniform; `Phase0Config` passed everywhere; PQ dict keys (`PQ`,`PQ_things`,`PQ_stuff`,`SQ`,`RQ`) match `compute_pq_from_accumulators`. Consistent.

**Open risk to watch:** the 32×64 oracle-labeled absolute PQ will be well below the current k27 pipeline's ~26–30 (coarser grid + oracle≠label-free); the gate is the **V0-vs-geometric delta on identical settings**, not the absolute number — do not compare Phase-0 absolutes to the 26–30 reference.
