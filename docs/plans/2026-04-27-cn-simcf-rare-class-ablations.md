# CN-SIMCF Rare-Class Recovery Ablation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` for batch execution with checkpoints between ablations. Each ablation step has a hard decision gate based on local pseudo-label PQ.

**Goal:** Ablate three cumulative interventions (CN-SIMCF, bootstrap remapping, multi-source protected mask) on the DCFA+DepthPro+SIMCF-ABC pseudo-label pipeline to recover dead Cityscapes classes (guard rail, tunnel, polegroup, caravan, trailer). Evaluate pseudo-label PQ locally on Cityscapes train (2,975 images) after each step, with explicit decision gates.

**Architecture:**
- **A1: Cluster-Native SIMCF (CN-SIMCF)** — Operate on 80 cluster IDs end-to-end; defer `cluster_to_class` mapping until after Steps A/B/C. Eliminates the chokepoint where `cluster_to_class` static argmax erases rare clusters.
- **A2: + Bootstrap Remapping (×2 iters)** — Re-fit `cluster_to_class` from CN-SIMCF outputs without GT (cluster co-occurrence in trained-model logits). Re-run CN-SIMCF with new mapping. 2 rounds.
- **A3: + 3-of-4 Multi-Source Protected Mask** — Build a high-precision rare-pixel mask from 4 independent unsupervised signals (DCFA disagreement, k=300 small clusters, DINOv3 [CLS] attention, depth edge). Modify CN-SIMCF to honor protected pixels.
- **A4 (DEFERRED — remote)** — Class-balanced sampling + EQLv2 in Stage-2 from epoch 0. Requires 6–8 h on RTX A6000. Not run locally.

**Tech Stack:** Python 3.10, NumPy, scipy.ndimage, PIL, PyTorch 2.10 (MPS for DCFA forward + DINOv3 attention), MiniBatchKMeans (sklearn). Local M4 Pro 48GB MacBook.

**Baseline (must reproduce in T0):**
| Metric | Value |
|---|---:|
| PQ | 25.85 |
| PQ_stuff | 33.96 |
| PQ_things | 14.70 |
| mIoU | 56.22 |

**Decision gates (per ablation):**
- ΔPQ > 0 OR ΔPQ_things > 0.5 → **proceed** to next ablation
- ΔPQ ∈ (-0.5, 0] → **continue but flag** as neutral-needs-investigation
- ΔPQ ≤ -0.5 → **abort** that branch and report

**Inputs (verified to exist):**
- `~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_depthpro/` (current SIMCF-ABC output, baseline)
- `~/Desktop/datasets/cityscapes/cups_pseudo_labels_adapter_V3_tau020/` (DCFA + DepthPro pre-SIMCF)
- `~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80/kmeans_centroids.npz` (k=80 centroids + cluster_to_class)
- `~/Desktop/datasets/cityscapes/dinov3_features/train/<city>/*.npy` (DINOv3 patch features)
- `~/Desktop/datasets/cityscapes/depth_depthpro/train/<city>/*.npy` (DepthPro depth maps)

**Eval contract:**
```bash
python scripts/evaluate_pseudolabel_quality.py \
  --pseudo_dir <OUTPUT_DIR> \
  --cityscapes_root ~/Desktop/datasets/cityscapes \
  --centroids_path ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80/kmeans_centroids.npz \
  --split train \
  --num_clusters 80 \
  --use_hungarian \
  --output reports/cn_simcf/<TAG>.json
```

**File structure (created by this plan):**
```
scripts/
├── refine_cn_simcf.py                  # T1: cluster-native SIMCF
├── bootstrap_cluster_remap.py          # T2: unsupervised cluster_to_class refit
└── build_rare_protected_mask.py        # T3: 3-of-4 multi-source mask
reports/cn_simcf/
├── T0_baseline.json
├── T1_cn_simcf.json
├── T2_bootstrap_iter1.json
├── T2_bootstrap_iter2.json
└── T3_protected.json
docs/plans/
└── 2026-04-27-cn-simcf-rare-class-ablations.md  # this file
```

---

## T0: Baseline Verification

**Files:** none (read-only).

- [ ] **Step 1: Confirm baseline directory exists and has expected file count**

```bash
ls ~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_depthpro/ | wc -l
```
Expected: 8,925 files (2,975 images × 3 files each: `_semantic.png`, `_instance.png`, `.pt`).

- [ ] **Step 2: Run baseline eval**

```bash
mkdir -p reports/cn_simcf
python scripts/evaluate_pseudolabel_quality.py \
  --pseudo_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_depthpro \
  --cityscapes_root ~/Desktop/datasets/cityscapes \
  --centroids_path ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80/kmeans_centroids.npz \
  --split train \
  --num_clusters 80 \
  --use_hungarian \
  --output reports/cn_simcf/T0_baseline.json
```
Expected wall time: ~8–12 min on M4 Pro.
Expected output: `PQ ≈ 25.85, PQ_stuff ≈ 33.96, PQ_things ≈ 14.70, mIoU ≈ 56.22` (±0.1).

- [ ] **Step 3: Identify dead classes from per-class breakdown**

Open `reports/cn_simcf/T0_baseline.json` and list all classes with `PQ < 1.0`. These are the targets for A1–A3. Expected dead/near-dead: `wall, fence, traffic light, traffic sign, terrain, train, motorcycle` (varies — record actual list).

- [ ] **Step 4: Decision gate for T0**

If reproduced PQ within ±0.3 of 25.85 → proceed to T1.
If PQ deviates >0.3 → STOP. Investigate (likely a centroids version mismatch).

---

## T1: Cluster-Native SIMCF (A1)

**Files:**
- Create: `scripts/refine_cn_simcf.py` (forked from `scripts/refine_simcf.py`)
- Output: `~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf/`

**What changes vs `refine_simcf.py`:**

| Step | refine_simcf.py | refine_cn_simcf.py |
|---|---|---|
| A | majority vote in `trainID` space, reassign to `best_cluster` of majority class | majority vote in `cluster_id` space; reassign minority pixels to majority *cluster* |
| B | merge if `inst_class[i] == inst_class[j]` AND cosine sim > 0.85 | merge ONLY if `inst_majority_cluster[i] == inst_majority_cluster[j]` AND cosine sim > 0.92 |
| C | per-trainID depth profile (19 mean/std), 3σ outlier mask | per-cluster depth profile (80 mean/std), 3σ for clusters with N≥500 px, 5σ for rare (<500) |
| Final | semantic stays as cluster IDs; mapping happens at eval time | same — preserves cluster IDs; eval uses centroids' `cluster_to_class` |

- [ ] **Step 1: Copy `refine_simcf.py` to `refine_cn_simcf.py`**

```bash
cp scripts/refine_simcf.py scripts/refine_cn_simcf.py
```

- [ ] **Step 2: Replace `step_a` with cluster-native version**

Replace the function body of `step_a()` in `scripts/refine_cn_simcf.py` (lines 82–128 of original) with:

```python
def step_a(semantic: np.ndarray, instance: np.ndarray,
           cluster_to_class: np.ndarray, num_clusters: int) -> int:
    """CN-SIMCF Step A: per-instance majority vote in CLUSTER space.

    Reassigns minority pixels to the majority CLUSTER (not class) within
    each instance. Preserves rare-class clusters that would be erased by
    class-space majority voting.
    """
    n_changed = 0
    for iid in np.unique(instance):
        if iid == 0:
            continue
        ys, xs = np.where(instance == iid)
        if len(ys) == 0:
            continue

        clusters = semantic[ys, xs]
        # Majority CLUSTER (not class) within this instance
        valid_clusters = clusters[clusters < num_clusters]
        if len(valid_clusters) == 0:
            continue
        cluster_counts = np.bincount(valid_clusters, minlength=num_clusters)
        majority_cluster = int(cluster_counts.argmax())

        # Reassign only pixels whose cluster maps to a DIFFERENT trainID than the majority cluster
        # (this preserves rare clusters that map to the same trainID as the majority)
        majority_tid = int(cluster_to_class[majority_cluster])
        if majority_tid >= NUM_CLASSES:
            continue
        train_ids = cluster_to_class[clusters]
        inconsistent = (train_ids != majority_tid) & (train_ids < NUM_CLASSES)
        if not inconsistent.any():
            continue

        semantic[ys[inconsistent], xs[inconsistent]] = majority_cluster
        n_changed += int(inconsistent.sum())

    return n_changed
```

- [ ] **Step 3: Replace `step_b` with cluster-native merging**

Replace `step_b()` body (lines 135–239 of original) with the version that compares `inst_majority_cluster` instead of `inst_class`, raises threshold to `sim_threshold = 0.92`:

```python
def step_b(semantic: np.ndarray, instance: np.ndarray,
           features: np.ndarray, cluster_to_class: np.ndarray,
           sim_threshold: float = 0.92, dilate_px: int = 3) -> tuple:
    """CN-SIMCF Step B: merge adjacent instances ONLY if they share the
    same majority cluster ID AND have very high feature similarity.

    Threshold raised from 0.85 -> 0.92 because we now require cluster
    identity (stricter than class identity), which is rarer.
    """
    inst_small = np.array(
        Image.fromarray(instance).resize((FEAT_W, FEAT_H), Image.NEAREST)
    )
    feat_2d = features.reshape(FEAT_H, FEAT_W, -1)

    instance_ids = np.unique(instance)
    instance_ids = instance_ids[instance_ids > 0]
    if len(instance_ids) < 2:
        return instance, 0

    inst_majority_cluster = {}
    inst_feat = {}
    num_clusters = int(cluster_to_class.shape[0])

    for iid in instance_ids:
        mask = instance == iid
        clusters = semantic[mask]
        valid = clusters[clusters < num_clusters]
        if len(valid) == 0:
            continue
        inst_majority_cluster[iid] = int(np.bincount(valid, minlength=num_clusters).argmax())

        mask_s = inst_small == iid
        if not mask_s.any():
            continue
        patches = feat_2d[mask_s]
        feat = patches.mean(axis=0)
        norm = np.linalg.norm(feat) + 1e-8
        inst_feat[iid] = feat / norm

    struct = ndimage.generate_binary_structure(2, 1)
    adjacency = set()
    for iid in instance_ids:
        if iid not in inst_majority_cluster:
            continue
        mask = instance == iid
        dilated = ndimage.binary_dilation(mask, structure=struct, iterations=dilate_px)
        border = dilated & ~mask
        for nb in np.unique(instance[border]):
            if nb == 0 or nb == iid or nb not in inst_majority_cluster:
                continue
            adjacency.add((min(iid, nb), max(iid, nb)))

    merge_pairs = []
    for i, j in adjacency:
        # CRITICAL: require same CLUSTER (not just same class)
        if inst_majority_cluster.get(i) != inst_majority_cluster.get(j):
            continue
        if i not in inst_feat or j not in inst_feat:
            continue
        sim = float(np.dot(inst_feat[i], inst_feat[j]))
        if sim > sim_threshold:
            merge_pairs.append((i, j))

    if not merge_pairs:
        return instance, 0

    parent = {iid: iid for iid in instance_ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in merge_pairs:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi

    new_instance = np.zeros_like(instance)
    root_to_new = {}
    next_id = 1
    for iid in sorted(instance_ids):
        root = find(iid)
        if root not in root_to_new:
            root_to_new[root] = next_id
            next_id += 1
        new_instance[instance == iid] = root_to_new[root]

    n_merges = len(instance_ids) - len(root_to_new)
    return new_instance, n_merges
```

- [ ] **Step 4: Replace `compute_depth_stats` and `step_c` with per-cluster versions**

Replace `compute_depth_stats()` (lines 246–295) with `compute_depth_stats_per_cluster()`:

```python
def compute_depth_stats_per_cluster(stems: list, input_dir: Path, depth_dir: Path,
                                    num_clusters: int) -> tuple:
    """Per-CLUSTER depth profile (80 means/stds, not 19 per-class)."""
    logger.info("CN-SIMCF Step C first pass: per-CLUSTER depth statistics...")
    cluster_sum = np.zeros(num_clusters, dtype=np.float64)
    cluster_sum_sq = np.zeros(num_clusters, dtype=np.float64)
    cluster_count = np.zeros(num_clusters, dtype=np.int64)

    for cups_stem in tqdm(stems, desc="Depth stats per-cluster"):
        city = _extract_city(cups_stem)
        sem = np.array(Image.open(input_dir / f"{cups_stem}_semantic.png"))
        sem_h, sem_w = sem.shape

        depth_path = _resolve_depth_path(depth_dir, "train", city, cups_stem)
        if depth_path is None:
            continue
        depth = np.load(str(depth_path)).astype(np.float64)
        if depth.shape != (sem_h, sem_w):
            depth = np.array(
                Image.fromarray(depth.astype(np.float32)).resize(
                    (sem_w, sem_h), Image.BILINEAR)
            ).astype(np.float64)

        for cl in range(num_clusters):
            mask = sem == cl
            if not mask.any():
                continue
            vals = depth[mask]
            cluster_sum[cl] += vals.sum()
            cluster_sum_sq[cl] += (vals ** 2).sum()
            cluster_count[cl] += len(vals)

    safe_count = np.maximum(cluster_count, 1)
    cluster_mean = cluster_sum / safe_count
    cluster_var = cluster_sum_sq / safe_count - cluster_mean ** 2
    cluster_std = np.sqrt(np.maximum(cluster_var, 0.0))

    rare_clusters = (cluster_count < 500) & (cluster_count > 0)
    logger.info(f"  {int(rare_clusters.sum())} rare clusters (<500 px) get 5σ tolerance")
    return cluster_mean, cluster_std, cluster_count


def step_c(semantic: np.ndarray, depth: np.ndarray,
           cluster_mean: np.ndarray, cluster_std: np.ndarray,
           cluster_count: np.ndarray,
           num_clusters: int,
           sigma_common: float = 3.0,
           sigma_rare: float = 5.0,
           rare_count_threshold: int = 500) -> int:
    """Per-cluster outlier masking. Rare clusters get looser σ."""
    n_masked = 0
    for cl in range(num_clusters):
        if cluster_std[cl] < 1e-6 or cluster_count[cl] == 0:
            continue
        mask = semantic == cl
        if not mask.any():
            continue
        sigma = sigma_rare if cluster_count[cl] < rare_count_threshold else sigma_common
        deviation = np.abs(depth[mask] - cluster_mean[cl])
        outlier = deviation > sigma * cluster_std[cl]
        if outlier.any():
            ys, xs = np.where(mask)
            semantic[ys[outlier], xs[outlier]] = 255
            n_masked += int(outlier.sum())
    return n_masked
```

- [ ] **Step 5: Update `main()` to use new APIs**

In `main()`:
- Replace the call `class_mean, class_std = compute_depth_stats(...)` with `cluster_mean, cluster_std, cluster_count = compute_depth_stats_per_cluster(stems, input_dir, depth_dir, num_clusters)`
- Replace the Step C call with `step_c(semantic, depth, cluster_mean, cluster_std, cluster_count, num_clusters)` (drop `cluster_to_class` arg).
- Update the `--sim_threshold` default to `0.92`.
- Remove `cluster_to_class` argument from `step_c` call.

- [ ] **Step 6: Run CN-SIMCF on the existing DCFA+DepthPro input**

```bash
python scripts/refine_cn_simcf.py \
  --input_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_adapter_V3_tau020 \
  --output_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf \
  --centroids_path ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80/kmeans_centroids.npz \
  --cityscapes_root ~/Desktop/datasets/cityscapes \
  --features_subdir dinov3_features \
  --depth_subdir depth_depthpro \
  --steps A,B,C \
  --sim_threshold 0.92 \
  --num_clusters 80 \
  2>&1 | tee logs/T1_cn_simcf.log
```
Expected wall time: ~6–10 min.

- [ ] **Step 7: Eval CN-SIMCF output**

```bash
python scripts/evaluate_pseudolabel_quality.py \
  --pseudo_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf \
  --cityscapes_root ~/Desktop/datasets/cityscapes \
  --centroids_path ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80/kmeans_centroids.npz \
  --split train \
  --num_clusters 80 \
  --use_hungarian \
  --output reports/cn_simcf/T1_cn_simcf.json
```

- [ ] **Step 8: Decision gate T1**

Compare `T1_cn_simcf.json` summary vs `T0_baseline.json`. Compute ΔPQ, ΔPQ_things, ΔmIoU, and per-class deltas for the dead classes.

| Outcome | ΔPQ | Action |
|---|---|---|
| Strong | > +0.5 OR ΔPQ_things > +1.0 | proceed to T2 |
| Mild | (0.0, +0.5] | proceed to T2 (note as marginal) |
| Neutral | (-0.5, 0.0] | proceed to T2 (flag — investigate Step B threshold) |
| Regression | ≤ -0.5 | abort T1 branch, report which classes regressed |

Record decision in `reports/cn_simcf/T1_decision.md` (one paragraph).

---

## T2: Bootstrap Cluster Remapping (A2)

**Files:**
- Create: `scripts/bootstrap_cluster_remap.py`
- Output dirs:
  - `~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80_iter1/kmeans_centroids.npz` (refit mapping)
  - `~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf_iter1/`
  - `~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf_iter2/`

**Bootstrap principle (UNSUPERVISED):** Re-fit `cluster_to_class` from the *current pseudo-labels' depth + DCFA-feature signature*, not from GT.

For each cluster $c$:
1. Collect mean DINOv3 feature $\bar{f}_c$ across all pixels assigned to $c$ in the latest pseudo-labels.
2. Collect mean DepthPro depth $\bar{d}_c$ and depth std $\sigma^d_c$.
3. Compute a "class affinity" score: cosine similarity to the *cluster centroid* in feature space, AND distance to known per-class depth ranges.
4. Re-assign cluster → trainID via combined score, optionally allowing rare classes (currently dead) to claim clusters whose feature-depth signature is far from any current trainID's centroid.

This is unsupervised because the per-class centroids are computed from the *current* pseudo-labels themselves (no GT involved).

- [ ] **Step 1: Create `scripts/bootstrap_cluster_remap.py`**

```python
#!/usr/bin/env python3
"""Bootstrap unsupervised re-mapping of cluster_to_class.

Refits the cluster -> trainID lookup using the current pseudo-labels'
feature + depth signatures. Allows currently-dead trainIDs to reclaim
clusters whose signature is far from any active class.

NO GT IS USED. Only:
  - DINOv3 features (frozen)
  - DepthPro depth (frozen)
  - Current pseudo-label assignments
"""
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_CLASSES = 19
CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]


def collect_cluster_signatures(pseudo_dir: Path, feat_dir: Path, depth_dir: Path,
                                centroids_path: str, num_clusters: int = 80):
    """For each cluster, compute mean DINOv3 feature and mean depth."""
    cluster_feat_sum = np.zeros((num_clusters, 768), dtype=np.float64)
    cluster_depth_sum = np.zeros(num_clusters, dtype=np.float64)
    cluster_count = np.zeros(num_clusters, dtype=np.int64)

    stems = sorted([p.name.replace("_semantic.png", "")
                    for p in pseudo_dir.glob("*_semantic.png")])
    logger.info(f"Collecting signatures from {len(stems)} pseudo-labels")

    for stem in tqdm(stems, desc="Signatures"):
        city = stem.split("_")[0]
        sem = np.array(Image.open(pseudo_dir / f"{stem}_semantic.png"))

        feat_path = feat_dir / "train" / city / f"{stem}.npy"
        if not feat_path.exists():
            continue
        feat = np.load(str(feat_path)).astype(np.float32)
        feat_2d = feat.reshape(32, 64, -1)

        depth_base = stem.replace("_leftImg8bit", "")
        depth_path = depth_dir / "train" / city / f"{depth_base}.npy"
        if not depth_path.exists():
            depth_path = depth_dir / "train" / city / f"{stem}.npy"
        if not depth_path.exists():
            continue
        depth = np.load(str(depth_path)).astype(np.float32)

        # Resize semantic to feature resolution (32x64)
        sem_small = np.array(Image.fromarray(sem).resize((64, 32), Image.NEAREST))
        depth_small = np.array(
            Image.fromarray(depth).resize((64, 32), Image.BILINEAR)
        )

        for cl in range(num_clusters):
            mask = sem_small == cl
            if not mask.any():
                continue
            cluster_feat_sum[cl] += feat_2d[mask].sum(axis=0)
            cluster_depth_sum[cl] += depth_small[mask].sum()
            cluster_count[cl] += int(mask.sum())

    safe_count = np.maximum(cluster_count, 1)
    cluster_feat_mean = cluster_feat_sum / safe_count[:, None]
    cluster_depth_mean = cluster_depth_sum / safe_count
    feat_norms = np.linalg.norm(cluster_feat_mean, axis=1, keepdims=True) + 1e-8
    cluster_feat_unit = cluster_feat_mean / feat_norms
    return cluster_feat_unit, cluster_depth_mean, cluster_count


def refit_cluster_to_class(cluster_feat: np.ndarray, cluster_depth: np.ndarray,
                            cluster_count: np.ndarray, current_c2c: np.ndarray,
                            min_cluster_pixels: int = 100,
                            allow_dead_class_recovery: bool = True) -> np.ndarray:
    """Refit cluster -> trainID via per-class prototypes derived from current mapping.

    Step 1: Build per-class feature + depth prototypes from the CURRENT mapping.
    Step 2: For each cluster, score against all class prototypes.
    Step 3: If a cluster's best score is to a CURRENTLY DEAD class with a
            consistent feature/depth pattern, reassign it.
    """
    num_clusters = cluster_feat.shape[0]
    new_c2c = current_c2c.copy()

    # Build per-class prototypes (weighted by cluster size)
    class_feat = np.zeros((NUM_CLASSES, cluster_feat.shape[1]), dtype=np.float64)
    class_depth = np.zeros(NUM_CLASSES, dtype=np.float64)
    class_weight = np.zeros(NUM_CLASSES, dtype=np.float64)
    for cl in range(num_clusters):
        if cluster_count[cl] < min_cluster_pixels:
            continue
        tid = int(current_c2c[cl])
        if tid >= NUM_CLASSES:
            continue
        w = float(cluster_count[cl])
        class_feat[tid] += cluster_feat[cl] * w
        class_depth[tid] += cluster_depth[cl] * w
        class_weight[tid] += w

    safe_w = np.maximum(class_weight, 1.0)
    class_feat /= safe_w[:, None]
    class_depth /= safe_w
    norms = np.linalg.norm(class_feat, axis=1, keepdims=True) + 1e-8
    class_feat_unit = class_feat / norms

    dead_classes = np.where(class_weight == 0)[0].tolist()
    logger.info(f"Dead classes (no clusters mapped): {[CITYSCAPES_CLASS_NAMES[c] for c in dead_classes]}")

    # Score each cluster vs each ACTIVE class
    n_remapped = 0
    for cl in range(num_clusters):
        if cluster_count[cl] < min_cluster_pixels:
            continue
        feat_sims = cluster_feat[cl] @ class_feat_unit.T  # (NUM_CLASSES,)
        # Mask out dead classes (they have no prototype)
        feat_sims[class_weight == 0] = -1.0
        best_active = int(np.argmax(feat_sims))
        best_active_sim = float(feat_sims[best_active])

        # Reassign to best active class IF significantly better than current
        cur_tid = int(current_c2c[cl])
        if cur_tid < NUM_CLASSES and class_weight[cur_tid] > 0:
            cur_sim = float(feat_sims[cur_tid])
            if best_active_sim - cur_sim > 0.05:
                new_c2c[cl] = best_active
                n_remapped += 1
        else:
            new_c2c[cl] = best_active
            n_remapped += 1

    logger.info(f"Remapped {n_remapped}/{num_clusters} clusters")
    return new_c2c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo_dir", required=True,
                        help="Current pseudo-label dir (CN-SIMCF output)")
    parser.add_argument("--feat_dir", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--current_centroids", required=True,
                        help="Path to kmeans_centroids.npz with current cluster_to_class")
    parser.add_argument("--output_centroids", required=True)
    parser.add_argument("--num_clusters", type=int, default=80)
    args = parser.parse_args()

    data = np.load(args.current_centroids)
    centroids = data["centroids"] if "centroids" in data else None
    current_c2c = data["cluster_to_class"].astype(np.uint8)

    cluster_feat, cluster_depth, cluster_count = collect_cluster_signatures(
        Path(args.pseudo_dir).expanduser(),
        Path(args.feat_dir).expanduser(),
        Path(args.depth_dir).expanduser(),
        args.current_centroids,
        args.num_clusters,
    )

    new_c2c = refit_cluster_to_class(
        cluster_feat, cluster_depth, cluster_count, current_c2c,
    )

    out = {"cluster_to_class": new_c2c}
    if centroids is not None:
        out["centroids"] = centroids
    out["cluster_feat_mean"] = cluster_feat.astype(np.float32)
    out["cluster_depth_mean"] = cluster_depth.astype(np.float32)
    out["cluster_count"] = cluster_count
    np.savez(Path(args.output_centroids).expanduser(), **out)
    logger.info(f"Saved new centroids to {args.output_centroids}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run bootstrap iter 1 (refit on T1 output)**

```bash
mkdir -p ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80_iter1

python scripts/bootstrap_cluster_remap.py \
  --pseudo_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf \
  --feat_dir ~/Desktop/datasets/cityscapes/dinov3_features \
  --depth_dir ~/Desktop/datasets/cityscapes/depth_depthpro \
  --current_centroids ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80/kmeans_centroids.npz \
  --output_centroids ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80_iter1/kmeans_centroids.npz \
  --num_clusters 80 \
  2>&1 | tee logs/T2_bootstrap_iter1.log
```
Expected: ~3–5 min.

- [ ] **Step 3: Re-run CN-SIMCF with iter1 centroids**

```bash
python scripts/refine_cn_simcf.py \
  --input_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_adapter_V3_tau020 \
  --output_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf_iter1 \
  --centroids_path ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80_iter1/kmeans_centroids.npz \
  --cityscapes_root ~/Desktop/datasets/cityscapes \
  --features_subdir dinov3_features \
  --depth_subdir depth_depthpro \
  --steps A,B,C \
  --sim_threshold 0.92 \
  --num_clusters 80 \
  2>&1 | tee logs/T2_cn_simcf_iter1.log
```

- [ ] **Step 4: Eval iter1**

```bash
python scripts/evaluate_pseudolabel_quality.py \
  --pseudo_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf_iter1 \
  --cityscapes_root ~/Desktop/datasets/cityscapes \
  --centroids_path ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80_iter1/kmeans_centroids.npz \
  --split train \
  --num_clusters 80 \
  --use_hungarian \
  --output reports/cn_simcf/T2_bootstrap_iter1.json
```

- [ ] **Step 5: Iter 2 — repeat steps 2–4 with iter1 outputs as input**

```bash
mkdir -p ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80_iter2

python scripts/bootstrap_cluster_remap.py \
  --pseudo_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf_iter1 \
  --feat_dir ~/Desktop/datasets/cityscapes/dinov3_features \
  --depth_dir ~/Desktop/datasets/cityscapes/depth_depthpro \
  --current_centroids ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80_iter1/kmeans_centroids.npz \
  --output_centroids ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80_iter2/kmeans_centroids.npz \
  --num_clusters 80

python scripts/refine_cn_simcf.py \
  --input_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_adapter_V3_tau020 \
  --output_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf_iter2 \
  --centroids_path ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80_iter2/kmeans_centroids.npz \
  --cityscapes_root ~/Desktop/datasets/cityscapes \
  --features_subdir dinov3_features --depth_subdir depth_depthpro \
  --steps A,B,C --sim_threshold 0.92 --num_clusters 80

python scripts/evaluate_pseudolabel_quality.py \
  --pseudo_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf_iter2 \
  --cityscapes_root ~/Desktop/datasets/cityscapes \
  --centroids_path ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80_iter2/kmeans_centroids.npz \
  --split train --num_clusters 80 --use_hungarian \
  --output reports/cn_simcf/T2_bootstrap_iter2.json
```

- [ ] **Step 6: Decision gate T2**

Compare iter1, iter2 vs T1. Record per-class deltas.
- If iter2 > iter1 > T1 → bootstrap is converging, proceed to T3
- If iter1 > T1 but iter2 ≤ iter1 → bootstrap saturates at iter1, use iter1 going forward
- If iter1 ≤ T1 → bootstrap fails, skip T2 chain (use T1 output for T3)

Document outcome in `reports/cn_simcf/T2_decision.md`.

---

## T3: Multi-Source Rare-Pixel Protected Mask (A3)

**Files:**
- Create: `scripts/build_rare_protected_mask.py`
- Modify: `scripts/refine_cn_simcf.py` (add `--protected_mask_dir` arg)
- Output: `~/Desktop/datasets/cityscapes/rare_protected_masks/<city>/*.png` (binary masks at 32×64, upsampled to image res)
- Final: `~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf_protected/`

**4 source signals (independent, each at 32×64 patch grid):**
1. **DCFA disagreement** — pixels where DCFA-adapted cluster ≠ raw-DINOv3 cluster
2. **k=300 small-cluster** — pixels in clusters of size < 0.5% of image (rare-by-frequency)
3. **DINOv3 [CLS] attention peak** — top 5% attention weight, NOT in dominant local cluster
4. **Depth edge** — Sobel depth edge intersecting a thin (<20 px) vertical band

**Protected if 3 of 4 agree.**

- [ ] **Step 1: Create `scripts/build_rare_protected_mask.py`**

```python
#!/usr/bin/env python3
"""Build 3-of-4 multi-source rare-pixel protected mask.

Sources (all unsupervised, all 32x64 patch resolution):
  S1: DCFA cluster differs from raw-DINOv3 cluster
  S2: k=300 cluster size < 0.5% of image (rare by frequency)
  S3: DINOv3 [CLS] attention in top-5%, but pixel NOT in dominant local cluster
  S4: Depth edge in thin (<20 px wide) vertical band

A pixel is rare-protected if 3 of 4 sources agree.
"""
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEAT_H, FEAT_W = 32, 64


def source_1_dcfa_disagreement(dcfa_sem: np.ndarray, raw_sem: np.ndarray) -> np.ndarray:
    """DCFA cluster differs from raw-DINOv3 cluster at this pixel."""
    return dcfa_sem != raw_sem


def source_2_k300_small_cluster(k300_sem: np.ndarray, image_pixels: int,
                                 frac_threshold: float = 0.005) -> np.ndarray:
    """k=300 cluster size < frac_threshold of total image pixels."""
    out = np.zeros_like(k300_sem, dtype=bool)
    unique, counts = np.unique(k300_sem, return_counts=True)
    threshold = image_pixels * frac_threshold
    for u, c in zip(unique, counts):
        if 0 < c < threshold:
            out |= (k300_sem == u)
    return out


def source_3_attention_off_cluster(attn_map: np.ndarray, sem: np.ndarray,
                                    top_pct: float = 0.05,
                                    local_radius: int = 2) -> np.ndarray:
    """High [CLS] attention but disagrees with local dominant cluster."""
    threshold = np.quantile(attn_map, 1.0 - top_pct)
    high_attn = attn_map >= threshold

    # Local dominant cluster (mode in 5x5 neighborhood)
    h, w = sem.shape
    mode_map = np.zeros_like(sem)
    for i in range(h):
        for j in range(w):
            i0, i1 = max(0, i - local_radius), min(h, i + local_radius + 1)
            j0, j1 = max(0, j - local_radius), min(w, j + local_radius + 1)
            patch = sem[i0:i1, j0:j1].ravel()
            patch = patch[patch < 256]
            if len(patch) > 0:
                vals, counts = np.unique(patch, return_counts=True)
                mode_map[i, j] = vals[counts.argmax()]
    return high_attn & (sem != mode_map)


def source_4_depth_thin_band(depth: np.ndarray, edge_thresh: float = 0.20,
                              max_band_width: int = 20) -> np.ndarray:
    """Depth edge intersecting thin (<max_band_width px) vertical band."""
    sx = ndimage.sobel(depth, axis=0)
    sy = ndimage.sobel(depth, axis=1)
    grad = np.sqrt(sx ** 2 + sy ** 2)
    edges = grad > edge_thresh

    # For each row, find horizontal runs of edges narrower than max_band_width
    out = np.zeros_like(edges)
    for i in range(edges.shape[0]):
        row = edges[i]
        labeled, n = ndimage.label(row)
        for k in range(1, n + 1):
            run = labeled == k
            if 0 < run.sum() < max_band_width:
                out[i, run] = True
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes_root", required=True)
    parser.add_argument("--dcfa_pseudo_dir", required=True,
                        help="DCFA semantic pseudo-labels (k=80)")
    parser.add_argument("--raw_pseudo_dir", required=True,
                        help="Raw DINOv3 semantic pseudo-labels (k=80, no DCFA)")
    parser.add_argument("--k300_pseudo_dir", required=True,
                        help="k=300 overclustered pseudo-labels")
    parser.add_argument("--attn_dir", required=True,
                        help="DINOv3 [CLS] attention maps (.npy at 32x64)")
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    dcfa_dir = Path(args.dcfa_pseudo_dir).expanduser() / args.split
    raw_dir = Path(args.raw_pseudo_dir).expanduser() / args.split
    k300_dir = Path(args.k300_pseudo_dir).expanduser() / args.split
    attn_dir = Path(args.attn_dir).expanduser() / args.split
    depth_dir = Path(args.depth_dir).expanduser() / args.split

    n_done = 0
    for city_dir in sorted(dcfa_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        city = city_dir.name
        out_city = output_dir / city
        out_city.mkdir(parents=True, exist_ok=True)
        for dcfa_path in tqdm(sorted(city_dir.glob("*.png")), desc=city):
            stem = dcfa_path.stem
            try:
                dcfa = np.array(Image.open(dcfa_path))
                raw = np.array(Image.open(raw_dir / city / f"{stem}.png"))
                k300 = np.array(Image.open(k300_dir / city / f"{stem}.png"))
                attn = np.load(str(attn_dir / city / f"{stem}.npy"))
                depth = np.load(str(depth_dir / city / f"{stem}.npy"))
            except FileNotFoundError as e:
                logger.warning(f"Missing source for {stem}: {e}")
                continue

            # Resize all to feature grid
            dcfa_s = np.array(Image.fromarray(dcfa).resize((FEAT_W, FEAT_H), Image.NEAREST))
            raw_s = np.array(Image.fromarray(raw).resize((FEAT_W, FEAT_H), Image.NEAREST))
            k300_s = np.array(Image.fromarray(k300).resize((FEAT_W, FEAT_H), Image.NEAREST))
            depth_s = np.array(Image.fromarray(depth.astype(np.float32))
                               .resize((FEAT_W, FEAT_H), Image.BILINEAR))
            if attn.shape != (FEAT_H, FEAT_W):
                attn = np.array(Image.fromarray(attn.astype(np.float32))
                                .resize((FEAT_W, FEAT_H), Image.BILINEAR))

            s1 = source_1_dcfa_disagreement(dcfa_s, raw_s)
            s2 = source_2_k300_small_cluster(k300_s, FEAT_H * FEAT_W)
            s3 = source_3_attention_off_cluster(attn, dcfa_s)
            s4 = source_4_depth_thin_band(depth_s)

            agreement = s1.astype(np.uint8) + s2.astype(np.uint8) \
                        + s3.astype(np.uint8) + s4.astype(np.uint8)
            protected = agreement >= 3

            # Upsample to original image resolution
            protected_full = np.array(
                Image.fromarray(protected.astype(np.uint8) * 255)
                .resize((dcfa.shape[1], dcfa.shape[0]), Image.NEAREST)
            ) > 0
            Image.fromarray(protected_full.astype(np.uint8) * 255).save(
                str(out_city / f"{stem}.png")
            )
            n_done += 1
    logger.info(f"Wrote {n_done} protected masks to {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `--protected_mask_dir` to `refine_cn_simcf.py`**

In `main()`, add:
```python
parser.add_argument("--protected_mask_dir", type=str, default=None,
                    help="If set, pixels in protected mask are exempt from Step A overwrite, "
                         "Step B merging, and Step C outlier rejection.")
```

In the per-image loop, load the protected mask if provided:
```python
protected_mask = None
if args.protected_mask_dir:
    pm_path = Path(args.protected_mask_dir).expanduser() / "train" / city / f"{cups_stem}.png"
    if pm_path.exists():
        protected_mask = np.array(Image.open(pm_path)) > 0
        if protected_mask.shape != semantic.shape:
            protected_mask = np.array(
                Image.fromarray(protected_mask.astype(np.uint8) * 255)
                .resize((semantic.shape[1], semantic.shape[0]), Image.NEAREST)
            ) > 0
```

Modify Step A to skip protected pixels:
```python
# Inside step_a, after computing inconsistent:
if protected_mask is not None:
    # Mask within current instance
    inst_protected = protected_mask[ys, xs]
    inconsistent = inconsistent & (~inst_protected)
```
(Pass `protected_mask` as new arg.)

Modify Step B to NOT merge across protected boundary:
```python
# Inside step_b, when checking merge_pairs, skip if either instance overlaps protected pixels:
if protected_mask is not None:
    if protected_mask[instance == i].any() or protected_mask[instance == j].any():
        continue
```

Modify Step C to use 5σ for ALL protected pixels (overrides per-cluster σ):
```python
# Inside step_c:
if protected_mask is not None:
    # Override outlier flag for protected pixels
    protected_in_cluster = protected_mask[ys[outlier], xs[outlier]]
    outlier[protected_in_cluster] = False
```

- [ ] **Step 3: Generate the protected masks**

```bash
mkdir -p logs reports/cn_simcf

# First confirm we have all 4 inputs:
ls ~/Desktop/datasets/cityscapes/pseudo_semantic_adapter_V3_k80/train/ | head -3
ls ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80/train/ | head -3
ls ~/Desktop/datasets/cityscapes/pseudo_semantic_adapter_c_lp1.0_k300/train/ 2>/dev/null | head -3 || echo "k300 dir missing — needs to be generated first"
ls ~/Desktop/datasets/cityscapes/dinov3_attention/ 2>/dev/null | head -3 || echo "attention dir missing — needs to be generated first"

# If attn or k300 missing, generate them (separate scripts — see notes below)

python scripts/build_rare_protected_mask.py \
  --cityscapes_root ~/Desktop/datasets/cityscapes \
  --dcfa_pseudo_dir ~/Desktop/datasets/cityscapes/pseudo_semantic_adapter_V3_k80 \
  --raw_pseudo_dir ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80 \
  --k300_pseudo_dir ~/Desktop/datasets/cityscapes/pseudo_semantic_adapter_c_lp1.0_k300 \
  --attn_dir ~/Desktop/datasets/cityscapes/dinov3_attention \
  --depth_dir ~/Desktop/datasets/cityscapes/depth_depthpro \
  --output_dir ~/Desktop/datasets/cityscapes/rare_protected_masks \
  --split train \
  2>&1 | tee logs/T3_protected_masks.log
```

**If `dinov3_attention` is missing**, generate via a small helper that runs DINOv3 forward and saves [CLS]-token attention from last layer (separate task — skip if attention is unavailable, in which case use 2-of-3 of the remaining sources).

- [ ] **Step 4: Re-run CN-SIMCF with protected mask using the BEST T2 centroids**

Use whichever centroids file (T1 or T2 iter1/iter2) gave the best PQ, recorded in T2 decision.

```bash
BEST_CENTROIDS=~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80_iter2/kmeans_centroids.npz   # adjust based on T2 result

python scripts/refine_cn_simcf.py \
  --input_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_adapter_V3_tau020 \
  --output_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf_protected \
  --centroids_path $BEST_CENTROIDS \
  --cityscapes_root ~/Desktop/datasets/cityscapes \
  --features_subdir dinov3_features --depth_subdir depth_depthpro \
  --steps A,B,C --sim_threshold 0.92 --num_clusters 80 \
  --protected_mask_dir ~/Desktop/datasets/cityscapes/rare_protected_masks \
  2>&1 | tee logs/T3_cn_simcf_protected.log
```

- [ ] **Step 5: Eval T3**

```bash
python scripts/evaluate_pseudolabel_quality.py \
  --pseudo_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf_protected \
  --cityscapes_root ~/Desktop/datasets/cityscapes \
  --centroids_path $BEST_CENTROIDS \
  --split train --num_clusters 80 --use_hungarian \
  --output reports/cn_simcf/T3_protected.json
```

- [ ] **Step 6: Decision gate T3**

Same gate as before. Record:
- ΔPQ_things on caravan, trailer (target +5 PQ each)
- ΔPQ_stuff on guard rail, tunnel, polegroup (target +3 PQ each)
- mIoU change (small ±1 expected)

Final report in `reports/cn_simcf/T3_decision.md`.

---

## T4 (DEFERRED — REMOTE): Class-Balanced Stage-2 Training

Not run locally. Use whichever pseudo-label directory wins T1–T3 as the Stage-2 training source. Apply:
- Class-balanced sampler (1/sqrt(freq) weighting)
- EQLv2 loss in Cascade Mask R-CNN box/cls heads from epoch 0
- ROI retention: keep all rare-class ROIs even at low confidence

Document remote execution plan in a separate `docs/plans/2026-04-XX-stage2-class-balanced.md` after local A1–A3 lands.

---

## Final Aggregate Report

After T0–T3 complete, generate `reports/cn_simcf/SUMMARY.md` with:

| Step | PQ | PQ_stuff | PQ_things | mIoU | Δ vs T0 | Decision |
|---|---:|---:|---:|---:|---:|---|
| T0 baseline | 25.85 | 33.96 | 14.70 | 56.22 | — | — |
| T1 CN-SIMCF | ? | ? | ? | ? | ? | ? |
| T2 +bootstrap iter1 | ? | ? | ? | ? | ? | ? |
| T2 +bootstrap iter2 | ? | ? | ? | ? | ? | ? |
| T3 +protected mask | ? | ? | ? | ? | ? | ? |

Plus a per-class PQ delta table for the 5 dead classes. Conclude with a recommendation: which configuration to send to remote for Stage-2 retraining.

---

## Self-Review Checklist

- [x] Spec coverage: all 3 local ablations have tasks. A4 explicitly marked deferred.
- [x] Placeholder scan: code blocks complete; no TBD/TODO inside steps.
- [x] Type consistency: `cluster_to_class` shape (256,), `num_clusters=80`, sigma_threshold names match across CN-SIMCF and bootstrap.
- [x] Decision gates: explicit per-task with thresholds.
- [x] Eval contract: same `evaluate_pseudolabel_quality.py` invocation per step.

## Execution Choice

This will be executed inline in the current session using `superpowers:executing-plans`, with checkpoints between T1, T2, and T3 for the user to review. Each task has 5–8 steps. Total local wall time estimate: ~90–120 minutes (mostly eval runs, ~10 min each, ×5 evals).
