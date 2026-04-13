# Plan: 5 Novel Instance Decomposition Approaches — DA3 Ablation Study

## Context
Current best instance decomposition: DA3 (Depth Anything V3) Sobel+CC at τ=0.03, A_min=1000 → **PQ_things=20.90** on Cityscapes val (500 images). Bottleneck: co-planar objects (person PQ=6.36, motorcycle PQ=0.0). Learned merge already failed (19.92). Need novel approaches grounded in 2024-2026 CV research to push PQ_things beyond 21.5 for NeurIPS 2026.

**Prior work (all evaluated, none beat DA3 Sobel+CC on DA3 depth):**
- Morse flow: 16.66 | TDA persistence: 16.70 | Mumford-Shah spectral: 18.71
- Optimal transport: 2.45 | Contrastive embed: 6.78 | Learned merge: 19.92
- All above used SPIdepth depth. No alternative method has been evaluated with DA3 depth yet.

---

## Execution Environment

```
PYTHON=/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python
CS_ROOT=/Users/qbit-glitch/Desktop/datasets/cityscapes
DEPTH_SUB=depth_dav3
SEM_SUB=pseudo_semantic_raw_dinov3_k80
FEAT_SUB=dinov2_features
CENTROIDS=$CS_ROOT/$SEM_SUB/kmeans_centroids.npz
```

## Output Contract (all methods)

```python
def xxx_instances(semantic, depth, thing_ids, ..., features=None) -> List[(mask, class_id, score)]
```
- `semantic`: (H,W) uint8 trainID
- `depth`: (H,W) float32 [0,1]
- `features`: (32, 64, 768) float32 DINOv2 patches
- Returns: list of (bool mask (H,W), int class_id, float score)

### Reusable Infrastructure
- **Eval framework**: `ablate_instance_methods.py` — discover_files, evaluate_panoptic_single, compute_pq_from_accumulators, expand_grid
- **Method registry**: `instance_methods/__init__.py` — METHODS dict
- **Shared utils**: `instance_methods/utils.py` — dilation_reclaim(), load_features(), cosine_similarity_regions(), upsample_features()
- **Sobel+CC**: `instance_methods/sobel_cc.py` — Stage 1 oversegmentation
- **Spectral clustering**: `instance_methods/mumford_shah.py` — _build_affinity_matrix(), spectral clustering
- **Depth generation**: `generate_depth_multimodel.py` — supports depthpro model (already integrated)
- **UNet reference**: `refine_net.py` — DepthGuidedUNet architecture
- **Training pattern**: `train_merge_predictor.py` — multi-threshold self-supervision

---

## The 5 Approaches (Novelty Rank Order)

### Approach #1: DINOv2 Feature Gradient Edges + Depth Fusion
**Research**: DiffCut (NeurIPS 2024), feature-guided boundary detection
**Novelty**: HIGH — first to combine DINOv2 feature spatial gradients with depth edges for unsupervised instance splitting
**Why it targets the bottleneck**: Co-planar objects (person, car) have identical depth but DIFFERENT DINOv2 features. Feature gradients detect boundaries that depth Sobel misses.

**Algorithm**:
1. Load DINOv2 features (32×64×768), PCA reduce to (32×64×64)
2. Upsample to eval resolution (512×1024×64) via bilinear
3. Compute spatial gradients: feature_grad = max_channel(|∂f/∂x|, |∂f/∂y|)
4. Compute depth gradients: depth_grad = sqrt(gx² + gy²) (existing Sobel)
5. Fuse: combined_edges = (feature_grad > τ_feat) | (depth_grad > τ_depth)
6. Standard CC pipeline on fused edge map
7. dilation_reclaim() post-processing

**Files**:
- CREATE: `instance_methods/feature_edge_cc.py` (~120 lines)
- MODIFY: `instance_methods/__init__.py` (+2 lines)

**Default config**:
- `feat_grad_threshold`: 0.15
- `depth_grad_threshold`: 0.03 (matches DA3 optimal)
- `fusion_mode`: "union"
- `min_area`: 1000
- `pca_dim`: 64

**Compute**: ~0.1s/image × 500 = ~1 min
**Dependencies**: None (numpy, scipy, PIL — all installed)

---

### Approach #2: Depth-Feature Joint NCut (Recursive Normalized Cut)
**Research**: DiffCut (NeurIPS 2024), Spectral Grouping (Shi & Malik extended)
**Novelty**: HIGH — recursive NCut on joint depth-feature affinity, principled graph-theoretic decomposition
**Why**: Spectral methods reason about global connectivity, not just local edges. Can separate co-planar objects if they differ in appearance.

**Algorithm**:
1. Work at feature resolution (32×64) for tractability
2. Per thing-class region: build joint affinity matrix
   - A[i,j] = exp(-α|d_i-d_j|²/σ_d²) × exp(-β||f_i-f_j||²/σ_f²) × spatial_decay(dist)
3. Compute normalized Laplacian: L_norm = I - D^(-1/2) A D^(-1/2)
4. Find 2nd eigenvector (Fiedler vector) via `scipy.sparse.linalg.eigsh`
5. Bipartition by thresholding Fiedler vector
6. Evaluate NCut cost; if cost < τ_ncut, accept split and recurse on each partition
7. Map patch-level instances to pixel-level via nearest-neighbor upsampling
8. dilation_reclaim()

**Reuse**: `mumford_shah.py` `_build_affinity_matrix()` (extend with recursive NCut logic)

**Files**:
- CREATE: `instance_methods/joint_ncut.py` (~200 lines)
- MODIFY: `instance_methods/__init__.py` (+2 lines)

**Default config**:
- `alpha` (depth weight): 1.0
- `beta` (feature weight): 1.0
- `ncut_threshold`: 0.05
- `min_area`: 1000
- `sigma_d, sigma_f`: auto (median heuristic)

**Compute**: ~2-5s/image × 500 = ~25 min
**Dependencies**: scipy.sparse.linalg (installed)

---

### Approach #3: Learned Depth Edge Detector (Self-Supervised)
**Research**: Multi-scale self-supervised edge learning, HED-style boundary detection
**Novelty**: MEDIUM-HIGH — self-supervised boundary prediction without any annotations, trained via depth multi-threshold consensus
**Why**: A neural edge detector can learn non-linear boundary signals that Sobel cannot capture. Combines depth geometry + DINOv2 appearance + spatial context.

**Algorithm — Training**:
1. Generate edge labels via multi-threshold consensus on DA3 depth:
   - Run Sobel at τ_low=0.02 → fine edges (many false positives)
   - Run Sobel at τ_high=0.05 → coarse edges (high precision)
   - Label: pixel is "true edge" if edge at τ_high, "non-edge" if not edge at τ_low
   - Pixels between thresholds: ignored (ambiguous)
2. Input channels: depth (1) + Sobel gx,gy (2) + PCA-reduced DINOv2 features (64) = 67 channels
3. Architecture: lightweight 4-layer ConvNet (NOT full UNet — keep fast)
   - Conv(67→64, 3×3) → ReLU → Conv(64→64, 3×3) → ReLU → Conv(64→32, 3×3) → ReLU → Conv(32→1, 1×1) → Sigmoid
   - ~150K parameters, runs at 512×1024
4. Loss: BCE with class balancing (edges are ~5% of pixels)
5. Train: 20 epochs on 2975 Cityscapes train images, Adam lr=1e-3

**Algorithm — Inference**:
1. Run trained edge detector → per-pixel edge probability
2. Threshold: edge_prob > τ_edge → binary edge map
3. Standard CC pipeline on edge map
4. dilation_reclaim()

**Files**:
- CREATE: `instance_methods/learned_edge_cc.py` (~150 lines)
- CREATE: `train_learned_edge.py` (~250 lines)
- MODIFY: `instance_methods/__init__.py` (+2 lines)

**Training config**: depth + Sobel + DINOv2 PCA (67ch), 20 epochs, Adam lr=1e-3

**Default inference config**:
- `edge_threshold`: 0.5
- `min_area`: 1000
- `dilation_iters`: 3

**Compute**: Training ~5 min. Inference: ~1 min (500 images).
**Dependencies**: PyTorch (installed)

---

### Approach #4: Apple Depth Pro Swap
**Research**: Depth Pro (ICLR 2025, Apple) — boundary-optimized monocular depth
**Novelty**: MEDIUM — model comparison, but DepthPro's boundary metric optimization directly addresses our edge quality bottleneck
**Why**: If DepthPro produces sharper depth edges between objects, Sobel+CC may work better out of the box. Quick A/B test.

**Algorithm**:
1. Generate DepthPro depth maps using existing `generate_depth_multimodel.py --model depthpro`
2. Run standard Sobel+CC threshold sweep
3. Compare vs DA3 per-class

**Files**:
- No new instance method files needed
- Run existing Sobel+CC with `--depth_subdir depth_depthpro`

**Step 1: Generate depth maps** (~30 min for 500 val images):
```bash
nohup $PYTHON -u mbps_pytorch/generate_depth_multimodel.py \
    --cityscapes_root $CS_ROOT --model depthpro --split val \
    > logs/generate_depthpro.log 2>&1 &
```

**Step 2: Evaluate** (~1 min):
```bash
nohup $PYTHON -u mbps_pytorch/ablate_instance_methods.py \
    --cityscapes_root $CS_ROOT --method sobel_cc \
    --depth_subdir depth_depthpro --semantic_subdir $SEM_SUB \
    --centroids_path $CENTROIDS \
    > logs/eval_depthpro.log 2>&1 &
```

**Default config**: `grad_threshold`: 0.03, `min_area`: 1000 (same as DA3 optimal)

**Compute**: Generate ~30 min + eval ~1 min = ~31 min total
**Dependencies**: `transformers` (installed), DepthPro HF model auto-downloaded

---

### Approach #5: Local Plane Decomposition (GeoDepth-inspired)
**Research**: GeoDepth (CVPR 2025) — depth discontinuity awareness via plane fitting
**Novelty**: MEDIUM — geometric plane-based scene decomposition, novel for unsupervised instance segmentation
**Why**: Objects in driving scenes occupy distinct depth planes. Adjacent patches on different planes → instance boundary. Complementary to gradient-based edges.

**Algorithm**:
1. Divide depth map into patches (e.g., 16×16 pixels)
2. Per patch: fit plane z = ax + by + c via SVD (least squares)
3. Compute per-patch surface normal: n = (-a, -b, 1) / ||(-a, -b, 1)||
4. For adjacent patch pairs: compute normal angle = arccos(n1 · n2)
5. Also compute depth residual: mean |z_actual - z_fitted| per patch
6. Boundary map: patches where normal_angle > τ_angle OR residual > τ_residual
7. Upsample patch-level boundary to pixel-level
8. Standard CC pipeline on boundary map
9. dilation_reclaim()

**Files**:
- CREATE: `instance_methods/plane_decomp.py` (~180 lines)
- MODIFY: `instance_methods/__init__.py` (+2 lines)

**Default config**:
- `patch_size`: 16
- `normal_angle_threshold`: 15 (degrees)
- `residual_threshold`: 0.02
- `min_area`: 1000

**Compute**: ~0.05s/image × 500 ≈ 30s
**Dependencies**: numpy (installed)

---

## Execution Order & Timeline

### Phase A: Implement + Evaluate (All in one pass)

| Step | Approach | Implement | Eval | Total |
|------|----------|-----------|------|-------|
| A1 | #1 Feature Gradient Edges | ~30 min | ~1 min | ~31 min |
| A2 | #2 Joint NCut | ~45 min | ~25 min | ~70 min |
| A3 | #3 Learned Edge Detector | ~45 min | ~6 min | ~51 min |
| A4 | #4 DepthPro Generation + Eval | ~5 min | ~31 min | ~36 min |
| A5 | #5 Plane Decomposition | ~30 min | ~30s | ~31 min |

**Total**: ~4 hours (implement all 5 methods + evaluate each once with default config).

### Phase B: Analysis

| Step | Task |
|------|------|
| B1 | Compare all 5 approaches + DA3 baseline in unified table |
| B2 | Per-class analysis (person, car, bicycle) |
| B3 | Update `reports/novel_instance_ablation_final_report.md` |

---

## Execution Commands

```bash
PYTHON=/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python
CS_ROOT=/Users/qbit-glitch/Desktop/datasets/cityscapes

# A1: Feature Gradient Edges (Approach #1)
nohup $PYTHON -u mbps_pytorch/ablate_instance_methods.py \
    --cityscapes_root $CS_ROOT --method feature_edge_cc \
    --depth_subdir depth_dav3 --semantic_subdir pseudo_semantic_raw_dinov3_k80 \
    --centroids_path $CENTROIDS \
    > logs/eval_feature_edge_cc.log 2>&1 &

# A2: Joint NCut (Approach #2)
nohup $PYTHON -u mbps_pytorch/ablate_instance_methods.py \
    --cityscapes_root $CS_ROOT --method joint_ncut \
    --depth_subdir depth_dav3 --semantic_subdir pseudo_semantic_raw_dinov3_k80 \
    --centroids_path $CENTROIDS \
    > logs/eval_joint_ncut.log 2>&1 &

# A3: Train + Evaluate Learned Edge Detector (Approach #3)
nohup $PYTHON -u mbps_pytorch/train_learned_edge.py \
    --cityscapes_root $CS_ROOT --epochs 20 \
    --depth_subdir depth_dav3 --semantic_subdir pseudo_semantic_raw_dinov3_k80 \
    > logs/train_learned_edge.log 2>&1 &
# Then evaluate:
nohup $PYTHON -u mbps_pytorch/ablate_instance_methods.py \
    --cityscapes_root $CS_ROOT --method learned_edge_cc \
    --depth_subdir depth_dav3 --semantic_subdir pseudo_semantic_raw_dinov3_k80 \
    --centroids_path $CENTROIDS \
    > logs/eval_learned_edge_cc.log 2>&1 &

# A4: Generate DepthPro maps + Evaluate (Approach #4)
nohup $PYTHON -u mbps_pytorch/generate_depth_multimodel.py \
    --cityscapes_root $CS_ROOT --model depthpro --split val \
    > logs/generate_depthpro.log 2>&1 &
# Then evaluate:
nohup $PYTHON -u mbps_pytorch/ablate_instance_methods.py \
    --cityscapes_root $CS_ROOT --method sobel_cc \
    --depth_subdir depth_depthpro --semantic_subdir pseudo_semantic_raw_dinov3_k80 \
    --centroids_path $CENTROIDS \
    > logs/eval_depthpro.log 2>&1 &

# A5: Plane Decomposition (Approach #5)
nohup $PYTHON -u mbps_pytorch/ablate_instance_methods.py \
    --cityscapes_root $CS_ROOT --method plane_decomp \
    --depth_subdir depth_dav3 --semantic_subdir pseudo_semantic_raw_dinov3_k80 \
    --centroids_path $CENTROIDS \
    > logs/eval_plane_decomp.log 2>&1 &
```

---

## Files to Create/Modify

| File | Action | Lines | Purpose |
|------|--------|-------|---------|
| `instance_methods/feature_edge_cc.py` | CREATE | ~120 | Approach #1: DINOv2 feature gradients + depth edges → CC |
| `instance_methods/joint_ncut.py` | CREATE | ~200 | Approach #2: Recursive NCut on joint affinity |
| `instance_methods/learned_edge_cc.py` | CREATE | ~150 | Approach #3: Learned edge inference |
| `instance_methods/plane_decomp.py` | CREATE | ~180 | Approach #5: Local plane fitting → boundary |
| `train_learned_edge.py` | CREATE | ~250 | Approach #3: Self-supervised training |
| `instance_methods/__init__.py` | MODIFY | +8 | Register 4 new methods |
| `ablate_instance_methods.py` | MODIFY | +20 | Add default configs for 4 new methods |

**No new files for Approach #4** (DepthPro) — uses existing sobel_cc with `--depth_subdir depth_depthpro`.

---

## Evaluation Protocol

### Fairness Constraints (all methods, identical setup)
- **Depth**: DA3 (`depth_dav3/`) — except Approach #4 which tests DepthPro
- **Semantics**: DINOv3 k=80 overclustering (`pseudo_semantic_raw_dinov3_k80/`)
- **Features**: DINOv2 ViT-B/14 patches (`dinov2_features/`)
- **Resolution**: 512×1024
- **Post-processing**: dilation_iters=3, same min_area
- **Evaluation**: 500 Cityscapes val images, 19-class PQ/PQ_things/PQ_stuff

### Output
Each config → JSON in `results/ablation_instance_methods/ablation_{method}_{suffix}.json`

---

## Success Criteria

| Target | PQ_things | Meaning |
|--------|-----------|---------|
| **Kill** | < 20.0 | Worse than DA3 baseline (20.90) → stop |
| **Minimum** | ≥ 21.0 | Marginal improvement (+0.1 absolute) |
| **Good** | ≥ 21.5 | Target met, NeurIPS ablation table entry |
| **Excellent** | ≥ 22.0 | Strong novel contribution |
| **Person PQ** | ≥ 8.0 | Doubles baseline (6.36), co-planar fix validated |

---

## DA3 Baseline Reference (for calibration)

| τ | A_min | PQ_things | inst/img | Source |
|---|-------|-----------|----------|--------|
| 0.01 | 1000 | 19.90 | 5.0 | depth_model_ablation |
| 0.02 | 1000 | 20.68 | 4.6 | depth_model_ablation |
| **0.03** | **1000** | **20.90** | **4.4** | **depth_model_ablation (BEST)** |
| 0.05 | 1000 | 20.51 | 4.3 | depth_model_ablation |
| 0.10 | 1000 | 19.38 | 4.2 | depth_model_ablation |

Per-class (DA3 τ=0.03):
- person PQ=6.36, car PQ=17.38, bicycle PQ=?, rider PQ=12.12
- truck PQ=34.82, bus PQ=50.12, train PQ=36.43, motorcycle PQ=0.0

---

## Prior Completed Work (Reference)

### Phase 1-2: SPIdepth Ablation — COMPLETE (2026-03-31)
244 configs across 6 methods (morse, tda, ot, mumford_shah, contrastive, learned_merge) on SPIdepth.
No method beat Sobel+CC (PQ_th=19.41). Mumford-Shah closest at 18.71.
Report: `reports/novel_instance_ablation_final_report.md`

### Phase 3: Contrastive Embedding — FAILED (2026-03-31)
InfoNCE + L2-normalized embeddings + HDBSCAN = wrong formulation.
Best PQ_things=7.32 (untrained), declined to 4.26 after training.
Key lesson: Embedding→HDBSCAN is wrong; boundary→CC is the only proven pipeline.

### Phase 4: Learned Fragment Merge — FAILED (2026-04-01)
85.8% pair accuracy but PQ_things=19.92 on DA3 (worse than 20.90 baseline).
Confirmed: merge doesn't help regardless of depth source.

### Depth Model Ablation — COMPLETE (2026-03-31)
DA3 best (PQ_th=20.90, τ=0.03) > DA2-L (20.20) > SPIdepth (19.41).
Report: `reports/depth_model_ablation_study.md`

---

## Verification Checklist
1. Each new method produces valid instance masks (spot-check 5 images)
2. All JSON results match schema of existing ablation JSONs
3. DA3 Sobel+CC baseline reproduces PQ_things=20.90 through ablate script
4. Per-class breakdown shows person/car direction
5. Final comparison table: all 5 approaches + DA3 baseline + prior 6 methods
