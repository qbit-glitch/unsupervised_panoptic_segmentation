# Plan: Novel Instance Decomposition Ablation Study

## Context
NeurIPS review audit W1 identifies Sobel+CC as "textbook image processing with zero algorithmic novelty." We need principled replacements. This plan covers implementing 6 novel methods + 1 combined method as ablations, evaluated on Cityscapes val (500 images) against the Sobel+CC baseline (PQ=26.74, PQ_things=19.41).

The brainstorm is saved at `reports/novel_instance_decomposition_brainstorm.md`.

---

## Architecture: Unified Script + Method Modules

### File Structure
```
mbps_pytorch/instance_methods/
  ├── __init__.py
  ├── utils.py              # Shared: dilation reclaim, feature loading, save/load
  ├── sobel_cc.py           # Refactored baseline from generate_depth_guided_instances.py
  ├── morse_flow.py         # Method 6: Watershed + feature merge
  ├── tda_persistence.py    # Method 1: Persistent homology
  ├── optimal_transport.py  # Method 5: Sinkhorn instance assignment
  ├── mumford_shah.py       # Method 2: Graph-cut energy minimization
  ├── contrastive_embed.py  # Method 4: Learned embedding + HDBSCAN
  └── slot_attention.py     # Method 3: Depth-conditioned slot attention

mbps_pytorch/ablate_instance_methods.py  # Unified sweep/eval CLI
mbps_pytorch/train_contrastive_embed.py  # Training for Method 4
mbps_pytorch/train_slot_instances.py     # Training for Method 3
```

### Output Contract (all methods)
Every method implements:
```python
def xxx_instances(semantic, depth, thing_ids, ..., features=None) -> List[(mask, class_id, score)]
```
- `semantic`: (H,W) uint8 trainID
- `depth`: (H,W) float32 [0,1]
- `features`: (2048, 768) float16 (optional, for feature-aware methods)
- Returns: list of (bool mask (H,W), int class_id, float score)

### Unified CLI
```bash
python mbps_pytorch/ablate_instance_methods.py \
    --cityscapes_root /path/to/cityscapes \
    --method morse \
    --split val \
    --sweep \
    --output_dir results/ablation_instance_methods/
```

### Critical Reusable Code
- **Eval loop template**: `mbps_pytorch/sweep_k50_spidepth.py` — `run_single_config()`, `evaluate_panoptic_single()`, `compute_pq_from_accumulators()`
- **Sobel+CC baseline**: `mbps_pytorch/generate_depth_guided_instances.py:75-154` — input/output contract + dilation reclamation
- **Feature loading**: `mbps_pytorch/generate_feature_depth_instances.py` — DINOv2 feature loading + upsampling
- **HDBSCAN/clustering**: `mbps_pytorch/generate_dino_cluster_instances.py:109-199` — `get_patch_features_for_cc()`, `assign_pixels_to_clusters()`
- **PQ metric**: `mbps_pytorch/evaluation/panoptic_quality.py`

---

## Implementation Order (4 Phases)

### Phase 1: Quick Wins (Days 1-2)

#### Step 1: Scaffold — `instance_methods/` package + unified eval script
- Create `__init__.py`, `utils.py` (extract dilation reclaim, feature loading, NPZ save/load from existing code)
- Create `sobel_cc.py` (refactor `depth_guided_instances()` from `generate_depth_guided_instances.py`)
- Create `ablate_instance_methods.py` with method dispatch, eval loop (mirror `sweep_k50_spidepth.py`), JSON output
- Verify baseline Sobel+CC reproduces PQ=26.74 through new script

#### Step 2: Method 6 — Morse/Gradient Flow Decomposition
**Algorithm**:
1. Gaussian blur depth (σ configurable)
2. `skimage.morphology.h_minima(depth, h=min_basin_depth)` — suppress shallow local minima
3. `skimage.segmentation.watershed(-depth, markers=labeled_minima)` — find basins of attraction
4. Per thing class: intersect basin map with semantic mask → candidate instances
5. Feature-based merge: for adjacent same-class instances, compute mean DINOv2 cosine similarity → merge if > `merge_threshold`
6. Filter by `min_area`, dilation reclaim

**Hyperparameter grid (fast: 64 configs)**:
| Param | Values |
|-------|--------|
| `min_basin_depth` | 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15 |
| `merge_threshold` | 0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.95 |
| `depth_blur_sigma` | 1.0 (fixed) |
| `min_area` | 1000 (fixed) |

**Sub-ablations**: Morse-raw (no feature merge, merge_threshold=0.0), Morse-merge (full)

**Dependencies**: scikit-image (check if installed, else `pip install scikit-image`)
**Compute**: ~250ms/image, 500 images × 64 configs = ~2 hours CPU
**Implementation**: ~100 lines core + ~50 lines feature merge

#### Step 3: Method 1 — TDA/Persistent Homology
**Algorithm**:
1. Gaussian blur depth
2. Build cubical complex on depth: `gudhi.CubicalComplex(top_dimensional_cells=depth.flatten(), dimensions=[H,W])`
3. Compute 0-dim persistence → persistence diagram (birth, death pairs)
4. **Practical approach**: Use watershed oversegmentation, then build region adjacency graph weighted by depth difference at boundaries (= boundary "persistence"). Merge regions with persistence < `tau_persist`. This is the efficient implementation of persistence-guided decomposition.
5. Per thing class: intersect with semantic mask, filter by min_area

**Hyperparameter grid (fast: 36 configs)**:
| Param | Values |
|-------|--------|
| `tau_persist` | 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20 |
| `filtration_mode` | "depth_direct", "gradient_mag" |
| `min_area` | 500, 1000 |

**Sub-ablations**: TDA-depth (filtration on D), TDA-grad (filtration on |∇D|), TDA+merge (add feature merge)

**Dependencies**: `pip install gudhi` (~80MB)
**Compute**: ~1s/image, 500 × 36 = ~5 hours CPU

### Phase 2: Medium Effort (Days 3-4)

#### Step 4: Method 5 — Optimal Transport
**Algorithm**:
1. Per thing class: extract patches, build descriptors x_i = [feat_i, depth_i × depth_scale, pos_i]
2. K-means on descriptors → K_proto prototypes
3. Cost matrix C[i,k] = ||x_i - proto_k||²
4. Sinkhorn: `T = diag(u) @ exp(-C/ε) @ diag(v)` with 100 iterations
5. Assign each patch to argmax_k T[i,k]
6. Convert patch assignments → pixel masks via nearest-neighbor upsample
7. Connected components + min_area filter

**Hyperparameter grid (fast: 72 configs)**:
| Param | Values |
|-------|--------|
| `K_proto` | 5, 10, 15, 20 |
| `epsilon` | 0.01, 0.05, 0.1 |
| `depth_scale` | 1.0, 5.0, 10.0 |
| `min_area` | 500, 1000 |

**Sub-ablations**: OT-depth-only (β_feat=0), OT-feat-only (depth_scale=0), OT-joint

**Dependencies**: None (implement Sinkhorn in ~20 lines numpy)
**Compute**: ~100ms/image, 500 × 72 = ~1 hour CPU

#### Step 5: Method 2 — Mumford-Shah Energy Minimization
**Algorithm**:
1. Work at (128, 256) resolution (downsample depth/semantic, upsample features)
2. Per thing class: build 4-connected pixel graph
3. Unary: (depth_i - μ_depth_R)² × α + ||feat_i - μ_feat_R||² × β
4. Pairwise: γ × exp(-||feat_i - feat_j||² / 2σ²_feat) × exp(-|d_i - d_j|² / 2σ²_depth)
5. Alpha-expansion with n_init_labels=20 via pymaxflow
6. Filter empty labels, min_area, dilation reclaim

**Hyperparameter grid (fast: 32 configs)**:
| Param | Values |
|-------|--------|
| `alpha/beta ratio` | 0.1, 1, 10, 100 |
| `gamma` | 1, 2, 5, 10 |
| `min_area` | 500, 1000 |

**Sub-ablations**: MS-depth-only (β=0), MS-feat-only (α=0), MS-joint, MS-spectral (spectral clustering fallback)

**Dependencies**: `pip install PyMaxflow` (~5MB). Fallback: spectral clustering via sklearn
**Compute**: ~5-8s/image at (128,256), 500 × 32 = ~1.5 hours CPU

### Phase 3: Learned Methods (Days 5-7)

#### Step 6: Method 4 — Contrastive Depth-Feature Embedding
**Algorithm (Training)**:
1. Input per patch: x_i = [feat_i (768), depth_i (1) × depth_weight, pos_i (2) × pos_weight] → 771-dim
2. MLP: Linear(771→256) → ReLU → Linear(256→128) → L2-normalize
3. Positive pairs: same class, ≤3 patches apart, |Δdepth| < delta_depth
4. Negative pairs: same class, separated by depth edge (gradient > tau_edge)
5. InfoNCE loss, Adam lr=3e-4, 20 epochs on train split (2975 images)

**Algorithm (Inference)**:
1. Embed all patches → z_i ∈ R^128
2. Per thing class: HDBSCAN on class patches' embeddings
3. Patch clusters → pixel masks via nearest-neighbor, connected components, min_area filter

**Training sweep (5 runs)**:
| Param | Values |
|-------|--------|
| `tau` (InfoNCE temp) | 0.07, 0.10 |
| `delta_depth` | 0.02, 0.05 |
| `depth_weight` | 0, 2, 5 |

**Inference sweep per trained model (48 configs)**:
| Param | Values |
|-------|--------|
| `hdbscan_min_cluster` | 5, 8 |
| `min_area` | 500, 1000 |

**Sub-ablations**: CE-raw (HDBSCAN on raw DINOv2, no training), CE-depth-only, CE-feat-only, CE-joint, CE-iter (iterative self-training)

**Dependencies**: None (PyTorch + sklearn HDBSCAN already available)
**Compute**: Training ~5 min/run × 5 = 25 min. Inference ~50s/config × 48 = 40 min. Total ~1 hour.

#### Step 7: Method 3 — Depth-Conditioned Slot Attention
**Architecture**:
1. Input: features F (2048, 768) + depth D pooled to (32, 64, 1)
2. Project: Linear(769 → d_slot) + positional encoding
3. Depth-initialized slots: depth histogram peaks → initial slot features
4. Slot attention iterations: depth-weighted cross-attention + GRU update
5. Reconstruction loss: ||X_proj - Σ_k A[:,k] × slot_k||²
6. Output: argmax attention → hard instance masks

**Training sweep (54 configs)**:
| Param | Values |
|-------|--------|
| `d_slot` | 64, 128, 256 |
| `n_slots` | 16, 32, 64 |
| `n_iters` | 3, 5 |
| `sigma_depth` | 0.02, 0.05, 0.10 |

**Sub-ablations**: SA-vanilla (random init, no depth), SA-depth-init, SA-depth-attn, SA-full

**Dependencies**: None (pure PyTorch)
**Compute**: Training ~37 min/run × 54 = ~33 hours on MPS (parallelize on 2× 1080 Ti → ~9 hours). Inference ~10s per model × 54 = ~9 min.

### Phase 4: Combined Method (Days 8-9)

#### Step 8: Morse + Contrastive Two-Stage
**Algorithm**:
- **Stage A**: Run best Morse config → initial instances (good for depth-separated objects)
- **Stage B**: Train contrastive head using Morse output as weak supervision (same-basin = positive, cross-basin = negative). At inference: for large instances (area > split_threshold), run HDBSCAN to check if it should be split. For adjacent small instances, merge if contrastive similarity > merge_threshold.

**Additional params**:
| Param | Values |
|-------|--------|
| `split_threshold` | 2000, 5000, 10000 |
| `contrastive_merge_threshold` | 0.6, 0.7, 0.8, 0.9 |

---

## Evaluation Protocol

### Metrics (all methods, identical setup)
- **Primary**: PQ, PQ_things, PQ_stuff (19-class, Cityscapes val, 500 images)
- **Diagnostic**: Per-class PQ/RQ/TP/FP/FN (especially person, car)
- **Efficiency**: seconds/image, total sweep time
- **Instance count**: avg instances/image (detects over/under-segmentation)

### Fairness Constraints
- Same k=80 semantic labels (mapped to 19 trainIDs)
- Same SPIdepth depth maps
- Same DINOv2 ViT-B/14 features (for feature-aware methods)
- Same eval resolution 512×1024
- Same thing/stuff split (trainIDs 11-18 = things)
- Same post-processing (dilation_iters=3)

### Statistical Significance
- Deterministic methods (1, 2, 5, 6): single run
- Learned methods (3, 4): 3 seeds, report mean ± std
- Paired bootstrap test (1000 resamples) for PQ_things vs baseline

### Output
Each config → JSON: `{method, config, PQ, PQ_stuff, PQ_things, per_class, avg_instances, seconds_per_image}`

---

## Success Criteria

| Target | PQ_things | Meaning |
|--------|-----------|---------|
| **Minimum** | ≥ 21.0 | Direction validated (+8% relative over 19.41) |
| **Good** | ≥ 23.0 | NeurIPS-worthy ablation table entry |
| **Excellent** | ≥ 25.0 | Comfortably beats CUPS on pseudo-label quality alone |
| **Person PQ** | ≥ 8.0 | Doubles baseline (4.2), proves co-planar fix works |
| **Kill threshold** | < 19.41 | Worse than baseline → stop tuning, move on |

---

## Dependencies to Install
```bash
pip install gudhi          # Method 1 (TDA), ~80MB
pip install scikit-image   # Method 6 (watershed), ~30MB
pip install PyMaxflow      # Method 2 (graph cuts), ~5MB
# POT optional for Method 5 (implement Sinkhorn manually instead)
```

---

## Total Time Estimate
| Phase | Duration | Methods |
|-------|----------|---------|
| Phase 1 | 2 days | Scaffold + Morse + TDA |
| Phase 2 | 2 days | OT + Mumford-Shah |
| Phase 3 | 3 days | Contrastive + Slot Attention |
| Phase 4 | 2 days | Combined Morse+Contrastive |
| **Total** | **~9 days** | 6 methods + 1 combined + full comparison |

## Verification
1. Baseline reproduces PQ=26.74 through new unified script
2. Each method produces valid NPZ instance masks (spot-check 5 images visually)
3. All JSON results have consistent schema
4. Final comparison table with all 7 methods + baseline
5. Per-class breakdown confirms person/car improvement direction

---

## Phase 2: Hyperparameter Sweeps — COMPLETE (2026-03-31)

**Result**: 244 configs across 6 methods. NO method beats Sobel+CC (PQ_th=19.41). Mumford-Shah closest at 18.71. 100-image subsets overestimate by ~4.5 PQ_things.
See `reports/novel_instance_ablation_final_report.md` for full analysis.

---

## Phase 3: Contrastive Depth-Feature Embedding — FAILED (2026-03-31)

### Result
InfoNCE + L2-normalized embeddings + HDBSCAN = fundamentally wrong formulation.
- Best PQ_things=7.32 at epoch 1 (essentially untrained), **declined** to 4.26 by epoch 20.
- Training loss dropped (0.34→0.04) but PQ_things got WORSE — metric mismatch.
- Root cause: L2 normalization maps all embeddings to unit hypersphere → uniform density → HDBSCAN
  (density-based) cannot find cluster modes. More training = more uniform = worse clustering.
- All 5 runs showed same pattern. Run 2 (depth_weight=5.0) best at epoch 1: PQ_th=8.93.
- Scripts created and working: `train_contrastive_embed.py`, `eval_contrastive_learned.py`,
  `scripts/train_contrastive_all_runs.sh`.

### Key Lesson
**Embedding → HDBSCAN is the wrong pipeline for this task.** The three best methods
(Sobel 19.41, Mumford-Shah 18.71, TDA 16.70) all use boundary/region → CC at their core.
Any learned approach must feed into the proven CC pipeline, not replace it with clustering.

---

## Phase 4: Learned Fragment Merge — Over-Segment + Learned Grouping (CURRENT)

### Context
Phase 3 proved that embedding → clustering fails. The target is PQ_things ≥ 18.0 with a
**learned** method (algorithmic novelty for NeurIPS Reviewer W1). The key insight: keep the
proven CC pipeline, but replace the hard depth threshold with a learned merge decision.

### Architecture: Two-Stage Over-Segment + Learned Merge

```
Stage 1: Sobel+CC at LOW threshold (τ=0.10)
  → Over-segmented fragments (many small CCs, some spuriously split)
  → PQ_things=18.17 at τ=0.10/A_min=1000 (baseline for this stage)

Stage 2: Learned pairwise merge predictor
  → For each adjacent same-class fragment pair:
      extract depth+feature+spatial descriptors → MLP → merge probability
  → Union-find on merge decisions → final instances
  → Dilation reclaim
```

### Why This Will Work (and Why Phase 3 Didn't)

| Phase 3 (Failed) | Phase 4 (New) |
|-------------------|---------------|
| Embedding → HDBSCAN (unproven) | Boundary → CC → merge (proven pipeline) |
| L2-norm → uniform density | Sigmoid → direct merge probability |
| InfoNCE (contrastive proxy) | BCE (direct task loss) |
| HDBSCAN needs density modes | Union-find just needs thresholding |
| Training ↑ ≠ PQ ↑ | Training directly optimizes merge accuracy |

### Self-Supervised Training Signal (NO GT required)

Multi-threshold Sobel consensus provides clean merge labels:
1. Run Sobel+CC at τ_low=0.10 → over-segmented fragments
2. Run Sobel+CC at τ_high=0.20 → "oracle" grouping (best known PQ)
3. For each adjacent same-class fragment pair at τ_low:
   - Both overlap SAME CC at τ_high → label=MERGE (1)
   - Overlap DIFFERENT CCs at τ_high → label=NO-MERGE (0)

This is depth-derived self-supervision: higher-threshold output supervises lower-threshold
merging. No instance annotations needed.

### Pairwise Fragment Descriptors

For each candidate merge pair (fragment A, fragment B):

| Feature | Dim | Source |
|---------|-----|--------|
| Mean DINOv2 feature A (PCA) | 64 | `dinov2_features/*.npy` |
| Mean DINOv2 feature B (PCA) | 64 | `dinov2_features/*.npy` |
| \|feat_A − feat_B\| | 64 | element-wise |
| cosine(feat_A, feat_B) | 1 | `utils.cosine_similarity_regions()` |
| Mean depth A, B | 2 | `depth_spidepth/*.npy` |
| \|depth_A − depth_B\| | 1 | absolute diff |
| log(area_A), log(area_B) | 2 | fragment pixel counts |
| Centroid distance (normalized) | 1 | Euclidean / image diagonal |
| Boundary cosine similarity | 1 | features at shared boundary |
| **Total** | **~200** | |

PCA 768→64 fitted on train DINOv2 features. Reduces noise and makes MLP tractable.

### Model: MergePredictor

```python
class MergePredictor(nn.Module):
    """Binary classifier: should two fragments merge?"""
    def __init__(self, input_dim=200, hidden_dim=128):
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    def forward(self, x):
        return self.net(x)  # raw logit → BCE with logits
```

~26K parameters. No BatchNorm (avoids single-sample crashes). Dropout for regularization.

### Training

- Loss: `BCEWithLogitsLoss` (class-balanced via pos_weight)
- Optimizer: Adam, lr=1e-3, weight_decay=1e-4
- Epochs: 20, early stopping on val merge accuracy
- Data: ~2975 train images × ~5 merge pairs/image ≈ 15K pairs
- Validation: 500 val images, same pair extraction
- Expected training time: ~2 min on MPS

### Inference Pipeline

```python
def learned_merge_instances(semantic, depth, thing_ids, features,
                            model, pca, tau_low=0.10,
                            merge_threshold=0.5, min_area=1000,
                            dilation_iters=3, depth_blur_sigma=1.0):
    # Stage 1: Over-segment
    fragments = sobel_cc_instances(semantic, depth, thing_ids,
                                   grad_threshold=tau_low,
                                   min_area=min_area // 2,  # lower threshold for fragments
                                   dilation_iters=0)  # no reclaim yet

    # Stage 2: Pairwise merge
    pairs = find_adjacent_same_class_pairs(fragments)
    for (frag_a, frag_b) in pairs:
        desc = extract_pairwise_descriptor(frag_a, frag_b, features, depth, pca)
        logit = model(desc)
        if sigmoid(logit) > merge_threshold:
            union(frag_a, frag_b)

    # Finalize
    merged = apply_union_find(fragments)
    return dilation_reclaim(merged, semantic, thing_ids,
                            min_area=min_area, dilation_iters=dilation_iters)
```

### Implementation Steps

#### Step 0: Quick Baseline — Sobel+CC + Feature Merge (NO training)
**Goal**: Establish if simple feature-based merge helps at all.
- Add optional feature merge to `sobel_cc.py` (reuse `_feature_merge_adjacent` from `morse_flow.py`)
- Sweep: τ ∈ {0.10, 0.15, 0.20} × merge_threshold ∈ {0.6, 0.7, 0.8, 0.9} × min_area ∈ {500, 1000}
- If PQ_things ≥ 18.0 here → faster win, proceed to learned merge for additional novelty
- **Time**: ~30 min implement + ~1 hour sweep

#### Step 1: Generate Training Data
**Goal**: Extract fragment pairs with merge labels from multi-threshold consensus.
- For each train image: run Sobel+CC at τ ∈ {0.05, 0.10, 0.15, 0.20, 0.25}
- Extract adjacent same-class fragment pairs at τ_low
- Label: merge/no-merge based on τ_high grouping
- Fit PCA(768→64) on fragment-level DINOv2 means
- Save: pairs NPZ + PCA model
- **Time**: ~20 min compute

#### Step 2: Train MergePredictor
**Goal**: Learn merge decision from fragment descriptors.
- Train with BCE, class-balanced, 20 epochs
- Log: accuracy, precision, recall, AUC on val set
- Save best checkpoint by val accuracy
- **Time**: ~5 min

#### Step 3: Evaluate Learned Merge Pipeline
**Goal**: End-to-end evaluation on Cityscapes val (500 images).
- Sweep 18 configs: τ_low ∈ {0.05, 0.10, 0.15} × merge_threshold ∈ {0.3, 0.5, 0.7} × min_area ∈ {500, 1000}
- Compare PQ, PQ_things, per-class breakdown against all prior methods
- **Time**: ~1.5 hours

#### Step 4: Analysis and Comparison
**Goal**: Determine if target is met, update ablation report.
- Per-class analysis: where does learned merge help vs Sobel+CC?
- Compare full 8-method table (6 from Phase 2 + contrastive_learned + learned_merge)
- If PQ_things ≥ 18.0: update `reports/novel_instance_ablation_final_report.md`

### Sobel+CC Sweep Reference (for calibration)

| τ | A_min | PQ_things | inst/img |
|---|-------|-----------|----------|
| 0.10 | 500 | 16.88 | 5.7 |
| 0.10 | 1000 | 18.17 | 4.2 |
| 0.15 | 1000 | 18.85 | 4.3 |
| 0.20 | 1000 | **19.41** | 4.3 |
| 0.25 | 1000 | 19.12 | 4.3 |

Gap to close: 19.41 - 18.17 = 1.24 PQ_things (from τ=0.10 to τ=0.20).
Learned merge needs to recover ≥60% of this gap to hit 18.0.

### Per-Class Analysis (Sobel+CC τ=0.20, where gains are possible)

| Class | PQ | TP | FP | FN | Bottleneck |
|-------|----|----|----|----|------------|
| car | 16.49 | 648 | 386 | 3987 | Over+under-seg |
| person | 4.02 | 123 | 277 | 3253 | Co-planar ceiling |
| bicycle | 5.80 | 77 | 323 | 1086 | Over-seg (many FP) |
| rider | 9.23 | 57 | 113 | 484 | Both |
| truck | 35.52 | 33 | 16 | 60 | Already decent |
| bus | 47.76 | 48 | 12 | 50 | Already decent |
| train | 36.43 | 11 | 9 | 12 | Already decent |
| motorcycle | 0.00 | 0 | 0 | 149 | No detections |

**Best targets for learned merge**: car (reduce 386 FP), bicycle (reduce 323 FP).
Feature-based merge can reduce FP by correctly grouping over-segmented fragments.

### Files to Create/Modify

| File | Action | Lines | Purpose |
|------|--------|-------|---------|
| `mbps_pytorch/train_merge_predictor.py` | CREATE | ~300 | Training script + data extraction |
| `mbps_pytorch/instance_methods/learned_merge.py` | CREATE | ~180 | Inference method |
| `mbps_pytorch/instance_methods/__init__.py` | MODIFY | +2 | Register new method |
| `mbps_pytorch/eval_learned_merge.py` | CREATE | ~120 | Dedicated eval script |

**Reuse**:
- `mbps_pytorch/instance_methods/sobel_cc.py` — Stage 1 over-segmentation
- `mbps_pytorch/instance_methods/utils.py` — `dilation_reclaim`, `load_features`, `cosine_similarity_regions`
- `mbps_pytorch/merge_depth_instances.py` — union-find merge pattern (lines 168-224)
- `mbps_pytorch/ablate_instance_methods.py` — eval infrastructure (discover_files, evaluate_panoptic_single, etc.)

### Execution Commands

```bash
PYTHON=/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python
CS_ROOT=/Users/qbit-glitch/Desktop/datasets/cityscapes

# Step 0: Quick baseline (feature merge, no training)
nohup $PYTHON -u mbps_pytorch/eval_learned_merge.py \
    --cityscapes_root $CS_ROOT --mode feature_merge --sweep \
    > logs/eval_feature_merge.log 2>&1 &

# Step 1-2: Generate data + train
nohup $PYTHON -u mbps_pytorch/train_merge_predictor.py \
    --cityscapes_root $CS_ROOT --epochs 20 \
    > logs/train_merge_predictor.log 2>&1 &

# Step 3: Evaluate learned merge
nohup $PYTHON -u mbps_pytorch/eval_learned_merge.py \
    --cityscapes_root $CS_ROOT \
    --checkpoint checkpoints/merge_predictor/best.pth \
    --pca_path checkpoints/merge_predictor/pca.npz \
    --sweep \
    > logs/eval_learned_merge.log 2>&1 &
```

### Kill / Success Criteria

| Threshold | PQ_things | Action |
|-----------|-----------|--------|
| Kill | < 17.0 | Stop. Frame Mumford-Shah (18.71) as novel contribution instead |
| Minimum | ≥ 18.0 | Report as successful learned method |
| Good | ≥ 19.0 | Approaches Sobel+CC, strong ablation entry |
| Excellent | ≥ 19.5 | Beats Sobel+CC, proves features add value |

### NeurIPS Framing

"We propose a two-stage instance decomposition: (1) depth-based oversegmentation via low-threshold
edge detection produces candidate fragments, (2) a learned pairwise merge predictor trained with
self-supervised multi-threshold depth consensus decides which fragments to group. This replaces
the single hard gradient threshold of Sobel+CC with a learned decision boundary in the
depth-feature space, achieving PQ_things=X.XX on Cityscapes val without instance annotations."

### Verification
1. Step 0: Feature merge ≥ τ=0.10 baseline (18.17) — features don't hurt
2. Step 1: Training data has ~40-60% positive (merge) labels — balanced
3. Step 2: Merge accuracy > 70% on val (random baseline = 50%)
4. Step 3: PQ_things ≥ 18.0 on full 500 val images
5. Step 3: PQ_stuff remains ~32 (instances don't affect stuff)
6. Step 4: Per-class analysis shows car/bicycle FP reduction
