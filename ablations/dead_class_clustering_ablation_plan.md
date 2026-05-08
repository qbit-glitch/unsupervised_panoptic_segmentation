# Dead-Class Clustering Ablation Plan — Stage-0 Recovery

**Date**: 2026-05-08
**Baseline**: ViT-L/16 + spherical k-means k=100
- PQ=20.88, mIoU=39.14, PQ_things=7.67
- **7 dead classes** (IoU=0, PQ=0): wall, fence, traffic light, rider, truck, train, motorcycle

**Goal**: Recover dead classes at the clustering level by replacing/augmenting the k-means step with methods that can discover minority semantics.

**Features**: `{CS}/dinov3_features_vitl16/{train,val}/` — 1024-dim, 2048 patches (32×64 grid) per image.

**Evaluation**: Same as `scripts/evaluate_pseudolabel_quality.py` — mIoU (19-class), PQ, per-class IoU/PQ on Cityscapes val (500 images). Hungarian matching from pseudo-clusters to GT classes.

---

## Common Protocol

1. Each method produces pseudo-label PNGs in `{CS}/pseudo_semantic_raw_dinov3_k100_{method}_vitl16/{train,val}/`
2. Evaluate with: `python scripts/evaluate_pseudolabel_quality.py --pred_dir {output_dir}/val --split val`
3. Save results to: `results/clustering_ablation/eval_{method}_vitl16_k100.json`
4. Report: per-class IoU for all 19 classes, mIoU, PQ, PQ_stuff, PQ_things
5. All methods use seed=42, k=100 unless method is adaptive-k

---

## Ablation 1: FeatUp / Shift-Average Feature Upsampling

### Hypothesis
Traffic light and pole are dead because they're smaller than the 16×16 patch size (occupy <1 patch at 32×64). Sub-patch features would give these objects dedicated feature vectors.

### Two Sub-Approaches

#### 1a: Shift-and-Average (Training-Free)
**Paper**: "Upsampling DINOv2 Features" (Advanced Intelligent Systems, 2026)

**Method**: Shift input image by sub-patch offsets (4 shifts: +8px horizontal, +8px vertical, +8px both, original), extract features for each shift, average in a 2× upsampled grid (64×128).

**Implementation**:
```
1. For each image, load raw image (512×1024)
2. Create 4 shifted versions: (0,0), (8,0), (0,8), (8,8)
3. Extract ViT-L/16 features for each shift → 4 × (2048, 1024)
4. Interleave into 64×128 grid → (8192, 1024)
5. L2-normalize
6. Run spherical k-means k=100 on all upsampled features
7. Assign clusters, resize labels to 512×1024
```

**Requirements**: DINOv2 ViT-L/16 model weights (HuggingFace `facebook/dinov2-large`), ~4× more feature extraction time.

**Feature extraction**: Must re-extract features at shifted positions (can't use cached features). Estimate ~2-3 hours for train+val on MPS.

**Output dir**: `pseudo_semantic_raw_dinov3_k100_shift_avg_vitl16/`

#### 1b: FeatUp Pretrained Upsampler
**Paper**: "FeatUp: A Model-Agnostic Framework for Features at Any Resolution" (ICLR 2024)

**Method**: Use pretrained JBU (Joint Bilateral Upsampling) network to upsample 32×64 → 128×256 features.

**Implementation**:
```
1. pip install featup (or clone from GitHub)
2. Load pretrained FeatUp for DINOv2
3. For each image: backbone features (32×64) → FeatUp → (128×256, 1024)
4. L2-normalize upsampled features
5. Spherical k-means k=100
6. Assign, resize to 512×1024
```

**Requirements**: `pip install featup`, pretrained weights (~100MB), GPU/MPS for upsampling.

**Output dir**: `pseudo_semantic_raw_dinov3_k100_featup_vitl16/`

### Expected Impact
| Class | Baseline IoU | Expected | Rationale |
|-------|-------------|----------|-----------|
| traffic light | 0.0 | 5-15 | Sub-patch resolution recovers small objects |
| pole | 16.0 | 20-30 | Better spatial precision |
| traffic sign | 46.0 | 48-52 | Marginal improvement |
| wall | 0.0 | 0-5 | Unlikely — wall is large, not a resolution issue |
| fence | 0.0 | 0-10 | Thin structure, may benefit from higher res |

### Compute Budget
- 1a: ~3h feature re-extraction + 30min k-means = ~3.5h
- 1b: ~1h upsampling (if pretrained weights exist) + 30min k-means = ~1.5h

---

## Ablation 2: Recursive Deep Spectral Clustering

### Hypothesis
Rider/motorcycle fail because k-means assigns them to bicycle/car clusters. Recursive NCut first separates "person-on-vehicle" from "vehicle", then splits rider from bicycle at finer levels.

### Method
**Paper**: "Hierarchy-Agnostic Unsupervised Segmentation" (NeurIPS 2024)

**Implementation**:
```
1. Clone https://github.com/[recursive-deep-spectral-clustering]
2. For each image:
   a. Build affinity matrix from ViT-L/16 features (2048×2048, cosine similarity)
   b. Compute top-k eigenvectors of normalized Laplacian
   c. Recursively bipartition using Fiedler vector (2nd eigenvector)
   d. Stop when partition cost exceeds threshold τ_cut
   e. Collect leaf segments → per-image cluster IDs
3. Global cluster alignment:
   a. Extract prototype (mean feature) per segment across all train images
   b. Cluster prototypes into k=100 global clusters via spherical k-means
   c. Assign each per-image segment to its nearest global cluster
4. Output pseudo-labels
```

**Key Hyperparameters**:
- `τ_cut`: partition cost threshold (controls granularity, sweep: 0.01, 0.02, 0.05)
- `max_depth`: maximum recursion depth (default: 8)
- `min_segment_size`: minimum pixels per segment (default: 50 patches = ~800 pixels)
- `n_eigenvectors`: number of eigenvectors for NCut (default: 2, try up to 5)

**Output dir**: `pseudo_semantic_raw_dinov3_k100_recursive_ncut_vitl16/`

### Expected Impact
| Class | Baseline IoU | Expected | Rationale |
|-------|-------------|----------|-----------|
| rider | 0.0 | 5-15 | Recursive split separates person-on-vehicle |
| motorcycle | 0.0 | 3-10 | Separated from car/bicycle |
| wall | 0.0 | 5-15 | Separated from building at fine level |
| fence | 0.0 | 5-10 | Separated from vegetation/building |
| truck | 0.0 | 3-8 | Separated from bus |

### Compute Budget
- Affinity matrix: 2048×2048 per image → ~5s/image on MPS
- Full dataset: ~5s × 3475 = ~5h
- Global clustering: ~10min

---

## Ablation 3: DiffCut — Diffusion Features + Recursive NCut

### Hypothesis
Diffusion features (from Stable Diffusion UNet encoder) capture texture gradients and boundaries that DINOv2 misses. Wall/fence are dead because DINOv2 features are classification-oriented and don't encode texture differences between wall and building.

### Method
**Paper**: "DiffCut: Catalyzing Zero-Shot Semantic Segmentation" (NeurIPS 2024)

**Implementation**:
```
1. pip install diffusers
2. Load Stable Diffusion v2.1 (or v1.5)
3. For each image:
   a. Encode with DDIM (t=50 or t=100 noise level)
   b. Extract UNet encoder features at multiple scales
   c. Concatenate multi-scale features → dense feature map
   d. Apply recursive NCut (same as Ablation 2)
4. Two evaluation modes:
   a. DiffCut standalone: cluster diffusion features alone
   b. DiffCut + DINOv2 ensemble: weighted combination of DiffCut and DINOv2 cluster assignments
```

**Key Hyperparameters**:
- `t`: diffusion timestep for feature extraction (sweep: 50, 100, 200)
- `layers`: which UNet layers to use (default: encoder blocks 2,3)
- `ensemble_weight`: α for combining with DINOv2 (0.0 = DINOv2 only, 1.0 = DiffCut only, sweep: 0.3, 0.5, 0.7)

**Output dirs**:
- `pseudo_semantic_raw_dinov3_k100_diffcut_standalone_vitl16/`
- `pseudo_semantic_raw_dinov3_k100_diffcut_ensemble_vitl16/`

### Expected Impact
| Class | Baseline IoU | Expected | Rationale |
|-------|-------------|----------|-----------|
| wall | 0.0 | 10-20 | Diffusion features encode texture differences |
| fence | 0.0 | 8-15 | Texture + boundary encoding |
| traffic light | 0.0 | 3-8 | Marginal — still small |
| truck | 0.0 | 5-10 | Texture distinguishes from bus |

### Compute Budget
- Feature extraction: ~3s/image on MPS (SD inference)
- Full dataset: ~3h
- NCut + clustering: ~5h (same as Ablation 2)
- Total: ~8h per variant

### Requirements
- `pip install diffusers transformers accelerate`
- Stable Diffusion weights (~5GB download)
- ~10GB VRAM (fits on MPS 48GB)

---

## Ablation 4: PPAP — Progressive Proxy Anchor Propagation

### Hypothesis
K-means allocates clusters proportional to feature-space density — rare classes get zero clusters. PPAP's proxy anchors explicitly seek distinct semantic modes regardless of density.

### Method
**Paper**: "Progressive Proxy Anchor Propagation for Unsupervised Semantic Segmentation" (ECCV 2024)

**Implementation**:
```
1. Clone https://github.com/hynnsk/PPAP
2. Extract the proxy anchor clustering module
3. Adapt to work with our pre-extracted ViT-L/16 features:
   a. Initialize k=100 proxy anchors (can start from k-means centroids)
   b. For each epoch (10-20 epochs):
      - For each mini-batch of features:
        * Compute similarity to all proxies
        * Pull features toward nearest proxy (positive)
        * Push features away from non-nearest proxies (negative)
        * Progressively relocate proxy toward dense regions
   c. ProxyNCA+ loss: ensures proxies spread to cover all modes
4. Assign clusters using final proxy locations
```

**Key Hyperparameters**:
- `k`: number of proxies (100, matching baseline)
- `lr_proxy`: proxy learning rate (1e-3)
- `epochs`: training epochs (10, 20)
- `temperature`: contrastive temperature (0.1, 0.2)
- `margin`: ProxyNCA+ margin (0.1)

**Output dir**: `pseudo_semantic_raw_dinov3_k100_ppap_vitl16/`

### Expected Impact
| Class | Baseline IoU | Expected | Rationale |
|-------|-------------|----------|-----------|
| motorcycle | 0.0 | 5-15 | Proxy finds motorcycle mode even if sparse |
| rider | 0.0 | 5-10 | Proxy separates rider from person |
| truck | 0.0 | 5-12 | Proxy distinguishes from bus |
| train | 0.0 | 3-8 | Very rare, proxy may still miss |
| wall | 0.0 | 3-8 | Feature-space separation from building |

### Compute Budget
- Training: ~30min on MPS (features already extracted)
- Assignment: ~5min
- Total: ~40min

---

## Ablation 5: PCL + Balanced Sinkhorn

### Hypothesis
Standard k-means centroids drift toward the mean of large clusters. ProtoNCE actively pushes small-cluster prototypes away from large-cluster prototypes, preventing absorption.

### Method
**Paper**: "Prototypical Contrastive Learning of Unsupervised Representations" (ICLR 2021)

**Implementation** (~150 lines new PyTorch code):
```
1. Initialize k=100 prototypes from k-means centroids (warm start)
2. For each epoch (20-50 epochs):
   a. E-step: assign features to nearest prototype (cosine similarity)
   b. Apply Sinkhorn normalization to balance assignments
   c. M-step: compute ProtoNCE loss
      L_ProtoNCE = -log(exp(z·c+/τ) / Σ_j exp(z·c_j/τ))
      where c+ is the assigned prototype, c_j are all prototypes
   d. Update prototypes via gradient descent
3. Final assignment: cosine similarity to optimized prototypes
```

**Key Hyperparameters**:
- `k`: 100 prototypes
- `τ`: temperature (0.05, 0.1, 0.2)
- `lr`: prototype learning rate (1e-2, 1e-3)
- `epochs`: 20, 50
- `sinkhorn_iters`: 3, 5, 10
- `sinkhorn_eps`: Sinkhorn regularization (0.05, 0.1)
- `momentum`: prototype EMA momentum (0.9, 0.99)

**Output dir**: `pseudo_semantic_raw_dinov3_k100_pcl_sinkhorn_vitl16/`

### Expected Impact
| Class | Baseline IoU | Expected | Rationale |
|-------|-------------|----------|-----------|
| wall | 0.0 | 5-12 | Sinkhorn prevents building from absorbing wall |
| fence | 0.0 | 3-10 | Same anti-absorption mechanism |
| truck | 0.0 | 5-10 | Pushed away from bus prototype |
| motorcycle | 0.0 | 3-8 | Separated from bicycle |
| train | 0.0 | 2-5 | Too rare even for balanced methods |

### Compute Budget
- Training: ~20-40min on MPS
- Assignment: ~5min
- Total: ~45min

---

## Ablation 6: SDCluster — Prototype Constraint + Semantic Consistency

### Hypothesis
Dead prototypes can be explicitly detected (zero/low assignment count) and re-initialized during training. The semantic consistency constraint ensures re-initialized prototypes land in semantically meaningful regions.

### Method
**Paper**: "SDCluster: A clustering based self-supervised pre-training method" (ISPRS 2025)

**Implementation**:
```
1. Initialize k=100 prototypes from k-means
2. Training loop with three components:
   a. Standard cluster assignment (cosine similarity)
   b. Prototype constraint module:
      - Monitor assignment counts per prototype
      - If count < threshold (e.g., < 0.5% of patches): re-initialize
      - Re-initialization: perturb the largest cluster's centroid
        or sample from high-uncertainty region
   c. Semantic consistency constraint:
      - Spatial smoothness: adjacent patches should share clusters
      - Feature coherence: cluster members should be tightly grouped
      - L_consistency = L_spatial + L_coherence
3. Alternating optimization: update assignments → update prototypes → check dead prototypes
```

**Key Hyperparameters**:
- `dead_threshold`: fraction below which prototype is considered dead (0.005, 0.01)
- `reinit_strategy`: "perturb_largest" | "sample_uncertain" | "furthest_point"
- `spatial_weight`: weight for spatial smoothness (0.1, 0.3)
- `coherence_weight`: weight for within-cluster coherence (0.1, 0.3)
- `epochs`: 30-50 (needs more iterations for re-initialization to converge)

**Output dir**: `pseudo_semantic_raw_dinov3_k100_sdcluster_vitl16/`

### Expected Impact
| Class | Baseline IoU | Expected | Rationale |
|-------|-------------|----------|-----------|
| wall | 0.0 | 8-18 | Dead prototype explicitly re-initialized |
| fence | 0.0 | 5-15 | Same mechanism |
| traffic light | 0.0 | 3-10 | Re-init finds small-object mode |
| rider | 0.0 | 5-12 | Re-init separates from person |
| motorcycle | 0.0 | 3-10 | Re-init finds motorcycle mode |
| truck | 0.0 | 5-10 | Re-init separates from bus |
| train | 0.0 | 2-8 | Re-init may still fail (too rare) |

### Compute Budget
- Training: ~1-2h on MPS (iterative re-initialization needs convergence)
- Assignment: ~5min
- Total: ~2h

---

## Execution Order

| Priority | Ablation | Effort | Time Est. | Rationale |
|----------|----------|--------|-----------|-----------|
| **P0** | 1a: Shift-Average | Low | 3.5h | Training-free, tests resolution hypothesis immediately |
| **P1** | 5: PCL + Sinkhorn | Medium | 45min | Fastest trainable method, directly addresses cluster imbalance |
| **P2** | 4: PPAP | Medium | 40min | Similar speed, tests proxy anchor hypothesis |
| **P3** | 6: SDCluster | Medium | 2h | Explicit dead prototype recovery |
| **P4** | 2: Recursive NCut | Medium | 5h | Tests spectral decomposition hypothesis |
| **P5** | 1b: FeatUp | Low-Med | 1.5h | Requires package install, but fast if it works |
| **P6** | 3: DiffCut | High | 8h | Longest, requires SD weights, but novel features |

---

## Success Criteria

**Per-ablation**: Any dead class IoU > 0 counts as recovery. Target: ≥3 dead classes recovered per method.

**Overall**: Best method should achieve:
- mIoU > 42 (baseline: 39.14, +3 from recovered classes)
- PQ > 22 (baseline: 20.88)
- ≥5 of 7 dead classes with IoU > 0

**Downstream validation**: Best method's pseudo-labels will be propagated through DCFA+SIMCF-ABC+CUPS to measure Stage-2/3 impact.
