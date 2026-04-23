# Pseudo-Label Pipeline Data-Flow Architecture

> **Configuration:** k=80, DepthPro, τ=0.20, A_min=1000, DCFA + SimCF-ABC

---

## 1. Semantic Pseudo-Label Generation (DCFA + k=80 Overclustering)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  INPUT: RGB Image (Cityscapes)                                                          │
│  Tensor:  (H_orig, W_orig, 3)  =  (1024, 2048, 3)  uint8                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Resize + Normalize
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  Resized RGB for CAUSE-TR                                                               │
│  Tensor:  (H, W, 3)  =  (322, 644, 3)  float32   [H,W = multiple of patch_size=14]      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ DINOv2 ViT-B/14 Backbone (frozen)
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  Patch Embedding                                                                        │
│  Input:   (322, 644, 3)                                                                 │
│  Output:  (N_patches, 768)  =  (1058, 768)   [N_patches = 23×46 = H/14 × W/14]          │
│  Note:    Sliding-window inference with 322×322 crops, aggregated via averaging         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Segment_TR Head (CAUSE-TR, frozen)
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  90-Dimensional Code Features                                                           │
│  Input:   (1058, 768)                                                                   │
│  Output:  (90, H, W)  =  (90, 322, 644)   float32                                       │
│  Note:    Per-pixel dense features after patch→pixel interpolation                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Adaptive Average Pool to Patch Grid
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  Patch-Resolution Codes for DCFA                                                        │
│  Input:   (90, 322, 644)                                                                │
│  Output:  (90, H_p, W_p)  =  (90, 23, 46)   float32                                     │
│  Note:    H_p = 322/14 = 23,  W_p = 644/14 = 46                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Permute + Flatten
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  Flattened Codes                                                                        │
│  Tensor:  (1, N, 90)  =  (1, 1058, 90)   float32                                        │
│  where N = H_p × W_p = 23 × 46 = 1058                                                   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ├────────────────────────────────────────┐
                                    │                                        │
                                    ▼ DepthPro Depth Map                     ▼ DINOv2 768D (optional for DCFA-X)
┌─────────────────────────────────────────────────────────┐   ┌──────────────────────────────────────────┐
│  Raw DepthPro Depth                                     │   │  DINOv2 Full Features (for DCFA-X only)  │
│  Tensor:  (512, 1024)  float32  [saved as .npy]         │   │  Tensor:  (1, 1058, 768)  float32        │
│  Range:   metric depth (log-scaled, normalized to       │   │  Note:    Pre-extracted and cached       │
│           [0, 1] via min-max per dataset)               │   │           as `{stem}_dino768.npy`        │
└─────────────────────────────────────────────────────────┘   └──────────────────────────────────────────┘
                                    │                                        │
                                    ▼ Block Average Downsample               │
┌─────────────────────────────────────────────────────────┐                  │
│  Downsampled Depth (to patch grid)                      │                  │
│  Tensor:  (H_p, W_p)  =  (23, 46)  float32              │                  │
│  Method:  mean-pooling over non-overlapping blocks      │                  │
└─────────────────────────────────────────────────────────┘                  │
                                    │                                        │
                                    ▼ Flatten                                │
┌─────────────────────────────────────────────────────────┐                  │
│  Flattened Depth Values                                 │                  │
│  Tensor:  (1, N)  =  (1, 1058)  float32                 │                  │
└─────────────────────────────────────────────────────────┘                  │
                                    │                                        │
                                    ▼ Sinusoidal Positional Encoding         │
┌─────────────────────────────────────────────────────────┐                  │
│  16-Dimensional Sinusoidal Depth Features               │                  │
│  Tensor:  (1, N, 16)  =  (1, 1058, 16)  float32         │                  │
│  Formula: e(d) = [sin(ω₁πd), cos(ω₁πd), ...,            │                  │
│                  sin(ω₈πd), cos(ω₈πd)]                  │                  │
│  where ωₖ ∈ {1, 2, 4, 8, 16, 32, 64, 128}               │                  │
└─────────────────────────────────────────────────────────┘                  │
                                    │                                        │
                                    ▼ DCFA (Depth-Conditioned Feature Adapter)│
┌────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────┐
│  DCFA v3 Architecture (V3_dd16_h384_l2)                                                                          │
│                                                                                                                   │
│  Input:   codes  (1, 1058, 90)  +  depth  (1, 1058, 16)                                                          │
│  Concat:  (1, 1058, 106)                                                                                         │
│                                                                                                                   │
│  Layer 1:  Linear(106 → 384)  +  LayerNorm(384)  +  ReLU   →  (1, 1058, 384)                                    │
│  Layer 2:  Linear(384 → 384)  +  LayerNorm(384)  +  ReLU   →  (1, 1058, 384)                                    │
│  Output:   Linear(384 → 90)  [zero-initialized]            →  (1, 1058, 90)  residual                           │
│                                                                                                                   │
│  Skip:     codes + residual                                →  (1, 1058, 90)  adjusted_codes                      │
│                                                                                                                   │
│  Parameters: ~40K total                                                                                          │
│  Checkpoint: `results/depth_adapter/V3_dd16_h384_l2/best.pt`                                                     │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Reshape + Interpolate
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  DCFA-Adjusted Features (back to pixel resolution)                                      │
│  Input:   (1, 1058, 90)                                                                 │
│  Reshape: (90, 23, 46)                                                                  │
│  Upsample:(90, 322, 644)  via bilinear interpolation                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ├─────────────────────────────────────────────────────┐
                                    │                                                     │
                                    ▼ (Optional) Depth Feature Concatenation              ▼ L2 Normalize + k-means Assignment
┌─────────────────────────────────────────────────────────────────┐   ┌─────────────────────────────────────────────────────┐
│  Depth Features at Pixel Resolution (variant-dependent)         │   │  Feature Normalization                              │
│  Tensor:  (D_depth, 322, 644)  where D_depth ∈ {0,1,3,16}     │   │  Input:   (90, 322, 644)  or  (90+D, 322, 644)      │
│  Note:    Only used if variant != "none". For DCFA-only       │   │  Norm:    L2-normalize per pixel: x / ||x||₂        │
│           pipeline, variant="none" and this branch is empty.  │   │  Output:  (C, 322, 644)  where C = 90 or 90+D       │
└─────────────────────────────────────────────────────────────────┘   └─────────────────────────────────────────────────────┘
                                    │                                                     │
                                    └────────────────────►◄───────────────────────────────┘
                                                          │
                                                          ▼ Concatenate (if depth variant)
                                    ┌─────────────────────────────────────────────────────┐
                                    │  Combined Features                                  │
                                    │  Tensor:  (C, 322, 644)  where C = 90 + D_depth     │
                                    └─────────────────────────────────────────────────────┘
                                                          │
                                                          ▼ Cosine Similarity with Centroids
                                    ┌─────────────────────────────────────────────────────┐
                                    │  k-means Centroids (fitted on val set, 500 images)  │
                                    │  Shape:   (80, C)  where C = 90 (+ optional depth)  │
                                    │  Method:  MiniBatchKMeans, L2-normalized            │
                                    │  Mapping: Majority-vote φ: {0..79} → {0..18}        │
                                    └─────────────────────────────────────────────────────┘
                                                          │
                                                          ▼ Matrix Multiply
                                    ┌─────────────────────────────────────────────────────┐
                                    │  Similarity Map                                     │
                                    │  Tensor:  (80, 322, 644)  float32                   │
                                    │  Op:      sim = centroids @ feat_norm.reshape(C,-1) │
                                    └─────────────────────────────────────────────────────┘
                                                          │
                                                          ▼ Max over clusters per class
                                    ┌─────────────────────────────────────────────────────┐
                                    │  Class Logits (19 classes)                          │
                                    │  Tensor:  (19, 322, 644)  float32                   │
                                    │  Op:      For each class c:                        │
                                    │           logits[c] = max(sim[clusters_mapped_to_c])│
                                    └─────────────────────────────────────────────────────┘
                                                          │
                                                          ▼ DenseCRF (optional)
                                    ┌─────────────────────────────────────────────────────┐
                                    │  CRF-Refined Logits                                 │
                                    │  Input:   softmax(logits × 5)                       │
                                    │  Output:  (19, 322, 644)  float32                   │
                                    │  Params:  sxy=1, compat=3 (Gaussian)                │
                                    │           sxy=67, srgb=3, compat=4 (Bilateral)      │
                                    │           max_iter=10                               │
                                    └─────────────────────────────────────────────────────┘
                                                          │
                                                          ▼ Argmax
                                    ┌─────────────────────────────────────────────────────┐
                                    │  Semantic Prediction (resized resolution)           │
                                    │  Tensor:  (322, 644)  uint8  ∈ {0..18}              │
                                    └─────────────────────────────────────────────────────┘
                                                          │
                                                          ▼ Nearest-Neighbor Upsample
                                    ┌─────────────────────────────────────────────────────┐
                                    │  Semantic Pseudo-Label S (full resolution)          │
                                    │  Tensor:  (1024, 2048)  uint8  ∈ {0..18}            │
                                    │  Note:    Saved as `{stem}.png`                     │
                                    └─────────────────────────────────────────────────────┘
```

---

## 2. Instance Pseudo-Label Generation (DepthPro + Sobel + CC)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  INPUT: DepthPro Monocular Depth Map                                                    │
│  Tensor:  (512, 1024)  float32  [metric depth, saved as .npy]                           │
│  Range:   normalized to [0, 1] via dataset min-max                                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Resize to Semantic Resolution
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  Resized Depth Map                                                                      │
│  Tensor:  (1024, 2048)  float32  [bilinear interpolation to match semantic map S]       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Gaussian Smoothing
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  Smoothed Depth                                                                         │
│  Tensor:  (1024, 2048)  float64   [σ = 1.0 pixel]                                       │
│  Op:      scipy.ndimage.gaussian_filter(depth, sigma=1.0)                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Sobel Gradient
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  Gradient Components                                                                    │
│  Gx:      (1024, 2048)  float64   [sobel(depth, axis=1)]                                │
│  Gy:      (1024, 2048)  float64   [sobel(depth, axis=0)]                                │
│  |∇D|:    (1024, 2048)  float64   [sqrt(Gx² + Gy²)]                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Threshold
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  Binary Depth Edge Map E                                                                │
│  Tensor:  (1024, 2048)  bool                                                            │
│  Op:      E = (|∇D| > τ)   where τ = 0.20                                               │
│  Density: ~3-5% of pixels (parameter-dependent on scene complexity)                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Per-Class Connected Component Analysis
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  FOR each thing class c ∈ {11, 12, 13, 14, 15, 16, 17, 18}:                             │
│                                                                                         │
│  1. Class mask:    M_c = (S == c)                    →  (1024, 2048)  bool             │
│  2. Split mask:    M_c' = M_c ∧ ¬E                  →  (1024, 2048)  bool             │
│  3. Label CC:      {C₁, ..., Cₙ} = CC(M_c')         →  n components                    │
│  4. Area filter:   keep Cᵢ iff |Cᵢ| ≥ A_min = 1000  px                                │
│  5. Dilation:      dilate(Cᵢ, iterations=3)                                           │
│  6. Reclaim:       final_mask = Cᵢ ∪ (dilated ∧ M_c ∧ ¬assigned)                      │
│  7. Re-filter:     keep iff area(final_mask) ≥ A_min                                  │
│                                                                                         │
│  Output per class: list of (mask, class_id, normalized_score)                          │
│  Typical count:    ~17 valid instances per image (from ~292 raw CCs)                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Instance Map Assembly
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  Instance Pseudo-Label Map I                                                            │
│  Tensor:  (1024, 2048)  int32                                                           │
│  Encoding:  0 = background/unassigned                                                   │
│            1..M = instance IDs (M ≈ 17 per image)                                       │
│  Note:     Saved as `{stem}_instances.npz` with keys:                                   │
│            - "masks": (M, 1024, 2048)  bool                                             │
│            - "scores": (M,)  float32  [area/max_area]                                   │
│            - "num_valid": int                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. SIMCF-ABC Cross-Modal Consistency Filtering

### Step A: Instance Validates Semantics (Majority Vote)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  INPUTS:                                                                                │
│    S:  (1024, 2048)  uint8  ∈ {0..18}  — semantic map (from Stage 1)                   │
│    I:  (1024, 2048)  int32             — instance map (from Stage 2)                   │
│    φ:  {0..79} → {0..18, 255}          — cluster-to-class mapping (from k-means fit)   │
│                                                                                         │
│  Note: S is actually stored at original cluster resolution (80 clusters) before         │
│        mapping to 19 classes. For Step A, we work with the raw cluster map              │
│        S_raw ∈ {0..79}^(1024×2048) and apply φ on-the-fly.                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Per-Instance Majority Vote
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  FOR each instance region Iₖ (pixels where I == k, k > 0):                              │
│                                                                                         │
│  1. Majority class:                                                                     │
│     t* = argmax_{t∈{0..18}} Σ_{p∈Iₖ} 𝟙[φ(S_raw(p)) == t]                              │
│                                                                                         │
│  2. Inconsistent pixels:                                                                │
│     ℳₖ = {p ∈ Iₖ : φ(S_raw(p)) ≠ t*}                                                  │
│                                                                                         │
│  3. Replacement cluster:                                                                │
│     s* = argmax_{s:φ(s)=t*} Σ_{q∈Iₖ} 𝟙[S_raw(q) == s]                                 │
│                                                                                         │
│  4. Reassignment:                                                                       │
│     ∀p ∈ ℳₖ:  S_raw(p) ← s*                                                             │
│                                                                                         │
│  OUTPUT:  S_A  (1024, 2048)  uint8  ∈ {0..79}  — refined cluster map                   │
│  Effect:  Typically 0 pixels changed when instances derive from same semantic source;   │
│           structural no-op for CUPS-derived labels.                                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Step B: Semantics Validate Instances (Feature-Guided Merging)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  INPUTS:                                                                                │
│    S_A:    (1024, 2048)  uint8  ∈ {0..79}    — refined cluster map from Step A         │
│    I:      (1024, 2048)  int32               — instance map from Stage 2               │
│    F:      (N_patch, 768)  float32           — frozen DINOv2/DINOv3 features            │
│               N_patch = 1058 for 322×644 feature resolution                             │
│               Note: F is L2-normalized: ||F(p)||₂ = 1                                   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Build Adjacency Graph
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  Graph G = (V, E)                                                                       │
│  V = {k : |Iₖ| > 0, k > 0}         — valid instance IDs                                │
│  E = {(a,b) : dilate(I_a, r=3) ∩ I_b ≠ ∅}  — 3-pixel dilation overlap                 │
│                                                                                         │
│  Typical: |V| ≈ 17, |E| ≈ 30 per image                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Compute Per-Instance Features
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  FOR each instance k ∈ V:                                                               │
│                                                                                         │
│  1. Downsample instance mask to patch grid:                                             │
│     Ĩₖ = resize(Iₖ, (H_p, W_p))  →  (23, 46)  bool                                     │
│                                                                                         │
│  2. Mean-pool DINO features over instance patches:                                      │
│     f̄ₖ = (1/|Ĩₖ|) Σ_{p∈Ĩₖ} F(p)    →  (768,)  float32                                │
│                                                                                         │
│  3. L2-normalize:                                                                       │
│     f̂ₖ = f̄ₖ / ||f̄ₖ||₂              →  (768,)  float32, ||f̂ₖ||₂ = 1                   │
│                                                                                         │
│  OUTPUT:  {f̂ₖ}_{k∈V}  — one 768D vector per instance                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Merge Decision
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  FOR each edge (a, b) ∈ E:                                                              │
│                                                                                         │
│  1. Semantic check:  merge only if φ(majority_cluster(S_A[I_a])) ==                     │
│                      φ(majority_cluster(S_A[I_b]))  [same class]                        │
│                                                                                         │
│  2. Feature similarity:                                                                 │
│     cos(f̂_a, f̂_b) = f̂_a · f̂_b   [both are unit vectors]                                │
│                                                                                         │
│  3. Merge if:  cos(f̂_a, f̂_b) > τ_sim = 0.85                                           │
│                                                                                         │
│  4. Union-Find with path compression resolves transitive merges.                        │
│                                                                                         │
│  OUTPUT:  I_B  — merged instance map, renumbered contiguously                           │
│  Effect:  ~2.4 merges per image (7,252 total over 2,975 training images)                │
│           Median instance size: 5,502 px → 14,965 px                                    │
│           Stuff contamination:  50.7% → 28.0%                                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Step C: Depth Validates Semantics (Statistical Outlier Masking)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  INPUTS:                                                                                │
│    S_B:    (1024, 2048)  uint8  ∈ {0..79}    — cluster map from Step B                 │
│    D:      (1024, 2048)  float32             — DepthPro depth map (resized)             │
│    φ:      {0..79} → {0..18}                 — cluster-to-class mapping                │
│                                                                                         │
│  Note: Two-pass algorithm. Pass 1 is global over all 2,975 training images.            │
│        Pass 2 is per-image.                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Pass 1: Global Depth Statistics (offline)
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  FOR each class c ∈ {0..18}:                                                            │
│                                                                                         │
│  μ_c    = (1/N_c) Σ_{n=1}^{2975} Σ_{p:φ(S⁽ⁿ⁾(p))=c} D⁽ⁿ⁾(p)                           │
│  σ_c²   = (1/N_c) Σ_{n=1}^{2975} Σ_{p:φ(S⁽ⁿ⁾(p))=c} (D⁽ⁿ⁾(p) - μ_c)²                  │
│                                                                                         │
│  where N_c = total pixels across all images assigned to class c                         │
│                                                                                         │
│  Storage:  (19, 2)  array of (μ_c, σ_c)  — computed once, reused per image              │
│  Method:   Welford's online algorithm for numerical stability                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Pass 2: Per-Image Outlier Masking
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  FOR each pixel p ∈ {1..(1024×2048)}:                                                   │
│                                                                                         │
│  c = φ(S_B(p))        — mapped class label                                              │
│                                                                                         │
│  IF |D(p) - μ_c| > λ_σ · σ_c    where λ_σ = 3.0:                                       │
│      S_C(p) ← 255     — mark as ignore (training mask)                                  │
│  ELSE:                                                                                  │
│      S_C(p) ← S_B(p)  — keep original cluster assignment                                │
│                                                                                         │
│  OUTPUT:  S_C  (1024, 2048)  uint8  ∈ {0..79, 255}                                     │
│  Effect:  ~1.36% of pixels masked (85M pixels over 2,975 images)                        │
│           Primarily removes mislabeled sky/building boundaries at depth discontinuities │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Panoptic Assembly (Instance-First Merge)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  INPUTS:                                                                                │
│    S_C:    (1024, 2048)  uint8  ∈ {0..79, 255}   — final semantic cluster map          │
│    I_B:    (1024, 2048)  int32                   — final merged instance map             │
│    φ:      {0..79} → {0..18}                     — cluster-to-class mapping             │
│                                                                                         │
│  CONSTANTS:                                                                             │
│    STUFF_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}   — 11 stuff classes                 │
│    THING_IDS = {11, 12, 13, 14, 15, 16, 17, 18}     — 8 thing classes                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Instance-First Assignment
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  1. Initialize panoptic map P = zeros(1024, 2048), next_id = 1                         │
│                                                                                         │
│  2. FOR each instance mask m in I_B (sorted by descending area):                        │
│       cls = majority_class(S_C[m])  ∈ THING_IDS                                         │
│       new_pixels = m ∧ (P == 0)                                                         │
│       IF new_pixels.sum() ≥ 10:                                                         │
│           P[new_pixels] = next_id                                                       │
│           segment_id[next_id] = cls                                                     │
│           next_id += 1                                                                  │
│                                                                                         │
│  3. FOR each stuff class c ∈ STUFF_IDS:                                                 │
│       mask = (φ(S_C) == c) ∧ (P == 0)   [unassigned pixels of class c]                 │
│       IF mask.sum() ≥ 64:                                                               │
│           P[mask] = next_id                                                             │
│           segment_id[next_id] = c                                                       │
│           next_id += 1                                                                  │
│                                                                                         │
│  4. Remaining unassigned pixels (P == 0) are implicitly ignored during training.        │
│                                                                                         │
│  OUTPUT:  Panoptic pseudo-label map P  (1024, 2048)  int32                             │
│           Segment metadata:  dict[segment_id → class_label]                             │
│                                                                                         │
│  FORMAT:  Saved as CUPS-compatible `.pt` distribution files:                            │
│           - "sem_seg": (19, 1024, 2048)  float32  [soft semantic logits]               │
│           - "instances": (M, 1024, 2048)  float32  [instance mask logits]              │
│           - "panoptic": (1024, 2048)  int32  [panoptic IDs]                            │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Complete Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         END-TO-END DIMENSION TRACKING                                   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  RGB Input           (1024, 2048, 3)          uint8                                     │
│       │                                                                                 │
│       ▼ Resize        (322, 644, 3)           float32                                   │
│       │                                                                                 │
│       ▼ DINOv2/14     (1058, 768)             float32   [patch features]                │
│       │                                                                                 │
│       ▼ Segment_TR    (90, 322, 644)          float32   [pixel-dense codes]             │
│       │                                                                                 │
│       ▼ Pool→DCFA     (90, 23, 46)            float32   [patch-grid codes]              │
│       │            ┌─────────────────────────────────────┐                              │
│       │            │ DCFA: (1,1058,90) + (1,1058,16)    │                              │
│       │            │       → (1,1058,90) adjusted       │                              │
│       │            └─────────────────────────────────────┘                              │
│       ▼ Upsample      (90, 322, 644)          float32   [adjusted pixel codes]          │
│       │                                                                                 │
│       ▼ k-means       (19, 322, 644)          float32   [class logits]                  │
│       │                                                                                 │
│       ▼ CRF + Argmax  (322, 644)              uint8     ∈ {0..18}                       │
│       │                                                                                 │
│       ▼ Upsample      (1024, 2048)            uint8     = S (semantic map)              │
│                                                                                         │
│  DepthPro Input      (512, 1024)              float32   [raw depth]                     │
│       │                                                                                 │
│       ▼ Resize        (1024, 2048)            float32   [match semantic resolution]     │
│       │                                                                                 │
│       ▼ Gaussian      (1024, 2048)            float64   [σ = 1.0]                       │
│       │                                                                                 │
│       ▼ Sobel + Thresh (1024, 2048)           bool      [τ = 0.20] = E                  │
│       │                                                                                 │
│       ▼ Per-Class CC  List[(M,1024,2048)]     bool      [A_min = 1000]                  │
│       │                                                                                 │
│       ▼ Assemble      (1024, 2048)            int32     = I (instance map)              │
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┐        │
│  │ SIMCF-ABC                                                                   │        │
│  │   Step A:  S + I  →  S_A   (instance majority vote on semantics)           │        │
│  │   Step B:  S_A + I + F  →  I_B  (DINO-guided instance merging, τ_sim=0.85) │        │
│  │   Step C:  S_A + D  →  S_C  (depth outlier masking, λ_σ=3.0)               │        │
│  └─────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                         │
│       │                                                                                 │
│       ▼ Panoptic Merge  (1024, 2048)          int32     = P (panoptic IDs)              │
│                                                                                         │
│  OUTPUT:  CUPS-format `.pt` files with sem_seg, instances, panoptic tensors            │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```
       
---

## 6. Key Parameter Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| `k` | 80 | K-means overclustering factor |
| `patch_size` | 14 | DINOv2 ViT-B patch size |
| `crop_size` | 322 | CAUSE-TR sliding window crop |
| `H_orig × W_orig` | 1024 × 2048 | Cityscapes original resolution |
| `H × W` | 322 × 644 | Resized feature resolution |
| `H_p × W_p` | 23 × 46 | Patch grid for DCFA |
| `N` | 1058 | Total patches per image |
| `code_dim` | 90 | CAUSE-TR output dimension |
| `depth_dim` | 16 | Sinusoidal depth encoding |
| `DCFA hidden` | 384 | MLP hidden dimension |
| `DCFA layers` | 2 | MLP depth |
| `τ` (Sobel) | 0.20 | Depth gradient threshold |
| `σ_blur` | 1.0 | Gaussian smoothing sigma |
| `A_min` | 1000 | Minimum instance area (pixels) |
| `dilation_iters` | 3 | Boundary reclamation dilation |
| `τ_sim` | 0.85 | DINO cosine similarity merge threshold |
| `λ_σ` | 3.0 | Depth outlier Z-score threshold |
| `τ_crf` | 5.0 | CRF temperature (logits multiplier) |
| `freq_bands` | 8 | Sinusoidal encoding frequencies |

---

## 7. Checkpoint Files Produced

```
cityscapes_root/
└── cups_pseudo_labels_dcfa_simcf_abc/
    ├── train/
    │   ├── aachen/
    │   │   ├── aachen_000000_000019.png           # Semantic cluster map (k=80 values)
    │   │   ├── aachen_000000_000019_instances.npz  # Instance masks + scores
    │   │   └── aachen_000000_000019.pt             # CUPS-format distribution
    │   └── ... (2,975 images)
    ├── val/
    │   └── ... (500 images)
    └── kmeans_centroids.npz                        # Shared centroids + mapping
        ├── centroids:        (80, 90)   float64    # L2-normalized k-means centroids
        ├── cluster_to_class: (80,)      uint8      # Majority-vote class assignment
        ├── variant:          "none"                # Depth variant used for fitting
        └── alpha:            1.0                   # Depth scaling (unused for variant=none)
```
