# MBPS Pseudo-Label Generation Pipeline: k=80 + DepthPro + DCFA + SIMCF-ABC

> **Verified by author:** This pipeline uses **DINOv2 ViT-B/14** (CAUSE/Segment_TR) for semantic feature extraction. **DepthPro** is the sole input for instance generation. **SIMCF-ABC operates purely on semantic labels, instance labels, and depth maps** — no DINOv3 features are used anywhere in pseudo-label generation. DINOv3 ViT-B/16 appears only as the frozen backbone in downstream Stage-2 training.

---

## Complete Architectural Pipeline (ASCII Art)

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                          INPUT: Cityscapes Training Set (2,975 images)                                           ║
║                                                                                                                                                  ║
║   ┌──────────────────────────────┐        ┌──────────────────────────────┐                                                                       ║
║   │  RGB Image 1024×2048         │        │  DepthPro Monocular Depth    │                                                                       ║
║   │  (leftImg8bit.png)           │        │  (512×1024, normalized [0,1])│                                                                       ║
║   └──────────────┬───────────────┘        └──────────────┬───────────────┘                                                                       ║
╚══════════════════╪════════════════════════════════════════╪═════════════════════════════════════════════════════════════════════════════════════╝
                   │                                        │
                   │                                        │
                   ▼                                        │
                                                            │
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — FEATURE LEVEL: SEMANTIC PSEUDO-LABEL GENERATION (DCFA + k=80 Overclustering)                                                            │
│  Backbone: DINOv2 ViT-B/14 (CAUSE) → 90D Segment_TR codes  |  Raises mIoU: 52.69% → 55.29%  |  Contributes +0.68 PQ                               │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────┐              ┌─────────────────────────────────────────┐
    │  CAUSE Segment_TR Codes     │              │  Sinusoidal Depth Encoding              │
    │  90-D per patch (32×64)     │              │  8 freq bands × [sin, cos] = 16-D       │
    │  from DINOv2 ViT-B/14       │              │  freqs: {1,2,4,8,16,32,64,128} × πd     │
    └─────────────┬───────────────┘              └─────────────────┬───────────────────────┘
                  │                                                │
                  │    ┌───────────────────────────────────────────┘
                  │    │
                  ▼    ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  DCFA: Depth-Conditioned Feature Adapter  (~40K params)                                 │
    │  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
    │  │  Input: [90-D codes  ‖  16-D sinusoidal depth]  →  (106-D)                      │   │
    │  │                                                                                 │   │
    │  │  Layer 1:  Linear(106 → 384) + LayerNorm + ReLU                                 │   │
    │  │  Layer 2:  Linear(384 → 384) + LayerNorm + ReLU                                 │   │
    │  │  Output:   Linear(384 → 90)  ← ZERO-INITIALIZED (starts as identity)            │   │
    │  │              ↓                                                                    │   │
    │  │  Skip:     adjusted_codes = codes + output_residual                               │   │
    │  └─────────────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                         │
    │  Training Loss:  L = L_depth-corr  +  λ_preserve × ||A_θ(f,d) − f||²   (λ = 20.0)     │
    │  Best Checkpoint:  results/depth_adapter/V3_dd16_h384_l2/best.pt                        │
    └────────────────────────────────────────────────┬────────────────────────────────────────┘
                                                     │
                                                     ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  MiniBatchKMeans Clustering  (k = 80 clusters)                                          │
    │  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
    │  │  • Fit on L2-normalized adjusted 90-D codes from all train images               │   │
    │  │  • batch_size = 10,000   |   max_iter = 300   |   n_init = 3   |   random_state=42│   │
    │  │  • 80 cluster centroids learned in 90-D space                                   │   │
    │  └─────────────────────────────────────────────────────────────────────────────────┘   │
    └────────────────────────────────────────────────┬────────────────────────────────────────┘
                                                     │
                                                     ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  Cluster Assignment & Upsampling                                                        │
    │  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
    │  │  • Predict cluster ID (0–79) for each of 32×64 = 2,048 patches per image        │   │
    │  │  • Nearest-neighbor upsample to 512×1024 pixel resolution                       │   │
    │  │  • Output: semantic cluster map  (uint8 PNG: 0–79 + 255 ignore)                 │   │
    │  └─────────────────────────────────────────────────────────────────────────────────┘   │
    └────────────────────────────────────────────────┬────────────────────────────────────────┘
                                                     │
                                                     ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  Cluster-to-Class Mapping (Validation-Set Majority Vote)                                │
    │  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
    │  │  For each cluster c ∈ [0,79]:  class(c) = argmax_t  count(cluster=c, GT=t)      │   │
    │  │  • Maps 80 clusters → 19 Cityscapes trainIDs                                    │   │
    │  │  • Road dominates (~31/80 clusters); rare classes may get 0–1 clusters          │   │
    │  └─────────────────────────────────────────────────────────────────────────────────┘   │
    └────────────────────────────────────────────────┬────────────────────────────────────────┘
                                                     │
                    ┌────────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  SEMANTIC PSEUDO-LABEL  (trainIDs 0–18)                                                 │
    │  Resolution: 512×1024  |  Classes: 19  |  Overclustered: 80 clusters before mapping   │
    └─────────────────────────────────────────────────────────────────────────────────────────┘


┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — INSTANCE LEVEL: DEPTH-GUIDED INSTANCE PSEUDO-LABEL GENERATION (DepthPro + Sobel + CC)                                                    │
│  Input: ONLY DepthPro depth map + semantic thing masks  |  NO vision features used here                                                            │
│  Reduces ~292 raw CCs/image → ~17 valid instances  |  PQ_things baseline: 12.31                                                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  DepthPro Depth Map (512×1024)                                                          │
    └────────────────────────────────────────────────┬────────────────────────────────────────┘
                                                     │
                    ┌────────────────────────────────┼────────────────────────────────┐
                    │                                │                                │
                    ▼                                ▼                                ▼
    ┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
    │  Gaussian Smoothing      │    │  Sobel Edge Detection    │    │  Class Mask Extraction   │
    │  σ = 0.0 (no blur for    │    │  Gx = Sobel_x * D        │    │  For each thing class    │
    │  DepthPro — already      │    │  Gy = Sobel_y * D        │    │  c ∈ {person,rider,car,  │
    │  clean)                  │    │  ||∇D|| = √(Gx²+Gy²)    │    │  truck,bus,train,        │
    │                          │    │                          │    │  motorcycle,bicycle}     │
    └────────────┬─────────────┘    └────────────┬─────────────┘    └────────────┬─────────────┘
                 │                               │                               │
                 └───────────────────────────────┼───────────────────────────────┘
                                                 │
                                                 ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  Depth Edge Threshold:  ||∇D|| > τ = 0.20                                               │
    └────────────────────────────────────────────────┬────────────────────────────────────────┘
                                                     │
                                                     ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  Per-Class Connected Component Extraction                                               │
    │  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
    │  │  For each thing class c:                                                          │   │
    │  │    1. M_c = {pixels where semantic == c}                                          │   │
    │  │    2. M'_c = M_c \ depth_edges   (split at depth discontinuities)                 │   │
    │  │    3. {CC₁, CC₂, ...} = connected_components(M'_c)                                │   │
    │  │    4. Filter: keep CCs with area ≥ A_min = 1000 pixels                            │   │
    │  │    5. Dilate by 3 iterations → reclaim boundary pixels                            │   │
    │  └─────────────────────────────────────────────────────────────────────────────────┘   │
    └────────────────────────────────────────────────┬────────────────────────────────────────┘
                                                     │
                                                     ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  INSTANCE PSEUDO-LABELS                                                                 │
    │  • ~17 valid instances per image (down from ~292 raw CCs)                               │
    │  • Scores = area / max_area (normalized by largest instance)                            │
    │  • Output: _instance.png (uint16)  +  NPZ masks                                         │
    └─────────────────────────────────────────────────────────────────────────────────────────┘


┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — LABEL LEVEL: SEMANTIC-INSTANCE MUTUAL CONSISTENCY FILTERING (SIMCF-ABC)                                                                  │
│  Inputs: semantic labels + instance labels + DepthPro depth maps  |  NO DINOv2/DINOv3 features used                                                 │
│  Raises PQ: 24.54 → 25.27 (SIMCF alone)  |  Full DCFA+SIMCF: 24.54 → 25.85 (+1.31 PQ)                                                              │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────┐              ┌─────────────────────────────┐
    │  Semantic Pseudo-Label      │              │  Instance Pseudo-Label      │
    │  (cluster IDs 0–79)         │              │  (uint16 instance IDs)      │
    └─────────────┬───────────────┘              └─────────────┬───────────────┘
                  │                                            │
                  │         ┌──────────────────────────────────┘
                  │         │
                  ▼         ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  STEP A: INSTANCE VALIDATES SEMANTICS  (Majority-Vote Consistency)                      │
    │  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
    │  │  For each instance I_k:                                                           │   │
    │  │    1. t* = majority_trainID( semantic[pixels in I_k] )                            │   │
    │  │    2. Find best_cluster = most frequent cluster ID that maps to t* in I_k         │   │
    │  │    3. Reassign all inconsistent pixels in I_k to best_cluster                     │   │
    │  │                                                                                   │   │
    │  │  Effect: Structurally a no-op for CUPS-derived labels (0 pixels changed)          │   │
    │  └─────────────────────────────────────────────────────────────────────────────────┘   │
    └────────────────────────────────────────────────┬────────────────────────────────────────┘
                                                     │
                                                     ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  STEP B: SEMANTICS VALIDATE INSTANCES  (Instance Merging)  ★ MOST CRITICAL STEP        │
    │  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
    │  │  Inputs: semantic map + instance map + cluster_to_class mapping                   │   │
    │  │  NOTE: No external vision features (DINOv2/DINOv3) are used in this step.         │   │
    │  │        Merging decisions are based on adjacency + semantic class consistency.     │   │
    │  │                                                                                   │   │
    │  │  1. Build adjacency graph: dilate each mask by d=3px, check overlap               │   │
    │  │  2. For adjacent instances with same mapped trainID: merge via union-find         │   │
    │  │  3. Renumber instances contiguously                                               │   │
    │  │                                                                                   │   │
    │  │  Results:  44 → 22 instances/image  |  median 5,502→14,965 px  |  7,252 total    │   │
    │  │           stuff contamination: 50.7% → 28.0%  |  PQ_things +1.33  |  bus +8.8 PQ │   │
    │  └─────────────────────────────────────────────────────────────────────────────────┘   │
    └────────────────────────────────────────────────┬────────────────────────────────────────┘
                                                     │
                                                     ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  STEP C: DEPTH VALIDATES SEMANTICS  (Statistical Outlier Masking)                       │
    │  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
    │  │  Input: DepthPro depth map                                                        │   │
    │  │                                                                                   │   │
    │  │  Pass 1 — Global Statistics (across all 2,975 training images):                   │   │
    │  │    For each class c ∈ [0,18]:                                                     │   │
    │  │      μ_c = mean( DepthPro[pixels where semantic maps to c] )                      │   │
    │  │      σ_c = std(  DepthPro[pixels where semantic maps to c] )                      │   │
    │  │                                                                                   │   │
    │  │  Pass 2 — Per-Image Masking:                                                      │   │
    │  │    For each pixel p with mapped class c:                                          │   │
    │  │      if | D(p) − μ_c | > 3 × σ_c :   semantic[p] ← 255 (ignore)                   │   │
    │  │                                                                                   │   │
    │  │  Result: ~85M pixels masked (1.36% of all pixels)  |  PQ_stuff +0.30              │   │
    │  └─────────────────────────────────────────────────────────────────────────────────┘   │
    └────────────────────────────────────────────────┬────────────────────────────────────────┘
                                                     │
                    ┌────────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  REFINED SEMANTIC + INSTANCE MAPS                                                       │
    │  • Semantic: cluster IDs with outliers masked (255)                                     │
    │  • Instance: merged, renumbered uint16 IDs                                              │
    └─────────────────────────────────────────────────────────────────────────────────────────┘


┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4 — PANOPTIC MERGE: SEMANTIC + INSTANCE → PANOPTIC PSEUDO-LABEL                                                                               │
│  Encoding: panoptic_id = class_id × 1000 + instance_id                                                                                               │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────┐              ┌─────────────────────────────┐              ┌─────────────────────────────┐
    │  Refined Semantic Map       │              │  Refined Instance Map       │              │  stuff_things.json          │
    │  (trainIDs 0–18 + 255)      │              │  (uint16 instance IDs)      │              │  (unsupervised split)       │
    └─────────────┬───────────────┘              └─────────────┬───────────────┘              └─────────────┬───────────────┘
                  │                                            │                                            │
                  │         ┌──────────────────────────────────┘                                            │
                  │         │                                                                             │
                  ▼         ▼                                                                             ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │  generate_panoptic_map()                                                                                                                          │
    │                                                                                                                                                   │
    │  Step 1 — PLACE THINGS (highest priority, they can overlap stuff)                                                                                 │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
    │  │  • Sort instances by score descending (largest/most confident first)                                                                            ││
    │  │  • For each instance mask:                                                                                                                      ││
    │  │      majority_cls = mode( semantic[pixels under mask] )   (skip if < 10 px)                                                                     ││
    │  │      Skip if majority_cls ∉ thing_ids                                                                                                           ││
    │  │      valid_mask = mask & ~already_assigned                                                                                                      ││
    │  │      panoptic_id = majority_cls × 1000 + instance_counter[majority_cls]                                                                         ││
    │  │      Mark pixels as assigned                                                                                                                    ││
    │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
    │                                                                                                                                                   │
    │  Step 2 — PLACE STUFF (fill remaining unassigned pixels)                                                                                          │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
    │  │  • For each stuff class c:  mask = (semantic == c) & ~assigned                                                                                  ││
    │  │  • Skip if area < min_stuff_area = 64 px                                                                                                        ││
    │  │  • panoptic_id = c × 1000 + 0    (instance_id = 0 for all stuff)                                                                                ││
    │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
    │                                                                                                                                                   │
    │  Step 3 — FALLBACK CC (uncovered thing pixels get CC instances)                                                                                   │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
    │  │  • For each thing class c:  remaining = (semantic == c) & ~assigned                                                                             ││
    │  │  • connected_components(remaining) → new instances                                                                                              ││
    │  │  • panoptic_id = c × 1000 + next_instance_id    (low confidence, score=0.1)                                                                     ││
    │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                                                                              │
                                                                                              ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  OUTPUT: PANOPTIC PSEUDO-LABELS                                                         │
    │  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
    │  │  • _panoptic.npy  — int32 Cityscapes-encoded panoptic map                       │   │
    │  │  • _panoptic.png  — uint16 visualization                                        │   │
    │  │  • segment_info   — JSON list per segment with id, category_id, isthing, area   │   │
    │  │  • 2,975 images × 3 files = ~8,925 output files                                 │   │
    │  └─────────────────────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────────────────────┘


┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  DOWNSTREAM: SUPERVISED TRAINING ON PSEUDO-LABELS                                                                                                  │
│  NOTE: DINOv3 ViT-B/16 is the frozen training backbone for Mask2Former / Cascade Mask R-CNN — NOT used in pseudo-label generation.                 │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  Panoptic Pseudo-Labels (PQ = 25.85)                                                    │
    └────────────────────────────────────────────────┬────────────────────────────────────────┘
                                                     │
                    ┌────────────────────────────────┼────────────────────────────────┐
                    │                                │                                │
                    ▼                                ▼                                ▼
    ┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
    │  Stage 2: Mask2Former    │    │  Stage 2: Cascade Mask   │    │  Stage 3: Self-Training  │
    │  (frozen DINOv3 backbone)│    │  R-CNN                  │    │  (refinement rounds)     │
    │  ~28% PQ                 │    │                         │    │  Final: 35.83% PQ        │
    └──────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘
```

---

## Feature Usage Summary (Author-Verified)

| Stage | What Happens | Inputs Used |
|-------|-------------|-------------|
| **Semantic Generation** | DCFA adapts CAUSE 90D codes; k-means (k=80) clusters them | **DINOv2 ViT-B/14** (CAUSE `Segment_TR`) + **DepthPro** depth |
| **Instance Generation** | Depth edges split semantic thing masks into instances | **DepthPro ONLY** — no vision features |
| **SIMCF Step A** | Majority vote within instances | Semantic + instance labels only |
| **SIMCF Step B** | Merge adjacent same-class instances | Semantic + instance labels only — **no DINOv2/DINOv3 features** |
| **SIMCF Step C** | Mask depth outliers | **DepthPro** depth maps |
| **Panoptic Merge** | Combine semantic + instance into panoptic IDs | Semantic + instance + stuff/things split |
| **Training Backbone** | Frozen feature extractor for panoptic model | **DINOv3 ViT-B/16** (downstream only) |

**Critical Correction:** Step B does **not** use DINOv3 (or any) vision features for merge decisions in this production pipeline. Merging is based purely on **adjacency + semantic class consistency**.

---

## Key Parameters Reference

| Component | Parameter | Value | Description |
|-----------|-----------|-------|-------------|
| **DCFA** | `code_dim` | 90 | CAUSE Segment_TR dimension (from DINOv2) |
| **DCFA** | `depth_dim` | 16 | Sinusoidal depth encoding dimension |
| **DCFA** | `hidden_dim` | 384 | MLP hidden layer size |
| **DCFA** | `num_layers` | 2 | MLP depth |
| **DCFA** | `λ_preserve` | 20.0 | Preservation loss weight |
| **K-Means** | `k` | 80 | Overclustering factor |
| **K-Means** | `batch_size` | 10,000 | MiniBatchKMeans batch size |
| **K-Means** | `max_iter` | 300 | Maximum iterations |
| **Instance Gen** | `τ` (edge) | 0.20 | Depth Sobel gradient threshold |
| **Instance Gen** | `A_min` | 1000 px | Minimum instance area |
| **Instance Gen** | `dilation` | 3 | Boundary reclamation iterations |
| **SIMCF-A** | — | — | Majority vote (no-op for CUPS) |
| **SIMCF-B** | `dilate_px` | 3 | Adjacency detection dilation |
| **SIMCF-C** | `σ_threshold` | 3.0 | Depth outlier sigma multiplier |
| **Panoptic** | `min_stuff_area` | 64 | Minimum stuff segment area |
| **Panoptic** | `label_divisor` | 1000 | Cityscapes encoding divisor |

---

## Ablation Results (Cityscapes Train, 2,975 images)

| Variant | PQ | PQ_stuff | PQ_things | mIoU | ΔPQ |
|---------|-----|----------|-----------|------|------|
| A0: Raw k=80 + DepthPro | 24.54 | 33.43 | 12.31 | 56.56 | — |
| + DCFA only | 25.22 | 33.99 | 13.16 | 56.16 | +0.68 |
| + SIMCF-ABC only | 25.27 | 33.73 | 13.64 | 56.57 | +0.73 |
| **+ DCFA + DepthPro + SIMCF-ABC (FULL)** | **25.85** | **33.96** | **14.70** | **56.22** | **+1.31** |

---

*Generated from deep codebase analysis of the MBPS project. Author-verified: no DINOv3 features are used in pseudo-label generation.*
