# Full Pseudo-Label Generation Pipeline: MBPS (k=80 + DepthPro + DCFA + SIMCF-ABC)

> **Author-verified pipeline.** Semantic features come from **DINOv2 ViT-B/14** via CAUSE/Segment_TR (90D codes). Instance generation uses **DepthPro only** — no vision features. SIMCF-ABC operates purely on semantic labels, instance labels, and DepthPro depth maps. DINOv3 ViT-B/16 is used **only** as the frozen downstream training backbone (Stage 2 Mask2Former / Cascade Mask R-CNN), not in pseudo-label generation.

---

## ASCII Architecture Diagram

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
│  STAGE 1 — FEATURE LEVEL: SEMANTIC PSEUDO-LABEL GENERATION                                                                                         │
│  Backbone: DINOv2 ViT-B/14 (CAUSE) → 90D Segment_TR codes                                                                                         │
│  Output: Semantic cluster map (512×1024, uint8: 0–79 + 255 ignore)                                                                                │
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
    │  │  Input: [90-D codes ‖ 16-D sinusoidal depth] → (106-D)                          │   │
    │  │  Layer 1: Linear(106 → 384) + LayerNorm + ReLU                                  │   │
    │  │  Layer 2: Linear(384 → 384) + LayerNorm + ReLU                                  │   │
    │  │  Output:  Linear(384 → 90)  ← ZERO-INITIALIZED (starts as identity)             │   │
    │  │  Skip:    adjusted_codes = codes + output_residual                                │   │
    │  └─────────────────────────────────────────────────────────────────────────────────┘   │
    │  Loss: L_depth-corr + λ_preserve × ||A_θ(f,d) − f||²   (λ = 20.0)                     │
    │  Checkpoint: results/depth_adapter/V3_dd16_h384_l2/best.pt                              │
    └────────────────────────────────────────────────┬────────────────────────────────────────┘
                                                     │
                                                     ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  MiniBatchKMeans Clustering  (k = 80 clusters)                                          │
    │  • Fit on L2-normalized adjusted 90-D codes from all train images                       │
    │  • batch_size = 10,000  |  max_iter = 300  |  n_init = 3  |  random_state = 42          │
    │  • 80 cluster centroids learned in 90-D space                                           │
    └────────────────────────────────────────────────┬────────────────────────────────────────┘
                                                     │
                                                     ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  Cluster Assignment & Upsampling                                                        │
    │  • Predict cluster ID (0–79) for each of 32×64 = 2,048 patches per image                │
    │  • Nearest-neighbor upsample to 512×1024 pixel resolution                               │
    └────────────────────────────────────────────────┬────────────────────────────────────────┘
                                                     │
                                                     ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  Cluster-to-Class Mapping (Validation-Set Majority Vote)                                │
    │  • class(c) = argmax_t count(cluster=c, GT=t) for each cluster c ∈ [0,79]              │
    │  • Maps 80 clusters → 19 Cityscapes trainIDs                                            │
    │  • Road dominates (~31/80 clusters); rare classes may get 0–1 clusters                  │
    └────────────────────────────────────────────────┬────────────────────────────────────────┘
                                                     │
                    ┌────────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  SEMANTIC PSEUDO-LABEL  (trainIDs 0–18, 512×1024)                                       │
    └─────────────────────────────────────────────────────────────────────────────────────────┘


┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — INSTANCE LEVEL: DEPTH-GUIDED INSTANCE GENERATION                                                                                        │
│  Input: ONLY DepthPro depth map + semantic thing masks  |  NO vision features used                                                                 │
│  Output: Instance masks (512×1024, uint16 instance IDs)                                                                                            │
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
    │  σ = 0.0 (no blur)       │    │  Gx = Sobel_x * D        │    │  For each thing class    │
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
    │  1. M_c = {pixels where semantic == c}                                                  │
    │  2. M'_c = M_c \ depth_edges   (split at depth discontinuities)                         │
    │  3. {CC₁, CC₂, ...} = connected_components(M'_c)                                        │
    │  4. Filter: keep CCs with area ≥ A_min = 1000 pixels                                    │
    │  5. Dilate by 3 iterations → reclaim boundary pixels                                    │
    └────────────────────────────────────────────────┬────────────────────────────────────────┘
                                                     │
                                                     ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  INSTANCE PSEUDO-LABELS                                                                 │
    │  • ~17 valid instances per image (down from ~292 raw CCs)                               │
    │  • Scores = area / max_area (normalized by largest instance)                            │
    │  • Output: _instance.png (uint16) + NPZ masks                                           │
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
                  ▼                                            ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  STEP A: INSTANCE VALIDATES SEMANTICS  (Majority Vote)                                  │
    │  • Within each instance I_k, majority-vote the mapped trainID t*                        │
    │  • Reassign inconsistent pixels to the best cluster mapping to t*                       │
    │  • Effect: Structurally a no-op for CUPS-derived labels (0 pixels changed)              │
    └─────────────────────────────────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  STEP B: SEMANTICS VALIDATE INSTANCES  (Instance Merging)  ★ MOST CRITICAL STEP        │
    │  • NO external vision features (DINOv2/DINOv3) are used in this step                    │
    │  • Merging decisions are based on adjacency + semantic class consistency only           │
    │  • Build adjacency graph: dilate each mask by d=3px, check overlap                      │
    │  • For adjacent instances with same mapped trainID: merge via union-find                │
    │  • Renumber instances contiguously                                                      │
    │  • Results: 44 → 22 instances/image  |  median 5,502→14,965 px  |  PQ_things +1.33      │
    └─────────────────────────────────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  STEP C: DEPTH VALIDATES SEMANTICS  (3-Sigma Outlier Masking)                           │
    │  • Pass 1 (global): Compute per-class depth mean μ_c and std σ_c across all 2,975 imgs  │
    │  • Pass 2 (per-image): Mask pixel as ignore (255) if |D(p) − μ_c| > 3 × σ_c             │
    │  • Result: ~85M pixels masked (1.36% of all pixels)  |  PQ_stuff +0.30                  │
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
                  └────────────────────────┬───────────────────┘                                            │
                                           │                                                                │
                                           ▼                                                                ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │  generate_panoptic_map()                                                                                                                          │
    │                                                                                                                                                   │
    │  Step 1 — PLACE THINGS (highest priority, they can overlap stuff)                                                                                 │
    │    • Sort instances by score descending (largest/most confident first)                                                                            │
    │    • For each instance mask: majority_cls = mode(semantic[pixels under mask])                                                                     │
    │    • Skip if majority_cls ∉ thing_ids  |  Skip if < 10 pixels                                                                                     │
    │    • valid_mask = mask & ~already_assigned                                                                                                        │
    │    • panoptic_id = majority_cls × 1000 + instance_counter[majority_cls]                                                                           │
    │                                                                                                                                                   │
    │  Step 2 — PLACE STUFF (fill remaining unassigned pixels)                                                                                          │
    │    • For each stuff class c: mask = (semantic == c) & ~assigned                                                                                   │
    │    • Skip if area < min_stuff_area = 64 px                                                                                                        │
    │    • panoptic_id = c × 1000 + 0    (instance_id = 0 for all stuff)                                                                                │
    │                                                                                                                                                   │
    │  Step 3 — FALLBACK CC (uncovered thing pixels get CC instances)                                                                                   │
    │    • For each thing class c: remaining = (semantic == c) & ~assigned                                                                              │
    │    • connected_components(remaining) → new instances                                                                                              │
    │    • panoptic_id = c × 1000 + next_instance_id    (low confidence, score=0.1)                                                                     │
    └─────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                                                                              │
                                                                                              ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  OUTPUT: PANOPTIC PSEUDO-LABELS                                                         │
    │  • _panoptic.npy  — int32 Cityscapes-encoded panoptic map                               │
    │  • _panoptic.png  — uint16 visualization                                                │
    │  • segment_info   — JSON list per segment with id, category_id, isthing, area           │
    │  • 2,975 images × 3 files = ~8,925 output files                                         │
    └─────────────────────────────────────────────────────────────────────────────────────────┘


┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  DOWNSTREAM: SUPERVISED TRAINING                                                                                                                     │
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

## Complete Textual Prompt (For Documentation / AI Generation)

### Overview

We present a four-stage depth-conditioned pseudo-label generation pipeline for unsupervised panoptic segmentation on Cityscapes. The pipeline takes only RGB images and monocular depth maps (DepthPro) as inputs, and produces high-quality panoptic pseudo-labels used to train Mask2Former and Cascade Mask R-CNN. The pipeline raises pseudo-label PQ from 24.54 to 25.85 (+1.31) through three orthogonal interventions at distinct representation levels: feature-level depth adaptation (DCFA), instance-level depth-guided splitting (DepthPro + CC), and label-level cross-modal consistency filtering (SIMCF-ABC).

### Stage 1: Semantic Pseudo-Label Generation (DCFA + k=80 Overclustering)

**Input:** Cityscapes RGB image (1024×2048).

**Feature Extraction:** We use CAUSE (Segment_TR) with a frozen DINOv2 ViT-B/14 backbone to extract 90-dimensional patch-level feature codes. The backbone is loaded from `dinov2_vit_base_14.pth`. Features are extracted via a sliding-window approach with 322×322 crops and bilinear interpolation, producing a 90×H×W feature map.

**Depth Conditioning (DCFA):** A lightweight Depth-Conditioned Feature Adapter (~40K parameters) adjusts the 90D codes before clustering. The adapter is a 2-layer MLP with hidden dimension 384:
- Input: concatenation of 90D CAUSE codes and 16D sinusoidal depth encoding
- Sinusoidal encoding uses 8 octave-spaced frequencies: {1, 2, 4, 8, 16, 32, 64, 128} × πd
- Layer 1: Linear(106 → 384) + LayerNorm + ReLU
- Layer 2: Linear(384 → 384) + LayerNorm + ReLU
- Output: Linear(384 → 90) with **zero-initialized weights and biases**
- Skip connection: `adjusted_codes = codes + output_residual`

The zero-initialized output guarantees the adapter starts as an identity mapping and can only improve clustering. Training uses depth-correlation loss plus a preservation term: `L = L_depth-corr + 20.0 × ||A_θ(f,d) − f||²`. Best checkpoint: `results/depth_adapter/V3_dd16_h384_l2/best.pt`.

**Clustering:** MiniBatchKMeans with k=80 clusters is fit on L2-normalized adjusted 90D codes from all 2,975 training images. Parameters: batch_size=10,000, max_iter=300, n_init=3, random_state=42. The 80 cluster centroids live in 90D feature space.

**Assignment & Mapping:** Each image's patches (32×64 = 2,048 patches) are assigned to the nearest cluster centroid. Cluster IDs (0–79) are upsampled via nearest-neighbor to 512×1024 pixel resolution. For evaluation and training, clusters are mapped to 19 Cityscapes trainIDs via majority vote against ground-truth validation labels. Road dominates (~31/80 clusters); rare classes may receive 0–1 clusters.

**Output:** Semantic pseudo-label as uint8 PNG (cluster IDs 0–79 + 255 for ignore).

### Stage 2: Depth-Guided Instance Generation (DepthPro + Sobel + CC)

**Input:** Semantic pseudo-label (trainIDs 0–18) + DepthPro monocular depth map (512×1024, normalized [0,1]).

**No vision features are used.** Instance boundaries come purely from depth discontinuities.

**Depth Preprocessing:** Gaussian smoothing with σ = 0.0 (no blur — DepthPro maps are already clean).

**Edge Detection:** Sobel gradient magnitude is computed on the depth map:
- `Gx = Sobel_x * D`, `Gy = Sobel_y * D`
- `||∇D|| = √(Gx² + Gy²)`
- Depth edges: pixels where `||∇D|| > τ = 0.20`

**Per-Class Connected Components:** For each thing class c ∈ {person, rider, car, truck, bus, train, motorcycle, bicycle}:
1. `M_c = {pixels where semantic == c}`
2. `M'_c = M_c \ depth_edges` (split mask at depth discontinuities)
3. Extract connected components: `{CC₁, CC₂, ...} = CC(M'_c)`
4. Area filter: keep only CCs with `|CC_i| ≥ A_min = 1000` pixels
5. Boundary reclamation: dilate each CC by 3 iterations, reclaim unassigned class-c pixels

**Output:** ~17 valid instances per image (down from ~292 raw connected components). Instance scores are normalized by area (`score = area / max_area`). Saved as uint16 PNG (`_instance.png`) and NPZ masks.

### Stage 3: SIMCF-ABC — Semantic-Instance Mutual Consistency Filtering

SIMCF-ABC is a three-step post-processing pipeline that cleans cross-modal inconsistencies between independently-generated semantic and instance labels. **No DINOv2 or DINOv3 vision features are used.** The inputs are: semantic cluster map, instance mask, cluster-to-class mapping, and DepthPro depth maps.

**Step A — Instance Validates Semantics (Majority Vote):**
For each instance region `I_k`, compute the majority trainID `t*` from semantic pixels under the mask. Reassign all inconsistent pixels to the most frequent cluster ID within `I_k` that maps to `t*`. For CUPS-derived labels where instances are connected components of individual cluster IDs, this step is structurally a no-op (0 pixels changed).

**Step B — Semantics Validate Instances (Instance Merging):** ★ Most Critical Step
Adjacent instances of the same semantic class that were over-split by depth noise are merged. The merge decision is based purely on **adjacency and semantic class consistency** — no external vision features are used.
1. Build adjacency graph via 3-pixel morphological dilation on each instance mask
2. For adjacent instance pairs with the same mapped trainID: mark for merge
3. Resolve transitive merges via union-find (disjoint set)
4. Renumber instances contiguously

Results: 44 → 22 instances per image; median instance size grows 5,502 → 14,965 px; stuff contamination drops from 50.7% to 28.0%; PQ_things improves by +1.33 (bus class +8.8 PQ).

**Step C — Depth Validates Semantics (3-Sigma Outlier Masking):**
Pixels with anomalous depth values for their semantic class are masked as ignore (255) to prevent the network from learning from likely-incorrect labels.
- Pass 1 (global): Compute per-class depth mean `μ_c` and standard deviation `σ_c` across all 2,975 training images
- Pass 2 (per-image): For each pixel p with mapped class c, mask if `|D(p) − μ_c| > 3 × σ_c`

Result: ~85M pixels masked (1.36% of all pixels), mainly boundary outliers. Contributes +0.30 PQ_stuff.

**Compositional Effect:** DCFA (+0.68 PQ) and SIMCF-ABC (+0.73 PQ) compose near-additively to +1.31 PQ because each corrects orthogonal error sources.

### Stage 4: Panoptic Merge

The refined semantic and instance maps are fused into a single panoptic label following Cityscapes encoding conventions.

**Encoding:** `panoptic_id = class_id × 1000 + instance_id`
- Stuff pixels: `instance_id = 0` (e.g., road = 0 × 1000 + 0 = 0)
- Thing pixels: `instance_id = 1, 2, 3, ...` per class (e.g., car #1 = 13 × 1000 + 1 = 13001)

**Algorithm:**
1. **Place Things First:** Sort instances by score descending. For each instance mask, take majority semantic class. Skip if not a thing class or < 10 pixels. Assign only to unassigned pixels (`valid_mask = mask & ~assigned`).
2. **Place Stuff:** For each stuff class, fill remaining unassigned pixels. Skip if area < 64 pixels. Instance ID is always 0.
3. **Fallback CC:** For uncovered thing pixels (no instance mask overlapped them), create new instances via connected components. These get low confidence (score = 0.1).

**Output per image:**
- `_panoptic.npy` — int32 encoded panoptic map
- `_panoptic.png` — uint16 visualization
- `segment_info` — JSON metadata per segment (id, category_id, isthing, area, score)

### Downstream Training

The panoptic pseudo-labels (PQ = 25.85) are used to supervise:
- **Stage 2:** Mask2Former or Cascade Mask R-CNN with frozen DINOv3 ViT-B/16 backbone (~28% PQ)
- **Stage 3:** Self-training rounds to refine the model (final: 35.83% PQ)

DINOv3 is used **only** as the training backbone, not in pseudo-label generation.

### Key Parameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| DCFA | code_dim | 90 (DINOv2 CAUSE) |
| DCFA | depth_dim | 16 (sinusoidal) |
| DCFA | hidden_dim | 384 |
| DCFA | num_layers | 2 |
| DCFA | λ_preserve | 20.0 |
| K-Means | k | 80 |
| K-Means | batch_size | 10,000 |
| Instance Gen | τ (edge) | 0.20 |
| Instance Gen | A_min | 1000 px |
| Instance Gen | dilation | 3 |
| SIMCF-B | dilate_px | 3 |
| SIMCF-C | σ_threshold | 3.0 |
| Panoptic | min_stuff_area | 64 px |
| Panoptic | label_divisor | 1000 |

### Ablation Results (Cityscapes Train, 2,975 images)

| Variant | PQ | PQ_stuff | PQ_things | ΔPQ |
|---------|-----|----------|-----------|------|
| Raw k=80 + DepthPro | 24.54 | 33.43 | 12.31 | — |
| + DCFA only | 25.22 | 33.99 | 13.16 | +0.68 |
| + SIMCF-ABC only | 25.27 | 33.73 | 13.64 | +0.73 |
| **Full (DCFA + DepthPro + SIMCF-ABC)** | **25.85** | **33.96** | **14.70** | **+1.31** |
