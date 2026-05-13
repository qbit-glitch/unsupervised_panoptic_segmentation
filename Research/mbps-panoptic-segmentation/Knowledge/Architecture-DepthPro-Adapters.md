---
type: knowledge
title: "Architecture: Apple DepthPro with DoRA Adapters (Stage 1)"
project: mbps-panoptic-segmentation
tags:
  - architecture
  - adapters
  - dora
  - depthpro
  - stage-1
  - depth-pseudo-labels
  - instance-pseudo-labels
  - professor-presentation
updated: 2026-04-24
---

# Apple DepthPro Architectural Modifications for MBPS Stage 1
## Depth-Guided Instance Pseudo-Label Generation with DoRA Adapters

> **Context:** NeurIPS professor meeting (2026-04-24). Professor advised: DoRA/LoRA adapters belong ONLY in Stage 1 (pseudo-label generation with frozen models), NEVER in Stage 2/3 (full backpropagation).

---

## 1. Original DepthPro Architecture (Before Adapters)

Apple DepthPro is a **metric monocular depth estimation** model. Unlike single-encoder depth models (e.g., Depth Anything), DepthPro contains **three internal DINOv2-Large vision transformers** working in concert:

| Component | Internal Path | Purpose | Layers | Dim |
|-----------|---------------|---------|--------|-----|
| Patch Encoder | `depth_pro.encoder.patch_encoder.model` | Encodes local patch tokens | 24 | 1024 |
| Image Encoder | `depth_pro.encoder.image_encoder.model` | Encodes global image context | 24 | 1024 |
| FOV Encoder | `fov_model.fov_encoder.model` | Encodes focal length / field-of-view | 24 | 1024 |

Each encoder uses the **HuggingFace `Dinov2Layer`** architecture with **separate Q/K/V projections** (not fused `qkv` as in some implementations).

### (A) Original DepthPro Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         APPLE DEPTHPRO (FROZEN)                             │
│                         ~1B parameters, FULLY FROZEN                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT IMAGE (3×H×W)                                                       │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              DEPTH_PRO.ENCODER (Dual-Encoder Backbone)              │   │
│   │  ┌─────────────────────────┐    ┌─────────────────────────┐         │   │
│   │  │   PATCH ENCODER         │    │   IMAGE ENCODER         │         │   │
│   │  │   (DINOv2-Large, 24L)   │    │   (DINOv2-Large, 24L)   │         │   │
│   │  │   • Patch embeddings    │    │   • Patch embeddings    │         │   │
│   │  │   • 24× Dinov2Layer     │    │   • 24× Dinov2Layer     │         │   │
│   │  │     ├─ attn.q (1024→1024)│   │     ├─ attn.q (1024→1024)│        │   │
│   │  │     ├─ attn.k (1024→1024)│   │     ├─ attn.k (1024→1024)│        │   │
│   │  │     ├─ attn.v (1024→1024)│   │     ├─ attn.v (1024→1024)│        │   │
│   │  │     ├─ attn.out.dense    │   │     ├─ attn.out.dense    │        │   │
│   │  │     ├─ mlp.fc1 (1024→4096)│  │     ├─ mlp.fc1 (1024→4096)│       │   │
│   │  │     └─ mlp.fc2 (4096→1024)│  │     └─ mlp.fc2 (4096→1024)│       │   │
│   │  │   • CLS token + registers │   │   • CLS token + registers │       │   │
│   │  └─────────────────────────┘    └─────────────────────────┘         │   │
│   │                              │                                        │   │
│   │                         [FUSION / CROSS-ATTENTION]                    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              DPT-STYLE DECODER HEAD                                 │   │
│   │  • Reassemble tokens at multiple scales                             │   │
│   │  • Refine + upsample to full resolution                             │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              FOV_MODEL                                              │   │
│   │  ┌─────────────────────────┐                                        │   │
│   │  │   FOV ENCODER           │  ← Predicts focal length (intrinsics) │   │
│   │  │   (DINOv2-Large, 24L)   │     FROZEN — not adapted              │   │
│   │  │   • 24× Dinov2Layer     │                                        │   │
│   │  └─────────────────────────┘                                        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ▼                                                                    │
│   METRIC DEPTH MAP (1×H×W)                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key observations:**
- The **patch encoder** captures fine-grained local structure (important for object boundaries).
- The **image encoder** captures global scene context (important for depth scale consistency).
- The **FOV encoder** is purely geometric; it estimates camera intrinsics and does not contribute to depth quality for segmentation purposes.

---

## 2. What Changes with DoRA Adapters

### Adapter Injection Strategy

We apply **DoRA (Weight-Decomposed Low-Rank Adaptation, Liu et al., ICML 2024)** to **only two of the three encoders**:

| Encoder | Adapted? | Rationale |
|---------|----------|-----------|
| Patch Encoder | ✅ **YES** | Directly controls local depth discontinuities → critical for instance boundaries |
| Image Encoder | ✅ **YES** | Controls global depth scale and scene understanding |
| FOV Encoder | ❌ **NO** | Only estimates focal length; adapting it would not improve depth boundaries for segmentation |

### Tiered Layer Adaptation Strategy

DINOv2-Large has 24 transformer blocks. We use a **tiered strategy** controlled by `late_block_start=18`:

| Block Range | Count | Adapted Layers per Block | Rationale |
|-------------|-------|--------------------------|-----------|
| **Early** (0–17) | 18 blocks | `attention.query` + `attention.value` only | Early layers extract low-level features (edges, textures). Adapting only Q+V preserves stable low-level representations while allowing directional attention updates. |
| **Late** (18–23) | 6 blocks | `attention.query`, `key`, `value`, `output.dense`, `mlp.fc1`, `mlp.fc2` | Late layers extract high-level semantic features. Full adaptation allows the model to recompose semantic-aware depth representations. |

**Why tiered?**
- Early ViT layers are known to be **low-level feature extractors** (Gabor-like filters, edges).
- Late layers encode **semantic object parts and scene context**.
- Adapting MLPs in early layers causes **unstable training** and **representation collapse** in self-supervised settings. Restricting early blocks to Q+V avoids this.

### Exact Linear Layers Adapted (HF DINOv2 Structure)

Per `Dinov2Layer`, the following `nn.Linear` modules are conditionally wrapped:

```
layer.attention.attention.query      → "query"   (always, blocks 0-23)
layer.attention.attention.key        → "key"     (only late, blocks 18-23)
layer.attention.attention.value      → "value"   (always, blocks 0-23)
layer.attention.output.dense         → "proj"    (only late, blocks 18-23)
layer.mlp.fc1                        → "fc1"     (only late, blocks 18-23)
layer.mlp.fc2                        → "fc2"     (only late, blocks 18-23)
```

### (B) Adapter-Injected DepthPro Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEPTHPRO WITH DoRA ADAPTERS (STUDENT)                    │
│              ~1B parameters FROZEN  +  ~1.66M parameters TRAINABLE          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT IMAGE (3×H×W)                                                       │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              DEPTH_PRO.ENCODER (Dual-Encoder Backbone)              │   │
│   │                                                                     │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │              PATCH ENCODER  (DINOv2-Large, 24L)             │   │   │
│   │  │                                                             │   │   │
│   │  │   Blocks 0–17 (Early):  ┌─────────────────────────────┐    │   │   │
│   │  │                         │  Q ──► [DoRA r=4]            │    │   │   │
│   │  │                         │  K ──► [FROZEN]              │    │   │   │
│   │  │                         │  V ──► [DoRA r=4]            │    │   │   │
│   │  │                         │  Proj ──► [FROZEN]           │    │   │   │
│   │  │                         │  FC1 ──► [FROZEN]            │    │   │   │
│   │  │                         │  FC2 ──► [FROZEN]            │    │   │   │
│   │  │                         └─────────────────────────────┘    │   │   │
│   │  │   Blocks 18–23 (Late): ┌─────────────────────────────┐    │   │   │
│   │  │                        │  Q ──► [DoRA r=4]            │    │   │   │
│   │  │                        │  K ──► [DoRA r=4]            │    │   │   │
│   │  │                        │  V ──► [DoRA r=4]            │    │   │   │
│   │  │                        │  Proj ──► [DoRA r=4]         │    │   │   │
│   │  │                        │  FC1 ──► [DoRA r=4]          │    │   │   │
│   │  │                        │  FC2 ──► [DoRA r=4]          │    │   │   │
│   │  │                        └─────────────────────────────┘    │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                     │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │              IMAGE ENCODER  (DINOv2-Large, 24L)             │   │   │
│   │  │              [IDENTICAL TIERED DoRA STRUCTURE]              │   │   │
│   │  │              Blocks 0–17: Q+V only                          │   │   │
│   │  │              Blocks 18–23: All 6 layers                     │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              DPT-STYLE DECODER HEAD  [FROZEN]                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              FOV_MODEL                                              │   │
│   │  ┌─────────────────────────┐                                        │   │
│   │  │   FOV ENCODER           │  ← [COMPLETELY FROZEN, NO ADAPTERS]  │   │
│   │  │   (DINOv2-Large, 24L)   │                                        │   │
│   │  └─────────────────────────┘                                        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ▼                                                                    │
│   ADAPTED METRIC DEPTH MAP (1×H×W)                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
        LEGEND:  [DoRA r=4] = trainable low-rank adapter  |  [FROZEN] = frozen
```

### Parameter Count Table (Exact Calculation)

**DoRA adds 3 parameter tensors per adapted linear layer:**
- `lora_A`: `rank × in_features`
- `lora_B`: `out_features × rank`
- `lora_magnitude`: `out_features × 1`

At **rank=4, alpha=4.0**:

| Layer Type | In → Out | lora_A | lora_B | magnitude | **Total per Layer** |
|------------|----------|--------|--------|-----------|---------------------|
| Q / K / V | 1024 → 1024 | 4,096 | 4,096 | 1,024 | **9,216** |
| attention.output.dense | 1024 → 1024 | 4,096 | 4,096 | 1,024 | **9,216** |
| mlp.fc1 | 1024 → 4096 | 4,096 | 16,384 | 4,096 | **24,576** |
| mlp.fc2 | 4096 → 1024 | 16,384 | 4,096 | 1,024 | **21,504** |

**Per-Encoder Breakdown:**

| Block Range | # Blocks | Adapted Layers / Block | Params / Block | Subtotal |
|-------------|----------|------------------------|----------------|----------|
| Early (0–17) | 18 | Q + V (2 layers) | 18,432 | **331,776** |
| Late (18–23) | 6 | Q + K + V + Proj + FC1 + FC2 (6 layers) | 82,944 | **497,664** |
| **Per Encoder Total** | 24 | — | — | **829,440** |

**Grand Totals:**

| Component | Trainable Params | Status |
|-----------|-----------------|--------|
| Patch Encoder adapters | 829,440 | ✅ Trainable |
| Image Encoder adapters | 829,440 | ✅ Trainable |
| FOV Encoder | 0 | ❌ Frozen |
| All other DepthPro weights | ~1,000,000,000 | ❌ Frozen |
| **Total Added** | **1,658,880** (~1.66M) | — |
| **% of Full Model** | **~0.17%** | — |

> **Note:** This is at `late_block_start=18`. Using `late_block_start=12` would yield ~1.2M per encoder (~2.4M total).

---

## 3. Training Architecture (Self-Supervised)

The professor's instruction is critical: **LoRA/DoRA adapters are ONLY used in Stage 1 (pseudo-label generation) where models are frozen.** The training is **fully self-supervised** — no ground-truth depth or segmentation labels are used.

### (C) Self-Supervised Training Setup Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           STAGE 1: SELF-SUPERVISED DoRA ADAPTER TRAINING                    │
│              Target: Adapt depth quality for instance boundaries            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────┐              ┌─────────────────────────┐     │
│   │      STUDENT NETWORK    │              │     TEACHER NETWORK     │     │
│   │   DepthPro + DoRA adapters              │   DepthPro (ORIGINAL)   │     │
│   │   ~1.66M params trainable               │   ALL params frozen     │     │
│   │   All base weights frozen               │   No adapters           │     │
│   │   (requires_grad=False)                 │   (inference only)      │     │
│   └─────────────────────────┘              └─────────────────────────┘     │
│              │                                        │                     │
│              │    Shared Input: Raw RGB Image         │                     │
│              │    (Cityscapes unlabeled train set)    │                     │
│              ▼                                        ▼                     │
│         Student Depth                           Teacher Depth               │
│         (adapted)                               (frozen, higher quality)    │
│              │                                        │                     │
│              └──────────────┬─────────────────────────┘                     │
│                             ▼                                               │
│              ┌─────────────────────────────────────────────┐                │
│              │           COMBINED LOSS                     │                │
│              │                                             │                │
│              │  L_distillation = MSE(student, teacher)     │                │
│              │    → Keeps student close to pretrained      │                │
│              │       metric depth distribution               │                │
│              │                                             │                │
│              │  L_ranking = MarginRankingLoss(pairs)       │                │
│              │    → Enforces correct depth ordering:       │                │
│              │       if d_i > d_j, then pred_i > pred_j    │                │
│              │       (critical for boundary contrast)        │                │
│              │                                             │                │
│              │  L_scale_inv = Scale-Invariant Log Loss       │                │
│              │    → Robust to global scale ambiguities       │                │
│              │       (Eigen et al. formulation)              │                │
│              │                                             │                │
│              │  L_total = w₁·L_dist + w₂·L_rank + w₃·L_si  │                │
│              │                                             │                │
│              └─────────────────────────────────────────────┘                │
│                             │                                               │
│                             ▼                                               │
│              ┌─────────────────────────────────────────────┐                │
│              │   OPTIMIZER: AdamW (lr=1e-4, WD=1e-4)       │                │
│              │   Only adapter params enter optimizer!        │                │
│              │   Base weights untouched                      │                │
│              └─────────────────────────────────────────────┘                │
│                                                                             │
│   OUTPUT: Checkpoint with adapter weights (best.pt, epoch_NNN.pt)           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Loss Functions (Code-Verified)

| Loss | Implementation | Weight | Purpose |
|------|---------------|--------|---------|
| **Distillation** | `F.mse_loss(student, teacher.detach())` | 1.0 | Prevents catastrophic forgetting of metric depth |
| **Ranking** | `F.margin_ranking_loss(d_i, d_j, sign(d_i−d_j), margin=0.1)` | 0.1 | Enforces local depth ordering → sharper boundaries |
| **Scale-Invariant** | `mean(diff²) − λ·(sum(diff))²/n²` | 0.5 | Handles scale ambiguity in self-supervision |

### Why Self-Supervised?

- **No depth labels** exist for Cityscapes training images at the required quality.
- The **frozen teacher** acts as a pseudo-supervisor, providing high-quality target depth maps.
- The **ranking loss** specifically targets the property we care about for instance segmentation: **depth discontinuities at object boundaries**. Correct relative ordering between foreground and background pixels is more important than absolute metric accuracy for downstream connected-components splitting.

---

## 4. Inference Architecture

After training, adapters are loaded into a fresh DepthPro model, all base parameters are frozen, and depth maps are generated for the entire unlabeled training set.

### (D) Complete Stage 1 Depth Pipeline Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│        STAGE 1 INFERENCE: DEPTH-GUIDED INSTANCE PSEUDO-LABELS               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   STEP 1: LOAD ADAPTED DEPTHPRO                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  model = AutoModelForDepthEstimation.from_pretrained("apple/DepthPro-hf")│
│   │  inject_lora_into_depthpro(model, variant="dora", rank=4, ...)      │   │
│   │  freeze_non_adapter_params(model)    ← freeze everything except DoRA│   │
│   │  model.load_state_dict(ckpt["model"], strict=False)                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│   STEP 2: GENERATE ADAPTED DEPTH MAPS (per image)                           │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  for img in unlabeled_train_set:                                    │   │
│   │      inputs = processor(images=img, return_tensors="pt")            │   │
│   │      depth = model(**inputs).predicted_depth   ← uses adapted encoders│  │
│   │      depth_norm = (depth - min) / (max - min)   ← [0,1] normalize   │   │
│   │      save depth_norm as .npy file                                   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│   STEP 3: DEPTH-GUIDED INSTANCE DECOMPOSITION                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  for each semantic class c in {person, car, rider, truck, bus,     │   │
│   │                                train, motorcycle, bicycle}:         │   │
│   │                                                                     │   │
│   │      1. GAUSSIAN BLUR depth map (σ=1.0)  ← reduce noise           │   │
│   │                                                                     │   │
│   │      2. SOBEL EDGE DETECTION on depth:                              │   │
│   │         gx = sobel(depth, axis=1)                                   │   │
│   │         gy = sobel(depth, axis=0)                                   │   │
│   │         grad_mag = √(gx² + gy²)                                     │   │
│   │         depth_edges = grad_mag > τ   (τ=0.20 default)               │   │
│   │                                                                     │   │
│   │      3. MASK = semantic_mask(class=c) ∧ ¬depth_edges                │   │
│   │         ← Remove pixels that are depth discontinuities              │   │
│   │                                                                     │   │
│   │      4. CONNECTED COMPONENTS on MASK                                │   │
│   │         ← Each connected component = one instance hypothesis        │   │
│   │                                                                     │   │
│   │      5. DILATE + RECLAIM (3 iterations)                             │   │
│   │         ← Grow instance masks to reclaim edge pixels                │   │
│   │                                                                     │   │
│   │      6. FILTER by min_area (1000 px)                                │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│   STEP 4: OUTPUT PSEUDO-LABELS                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  • instance_map.png  → integer mask (instance IDs)                  │   │
│   │  • instances.npz     → masks[ N×H×W ], classes[N], scores[N]        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Design Rationale Summary

| Design Choice | Rationale |
|---------------|-----------|
| **Only Patch + Image encoders adapted** | These control depth quality. FOV encoder only predicts focal length; adapting it wastes parameters. |
| **FOV encoder frozen** | Camera intrinsics are dataset-dependent but do not improve boundary detection. |
| **Tiered strategy (Q+V early, all late)** | Early layers learn low-level features that should remain stable. Late layers learn semantics where adaptation is most impactful. |
| **DoRA instead of LoRA** | DoRA decomposes weight into magnitude and direction, enabling more expressive updates with the same rank. Better for fine-grained depth boundary refinement. |
| **Self-supervised losses** | No labels needed. Distillation preserves metric quality; ranking loss directly optimizes for boundary contrast. |
| **Rank=4** | Extremely parameter-efficient (~0.17% of model). Higher ranks (e.g., 16) showed diminishing returns in ablations. |
| **Sobel + Connected Components** | Classic, interpretable, and fast. Depth discontinuities naturally separate instances of the same semantic class. |

---

## 6. What the Professor Should Understand

> **"In Stage 1, we treat Apple DepthPro as a frozen foundation model and train tiny DoRA adapters (~1.66M parameters, 0.17% of the model) on its patch and image encoders. These adapters are trained self-supervised using the frozen model as a teacher, with a ranking loss that specifically sharpens depth boundaries. At inference, the adapted depth maps are fed into a classical Sobel + Connected Components pipeline to decompose 'thing' classes into instance masks. The FOV encoder is left completely untouched. All adapter training is confined to Stage 1; no adapters exist in Stages 2 or 3, per your instruction."**

---

## 7. Key Files in the Codebase

| File | Role |
|------|------|
| `mbps_pytorch/models/adapters/depthpro_adapter.py` | Injects DoRA into HF DINOv2 encoders inside DepthPro |
| `mbps_pytorch/models/adapters/lora_layers.py` | `DoRALinear` implementation (weight-decomposed LoRA) |
| `mbps_pytorch/train_depth_adapter_lora.py` | Self-supervised training loop (distillation + ranking + SI loss) |
| `mbps_pytorch/generate_instance_pseudolabels_adapted.py` | Loads adapters + generates depth-guided instance pseudo-labels |
| `mbps_pytorch/instance_methods/sobel_cc.py` | Sobel edge detection + connected components decomposition |

---

This architecture is **parameter-efficient, self-supervised, and strictly confined to Stage 1**, fully respecting the constraint that adapters only modify frozen pretrained models during pseudo-label generation.
