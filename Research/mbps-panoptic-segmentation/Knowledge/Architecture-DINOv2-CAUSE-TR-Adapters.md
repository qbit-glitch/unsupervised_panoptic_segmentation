---
type: knowledge
title: "Architecture: DINOv2+CAUSE-TR with DoRA Adapters (Stage 1)"
project: mbps-panoptic-segmentation
tags:
  - architecture
  - adapters
  - dora
  - cause-tr
  - dinov2
  - stage-1
  - semantic-pseudo-labels
  - professor-presentation
updated: 2026-04-24
---

# Architectural Modifications for DINOv2+CAUSE-TR in MBPS Stage 1
## Semantic Pseudo-Label Generation with DoRA Adapters

> **Context:** NeurIPS professor meeting (2026-04-24). Professor advised: DoRA/LoRA adapters belong ONLY in Stage 1 (pseudo-label generation with frozen models), NEVER in Stage 2/3 (full backpropagation).

---

## 1. Original Architecture (Before Adapters)

The baseline Stage 1 pipeline uses a **frozen** DINOv2 ViT-Base/14 backbone paired with the CAUSE-TR transformer-decoder head. No parameters are updated; the model acts purely as a fixed feature extractor.

### Component Specifications

| Component | Specification | Parameters |
|-----------|--------------|------------|
| **DINOv2 ViT-B/14** | 12 blocks, 768-dim, 12 heads, patch 14 | ~86M |
| **CAUSE-TR Segment_TR** | TRDecoder (1-layer Transformer), 90D reduction | ~2M |
| **Cluster Module** | 2048-codebook VQ, 2048-dim projection | ~185K |
| **Codebook** | Precomputed modularity (frozen) | 2048 × 768 |

### Forward Flow (Frozen)

```
Image (3×H×W)
    │
    ▼
┌─────────────────────────────────────┐
│  DINOv2 ViT-B/14 (FROZEN)          │
│  Patch Embed → [CLS] + 23×23 patches│
│  → 12 Transformer Blocks            │
│  Output: 529 tokens × 768-dim       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  CAUSE-TR Segment_TR Head (FROZEN) │
│  VQ-Query → TRDecoder → 90D codes  │
│  Output: 529 tokens × 90-dim        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  K-Means Clustering (pre-fit)       │
│  54 clusters on 90D codes           │
│  Output: PNG pseudo-labels          │
└─────────────────────────────────────┘
```

---

## 2. What Changes with DoRA Adapters

We inject **weight-decomposed LoRA (DoRA)** into the frozen backbone. DoRA decomposes each adapted weight matrix into:
- A **frozen magnitude vector** `m` (initialized from `||W₀||`)
- A **trainable direction update** via low-rank matrices `B` (out×r) and `A` (r×in)

### DoRA Forward Equation

```
V'      = W₀ + ΔW = W₀ + (α/r) · B · A
V_norm  = ||V'||   (row-wise L2 norm, dim=1, detached)
W'      = m ⊙ (V' / V_norm)
y       = x · W'ᵀ + b          (frozen bias)
```

The magnitude `m` is learned independently, while the direction is adapted via the low-rank path. This outperforms vanilla LoRA on feature adaptation tasks.

---

### 2.1 Tiered Injection Strategy

We use a **tiered strategy** (`late_block_start=6`) because early ViT blocks learn low-level patterns (edges, textures) while late blocks learn high-level semantic concepts. We want minimal intervention early and full adaptation late.

```
┌─────────────────────────────────────────────────────────────┐
│  BLOCKS 0 ──► 5   (Early)                                   │
│  ├── attn.qkv  ← DoRA injected                              │
│  ├── attn.proj  (frozen)                                    │
│  ├── mlp.fc1    (frozen)                                    │
│  └── mlp.fc2    (frozen)                                    │
│                                                             │
│  BLOCKS 6 ──► 11  (Late)                                    │
│  ├── attn.qkv   ← DoRA injected                             │
│  ├── attn.proj  ← DoRA injected                             │
│  ├── mlp.fc1    ← DoRA injected                             │
│  └── mlp.fc2    ← DoRA injected                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 CAUSE-TR Head Adaptation (Optional)

When `--adapt_cause` is enabled, we also inject DoRA into the **TRDecoder** inside `segment.head`:

| Layer | Dimensions | DoRA Params (r=4) |
|-------|-----------|-------------------|
| `self_attn.out_proj` | 768 → 768 | 6,912 |
| `multihead_attn.out_proj` | 768 → 768 | 6,912 |
| `linear1` (FFN expand) | 768 → 2048 | 13,312 |
| `linear2` (FFN project) | 2048 → 768 | 12,032 |
| **Total per decoder** | | **39,168** |

The EMA head (`head_ema`) remains **frozen** by design — it serves as the teacher.

---

### 2.3 Parameter Count Breakdown (r=4, α=4.0)

| Component | Layers Adapted | Trainable Params |
|-----------|---------------|------------------|
| **DINOv2 Early (0–5)** | 6 × `attn.qkv` | 87,552 |
| **DINOv2 Late (6–11)** | 6 × (`qkv` + `proj` + `fc1` + `fc2`) | 336,384 |
| **DINOv2 Subtotal** | 30 linear layers | **423,936** |
| **CAUSE-TR Head** | `out_proj` ×2 + `linear1` + `linear2` | **39,168** |
| **Grand Total** | | **~463K** |

> **Note:** The user-facing estimate of "~600K" is a conservative upper bound. At strict r=4 with the tiered strategy, the actual count is **~424K for backbone-only** and **~463K with CAUSE-TR head adaptation** — both well under 1% of the 86M backbone.

---

## 3. Training Architecture (Self-Supervised)

We employ a **student–teacher distillation** framework. The student is the adapter-wrapped model; the teacher is the **frozen original** (no adapters).

### Why This Design?
- The teacher provides a **stable target** from the original pretrained representation.
- The student learns to **deviate minimally but meaningfully** via the low-rank adapters.
- The EMA-smoothed CAUSE head prevents representation collapse during clustering.

---

### Diagram (C): Self-Supervised Training Setup

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT IMAGE (batch)                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
   ┌─────────┐                                ┌─────────┐
   │  Aug 1  │                                │  Aug 2  │  (optional, for
   │ (weak)  │                                │ (strong)│   cross-view loss)
   └────┬────┘                                └────┬────┘
        ▼                                           ▼
┌───────────────┐                          ┌───────────────┐
│   STUDENT     │                          │    TEACHER    │
│  (trainable)  │                          │   (frozen)    │
│               │                          │               │
│ ┌───────────┐ │                          │ ┌───────────┐ │
│ │ DINOv2    │ │                          │ │ DINOv2    │ │
│ │ + DoRA    │ │                          │ │ (orig W₀) │ │
│ │ adapters  │ │                          │ │ NO adapters│ │
│ └─────┬─────┘ │                          │ └─────┬─────┘ │
│       ▼       │                          │       ▼       │
│  feat_student │                          │  feat_teacher  │
│    [529×768]  │                          │    [529×768]  │
│       │       │                          │       │       │
│ ┌─────▼─────┐ │                          │ ┌─────▼─────┐ │
│ │ CAUSE-TR  │ │                          │ │ CAUSE-TR  │ │
│ │ head      │ │                          │ │ head      │ │
│ │ (+adapters│ │                          │ │ (frozen)  │ │
│ │  optional)│ │                          │ │           │ │
│ └─────┬─────┘ │                          │ └─────┬─────┘ │
│       ▼       │                          │       ▼       │
│  seg_student  │                          │  seg_teacher   │
│    [529×90]   │                          │    [529×90]   │
└───────┬───────┘                          └───────┬───────┘
        │                                          │
        └──────────────────┬───────────────────────┘
                           ▼
              ┌────────────────────────┐
              │      LOSS FUNCTIONS    │
              ├────────────────────────┤
              │ 1. DINO Distillation   │
              │    KL(feat_student ||  │
              │       feat_teacher)    │
              │    τ_student=0.1       │
              │    τ_teacher=0.07      │
              ├────────────────────────┤
              │ 2. Depth Correlation   │
              │    Align 90D codes with│
              │    precomputed depth   │
              │    maps (grid sample)  │
              │    λ_depth = 0.05      │
              ├────────────────────────┤
              │ 3. Cross-View Consist. │
              │    L2-cosine between   │
              │    weak & strong aug   │
              ├────────────────────────┤
              │ 4. CAUSE Cluster Loss  │
              │    VQ commitment on    │
              │    EMA head output     │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  EMA Update Step       │
              │  head_ema ← 0.99·ema   │
              │           + 0.01·head  │
              └────────────────────────┘
```

### Training Details

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW (lr=1e-4, wd=1e-4) |
| Scheduler | CosineAnnealing |
| Gradient Clipping | max_norm=1.0 |
| EMA Momentum | λ = 0.99 |
| Resolution | 322 × 322 (23 × 23 patches) |
| Epochs | 10 |

---

## 4. Inference Architecture

At inference, we **load the adapter checkpoint**, re-inject the exact same adapter topology, load the learned adapter weights, and freeze everything else.

### Diagram (D): Complete Stage 1 Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: MODEL INITIALIZATION                                           │
│  ─────────────────────────────────────                                  │
│  • Load DINOv2 ViT-B/14 base weights                                    │
│  • Inject DoRA adapters into SAME layers as training                    │
│    (early: qkv-only, late: qkv+proj+fc1+fc2)                            │
│  • Load trained adapter checkpoint (lora_A, lora_B, lora_magnitude)     │
│  • Freeze ALL non-adapter parameters                                    │
│  • Set model.eval()                                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: FEATURE EXTRACTION (per image)                                 │
│  ─────────────────────────────────────                                  │
│                                                                         │
│   Input Image (1024×2048)                                               │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────────────┐                                                   │
│   │ Sliding Window  │  322×322 crops, stride 154px (patch-aligned     │
│   │ (patch-aligned) │  half overlap, 11 patches × 14px = 154px)         │
│   └────────┬────────┘                                                   │
│            ▼                                                            │
│   ┌─────────────────┐                                                   │
│   │ DINOv2+DoRA     │  Output: 529 × 768 (patch tokens only, drop CLS)  │
│   │ (adapted)       │                                                   │
│   └────────┬────────┘                                                   │
│            ▼                                                            │
│   ┌─────────────────┐                                                   │
│   │ CAUSE-TR Head   │  Output: 529 × 90                                 │
│   │ (adapted or froz│                                                   │
│   └────────┬────────┘                                                   │
│            ▼                                                            │
│   ┌─────────────────┐                                                   │
│   │ transform()     │  Reshape to 90 × 23 × 23 spatial                  │
│   └────────┬────────┘                                                   │
│            ▼                                                            │
│   ┌─────────────────┐                                                   │
│   │ Average Overlap │  Accumulate + divide by visit count                │
│   │ (full resolution)│  per patch grid position                          │
│   └────────┬────────┘                                                   │
│            ▼                                                            │
│   Flatten to (H_p×W_p) × 90 feature vectors                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3: K-MEANS CLUSTERING (dataset-level)                             │
│  ─────────────────────────────────────                                  │
│  • Collect all 90D vectors from entire train set                        │
│  • Subsample to 500K for MiniBatchKMeans                                │
│  • Fit K=54 clusters (2× Cityscapes classes, for stuff/thing separation)│
│  • Save cluster_centers_ to kmeans_model.npz                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 4: PSEUDO-LABEL GENERATION                                        │
│  ─────────────────────────────────────                                  │
│  • Predict cluster ID for each patch                                    │
│  • Reshape to H_p × W_p                                                 │
│  • Upsample to original H × W via nearest-neighbor                      │
│  • Save as PNG (uint8 cluster IDs)                                      │
│                                                                         │
│   Output: leftImg8bit/ → pseudo_semantic_adapted/train/                 │
│           munich_000001_000000_leftImg8bit.png → cluster IDs 0..53      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Summary: What the Professor Should Understand

| Question | Answer |
|---------|--------|
| **Why adapters ONLY in Stage 1?** | Stage 1 models are frozen pretrained extractors — adapters are the *only* way to adapt them without catastrophic forgetting or massive compute. Stages 2/3 use full fine-tuning (Mask2Former + CUPS), so adapters are unnecessary. |
| **Why DoRA instead of LoRA?** | DoRA decouples magnitude and direction learning. Empirically, this yields better feature adaptation for dense prediction tasks where feature scale matters. |
| **Why tiered injection?** | Early blocks process low-level structure; late blocks process semantics. We minimally disturb early layers and fully adapt late layers. |
| **Why student–teacher distillation?** | The frozen teacher anchors the student to the original pretrained manifold. The student only learns deviations that improve pseudo-label quality (guided by depth and clustering losses). |
| **How many new params?** | ~424K–463K trainable parameters (0.5% of backbone) versus 86M frozen. This is extremely parameter-efficient. |
| **What is the output?** | Cluster-ID PNGs (54 classes). These are later mapped to 19 Cityscapes semantic classes via the modularity codebook and hungarian matching in downstream stages. |

---

## Appendix: Full Architecture Comparison Diagrams

### (A) Original Frozen Pipeline (Before Adapters)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STAGE 1 (ORIGINAL)                          │
│                         ─────────────────                           │
│                                                                     │
│   Image ──► ┌─────────────┐ ──► ┌─────────────┐ ──► ┌───────────┐ │
│             │  DINOv2-B/14│     │ CAUSE-TR    │     │  K-Means  │ │
│             │  [FROZEN]   │     │  [FROZEN]   │     │  54 clust │ │
│             │             │     │             │     │           │ │
│             │ W₀ fixed    │     │ TRDecoder   │     │ Pre-fit   │ │
│             │ 12 blocks   │     │ VQ queries  │     │ on 90D    │ │
│             │ 768-dim     │     │ 90-dim out  │     │ codes     │ │
│             └─────────────┘     └─────────────┘     └─────┬─────┘ │
│                                                           │       │
│                                                     Pseudo-Labels │
│                                                           PNG     │
└─────────────────────────────────────────────────────────────────────┘
        Trainable params: 0
```

### (B) Adapter-Injected Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 1 (WITH DoRA ADAPTERS)                     │
│                    ────────────────────────────                     │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  DINOv2 ViT-B/14  (86M params total, ~425K trainable)       │   │
│   │                                                             │   │
│   │   Block 0 ──┐  attn.qkv  ← DoRA (A,B,m)                    │   │
│   │   Block 1 ──┤  attn.proj  ───── frozen                     │   │
│   │   Block 2 ──┤  mlp.fc1    ───── frozen                     │   │
│   │   Block 3 ──┤  mlp.fc2    ───── frozen                     │   │
│   │   Block 4 ──┤                                              │   │
│   │   Block 5 ──┘  ← Early: qkv-only steering                  │   │
│   │                                                             │   │
│   │   Block 6 ──┐  attn.qkv  ← DoRA                            │   │
│   │   Block 7 ──┤  attn.proj ← DoRA                            │   │
│   │   Block 8 ──┤  mlp.fc1   ← DoRA                            │   │
│   │   Block 9 ──┤  mlp.fc2   ← DoRA                            │   │
│   │   Block 10 ─┤                                              │   │
│   │   Block 11 ─┘  ← Late: full adaptation                     │   │
│   │                                                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  CAUSE-TR Segment_TR  (~2M params, ~48K trainable optional) │   │
│   │                                                             │   │
│   │   TRDecoder:                                                │   │
│   │     self_attn.out_proj      ← DoRA (optional)               │   │
│   │     multihead_attn.out_proj ← DoRA (optional)               │   │
│   │     linear1                 ← DoRA (optional)               │   │
│   │     linear2                 ← DoRA (optional)               │   │
│   │     f1, f2 convs            ─── frozen                      │   │
│   │                                                             │   │
│   │   projection_head           ─── frozen                      │   │
│   │   linear (classifier)       ─── frozen                      │   │
│   │                                                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  Cluster Module (frozen codebook, optional trainable probe) │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│                        K-Means ──► Pseudo-Labels                  │
└─────────────────────────────────────────────────────────────────────┘
```

### (C) Self-Supervised Training Setup (Detailed)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   for each batch:                                                   │
│                                                                     │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│   │   Teacher    │     │   Student    │     │   Student    │       │
│   │  (no grad)   │     │  (img)       │     │  (img_aug)   │       │
│   │              │     │              │     │              │       │
│   │ feat_t =     │     │ feat_s =     │     │ feat_aug =   │       │
│   │  backbone₀(  │     │  backbone'(  │     │  backbone'(  │       │
│   │    img )     │     │    img )     │     │    img_aug)  │       │
│   │              │     │              │     │              │       │
│   │ seg_t =      │     │ seg_s =      │     │              │       │
│   │  head₀(feat_t│     │  head'(feat_s│     │              │       │
│   │  )           │     │  )           │     │              │       │
│   └──────┬───────┘     └──────┬───────┘     └──────┬───────┘       │
│          │                    │                    │               │
│          │                    │                    │               │
│          │         ┌─────────▼────────────────────┘               │
│          │         │                                              │
│          │         │  L_distill = KL(softmax(feat_s/0.1) ||      │
│          │         │               softmax(feat_t/0.07))           │
│          │         │                                              │
│          │         │  L_crossview = 1 - cos(feat_s, feat_aug)     │
│          │         │                                              │
│          │         │  L_depth = -corr(seg_s, depth_map)           │
│          │         │                                              │
│          │         │  L_cluster = VQ_loss(head_ema(feat_t))       │
│          │         │                                              │
│          │         │  L_total = w₁·L_distill + w₂·L_crossview +   │
│          │         │            w₃·L_depth + w₄·L_cluster         │
│          │         │                                              │
│          │         └─────────┬────────────────────────────────────┘
│          │                   │                                      │
│          │                   ▼                                      │
│          │         ┌─────────────────┐                              │
│          │         │  AdamW step on  │                              │
│          │         │  adapter_params │                              │
│          │         └─────────────────┘                              │
│          │                                                          │
│          │         ┌─────────────────┐                              │
│          └────────►│  EMA Update:    │                              │
│                    │  head_ema ←     │                              │
│                    │  0.99·ema +     │                              │
│                    │  0.01·head      │                              │
│                    └─────────────────┘                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### (D) Complete Stage 1 Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FULL PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Cityscapes Train Split (2,975 images)                                  │
│        │                                                                │
│        ├──► ┌─────────────────┐                                         │
│        │    │ Precompute Depth│  (DepthPro / Depth Anything v3)        │
│        │    │  1×H×W maps     │                                         │
│        │    └─────────────────┘                                         │
│        │                                                                │
│        └──► ┌─────────────────┐                                         │
│             │   AdapterTrain  │  (322×322 crops + weak augment)        │
│             │     Dataset     │                                         │
│             └────────┬────────┘                                         │
│                      │                                                  │
│                      ▼                                                  │
│           ┌────────────────────┐                                        │
│           │ train_semantic_    │                                        │
│           │ adapter.py         │  ◄── 10 epochs, AdamW, lr=1e-4       │
│           │                    │                                        │
│           │ Student: DINOv2'   │  (DoRA adapters active)               │
│           │ Teacher: DINOv2₀   │  (frozen original)                    │
│           └────────┬───────────┘                                        │
│                    │                                                    │
│                    ▼                                                    │
│           ┌────────────────────┐                                        │
│           │  best.pt checkpoint│                                        │
│           │  ├─ backbone state │  (lora_A, lora_B, lora_magnitude)     │
│           │  ├─ segment state  │  (optional adapter weights)            │
│           │  └─ adapter_config │  (metadata for reconstruction)         │
│           └────────┬───────────┘                                        │
│                    │                                                    │
│                    ▼                                                    │
│           ┌────────────────────┐                                        │
│           │ generate_semantic_ │                                        │
│           │ pseudolabels_      │                                        │
│           │ adapted.py         │                                        │
│           │                    │                                        │
│           │ • Reconstruct      │                                        │
│           │   architecture     │                                        │
│           │ • Load adapters    │                                        │
│           │ • Sliding window   │                                        │
│           │   inference        │                                        │
│           │ • K-Means K=54     │                                        │
│           └────────┬───────────┘                                        │
│                    │                                                    │
│                    ▼                                                    │
│           ┌────────────────────┐                                        │
│           │ pseudo_semantic_   │                                        │
│           │ adapted/train/     │                                        │
│           │ *.png (cluster IDs)│  ───► Stage 2: Mask2Former training   │
│           └────────────────────┘                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Final Talking Points for the Professor

1. **Efficiency**: We add <0.5M trainable parameters to an 86M model — this is **0.5% overhead** but yields measurable improvements in pseudo-label mIoU.
2. **Correctness**: The EMA teacher ensures the clustering manifold does not collapse during adapter training. The depth correlation loss grounds semantic clusters in geometric structure.
3. **Separation of Concerns**: Adapters are strictly confined to Stage 1. Stage 2 (Mask2Former) and Stage 3 (CUPS refinement) perform full backpropagation through their own architectures, so no adapter remnants leak into the panoptic model.
4. **Reproducibility**: The `adapter_config` dict serialized in every checkpoint guarantees the exact same injection topology is reconstructed at inference, preventing silent weight-dropping bugs.

This architecture allows the project to leverage massive pretrained models in a parameter-efficient way while generating higher-quality pseudo-labels that propagate downstream to improve final panoptic quality.

---

## Related Notes
- [[Knowledge/Adapter-Strategy]] — Overall adapter placement policy
- [[Daily/2026-04-24]] — Meeting notes with NeurIPS professor
- `analysis_docs/why_conv_dora_not_plain_lora.md` — Empirical evidence against adapters in Stage 2/3
- `analysis_docs/lora_adapter_implementation_plan.md` — Original adapter design
- `mbps_pytorch/train_semantic_adapter.py` — Training script
- `mbps_pytorch/generate_semantic_pseudolabels_adapted.py` — Inference script
- Commit `C213` — Bug fix for silent LoRA drop
