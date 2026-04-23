# Semantic Adapter Training Architecture (CAUSE-TR + DINOv2)

## Overview

We train **DoRA adapters** on the frozen DINOv2 backbone and CAUSE-TR head using **only self-supervised losses** — no ground truth labels. The adapters learn to shift the feature distribution from generic ImageNet/LVD to Cityscapes driving scenes.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         INPUT: Cityscapes RGB Image (322×644)                       │
│                                                                                     │
│    ┌─────────────────────────────┐    ┌─────────────────────────────┐               │
│    │      Teacher Branch         │    │      Student Branch         │               │
│    │      (FROZEN, no grad)      │    │      (ADAPTED, trainable)   │               │
│    └─────────────────────────────┘    └─────────────────────────────┘               │
│              │                                  │                                    │
│              ▼                                  ▼                                    │
│    ┌──────────────────┐            ┌──────────────────┐                              │
│    │  DINOv2 ViT-B/14 │            │  DINOv2 ViT-B/14 │                              │
│    │  (FROZEN weights)│            │  + DoRA Adapters │                              │
│    │                  │            │  (trainable)     │                              │
│    │  Blocks 0-23:    │            │  Blocks 0-5:     │                              │
│    │  all frozen      │            │  qkv only        │                              │
│    │                  │            │  Blocks 6-23:    │                              │
│    │  Output:         │            │  qkv+proj+fc1+fc2│                              │
│    │  (N, 768)        │            │                  │                              │
│    └──────────────────┘            │  Output:         │                              │
│              │                     │  (N, 768)        │                              │
│              │                     └──────────────────┘                              │
│              │                            │                                         │
│              ▼                            ▼                                         │
│    ┌──────────────────┐            ┌──────────────────┐                              │
│    │ CAUSE-TR Head    │            │ CAUSE-TR Head    │                              │
│    │ (FROZEN)         │            │ + DoRA Adapters  │                              │
│    │                  │            │ (trainable)      │                              │
│    │  TRDecoder:      │            │                  │                              │
│    │  self_attn,      │            │  TRDecoder:      │                              │
│    │  multihead_attn, │            │  self_attn.out,  │                              │
│    │  FFN frozen      │            │  multihead_attn, │                              │
│    │                  │            │  FFN adapted     │                              │
│    │  Output:         │            │                  │                              │
│    │  (90, 23, 23)    │            │  Output:         │                              │
│    │  codes           │            │  (90, 23, 23)    │                              │
│    └──────────────────┘            │  codes           │                              │
│              │                     └──────────────────┘                              │
│              │                            │                                         │
│              ▼                            ▼                                         │
│    ┌──────────────────┐            ┌──────────────────┐                              │
│    │ Teacher Features │            │ Student Features │                              │
│    │ feat_teacher     │            │ feat_student     │                              │
│    │ seg_feat_teacher │            │ seg_feat_student │                              │
│    └──────────────────┘            └──────────────────┘                              │
│              │                                  │                                    │
│              │              LOSSES               │                                    │
│              │         (no GT labels)            │                                    │
│              │                                  │                                    │
│              └──────────────────────────────────┘                                    │
│                            │                                                        │
│                            ▼                                                        │
│    ┌─────────────────────────────────────────────────────────┐                      │
│    │              SELF-SUPERVISED LOSS COMPUTATION           │                      │
│    └─────────────────────────────────────────────────────────┘                      │
│                            │                                                        │
│         ┌──────────────────┼──────────────────┐                                     │
│         │                  │                  │                                     │
│         ▼                  ▼                  ▼                                     │
│    ┌─────────┐      ┌─────────────┐    ┌─────────────┐                              │
│    │ Loss 1  │      │   Loss 2    │    │   Loss 3    │                              │
│    │DINO     │      │ Cross-View  │    │ Depth-Feature│                             │
│    │Self-Dist│      │ Consistency │    │ Correlation │                              │
│    └─────────┘      └─────────────┘    └─────────────┘                              │
│         │                  │                  │                                     │
│         └──────────────────┼──────────────────┘                                     │
│                            │                                                        │
│                            ▼                                                        │
│                   ┌────────────────┐                                                │
│                   │ Total Loss     │                                                │
│                   │ Backpropagate  │                                                │
│                   │ → Update ONLY  │                                                │
│                   │   adapters     │                                                │
│                   └────────────────┘                                                │
│                            │                                                        │
│                            ▼                                                        │
│                   ┌────────────────┐                                                │
│                   │ EMA Update     │                                                │
│                   │ student.head   │                                                │
│                   │ → head_ema     │                                                │
│                   └────────────────┘                                                │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Breakdown

### 1. DINOv2 Backbone with DoRA Adapters

```
DINOv2 ViT-B/14 (12 blocks, 768-dim per CAUSE repo)
│
├── Blocks 0-5 (Early Layers): qkv ONLY
│   └── Minimal adaptation for low-level features (edges, textures)
│
└── Blocks 6-11 (Late Layers): qkv + proj + fc1 + fc2
    └── Full adaptation for high-level semantics (cars, roads, pedestrians)

Total trainable params: ~600K (0.8% of 86M backbone)
```

**DoRA Formula per layer:**
```
W' = m ⊙ (W₀ + BA) / ||W₀ + BA||

Where:
  W₀ = frozen pretrained weight (768×768 or 768×3072)
  B, A = low-rank matrices (rank=4)
  m = learned magnitude vector
  ⊙ = element-wise multiplication
  Only B, A, m are trainable
```

---

### 2. CAUSE-TR Segment_TR Head with Adapters

```
Segment_TR Head
│
├── TRDecoder (Transformer Decoder)
│   ├── self_attn.out_proj     ← DoRA adapted
│   ├── multihead_attn.out_proj ← DoRA adapted
│   ├── linear1 (FFN expand)   ← DoRA adapted
│   └── linear2 (FFN project)  ← DoRA adapted
│
├── Output Projection
│   └── Transforms to 90-dim semantic codes
│
└── EMA Head (head_ema)
    └── Exponential moving average of student head
        → Used for stable cluster assignments
```

**Total trainable params: ~200K**

---

### 3. Loss Functions (No GT Needed!)

#### Loss 1: DINO Self-Distillation (Primary)

```
┌────────────────────────────────────────────────────────────┐
│  Teacher: frozen DINOv2 → feat_teacher                     │
│  Student: adapted DINOv2 → feat_student                    │
│                                                            │
│  L_distill = -Σ p_teacher * log(p_student)                 │
│                                                            │
│  Where:                                                    │
│    p_teacher = softmax(feat_teacher / τ_teacher)           │
│    p_student = softmax(feat_student / τ_student)           │
│                                                            │
│  τ_teacher = 0.07 (sharper)                                │
│  τ_student = 0.1  (softer)                                 │
│                                                            │
│  Effect: Student stays close to pretrained manifold        │
│          while adapting to Cityscapes domain               │
└────────────────────────────────────────────────────────────┘
```

#### Loss 2: Cross-View Consistency

```
┌────────────────────────────────────────────────────────────┐
│  Image I → Augmentation T₁ → img                           │
│  Image I → Augmentation T₂ → img_aug                       │
│                                                            │
│  feat_student = backbone(img)                              │
│  feat_aug     = backbone(img_aug)                          │
│                                                            │
│  L_cv = 1 - cosine_similarity(feat_student, feat_aug)      │
│                                                            │
│  Augmentations: color jitter, grayscale, random crops      │
│                                                            │
│  Effect: Features invariant to photometric changes         │
└────────────────────────────────────────────────────────────┘
```

#### Loss 3: Depth-Feature Correlation (DepthG)

```
┌────────────────────────────────────────────────────────────┐
│  Input:                                                     │
│    code_student = Segment_TR(feat_student) → (90, 23, 23)  │
│    depth_map    = DepthPro depth → downsampled to (23, 23) │
│                                                            │
│  For sampled patch pairs (i, j):                           │
│    cd_ij = cosine_similarity(code_i, code_j)               │
│    dd_ij = cosine_similarity(depth_i, depth_j)             │
│                                                            │
│  L_depth = -cd_ij * dd_ij                                  │
│                                                            │
│  Effect: Similar depth → similar features                  │
│          (geometric prior from monocular depth)            │
└────────────────────────────────────────────────────────────┘
```

#### Loss 4: CAUSE Cluster Loss (Optional)

```
┌────────────────────────────────────────────────────────────┐
│  Input: seg_feat_ema = head_ema(feat_teacher)              │
│                                                            │
│  cluster.forward_centroid(seg_feat_ema)                    │
│  → Computes contrastive cluster assignments                │
│  → Encourages compact clusters in feature space            │
│                                                            │
│  Effect: Maintains clusterability for k-means              │
└────────────────────────────────────────────────────────────┘
```

---

## Training Loop (Step-by-Step)

```
FOR each epoch:
  FOR each batch of Cityscapes images:
    
    1. FORWARD (Teacher, no grad):
       feat_teacher = frozen_dinov2(img)
       seg_teacher  = frozen_cause_head(feat_teacher)
    
    2. FORWARD (Student, with grad):
       feat_student = adapted_dinov2(img)      ← adapters active
       seg_student  = adapted_cause_head(feat_student)  ← adapters active
    
    3. COMPUTE LOSSES:
       L1 = distillation(feat_student, feat_teacher)
       L2 = cross_view(feat_student, feat_aug)     [if aug enabled]
       L3 = depth_correlation(seg_student, depth)  [if depth available]
       L4 = cluster_loss(seg_ema)                  [optional]
       
       L_total = w1*L1 + w2*L2 + w3*L3 + w4*L4
    
    4. BACKWARD:
       L_total.backward()
       
       → Gradients flow ONLY through adapter parameters
       → Pretrained weights (W₀) receive ZERO gradients
    
    5. UPDATE:
       optimizer.step()  ← updates B, A, m matrices
    
    6. EMA UPDATE:
       head_ema = 0.99 * head_ema + 0.01 * head_student

  END batch
END epoch
```

---

## What is Frozen vs. Trainable

| Component | Frozen? | Trainable? | Params |
|-----------|---------|------------|--------|
| DINOv2 W₀ (pretrained) | ✅ Yes | ❌ No | 86M |
| DoRA B, A matrices | ❌ No | ✅ Yes | ~400K |
| DoRA magnitude m | ❌ No | ✅ Yes | ~100K |
| CAUSE-TR head W₀ | ✅ Yes | ❌ No | ~2M |
| CAUSE-TR DoRA adapters | ❌ No | ✅ Yes | ~200K |
| EMA head | ✅ Yes | ❌ No | ~2M |
| **Total trainable** | | | **~700K** |

---

## Key Insight

```
┌─────────────────────────────────────────────────────────────────┐
│  WITHOUT ADAPTERS                                               │
│  ├── DINOv2 trained on LVD-142M (generic natural images)        │
│  └── Features: good for ImageNet, suboptimal for driving scenes │
│                                                                 │
│  WITH ADAPTERS                                                  │
│  ├── DINOv2 pretrained weights stay frozen (generic knowledge)  │
│  └── DoRA adapters learn Cityscapes-specific feature shifts     │
│      → Better clustering for k-means pseudo-labels              │
│      → Higher PQ downstream                                     │
└─────────────────────────────────────────────────────────────────┘
```

The adapters are like **tinted glasses** — the pretrained model already sees well, but the adapters adjust the "color" of features to match the Cityscapes domain. All without ever seeing a human annotation.
