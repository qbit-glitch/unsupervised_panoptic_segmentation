---
type: knowledge
title: "Architecture: Depth Anything V3 (DA3) with DoRA Adapters (Stage 1)"
project: mbps-panoptic-segmentation
tags:
  - architecture
  - adapters
  - dora
  - da3
  - depth-anything-v3
  - stage-1
  - depth-pseudo-labels
  - instance-pseudo-labels
  - professor-presentation
updated: 2026-04-24
---

# Depth Anything V3 (DA3) Architectural Modifications for MBPS Stage 1
## Depth-Guided Instance Pseudo-Label Generation with DoRA Adapters

> **Context:** NeurIPS professor meeting (2026-04-24). Professor advised: DoRA/LoRA adapters belong ONLY in Stage 1 (pseudo-label generation with frozen models), NEVER in Stage 2/3 (full backpropagation).

---

## Executive Summary

This document details the architectural modifications required to integrate **Depth Anything V3 (DA3)** into Stage 1 of the MBPS pipeline for depth-guided instance pseudo-label generation. Following the professor's guidance, **DoRA adapters are applied exclusively in Stage 1** where the pretrained depth model remains frozen, enabling domain-specific adaptation without catastrophic forgetting or destabilizing downstream training.

---

## 1. Original DA3 Architecture (Before Adapters)

### 1.1 Model Overview

| Attribute | Specification |
|-----------|--------------|
| **Model Class** | `depth_anything_3.api.DepthAnything3` |
| **Default Checkpoint** | `depth-anything/DA3MONO-LARGE` |
| **Encoder** | ViT-based (improved DINOv2/DINOv3-Large, ~307M params) |
| **Decoder** | DPT-style head (Dense Prediction Transformer) |
| **Output** | Relative monocular depth maps (H × W, normalized) |
| **API Pattern** | Custom API: `model.inference_batch(img)` — **not** HuggingFace `AutoModel` |
| **Preprocessing** | Handled internally by DA3 (not via HF Processor) |

### 1.2 Key Difference from DA2-Large

| Aspect | DA2-Large | DA3 (DA3MONO-LARGE) |
|--------|-----------|---------------------|
| **Loading** | `transformers.AutoModelForDepthEstimation` | `depth_anything_3.api.DepthAnything3.from_pretrained()` |
| **Forward Pass** | `model(**inputs).predicted_depth` | `model.inference_batch(img_tensor)` |
| **Architecture** | Standard HF Transformers | Custom implementation (non-standard module hierarchy) |
| **Preprocessing** | HF `AutoImageProcessor` | Internal to DA3 API |
| **Depth Quality** | Good general edges | Sharper instance boundaries, better fine structures |

### Diagram (A): Original DA3 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DA3MONO-LARGE  (FROZEN)                             │
│  Custom API: depth_anything_3.api.DepthAnything3.from_pretrained()         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT: RGB Image Tensor  (B, 3, H, W)                                     │
│                      │                                                      │
│                      ▼                                                      │
│   ┌────────────────────────────────────────┐                                │
│   │     DA3 INTERNAL PREPROCESSING         │  <-- NOT HF Processor          │
│   │   (resize, normalize, patchify)        │                                │
│   └────────────────────────────────────────┘                                │
│                      │                                                      │
│                      ▼                                                      │
│   ╔═══════════════════════════════════════════════════════════════════════╗ │
│   ║                    VIT ENCODER  (DINOv2/DINOv3-Large)                ║ │
│   ║  ┌─────────┐  ┌─────────┐  ┌─────────┐        ┌─────────┐          ║ │
│   ║  │ Block 0 │→ │ Block 1 │→ │ Block 2 │→ ... → │ Block N │  N≈24    ║ │
│   ║  │  (attn) │  │  (attn) │  │  (attn) │        │  (attn) │          ║ │
│   ║  │  + MLP  │  │  + MLP  │  │  + MLP  │        │  + MLP  │          ║ │
│   ║  └─────────┘  └─────────┘  └─────────┘        └─────────┘          ║ │
│   ║                                                                     ║ │
│   ║  Module paths may vary from standard HF:                            ║ │
│   ║    • model.encoder.blocks[i]  (CAUSE-style)                         ║ │
│   ║    • model.vit.blocks[i]      (custom wrapper)                      ║ │
│   ║    • model.backbone.blocks[i] (backbone wrapper)                    ║ │
│   ║                                                                     ║ │
│   ║  ALL parameters frozen (no grad)                                   ║ │
│   ╚═══════════════════════════════════════════════════════════════════════╝ │
│                      │                                                      │
│                      ▼                                                      │
│   ┌────────────────────────────────────────┐                                │
│   │         DPT DECODER HEAD               │                                │
│   │  (feature fusion + upsample + refine)  │                                │
│   └────────────────────────────────────────┘                                │
│                      │                                                      │
│                      ▼                                                      │
│   OUTPUT: Relative Depth Map  (B, H, W)  [0,1] normalized                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. What Changes with DoRA Adapters

### 2.1 The Challenge: Non-Standard Architecture

DA3 does **not** use the standard HuggingFace `Dinov2Model` class hierarchy. Its internal module naming conventions differ from both:
- **CAUSE-TR style**: `encoder.blocks[i].attn.qkv`, `encoder.blocks[i].mlp.fc1`
- **HF Transformers style**: `encoder.layer[i].attention.attention.query`

Because of this, the adapter injection system uses a **fallback generic injection strategy**.

### 2.2 Injection Strategy: Tiered + Generic Fallback

The injection logic in `mbps_pytorch/models/adapters/depth_adapter.py` operates as follows:

```
Step 1: Try _find_encoder_blocks()
    ├── Check: model.encoder          (DAv3 / DepthPro HF)
    ├── Check: model.blocks
    ├── Check: model.vit.blocks
    ├── Check: model.backbone.blocks
    ├── Check: model.encoder.layer    (HF BERT-style)
    ├── Check: model.backbone.encoder.layer  (HF DAv2)
    │
    └── IF found → Use structured tiered injection
           Early blocks (0..5):  attn.qkv  only
           Late  blocks (6..N):  attn.qkv + attn.proj + mlp.fc1 + mlp.fc2
           
    └── IF NOT found → Fallback to _inject_generic_vit()
           Walk ALL named_modules()
           Match: "attn" in name.lower() AND isinstance(module, nn.Linear)
           Adapt EVERY matching linear layer
```

### 2.3 Why Generic Injection is Safe for DA3

Since DA3's encoder is fundamentally a ViT with attention blocks, the generic walker finds the same conceptual layers (Q, K, V projections, output projections) even if nested under unconventional names. The `_inject_generic_vit()` function:

1. Iterates `model.named_modules()`
2. Identifies modules where `"attn" in name.lower()` and `isinstance(module, nn.Linear)`
3. Wraps each with `DoRALinear` (or `LoRALinear` / `ConvDoRALinear`)
4. Replaces the original `nn.Linear` in its parent module

### 2.4 Tiered Strategy Applied to DA3

Even with generic injection, the **tiered philosophy** is preserved via `late_block_start=6`:

| Block Index | Adaptation Level | Rationale |
|-------------|------------------|-----------|
| **0 – 5** (Early) | Attention QKV only | Preserve low-level feature extraction (edges, textures); avoid overfitting to target domain |
| **6 – N** (Late) | Attention QKV + Proj + MLP fc1 + fc2 | Full adaptation for high-level semantic depth relationships; task-specific tuning |

> **Note**: In the generic fallback, the tiering is approximated by the order modules appear in `named_modules()`. For ViTs, this typically respects depth ordering.

### Diagram (B): Adapter-Injected DA3 (Generic ViT Strategy)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DA3 WITH DoRA ADAPTERS  (STUDENT)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT: RGB Image Tensor  (B, 3, H, W)                                     │
│                      │                                                      │
│                      ▼                                                      │
│   ┌────────────────────────────────────────┐                                │
│   │     DA3 INTERNAL PREPROCESSING         │  <-- STILL FROZEN              │
│   └────────────────────────────────────────┘                                │
│                      │                                                      │
│                      ▼                                                      │
│   ╔═══════════════════════════════════════════════════════════════════════╗ │
│   ║                    VIT ENCODER  WITH DoRA ADAPTERS                   ║ │
│   ║                                                                     ║ │
│   ║  ┌─────────────────────────────────────────────────────────────┐   ║ │
│   ║  │  EARLY BLOCKS  (0 .. late_block_start-1)                   │   ║ │
│   ║  │  ┌─────────┐  ┌─────────┐        ┌─────────┐              │   ║ │
│   ║  │  │ Block 0 │  │ Block 1 │  ...   │ Block 5 │              │   ║ │
│   ║  │  │ ┌─────┐ │  │ ┌─────┐ │        │ ┌─────┐ │              │   ║ │
│   ║  │  │ │ATTN │ │  │ │ATTN │ │        │ │ATTN │ │              │   ║ │
│   ║  │  │ │qkv  │ │  │ │qkv  │ │        │ │qkv  │ │              │   ║ │
│   ║  │  │ │███◄─┘ │  │ │███◄─┘ │        │ │███◄─┘ │  ◄── DoRA    │   ║ │
│   ║  │  │ └──┬──┘ │  │ └──┬──┘ │        │ └──┬──┘ │   (trainable)│   ║ │
│   ║  │  │  MLP    │  │  MLP    │        │  MLP    │   FROZEN     │   ║ │
│   ║  │  └─────────┘  └─────────┘        └─────────┘              │   ║ │
│   ║  └─────────────────────────────────────────────────────────────┘   ║ │
│   ║                              │                                      ║ │
│   ║                              ▼                                      ║ │
│   ║  ┌─────────────────────────────────────────────────────────────┐   ║ │
│   ║  │  LATE BLOCKS  (late_block_start .. N)                      │   ║ │
│   ║  │  ┌─────────┐  ┌─────────┐        ┌─────────┐              │   ║ │
│   ║  │  │ Block 6 │  │ Block 7 │  ...   │ Block N │              │   ║ │
│   ║  │  │ ┌─────┐ │  │ ┌─────┐ │        │ ┌─────┐ │              │   ║ │
│   ║  │  │ │ATTN │ │  │ │ATTN │ │        │ │ATTN │ │              │   ║ │
│   ║  │  │ │qkv █│ │  │ │qkv █│ │        │ │qkv █│ │  ◄── DoRA    │   ║ │
│   ║  │  │ │proj █│ │  │ │proj █│        │ │proj █│ │  ◄── DoRA    │   ║ │
│   ║  │  │ └──┬──┘ │  │ └──┬──┘ │        │ └──┬──┘ │              │   ║ │
│   ║  │  │ ┌──┴──┐ │  │ ┌──┴──┐ │        │ ┌──┴──┐ │              │   ║ │
│   ║  │  │ │MLP  │ │  │ │MLP  │ │        │ │MLP  │ │              │   ║ │
│   ║  │  │ │fc1 █│ │  │ │fc1 █│ │        │ │fc1 █│ │  ◄── DoRA    │   ║ │
│   ║  │  │ │fc2 █│ │  │ │fc2 █│ │        │ │fc2 █│ │  ◄── DoRA    │   ║ │
│   ║  │  │ └─────┘ │  │ └─────┘ │        │ └─────┘ │              │   ║ │
│   ║  │  └─────────┘  └─────────┘        └─────────┘              │   ║ │
│   ║  │  █ = trainable adapter params  (lora_A, lora_B, magnitude)│   ║ │
│   ║  └─────────────────────────────────────────────────────────────┘   ║ │
│   ╚═══════════════════════════════════════════════════════════════════════╝ │
│                      │                                                      │
│                      ▼                                                      │
│   ┌────────────────────────────────────────┐                                │
│   │         DPT DECODER HEAD               │  <-- FROZEN (no adapters)      │
│   │  (feature fusion + upsample + refine)  │                                │
│   └────────────────────────────────────────┘                                │
│                      │                                                      │
│                      ▼                                                      │
│   OUTPUT: Adapted Relative Depth Map  (B, H, W)                            │
│                                                                             │
│   LEGEND:  ███◄─ = DoRA-adapted linear layer                               │
│            ───   = Frozen pretrained weights                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.5 Parameter Count Estimates

Assuming DINOv2/DINOv3-Large encoder (~24 blocks, dim=1024):

| Component | DA2-Large (HF) | DA3 (Generic Fallback) |
|-----------|---------------|------------------------|
| **Encoder total params** | ~307M | ~307M (similar) |
| **DoRA layers adapted** | Structured: ~48–72 | Generic: all attn linears found |
| **Trainable params (r=4)** | ~1.2M | ~1.2–1.5M (depends on exact attn layout) |
| **% of total model** | ~0.4% | ~0.4–0.5% |

#### Breakdown per DoRALinear layer (r=4, dim=1024):

| Parameter | Shape | Count |
|-----------|-------|-------|
| `lora_A` | (4, 1024) | 4,096 |
| `lora_B` | (1024, 4) | 4,096 |
| `lora_magnitude` | (1024, 1) | 1,024 |
| **Per layer total** | | **9,216** |

With ~130–160 attention linear layers found across all blocks, total trainable ≈ **1.2M–1.5M parameters**.

---

## 3. Training Architecture (Self-Supervised)

### 3.1 Student-Teacher Setup

Since the professor restricts adapters to **Stage 1 only**, we train them in a fully **self-supervised** manner — no ground-truth depth annotations required.

| Component | Configuration |
|-----------|--------------|
| **Student** | DA3 + DoRA adapters on encoder (trainable) |
| **Teacher** | Same DA3, **no adapters**, fully frozen |
| **Data** | Unlabeled RGB images from target domain (Cityscapes/COCO) |
| **Optimizer** | AdamW, LR=1e-4, weight_decay=1e-4 |
| **Scheduler** | CosineAnnealing over epochs |

### 3.2 Loss Functions

Three complementary self-supervised objectives (same as DA2 adapter training):

| Loss | Formula / Description | Weight | Purpose |
|------|----------------------|--------|---------|
| **Self-Distillation** | `MSE(student_depth, teacher_depth.detach())` | 1.0 | Keep adapted outputs close to pretrained; prevent divergence |
| **Relative Depth Ranking** | Pairwise margin ranking on random pixel pairs | 0.1 | Preserve ordinal depth relationships |
| **Scale-Invariant Loss** | Eigen et al. log-space SI loss | 0.5 | Handle unknown absolute scale |

### 3.3 Forward Pass Difference

```python
# Teacher forward (no grad)
with torch.no_grad():
    teacher_out = model.inference_batch(img)  # DA3 custom API

# Student forward (adapters active)
student_out = model.inference_batch(img)      # Same API, adapters in encoder
```

> **Critical**: Both student and teacher share the same `model.inference_batch()` call. The only difference is that the student has DoRA layers injected and `requires_grad=True` on adapter parameters. The teacher path is computed first under `torch.no_grad()`, then the student path recomputes with gradients flowing through adapters.

### Diagram (C): Self-Supervised Training Setup

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            STAGE 1: SELF-SUPERVISED DoRA ADAPTER TRAINING                   │
│                     (DA3-specific architecture)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Unlabeled RGB Images  (from Cityscapes leftImg8bit/train)                │
│            │                                                                │
│            ├────────────────────────┐                                       │
│            │                        │                                       │
│            ▼                        ▼                                       │
│   ┌──────────────┐         ┌──────────────┐                                │
│   │   STUDENT    │         │   TEACHER    │                                │
│   │  DA3 + DoRA  │         │  DA3 (frozen)│                                │
│   │              │         │  (no adapters)│                               │
│   │  ┌────────┐  │         │  ┌────────┐  │                                │
│   │  │Encoder │  │         │  │Encoder │  │                                │
│   │  │+ DoRA  │  │         │  │(frozen)│  │                                │
│   │  └───┬────┘  │         │  └───┬────┘  │                                │
│   │      │       │         │      │       │                                │
│   │  ┌───▼───┐   │         │  ┌───▼───┐   │                                │
│   │  │Decoder│   │         │  │Decoder│   │                                │
│   │  │(frozen│   │         │  │(frozen│   │                                │
│   │  └───┬───┘   │         │  └───┬───┘   │                                │
│   └──────┼───────┘         └──────┼───────┘                                │
│          │                        │                                         │
│          ▼                        ▼                                         │
│    D_student                 D_teacher (detached)                           │
│    (B,H,W)                   (B,H,W)                                        │
│          │                        │                                         │
│          └────────┬───────────────┘                                         │
│                   ▼                                                         │
│   ┌──────────────────────────────────────────────┐                          │
│   │              LOSS COMPUTATION                │                          │
│   │                                              │                          │
│   │  ┌────────────────────────────────────────┐  │                          │
│   │  │  L_distill = MSE(D_s, D_t.detach())    │  │  w=1.0                 │
│   │  │      ↓ forces student ≈ teacher         │  │                        │
│   │  ├────────────────────────────────────────┤  │                          │
│   │  │  L_rank  = MarginRanking(pairs)        │  │  w=0.1                 │
│   │  │      ↓ preserves depth ordering         │  │                        │
│   │  ├────────────────────────────────────────┤  │                          │
│   │  │  L_si    = ScaleInvariant(D_s, D_t)    │  │  w=0.5                 │
│   │  │      ↓ handles unknown global scale     │  │                        │
│   │  └────────────────────────────────────────┘  │                          │
│   │                                              │                          │
│   │  L_total = w_d·L_distill + w_r·L_rank + w_si·L_si                      │
│   │                                              │                          │
│   │  Only gradients through STUDENT adapters ────┘                          │
│   │  (all base params frozen)                                               │
│   └──────────────────────────────────────────────┘                          │
│                   │                                                         │
│                   ▼                                                         │
│         ┌─────────────┐                                                     │
│         │  AdamW opt  │  ← updates only lora_A, lora_B, lora_magnitude    │
│         │  LR = 1e-4  │                                                     │
│         └─────────────┘                                                     │
│                                                                             │
│   CHECKPOINT SAVED: adapter weights only (best.pt, epoch_*.pt)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Inference Architecture

### 4.1 Loading Pipeline for Pseudo-Label Generation

At inference time (generating depth maps for instance boundary detection):

```python
# 1. Load base DA3 model
model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")

# 2. Inject DoRA adapters (same architecture as training)
inject_lora_into_depth_model(model, variant="dora", rank=4, alpha=4.0)

# 3. Freeze everything except adapters
freeze_non_adapter_params(model)

# 4. Load trained adapter checkpoint
ckpt = torch.load("best.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model"], strict=False)

# 5. Run inference
with torch.inference_mode():
    depth = model.inference_batch(img_tensor)  # Same custom API
```

### 4.2 Depth → Instance Pseudo-Label Pipeline

The adapted depth maps feed into the depth-guided connected components algorithm:

| Step | Operation | Parameters |
|------|-----------|------------|
| 1 | Load semantic pseudo-labels (from DINOv3) | trainIDs 0–18 |
| 2 | Load DA3 depth map | `.npy` file, normalized [0,1] |
| 3 | Gaussian blur on depth | `sigma=1.0` |
| 4 | Sobel gradient magnitude | `Gx, Gy` → `sqrt(Gx² + Gy²)` |
| 5 | Threshold depth edges | **`τ = 0.03`** (empirically optimal) |
| 6 | Per thing-class connected components | `min_area = 100` |
| 7 | Boundary pixel reclamation | `dilation_iters = 3` |
| 8 | Save as NPZ masks | Compatible with evaluator |

### Diagram (D): Complete Stage 1 Depth Pipeline → Instance Pseudo-Labels

```
┌─────────────────────────────────────────────────────────────────────────────┐
│       STAGE 1 INFERENCE: DEPTH-GUIDED INSTANCE PSEUDO-LABEL GENERATION      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  A. ADAPTED DEPTH GENERATION  (per image)                          │   │
│  │                                                                     │   │
│  │   RGB Image  (H, W, 3)                                              │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │   ┌─────────────────────────────────┐                               │   │
│  │   │  DA3 + DoRA adapters (loaded)   │                               │   │
│  │   │  • Encoder: DoRA active         │                               │   │
│  │   │  • Decoder: frozen              │                               │   │
│  │   │                                 │                               │   │
│  │   │  Forward: inference_batch()     │                               │   │
│  │   └─────────────────────────────────┘                               │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │   Depth Map D  (H, W)  ∈ [0, 1]  (adapted for target domain)       │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │   Save: depth_dav3/train/city/xxx_leftImg8bit.npy                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  B. DEPTH-GRADIENT EDGE DETECTION                                   │   │
│  │                                                                     │   │
│  │   Depth Map D                                                       │   │
│  │       │                                                             │   │
│  │       ├──► Gaussian Blur (σ=1.0) ──► D_smooth                       │   │
│  │       │                                                             │   │
│  │       ├──► Sobel Gx ──┐                                             │   │
│  │       │               ├──► Gradient Magnitude │∇D│                  │   │
│  │       └──► Sobel Gy ──┘                                             │   │
│  │                           │                                         │   │
│  │                           ▼                                         │   │
│  │                  Edge Mask E = │∇D│ > τ                            │   │
│  │                              τ = 0.03  ◄── OPTIMAL (DA3-specific)  │   │
│  │                                                                     │   │
│  │   ┌─────────────────────────────────────────┐                       │   │
│  │   │  WHY τ=0.03?                            │                       │   │
│  │   │  DA3 produces sharper depth edges than  │                       │   │
│  │   │  DA2. A low threshold captures fine     │                       │   │
│  │   │  instance boundaries without merging    │                       │   │
│  │   │  adjacent objects (cars, pedestrians).  │                       │   │
│  │   └─────────────────────────────────────────┘                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  C. INSTANCE GENERATION (per thing class)                           │   │
│  │                                                                     │   │
│  │   Inputs:  Semantic S (from DINOv3)  +  Edge Mask E                 │   │
│  │                                                                     │   │
│  │   For each thing class c ∈ {person, rider, car, truck, bus,         │   │
│  │                            train, motorcycle, bicycle}:             │   │
│  │                                                                     │   │
│  │       ┌─────────────────────────────────────────┐                   │   │
│  │       │  Class mask:  M_c = (S == c)            │                   │   │
│  │       │       │                                   │                   │   │
│  │       │       ├──► Remove edges: M'_c = M_c & ~E  │                   │   │
│  │       │       │                                   │                   │   │
│  │       │       └──► Connected Components on M'_c   │                   │   │
│  │       │              ┌────┐ ┌────┐ ┌────┐        │                   │   │
│  │       │              │CC-1│ │CC-2│ │CC-3│ ...     │                   │   │
│  │       │              └────┘ └────┘ └────┘        │                   │   │
│  │       │                   │                       │                   │   │
│  │       └──► Dilation-based boundary reclamation    │                   │   │
│  │              (3 iterations to recover edge pixels)│                   │   │
│  │                    │                              │                   │   │
│  │                    ▼                              │                   │   │
│  │              Filter: area >= 100 pixels           │                   │   │
│  │                    │                              │                   │   │
│  │                    ▼                              │                   │   │
│  │              Instance mask + class label + score  │                   │   │
│  │       └─────────────────────────────────────────┘                   │   │
│  │                                                                     │   │
│  │   Output: List[(mask, class_id, confidence_score)]                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  D. SAVE PSEUDO-LABELS                                              │   │
│  │                                                                     │   │
│  │   NPZ: masks  (N_inst, H, W)  bool                                  │   │
│  │        scores (N_inst,)       float32  [0,1]                        │   │
│  │        classes(N_inst,)       int32    trainIDs                     │   │
│  │                                                                     │   │
│  │   PNG: Instance ID map for visualization                            │   │
│  │                                                                     │   │
│  │   Path: pseudo_instance_depth/train/city/xxx.npz                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Design Rationale & Key Decisions

### 5.1 Why Generic Injection for DA3?

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **A. Hardcode DA3 paths** | Precise control | Brittle if DA3 updates internal structure | ❌ Rejected |
| **B. Require DA3 to use HF format** | Standard injection | DA3 has custom API; would require wrapping | ❌ Rejected |
| **C. Generic `named_modules()` walker** | Robust to internal changes; finds all attention | May adapt more layers than strictly needed; no explicit tiering | ✅ **Chosen** |

The generic walker is the pragmatic choice because DA3's internal architecture is opaque and may change across versions. It guarantees adapter injection succeeds.

### 5.2 Why Keep Decoder Frozen?

- **Stability**: The DPT decoder is already well-trained for depth upsampling. Adapting it risks artifacting.
- **Parameter efficiency**: Decoder layers are large (convolutional + fusion). Freezing them keeps trainable params < 0.5%.
- **Generalization**: Encoder adaptation transfers better across image resolutions; decoder adaptation overfits to training resolution.

### 5.3 Why τ = 0.03 for DA3?

From the March 2026 depth ablation study:

| Model | Optimal τ | PQ_things | Notes |
|-------|-----------|-----------|-------|
| SPIdepth | 0.15 | 17.30 | Noisy edges, needs higher threshold |
| DA2-Large | 0.03 | 20.20 | Good edges, low threshold works |
| **DA3** | **0.03** | **20.90** | Sharper edges, same τ, better PQ |

DA3 achieves **+0.70 PQ_things** over DA2-Large with the **same threshold**, indicating superior edge localization. The low τ is possible because DA3's depth predictions have less blur at object boundaries.

### 5.4 Comparison: DA3 vs DA2-Large Adapter Architecture

| Aspect | DA2-Large (HF) | DA3 (Custom API) |
|--------|----------------|------------------|
| **Loading** | `AutoModelForDepthEstimation` | `DepthAnything3.from_pretrained()` |
| **Forward** | `model(**inputs).predicted_depth` | `model.inference_batch(img)` |
| **Adapter injection** | Structured: `backbone.encoder.layer[i]` | Generic fallback: `_inject_generic_vit()` |
| **Tiering** | Explicit early/late split | Approximated by module walk order |
| **Trainable params** | ~1.2M (r=4) | ~1.2–1.5M (r=4) |
| **Training losses** | Distillation + Ranking + SI | Identical |
| **Optimal τ (instances)** | 0.03 | 0.03 |
| **PQ_things (Cityscapes)** | 20.20 | **20.90** (+3.5%) |
| **PQ (Cityscapes)** | ~26.5 | **27.37** |

---

## 6. Summary for the Professor

### What You Should Understand

1. **DA3 is the best depth model for MBPS Stage 1** based on empirical results: **PQ=27.37, PQ_things=20.90** on Cityscapes — outperforming DA2-Large and SPIdepth.

2. **Adapters are Stage 1 ONLY**, as advised. DoRA parameters (~1.2M) are trained self-supervised on unlabeled target-domain images, then frozen during Stage 2/3 panoptic training.

3. **DA3 requires generic adapter injection** because it uses a custom API (`depth_anything_3.api.DepthAnything3`) rather than standard HuggingFace transformers. The `_inject_generic_vit()` fallback safely finds all attention `nn.Linear` layers via `named_modules()`.

4. **The same self-supervised losses apply**: distillation from a frozen teacher, pairwise depth ranking, and scale-invariant consistency. No labels needed.

5. **Inference uses the exact same API**: `model.inference_batch(img)` works identically before and after adapter injection. The only difference is that adapter weights are loaded and active.

6. **The optimal depth-edge threshold τ=0.03 is shared with DA2**, but DA3 achieves higher PQ because its depth boundaries are sharper and more accurate — especially benefiting the **car class** (+10.3 PQ over SPIdepth).

### Bottom Line

The DA3 adapter architecture is a **drop-in replacement** for DA2-Large in the existing training script (`train_depth_adapter_lora.py`). The only code change is the model loader (`load_dav3_model()`) and the fallback injection path. Everything else — losses, training loop, inference API, and the downstream connected-components pipeline — remains unchanged. This makes DA3 integration **low-risk and high-reward** for the MBPS Stage 1 pipeline.
