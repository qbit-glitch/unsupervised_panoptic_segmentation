---
type: knowledge
title: "Architecture: Depth Anything V2 Large (DA2-Large) with DoRA Adapters (Stage 1)"
project: mbps-panoptic-segmentation
tags:
  - architecture
  - adapters
  - dora
  - da2
  - depth-anything-v2
  - stage-1
  - depth-pseudo-labels
  - instance-pseudo-labels
  - professor-presentation
updated: 2026-04-24
---

# Architectural Modifications for Depth Anything V2 Large in MBPS Stage 1
## Depth-Guided Instance Pseudo-Label Generation with DoRA Adapters

> **Context:** NeurIPS professor meeting (2026-04-24). Professor advised: DoRA/LoRA adapters belong ONLY in Stage 1 (pseudo-label generation with frozen models), NEVER in Stage 2/3 (full backpropagation).

---

## 1. Overview: Why Adapt DA2-Large in Stage 1?

In MBPS Stage 1, all pretrained models are **frozen** and used only for pseudo-label generation. The professor's directive is clear: **LoRA/DoRA adapters belong exclusively in Stage 1**, where they fine-tune frozen foundation models to produce higher-quality pseudo-labels without catastrophic forgetting or destabilizing downstream training.

DA2-Large (Depth Anything V2 Large) is a ~300M parameter model consisting of a **DINOv2-Large ViT encoder** (24 transformer blocks, 1024-dim, patch size 14) and a **DPT decoder head**. We inject tiered DoRA adapters into the encoder while keeping the DPT decoder frozen. The adapted student model is then self-supervised against a frozen teacher copy, producing depth maps that are post-processed into instance pseudo-labels via gradient-based connected components.

---

## 2. (A) Original DA2-Large Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Depth Anything V2 Large (Original)                       │
│                         ~300M parameters, ALL frozen                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input RGB Image (H × W × 3)                                               │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────┐                                   │
│   │  HF Image Processor                 │  ◄── AutoImageProcessor           │
│   │  (resize, normalize, patchify)      │      from_pretrained()            │
│   └─────────────────────────────────────┘                                   │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │           DINOv2-Large Encoder (backbone.encoder.layer)             │   │
│   │              24 Transformer Blocks × 1024-dim × 16 heads            │   │
│   │                                                                     │   │
│   │   Block 0 ──► Block 1 ──► ... ──► Block 17 ──► Block 23           │   │
│   │   [early]      [early]           [early]        [late]              │   │
│   │                                                                     │   │
│   │   Each Block:                                                       │   │
│   │   ┌─────────────────────────────────────────────────────────┐       │   │
│   │   │  Self-Attention (HF-style, separate projections)        │       │   │
│   │   │   ├─ attention.attention.query  (Linear 1024→1024)      │       │   │
│   │   │   ├─ attention.attention.key    (Linear 1024→1024)      │       │   │
│   │   │   ├─ attention.attention.value  (Linear 1024→1024)      │       │   │
│   │   │   └─ attention.output.dense     (Linear 1024→1024)      │       │   │
│   │   │                                                         │       │   │
│   │   │  MLP                                                      │       │   │
│   │   │   ├─ mlp.fc1  (Linear 1024→4096)  ┐                     │       │   │
│   │   │   └─ mlp.fc2  (Linear 4096→1024)  ◄── GELU, Dropout     │       │   │
│   │   └─────────────────────────────────────────────────────────┘       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ▼  Multi-scale feature maps [1/4, 1/8, 1/16, 1/32]                  │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │           DPT Decoder Head (Dense Prediction Transformer)           │   │
│   │                                                                     │   │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  │   │
│   │   │ Reassemble  │  │ Reassemble  │  │ Reassemble  │  │  Fusion  │  │   │
│   │   │  (1/4 scale)│  │  (1/8 scale)│  │ (1/16 scale)│  │  + Refine│  │   │
│   │   └─────────────┘  └─────────────┘  └─────────────┘  └──────────┘  │   │
│   │                              │                                      │   │
│   │                              ▼                                      │   │
│   │                    Final 1×1 Conv → Predicted Depth                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ▼                                                                    │
│   Output: Relative Depth Map (H/14 × W/14, upsampled to H × W)              │
│           ❌ No absolute metric scale                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key architectural facts**:
- **HF path**: `model.backbone.encoder.layer[i]` contains the 24 blocks
- **Attention is unfused**: `attention.attention.query`, `.key`, `.value` are separate `nn.Linear` layers (not a monolithic `qkv` like CAUSE-TR's DINOv2)
- **Output**: Relative depth only. This means scale-invariant losses are critical during adapter training.

---

## 3. (B) Adapter-Injected DA2-Large (Stage 1 Configuration)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              DA2-Large with Tiered DoRA Adapters (Stage 1)                  │
│                                                                             │
│   SAME Input Processor → SAME DPT Decoder (FROZEN)                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │           DINOv2-Large Encoder WITH DoRA Adapters                   │   │
│   │                                                                     │   │
│   │   ┌─────────────────────────────────────────────────────────────┐   │   │
│   │   │  EARLY BLOCKS: 0 ──► 17  (18 blocks)                        │   │   │
│   │   │  ─────────────────────────────────────────                  │   │   │
│   │   │  Adapted:  • attention.attention.query  [DoRA]              │   │   │
│   │   │            • attention.attention.value   [DoRA]             │   │   │
│   │   │                                                               │   │   │
│   │   │  Frozen:   • attention.attention.key                       │   │   │
│   │   │            • attention.output.dense                        │   │   │
│   │   │            • mlp.fc1, mlp.fc2                              │   │   │
│   │   │            • LayerNorms, biases                            │   │   │
│   │   └─────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                      │   │
│   │                              ▼                                      │   │
│   │   ┌─────────────────────────────────────────────────────────────┐   │   │
│   │   │  LATE BLOCKS: 18 ──► 23  (6 blocks)                         │   │   │
│   │   │  ─────────────────────────────────────────                  │   │   │
│   │   │  Adapted:  • attention.attention.query  [DoRA]              │   │   │
│   │   │            • attention.attention.key     [DoRA]             │   │   │
│   │   │            • attention.attention.value   [DoRA]             │   │   │
│   │   │            • attention.output.dense      [DoRA]  ◄─ "proj"  │   │   │
│   │   │            • mlp.fc1                     [DoRA]             │   │   │
│   │   │            • mlp.fc2                     [DoRA]             │   │   │
│   │   │                                                               │   │   │
│   │   │  Frozen:   • LayerNorms, biases, ALL other weights           │   │   │
│   │   └─────────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  DPT Decoder Head                                                   │   │
│   │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │   │
│   │  ❄️  COMPLETELY FROZEN  (adapt_decoder = False)                     │   │
│   │      No adapters injected in reassembly, fusion, or refine layers   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│                         Predicted Relative Depth                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tiered Rationale

| Block Range | Strategy | Motivation |
|---|---|---|
| **0 – 17** (early) | Adapt **Q + V only** | Early layers learn low-level features (edges, textures). Restricting adaptation to query and value preserves the pretrained key-distribution and prevents overfitting to domain-specific textures. |
| **18 – 23** (late) | Adapt **Q + K + V + proj + fc1 + fc2** | Late layers encode high-level semantics and global context. Full adaptation here allows the model to reshape semantic representations for better depth boundary alignment, which is critical for downstream instance segmentation. |

The `late_block_start=18` heuristic is hard-coded in `mbps_pytorch/models/adapters/depth_adapter.py`. The `_find_encoder_blocks()` function discovers the encoder via `model.backbone.encoder.layer` for the HF DA2 checkpoint.

### Parameter Count Breakdown (Exact, from Code)

At **rank = 4**, **alpha = 4.0**, HF-style DINOv2-Large (dim = 1024, MLP = 4096):

| Component | Layers per Block | Blocks | Params per Layer | Subtotal |
|---|---|---|---:|---:|
| **Early DoRA** (Q, V) | 2 | 18 | 9,216 | **331,776** |
| **Late DoRA** (Q, K, V, proj) | 4 | 6 | 9,216 | **221,184** |
| **Late DoRA** (fc1) | 1 | 6 | 24,576 | **147,456** |
| **Late DoRA** (fc2) | 1 | 6 | 21,504 | **129,024** |
| **Total DoRA** | | | | **~829K** |
| **Total LoRA** (no magnitude) | | | | **~737K** |

- **Trainable**: ~829K parameters (DoRA) or ~737K (LoRA)
- **Frozen**: ~299M parameters
- **Trainable ratio**: **~0.28% of total model**

> **Note**: The codebase uses `freeze_non_adapter_params()`, which strictly freezes everything except tensors containing `lora_`, `dwconv`, or `conv_gate` in their names. All pretrained weights, biases, and LayerNorms remain immutable.

---

## 4. (C) Self-Supervised Training Architecture

Because DA2-Large is trained on diverse unlabeled images, we do **not** have ground-truth depth for the target domain (Cityscapes/COCO). Instead, we use a **student-teacher self-distillation** paradigm:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│          Self-Supervised Adapter Training for Depth (Stage 1)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────┐         ┌─────────────────────────────┐   │
│   │        STUDENT              │         │        TEACHER              │   │
│   │   DA2-Large + DoRA adapters │         │   DA2-Large (frozen copy)   │   │
│   │   ───────────────────────── │         │   ───────────────────────── │   │
│   │   Encoder: adapters ACTIVE  │         │   Encoder: NO adapters      │   │
│   │   Decoder: FROZEN           │         │   Decoder: FROZEN           │   │
│   │   Trainable: ~829K params   │         │   Trainable: 0 params       │   │
│   └─────────────┬───────────────┘         └─────────────┬───────────────┘   │
│                 │                                       │                   │
│                 │    Same input image (augmented)       │                   │
│                 └──────────────┬────────────────────────┘                   │
│                                ▼                                            │
│                      ┌─────────────────┐                                    │
│                      │  Forward Pass   │                                    │
│                      └────────┬────────┘                                    │
│                               │                                             │
│              ┌────────────────┼────────────────┐                            │
│              ▼                ▼                ▼                            │
│    ┌─────────────────┐ ┌──────────────┐ ┌─────────────────┐                │
│    │  Student Depth  │ │ Teacher Depth│ │ Student Depth   │                │
│    │  (no aug)       │ │ (no aug)     │ │ (augmented)     │                │
│    │  D_student      │ │ D_teacher    │ │ D_student_aug   │                │
│    └────────┬────────┘ └──────┬───────┘ └────────┬────────┘                │
│             │                 │                  │                          │
│             └─────────────────┴──────────────────┘                          │
│                               │                                             │
│                               ▼                                             │
│              ┌─────────────────────────────────────┐                        │
│              │           LOSS COMPUTATION          │                        │
│              ├─────────────────────────────────────┤                        │
│              │                                     │                        │
│              │  1. DISTILLATION (MSE)              │                        │
│              │     L_mse = ||D_student − D_teacher||²                      │
│              │     Weight: 1.0                     │                        │
│              │                                     │                        │
│              │  2. RELATIVE DEPTH RANKING          │                        │
│              │     Sample pixel pairs (i, j)       │                        │
│              │     L_rank = MarginRankingLoss(     │                        │
│              │                 sign(D_i − D_j),    │                        │
│              │                 margin=0.1 )        │                        │
│              │     Weight: 0.1                     │                        │
│              │                                     │                        │
│              │  3. SCALE-INVARIANT CONSISTENCY     │                        │
│              │     L_si = (log D − log D*)² − λ(Σ(log D − log D*))²/n²   │
│              │     (Eigen et al.)                  │                        │
│              │     Weight: 0.5                     │                        │
│              │                                     │                        │
│              │  TOTAL = 1.0·L_mse + 0.1·L_rank + 0.5·L_si                │
│              │                                     │                        │
│              └─────────────────────────────────────┘                        │
│                               │                                             │
│                               ▼                                             │
│                    ┌─────────────────────┐                                  │
│                    │  AdamW (lr=1e-4)    │                                  │
│                    │  Cosine Annealing   │                                  │
│                    │  Grad Clip (max=1.0)│                                  │
│                    └─────────────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Loss Combination?

| Loss | Role | Critical for DA2-Large? |
|---|---|---|
| **MSE Distillation** | Keeps student close to teacher's strong zero-shot prior | Yes — prevents divergence |
| **Relative Ranking** | Enforces correct pairwise depth ordering at boundaries | Yes — improves edge sharpness for instance separation |
| **Scale-Invariant** | Compensates for relative (non-metric) depth output | **Essential** — DA2 has no absolute scale; log-space SI loss makes the model robust to arbitrary scale shifts |

The teacher is a **frozen exact copy** of the pretrained HF model with **no adapters injected**. Only the student's adapter parameters receive gradients.

---

## 5. (D) Complete Stage 1 Depth Pipeline → Instance Pseudo-Labels

Once adapter training converges (typically 10 epochs on Cityscapes train), the checkpoint is used in inference mode to generate instance pseudo-labels:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│     Stage 1 Inference: Depth → Instance Pseudo-Label Generation             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: LOAD ADAPTED MODEL                                                 │
│  ═══════════════════════════                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  checkpoint = torch.load("best.pt")                                 │    │
│  │  model = AutoModelForDepthEstimation.from_pretrained("depth-anything│    │
│  │           /Depth-Anything-V2-Large-hf")                             │    │
│  │  inject_lora_into_depth_model(model, variant="dora", rank=4,        │    │
│  │                               late_block_start=18)                  │    │
│  │  model.load_state_dict(checkpoint["model"])                         │    │
│  │  freeze_non_adapter_params(model)   ◄── Ensures only adapters run   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  STEP 2: GENERATE DEPTH MAP                                                 │
│  ════════════════════════════                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  inputs = processor(images=image, return_tensors="pt")              │    │
│  │  depth = model(**inputs).predicted_depth                            │    │
│  │  depth_np = depth.cpu().numpy()  ◄── Save as .npy                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  STEP 3: GRADIENT-BASED BOUNDARY EXTRACTION                                 │
│  ══════════════════════════════════════════                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Sobel_x = cv2.Sobel(depth_np, dx=1, dy=0)                         │    │
│  │  Sobel_y = cv2.Sobel(depth_np, dx=0, dy=1)                         │    │
│  │  grad_mag = sqrt(Sobel_x² + Sobel_y²)                              │    │
│  │                                                                     │    │
│  │  Binary mask = grad_mag > τ     ◄── τ = 0.03 (Cityscapes optimal)  │    │
│  │                                                                     │    │
│  │  Connected Components on (1 − binary_mask)                          │    │
│  │    • Each connected region = candidate instance                    │    │
│  │    • Filter by minimum area (typically 50–200 pixels)              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  STEP 4: INSTANCE PSEUDO-LABEL PNG                                          │
│  ════════════════════════════════                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  instance_id = 1, 2, 3, ... for each valid connected component      │    │
│  │  background = 0                                                       │    │
│  │                                                                     │    │
│  │  Save:  pseudo_labels/instance/<image_stem>_instance.png            │    │
│  │  Format: 16-bit or indexed PNG with instance IDs                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  OUTPUT ARTIFACTS:                                                          │
│    • depth_maps/da2_large/<stem>.npy   ← adapter-refined relative depth    │
│    • pseudo_labels/instance/<stem>.png ← thing-class candidate masks        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Hyperparameter Context (from March 2026 Ablations)

| Dataset | Optimal τ | Min Area | PQ_things (DA2-Large) |
|---|---|---:|---:|
| **Cityscapes** | 0.03 | 1000 | **20.20** |
| **COCO** | 0.08 | 1000 | **14.04** |

The depth-gradient threshold τ is the most sensitive hyperparameter. A lower τ (0.03) on Cityscapes captures fine-grained instance boundaries in structured driving scenes, while a higher τ (0.08) on COCO avoids oversegmentation in cluttered natural images.

---

## 6. Summary for the Professor

| Question | Answer |
|---|---|
| **Where do adapters go?** | **Only in the DINOv2-Large encoder** (24 blocks). Early blocks: Q+V only. Late blocks: full attention + MLP. |
| **Does the decoder change?** | **No.** `adapt_decoder=false`. The DPT head remains fully frozen to preserve zero-shot depth prior stability. |
| **How many parameters?** | **~829K trainable** (DoRA, r=4) out of ~300M total — **0.28%** of the model. This is parameter-efficient by design. |
| **Why self-supervised?** | No ground-truth depth exists for the target domain. Student-teacher distillation with MSE + ranking + scale-invariant losses adapts the model without labels. |
| **Why DA2-Large over DA3?** | On Cityscapes, DA3 edges DA2 slightly (PQ_things 20.90 vs 20.20). **But on COCO, DA2-Large beats DA3** (14.04 vs 13.76). DA2-Large is also more lightweight and easier to serve via HF Transformers. |
| **What is the output?** | `.npy` depth maps + instance candidate PNGs from Sobel-connected-components, fed into Stage 2/3 of MBPS. |

**Bottom line**: DA2-Large is kept frozen except for surgically placed DoRA adapters in the encoder. The adapters are trained with self-supervised losses that respect the model's relative-depth nature, then used at inference to generate sharper depth boundaries that improve instance pseudo-label quality for the panoptic segmentation pipeline.
