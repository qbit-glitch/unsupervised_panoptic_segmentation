# LoRA/DoRA Adapter Implementation Plan for MBPS Stage 1

## Executive Summary

This document describes the design and implementation of a parameter-efficient fine-tuning (PEFT) system using LoRA/DoRA adapters for the MBPS pseudo-label generation pipeline (Stage 1). The system adapts frozen pretrained models (DINOv2 + CAUSE-TR for semantics, DAv3/DepthPro for depth) to generate higher-quality pseudo-labels without requiring ground-truth labels.

---

## 1. Design Decisions

### 1.1 Which Models to Adapt?

| Component | Adapt? | Rationale |
|-----------|--------|-----------|
| DINOv2 ViT-B/14 Backbone | **Yes** | Primary feature extractor. Small adapter changes propagate through CAUSE-TR to improve semantic codes significantly. |
| CAUSE-TR Segment_TR Head | **Optional** | Small module (~2M params). Adapting it provides direct control over 90D code generation but adds complexity. Default: ON for maximum flexibility. |
| Depth Encoder (DAv3/DepthPro) | **Yes** | Depth encoder is typically a frozen DINOv2/DINOv3. Adapting it improves depth boundary quality. |
| Depth Decoder Head | **No (default)** | Decoder is lightweight; adapting it risks destabilizing scale estimates. Can be enabled via `--adapt_decoder`. |

### 1.2 Which Layers to Adapt?

**DINOv2 Backbone (Tiered Strategy):**
- **Early blocks (0 to late_block_start-1)**: `attn.qkv` only
  - Rationale: Early layers extract low-level features (edges, textures). Minimal adaptation prevents overfitting to noisy pseudo-labels.
- **Late blocks (late_block_start to N-1)**: `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2`
  - Rationale: Late layers encode semantic concepts. Full adaptation allows the model to adapt high-level representations to the target domain (Cityscapes).

**CAUSE-TR Head:**
- `tr.self_attn.out_proj`, `tr.multihead_attn.out_proj`
- `tr.linear1`, `tr.linear2` (FFN)
- Optional: `tr.f1`, `tr.f2` (1x1 conv projections) via Conv-LoRA

**Depth Models:**
- Same tiered strategy as DINOv2 (encoder blocks)
- Optional decoder adaptation: final projection layers only

### 1.3 Rank Selection

| Rank (r) | Alpha | Trainable Params | % of DINOv2 | Use Case |
|----------|-------|------------------|-------------|----------|
| 2 | 2.0 | ~300K | 0.4% | Ultra-low VRAM, first round self-training |
| 4 | 4.0 | ~600K | 0.8% | **Default balance** (recommended) |
| 8 | 8.0 | ~1.2M | 1.6% | Higher capacity, stable pseudo-labels |
| 16 | 16.0 | ~2.4M | 3.2% | Maximum capacity, risk of overfitting |

**Recommendation:** Start with `r=4, alpha=4.0` (alpha/r = 1.0). This adds ~0.8% trainable parameters to DINOv2 and provides sufficient capacity for domain adaptation without destabilizing pretrained features.

### 1.4 Variant Selection

| Variant | Description | Best For | Overhead |
|---------|-------------|----------|----------|
| **LoRA** | Standard low-rank adaptation | Baseline ablation | Minimal |
| **DoRA** | Weight-decomposed (magnitude + direction) | **Default** - protects pretrained feature scales | +magnitude vector per layer |
| **Conv-DoRA** | DoRA + 3x3 DWConv on activation space | Dense prediction (spatial inductive bias) | +9*r params per adapted linear |

**Recommendation:** Use **DoRA** as default. Use **Conv-DoRA** only if explicitly benchmarking spatial refinement, as it requires setting spatial dims before each forward pass.

### 1.5 Handling Special Tokens

DINOv2 uses CLS token + patch tokens. The adapter injection only modifies `nn.Linear` layers inside `Attention` and `Mlp` modules, which operate on the full sequence (CLS + patches). The forward pass naturally handles all tokens uniformly.

For **Conv-DoRA**, the spatial conv path only applies to patch tokens. Special tokens (CLS, registers if present) are zero-padded after the DWConv and before the up-projection, ensuring shape compatibility.

---

## 2. Architecture

### 2.1 File Structure

```
mbps_pytorch/models/adapters/
  __init__.py                 - Public API
  lora_layers.py              - Core LoRA/DoRA/Conv-DoRA layers + Conv2d LoRA
  dinov2_adapter.py           - DINOv2-specific injection (handles BlockChunk)
  cause_adapter.py            - CAUSE-TR head injection
  depth_adapter.py            - Depth model injection (generic ViT encoder)

train_semantic_adapter.py     - Self-supervised training for DINOv2+CAUSE-TR
train_depth_adapter_lora.py   - Self-supervised training for depth models
generate_semantic_pseudolabels_adapted.py  - K-means on adapted features
generate_instance_pseudolabels_adapted.py  - Depth-guided CC with adapted depth

configs/
  semantic_adapter_baseline.yaml
  depth_adapter_baseline.yaml
```

### 2.2 Adapter Layer Specifications

**LoRALinear:**
- `lora_A`: (rank, in_features) - Kaiming init
- `lora_B`: (out_features, rank) - Zero init
- Scaling: `alpha / rank`
- Forward: `W0 @ x + scaling * B @ A @ x + bias`

**DoRALinear:**
- `lora_magnitude`: (out_features, 1) - Column norms of W0
- `lora_A`, `lora_B`: Same as LoRA
- Forward: `m * (W0 + scaling * B @ A) / ||W0 + scaling * B @ A||_c @ x + bias`

**ConvDoRALinear:**
- Inherits DoRALinear
- `dwconv`: (rank, rank, 3, 3) depthwise conv - Zero init
- `conv_gate`: scalar - Zero init
- Forward: `DoRA(x) + gate * DWConv(reshape(A @ x))`

**LoRAConv2d (for CAUSE-TR 1x1 convs):**
- `lora_A`: (rank, in_channels, 1, 1) conv - Kaiming init
- `lora_B`: (out_channels, rank, 1, 1) conv - Zero init
- Forward: `W0 * x + scaling * B * A * x + bias`

---

## 3. Self-Supervised Training Objectives

### 3.1 Semantic Adapters (DINOv2 + CAUSE-TR)

**A. DINO Self-Distillation (Primary)**
- Student: DINOv2 with adapters
- Teacher: Frozen DINOv2 (original weights)
- Loss: Cross-entropy between student and teacher feature distributions
- Rationale: Pulls adapted features close to pretrained manifold while allowing domain-specific drift

**B. Cross-View Consistency**
- Two augmented views of the same image (color jitter, grayscale)
- Loss: Cosine similarity between student features
- Rationale: Enforces invariance to photometric perturbations

**C. Depth-Feature Alignment (DepthG)**
- Correlates CAUSE 90D codes with depth maps
- Loss: `-cd.clamp(0, 0.8) * (dd - shift)` where cd=code correlation, dd=depth correlation
- Rationale: Semantically similar regions should have similar depth

**D. CAUSE Cluster Loss**
- Continues CAUSE-TR's cluster assignment loss on adapted features
- Loss: `cluster.forward_centroid(seg_feat_ema)`
- Rationale: Maintains cluster structure for downstream k-means pseudo-labels

**Recommended Combination:** `distillation (w=1.0) + depth_cluster (w=0.05)`

### 3.2 Depth Adapters (DAv3 / DepthPro)

**A. Self-Distillation (Primary)**
- Student: Depth model with adapters
- Teacher: Frozen depth model
- Loss: MSE between student and teacher depth predictions
- Rationale: Prevents catastrophic drift from pretrained depth estimates

**B. Relative Depth Ranking**
- Samples pixel pairs, penalizes incorrect depth ordering
- Loss: Margin ranking loss with margin=0.1
- Rationale: Relative depth is more robust than absolute scale for instance boundaries

**C. Scale-Invariant Consistency**
- Log-space MSE with scale-invariant term
- Loss: `mean(diff^2) - lambda * (mean(diff))^2`
- Rationale: Matches depth scale independently per image

**Recommended Combination:** `distillation (w=1.0) + ranking (w=0.1)`

---

## 4. Integration with Pseudo-Label Pipeline

### 4.1 Training Flow

```
Stage 1a: Train Semantic Adapters
  Input: Cityscapes train images + precomputed depth maps
  Output: adapted_backbone.pt, adapted_segment_tr.pt
  
Stage 1b: Train Depth Adapters
  Input: Cityscapes train images
  Output: adapted_depth_model.pt

Stage 1c: Generate Adapted Pseudo-Labels
  Input: adapted models + Cityscapes images
  Output: pseudo_semantic_adapted/ , pseudo_instance_adapted/
```

### 4.2 Compatibility with Stage 2/3

The adapted pseudo-labels are saved in the **same format** as existing pseudo-labels:
- Semantic: PNG files with uint8 cluster IDs (0 to K-1)
- Instance: PNG files with uint16 instance IDs

Existing Stage 2/3 configs can be updated by simply changing the pseudo-label directory path:
```yaml
# In existing Stage 2/3 config
pseudo_semantic_dir: "cityscapes/pseudo_semantic_adapted/train"
pseudo_instance_dir: "cityscapes/pseudo_instance_adapted/train"
```

---

## 5. VRAM and Compute Estimates

| Configuration | Backbone Params | Adapter Params | Total Trainable | VRAM (batch=4) |
|---------------|-----------------|----------------|-----------------|----------------|
| DINOv2 + DoRA r=4 | 86M (frozen) | ~600K | ~600K | ~6 GB |
| DINOv2 + DoRA r=4 + CAUSE | 86M + 2M (frozen) | ~800K | ~800K | ~7 GB |
| DepthPro + DoRA r=4 | 300M (frozen) | ~1.2M | ~1.2M | ~10 GB |
| DINOv2 + Conv-DoRA r=4 | 86M (frozen) | ~700K | ~700K | ~6.5 GB |

**Training Time:** ~2-4 hours per epoch on Cityscapes train (2975 images) with batch_size=4 on a single A6000.

---

## 6. Usage Instructions

### 6.1 Train Semantic Adapters

```bash
python mbps_pytorch/train_semantic_adapter.py \
    --data_dir /data/datasets \
    --output_dir checkpoints/semantic_adapter_dora \
    --variant dora --rank 4 --alpha 4.0 \
    --late_block_start 6 \
    --adapt_cause \
    --losses distillation,depth_cluster \
    --lambda_depth 0.05 \
    --epochs 10 --lr 1e-4 --batch_size 4
```

### 6.2 Train Depth Adapters

```bash
python mbps_pytorch/train_depth_adapter_lora.py \
    --model_type depthpro \
    --data_dir /data/cityscapes/leftImg8bit/train \
    --output_dir checkpoints/depth_adapter_dora \
    --variant dora --rank 4 --alpha 4.0 \
    --losses distillation,ranking \
    --epochs 10 --lr 1e-4 --batch_size 4
```

### 6.3 Generate Adapted Semantic Pseudo-Labels

```bash
python mbps_pytorch/generate_semantic_pseudolabels_adapted.py \
    --checkpoint checkpoints/semantic_adapter_dora/best.pt \
    --data_dir /data/cityscapes/leftImg8bit/train \
    --output_dir /data/cityscapes/pseudo_semantic_adapted/train \
    --num_clusters 54
```

### 6.4 Generate Adapted Instance Pseudo-Labels

```bash
python mbps_pytorch/generate_instance_pseudolabels_adapted.py \
    --checkpoint checkpoints/depth_adapter_dora/best.pt \
    --model_type depthpro \
    --image_dir /data/cityscapes/leftImg8bit/train \
    --semantic_dir /data/cityscapes/pseudo_semantic_adapted/train \
    --output_dir /data/cityscapes/pseudo_instance_adapted/train
```

---

## 7. Ablations and Expected Gains

| Ablation | Expected PQ Gain | Notes |
|----------|-----------------|-------|
| Baseline (frozen) | 0.0 | Current Stage 1 pipeline |
| + DINOv2 adapters only | +0.5 to +1.0 PQ | Backbone feature refinement |
| + CAUSE-TR adapters | +0.3 to +0.5 PQ | Direct code optimization |
| + Depth adapters | +0.5 to +1.0 PQ | Better instance boundaries |
| **Combined** | **+1.5 to +2.5 PQ** | Synergistic improvement |

---

## 8. Implementation Checklist

- [x] LoRA/DoRA injection utilities (reusable across models)
- [x] DINOv2 backbone adapter wrapper (handles BlockChunk)
- [x] CAUSE-TR head adapter wrapper (linear + conv2d)
- [x] Depth model adapter wrapper (generic ViT encoder)
- [x] Self-supervised training loop for semantic adapters
- [x] Self-supervised training loop for depth adapters
- [x] Pseudo-label generation script using adapted models (semantic)
- [x] Pseudo-label generation script using adapted models (instance)
- [x] Config YAMLs for different adapter configurations
- [ ] Tests / smoke tests (to be added)

---

## 9. Future Work

1. **Progressive Rank Expansion:** Start with r=2, expand to r=4, then r=8 across pseudo-label refinement rounds (inspired by CUPS progressive LoRA).
2. **Adapter Composition:** Combine semantic and depth adapters into a joint adapter that shares some low-rank parameters.
3. **Cross-Domain Transfer:** Train adapters on multiple datasets (Cityscapes + COCO) for more generalizable pseudo-labels.
4. **Attention-Guided Depth:** Use DINOv2 attention maps to guide depth adapter training (where attention is high, depth boundaries should be sharp).

---

*Document version: 1.0*
*Date: 2026-04-21*
