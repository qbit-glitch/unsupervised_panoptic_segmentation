# Lightweight Backbone Research for Mobile Unsupervised Panoptic Segmentation

## 1. Objective

Select and ablate lightweight backbones (<8M total params) to distill our UNet-refined pseudo-labels (PQ=28.0) into a standalone mobile model. The resulting model takes raw RGB images and outputs panoptic segmentation — no DINOv2, no depth network at inference.

## 2. Backbone Candidates — Comprehensive Catalog

All models below have **ImageNet-pretrained weights in `timm`** and support `features_only=True` for multi-scale feature extraction.

### Tier 1: Ultra-Light (<3M backbone params) — Fastest Mobile Inference

| Model | Full Params | Backbone Params | Feature Channels | ImageNet Top-1 | Year | Architecture |
|-------|------------|-----------------|------------------|----------------|------|-------------|
| **RepGhostNet-0.8** | 3.3M | 0.3M | [12, 20, 32, 64, 128] | ~73% | 2024 | Reparameterized Ghost modules |
| **EfficientViT-B0** | 3.4M | 0.7M | [16, 32, 64, 128] | 71.6% | 2023 | Multi-scale linear attention (MIT) |
| **GhostNet-1.0** | 5.2M | 0.9M | [16, 24, 40, 80, 160] | 73.9% | 2020 | Ghost modules (cheap linear ops) |
| **MobileViT-XXS** | 1.3M | 1.0M | [16, 24, 48, 64, 320] | 69.0% | 2022 | CNN + ViT hybrid (Apple) |
| **MobileNetV4-S** | 3.8M | 1.3M | [32, 32, 64, 96, 960] | 73.8% | 2024 | UIB + Mobile MQA (Google) |

### Tier 2: Light (3-5M backbone params) — Best Accuracy/Size Tradeoff

| Model | Full Params | Backbone Params | Feature Channels | ImageNet Top-1 | Year | Architecture |
|-------|------------|-----------------|------------------|----------------|------|-------------|
| **MobileViT-XS** | 2.3M | 1.9M | [32, 48, 64, 80, 384] | 74.8% | 2022 | CNN + ViT hybrid (Apple) |
| **MobileNetV3-L** | 5.5M | 3.0M | [16, 24, 40, 112, 960] | 75.8% | 2019 | SE + h-swish (Google) |
| **MobileOne-S1** | 4.8M | 3.5M | [64, 96, 192, 512, 1280] | 75.9% | 2023 | Reparameterized (Apple) |
| **EfficientNet-B0** | 5.3M | 3.6M | [16, 24, 40, 112, 320] | 77.7% | 2019 | Compound scaling (Google) |
| **MobileViTv2-1.0** | 4.9M | 4.4M | [64, 128, 256, 384, 512] | 78.1% | 2023 | Separable self-attention (Apple) |

### Tier 3: Medium (5-8M backbone params) — Highest Accuracy Under Budget

| Model | Full Params | Backbone Params | Feature Channels | ImageNet Top-1 | Year | Architecture |
|-------|------------|-----------------|------------------|----------------|------|-------------|
| **RepViT-M0.9** | 5.5M | 4.7M | [48, 96, 192, 384] | 78.7% | 2024 | Reparameterized ViT (CVPR'24) |
| **RepViT-M1.0** | 7.3M | 6.4M | [56, 112, 224, 448] | 80.0% | 2024 | First <8M model to hit 80% Top-1 |
| **FastViT-T12** | 7.6M | ~6M | [64, 128, 256, 512] | 79.1% | 2023 | Structural reparam (Apple, ICCV'23) |

### Not Selected (reasons)

| Model | Why Excluded |
|-------|-------------|
| MobileNetV2 | Superseded by V3 and V4 |
| ShuffleNetV2 | Weaker features, older architecture |
| MNASNet | Superseded by EfficientNet |
| EfficientNet-B1 (7.8M) | Over budget when + decoder |
| DINOv2 ViT-S/14 (22M) | Way over 8M budget |
| DINO ViT-Ti/16 (5.7M) | No multi-scale features (single-scale ViT) |

## 3. Key Architecture Insights

### Why RepViT-M0.9/M1.0 Are Likely Winners
- **CVPR 2024**: Latest reparameterized architecture, Pareto-optimal on mobile
- RepViT-M0.9 is 2.0% more accurate than FastViT-T8 at same latency
- RepViT-M1.0 first model under 8M to hit 80% ImageNet top-1
- 4-stage feature pyramid with clean channel progression (48→96→192→384) — perfect for FPN
- Proven on semantic segmentation: outperforms EfficientFormer by 1.7 mIoU on ADE20K
- Structural reparameterization: multi-branch training → single-branch inference (faster)

### Why MobileNetV4-S Is Interesting
- **ECCV 2024**: Latest Google architecture, optimized for ALL mobile accelerators
- Universal Inverted Bottleneck (UIB) + Mobile MQA attention
- Only 3.8M full params / 1.3M backbone — extremely light
- 2× faster than MobileNetV3 on EdgeTPU
- But: 73.8% ImageNet accuracy is lower than RepViT

### Why MobileViTv2-1.0 Deserves Attention
- Separable self-attention: O(N) complexity instead of O(N²)
- 78.1% ImageNet accuracy with only 4.9M params
- Rich multi-scale features: [64, 128, 256, 384, 512] — widest channels in Tier 2
- Apple-designed for mobile deployment

### CNN vs ViT for This Task
- **CNNs** (MobileNetV3/V4, GhostNet): Faster inference, better optimized for TFLite/NNAPI
- **ViT hybrids** (RepViT, MobileViT, FastViT): Better feature quality for dense prediction
- **For panoptic seg**: ViT hybrids likely win because global attention helps instance separation

## 4. Decoder Head Design

The backbone provides multi-scale features. We need a lightweight decoder that produces:
- **19-class semantic logits** (stuff + things)
- **Instance embeddings** (for thing-class clustering)

### Decoder Option A: Simple FPN + Panoptic Head (~1-2M params)
```
Backbone features (4-5 scales)
  → Lightweight FPN (1×1 conv to align channels → upsample → add)
  → Semantic head: 3×3 conv → 19-class logits
  → Instance head: 3×3 conv → 16-dim embeddings → mean-shift clustering
  → Panoptic merge
```

### Decoder Option B: LRASPP (Lite R-ASPP) (~0.3M params)
```
Backbone features (last 2 scales only)
  → Low-level: 1×1 conv (48 channels)
  → High-level: Global avg pool → 1×1 conv → sigmoid → multiply
  → Concat → 1×1 conv → bilinear upsample → logits
```
Designed specifically for MobileNetV3. Extremely light. No instance head though.

### Decoder Option C: DeepLabV3+ Lite (~1.5M params)
```
Backbone features
  → ASPP with depthwise separable convs (rates 1, 6, 12)
  → Low-level skip connection
  → Semantic + instance heads
```

### Recommendation: Decoder Option A (Simple FPN + Panoptic Head)
- Works with any backbone (not tied to MobileNetV3 like LRASPP)
- Supports both semantic and instance outputs
- ~1-2M params — stays well within 8M total budget
- Used by CUPS (their Cascade Mask R-CNN also uses FPN)

## 5. Ablation Plan

### 5.1 Backbone Ablation (6 runs)

We select 6 backbones spanning the three tiers, paired with the same FPN decoder:

| Run | Backbone | Backbone Params | Total (est.) | Rationale |
|-----|----------|-----------------|-------------|-----------|
| **B1** | RepViT-M0.9 | 4.7M | ~6.5M | **Expected winner** — CVPR'24 SOTA, best accuracy/size |
| **B2** | MobileNetV4-S | 1.3M | ~3M | **Smallest** — tests minimum viable backbone |
| **B3** | MobileNetV3-L | 3.0M | ~5M | **Baseline** — most widely deployed mobile backbone |
| **B4** | EfficientNet-B0 | 3.6M | ~5.5M | **Classic** — established dense prediction backbone |
| **B5** | MobileViTv2-1.0 | 4.4M | ~6M | **Best ViT hybrid** — separable attention, rich features |
| **B6** | GhostNet-1.0 | 0.9M | ~3M | **Ultra-light control** — tests if cheap features suffice |

### 5.2 Fixed Training Protocol

| Parameter | Value |
|-----------|-------|
| Pseudo-labels | UNet-refined, PQ=28.0 (from best checkpoint) |
| Input resolution | 512×1024 (half Cityscapes for mobile) |
| Backbone | Frozen for first 5 epochs, then unfreeze last 2 stages |
| Optimizer | AdamW, LR=1e-4 (backbone) / 1e-3 (decoder) |
| Schedule | Cosine decay, 50 epochs |
| Augmentation | Random crop 384×768, horizontal flip, color jitter, copy-paste |
| Batch size | 4 (2× GTX 1080 Ti) or 8 (with gradient accumulation) |
| Loss | CE (semantic) + discriminative embedding loss (instance) |
| Eval | PQ, PQ_stuff, PQ_things, mIoU on Cityscapes val |

### 5.3 Decoder Ablation (after best backbone selected)

| Run | Decoder | Params | Rationale |
|-----|---------|--------|-----------|
| D1 | Simple FPN + panoptic head | ~1.5M | Balanced |
| D2 | LRASPP (semantic only) | ~0.3M | Minimum viable |
| D3 | DeepLabV3+ Lite | ~1.5M | ASPP multi-scale |

### 5.4 ONNX/TFLite Export & Mobile Benchmark

After selecting best backbone + decoder:
1. Export to ONNX → measure latency on CPU
2. Convert to TFLite with INT8 quantization
3. Benchmark on Android (Pixel 8 / Samsung Galaxy) via NNAPI
4. Measure: latency, FPS, memory footprint, model size (MB)

## 6. Expected Outcomes

| Backbone | Expected PQ | Expected Latency (mobile) | Risk |
|----------|------------|--------------------------|------|
| RepViT-M0.9 | 25-27 | ~200-400ms | Low |
| MobileNetV4-S | 22-25 | ~100-200ms | Medium (weak features) |
| MobileNetV3-L | 24-26 | ~150-300ms | Low |
| EfficientNet-B0 | 24-27 | ~200-400ms | Low |
| MobileViTv2-1.0 | 25-27 | ~300-500ms | Low |
| GhostNet-1.0 | 20-24 | ~80-150ms | High (very small) |

## 7. Implementation Order

1. **Step 1**: Generate refined pseudo-labels using best UNet checkpoint (PQ=28.0) on Cityscapes train
2. **Step 2**: Implement generic training script: `train_mobile_panoptic.py`
   - Takes any timm backbone via `--backbone <timm_model_name>`
   - Simple FPN decoder with semantic + instance heads
   - Loads pseudo-labels as training targets
3. **Step 3**: Run backbone ablation (B1-B6) — ~2 days on 2× GTX 1080 Ti
4. **Step 4**: Select winner, run decoder ablation (D1-D3) — ~1 day
5. **Step 5**: ONNX export + quantization + mobile benchmark
6. **Step 6**: Build Android demo app

## 8. References

- [RepViT (CVPR 2024)](https://arxiv.org/pdf/2307.09283) — Revisiting Mobile CNN from ViT perspective
- [MobileNetV4 (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05647.pdf) — Universal models for mobile ecosystem
- [FastViT (ICCV 2023)](https://arxiv.org/pdf/2303.14189) — Fast hybrid ViT using structural reparameterization
- [MobileViTv2 (2023)](https://arxiv.org/abs/2206.02680) — Separable self-attention for mobile vision
- [EfficientViT (ICCV 2023)](https://arxiv.org/abs/2205.14756) — Multi-scale linear attention
- [MobileOne (CVPR 2023)](https://arxiv.org/abs/2206.04040) — Apple's reparameterized mobile backbone
- [GhostNetV2 (NeurIPS 2022)](https://arxiv.org/abs/2211.12905) — Cheap operations for efficient features
- [CUPS (CVPR 2025)](https://visinf.github.io/cups/) — Baseline: scene-centric unsupervised panoptic segmentation
