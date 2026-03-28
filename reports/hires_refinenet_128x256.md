# HiRes RefineNet: High-Resolution Cross-Modal Semantic Refinement for Unsupervised Panoptic Segmentation

## 1. Motivation

Unsupervised panoptic segmentation relies on pseudo-labels generated from self-supervised features. Our best pseudo-labels, produced via k=80 overclustering of CAUSE-TR features with depth-guided instance splitting, achieve PQ=26.74 (PQ_stuff=32.08, PQ_things=19.41) on Cityscapes val. While prior work with CSCMRefineNet at 32x64 patch resolution improved semantic quality (mIoU +5.3%, PQ_stuff +1.30), it consistently degraded PQ_things from 19.41 to 17.10 across all hyperparameter configurations and architectural ablations (TAD, BPL, ASPP-lite).

We identified the root cause as a fundamental resolution bottleneck: at 32x64, the feature map contains only 2,048 spatial positions for a 1024x2048 image. Small thing instances (bicycles, riders, motorcycles) that occupy fewer than 500 pixels in the original image collapse to sub-pixel representations, causing adjacent instances to merge during refinement. No loss function or receptive field modification can recover spatial precision that has been destroyed by downsampling.

This motivates operating at 128x256 resolution (32,768 spatial positions) — a 16x increase over 32x64 that preserves thing-object boundaries while remaining computationally tractable. We systematically ablate (1) the upsampling strategy from native DINOv2 resolution to 128x256, and (2) the refinement architecture, comparing five approaches spanning convolutions, attention, and state-space models.

## 2. Architecture: HiRes RefineNet

### 2.1 Overall Pipeline

```
                          HiRes RefineNet Pipeline
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                                                                          │
  │  DINOv2 ViT-B/14 (frozen)                                               │
  │  768-dim, 32x64 patches                                                 │
  │       │                                                                  │
  │       ├──────────────────────┐                                           │
  │       │                      │                                           │
  │       ▼                      ▼                                           │
  │  ┌──────────┐         ┌──────────────────┐                               │
  │  │ Semantic  │         │  Depth-Feature   │                               │
  │  │Projection│         │   Projection     │                               │
  │  │ Conv 1x1 │         │ Conv 1x1 + FiLM  │◄── Depth + Sobel grads       │
  │  │ 768→192  │         │ 768→192          │    (128x256)                  │
  │  └────┬─────┘         └────────┬─────────┘                               │
  │       │  (192, 32x64)          │  (192, 32x64)                           │
  │       │                        │                                         │
  │       ▼                        ▼                                         │
  │  ┌─────────────────────────────────────────┐                             │
  │  │          Upsampling Module               │                             │
  │  │   32x64 ──────────────────► 128x256     │                             │
  │  │                                          │                             │
  │  │   Strategy A: Bilinear interpolation     │                             │
  │  │   Strategy B: 2-stage transposed conv    │                             │
  │  │   Strategy C: PixelShuffle               │                             │
  │  └──────────────┬──────────────────────────┘                             │
  │                  │  (192, 128x256) × 2 streams                           │
  │                  ▼                                                       │
  │  ┌─────────────────────────────────────────┐                             │
  │  │      Refinement Blocks × 4               │                             │
  │  │                                          │                             │
  │  │   ┌─────────┐      ┌─────────┐          │                             │
  │  │   │Semantic │◄────►│ Depth   │          │                             │
  │  │   │ Stream  │ α, β │ Stream  │          │                             │
  │  │   └─────────┘      └─────────┘          │                             │
  │  │                                          │                             │
  │  │   Variant 1: Conv2d (DW-Sep 3x3)        │                             │
  │  │   Variant 2: Windowed Self-Attention     │                             │
  │  │   Variant 3: VisionMamba2 (Bidirectional)│                             │
  │  │   Variant 4: Spatial Mamba (SS2D 4-way)  │                             │
  │  │   Variant 5: MambaOut (Gated CNN)        │                             │
  │  └──────────────┬──────────────────────────┘                             │
  │                  │  (192, 128x256)                                       │
  │                  ▼                                                       │
  │  ┌─────────────────────────────────────────┐                             │
  │  │      Classification Head                 │                             │
  │  │      Conv2d(192, 19, 1) at 128x256      │                             │
  │  │      → upsample to 1024x2048 for eval   │                             │
  │  └─────────────────────────────────────────┘                             │
  │                                                                          │
  └──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Upsampling Strategies

The DINOv2 ViT-B/14 backbone produces features at 32x64 patch resolution (stride 14). To operate at 128x256, we require a 4x spatial upsampling. We ablate three strategies with increasing parametric complexity:

#### Strategy A: Bilinear Interpolation

```
  Input (B, 192, 32, 64)
       │
       ▼
  F.interpolate(scale_factor=4, mode='bilinear')
       │
       ▼
  Output (B, 192, 128, 256)

  Parameters: 0
  Computation: Negligible
```

The simplest approach. Preserves the feature distribution exactly but introduces spatial smoothness — upsampled features lack high-frequency boundary detail.

#### Strategy B: Two-Stage Transposed Convolution

```
  Input (B, 192, 32, 64)
       │
       ▼
  ConvTranspose2d(192, 192, k=4, s=2, p=1)  ──► (B, 192, 64, 128)
  GroupNorm(1, 192) + GELU
       │
       ▼
  ConvTranspose2d(192, 192, k=4, s=2, p=1)  ──► (B, 192, 128, 256)
  GroupNorm(1, 192) + GELU
       │
       ▼
  Output (B, 192, 128, 256)

  Parameters: ~300K
  Computation: Low
```

Learnable upsampling in two stages. Each stage doubles spatial resolution via stride-2 transposed convolution. The learned kernels can sharpen boundaries during upsampling — directly addressing the PQ_things degradation mechanism.

#### Strategy C: PixelShuffle

```
  Input (B, 192, 32, 64)
       │
       ▼
  Conv2d(192, 192 × 16, kernel=1)    ──► (B, 3072, 32, 64)
  PixelShuffle(upscale_factor=4)      ──► (B, 192, 128, 256)
  GroupNorm(1, 192) + GELU
       │
       ▼
  Output (B, 192, 128, 256)

  Parameters: ~590K
  Computation: Low-Medium
```

Sub-pixel convolution (Shi et al., CVPR 2016). A 1x1 convolution expands channels by 16x, then PixelShuffle rearranges channels into spatial dimensions. Learns a per-subpixel mapping from the low-res feature space.

### 2.3 Refinement Block Architectures

All blocks operate at 128x256 resolution with dual-stream (semantic + depth) cross-modal gating. Each variant shares the same cross-modal coupling mechanism: a learned sigmoid gate modulates information flow between streams via parameters alpha and beta.

#### Variant 1: Conv2d (CoupledConvBlock)

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    CoupledConvBlock                          │
  │                                                             │
  │  sem (192, 128, 256)              depth (192, 128, 256)     │
  │       │                                │                    │
  │       ▼                                ▼                    │
  │  GroupNorm                         GroupNorm                │
  │       │                                │                    │
  │       │◄──── α · σ(Conv1x1(d)) ────────┤                    │
  │       ├──── β · σ(Conv1x1(s)) ────────►│                    │
  │       │                                │                    │
  │       ▼                                ▼                    │
  │  DW-Conv 3×3                      DW-Conv 3×3              │
  │  GN + GELU                        GN + GELU                │
  │  Conv 1×1 (192→384)               Conv 1×1 (192→384)       │
  │  GELU                             GELU                     │
  │  Conv 1×1 (384→192)               Conv 1×1 (384→192)       │
  │       │                                │                    │
  │       + residual                       + residual           │
  │       │                                │                    │
  └───────┴────────────────────────────────┴────────────────────┘

  Parameters: ~460K per block
  FLOPs at 128x256: ~3.8G per block
```

The proven baseline from 32x64 experiments. At 128x256, each depthwise 3x3 convolution processes 16x more spatial positions. Local receptive field (3x3) captures fine-grained thing boundaries.

#### Variant 2: Windowed Self-Attention

```
  ┌─────────────────────────────────────────────────────────────┐
  │               Windowed Attention Block                      │
  │                                                             │
  │  sem (192, 128, 256)              depth (192, 128, 256)     │
  │       │                                │                    │
  │       ▼                                ▼                    │
  │  LayerNorm                         LayerNorm                │
  │       │                                │                    │
  │       │◄──── α · σ(Linear(d)) ─────────┤                    │
  │       ├──── β · σ(Linear(s)) ─────────►│                    │
  │       │                                │                    │
  │       ▼                                ▼                    │
  │  Partition 8×8 windows            Partition 8×8 windows     │
  │  (512 windows × 64 tokens)        (512 windows × 64 tokens)│
  │       │                                │                    │
  │       ▼                                ▼                    │
  │  MHSA (4 heads, d_k=48)          MHSA (4 heads, d_k=48)   │
  │  + relative position bias          + relative position bias │
  │       │                                │                    │
  │       ▼                                ▼                    │
  │  Unpartition                       Unpartition              │
  │  + residual                        + residual               │
  │       │                                │                    │
  │       ▼                                ▼                    │
  │  FFN (192→384→192)                FFN (192→384→192)         │
  │  + residual                        + residual               │
  │       │                                │                    │
  └───────┴────────────────────────────────┴────────────────────┘

  Parameters: ~520K per block
  FLOPs at 128x256: ~2.1G per block
  Window: 8×8 = 64 tokens per window, O(64^2) = O(4096) attention per window
  Alternating blocks use shifted windows (Swin-style) for cross-window communication
```

Windowed attention restricts self-attention to local 8x8 windows, reducing complexity from O(32K^2) to O(512 x 64^2). Shifted windows on alternating blocks enable cross-window information flow. Relative position bias encodes spatial relationships within each window.

#### Variant 3: VisionMamba2 (Bidirectional)

```
  ┌─────────────────────────────────────────────────────────────┐
  │               VisionMamba2 Bidirectional Block              │
  │                                                             │
  │  sem (192, 128, 256)              depth (192, 128, 256)     │
  │       │                                │                    │
  │       ▼                                ▼                    │
  │  LayerNorm                         LayerNorm                │
  │       │                                │                    │
  │       │◄──── α · σ(Linear(d)) ─────────┤                    │
  │       ├──── β · σ(Linear(s)) ─────────►│                    │
  │       │                                │                    │
  │       ▼                                ▼                    │
  │  Raster scan                       Raster scan              │
  │  (128,256) → 32768 tokens          (128,256) → 32768 tokens│
  │       │                                │                    │
  │       ├──► Mamba2_fwd ──┐              ├──► Mamba2_fwd ──┐  │
  │       └──► Mamba2_bwd ──┤              └──► Mamba2_bwd ──┤  │
  │                         ▼                                ▼  │
  │              Sigmoid gate merge              Sigmoid gate    │
  │                    │                              │         │
  │       Unscan to (128, 256)          Unscan to (128, 256)    │
  │       + residual                    + residual              │
  │       │                                │                    │
  │       ▼                                ▼                    │
  │  FFN (192→384→192)                FFN (192→384→192)         │
  │  + residual                        + residual               │
  │       │                                │                    │
  └───────┴────────────────────────────────┴────────────────────┘

  Parameters: ~1.2M per block (2 Mamba2 layers per stream × 2 streams)
  FLOPs at 128x256: O(N) linear in sequence length
  Sequence length: 32,768 tokens
```

Our existing VisionMamba2 with bidirectional scanning. Forward and backward scans capture global context in both directions. The sigmoid gate learns to merge directional information. Linear O(N) complexity scales well to 32K tokens.

#### Variant 4: Spatial Mamba (SS2D Cross-Scan)

```
  ┌─────────────────────────────────────────────────────────────┐
  │               Spatial Mamba SS2D Block                      │
  │                                                             │
  │  sem (192, 128, 256)                                        │
  │       │                                                     │
  │       ▼                                                     │
  │  4-way cross-scan:                                          │
  │                                                             │
  │  ──────►  row-major forward                                 │
  │  ◄──────  row-major backward                                │
  │  │││││▼   col-major forward                                 │
  │  ▲│││││   col-major backward                                │
  │                                                             │
  │  Batch all 4 directions: (4B, 32768, 192)                   │
  │       │                                                     │
  │       ▼                                                     │
  │  Single shared Mamba2 layer                                 │
  │       │                                                     │
  │       ▼                                                     │
  │  Cross-unscan + learned 4→1 projection                      │
  │  Linear(4 × 192, 192)                                      │
  │       │                                                     │
  │       + residual + FFN                                      │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

  Parameters: ~800K per block (1 shared Mamba2 layer + merge proj per stream)
  FLOPs at 128x256: O(4N) — 4 directions batched
  Advantage: Captures both row and column spatial structure
```

VMamba-style Selective Scan 2D. Four scanning directions (row-fwd, row-bwd, col-fwd, col-bwd) capture spatial relationships in all orientations. A single shared Mamba2 layer processes all directions in a batched call, and a learned projection merges the four output views. This variant tests whether explicit 2D spatial awareness improves over simple bidirectional scanning.

#### Variant 5: MambaOut (Gated CNN)

```
  ┌─────────────────────────────────────────────────────────────┐
  │               MambaOut Block (Yu et al., 2024)              │
  │                                                             │
  │  sem (192, 128, 256)              depth (192, 128, 256)     │
  │       │                                │                    │
  │       ▼                                ▼                    │
  │  GroupNorm                         GroupNorm                │
  │       │                                │                    │
  │       │◄──── α · σ(Conv1x1(d)) ────────┤                    │
  │       ├──── β · σ(Conv1x1(s)) ────────►│                    │
  │       │                                │                    │
  │       ▼                                ▼                    │
  │  Conv 1×1 (192→384)               Conv 1×1 (192→384)       │
  │       │                                │                    │
  │       ├──► DW-Conv 7×7 ──► SiLU ──► gate                   │
  │       └──► Linear ────────────────► value                   │
  │            gate ⊙ value                gate ⊙ value         │
  │       │                                │                    │
  │       ▼                                ▼                    │
  │  Conv 1×1 (384→192)               Conv 1×1 (384→192)       │
  │       │                                │                    │
  │       + residual                       + residual           │
  │       │                                │                    │
  └───────┴────────────────────────────────┴────────────────────┘

  Parameters: ~580K per block
  FLOPs at 128x256: ~4.5G per block
  Key insight: Mamba = gated conv + SSM. MambaOut removes SSM, keeps gating.
```

MambaOut (Yu et al., 2024) demonstrated that for visual recognition tasks, Mamba's selective state-space mechanism provides minimal benefit — the gated convolution structure is the primary source of representational power. This variant replaces the SSM with a larger depthwise convolution (7x7) and SiLU-gated value projection. It serves as a control: if MambaOut matches VisionMamba2/SpatialMamba, the SSM is not contributing to refinement quality.

### 2.4 Loss Functions

All losses computed at 128x256 resolution. Pseudo-labels downsampled from 1024x2048 via nearest-neighbor; depth maps from 512x1024 via bilinear interpolation.

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    Loss Computation                         │
  │                                                             │
  │  Pseudo-labels (1024x2048)──► nearest ──► (128x256)        │
  │  Depth maps (512x1024) ──► bilinear ──► (128x256)          │
  │                                                             │
  │  L_total = w_d · L_distill                                  │
  │          + λ_align · L_align                                │
  │          + λ_proto · L_proto                                │
  │          + λ_ent · L_entropy                                │
  │                                                             │
  │  w_d: cosine warmdown from 1.0 → 0.85 (distill floor)      │
  │  λ_align = 0.25  (depth-boundary alignment)                │
  │  λ_proto = 0.025 (feature prototype consistency)            │
  │  λ_ent   = 0.025 (entropy minimization)                    │
  │  label_smoothing = 0.1                                      │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
```

## 3. Experimental Setup

### 3.1 Dataset and Evaluation

- **Dataset**: Cityscapes (2975 train, 500 val), 1024x2048 resolution
- **Pseudo-labels**: k=80 overclustered CAUSE-TR features, mapped to 19 trainIDs
- **Input features**: Pre-extracted DINOv2 ViT-B/14 (768-dim, 32x64 patches, float16)
- **Depth**: SPIdepth monocular depth (512x1024, float32)
- **Evaluation**: Panoptic Quality (PQ, PQ_stuff, PQ_things), mIoU, changed_pct

### 3.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Resolution | 128x256 |
| bridge_dim | 192 |
| num_blocks | 4 |
| batch_size | 4 (2 if OOM) |
| lr | 5e-5 |
| scheduler | Cosine annealing |
| epochs | 20 |
| eval_interval | 2 |
| distill_min | 0.85 |
| lambda_align | 0.25 |
| lambda_proto | 0.025 |
| lambda_ent | 0.025 |
| label_smoothing | 0.1 |
| num_classes | 19 |
| device | MPS (Apple M4 Pro, 48GB) |

### 3.3 Baselines

| Config | Resolution | PQ | PQ_stuff | PQ_things | mIoU |
|--------|-----------|-----|----------|-----------|------|
| Input pseudo-labels | 1024x2048 | 26.74 | 32.08 | 19.41 | ~50% |
| CSCMRefineNet Run D | 32x64 | 26.52 | 33.38 | 17.10 | 55.31 |

## 4. Phase 1 Results: Upsampling Strategy Ablation

Architecture fixed to Conv2d (CoupledConvBlock). Only the upsampling module varies.

| Run | Upsampling | Extra Params | Best PQ | PQ_stuff | PQ_things | mIoU | Best Epoch | changed% | Time/Epoch |
|-----|-----------|-------------|---------|----------|-----------|------|------------|----------|------------|
| U-B | Transposed Conv | ~300K | 27.50 | 34.77 | 17.58 | 56.89 | 12 (PQ) / 16 (things) | 6.12% | ~948s |
| U-A | Bilinear | 0 | 27.29 | 34.92 | 16.85 | 57.01 | 14 (PQ) / 18 (mIoU) | 6.21% | ~920s |
| U-C | PixelShuffle | ~590K | 27.26 | 34.82 | 16.86 | 57.15 | 12 (PQ) / 16 (mIoU) | 6.23% | ~960s |

### 4.1 Analysis

Transposed convolution (U-B) achieves the highest PQ (27.50) and PQ_things (17.58), outperforming both bilinear (U-A) and PixelShuffle (U-C) by a consistent margin of 0.72--0.73 PQ_things across all evaluation epochs. The advantage concentrates on large thing classes (truck +3.74, train +2.20 vs bilinear), where learned upsampling kernels sharpen boundary features during spatial expansion. Bilinear and PixelShuffle achieve marginally higher mIoU (57.01, 57.15 vs 56.89) and PQ_stuff (34.92, 34.82 vs 34.77), confirming that smooth upsampling benefits semantic classification but not instance separation. PixelShuffle's 590K extra parameters provide no advantage over zero-parameter bilinear, and its slow convergence (PQ=21.69 at epoch 2 vs 26.49 for U-B) indicates the 192x16=3072 channel expansion requires substantial training to learn useful spatial rearrangement patterns. The dissociation between semantic quality (mIoU) and panoptic quality (PQ, PQ_things) underscores that upsampling strategies must be evaluated on panoptic metrics for panoptic segmentation tasks.

### 4.2 Selected Upsampling Strategy

**Transposed Convolution (U-B)** is selected for Phase 2 architecture ablation based on best PQ (27.50) and best PQ_things (17.58). The 300K parameter overhead is justified by the +0.73 PQ_things improvement over parameter-free bilinear.

## 5. Phase 2 Results: Architecture Ablation

Upsampling fixed to best from Phase 1. Five refinement architectures compared.

| Run | Architecture | Params | Best PQ | PQ_stuff | PQ_things | mIoU | Best Epoch | changed% | Time/Epoch |
|-----|-------------|--------|---------|----------|-----------|------|------------|----------|------------|
| A-Conv | Conv2d | ~1.83M | — | — | — | — | — | — | — |
| A-Attn | Windowed Self-Attention | ~2.1M | — | — | — | — | — | — | — |
| A-VM2 | VisionMamba2 (Bidir.) | ~4.8M | — | — | — | — | — | — | — |
| A-SM | Spatial Mamba (SS2D) | ~3.2M | — | — | — | — | — | — | — |
| A-MOut | MambaOut (Gated CNN) | ~2.3M | — | — | — | — | — | — | — |

### 5.1 PQ_things Recovery Analysis

*Key question: Does 128x256 resolution recover the PQ_things regression observed at 32x64?*

| Config | PQ_things | Delta vs Input (19.41) |
|--------|-----------|----------------------|
| 32x64 Conv2d (Run D) | 17.10 | -2.31 |
| 128x256 A-Conv | — | — |
| 128x256 A-Attn | — | — |
| 128x256 A-VM2 | — | — |
| 128x256 A-SM | — | — |
| 128x256 A-MOut | — | — |

### 5.2 Computational Efficiency

| Architecture | Params | Time/Epoch | Memory (BS=4) | PQ/Param Efficiency |
|-------------|--------|------------|---------------|-------------------|
| Conv2d | — | — | — | — |
| Windowed Attn | — | — | — | — |
| VisionMamba2 | — | — | — | — |
| Spatial Mamba | — | — | — | — |
| MambaOut | — | — | — | — |

### 5.3 Analysis

*To be completed after Phase 2 runs.*

## 6. Discussion

### 6.1 Resolution vs Architecture

*Does the resolution increase (32x64 → 128x256) matter more than the architecture choice? Compare the worst 128x256 model against the best 32x64 model.*

### 6.2 SSM Contribution

*Does the SSM component in VisionMamba2/SpatialMamba provide measurable benefit over MambaOut (gated CNN without SSM)? If not, Mamba's recurrence is unnecessary for semantic refinement at this scale.*

### 6.3 Global vs Local Context

*Windowed attention and Conv2d operate locally. VisionMamba2 and SpatialMamba capture global context. Does global context help refinement, or is local boundary precision sufficient?*

### 6.4 Limitations

- Features are upsampled from 32x64, not natively computed at 128x256. The upsampled features cannot contain spatial detail finer than the original 32x64 grid.
- MPS float32 constraint increases memory usage and training time vs CUDA mixed precision.
- Panoptic evaluation includes connected-component analysis which may interact with resolution differently.

## 7. Conclusion

*To be completed after all experiments.*
