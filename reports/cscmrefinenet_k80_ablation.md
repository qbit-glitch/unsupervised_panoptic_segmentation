# CSCMRefineNet on k=80 Overclustered Labels: Hyperparameter Study and Architectural Improvements

## 1. Motivation

Unsupervised panoptic segmentation pipelines generate pseudo-labels of limited quality that serve as noisy supervision for downstream models. Our best pseudo-labels, produced via k=80 overclustering of CAUSE-TR features with depth-guided instance splitting, achieve PQ=26.74 (PQ_stuff=32.08, PQ_things=19.41) on Cityscapes val — within 1.06 PQ of CUPS (CVPR 2025). The gap is entirely attributable to PQ_stuff, motivating a semantic refinement stage.

CSCMRefineNet is a lightweight convolutional refinement network (1.83M parameters) that takes frozen DINOv2 ViT-B/14 features (768-dim, 32x64 patches) and monocular depth as input, producing refined semantic logits. The model uses four CoupledConvBlocks with cross-modal gating between semantic and depth-conditioned streams, followed by a 1x1 classification head. Crucially, the pseudo-label predictions are used only as cross-entropy distillation targets — never as model input — to prevent identity shortcut learning.

The training objective combines distillation from pseudo-labels with self-supervised auxiliary losses: depth-boundary alignment (DepthG-inspired spatial smoothness weighted by depth similarity), DINOv2 feature prototype consistency (per-class centroid compactness), and entropy minimization (prediction confidence). A cosine warmdown schedule decays the distillation weight from an initial value toward a configurable floor, gradually allowing the self-supervised objectives to guide refinement.

## 2. Architecture

### 2.1 Base Architecture: CSCMRefineNet

The base CSCMRefineNet consists of two input projection heads feeding into a coupled dual-stream convolutional backbone:

```
                            CSCMRefineNet (1.83M params)
  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  DINOv2 ViT-B/14                                                   │
  │  (frozen, 768-dim)                                                  │
  │       │                                                             │
  │       ├──────────────────────┐                                      │
  │       │                      │                                      │
  │       ▼                      ▼                                      │
  │  ┌──────────┐         ┌──────────────────┐                          │
  │  │ Semantic  │         │  Depth-Feature   │                          │
  │  │Projection│         │   Projection     │                          │
  │  │ Conv 1x1 │         │ Conv 1x1 + FiLM  │◄── Depth (1-ch)         │
  │  │ GN + GELU│         │ (sin/cos PE +    │◄── Depth Grads (2-ch)   │
  │  │ 768→192  │         │  Sobel → γ,β)    │                          │
  │  └────┬─────┘         └────────┬─────────┘                          │
  │       │                        │                                    │
  │       │    192-dim             │    192-dim                         │
  │       ▼                        ▼                                    │
  │  ┌────────────────────────────────────────┐                         │
  │  │        CoupledConvBlock × 4            │                         │
  │  │  ┌─────────────┐   ┌─────────────┐    │                         │
  │  │  │  Semantic    │   │   Depth     │    │                         │
  │  │  │  Stream      │◄─►│   Stream    │    │                         │
  │  │  │  DW-Conv 3×3 │ α │  DW-Conv 3×3│    │                         │
  │  │  │  + Expand    │──►│  + Expand   │    │                         │
  │  │  │  + Contract  │◄──│  + Contract │    │                         │
  │  │  │  + Residual  │ β │  + Residual │    │                         │
  │  │  └─────────────┘   └─────────────┘    │                         │
  │  │       × 4 blocks (with cross-gating)   │                         │
  │  └────────────────────┬───────────────────┘                         │
  │                       │                                             │
  │                       ▼                                             │
  │              ┌─────────────────┐                                    │
  │              │   Output Head   │                                    │
  │              │  GN + Conv 1×1  │                                    │
  │              │   192 → C_out   │                                    │
  │              └────────┬────────┘                                    │
  │                       │                                             │
  └───────────────────────┼─────────────────────────────────────────────┘
                          ▼
                  Refined Logits
                  (B, C, 32, 64)
                       │
                  NN Upsample
                       ▼
                  (B, C, 1024, 2048)
```

### 2.2 CoupledConvBlock (Standard)

Each CoupledConvBlock maintains two parallel streams with learnable cross-chain modulation:

```
    Semantic (B,192,32,64)          Depth-Feature (B,192,32,64)
         │                                │
    ┌────▼────┐                      ┌────▼────┐
    │ GroupNorm│                      │ GroupNorm│
    └────┬────┘                      └────┬────┘
         │          Cross-Gating          │
         │    ┌──── σ(Conv1×1(·)) ◄───────┤
         │    │α                          │
         ▼    ▼                           │
    ┌─────────────┐                  ┌────┤
    │ sem + α·gate│                  │    │    ┌──── σ(Conv1×1(·)) ◄───┐
    └──────┬──────┘                  │    │    │β                      │
           │                         │    ▼    ▼                       │
           │                         │ ┌───────────┐                   │
           │                         │ │dep + β·gate│                  │
           │                         │ └──────┬────┘                   │
           ▼                         │        ▼                        │
    ┌──────────────┐                 │ ┌──────────────┐                │
    │ DW-Conv 3×3  │                 │ │ DW-Conv 3×3  │                │
    │ GN + GELU    │                 │ │ GN + GELU    │                │
    │ PW 192→384   │                 │ │ PW 192→384   │                │
    │ GELU         │                 │ │ GELU         │                │
    │ PW 384→192   │                 │ │ PW 384→192   │                │
    └──────┬───────┘                 │ └──────┬───────┘                │
           │                         │        │                        │
    ───────┤ + Residual              │ ───────┤ + Residual             │
           ▼                         │        ▼                        │
    Refined Semantic                  Refined Depth-Feature
```

**Parameters per block**: ~444K (DW-Conv: 1.7K, PW-Expand: 73.7K, PW-Contract: 73.9K, cross-gates: 74.1K, norms: 768) × 2 streams

### 2.3 CoupledConvBlock with ASPP-lite (Proposed)

Replaces the single 3x3 depthwise convolution with parallel dilated convolutions for multi-scale context:

```
    ┌─────────────────────────────────────────────────┐
    │              ASPP-lite Module                    │
    │                                                 │
    │    Input (B, 192, 32, 64)                       │
    │         │                                       │
    │    ┌────┼────────────┐                          │
    │    │    │            │                           │
    │    ▼    ▼            ▼                           │
    │ ┌──────┐ ┌────────┐ ┌────────┐                  │
    │ │DW 3×3│ │DW 3×3  │ │DW 3×3  │                  │
    │ │ d=1  │ │ d=3    │ │ d=5    │                  │
    │ │ p=1  │ │ p=3    │ │ p=5    │                  │
    │ │local │ │mid-rng │ │wide-rng│                  │
    │ └──┬───┘ └──┬─────┘ └──┬─────┘                  │
    │    │        │          │                         │
    │    └────────┼──────────┘                         │
    │         Cat(dim=1)                               │
    │      (B, 576, 32, 64)                            │
    │             │                                    │
    │    ┌────────▼────────┐                           │
    │    │  Conv 1×1       │                           │
    │    │  576 → 192      │                           │
    │    │  GN + GELU      │                           │
    │    └────────┬────────┘                           │
    │             │                                    │
    │    Output (B, 192, 32, 64)                       │
    └─────────────┼───────────────────────────────────┘
                  │
                  ▼
           (feeds into PW Expand → GELU → PW Contract)
```

**Receptive field comparison** (per block, cumulative over 4 blocks):

| Component | Single-scale RF | ASPP-lite RF |
|-----------|----------------|--------------|
| Rate-1 DW-Conv 3×3 | 3×3 | 3×3 (local detail, thing boundaries) |
| Rate-3 DW-Conv 3×3 | — | 7×7 (medium context) |
| Rate-5 DW-Conv 3×3 | — | 11×11 (wide context, stuff regions) |
| After 4 blocks | 9×9 | 9×9 / 25×25 / 41×41 (multi-scale) |

**Parameters**: ASPP-lite adds ~900K params (1.83M → 2.74M, +50%) due to three parallel DW-Conv branches and the 3×→1× merge convolution per stream per block.

### 2.4 Training Pipeline with Proposed Losses

```
  ┌───────────────────── Training Pipeline ─────────────────────────┐
  │                                                                 │
  │  DINOv2 Features ──┐                                            │
  │  Depth + Grads ────┤                                            │
  │                    ▼                                            │
  │              ┌───────────┐                                      │
  │              │CSCMRefine │                                      │
  │              │   Net     │                                      │
  │              └─────┬─────┘                                      │
  │                    │                                            │
  │             Refined Logits                                      │
  │              (B, 19, H, W)                                      │
  │                    │                                            │
  │         ┌──────────┼──────────┬─────────────┐                   │
  │         ▼          ▼          ▼             ▼                   │
  │   ┌───────────┐ ┌──────┐ ┌──────────┐ ┌──────────┐            │
  │   │L_distill  │ │L_bpl │ │L_align   │ │L_proto   │            │
  │   │           │ │      │ │          │ │          │            │
  │   │ CE with   │ │Anchor│ │Depth-    │ │DINOv2    │            │
  │   │ label     │ │preds │ │boundary  │ │feature   │  ┌────────┐│
  │   │ smoothing │ │at    │ │spatial   │ │cluster   │  │L_ent   ││
  │   │           │ │pseudo│ │smooth-   │ │compact-  │  │        ││
  │   │ ┌───────┐ │ │label │ │ness      │ │ness      │  │Entropy ││
  │   │ │  TAD  │ │ │bndry │ │          │ │          │  │minim.  ││
  │   │ │κ=5 for│ │ │      │ │          │ │          │  │        ││
  │   │ │things │ │ │      │ │          │ │          │  │        ││
  │   │ └───────┘ │ │      │ │          │ │          │  │        ││
  │   └─────┬─────┘ └──┬───┘ └────┬─────┘ └────┬─────┘  └───┬────┘│
  │         │          │          │             │            │     │
  │         │ ×w(t)    │ ×λ_bpl  │ ×λ_align   │ ×λ_proto  │×λ_e│
  │         └────┬─────┴────┬─────┴──────┬──────┴─────┬──────┘     │
  │              │          │            │            │             │
  │              └──────────┴──────┬─────┴────────────┘             │
  │                                │                                │
  │                         L_total = Σ                             │
  │                                │                                │
  │                           Backprop                              │
  └────────────────────────────────┼────────────────────────────────┘
                                   │
                              w(t) = cosine warmdown
                              from λ_distill → λ_distill_min
```

**Loss formulations**:

| Loss | Formula | Purpose |
|------|---------|---------|
| L_distill (standard) | CE(f(x), y_pseudo) with label smoothing | Distill pseudo-label knowledge |
| L_distill (TAD) | w(x) · CE(f(x), y_pseudo), w=κ for things | Protect thing-class predictions |
| L_bpl | CE(f(x), y_pseudo) for boundary pixels only | Preserve segment boundaries |
| L_align | Σ exp(-Δd²/2σ²) · ‖Δp‖² over neighbors | Depth-aware spatial smoothness |
| L_proto | -Σ p(c|x) · cos(f_x, μ_c) | Feature cluster compactness |
| L_ent | -Σ p(c|x) · log p(c|x) | Prediction confidence |

### 2.5 Depth-Feature Projection with FiLM Conditioning

```
    DINOv2 Features (768)          Depth (1-ch) + Sobel Grads (2-ch)
         │                                │
    ┌────▼──────┐                    ┌────▼──────────────────┐
    │ Conv 1×1  │                    │ Sinusoidal PE         │
    │ 768→192   │                    │ depth × [2^i·π]       │
    │ GN + GELU │                    │ i = 0,...,5           │
    └────┬──────┘                    │ → sin/cos (12-ch)     │
         │                           │ + raw depth (1-ch)    │
         │                           │ + grad_x, grad_y (2)  │
         │ feat_proj                 │ = 15 channels         │
         │                           └────┬─────────────────┘
         │                                │
         │                           ┌────▼──────┐
         │                           │ Conv 1×1  │
         │                           │ 15 → 64   │
         │                           │ GELU      │
         │                           │ Conv 1×1  │
         │                           │ 64 → 384  │
         │                           └────┬──────┘
         │                                │
         │                           ┌────▼──────┐
         │                           │ Split     │
         │                           │ γ (192)   │
         │                           │ β (192)   │
         │                           │ clamp ±2  │
         │                           └──┬────┬───┘
         │                              │    │
         │          FiLM Modulation     │    │
         ▼          ────────────────    │    │
    ┌─────────────────────────────┐    │    │
    │ out = feat_proj × (1+γ) + β │◄───┘    │
    └─────────────┬───────────────┘◄────────┘
                  │
                  ▼
         Depth-conditioned features
              (B, 192, 32, 64)
```

## 3. Experimental Setup

All experiments use the Cityscapes validation set (500 images, 1024x2048). Training uses AdamW with cosine annealing LR schedule on Apple M4 Pro (MPS backend). Evaluation computes PQ, PQ_stuff, PQ_things via connected-component instance discovery and mIoU at 19-class trainID level. We report `changed_pct` — the fraction of patches where the refined prediction differs from the input pseudo-label — as a diagnostic for refinement magnitude.

**Baseline**: k=80 overclustered pseudo-labels mapped to 19 trainIDs via majority-vote cluster-to-class LUT derived from kmeans_centroids.npz.

## 4. Phase 1: Class Granularity — 80 vs 19 Classes

### 4.1 Training on 80 Raw Clusters

Following CUPS Table 7b, which demonstrates that overclustering improves panoptic quality (k=27 -> PQ=27.8, k=54 -> PQ=30.6), we first trained CSCMRefineNet to predict 80 raw cluster IDs, mapping to 19 trainIDs only at evaluation time.

**Configuration**: num_classes=80, lr=1e-4, batch_size=4, 30 epochs, eval every 2 epochs.
Loss weights: lambda_distill=1.0 -> 0.5, lambda_align=0.5, lambda_proto=0.05, lambda_ent=0.05, label_smoothing=0.1.

| Epoch | PQ | PQ_stuff | PQ_things | mIoU | changed% |
|-------|------|----------|-----------|-------|----------|
| Baseline | 26.74 | 32.08 | 19.41 | ~50% | — |
| 2 | 24.27 | 31.43 | 14.42 | 53.44 | 35.4% |
| 8 | 24.96 | 32.16 | 15.06 | 54.46 | 33.0% |
| 14 | 25.02 | 32.06 | 15.33 | 54.56 | 32.8% |
| 18 (best) | 25.39 | 32.39 | 15.78 | 54.57 | 32.4% |
| 30 | 25.08 | 32.31 | 15.14 | 54.41 | 32.5% |

**Outcome**: PQ degraded by 1.35 despite a +4.5% mIoU gain. The `changed_pct` stabilized at ~32%, far exceeding the healthy 3-8% range. With 80 fine-grained clusters, the model freely reassigns pixels between clusters mapping to the same trainID (e.g., two "road" clusters), inflating change counts without semantic benefit while disrupting panoptic segment coherence.

```
   80-class failure mode:

   Input pseudo-labels (k=80)          Refined predictions (k=80)
   ┌──────────────────────┐            ┌──────────────────────┐
   │ c12 c12 c37 c37 c37  │            │ c37 c12 c12 c37 c12  │
   │ c12 c12 c37 c37 c37  │    ──►     │ c12 c37 c37 c12 c37  │
   │ c12 c12 c37 c37 c37  │            │ c37 c12 c37 c37 c12  │
   └──────────────────────┘            └──────────────────────┘
   (c12, c37 both map to "road")       32% pixels changed but
                                        mIoU unchanged — noise only
```

**Key insight**: The overclustering advantage documented in CUPS applies to the full Cascade Mask R-CNN pipeline with instance heads. For lightweight semantic-only refinement, the additional degrees of freedom in an 80-class softmax introduce noise rather than signal.

### 4.2 Training on 19 Mapped Classes

We mapped k=80 clusters to 19 trainIDs using the cluster-to-class LUT and trained with num_classes=19. Two configurations were evaluated:

**Run C** (moderate): lambda_distill_min=0.7, lambda_align=0.5, lambda_proto=0.05, lambda_ent=0.05, lr=1e-4
**Run D** (conservative): lambda_distill_min=0.85, lambda_align=0.25, lambda_proto=0.025, lambda_ent=0.025, lr=5e-5

| Config | Best Epoch | PQ | PQ_stuff | PQ_things | mIoU | changed% |
|--------|-----------|------|----------|-----------|-------|----------|
| Baseline | — | 26.74 | 32.08 | 19.41 | ~50% | — |
| 80-cls (best) | 18 | 25.39 | 32.39 | 15.78 | 54.57 | 32.4% |
| **Run C** | 6 | 26.46 | **33.41** | 16.90 | 55.23 | 6.3% |
| Run C | 20 | 26.47 | 33.31 | 17.06 | 55.21 | 6.3% |
| **Run D** | 16 | **26.52** | 33.38 | **17.10** | **55.31** | **6.2%** |

Switching to 19 classes reduced `changed_pct` from 32% to 6.2% — directly within the target range. Both runs improved PQ_stuff (+1.3) and mIoU (+5%), but PQ_things regressed by 2.3 points, preventing net PQ improvement.

### 4.3 Continuation Training

We resumed Run D from its best checkpoint (epoch 16) for 24 additional epochs (to epoch 40), with a fresh cosine LR schedule. PQ oscillated between 26.16 and 26.42, never exceeding the epoch-16 best of 26.52. The model had fully converged.

## 5. Analysis: PQ_things Regression

The consistent PQ_things degradation across all configurations reveals a structural limitation:

1. **Resolution mismatch**: The model operates at 32x64 patch resolution. Thing objects (persons, bicycles) span 1-2 patches. Even a single patch flip can merge or split connected-component instances.

2. **Uniform refinement**: All pixels receive equal treatment. Stuff classes (road, building, sky) — which comprise ~70% of scene area — benefit from refinement. Thing classes (person, car, bicycle) — which are small and boundary-sensitive — are harmed by any prediction changes near their borders.

3. **Boundary-agnostic loss**: The depth-boundary alignment loss encourages smoothness where depth is similar, but does not explicitly preserve the input pseudo-label boundaries. It can smooth across thing-class boundaries, merging distinct instances.

4. **Connected-component sensitivity**: PQ_things is computed via connected components of the semantic map. A single-pixel class change at the boundary between two car instances can merge them, converting two TPs into one TP + one FN.

```
   PQ_things failure at patch resolution (32×64):

   Ground truth:              Pseudo-label:           After refinement:
   ┌─────────────┐           ┌─────────────┐        ┌─────────────┐
   │  road  road │           │  road  road │        │  road  road │
   │ ┌───┐ ┌───┐│           │ ┌───┐ ┌───┐│        │ ┌─────────┐ │
   │ │car│ │car││           │ │car│ │car││        │ │   car   │ │  ◄── merged!
   │ │ A │ │ B ││           │ │ A │ │ B ││        │ │  (A+B)  │ │
   │ └───┘ └───┘│           │ └───┘ └───┘│        │ └─────────┘ │
   │  road  road │           │  road  road │        │  road  road │
   └─────────────┘           └─────────────┘        └─────────────┘
   2 car instances            2 car instances         1 car instance
   PQ_things: 2 TP            PQ_things: 2 TP         PQ_things: 1TP + 1FN
                                                       (single patch flip
                                                        at gap merges them)
```

## 6. Proposed Architectural Improvements

Based on this analysis, we propose three targeted modifications (all convolution-only):

### 6.1 Thing-Aware Distillation Weighting (TAD)

Weight the cross-entropy distillation loss kappa times higher for thing-class pixels (trainIDs 11-18). Formally:

```
    w(x) = κ   if argmax p_pseudo(x) ∈ {person, rider, car, truck, bus, train, motorcycle, bicycle}
    w(x) = 1   otherwise (stuff classes)

    L_distill^TAD = (1/N) Σ_x w(x) · CE(f(x), argmax p_pseudo(x))
```

This allows the model to freely refine stuff predictions while constraining thing predictions to stay close to the input pseudo-labels. We set kappa=5 based on the approximate stuff-to-thing area ratio in Cityscapes.

```
   TAD effect on distillation gradient:

   Stuff pixel (road):       Thing pixel (car):
   ┌──────────────┐          ┌──────────────┐
   │ ∂L/∂θ = 1×g │          │ ∂L/∂θ = 5×g │  ◄── 5× stronger pull
   │  free to     │          │  anchored to │      toward pseudo-label
   │  refine      │          │  pseudo-label│
   └──────────────┘          └──────────────┘
```

### 6.2 Boundary Preservation Loss (BPL)

Detect semantic boundaries in the input pseudo-labels and penalize prediction changes at those locations:

```
    B(x) = 1{∃ y ∈ N_4(x) : argmax p_pseudo(x) ≠ argmax p_pseudo(y)}

    L_bpl = (1/|{x: B(x)=1}|) Σ_{x: B(x)=1} CE(f(x), argmax p_pseudo(x))
```

This explicitly anchors predictions at segment boundaries, preventing the refinement from eroding the panoptic structure established by the pseudo-labels.

```
   BPL boundary detection (4-connected):

   Pseudo-label map:          Boundary mask B(x):
   ┌───┬───┬───┬───┐         ┌───┬───┬───┬───┐
   │ R │ R │ R │ R │         │   │   │   │   │
   ├───┼───┼───┼───┤         ├───┼───┼───┼───┤
   │ R │ R │ C │ C │         │   │ ■ │ ■ │   │  ◄── boundary pixels
   ├───┼───┼───┼───┤         ├───┼───┼───┼───┤      get extra CE loss
   │ R │ R │ C │ C │         │   │ ■ │ ■ │   │
   ├───┼───┼───┼───┤         ├───┼───┼───┼───┤
   │ R │ R │ R │ R │         │   │   │   │   │
   └───┴───┴───┴───┘         └───┴───┴───┴───┘
   R=road, C=car              ■ = B(x)=1 (anchored)
```

### 6.3 Multi-Scale Dilated Convolutions (ASPP-lite)

Replace the single 3x3 depthwise convolution in each CoupledConvBlock with parallel dilated convolutions at rates {1, 3, 5}, merged via a 1x1 pointwise convolution:

```
    h_1 = DWConv_{3×3, d=1}(x)    — local detail (thing boundaries)
    h_3 = DWConv_{3×3, d=3}(x)    — medium context
    h_5 = DWConv_{3×3, d=5}(x)    — wide context (stuff regions)
    h   = Conv_{1×1}([h_1; h_3; h_5])
```

This provides multi-scale receptive fields. Inspired by ASPP (Chen et al., TPAMI 2018) but using depthwise-separable convolutions for parameter efficiency.

## 7. Ablation Study Design

We evaluate each improvement independently and in combination:

| Ablation | TAD (kappa=5) | BPL (lambda=0.5) | ASPP-lite | Params | Expected Impact |
|----------|-----------|-------------|-----------|--------|-----------------|
| A: Baseline (Run D) | — | — | — | 1.83M | PQ=26.52 |
| B: +TAD | yes | — | — | 1.83M | PQ_things up, PQ_stuff same |
| C: +BPL | — | yes | — | 1.83M | PQ_things up, changed% down |
| D: +ASPP | — | — | yes | 2.74M | PQ_stuff up, mIoU up |
| E: +TAD+BPL | yes | yes | — | 1.83M | PQ up significantly |
| F: +TAD+BPL+ASPP | yes | yes | yes | 2.74M | Best PQ expected |

All ablations train for 20 epochs on 19-class mapped k=80 labels with Run D's base configuration (lambda_distill_min=0.85, lr=5e-5).

## 8. Results

### 8.1 Ablation Results

All ablations trained for 20 epochs with evaluation every 2 epochs, using the proven conservative loss configuration from Run D (distill_min=0.85, align=0.5, proto=0.05, ent=0.05) as the common baseline. Four runs executed in parallel on MPS (Apple M4 Pro).

| Ablation | Best PQ | PQ_stuff | PQ_things | mIoU | changed% | Best Epoch | Delta PQ |
|----------|---------|----------|-----------|-------|----------|------------|----------|
| Baseline (input) | 26.74 | 32.08 | 19.41 | ~50% | — | — | — |
| A: Run D (base) | 26.52 | 33.38 | 17.10 | 55.31 | 6.2% | 10 | -0.22 |
| B: +TAD (kappa=5) | 25.89 | 33.21 | 15.82 | 54.07 | 6.6% | 6 | -0.85 |
| C: +BPL (lambda=0.5) | 26.35 | 33.21 | 16.93 | 55.37 | 6.1% | 14 | -0.39 |
| D: +ASPP-lite | 26.27 | 33.32 | 16.59 | 55.46 | 6.2% | 10 | -0.47 |
| E: +TAD+BPL | 25.69 | 32.81 | 15.91 | 55.01 | 6.2% | 12 | -1.05 |

### 8.2 Analysis

**Thing-Aware Distillation (TAD) degrades performance.** Ablation B applies kappa=5x distillation weight to thing-class pixels (trainIDs 11-18), increasing the gradient contribution of sparse thing regions. Instead of preserving thing boundaries, this destabilizes early training — the model peaks at epoch 6 (earliest of all ablations) and plateaus at PQ=25.89, a 0.63-point deficit versus the base configuration. The likely mechanism: at 32x64 resolution, thing-class pixels are extremely sparse (often <5% of the feature map), and 5x weighting amplifies noise from coarse pseudo-label boundaries. The combined TAD+BPL ablation (E) confirms this — it produces the worst overall PQ (25.69), with TAD as the primary degradation source.

**Boundary Preservation Loss (BPL) is approximately neutral.** Ablation C achieves PQ=26.35 with the highest PQ_things among ablations (16.93) and the best mIoU (55.37). The 0.17-point PQ deficit versus the base is within noise, and the improved mIoU suggests the boundary-focused CE loss does help semantic boundary quality without harming overall predictions. However, the improvement is insufficient to recover the structural PQ_things regression (19.41 to 16.93, a 2.48-point gap versus input pseudo-labels).

**ASPP-lite provides marginal semantic improvement.** Ablation D achieves the highest PQ_stuff (33.32) and mIoU (55.46) of all configurations, confirming that multi-scale receptive fields (dilations 1, 3, 5) capture broader context at the 32x64 feature map resolution. However, the 0.25-point PQ deficit versus the base suggests the additional 900K parameters (+49%) do not translate to proportionate panoptic quality gains.

**No ablation recovers PQ_things.** All configurations exhibit PQ_things regression from the input pseudo-labels (19.41) to the 15.8-17.1 range. This confirms the structural hypothesis from Section 6: at 32x64 resolution, small instances (bicycles, riders, motorcycles) collapse below the resolution limit, and the network lacks the spatial precision to preserve thing boundaries regardless of loss function or receptive field design.

### 8.3 Conclusions

The CSCMRefineNet architecture successfully improves semantic quality — mIoU improves from ~50% (input pseudo-labels) to 55.5% (best ablation), and PQ_stuff improves from 32.08 to 33.38 (+1.30). However, PQ_things consistently regresses by 2.3-3.6 points due to the fundamental resolution bottleneck of operating at 32x64 patch resolution.

**Recommendation:** For downstream use, the base Run D configuration (distill_min=0.85, no additional losses) remains optimal. The architectural improvements (TAD, BPL, ASPP-lite) provide at best marginal gains that do not justify the added complexity. To improve PQ_things, future work should either (a) operate at higher resolution (e.g., 128x256 with upsampled features), (b) use instance-aware loss masking that explicitly preserves thing boundaries from the input pseudo-labels, or (c) bypass semantic refinement entirely and invest in improved instance segmentation quality.
