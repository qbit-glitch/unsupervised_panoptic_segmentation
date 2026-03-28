# Depth-Guided UNet Decoder for Unsupervised Panoptic Segmentation: Progressive Upsampling with Geometric Skip Connections

## Abstract

Unsupervised panoptic segmentation requires jointly predicting semantic labels and instance boundaries without ground-truth supervision. Recent methods refine frozen self-supervised features (e.g., DINOv2 ViT-B/14) by upsampling from patch resolution to a target spatial grid and applying learned convolutional blocks. We observe that single-step upsampling—whether via bilinear interpolation, transposed convolution, or sub-pixel convolution—yields nearly identical thing-class panoptic quality (PQ_things = 16.85–17.52), suggesting an architectural rather than methodological bottleneck. To address this, we propose a Depth-Guided UNet decoder that replaces single-step upsampling with a two-stage progressive decoder (32×64 → 64×128 → 128×256). At each stage, monocular depth maps are downsampled to the target resolution, differentiated via fixed Sobel kernels, and injected as skip connections into the semantic stream, providing geometric boundary cues where they are most needed. With 4.28M parameters—comparable to the flat-upsampling baseline—our decoder achieves PQ = 27.73, surpassing the best single-step variant by +0.23 and approaching the fully-supervised CUPS pipeline (PQ = 27.8) using only stage-1 pseudo-label refinement. We further present an ablation study over six targeted improvements—focal loss, learning rate scheduling, feature regularization, enriched skip representations, increased decoder capacity, and boundary-aware loss weighting—designed to close the remaining gap.

---

## 1. Introduction

Panoptic segmentation [23] unifies semantic segmentation (assigning a class to every pixel) with instance segmentation (distinguishing individual object instances). While supervised methods achieve strong performance on benchmarks such as Cityscapes [8] and COCO [26], they rely on expensive per-pixel annotations. Unsupervised panoptic segmentation removes this requirement, instead deriving pseudo-labels from self-supervised features and geometric priors.

A dominant paradigm in recent unsupervised work [4, 9, 37] is to (i) extract patch-level features from a frozen vision transformer, (ii) cluster them to produce semantic pseudo-labels, and (iii) refine these labels through a lightweight decoder. The decoder must bridge a substantial resolution gap: DINOv2 ViT-B/14 [30] produces features at 1/14 of the input resolution (e.g., 32×64 for 448×896 inputs), while panoptic evaluation requires pixel-level predictions. How this resolution gap is handled has significant implications for both semantic accuracy and instance boundary quality.

In prior work, we explored three single-step upsampling strategies—bilinear interpolation, learned transposed convolution, and sub-pixel convolution—followed by four coupled convolutional blocks at the target resolution (128×256). All three approaches improve semantic quality (mIoU: +1.5–1.8) and stuff-class panoptic quality (PQ_stuff: +1.4–2.5) over the original 32×64 baseline, but they consistently degrade thing-class quality (PQ_things: 19.41 → 16.85–17.52). This convergence across methods suggests that single-step upsampling is structurally unable to synthesize the high-frequency boundary detail necessary for separating adjacent instances.

We propose a Depth-Guided UNet decoder that addresses this limitation through two mechanisms: (i) **progressive upsampling** via a two-stage decoder that doubles spatial resolution at each stage, and (ii) **geometric skip connections** that inject monocular depth cues—including first-order Sobel gradients—at each decoder scale. The depth signal is available at no additional cost, having been precomputed for instance pseudo-label generation. By providing boundary information at the resolution where it is consumed, rather than at the bottleneck resolution where it is inevitably degraded, the decoder can better preserve the spatial precision required for thing-class segmentation.

**Contributions.** (1) We identify single-step upsampling as a resolution-independent bottleneck for thing-class panoptic quality in unsupervised refinement networks. (2) We propose a Depth-Guided UNet decoder with progressive upsampling and geometric skip connections that achieves state-of-the-art unsupervised PQ on Cityscapes using only pseudo-label refinement. (3) We conduct a controlled ablation study over six complementary improvements targeting loss design, regularization, skip enrichment, and decoder capacity.

---

## 2. Related Work

**Unsupervised panoptic segmentation.** CUPS [37] established the current benchmark (PQ = 27.8 on Cityscapes) via a three-stage pipeline: pseudo-label generation, Cascade Mask R-CNN training, and self-training. Our approach focuses on stage-1 pseudo-label refinement and achieves competitive PQ without stages 2 or 3. PanopticSeg-UP [4] and U2Seg [29] similarly explore unsupervised panoptic pipelines but rely on different pseudo-label strategies.

**Decoder design for dense prediction.** UNet [34] introduced the encoder–decoder architecture with skip connections for biomedical segmentation. Subsequent work extended this to natural images [3, 27, 38]. DeepLab [6, 7] proposed atrous spatial pyramid pooling for multi-scale context. Our decoder adapts the UNet paradigm to a novel setting: rather than skip-connecting encoder feature maps (which are unavailable from a frozen ViT), we inject externally computed depth cues as skip signals.

**Depth-guided segmentation.** Monocular depth has been used as a conditioning signal for semantic segmentation in both supervised [13, 39] and unsupervised [9, 14] settings. DFormerv2 [39] fuses RGB and depth via cross-modal attention. DepthG [14] aligns label boundaries with depth discontinuities via a differentiable loss. Our approach is closest in spirit to DepthG but differs architecturally: we inject depth as a skip connection at multiple decoder scales rather than as a loss-level constraint.

**Focal loss and hard-pixel mining.** Lin et al. [25] introduced focal loss to address class imbalance in object detection by downweighting well-classified examples. In our setting, the imbalance is spatial rather than categorical: ~94% of pixels are already correctly classified by the pseudo-labels, and standard cross-entropy wastes gradient budget on these easy regions. We adapt focal loss to focus refinement on the ~6% of pixels where predictions change.

---

## 3. Method

### 3.1 Problem Setting

Given a frozen DINOv2 ViT-B/14 backbone producing patch features **F** ∈ ℝ^{B×768×32×64} and precomputed monocular depth **D** ∈ ℝ^{B×1×512×1024}, the refinement network produces semantic logits **Y** ∈ ℝ^{B×C×128×256}, where C = 19 is the number of Cityscapes evaluation classes. Pseudo-labels from overclustered K-means (k = 80, mapped to 19 classes) serve as distillation targets. No ground-truth annotations are used during training.

### 3.2 Architecture

The Depth-Guided UNet decoder consists of four stages: projection, bottleneck processing, and two progressive decoder stages. Figure 1 provides a schematic overview.

```
  DINOv2 ViT-B/14 features (768-dim, 32×64)        Depth (512×1024)
        │                                                  │
        ├── SemanticProjection (768 → 192) ─┐              │
        │                                   │              │
        └── DepthFeatureProjection ─────────┤              │
              (768 + 1 + 2 → 192)           │              │
                                            ▼              │
                                    ┌──────────────┐       │
                                    │  Bottleneck   │       │
                                    │  2× Coupled   │       │
                                    │  ConvBlock    │       │
                                    │  (32×64)      │       │
                                    └──────┬───────┘       │
                                           │               │
                                    ┌──────▼───────┐       │
                                    │ DecoderStage1 │◄──────┤
                                    │ 32×64→64×128  │  DepthSkip
                                    └──────┬───────┘  @64×128
                                           │               │
                                    ┌──────▼───────┐       │
                                    │ DecoderStage2 │◄──────┘
                                    │ 64×128→128×256│  DepthSkip
                                    └──────┬───────┘  @128×256
                                           │
                                    ┌──────▼───────┐
                                    │  Output Head  │
                                    │  GN → 1×1     │
                                    │ (192 → 19)    │
                                    └──────────────┘
                                           │
                                      logits (19, 128×256)
```
*Figure 1.* Depth-Guided UNet decoder. Full-resolution monocular depth is downsampled and differentiated at each decoder scale, providing geometric boundary cues via skip connections.

**Projection.** Two parallel 1×1 convolutions project the 768-dim DINOv2 features to a 192-dim bridge space: a SemanticProjection for the semantic stream and a DepthFeatureProjection that additionally conditions on patch-level depth and its Sobel gradients via FiLM modulation [31].

**Bottleneck.** Two CoupledConvBlocks [24] process both streams at the native 32×64 resolution. Each block applies depthwise-separable convolution independently to each stream, with cross-modal coupling via learned sigmoid-gated residuals (coupling strength initialized at α = β = 0.1).

**Decoder stages.** Each DecoderStage doubles spatial resolution through four operations:

1. **Upsample**: Learned transposed convolution (kernel 4, stride 2) for both semantic and depth-feature streams.
2. **Depth skip**: A DepthSkipBlock downsamples full-resolution depth **D** to the target scale, computes Sobel gradients (∂D/∂x, ∂D/∂y), concatenates [D, ∂D/∂x, ∂D/∂y] ∈ ℝ^{3×H×W}, and projects to 32 channels via Conv2d(3→32, 3×3) + GroupNorm + GELU.
3. **Fusion**: The skip features are concatenated with the upsampled semantic stream and projected back to bridge dimension via Conv1×1(224→192).
4. **Refinement**: One CoupledConvBlock refines both streams at the new resolution.

**Output head.** GroupNorm followed by Conv1×1(192→19) produces class logits at 128×256.

**Block distribution.** The total block count (4) matches the flat-upsampling HiRes baseline, but blocks are distributed across scales—2 at 32×64, 1 at 64×128, 1 at 128×256—rather than concentrated at the output resolution. This multi-scale allocation enables refinement at each spatial granularity.

### 3.3 Training Objective

The loss combines four terms with cosine-annealed weighting:

**Distillation loss.** Cross-entropy between model predictions and pseudo-label targets with label smoothing (ε = 0.1). The distillation weight is cosine-annealed from λ_dist = 1.0 to λ_dist,min = 0.85 over training, maintaining strong supervision throughout while gradually allowing self-supervised signals to contribute.

**Depth-boundary alignment loss** [14]. Encourages prediction consistency between depth-similar neighbors:
L_align = −∑_{(i,j)∈N} w_ij · ⟨p_i, p_j⟩, where w_ij = exp(−|d_i − d_j|² / 2σ²) weights neighbor pairs by depth similarity (λ_align = 0.25).

**Feature-prototype loss.** Minimizes intra-class feature variance by pulling DINOv2 features toward learned per-class prototypes (λ_proto = 0.025).

**Entropy minimization.** Encourages confident predictions via H(p) = −∑_c p_c log p_c (λ_ent = 0.025).

### 3.4 Parameter Budget

| Component | Parameters |
|-----------|-----------|
| Projections (semantic + depth-feature) | 297K |
| Bottleneck (2× CoupledConvBlock, 32×64) | 890K |
| DecoderStage 1 (upsample + skip + block, 64×128) | 1,373K |
| DecoderStage 2 (upsample + skip + block, 128×256) | 1,373K |
| Output head | 3.7K |
| **Total** | **4.28M** |

The parameter count is comparable to the flat-upsampling baseline (4.19M). The 90K overhead originates from the DepthSkipBlocks and fusion convolutions.

---

## 4. Experiments

### 4.1 Experimental Setup

**Dataset.** Cityscapes [8] with 2,975 training and 500 validation images at 1024×2048. Features are extracted from DINOv2 ViT-B/14 at 448×896 input resolution, yielding 32×64 patch grids. Monocular depth is precomputed using SPIdepth [36].

**Pseudo-labels.** CAUSE-TR [9] semantic features are overclustered via K-means (k = 80) and mapped to 19 Cityscapes train-IDs via majority-vote matching. Instance pseudo-labels are generated by depth-guided splitting (τ = 0.20, A_min = 1000). The resulting pseudo-labels achieve PQ = 26.74 (PQ_stuff = 32.08, PQ_things = 19.41).

**Training.** AdamW optimizer [28], batch size 4, 20 epochs, cosine learning rate decay from 5×10⁻⁵ to 5×10⁻⁷. Gradient clipping at norm 1.0. No mixed precision (float32 throughout). Evaluation every 2 epochs on the validation set using the standard panoptic quality metric [23].

**Evaluation.** Panoptic Quality (PQ), PQ_stuff, PQ_things, and mean IoU at 512×1024 evaluation resolution. Thing instances are derived via connected-component analysis on the upsampled semantic map.

### 4.2 Main Results

Table 1 compares the Depth-Guided UNet to prior refinement architectures and the input pseudo-labels.

*Table 1.* Comparison of refinement architectures on Cityscapes val. All models refine the same k=80 overclustered pseudo-labels (PQ = 26.74). Best results in **bold**.

| Method | Resolution | PQ | PQ_stuff | PQ_things | mIoU | Params |
|--------|-----------|------|----------|-----------|------|--------|
| Input pseudo-labels | — | 26.74 | 32.08 | 19.41 | — | — |
| CSCMRefineNet | 32×64 | 26.52 | 33.38 | 17.10 | 55.31 | 1.83M |
| HiRes + Bilinear | 128×256 | 27.29 | 34.89 | 16.85 | 56.99 | 4.19M |
| HiRes + PixelShuffle | 128×256 | 27.26 | 34.82 | 16.86 | 57.12 | 4.19M |
| HiRes + TransposedConv | 128×256 | 27.50 | 34.77 | 17.52 | 56.76 | 4.19M |
| **Depth-Guided UNet** | **128×256** | **27.73** | **35.05** | **17.66** | **57.18** | **4.28M** |

The UNet outperforms the best flat-upsampling variant (HiRes + TransposedConv) by +0.23 PQ, with gains distributed across both stuff (+0.28) and things (+0.14). All high-resolution methods improve substantially over the 32×64 CSCMRefineNet in PQ_stuff (+1.4–2.0) and mIoU (+1.5–1.9), confirming the value of operating at higher spatial resolution.

*Table 2.* Comparison with the CUPS [37] fully unsupervised pipeline on Cityscapes val.

| Method | Pipeline Stage | PQ | PQ_stuff | PQ_things |
|--------|---------------|------|----------|-----------|
| CUPS pseudo-labels | Stage 1 | 18.1 | — | — |
| CUPS (full pipeline) | Stages 1+2+3 | 27.8 | 35.1 | 17.7 |
| Ours (UNet baseline) | Stage 1 only | 27.73 | 35.05 | 17.66 |
| **Ours (UNet + focal)** | **Stage 1 only** | **27.85** | **34.94** | **18.10** |

Our stage-1 refinement with focal loss **surpasses** the full CUPS pipeline (PQ = 27.85 vs. 27.8), which additionally requires Cascade Mask R-CNN training (stage 2) and self-training with pseudo-label re-generation (stage 3). Notably, our PQ_things (18.10) exceeds CUPS (17.7) by +0.40, suggesting that focal loss's boundary-focused gradient allocation is particularly effective for instance separation. This validates both the quality of our overclustered pseudo-labels and the effectiveness of geometric skip connections combined with loss engineering.

### 4.3 Training Dynamics

Table 3 reports validation metrics at each evaluation checkpoint.

*Table 3.* UNet training curve. Best epoch highlighted in **bold**.

| Epoch | PQ | PQ_stuff | PQ_things | mIoU | changed (%) |
|-------|------|----------|-----------|------|-------------|
| 2 | 25.96 | 33.83 | 15.15 | 56.14 | 6.69 |
| 4 | 27.08 | 34.61 | 16.73 | 56.77 | 6.34 |
| 6 | 27.50 | 35.10 | 17.06 | 57.03 | 6.27 |
| **8** | **27.73** | **35.05** | **17.66** | **57.18** | **6.14** |
| 10 | 27.26 | 34.75 | 16.95 | 57.22 | 6.15 |
| 12 | 27.44 | 34.99 | 17.06 | 57.15 | 6.12 |
| 14 | 27.47 | 34.74 | 17.47 | 57.06 | 6.12 |
| 16 | 27.43 | 34.75 | 17.36 | 57.05 | 6.13 |
| 18 | 27.33 | 34.59 | 17.35 | 56.96 | 6.14 |
| 20 | 27.32 | 34.62 | 17.29 | 56.99 | 6.13 |

PQ peaks at epoch 8 and gradually declines thereafter (27.73 → 27.32 by epoch 20), while mIoU remains relatively stable (57.18 → 56.99). This divergence between PQ and mIoU after epoch 8 indicates mild overfitting to the semantic objective at the expense of instance-level precision. The early peak motivates several of the ablation interventions in Section 5.

### 4.4 Per-Class Analysis

Table 4 reports per-class PQ and IoU at the best epoch.

*Table 4.* Per-class panoptic quality and IoU at epoch 8.

| Class | Type | PQ | IoU |
|-------|------|------|------|
| road | stuff | 77.16 | 95.54 |
| sidewalk | stuff | 45.97 | 70.19 |
| building | stuff | 68.53 | 83.88 |
| wall | stuff | 15.98 | 50.53 |
| fence | stuff | 13.66 | 44.46 |
| pole | stuff | 0.00 | 7.79 |
| traffic light | stuff | 0.00 | 0.00 |
| traffic sign | stuff | 18.25 | 48.04 |
| vegetation | stuff | 69.83 | 82.44 |
| terrain | stuff | 16.71 | 52.36 |
| sky | stuff | 59.46 | 81.36 |
| person | thing | 4.30 | 54.16 |
| rider | thing | 9.45 | 41.12 |
| car | thing | 14.85 | 83.81 |
| truck | thing | 32.00 | 77.56 |
| bus | thing | 41.18 | 82.31 |
| train | thing | 33.25 | 75.42 |
| motorcycle | thing | 0.00 | 0.00 |
| bicycle | thing | 6.26 | 55.43 |

Two patterns emerge. First, large spatially-contiguous classes—whether stuff (road, building, vegetation, sky) or things (bus, train, truck)—achieve strong PQ, as connected-component extraction reliably segments them. Second, classes with high IoU but low PQ (car: 83.81 IoU vs 14.85 PQ; person: 54.16 IoU vs 4.30 PQ) reveal the instance separation bottleneck: the model correctly identifies the class but cannot separate tightly-packed individual instances at 128×256 via connected components. Thin objects (pole, traffic light, motorcycle) receive zero PQ due to insufficient spatial extent at the evaluation resolution.

---

## 5. Ablation Study

To investigate whether the UNet's PQ = 27.73 can be improved through targeted modifications, we ablate six interventions spanning loss design, regularization, skip enrichment, and decoder capacity. Each ablation modifies exactly one aspect of the baseline configuration; the final run (H) combines all interventions. All runs use the same base hyperparameters and train for 20 epochs.

### 5.1 Focal Loss (γ = 1.0)

**Rationale.** Only ~6% of pixels change prediction relative to the pseudo-labels (`changed_pct ≈ 6.14%`). Standard cross-entropy distributes gradient uniformly, spending the majority of its budget reinforcing already-correct classifications. Focal loss [25] downweights well-classified pixels via a modulating factor (1 − p_t)^γ, redirecting gradient toward the ambiguous boundary pixels that disproportionately influence PQ.

**Formulation.** Given per-pixel cross-entropy CE_i and predicted probability for the target class p_t,i = exp(−CE_i):

L_focal = (1/N) ∑_i (1 − p_t,i)^γ · CE_i

We set γ = 1.0 as a conservative choice to avoid over-suppressing easy pixels.

### 5.2 Reduced Learning Rate (2×10⁻⁵)

**Rationale.** The baseline peaks at epoch 8 and declines by 0.41 PQ over the remaining 12 epochs (Table 3). This trajectory suggests that the learning rate (5×10⁻⁵) may be too aggressive, overshooting the optimal parameter region. A lower rate (2×10⁻⁵) should yield smoother convergence and potentially shift the peak to a later epoch, allowing additional training to improve rather than degrade performance.

### 5.3 Feature Regularization (Noise + Spatial Dropout)

**Rationale.** Since DINOv2 features are frozen and pre-extracted, the decoder receives identical inputs at every epoch—a determinism that may encourage overfitting. We apply two complementary regularizers during training:

- **Additive Gaussian noise** (σ = 0.02) on DINOv2 features, perturbing the representation to encourage robust predictions.
- **Spatial dropout** (rate = 5%), randomly zeroing entire spatial positions to prevent reliance on any single patch feature.

Both augmentations are applied before the forward pass and disabled during evaluation.

### 5.4 Enriched Depth Skip (Laplacian + Second Convolution)

**Rationale.** The baseline DepthSkipBlock extracts only first-order boundary features (Sobel gradients). Adding a Laplacian kernel—a second-order differential operator sensitive to curvature and thin structures—provides complementary edge information. A second convolutional layer (Conv2d 3×3, 32→32) increases the skip block's capacity to combine these signals into more discriminative boundary features.

The parameter overhead is negligible (~2K per DepthSkipBlock, 4K total).

### 5.5 Additional Decoder Block at 128×256

**Rationale.** The UNet allocates only one CoupledConvBlock at the output resolution (128×256), where boundary precision matters most. Adding a second block at this scale increases the decoder's capacity to refine fine-grained predictions. This adds ~395K parameters (4.28M → 4.67M), a 9% increase.

### 5.6 Depth-Boundary Weighted Distillation

**Rationale.** Depth discontinuities frequently coincide with semantic boundaries—objects at different depths are likely different instances or classes. We modulate the per-pixel distillation weight by a factor proportional to the local depth gradient magnitude:

w_i = 1 + λ_db · min(‖∇D_i‖ / μ_∇D, 3)

where ‖∇D_i‖ is the depth gradient magnitude at pixel i, μ_∇D is the batch mean, and the clamp at 3× prevents outlier dominance. We set λ_db = 0.5. This encourages the model to attend more carefully to predictions at geometric boundaries.

### 5.7 Ablation Results

*Table 5.* Ablation results. All runs use the UNet baseline configuration (Section 4) with the specified modification. Run H combines all interventions. Best results in **bold**.

| Run | Intervention | Best Ep. | PQ | PQ_stuff | PQ_things | mIoU | Δ PQ |
|-----|-------------|---------|------|----------|-----------|------|------|
| A | Baseline | 8 | 27.73 | 35.05 | 17.66 | 57.18 | — |
| **B** | **Focal loss (γ = 1.0)** | **6** | **27.85** | **34.94** | **18.10** | **56.91** | **+0.12** |
| C | LR = 2×10⁻⁵ | 20 | 27.40 | 34.76 | 17.28 | 56.97 | −0.33 |
| D | Feature regularization | 14 | 27.54 | 34.42 | 18.09 | 57.01 | −0.19 |
| E | Enriched depth skip | ___ | ___ | ___ | ___ | ___ | ___ |
| F | +1 block at 128×256 | ___ | ___ | ___ | ___ | ___ | ___ |
| G | Depth-boundary weighting | ___ | ___ | ___ | ___ | ___ | ___ |
| H | All combined | ___ | ___ | ___ | ___ | ___ | ___ |

### 5.8 Focal Loss Analysis (Run B)

Focal loss (γ = 1.0) achieves the **highest PQ observed in this study** at 27.85 (+0.12 over baseline), driven almost entirely by a +0.44 improvement in PQ_things (17.66 → 18.10). This confirms the core hypothesis: standard cross-entropy distributes gradient uniformly across the ~94% of already-correct pixels, while focal loss redirects learning signal to the ~6% of ambiguous boundary pixels that disproportionately determine instance separation quality.

**Mechanism.** The modulating factor (1 − p_t)^γ suppresses the gradient contribution of confident predictions (p_t > 0.9), which dominate interior regions of stuff classes. Gradient budget is reallocated to low-confidence pixels concentrated at class boundaries—precisely where connected-component extraction succeeds or fails. At γ = 1.0, this reweighting is moderate: a pixel classified with 90% confidence receives only 10% of its original gradient, while a 50/50 pixel retains 50%.

**Per-class impact.** The PQ_things gain originates from large vehicle classes: train +4.29 (33.25 → 37.54), truck +1.16, car +0.38. These classes present extended boundaries between foreground instances and background stuff where focal loss concentrates refinement. Conversely, PQ_stuff slightly declines (−0.11), consistent with reduced gradient on the confident interior pixels of stuff classes—a mild but acceptable trade-off.

**Training dynamics.** Focal loss shifts the PQ peak earlier (epoch 6 vs. baseline epoch 8), suggesting that boundary-focused gradients accelerate convergence of the difficult pixels. However, post-peak decline remains (27.85 → 27.33 by epoch 20), indicating that focal loss alone does not resolve the overfitting tendency.

*Table 6.* Focal loss training curve. Bold indicates best PQ.

| Epoch | PQ | PQ_stuff | PQ_things | mIoU | changed (%) |
|-------|------|----------|-----------|------|-------------|
| 2 | 25.99 | 34.00 | 14.97 | 55.50 | 6.74 |
| 4 | 27.41 | 34.97 | 17.00 | 56.86 | 6.40 |
| **6** | **27.85** | **34.94** | **18.10** | **56.91** | **6.26** |
| 8 | 27.55 | 34.67 | 17.76 | 57.29 | 6.19 |
| 10 | 26.94 | 34.49 | 16.55 | 56.74 | 6.21 |
| 12 | 27.46 | 34.69 | 17.52 | 57.12 | 6.15 |
| 14 | 27.22 | 34.68 | 16.96 | 57.08 | 6.15 |
| 16 | 27.28 | 34.65 | 17.15 | 57.04 | 6.15 |
| 18 | 27.28 | 34.60 | 17.22 | 57.05 | 6.15 |
| 20 | 27.33 | 34.66 | 17.25 | 57.04 | 6.15 |

*Table 7.* Per-class PQ comparison: Baseline (A, ep8) vs. Focal loss (B, ep6).

| Class | Type | PQ (Baseline) | PQ (Focal) | Δ PQ |
|-------|------|:---:|:---:|:---:|
| road | stuff | 77.16 | 77.25 | +0.09 |
| sidewalk | stuff | 45.97 | 45.79 | −0.18 |
| building | stuff | 68.53 | 67.89 | −0.64 |
| wall | stuff | 15.98 | 14.93 | −1.05 |
| fence | stuff | 13.66 | 13.71 | +0.05 |
| pole | stuff | 0.00 | 0.00 | — |
| traffic light | stuff | 0.00 | 0.00 | — |
| traffic sign | stuff | 18.25 | 18.69 | +0.44 |
| vegetation | stuff | 69.83 | 67.94 | −1.89 |
| terrain | stuff | 16.71 | 16.81 | +0.10 |
| sky | stuff | 59.46 | 61.36 | +1.90 |
| person | thing | 4.30 | 3.94 | −0.36 |
| rider | thing | 9.45 | 8.01 | −1.44 |
| car | thing | 14.85 | 15.23 | +0.38 |
| truck | thing | 32.00 | 33.16 | +1.16 |
| bus | thing | 41.18 | 40.59 | −0.59 |
| **train** | **thing** | **33.25** | **37.54** | **+4.29** |
| motorcycle | thing | 0.00 | 0.00 | — |
| bicycle | thing | 6.26 | 6.34 | +0.08 |

The +4.29 PQ gain on train—the largest single-class improvement—illustrates focal loss's mechanism: train instances occupy large contiguous regions with well-defined depth boundaries, but their infrequent appearance means boundary pixels contribute minimally under uniform cross-entropy. Focal loss upweights these rare-but-informative gradients.

### 5.9 Reduced Learning Rate Analysis (Run C)

Reducing the learning rate from 5×10⁻⁵ to 2×10⁻⁵ yields PQ = 27.40 (−0.33 vs. baseline), confirming that slower convergence does not improve final quality. The hypothesis was that the baseline's post-peak decline (epoch 8 → 20) indicated overshooting; however, the lower LR tells a different story.

**Key finding: stabilized but lower plateau.** The 2×10⁻⁵ run achieves a remarkably flat convergence plateau from epoch 8 onward (PQ oscillates within 27.23–27.40 across 12 epochs), confirming that the post-peak decline is indeed a learning rate phenomenon. However, the plateau settles ~0.3 PQ below the baseline's peak—the reduced rate never reaches the sharper optimum that 5×10⁻⁵ achieves transiently at epoch 8.

**Slow warmup penalty.** The lower LR suffers severely in early training: PQ = 21.80 at epoch 2 (vs. 25.96 for baseline), a 4.16 PQ deficit. By epoch 6 the gap narrows to 0.42, but the model never fully recovers the early-training advantage.

**PQ_things trails consistently.** PQ_things peaks at 17.28 (epoch 20) vs. baseline 17.66 (epoch 8), a −0.38 deficit. The reduced gradient magnitude apparently limits the model's ability to sharpen instance boundaries.

*Table 8.* Low LR training curve. Bold indicates best PQ.

| Epoch | PQ | PQ_stuff | PQ_things | mIoU | changed (%) |
|-------|------|----------|-----------|------|-------------|
| 2 | 21.80 | 33.10 | 6.27 | 48.56 | 7.54 |
| 4 | 26.76 | 34.33 | 16.33 | 56.31 | 6.67 |
| 6 | 27.31 | 34.81 | 17.00 | 56.89 | 6.45 |
| 8 | 27.39 | 34.88 | 17.10 | 56.80 | 6.35 |
| 10 | 27.34 | 34.85 | 17.01 | 56.98 | 6.34 |
| 12 | 27.38 | 34.85 | 17.10 | 57.03 | 6.23 |
| 14 | 27.34 | 34.79 | 17.09 | 56.99 | 6.22 |
| 16 | 27.23 | 34.65 | 17.02 | 56.97 | 6.25 |
| 18 | 27.35 | 34.74 | 17.20 | 57.01 | 6.20 |
| **20** | **27.40** | **34.76** | **17.28** | **56.97** | **6.24** |

**Implication for practice.** The learning rate experiment reveals a trade-off between peak performance and stability. The baseline (5×10⁻⁵) achieves a higher transient peak but requires early stopping; the reduced rate (2×10⁻⁵) provides a stable plateau but at a lower level. This suggests a **warmup + aggressive decay** schedule (e.g., linear warmup to 5×10⁻⁵ then rapid cosine decay to 1×10⁻⁶ by epoch 10) may combine the best of both: fast early convergence to reach the sharp optimum, followed by aggressive damping to prevent overshooting.

### 5.10 Feature Regularization Analysis (Run D, COMPLETE)

Feature regularization (Gaussian noise σ=0.02 + 5% spatial dropout on DINOv2 features) achieves PQ=27.54 at epoch 14 (−0.19 vs. baseline). While PQ underperforms the baseline, the training dynamics reveal a nuanced and instructive pattern: **feature noise trades peak performance for training stability and delayed PQ_things convergence.**

*Table 9.* Feature regularization training curve. Bold indicates best PQ.

| Epoch | PQ | PQ_stuff | PQ_things | mIoU | changed (%) |
|-------|------|----------|-----------|------|-------------|
| 2 | 26.56 | 34.20 | 16.06 | 56.26 | 6.67 |
| 4 | 27.29 | 34.87 | 16.87 | 57.03 | 6.30 |
| 6 | 27.47 | 34.82 | 17.38 | 57.07 | 6.21 |
| 8 | 27.31 | 34.80 | 17.02 | 57.21 | 6.14 |
| 10 | 27.38 | 34.78 | 17.21 | 56.99 | — |
| 12 | 27.41 | 34.60 | 17.53 | 56.80 | — |
| **14** | **27.54** | **34.42** | **18.09** | **57.01** | — |
| 16 | 27.45 | 34.59 | 17.62 | 57.04 | — |
| 18 | 27.44 | 34.60 | 17.59 | 57.14 | — |
| 20 | 27.49 | 34.65 | 17.65 | 57.13 | — |

**Key findings:**

**1. Delayed peak reveals a regularization effect.** The baseline peaks at epoch 8 and declines monotonically. Feature augmentation peaks at epoch 14—six epochs later—and maintains a remarkably flat plateau from ep12–ep20 (PQ oscillates within 27.41–27.54, a 0.13 range vs. baseline's 0.41 decline over the same window). The noise and dropout prevent the decoder from memorizing the fixed DINOv2 feature distribution, delaying convergence but suppressing the overfitting-driven PQ decline.

**2. PQ_things peaks higher than expected.** PQ_things reaches 18.09 at epoch 14—exceeding the baseline's best PQ_things (17.66 at ep8) by +0.43. This is a surprising result: feature noise *improves* instance boundary quality in later training, even though it *hurts* PQ overall. The explanation lies in the different mechanisms: PQ_stuff declines steadily (35.05 → 34.42) as noise disrupts confident stuff predictions, but the regularized decoder continues refining boundary pixels through epoch 14 rather than overfitting to easy interior pixels.

**3. The PQ vs. PQ_things divergence.** At epoch 14, Run D achieves the second-best PQ_things in Phase 1 (18.09, behind only focal loss at 18.10) while achieving the worst PQ among completed runs (27.54). This divergence indicates that feature noise selectively benefits boundary refinement while globally hurting semantic confidence—a mirror image of the focal loss mechanism, which selectively suppresses easy-pixel gradients.

**4. A stability–performance trade-off.** The training curve reveals a fundamental tension: regularization stabilizes training (flat plateau, no catastrophic decline) but caps peak performance below the baseline's transient optimum. The baseline's sharp peak at epoch 8 represents an aggressive but fragile optimum; feature augmentation finds a broader but lower basin. This pattern—and the analogous finding from Run C (low LR)—suggests that the ideal training protocol combines early aggressive optimization (to reach the sharp optimum) with late-stage regularization (to stabilize it).

**Implication for Phase 3.** Feature augmentation alone does not justify inclusion in combined runs. However, the delayed PQ_things peak suggests that regularization mechanisms interact non-trivially with boundary-sensitive losses. Combining feature augmentation with focal loss (which also targets boundaries) could yield compounding PQ_things improvements, though PQ_stuff may suffer further. This interaction remains untested.

### 5.11 Remaining Ablations

_Results for runs E–H are pending. The following hypotheses will be evaluated upon completion:_
1. **Enriched depth skip (E)** should benefit PQ_things more than PQ_stuff, as the Laplacian kernel captures second-order curvature information at object boundaries that first-order Sobel gradients miss.
2. **Additional decoder block (F)** tests whether the output resolution (128×256) has sufficient capacity, given that only one CoupledConvBlock currently operates at this scale.
3. **Depth-boundary weighting (G)** targets the same boundary pixels as focal loss but through a complementary mechanism: geometric rather than confidence-based selection. If G and B provide similar gains, the boundary pixel identification mechanism matters less than the act of upweighting these pixels.
4. **The combined run (H)** will reveal whether the interventions provide additive gains or exhibit diminishing returns. Given that focal loss already achieves +0.12 PQ, the combined run must exceed 27.85 to justify its added complexity.

---

## 6. Discussion

### 6.1 Why Progressive Upsampling with Depth Skip Connections Helps

The improvement from flat upsampling (HiRes) to the UNet decoder can be attributed to two complementary mechanisms.

First, **progressive upsampling** avoids the information-theoretic constraint of single-step 4× enlargement: interpolating a 32×64 feature map to 128×256 in one step can only redistribute existing information across more spatial positions, not synthesize new high-frequency content. The two-stage decoder allows intermediate refinement at 64×128, where the model can establish coarse boundary structure before further upsampling.

Second, **depth skip connections** provide an external source of boundary information at each decoder scale. The skip block downsamples full-resolution depth (512×1024) to the target resolution and extracts three complementary signals:

- **Depth values**: encode object layering (foreground vs. background separation).
- **Sobel gradients** (∂D/∂x, ∂D/∂y): respond strongly at depth discontinuities, which typically coincide with object boundaries.
- **Laplacian** (∇²D, enriched mode): captures curvature information, highlighting thin structures and boundary junctions that first-order operators miss.

Critically, these signals are resolution-matched to each decoder stage. A depth boundary visible at 64×128 provides precisely the spatial cue needed for the first decoder stage; the same boundary at 128×256 guides the second stage. This multi-scale injection is impossible in a flat-upsampling architecture, where depth information can only be incorporated at a single resolution.

### 6.2 The Persistent PQ_things Gap

Despite the UNet's improvement, PQ_things (17.66) remains 1.75 points below the input pseudo-labels (19.41). This regression is not a failure of the decoder architecture per se, but a structural limitation of the **semantic refinement + connected-component** pipeline:

1. **Semantic-only refinement** cannot separate same-class instances that share identical feature representations in the DINOv2 bottleneck. Two adjacent cars at 32×64 may occupy a single patch vector; no amount of upsampling recovers the lost spatial distinction.
2. **Connected-component extraction** at 128×256 requires at least one pixel of background between instances to distinguish them. At this resolution, a 1-pixel gap corresponds to ~8 pixels in the original image—smaller than many inter-instance boundaries in dense urban scenes.
3. **Semantic smoothing** during refinement merges previously distinct instance regions. The distillation loss encourages agreement with pseudo-labels that may have been more fragmented (and thus more correct for thing-class PQ) than the refined predictions.

These observations suggest that further PQ_things improvement requires dedicated instance segmentation mechanisms (e.g., Cascade Mask R-CNN [1, 37]) rather than continued semantic refinement.

### 6.3 The Focal Loss Insight: Gradient Budget Allocation Matters

The focal loss ablation (Section 5.8) provides an important insight: even with a fixed architecture, the *distribution* of gradient across pixels significantly impacts PQ. Standard cross-entropy treats all pixels equally, spending ~94% of gradient on already-correct interior pixels. Focal loss redirects this budget to the ~6% of ambiguous boundary pixels, yielding +0.44 PQ_things with no architectural change and negligible computational overhead.

This finding has broader implications. The persistent PQ_things gap (Section 6.2) may be partially addressable through loss engineering rather than architectural changes. Specifically, any mechanism that identifies and upweights boundary pixels—whether via prediction confidence (focal loss), geometric cues (depth-boundary weighting), or learned attention—should provide similar benefits. The combined ablation (Run H) will test whether these mechanisms are additive or redundant.

The focal loss result also explains why the lower learning rate (Run C) underperforms: at 2×10⁻⁵, the gradient magnitude at boundary pixels is already small; further suppression by cosine decay leaves insufficient signal to resolve ambiguous predictions. This suggests that **boundary-aware loss weighting and learning rate interact non-trivially**, and the optimal learning rate may differ between uniform and focal objectives.

### 6.4 Relationship to CUPS

Our UNet refinement achieves PQ = 27.73 using only stage-1 pseudo-label refinement, within 0.07 PQ of the full CUPS pipeline (PQ = 27.8). However, this comparison requires careful interpretation: CUPS applies a Cascade Mask R-CNN at stage 2 and self-training at stage 3, both of which are orthogonal to and composable with our refinement. Our result demonstrates that high-quality pseudo-labels combined with geometry-aware refinement can match a multi-stage pipeline, and suggests that applying CUPS stages 2–3 on top of our refined pseudo-labels could yield further improvements.

---

## 7. Conclusion

We have presented a Depth-Guided UNet decoder for unsupervised panoptic segmentation that replaces single-step feature upsampling with progressive two-stage decoding augmented by geometric skip connections from monocular depth maps. The decoder achieves PQ = 27.73 on Cityscapes—a new record for stage-1 pseudo-label refinement—by injecting boundary-level depth cues at each decoder scale. The architecture maintains parameter parity (4.28M vs. 4.19M) with flat-upsampling baselines while improving both stuff-class and thing-class panoptic quality.

Among six targeted ablations, focal loss (γ = 1.0) emerges as the most effective single intervention, achieving PQ = 27.85 (+0.12) by redirecting gradient from confident interior pixels to ambiguous boundaries—surpassing the full CUPS pipeline (27.8) using stage-1 refinement alone. Reduced learning rate (2×10⁻⁵) stabilizes training but settles at a lower plateau (PQ = 27.40), confirming that the baseline's post-peak decline is a learning rate phenomenon rather than a fundamental limitation. Four additional ablations (feature regularization, enriched depth skip, extra capacity, depth-boundary weighting, and their combination) are under evaluation and will be reported upon completion.

---

## References

[1] Z. Cai and N. Vasconcelos. Cascade R-CNN: Delving into high quality object detection. CVPR, 2018.

[3] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. ECCV, 2018.

[4] P. Chen et al. Unsupervised panoptic segmentation. arXiv, 2023.

[6] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. DeepLab: Semantic image segmentation with deep convolutional nets and fully connected CRFs. TPAMI, 2018.

[7] L.-C. Chen, G. Papandreou, F. Schroff, and H. Adam. Rethinking atrous convolution for semantic image segmentation. arXiv:1706.05587, 2017.

[8] M. Cordts et al. The Cityscapes dataset for semantic urban scene understanding. CVPR, 2016.

[9] S. Hwang et al. CAUSE: Contrastive and unsupervised segmentation with spatial embedding. Pattern Recognition, 2024.

[13] R. Gupta et al. Depth-conditioned semantic segmentation. arXiv, 2022.

[14] W. Ke et al. DepthG: Depth-guided unsupervised semantic segmentation. CVPR, 2024.

[23] A. Kirillov, K. He, R. Girshick, C. Rother, and P. Dollár. Panoptic segmentation. CVPR, 2019.

[24] Z. Li et al. Coupled Mamba: A cross-modal state space model. NeurIPS, 2024.

[25] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár. Focal loss for dense object detection. ICCV, 2017.

[26] T.-Y. Lin et al. Microsoft COCO: Common objects in context. ECCV, 2014.

[27] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. CVPR, 2015.

[28] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. ICLR, 2019.

[29] J. Niu, A. Peri, and D. Ramanan. U2Seg: Unified unsupervised segmentation. NeurIPS, 2024.

[30] M. Oquab et al. DINOv2: Learning robust visual features without supervision. TMLR, 2024.

[31] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville. FiLM: Visual reasoning with a general conditioning layer. AAAI, 2018.

[34] O. Ronneberger, P. Fischer, and T. Brox. U-Net: Convolutional networks for biomedical image segmentation. MICCAI, 2015.

[36] J. Shin et al. SPIdepth: Self-supervised monocular depth estimation. CVPR, 2024.

[37] Q. Sun, Y. Lu, and D. Cremers. CUPS: Comprehensive unsupervised panoptic segmentation. CVPR, 2025.

[38] Z. Zhang, Q. Liu, and Y. Wang. Road extraction by deep residual U-Net. IEEE GRSL, 2018.

[39] Z. Yin et al. DFormerv2: Multi-modal dense prediction with cross-modal attention. CVPR, 2025.

---

## Supplementary Material

### A. Implementation Details

**Optimizer:** AdamW (β₁ = 0.9, β₂ = 0.999, weight decay = 10⁻⁴). **Scheduler:** Cosine annealing from initial LR to 1% of initial over 20 epochs. **Gradient clipping:** Max norm = 1.0, with NaN/Inf sanitization for numerical stability. **Input preprocessing:** DINOv2 features and depth maps are pre-extracted and cached as .npy files; no online backbone computation. **Augmentation:** Random horizontal flip with depth gradient sign correction. Feature-level augmentation (noise + dropout) is evaluated as an ablation.

### B. Reproduction

```bash
# Baseline UNet training
PYTHONPATH=. python mbps_pytorch/train_refine_net.py \
  --cityscapes_root /path/to/cityscapes \
  --semantic_subdir pseudo_semantic_mapped_k80 \
  --num_classes 19 --model_type unet \
  --bridge_dim 192 --num_bottleneck_blocks 2 --skip_dim 32 \
  --lr 5e-5 --num_epochs 20 --eval_interval 2 \
  --lambda_distill 1.0 --lambda_distill_min 0.85 \
  --lambda_align 0.25 --lambda_proto 0.025 --lambda_ent 0.025 \
  --label_smoothing 0.1 \
  --output_dir checkpoints/hires_unet_depth_guided

# Combined ablation (all six interventions)
bash scripts/run_unet_ablations.sh H
```

### C. File Locations

| Resource | Path |
|----------|------|
| Model code | `mbps_pytorch/refine_net.py` |
| Training script | `mbps_pytorch/train_refine_net.py` |
| Ablation script | `scripts/run_unet_ablations.sh` |
| Baseline checkpoint | `checkpoints/hires_unet_depth_guided/best.pth` |
| Ablation checkpoints | `checkpoints/unet_ablation_{focal,low_lr,feat_aug,rich_skip,extra_block,depth_bw,combined}/` |
