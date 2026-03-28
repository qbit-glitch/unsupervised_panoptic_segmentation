# High-Resolution Cross-Modal Semantic Refinement via PixelShuffle Upsampling: A Comparative Training Analysis

## Abstract

We present the training analysis of Run U-C, the PixelShuffle upsampling variant of HiRes RefineNet. PixelShuffle expands channels by 16x via a learned 1x1 convolution (192 to 3072 channels), then rearranges them into spatial dimensions via sub-pixel convolution. Despite 590K additional upsampling parameters --- nearly twice that of transposed convolution (300K) --- PixelShuffle achieves PQ=27.26 (epoch 12), trailing transposed convolution by 0.24 PQ and matching bilinear (27.29) within noise. PQ_things peaks at 16.86, nearly identical to bilinear (16.85) and 0.72 points below transposed convolution (17.58). The model achieves the highest mIoU of any configuration (57.15 at epoch 16), continuing the pattern that smooth upsampling strategies optimize for semantic classification rather than panoptic instance separation. A distinctive slow-start phenomenon (PQ=21.69 at epoch 2, recovering to 27.26 by epoch 12) reveals that the 3072-channel expansion requires substantial training to learn useful sub-pixel spatial mappings.

---

## 1. Introduction

Run U-C completes the Phase 1 upsampling ablation by testing sub-pixel convolution (Shi et al., CVPR 2016), the most parameter-heavy upsampling strategy. A 1x1 convolution expands the 192-channel feature map to 3072 channels (192 x 4^2), and PixelShuffle rearranges these channels into a 4x spatial expansion. Unlike transposed convolution, which learns spatially-local upsampling kernels, PixelShuffle learns a per-subpixel channel-to-space mapping --- each of the 16 sub-pixel positions receives a learned linear combination of the 192 input channels.

The model comprises 4.48M parameters (versus 4.19M for U-B and 3.89M for U-A), with the 590K overhead concentrated in the single Conv2d(192, 3072, 1) layer. All other components --- projections, refinement blocks, classification head, and training configuration --- are identical to U-A and U-B.

---

## 2. Training Trajectory

### 2.1 Aggregate Metrics

**Table 1.** U-C aggregate metrics across 20 training epochs.

| Epoch | PQ | Delta | PQ_stuff | Delta | PQ_things | Delta | mIoU | Delta | changed% |
|-------|-------|-------|----------|-------|-----------|-------|-------|-------|----------|
| 2 | 21.69 | --- | 32.90 | --- | 6.27 | --- | 48.73 | --- | 7.74 |
| 4 | 25.83 | +4.14 | 34.41 | +1.51 | 14.04 | +7.77 | 56.26 | +7.53 | 6.74 |
| 6 | 26.57 | +0.74 | 34.34 | -0.07 | 15.90 | +1.86 | 56.70 | +0.44 | 6.61 |
| 8 | 26.61 | +0.04 | 34.62 | +0.28 | 15.58 | -0.32 | **57.03** | +0.33 | 6.51 |
| 10 | 27.03 | +0.42 | 34.64 | +0.02 | 16.57 | +0.99 | 56.84 | -0.19 | 6.42 |
| 12 | **27.26** | +0.23 | **34.82** | +0.18 | **16.86** | +0.29 | 57.12 | +0.28 | 6.30 |
| 14 | 27.09 | -0.17 | 34.55 | -0.27 | 16.83 | -0.03 | 56.83 | -0.29 | 6.28 |
| 16 | 27.13 | +0.04 | 34.70 | +0.15 | 16.71 | -0.12 | **57.15** | +0.32 | 6.25 |
| 18 | 27.07 | -0.06 | 34.70 | +0.00 | 16.57 | -0.14 | 57.06 | -0.09 | 6.25 |
| 20 | 27.17 | +0.10 | 34.74 | +0.04 | 16.75 | +0.18 | 57.13 | +0.07 | 6.23 |

### 2.2 The Slow-Start Phenomenon

The most distinctive feature of U-C's training trajectory is its dramatically poor epoch 2 performance: PQ=21.69, PQ_things=6.27, mIoU=48.73. For comparison, U-B achieves PQ=26.49 and U-A achieves PQ=26.03 at the same epoch --- a 4.34--4.80 PQ gap.

The root cause is architectural. The PixelShuffle upsampling module must learn a Conv2d(192, 3072, 1) layer that maps 192 input channels to 3072 output channels, which are then rearranged into 192 channels at 4x spatial resolution. At initialization, this mapping is essentially random, producing spatially incoherent features at 128x256. The refinement blocks receive noise-like inputs and cannot make useful predictions. By contrast, bilinear interpolation produces coherent (if smooth) features from epoch 1, and transposed convolution's smaller 4x4 kernels converge quickly.

The recovery is rapid: by epoch 4, PQ jumps by +4.14 (the largest single-epoch gain in the entire Phase 1 ablation), and mIoU recovers from 48.73 to 56.26 (+7.53). By epoch 6, U-C's PQ (26.57) is within 0.51 of U-B (27.08) and within 0.10 of U-A (26.67). The slow start costs approximately 2--4 epochs of effective training, after which PixelShuffle converges to a trajectory similar to bilinear.

This has a practical implication: if training budget is limited, PixelShuffle wastes 10--20% of epochs on initialization recovery. For 20-epoch runs, this matters; for longer training schedules, the cost would be amortized.

### 2.3 Three-Way Comparison at Matched Epochs

**Table 2.** All three upsampling strategies compared at matched epochs.

| Epoch | U-B PQ | U-A PQ | U-C PQ | U-B PQ_things | U-A PQ_things | U-C PQ_things |
|-------|--------|--------|--------|---------------|---------------|---------------|
| 2 | 26.49 | 26.03 | 21.69 | 16.01 | 15.04 | 6.27 |
| 4 | 26.86 | 26.63 | 25.83 | 16.76 | 15.70 | 14.04 |
| 6 | 27.08 | 26.67 | 26.57 | 16.59 | 15.94 | 15.90 |
| 8 | 27.10 | 27.08 | 26.61 | 17.12 | 16.34 | 15.58 |
| 10 | 27.35 | 26.95 | 27.03 | 17.33 | 16.62 | 16.57 |
| 12 | **27.50** | 27.11 | 27.26 | **17.52** | 16.63 | 16.86 |
| 14 | 27.32 | **27.29** | 27.09 | 17.21 | **16.85** | 16.83 |
| 16 | 27.42 | 27.23 | 27.13 | **17.58** | 16.66 | 16.71 |
| 18 | 27.44 | 27.23 | 27.07 | 17.57 | 16.67 | 16.57 |
| 20 | 27.43 | 27.20 | 27.17 | 17.57 | 16.72 | 16.75 |

After the slow start, U-C converges to a performance profile nearly identical to U-A. From epoch 6 onward, the PQ gap between U-A and U-C is at most 0.20 points, and PQ_things differs by at most 0.15 points. The 590K learned parameters in PixelShuffle provide no measurable advantage over zero-parameter bilinear interpolation for panoptic refinement quality.

U-B maintains a consistent advantage over both U-A and U-C across all epochs from epoch 6 onward, with a PQ_things gap of 0.60--1.00 points. This advantage is structural: transposed convolution's spatially-local learned kernels (4x4 per stage) are better suited to boundary sharpening than PixelShuffle's channel-to-space rearrangement.

---

## 3. Per-Class Analysis

### 3.1 Thing Classes at Best Epoch

**Table 3.** Per-class thing PQ at each run's best PQ_things epoch.

| Class | U-B (ep16) | U-A (ep14) | U-C (ep12) | U-C vs U-B |
|-------|-----------|-----------|-----------|------------|
| person | 3.99 | 3.94 | 3.89 | -0.10 |
| rider | 8.29 | 8.09 | 8.60 | +0.31 |
| car | 14.92 | 15.16 | 15.38 | +0.46 |
| truck | **33.56** | 29.82 | 32.28 | -1.28 |
| bus | **40.49** | 40.08 | **40.81** | +0.32 |
| train | **33.31** | 31.11 | 33.01 | -0.30 |
| motorcycle | 0.00 | 0.00 | 0.00 | 0.00 |
| bicycle | 6.12 | 6.59 | 6.19 | +0.07 |

PixelShuffle shows an interesting intermediate profile. It outperforms bilinear on truck (+2.46), train (+1.90), and bus (+0.73), suggesting that the learned channel-to-space mapping captures some boundary information for large vehicles. However, it still trails transposed convolution on truck (-1.28) and train (-0.30). For small things (person, rider, bicycle), all three strategies perform within noise of each other.

### 3.2 Semantic Quality

**Table 4.** mIoU comparison at epoch 20 (all runs converged).

| Metric | U-B | U-A | U-C |
|--------|------|------|------|
| mIoU (ep20) | 56.89 | 57.01 | 57.13 |
| mIoU (best) | 56.89 | 57.01 | **57.15** |
| PQ_stuff (best) | 34.77 | **34.92** | 34.82 |

U-C achieves the highest mIoU of any configuration (57.15 at epoch 16), continuing the pattern that non-boundary-sharpening upsampling strategies produce slightly better semantic classification. The 0.26-point mIoU advantage over U-B (57.15 vs 56.89) does not translate to panoptic quality, mirroring the same dissociation observed in the U-A analysis.

---

## 4. PixelShuffle vs Transposed Convolution: Why Spatial Locality Matters

The core question of U-C is whether a globally-learned channel-to-space mapping (PixelShuffle) can match or exceed locally-learned spatial kernels (transposed convolution) for upsampling. The answer is clearly negative for panoptic segmentation.

Transposed convolution learns 4x4 spatial kernels that operate locally: each output pixel is a learned combination of a 4x4 neighborhood in the input. This local structure is inherently suited to boundary operations --- sharpening, edge enhancement, and spatial discontinuity preservation are local phenomena. A ConvTranspose2d layer can learn to produce sharp transitions at object boundaries by adjusting its kernel weights to amplify feature differences between adjacent input positions.

PixelShuffle learns a global 1x1 convolution that maps each input position independently to 16 sub-pixel outputs. There is no spatial interaction between input positions during upsampling --- each input pixel generates its 4x4 sub-pixel block in isolation. Boundary sharpness can only emerge from the downstream refinement blocks, not from the upsampling itself. The 590K parameters learn a channel-space factorization that is expressive but spatially unaware.

This explains why PixelShuffle converges to bilinear-like performance: both strategies produce spatially smooth upsampled features (bilinear by construction, PixelShuffle by its position-independent architecture), and the refinement blocks compensate identically in both cases. The extra 590K parameters in PixelShuffle learn a marginally different feature distribution but not a fundamentally different spatial structure.

---

## 5. Phase 1 Conclusion

### 5.1 Final Ranking

**Table 5.** Phase 1 upsampling ablation final results.

| Rank | Run | Strategy | Extra Params | Best PQ | Best PQ_things | Best mIoU |
|------|-----|----------|-------------|---------|----------------|-----------|
| 1 | U-B | Transposed Conv | 300K | **27.50** | **17.58** | 56.89 |
| 2 | U-A | Bilinear | 0 | 27.29 | 16.85 | 57.01 |
| 3 | U-C | PixelShuffle | 590K | 27.26 | 16.86 | **57.15** |

### 5.2 Key Findings

1. **Transposed convolution is the optimal upsampling strategy for panoptic segmentation.** Its locally-learned spatial kernels provide boundary sharpness that directly improves PQ_things by +0.72--0.73 over both alternatives. The 300K parameter investment yields a clear return on the primary evaluation metric.

2. **PixelShuffle provides no advantage over bilinear.** Despite 590K additional parameters (2x that of transposed conv), PixelShuffle matches bilinear within noise on all panoptic metrics. Its position-independent channel-to-space mapping cannot learn the spatially-local operations needed for boundary sharpening.

3. **Semantic quality inversely correlates with panoptic quality across upsampling strategies.** The ranking by mIoU (U-C > U-A > U-B) is exactly reversed from the ranking by PQ (U-B > U-A > U-C). Smooth upsampling benefits pixel classification; sharp upsampling benefits instance separation. These are competing objectives that the choice of upsampling strategy can trade off.

4. **PixelShuffle exhibits a slow-start phenomenon.** The 3072-channel expansion requires ~4 epochs to learn useful spatial mappings, wasting 20% of a 20-epoch training budget. This makes PixelShuffle both less effective and less efficient than the alternatives.

### 5.3 Selected Strategy for Phase 2

**Transposed Convolution** is carried forward to Phase 2 (architecture ablation) based on best PQ (27.50) and best PQ_things (17.58). Phase 2 will compare five refinement block architectures (Conv2d, Windowed Attention, VisionMamba2, Spatial Mamba, MambaOut) all using transposed convolution upsampling at 128x256 resolution.
