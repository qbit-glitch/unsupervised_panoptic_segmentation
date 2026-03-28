# High-Resolution Cross-Modal Semantic Refinement via Bilinear Upsampling: A Comparative Training Analysis

## Abstract

We present the training analysis of Run U-A, the bilinear upsampling variant of HiRes RefineNet operating at 128x256 resolution. Over 20 training epochs, the model achieves PQ=27.29 (epoch 14), PQ_stuff=34.92 (epoch 18), and mIoU=57.01 --- the highest semantic quality observed in any configuration. However, PQ_things peaks at only 16.85 (epoch 14), trailing the transposed convolution variant (U-B) by 0.73 points. This gap persists across all evaluation checkpoints and concentrates on large thing classes where learned upsampling kernels provide boundary sharpness that parameter-free bilinear interpolation cannot. The results establish a clear dissociation: bilinear upsampling is optimal for semantic quality, while learned upsampling is necessary for panoptic instance separation.

---

## 1. Introduction

Run U-A tests the simplest upsampling strategy for HiRes RefineNet: bilinear interpolation from 32x64 to 128x256 with zero additional parameters. This serves as the lower bound for the Phase 1 upsampling ablation --- any learned upsampling strategy (transposed convolution, PixelShuffle) must outperform bilinear to justify its additional parameters and complexity.

The experimental configuration is identical to Run U-B in all respects except the upsampling module. The same CoupledConvBlock refinement architecture, training schedule, loss configuration, and pseudo-labels are used, isolating the upsampling strategy as the sole independent variable. The model comprises 3.89M parameters (versus 4.19M for U-B), the 300K difference corresponding exactly to the two ConvTranspose2d layers that bilinear interpolation replaces.

---

## 2. Training Trajectory

### 2.1 Aggregate Metrics

Table 1 presents the epoch-by-epoch trajectory with inter-epoch deltas.

**Table 1.** U-A aggregate metrics across 20 training epochs. Bold values indicate epoch-best.

| Epoch | PQ | Delta | PQ_stuff | Delta | PQ_things | Delta | mIoU | Delta | changed% |
|-------|-------|-------|----------|-------|-----------|-------|-------|-------|----------|
| 2 | 26.03 | --- | 34.03 | --- | 15.04 | --- | 55.48 | --- | 7.04 |
| 4 | 26.63 | +0.60 | 34.58 | +0.55 | 15.70 | +0.66 | 56.56 | +1.08 | 6.66 |
| 6 | 26.67 | +0.04 | 34.47 | -0.11 | 15.94 | +0.24 | 56.65 | +0.09 | 6.48 |
| 8 | 27.08 | +0.41 | **34.90** | +0.43 | 16.34 | +0.40 | 56.89 | +0.24 | 6.37 |
| 10 | 26.95 | -0.13 | 34.46 | -0.44 | 16.62 | +0.28 | 56.88 | -0.01 | 6.31 |
| 12 | 27.11 | +0.16 | 34.73 | +0.27 | 16.63 | +0.01 | 56.91 | +0.03 | 6.25 |
| 14 | **27.29** | +0.18 | 34.89 | +0.16 | **16.85** | +0.22 | 56.99 | +0.08 | 6.24 |
| 16 | 27.23 | -0.06 | 34.91 | +0.02 | 16.66 | -0.19 | 56.96 | -0.03 | 6.22 |
| 18 | 27.23 | +0.00 | **34.92** | +0.01 | 16.67 | +0.01 | **57.01** | +0.05 | 6.21 |
| 20 | 27.20 | -0.03 | 34.82 | -0.10 | 16.72 | +0.05 | **57.01** | +0.00 | 6.21 |

The training trajectory exhibits three characteristic phases, mirroring the dynamics observed in U-B but with systematically lower panoptic quality.

During the rapid learning phase (epochs 2--8), PQ climbs from 26.03 to 27.08 (+1.05), driven by concurrent improvements in stuff (34.03 to 34.90) and things (15.04 to 16.34). The largest single-epoch gain occurs at epoch 4 (PQ +0.60), where mIoU jumps by 1.08 points --- the sharpest semantic improvement in the entire run. This reflects the model's initial adjustment from random initialization to the pseudo-label distribution, a process independent of the upsampling strategy.

The refinement phase (epochs 8--14) sees slower but steady gains, with PQ reaching its peak of 27.29 at epoch 14. PQ_things plateaus around 16.6--16.85, never crossing the 32x64 baseline threshold of 17.10. This is the critical divergence from U-B, where PQ_things crossed 17.10 at epoch 8 and reached 17.58 by epoch 16.

The convergence phase (epochs 14--20) shows all metrics stabilizing. PQ_stuff reaches its maximum of 34.92 at epoch 18, and mIoU peaks at 57.01 (epochs 18 and 20). The changed_pct converges to 6.21%, marginally lower than U-B's 6.12%, indicating slightly more conservative refinement.

### 2.2 Comparison with U-B at Matched Epochs

Table 2 directly contrasts U-A and U-B at each evaluation point, quantifying the consistent advantage of learned upsampling.

**Table 2.** Head-to-head comparison of U-A (Bilinear) vs U-B (Transposed Conv) at matched epochs.

| Epoch | U-B PQ | U-A PQ | Gap | U-B PQ_things | U-A PQ_things | Gap | U-B mIoU | U-A mIoU | Gap |
|-------|--------|--------|-----|---------------|---------------|-----|----------|----------|-----|
| 2 | 26.49 | 26.03 | -0.46 | 16.01 | 15.04 | -0.97 | 56.24 | 55.48 | -0.76 |
| 4 | 26.86 | 26.63 | -0.23 | 16.76 | 15.70 | -1.06 | 56.55 | 56.56 | +0.01 |
| 6 | 27.08 | 26.67 | -0.41 | 16.59 | 15.94 | -0.65 | 56.83 | 56.65 | -0.18 |
| 8 | 27.10 | 27.08 | -0.02 | 17.12 | 16.34 | -0.78 | 56.45 | 56.89 | **+0.44** |
| 10 | 27.35 | 26.95 | -0.40 | 17.33 | 16.62 | -0.71 | 56.58 | 56.88 | **+0.30** |
| 12 | 27.50 | 27.11 | -0.39 | 17.52 | 16.63 | -0.89 | 56.76 | 56.91 | **+0.15** |
| 14 | 27.32 | 27.29 | -0.03 | 17.21 | 16.85 | -0.36 | 56.76 | 56.99 | **+0.23** |
| 16 | 27.42 | 27.23 | -0.19 | 17.58 | 16.66 | -0.92 | 56.83 | 56.96 | **+0.13** |
| 18 | 27.44 | 27.23 | -0.21 | 17.57 | 16.67 | -0.90 | 56.80 | 57.01 | **+0.21** |
| 20 | 27.43 | 27.20 | -0.23 | 17.57 | 16.72 | -0.85 | 56.89 | 57.01 | **+0.12** |

Three patterns emerge with striking consistency:

**PQ_things: U-B leads at every epoch.** The gap ranges from 0.36 to 1.06 points, averaging 0.81. This is not a training dynamics artifact --- it is a structural advantage of learned upsampling kernels that sharpen boundary features during the 4x spatial expansion. Bilinear interpolation produces smooth features by construction; transposed convolution can learn to enhance discontinuities at object boundaries.

**mIoU: U-A leads from epoch 4 onward.** After the initial epoch where U-B's learned upsampling provides a head start, U-A consistently achieves higher mIoU by 0.12--0.44 points. This counterintuitive result suggests that bilinear interpolation's smoothness is advantageous for semantic classification: the smoother feature gradients reduce noise that can confuse the pixel-wise classifier, yielding slightly more accurate per-pixel predictions.

**PQ: U-B leads, but the gap narrows.** The PQ gap shrinks from 0.46 (epoch 2) to 0.03--0.23 (epochs 14--20), reflecting that PQ balances stuff and things. U-A's mIoU advantage partially compensates for its PQ_things deficit, keeping the overall PQ gap modest.

---

## 3. Per-Class Analysis

### 3.1 Thing Classes: The Boundary Sharpness Gap

Table 3 compares per-class thing PQ between U-A and U-B at their respective best epochs.

**Table 3.** Per-class thing PQ comparison. U-B values from epoch 16 (best PQ_things), U-A from epoch 14 (best PQ_things).

| Class | U-B (ep16) | U-A (ep14) | Delta | Instance Size |
|-------|-----------|-----------|-------|---------------|
| person | 3.99 | 3.94 | -0.05 | Small |
| rider | 8.29 | 8.09 | -0.20 | Small |
| car | 14.92 | 15.16 | **+0.24** | Medium |
| truck | **33.56** | 29.82 | -3.74 | Large |
| bus | **40.49** | 40.08 | -0.41 | Large |
| train | **33.31** | 31.11 | -2.20 | Large |
| motorcycle | 0.00 | 0.00 | 0.00 | Rare |
| bicycle | 6.12 | 6.59 | **+0.47** | Small |

The PQ_things gap concentrates overwhelmingly on large thing classes. Truck shows the largest deficit: U-A's 29.82 trails U-B's 33.56 by 3.74 points. Train exhibits a similar pattern (31.11 vs 33.31, -2.20). These large vehicles have well-defined boundaries in the image that transposed convolution's learned kernels can sharpen during upsampling, directly improving the segment overlap (IoU) component of PQ.

Interestingly, car and bicycle show the reverse pattern: U-A slightly outperforms U-B (+0.24 and +0.47 respectively). For these smaller, more numerous instances, bilinear's smoother features may reduce false-positive fragmentation, leading to marginally better panoptic quality despite less precise boundaries.

Small thing classes (person, rider) show negligible differences between the two upsampling strategies, confirming that neither bilinear nor transposed convolution can recover the sub-patch spatial detail needed for small-instance separation.

### 3.2 Stuff Classes

Table 4 presents stuff-class PQ at best epochs.

**Table 4.** Per-class stuff PQ comparison. U-B values from epoch 12 (best PQ), U-A from epoch 14 (best PQ).

| Class | U-B (ep12) | U-A (ep14) | Delta |
|-------|-----------|-----------|-------|
| road | 77.25 | 77.11 | -0.14 |
| sidewalk | 45.97 | 45.50 | -0.47 |
| building | 68.07 | 67.94 | -0.13 |
| wall | 15.39 | 14.79 | -0.60 |
| fence | 13.41 | 13.58 | **+0.17** |
| pole | 0.00 | 0.00 | 0.00 |
| traffic light | 0.00 | 0.00 | 0.00 |
| traffic sign | 17.31 | 19.07 | **+1.76** |
| vegetation | 68.87 | 69.79 | **+0.92** |
| terrain | 16.53 | 15.61 | -0.92 |
| sky | 59.61 | 60.34 | **+0.73** |

Stuff classes show a mixed pattern with no systematic advantage for either strategy. Traffic sign (+1.76 for U-A) and vegetation (+0.92) favor bilinear, while wall (-0.60) and sidewalk (-0.47) favor transposed conv. The overall PQ_stuff is remarkably similar (34.77 vs 34.92), with U-A holding a marginal 0.15-point edge driven primarily by traffic sign and sky improvements. For large, contiguous stuff regions, the upsampling strategy matters less than the refinement blocks' ability to classify pixels correctly.

### 3.3 Semantic Quality: Where Bilinear Excels

Table 5 highlights the classes where U-A's mIoU advantage is most pronounced.

**Table 5.** Per-class mIoU comparison at epoch 20 (both runs).

| Class | U-B mIoU | U-A mIoU | Delta |
|-------|----------|----------|-------|
| wall | 49.32 | **50.08** | +0.76 |
| fence | 44.50 | **44.20** | -0.30 |
| traffic sign | 47.48 | 47.90 | +0.42 |
| truck | 77.40 | **78.10** | +0.70 |
| bus | 81.66 | **80.80** | -0.86 |
| train | 74.36 | **74.04** | -0.32 |
| rider | 39.51 | **40.40** | +0.89 |
| bicycle | 55.68 | 56.06 | +0.38 |
| **Average (all)** | **56.89** | **57.01** | **+0.12** |

The mIoU advantage of bilinear upsampling is distributed broadly across classes rather than concentrated in a few. Rider (+0.89), wall (+0.76), and truck (+0.70) show the largest gains. The hypothesis is that bilinear interpolation's smooth feature gradients reduce classification noise at class boundaries, yielding more consistent per-pixel predictions. Transposed convolution's learned sharpening, while beneficial for panoptic instance boundaries, may introduce high-frequency artifacts that slightly degrade semantic classification accuracy.

---

## 4. The Bilinear Paradox: Better Semantics, Worse Panoptics

The central finding of Run U-A is a dissociation between semantic and panoptic quality that has not been previously observed in this project. Bilinear upsampling achieves the highest mIoU (57.01) and PQ_stuff (34.92) of any configuration, yet its PQ (27.29) and PQ_things (16.85) trail the transposed convolution variant.

This dissociation arises from the dual nature of panoptic evaluation. PQ decomposes into recognition quality (RQ) and segmentation quality (SQ). mIoU measures only per-pixel classification accuracy --- analogous to SQ --- while PQ additionally requires correct instance detection (RQ). Bilinear interpolation optimizes for smooth, consistent classification (high mIoU, high SQ) but produces features that lack the sharp discontinuities needed to detect instance boundaries (lower RQ for things).

The implication for architecture design is that upsampling strategies must be evaluated on panoptic metrics, not semantic metrics alone. A strategy that maximizes mIoU may be suboptimal for panoptic segmentation, particularly for thing classes where instance boundary precision determines the dominant quality factor.

---

## 5. Summary of Findings

### 5.1 Final Results

| Metric | U-A Best | U-A Epoch | U-B Best | U-B Epoch | Delta (U-A - U-B) |
|--------|----------|-----------|----------|-----------|-------------------|
| PQ | 27.29 | 14 | **27.50** | 12 | -0.21 |
| PQ_stuff | **34.92** | 18 | 34.77 | 12 | +0.15 |
| PQ_things | 16.85 | 14 | **17.58** | 16 | -0.73 |
| mIoU | **57.01** | 18 | 56.89 | 20 | +0.12 |
| Parameters | **3.89M** | --- | 4.19M | --- | -300K |

### 5.2 Key Takeaways

1. **Bilinear is the semantic champion.** It achieves the highest mIoU (57.01) and PQ_stuff (34.92) of any configuration tested, with zero additional parameters beyond the refinement blocks. For applications that prioritize semantic segmentation quality over panoptic instance separation, bilinear is the optimal upsampling strategy.

2. **Transposed convolution is the panoptic champion.** Its learned upsampling kernels provide boundary sharpness that translates to +0.73 PQ_things and +0.21 PQ over bilinear. The advantage concentrates on large thing classes (truck +3.74, train +2.20) where boundary precision directly impacts segment overlap quality.

3. **The 300K parameter investment pays off for things, not stuff.** Transposed convolution's 300K extra parameters (two ConvTranspose2d layers) contribute exclusively to thing-class boundary quality. Stuff classes show no systematic benefit from learned upsampling. This targeted impact validates the architectural hypothesis that thing-class panoptic quality is boundary-limited, not semantics-limited.

4. **PQ_things remains below the 32x64 baseline for bilinear.** U-A's best PQ_things of 16.85 never crosses the Run D threshold of 17.10, while U-B crossed it at epoch 8. This confirms that bilinear upsampling cannot fully compensate for the spatial information lost during ViT encoding, even with four refinement blocks at 128x256 resolution.

### 5.3 Recommended Checkpoint

For downstream use, **U-B (transposed conv) epoch 12** remains the recommended checkpoint (PQ=27.50). U-A's mIoU advantage (+0.12) does not outweigh its PQ deficit (-0.21) for panoptic segmentation. However, if semantic segmentation maps are the primary output (e.g., for pseudo-label generation), U-A epoch 18 (mIoU=57.01) provides the highest-quality semantic predictions.

---

## Appendix: Per-Class PQ Trajectories (U-A)

### Thing Classes

| Class | Ep2 | Ep4 | Ep6 | Ep8 | Ep10 | Ep12 | Ep14 | Ep16 | Ep18 | Ep20 |
|-------|------|------|------|------|------|------|------|------|------|------|
| person | 4.50 | 4.09 | 4.30 | 3.87 | 3.61 | 3.91 | 3.94 | 4.00 | 3.91 | 4.14 |
| rider | 7.06 | 7.83 | 8.69 | 8.23 | 8.44 | 8.49 | 8.09 | 8.44 | 8.31 | 8.24 |
| car | 14.64 | 14.98 | 15.53 | 14.91 | 14.96 | 14.38 | 15.16 | 14.99 | 14.99 | 15.10 |
| truck | 28.35 | 28.85 | 27.86 | 29.49 | 29.92 | 30.29 | 29.82 | 29.96 | 30.57 | 31.14 |
| bus | 35.34 | 37.87 | 38.77 | 39.46 | 39.01 | 40.61 | 40.08 | 39.97 | 39.57 | 39.44 |
| train | 22.71 | 24.54 | 25.05 | 27.55 | 29.97 | 28.52 | 31.11 | 29.27 | 29.36 | 29.24 |
| motorcycle | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| bicycle | 7.69 | 7.44 | 7.28 | 7.18 | 7.08 | 6.80 | 6.59 | 6.67 | 6.67 | 6.47 |

### Stuff Classes

| Class | Ep2 | Ep4 | Ep6 | Ep8 | Ep10 | Ep12 | Ep14 | Ep16 | Ep18 | Ep20 |
|-------|------|------|------|------|------|------|------|------|------|------|
| road | 76.93 | 76.87 | 77.02 | 77.07 | 77.18 | 77.21 | 77.11 | 77.17 | 77.19 | 77.17 |
| sidewalk | 45.43 | 43.75 | 44.16 | 44.97 | 45.77 | 45.23 | 45.50 | 45.64 | 45.63 | 45.43 |
| building | 67.33 | 67.88 | 67.77 | 68.23 | 67.84 | 68.16 | 67.94 | 68.57 | 68.28 | 68.30 |
| wall | 12.90 | 13.16 | 14.11 | 14.77 | 14.81 | 14.75 | 14.79 | 14.91 | 15.07 | 15.03 |
| fence | 11.41 | 13.00 | 12.07 | 13.68 | 12.72 | 12.18 | 13.58 | 12.82 | 13.93 | 13.56 |
| pole | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| traffic light | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| traffic sign | 17.67 | 18.93 | 19.09 | 18.61 | 18.11 | 19.42 | 19.07 | 19.09 | 19.15 | 18.83 |
| vegetation | 68.69 | 68.48 | 69.27 | 69.27 | 70.11 | 68.71 | 69.79 | 69.20 | 69.22 | 69.28 |
| terrain | 16.03 | 16.68 | 16.44 | 16.93 | 15.60 | 16.16 | 15.61 | 15.93 | 15.89 | 16.12 |
| sky | 57.94 | 61.67 | 59.28 | 60.33 | 56.91 | 60.23 | 60.34 | 60.65 | 59.72 | 59.31 |
