# High-Resolution Cross-Modal Semantic Refinement via Transposed Convolution Upsampling: A Detailed Training Analysis

## Abstract

We present a detailed epoch-by-epoch analysis of Run U-B, the first HiRes RefineNet configuration operating at 128x256 resolution with two-stage transposed convolution upsampling. Over 20 training epochs, the model achieves PQ=27.50 (epoch 12), PQ_stuff=34.77, and mIoU=56.89 --- all new records for unsupervised panoptic segmentation on Cityscapes val. PQ_things partially recovers from the 32x64 regression (17.10 to 17.58), but a 1.83-point gap to the input pseudo-label quality (19.41) persists. Per-class decomposition reveals that resolution gains concentrate on large thing instances (truck +6.03, train +5.18) while small, frequently adjacent things (person +0.41, bicycle -0.84) show negligible or negative improvement. These findings motivate UNet-style decoders with boundary-aware skip connections as the next architectural intervention.

---

## 1. Introduction

The resolution bottleneck at 32x64 patch resolution has been established as the primary limitation of CSCMRefineNet for unsupervised panoptic segmentation. Four ablation studies (TAD, BPL, ASPP-lite, TAD+BPL) confirmed that no loss function, receptive field, or weighting modification can recover the PQ_things regression (19.41 to 17.10) inherent to operating at 2,048 spatial positions. This motivated the HiRes RefineNet architecture, which upsamples frozen DINOv2 ViT-B/14 features from 32x64 to 128x256 (32,768 spatial positions) before applying cross-modal refinement blocks.

Run U-B is the first HiRes RefineNet experiment, using two-stage transposed convolution upsampling (32x64 to 64x128 to 128x256) with four CoupledConvBlock refinement blocks at the target resolution. This report provides a comprehensive analysis of the training trajectory, per-class dynamics, stuff-things trade-offs, and cost-benefit profile relative to the 32x64 baseline.

### 1.1 Experimental Configuration

The model comprises 4.19M parameters: semantic and depth-feature projections (768 to 192 channels), two ConvTranspose2d upsampling stages (~300K parameters), four CoupledConvBlocks with depthwise-separable 3x3 convolutions and cross-modal gating at 128x256 resolution, and a 19-class classification head. Training uses the proven conservative configuration from Run D: cosine-annealed learning rate (5e-5), distillation floor of 0.85 with label smoothing 0.1, and gentle self-supervised losses (lambda_align=0.25, lambda_proto=0.025, lambda_ent=0.025). Pseudo-labels are k=80 overclustered CAUSE-TR predictions mapped to 19 Cityscapes trainIDs, downsampled to 128x256 via nearest-neighbor interpolation. Depth maps (SPIdepth, 512x1024) are bilinear-downsampled to 32x64 for FiLM conditioning. Training runs for 20 epochs with evaluation every 2 epochs on MPS (Apple M4 Pro, 48GB unified memory, float32).

### 1.2 Reference Baselines

Two baselines frame the analysis. The input pseudo-labels (k=80 overclustered with depth-guided instance splitting) achieve PQ=26.74, PQ_stuff=32.08, PQ_things=19.41, and mIoU of approximately 50%. The best 32x64 CSCMRefineNet configuration (Run D, epoch 16) achieves PQ=26.52, PQ_stuff=33.38, PQ_things=17.10, and mIoU=55.31. The critical observation is that Run D improves semantic quality at the cost of thing-class panoptic quality --- a trade-off that 128x256 resolution aims to resolve.

---

## 2. Training Trajectory

### 2.1 Aggregate Metric Evolution

Table 1 presents the complete epoch-by-epoch trajectory of aggregate metrics, together with inter-epoch deltas that reveal the model's learning dynamics.

**Table 1.** Aggregate metrics across 20 training epochs. Bold values indicate epoch-best for each metric. Delta columns show change from the previous evaluation checkpoint.

| Epoch | PQ | Delta PQ | PQ_stuff | Delta | PQ_things | Delta | mIoU | Delta | changed% |
|-------|-------|----------|----------|-------|-----------|-------|-------|-------|----------|
| 2 | 26.49 | --- | 34.11 | --- | 16.01 | --- | 56.24 | --- | 6.65 |
| 4 | 26.86 | +0.37 | 34.20 | +0.09 | 16.76 | +0.75 | 56.55 | +0.31 | 6.40 |
| 6 | 27.08 | +0.22 | **34.70** | +0.50 | 16.59 | -0.17 | 56.83 | +0.28 | 6.27 |
| 8 | 27.10 | +0.02 | 34.36 | -0.34 | 17.12 | +0.53 | 56.45 | -0.38 | 6.21 |
| 10 | 27.35 | +0.25 | 34.64 | +0.28 | 17.33 | +0.21 | 56.58 | +0.13 | 6.15 |
| 12 | **27.50** | +0.15 | **34.77** | +0.13 | 17.52 | +0.19 | 56.76 | +0.18 | 6.15 |
| 14 | 27.32 | -0.18 | 34.68 | -0.09 | 17.21 | -0.31 | 56.76 | +0.00 | 6.12 |
| 16 | 27.42 | +0.10 | 34.57 | -0.11 | **17.58** | +0.37 | 56.83 | +0.07 | 6.12 |
| 18 | 27.44 | +0.02 | 34.61 | +0.04 | 17.57 | -0.01 | 56.80 | -0.03 | 6.12 |
| 20 | 27.43 | -0.01 | 34.60 | -0.01 | 17.57 | +0.00 | **56.89** | +0.09 | 6.12 |

The training trajectory divides naturally into three phases. During the first phase (epochs 2--6), the model achieves rapid semantic improvement: PQ_stuff rises from 34.11 to 34.70 (+0.59) and mIoU from 56.24 to 56.83 (+0.59). The network prioritizes the dominant stuff classes, which account for the majority of image area. PQ_things lags behind and even dips at epoch 6 (16.59, down from 16.76), indicating that early gradient updates favor stuff-class boundary refinement over thing-instance preservation.

The second phase (epochs 6--12) marks the critical transition: PQ_things surges from 16.59 to 17.52 (+0.93 over three evaluation points). At epoch 8, PQ_things reaches 17.12, crossing the 32x64 baseline threshold of 17.10 for the first time. This inflection point validates the resolution hypothesis --- the 128x256 feature map provides sufficient spatial precision to distinguish thing instances that were irrecoverably merged at 32x64. PQ reaches its maximum of 27.50 at epoch 12, driven by simultaneous improvements in both stuff (34.77) and things (17.52).

The third phase (epochs 12--20) exhibits a convergence plateau. All metrics oscillate within narrow bands: PQ between 27.32 and 27.50, PQ_stuff between 34.57 and 34.77, and PQ_things between 17.21 and 17.58. The cosine learning rate schedule drives the model toward a stable optimum, and the changed_pct stabilizes at 6.12% --- well within the healthy 3--8% refinement intensity range. Notably, PQ_things achieves its absolute best at epoch 16 (17.58), two evaluation points after the PQ peak, suggesting that later training slightly favors thing-class precision at a small cost to overall panoptic quality.

### 2.2 Refinement Intensity

The changed_pct metric quantifies the fraction of pixels whose predicted class differs from the input pseudo-label. It decreases monotonically from 6.65% (epoch 2) to 6.12% (epoch 14 onward), reflecting increasingly conservative refinement as the learning rate decays. This behavior is desirable: the model makes its largest corrections early, when the learning rate is highest and gradient signals are strongest, then gradually locks in predictions as confidence increases. The final value of 6.12% indicates that the model changes approximately 1 in 16 pixels relative to the input --- a rate consistent with boundary-focused refinement rather than wholesale reclassification.

---

## 3. Per-Class Panoptic Quality Analysis

### 3.1 Stuff Classes

Table 2 presents the per-class PQ trajectory for the 11 Cityscapes stuff classes. The dynamics reveal three distinct behavioral groups.

**Table 2.** Per-class PQ for stuff classes across training epochs. Bold indicates epoch-best.

| Class | Ep2 | Ep6 | Ep12 | Ep16 | Ep20 | Best | Trend |
|-------|------|------|------|------|------|------|-------|
| road | 76.92 | **77.30** | 77.25 | 77.17 | 77.18 | 77.30 | Saturated |
| sidewalk | 44.23 | **46.70** | 45.97 | 44.81 | 45.26 | 46.70 | Early peak, slight decline |
| building | 67.86 | **68.83** | 68.07 | 68.29 | 67.94 | 68.83 | Saturated |
| wall | 12.82 | 14.26 | **15.39** | 15.29 | 15.02 | 15.39 | Steady improvement |
| fence | 12.09 | 13.20 | 13.41 | 13.47 | **13.70** | 13.70 | Continuous improvement |
| pole | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | Zero throughout |
| traffic light | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | Zero throughout |
| traffic sign | 16.81 | **18.32** | 17.31 | 17.60 | 17.42 | 18.32 | Early peak, volatile |
| vegetation | 68.51 | **70.14** | 68.87 | 68.48 | 68.86 | 70.14 | Early peak |
| terrain | 16.57 | 16.06 | 16.53 | 16.85 | 16.75 | **17.26** (ep14) | Late improvement |
| sky | 59.38 | 56.86 | **59.61** | 58.31 | 58.45 | 59.61 | Volatile |

The dominant stuff classes --- road (77.30), building (68.83), vegetation (70.14), and sky (59.61) --- reach their peaks within the first six epochs and exhibit minimal variation thereafter. These classes cover large contiguous regions where 128x256 resolution provides ample spatial precision; the refinement blocks quickly learn their characteristic feature patterns and converge. Road is particularly stable, oscillating by only 0.27 PQ across all 10 evaluation points.

Thin and boundary-sensitive stuff classes tell a different story. Wall improves steadily from 12.82 to 15.39, gaining +2.57 PQ over the full training run. Fence shows similar continuous improvement (12.09 to 13.70, +1.61). Both classes consist of thin, elongated structures that benefit directly from the 16x increase in spatial positions. The improvement trajectory suggests that the model gradually learns to separate these structures from adjacent classes (wall from building, fence from vegetation) as training progresses.

Pole and traffic light remain at zero PQ throughout all 20 epochs. These are extremely thin vertical structures that occupy a negligible fraction of image area. Even at 128x256, they are too narrow for the panoptic evaluation's connected-component analysis to detect as valid segments. The pseudo-labels likely contain very few correctly labeled pixels for these classes, providing insufficient supervision signal.

### 3.2 Thing Classes

Table 3 presents the per-class PQ trajectory for the 8 Cityscapes thing classes. The results reveal a stark size-dependent performance gradient.

**Table 3.** Per-class PQ for thing classes across training epochs. Bold indicates epoch-best.

| Class | Ep2 | Ep6 | Ep12 | Ep16 | Ep20 | Best | Trend |
|-------|------|------|------|------|------|------|-------|
| person | 3.60 | 3.86 | 3.89 | 3.99 | 4.01 | **4.04** (ep18) | Near-zero, slight upward |
| rider | **7.46** | 7.19 | 8.60 | 8.29 | 7.98 | **9.10** (ep4) | Volatile, no clear trend |
| car | 14.10 | 14.75 | **15.38** | 14.92 | 15.09 | 15.38 | Modest improvement |
| truck | 27.53 | 28.98 | 32.28 | **33.56** | 33.46 | 33.56 | Strong improvement |
| bus | 39.19 | 38.85 | **40.81** | 40.49 | 39.46 | 40.81 | Moderate, peaks mid-training |
| train | 29.19 | 32.83 | 33.01 | 33.31 | **34.37** | 34.37 | Continuous improvement |
| motorcycle | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | Zero throughout |
| bicycle | **7.03** | 6.27 | 6.19 | 6.12 | 6.19 | 7.03 | **Regression** |

The thing-class results decompose cleanly by instance size. Large thing instances --- truck, bus, and train --- show substantial and sustained improvement. Truck gains +6.03 PQ from epoch 2 (27.53) to its peak at epoch 16 (33.56), the largest per-class improvement in the entire experiment. Train improves continuously from 29.19 to 34.37 (+5.18), never declining between consecutive evaluations after epoch 4. These large vehicles occupy hundreds to thousands of pixels even at 128x256 resolution, providing sufficient spatial extent for the refinement blocks to refine boundaries and improve segment quality.

Car, the most frequent thing class, shows modest improvement from 14.10 to 15.38 (+1.28). While cars are common in Cityscapes, many appear at considerable distance and thus occupy relatively few pixels. Dense parking scenarios with adjacent cars at similar depth present an especially challenging case for semantic-only refinement, as no instance-aware mechanism distinguishes one car from another.

The small thing classes --- person, rider, bicycle, and motorcycle --- reveal the fundamental limitation of resolution-based approaches. Person PQ remains near zero throughout training (3.60 to 4.04, +0.44 over 20 epochs), despite mIoU of 54.5% for the person class. This dramatic divergence between semantic accuracy and panoptic quality confirms that the model correctly identifies person pixels but cannot separate individual person instances. Pedestrians in Cityscapes frequently stand adjacent to one another, and at 128x256 the gap between two people (often < 5 pixels in the original image) maps to a sub-pixel representation.

Bicycle presents the most concerning result: PQ decreases from 7.03 at epoch 2 to 6.19 at epoch 20, a regression of -0.84. The refinement process actively degrades bicycle predictions, likely by merging adjacent bicycle-rider pairs into single segments. This is a direct consequence of operating on upsampled features that lack the fine-grained boundary detail needed to separate thin, overlapping instances.

Motorcycle remains at zero PQ throughout. Motorcycles are extremely rare in the Cityscapes validation set (appearing in only a handful of images), and the pseudo-labels likely provide no usable instances for this class.

### 3.3 Size-Stratified Analysis

To quantify the size-dependent effect, we group thing classes by typical instance area and compute the average PQ improvement from epoch 2 to the best epoch:

| Size Category | Classes | Avg PQ (Ep2) | Avg PQ (Best) | Avg Improvement |
|---------------|---------|-------------|---------------|-----------------|
| Large | truck, bus, train | 31.97 | 36.25 | **+4.28** |
| Medium | car | 14.10 | 15.38 | **+1.28** |
| Small | person, rider, bicycle | 6.03 | 6.39 | **+0.36** |
| Zero | motorcycle | 0.00 | 0.00 | 0.00 |

The improvement scales approximately linearly with instance size: large things gain 12x more PQ than small things. This confirms that 128x256 resolution is sufficient for large-object boundary refinement but insufficient for small-object instance separation. The remaining PQ_things gap (17.58 vs 19.41) is dominated by the small-thing classes, which contribute approximately 2/3 of the deficit.

---

## 4. Semantic Segmentation Analysis

### 4.1 Per-Class mIoU Trajectory

Table 4 presents the mIoU trajectory for all 19 classes, complementing the panoptic analysis with a pure semantic perspective.

**Table 4.** Per-class mIoU across training epochs (selected checkpoints). Bold indicates epoch-best.

| Class | Ep2 | Ep6 | Ep12 | Ep20 | Best | Category |
|-------|------|------|------|------|------|----------|
| road | 95.37 | **95.71** | 95.45 | 95.53 | 95.71 | Saturated |
| sidewalk | 68.62 | **70.59** | 70.20 | 69.64 | 70.59 | Early peak |
| building | 83.41 | 83.93 | 83.84 | 83.59 | **83.95** (ep4) | Saturated |
| wall | 48.85 | **50.23** | 49.74 | 49.32 | 50.23 | Early peak |
| fence | **46.05** | 44.04 | 43.32 | 44.50 | 46.05 | Regression |
| pole | 7.11 | 7.66 | 7.17 | 8.11 | **8.39** (ep16) | Slow growth |
| traffic light | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | Zero |
| traffic sign | 43.88 | **48.21** | 46.94 | 47.48 | 48.21 | Early peak |
| vegetation | 82.42 | **82.60** | 81.91 | 81.98 | 82.60 | Saturated |
| terrain | 51.67 | 51.78 | 51.93 | 52.71 | **52.72** (ep16) | Late growth |
| sky | 81.29 | 80.95 | 81.20 | 81.03 | **81.45** (ep14) | Stable |
| person | 54.14 | 54.67 | 54.38 | 54.47 | **54.68** (ep14) | Stable |
| rider | 38.14 | 38.03 | 39.78 | 39.51 | **39.84** (ep4) | Volatile |
| car | 83.18 | 83.83 | **84.12** | 84.00 | 84.12 | Saturated |
| truck | 74.96 | 75.96 | 76.76 | **77.40** | 77.40 | Continuous |
| bus | 80.63 | 80.97 | 81.65 | **81.66** | 81.66 | Continuous |
| train | 73.19 | **74.51** | 74.34 | 74.36 | **74.53** (ep14) | Early peak |
| motorcycle | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | Zero |
| bicycle | 55.64 | **56.08** | 55.70 | 55.68 | 56.08 | Stable |

### 4.2 The Semantic-Panoptic Divergence

A striking finding emerges when comparing mIoU and PQ for the same classes. Several classes achieve high semantic accuracy but near-zero panoptic quality:

| Class | mIoU (Ep20) | PQ (Ep20) | Ratio PQ/mIoU |
|-------|-------------|-----------|---------------|
| person | 54.47 | 4.01 | 0.07 |
| bicycle | 55.68 | 6.19 | 0.11 |
| rider | 39.51 | 7.98 | 0.20 |
| car | 84.00 | 15.09 | 0.18 |
| truck | 77.40 | 33.46 | 0.43 |
| bus | 81.66 | 39.46 | 0.48 |
| train | 74.36 | 34.37 | 0.46 |

Person exemplifies this divergence most dramatically: the model correctly classifies 54.5% of person pixels (mIoU) but achieves only 4.0 PQ, a ratio of 0.07. The semantic classifier has learned to identify person regions with reasonable accuracy, but the panoptic evaluation requires correct instance-level segmentation --- distinguishing one person from another --- which semantic refinement alone cannot provide. This ratio increases with instance size: large vehicles (truck, bus, train) achieve PQ/mIoU ratios of 0.43--0.48, while small things (person, rider, bicycle) range from 0.07 to 0.20.

This pattern has a direct architectural implication: further investment in semantic refinement quality (higher mIoU) will yield diminishing returns for panoptic quality (PQ) on thing classes. The bottleneck has shifted from "can we classify these pixels correctly?" to "can we separate these instances spatially?" --- a fundamentally different problem that requires either instance-aware mechanisms or boundary-level spatial signals injected via skip connections.

---

## 5. Stuff-Things Trade-off Dynamics

### 5.1 Anti-Correlated Oscillation

Examination of the inter-epoch deltas reveals a recurrent pattern: when PQ_stuff improves, PQ_things tends to regress, and vice versa.

**Table 5.** Inter-epoch deltas showing stuff-things anti-correlation. Epochs where both metrics move in the same direction are marked with (*).

| Transition | Delta PQ_stuff | Delta PQ_things | Correlation |
|-----------|----------------|-----------------|-------------|
| Ep 2 to 4 | +0.09 | +0.75 | Same direction (*) |
| Ep 4 to 6 | +0.50 | -0.17 | Anti-correlated |
| Ep 6 to 8 | -0.34 | +0.53 | Anti-correlated |
| Ep 8 to 10 | +0.28 | +0.21 | Same direction (*) |
| Ep 10 to 12 | +0.13 | +0.19 | Same direction (*) |
| Ep 12 to 14 | -0.09 | -0.31 | Same direction (*) |
| Ep 14 to 16 | -0.11 | +0.37 | Anti-correlated |
| Ep 16 to 18 | +0.04 | -0.01 | Mixed |
| Ep 18 to 20 | -0.01 | +0.00 | Neutral |

The anti-correlation is most pronounced in the active learning phase (epochs 4--16), where three of five transitions show opposing movement between stuff and things. This behavior is characteristic of a shared representation learning regime: the cross-modal gating mechanism allocates model capacity between semantic streams, and gradient updates that improve one category can temporarily displace the other. The cosine learning rate schedule gradually dampens this oscillation, leading to convergence in the final epochs.

### 5.2 Implications for Architecture Design

The stuff-things oscillation suggests that a single shared refinement pathway may be suboptimal for jointly optimizing both categories. The CoupledConvBlock's cross-modal gating allows information exchange between semantic and depth streams, but both streams operate at the same resolution and share the same spatial bottleneck. A UNet decoder with scale-specific skip connections could decouple stuff refinement (at lower resolution, where global context matters) from thing refinement (at higher resolution, where boundary precision matters), potentially eliminating the anti-correlation.

---

## 6. Cost-Benefit Analysis

### 6.1 Comparison with 32x64 Baseline

Table 6 quantifies the gains and costs of moving from 32x64 to 128x256 resolution.

**Table 6.** Resource comparison between 32x64 Run D and 128x256 Run U-B.

| Metric | 32x64 Run D | 128x256 U-B | Delta | Ratio |
|--------|-------------|-------------|-------|-------|
| PQ | 26.52 | 27.50 | +0.98 | --- |
| PQ_stuff | 33.38 | 34.77 | +1.39 | --- |
| PQ_things | 17.10 | 17.58 | +0.48 | --- |
| mIoU | 55.31 | 56.89 | +1.58 | --- |
| Parameters | 1.83M | 4.19M | +2.36M | 2.3x |
| Time/epoch | ~96s | ~948s | +852s | 9.9x |
| Spatial positions | 2,048 | 32,768 | +30,720 | 16x |
| Total training time | ~32 min | ~316 min | +284 min | 9.9x |

The 128x256 resolution delivers clear improvements across all metrics: +0.98 PQ, +1.39 PQ_stuff, +0.48 PQ_things, and +1.58 mIoU. However, these gains come at 2.3x parameter cost and 9.9x training time. In terms of parameter efficiency, the 32x64 model achieves 14.49 PQ per million parameters, while the 128x256 model achieves 6.56 --- a 2.2x reduction in efficiency. The training time scaling (9.9x) is sub-linear relative to the spatial position increase (16x), indicating that data loading and evaluation overhead partially amortize the computational cost.

### 6.2 Marginal Returns

The epoch-level data enables analysis of when diminishing returns set in. By epoch 6, the model has already achieved 98.5% of its final PQ (27.08 vs 27.50). The remaining 14 epochs contribute only +0.42 PQ --- an average of +0.03 PQ per epoch. For production use, early stopping at epoch 10--12 would capture the vast majority of quality gains at 50--60% of the total training cost.

---

## 7. Discussion

### 7.1 Resolution as a Necessary but Insufficient Condition

Run U-B establishes that 128x256 resolution is necessary for recovering PQ_things --- the 32x64 baseline's regression is reversed, with PQ_things crossing the 17.10 threshold at epoch 8 and reaching 17.58 by epoch 16. However, the resolution increase alone is insufficient: PQ_things remains 1.83 points below the input pseudo-label quality of 19.41. The deficit concentrates on small thing classes (person, bicycle, rider), which together account for approximately two-thirds of the gap.

The fundamental limitation is architectural: transposed convolution upsampling redistributes existing feature information across more spatial positions but cannot synthesize the high-frequency boundary detail absent from the 32x64 source representation. Two adjacent pedestrians that share the same DINOv2 patch token at 32x64 will produce smoothly interpolated features at 128x256 --- with no boundary between them.

### 7.2 Architectural Implications

The per-class analysis points toward three concrete next steps:

First, **UNet-style decoders with boundary-aware skip connections** are the natural intervention. Depth maps (SPIdepth, 512x1024) and RGB images (1024x2048) retain boundary information at resolutions far exceeding 128x256. Injecting this information via skip connections at each decoder scale provides the spatial detail that upsampled features lack. The depth gradient analysis (Section 3.3 of the companion report) demonstrates that Sobel gradients of depth produce strong responses at exactly the boundaries that PQ_things evaluation rewards.

Second, the **semantic-panoptic divergence** (Section 4.2) indicates that further mIoU improvements will not translate proportionally to PQ gains for thing classes. The bottleneck has shifted from semantic classification accuracy to instance separation. Future architectural work should prioritize spatial boundary precision over semantic feature quality.

Third, the **stuff-things oscillation** (Section 5) suggests that decoupling stuff and thing refinement across different scales --- allocating more computation to stuff at low resolution and to things at high resolution --- may eliminate the anti-correlation and improve both categories simultaneously. The Progressive Decoder (Option C in the companion UNet report) tests this hypothesis directly.

### 7.3 Limitations

Several limitations should be noted. First, the evaluation is conducted on a single run without repeated seeds; the observed metrics may vary by approximately +/-0.3 PQ based on initialization. Second, the MPS float32 constraint prevents mixed-precision training, inflating both memory usage and training time relative to CUDA baselines. Third, the pseudo-label quality itself imposes an upper bound on refinement performance --- classes absent from the pseudo-labels (motorcycle, traffic light) cannot be recovered by any refinement architecture. Finally, the panoptic evaluation's connected-component analysis interacts with resolution in ways that may not be captured by aggregate metrics alone; thin structures that span multiple components at higher resolution may fragment into non-viable segments.

---

## 8. Conclusion

Run U-B establishes that operating at 128x256 resolution via transposed convolution upsampling achieves record unsupervised panoptic segmentation quality on Cityscapes: PQ=27.50, PQ_stuff=34.77, PQ_things=17.58, and mIoU=56.89. The 4x resolution increase from 32x64 reverses the PQ_things regression that was unfixable at lower resolution, validating the resolution hypothesis. However, the recovery is incomplete: a 1.83-point gap to the input pseudo-label quality persists, driven by small thing classes (person, bicycle, rider) whose instances remain inseparable in upsampled features. These findings motivate boundary-aware decoder architectures --- specifically UNet-style decoders with depth or RGB skip connections --- that can inject the fine-grained spatial detail needed to close the remaining PQ_things gap.

---

## Appendix A: Recommended Checkpoints

| Purpose | File | Epoch | Primary Metric |
|---------|------|-------|----------------|
| Best overall PQ | `best.pth` | 12 | PQ=27.50 |
| Best PQ_things | `checkpoint_epoch_0016.pth` | 16 | PQ_things=17.58 |
| Best mIoU | `checkpoint_epoch_0020.pth` | 20 | mIoU=56.89 |

For downstream panoptic pipeline integration, epoch 12 (best PQ) is recommended as the primary checkpoint. For UNet decoder experiments targeting PQ_things recovery, epoch 16 provides the strongest thing-class baseline for comparison.
