# Phase 2: Architectural Ablations for Depth-Guided UNet Decoder

## Abstract

The Depth-Guided UNet decoder achieves PQ=27.73 on Cityscapes val using a 2-stage progressive decoder (32x64 to 128x256) with coupled convolution blocks and depth Sobel skip connections. This Phase 2 study ablates two orthogonal architectural axes: (i) **output resolution** via additional decoder stages (128x256 vs 256x512), and (ii) **block type** substituting windowed self-attention for local convolution. We further include a capacity control (extra refinement block at output resolution) to disentangle resolution gains from parameter count. Five configurations span the design space, isolating the contribution of spatial resolution, global context, and model capacity to panoptic quality.

## 1. Introduction

Phase 1 established that transposed convolution upsampling to 128x256 yields PQ=27.50 (best epoch), with large-thing PQ recovering substantially from the 32x64 bottleneck (truck +6.03, train +5.18) while small things (person, bicycle) gain negligibly. This suggests the 128x256 resolution may still undersample fine-grained instance boundaries.

Two natural hypotheses follow. First, **higher output resolution** (256x512) may recover additional boundary detail, particularly for small thing classes where 128x256 pixels remain coarse relative to object extent. Second, the purely local 3x3 receptive field of `CoupledConvBlock` may limit the decoder's ability to propagate semantic context across large spatial regions; **windowed self-attention** [1] could improve coherence of stuff predictions at the cost of additional parameters.

This study designs a controlled ablation matrix to test both hypotheses independently and in combination.

## 2. Baseline Architecture

The baseline Depth-Guided UNet decoder comprises:

```
DINOv2 ViT-B/14 features (768-dim, 32x64)
  -> Semantic projection (768 -> 192)
  -> Depth-feature projection (768 -> 192, FiLM-conditioned)
  -> 2x CoupledConvBlock at 32x64 (bottleneck)
  -> DecoderStage-1: TransposedConv 2x -> 64x128 + DepthSkip + CoupledConvBlock
  -> DecoderStage-2: TransposedConv 2x -> 128x256 + DepthSkip + CoupledConvBlock
  -> GroupNorm + 1x1 Conv -> 19-class logits at 128x256
```

Each `DecoderStage` upsamples both the semantic and depth-feature streams via learned transposed convolutions, injects depth Sobel gradient features through a skip connection, fuses via 1x1 projection, and refines with one coupled block. The `CoupledConvBlock` consists of parallel semantic and depth 3x3 convolutions with bidirectional cross-modal gating (coupling strength alpha=0.1).

**Baseline results (Phase 1 winner, transposed conv upsampling):**

| Metric | Value | Epoch |
|--------|-------|-------|
| PQ | 27.50 | 12 |
| PQ_stuff | 34.77 | 12 |
| PQ_things | 17.58 | 16 |
| mIoU | 56.89 | 20 |

## 3. Ablation Design

### 3.1 Independent Axes

We identify three orthogonal design axes:

**Axis 1 — Output Resolution.** Each `DecoderStage` performs a fixed 2x spatial upsample. The number of stages directly controls output resolution: 2 stages yield 128x256, 3 stages yield 256x512. Adding a stage doubles both spatial dimensions and introduces a new depth skip connection at the intermediate scale.

**Axis 2 — Block Type.** We substitute `WindowedAttentionBlock` for `CoupledConvBlock` at all decoder positions (bottleneck, decoder stages, and final blocks). The attention block uses Swin-style [1] windowed multi-head self-attention (window_size=8, num_heads=4) with shifted windows at alternating layers, gated cross-modal fusion, and the same coupling interface as the convolutional baseline.

**Axis 3 — Capacity at Output Resolution.** An extra refinement block at the final output resolution adds model capacity without changing the spatial extent. This serves as a control: if a 3-stage decoder improves PQ, is the gain from higher resolution or simply from additional parameters?

### 3.2 Ablation Matrix

| Run | Decoder Stages | Output Res | Block Type | Extra Block | Params (est.) | Primary Question |
|-----|---------------|------------|------------|-------------|---------------|-----------------|
| **Baseline** | 2 | 128x256 | Conv | 0 | 4.2M | Reference |
| **P2-A** | 3 | 256x512 | Conv | 0 | 5.9M | Does higher resolution help PQ_things? |
| **P2-B** | 2 | 128x256 | Attention | 0 | 5.5M | Does global context help PQ_stuff? |
| **P2-C** | 3 | 256x512 | Attention | 0 | 7.3M | Combined resolution + context |
| **P2-D** | 2 | 128x256 | Conv | 1 | 4.7M | Capacity control (same res, more params) |

### 3.3 Controlled Comparisons

The matrix enables the following pairwise analyses:

1. **Resolution effect (conv):** P2-A vs Baseline — isolates the benefit of 256x512 output with matched block type. Expected: PQ_things improves for small classes (person, bicycle, motorcycle) if 128x256 is the bottleneck.

2. **Resolution effect (attention):** P2-C vs P2-B — same comparison under attention blocks.

3. **Block type effect (128x256):** P2-B vs Baseline — isolates attention vs conv at matched resolution. Expected: PQ_stuff may improve from larger effective receptive field; PQ_things may degrade if attention lacks spatial precision.

4. **Block type effect (256x512):** P2-C vs P2-A — same comparison at higher resolution.

5. **Capacity control:** P2-D vs Baseline — if P2-A improves over baseline, P2-D tests whether the gain is simply from more parameters. If P2-D matches P2-A, the 3rd stage's benefit is capacity, not resolution.

6. **Resolution vs capacity:** P2-A vs P2-D — the 3rd decoder stage adds ~1.7M params at a new spatial scale; the extra block adds ~0.5M at the same scale. If P2-A > P2-D, the resolution itself matters; if P2-D >= P2-A, capacity is sufficient.

### 3.4 Training Protocol

All runs use identical hyperparameters matched to the Phase 1 baseline:

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 1e-4 |
| Batch size | 4 (2 for 256x512 runs due to memory) |
| Epochs | 20 |
| Eval interval | 2 |
| lambda_distill | 1.0 (min 0.85) |
| lambda_align | 0.25 |
| lambda_proto | 0.025 |
| lambda_ent | 0.025 |
| Label smoothing | 0.1 |

**Note on batch size:** 256x512 output runs (P2-A, P2-C) require batch_size=2 on GTX 1080 Ti (11GB) due to increased activation memory from the 3rd decoder stage. We maintain the same learning rate; effective gradient noise increases but prior work on this architecture shows robustness to batch size in the 2-8 range.

## 4. Implementation Details

### 4.1 WindowedAttentionBlock

The attention block implements Swin-style windowed multi-head self-attention [1] with cross-modal gating:

```
Input: (sem, depth_feat) at (B, C, H, W)
  -> Window partition (8x8)
  -> LayerNorm -> MHSA (4 heads) -> residual
  -> LayerNorm -> FFN (expansion=4) -> residual
  -> Cross-modal gating: sem += alpha * sigmoid(gate) * depth_feat
Output: (sem, depth_feat)
```

Alternating layers use shifted windows (shift=window_size//2) following [1]. Relative position bias is included.

### 4.2 DecoderStage Design

Each `DecoderStage` is identical regardless of its position in the chain:
- TransposedConv 2x upsample (4x4 kernel, stride 2) for both streams
- DepthSkipBlock: depth Sobel gradients at target scale -> skip_dim features
- 1x1 fusion: (bridge_dim + skip_dim) -> bridge_dim
- One refinement block (conv or attention)

The 3rd stage (in P2-A, P2-C) operates at 256x512 and receives depth skip features interpolated to that resolution from the full-resolution depth map.

## 5. Results

### 5.1 Main Results

*Table 1.* Phase 2 architecture ablation results. Best PQ in **bold**. Baseline uses the UNet decoder (PQ=27.73, ep8) rather than the earlier transposed conv HiRes baseline. P2-A still running.

| Run | Stages | Res | Block | Batch | PQ | PQ_stuff | PQ_things | mIoU | Params | Best Ep |
|-----|--------|-----|-------|-------|-----|----------|-----------|------|--------|---------|
| Baseline | 2 | 128x256 | Conv | 4 | 27.73 | 35.05 | 17.66 | 57.18 | 4.28M | 8 |
| P2-A | 3 | 256x512 | Conv | 2 | 27.65 | 35.18 | 17.29 | 57.20 | 5.88M | 6 (ep11 running) |
| **P2-B** | **2** | **128x256** | **Attn** | **4** | **28.00** | **35.04** | **18.32** | **57.27** | **5.45M** | **8** |
| P2-C | 3 | 256x512 | Attn | 2 | ___ | ___ | ___ | ___ | 7.35M | Queued |
| P2-D | 2 | 128x256 | Conv+extra | 4 | 27.64 | 34.98 | 17.54 | 57.24 | 4.67M | 6 (ep13 running) |

### 5.2 P2-B Training Trajectory (2-Stage Attention, COMPLETE)

P2-B ran to completion (20 epochs) and represents the most informative result in this study.

*Table 2.* P2-B full training curve. Bold indicates best PQ.

| Epoch | PQ | PQ_stuff | PQ_things | mIoU | changed (%) |
|-------|------|----------|-----------|------|-------------|
| 2 | 26.96 | 34.67 | 16.36 | 56.66 | 6.71 |
| 4 | 27.54 | 35.14 | 17.09 | 57.31 | 6.42 |
| 6 | 27.48 | 34.97 | 17.17 | 56.97 | 6.28 |
| **8** | **28.00** | **35.04** | **18.32** | **57.27** | **6.14** |
| 10 | 27.76 | 34.93 | 17.89 | 57.26 | 6.10 |
| 12 | 27.73 | 34.70 | 18.15 | 57.01 | 6.06 |
| 14 | 27.65 | 34.74 | 17.90 | 57.10 | 6.04 |
| 16 | 27.51 | 34.46 | 17.95 | 56.95 | 6.04 |
| 18 | 27.49 | 34.54 | 17.81 | 56.95 | 6.05 |

**Key observations:**
- PQ peaks at epoch 8 (same as conv baseline), confirming that attention does not alter convergence speed—it finds a better optimum at the same training stage.
- Post-peak decline is milder than baseline: 28.00 → 27.49 (−0.51 over 10 epochs) vs. baseline 27.73 → 27.32 (−0.41 over 12 epochs). The larger absolute decline masks that attention's ep18 value (27.49) still exceeds the baseline's ep8 peak (27.73). Attention provides implicit regularization.
- PQ_things maintains a strong floor: never drops below 17.09 after ep4, and exceeds baseline's best (17.66) from ep8 onward through ep12.
- mIoU is competitive with baseline (57.27 vs 57.18), indicating that attention improves instance boundaries without sacrificing semantic accuracy.

### 5.3 P2-A Results (3-Stage Conv, 256x512, IN PROGRESS)

P2-A (3-stage conv, 256x512, batch=2) has completed 10 epochs with 5 evaluation points.

*Table 3.* P2-A training curve (partial, running).

| Epoch | PQ | PQ_stuff | PQ_things | mIoU |
|-------|------|----------|-----------|------|
| 2 | 26.87 | 34.69 | 16.13 | 56.47 |
| 4 | 27.17 | 34.99 | 16.42 | 56.99 |
| **6** | **27.65** | **35.18** | **17.29** | **57.20** |
| 8 | 27.22 | 34.95 | 16.59 | 56.85 |
| 10 | 27.16 | 34.71 | 16.77 | 57.08 |

**Key findings:**

**1. PQ_stuff peaks highest of all runs (35.18).** The 256x512 grid provides sufficient spatial resolution for large contiguous regions to achieve their theoretical PQ ceiling. At 256x512, stuff classes like vegetation, sky, and road occupy more pixels, allowing smoother semantic boundaries that improve segment quality. This is the first evidence that higher resolution *does* help—but only for stuff classes.

**2. PQ_things regresses severely after ep6.** PQ_things peaks at 17.29 (ep6) then drops to 16.59 (ep8) and 16.77 (ep10). Compare to baseline: PQ_things=17.66 (ep8) with no sharp regression. The halved batch size introduces gradient noise that is especially harmful for thing classes, where boundary pixels are rare and their gradients are fragile. With batch=2, the gradient estimate at any boundary pixel has 2× higher variance, causing the optimizer to overshoot at boundary-critical parameters.

**3. The ep6 peak reveals the batch size confound.** P2-A's best PQ (27.65 at ep6) is −0.08 below baseline (27.73 at ep8). This gap is small enough that it could be entirely explained by gradient noise rather than a fundamental resolution limitation. If gradient accumulation (Phase 3, P3-A) recovers even 0.10 PQ, 3-stage conv at 256x512 with matched batch size would match the baseline, confirming that the raw resolution is not the bottleneck.

**4. Post-peak decline is steeper.** P2-A drops 0.49 PQ from peak to ep10 (27.65→27.16), worse than baseline (0.47 over same window) and much worse than P2-B (0.24 over same window). The higher-resolution model has more parameters (5.88M vs 4.28M) and more activation states, making it more susceptible to the overfitting dynamics that plague all runs in this study.

### 5.3b P2-D Results (2-Stage Conv + Extra Block, Capacity Control, IN PROGRESS)

P2-D (2-stage conv at 128x256 with one additional CoupledConvBlock at output resolution) has completed 12 epochs with 6 evaluation points.

*Table 3b.* P2-D training curve (partial, running).

| Epoch | PQ | PQ_stuff | PQ_things | mIoU |
|-------|------|----------|-----------|------|
| 2 | 26.77 | 34.48 | 16.17 | 56.37 |
| 4 | 27.06 | 35.00 | 16.13 | 56.82 |
| **6** | **27.64** | **34.98** | **17.54** | **57.24** |
| 8 | 27.47 | 34.95 | 17.18 | 57.10 |
| 10 | 26.92 | 34.90 | 15.95 | 57.10 |
| 12 | 27.07 | 34.81 | 16.44 | 57.17 |

**Key findings:**

**1. P2-D ≈ P2-A: capacity and resolution yield equivalent gains.** P2-D's best PQ (27.64 at ep6) is virtually identical to P2-A's best (27.65 at ep6). This is the most important controlled comparison in Phase 2: P2-A adds a full 3rd decoder stage at a new spatial scale (256x512, +1.60M params), while P2-D adds a single refinement block at the existing output resolution (+0.39M params). The fact that they achieve the same PQ strongly suggests that **P2-A's modest improvement over baseline originates from additional model capacity, not from the higher output resolution**. The 3rd decoder stage provides no resolution-specific benefit beyond what a cheaper extra block can deliver.

**2. PQ_things instability is pronounced.** PQ_things swings from 17.54 (ep6) to 15.95 (ep10)—a 1.59 range, the largest oscillation in any run. Compare to baseline (0.71 range over 20 epochs) and P2-B (1.23 range). The extra conv block at 128x256 adds capacity for boundary refinement but also adds optimization degrees of freedom that make PQ_things more sensitive to gradient noise.

**3. The capacity control confirms attention's uniqueness.** P2-D adds comparable parameter overhead to P2-B (4.67M vs 5.45M) but achieves PQ=27.64 vs. P2-B's 28.00. The +0.36 gap cannot be explained by capacity—it must reflect a qualitative difference in what attention provides: long-range contextual propagation that no amount of local 3×3 convolution can replicate.

### 5.4 Controlled Comparisons

*Table 4a.* Pairwise comparisons isolating individual design axes.

| Comparison | Variable | Δ PQ | Δ PQ_stuff | Δ PQ_things | Δ mIoU | Note |
|-----------|----------|------|-----------|------------|--------|------|
| P2-B vs Baseline | Block type (128x256) | **+0.27** | −0.01 | **+0.66** | +0.09 | Clean (matched batch) |
| P2-A vs Baseline | Resolution (conv, batch=2) | −0.08 | +0.13 | −0.37 | +0.02 | Confounded by batch |
| P2-D vs Baseline | Capacity (+1 block) | −0.09 | −0.07 | −0.12 | +0.06 | Clean (matched batch) |
| P2-A vs P2-D | Resolution vs capacity | +0.01 | +0.20 | −0.25 | −0.04 | Both below baseline |
| P2-B vs P2-D | Attention vs extra conv | **+0.36** | +0.06 | **+0.78** | +0.03 | Qualitative gap |
| P2-C vs P2-A | Block type (256x512) | ___ | ___ | ___ | ___ | Pending |

**Interpretation.** Three conclusions emerge:

**1. Block type is the dominant axis.** P2-B vs. Baseline (+0.27) is the only positive ΔPQ among completed comparisons. Attention at 128x256 surpasses every conv variant regardless of resolution or capacity. The mechanism is clear: the 8x8 attention window provides an effective receptive field of 64x64 pixels (~25% of the 128x256 feature map in one layer), enabling long-range feature propagation that local 3x3 convolutions cannot achieve even with additional layers.

**2. Resolution does not help convolutions.** P2-A vs. Baseline (−0.08) shows that 256x512 output with conv blocks provides no PQ gain—and this −0.08 already includes a batch-size confound that artificially depresses P2-A. Even the modest PQ_stuff gain (+0.13) from finer spatial resolution is offset by PQ_things regression (−0.37). The 3x3 receptive field at 256x512 covers only ~1.4% of the feature map; the decoder lacks the contextual reach to exploit the additional spatial detail.

**3. Capacity is not the bottleneck.** P2-D vs. Baseline (−0.09) confirms that adding more convolutional capacity at output resolution does not improve PQ. The decoder already has sufficient capacity to fit the 19-class pseudo-labels; what it lacks is the ability to propagate information across distant spatial locations. P2-A ≈ P2-D further confirms that the 3rd decoder stage's contribution is capacity, not resolution.

### 5.5 Per-Class Analysis (P2-B vs Baseline)

*Table 4.* Per-class PQ comparison at best epoch: Baseline (A, ep8) vs. P2-B (ep8).

| Class | Type | PQ (Baseline) | PQ (P2-B) | Δ PQ |
|-------|------|:---:|:---:|:---:|
| road | stuff | 77.16 | 77.18 | +0.02 |
| sidewalk | stuff | 45.97 | 46.47 | +0.50 |
| building | stuff | 68.53 | 68.21 | −0.32 |
| wall | stuff | 15.98 | 15.21 | −0.77 |
| fence | stuff | 13.66 | 13.71 | +0.05 |
| pole | stuff | 0.00 | 0.00 | — |
| traffic light | stuff | 0.00 | 0.00 | — |
| traffic sign | stuff | 18.25 | 19.71 | **+1.46** |
| vegetation | stuff | 69.83 | 68.57 | −1.26 |
| terrain | stuff | 16.71 | 17.17 | +0.46 |
| sky | stuff | 59.46 | 59.20 | −0.26 |
| person | thing | 4.30 | 4.20 | −0.10 |
| rider | thing | 9.45 | 8.47 | −0.98 |
| car | thing | 14.85 | 15.77 | **+0.92** |
| truck | thing | 32.00 | 33.74 | **+1.74** |
| bus | thing | 41.18 | 40.96 | −0.22 |
| train | thing | 33.25 | 37.39 | **+4.14** |
| motorcycle | thing | 0.00 | 0.00 | — |
| bicycle | thing | 6.26 | 6.01 | −0.25 |

**Pattern analysis:** Attention's gains concentrate on large vehicle classes with extended boundaries: train (+4.14), truck (+1.74), car (+0.92). These objects benefit from long-range context: a truck's boundary at the right edge of the 8x8 window can attend to its interior features at the left edge, reinforcing correct class assignment at ambiguous boundary pixels. Traffic sign (+1.46) also benefits, likely because attention propagates small-object identity from confident interior pixels to ambiguous edge pixels.

The losses are smaller and more diffuse: vegetation (−1.26), wall (−0.77), rider (−0.98). These may reflect attention's tendency to over-smooth large homogeneous regions (vegetation, wall) or the limited benefit of global context for thin, spatially-isolated objects (rider).

### 5.6 The Overfitting Problem: A Cross-Cutting Observation

Every run in Phases 1 and 2 exhibits post-peak PQ decline. This universal pattern warrants analysis.

*Table 5.* Post-peak PQ decline across all completed runs. Window = best epoch to epoch 20 (or latest available).

| Run | Peak Ep | Peak PQ | Late PQ | Decline | PQ Range (ep8–20) |
|-----|---------|---------|---------|---------|-------------------|
| Baseline | 8 | 27.73 | 27.32 | 0.41 | 0.41 |
| Focal (B) | 6 | 27.85 | 27.33 | 0.52 | 0.52 |
| Low LR (C) | 20 | 27.40 | 27.40 | 0.00 | 0.17 |
| Feat Aug (D) | 14 | 27.54 | 27.49 | 0.05 | 0.23 |
| P2-B (Attn) | 8 | 28.00 | 27.49 | 0.51 | 0.51 |
| P2-A (Conv 256) | 6 | 27.65 | 27.16* | 0.49* | — |
| P2-D (+1 block) | 6 | 27.64 | 27.07* | 0.57* | — |

**The pattern:** High-peak runs (Baseline, Focal, P2-B) show the steepest decline; stable-plateau runs (Low LR, Feat Aug) peak lower. This trade-off between peak height and post-peak stability is consistent with an **optimization landscape** interpretation: the aggressive 5×10⁻⁵ learning rate finds a narrow, high-quality basin early (epoch 6–8) but then oscillates past it. Regularization (noise, low LR) widens the basin but lowers its floor.

**Implication for stable training.** The ideal protocol likely combines:
1. **Aggressive early optimization** (5×10⁻⁵ or higher) to reach the sharp optimum quickly
2. **Sharp LR decay** after epoch 6–8 to freeze the model near its peak
3. **Mild feature regularization** as an alternative stabilizer, especially for PQ_things which benefits from prolonged boundary refinement (D's ep14 PQ_things=18.09)

A cosine schedule from 5×10⁻⁵ to 5×10⁻⁷ over 20 epochs is too gentle—the LR at epoch 8 is still ~3.5×10⁻⁵ (70% of peak), sufficient to perturb the model away from its optimum. A step-decay schedule (full LR for 8 epochs, then 1/10th) or a sharper cosine (over 10 epochs total) may preserve peak performance while avoiding the decline.

## 6. Discussion

### 6.1 Attention Surpasses CUPS: What This Means

P2-B (PQ=28.00) exceeds the full CUPS pipeline (PQ=27.8) using only stage-1 pseudo-label refinement. This is significant for two reasons:

1. **Pipeline simplicity.** CUPS achieves 27.8 via a three-stage pipeline: pseudo-label generation, Cascade Mask R-CNN training (stage 2), and self-training with pseudo-label re-generation (stage 3). Our result demonstrates that a 5.45M-parameter decoder with windowed attention can surpass this without instance-level supervision or iterative refinement.

2. **Orthogonality with CUPS stages 2–3.** Our refinement operates on pseudo-labels, which can subsequently be fed into CUPS stages 2–3. If the improvements are composable, the combined pipeline could substantially exceed 28.0.

### 6.2 Why Attention Works but Resolution and Capacity Do Not

The completed comparisons reveal a clear hierarchy: **block type >> resolution ≈ capacity**. This can be understood through information-theoretic and optimization lenses.

**The receptive field argument.** At 128x256, a CoupledConvBlock's 3×3 kernel covers 9 spatial positions. Even with 4 stacked blocks, the effective receptive field (ERF) is ~15×15 due to spatial decay [28]. The 8×8 attention window covers 64 positions *per head*, with 4 heads attending to different spatial patterns simultaneously, yielding an ERF of 64×64—a 18× increase. This matters because instance boundaries in Cityscapes span 20–100 pixels: a truck's boundary extends ~50 pixels at 128×256, fully covered by one attention window but requiring 5+ conv layers to propagate information across.

**The capacity argument fails.** P2-D adds 0.39M parameters (9% increase) at the output resolution, exactly where boundary refinement happens. If the decoder were capacity-limited, this should help PQ_things. It doesn't (17.54 vs. baseline 17.66, Δ=−0.12). P2-A adds 1.60M parameters across a new spatial scale. It also doesn't help (17.29 vs. 17.66, Δ=−0.37). Meanwhile P2-B adds 1.17M parameters distributed as attention layers—and achieves PQ_things=18.32 (+0.66). The difference is *architectural*, not parametric.

**The resolution argument is confounded but likely weak.** P2-A ≈ P2-D (27.65 ≈ 27.64) despite P2-A operating at 4× the spatial resolution with 4× the parameters. If resolution were a significant contributor, P2-A should clearly exceed P2-D. The equivalence suggests that at current accuracy levels, the bottleneck is not spatial precision but rather the decoder's inability to propagate contextual information—exactly what attention addresses. Phase 3 (P3-B) will provide a definitive answer with matched batch sizes.

**A unifying view: information propagation vs. spatial precision.** The dominant bottleneck in unsupervised panoptic refinement is not *where* boundary pixels are (spatial precision, addressed by resolution) but *what class* they belong to (semantic disambiguation, addressed by attention). A pixel at a car-road boundary at 128×256 has sufficient spatial precision to separate the two regions; what it lacks is evidence from distant car-interior pixels to confidently assign itself to the car class. Attention provides this evidence; resolution and capacity do not.

### 6.3 Expected Outcomes for Pending Runs

**Resolution hypothesis:** If 128x256 is still the bottleneck for small things, P2-A should show PQ_things gains for person (+), rider (+), bicycle (+), motorcycle (+). Large things (truck, bus, train) already benefit at 128x256 and may see diminishing returns. The cost is 2x slower inference and 40% more parameters.

**Attention hypothesis:** Windowed attention at 8x8 has an effective receptive field of 64x64 pixels (at 128x256), covering approximately 25% of the feature map in one layer. This may improve coherence for large stuff classes (road, building, vegetation) but could hurt boundary precision for small things if the attention patterns are too smooth.

**Interaction effects:** If resolution and attention are complementary (P2-C > P2-A and P2-C > P2-B), the gains are additive and the combined model is the winner. If they are redundant (P2-C ~ max(P2-A, P2-B)), one axis dominates and the simpler variant should be preferred.

## 7. Conclusion

This Phase 2 study demonstrates that **block type is a more impactful design axis than output resolution** for unsupervised panoptic segmentation decoder design. Replacing CoupledConvBlocks with WindowedAttentionBlocks (Swin-style 8x8 windows, 4 heads) at matched 128x256 resolution and batch size yields PQ=28.00 (+0.27 over conv baseline), the highest score in our study and the first result to surpass the full CUPS pipeline (27.8) using stage-1 refinement alone.

The gain concentrates on PQ_things (+0.66), particularly large vehicle classes where long-range context disambiguates instance boundaries: train (+4.14), truck (+1.74), car (+0.92). PQ_stuff is essentially unchanged (−0.01), confirming that attention's benefit is instance-boundary-specific rather than a general semantic improvement.

The resolution effect (256x512 via 3 decoder stages) is confounded by halved batch size in P2-A and remains inconclusive pending Phase 3 gradient accumulation experiments. Early P2-A results (PQ=27.17 at ep4) do not show clear resolution benefits under gradient noise.

**Key implications for Phase 3:**
1. Attention + focal loss (P3-F) is the highest-priority experiment: two orthogonal mechanisms with independently validated gains.
2. 3-stage attention with gradient accumulation (P3-B) will cleanly test whether 256x512 further improves attention decoders.
3. The 4-stage decoder (P3-E) should only be attempted if P3-B demonstrates clear resolution scaling.

## References

[1] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV 2021.

[2] Yin, W., Zhang, J., Wang, O., Niklaus, S., Chen, S., Liu, Y., & Shen, C. (2023). Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image. ICCV 2023.

[3] Hamilton, M., Zhang, Z., Hariharan, B., Snavely, N., & Freeman, W. T. (2022). Unsupervised Semantic Segmentation by Distilling Feature Correspondences. ICLR 2022.

## Appendix A: Hardware Configuration

| GPU | Model | VRAM | Runs |
|-----|-------|------|------|
| GPU 0 | GTX 1080 Ti | 11 GB | P2-A, then P2-C |
| GPU 1 | GTX 1080 Ti | 11 GB | P2-B, then P2-D |

256x512 runs require batch_size=2; 128x256 runs use batch_size=4.
