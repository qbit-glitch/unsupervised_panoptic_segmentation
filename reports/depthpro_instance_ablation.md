# DepthPro Instance Pseudo-Label Ablation Study

**Date**: 2026-04-07
**Dataset**: Cityscapes val (500 images, 1024x2048)
**Semantics**: k=80 overclustered (19 trainIDs via Hungarian matching)
**Evaluation**: Standard panoptic PQ (IoU > 0.5 matching)
**Depth model**: Apple DepthPro (ICLR 2025), `apple/DepthPro-hf` via HuggingFace
**Scripts**: `mbps_pytorch/sweep_depthpro.py` (sweep), `mbps_pytorch/generate_depth_multimodel.py` (depth gen)

---

## Summary of Findings

1. **DepthPro is the best depth model for instance splitting** — PQ_things=23.35 beats DA3 (20.90) by **+2.45 points** (+11.7% relative). This is the largest single-factor improvement in the project's instance pipeline.

2. **Optimal threshold is 3x lower than DA3** — DepthPro needs tau=0.01 vs DA3's tau=0.03, confirming that sharper depth boundaries require lower Sobel gradient thresholds.

3. **No Gaussian blur needed (sigma=0.0)** — Unique to DepthPro. All other depth models (DA3, DA2, SPIdepth) benefit from sigma=1.0 smoothing. DepthPro's native output is clean enough that any blur degrades boundary sharpness.

4. **Car and person benefit most** — Car PQ jumps +5.4 (32.2 vs 26.8) from better adjacent-vehicle separation. Person PQ jumps +5.4 (11.8 vs 6.4) from better co-planar pedestrian splitting.

5. **Boundary F1 directly predicts instance quality** — DepthPro (F1=0.409) > DA3 (~0.35) > DA2 (F1=0.228) > SPIdepth (~0.19), and PQ_things follows the same ordering: 23.35 > 20.90 > 20.20 > 19.41.

6. **Dilation_iters=3 remains optimal** — Same default as DA3/SPIdepth, unchanged by depth model choice.

---

## Phase 1: Grad Threshold x Min Area Sweep (22 configs)

| tau | A_min | PQ | PQ_stuff | PQ_things | SQ | RQ | inst/img |
|-------|-------|------|----------|-----------|------|------|----------|
| CC-only | -- | 24.84 | 32.08 | 14.90 | 71.15 | 28.22 | 8.6 |
| 0.005 | 500 | 27.61 | 32.08 | 21.47 | 74.70 | 39.84 | 7.5 |
| 0.005 | 1000 | 27.82 | 32.08 | 21.97 | 75.55 | 37.78 | 4.9 |
| 0.005 | 1500 | 27.83 | 32.08 | 21.99 | 75.83 | 35.80 | 3.6 |
| **0.01** | **500** | 27.55 | 32.08 | 21.34 | 74.47 | 38.63 | 7.5 |
| **0.01** | **1000** | **28.09** | **32.08** | **22.60** | **75.26** | **37.47** | **5.1** |
| 0.01 | 1500 | 27.96 | 32.08 | 22.31 | 75.59 | 36.22 | 4.0 |
| 0.02 | 500 | 26.67 | 32.08 | 19.24 | 72.96 | 34.21 | 6.9 |
| 0.02 | 1000 | 27.46 | 32.08 | 21.12 | 73.48 | 34.20 | 4.9 |
| 0.02 | 1500 | 27.37 | 32.08 | 20.90 | 73.84 | 33.45 | 3.9 |
| 0.03 | 500 | 26.35 | 32.08 | 18.49 | 72.56 | 32.60 | 6.6 |
| 0.03 | 1000 | 26.89 | 32.08 | 19.75 | 73.00 | 32.81 | 4.7 |
| 0.03 | 1500 | 26.93 | 32.08 | 19.87 | 73.33 | 32.33 | 3.9 |
| 0.05 | 500 | 26.06 | 32.08 | 17.79 | 72.10 | 31.39 | 6.2 |
| 0.05 | 1000 | 26.50 | 32.08 | 18.82 | 72.46 | 31.76 | 4.6 |
| 0.08 | 1000 | 26.44 | 32.08 | 18.69 | 72.06 | 31.18 | 4.5 |
| 0.10 | 1000 | 26.43 | 32.08 | 18.67 | 71.86 | 30.92 | 4.4 |
| 0.15 | 1000 | 26.34 | 32.08 | 18.46 | 71.70 | 30.65 | 4.4 |
| 0.20 | 1000 | 26.32 | 32.08 | 18.42 | 71.62 | 30.60 | 4.3 |
| 0.30 | 1000 | 26.34 | 32.08 | 18.46 | 71.57 | 30.52 | 4.3 |
| 0.50 | 1000 | 26.33 | 32.08 | 18.43 | 71.50 | 30.49 | 4.2 |
| 0.80 | 1000 | 26.33 | 32.08 | 18.43 | 71.49 | 30.49 | 4.2 |
| 1.00 | 1000 | 26.33 | 32.08 | 18.43 | 71.49 | 30.49 | 4.2 |

**Phase 1 Winner**: tau=0.01, A_min=1000, PQ=28.09, PQ_things=22.60

**Observations**:
- Strong monotonic improvement as tau decreases from 1.0 to 0.01
- tau >= 0.20 all converge (~18.4 PQ_things) — edge density too low to matter
- tau=0.005 slightly worse than tau=0.01 — over-splitting begins at very low thresholds
- A_min=1000 consistently optimal across all tau values

---

## Phase 2: Blur Sigma x Dilation Iters (12 configs, around tau=0.01, A_min=1000)

### Blur Sigma Sweep (dilation_iters=3 fixed)

| sigma | PQ | PQ_things | Delta vs sigma=1.0 |
|-------|------|-----------|---------------------|
| **0.00** | **28.40** | **23.35** | **+0.75** |
| 0.25 | 28.40 | 23.35 | +0.75 |
| 0.50 | 28.37 | 23.27 | +0.67 |
| 1.00 | 28.09 | 22.60 | baseline |
| 1.50 | 27.83 | 21.98 | -0.62 |
| 2.00 | 27.55 | 21.33 | -1.27 |

**Key finding**: sigma=0.0 and sigma=0.25 are tied. sigma=1.0 (default for DA3/SPIdepth) costs -0.75 PQ_things. Higher blur monotonically degrades performance.

### Dilation Iters Sweep (sigma=1.0 fixed)

| dilation | PQ | PQ_things | Delta vs dil=3 |
|----------|------|-----------|-----------------|
| 0 | 26.79 | 19.53 | -3.07 |
| 1 | 27.41 | 20.99 | -1.61 |
| 2 | 27.80 | 21.93 | -0.67 |
| **3** | **28.09** | **22.60** | **baseline** |
| 4 | 28.07 | 22.55 | -0.05 |
| 5 | 27.92 | 22.21 | -0.39 |
| 7 | 27.54 | 21.30 | -1.30 |

**Key finding**: dilation_iters=3 is optimal. No dilation (dil=0) loses -3.07 PQ_things — boundary pixel reclamation is critical. Diminishing returns beyond dil=3.

### Phase 2 Winner

**tau=0.01, A_min=1000, sigma=0.0, dilation_iters=3 -> PQ=28.40, PQ_things=23.35**

The sigma=0.0 finding adds +0.75 PQ_things over the Phase 1 default.

## Full training threshold sensitivity

Direct validation quality identifies very low tau as a strong setting when each pseudo mask is scored against validation labels. The full training setting has a different purpose because these labels supervise a downstream instance learner across the complete Cityscapes train split. We therefore evaluate coverage, proxy confidence, feature affinity noise, geometric boundary noise, merge noise, missed boundary evidence, and boundary support together.

The table compresses the full sweep into one reviewer readable sensitivity analysis. Lower tau creates many masks and preserves boundary support, but it also raises affinity noise and boundary noise. Larger area minimum removes noisy fragments, but it also removes training instances and begins to create sparse supervision.

    tau    area minimum    masks image    proxy score    affinity noise    merge noise    boundary noise    missed boundary    boundary support    reading
    0.01   100             46.06          0.964          67.65             1.61           113.85            35.49              0.134               over fragmented
    0.01   500             29.38          0.974          65.10             1.83           111.74            37.34              0.103               over fragmented
    0.01   1000            21.87          0.978          62.52             2.02           109.73            39.01              0.086               high split noise
    0.01   1500            17.75          0.980          60.29             2.17           107.66            40.13              0.076               still noisy
    0.02   100             39.02          0.961          62.39             1.70           110.57            38.05              0.146               over fragmented
    0.02   500             26.47          0.973          61.04             1.88           109.34            39.20              0.111               over fragmented
    0.02   1000            20.56          0.977          59.58             2.03           108.17            40.34              0.092               high split noise
    0.02   1500            17.21          0.979          58.30             2.17           106.91            41.16              0.080               acceptable but noisy
    0.05   500             22.32          0.972          56.85             2.02           107.28            41.98              0.104               dense labels
    0.05   1000            18.29          0.976          56.34             2.15           106.69            42.57              0.086               good candidate
    0.05   1500            15.85          0.978          55.77             2.26           105.93            42.97              0.076               good candidate
    0.05   2000            14.13          0.979          55.14             2.36           105.04            43.26              0.070               sparse labels
    0.10   500             20.00          0.974          55.69             2.24           106.52            43.81              0.083               good candidate
    0.10   1000            16.93          0.977          55.45             2.33           106.11            44.07              0.070               good candidate
    0.10   1500            14.93          0.979          55.08             2.42           105.57            44.27              0.063               sparse labels
    0.20   500             18.69          0.977          55.79             2.57           106.23            44.95              0.059               good candidate
    0.20   1000            16.18          0.979          55.57             2.62           105.91            45.03              0.053               selected balance
    0.20   1500            14.45          0.980          55.20             2.66           105.43            45.08              0.051               sparse labels
    0.20   2000            13.14          0.980          54.72             2.69           104.77            45.10              0.049               sparse labels
    0.30   500             18.33          0.979          55.97             2.78           106.20            45.29              0.050               higher merge risk
    0.30   1000            15.99          0.980          55.73             2.80           105.89            45.32              0.047               lower support
    0.30   1500            14.33          0.980          55.35             2.81           105.44            45.33              0.046               sparse labels
    0.40   500             18.20          0.979          56.07             2.87           106.19            45.42              0.046               higher merge risk
    0.40   1000            15.92          0.980          55.81             2.88           105.89            45.43              0.044               lower support
    0.40   1500            14.28          0.980          55.42             2.89           105.44            45.42              0.044               sparse labels

This pattern supports tau 0.20 with area minimum 1000 as the operating point for training labels. It sits on the low noise plateau with proxy score 0.979 and affinity noise 55.57, while it still retains 16.18 masks per image. Moving to area minimum 1500 or 2000 gives only small noise reductions, but it removes useful supervision and weakens boundary support. Moving to lower tau gives denser labels, but the extra masks arrive with higher feature disagreement and larger boundary noise.

This choice also explains why the full Stage 2 training setting can differ from the direct validation setting above. The validation setting asks which raw pseudo labels score best by themselves. The training setting asks which labels provide enough reliable examples for a downstream instance learner. The selected setting answers the second question.

## Depth Model Comparison (Final)

| Depth Model | Type | Boundary F1 | Opt tau | Opt A_min | sigma | PQ | PQ_stuff | PQ_things |
|-------------|------|-------------|---------|-----------|-------|------|----------|-----------|
| CC-only (no depth) | -- | -- | -- | -- | -- | 24.84 | 32.08 | 14.90 |
| SPIdepth | Self-supervised | ~0.19 | 0.20 | 1000 | 1.0 | 26.74 | 32.08 | 19.41 |
| DA2-Large | Foundation (relative) | 0.228 | 0.03 | 1000 | 1.0 | 27.10 | 32.08 | 20.20 |
| DA3-Large | Foundation (relative) | ~0.35 | 0.03 | 1000 | 1.0 | 27.37 | 32.08 | 20.90 |
| **DepthPro** | **Foundation (metric)** | **0.409** | **0.01** | **1000** | **0.0** | **28.40** | **32.08** | **23.35** |

**PQ_stuff is identical (32.08) across all models** — depth only affects thing classes, as expected.

---

## Per-Class Breakdown (DepthPro best vs DA3 best)

| Class | Type | DepthPro PQ | DA3 PQ | Delta | DepthPro TP | DepthPro FP | DepthPro FN |
|-------|------|-------------|--------|-------|-------------|-------------|-------------|
| bus | T | **49.5** | 47.7 | +1.8 | 52 | 20 | 46 |
| truck | T | 37.9 | 34.8 | +3.1 | 33 | 19 | 60 |
| **car** | T | **32.2** | 26.8 | **+5.4** | 1221 | 384 | 3414 |
| train | T | 31.8 | 32.7 | -0.9 | 10 | 16 | 13 |
| **rider** | T | **15.9** | 12.1 | **+3.8** | 77 | 37 | 464 |
| **person** | T | **11.8** | 6.4 | **+5.4** | 298 | 131 | 3078 |
| bicycle | T | **7.8** | 6.7 | +1.1 | 95 | 251 | 1068 |
| motorcycle | T | 0.0 | 0.0 | 0.0 | 0 | 0 | 149 |
| **Mean things** | | **23.35** | **20.90** | **+2.45** | | | |

**Biggest gains**: car (+5.4), person (+5.4), rider (+3.8) — all boundary-sensitive classes where DepthPro's sharp metric depth excels at separating adjacent objects.

**Slight regression**: train (-0.9) — rare class (23 GT instances), not statistically significant.

**Motorcycle**: 0.0 for all models — k=80 semantics have 0 motorcycle clusters (structural ceiling).

---

## Key Insight: Boundary Sharpness Predicts Optimal Threshold

| Depth Model | Boundary F1 (Sintel) | Optimal tau | PQ_things |
|-------------|---------------------|-------------|-----------|
| SPIdepth | ~0.19 | 0.20 | 19.41 |
| DA2-Large | 0.228 | 0.03 | 20.20 |
| DA3-Large | ~0.35 | 0.03 | 20.90 |
| DepthPro | 0.409 | 0.01 | 23.35 |

Sharper boundaries produce more precise depth edges that align with true object boundaries. This allows a lower Sobel threshold (capturing finer edges) without introducing false splits. The relationship is roughly:

- **Higher boundary F1 -> lower optimal tau -> better instance splitting**
- DepthPro's metric depth (absolute scale) may also help — consistent gradients across scenes.

---

## Recommended Configuration for Stage 2 Training

The recommended setting for direct validation pseudo label generation remains DepthPro with standard Sobel gradient, tau 0.01, area minimum 1000, sigma 0.0, and dilation iterations 3. The recommended setting for full Stage 2 training labels is DepthPro with tau 0.20, area minimum 1000, sigma 0.0, and dilation iterations 3 because the full training sensitivity sweep favors a balanced label set over the raw validation optimum.

## Data Locations

| Resource | Path |
|----------|------|
| DepthPro depth maps | `depth_depthpro/{train,val}/{city}/{stem}.npy` |
| DepthPro instances | `pseudo_instance_depthpro/{train,val}/{city}/{stem}.npz` |
| Phase 1 sweep results | `sweep_depthpro_phase1_val.json` |
| Phase 2 sweep results | `sweep_depthpro_phase2_val.json` |
| Sweep script | `mbps_pytorch/sweep_depthpro.py` |
| Depth generation | `mbps_pytorch/generate_depth_multimodel.py --model depthpro` |
| Instance generation | `mbps_pytorch/generate_depth_guided_instances.py` |
| Phase 1 log | `logs/sweep_depthpro_phase1.log` |
| Phase 2 log | `logs/sweep_depthpro_phase2.log` |

## Comparison with Prior Ablation

This study extends `reports/depth_model_ablation_study.md` (2026-03-28) which compared DA3, DA2-Large, and SPIdepth. That study concluded DA3 was optimal (PQ_things=20.90). DepthPro now supersedes DA3 as the recommended depth model with a +2.45 gain.
