# MBPS Paper Figure Plan

This file lists the recommended figure set for the current paper draft. The goal is to make every figure answer a reviewer-facing question rather than merely illustrate a module.

## Main Paper Figures

### Figure 1: Method Overview

**Purpose:** Communicate the full paper thesis in one glance: a single RGB image can provide panoptic pseudo-labels when frozen semantic appearance, monocular geometry, and cross-modal agreement are composed carefully enough to seed CUPS-style training.

**Panels:**

| Panel | Content |
|---|---|
| 1 | Input RGB image |
| 2 | Frozen DINOv2 + CAUSE-TR semantic code extraction |
| 3 | DCFA depth-conditioned semantic correction |
| 4 | Monocular depth edge extraction |
| 5 | SIMCF-ABC semantic-instance-depth agreement filter |
| 6 | CUPS bootstrapping + self-training with DINOv3 Cascade Mask R-CNN |
| 7 | Final panoptic prediction |

**Notes:**

- This can be a polished schematic.
- Generated/vector-style art is acceptable for the architecture diagram.
- Keep the central message visible: no stereo/video at pseudo-label generation time.

### Figure 2: Stage-1 Pseudo-Label Evolution

**Purpose:** Show that the monocular pseudo-label source is concrete, interpretable, and visually plausible before downstream model training.

**Use real Cityscapes examples.**

**Columns:**

| Column | Content |
|---|---|
| 1 | RGB image |
| 2 | Monocular depth |
| 3 | Depth edges |
| 4 | Raw `K=80` semantic clusters |
| 5 | Raw depth instances |
| 6 | DCFA + SIMCF-ABC pseudo-label |
| 7 | Ground truth, for evaluation visualization only |

**Notes:**

- Do not use generated images for this figure.
- Select examples where the progression from raw clusters to refined pseudo-labels is visually clear.
- This figure supports the claim that the pseudo-labels contain structured errors rather than random noise.

### Figure 3: Monocular Depth Replaces Stereo/Video Cue

**Purpose:** Defend the central claim that monocular depth is a useful instance cue in general, not a lucky artifact of one estimator.

**Suggested layout:**

| Region | Content |
|---|---|
| Left | Same RGB image with CC-only instances, DepthPro instances, and DA3 instances |
| Right | Bar plot of `PQ_th` from the depth model comparison table |

**Bar plot values:**

| Depth source | PQ_th |
|---|---:|
| No depth / CC-only | 14.93 |
| SPIdepth | 19.41 |
| DepthPro | 23.35 |
| Depth Anything v2 / DA2 | 20.20 |
| Depth Anything v3 / DA3 | 20.90 |

**Notes:**

- The visual side should show the same scene under different instance-generation cues.
- The quantitative side should emphasize thing PQ because the depth cue primarily affects instance quality.

### Figure 4: DCFA + SIMCF Complementarity

**Purpose:** Show that DCFA and SIMCF correct different failure modes and combine constructively.

**Use a grouped bar chart from the component ablation table.**

**Values:**

| Variant | PQ | PQ_th | PQ_st |
|---|---:|---:|---:|
| Raw `K=80` + depth | 24.54 | 12.31 | 33.43 |
| DCFA only | 25.22 | 13.16 | 33.99 |
| SIMCF only | 25.27 | 13.64 | 33.73 |
| DCFA + SIMCF | 25.85 | 14.70 | 33.96 |

**Preferred visualization:**

- Primary bars: `PQ_th`.
- Secondary markers or small labels: total `PQ`.
- Optional muted reference line at raw baseline.

**Notes:**

- This figure should read as evidence for complementarity, not as a large absolute-gain claim.
- Caption should state that DCFA acts before clustering, while SIMCF acts after semantic-instance generation.

### Figure 5: Qualitative Success and Failure Cases

**Purpose:** Build reviewer trust by showing both where the method works and where it fails.

**Rows:**

| Row | Case |
|---|---|
| 1 | Success: large stuff regions such as road, building, vegetation, or sky |
| 2 | Success: large vehicles or depth-separated cars |
| 3 | Failure: co-planar pedestrians or vehicles merge |
| 4 | Failure: thin structures vanish |
| 5 | Failure: rare/dead class collapse |

**Columns:**

| Column | Content |
|---|---|
| 1 | RGB image |
| 2 | Monocular depth |
| 3 | Prediction |
| 4 | Ground truth |
| 5 | Error annotation overlay |

**Notes:**

- Use real project outputs only.
- Annotate failure modes consistently.
- This is mandatory for a credible submission because the per-class table shows clear weak classes.

## Supplement Figures

### Figure S1: Detailed Architecture / Algorithm Diagram

**Purpose:** Give the implementation-level version of Figure 1.

**Include:**

| Component | Detail |
|---|---|
| Semantic code | CAUSE-TR 90D code |
| Depth encoding | 16D sinusoidal depth encoding |
| DCFA | zero-initialized residual adapter |
| Instance cue | Sobel depth splitting + connected components + area filtering + dilation |
| SIMCF-A | instance-to-semantic consistency |
| SIMCF-B | semantic-to-instance fragment merging with DINOv3 similarity |
| SIMCF-C | depth-to-semantic outlier masking |
| Export | CUPS-compatible semantic PNGs, instance PNGs, and pixel-distribution tensors |

### Figure S2: Per-Class Breakdown Visualization

**Purpose:** Make the full Cityscapes per-class table easier to read.

**Visualization:**

- Sorted bar chart of per-class PQ.
- Color stuff and thing classes differently.
- Mark dead or near-dead classes.

**Classes to highlight:**

| Group | Examples |
|---|---|
| Strong stuff | road, building, vegetation, sky |
| Thin structures | pole, traffic light, traffic sign, fence |
| Common things | person, car, bicycle |
| Rare/dead things | caravan, trailer, motorcycle, train |

### Figure S3: Cross-Dataset Transfer

**Purpose:** Show where the trained model transfers and where class-space mismatch breaks it.

**Values:**

| Dataset | Class-space relation | PQ | PQ_th | PQ_st | mIoU |
|---|---|---:|---:|---:|---:|
| Cityscapes | source | 35.83 | 36.26 | 35.56 | 44.56 |
| KITTI | aligned driving classes | 34.85 | 31.94 | 36.40 | 46.87 |
| Mapillary Vistas v2 | mostly aligned driving classes | 39.19 | 32.06 | 44.37 | 58.87 |
| MOTSChallenge | 2-class driving subset | 61.10 | 25.52 | 96.68 | 92.10 |
| COCO-Stuff-27 | disjoint/coarse classes | 7.83 | 7.83 | 7.84 | 14.22 |

**Notes:**

- Visually separate aligned driving datasets from COCO-Stuff-27.
- Caption should frame COCO as class-space mismatch, not as evidence that the method fails on all non-Cityscapes data.

### Figure S4: Extended Failure Gallery

**Purpose:** Provide honest qualitative coverage beyond the main paper.

**Include 6-10 examples across:**

| Failure type | Required visualization |
|---|---|
| Co-planar people or vehicles merge | RGB, depth, prediction, ground truth, error overlay |
| Depth over-fragments one object | RGB, depth gradient, instance fragments |
| Thin objects vanish | RGB, semantic map, prediction, ground truth |
| Rare class collapses | RGB, semantic clusters, prediction, ground truth |
| COCO class-space mismatch | RGB, prediction, mapped ground truth |

## Priority

| Priority | Figure |
|---|---|
| Mandatory | Figure 1: Method Overview |
| Mandatory | Figure 2: Stage-1 Pseudo-Label Evolution |
| Mandatory | Figure 5: Qualitative Success and Failure Cases |
| Highly valuable | Figure 3: Monocular Depth Replaces Stereo/Video Cue |
| Highly valuable | Figure 4: DCFA + SIMCF Complementarity |
| Supplement-critical | Figure S1: Detailed Architecture / Algorithm Diagram |
| Supplement-critical | Figure S2: Per-Class Breakdown Visualization |
| Supplement-critical | Figure S3: Cross-Dataset Transfer |
| Supplement-critical | Figure S4: Extended Failure Gallery |

## Generation Policy

- Do not generate qualitative outputs with image generation tools; qualitative figures must come from real project artifacts.
- Generated or vector-style visuals are acceptable only for architecture schematics and polished explanatory diagrams.
- Every figure should answer a reviewer question:
  - Why is monocular input enough?
  - What does each cue contribute?
  - How do pseudo-labels evolve into final predictions?
  - Where does the method fail?
  - Which claims are supported by ablations?
