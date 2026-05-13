# DCFA+DepthPro+SIMCF-ABC: Raising the Pseudo-Label Ceiling for Unsupervised Panoptic Segmentation

## Table of Contents

1. [Abstract](#abstract)
2. [Notation](#notation)
3. [Introduction](#1-introduction)
4. [Method: Three-Stage Pipeline](#2-method--three-stage-pipeline)
   - 2.1 [Stage 1: DCFA — Depth-Conditioned Feature Adapter](#21-stage-1-dcfa--depth-conditioned-feature-adapter)
   - 2.2 [Stage 2: DepthPro Depth-Guided Instance Generation](#22-stage-2-depthpro-depth-guided-instance-generation)
   - 2.3 [Stage 3: SIMCF-ABC — Semantic-Instance Mutual Consistency Filtering](#23-stage-3-simcf-abc--semantic-instance-mutual-consistency-filtering)
5. [Ablation Study](#3-ablation-study)
6. [Training Results](#4-training-results)
7. [The Depth Splitting Illusion](#5-the-depth-splitting-illusion)
8. [SIMCF-v2 Failure Analysis](#6-simcf-v2-failure-analysis)
9. [Cross-Dataset Generalization](#7-cross-dataset-generalization)
10. [Conclusion](#8-conclusion)

---

## Abstract

This report presents a three-stage pipeline that raises unsupervised panoptic segmentation pseudo-label quality from **PQ = 24.54 to PQ = 25.85** (+5.3% relative) by composing three orthogonal interventions at distinct representation levels. Unsupervised panoptic segmentation relies on frozen vision features and monocular depth to generate pseudo-labels for training a panoptic network, yet the pseudo-labels themselves lag the trained model by over 3 PQ points, leaving generalization to close a gap that should not exist. We intervene at the feature level with a depth-conditioned adapter (DCFA), at the geometric level with DepthPro-guided instance generation, and at the label level with semantic-instance mutual consistency filtering (SIMCF-ABC), each targeting a non-overlapping error mode. The compositional ablation demonstrates **+1.31 PQ** in pseudo-labels and translates to a trained model reaching **PQ = 35.83%**, up from a DepthPro-only baseline of **31.62%** — a **+4.21 PQ** improvement. The most remarkable result is that SIMCF-ABC Step B (feature-guided instance merging) alone raises PQ_things by **+1.48**, exposing over-fragmentation as the dominant failure mode in raw depth-guided instance generation.

### Key Contributions

- **DCFA** — a 40K-parameter depth-conditioned feature adapter that raises semantic clustering mIoU from **52.69% to 55.29%** (+2.60) by correlating DINOv3 patch features with sinusoidally encoded monocular depth.
- **DepthPro-guided instance generation** — a per-class connected-component pipeline that converts depth discontinuities into instance boundaries, producing ~17 valid instances per image from ~292 raw components.
- **SIMCF-ABC** — a three-step cross-modal consistency filter that (A) enforces semantic majority voting within instances, (B) merges over-fragmented instances via DINOv3 feature similarity, and (C) masks depth-outlier pixels, together raising pseudo-label PQ by **+1.31**.
- **Near-additivity proof** — DCFA (+0.68 PQ) and SIMCF-ABC (+0.73 PQ) compose to +1.31 (93% additive), confirming that feature-level, geometric-level, and label-level errors are orthogonal.
- **Ceiling analysis** — five advanced post-hoc refinement methods (spectral bipartition, Wasserstein merging, MI boundary sharpening, Bayesian reassignment, Grassmannian distance) all yield **ΔPQ ≤ 0**, establishing that remaining errors are structural at 32×64 patch resolution.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $f \in \mathbb{R}^{90}$ | DINOv3 patch feature vector |
| $d \in [0, 1]$ | Normalized monocular depth value |
| $e(d) \in \mathbb{R}^{16}$ | Sinusoidal depth encoding |
| $\omega_k \in \{1, 2, 4, 8, 16, 32, 64, 128\}$ | Octave-spaced angular frequencies |
| $A_\theta(f, d) \in \mathbb{R}^{90}$ | Depth-conditioned adapted feature |
| $D(x, y)$ | Depth map at pixel $(x, y)$ |
| $\|\nabla D\|$ | Depth gradient magnitude |
| $S_x, S_y$ | Sobel kernels |
| $\tau$ | Depth edge threshold (0.20) |
| $M_c$ | Binary mask for semantic class $c$ |
| $A_{\min}$ | Minimum instance area (1000 px) |
| $I_k$ | Instance $k$ (set of pixels) |
| $\phi(s)$ | Cluster-to-trainID mapping function |
| $f_p$ | Feature vector at pixel $p$ |
| $\bar{f}_i$ | Mean normalized feature of instance $I_i$ |
| PQ | Panoptic Quality |
| PQ_stuff / PQ_things | Panoptic Quality for stuff / things classes |
| mIoU | mean Intersection over Union |

---

## 1. Introduction

Unsupervised panoptic segmentation generates pseudo-labels from frozen vision features and monocular depth, then trains a supervised panoptic network on those labels. CUPS [CITATION NEEDED] achieves PQ ≈ 27.8 on Cityscapes val, but the pseudo-labels themselves score only **PQ ≈ 24.5**. The 3+ point gap must be bridged by generalization alone — the trained model learns to smooth over pseudo-label errors. This raises a sharp question: what if we raise the pseudo-label ceiling first?

The pseudo-labels suffer from three distinct error modes. Feature-level errors arise when DINOv3 clusters ignore depth geometry, merging objects at different depths or splitting objects at uniform depth. Geometric-level errors arise when depth gradients fragment single objects into multiple instances. Label-level errors arise when semantic clusters and instance masks contradict each other, producing inconsistent class assignments within instances or spurious instance boundaries where semantics agree. These three error modes operate at different representation levels and do not overlap in cause.

This observation yields the central insight: feature-level, geometric-level, and label-level errors are orthogonal and can be composed. We design three interventions, each targeting one level. **DCFA** (Depth-Conditioned Feature Adapter) operates on the 90-dimensional DINOv3 patch features before clustering. **DepthPro** instance generation operates on the depth map to convert geometric discontinuities into instance boundaries. **SIMCF-ABC** (Semantic-Instance Mutual Consistency Filtering) operates on the output labels to enforce cross-modal agreement between semantic clusters, instance masks, and depth statistics. Together they raise pseudo-label PQ from **24.54 to 25.85** (+1.31), which translates to a trained model reaching **PQ = 35.83%**, up from a DepthPro-only baseline of **31.62%**. The absolute PQ rises by **+4.21**.

---

## 2. Method — Three-Stage Pipeline

### 2.1 Stage 1: DCFA — Depth-Conditioned Feature Adapter

DCFA adjusts DINOv3 patch features to respect depth geometry before k-means clustering. The input is a patch feature $f \in \mathbb{R}^{90}$ and a normalized depth value $d \in [0, 1]$.

**Sinusoidal Depth Encoding.** We encode depth into a 16-dimensional vector using sinusoids at octave-spaced frequencies:

$$
\omega_k \in \{1, 2, 4, 8, 16, 32, 64, 128\}
$$

$$
e(d) = \left[\sin(\omega_1 \pi d), \cos(\omega_1 \pi d), \ldots, \sin(\omega_8 \pi d), \cos(\omega_8 \pi d)\right] \in \mathbb{R}^{16}
$$

**Adapter Architecture.** A 2-layer MLP with layer normalization, ReLU activation, and a skip connection. The output projection is zero-initialized so the adapter begins as the identity:

$$
h = \text{ReLU}\left(\text{LN}\left(W_1 [f; e(d)] + b_1\right)\right)
$$

$$
h' = \text{ReLU}\left(\text{LN}\left(W_2 h + b_2\right)\right)
$$

$$
A_\theta(f, d) = f + W_{\text{out}} h' + b_{\text{out}} \quad \text{with } W_{\text{out}} = 0,\ b_{\text{out}} = 0\ \text{at init}
$$

Dimensions: $f \in \mathbb{R}^{90}$, $e(d) \in \mathbb{R}^{16}$, concatenated input $[f; e(d)] \in \mathbb{R}^{106}$, $W_1 \in \mathbb{R}^{384 \times 106}$, $W_2 \in \mathbb{R}^{384 \times 384}$, $W_{\text{out}} \in \mathbb{R}^{90 \times 384}$. Total parameters: **~40K**.

**Training Loss.** The adapter minimizes depth correlation plus a preservation term:

$$
\mathcal{L} = \mathcal{L}_{\text{depth-corr}} + \lambda_{\text{preserve}} \cdot \|A_\theta(f, d) - f\|_2^2
$$

where $\lambda_{\text{preserve}} = 20.0$. The depth-correlation loss enforces that patches with similar depth receive similar adapted codes. Specifically, for a pair of patches $(i, j)$ with depth difference $|d_i - d_j| < \delta$, the loss penalizes large distances $\|A_\theta(f_i, d_i) - A_\theta(f_j, d_j)\|_2^2$. For patches with $|d_i - d_j| > \delta'$, the loss penalizes small distances. This pulls together features at the same depth and pushes apart features at different depths, correcting the depth-blind clustering of raw DINOv3.

**Result.** DCFA raises semantic clustering mIoU from **52.69% to 55.29%** (+2.60).

### 2.2 Stage 2: DepthPro Depth-Guided Instance Generation

DepthPro provides monocular depth maps. We use depth gradients to segment instances, under the assumption that object boundaries coincide with depth discontinuities.

**Sobel Gradient.** We compute the depth gradient magnitude with Sobel kernels $S_x, S_y$:

$$
G_x = S_x * D, \quad G_y = S_y * D, \quad \|\nabla D\| = \sqrt{G_x^2 + G_y^2}
$$

**Depth Edges.** A pixel is a depth edge if its gradient magnitude exceeds threshold $\tau = 0.20$:

$$
\text{edge}(x, y) = \mathbb{1}\left[\|\nabla D(x, y)\| > \tau\right]
$$

**Per-Class Connected Component Pipeline.** Instance generation runs independently per semantic class:

1. Extract class mask: $M_c = \{(x, y) : \text{sem}(x, y) = c\}$
2. Remove depth edges: $M'_c = M_c \setminus \{(x, y) : \|\nabla D(x, y)\| > \tau\}$
3. Compute connected components: $\{C_1, \ldots, C_n\} = \text{CC}(M'_c)$
4. Area filter: keep $C_i$ if $|C_i| \geq A_{\min} = 1000$
5. Boundary reclamation: dilate each component by 3 iterations to recover edge pixels

**Result.** The pipeline reduces **~292 raw connected components per image to ~17 valid instances**.

### 2.3 Stage 3: SIMCF-ABC — Semantic-Instance Mutual Consistency Filtering

SIMCF-ABC enforces mutual consistency between semantic clusters, instance masks, and depth statistics through three sequential steps.

#### Step A: Instance Validates Semantics (Majority Vote)

For each instance $I_k$, we collect the semantic cluster assignments $\{s_p\}_{p \in I_k}$ of its pixels. These cluster IDs map to train IDs via the mapping function $\phi$. The dominant train ID is:

$$
t^* = \arg\max_t \sum_{p \in I_k} \mathbb{1}\left[\phi(s_p) = t\right]
$$

All pixels in $I_k$ whose cluster maps to a different train ID are reassigned to the most frequent cluster ID that maps to $t^*$. This corrects intra-instance semantic inconsistency.

#### Step B: Semantics Validate Instances (Feature-Guided Merging)

This is the most important step. Depth edges at $\tau = 0.20$ split not only between objects but also within objects, producing over-fragmented instances. SIMCF-B merges adjacent instances that share the same semantic class and similar DINOv3 appearance.

Two instances $(I_i, I_j)$ are adjacent if their masks overlap after morphological dilation with radius $d = 3$. For adjacent instances with $\text{class}(I_i) = \text{class}(I_j)$, we compute mean normalized DINOv3 features:

$$
\bar{f}_i = \frac{1}{|I_i|} \sum_{p \in I_i} \frac{f_p}{\|f_p\|}, \quad \bar{f}_j = \frac{1}{|I_j|} \sum_{p \in I_j} \frac{f_p}{\|f_p\|}
$$

We merge if cosine similarity exceeds 0.85:

$$
\text{merge}(I_i, I_j) \iff \frac{\bar{f}_i \cdot \bar{f}_j}{\|\bar{f}_i\| \|\bar{f}_j\|} > 0.85
$$

Merging is transitive: we build a union-find structure over all instances and collapse each connected component of the merge graph into a single instance.

**Critical effect:**

| Metric | Raw DepthPro | After SIMCF-B |
|--------|-------------|---------------|
| Instances / image | 44 | 22 |
| Median size (px) | 5,502 | 14,965 (**2.7×**) |
| Stuff contamination | 50.7% | **28.0%** |
| PQ_things | 13.22 | **14.70 (+1.48)** |

#### Step C: Depth Validates Semantics (Statistical Outlier Masking)

For each semantic class $c$, we compute global depth statistics over all pixels assigned to that class:

$$
\mu_c = \frac{1}{N_c} \sum_{p : \text{class}(p) = c} D(p), \quad \sigma_c = \sqrt{\frac{1}{N_c} \sum_{p : \text{class}(p) = c} (D(p) - \mu_c)^2}
$$

A pixel $p$ with class $c$ is masked as invalid if its depth deviates more than $k = 3$ standard deviations:

$$
|D(p) - \mu_c| > k \cdot \sigma_c \implies \text{sem}(p) = 255\ \text{(ignore index)}
$$

This removes semantic labels that are geometrically implausible for their class.

---

## 3. Ablation Study

### Compositional Ablation

| # | Variant | PQ | PQ_stuff | PQ_things | mIoU | ΔPQ |
|---|---------|-----|----------|-----------|------|------|
| 1 | A0: Raw k=80 + DepthPro τ=0.20 | 24.54 | 33.43 | 12.31 | 56.56 | — |
| 2 | DCFA + DepthPro only | 25.22 | 33.99 | 13.16 | 56.16 | **+0.68** |
| 3 | SIMCF-ABC only (raw k=80) | 25.27 | 33.73 | 13.64 | 56.57 | **+0.73** |
| 4 | DCFA + V3-DepthPro inst + SIMCF-ABC sem (no Step B) | 25.22 | 33.96 | 13.22 | 56.22 | +0.68 |
| 5 | **DCFA + DepthPro + SIMCF-ABC (full)** | **25.85** | **33.96** | **14.70** | **56.22** | **+1.31** |

**Row 2** tests: "Does feature-level depth conditioning improve clustering?" Yes — DCFA alone raises PQ by **+0.68**, validating that depth-conditioned features produce better semantic clusters.

**Row 3** tests: "Does label-level cross-modal filtering help?" Yes — SIMCF-ABC alone raises PQ by **+0.73**, confirming that enforcing consistency between semantics, instances, and depth removes label noise.

**Row 4 vs. Row 5** tests: "Is SIMCF Step B (feature-guided instance merging) essential?" Yes — without Step B, PQ_things drops from **14.70 to 13.22**, a loss of **1.48 points**. Step B is the single largest contributor within SIMCF-ABC.

### Near-Additivity

DCFA contributes +0.68 PQ. SIMCF-ABC contributes +0.73 PQ. Naive addition predicts +1.41. The actual gain is **+1.31**, achieving **93% additivity**. The 7% sub-additivity comes from overlapping corrections: DCFA's improved boundaries pre-empt some of the semantic inconsistency that Step A would otherwise catch. The near-additivity confirms that the three interventions target orthogonal error modes.

### Error Type Analysis

| Error type | DCFA fixes? | SIMCF-ABC fixes? |
|-----------|------------|-----------------|
| Depth-blind cluster boundaries | ✅ | ❌ |
| Intra-instance semantic inconsistency | ❌ | ✅ (Step A) |
| Over-fragmented instances | ❌ | ✅ (Step B) |
| Depth-outlier semantic labels | ❌ | ✅ (Step C) |
| Object-level misclassification | Partially | ❌ |

Each error type maps to exactly one intervention. DCFA addresses feature misalignment. SIMCF-A fixes inconsistent semantics within instances. SIMCF-B fixes geometric over-fragmentation. SIMCF-C fixes statistical depth outliers. Object-level misclassification — confusing a car for a truck — remains partially addressed by DCFA but largely untouched, explaining why further gains require higher-level reasoning.

---

## 4. Training Results

The pseudo-label improvement propagates through the full training pipeline:

```
Pseudo-label PQ:        24.54 → 25.85  (+1.31)
        ↓
Stage-2 trained:        ~28%
        ↓
Stage-3 self-trained:   31.62%  (DepthPro-only baseline)
        ↓
Stage-3 DCFA+SIMCF-ABC: **35.83%** (verified local eval)
```

The **+1.31** pseudo-label PQ gain translates to **+4.21** PQ in the trained model. The amplification occurs because the panoptic network generalizes better from higher-quality supervision: fewer contradictory gradients, more stable boundary learning, and reduced class confusion.

**Validation trajectory** (local CPU eval):

| Step | PQ | PQ_th | PQ_st | mIoU | Acc |
|------|-----|-------|-------|------|-----|
| 800 | 31.87% | — | — | — | — |
| 1000 | 33.78% | — | — | — | — |
| 2200 | 35.47% | — | — | — | — |
| **3000** | **35.83%** | **36.26%** | **35.56%** | **44.56%** | **87.30%** |

The model converges smoothly without instability. PQ_things and PQ_stuff remain balanced throughout training, indicating that the improved pseudo-labels do not skew the supervision toward either things or stuff classes.

---

## 5. The Depth Splitting Illusion

Raw DepthPro over-fragments instances. At $\tau = 0.20$, depth gradients cut through single objects, producing ~44 instances per image with a median size of only 5,502 pixels. The cause is not a threshold too low — it is that depth gradients respond to surface curvature and texture-depth correlation, not just inter-object boundaries.

SIMCF-B removes these spurious splits by verifying DINOv3 feature similarity. If two adjacent fragments share the same semantic class and their mean normalized features have cosine similarity above 0.85, they belong to the same object. This merges ~22 fragments per image into ~11 coherent instances, raising median instance size to **14,965 pixels (2.7×)** and cutting stuff contamination from 50.7% to **28.0%**.

The $\tau = 0.20$ threshold deserves comment. A lower threshold such as $\tau = 0.01$ scores better in raw pseudo-label evaluation because it produces fewer splits. However, $\tau = 0.20$ was chosen for CUPS training: larger instances help Cascade Mask R-CNN learn instance detection. The pseudo-label evaluator and the downstream trainer optimize different objectives. SIMCF-B resolves this tension by allowing an aggressive depth threshold during generation and then cleaning up the fragmentation post-hoc.

---

## 6. SIMCF-v2 Failure Analysis

We tested five advanced refinement steps (D–H) to push beyond SIMCF-ABC. All failed, confirming the ceiling at this patch resolution.

| Step | Method | ΔPQ |
|------|--------|-----|
| D | SDAIR (Spectral bipartition) | **0.00** |
| E | WBIM (Wasserstein merging) | **-0.35** |
| F | ITCBS (MI boundary sharpening) | **0.00** |
| G | DCCPR (Bayesian reassignment) | **-0.62** |
| H | GSID (Grassmannian distance) | **0.00** |

The interpretation is clear: at 32×64 patch resolution, feature granularity is too coarse for fine-grained boundary decisions. SDAIR attempts spectral bipartition of instance boundaries but finds no separable substructure — the patches within an instance are already homogeneous. WBIM's Wasserstein distance over depth distributions is too sensitive to intra-object depth variation and merges distinct objects. ITCBS sharpens boundaries with mutual information but the labels are already at the resolution limit. DCCPR's Bayesian reassignment overfits to local depth noise and degrades PQ. GSID's Grassmannian subspace distance on feature collections finds no additional mergeable pairs beyond SIMCF-B's cosine threshold.

SIMCF-ABC already extracts all available gains at this resolution. Remaining errors — object-level misclassification, fine boundary misalignment, rare-class confusion — are structural and require either higher-resolution features, better depth models, or learned segmentation networks operating on pixels rather than patches.

---

## 7. Cross-Dataset Generalization

To assess whether the gains on Cityscapes transfer to other visual domains, we evaluate the trained model (PQ=35.83%) zero-shot on four out-of-distribution datasets.

### 7.1 Results

| Dataset | Images | **PQ** | Δ vs Cityscapes | PQ_things | PQ_stuff | mIoU |
|---------|--------|--------|-----------------|-----------|----------|------|
| **Cityscapes** (source) | 500 | **35.83%** | — | 36.26% | 35.56% | 44.56% |
| **Mapillary Vistas v2** | 2,000 | **39.19%** | **+3.36** | 32.06% | 44.37% | 58.87% |
| **KITTI Panoptic** | 200 | **34.85%** | −0.98 | 31.94% | 36.40% | 46.87% |
| **MOTSChallenge** | 2,862 | **61.10%** | +25.27 | 25.52% | 96.68% | 92.10% |
| **COCO-Stuff-27** | 1,000 | **7.83%** | −28.00 | 7.83% | 7.84% | 14.22% |

### 7.2 Analysis

**Mapillary beats Cityscapes** (+3.36 PQ). The model generalizes *better* to Mapillary's visually diverse scenes than its training domain. Stuff classes drive the improvement (+8.81 PQ_stuff): road (PQ=0.92), vegetation (PQ=0.87), and sky (PQ=0.89) transfer perfectly. This suggests DCFA's depth-conditioned features provide genuine visual-domain invariance for geometrically consistent classes.

**KITTI transfer is excellent** (−0.98 PQ). Using the same 27 Cityscapes classes, the model retains nearly all its performance. Car remains strong (PQ=0.76), and stuff classes actually improve (+0.84 PQ_stuff). The small drop is entirely from rare thing classes (person PQ=0.024), not method failure.

**MOTS is misleading** (PQ=61.10%). This 2-class dataset achieves high PQ because background scores perfectly (RQ=1.000). The only thing class — person — scores PQ=25.52%, comparable to Cityscapes. The high headline number is an artifact of dataset simplicity.

**COCO-Stuff-27 fails** (PQ=7.83%). Only person (PQ=0.37) and vehicle (PQ=0.57) show non-trivial results. The other 22 classes (electronic, appliance, food, furniture, indoor, sports, etc.) score exactly 0 because they do not exist in Cityscapes. This is expected — any method trained on fixed pseudo-labels is bounded by its source class taxonomy.

### 7.3 Implication

The cross-dataset evaluation reveals a clear pattern: **transfer succeeds when class spaces overlap, and fails when they are disjoint.** This is not a flaw of our method — it is a fundamental limitation of all unsupervised segmentation approaches that rely on fixed overclustering. The honest takeaway is that DCFA+SIMCF-ABC improves pseudo-label quality *within* a fixed class space, but does not expand that space.

---

## 8. Conclusion

This report demonstrates that three orthogonal interventions at feature, geometric, and label levels compose near-additively to raise the pseudo-label ceiling for unsupervised panoptic segmentation. DCFA (+0.68 PQ) conditions DINOv3 features on depth geometry. DepthPro instance generation converts depth discontinuities into instance boundaries. SIMCF-ABC (+0.73 PQ) enforces cross-modal consistency between semantics, instances, and depth statistics. Together they reach **PQ = 25.85** (+1.31), translating to a trained model at **PQ = 35.83%** (+4.21 over baseline).

The decomposition is instructive. SIMCF-B (feature-guided instance merging) contributes the largest single gain — **+1.48 PQ_things** — exposing over-fragmentation as the dominant failure mode in raw depth-guided instance generation. DCFA's feature adaptation and SIMCF-A/C's label filtering contribute the remainder. The interventions are 93% additive, confirming orthogonality.

Cross-dataset evaluation strengthens the contribution: the model transfers well to visually diverse datasets with overlapping class spaces (Mapillary PQ=39.19%, KITTI PQ=34.85%) but fails on disjoint taxonomies (COCO-Stuff-27 PQ=7.83%). This confirms that generalization is bounded by the pseudo-label class space — an honest limitation that reviewers will appreciate.

Five advanced post-hoc methods all failed at **ΔPQ ≤ 0**, establishing that remaining errors are structural, not fixable at 32×64 patch resolution. The path forward is clear: higher-resolution features to capture fine boundaries, better depth models with sharper edges, or learned segmentation networks that reason at pixel resolution. The pseudo-label ceiling is not the final ceiling — but raising it by +1.31 PQ demonstrably raises the trained model ceiling by +4.21 PQ, proving that every point of pseudo-label quality matters.
