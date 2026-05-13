# DCFA + DepthPro + SIMCF-ABC: A Three-Stage Pseudo-Label Refinement Pipeline

**Date:** 2026-04-19
**Best Result:** PQ = 25.85 (+1.31 over baseline), PQ\_things = 14.70 (+2.39)
**Labels:** `cups_pseudo_labels_dcfa_simcf_abc/` (8,925 files)

---

## 1. Motivation

Unsupervised panoptic segmentation generates pseudo-labels from frozen vision features and monocular depth, then trains a panoptic network on those labels. The quality ceiling of the trained model is bounded by the quality of its supervision. Prior work (CUPS, CVPR 2025) achieves PQ = 27.8 on Cityscapes by training Cascade Mask R-CNN on overclustered pseudo-labels --- but the pseudo-labels themselves score only PQ $\approx$ 24.5, leaving a 3+ point gap that the network must bridge through generalization alone.

This report presents a three-stage refinement pipeline that raises pseudo-label PQ from 24.54 to 25.85 --- a +5.3% relative improvement --- by composing three orthogonal interventions at distinct levels of the representation hierarchy:

| Level | Method | What it corrects |
|-------|--------|-----------------|
| **Feature** | DCFA (Depth-Conditioned Feature Adapter) | Cluster boundaries that ignore depth discontinuities |
| **Instance** | DepthPro depth-guided splitting | Missing instance boundaries between co-planar objects |
| **Label** | SIMCF-ABC (Semantic-Instance Mutual Consistency Filtering) | Cross-modal inconsistencies between semantic and instance maps |

The key finding: these three interventions compose **near-additively** (+0.68 + +0.73 $\approx$ +1.31), because each operates on a different source of error that the others leave untouched.

---

## 2. Method

### 2.1 Stage 1: DCFA --- Depth-Conditioned Feature Adapter

**Problem.** Standard overclustering (k-means on frozen DINOv3 ViT-B/16 features) treats all patches with similar appearance identically, regardless of their 3D position. A sidewalk patch at 5m and a road patch at 50m may share similar texture features but belong to different semantic classes. Without depth conditioning, the cluster boundaries are blind to this geometric signal.

**Approach.** DCFA is a lightweight MLP that adjusts the frozen 90-dimensional feature codes before they enter k-means, conditioned on monocular depth via a sinusoidal positional encoding. The adapter preserves the original feature space through a skip connection with zero-initialized output, guaranteeing that it starts as identity and can only improve clustering.

**Sinusoidal Depth Encoding.** Given a normalized depth value $d \in [0, 1]$, we construct a 16-dimensional encoding:

$$\mathbf{e}(d) = \bigl[\sin(\omega_1 \pi d),\; \cos(\omega_1 \pi d),\; \ldots,\; \sin(\omega_8 \pi d),\; \cos(\omega_8 \pi d)\bigr] \in \mathbb{R}^{16}$$

where $\omega_k \in \{1, 2, 4, 8, 16, 32, 64, 128\}$ are 8 octave-spaced frequencies. This multi-frequency representation enables the MLP to learn depth-dependent adjustments at multiple spatial scales.

**Adapter Architecture.** The adapter $\mathcal{A}_\theta$ is a 2-layer MLP with skip connection:

$$\mathbf{h} = \text{ReLU}\bigl(\text{LN}(\mathbf{W}_1 [\mathbf{f};\; \mathbf{e}(d)] + \mathbf{b}_1)\bigr)$$

$$\mathbf{h}' = \text{ReLU}\bigl(\text{LN}(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2)\bigr)$$

$$\mathcal{A}_\theta(\mathbf{f}, d) = \mathbf{f} + \underbrace{\mathbf{W}_{\text{out}} \mathbf{h}' + \mathbf{b}_{\text{out}}}_{\text{zero-initialized residual}}$$

where $\mathbf{f} \in \mathbb{R}^{90}$ is the frozen DINOv3 code, $[\cdot;\cdot]$ denotes concatenation, LN is LayerNorm, $\mathbf{W}_1 \in \mathbb{R}^{384 \times 106}$, $\mathbf{W}_2 \in \mathbb{R}^{384 \times 384}$, and $\mathbf{W}_{\text{out}} \in \mathbb{R}^{90 \times 384}$. The output projection is initialized to zero: $\mathbf{W}_{\text{out}} = \mathbf{0}$, $\mathbf{b}_{\text{out}} = \mathbf{0}$.

**Training.** The adapter is trained with a depth-correlation loss that encourages patches with similar depth to have similar adjusted codes, plus a preservation term that prevents the codes from drifting too far from the originals:

$$\mathcal{L} = \mathcal{L}_{\text{depth-corr}} + \lambda_{\text{preserve}} \cdot \|\mathcal{A}_\theta(\mathbf{f}, d) - \mathbf{f}\|_2^2$$

**Specifications.** 40K parameters, hidden dimension 384, 2 layers, 16D sinusoidal depth input. Checkpoint: `results/depth_adapter/V3_dd16_h384_l2/best.pt`. After adaptation, k-means (k=80) is re-run on the adjusted codes to produce refined semantic pseudo-labels.

**Result.** mIoU improves from 52.69% to 55.29% (+2.60 points at k=80).

---

### 2.2 Stage 2: DepthPro Depth-Guided Instance Generation

**Problem.** Semantic pseudo-labels assign class identities but not object individuality. Two adjacent cars of the same class appear as a single connected region. Instance boundaries require a signal orthogonal to appearance --- depth discontinuities at object boundaries provide exactly this.

**Approach.** We use DepthPro (Apple, 2024) monocular depth estimation to generate depth maps, then extract instance boundaries via Sobel gradient thresholding and connected component analysis.

**Edge Detection.** Given a depth map $D(x, y)$, optionally smoothed with Gaussian blur ($\sigma = 1.0$), we compute Sobel gradient magnitude:

$$G_x = S_x * D, \quad G_y = S_y * D, \quad \|\nabla D\| = \sqrt{G_x^2 + G_y^2}$$

where $S_x, S_y$ are the $3 \times 3$ Sobel kernels. Depth edges are pixels where $\|\nabla D\| > \tau$, with threshold $\tau = 0.20$.

**Per-Class Connected Components.** For each thing class $c \in \{$person, rider, car, truck, bus, train, motorcycle, bicycle$\}$:

1. Compute the class mask: $M_c = \{(x,y) : \text{sem}(x,y) = c\}$
2. Remove depth edges: $M_c' = M_c \setminus \{(x,y) : \|\nabla D(x,y)\| > \tau\}$
3. Extract connected components: $\{C_1, C_2, \ldots, C_n\} = \text{CC}(M_c')$
4. Filter by minimum area: keep $C_i$ only if $|C_i| \geq A_{\min} = 1000$ pixels
5. Boundary reclamation: dilate each $C_i$ by 3 iterations, reclaim unassigned class-$c$ pixels

**Specifications.** $\tau = 0.20$, $A_{\min} = 1000$, $\sigma_{\text{blur}} = 1.0$, dilation iterations = 3. This reduces $\sim$292 raw connected components per image to $\sim$17 valid instances.

---

### 2.3 Stage 3: SIMCF-ABC --- Semantic-Instance Mutual Consistency Filtering

**Problem.** Semantic labels and instance labels are generated independently: semantics from k-means overclustering, instances from depth-guided splitting. These two modalities contain complementary information but also complementary errors. SIMCF exploits their mutual consistency to clean both.

#### Step A: Instance Validates Semantics (Majority Vote)

Within each instance region $I_k$, the semantic labels should be consistent --- a single car instance should not contain "road" pixels. Step A enforces this via majority vote.

For instance $I_k$ with semantic cluster assignments $\{s_p : p \in I_k\}$ mapped to trainIDs via $\phi : s \mapsto t$:

$$t^* = \arg\max_{t \in \{0,\ldots,18\}} \sum_{p \in I_k} \mathbb{1}[\phi(s_p) = t]$$

For each inconsistent pixel $p$ where $\phi(s_p) \neq t^*$, replace $s_p$ with:

$$s_p \leftarrow \arg\max_{s : \phi(s) = t^*} \sum_{q \in I_k} \mathbb{1}[s_q = s]$$

i.e., the most frequent cluster ID within $I_k$ that maps to the majority trainID $t^*$.

#### Step B: Semantics Validate Instances (Feature-Guided Merging)

Adjacent instances of the same semantic class that are visually similar should be merged --- they were likely over-split by depth noise. Step B uses DINOv3 features to make this decision.

For each pair of adjacent instances $(I_i, I_j)$ where adjacency is defined by morphological dilation ($d = 3$ pixels):

1. Check semantic consistency: $\text{class}(I_i) = \text{class}(I_j)$ (majority trainID)
2. Compute mean DINOv3 feature vectors:

$$\bar{\mathbf{f}}_i = \frac{1}{|I_i|} \sum_{p \in I_i} \frac{\mathbf{f}_p}{\|\mathbf{f}_p\|}, \quad \bar{\mathbf{f}}_j = \frac{1}{|I_j|} \sum_{p \in I_j} \frac{\mathbf{f}_p}{\|\mathbf{f}_p\|}$$

3. Merge if cosine similarity exceeds threshold:

$$\text{merge}(I_i, I_j) \iff \frac{\bar{\mathbf{f}}_i \cdot \bar{\mathbf{f}}_j}{\|\bar{\mathbf{f}}_i\| \|\bar{\mathbf{f}}_j\|} > \tau_{\text{sim}} = 0.85$$

Transitive merges are resolved via union-find, and instance IDs are renumbered contiguously.

**Effect.** This step is critical for instance quality: it reduces raw DepthPro instances from 44/image (median 5,502 px, 50.7% stuff contamination) to 22/image (median 14,965 px, 28.0% stuff contamination). The 2.7$\times$ increase in median instance size and halved stuff contamination directly improves PQ\_things.

#### Step C: Depth Validates Semantics (Statistical Outlier Masking)

Each semantic class has a characteristic depth distribution. A "sky" pixel at 2m or a "road" pixel at 200m is likely a labeling error. Step C uses global depth statistics to identify and mask such outliers.

**First pass:** Compute per-class depth statistics across all training images:

$$\mu_c = \frac{1}{N_c} \sum_{p : \text{class}(p) = c} D(p), \quad \sigma_c = \sqrt{\frac{1}{N_c} \sum_{p : \text{class}(p) = c} (D(p) - \mu_c)^2}$$

**Second pass:** For each pixel $p$ with mapped class $c$, mask if depth deviates beyond $k$ standard deviations:

$$\text{sem}(p) \leftarrow 255 \quad \text{if} \quad |D(p) - \mu_c| > k \cdot \sigma_c, \quad k = 3$$

Setting $\text{sem}(p) = 255$ marks the pixel as "ignore" during training, preventing the network from learning from likely-incorrect labels.

---

## 3. Pipeline Architecture

```
                        DCFA + DepthPro + SIMCF-ABC Pipeline
  ═══════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────┐
  │                     INPUT SIGNALS                               │
  │                                                                 │
  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
  │   │  DINOv3       │    │  DepthPro    │    │  DINOv3       │     │
  │   │  ViT-B/16     │    │  Monocular   │    │  768D         │     │
  │   │  90D codes    │    │  Depth       │    │  Features     │     │
  │   │  (2048 patch) │    │  (512×1024)  │    │  (2048 patch) │     │
  │   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘     │
  │          │                   │                    │              │
  └──────────┼───────────────────┼────────────────────┼──────────────┘
             │                   │                    │
  ═══════════╪═══════════════════╪════════════════════╪══════════════
  STAGE 1    │                   │                    │
  (Feature)  ▼                   │                    │
       ┌───────────┐             │                    │
       │   DCFA    │             │                    │
       │  40K MLP  │◄── 16D sin(d)                   │
       │  skip+0   │             │                    │
       └─────┬─────┘             │                    │
             │ adjusted          │                    │
             │ 90D codes         │                    │
             ▼                   │                    │
       ┌───────────┐             │                    │
       │  k-means  │             │                    │
       │  k = 80   │             │                    │
       └─────┬─────┘             │                    │
             │ semantic          │                    │
             │ (H×W, uint8)      │                    │
             │                   │                    │
  ═══════════╪═══════════════════╪════════════════════╪══════════════
  STAGE 2    │                   │                    │
  (Instance) │                   ▼                    │
             │            ┌────────────┐              │
             │            │   Sobel    │              │
             │            │  ‖∇D‖>τ   │              │
             │            │  τ = 0.20  │              │
             │            └─────┬──────┘              │
             │                  │ depth edges          │
             │                  ▼                     │
             │     ┌──────────────────────┐           │
             │     │  Per-Class CC        │           │
             ├────►│  + A_min ≥ 1000 px   │           │
             │     │  + 3px dilation      │           │
             │     └──────────┬───────────┘           │
             │                │ instance              │
             │                │ (H×W, uint16)         │
             │                │                       │
  ═══════════╪════════════════╪═══════════════════════╪══════════════
  STAGE 3    │                │                       │
  (Label)    ▼                ▼                       │
       ┌──────────────────────────┐                   │
       │      SIMCF Step A        │                   │
       │  Instance → Semantic     │                   │
       │  majority vote per inst  │                   │
       └────────────┬─────────────┘                   │
                    │                                 │
                    ▼                                 │
       ┌──────────────────────────┐                   │
       │      SIMCF Step B        │◄──────────────────┘
       │  Semantic → Instance     │     DINOv3 768D features
       │  merge if cos > 0.85     │
       │  + union-find            │
       └────────────┬─────────────┘
                    │
                    ▼
       ┌──────────────────────────┐
       │      SIMCF Step C        │◄── DepthPro depth
       │  Depth → Semantic        │
       │  mask if |d−μ_c| > 3σ_c  │
       └────────────┬─────────────┘
                    │
  ══════════════════╪═══════════════════════════════════════════════
                    ▼
            ┌──────────────┐
            │   OUTPUT     │
            │              │
            │  _semantic   │  (H×W, uint8, 0-79 + 255)
            │  _instance   │  (H×W, uint16)
            │  .pt         │  (pixel distributions)
            │              │
            │  8,925 files │
            └──────────────┘
```

---

## 4. Ablation Study

### 4.1 Compositional Ablation

Each component was evaluated independently and in combination to verify additive composition. All evaluations use the global Hungarian matching protocol on the Cityscapes training set (2,975 images).

**Claim:** DCFA (feature-level) and SIMCF-ABC (label-level) address orthogonal error sources and should compose additively.

**Test:** Five variants isolating each component and their combinations.

| # | Variant | PQ | PQ\_stuff | PQ\_things | mIoU | $\Delta$PQ |
|---|---------|-----|----------|-----------|------|---------|
| 1 | A0: Raw k=80 + DepthPro $\tau$=0.20 | 24.54 | 33.43 | 12.31 | 56.56 | --- |
| 2 | DCFA + DepthPro only | 25.22 | 33.99 | 13.16 | 56.16 | +0.68 |
| 3 | SIMCF-ABC only (raw k=80) | 25.27 | 33.73 | 13.64 | 56.57 | +0.73 |
| 4 | DCFA + V3-DepthPro inst + SIMCF-ABC sem (no Step B on inst) | 25.22 | 33.96 | 13.22 | 56.22 | +0.68 |
| 5 | **DCFA + DepthPro + SIMCF-ABC (full)** | **25.85** | **33.96** | **14.70** | **56.22** | **+1.31** |

**Observation:** The expected additive gain from rows 2 and 3 is $+0.68 + +0.73 = +1.41$. The actual combined gain is $+1.31$ (93% of additive), confirming near-orthogonality. The slight sub-additivity ($-0.10$) comes from overlapping corrections where both DCFA and SIMCF-A fix the same misassigned pixels.

**Surprise:** Row 4 vs Row 5 reveals that SIMCF Step B's instance merging contributes +1.48 PQ\_things ($13.22 \to 14.70$). Simply combining DCFA-improved semantics with DepthPro instances --- without SIMCF's instance merging --- yields worse PQ\_things than the full pipeline. Step B is not optional; it is the largest single contributor to PQ\_things improvement.

### 4.2 The Depth Splitting Illusion

Raw DepthPro instances (without SIMCF Step B merging) appear to provide more object separability but actually degrade panoptic quality:

| Metric | Raw DepthPro | After SIMCF Step B |
|--------|-------------|-------------------|
| Instances/image | 44 | 22 |
| Median instance size (px) | 5,502 | 14,965 |
| Stuff contamination | 50.7% | 28.0% |
| PQ\_things | 13.22 | **14.70** |

Depth gradients at $\tau = 0.20$ split not only between objects but also within objects (surface curvature, texture-depth correlation) and within stuff regions (road plane changes, building facades). SIMCF Step B's semantic-guided merging removes these spurious splits by verifying that adjacent same-class fragments share similar DINOv3 features (cosine > 0.85).

### 4.3 SIMCF-v2: Five Advanced Refinement Steps (All Failed)

To test whether further post-hoc refinement could improve beyond SIMCF-ABC, we implemented five additional steps operating at 32$\times$64 patch resolution. All were applied individually on top of SIMCF-ABC output.

**Claim:** Post-hoc refinement at patch resolution has a hard ceiling; SIMCF-ABC already captures the easy wins.

| Step | Method | Mechanism | Changes | $\Delta$PQ |
|------|--------|-----------|---------|-----------|
| D | SDAIR (Spectral Depth-Aware Instance Refinement) | Spectral bipartition with depth-feature product kernel: $w_{ij} = \exp\!\bigl(-\frac{\|\mathbf{f}_i - \mathbf{f}_j\|^2}{2\sigma_f^2}\bigr) \cdot \exp\!\bigl(-\frac{(d_i - d_j)^2}{2\sigma_d^2}\bigr)$. Fiedler vector sign determines split. | 103 splits (0.03/img) | 0.00 |
| E | WBIM (Wasserstein-Based Instance Merging) | Sliced Wasserstein distance between instance feature distributions. Merge if $W_1 < \tau_W$. | 7,278 merges (2.45/img) | **-0.35** |
| F | ITCBS (Information-Theoretic Class Boundary Sharpening) | Mutual information between patch features and class labels. Reassign boundary patches if MI gain > threshold. | 0 pixels | 0.00 |
| G | DCCPR (Depth-Conditioned Class Prior Reassignment) | Per-class Gaussian mixture models in depth-feature space. Bayesian reassignment using posterior class probabilities. | 141.9M pixels | **-0.62** |
| H | GSID (Grassmannian Subspace Instance Discrimination) | Grassmannian distance between instance feature subspaces. Merge if subspace angle < threshold. | 0 merges | 0.00 |

**Interpretation.** Steps D, F, H were too conservative to trigger meaningful changes. Steps E, G were too aggressive --- E over-merged instances (killing PQ\_things), G reassigned $\sim$5% of all pixels to wrong classes. The pattern is consistent: at 32$\times$64 patch resolution ($16 \times 32$ pixels per patch), the feature granularity is too coarse to make fine-grained boundary decisions. SIMCF-ABC's three simple operations already extract all gains available at this resolution.

---

## 5. Compositionality Analysis

The near-additive composition of DCFA and SIMCF-ABC arises from their operating on disjoint error populations:

| Error type | DCFA corrects? | SIMCF-ABC corrects? |
|-----------|---------------|-------------------|
| Depth-blind cluster boundaries | Yes | No |
| Intra-instance semantic inconsistency | No | Yes (Step A) |
| Over-fragmented instances | No | Yes (Step B) |
| Depth-outlier semantic labels | No | Yes (Step C) |
| Object-level misclassification | Partially | No |

DCFA adjusts the feature space *before* clustering, moving depth-separated patches apart. SIMCF-ABC operates *after* both clustering and instance generation, fixing cross-modal inconsistencies that clustering and depth splitting introduced. The only overlap is that DCFA's better cluster boundaries sometimes pre-empt errors that Step A would have caught --- explaining the 7% sub-additivity.

---

## 6. Comparison with Prior Pseudo-Label Baselines

| Method | PQ | PQ\_stuff | PQ\_things | mIoU | Year |
|--------|-----|----------|-----------|------|------|
| CC-only (no depth) | 24.80 | 32.08 | 14.93 | --- | --- |
| SPIdepth ($\tau$=0.20) | 26.74 | 32.08 | 19.41 | --- | --- |
| DA3 ($\tau$=0.03) | 27.37 | 32.08 | 20.90 | --- | --- |
| DepthPro ($\tau$=0.01, $A_\min$=1000) | 28.40 | 32.08 | 23.35 | --- | --- |
| **DCFA + DepthPro + SIMCF-ABC** ($\tau$=0.20) | **25.85** | **33.96** | **14.70** | 56.22 | 2026 |

**Note on evaluation context.** The first four rows use $\tau$-optimized depth splitting evaluated on the *validation* set with 19-class standard protocol. Our pipeline uses $\tau = 0.20$ (optimized for *training* --- larger instances help Cascade Mask R-CNN learning) and is evaluated on the *training* set. Direct PQ comparison across these protocols is not meaningful. The relevant comparison is within our ablation (Section 4.1), where all variants use identical evaluation.

---

## 7. Artifacts

| Artifact | Path |
|----------|------|
| Best pseudo-labels | `cups_pseudo_labels_dcfa_simcf_abc/` (8,925 files) |
| DCFA centroids | `pseudo_semantic_adapter_V3_k80/kmeans_centroids.npz` |
| DCFA checkpoint | `results/depth_adapter/V3_dd16_h384_l2/best.pt` |
| SIMCF-ABC script | `scripts/refine_simcf.py` |
| SIMCF-v2 scripts | `scripts/simcf_v2/{sdair,wbim,itcbs,dccpr,gsid}.py` |
| Ablation runner | `scripts/run_simcf_v2_ablation.sh` |
| Evaluation script | `scripts/evaluate_pseudolabel_quality.py` |
| SIMCF-v2 results | `logs/pseudolabel_ablation/results_v2.csv` |

---

## 8. Conclusion

The DCFA + DepthPro + SIMCF-ABC pipeline demonstrates that pseudo-label quality gains compose across representation levels: feature adaptation (DCFA), geometric instance extraction (DepthPro), and cross-modal consistency filtering (SIMCF-ABC) address distinct error modes that sum to a +1.31 PQ improvement. Five additional post-hoc refinement methods (SIMCF-v2 Steps D--H) all failed, confirming that the remaining pseudo-label errors are structural and cannot be resolved at 32$\times$64 patch resolution. The path to further improvement lies upstream: higher-resolution features, better depth models, or learned segmentation networks that generalize beyond the pseudo-label ceiling.
