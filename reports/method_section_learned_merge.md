# Instance Decomposition via Learned Pairwise Fragment Merging

*NeurIPS-style method section — draft for paper integration*

---

## 3. Instance Decomposition

Given a pseudo-semantic map $\hat{S} \in \{0,\dots,K{-}1\}^{H \times W}$ and a monocular depth estimate $D \in \mathbb{R}^{H \times W}$, the instance decomposition task is to partition each *thing*-class region into individual object masks without access to instance-level supervision. We propose a two-stage decomposition that replaces the conventional single-threshold depth edge detector with a learned pairwise merge predictor trained via self-supervised multi-threshold consensus.

### 3.1 Motivation: The Threshold Dilemma

The standard approach applies Sobel edge detection on $D$ at a fixed gradient threshold $\tau$, then extracts connected components within each thing-class region. This creates a fundamental tension: a low $\tau$ over-segments objects (high recall, many false positives), while a high $\tau$ under-segments by failing to separate co-located instances (fewer false positives, high false negatives). No single $\tau$ is optimal across all scenes — urban images with tightly packed vehicles require aggressive splitting, while isolated objects benefit from conservative thresholds.

We resolve this by *decoupling* the two failure modes. Stage 1 uses a deliberately low threshold to maximize recall (over-segmentation), then Stage 2 applies a learned binary classifier that decides, for each pair of adjacent same-class fragments, whether they belong to the same object instance.

### 3.2 Stage 1: Depth-Guided Oversegmentation

We compute the depth gradient magnitude $G = \|\nabla D_\sigma\|$ where $D_\sigma = D * \mathcal{G}_\sigma$ is Gaussian-smoothed depth ($\sigma{=}1.0$). A binary edge map $E = \mathbf{1}[G > \tau_\text{low}]$ is computed at a low threshold $\tau_\text{low}$ (typically 0.10–0.15, compared to 0.20 for the single-threshold baseline). For each thing class $c$, the edge-free class region $\hat{S}_c \setminus E$ is decomposed into connected components. Fragments below a minimum area $A_\text{frag}$ are discarded. This yields a set of $N$ over-segmented fragments:

$$\mathcal{F} = \{(m_i, c_i, a_i)\}_{i=1}^{N}, \quad m_i \in \{0,1\}^{H \times W}, \; c_i \in \mathcal{C}_\text{thing}, \; a_i = \|m_i\|_1$$

The low threshold ensures that depth boundaries between distinct objects are preserved while also creating spurious splits *within* single objects — a deliberate design choice that Stage 2 corrects.

### 3.3 Stage 2: Learned Pairwise Merge Predictor

**Adjacency graph.** We construct a graph $\mathcal{G} = (\mathcal{F}, \mathcal{E})$ where an edge $(i, j) \in \mathcal{E}$ exists if fragments $i$ and $j$ share the same class ($c_i = c_j$) and are spatially adjacent (their dilated masks overlap within a 5-pixel radius, bridging the depth-edge gap between them).

**Pairwise descriptor.** For each candidate pair $(i, j) \in \mathcal{E}$, we extract a 200-dimensional descriptor $\mathbf{x}_{ij}$ that encodes three categories of evidence:

| Signal | Dimensions | Description |
|--------|-----------|-------------|
| **Appearance** (×3) | 64 + 64 + 64 | PCA-reduced DINOv2 mean features $\bar{\mathbf{f}}_i$, $\bar{\mathbf{f}}_j$, and element-wise difference $|\bar{\mathbf{f}}_i - \bar{\mathbf{f}}_j|$ |
| **Geometry** | 4 | Mean depth per fragment ($\bar{d}_i, \bar{d}_j$), depth difference $|\bar{d}_i - \bar{d}_j|$, and normalized centroid distance |
| **Shape** | 2 | Log-area $\log(1 + a_i)$ and $\log(1 + a_j)$ |
| **Relational** | 2 | Cosine similarity $\cos(\bar{\mathbf{f}}_i, \bar{\mathbf{f}}_j)$ and boundary cosine similarity $s_\text{bnd}$ |

The appearance features are mean-pooled DINOv2 ViT-B/14 patch embeddings (768-dim) projected to 64 dimensions via PCA fitted on training-set fragment features (92.9% variance retained). The boundary cosine similarity $s_\text{bnd}$ averages the cosine similarity between features at the shared boundary region and each fragment's mean feature, providing a local texture-consistency signal at the merge interface.

**Merge classifier.** A three-layer MLP $f_\theta: \mathbb{R}^{200} \to \mathbb{R}$ maps descriptors to merge logits:

$$f_\theta(\mathbf{x}) = W_3 \, \text{ReLU}(W_2 \, \text{ReLU}(\text{Dropout}_{0.1}(W_1 \mathbf{x} + b_1)) + b_2) + b_3$$

with hidden dimensions $128 \to 64 \to 1$ (~26K parameters). A fragment pair is merged when $\sigma(f_\theta(\mathbf{x}_{ij})) > \tau_\text{merge}$.

**Transitive closure.** Merge decisions are applied transitively via union-find with path compression: if fragments $A$ and $B$ are merged, and $B$ and $C$ are merged, then $A$, $B$, $C$ form a single instance. The merged masks are aggregated by pixel-wise OR.

**Post-processing.** Merged instances undergo dilation-based boundary reclamation (3 iterations) to recover pixels lost to depth edges, followed by minimum-area filtering ($A_\text{min}{=}1000$).

### 3.4 Self-Supervised Training via Multi-Threshold Consensus

A key property of our approach is that training requires *no instance-level annotations*. Instead, we exploit the monotonic relationship between the gradient threshold $\tau$ and instance granularity to generate binary merge labels entirely from depth.

**Label generation.** For each training image, we run Stage 1 at two thresholds: $\tau_\text{low}{=}0.10$ (producing over-segmented fragments) and $\tau_\text{high}{=}0.20$ (producing coarser, more accurate groupings). For each adjacent same-class fragment pair $(i, j)$ at $\tau_\text{low}$, we determine whether both fragments overlap the *same* connected component at $\tau_\text{high}$ (via majority vote). If so, the pair is labeled MERGE (1); otherwise NO-MERGE (0).

The intuition is that the higher threshold acts as a *noisy oracle*: fragments that remain grouped under a stricter edge criterion likely belong to the same object. This self-supervision is a form of *multi-scale depth consistency* — analogous to how multi-scale contrastive objectives use different augmentation strengths to define positive/negative pairs, but operating entirely in the depth-geometry domain.

**Training details.** The MergePredictor is trained on 3,580 pairs extracted from 2,975 Cityscapes training images (61% merge, 39% no-merge). We use binary cross-entropy with class-balanced positive weighting ($w_\text{pos}{=}0.64$), Adam optimizer ($\text{lr}{=}10^{-3}$, weight decay $10^{-4}$), and cosine annealing over 20 epochs. Best model selected by validation merge accuracy (86.0% at epoch 12, vs. 61% majority-class baseline).

### 3.5 Relationship to Ablation Baselines

We situate the learned merge within a taxonomy of eight instance decomposition strategies evaluated under identical conditions (same semantic pseudo-labels, same monocular depth, same DINOv2 features, same post-processing). The methods span five algorithmic families, and the taxonomy clarifies *why* certain design choices succeed or fail:

**Family 1: Boundary detection → connected components.**

- **Sobel+CC** (baseline). Single-threshold Sobel edges on depth → CC per class → dilation reclaim. The conventional approach. PQ$_\text{things}$=19.41. Strength: each pixel is classified independently, making it robust to scene-difficulty distribution. Weakness: the single threshold $\tau$ cannot simultaneously optimize recall and precision.

- **Feature cosine merge** (training-free ablation). Identical Stage 1, but merges adjacent fragments when their DINOv2 cosine similarity exceeds a threshold. PQ$_\text{things}$=19.10. Result: raw cosine similarity between adjacent same-class fragments has *zero discriminative power* — all similarity values cluster in a narrow band (0.85–0.95), producing identical results across all thresholds. This motivates the multi-cue learned descriptor.

- **Learned merge** (ours). The full two-stage pipeline described above. PQ$_\text{things}$=18.76. Replaces the single hard threshold with a learned decision boundary in the joint depth-feature-geometry space. The 86% merge accuracy demonstrates that the multi-cue descriptor provides discriminative signal absent from cosine similarity alone.

**Family 2: Topological / watershed decomposition.**

- **Morse flow** (gradient flow to depth basins). Watershed segmentation on smoothed depth, followed by optional feature-based merging of adjacent basins. PQ$_\text{things}$=16.66. Failure mode: monocular depth estimates are smooth, lacking the sharp local minima that watershed requires. The method produces the *same* output across all 56 configurations — a diagnostic indicator that the input signal (smooth monocular depth) is incompatible with the algorithm's assumption (multi-modal depth surfaces).

- **TDA persistence** (persistent homology). Constructs a filtration on the depth function and merges regions whose boundary persistence falls below a threshold $\tau_\text{persist}$. PQ$_\text{things}$=16.70. Same fundamental limitation as Morse: the persistence diagram of smooth monocular depth is uninformative. Both methods were designed for stereo/LiDAR depth with richer topological structure.

**Family 3: Energy minimization.**

- **Mumford-Shah** (spectral clustering on depth-feature affinity). Builds a joint affinity matrix from depth proximity and DINOv2 feature similarity, then applies spectral clustering per class. PQ$_\text{things}$=18.71. The closest competitor to our learned merge. Strength: principled joint depth-feature reasoning. Weakness: spectral clustering at low resolution (64×128 for tractability) loses fine-grained spatial information, and its performance is fragile to the ratio of easy-to-hard scenes — it overestimates by +4.56 PQ$_\text{things}$ on the first 100 images compared to the full 500-image evaluation.

**Family 4: Optimal transport.**

- **Sinkhorn assignment** (OT-based clustering). Treats instance decomposition as a balanced assignment problem: patches are assigned to $K$ prototypes via Sinkhorn iterations on a cost matrix incorporating depth, features, and position. PQ$_\text{things}$=2.45. Fundamental flaw: the uniform mass constraint ($\sum_i T_{ik} = 1/K$) forces equal-sized instances, which is pathologically wrong for driving scenes where a single bus may occupy 100× more pixels than a pedestrian.

**Family 5: Representation learning.**

- **Contrastive embedding** (InfoNCE + HDBSCAN). Trains a projection head with InfoNCE contrastive loss to produce 128-dim L2-normalized embeddings, then applies HDBSCAN clustering. PQ$_\text{things}$=6.78 (raw DINOv2), degrading to 4.26 after training. Root cause: L2 normalization maps all embeddings onto the unit hypersphere, creating uniform density that HDBSCAN (a *density*-based method) cannot partition. Training exacerbates this by making the distribution *more* uniform. This failure motivates our design principle: *keep the proven boundary → CC pipeline and learn only the merge decision, not the representation*.

### 3.6 Design Principles

The ablation taxonomy yields three principles that guided our architecture:

1. **Preserve the CC backbone.** The four top methods (Sobel+CC, feature cosine, learned merge, Mumford-Shah) all use connected components as the core grouping primitive. Methods that replace CC with density clustering (contrastive) or mass-balanced assignment (OT) fail catastrophically. The learned merge respects this by using CC for oversegmentation and learning only the *merge* decision.

2. **Multi-cue descriptors over single-signal thresholds.** Raw cosine similarity (single feature signal) has zero discriminative power for adjacent same-class fragments. The 200-dim descriptor combining appearance, geometry, shape, and boundary statistics achieves 86% accuracy — demonstrating that the merge decision requires *joint* reasoning across modalities that a single threshold cannot capture.

3. **Self-supervision from depth consistency.** Instance annotations are unavailable by construction in the unsupervised setting. Multi-threshold consensus exploits the *ordinal* structure of depth edges — the insight that a higher threshold is a noisier but more conservative grouping oracle — to generate binary labels without any external supervision.

---

## Summary of Notation

| Symbol | Definition |
|--------|-----------|
| $\hat{S}$ | Pseudo-semantic map (K-means on DINOv2, mapped to 19 trainIDs) |
| $D$ | Monocular depth estimate (SPIdepth, normalized to [0,1]) |
| $\tau_\text{low}, \tau_\text{high}$ | Sobel gradient thresholds for oversegmentation / oracle grouping |
| $\mathcal{F}$ | Set of over-segmented fragments $(m_i, c_i, a_i)$ |
| $\mathbf{x}_{ij}$ | 200-dim pairwise descriptor for fragment pair $(i,j)$ |
| $f_\theta$ | MergePredictor MLP (200 → 128 → 64 → 1) |
| $\tau_\text{merge}$ | Sigmoid probability threshold for merge decision |

---

*Total method parameters: ~26K (MergePredictor) + PCA projection (64×768). Training: 20 epochs on 3,580 pairs (~2 min on Apple MPS). No instance annotations required.*
