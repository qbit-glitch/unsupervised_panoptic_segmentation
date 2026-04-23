# SIMCF: Semantic-Instance Mutual Consistency Filtering

**Full name:** Semantic-Instance Mutual Consistency Filtering with Depth-Aware Boundary Correction

**Acronym:** SIMCF-ABC (steps A, B, C applied sequentially)

---

## 1. Pipeline Overview

```
                         SIMCF Pipeline
    ========================================================

    INPUTS (per image, independently generated)
    ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐
    │  Semantic Map    │  │  Instance Map    │  │  Depth Map   │
    │  S ∈ {0..79}^HW │  │  I ∈ N^HW        │  │  D ∈ R+^HW   │
    │  (k=80 clusters)│  │  (DepthPro+CC)   │  │  (DepthPro)  │
    └────────┬────────┘  └────────┬─────────┘  └──────┬───────┘
             │                    │                    │
             │  ┌─────────────────────────────┐       │
             │  │        STEP A               │       │
             │  │  Instance → Semantic        │       │
             │  │                             │       │
             │  │  For each instance Iₖ:      │       │
             │  │    c* = mode(φ(S[Iₖ]))      │       │
             │  │    Fix: S[p] ← argmax       │       │
             │  │      cluster matching c*    │       │
             │  │                             │       │
             │  │  [0 pixels changed — no-op  │       │
             │  │   when I derives from S]    │       │
             │  └──────────────┬──────────────┘       │
             │                 │                       │
    ┌────────┴─────────────────┴───────────┐          │
    │              STEP B                   │          │
    │  Semantic → Instance                  │          │
    │                                       │          │
    │  ┌─────────────────────────────────┐  │          │
    │  │  1. Build adjacency graph G     │  │          │
    │  │     Nodes: instances {Iₖ}       │  │          │
    │  │     Edges: 3px dilation overlap │  │          │
    │  │                                 │  │          │
    │  │  2. For each edge (Iₐ, Iᵦ):    │  │          │
    │  │     IF c(Iₐ) = c(Iᵦ)           │  │          │
    │  │     AND cos(fₐ, fᵦ) > τ_sim    │  │          │
    │  │     THEN merge via union-find   │  │          │
    │  │                                 │  │          │
    │  │  τ_sim = 0.85                   │  │          │
    │  │  f = mean-pool DINOv3 patches   │  │          │
    │  │                                 │  │          │
    │  │  [7,252 merges, 2.4/img]        │  │          │
    │  └─────────────────────────────────┘  │          │
    └──────────────────┬────────────────────┘          │
                       │                               │
    ┌──────────────────┴───────────────────────────────┴──┐
    │                    STEP C                            │
    │  Depth → Semantic                                    │
    │                                                      │
    │  Pass 1 (global): Compute per-class depth profiles   │
    │    μ_c = E[D | φ(S)=c],  σ_c = Std[D | φ(S)=c]     │
    │                                                      │
    │  Pass 2 (per-image): Mask outliers                   │
    │    ∀p: if |D(p) - μ_c(p)| > 3σ_c(p) → S(p) = 255   │
    │                                                      │
    │  [85M pixels masked, 1.36%]                          │
    └──────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────┴──────────────────┐
    │           OUTPUT                     │
    │  S' ∈ {0..79, 255}^HW  (refined)   │
    │  I' ∈ N^HW             (merged)    │
    │  + regenerated .pt distributions    │
    └─────────────────────────────────────┘
```

---

## 2. Mathematical Formulation

### Notation

| Symbol | Definition |
|--------|-----------|
| S(p) | Semantic cluster ID at pixel p, S: Ω → {0, ..., K-1}, K=80 |
| I(p) | Instance ID at pixel p, I: Ω → N₀ |
| D(p) | Depth at pixel p, D: Ω → R₊ |
| F(p) | DINOv3 feature at patch p, F: Ω_patch → R^768, ‖F(p)‖₂=1 |
| φ(s) | Cluster-to-class mapping, φ: {0..79} → {0..18, 255} |
| Iₖ | Set of pixels belonging to instance k: {p ∈ Ω : I(p) = k} |
| Ω | Image domain, H × W = 1024 × 2048 |
| Ω_patch | Patch domain, 32 × 64 = 2048 patches |

### Step A — Instance Validates Semantics

For each instance k with Iₖ ≠ ∅:

1. **Majority class:**
$$c_k^* = \arg\max_{c \in \{0..18\}} \sum_{p \in I_k} \mathbb{1}[\phi(S(p)) = c]$$

2. **Inconsistent pixels:**
$$\mathcal{M}_k = \{p \in I_k : \phi(S(p)) \neq c_k^* \wedge \phi(S(p)) < 19\}$$

3. **Best replacement cluster:**
$$s_k^* = \arg\max_{s \in \{0..79\}} \sum_{p \in I_k \setminus \mathcal{M}_k} \mathbb{1}[S(p) = s]$$

4. **Reassignment:** $\forall p \in \mathcal{M}_k: S(p) \leftarrow s_k^*$

### Step B — Semantics Validate Instances

1. **Adjacency graph** $G = (V, E)$:
    - $V = \{k : |I_k| > 0, k > 0\}$
    - $(a, b) \in E \iff \text{dilate}(I_a, r=3) \cap I_b \neq \emptyset$

2. **Per-instance features** (mean-pooled at patch resolution):
$$\bar{f}_k = \frac{1}{|\tilde{I}_k|} \sum_{p \in \tilde{I}_k} F(p), \quad \hat{f}_k = \frac{\bar{f}_k}{\|\bar{f}_k\|_2}$$

where $\tilde{I}_k$ is instance $k$ downsampled to 32×64.

3. **Merge criterion** for edge $(a, b) \in E$:
$$\text{merge}(a, b) \iff c_a^* = c_b^* \wedge \langle \hat{f}_a, \hat{f}_b \rangle > \tau_{\text{sim}}$$

with $\tau_{\text{sim}} = 0.85$.

4. **Union-find** with path compression resolves transitive merges. Renumber contiguously.

### Step C — Depth Validates Semantics

1. **Global depth statistics** (Welford's online algorithm):
$$\mu_c = \frac{1}{N_c} \sum_{n=1}^{N} \sum_{p: \phi(S^{(n)}(p))=c} D^{(n)}(p)$$
$$\sigma_c^2 = \frac{1}{N_c} \sum_{n=1}^{N} \sum_{p: \phi(S^{(n)}(p))=c} \left(D^{(n)}(p) - \mu_c\right)^2$$

where $N$ = 2,975 images and $N_c = \sum_n |\{p : \phi(S^{(n)}(p)) = c\}|$.

2. **Outlier masking:**
$$S(p) \leftarrow 255 \quad \text{if} \quad \frac{|D(p) - \mu_{\phi(S(p))}|}{\sigma_{\phi(S(p))}} > \lambda_\sigma$$

with $\lambda_\sigma = 3.0$.

---

## 3. Key Properties

**Compositionality.** Steps A, B, C are sequentially composable: A modifies S, B modifies I, C modifies S. No circular dependencies.

**Monotonicity.** Step A never increases semantic inconsistency within instances. Step B never splits instances. Step C never assigns new classes, only masks.

**Modality coupling:**

| Step | Reads | Modifies | Cross-modal signal |
|------|-------|----------|-------------------|
| A | I, S, φ | S | Instance structure → semantic labels |
| B | S, I, F, φ | I | Semantic class + feature geometry → instance topology |
| C | S, D, φ | S | Depth statistics → semantic plausibility |

**Computational cost:** O(N × H × W) per step. No GPU required. Total: 17 minutes on M4 Pro for 2,975 images.

---

## 4. Empirical Impact Summary

| Metric | Baseline (A0) | SIMCF-ABC (A3) | Delta |
|--------|------------:|---------------:|------:|
| PQ | 24.54 | **25.27** | **+0.73** |
| PQ_stuff | 33.43 | 33.73 | +0.30 |
| PQ_things | 12.31 | **13.64** | **+1.33** |
| mIoU | 56.56 | 56.57 | +0.01 |
| Ignore % | 0.0 | 1.4 | +1.4 |

**Dominant contributor:** Step B (instance merging) drives +1.33 PQ_things. Bus gains +8.8 PQ from de-fragmentation. Step C contributes +0.30 PQ_stuff via boundary sharpening. Step A is structurally inert for CUPS-derived labels.

---

## 5. Proposed Extensions (SIMCF-v2)

Five enhancement passes (D, E, F, G, H) proposed for ablation:

| Pass | Name | Replaces/Extends | Target |
|------|------|-----------------|--------|
| D | SDAIR — Spectral Depth-Aware Instance Refinement | New (instance splitting) | Person PQ 2.6→10+ |
| E | WBIM — Wasserstein Barycentric Instance Merging | Step B cosine | Car FP reduction |
| F | ITCBS — Info-Theoretic Cluster Boundary Sharpening | New (boundary) | Road/sidewalk confusion |
| G | DCCPR — Depth-Conditioned Class Prior Regularization | Step C masking→reassignment | Recover masked pixels |
| H | GSID — Grassmannian Subspace Instance Discrimination | Step B cosine (alternative) | Rider regression fix |
