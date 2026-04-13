# Novel Instance Decomposition: Brainstormed Approaches

**Date:** 2026-03-28
**Context:** NeurIPS review audit W1 identifies Sobel+CC as "textbook image processing with zero algorithmic novelty." These 6 approaches are principled replacements.

---

## 1. Persistent Homology on Depth Fields (TDA-based)

**Core idea**: Instead of picking a single Sobel threshold (arbitrary), use persistent homology to find boundaries that are stable across ALL thresholds simultaneously.

**How it works**:
- Build a filtration on the depth map: sweep a threshold from 0 to max gradient magnitude
- At each threshold, connected components appear (birth) and merge (death)
- Plot a persistence diagram: each point (birth, death) represents a depth boundary
- **Long-lived features** (high persistence = death - birth) are real object boundaries
- **Short-lived features** are noise
- The persistence diagram gives a principled, threshold-free decomposition

**Why it's novel**: TDA has been used in medical imaging and point clouds but never for unsupervised panoptic pseudo-label generation. The connection between persistent features in depth fields and object instances is unexplored.

**Why it's principled**: Persistence diagrams have stability guarantees (small perturbation in depth = small perturbation in diagram). This is provably robust to depth noise, unlike a fixed threshold.

**Addresses**: W1 (algorithmic novelty), and partly the co-planar problem (persistence naturally handles multi-scale depth structure).

**Feasibility**: Libraries exist (giotto-tda, ripser, gudhi). Compute is ~seconds per image.

---

## 2. Mumford-Shah Functional in Depth-Feature Space

**Core idea**: Frame instance decomposition as energy minimization. Find the partition that simultaneously explains depth AND feature variation with minimal boundary length.

**The energy**:
```
E(Π) = Σ_i [ α·Var_depth(R_i) + β·Var_feature(R_i) ] + γ·|∂Π|
```
where R_i are regions, Var is within-region variance, and |∂Π| is total boundary length.

**Why it's novel**: Mumford-Shah is classical for image segmentation but has never been applied to joint depth+SSL-feature spaces for unsupervised instance discovery. The feature variance term is what handles co-planar objects — two people at the same depth have different DINOv2 features, so the energy prefers to split them.

**Why it's principled**: Mumford-Shah has 40 years of theory (existence of minimizers, Γ-convergence, Ambrosio-Tortorelli approximation). You can solve it via graph cuts (Boykov-Kolmogorov) with provable approximation bounds.

**Solves co-planar problem**: The depth variance term alone can't split co-planar objects, but the feature variance term can. The energy naturally balances when to use depth vs appearance cues.

**Feasibility**: Graph-cut solvers are fast. PyMaxflow or GCO libraries. Minutes per image.

---

## 3. Depth-Conditioned Slot Attention (Learned, Self-Supervised)

**Core idea**: Slot attention (Locatello et al. 2020) discovers objects as competing "slots" that explain the scene. Condition slots on depth to give geometric priors for where objects are.

**Architecture**:
- Input: DINOv2/DINOv3 patch features + depth map
- Depth-initialized slots: instead of random slot init, initialize slot positions from depth modes (peaks in depth histogram = likely objects at different depths)
- Cross-attention between slots and features, with depth-weighted attention (nearby-in-depth pixels attend more strongly to the same slot)
- Output: soft instance masks per slot

**Training**: Self-supervised reconstruction loss — slots must reconstruct the DINOv2 features. No labels needed.

**Why it's novel**: Slot attention for panoptic segmentation exists (DINOSAUR), but depth-conditioned slot initialization + depth-weighted attention is new. This is a learned component that uses depth as a structural prior rather than a hard boundary signal.

**Addresses**: W1 (learned component), co-planar problem (feature-based slot competition), W4 indirectly (slots learn to use depth as a soft cue, not a hard requirement).

**Feasibility**: Slot attention is lightweight (~1M params). Training on pseudo-labels takes hours on a single GPU.

---

## 4. Contrastive Depth-Feature Embedding with Automatic Clustering

**Core idea**: Learn a joint embedding space where "same instance" pixels are close and "different instance" pixels are far, using depth discontinuities as weak self-supervision.

**Method**:
- Take DINOv2 patch features (384-dim) + depth (1-dim) + position (2-dim)
- Train a lightweight projection head (MLP) with contrastive loss:
  - **Positive pairs**: pixels that are spatially close AND have similar depth (likely same instance)
  - **Negative pairs**: pixels separated by a depth discontinuity (likely different instances)
- The learned embedding captures both appearance and geometry
- Cluster the embeddings with HDBSCAN (density-based, no k required)

**Why it's novel**: Using depth discontinuities as a self-supervised signal for contrastive instance embedding learning. The depth doesn't define instances directly — it provides training signal for a learned model that generalizes beyond depth.

**Key advantage**: The learned model can discover instances that depth alone can't (co-planar objects) because the contrastive training teaches it to use appearance features for the hard cases while depth bootstraps the easy cases.

**This is essentially a principled version of iterative self-training**: depth gives noisy labels → learn embeddings → embeddings give better labels → iterate.

**Feasibility**: Very feasible. Small MLP, contrastive loss, HDBSCAN. Hours of training.

---

## 5. Optimal Transport Instance Decomposition

**Core idea**: Model instance decomposition as an optimal transport problem — what is the minimum-cost way to assign pixels to instance prototypes?

**Formulation**:
- Source distribution: pixels with features (depth, DINOv2, position)
- Target: K instance prototypes (discovered automatically)
- Transport cost: Wasserstein distance in depth-feature space
- Constraint: each instance must be spatially contiguous (entropic regularization + spatial penalty)

**Solve via Sinkhorn iterations** with a spatial contiguity regularizer. The Sinkhorn algorithm is differentiable, so you can backprop through it.

**Why it's novel**: OT for instance segmentation is barely explored. The spatial contiguity constraint in Wasserstein space is a new formulation. And OT naturally handles the "how many instances" question via the transport plan's structure.

**Feasibility**: Sinkhorn is efficient (POT library). But scaling to full images might need hierarchical/superpixel approximation.

---

## 6. Depth Gradient Flow Decomposition (Morse Theory-inspired)

**Core idea**: Use the gradient flow of the depth map to define a canonical decomposition into "basins" — regions where all gradient flow lines converge to the same local minimum.

**How it works**:
- Compute the gradient field ∇D of the depth map
- Trace flow lines from each pixel following -∇D (steepest descent)
- All pixels whose flow lines converge to the same sink form one basin
- Basins = candidate instances
- Merge basins that have similar features (DINOv2 cosine similarity > threshold)

**Why it's principled**: This is Morse theory — the decomposition into ascending/descending manifolds is a fundamental object in differential topology. The basins are determined by the critical points (minima, saddle points) of the depth function, not by an arbitrary threshold.

**Key insight vs Sobel+CC**: Sobel+CC asks "is the gradient above a threshold?" (binary, local). Morse decomposition asks "where does this pixel's gradient flow lead?" (global, structural). The Morse decomposition is invariant to smooth deformations of the depth map.

**Addresses co-planar issue partially**: Objects at different heights on the same depth plane create different flow basins. But truly co-planar flat objects (two people standing side-by-side at exact same depth) still need the feature merge step.

**Feasibility**: Gradient flow tracing is fast (flood-fill-like). Can be done in numpy. The feature-based merge step adds the learned/appearance dimension.

---

## Ranking for NeurIPS Impact

| Rank | Approach | Novelty | Theory | Fixes Co-planar | Feasibility |
|------|----------|---------|--------|-----------------|-------------|
| 1 | **Mumford-Shah (2)** | High | Very strong | Yes (feature term) | High |
| 2 | **Contrastive Embedding (4)** | High | Moderate | Yes (learned) | Very high |
| 3 | **Persistent Homology (1)** | Very high | Very strong | Partial | High |
| 4 | **Morse Flow (6)** | High | Strong | Partial | Very high |
| 5 | **Slot Attention (3)** | Moderate | Moderate | Yes (learned) | High |
| 6 | **Optimal Transport (5)** | Very high | Strong | Yes | Moderate |

## Recommended Combination

**Morse/Persistent Homology + Contrastive Refinement** (two-stage):

1. **Stage A** — Principled geometric decomposition (Morse flow OR persistent homology) replaces Sobel+CC. Gives mathematical rigor and threshold-free operation. Handles depth-separated objects.

2. **Stage B** — Contrastive depth-feature embedding refines Stage A output. Learns to split co-planar objects using appearance cues, trained with Stage A's output as weak supervision. Handles the hard cases (persons, adjacent cars).

This gives:
- Mathematical novelty (TDA/Morse theory, never used for panoptic pseudo-labels)
- A learned component (contrastive embedding)
- A principled framework instead of ad-hoc thresholds
- Direct improvement on the person PQ=4.2 bottleneck
- A clean story: "geometry for the easy cases, learned appearance for the hard cases"
