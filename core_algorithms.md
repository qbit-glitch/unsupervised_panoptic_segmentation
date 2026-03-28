# CutS3D: Core Algorithms in CLRS Format

> Reference: Sick et al., "CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation," ICCV 2025.

---

## Algorithm 1: COMPUTE-AFFINITY-MATRIX

Computes the semantic affinity matrix from DINO ViT-S/8 patch features using cosine similarity.

```
COMPUTE-AFFINITY-MATRIX(F)
────────────────────────────────────────────────────────
  Input:  F ∈ ℝ^{K×C}        — patch-level feature map (K = H'×W' patches, C = 384 channels)
  Output: W ∈ ℝ^{K×K}        — semantic affinity matrix

  1  for i = 1 to K
  2      f̂_i ← F[i] / ‖F[i]‖₂                          ▷ L2-normalize each patch feature
  3  for i = 1 to K
  4      for j = 1 to K
  5          W[i,j] ← f̂_i · f̂_j                         ▷ cosine similarity (dot product of unit vectors)
  6  return W
```

**Complexity:** O(K² · C) time, O(K²) space.
**Note:** W[i,j] ∈ [-1, 1]. This is the raw cosine affinity used before Spatial Importance Sharpening.

---

## Algorithm 2: NORMALIZED-CUT (NCut)

Solves the Normalized Cut problem on the semantic affinity graph to obtain the second-smallest eigenvector, yielding a semantic bipartition.

```
NORMALIZED-CUT(W, τ_ncut)
────────────────────────────────────────────────────────
  Input:  W ∈ ℝ^{K×K}        — semantic affinity matrix
          τ_ncut ∈ ℝ          — binarization threshold for second eigenvector
  Output: B ∈ {0,1}^K         — semantic bipartition (foreground mask)
          λ_max, λ_min        — maximum/minimum absolute eigenvalues
          x ∈ ℝ^K             — second-smallest eigenvector

  1  Z ← DIAG-MATRIX(K)                                  ▷ Z[i,i] = Σ_j W[i,j]  (degree matrix)
  2  for i = 1 to K
  3      Z[i,i] ← Σ_{j=1}^{K} W[i,j]
  4  ▷ Solve generalized eigenvalue problem: (Z - W)x = λZx
  5  L ← Z - W                                           ▷ graph Laplacian
  6  Z_inv_sqrt ← DIAG(1/√Z[i,i] for i = 1..K)
  7  L_norm ← Z_inv_sqrt · L · Z_inv_sqrt                ▷ symmetric normalized Laplacian
  8  eigenvalues, eigenvectors ← EIGH(L_norm)             ▷ ascending eigenvalue decomposition
  9  x ← eigenvectors[:, 1]                               ▷ second-smallest eigenvector (Fiedler vector)
 10  ▷ Identify semantically extreme points
 11  λ_max ← argmax_e |eigenvalues[e]|                    ▷ index of max absolute eigenvalue
 12  λ_min ← argmin_e |eigenvalues[e]|                    ▷ index of min absolute eigenvalue
 13  ▷ Binarize to obtain foreground/background partition
 14  for i = 1 to K
 15      if x[i] > τ_ncut
 16          B[i] ← 1                                     ▷ foreground
 17      else
 18          B[i] ← 0                                     ▷ background
 19  ▷ Choose partition containing fewer patches as foreground
 20  if Σ B[i] > K/2
 21      B ← 1 - B                                        ▷ flip if majority is foreground
 22  return B, λ_max, λ_min, x
```

**Complexity:** O(K³) for eigendecomposition (dominates), O(K²) space.

---

## Algorithm 3: SPATIAL-IMPORTANCE-SHARPENING

Computes a Spatial Importance map from depth to sharpen the semantic affinity matrix at 3D object boundaries.

```
SPATIAL-IMPORTANCE-SHARPENING(W, D, σ_gauss, β)
────────────────────────────────────────────────────────
  Input:  W ∈ ℝ^{K×K}        — semantic affinity matrix
          D ∈ ℝ^{H'×W'}      — depth map (resized to patch resolution)
          σ_gauss ∈ ℝ         — Gaussian kernel standard deviation
          β ∈ ℝ               — lower bound for normalized importance (default 0.45)
  Output: W' ∈ ℝ^{K×K}       — sharpened affinity matrix

  ▷ Step 1: Compute Spatial Importance (Eq. 1)
  1  G_σ ← GAUSSIAN-KERNEL(σ_gauss)                      ▷ 2D Gaussian low-pass filter
  2  D_blur ← G_σ ∗ D                                    ▷ convolve depth with Gaussian
  3  ΔD ← |D_blur - D|                                   ▷ absolute high-frequency depth residual

  ▷ Step 2: Normalize to [β, 1.0] (Eq. 2)
  4  ΔD_min ← min(ΔD)
  5  ΔD_max ← max(ΔD)
  6  for each pixel (i,j)
  7      ΔD_n[i,j] ← (1 - β) · (ΔD[i,j] - ΔD_min) / (ΔD_max - ΔD_min + ε) + β

  ▷ Step 3: Flatten to patch-level importance vector
  8  δ ∈ ℝ^K ← FLATTEN(ΔD_n)                             ▷ one importance value per patch

  ▷ Step 4: Sharpen affinity via element-wise exponentiation (Eq. 3)
  9  for i = 1 to K
 10      for j = 1 to K
 11          exponent ← 1 - δ[i] · δ[j]                  ▷ joint importance of patch pair
 12          W'[i,j] ← W[i,j]^exponent                   ▷ sharpen: high importance → exponent≈0 → W'≈1
 13  return W'
```

**Intuition:** Where ΔD is high (object boundaries), the exponent approaches 0, pushing W'→1, which preserves strong affinities across boundaries. Where ΔD is low (flat regions), exponent approaches 1-β², leaving affinities mostly unchanged.

**Note on Eq. 3 in paper:** The paper writes `W_{i,j}^{1-ΔD_{n_{i,j}}}` using a single index. In practice, for a pairwise matrix, the sharpening uses the importance at patch i (the row). We use the geometric mean of both patches' importance for symmetry.

---

## Algorithm 4: LOCALCUT — Cutting Instances in 3D

Separates individual instances from a semantic group by performing MinCut on a k-NN graph constructed in 3D space.

```
LOCALCUT(B, D, F, τ_knn, k, λ_max, λ_min)
────────────────────────────────────────────────────────
  Input:  B ∈ {0,1}^K         — semantic bipartition (foreground mask)
          D ∈ ℝ^{H'×W'}       — depth map at patch resolution
          F ∈ ℝ^{K×C}         — patch features
          τ_knn ∈ ℝ            — edge weight threshold for k-NN graph
          k ∈ ℤ⁺              — number of nearest neighbors
          λ_max               — index of point at max absolute eigenvalue (from NCut)
          λ_min               — index of point at min absolute eigenvalue (from NCut)
  Output: M ∈ {0,1}^K         — instance binary mask

  ▷ Step 1: Unproject to 3D point cloud
  1  P ← PIXELS-TO-3D(D)                                 ▷ P ∈ ℝ^{K×3}, orthographic unprojection
  2  for i = 1 to K
  3      if B[i] = 0                                      ▷ background points
  4          P[i].z ← z_background                        ▷ set to far-plane depth (pushes bg away in 3D)

  ▷ Step 2: Construct k-NN graph in 3D
  5  V ← {p_1, p_2, ..., p_K}                            ▷ vertex set = all 3D points
  6  E ← ∅                                               ▷ edge set
  7  for i = 1 to K
  8      neighbors ← k-NEAREST-NEIGHBORS(P, P[i], k)     ▷ k closest by Euclidean distance
  9      for each j ∈ neighbors
 10          c ← ‖P[i] - P[j]‖₂                          ▷ Euclidean distance as edge weight
 11          if c ≤ τ_knn
 12              E ← E ∪ {(i, j, c)}                     ▷ add edge with capacity c
 13  G^3D ← (V, E)

  ▷ Step 3: Define source and sink from NCut eigenvalues
 14  s ← λ_max                                           ▷ source = most foreground point
 15  t ← λ_min                                           ▷ sink = most background point

  ▷ Step 4: Solve MinCut via max-flow (Dinic's algorithm)
 16  max_flow ← DINIC-MAX-FLOW(G^3D, s, t)               ▷ find maximum flow
 17  S_cut, T_cut ← EXTRACT-MIN-CUT(G^3D, s)             ▷ partition into source/sink side

  ▷ Step 5: Extract instance mask
 18  for i = 1 to K
 19      if i ∈ S_cut and B[i] = 1
 20          M[i] ← 1                                     ▷ instance pixel (foreground + source side)
 21      else
 22          M[i] ← 0

  23 return M
```

**Complexity:** O(K · k · log K) for k-NN construction, O(V² · E) for Dinic's max-flow.

---

## Algorithm 5: DINIC-MAX-FLOW

Dinic's algorithm for maximum flow / minimum cut. Used inside LocalCut.

```
DINIC-MAX-FLOW(G, s, t)
────────────────────────────────────────────────────────
  Input:  G = (V, E)          — directed graph with edge capacities c(u,v)
          s                   — source node
          t                   — sink node
  Output: max_flow ∈ ℝ        — maximum flow value

  1  Initialize flow f(u,v) ← 0 for all (u,v) ∈ E
  2  max_flow ← 0

  3  while BFS-LEVEL-GRAPH(G_f, s, t) ≠ NIL do           ▷ while augmenting path exists
  4      level ← BFS-LEVEL-GRAPH(G_f, s, t)              ▷ build level graph via BFS
  5      repeat
  6          Δ ← DFS-BLOCKING-FLOW(G_f, s, t, level, ∞)  ▷ find blocking flow via DFS
  7          max_flow ← max_flow + Δ
  8      until Δ = 0
  9  return max_flow

BFS-LEVEL-GRAPH(G_f, s, t)
  ▷ Standard BFS from s in residual graph G_f
  ▷ Returns level array or NIL if t unreachable

DFS-BLOCKING-FLOW(G_f, u, t, level, pushed)
  1  if u = t
  2      return pushed
  3  for each edge (u, v) in G_f with residual capacity > 0
  4      if level[v] = level[u] + 1
  5          d ← DFS-BLOCKING-FLOW(G_f, v, t, level, min(pushed, residual(u,v)))
  6          if d > 0
  7              f(u,v) ← f(u,v) + d
  8              f(v,u) ← f(v,u) - d
  9              return d
 10  return 0

EXTRACT-MIN-CUT(G_f, s)
  ▷ After max-flow, BFS from s in residual graph
  ▷ S_cut = reachable vertices, T_cut = unreachable
```

**Complexity:** O(V² · E) for general graphs.

---

## Algorithm 6: SPATIAL-CONFIDENCE

Computes per-patch confidence maps by performing LocalCut at multiple τ_knn thresholds and measuring mask stability.

```
SPATIAL-CONFIDENCE(B, D, F, τ_knn_min, τ_knn_max, T, k, λ_max, λ_min)
────────────────────────────────────────────────────────
  Input:  B ∈ {0,1}^K          — semantic bipartition
          D ∈ ℝ^{H'×W'}        — depth map
          F ∈ ℝ^{K×C}          — patch features
          τ_knn_min ∈ ℝ         — minimum knn threshold (default 0.5)
          τ_knn_max ∈ ℝ         — maximum knn threshold
          T ∈ ℤ⁺               — number of threshold samples
          k ∈ ℤ⁺               — k for k-NN graph
          λ_max, λ_min          — source/sink indices from NCut
  Output: SC ∈ [0,1]^K         — spatial confidence map

  ▷ Step 1: Compute binary cuts at T different thresholds (Eq. 4)
  1  SC ← zeros(K)
  2  for t = 1 to T
  3      τ_t ← τ_knn_min + t · (τ_knn_max - τ_knn_min) / T    ▷ linearly sample threshold
  4      BC_t ← LOCALCUT(B, D, F, τ_t, k, λ_max, λ_min)       ▷ binary cut at threshold τ_t
  5      SC ← SC + BC_t

  ▷ Step 2: Average over all T cuts
  6  SC ← SC / T                                               ▷ SC[i] ∈ [0, 1]

  7  return SC
```

**Intuition:** Patches consistently assigned to the instance across all thresholds get SC → 1 (high confidence). Patches that flip between foreground/background at different thresholds get intermediate SC values (low confidence, likely at object boundaries).

**Complexity:** O(T × LOCALCUT_cost).

---

## Algorithm 7: CONFIDENT-COPY-PASTE-SELECTION

Selects only high-confidence pseudo-masks for copy-paste data augmentation during CAD training.

```
CONFIDENT-COPY-PASTE-SELECTION(masks, SC_maps, top_fraction)
────────────────────────────────────────────────────────
  Input:  masks                — list of N pseudo-masks for an image
          SC_maps              — corresponding spatial confidence maps
          top_fraction ∈ (0,1] — fraction of masks to keep
  Output: selected_masks       — subset of high-confidence masks

  1  scores ← empty array of size N
  2  for i = 1 to N
  3      scores[i] ← MEAN(SC_maps[i])                    ▷ average confidence over all patches in mask
  4  sorted_indices ← ARGSORT-DESCENDING(scores)
  5  n_select ← ⌈top_fraction × N⌉
  6  selected_masks ← {masks[sorted_indices[j]] : j = 1 to n_select}
  7  return selected_masks
```

---

## Algorithm 8: CONFIDENCE-ALPHA-BLENDING

Alpha-blends pasted objects using Spatial Confidence maps during copy-paste augmentation.

```
CONFIDENCE-ALPHA-BLENDING(I_S, I_T, M, SC)
────────────────────────────────────────────────────────
  Input:  I_S ∈ ℝ^{H×W×3}     — source image (object to paste)
          I_T ∈ ℝ^{H×W×3}     — target image (destination)
          M ∈ {0,1}^{H×W}     — binary instance mask (upsampled to pixel resolution)
          SC ∈ [0,1]^{H×W}    — spatial confidence map (upsampled to pixel resolution)
  Output: I_aug ∈ ℝ^{H×W×3}   — augmented image

  ▷ Eq. 5 from paper
  1  for each pixel (i, j)
  2      if M[i,j] = 0
  3          α[i,j] ← 0                                   ▷ outside mask: no blending
  4      else
  5          α[i,j] ← SC[i,j]                             ▷ confidence as alpha
  6      I_aug[i,j] ← α[i,j] · I_S[i,j] + (1 - α[i,j]) · I_T[i,j]
  7  return I_aug
```

**Intuition:** High-confidence regions are fully pasted (opaque), low-confidence regions are blended with the target, creating smoother augmentations.

---

## Algorithm 9: SPATIAL-CONFIDENCE-SOFT-TARGET-LOSS

Computes per-patch weighted BCE loss using Spatial Confidence for CAD training.

```
SPATIAL-CONFIDENCE-SOFT-TARGET-LOSS(M̂, M, SC)
────────────────────────────────────────────────────────
  Input:  M̂ ∈ [0,1]^{H×W}     — predicted mask (sigmoid output)
          M ∈ {0,1}^{H×W}      — target pseudo-mask
          SC ∈ [0,1]^{H×W}     — spatial confidence map
  Output: L_mask ∈ ℝ            — scalar loss

  ▷ Eq. 6 from paper
  1  L_mask ← 0
  2  for each pixel (i, j)
  3      bce ← -[M[i,j] · log(M̂[i,j]) + (1 - M[i,j]) · log(1 - M̂[i,j])]
  4      L_mask ← L_mask + SC[i,j] · bce
  5  ▷ Note: SC[i,j] = 1 for pixels outside the confidence map (non-instance regions)
  6  return L_mask / (H × W)
```

**Key insight:** Instead of multiplying the entire mask loss by a single scalar (as in CuVLER's soft target loss), CutS3D weights each pixel independently. Confident patches contribute more to the gradient; uncertain boundary patches contribute less.

---

## Algorithm 10: CUTS3D-PSEUDO-MASK-EXTRACTION (Full Pipeline)

The complete CutS3D pseudo-mask extraction pipeline combining all components.

```
CUTS3D-PSEUDO-MASK-EXTRACTION(I, N_max, τ_ncut, τ_knn, k, σ_gauss, β)
────────────────────────────────────────────────────────
  Input:  I ∈ ℝ^{H×W×3}       — input RGB image (resized to 480×480)
          N_max ∈ ℤ⁺           — maximum number of segmentation iterations
          τ_ncut ∈ ℝ            — NCut binarization threshold
          τ_knn ∈ ℝ             — k-NN graph edge threshold for LocalCut
          k ∈ ℤ⁺               — k for k-NN graph
          σ_gauss ∈ ℝ           — Gaussian blur sigma for Spatial Importance
          β ∈ ℝ                — lower bound for Spatial Importance normalization
  Output: masks                — list of instance pseudo-masks
          SC_maps              — list of spatial confidence maps

  ▷ Step 1: Extract features and depth
  1  F ← DINO-VIT-S/8(I)                                 ▷ F ∈ ℝ^{K×C}, K = (H/8)×(W/8), C = 384
  2  D ← MONOCULAR-DEPTH(I)                              ▷ D ∈ ℝ^{H×W}, e.g., ZoeDepth
  3  D_patch ← RESIZE(D, H' = H/8, W' = W/8)            ▷ resize depth to patch resolution

  ▷ Step 2: Compute sharpened affinity matrix
  4  W ← COMPUTE-AFFINITY-MATRIX(F)                      ▷ Algorithm 1
  5  W ← SPATIAL-IMPORTANCE-SHARPENING(W, D_patch, σ_gauss, β)  ▷ Algorithm 3

  ▷ Step 3: Iterative mask extraction
  6  masks ← []
  7  SC_maps ← []
  8  active_mask ← ones(K)                               ▷ track remaining active patches

  9  for n = 1 to N_max
 10      ▷ NCut on remaining patches
 11      W_sub ← W[active, active]                       ▷ subgraph of active patches
 12      B, λ_max, λ_min, x ← NORMALIZED-CUT(W_sub, τ_ncut)  ▷ Algorithm 2

 13      ▷ Early stopping: if predicting "the rest" (inverse of all previous)
 14      if B represents the complement of all prior masks
 15          break

 16      ▷ LocalCut: cut instance in 3D
 17      M ← LOCALCUT(B, D_patch, F, τ_knn, k, λ_max, λ_min)  ▷ Algorithm 4

 18      ▷ CRF refinement
 19      M_refined ← CRF-REFINE(M, I)                    ▷ dense CRF post-processing

 20      ▷ Size filtering
 21      if SUM(M_refined) / K < min_mask_size
 22          continue                                     ▷ skip tiny masks

 23      ▷ Compute Spatial Confidence
 24      SC ← SPATIAL-CONFIDENCE(B, D_patch, F, τ_knn_min, τ_knn_max, T, k, λ_max, λ_min)  ▷ Alg 6

 25      ▷ Store results
 26      masks.APPEND(M_refined)
 27      SC_maps.APPEND(SC)

 28      ▷ Remove segmented patches from active set
 29      for i = 1 to K
 30          if M_refined[i] = 1
 31              active_mask[i] ← 0

 32  return masks, SC_maps
```

---

## Algorithm 11: CUTS3D-CAD-TRAINING (Detector Training with Self-Training)

Full pipeline for training the Class-Agnostic Detector on CutS3D pseudo-masks.

```
CUTS3D-CAD-TRAINING(dataset, R_self_train)
────────────────────────────────────────────────────────
  Input:  dataset              — ImageNet-1K (IN1K) training set (~1.3M images)
          R_self_train ∈ ℤ⁺    — number of self-training rounds (default 3)
  Output: θ_CAD                — trained CAD parameters

  ▷ Phase 1: Generate pseudo-masks on entire IN1K
  1  pseudo_data ← {}
  2  for each image I in dataset
  3      masks, SC_maps ← CUTS3D-PSEUDO-MASK-EXTRACTION(I, ...)  ▷ Algorithm 10
  4      pseudo_data[I] ← (masks, SC_maps)

  ▷ Phase 2: Initial CAD training on pseudo-masks
  5  θ_CAD ← INITIALIZE-CASCADE-MASK-RCNN()
  6  for each epoch
  7      for each batch (I, masks, SC_maps)
  8          ▷ Confident copy-paste augmentation
  9          selected ← CONFIDENT-COPY-PASTE-SELECTION(masks, SC_maps, top_frac)  ▷ Alg 7
 10          I_aug ← CONFIDENCE-ALPHA-BLENDING(I_source, I_target, M, SC)         ▷ Alg 8

 11          ▷ Forward pass through CAD
 12          M̂, scores ← CASCADE-MASK-RCNN(I_aug; θ_CAD)

 13          ▷ Compute loss with Spatial Confidence Soft Target Loss
 14          L ← SPATIAL-CONFIDENCE-SOFT-TARGET-LOSS(M̂, M_target, SC) + L_box + L_cls  ▷ Alg 9
 15          θ_CAD ← θ_CAD - η · ∇_θ L

  ▷ Phase 3: Self-training rounds
 16  for r = 1 to R_self_train
 17      ▷ Generate new pseudo-labels using current CAD
 18      for each image I in dataset
 19          masks_new ← CASCADE-MASK-RCNN(I; θ_CAD)      ▷ predict with current model
 20      ▷ Retrain CAD on new pseudo-labels
 21      θ_CAD ← TRAIN-CAD(masks_new, ...)                ▷ same loss as Phase 2

 22  return θ_CAD
```

---

## Algorithm 12: PIXELS-TO-3D (Orthographic Unprojection)

Projects 2D pixel coordinates to 3D points using a depth map.

```
PIXELS-TO-3D(D, fx, fy, cx, cy)
────────────────────────────────────────────────────────
  Input:  D ∈ ℝ^{H×W}          — depth map
          fx, fy ∈ ℝ            — focal lengths
          cx, cy ∈ ℝ            — principal point coordinates
  Output: P ∈ ℝ^{H×W×3}        — 3D point cloud

  1  for v = 0 to H-1
  2      for u = 0 to W-1
  3          z ← D[v, u]
  4          x ← (u - cx) · z / fx
  5          y ← (v - cy) · z / fy
  6          P[v, u] ← (x, y, z)
  7  return P
```

---

## Algorithm 13: CRF-REFINE (Dense CRF Post-Processing)

Refines binary masks using a dense Conditional Random Field.

```
CRF-REFINE(M, I, θ_crf)
────────────────────────────────────────────────────────
  Input:  M ∈ {0,1}^K          — coarse binary mask at patch resolution
          I ∈ ℝ^{H×W×3}        — original RGB image
          θ_crf                 — CRF parameters (sxy, srgb, compat)
  Output: M_refined ∈ {0,1}^{H×W}  — refined mask at pixel resolution

  1  M_up ← UPSAMPLE(M, H, W)                            ▷ bilinear upsampling to image resolution
  2  unary ← COMPUTE-UNARY(M_up)                          ▷ -log(P(label)) for each pixel
  3  ▷ Pairwise potentials: appearance (bilateral) + smoothness (Gaussian)
  4  pairwise_bilateral ← BILATERAL-KERNEL(sxy, srgb, I)
  5  pairwise_gaussian ← GAUSSIAN-KERNEL(sxy_smooth)
  6  M_refined ← MEAN-FIELD-INFERENCE(unary, pairwise_bilateral + pairwise_gaussian, n_iters)
  7  return M_refined
```

---

## Summary of Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| NCut threshold | τ_ncut | 0.0 | Binarization threshold for eigenvector |
| k-NN neighbors | k | 10 | Neighbors in 3D point cloud graph |
| k-NN threshold | τ_knn | 0.115 | Edge weight threshold for LocalCut |
| τ_knn minimum | τ_knn_min | 0.5 × τ_knn | Minimum for Spatial Confidence sweep |
| τ_knn maximum | τ_knn_max | τ_knn | Maximum for Spatial Confidence sweep |
| SC samples | T | 6 | Number of threshold samples for SC |
| Gaussian sigma | σ_gauss | 3.0 | Blur kernel for Spatial Importance |
| SI lower bound | β | 0.45 | Lower bound of normalized SI map |
| Max iterations | N_max | 3 | Max segmentation iterations per image |
| Self-training rounds | R | 3 | Number of self-training rounds |
| Image size | - | 480×480 | Input resolution |
| Patch size | - | 8×8 | DINO ViT-S/8 patch size |
| Feature dim | C | 384 | DINO feature channels |
| SC_knn_min | - | 0.5 | Minimum SC threshold |

---

## Equation Reference

| Eq. | Formula | Location |
|-----|---------|----------|
| (1) | ΔD = \|G_σ ∗ D − D\| | Spatial Importance |
| (2) | ΔD_n = (1−β)·(ΔD − min ΔD)/(max ΔD − min ΔD) + β | SI Normalization |
| (3) | W_{i,j} = W_{i,j}^{1−ΔD_{n_{i,j}}} | Affinity Sharpening |
| (4) | SC_{i,j} = (1/T) Σ_t BC_{i,j}(t) | Spatial Confidence |
| (5) | I^aug = SC · I^S + (1−SC) · I^T | Alpha-Blending |
| (6) | L_mask = Σ SC_{i,j} · BCE(M̂_{i,j}, M_{i,j}) | SC Soft Target Loss |
