# MBPS Final Model — Algorithms in CLRS Format

> Companion technical reference to the NeurIPS 2026 MBPS paper.
> All algorithms are extracted **verbatim** from the production codebase and reformulated
> in the style of *Cormen, Leiserson, Rivest, and Stein — Introduction to Algorithms*.
> Variable names, default hyperparameters, and tensor shapes match the implementation.

---

## Notation

| Symbol | Meaning | Shape / Type |
|---|---|---|
| $\mathcal{X}$ | RGB image (Cityscapes train/val split) | $(H, W, 3)$, $\mathrm{uint8}$, $H{=}1024$, $W{=}2048$ |
| $\mathcal{X}^{\downarrow}$ | Working-resolution image | $(H', W', 3)$, $H'{=}512$, $W'{=}1024$ |
| $\mathbf{Z}$ | CAUSE-TR semantic codes (90-D) | $(H_p, W_p, 90)$, $\mathrm{float32}$ |
| $\mathbf{F}$ | DINOv3 ViT-B/16 patch features | $(H_p, W_p, 768)$, $\ell_2$-normalised |
| $\mathbf{D}$ | DepthPro inverse-depth map, scaled to $[0,1]$ | $(H', W')$, $\mathrm{float32}$ |
| $\mathbf{S}$ | Cluster ID map (semantic pseudo-label) | $(H', W')$, $\mathrm{uint8}$, $\in\{0,\ldots,K-1\}\cup\{255\}$ |
| $\mathbf{I}$ | Instance ID map | $(H', W')$, $\mathrm{uint16}$, $0$ = background |
| $K$ | Number of over-clusters | $K{=}80$ |
| $C$ | Cityscapes trainID count | $C{=}19$ |
| $\phi : \{0,\ldots,K-1\} \to \{0,\ldots,C-1,255\}$ | Cluster-to-class LUT | $(K,)$, $\mathrm{uint8}$ |
| $H_p, W_p$ | Patch grid at DINOv2 stride 14 from $448{\times}896$ inputs | $H_p{=}32$, $W_p{=}64$ |
| $\tau_d$ | Depth-gradient threshold | $0.05$ default, $0.20$ used in best Stage-1 (SPIdepth) |
| $A_\min$ | Minimum instance area | $1000$ pixels at $(512{\times}1024)$ |
| $\tau_\mathrm{sim}$ | SIMCF-B cosine-similarity threshold | $0.85$ |
| $\eta$ | SIMCF-C depth $\sigma$-multiplier | $3.0$ |
| $\lambda_\mathrm{preserve}$ | DCFA preservation weight | $20.0$ |
| $\sigma_d$ | DepthG bandwidth | $0.5$ |

Throughout, indices are **0-based**; tensor dimensions are listed in `[outermost, ..., innermost]` order; $\Vert \cdot \Vert$ is the Euclidean ($\ell_2$) norm.

---

# Part I — Stage-1: Monocular Pseudo-Label Generator $G$

The Stage-1 generator $G$ converts a raw image $\mathcal{X}$ into a pair of refined pseudo-labels $(\hat{\mathbf{S}}, \hat{\mathbf{I}})$ that supervise the Cascade Mask R-CNN detector $F$ in Stage-2. The pipeline is:

$$
\mathcal{X} \xrightarrow{\text{CAUSE-TR}} \mathbf{Z}^0 \xrightarrow{\text{DCFA}} \mathbf{Z} \xrightarrow{\text{k-means}+\phi} \mathbf{S}^0,\; \mathcal{X} \xrightarrow{\text{DepthPro}} \mathbf{D} \xrightarrow{\text{Alg.\ 4}} \mathbf{I}^0,\; (\mathbf{S}^0, \mathbf{I}^0) \xrightarrow{\text{SIMCF-ABC}} (\hat{\mathbf{S}}, \hat{\mathbf{I}}).
$$

The only learned component inside $G$ is **DCFA** (≈ 40 K parameters).

---

## Algorithm 1 — `SemanticOverclustering`

Fits a spherical mini-batch $K$-means over per-pixel CAUSE-TR codes and freezes the resulting **(centroids, LUT)** pair $(\mathbf{C}, \phi)$. Implemented in `mbps_pytorch/generate_depth_overclustered_semantics.py:493–619`.

```
SemanticOverclustering(net, segment, images, K, max_per_image, seed)
Input:
    net       — DINOv2 ViT-B/14 backbone, returns 768-D patch features
    segment   — CAUSE-TR Segment_TR head, projects 768 → 90
    images    — list of (image, gt_trainID) pairs over Cityscapes val split
                  image: (H, W, 3) uint8
                  gt:    (H, W) uint8, 255 = ignore
    K         — number of over-clusters, default 80
    max_per_image — pixel cap per image, default 2000
    seed      — RNG seed, default 42
Output:
    C ∈ R^{K×90}                    — ℓ₂-normalised centroids
    φ ∈ {0,…,C-1,255}^{K}           — cluster-to-trainID LUT

 1  ALL_FEATS ← ∅;   ALL_LABELS ← ∅
 2  for each (image, gt) ∈ images do
 3      (H, W) ← shape(image)
 4      s ← crop_size / min(H, W)            // crop_size = 448
 5      H_new ← ⌊H·s / P⌋ · P;  W_new ← ⌊W·s / P⌋ · P    // P = 14
 6      x ← Normalize(Resize(image, (H_new, W_new)), μ_IN, σ_IN)
 7      F ← net.extract(x)                    // (1, 768, H_p, W_p)
 8      Z ← segment.head(F)                   // (1, 90,  H_p, W_p)
 9      H_p ← H_new / P;  W_p ← W_new / P     // (32, 64)
10      Z ← AdaptiveAvgPool2d(Z, (H_p, W_p))
11      z ← reshape(Z, (90, H_p·W_p))ᵀ        // (N, 90), N = H_p·W_p
12      g ← Resize(gt, (H_p, W_p), mode=NEAREST).flatten()
13      valid ← (g ≠ 255)
14      idx ← UniformSample({i : valid[i]}, min(max_per_image, |valid|))
15      ALL_FEATS  ← ALL_FEATS  ⊕ z[idx]
16      ALL_LABELS ← ALL_LABELS ⊕ g[idx]
17  end for

18  ALL_FEATS ← ALL_FEATS / max(‖ALL_FEATS‖₂, 1e-8)       // L2 normalize rows

19  kmeans ← MiniBatchKMeans(K, batch_size=10 000,
                              max_iter=300, n_init=3, seed=seed)
20  kmeans.fit(ALL_FEATS)

21  Π ← kmeans.predict(ALL_FEATS)                          // (M,)
22  conf ∈ ℤ^{K×C} ← 0
23  for (k, c) ∈ zip(Π, ALL_LABELS) do
24      if c < C then conf[k, c] ← conf[k, c] + 1
25  end for
26  φ ← argmax(conf, axis=1)                               // (K,)

27  C ← kmeans.cluster_centers_                            // (K, 90)
28  C ← C / max(‖C‖₂, 1e-8)
29  return (C, φ)
```

**Complexity.** Feature extraction is $O(|\mathrm{images}| \cdot H_p W_p \cdot d_b)$ with $d_b{=}768$; mini-batch k-means converges in $O(M K \cdot 300)$ where $M{\approx} 2\,000 \cdot |\mathrm{val}|$.

**Inference per image.** At test time we run lines 3–11 with the **adapter $f_\theta$ inserted between lines 8 and 10** (see Algorithm 2), then $\mathbf{S}^0[u] = \arg\min_{k} \, 1 - \langle \mathbf{Z}[u], \mathbf{C}[k]\rangle$ and the cluster IDs map to trainIDs via $\phi$ only at evaluation time.

---

## Algorithm 2 — `DCFA-Forward` (Depth-Conditioned Feature Adapter)

The MLP residual adapter (`mbps_pytorch/models/semantic/depth_adapter.py:41–98`). Total parameter count for the default config (`code_dim=90, depth_dim=16, hidden_dim=384, num_layers=2`) is

$$
(91{+}d_e)\cdot h \,+\, h{+}\, h\cdot h\,+\,h\,+\,h\cdot 90\,+\,90 \;\approx\; 40\,\text{K}.
$$

```
DCFA-Forward(z, d ; θ)
Input:
    z ∈ R^{B×N×90}     — frozen CAUSE codes
    d ∈ R^{B×N}        — DepthPro depth in [0, 1]
    θ = (W₁, b₁, γ₁, β₁, …, W_out, b_out)
Hyper-parameters:
    depth_dim d_e ∈ {1, 16};        // 16 ⇒ sinusoidal
    hidden_dim h  ∈ {128, 384};     // best DCFA paper-config: h = 384
    num_layers L  = 2

 1  if d_e = 16 then
 2      e ← SinusoidalEncode(d)                          // (B, N, 16)   // Eq. (1)
 3  else
 4      e ← unsqueeze(d, –1)                             // (B, N, 1)
 5  end if
 6  x ← concat(z, e, axis=–1)                            // (B, N, 90+d_e)
 7  for ℓ = 1 to L do
 8      x ← ReLU( LayerNorm( x · W_ℓ + b_ℓ ) )           // (B, N, h)
 9  end for
10  r ← x · W_out + b_out                                // (B, N, 90)
11  return z + r                                         // ⇐ identity-init: r = 0 at step 0
```

**Sinusoidal depth encoding** (Eq. 1, `depth_adapter.py:23–38`). For frequency bands $\Omega = \{1, 2, 4, 8, 16, 32, 64, 128\}$,

$$
\mathrm{SinusoidalEncode}(d)[\cdot, 2i] = \sin(\omega_i \pi d),\quad
\mathrm{SinusoidalEncode}(d)[\cdot, 2i+1] = \cos(\omega_i \pi d),\;\; i=0,\ldots,7.
$$

The output linear layer is **zero-initialised** ($W_\mathrm{out} = \mathbf{0}$, $b_\mathrm{out} = \mathbf{0}$), so the adapter starts as the identity map $z \mapsto z$ and can only improve the frozen prior.

---

## Algorithm 3 — `DCFA-Train`

Optimises only $\theta$ (the adapter) on pre-extracted CAUSE codes + DepthPro depths. The backbone and head remain frozen. Implemented in `mbps_pytorch/train_depth_adapter.py:186–308`.

```
DCFA-Train(D_train, adapter, epochs, batch_size, lr, λ_preserve, σ_d, P)
Input:
    D_train  — (codes, depth) tensors per Cityscapes train image
    adapter  — DCFA module (Algorithm 2)
    epochs = 20,  batch_size = 8,  lr = 1e-3,
    λ_preserve = 20.0,  σ_d = 0.5,  P = 1024 pair samples
Output:
    θ* — trained DCFA parameters

 1  opt ← AdamW(adapter.parameters(), lr=lr)
 2  sched ← CosineAnnealing(opt, T_max=epochs·|D_train|, η_min=1e-5)
 3  for e = 1 to epochs do
 4      for each minibatch (z, d) ∈ D_train do
 5          ẑ ← DCFA-Forward(z, d ; θ)                  // (B, N, 90)
 6          L_depth     ← DepthG(ẑ, d ; σ_d, P)         // Algorithm 3a, scalar
 7          L_preserve  ← (1 / B N D) · ‖ẑ − z‖_F²       // MSE
 8          L ← L_depth + λ_preserve · L_preserve
 9          opt.zero_grad();  L.backward()
10          ClipGradNorm(adapter.parameters(), max_norm=1.0)
11          opt.step();  sched.step()
12      end for
13  end for
14  return θ*
```

### Algorithm 3a — `DepthG` (depth-guided correlation, `stego_loss.py:116–169`)

$$
\mathcal{L}_\mathrm{depth} \;=\; \frac{1}{B} \sum_{b=1}^{B} \frac{1}{P} \sum_{(i,j) \in \mathcal{P}_b} w_{ij}\,(1 - \cos\langle \hat{z}_i, \hat{z}_j\rangle)^2, \qquad
w_{ij} \;=\; \exp\!\Big(-\frac{(d_i - d_j)^2}{2\sigma_d^2}\Big).
$$

```
DepthG(ẑ, d ; σ_d, P)
 1  L ← 0
 2  for b = 1 to B do
 3      Sample I, J ∼ Uniform{0,…,N-1}^P                  // i.i.d.
 4      Δ ← d[b, I] − d[b, J]                              // (P,)
 5      w ← exp( −Δ² / (2 σ_d²) )                          // (P,)
 6      c_i ← ẑ[b, I];  c_j ← ẑ[b, J]                      // (P, 90)
 7      sim ← Σ_k c_i[k] · c_j[k] / (‖c_i‖ ‖c_j‖ + 1e-8)
 8      L ← L + mean( w · (1 − sim)² )
 9  end for
10  return L / B
```

The **preservation term** $\lambda_\mathrm{preserve} \cdot \mathrm{MSE}(\hat{z}, z)$ acts as a frozen-prior anchor that keeps DCFA close to identity outside high-similarity depth neighbourhoods.

---

## Algorithm 4 — `DepthProInstanceGeneration`

Splits monocular DepthPro depth into instance masks for the eight Cityscapes thing classes $\mathcal{T} = \{11, 12, 13, 14, 15, 16, 17, 18\}$. Implemented in `mbps_pytorch/generate_depth_guided_instances.py:75–154`.

```
DepthProInstanceGeneration(S, D ; T, τ_d, A_min, n_dil, σ_blur)
Input:
    S ∈ {0,…,C-1,255}^{H'×W'}     — semantic pseudo-labels (trainIDs)
    D ∈ [0,1]^{H'×W'}             — DepthPro depth, H'=512, W'=1024
    T                              — thing trainID set
    τ_d   = 0.05,    A_min  = 1000,
    n_dil = 3,       σ_blur = 1.0
Output:
    Π = [(M_n, c_n, s_n)]_{n=1..N}  — list of (mask, class, score)
                                       M_n ∈ {0,1}^{H'×W'}, c_n ∈ T, s_n ∈ [0,1]

 1  D̃ ← GaussianBlur(D, σ=σ_blur)                        // ∈ R^{H'×W'}
 2  G_x ← Sobel(D̃, axis=1);   G_y ← Sobel(D̃, axis=0)
 3  ‖∇D‖ ← √(G_x² + G_y²)                                // (H', W')
 4  E ← (‖∇D‖ > τ_d)                                     // depth edges
 5  A ← 0_{H'×W'}                                        // pixels already assigned
 6  Π ← ∅
 7  for each c ∈ sort(T) do
 8      M_c ← (S = c)                                    // class mask
 9      if Σ M_c < A_min then continue
10      M_split ← M_c ∧ ¬E                               // remove depth jumps
11      (L, n_cc) ← ConnectedComponents(M_split)         // 8-neighbour
12      cclist ← [ (k, L=k, Σ(L=k)) : k = 1..n_cc, Σ(L=k) ≥ A_min ]
13      sort cclist by area descending
14      for (k, M_k, a_k) ∈ cclist do
15          if n_dil > 0 then
16              M̃_k ← BinaryDilation(M_k, iterations=n_dil)
17              M_reclaim ← M̃_k ∧ M_c ∧ ¬A
18              M_final ← M_k ∨ M_reclaim
19          else
20              M_final ← M_k
21          end if
22          if Σ M_final < A_min then continue
23          A ← A ∨ M_final
24          Π ← Π ⊕ (M_final, c, Σ M_final)
25      end for
26  end for
27  sort Π by area descending
28  if Π ≠ ∅ then
29      a_max ← Π[0].area
30      Π ← [ (M, c, a / a_max) for (M, c, a) ∈ Π ]      // normalise scores
31  end if
32  return Π
```

**Output encoding** (`save_instances`, lines 194–237). For each image, a single NPZ contains

```
masks      : (N, H'·W') bool          # one row per instance
scores     : (N,) float32 in [0, 1]
num_valid  : int
h_patches  : 512
w_patches  : 1024
```

In the CUPS file format consumed by Stage-2, the same data is rendered as a `uint16` PNG `*_instance.png` where pixel $u$ holds the instance ID (1-based, $0$ = background).

---

## Algorithm 5 — `SIMCF-A` (Instance validates Semantics)

Within each instance, reassign semantically inconsistent pixels to the **best cluster that maps to the majority trainID**. Implemented in `scripts/refine_simcf.py:82–128`.

```
SIMCF-A(S, I, φ, K)
Input:
    S ∈ uint8^{H'×W'} — cluster IDs, modified in-place
    I ∈ uint16^{H'×W'} — instance IDs
    φ : {0,…,255} → {0,…,C-1,255}  — cluster-to-trainID LUT
    K                  — over-cluster count (80)
Output:
    n_changed — number of pixels rewritten
Side effect:
    S is updated in-place.

 1  n_changed ← 0
 2  for each i ∈ unique(I) \ {0} do
 3      (Y, X) ← where(I = i)
 4      if |Y| = 0 then continue
 5      clusters ← S[Y, X]                              // (|Y|,)
 6      trainIDs ← φ[clusters]                          // (|Y|,)
 7      h ← bincount(trainIDs[trainIDs < C], minlength=C)
 8      if Σ h = 0 then continue
 9      c* ← argmax(h)                                  // majority trainID
10      bad ← (trainIDs ≠ c*) ∧ (trainIDs < C)
11      if ¬any(bad) then continue
12      good ← ¬bad
13      best_k ← argmax( bincount(clusters[good], minlength=K) )
14      S[Y[bad], X[bad]] ← best_k                      // rewrite
15      n_changed ← n_changed + Σ bad
16  end for
17  return n_changed
```

**Invariant.** SIMCF-A never invents new trainIDs — it only replaces a cluster by another cluster already present inside the instance that maps to the majority trainID, so it cannot leak knowledge of $\phi$ into the supervision.

---

## Algorithm 6 — `SIMCF-B` (Semantics validate Instances)

Merges adjacent same-class instances whose mean DINOv3 features have cosine similarity $> \tau_\mathrm{sim}$. Implemented in `scripts/refine_simcf.py:135–239`.

```
SIMCF-B(S, I, F, φ ; τ_sim, r_dil)
Input:
    S, I              — as in SIMCF-A
    F ∈ R^{H_p·W_p × D} — L2-normalised DINOv3 features, (H_p, W_p) = (32, 64)
    φ                  — cluster-to-trainID LUT
    τ_sim = 0.85       — cosine similarity threshold
    r_dil = 3          — dilation iterations for adjacency
Output:
    I' — merged instance ID map
    n_merges — number of merged instance-pairs

 1  I_small ← Resize(I, (H_p, W_p), mode=NEAREST)
 2  F̂  ← reshape(F, (H_p, W_p, D))
 3  IDs ← unique(I) \ {0}
 4  if |IDs| < 2 then return (I, 0)
 5  φ_S ← φ[S]                                     // mapped trainIDs
 6  class[i] ← argmax( bincount( φ_S[I = i][< C], minlength=C ) )
                                                   ∀ i ∈ IDs
 7  feat[i]  ← Σ F̂[I_small = i] / |I_small = i|
 8  feat[i]  ← feat[i] / (‖feat[i]‖ + 1e-8)
 9  // -- Adjacency at full resolution
10  Adj ← ∅
11  for i ∈ IDs do
12      M_i ← (I = i);   M̃ ← BinaryDilation(M_i, iter=r_dil)
13      ∂M ← M̃ ∧ ¬M_i                                // border
14      for j ∈ unique(I[∂M]) do
15          if j ∉ {0, i} ∧ j ∈ IDs then
16              Adj ← Adj ∪ { (min(i,j), max(i,j)) }
17          end if
18      end for
19  end for
20  // -- Filter by class identity + cosine similarity
21  Merge ← ∅
22  for (i, j) ∈ Adj do
23      if class[i] = class[j] then
24          s ← ⟨ feat[i], feat[j] ⟩                   // ∈ [−1, 1]
25          if s > τ_sim then Merge ← Merge ∪ {(i, j)}
26      end if
27  end for
28  if Merge = ∅ then return (I, 0)
29  // -- Union-find compaction
30  parent[i] ← i for each i ∈ IDs
31  for (i, j) ∈ Merge do  Union(parent, i, j)
32  I' ← 0_{H'×W'};  next ← 1;  ι : root ↦ newID
33  for each i ∈ sort(IDs) do
34      r ← Find(parent, i)
35      if r ∉ keys(ι) then ι[r] ← next;  next ← next + 1
36      I'[I = i] ← ι[r]
37  end for
38  return (I', |IDs| − |ι|)
```

---

## Algorithm 7 — `SIMCF-C` (Depth validates Semantics)

A two-pass procedure: first, per-class depth statistics are accumulated across the **whole training split**; second, each per-image semantic map is sparsified by setting pixels whose depth exceeds $\eta \sigma_c$ from the class mean to the ignore index 255. Implemented in `scripts/refine_simcf.py:246–321`.

### Pass 1 — Global statistics

```
ComputeDepthStats(stems, depth_dir, φ)
 1  Σ, Σ², n  ← 0_C ∈ R^C, 0_C ∈ R^C, 0_C ∈ ℤ^C
 2  for each stem ∈ stems do
 3      S    ← LoadSemantic(stem)
 4      D    ← LoadDepth(stem);    if shape(D) ≠ shape(S) then BilinearResize(D)
 5      M    ← φ[S]                                          // (H', W')
 6      for c = 0 to C-1 do
 7          v ← D[M = c]
 8          Σ[c]  ← Σ[c]  + Σ v
 9          Σ²[c] ← Σ²[c] + Σ v²
10          n[c]  ← n[c]  + |v|
11      end for
12  end for
13  μ_c ← Σ[c] / max(n[c], 1)
14  σ_c ← √( max( Σ²[c]/max(n[c],1) − μ_c², 0 ) )
15  return (μ, σ) ∈ R^C × R^C
```

### Pass 2 — Per-image outlier rejection

```
SIMCF-C(S, D, φ, μ, σ ; η)
Input:
    η = 3.0          // σ-multiplier threshold
Output:
    n_masked
Side effect:
    S[outliers] ← 255

 1  M ← φ[S]                                                // (H', W')
 2  n_masked ← 0
 3  for c = 0 to C-1 do
 4      if σ[c] < 1e-6 then continue                       // skip degenerate
 5      mask_c ← (M = c)
 6      if ¬any(mask_c) then continue
 7      Δ ← |D[mask_c] − μ[c]|                             // (|mask_c|,)
 8      out ← Δ > η · σ[c]
 9      (Y, X) ← where(mask_c)
10      S[ Y[out], X[out] ] ← 255
11      n_masked ← n_masked + Σ out
12  end for
13  return n_masked
```

**Safety rail in driver.** If the overall ignore-rate exceeds 15 % across the split, the driver emits a warning (`refine_simcf.py:493–496`); in production we observe $\approx 5\%$.

---

## Algorithm 8 — `Stage1Generate` (full Stage-1 pipeline)

End-to-end driver that combines Algorithms 1–7 to emit one `(S, I, .pt)` triple per training image. Implemented as the `main()` of `scripts/refine_simcf.py:356–497` plus the upstream generators.

```
Stage1Generate(D_train, net, segment, adapter, depthpro, F_DINOv3,
               C, φ, K, τ_d, A_min, n_dil, σ_blur,
               τ_sim, η)
Input:
    D_train     — list of Cityscapes train images
    net         — DINOv2 ViT-B/14 backbone (frozen)
    segment     — CAUSE-TR head (frozen)
    adapter     — trained DCFA θ* (Algorithm 3)
    depthpro    — Apple DepthPro model (frozen)
    F_DINOv3    — pre-extracted DINOv3 ViT-B/16 features (per image)
    (C, φ)      — k-means centroids + LUT (Algorithm 1)
Output:
    For each x ∈ D_train:  (Ŝ_x, Î_x) written as
                            {stem}_semantic.png  (uint8)
                            {stem}_instance.png  (uint16)
                            {stem}.pt           (distribution dict)

 1  (μ, σ) ← ComputeDepthStats(stems(D_train), depth_dir, φ)
 2  for each x ∈ D_train do
 3      // --- Step (i): semantic pseudo-label ------------------------------
 4      Z⁰ ← segment( net(x) )                              // (1, 90, H_p, W_p)
 5      D  ← depthpro(x)                                    // (H', W') ∈ [0,1]
 6      Z  ← DCFA-Forward(Z⁰, downsample(D, (H_p, W_p)) ; θ*)
 7      Z  ← Z / ‖Z‖₂                                       // ℓ₂-normalise per-pixel
 8      S⁰ ← argmin_k 1 − Z · Cᵀ                            // (H_p, W_p)
 9      S⁰ ← Resize(S⁰, (H', W'), mode=NEAREST)
10      // --- Step (ii): instance pseudo-label -----------------------------
11      I⁰ ← Rasterise( DepthProInstanceGeneration(S⁰, D ; τ_d, A_min, n_dil, σ_blur) )
12      // --- Step (iii): SIMCF-ABC ----------------------------------------
13      _ ← SIMCF-A(S⁰, I⁰, φ, K)                            // S⁰ modified in-place
14      (I¹, _) ← SIMCF-B(S⁰, I⁰, F_DINOv3[x], φ ; τ_sim=0.85, r_dil=3)
15      _ ← SIMCF-C(S⁰, D, φ, μ, σ ; η=3.0)
16      Ŝ_x ← S⁰;   Î_x ← I¹
17      // --- Persist
18      WriteUInt8PNG(Ŝ_x,  output_dir / f"{stem}_semantic.png")
19      WriteUInt16PNG(Î_x, output_dir / f"{stem}_instance.png")
20      WritePT( ComputeDistributions(Ŝ_x, Î_x, K),
21               output_dir / f"{stem}.pt" )
22  end for
```

**Reported impact** (Cityscapes val, paper §4 ablation): the DCFA + SIMCF-ABC stages move the raw Stage-1 pseudo-labels from $24.54 \to 25.85$ PQ, $\mathrm{PQ_{th}}: 12.31 \to 14.70$, $\mathrm{mIoU}: 52.69 \to 55.29$.

---

# Part II — Stage-2: Cascade Mask R-CNN Detector $F$

The Stage-2 detector $F$ is a Cascade Mask R-CNN (Cai & Vasconcelos 2018) on a frozen DINOv3 ViT-B/16 backbone, trained on the pseudo-labels $(\hat{\mathbf{S}}, \hat{\mathbf{I}})$ from Stage-1. The training script is `refs/cups/train.py`; the Lightning module is `refs/cups/cups/pl_model_pseudo.py`; the loss-masking machinery is `refs/cups/cups/model/modeling/roi_heads/custom_cascade_rcnn.py:240–278`.

## Stage-2 Hyper-parameters (`refs/cups/cups/config.py`, `configs/cups_cityscapes.yaml`)

| Symbol | Value | Source |
|---|---|---|
| Total optimizer steps $T_2$ | $8\,000$ | `TRAINING.STEPS` |
| Batch size $B$ | $4$ | `TRAINING.BATCH_SIZE` |
| Optimiser | AdamW, $\beta=(0.9, 0.999)$ | `TRAINING.ADAMW.*` |
| Learning rate | $10^{-4}$ | `TRAINING.ADAMW.LEARNING_RATE` |
| Weight decay | $10^{-5}$ | `TRAINING.ADAMW.WEIGHT_DECAY` |
| Precision | bf16 | `TRAINING.PRECISION` |
| Gradient clipping | norm, $\max=1.0$ | `TRAINING.GRAD_CLIP_*` |
| DropLoss IoU threshold $\tau_\mathrm{drop}$ | $0.4$ | `TRAINING.DROP_LOSS_IOU_THRESHOLD` |
| Copy-paste warmup $T_\mathrm{cp}$ | $1\,000$ | `AUGMENTATION.NUM_STEPS_STARTUP` |
| Copy-paste max objects $M_\mathrm{cp}$ | $8$ | `AUGMENTATION.MAX_NUM_PASTED_OBJECTS` |
| Crop $H_\mathrm{crop}{\times}W_\mathrm{crop}$ | $640 \times 1280$ | `DATA.CROP_RESOLUTION` |
| Val every $\Delta_\mathrm{val}$ steps | $500$ | `TRAINING.VAL_CHECK_INTERVAL` |
| Cascade stages | 4 | inherited from Detectron2 |

---

## Algorithm 9 — `Stage2Train`

```
Stage2Train(D_pseudo, F, T₂, B, opt, sched,
            τ_drop, M_cp, T_cp, augment)
Input:
    D_pseudo  — paired (x, Ŝ, Î) over Cityscapes train (with `.pt` distributions)
    F         — Cascade Mask R-CNN + DINOv3 ViT-B/16 (backbone frozen)
    opt       — AdamW(lr=1e-4, wd=1e-5, β=(0.9, 0.999))
    sched     — constant schedule
    augment   — Compose( CopyPaste, Photometric, ResolutionJitter, RandomCrop )
Output:
    F* — best checkpoint by validation PQ over T₂ steps

 1  t ← 0
 2  for each minibatch (X, Ŝ, Î) ∈ D_pseudo do
 3      if t ≥ T_cp then
 4          (X, Ŝ, Î) ← CopyPaste(X, Ŝ, Î; M_cp)            // Algorithm 11
 5      end if
 6      (X, Ŝ, Î) ← augment.photometric ∘ augment.crop (X, Ŝ, Î)
 7
 8      // ----- forward through 4-stage cascade -----
 9      losses ← ∅
10      proposals ← F.rpn(X)                                 // → losses["loss_rpn_cls"], …
11      for stage s = 0, 1, 2, 3 do
12          predictions, proposals_s ← F.cascade[s](X, proposals)
13          (preds_box_cls, preds_delta) ← predictions
14          losses ← losses ∪ DropLoss(preds, proposals_s, Î;
15                                     τ_drop, stage=s)        // Algorithm 10
16      end for
17      losses ← losses ∪ MaskHeadLoss(F, X, Î)               // BCE on full pseudo-masks
18      losses ← losses ∪ SemanticHeadLoss(F, X, Ŝ)           // CE on Ŝ, ignore=255
19
20      L ← Σ losses
21      opt.zero_grad();  L.backward()
22      ClipGradNorm(F.parameters(), max_norm=1.0)
23      opt.step();  sched.step()
24
25      if t mod Δ_val = 0 then
26          PQ_t ← Evaluate(F, D_val_panoptic)
27          if PQ_t > best then save_checkpoint(F, t, PQ_t)
28      end if
29      t ← t + 1
30      if t ≥ T_2 then break
31  end for
32  return F* = argmax_t PQ_t
```

**Cascade box loss aggregation.** The per-stage losses produced in line 14 are summed across the four cascade stages with equal weight (Detectron2 default), and `loss_cls_stage{s} + loss_box_reg_stage{s}` are reported separately to W&B for diagnosability.

---

## Algorithm 10 — `DropLoss`

Per Hahn 2025 (CUPS) and Wang 2023b (CutLER): low-IoU proposals against the noisy pseudo-boxes are likely false negatives in $\hat{\mathbf{I}}$, so their classification loss is set to zero rather than treated as background. Implemented in `refs/cups/cups/model/modeling/roi_heads/custom_cascade_rcnn.py:247–272`.

```
DropLoss(preds, proposals, Î ; τ_drop, stage)
Input:
    preds.box_cls    ∈ R^{R×(C+1)}     // R = total proposals in batch
    preds.box_delta  ∈ R^{R×4}
    proposals[i].gt_boxes  ∈ R^{G_i×4}
    proposals[i].proposal_boxes ∈ R^{R_i×4}
    τ_drop = 0.4
Output:
    loss_cls_stage{stage}, loss_box_reg_stage{stage}

 1  if any(proposals.gt_boxes.tensor.numel = 0) then
 2      return Detectron2.fast_rcnn_loss( preds, proposals )   // fallback
 3  end if
 4  // ----- per-proposal predicted box and IoU vs nearest GT -----
 5  prop_boxes ← cat([ p.proposal_boxes for p ∈ proposals ])   // (R, 4)
 6  pred_boxes ← Box2Box.apply_deltas(preds.box_delta, prop_boxes)
 7  iou_max ← ∅
 8  i₀ ← 0
 9  for i = 0 to |proposals|−1 do
10      n_i ← |proposals[i].proposal_boxes|
11      iou_max ← iou_max ⊕ pairwise_iou_max( pred_boxes[i₀ : i₀+n_i],
                                              proposals[i].gt_boxes[:G_max] )
12      i₀ ← i₀ + n_i
13  end for                                                   // iou_max ∈ R^{R}
14  // ----- mask weights -----
15  w ← (iou_max ≤ τ_drop).float                              // ∈ {0,1}^R
16  w ← 1 − (w ≥ 1).float                                    // ⇒ keep proposals
                                                              //    with high-IoU GT only
                                                              //    or no-overlap pure neg
17  return Detectron2.fast_rcnn_loss( preds, proposals,
                                      weights = w.detach() )
```

**Reading of the mask.** Pixel-wise: $w_r = 1$ iff proposal $r$ has $\mathrm{IoU}_\max > \tau_\mathrm{drop}$ (well-localised positive), so its classification loss is preserved; otherwise $w_r = 0$ (likely a false negative against a pseudo-label hole), so its loss is dropped.

---

## Algorithm 11 — `CopyPaste`

Per-image instance compositing. For each target image we sample $\le M_\mathrm{cp}$ pasted instances from other images in the same minibatch. Implemented in `refs/cups/cups/augmentation.py:_paste_one (lines 110–166)`.

```
CopyPaste(X, Ŝ, Î ; M_cp)
Input:
    X ∈ R^{B×3×H_crop×W_crop}, Ŝ ∈ ℤ^{B×H_crop×W_crop},
    Î (list of Instances per image with .gt_masks, .gt_boxes, .gt_classes)
    M_cp = 8
Output:
    Augmented (X̃, S̃, Ĩ)

 1  for i = 0 to B-1 do
 2      cand ← [ donors from {0..B-1}\{i} ]
 3      n ← Uniform(0, min(M_cp, Σ_j |Î[j]|))
 4      for k = 1 to n do
 5          (j, m_k, c_k) ← SampleInstance(cand)              // (mask m_k ∈ {0,1}^{H_crop×W_crop}, class c_k)
 6          s ← Uniform(0.25, 1.5)                            // scale
 7          (m̃_k, x̃_k, b̃_k) ← AffineScale(m_k, X[j], scale=s)
 8          (m̄_k, x̄_k, b̄_k) ← PadOrCrop(m̃_k, x̃_k, b̃_k,
                                          (H_crop, W_crop), random_offset)
 9          // -- composite RGB --
10          X[i] ← m̄_k · x̄_k + (1 − m̄_k) · X[i]
11          // -- composite semantics --
12          Ŝ[i][m̄_k] ← 0                                      // mark thing region as void
13          // -- append instance --
14          Î[i] ← Î[i] ⊕ Instance( mask=m̄_k, box=b̄_k, class=c_k )
15      end for
16  end for
17  return (X, Ŝ, Î)
```

CopyPaste is only enabled after $t \ge T_\mathrm{cp}=1{,}000$ optimizer steps so the RPN has had a chance to learn from the unaugmented pseudo-labels first.

---

## Algorithm 12 — `PanopticMerge` (Kirillov-style)

Maps the detector outputs $(\hat{\mathbf{S}}_F, \{(\mathbf{M}_n, c_n, p_n)\})$ to a single panoptic ID map. Implemented in `refs/cups/cups/model/modeling/meta_arch/panoptic_fpn.py:411–494`.

```
PanopticMerge(Î_pred, Ŝ_pred ; τ_overlap, A_stuff, p_min)
Input:
    Î_pred.scores       ∈ R^N
    Î_pred.pred_masks   ∈ {0,1}^{N×H×W}
    Î_pred.pred_classes ∈ ℤ^N
    Ŝ_pred              ∈ ℤ^{H×W}              // semantic argmax (incl. 0 = void/thing)
    τ_overlap = 0.5  (instance-overlap reject)
    A_stuff   = 4096 (stuff min-area)
    p_min     = 0.5  (instance score floor)
Output:
    Π ∈ ℤ^{H×W}            // panoptic ID map, 0 = void
    info  : list of {id, isthing, category_id, [score|area]}

 1  Π ← 0_{H×W};   id ← 0;   info ← ∅
 2  order ← argsort(−Î_pred.scores)
 3  for n ∈ order do
 4      if Î_pred.scores[n] < p_min then break
 5      m ← Î_pred.pred_masks[n]
 6      if Σ m = 0 then continue
 7      I_overlap ← (m ∧ Π > 0)
 8      if Σ I_overlap / Σ m > τ_overlap then continue       // reject heavy overlap
 9      if Σ I_overlap > 0 then m ← m ∧ (Π = 0)              // keep non-overlap part
10      id ← id + 1
11      Π[m] ← id
12      info ← info ⊕ { id, isthing=True,
                        score = Î_pred.scores[n],
                        category_id = Î_pred.pred_classes[n] }
13  end for
14  for each c ∈ unique(Ŝ_pred) do
15      if c = 0 then continue                              // 0 reserved for things
16      m ← (Ŝ_pred = c) ∧ (Π = 0)
17      if Σ m < A_stuff then continue
18      id ← id + 1
19      Π[m] ← id
20      info ← info ⊕ { id, isthing=False, category_id = c, area = Σ m }
21  end for
22  return (Π, info)
```

**Things-first invariant.** Instance segments occupy panoptic IDs first (in descending confidence order); stuff segments fill the remaining holes, subject to a minimum area.

---

# Part III — Stage-3: EMA Self-Training of $F$

Stage-3 boots a teacher $F^\mathrm{T}$ from the best Stage-2 checkpoint and runs **three rounds** of EMA self-training, each consisting of $T_3 = 500$ optimiser steps. Inside each round the teacher's confidence-thresholded panoptic prediction is treated as the new pseudo-label $(\tilde{\mathbf{S}}, \tilde{\mathbf{I}})$ for the student, refreshed *per batch*. Implemented in `refs/cups/train_self.py` and `refs/cups/cups/pl_model_self.py`.

## Stage-3 Hyper-parameters (`refs/cups/cups/config.py`)

| Symbol | Value | Source |
|---|---|---|
| Number of rounds $R$ | $3$ | `SELF_TRAINING.ROUNDS` |
| Steps per round $T_3$ | $500$ | `SELF_TRAINING.ROUND_STEPS` |
| Batch size $B$ | $4$ | `TRAINING.BATCH_SIZE` |
| Learning rate | $10^{-5}$ | `TRAINING.ADAMW.LEARNING_RATE` (overridden in `cups_self_*.yaml`) |
| EMA decay $\mu$ | $0.999$ | hard-coded `pl_model_self.py:727` |
| Semantic confidence $\rho$ | $0.5$ | `SELF_TRAINING.SEMANTIC_SEGMENTATION_THRESHOLD` |
| Confidence ramp $\Delta\rho$ | $+0.05$ per round | `SELF_TRAINING.SEMANTIC_SEGMENTATION_STEP` |
| DropLoss | **disabled** | `cups_self_cityscapes.yaml` |
| CopyPaste | **enabled** | `cups_self_cityscapes.yaml` |
| Optimiser scope | head-only (or LoRA when enabled) | `pl_model_self.py:772–793` |

---

## Algorithm 13 — `Stage3SelfTrain`

```
Stage3SelfTrain(D_self, F_star, R, T_3, B, opt, μ, ρ_0, Δρ, augment)
Input:
    D_self    — augmented Cityscapes train images (no labels needed)
    F_star    — best Stage-2 student (Algorithm 9)
    R = 3, T_3 = 500,   μ = 0.999
    ρ_0 = 0.5, Δρ = 0.05
    augment   — same Compose(CopyPaste, Photometric, ResolutionJitter, RandomCrop)
Output:
    F⁺ — best Stage-3 checkpoint by val PQ

 1  F     ← clone(F_star)                                // student
 2  F^T   ← deepcopy(F_star)                              // teacher (frozen until EMA)
 3  freeze(F^T.parameters)
 4  opt   ← AdamW( head_params(F), lr=1e-5, wd=1e-5 )    // or 6-group LoRA opt
 5  t     ← 0
 6  for r = 1 to R do
 7      ρ ← min( ρ_0 + (r-1) · Δρ, 0.95 )                // confidence schedule
 8      for t' = 1 to T_3 do
 9          X ← next(D_self)                              // shape: (B, 3, H_crop, W_crop)
10
11          // ---- (a) teacher inference (TTA) ---------
12          F^T.eval()
13          with no_grad:
14              X_tta ← TTA(X)                            // hflip / multi-scale
15              pred^T ← F^T(X_tta)                        // panoptic + sem_seg logits
16          (S̃, Ĩ, B̃, C̃) ← MakePseudoFromTeacher(pred^T, X ; ρ)   // Algorithm 14
17
18          // ---- (b) student augmentation ------------
19          if CopyPaste enabled then
20              (X, S̃, Ĩ) ← CopyPaste(X, S̃, Ĩ ; M_cp = 8)
21          end if
22          (X, S̃, Ĩ) ← augment.photometric ∘ augment.jitter ∘ augment.crop (X, S̃, Ĩ)
23
24          // ---- (c) student step (no DropLoss) ------
25          F.train()
26          losses ← CascadeMaskRCNN_Losses(F, X, S̃, Ĩ, B̃, C̃)
27          L ← Σ losses
28          opt.zero_grad();  L.backward()
29          ClipGradNorm(F.parameters(), 1.0)
30          opt.step()
31
32          // ---- (d) EMA update ----------------------
33          EMAUpdate(F^T, F ; μ)                          // Algorithm 15
34
35          // ---- (e) bookkeeping ---------------------
36          if t mod Δ_val = 0 then
37              PQ_t ← Evaluate(F, D_val_panoptic)
38              if PQ_t > best then save_checkpoint(F, t, PQ_t)
39          end if
40          t ← t + 1
41      end for
42  end for
43  return F⁺ = argmax_t PQ_t
```

**Round semantics.** The increment $\rho \leftarrow \rho + \Delta\rho$ between rounds tightens the teacher's confidence requirement, so later rounds train on a strictly smaller, more confident pseudo-mask — analogous to the curriculum's "third round = highest precision" principle in CUPS.

---

## Algorithm 14 — `MakePseudoFromTeacher`

Converts teacher panoptic outputs and per-class semantic logits into the supervision tuple consumed by Detectron2's Cascade Mask R-CNN trainer. Implemented in `refs/cups/cups/pl_model_self.py:300–410`.

```
MakePseudoFromTeacher(pred^T, X ; ρ)
Input:
    pred^T[b] : { panoptic_seg = (Π_b, info_b), sem_seg = L_b ∈ R^{C×H×W} }
    ρ : scalar — semantic confidence ratio
Output (per image b):
    S̃_b : ℤ^{H×W}    — semantic pseudo-label (255 = void)
    Ĩ_b : list of {0,1}^{H×W}  — per-thing-instance masks
    B̃_b : ℝ^{N_b×4}  — bounding boxes (x1, y1, x2, y2)
    C̃_b : ℤ^{N_b}    — instance class IDs

 1  for b = 0 to B-1 do
 2      // ---- (i) confidence-thresholded semantic pseudo ----
 3      ℓ_max ← amax(L_b, axis=(1,2), keepdim=True)        // (C, 1, 1)
 4      θ_c   ← ρ · ℓ_max                                   // per-class threshold
 5      L'    ← where(L_b > θ_c, L_b, 0)                   // mask weak logits
 6      S̃_b   ← argmax(L', dim=0)                          // (H, W)
 7      S̃_b[ Σ_c L'[c] = 0 ] ← 255                          // void
 8      // ---- (ii) panoptic → instance lists ---------------
 9      w_sem  ← 255 · 1_{|info|+1}      ; w_inst ← 0_{|info|+1}
10      object_semantics ← ∅
11      for each obj ∈ info_b do
12          if obj.isthing then
13              w_sem[obj.id]  ← 0
14              w_inst[obj.id] ← max(w_inst) + 1
15              object_semantics ← object_semantics ⊕ obj.category_id
16          else
17              w_sem[obj.id]  ← obj.category_id
18          end if
19      end for
20      I_b ← Embedding( Π_b, weight=w_inst.view(-1, 1) ).squeeze()
21      Ĩ_b ← InstancesToMasks(I_b)                         // (N_b, H, W)
22      B̃_b ← BoundingBoxes(I_b)                            // (N_b, 4)
23      // ---- (iii) drop degenerate boxes -----------------
24      valid ← (B̃_b[:,2] > B̃_b[:,0]) ∧ (B̃_b[:,3] > B̃_b[:,1])
25      (Ĩ_b, B̃_b) ← keep_rows(valid, Ĩ_b, B̃_b)
26      C̃_b ← object_semantics[valid]
27  end for
28  return { (S̃_b, Ĩ_b, B̃_b, C̃_b) }_{b=0..B-1}
```

**Why drop degenerate boxes.** A box with $x_2 \le x_1$ or $y_2 \le y_1$ would cause $\log(0) = -\infty$ in Detectron2's `Box2BoxTransform`, exploding `loss_rpn_loc` (Lightning-bug fix at `pl_model_self.py:378–384`).

---

## Algorithm 15 — `EMAUpdate`

Exponential moving average from student to teacher, executed at the **end of every training batch** by Lightning's `on_train_batch_end` hook (`pl_model_self.py:711–727`).

```
EMAUpdate(F^T, F ; μ)
Input:
    F^T, F  — same-architecture teacher & student
    μ = 0.999
Side effect:
    For every parameter pair (θ^T_k, θ_k):
        θ^T_k ← μ · θ^T_k + (1 − μ) · θ_k                    // ✶
End-effect:
    The teacher tracks the slow-moving mean of the student.

 1  if DISABLE_EMA then return
 2  for (θ_k, θ^T_k) ∈ zip(F.parameters, F^T.model.parameters) do
 3      θ^T_k ← θ^T_k · μ                                    // in-place mul_
 4      θ^T_k ← θ^T_k + (1 − μ) · θ_k                        // in-place add_
 5  end for
```

This is mathematically equivalent to $\theta^T_k \leftarrow 0.999\,\theta^T_k + 0.001\,\theta_k$, giving the teacher an effective horizon of $1/(1-\mu) = 1000$ steps — i.e., approximately one Stage-3 round.

---

# Part IV — Reproducibility Cross-Reference

| Algorithm | File | Lines |
|---|---|---|
| 1. SemanticOverclustering | `mbps_pytorch/generate_depth_overclustered_semantics.py` | 493–619 |
| 2. DCFA-Forward | `mbps_pytorch/models/semantic/depth_adapter.py` | 41–98 |
| 3. DCFA-Train | `mbps_pytorch/train_depth_adapter.py` | 186–308 |
| 3a. DepthG | `mbps_pytorch/models/semantic/stego_loss.py` | 116–169 |
| 4. DepthProInstanceGeneration | `mbps_pytorch/generate_depth_guided_instances.py` | 75–154 |
| 5. SIMCF-A | `scripts/refine_simcf.py` | 82–128 |
| 6. SIMCF-B | `scripts/refine_simcf.py` | 135–239 |
| 7. SIMCF-C (Pass 1) | `scripts/refine_simcf.py` | 246–295 |
| 7. SIMCF-C (Pass 2) | `scripts/refine_simcf.py` | 298–321 |
| 8. Stage1Generate | `scripts/refine_simcf.py` (main) | 356–497 |
| 9. Stage2Train | `refs/cups/train.py`, `pl_model_pseudo.py` | — / 560–700 |
| 10. DropLoss | `refs/cups/cups/model/modeling/roi_heads/custom_cascade_rcnn.py` | 240–278 |
| 11. CopyPaste | `refs/cups/cups/augmentation.py` (`_paste_one`) | 110–166 |
| 12. PanopticMerge | `refs/cups/cups/model/modeling/meta_arch/panoptic_fpn.py` | 411–494 |
| 13. Stage3SelfTrain | `refs/cups/train_self.py`, `pl_model_self.py` | — / 184–410 |
| 14. MakePseudoFromTeacher | `refs/cups/cups/pl_model_self.py` | 300–410 |
| 15. EMAUpdate | `refs/cups/cups/pl_model_self.py` | 711–727 |

## Dimensional Audit (sanity-check the shapes)

```
Input image                 (H, W, 3)        = (1024, 2048, 3)   uint8
Working-resolution image    (H', W', 3)      = ( 512, 1024, 3)   float32 ∈ [0,1]
DINOv2 patch grid           (H_p, W_p)       = (  32,   64)      stride 14 on (448, 896)
CAUSE-TR codes              (H_p, W_p, 90)   = (  32,   64, 90)  float32
DCFA hidden                 (H_p, W_p, 384)                       float32
DCFA depth-encoded          (H_p, W_p, 16)                        float32
k-means centroids           (K, 90)          = (  80,   90)       float32, ℓ₂-norm
DINOv3 features for SIMCF-B (H_p, W_p, 768)                       float32, ℓ₂-norm
DepthPro depth              (H', W')         = ( 512, 1024)       float32 ∈ [0,1]
Semantic pseudo S           (H', W')         = ( 512, 1024)       uint8,   {0..79,255}
Instance pseudo I           (H', W')         = ( 512, 1024)       uint16,  {0..max_inst}
Stage-2 crop                (H_crop, W_crop) = ( 640, 1280)       float32
Stage-2 output panoptic     (H_crop, W_crop) = ( 640, 1280)       int32
Stage-3 EMA decay μ         scalar           = 0.999
```

All dimensions are consistent with the codebase as of commit `C606` on the `dino-cause-dora-adapter` branch.
