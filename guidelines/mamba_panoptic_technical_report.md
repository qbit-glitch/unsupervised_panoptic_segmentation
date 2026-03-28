# Unsupervised Mamba-Bridge Panoptic Segmentation: Technical Specifications

## Complete Mathematical Framework, CLRS Algorithms, and Implementation Guide

---

# Part I: Resolving the Seven Critical Technical Challenges

## Challenge 1: Feature Dimension Mismatch Resolution

### Problem Statement
DepthG produces semantic codes $\mathbf{s} \in \mathbb{R}^{D_s \times H \times W}$ where $D_s = 90$, while CutS3D operates on DINO features $\mathbf{f} \in \mathbb{R}^{D_f \times H \times W}$ where $D_f = 384$. Direct concatenation creates imbalanced representations.

### Solution: Adaptive Projection Bridge (APB)

**Definition 1.1 (Semantic Projection):**
$$\mathbf{s}' = \text{LayerNorm}(\mathbf{W}_s \mathbf{s} + \mathbf{b}_s)$$
where $\mathbf{W}_s \in \mathbb{R}^{D_b \times D_s}$, $\mathbf{b}_s \in \mathbb{R}^{D_b}$, and $D_b$ is the bridge dimension.

**Definition 1.2 (Instance Projection):**
$$\mathbf{f}' = \text{LayerNorm}(\mathbf{W}_f \mathbf{f} + \mathbf{b}_f)$$
where $\mathbf{W}_f \in \mathbb{R}^{D_b \times D_f}$, $\mathbf{b}_f \in \mathbb{R}^{D_b}$.

**Definition 1.3 (Inverse Projections for Reconstruction):**
$$\hat{\mathbf{s}} = \mathbf{W}_s^{\dagger} \mathbf{z}_s, \quad \hat{\mathbf{f}} = \mathbf{W}_f^{\dagger} \mathbf{z}_f$$
where $\mathbf{z}_s, \mathbf{z}_f$ are the Mamba-processed features and $\mathbf{W}^{\dagger}$ denotes learned pseudo-inverse projections.

**Theorem 1.1 (Optimal Bridge Dimension):**
The optimal bridge dimension $D_b^*$ minimizes reconstruction error while maximizing cross-modal correlation:
$$D_b^* = \arg\min_{D_b} \left[ \|\mathbf{s} - \hat{\mathbf{s}}\|_2^2 + \|\mathbf{f} - \hat{\mathbf{f}}\|_2^2 - \lambda \cdot \text{CKA}(\mathbf{s}', \mathbf{f}') \right]$$

where CKA is Centered Kernel Alignment. Empirically, $D_b \in \{128, 192, 256\}$ works best.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHM 1: ADAPTIVE-PROJECTION-BRIDGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Semantic features s ∈ ℝ^(D_s × N), Instance features f ∈ ℝ^(D_f × N)
       Bridge dimension D_b, Temperature τ
Output: Aligned features s' ∈ ℝ^(D_b × N), f' ∈ ℝ^(D_b × N)

 1  ▷ Initialize learnable projections
 2  W_s ← Xavier_Init(D_b, D_s)
 3  W_f ← Xavier_Init(D_b, D_f)
 4  
 5  ▷ Project to bridge space
 6  s_proj ← W_s · s                           ▷ ℝ^(D_b × N)
 7  f_proj ← W_f · f                           ▷ ℝ^(D_b × N)
 8  
 9  ▷ Apply LayerNorm for stable gradients
10  s' ← LayerNorm(s_proj, dim=0)
11  f' ← LayerNorm(f_proj, dim=0)
12  
13  ▷ Compute alignment loss for training
14  K_s ← s' · s'^T                            ▷ Gram matrix
15  K_f ← f' · f'^T
16  L_align ← 1 - CKA(K_s, K_f)
17  
18  return s', f', L_align
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Challenge 2: Training Dynamics Conflict Resolution

### Problem Statement
DepthG's contrastive loss pulls same-class features together while pushing different-class features apart. CutS3D's instance losses focus on boundary precision. These objectives can conflict: contrastive losses may blur boundaries, while boundary losses may fragment semantic regions.

### Solution: Gradient-Balanced Curriculum Learning (GBCL)

**Definition 2.1 (Gradient Magnitude Ratio):**
$$\rho_t = \frac{\|\nabla_\theta \mathcal{L}_{\text{semantic}}\|_2}{\|\nabla_\theta \mathcal{L}_{\text{instance}}\|_2 + \epsilon}$$

**Definition 2.2 (Adaptive Loss Weighting):**
$$\mathcal{L}_{\text{total}} = \alpha_t \cdot \mathcal{L}_{\text{semantic}} + \beta_t \cdot \mathcal{L}_{\text{instance}} + \gamma_t \cdot \mathcal{L}_{\text{bridge}}$$

where the weights evolve as:
$$\alpha_t = \alpha_0 \cdot \sigma\left(\frac{t - t_{\text{warm}}}{T}\right), \quad \beta_t = \beta_0 \cdot \left(1 - e^{-t/\tau_\beta}\right)$$

**Theorem 2.1 (Curriculum Convergence):**
Under GBCL with proper scheduling, the joint optimization converges to a Pareto-optimal solution satisfying:
$$\nabla_\theta \mathcal{L}_{\text{semantic}} \cdot \nabla_\theta \mathcal{L}_{\text{instance}} \geq 0$$
(gradient directions become non-conflicting after warmup).

### Three-Phase Curriculum Strategy

**Phase A (Epochs 1-20): Semantic Foundation**
- Train DepthG branch only with full $\mathcal{L}_{\text{semantic}}$
- Freeze instance branch, $\beta_t = 0$
- Goal: Establish stable semantic cluster structure

**Phase B (Epochs 21-40): Instance Integration**  
- Unfreeze instance branch, gradually increase $\beta_t$
- Apply gradient projection to remove conflicting components:
$$\nabla_\theta^{\text{proj}} = \nabla_\theta \mathcal{L}_{\text{instance}} - \frac{\langle \nabla_\theta \mathcal{L}_{\text{instance}}, \nabla_\theta \mathcal{L}_{\text{semantic}} \rangle^-}{\|\nabla_\theta \mathcal{L}_{\text{semantic}}\|_2^2} \nabla_\theta \mathcal{L}_{\text{semantic}}$$
where $\langle \cdot, \cdot \rangle^- = \min(0, \langle \cdot, \cdot \rangle)$

**Phase C (Epochs 41-60): Joint Refinement**
- Full joint training with bridge losses
- EMA momentum network for pseudo-label refinement

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHM 2: GRADIENT-BALANCED-CURRICULUM-LEARNING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Model θ, Epochs T, Warmup t_warm, Phase transitions t_A, t_B
       Initial weights α₀, β₀, γ₀
Output: Trained model θ*

 1  ▷ Phase A: Semantic Foundation
 2  for t ← 1 to t_A do
 3      L_sem ← ComputeSemanticLoss(θ)
 4      θ ← θ - η · ∇_θ L_sem
 5  end for
 6  
 7  ▷ Phase B: Instance Integration with Gradient Projection
 8  for t ← t_A + 1 to t_B do
 9      L_sem ← ComputeSemanticLoss(θ)
10      L_inst ← ComputeInstanceLoss(θ)
11      
12      g_sem ← ∇_θ L_sem
13      g_inst ← ∇_θ L_inst
14      
15      ▷ Project away conflicting gradient component
16      conflict ← min(0, ⟨g_inst, g_sem⟩)
17      g_inst_proj ← g_inst - (conflict / (‖g_sem‖² + ε)) · g_sem
18      
19      ▷ Compute adaptive weight
20      β_t ← β₀ · (1 - exp(-(t - t_A) / τ_β))
21      
22      g_total ← g_sem + β_t · g_inst_proj
23      θ ← θ - η · g_total
24  end for
25  
26  ▷ Phase C: Joint Refinement with Bridge
27  θ_ema ← θ                                  ▷ Initialize EMA
28  for t ← t_B + 1 to T do
29      L_sem ← ComputeSemanticLoss(θ)
30      L_inst ← ComputeInstanceLoss(θ)
31      L_bridge ← ComputeBridgeLoss(θ)
32      L_consist ← ComputeConsistencyLoss(θ, θ_ema)
33      
34      ▷ Full loss with balanced weights
35      ρ ← ‖∇_θ L_sem‖ / (‖∇_θ L_inst‖ + ε)
36      α_t ← α₀ / (1 + log(1 + ρ))
37      β_t ← β₀ · (1 + log(1 + 1/ρ))
38      
39      L_total ← α_t · L_sem + β_t · L_inst + γ₀ · L_bridge + L_consist
40      θ ← θ - η · ∇_θ L_total
41      
42      ▷ Update EMA
43      θ_ema ← μ · θ_ema + (1 - μ) · θ
44  end for
45  
46  return θ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Challenge 3: Stuff-Things Disambiguation Without Motion

### Problem Statement
CUPS uses stereo video motion (optical flow, scene flow) to distinguish stuff (static background) from things (moving objects). Without stereo video, we need alternative geometric and feature-based cues.

### Solution: Multi-Cue Stuff-Things Classifier (MC-STC)

We derive three complementary cues from monocular depth and DINO features:

**Cue 1: Depth Boundary Density (DBD)**

**Definition 3.1 (Depth Gradient Magnitude):**
$$G_d(x, y) = \sqrt{\left(\frac{\partial D}{\partial x}\right)^2 + \left(\frac{\partial D}{\partial y}\right)^2}$$

**Definition 3.2 (Depth Boundary Density for region $R$):**
$$\text{DBD}(R) = \frac{1}{|R|} \sum_{(x,y) \in R} \mathbb{1}[G_d(x,y) > \tau_d]$$

**Intuition:** Things have sharp depth boundaries (cars, people have distinct depth from background), while stuff regions (sky, road) have smooth depth gradients.

**Cue 2: Feature Cluster Compactness (FCC)**

**Definition 3.3 (Intra-class Feature Variance):**
For semantic cluster $c$ with features $\{f_i\}_{i \in c}$:
$$\text{FCC}(c) = 1 - \frac{\text{tr}(\Sigma_c)}{\text{tr}(\Sigma_{\text{total}})}$$
where $\Sigma_c = \frac{1}{|c|}\sum_{i \in c}(f_i - \mu_c)(f_i - \mu_c)^T$

**Intuition:** Thing classes (specific objects) form tighter clusters in feature space than stuff classes (diverse textures like grass, sky).

**Cue 3: Instance Decomposition Frequency (IDF)**

**Definition 3.4 (Instance Decomposition Frequency):**
$$\text{IDF}(c) = \frac{\mathbb{E}[\text{num\_instances}(R) \mid \text{semantic}(R) = c]}{\mathbb{E}[\text{area}(R) \mid \text{semantic}(R) = c]}$$

**Intuition:** Regions frequently decomposed into multiple instances by CutS3D are likely things.

### Stuff-Things Classification

**Definition 3.5 (Stuff-Things Score):**
$$\text{ST}(c) = w_1 \cdot \text{DBD}(c) + w_2 \cdot \text{FCC}(c) + w_3 \cdot \text{IDF}(c)$$

**Classification Rule:**
$$\text{class}(c) = \begin{cases} \text{thing} & \text{if } \text{ST}(c) > \tau_{\text{st}} \\ \text{stuff} & \text{otherwise} \end{cases}$$

**Theorem 3.1 (Optimal Threshold):**
The optimal threshold $\tau_{\text{st}}^*$ maximizes the F1-score on a held-out validation set (using GT stuff/thing labels for threshold tuning only—the classifier itself remains unsupervised):
$$\tau_{\text{st}}^* = \arg\max_\tau F_1(\text{ST}, \tau)$$

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHM 3: MULTI-CUE-STUFF-THINGS-CLASSIFIER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Depth map D ∈ ℝ^(H×W), DINO features F ∈ ℝ^(C×H×W)
       Semantic clusters {c₁, ..., c_K}, Instance masks {m₁, ..., m_M}
       Weights w₁, w₂, w₃, Threshold τ_st
Output: stuff_classes, thing_classes

 1  ▷ Compute Depth Boundary Density for each cluster
 2  G_d ← SobelFilter(D)                       ▷ Depth gradient magnitude
 3  for each cluster c_k do
 4      R_k ← pixels where semantic_label = k
 5      boundary_pixels ← count(G_d[R_k] > τ_d)
 6      DBD[k] ← boundary_pixels / |R_k|
 7  end for
 8  
 9  ▷ Compute Feature Cluster Compactness
10  F_flat ← Flatten(F, spatial_dims)          ▷ ℝ^(C × N)
11  Σ_total ← Cov(F_flat)
12  for each cluster c_k do
13      F_k ← F_flat[:, semantic_label = k]
14      Σ_k ← Cov(F_k)
15      FCC[k] ← 1 - trace(Σ_k) / trace(Σ_total)
16  end for
17  
18  ▷ Compute Instance Decomposition Frequency
19  for each cluster c_k do
20      instances_in_k ← 0
21      area_k ← 0
22      for each instance mask m_j do
23          overlap ← |m_j ∩ R_k| / |m_j|
24          if overlap > 0.5 then
25              instances_in_k ← instances_in_k + 1
26          end if
27      end for
28      area_k ← |R_k|
29      IDF[k] ← instances_in_k / (area_k / (H × W))
30  end for
31  
32  ▷ Normalize cues to [0, 1]
33  DBD ← (DBD - min(DBD)) / (max(DBD) - min(DBD))
34  FCC ← (FCC - min(FCC)) / (max(FCC) - min(FCC))
35  IDF ← (IDF - min(IDF)) / (max(IDF) - min(IDF))
36  
37  ▷ Compute Stuff-Things Score
38  for each cluster c_k do
39      ST[k] ← w₁ · DBD[k] + w₂ · FCC[k] + w₃ · IDF[k]
40  end for
41  
42  ▷ Classify
43  thing_classes ← {k : ST[k] > τ_st}
44  stuff_classes ← {k : ST[k] ≤ τ_st}
45  
46  return stuff_classes, thing_classes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Learning the Weights

Rather than hand-tuning $w_1, w_2, w_3$, we learn them via a small MLP on a held-out split:

**Definition 3.6 (Learned Stuff-Things Classifier):**
$$p(\text{thing} | c) = \sigma(\text{MLP}([\text{DBD}(c); \text{FCC}(c); \text{IDF}(c)]))$$

where MLP has architecture: Linear(3, 16) → ReLU → Linear(16, 8) → ReLU → Linear(8, 1)

---

## Challenge 4: Depth Consistency Across Branches

### Solution: Unified Depth Conditioning Module (UDCM)

**Definition 4.1 (Depth Feature Encoding):**
$$\mathbf{d} = \text{DepthEncoder}(D) = \text{Conv}_{1\times1}(\text{PositionalEncode}(D))$$
where $\mathbf{d} \in \mathbb{R}^{D_b \times H \times W}$

**Definition 4.2 (Depth-Conditioned Semantic Features):**
$$\mathbf{s}_d = \mathbf{s}' \odot \sigma(\mathbf{W}_d^s \mathbf{d}) + \mathbf{s}'$$

**Definition 4.3 (Depth-Conditioned Instance Features):**
$$\mathbf{f}_d = \mathbf{f}' \odot \sigma(\mathbf{W}_d^f \mathbf{d}) + \mathbf{f}'$$

**Definition 4.4 (Depth Consistency Loss):**
$$\mathcal{L}_{\text{depth}} = \sum_{(i,j) \in \mathcal{E}} |D_i - D_j| \cdot \left( \|\mathbf{s}_i - \mathbf{s}_j\|_2 + \|\mathbf{f}_i - \mathbf{f}_j\|_2 \right)$$
where $\mathcal{E}$ is the set of spatially adjacent pixel pairs.

**Intuition:** Pixels with similar depth should have similar semantic and instance features.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHM 4: UNIFIED-DEPTH-CONDITIONING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Depth D ∈ ℝ^(H×W), Semantic features s' ∈ ℝ^(D_b×H×W)
       Instance features f' ∈ ℝ^(D_b×H×W)
Output: Depth-conditioned features s_d, f_d, Consistency loss L_depth

 1  ▷ Encode depth with positional information
 2  D_pe ← PositionalEncode(D)                 ▷ Sinusoidal encoding
 3  d ← Conv1x1(D_pe)                          ▷ ℝ^(D_b×H×W)
 4  
 5  ▷ Compute depth-aware gates
 6  gate_s ← Sigmoid(W_d^s · d)
 7  gate_f ← Sigmoid(W_d^f · d)
 8  
 9  ▷ Apply gated conditioning (residual)
10  s_d ← s' ⊙ gate_s + s'
11  f_d ← f' ⊙ gate_f + f'
12  
13  ▷ Compute depth consistency loss
14  L_depth ← 0
15  for each adjacent pixel pair (i, j) do
16      depth_diff ← |D[i] - D[j]|
17      feat_diff ← ‖s_d[i] - s_d[j]‖₂ + ‖f_d[i] - f_d[j]‖₂
18      L_depth ← L_depth + depth_diff · feat_diff
19  end for
20  L_depth ← L_depth / num_pairs
21  
22  return s_d, f_d, L_depth
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Challenge 5: Mamba2/SSD for TPU (Memory & Compute)

### Mamba2 State Space Duality Formulation

Mamba2 reformulates the selective state space model as structured matrix multiplications, enabling efficient TPU execution.

**Definition 5.1 (Mamba2 SSD Layer):**
$$\mathbf{y} = \text{SSM}(\mathbf{A}, \mathbf{B}, \mathbf{C})(\mathbf{x}) = \mathbf{C} \cdot \mathbf{M} \cdot (\mathbf{B} \odot \mathbf{x})$$

where $\mathbf{M} \in \mathbb{R}^{L \times L}$ is a structured (semiseparable) matrix:
$$\mathbf{M}_{ij} = \begin{cases} \mathbf{C}_i^T \mathbf{A}^{i-j} \mathbf{B}_j & i \geq j \\ 0 & i < j \end{cases}$$

**Definition 5.2 (Chunked Computation for TPU):**
Divide sequence into chunks of size $P$ (typically 64-256 for TPU):
$$\mathbf{y}^{(k)} = \mathbf{C}^{(k)} \mathbf{h}^{(k-1)} + \text{IntraChunkSSM}(\mathbf{x}^{(k)})$$

where $\mathbf{h}^{(k)}$ is the hidden state passed between chunks.

**Theorem 5.1 (TPU FLOP Efficiency):**
Chunked SSD achieves $O(LP + LN/P)$ FLOPs where $L$ is sequence length, $P$ is chunk size, and $N$ is state dimension. This maps efficiently to TPU's 128×128 systolic arrays when $P = 128$.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHM 5: MAMBA2-SSD-TPU-FORWARD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Input sequence x ∈ ℝ^(B×L×D), Chunk size P = 128
       Learnable parameters: A ∈ ℝ^(D×N), B_proj ∈ ℝ^(D×N), C_proj ∈ ℝ^(D×N)
       Δ projection, D residual
Output: Output sequence y ∈ ℝ^(B×L×D)

 1  ▷ Compute input-dependent parameters
 2  Δ ← Softplus(Linear_Δ(x))                  ▷ ℝ^(B×L×D)
 3  B ← Linear_B(x)                            ▷ ℝ^(B×L×N)
 4  C ← Linear_C(x)                            ▷ ℝ^(B×L×N)
 5  
 6  ▷ Discretize A
 7  A_bar ← exp(Δ ⊙ A)                         ▷ ℝ^(B×L×D×N)
 8  
 9  ▷ Chunk the sequence for TPU efficiency
10  num_chunks ← L / P
11  x_chunks ← Reshape(x, [B, num_chunks, P, D])
12  
13  ▷ Initialize hidden state
14  h ← zeros(B, D, N)
15  y_chunks ← []
16  
17  for k ← 1 to num_chunks do
18      x_k ← x_chunks[:, k, :, :]             ▷ ℝ^(B×P×D)
19      B_k ← B[:, k*P:(k+1)*P, :]
20      C_k ← C[:, k*P:(k+1)*P, :]
21      A_bar_k ← A_bar[:, k*P:(k+1)*P, :, :]
22      
23      ▷ Intra-chunk SSM via matrix multiplication (TPU-friendly)
24      ▷ Build semiseparable matrix M for chunk
25      M_k ← BuildSemiseparableMatrix(A_bar_k, B_k, C_k, P)
26      
27      ▷ Contribution from previous hidden state
28      y_from_h ← einsum('bdn,bpn->bpd', h, C_k)
29      
30      ▷ Intra-chunk computation as matmul
31      y_intra ← einsum('bpq,bqd->bpd', M_k, x_k ⊙ B_k.unsqueeze(-1))
32      
33      y_k ← y_from_h + y_intra
34      y_chunks.append(y_k)
35      
36      ▷ Update hidden state for next chunk
37      h ← UpdateHiddenState(h, A_bar_k, B_k, x_k)
38  end for
39  
40  ▷ Concatenate and add residual
41  y ← Concat(y_chunks, dim=1)
42  y ← y + D_param · x                        ▷ Skip connection
43  
44  return y
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### JAX/Flax Implementation Notes for TPU

```python
# Key TPU optimizations for Mamba2:
# 1. Use jnp.einsum with optimal contraction paths
# 2. Pad sequences to multiples of 128 for systolic array alignment
# 3. Use bfloat16 for matrix multiplications
# 4. Shard batch dimension across TPU cores with pmap
```

---

## Challenge 6: Scan Direction for Semantic-Instance Fusion

### Solution: Bidirectional Cross-Modal Scan (BiCMS)

**Definition 6.1 (Concatenated Feature Sequence):**
$$\mathbf{z} = [\mathbf{s}_1, \mathbf{f}_1, \mathbf{s}_2, \mathbf{f}_2, ..., \mathbf{s}_N, \mathbf{f}_N]$$
(interleaved semantic-instance tokens at each spatial position)

**Definition 6.2 (Forward Scan):**
$$\mathbf{y}^{\rightarrow} = \text{Mamba2}(\mathbf{z})$$
Information flows: semantic → instance at each position, earlier positions → later positions.

**Definition 6.3 (Backward Scan):**
$$\mathbf{y}^{\leftarrow} = \text{Mamba2}(\text{reverse}(\mathbf{z}))$$

**Definition 6.4 (Bidirectional Fusion):**
$$\mathbf{y} = \mathbf{W}_{\text{fuse}}[\mathbf{y}^{\rightarrow}; \text{reverse}(\mathbf{y}^{\leftarrow})] + \mathbf{z}$$

**Theorem 6.1 (Information Flow Coverage):**
BiCMS ensures that each token receives information from:
- All preceding semantic tokens (forward scan)
- All succeeding instance tokens (backward scan)  
- Local semantic-instance interaction at its position (interleaving)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHM 6: BIDIRECTIONAL-CROSS-MODAL-SCAN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Semantic features S ∈ ℝ^(N×D_b), Instance features F ∈ ℝ^(N×D_b)
       Mamba2 layer, Fusion projection W_fuse
Output: Fused features Y_s ∈ ℝ^(N×D_b), Y_f ∈ ℝ^(N×D_b)

 1  ▷ Interleave semantic and instance tokens
 2  Z ← []
 3  for i ← 1 to N do
 4      Z.append(S[i])
 5      Z.append(F[i])
 6  end for
 7  Z ← Stack(Z)                               ▷ ℝ^(2N×D_b)
 8  
 9  ▷ Forward scan
10  Y_fwd ← Mamba2_Forward(Z)                  ▷ ℝ^(2N×D_b)
11  
12  ▷ Backward scan
13  Z_rev ← Reverse(Z)
14  Y_bwd_rev ← Mamba2_Forward(Z_rev)
15  Y_bwd ← Reverse(Y_bwd_rev)                 ▷ ℝ^(2N×D_b)
16  
17  ▷ Fuse bidirectional outputs
18  Y_concat ← Concat([Y_fwd, Y_bwd], dim=-1)  ▷ ℝ^(2N×2D_b)
19  Y_fused ← Linear(Y_concat, D_b) + Z        ▷ Residual connection
20  
21  ▷ Deinterleave back to semantic and instance
22  Y_s ← Y_fused[0::2]                        ▷ Even indices: semantic
23  Y_f ← Y_fused[1::2]                        ▷ Odd indices: instance
24  
25  return Y_s, Y_f
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Challenge 7: Pseudo-Label Quality Ceiling

### Solution: Confidence-Weighted Multi-Round Self-Training (CW-MRST)

**Definition 7.1 (Semantic Confidence):**
$$c_s(i) = \max_k p(y_i = k | \mathbf{x})$$
(maximum softmax probability across semantic classes)

**Definition 7.2 (Instance Confidence):**
From CutS3D's spatial confidence maps:
$$c_f(i) = \sigma\left(\frac{\text{score}(m_i)}{\tau_c}\right)$$
where $\text{score}(m_i)$ is the mask confidence from the instance head.

**Definition 7.3 (Joint Confidence):**
$$c(i) = c_s(i)^{\alpha} \cdot c_f(i)^{1-\alpha}$$
where $\alpha$ balances semantic vs instance confidence.

**Definition 7.4 (Confidence-Weighted Loss):**
$$\mathcal{L}_{\text{CW}} = \frac{\sum_i c(i) \cdot \ell(i)}{\sum_i c(i) + \epsilon}$$

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHM 7: CONFIDENCE-WEIGHTED-SELF-TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Model θ, Dataset D, Number of rounds R
       Confidence threshold schedule {τ_r}_{r=1}^R
       EMA momentum μ
Output: Refined model θ*

 1  θ_teacher ← θ                              ▷ Initialize teacher
 2  
 3  for r ← 1 to R do
 4      ▷ Generate pseudo-labels with teacher
 5      PseudoLabels ← {}
 6      Confidences ← {}
 7      
 8      for each image x ∈ D do
 9          ▷ Get semantic predictions
10          p_sem ← Softmax(SemanticHead(θ_teacher, x))
11          y_sem ← ArgMax(p_sem)
12          c_sem ← Max(p_sem)
13          
14          ▷ Get instance predictions
15          masks, scores ← InstanceHead(θ_teacher, x)
16          c_inst ← Sigmoid(scores / τ_c)
17          
18          ▷ Compute joint confidence
19          c_joint ← c_sem^α · MaskToPixelConfidence(c_inst, masks)^(1-α)
20          
21          ▷ Filter by confidence threshold
22          valid_mask ← c_joint > τ_r
23          
24          PseudoLabels[x] ← (y_sem, masks, valid_mask)
25          Confidences[x] ← c_joint
26      end for
27      
28      ▷ Train student with confidence-weighted loss
29      for epoch ← 1 to E_r do
30          for each batch (x, pseudo, conf) do
31              L_sem ← ConfidenceWeightedCE(pred_sem, pseudo_sem, conf)
32              L_inst ← ConfidenceWeightedMask(pred_inst, pseudo_inst, conf)
33              L_bridge ← BridgeLoss(θ)
34              
35              L_total ← L_sem + λ_inst · L_inst + λ_bridge · L_bridge
36              θ ← θ - η · ∇_θ L_total
37          end for
38      end for
39      
40      ▷ Update teacher via EMA
41      θ_teacher ← μ · θ_teacher + (1 - μ) · θ
42      
43      ▷ Increase confidence threshold for next round
44      τ_{r+1} ← τ_r + Δτ
45  end for
46  
47  return θ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

# Part II: Mathematical Derivations for Loss Functions

## Loss 1: Semantic Branch Loss ($\mathcal{L}_{\text{semantic}}$)

Building on DepthG's depth-guided feature correlation:

### STEGO Correspondence Loss

**Definition (Feature Correspondence):**
$$\mathcal{L}_{\text{STEGO}} = -\sum_{(i,j) \in \mathcal{P}^+} \log \frac{\exp(\mathbf{s}_i^T \mathbf{s}_j / \tau)}{\sum_{k \in \mathcal{N}(i)} \exp(\mathbf{s}_i^T \mathbf{s}_k / \tau)}$$

where $\mathcal{P}^+$ are positive pairs from KNN in DINO feature space.

### Depth-Guided Feature Correlation Loss

**Definition (Depth Correlation Weight):**
$$w_{ij}^d = \exp\left(-\frac{|D_i - D_j|^2}{2\sigma_d^2}\right)$$

**Definition (Depth-Guided Correlation Loss):**
$$\mathcal{L}_{\text{DepthG}} = \sum_{(i,j)} w_{ij}^d \cdot \left(1 - \frac{\mathbf{s}_i^T \mathbf{s}_j}{\|\mathbf{s}_i\| \|\mathbf{s}_j\|}\right)^2$$

**Combined Semantic Loss:**
$$\boxed{\mathcal{L}_{\text{semantic}} = \mathcal{L}_{\text{STEGO}} + \lambda_d \cdot \mathcal{L}_{\text{DepthG}}}$$

**Theorem (Gradient of $\mathcal{L}_{\text{DepthG}}$):**
$$\frac{\partial \mathcal{L}_{\text{DepthG}}}{\partial \mathbf{s}_i} = \sum_j w_{ij}^d \cdot 2\left(1 - \cos\theta_{ij}\right) \cdot \frac{\mathbf{s}_j - \cos\theta_{ij} \cdot \mathbf{s}_i}{\|\mathbf{s}_i\| \|\mathbf{s}_j\|}$$

This gradient pushes features of depth-similar pixels together.

---

## Loss 2: Instance Branch Loss ($\mathcal{L}_{\text{instance}}$)

Building on CutS3D's instance detection framework:

### Spatial Confidence-Weighted BCE

**Definition (Spatial Confidence Map):**
$$C_{\text{spatial}}(i) = \min_{j \in \mathcal{N}_k(i)} \text{score}(m_{j \rightarrow i})$$
where $m_{j \rightarrow i}$ is the mask prediction from anchor $j$ covering pixel $i$.

**Definition (Confidence-Weighted BCE):**
$$\mathcal{L}_{\text{BCE}}^{\text{CW}} = -\sum_i C_{\text{spatial}}(i) \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]$$

### DropLoss for Handling Missing Instances

**Definition (DropLoss):**
$$\mathcal{L}_{\text{Drop}} = \sum_{m \in \mathcal{M}} \mathbb{1}[\text{IoU}(m, \hat{m}^*) < \tau_{\text{drop}}] \cdot \|\mathbf{f}_m\|_2^2$$

where $\hat{m}^*$ is the best matching predicted mask. This regularizes features of unmatched instances.

**Combined Instance Loss:**
$$\boxed{\mathcal{L}_{\text{instance}} = \mathcal{L}_{\text{BCE}}^{\text{CW}} + \lambda_{\text{drop}} \cdot \mathcal{L}_{\text{Drop}} + \mathcal{L}_{\text{box}}}$$

---

## Loss 3: Mamba Bridge Loss ($\mathcal{L}_{\text{bridge}}$)

### Feature Reconstruction Loss

Ensuring the bridge doesn't destroy information:

$$\mathcal{L}_{\text{recon}} = \|\mathbf{s} - \hat{\mathbf{s}}\|_2^2 + \|\mathbf{f} - \hat{\mathbf{f}}\|_2^2$$

where $\hat{\mathbf{s}}, \hat{\mathbf{f}}$ are reconstructed from bridge outputs.

### Cross-Modal Alignment Loss

Using Centered Kernel Alignment (CKA):

**Definition (CKA):**
$$\text{CKA}(\mathbf{S}, \mathbf{F}) = \frac{\|\mathbf{S}^T\mathbf{F}\|_F^2}{\|\mathbf{S}^T\mathbf{S}\|_F \cdot \|\mathbf{F}^T\mathbf{F}\|_F}$$

$$\mathcal{L}_{\text{CKA}} = 1 - \text{CKA}(\mathbf{S}', \mathbf{F}')$$

### Mamba State Regularization

Preventing state explosion in SSM:

$$\mathcal{L}_{\text{state}} = \|\mathbf{h}_T\|_2^2$$

where $\mathbf{h}_T$ is the final hidden state.

**Combined Bridge Loss:**
$$\boxed{\mathcal{L}_{\text{bridge}} = \lambda_r \cdot \mathcal{L}_{\text{recon}} + \lambda_{\text{cka}} \cdot \mathcal{L}_{\text{CKA}} + \lambda_h \cdot \mathcal{L}_{\text{state}}}$$

---

## Loss 4: Cross-Branch Consistency Loss ($\mathcal{L}_{\text{consistency}}$)

### Semantic Uniformity Within Instances

**Definition (Instance-Semantic Entropy):**
$$H_k = -\sum_{c=1}^{C} p_k(c) \log p_k(c)$$

where $p_k(c) = \frac{|\{i \in m_k : y_i^{\text{sem}} = c\}|}{|m_k|}$ is the proportion of semantic class $c$ within instance mask $k$.

**Definition (Semantic Uniformity Loss):**
$$\mathcal{L}_{\text{uniform}} = \frac{1}{K}\sum_{k=1}^{K} H_k$$

**Goal:** Minimize entropy → each instance should have a single dominant semantic class.

### Boundary Alignment Loss

**Definition (Semantic Boundary):**
$$B_{\text{sem}}(i,j) = \mathbb{1}[y_i^{\text{sem}} \neq y_j^{\text{sem}}] \quad \text{for adjacent } (i,j)$$

**Definition (Instance Boundary):**
$$B_{\text{inst}}(i,j) = \mathbb{1}[\exists k: i \in m_k, j \notin m_k]$$

**Definition (Boundary Alignment Loss):**
$$\mathcal{L}_{\text{boundary}} = \sum_{(i,j) \in \mathcal{E}} \left| B_{\text{sem}}(i,j) - B_{\text{inst}}(i,j) \right|$$

### Depth-Boundary Coherence

Both boundaries should align with depth discontinuities:

**Definition (Depth Boundary):**
$$B_{\text{depth}}(i,j) = \mathbb{1}[|D_i - D_j| > \tau_D]$$

**Definition (Depth-Boundary Coherence Loss):**
$$\mathcal{L}_{\text{DBC}} = \sum_{(i,j)} \left( B_{\text{depth}}(i,j) - B_{\text{sem}}(i,j) \right)^2 + \left( B_{\text{depth}}(i,j) - B_{\text{inst}}(i,j) \right)^2$$

**Combined Consistency Loss:**
$$\boxed{\mathcal{L}_{\text{consistency}} = \lambda_u \cdot \mathcal{L}_{\text{uniform}} + \lambda_b \cdot \mathcal{L}_{\text{boundary}} + \lambda_{dbc} \cdot \mathcal{L}_{\text{DBC}}}$$

---

## Loss 5: Panoptic Quality Proxy Loss ($\mathcal{L}_{\text{PQ}}$)

Direct optimization of a differentiable PQ approximation:

### Soft Segment Matching

**Definition (Soft IoU):**
$$\text{IoU}_{\text{soft}}(m, \hat{m}) = \frac{\sum_i \min(m_i, \hat{m}_i)}{\sum_i \max(m_i, \hat{m}_i)}$$

### Differentiable PQ

**Definition (Soft True Positives):**
$$\text{TP}_{\text{soft}} = \sum_{(m, \hat{m})} \sigma\left(\frac{\text{IoU}_{\text{soft}}(m, \hat{m}) - 0.5}{\tau_{\text{pq}}}\right)$$

**Definition (Differentiable PQ Loss):**
$$\mathcal{L}_{\text{PQ}} = 1 - \frac{2 \cdot \text{TP}_{\text{soft}}}{\text{TP}_{\text{soft}} + |\mathcal{M}| + |\hat{\mathcal{M}}|}$$

---

## Total Loss Function

$$\boxed{\mathcal{L}_{\text{total}} = \underbrace{\alpha \cdot \mathcal{L}_{\text{semantic}}}_{\text{Depth-guided semantics}} + \underbrace{\beta \cdot \mathcal{L}_{\text{instance}}}_{\text{3D-aware instances}} + \underbrace{\gamma \cdot \mathcal{L}_{\text{bridge}}}_{\text{Mamba fusion}} + \underbrace{\delta \cdot \mathcal{L}_{\text{consistency}}}_{\text{Cross-branch coherence}} + \underbrace{\epsilon \cdot \mathcal{L}_{\text{PQ}}}_{\text{Panoptic quality}}}$$

**Recommended Hyperparameters:**
- $\alpha = 1.0$ (primary semantic weight)
- $\beta = 0.5 \rightarrow 1.0$ (curriculum: ramp up over training)
- $\gamma = 0.1$ (bridge regularization)
- $\delta = 0.3$ (consistency enforcement)
- $\epsilon = 0.2$ (panoptic refinement, added in Phase C)

---

# Part III: Complete Training Pipeline Algorithm

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHM 8: MAMBA-BRIDGE-PANOPTIC-SEGMENTATION (MBPS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Dataset D = {(x_i, D_i)}  ▷ Images with monocular depth
       Frozen DINO ViT-S/8 backbone φ
       Frozen ZoeDepth estimator ψ (if depth not provided)
       Hyperparameters: epochs T, learning rate η, batch size B
       Phase transitions: t_A, t_B, self-training rounds R

Output: Trained MBPS model θ* producing panoptic segmentation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▷ INITIALIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1  ▷ Initialize model components
 2  SemanticHead ← InitDepthGHead(in_dim=384, code_dim=90)
 3  InstanceHead ← InitCutS3DHead(in_dim=384)
 4  ProjectionBridge ← InitAdaptiveProjection(D_s=90, D_f=384, D_b=192)
 5  MambaBridge ← InitMamba2SSD(dim=192, num_layers=4, chunk_size=128)
 6  StuffThingsClassifier ← InitMLPClassifier(in_dim=3, hidden=16)
 7  PanopticMerger ← InitPanopticHead()
 8  
 9  θ ← {SemanticHead, InstanceHead, ProjectionBridge, 
         MambaBridge, StuffThingsClassifier, PanopticMerger}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▷ PHASE A: SEMANTIC FOUNDATION (epochs 1 to t_A)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10  for t ← 1 to t_A do
11      for each batch {(x, D)} ∈ D do
12          ▷ Extract DINO features
13          F ← φ(x)                            ▷ ℝ^(B×N×384)
14          
15          ▷ Estimate depth if not provided
16          if D is None then
17              D ← ψ(x)
18          end if
19          
20          ▷ Compute semantic codes
21          S ← SemanticHead(F)                 ▷ ℝ^(B×N×90)
22          
23          ▷ Compute semantic loss with depth guidance
24          L_STEGO ← ComputeSTEGOLoss(S, F)
25          L_DepthG ← ComputeDepthCorrelationLoss(S, D)
26          L_semantic ← L_STEGO + λ_d · L_DepthG
27          
28          ▷ Update semantic head only
29          θ_sem ← θ_sem - η · ∇_{θ_sem} L_semantic
30      end for
31  end for

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▷ PHASE B: INSTANCE INTEGRATION (epochs t_A+1 to t_B)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
32  for t ← t_A + 1 to t_B do
33      β_t ← β_0 · (1 - exp(-(t - t_A) / τ_β))  ▷ Curriculum weight
34      
35      for each batch {(x, D)} ∈ D do
36          F ← φ(x)
37          S ← SemanticHead(F)
38          
39          ▷ Generate instance pseudo-labels via CutS3D pipeline
40          Masks, Scores ← GenerateCutS3DPseudoMasks(F, D)
41          
42          ▷ Compute instance predictions
43          Masks_pred, Scores_pred ← InstanceHead(F)
44          
45          ▷ Compute losses
46          L_semantic ← ComputeSemanticLoss(S, D)
47          L_instance ← ComputeInstanceLoss(Masks_pred, Masks, Scores)
48          
49          ▷ Gradient conflict resolution
50          g_sem ← ∇_θ L_semantic
51          g_inst ← ∇_θ L_instance
52          conflict ← min(0, ⟨g_inst, g_sem⟩)
53          g_inst_proj ← g_inst - (conflict / (‖g_sem‖² + ε)) · g_sem
54          
55          ▷ Combined update
56          θ ← θ - η · (g_sem + β_t · g_inst_proj)
57      end for
58  end for

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▷ PHASE C: JOINT TRAINING WITH MAMBA BRIDGE (epochs t_B+1 to T)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
59  θ_ema ← θ                                   ▷ Initialize EMA teacher
60  
61  for t ← t_B + 1 to T do
62      for each batch {(x, D)} ∈ D do
63          F ← φ(x)                            ▷ DINO features
64          S ← SemanticHead(F)                 ▷ Semantic codes
65          
66          ▷ Project to bridge dimension
67          S', F', L_align ← AdaptiveProjectionBridge(S, F)
68          
69          ▷ Depth conditioning
70          S_d, F_d, L_depth ← UnifiedDepthConditioning(D, S', F')
71          
72          ▷ Bidirectional Mamba fusion
73          S_fused, F_fused ← BidirectionalCrossModalScan(S_d, F_d)
74          
75          ▷ Inverse projection back to original dimensions
76          S_out ← InverseProject_S(S_fused)   ▷ ℝ^(B×N×90)
77          F_out ← InverseProject_F(F_fused)   ▷ ℝ^(B×N×384)
78          
79          ▷ Generate predictions
80          Sem_pred ← ClusterAndCRF(S_out)
81          Masks_pred, Scores_pred ← InstanceHead(F_out)
82          
83          ▷ Stuff-Things classification
84          Clusters ← GetSemanticClusters(Sem_pred)
85          DBD ← ComputeDepthBoundaryDensity(Clusters, D)
86          FCC ← ComputeFeatureClusterCompactness(Clusters, F)
87          IDF ← ComputeInstanceDecompFreq(Clusters, Masks_pred)
88          ST_scores ← StuffThingsClassifier([DBD; FCC; IDF])
89          Things_clusters ← {c : ST_scores[c] > 0.5}
90          Stuff_clusters ← {c : ST_scores[c] ≤ 0.5}
91          
92          ▷ Merge into panoptic output
93          Panoptic_pred ← PanopticMerger(Sem_pred, Masks_pred, 
94                                          Things_clusters, Stuff_clusters)
95          
96          ▷ Compute all losses
97          L_semantic ← ComputeSemanticLoss(S_out, D)
98          L_instance ← ComputeInstanceLoss(Masks_pred, Scores_pred)
99          L_bridge ← L_align + λ_cka · ComputeCKALoss(S_fused, F_fused)
100                    + λ_h · ComputeStateRegularization(MambaBridge)
101         
102         L_uniform ← ComputeInstanceSemanticEntropy(Masks_pred, Sem_pred)
103         L_boundary ← ComputeBoundaryAlignment(Sem_pred, Masks_pred)
104         L_DBC ← ComputeDepthBoundaryCoherence(Sem_pred, Masks_pred, D)
105         L_consistency ← λ_u · L_uniform + λ_b · L_boundary + λ_dbc · L_DBC
106         
107         L_PQ ← ComputeDifferentiablePQLoss(Panoptic_pred, θ_ema)
108         
109         ▷ Total loss
110         L_total ← α · L_semantic + β · L_instance + γ · L_bridge 
111                   + δ · L_consistency + ε · L_PQ
112         
113         ▷ Update model
114         θ ← θ - η · ∇_θ L_total
115         
116         ▷ Update EMA teacher
117         θ_ema ← μ · θ_ema + (1 - μ) · θ
118     end for
119 end for

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▷ PHASE D: SELF-TRAINING REFINEMENT (R rounds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
120 for r ← 1 to R do
121     ▷ Generate high-confidence pseudo-labels
122     PseudoLabels ← GeneratePseudoLabels(θ_ema, D, conf_threshold=τ_r)
123     
124     ▷ Retrain with filtered pseudo-labels
125     for epoch ← 1 to E_r do
126         for each (x, pseudo, conf) ∈ PseudoLabels do
127             L_total ← ConfidenceWeightedLoss(θ, x, pseudo, conf)
128             θ ← θ - η_r · ∇_θ L_total
129         end for
130     end for
131     
132     ▷ Update teacher and increase threshold
133     θ_ema ← μ · θ_ema + (1 - μ) · θ
134     τ_{r+1} ← τ_r + 0.05
135 end for

136 return θ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

# Part IV: Panoptic Merging Algorithm

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHM 9: PANOPTIC-MERGING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Semantic segmentation Sem ∈ {1,...,K}^(H×W)
       Instance masks M = {m_1, ..., m_N}, each m_i ∈ {0,1}^(H×W)
       Instance scores S = {s_1, ..., s_N}
       Thing classes T ⊆ {1,...,K}
       Stuff classes St = {1,...,K} \ T
       Overlap threshold τ_overlap = 0.5

Output: Panoptic segmentation P ∈ (ℕ × {1,...,K})^(H×W)
        Each pixel (id, class) where id=0 for stuff, id>0 for things

 1  ▷ Initialize output
 2  P ← zeros(H, W, 2)                         ▷ (instance_id, class)
 3  instance_counter ← 0
 4  used_pixels ← zeros(H, W)                  ▷ Track assigned pixels
 5  
 6  ▷ Sort instance masks by score (descending)
 7  sorted_indices ← ArgSort(S, descending=True)
 8  
 9  ▷ Process thing instances (non-overlapping, higher score priority)
10  for i ∈ sorted_indices do
11      m_i ← M[i]
12      
13      ▷ Get majority semantic class within mask
14      class_counts ← Histogram(Sem[m_i], bins=K)
15      majority_class ← ArgMax(class_counts)
16      
17      ▷ Check if this is a thing class
18      if majority_class ∈ T then
19          ▷ Compute overlap with already assigned pixels
20          overlap ← Sum(m_i ∧ used_pixels) / Sum(m_i)
21          
22          if overlap < τ_overlap then
23              ▷ Assign this instance
24              instance_counter ← instance_counter + 1
25              valid_pixels ← m_i ∧ ¬used_pixels
26              P[valid_pixels] ← (instance_counter, majority_class)
27              used_pixels ← used_pixels ∨ valid_pixels
28          end if
29      end if
30  end for
31  
32  ▷ Process stuff regions (fill remaining pixels)
33  for each pixel (x, y) do
34      if used_pixels[x, y] = 0 then
35          semantic_class ← Sem[x, y]
36          if semantic_class ∈ St then
37              P[x, y] ← (0, semantic_class)   ▷ id=0 for stuff
38          else
39              ▷ Thing class pixel not covered by instance
40              ▷ Assign to nearest instance of same class or mark as void
41              nearest_id ← FindNearestInstanceOfClass(x, y, semantic_class, P)
42              if nearest_id ≠ None then
43                  P[x, y] ← (nearest_id, semantic_class)
44              else
45                  P[x, y] ← (0, semantic_class)  ▷ Treat as stuff
46              end if
47          end if
48      end if
49  end for
50  
51  return P
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

# Part V: Hyperparameter Recommendations

## Loss Weight Schedule

| Hyperparameter | Phase A | Phase B | Phase C | Self-Training |
|----------------|---------|---------|---------|---------------|
| α (semantic)   | 1.0     | 1.0     | 0.8     | 0.6           |
| β (instance)   | 0.0     | 0→1.0   | 1.0     | 1.0           |
| γ (bridge)     | 0.0     | 0.0     | 0.1     | 0.1           |
| δ (consistency)| 0.0     | 0.0     | 0.3     | 0.4           |
| ε (PQ proxy)   | 0.0     | 0.0     | 0.2     | 0.3           |

## Architecture Hyperparameters

| Component | Parameter | Recommended Value |
|-----------|-----------|-------------------|
| Projection Bridge | D_b | 192 |
| Mamba2 | num_layers | 4 |
| Mamba2 | state_dim (N) | 64 |
| Mamba2 | chunk_size (P) | 128 (TPU-aligned) |
| Stuff-Things MLP | hidden_dims | [16, 8] |
| CRF | iterations | 10 |

## Training Schedule

| Parameter | Value |
|-----------|-------|
| Total epochs (T) | 60 |
| Phase A end (t_A) | 20 |
| Phase B end (t_B) | 40 |
| Self-training rounds (R) | 3 |
| Initial LR | 1e-4 |
| LR schedule | Cosine decay |
| Batch size | 8 (per TPU core) |
| EMA momentum (μ) | 0.999 |
| Confidence threshold init (τ_1) | 0.7 |
| Confidence threshold increment (Δτ) | 0.05 |
