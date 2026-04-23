# LoRA/DoRA Adapter-Based Domain Adaptation for Stage-1 Pseudo-Label Generation

**Date:** 2026-04-21
**Authors:** Research Analysis (Senior NeurIPS Advisor)
**Scope:** Literature review and experimental protocol for adapting frozen Stage-1 models (DINOv2/CAUSE-TR, Depth Anything V3/DepthPro) via parameter-efficient fine-tuning to generate better pseudo-labels for unsupervised panoptic segmentation on Cityscapes.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [LoRA/DoRA for Domain Adaptation of Frozen Foundation Models](#2-loradora-for-domain-adaptation-of-frozen-foundation-models)
3. [Self-Supervised Training Objectives for Adapter Fine-Tuning](#3-self-supervised-training-objectives-for-adapter-fine-tuning)
4. [Parameter-Efficient Fine-Tuning Strategies](#4-parameter-efficient-fine-tuning-strategies)
5. [Evaluation Without Ground Truth](#5-evaluation-without-ground-truth)
6. [Specific Considerations for Our Pipeline](#6-specific-considerations-for-our-pipeline)
7. [Related Work Review](#7-related-work-review)
8. [Proposed Experimental Protocol](#8-proposed-experimental-protocol)
9. [Risk Assessment](#9-risk-assessment)
10. [Related Work Citations](#10-related-work-citations)

---

## 1. Executive Summary

### Key Findings

1. **The professor suggestion is well-founded and represents a significant paradigm shift.** Instead of adapting the backbone in Stage 2/3 (where our experiments show Conv-DoRA harms performance due to label noise amplification), we should apply **LoRA/DoRA adapters to the frozen Stage-1 generators** (DINOv2 + CAUSE-TR for semantic, DAv3/DepthPro for depth) to produce **higher-quality pseudo-labels** before Stage 2 even begins.

2. **ExPLoRA (Stanford, 2024)** provides the strongest precedent: extending unsupervised pre-training of DINOv2/MAE to new domains by unfreezing 1-2 ViT blocks and applying LoRA to all others, using the **original DINO self-distillation objective** on unlabeled target data. This achieves up to 8% improvement in linear probing while using <10% of parameters.

3. **For semantic features:** The most promising self-supervised objectives are:
   - **STEGO-style feature correspondence consistency** (contrastive correlation loss on DINO features)
   - **Depth-Feature Correlation (DepthG, CVPR 2024)** — directly aligns features with monocular depth, which we already compute
   - **Cross-view consistency** (different augmentations of the same image)
   - **InfoNCE / contrastive learning on patch features** with k-NN mining

4. **For depth estimation:** Self-supervised adaptation is highly feasible:
   - **Photometric consistency** across augmented views (classic self-supervised depth)
   - **Scale-invariant depth consistency** (SAG loss, ICCV 2021)
   - **Self-distillation from frozen teacher** (preserve zero-shot capability while adapting)
   - **Feature-depth alignment** (align depth features with DINO semantic features)

5. **DoRA > LoRA for feature adaptation.** DoRA (ICML 2024) decomposes weight updates into magnitude and direction, with only the direction being low-rank. It consistently outperforms LoRA on vision tasks and is more stable. **EVA** (Explained Variance Adaptation, 2024) further improves LoRA initialization by doing SVD on downstream activations — highly relevant for our domain shift.

6. **Conv-LoRA/DoRA is NOT recommended for feature extraction.** Our own experiments (see `why_conv_dora_not_plain_lora.md`) show that injecting spatial convolutions into DINOv3 global attention features destroys stuff-class performance. For Stage-1 feature adaptation, use **plain LoRA or DoRA on Q, V projections only**.

7. **Rank selection:** For ViT-B (86M params), ranks r=4-16 are sufficient. ExPLoRA uses r=8-16. Higher ranks (r=32) may overfit to pseudo-label noise. Alpha should equal rank or be 2x rank.

8. **Sequential training is safer than joint.** Train semantic adapter first (more stable), then depth adapter (can use improved semantic features as guidance), then optionally joint fine-tuning.

### Recommendations at a Glance

| Decision | Recommendation |
|----------|---------------|
| **Adapter type** | Plain DoRA (not Conv-DoRA) on Q, V attention weights |
| **Rank** | r=8 or r=16 for ViT-B; alpha = rank |
| **Semantic objective** | DepthG-style depth-feature correlation + STEGO correspondence |
| **Depth objective** | Self-distillation + photometric consistency + scale-invariant loss |
| **Training order** | Semantic first -> Depth second -> Joint optional |
| **Backbone choice** | Keep DINOv2 (proven with CAUSE-TR); DINOv3 is promising but less validated with CAUSE |
| **Depth model** | DepthPro for metric scale; DAv3 for relative depth and easier adaptation |
| **Evaluation** | Clustering metrics + proxy downstream Stage-2 PQ |

---

## 2. LoRA/DoRA for Domain Adaptation of Frozen Foundation Models

### 2.1 Core Concept: Test-Time Training vs. Target-Domain Pre-training

The literature distinguishes two paradigms relevant to our setting:

**Test-Time Adaptation (TTA):** Methods like TENT (Wang et al., 2021), EATA (Niu et al., 2022), SAR (Niu et al., 2023), and MEMO (Zhang et al., 2022) adapt a pre-trained model at inference time using only the test batch, without source data. These are primarily designed for classification and update normalization statistics or use entropy minimization. **They are NOT suitable for our problem** because:
- They process single images or small batches, not full datasets
- They do not learn persistent adaptations
- Entropy minimization causes collapse in dense prediction

**Target-Domain Self-Supervised Pre-training (TD-SSPT):** This is the right paradigm. We have the entire Cityscapes training set (~3K images) without labels. We can run **unsupervised self-distillation** on this target domain to adapt the feature extractor. The key paper here is **ExPLoRA** (Khanna et al., 2024):

> *"ExPLoRA continues the unsupervised pre-training objective on a new domain, unfreezing 1-2 pre-trained ViT blocks and tuning all other layers with LoRA... up to 8% improvement in linear probing top-1 accuracy on downstream tasks while using <10% of the number of parameters."*

### 2.2 Why This Works for Cityscapes

DINOv2 was pre-trained on LVD-142M (curated natural images). Cityscapes is a **narrow but different domain**: driving scenes, specific camera height/angle, European urban environments. The features need adaptation for:
- **Scale:** DINOv2 sees objects at many scales; Cityscapes has consistent scales (cars, pedestrians at known distances)
- **Semantic categories:** Road, sidewalk, building, vegetation — these are common but have different visual statistics
- **Texture/statistics:** Road textures, building facades, sky appearance differ from generic natural images

### 2.3 ExPLoRA Method Details

ExPLoRA recipe (directly applicable to us):

```
1. Initialize frozen ViT with DINOv2 weights
2. Unfreeze all parameters of subset U of L blocks (usually last 1-2 blocks)
3. Apply LoRA with rank r on Q and V weights in attention of remaining L-U blocks
4. Train on unlabeled target data T using the SAME unsupervised objective as pretraining (DINO/iBOT loss)
5. Result: a new foundation model for domain T
```

**Critical insight:** They use the **original DINO self-distillation loss**, not a downstream task loss. This preserves the feature quality while adapting to the domain.

For our pipeline, we can use:
- The DINOv2 teacher-student framework (if we can reproduce it)
- Or simpler: STEGO-style correspondence distillation (easier to implement)
- Or DepthG-style depth-feature correlation (we have depth already)

### 2.4 DoRA: Weight-Decomposed Low-Rank Adaptation (ICML 2024)

DoRA (Liu et al., 2024, NVIDIA) is the state-of-the-art LoRA variant:

```
Standard LoRA:   W_prime = W_0 + BA
DoRA:            W_prime = m * (W_0 + BA) / ||W_0 + BA||
```

Where:
- `W_0` is frozen pre-trained weight
- `BA` is low-rank update (only to direction)
- `m` is a learned magnitude vector (per output neuron)
- `*` is element-wise multiplication

**Why DoRA is better for our task:**
1. **Decouples magnitude and direction:** Full fine-tuning adjusts both freely; LoRA conflates them. DoRA matches FT more closely.
2. **More stable training:** Directional updates are normalized, preventing feature collapse.
3. **Better on vision tasks:** EVA experiments show DoRA+EVA outperforms LoRA on image classification (VTAB-1K) and decision-making.
4. **No extra inference cost:** Can be merged into W at inference.

**Implementation note:** The `dora_simple=True` flag detaches the normalization from gradients, saving ~24% memory with negligible accuracy loss.

### 2.5 EVA: Explained Variance Adaptation (2024)

EVA (Paischer et al., 2024) improves LoRA initialization:

```
1. Propagate minibatches of target data through frozen model
2. Compute SVD on activation vectors
3. Initialize LoRA A matrix with right-singular vectors (top-r by explained variance)
4. Redistribute ranks across weight matrices based on variance explained
```

**Relevance to us:** EVA is particularly effective when there is a domain shift (e.g., natural images -> driving scenes), because it uses target-domain activations to determine which directions need adaptation. For Cityscapes adaptation, EVA initialization could significantly outperform random LoRA init.

---

## 3. Self-Supervised Training Objectives for Adapter Fine-Tuning

Since we have no GT labels on Cityscapes training set, we need purely self-supervised losses. We organize by semantic features vs. depth.

### 3.1 For Semantic Features (DINOv2 + CAUSE-TR)

#### 3.1.1 Feature Correspondence Consistency (STEGO-style)

STEGO (Hamilton et al., NeurIPS 2022) distills DINO features into a segmentation head using:

```
L_corr(x, x_prime, b) = -Σ_{hwij} (F^DINO_{hwij} - b) * max(S_{hwij}, 0)
```

Where:
- `F^DINO` is DINO patch feature correspondence (cosine similarity)
- `S` is the segmentation head output correspondence
- `b` is a threshold (negative pressure)

**How to use for adapter training:** Instead of training a segmentation head, we train LoRA adapters on DINOv2 such that the **adapted features maintain the correspondence structure** while better fitting the target domain. We can add:
- A consistency loss: adapted features should match original DINO features on easy correspondences
- A domain-alignment loss: adapted features should have better clustering structure on target data

#### 3.1.2 Cross-View Consistency

This is the classic self-supervised approach:
1. Take image I, create two augmented views: T1(I), T2(I)
2. Extract features with adapted model: f1 = F(T1(I)), f2 = F(T2(I))
3. Loss: maximize cosine similarity of corresponding patches between f1 and f2

**Augmentation choices:** Color jitter, Gaussian blur, random crop, horizontal flip. **Avoid** strong geometric distortions that break patch correspondence.

#### 3.1.3 Depth-Feature Correlation (DepthG, CVPR 2024)

This is the **most directly applicable** loss for our pipeline. DepthG (Sick et al., CVPR 2024) proposes:

```
L_DFC = -Σ_{hwij} D_{hwij} * F_{hwij}
```

Where:
- `D` is depth correlation (normalized inverse depth difference)
- `F` is feature correlation
- The loss encourages features to be similar when depth is similar, and different when depth is different

**Key insight:** In 3D space, nearby points are likely the same object; distant points are likely different objects. Depth provides a **geometric prior** for feature learning.

**Our advantage:** We already compute depth maps (DAv3 or DepthPro) for instance pseudo-labels. We can reuse these depth maps to train the semantic feature adapter.

Implementation:
```python
# Depth correlation: small depth difference -> high correlation
depth_corr = exp(-|d_i - d_j| / sigma)

# Feature correlation: cosine similarity
feat_corr = normalize(f_i) @ normalize(f_j).T

# Loss: align them
loss = - (depth_corr * feat_corr).mean()
```

#### 3.1.4 InfoNCE / Contrastive Learning on Patch Features

Mine hard negative patches using:
- **k-NN in feature space:** For each patch, find k nearest patches in the same image (positives) and k farthest (negatives)
- **Cross-image mining:** Use a memory bank of patches from other images
- **Depth-aware sampling:** Use Farthest-Point Sampling on depth maps (as DepthG does) to get diverse spatial coverage

Loss:
```
L_InfoNCE = -log[ exp(f_i * f_j^+ / tau) / Σ_k exp(f_i * f_k / tau) ]
```

#### 3.1.5 Clustering Quality Objectives

**SwAV-style online clustering:**
- Compute cluster assignments (codes) for each patch feature using a set of prototypes
- Enforce consistency between codes of different augmented views
- No need for explicit negative samples

**SCAN-style self-labeling:**
1. Use nearest-neighbor mining to find confident pairs
2. Train with cross-entropy on confident pseudo-labels
3. Gradually increase confidence threshold

**Recommendation for our pipeline:** Combine Depth-Feature Correlation (primary) + Cross-View Consistency (regularization). This gives us:
- A strong geometric signal (depth)
- Invariance to appearance changes (augmentation)
- No need for memory banks or complex clustering

### 3.2 For Depth Estimation (DAv3 / DepthPro)

#### 3.2.1 Self-Distillation from Frozen Teacher

The safest approach: treat the frozen DAv3/DepthPro as a teacher and train adapters to match it, while allowing domain-specific adjustments.

```
L_distill = ||D_adapted - D_frozen||_2
```

This preserves the zero-shot capability while allowing minor domain adjustments. **Always use this as a baseline component.**

#### 3.2.2 Photometric Consistency Across Augmented Views

If we have multi-frame data (video sequences in Cityscapes), we can use:

```
L_photo = alpha*(1 - SSIM(I_t, I_{s->t})) + (1-alpha)*||I_t - I_{s->t}||_1
```

Where `I_{s->t}` is frame s warped to t using predicted depth and known ego-motion.

**Issue:** Cityscapes has static frames, moving objects, and occlusion. Photometric consistency alone is noisy. Combine with:
- Auto-masking static pixels (Monodepth2)
- Minimum reprojection loss (handle occlusions)
- Edge-aware smoothness regularization

#### 3.2.3 Scale-Invariant Depth Consistency

The SAG loss (Wang et al., ICCV 2021) enforces scale consistency:

```
L_SAG = ||(s_t R P_t + T) - P_s||_2
```

Where relative scale factor `s_t` is explicitly estimated. This prevents per-frame scale drift.

**For our single-image setting:** Use scale-invariant loss on pairs:
```
L_SI = Var(log D) + (mean(log D) - log D_true)^2  # if we had GT
L_SI_proxy = ||log D_i - log D_j - log(d_i/d_j)||_2  # relative consistency
```

#### 3.2.4 Feature-Depth Alignment

Align depth features with semantic features from the (adapted) DINOv2:

```
L_align = -cosine_sim(F_semantic, F_depth)
```

This encourages the depth model to use semantic cues (e.g., road surfaces are flat, buildings are vertical) and vice versa.

#### 3.2.5 Relative Depth Ranking Loss

Use ordinal depth relationships:
- Sample pairs of pixels (i, j)
- If d_i < d_j (closer), enforce D_i < D_j
- Loss: hinge loss on depth ordering

This is robust to absolute scale ambiguity and works with relative depth models like DAv3.

#### 3.2.6 Depth Anything V3 Specifics

DAv3 is already trained on 62M+ unlabeled images and has strong generalization. For adaptation:
- **Freeze encoder, adapt decoder:** DAv3 DPT decoder can be fine-tuned with ranking losses
- **LoRA on encoder:** If using LoRA, apply to the DINOv2 backbone inside DAv3 (it uses DINOv2 as encoder)
- **Metric fine-tuning:** DAv3 v2 has metric depth variants; if using those, add scale-invariant losses

#### 3.2.7 DepthPro Specifics

DepthPro (Apple, 2024) produces metric depth at native resolution. Key properties:
- Uses multi-scale ViT + DPT decoder
- Trained with Scale-and-Shift Invariant (SSI) loss
- Strong zero-shot, but domain bias exists (trained on diverse real + synthetic data)

**Adaptation strategy:**
- LoRA on the multi-scale encoder features
- Freeze the focal-length estimation head (not relevant for driving)
- Fine-tune decoder with SSI loss on target domain
- Self-distillation from frozen teacher as regularization

---

## 4. Parameter-Efficient Fine-Tuning Strategies

### 4.1 Where to Place Adapters

#### Vision Transformer Layers

Research consensus (ExPLoRA, Rein, Conv-LoRA ablations):

| Layer Type | Adapt? | Rationale |
|-----------|--------|-----------|
| **Early layers (0-3)** | Minimal | Low-level edges, textures; DINOv2 already generalizes well |
| **Mid layers (4-8)** | Moderate | Object parts, mid-level features; some domain adaptation helps |
| **Late layers (9-11)** | **Heavy** | High-level semantics; most domain-specific adaptation needed |
| **Q, V projections** | **Yes** | Query and Value are most important for attention patterns |
| **K projection** | Optional | Less critical; can skip to save parameters |
| **FFN/MLP (fc1, fc2)** | Yes | Adds non-linear adaptation capacity |
| **Output projection** | Yes | Final feature transformation |

**Recommended config for ViT-B/14 (12 blocks):**
```yaml
lora_target_modules: ["qkv", "proj", "fc1", "fc2"]
late_block_start: 6      # Only adapt blocks 6-11
unfrozen_blocks: [11]    # Fully unfreeze last block (ExPLoRA style)
lora_rank: 8
lora_alpha: 8
lora_dropout: 0.05
```

This yields ~2-4M trainable parameters out of 86M (2-5%).

### 4.2 Rank Selection

| Rank | Parameters (ViT-B) | Use Case |
|------|-------------------|----------|
| r=4 | ~1M | Very constrained; good if label noise is high |
| r=8 | ~2M | **Sweet spot** for domain adaptation |
| r=16 | ~4M | Stronger adaptation; risk of overfitting |
| r=32 | ~8M | Too many params for noisy pseudo-label setting |

**Evidence:**
- ExPLoRA uses r=8-16 for satellite/medical adaptation
- Rein++ uses r=16 for DGSS and finds it optimal
- Our own Stage-2/3 experiments show r=32 overfits (Conv-DoRA degraded PQ)

**Rule of thumb:** `alpha = rank` or `alpha = 2 * rank`. Higher alpha increases adapter influence.

### 4.3 DoRA vs. LoRA vs. Conv-LoRA

| Method | Params | Best For | Our Recommendation |
|--------|--------|----------|-------------------|
| **Plain LoRA** | Lowest | General adaptation | Baseline |
| **DoRA** | Same as LoRA | Stability, vision tasks | **Primary choice** |
| **Conv-LoRA** | +conv params | Supervised dense prediction | **Avoid** — clashes with global attention |
| **Rein** | ~2.5M tokens | Domain generalization | Interesting alternative |

**Why not Conv-LoRA for Stage-1:**
- Our Stage-2/3 analysis shows Conv-DoRA destroys stuff performance when labels are noisy
- Stage-1 feature extraction needs **global context** (roads, sky, buildings are spatially extended)
- 3x3 conv constrains adaptation to local neighborhoods
- Use **plain DoRA** instead

### 4.4 Initialization Strategies

| Method | How | When to Use |
|--------|-----|-------------|
| **Kaiming init** | Default LoRA (A=kaiming, B=zero) | Baseline |
| **EVA** | SVD on target activations | **Strongly recommended** for domain shift |
| **PiSSA** | SVD on pre-trained weights, freeze principal components | Preserves pre-trained knowledge |
| **OPLoRA** | Orthogonal projection to null space of pre-trained weights | Prevents catastrophic forgetting |

**Recommendation:** Try EVA init first (data-driven, domain-aware). If unstable, fall back to Kaiming.

### 4.5 Dropout and Regularization

- **LoRA dropout:** 0.0-0.1. Use 0.05 for regularization.
- **Weight decay:** 0.01 on adapters (not on backbone)
- **Learning rate:** 1e-4 to 5e-4 for adapters; 10x lower for unfrozen blocks
- **Warmup:** 500 steps linear warmup
- **Schedule:** Cosine decay over 10-20K steps

---

## 5. Evaluation Without Ground Truth

### 5.1 Clustering Metrics

Since semantic pseudo-labels come from k-means clustering, evaluate clustering quality directly:

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Silhouette Score** | Cohesion vs. separation of clusters | Higher is better (range [-1, 1]) |
| **Davies-Bouldin Index** | Ratio of within-cluster to between-cluster distance | Lower is better |
| **Calinski-Harabasz Index** | Ratio of between-cluster to within-cluster dispersion | Higher is better |
| **Cluster Purity** | % of dominant class in each cluster (needs GT for true eval) | — |

**Implementation:** Run k-means (K=80) on adapted features, compute metrics on validation set.

### 5.2 Feature Quality Metrics

| Metric | What It Measures | How to Compute |
|--------|-----------------|---------------|
| **CKA (Centered Kernel Alignment)** | Similarity between layer representations | CKA(adapted_layer_i, frozen_layer_i) — should be high for early layers, can diverge for late layers |
| **Feature diversity** | Average pairwise distance of patch features | Higher = more discriminative |
| **Rank of feature matrix** | Effective dimensionality | Should not collapse (rank >> 1) |

### 5.3 Proxy Tasks

| Proxy Task | Signal | How |
|-----------|--------|-----|
| **Depth prediction consistency** | Adapted semantic features should correlate with depth | Compute Depth-Feature Correlation loss on held-out set |
| **Semantic consistency across augments** | Features should be invariant | Extract features from I and T(I), measure cosine similarity |
| **k-NN retrieval accuracy** | Features should retrieve semantically similar patches | Manual inspection or weakly-supervised evaluation |

### 5.4 Downstream Validation (The Gold Standard)

**The only true evaluation:** Generate pseudo-labels with adapted Stage-1, train Stage-2, measure PQ.

```
Adapt Stage-1 -> Generate pseudo-labels -> Train Stage-2 -> Evaluate PQ
```

This is expensive but necessary. To reduce cost:
- Train Stage-2 for fewer epochs (e.g., 20K steps instead of 40K)
- Use smaller model (ResNet-50 instead of DINOv3 ViT-B)
- Evaluate on subset of validation set

### 5.5 Sanity Checks

1. **Feature visualization:** PCA of patch features should show coherent semantic regions
2. **Attention rollout:** Attention maps should focus on meaningful objects
3. **Depth alignment:** Scatter plot of feature similarity vs. depth difference should show negative correlation
4. **No collapse check:** All clusters should have >1% of pixels (prevent dead clusters)

---

## 6. Specific Considerations for Our Pipeline

### 6.1 DINOv2 vs. DINOv3

| Aspect | DINOv2 | DINOv3 |
|--------|--------|--------|
| **Pretraining data** | LVD-142M (~142M images) | 1.7B images (10x larger) |
| **Architecture** | ViT-S/B/L/g with registers | ViT-S/S+/B/L/H+ |
| **Proven with CAUSE-TR** | **Yes** — CAUSE paper uses DINOv2 | Not yet validated |
| **Our current pipeline** | Used in Stage 1 | Used in Stage 2 backbone |
| **Dense prediction** | Excellent features | Potentially better (more data) |
| **Adaptation risk** | Lower (well-understood) | Higher (less tested for unsupervised seg) |

**Recommendation:** Adapt **DINOv2** for Stage-1 semantic generation. It is proven with CAUSE-TR, and we know its failure modes (SINDER defects, register tokens). DINOv3 can remain the Stage-2 backbone.

**If experimenting with DINOv3 for Stage-1:** Use SINDER repair first (smooth regularization on singular values) to fix patch token artifacts.

### 6.2 CAUSE-TR Head: Adapt or Replace?

The CAUSE-TR (Segment_TR) head is already trained on ImageNet-kmeans and produces 90-dim features that are then clustered.

**Options:**
1. **Freeze CAUSE-TR, adapt DINOv2 only:** Safest. CAUSE-TR is a lightweight projection; adapting DINOv2 features upstream is sufficient.
2. **Adapt both DINOv2 + CAUSE-TR:** More flexible but risk of collapse. CAUSE-TR has fewer params, so LoRA on it is cheap.
3. **Replace CAUSE-TR with STEGO head:** STEGO correspondence loss might produce better features for clustering. But this deviates from proven pipeline.

**Recommendation:** Start with option 1 (adapt DINOv2 only, freeze CAUSE-TR). If underperforming, try option 2 with very small rank (r=4) on CAUSE-TR.

### 6.3 DepthPro vs. Depth Anything V3

| Aspect | DepthPro | DAv3 |
|--------|----------|------|
| **Output** | Metric depth (absolute scale) | Relative depth (or metric with fine-tuning) |
| **Resolution** | Native (up to 2.25MP) | Standard (up to 518x518) |
| **Sharpness** | Very sharp boundaries | Good, slightly softer |
| **Zero-shot on Cityscapes** | Strong | Strong |
| **Adaptability** | Harder (Apple license, less open) | **Easier** (open source, HuggingFace) |
| **Encoder** | Multi-scale ViT (BEiT) | DINOv2 ViT |
| **Instance masks** | Good gradient -> CC | Good gradient -> CC |

**Recommendation:**
- **Primary:** DAv3 for pseudo-label generation (easier to adapt, same DINOv2 encoder as semantic branch)
- **Secondary experiment:** DepthPro if metric scale is critical for instance separation
- **Adaptation:** DAv3 encoder IS DINOv2 — we can share the adapted weights between semantic and depth branches!

### 6.4 Preventing Catastrophic Forgetting

Key techniques:
1. **Freeze backbone + adapters:** LoRA/DoRA naturally prevent forgetting by construction
2. **Self-distillation loss:** `L = L_domain + lambda*L_distill` where `L_distill` matches frozen teacher
3. **OPLoRA (2025):** Project LoRA updates orthogonal to top singular vectors of pre-trained weights
4. **Small learning rate:** 1e-4 or lower for adapter updates
5. **Early stopping:** Monitor feature quality; stop if CKA drops too much

### 6.5 Joint vs. Sequential Training

**Sequential (Recommended):**
```
Phase 1: Train semantic adapter (DINOv2 + CAUSE-TR)
         Loss: Depth-Feature Correlation + Cross-View Consistency
         Duration: 10K steps

Phase 2: Train depth adapter (DAv3)
         Loss: Self-distillation + Photometric consistency + Feature-Depth Alignment (using adapted semantic features from Phase 1)
         Duration: 10K steps

Phase 3 (Optional): Joint fine-tuning
         Loss: Combine all objectives with small weights
         Duration: 5K steps
```

**Why sequential?**
- Semantic features are more stable to train first
- Depth adaptation can leverage improved semantic features
- Prevents objective competition early in training
- Easier to debug failure modes

---

## 7. Related Work Review

### 7.1 Conv-LoRA / Conv-DoRA for Dense Prediction

**Paper:** "Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model" (ICLR 2024)

- Proposes adding 3x3 depthwise conv inside LoRA bottleneck for SAM adaptation
- Shows gains on medical, remote sensing, and natural image segmentation
- **Critical caveat:** All experiments use **supervised fine-tuning with GT labels**
- Their ViT lacks the global attention quality of DINOv2/DINOv3
- **Our finding:** Conv-DoRA destroys stuff performance when labels are noisy (see `why_conv_dora_not_plain_lora.md`)

### 7.2 Test-Time Adaptation (TTA)

**Key papers:**
- **TENT** (Wang et al., 2021): Entropy minimization on BN layers
- **EATA** (Niu et al., 2022): Sample filtering + Fisher regularizer
- **SAR** (Niu et al., 2023): Sharpness-aware adaptation, removes high-gradient samples
- **MEMO** (Zhang et al., 2022): Marginal entropy minimization across augmentations

**Verdict:** Not directly applicable. TTA is for inference-time batch adaptation; we need persistent domain adaptation on the full training set.

### 7.3 Adapter-Based Domain Adaptation for Vision

**Rein / Rein++ (CVPR 2024 / T-PAMI 2025):**
- Introduces learnable "Rein tokens" that refine features within each ViT layer
- Rein++ adds domain adaptation capability (Rein-A) with logit/instance alignment
- Achieves 78.2% mIoU on GTAV->Cityscapes domain adaptation
- Uses only **1% extra parameters** vs. full fine-tuning
- **Key insight:** Fewer trainable parameters -> better generalization (inverted U-curve)

**Relevance:** Rein++ proves that parameter-efficient adaptation of VFMs for domain adaptation works. Their recipe: VFM backbone + Rein tokens + Mask2Former head. Our pipeline is similar but unsupervised.

### 7.4 DINOv2 Fine-Tuning Protocols

**SINDER (ECCV 2024):**
- Identifies "singular defects" in DINOv2 patch tokens (artifacts from leading singular vector)
- Proposes smooth regularization fine-tuning on small dataset (~30K ImageNet images)
- Improves unsupervised segmentation mIoU by +2-4% on Cityscapes
- **Actionable:** Before adapting DINOv2, consider SINDER repair to get cleaner features

**ExPLoRA (2024):**
- Extends DINOv2 pre-training to new domains with LoRA
- Unfreezes last 1-2 blocks, LoRA on rest
- Uses original DINO loss on unlabeled target data
- **Actionable:** This is our primary methodological template

### 7.5 Self-Supervised Fine-Tuning of DINO Features

**DepthG (CVPR 2024):**
- State-of-the-art unsupervised semantic segmentation on Cityscapes (23.1% mIoU with DINO-B/8)
- Uses Depth-Feature Correlation loss + Farthest-Point Sampling
- Does NOT adapt DINOv2 backbone (frozen)
- **Our extension:** Add LoRA adapters to DINOv2 and train with DepthG-style loss — could push mIoU even higher

**EAGLE (CVPR 2024):**
- Object-centric unsupervised semantic segmentation
- Uses eigenbasis of feature similarity matrix for clustering
- 22.1% mIoU on Cityscapes with frozen DINO-B/8
- **Actionable:** EiCue clustering could replace k-means in our pipeline

**GASeg (2025):**
- Latest SOTA: 23.2% mIoU on Cityscapes with frozen DINOv2-B/8
- Uses topological features + STEGO-style loss
- **Benchmark:** Our adapted features should aim to beat 23.2% mIoU

### 7.6 CAUSE Paper

The CAUSE paper (the foundation of our semantic pipeline) uses:
- Frozen DINOv2 ViT-B/14 backbone
- CAUSE-TR (Segment_TR) head trained on ImageNet-kmeans
- k-means K=80 clustering on 90-dim features
- Achieves ~31-35% mIoU on Cityscapes (with CRF)

**Fine-tuning protocol:** The CAUSE paper does NOT describe fine-tuning DINOv2 on target domains. This is our novel contribution.

---

## 8. Proposed Experimental Protocol

### Phase 0: Infrastructure (1-2 days)

1. **Implement plain DoRA adapter** for DINOv2 ViT-B/14
   - Target modules: qkv, proj, fc1, fc2
   - Rank r=8, alpha=8, dropout=0.05
   - Late blocks only: blocks 6-11
   - Unfreeze block 11 fully

2. **Implement Depth-Feature Correlation loss**
   - Use existing depth maps (DAv3 or DepthPro)
   - FPS sampling for feature pairs
   - Normalize depth correlations to [0, 1]

3. **Implement cross-view consistency**
   - Color jitter, blur, random crop, horizontal flip
   - Cosine similarity on corresponding patch features

### Phase 1: Semantic Adapter Training (3-5 days)

**Config:**
```yaml
model: DINOv2 ViT-B/14 + CAUSE-TR (frozen)
adapter: DoRA, r=8, blocks 6-11
optimizer: AdamW, lr=2e-4 for adapters, lr=2e-5 for block 11
schedule: Cosine, 10K steps, warmup 500
batch_size: 16
resolution: 518x518 (DINOv2 native)
```

**Loss:**
```
L_total = lambda_dfc * L_depth_feature_corr + lambda_cv * L_cross_view + lambda_distill * L_distill

Recommended: lambda_dfc = 1.0, lambda_cv = 0.5, lambda_distill = 0.1
```

**Data:** Cityscapes train set (~3K images), no labels.

**Evaluation:**
- Every 2K steps: run k-means K=80, compute clustering metrics
- Every 2K steps: generate pseudo-labels, run quick Stage-2 (20K steps, ResNet-50), measure PQ

**Expected outcome:** Clustering mIoU should improve from ~21% (frozen) to ~24-26% (adapted).

### Phase 2: Depth Adapter Training (2-3 days)

**Config:**
```yaml
model: Depth Anything V3 (DINOv2 ViT-B encoder)
adapter: DoRA on encoder, r=8
optimizer: AdamW, lr=1e-4
schedule: Cosine, 10K steps
```

**Loss:**
```
L_total = lambda_distill * ||D_adapted - D_frozen|| + lambda_rank * L_ranking + lambda_align * L_feature_align

Recommended: lambda_distill = 1.0, lambda_rank = 0.5, lambda_align = 0.3
```

**Feature alignment:** Use adapted semantic features from Phase 1.

**Evaluation:**
- Depth boundary consistency (gradient alignment with image edges)
- Instance mask quality from connected components
- Stage-2 PQ with adapted depth pseudo-labels

### Phase 3: Joint Refinement (Optional, 2-3 days)

If Phase 1 and 2 both show gains:
```yaml
Train both adapters jointly with small learning rates
Loss: weighted combination of all objectives
Duration: 5K steps
```

### Phase 4: Full Pipeline Evaluation (1 week)

1. Generate adapted pseudo-labels for full Cityscapes train
2. Train Stage-2 (Cascade Mask R-CNN + DINOv3 ViT-B) with adapted labels
3. Train Stage-3 (EMA self-training) with frozen backbone
4. Compare PQ against baseline (frozen Stage-1)

**Target improvement:** +2-5 PQ points from better pseudo-labels alone.

### Ablation Studies

| Experiment | What to Vary | Baseline | Expected Comparison |
|-----------|-------------|----------|-------------------|
| A1 | Adapter type | Plain DoRA | vs. LoRA vs. Conv-DoRA |
| A2 | Rank | r=8 | vs. r=4, r=16, r=32 |
| A3 | Block scope | Blocks 6-11 | vs. all blocks vs. only block 11 |
| A4 | Loss components | Full | ablate DFC, cross-view, distillation |
| A5 | Sequential vs. joint | Sequential | vs. joint training |
| A6 | Depth model | DAv3 | vs. DepthPro vs. frozen |
| A7 | DINO version | DINOv2 | vs. DINOv3 (with SINDER repair) |

### Test-Time Adaptation

```bibtex
@inproceedings{wang2021tent,
  title={Tent: Fully Test-Time Adaptation by Entropy Minimization},
  author={Wang, Dequan and Shelhamer, Evan and Liu, Shaoteng and Olshausen, Bruno and Darrell, Trevor},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{niu2022eata,
  title={Efficient Test-Time Adaptation with Entropy-Aware Sample Selection},
  author={Niu, Shuaicheng and Wu, Jiaxiang and Zhang, Yifan and Chen, Yao and Zheng, Shijian and Zhao, Peilin and Tan, Mingkui},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{niu2023sar,
  title={Towards Stable Test-Time Adaptation in Dynamic Wild World},
  author={Niu, Shuaicheng and Wu, Jiaxiang and Zhang, Yifan and Wen, Yao and Chen, Yao and Zhao, Peilin and Tan, Mingkui},
  booktitle={ICLR},
  year={2023}
}
```

### Self-Supervised Depth

```bibtex
@inproceedings{wang2021scale,
  title={Can Scale-Consistent Monocular Depth Be Learned in a Self-Supervised Scale-Invariant Manner?},
  author={Wang, Yiran and Yang, Zihang and Wang, Zhaowen and Wang, Yang and Yang, Yang and Zhao, Heng},
  booktitle={ICCV},
  year={2021}
}

@article{li2024depthpro,
  title={Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
  author={Li, Fabian and Singh, Manmohan and Bleyer, Michael and Bhardwaj, Chirag and Sajjadi, Mehdi and Kehrem, Prashanth and Hui, Trevor and Sharifi, Mohammad and Tanno, Ryutaro and Reig, Andrew},
  journal={arXiv:2410.02073},
  year={2024}
}

@article{yang2024depthanything,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}
```

### Unsupervised Semantic Segmentation

```bibtex
@inproceedings{kim2024eagle,
  title={EAGLE: Eigen Aggregation Learning for Object-Centric Unsupervised Semantic Segmentation},
  author={Kim, Chanyoung and Jeong, Joonmyung and Kim, Dongyoon and Roh, Byungseok and Chun, Sangdoo},
  booktitle={CVPR},
  year={2024}
}

@article{hahn2024primaps,
  title={Boosting Unsupervised Semantic Segmentation with Principal Mask Proposals},
  author={Hahn, Oliver and Araslanov, Nikita and Schaub-Meyer, Simone and Roth, Stefan},
  journal={TMLR},
  year={2024}
}

@article{gaseg2025,
  title={Topological Features for Robust Self-Supervised Segmentation},
  author={Authors},
  journal={arXiv:2512.23997},
  year={2025}
}
```

### Conv-LoRA for Dense Prediction

```bibtex
@inproceedings{yu2024convlora,
  title={Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model},
  author={Yu, Jie and others},
  booktitle={ICLR},
  year={2024}
}
```

---

*Document compiled from extensive literature review across NeurIPS, ICML, CVPR, ICCV, ECCV, and ICLR papers (2021-2025). Focus: parameter-efficient fine-tuning, domain adaptation, self-supervised learning, and unsupervised semantic segmentation.*

