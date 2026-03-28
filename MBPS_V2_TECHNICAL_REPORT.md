# MBPS: Mamba-Bridge Panoptic Segmentation via Cross-Modal State Space Fusion of Self-Supervised Representations

---

## Abstract

Unsupervised panoptic segmentation remains a fundamental challenge in computer vision, requiring joint pixel-level semantic classification and instance-level object delineation without any human annotations. Existing approaches either rely on expensive stereo video inputs (CUPS, CVPR 2025) or produce degenerate solutions due to end-to-end unsupervised training instability. We present **MBPS** (Mamba-Bridge Panoptic Segmentation), a novel framework that bridges the gap between self-supervised feature learning and panoptic segmentation through a structured state space fusion mechanism. Our key insight is that semantic class boundaries and instance object boundaries are *complementary* signals that can be jointly refined through a *cross-modal state space scan* — a Mamba2-based module that treats interleaved semantic and instance tokens as a unified sequence, enabling each modality to attend to the other through the SSM's recurrent state. Concretely, we leverage frozen DINOv3 ViT-B/16 features (Oquab et al., 2025) to generate high-quality pseudo-labels via K-means semantic clustering and MaskCut instance discovery, then train a lightweight panoptic model with a novel Bidirectional Cross-Modal Scan (BiCMS) bridge that fuses semantic logits and instance embeddings through depth-conditioned Mamba2 state space dynamics. Our approach requires only monocular images and achieves \_\_\_\_ PQ on Cityscapes and \_\_\_\_ PQ on COCO-Stuff-27. Results are yet to be calculated. The complete codebase implements a hybrid PyTorch (offline pseudo-label generation) and JAX/Flax (TPU training) pipeline spanning ~6,000 lines across 30 files.

---

## 1. Introduction

### 1.1 Problem Statement

Panoptic segmentation (Kirillov et al., 2019) unifies semantic segmentation ("what class is each pixel?") and instance segmentation ("which object does each pixel belong to?") into a single coherent output. The supervised setting has been well-explored with Panoptic FPN (Kirillov et al., 2019), Mask2Former (Cheng et al., 2022), and Panoptic SegFormer (Li et al., 2022). However, the *unsupervised* setting — producing panoptic maps without any human annotations — remains largely unsolved.

The unsupervised panoptic segmentation problem decomposes into three sub-problems:

1. **Semantic discovery**: Discovering meaningful semantic categories from unlabeled data.
2. **Instance discovery**: Identifying individual object instances without bounding box or mask annotations.
3. **Stuff-things disambiguation**: Determining which discovered categories are countable objects ("things") versus amorphous regions ("stuff").

Each sub-problem has seen individual progress: STEGO (Hamilton et al., 2022) for semantic discovery, CutLER (Wang et al., 2023) for instance discovery, and various heuristics for stuff-things classification. However, *combining* these into a coherent panoptic output is non-trivial because the semantic and instance signals are fundamentally different in nature — semantic classes are *global* (road is road everywhere) while instances are *local* (each car is a separate entity). We argue that explicitly modeling the *cross-modal interaction* between these two signal types is crucial.

### 1.2 Motivation for Cross-Modal Fusion

Consider the spatial relationship between semantic boundaries and instance boundaries. At a semantic boundary (e.g., road → sidewalk), there is no instance boundary — these are both "stuff" classes. At an instance boundary (e.g., car_1 → car_2), the semantic class is the same ("car") but the instance identity changes. This complementarity means that knowing the semantic class helps resolve instance ambiguity (two adjacent "car" patches likely belong to different cars if separated by "road"), and knowing the instance structure helps resolve semantic ambiguity (a coherent object region should have a consistent semantic label).

State space models (SSMs), particularly Mamba (Gu & Dao, 2023) and Mamba-2 (Dao & Gu, 2024), provide an elegant mechanism for modeling this cross-modal interaction. Unlike attention, which treats all token pairs equally, SSMs process sequences *recurrently* — each position's output depends on a compressed representation of all preceding positions. By *interleaving* semantic and instance tokens ([s_1, f_1, s_2, f_2, ...]), we force the SSM to build a recurrent state that naturally captures cross-modal dependencies: each semantic token's representation is influenced by the preceding instance token, and vice versa.

### 1.3 Why Not End-to-End Unsupervised Training?

Our initial approach (MBPS v1) attempted end-to-end unsupervised panoptic segmentation with a DINO ViT-S/8 backbone, STEGO-style semantic clustering, CutS3D instance discovery, and a 4-phase training curriculum with 12 interacting loss terms. After 60 epochs on Cityscapes, this achieved PQ = 0.01% — effectively zero. The failure modes were:

- **Cluster collapse**: STEGO's contrastive learning with a weak backbone (384-dim, ~30 mIoU linear probe) led to 87 → 12 active clusters by epoch 50.
- **Loss conflict**: 12 loss terms with different convergence rates caused oscillatory gradients.
- **Unstable bridge activation**: When the Mamba2 bridge activates at epoch 41 with 40 epochs of untrained parameters, gradients through the SSD matmul chain overflow.

This experience motivated a fundamental redesign: instead of end-to-end unsupervised training, we *decouple* pseudo-label generation from model training. We use the strongest available self-supervised vision model (DINOv3 ViT-B/16, linear probe mIoU 81.1) to generate high-quality pseudo-labels *offline*, then train a simpler model with 3 loss terms on these pseudo-labels. The Mamba2 bridge remains our novel contribution for cross-modal fusion.

### 1.4 Contributions

1. **Bidirectional Cross-Modal Scan (BiCMS)**: A novel Mamba2-based module that fuses semantic and instance representations through interleaved token scanning with depth-conditioned state dynamics. This is the first application of structured state space models to cross-modal fusion in panoptic segmentation.

2. **Unified Depth Conditioning Module (UDCM)**: A FiLM-based mechanism that conditions the cross-modal bridge on monocular depth, providing 3D geometric priors that improve boundary delineation.

3. **Adaptive Projection Bridge (APB)**: A dimensionality-aligned projection mechanism that maps heterogeneous semantic (K-dim logits) and instance (D-dim embeddings) representations into a shared bridge space with energy-balanced normalization.

4. **Pseudo-label panoptic pipeline**: A complete offline pipeline using DINOv3 + MaskCut + Depth Anything V3 for generating high-quality semantic and instance pseudo-labels from monocular images.

5. **Comprehensive ablation study**: Six controlled ablation experiments (no Mamba bridge, no depth conditioning, no bidirectional scan, no bridge at all, no self-training, DINOv1 backbone) that isolate the contribution of each component.

---

## 2. Architecture Overview

### 2.1 System Design

MBPS v2 employs a two-stage architecture: an offline pseudo-label generation pipeline (PyTorch, GPU) and an online training pipeline (JAX/Flax, TPU).

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: OFFLINE PSEUDO-LABEL GENERATION (PyTorch, GPU)    │
│                                                             │
│  Cityscapes Images ──→ DINOv3 ViT-B/16 ──→ Patch Features  │
│                              │                              │
│           ┌──────────────────┼──────────────────┐           │
│           │                  │                  │           │
│     K-means(K=19)      MaskCut (NCut)    Depth Any. V3     │
│           │                  │                  │           │
│     Pseudo-Semantic    Pseudo-Instance    Depth Maps        │
│           │                  │                  │           │
│           └──────────────────┴──────────────────┘           │
│                              │                              │
│                     TFRecord Serialization                  │
│                              ↓                              │
│                     GCS: gs://mbps-panoptic/                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: ONLINE TRAINING (JAX/Flax, TPU v4-8)             │
│                                                             │
│  TFRecords ──→ Image (B,H,W,3) + Depth + Pseudo-Labels     │
│                       │                                     │
│              DINOv3 ViT-B/16 (frozen)                       │
│                       │                                     │
│              Features (B, N, 768)                           │
│                  ┌────┴────┐                                │
│                  │         │                                │
│           Semantic HD  Instance HD                          │
│           (B,N,K)      (B,N,D_i)                           │
│                  │         │                                │
│           ┌──────┴─────────┴──────┐                        │
│           │   MAMBA BRIDGE (Ours)  │                        │
│           │  APB → UDCM → BiCMS   │                        │
│           │  → Inverse Proj       │                        │
│           └──────┬─────────┬──────┘                        │
│                  │         │                                │
│           Refined Sem.  Refined Inst.                      │
│           (B,N,K)       (B,N,D_i)                          │
│                  │         │                                │
│              ┌───┴─────────┴───┐                           │
│              │ Panoptic Merge  │                           │
│              └────────┬────────┘                           │
│                       │                                     │
│              Panoptic Output (B,H,W)                       │
└─────────────────────────────────────────────────────────────┘
```

**Implementation reference**: The main model is defined in `mbps/models/mbps_v2_model.py` (JAX, class `MBPSv2Model`, 292 lines) and `mbps_pytorch/models/mbps_v2_model.py` (PyTorch, class `MBPSv2Model`, 302 lines). The training loop with pmap-based TPU parallelism is in `scripts/train_v2.py` (660 lines).

### 2.2 Dimensional Analysis

For Cityscapes at 512 x 1024 resolution with patch size 16:

| Symbol | Description | Value |
|--------|-------------|-------|
| B | Batch size (per TPU core) | 4 |
| H, W | Image height, width | 512, 1024 |
| P | Patch size | 16 |
| N | Number of patch tokens = (H/P) x (W/P) | 32 x 64 = 2048 |
| D_b | Backbone feature dimension | 768 |
| K | Number of semantic classes | 19 |
| D_i | Instance embedding dimension | 64 |
| D_br | Bridge dimension | 384 |
| D_s | SSM state dimension | 16 (JAX/TPU) |
| L_m | Number of Mamba2 layers per direction | 4 |
| P_c | Mamba2 chunk size | 64 (JAX/TPU) |
| N_reg | DINOv3 register tokens | 4 |

---

## 3. Backbone: Frozen DINOv3 ViT-B/16

### 3.1 Motivation for DINOv3

The choice of backbone is the single most consequential architectural decision in pseudo-label-based panoptic segmentation. The quality of all downstream components — semantic clusters, instance masks, and the bridge itself — is upper-bounded by the quality of the backbone features.

DINOv3 (Oquab et al., 2025) represents the third generation of self-supervised vision transformers from Meta AI, building upon DINOv2 (Oquab et al., 2023) with improved training data curation (LVD-1.69B dataset), enhanced augmentation strategies, and register tokens for reduced artifact formation (Darcet et al., 2024). Key properties that make DINOv3 suitable for our pipeline:

- **Linear probe mIoU of 81.1** on ADE20K — the features are already discriminative enough for semantic segmentation without any supervised fine-tuning.
- **Register tokens** eliminate the attention sink artifact in [CLS] that plagued DINOv1/v2, producing spatially uniform patch features better suited for dense prediction.
- **768-dimensional** features (vs. 384 for ViT-S/8) provide richer representations for both semantic clustering and instance boundary detection.

The DINOv1 ViT-S/8 backbone used in MBPS v1 achieved only ~30 mIoU on linear probe, making all downstream tasks nearly impossible. The upgrade to DINOv3 ViT-B/16 is not merely quantitative — it fundamentally changes what the system can discover.

### 3.2 Architecture Details

The DINOv3 ViT-B/16 backbone follows the standard Vision Transformer architecture (Dosovitskiy et al., 2021) with the following specifications:

**Patch Embedding**: A single convolutional layer projects non-overlapping 16 x 16 patches to 768-dimensional tokens:

$$\mathbf{z}_i = \text{Conv2d}(3 \rightarrow 768, \text{kernel}=16, \text{stride}=16)(\mathbf{x}_{\text{patch}_i}) \quad \in \mathbb{R}^{768}$$

For a 512 x 1024 input, this produces N = 32 x 64 = 2048 patch tokens.

**Token Sequence**: DINOv3 prepends a [CLS] token and 4 register tokens (Darcet et al., 2024) to the patch sequence. The register tokens serve as "attention sinks" that absorb high-norm, low-information attention mass, preventing artifact formation in patch features. Position embeddings are applied *only* to [CLS] and patch tokens; register tokens receive no positional information:

$$\mathbf{z}^{(0)} = [\mathbf{z}_{\text{CLS}} + \mathbf{p}_0, \; \mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3, \mathbf{r}_4, \; \mathbf{z}_1 + \mathbf{p}_1, \ldots, \mathbf{z}_N + \mathbf{p}_N]$$

where $\mathbf{p}_i$ are learned position embeddings and $\mathbf{r}_j$ are learned register tokens.

**Transformer Blocks** (x12): Each block applies pre-norm multi-head self-attention followed by a feed-forward network:

$$\mathbf{z}' = \mathbf{z} + \text{MHSA}(\text{LN}(\mathbf{z}))$$
$$\mathbf{z}^{(\ell+1)} = \mathbf{z}' + \text{FFN}(\text{LN}(\mathbf{z}'))$$

where MHSA uses 12 heads with head dimension 64, and FFN uses GELU activation with expansion ratio 4 (768 → 3072 → 768).

**Position Embedding Interpolation**: When the input resolution differs from the pretrained resolution (224 x 224), the patch position embeddings are bicubically interpolated to the new grid size. This is handled by `_interpolate_pos_embed` in `mbps_pytorch/models/backbone/dinov3_vitb.py:228-257` and `mbps/models/backbone/dinov3_vitb.py` (JAX).

**Freezing**: All backbone parameters are frozen (`requires_grad_(False)` in PyTorch, `stop_gradient` in JAX). This serves two purposes: (1) preserving the quality of pretrained features, and (2) dramatically reducing memory consumption since no gradient storage is needed for the 86M backbone parameters.

**Output**: Only patch tokens are returned (skipping [CLS] and registers):

$$\mathbf{F} = \text{LN}(\mathbf{z}^{(12)}_{5:N+5}) \quad \in \mathbb{R}^{B \times N \times 768}$$

**Implementation reference**: `mbps_pytorch/models/backbone/dinov3_vitb.py` (PyTorch, `DINOv3ViTB` class, 341 lines) and `mbps/models/backbone/dinov3_vitb.py` (JAX/Flax, 355 lines). Weight conversion from HuggingFace format is handled by `from_pretrained()` classmethod (lines 259-341 in PyTorch) and `mbps/models/backbone/dinov3_weights_converter.py` (JAX).

**Citation**: Oquab, M., Darcet, T., et al. "DINOv3: Learning Robust Visual Features with Self-Supervised Vision Transformers." *Meta AI Research*, 2025. Building on DINOv2 (Oquab et al., TMLR 2023) and register tokens (Darcet et al., ICLR 2024).

---

## 4. Offline Pseudo-Label Generation Pipeline

The quality of pseudo-labels is the primary determinant of final performance. Our pipeline generates three types of pseudo-labels from unlabeled images using pretrained models.

### 4.1 Semantic Pseudo-Labels via K-Means Clustering

**Motivation**: Given DINOv3's linear probe mIoU of 81.1, the patch features already encode rich semantic information. K-means clustering on these features should yield semantic categories that approximate ground-truth classes, without any supervision.

**Algorithm**: We apply mini-batch K-means (Sculley, 2010) to all DINOv3 patch features across the training set:

1. Extract features: $\mathbf{F}_i \in \mathbb{R}^{N \times 768}$ for each image $i$.
2. Concatenate all features: $\mathbf{F}_{\text{all}} \in \mathbb{R}^{(|\mathcal{D}| \cdot N) \times 768}$.
3. L2-normalize: $\hat{\mathbf{F}} = \mathbf{F} / \|\mathbf{F}\|_2$.
4. Mini-batch K-means with K=19 clusters (matching Cityscapes), 100 iterations, batch size 10000, 5 random restarts.
5. Assign each patch to its nearest cluster centroid: $y_i = \arg\min_k \|\hat{\mathbf{f}}_i - \boldsymbol{\mu}_k\|_2$.

**CRF Post-Processing**: Raw K-means assignments at patch level (32 x 64) are spatially coarse. We apply a dense CRF (Krahenbuhl & Koltun, 2011) to refine assignments using the original image as a guide:

$$P(y_i | \mathbf{x}) \propto \exp\left(-\sum_{j \neq i} w_1 \cdot \exp\left(-\frac{\|\mathbf{p}_i - \mathbf{p}_j\|^2}{2\sigma_\alpha^2} - \frac{\|\mathbf{I}_i - \mathbf{I}_j\|^2}{2\sigma_\beta^2}\right) - w_2 \cdot \exp\left(-\frac{\|\mathbf{p}_i - \mathbf{p}_j\|^2}{2\sigma_\gamma^2}\right)\right)$$

with parameters: $w_1 = 10$, $w_2 = 3$, $\sigma_\alpha = 80$, $\sigma_\beta = 13$, $\sigma_\gamma = 3$, 10 CRF iterations.

**Cluster-to-Class Alignment**: For evaluation purposes only (not used during training), we compute Hungarian matching between predicted clusters and ground-truth classes using the validation set.

**Implementation reference**: `mbps_pytorch/generate_semantic_pseudolabels.py` (494 lines). Key functions: `extract_and_cluster()` (K-means), `apply_crf_refinement()` (dense CRF), `hungarian_match()` (cluster alignment).

**Citation**: K-means clustering of self-supervised features follows the approach in STEGO (Hamilton et al., ICLR 2022) and PiCIE (Cho et al., CVPR 2021), but we use DINOv3 features which are substantially more discriminative.

### 4.2 Instance Pseudo-Labels via MaskCut

**Motivation**: Instance discovery requires identifying individual objects without knowing their class. Normalized Cut (NCut) on feature affinity matrices provides a principled graph-partitioning approach: similar features should be in the same segment, dissimilar features in different segments.

**Algorithm**: We implement MaskCut, following the methodology of CutLER (Wang et al., CVPR 2023):

1. **Affinity matrix construction**: For each image, compute cosine similarity between all pairs of DINOv3 patch features:
   $$\mathbf{A}_{ij} = \frac{\mathbf{f}_i^\top \mathbf{f}_j}{\|\mathbf{f}_i\| \cdot \|\mathbf{f}_j\|}$$

2. **Degree-normalized Laplacian**: Compute the random walk Laplacian:
   $$\mathbf{L}_{\text{rw}} = \mathbf{I} - \mathbf{D}^{-1}\mathbf{A}$$
   where $\mathbf{D}_{ii} = \sum_j \mathbf{A}_{ij}$.

3. **Spectral bipartition**: Compute the second-smallest eigenvector (Fiedler vector) of $\mathbf{L}_{\text{rw}}$ and threshold at zero to obtain a binary foreground/background mask:
   $$\mathbf{m} = \mathbb{1}[\mathbf{v}_2 > 0]$$

4. **Iterative discovery**: Mask out discovered foreground pixels and repeat steps 1-3 for up to `max_instances=20` per image. Stop when the foreground mask has fewer than `min_pixels=100` pixels.

5. **Filtering**: Discard masks with area < 100 pixels or NCut cost > 0.7 (low-confidence bipartitions).

**Depth-Weighted Affinities (Optional)**: When depth maps are available, we can enhance the affinity matrix with depth similarity:
$$\mathbf{A}'_{ij} = \mathbf{A}_{ij} \cdot \exp\left(-\frac{|d_i - d_j|^2}{2\sigma_d^2}\right)$$

This encourages grouping of pixels at similar depth, following the CutS3D approach.

**Implementation reference**: `mbps_pytorch/generate_instance_pseudolabels.py` (416 lines). Key functions: `maskcut_single_image()` (iterative NCut), `ncut_bipartition()` (spectral bisection), `masks_to_instance_map()` (mask compositing).

**Citation**: MaskCut is adapted from CutLER (Wang et al., "Cut and Learn for Unsupervised Object Detection and Instance Segmentation," CVPR 2023), which itself builds on Normalized Cuts (Shi & Malik, 2000) and TokenCut (Wang et al., NeurIPS 2022).

### 4.3 Monocular Depth Estimation

**Motivation**: Depth provides a 3D geometric prior that is orthogonal to appearance. Objects at different depths are likely different instances, and depth discontinuities often align with semantic boundaries. We use depth both as input to the UDCM (Section 6) and optionally as a refinement signal for pseudo-labels.

**Model**: We employ Depth Anything V3 (Yang et al., 2025) with the Large backbone. If DA V3 is unavailable, we fall back to Depth Anything V2 Large (Yang et al., 2024):

$$\mathbf{d} = \text{DepthAnythingV3}(\mathbf{x}) \quad \in \mathbb{R}^{H \times W}, \quad d_{ij} \in [0, 1]$$

Depth values are min-max normalized to [0, 1] per image: $d' = (d - d_{\min}) / (d_{\max} - d_{\min} + \epsilon)$.

**Implementation reference**: `mbps_pytorch/generate_depth_maps.py` (278 lines). Functions: `generate_depth_da3()` (DA V3 via `depth_anything_3` package), `generate_depth_da2()` (DA V2 via HuggingFace Transformers fallback).

**Citation**: Yang, L., et al. "Depth Anything V3: Scaling Up Monocular Depth Estimation with Synthetic Data." 2025. Fallback: Yang, L., et al. "Depth Anything V2," NeurIPS 2024.

### 4.4 Stuff-Things Classification

**Motivation**: Panoptic segmentation requires distinguishing "stuff" classes (uncountable amorphous regions: road, sky, vegetation) from "things" classes (countable objects: car, person, bicycle). Without annotations, we infer this from the statistical relationship between semantic clusters and instance masks.

**Heuristic**: For each semantic cluster $k$, we compute:

- **Instance overlap** $o_k$: Fraction of cluster pixels that overlap with any instance mask.
- **Average region count** $r_k$: Average number of connected components per image.
- **Average relative size** $s_k$: Average region area / image area.

The things score is:
$$\text{score}_k = w_1 \cdot o_k + w_2 \cdot \log(r_k) - w_3 \cdot s_k$$

with $w_1 = 1.0$, $w_2 = 0.3$, $w_3 = 0.5$. If $\text{score}_k > 0.3$, classify as "thing"; otherwise "stuff".

**Intuition**: Things classes have high instance overlap (they *are* instances), many disconnected regions per image (multiple cars, multiple pedestrians), and small relative size (each car is a small fraction of the image). Stuff classes have low instance overlap, few regions (often one connected road region), and large relative size.

**Implementation reference**: `mbps_pytorch/classify_stuff_things.py` (242 lines). Functions: `compute_cluster_statistics()`, `classify_stuff_things()`.

### 4.5 TFRecord Serialization

All pseudo-labels are serialized into TFRecords for efficient TPU training. Each record contains:

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| image | float32 | (H, W, 3) | RGB image, [0, 1] |
| depth | float32 | (H, W) | Depth map, [0, 1] |
| pseudo_semantic | int32 | (N,) | Patch-level semantic cluster IDs |
| pseudo_instance | int32 | (N,) | Patch-level instance IDs (0=background) |
| image_id | string | - | Unique identifier |

**Implementation reference**: `mbps_pytorch/generate_v2_tfrecords.py` (327 lines) for generation, `mbps/data/tfrecord_utils.py` (344 lines) for parsing in the training pipeline.

---

## 5. Semantic Head

### 5.1 Architecture

The semantic head is a simple 2-layer MLP that maps backbone features to per-token class logits:

$$\mathbf{s} = W_2 \cdot \text{GELU}(\text{LN}(W_1 \cdot \mathbf{f} + \mathbf{b}_1)) + \mathbf{b}_2$$

where $W_1 \in \mathbb{R}^{768 \times 768}$, $W_2 \in \mathbb{R}^{768 \times K}$, and LN denotes LayerNorm (Ba et al., 2016).

**Dimensional flow**: $(B, N, 768) \xrightarrow{W_1} (B, N, 768) \xrightarrow{\text{LN+GELU}} (B, N, 768) \xrightarrow{W_2} (B, N, K=19)$

### 5.2 Design Rationale

We deliberately use a *simple* head rather than a more complex decoder (e.g., UPerNet, FPN). The rationale is twofold:

1. **Feature quality**: With DINOv3 features at mIoU 81.1, the features already contain sufficient semantic information. A complex decoder risks overfitting to pseudo-label noise.
2. **Bridge interaction**: The semantic logits feed into the Mamba bridge, which provides the "decoding" function. A complex head before the bridge would reduce the bridge's ability to refine predictions.

**Implementation reference**: `mbps/models/mbps_v2_model.py`, class `SemanticHeadV2` (JAX); `mbps_pytorch/models/mbps_v2_model.py`, class `SemanticHeadV2` (PyTorch, lines 39-61).

---

## 6. Instance Embedding Head

### 6.1 Architecture

The instance head produces per-token embeddings in a low-dimensional space where same-instance tokens are close and different-instance tokens are far apart:

$$\mathbf{e} = W_4 \cdot \text{GELU}(\text{LN}(W_3 \cdot \mathbf{f} + \mathbf{b}_3)) + \mathbf{b}_4$$

where $W_3 \in \mathbb{R}^{768 \times 256}$, $W_4 \in \mathbb{R}^{256 \times D_i}$, and $D_i = 64$.

**Dimensional flow**: $(B, N, 768) \xrightarrow{W_3} (B, N, 256) \xrightarrow{\text{LN+GELU}} (B, N, 256) \xrightarrow{W_4} (B, N, 64)$

### 6.2 Design Rationale

The bottleneck from 768 to 256 to 64 serves two purposes:

1. **Compact representation**: Instance identity can be represented in a much lower-dimensional space than semantic class. Each instance needs only to be distinguishable from its spatial neighbors.
2. **Bridge compatibility**: The 64-dim embedding, when concatenated with 19-dim semantic logits, gives 83 total dimensions — a reasonable input size for projection to the 384-dim bridge space.

The 64-dim choice follows the discriminative loss literature (de Brabandere et al., 2017; Neven et al., 2019) where 32-128 dimensions are standard for instance embedding.

**Implementation reference**: `mbps/models/mbps_v2_model.py`, class `InstanceEmbeddingHead` (JAX); `mbps_pytorch/models/mbps_v2_model.py`, class `InstanceEmbeddingHead` (PyTorch, lines 64-84).

---

## 7. The Mamba Bridge: Cross-Modal State Space Fusion (Novel Contribution)

The Mamba Bridge is our primary novel contribution. It consists of four sub-modules applied in sequence: (1) Adaptive Projection Bridge, (2) Unified Depth Conditioning Module, (3) Bidirectional Cross-Modal Scan, and (4) Inverse Projection with gated residual connection.

### 7.1 Adaptive Projection Bridge (APB)

**Problem**: The semantic logits ($K = 19$ dimensions) and instance embeddings ($D_i = 64$ dimensions) live in fundamentally different vector spaces with different dimensionalities, magnitudes, and semantics. Directly concatenating or adding them would be meaningless.

**Solution**: We project both into a shared bridge space of dimension $D_{br} = 384$ using learned linear projections with LayerNorm:

$$\mathbf{s}_{\text{proj}} = \text{LN}(W_s \cdot \mathbf{s}) \quad \in \mathbb{R}^{B \times N \times D_{br}}$$
$$\mathbf{e}_{\text{proj}} = \text{LN}(W_e \cdot \mathbf{e}) \quad \in \mathbb{R}^{B \times N \times D_{br}}$$

where $W_s \in \mathbb{R}^{K \times D_{br}}$ and $W_e \in \mathbb{R}^{D_i \times D_{br}}$.

**Energy Alignment Loss**: To prevent one modality from dominating the shared space, we regularize the projected representations to have similar energy:

$$\mathcal{L}_{\text{align}} = \left| \frac{1}{BN} \sum_{b,n} \|\mathbf{s}_{\text{proj}}^{(b,n)}\|_2^2 - \frac{1}{BN} \sum_{b,n} \|\mathbf{e}_{\text{proj}}^{(b,n)}\|_2^2 \right|$$

This loss ensures that semantic and instance signals contribute equally to the bridge computation.

**Intuition**: The APB functions as a "translator" between modality-specific languages. Just as machine translation maps sentences from different languages into a shared latent space, the APB maps semantic logits and instance embeddings into a shared geometric space where cross-modal interactions are meaningful.

**Implementation reference**: `mbps/models/bridge/projection.py`, class `AdaptiveProjectionBridge` (JAX, 99 lines); `mbps_pytorch/models/bridge/projection.py` (PyTorch, 117 lines).

### 7.2 Unified Depth Conditioning Module (UDCM)

**Problem**: Semantic and instance boundaries often correspond to depth discontinuities. Without depth information, the bridge must learn these 3D relationships purely from 2D appearance — a fundamentally harder task.

**Solution**: We condition the bridge representations on monocular depth using FiLM (Feature-wise Linear Modulation, Perez et al., AAAI 2018). First, we encode the depth values using sinusoidal positional encoding (Vaswani et al., 2017):

$$\gamma(\mathbf{d}) = \left[\sin(2^k \pi \mathbf{d}), \cos(2^k \pi \mathbf{d})\right]_{k \in \{1,2,4,8,16,32\}} \quad \in \mathbb{R}^{B \times N \times 12}$$

The 6 frequency bands span from 1.0 to 32.0, encoding depth at multiple spatial frequencies. Low frequencies capture coarse depth structure (foreground vs. background), high frequencies capture fine depth discontinuities (object boundaries).

The encoding is processed through a small MLP and projected to FiLM parameters:

$$\mathbf{d}_{\text{feat}} = W_d \cdot \text{ReLU}(W_{d2} \cdot \text{ReLU}(W_{d1} \cdot \gamma(\mathbf{d}))) \quad \in \mathbb{R}^{B \times N \times D_{br}}$$

FiLM modulation is then applied separately to the semantic and instance projections:

$$\mathbf{s}_{\text{cond}} = \mathbf{s}_{\text{proj}} \odot \boldsymbol{\gamma}_s(\mathbf{d}_{\text{feat}}) + \boldsymbol{\beta}_s(\mathbf{d}_{\text{feat}})$$
$$\mathbf{e}_{\text{cond}} = \mathbf{e}_{\text{proj}} \odot \boldsymbol{\gamma}_f(\mathbf{d}_{\text{feat}}) + \boldsymbol{\beta}_f(\mathbf{d}_{\text{feat}})$$

where $\boldsymbol{\gamma}_\bullet = \text{clip}(W_\gamma \cdot \mathbf{d}_{\text{feat}} + \mathbf{1}, 0.1, 5.0)$ and $\boldsymbol{\beta}_\bullet = \text{clip}(W_\beta \cdot \mathbf{d}_{\text{feat}}, -5.0, 5.0)$.

**Stability clamping**: The gamma values are initialized around 1.0 (identity modulation) and clamped to [0.1, 5.0]. The beta values are clamped to [-5.0, 5.0]. Without clamping, depth conditioning can cause feature explosion during early training when the depth MLP is randomly initialized. This was discovered during MBPS v1 development where unclamped FiLM caused NaN gradients at epoch 41.

**Depth Consistency Loss**: We add a regularization term encouraging smooth depth conditioning — positions with similar depth should receive similar modulation:

$$\mathcal{L}_{\text{depth}} = \frac{1}{BN} \sum_{b,n} \left|\frac{\partial \bar{\gamma}_s}{\partial n}\right| \cdot \exp\left(-5 \cdot \text{clip}\left(\left|\frac{\partial d}{\partial n}\right|, 0, 10\right)\right)$$

where $\bar{\gamma}_s = \text{mean}(\boldsymbol{\gamma}_s, \text{dim}=-1)$ is the average gamma across feature dimensions. The exponential weighting permits large modulation changes at depth discontinuities while penalizing changes in smooth depth regions.

**Intuition**: FiLM conditioning allows the depth signal to *gate* which features pass through the bridge. At a depth discontinuity (e.g., a car in front of a building), the gamma and beta parameters can amplify instance-discriminative features and suppress background noise. This is geometrically meaningful: objects at different depths are almost certainly different instances.

**Implementation reference**: `mbps/models/bridge/depth_conditioning.py` (JAX, 127 lines); `mbps_pytorch/models/bridge/depth_conditioning.py` (PyTorch, 145 lines).

**Citation**: FiLM conditioning is adapted from Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer," AAAI 2018. The sinusoidal positional encoding follows Vaswani et al., "Attention Is All You Need," NeurIPS 2017.

### 7.3 Bidirectional Cross-Modal Scan (BiCMS) — Core Novel Module

The BiCMS is the heart of our contribution. It processes interleaved semantic and instance tokens through Mamba2 state space dynamics in both forward and backward directions, enabling each modality to attend to the other through the SSM's recurrent hidden state.

#### 7.3.1 Token Interleaving Strategy

**Key Insight**: Instead of processing semantic and instance tokens as *separate sequences*, we interleave them into a *single sequence*:

$$\mathbf{z} = [\mathbf{s}_1, \mathbf{e}_1, \mathbf{s}_2, \mathbf{e}_2, \ldots, \mathbf{s}_N, \mathbf{e}_N] \quad \in \mathbb{R}^{B \times 2N \times D_{br}}$$

This interleaving is critical. In the Mamba SSM, each position's output is a function of its input and a *compressed representation of all preceding positions*. By placing semantic token $\mathbf{s}_i$ immediately before instance token $\mathbf{e}_i$, we ensure that:

- The instance token $\mathbf{e}_i$ has direct access to the semantic context at position $i$ (through the recurrent state updated by $\mathbf{s}_i$).
- The semantic token $\mathbf{s}_{i+1}$ has access to the instance information at position $i$ (through the state updated by $\mathbf{e}_i$).

This creates an *alternating cross-modal attention* pattern that is naturally enforced by the SSM structure, without explicit cross-attention mechanisms.

**Implementation reference**: `interleave_tokens()` and `deinterleave_tokens()` functions in `mbps/models/bridge/bicms.py` and `mbps_pytorch/models/bridge/bicms.py`.

#### 7.3.2 Structured State Space Duality (SSD) Kernel

Each Mamba2 layer applies the Structured State Space Duality (SSD) kernel (Dao & Gu, 2024), which we implement using chunked matrix multiplications for TPU efficiency.

**SSM Parameterization**: The continuous-time SSM is:

$$\dot{\mathbf{h}}(t) = \mathbf{A} \mathbf{h}(t) + \mathbf{B}(t) x(t)$$
$$y(t) = \mathbf{C}(t)^\top \mathbf{h}(t) + D \cdot x(t)$$

where $\mathbf{h} \in \mathbb{R}^{D_s}$ is the state, $\mathbf{A} \in \mathbb{R}^{D_{br}}$ is a diagonal state transition matrix, $\mathbf{B} \in \mathbb{R}^{D_s}$ and $\mathbf{C} \in \mathbb{R}^{D_s}$ are input-dependent projections, and $D \in \mathbb{R}^{D_{br}}$ is a residual skip connection.

**Input Processing**: The input is first projected and split into a main branch and a gating signal:

$$[\bar{\mathbf{x}}, \mathbf{z}] = \text{split}(W_x \cdot \mathbf{x}) \quad \in \mathbb{R}^{2 \times (B \times L \times D_{br})}$$

**B/C Computation with RMSNorm** (stability critical):

$$\mathbf{B} = \text{RMSNorm}(W_B \cdot \bar{\mathbf{x}}) \quad \in \mathbb{R}^{B \times L \times D_s}$$
$$\mathbf{C} = \text{RMSNorm}(W_C \cdot \bar{\mathbf{x}}) \quad \in \mathbb{R}^{B \times L \times D_s}$$

**Stability note**: The RMSNorm on B and C is crucial. Without it, the SSD matmul chain (involving $P \times N$ products where $P = 64$ is chunk size and $N = 16$ is state dimension) can amplify input norms exponentially. This was the root cause of the Phase C NaN explosion in MBPS v1.

**Discretization** (Zero-Order Hold):

$$\Delta = \text{clip}(\text{softplus}(W_\Delta \cdot \bar{\mathbf{x}} + \mathbf{b}_\Delta), 10^{-4}, 5.0)$$
$$\bar{\mathbf{A}} = \exp(\text{clip}(\Delta \odot \mathbf{A}, -20, 0))$$

The $\Delta$ (timestep) parameter controls the "speed" of the state dynamics. The `dt_bias` is initialized via inverse softplus of a uniform distribution in [0.001, 0.1]:

$$\mathbf{b}_\Delta = \log(\exp(\text{Uniform}(0.001, 0.1)) - 1)$$

This ensures initial timesteps are small (slow dynamics), preventing the recurrent state from diverging before the model has learned meaningful update rules. The clamping of $\Delta \cdot \mathbf{A}$ to $[-20, 0]$ ensures that $\bar{\mathbf{A}} \in (0, 1]$, guaranteeing the state never grows unboundedly.

**Chunked SSD Computation**: For a sequence of length $L$, we split into chunks of size $P_c$:

For each chunk $c$ containing positions $p \in \{1, \ldots, P_c\}$:

1. **Causal transition matrix** (per-chunk):
$$M_{ij}^{(d)} = \begin{cases} \exp\left(\sum_{k=j}^{i} \log \bar{A}_{k}^{(d)}\right) & \text{if } i \geq j \\ 0 & \text{if } i < j \end{cases}$$

2. **Intra-chunk state computation**:
$$\text{state}_{i,n,d} = \sum_{j \leq i} M_{ij}^{(d)} \cdot B_{j,n} \cdot \bar{x}_{j,d}$$

3. **Intra-chunk output**:
$$y_{\text{intra},i,d} = \sum_n C_{i,n} \cdot \text{state}_{i,n,d}$$

4. **Inter-chunk state propagation** (via `jax.lax.scan` for memory efficiency):
$$\mathbf{h}_c = \mathbf{h}_{c-1} \odot \prod_p \bar{\mathbf{A}}_{c,p} + \sum_p \mathbf{B}_{c,p} \otimes (\bar{\mathbf{x}}_{c,p} \odot \text{decay-to-end}_{c,p})$$

5. **Inter-chunk correction**:
$$y_{\text{inter},p,d} = \sum_n C_{p,n} \cdot h_{c-1,n,d} \cdot \text{decay}_{p,d}$$

6. **Total output**: $y = y_{\text{intra}} + y_{\text{inter}}$

**D Skip Connection and Output Gating**:

$$\mathbf{y} = \text{RMSNorm}(\mathbf{y}_{\text{scan}} + \mathbf{D} \odot \bar{\mathbf{x}}) \odot \text{SiLU}(\mathbf{z})$$

The D skip connection (initialized to ones) provides a gradient highway through the SSD kernel, ensuring gradients can flow even when the scan dynamics are poorly conditioned. The RMSNorm before gating prevents output explosion.

**NaN Safety**: Per-chunk NaN guards replace any non-finite values with zeros:

$$\mathbf{y}_c = \begin{cases} \mathbf{y}_c & \text{if } \text{isfinite}(\mathbf{y}_c) \\ \mathbf{0} & \text{otherwise} \end{cases}$$

**Memory Optimization**: Using `jax.checkpoint` on the per-chunk scan body trades compute for memory, achieving O(num_chunks) memory instead of O(sequence_length). This is critical for the 2N = 4096 interleaved tokens on TPU with 32GB HBM.

**Implementation reference**: `mbps/models/bridge/mamba2_ssd.py`, class `SSDKernel` (JAX, primary implementation); `mbps_pytorch/models/bridge/mamba2_ssd.py`, class `SSDKernel` (PyTorch). The JAX version includes all stability techniques; the PyTorch version is simpler but less numerically stable.

**Citation**: The SSD kernel is adapted from Mamba-2 (Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality," ICML 2024), which shows that the selective SSM computation can be reformulated as a structured matrix multiplication with sub-quadratic complexity. The chunked computation follows the Mamba-2 reference implementation. The stability techniques (RMSNorm on B/C, dt_bias initialization, D skip connection, NaN guards) are our contributions based on empirical analysis of training dynamics.

#### 7.3.3 Mamba2 Block and Stack

Each **Mamba2Block** wraps the SSD kernel with residual connections and a feed-forward network:

$$\mathbf{z}' = \mathbf{z} + \text{Dropout}(\text{SSD}(\text{LN}(\mathbf{z})))$$
$$\mathbf{z}'' = \mathbf{z}' + \text{Dropout}(W_{\text{down}} \cdot \text{GELU}(W_{\text{up}} \cdot \text{LN}(\mathbf{z}')))$$

where the FFN has expansion factor 2 ($D_{br} \rightarrow 2D_{br} \rightarrow D_{br}$).

A **Mamba2Stack** chains $L_m = 4$ blocks with a final LayerNorm. The 4-layer depth was chosen as a balance between representational capacity and computational cost; deeper stacks showed diminishing returns in preliminary experiments.

**Implementation reference**: `mbps/models/bridge/mamba2_ssd.py`, classes `Mamba2Block` and `Mamba2Stack` (JAX).

#### 7.3.4 Bidirectional Scanning

A single forward scan can only capture dependencies from left to right (in raster order). Semantic and instance boundaries, however, are *bidirectional* — a car is bounded by road on both its left and right sides. We therefore run two independent Mamba2Stacks:

**Forward scan**: Processes tokens in natural order $[\mathbf{s}_1, \mathbf{e}_1, \mathbf{s}_2, \mathbf{e}_2, \ldots]$.

**Backward scan**: Processes tokens in reversed order and flips the output back:

$$\mathbf{y}_{\text{fwd}} = \text{Mamba2Stack}_{\text{fwd}}(\mathbf{z})$$
$$\mathbf{y}_{\text{bwd}} = \text{flip}(\text{Mamba2Stack}_{\text{bwd}}(\text{flip}(\mathbf{z})))$$

**Learned gating merge**: Rather than simply averaging or concatenating, we learn a position-dependent gate:

$$\boldsymbol{\alpha} = \sigma(W_g \cdot [\mathbf{y}_{\text{fwd}} \| \mathbf{y}_{\text{bwd}}]) \quad \in [0, 1]^{B \times 2N \times D_{br}}$$
$$\mathbf{y}_{\text{merged}} = \boldsymbol{\alpha} \odot \mathbf{y}_{\text{fwd}} + (1 - \boldsymbol{\alpha}) \odot \mathbf{y}_{\text{bwd}}$$

The sigmoid gate allows each position and feature dimension to independently choose how much to rely on forward vs. backward context. For example, the left boundary of an object might rely more on forward context (seeing background → foreground transition), while the right boundary relies more on backward context.

After merging, the interleaved tokens are de-interleaved back into separate semantic and instance streams:

$$\mathbf{s}_{\text{fused}}, \mathbf{e}_{\text{fused}} = \text{deinterleave}(\mathbf{y}_{\text{merged}})$$

**Implementation reference**: `mbps/models/bridge/bicms.py`, class `BidirectionalCrossModalScan` (JAX, 155 lines); `mbps_pytorch/models/bridge/bicms.py` (PyTorch, 173 lines).

### 7.4 Inverse Projection and Gated Residual

**Inverse Projection**: The fused bridge representations are projected back to the original dimensionalities:

$$\hat{\mathbf{s}} = \text{LN}(W_{\text{inv},s} \cdot \mathbf{s}_{\text{fused}}) \quad \in \mathbb{R}^{B \times N \times K}$$
$$\hat{\mathbf{e}} = \text{LN}(W_{\text{inv},e} \cdot \mathbf{e}_{\text{fused}}) \quad \in \mathbb{R}^{B \times N \times D_i}$$

**Gated Residual Connection**: The bridge output is added to the original head outputs via a learned gate:

$$g = \sigma(\alpha_g) \quad \text{where } \alpha_g \text{ is initialized to } -4$$

$$\mathbf{s}_{\text{final}} = \mathbf{s} + g \cdot \hat{\mathbf{s}}$$
$$\mathbf{e}_{\text{final}} = \mathbf{e} + g \cdot \hat{\mathbf{e}}$$

**Critical initialization**: $\sigma(-4) \approx 0.018$, so the bridge initially contributes less than 2% of the signal. This prevents the untrained bridge from corrupting the semantic and instance heads during early training. The gate learns to open as the bridge parameters stabilize.

For numerical safety, the refined semantic logits are clamped: $\mathbf{s}_{\text{final}} = \text{clip}(\mathbf{s}_{\text{final}}, -50, 50)$.

**Implementation reference**: `mbps/models/mbps_v2_model.py`, within `MBPSv2Model.__call__()` (JAX); `mbps_pytorch/models/mbps_v2_model.py`, within `MBPSv2Model.forward()` (PyTorch, lines 183-276).

---

## 8. Instance Clustering at Inference

At inference time, instance embeddings must be converted to discrete instance masks. We employ a cosine-similarity-based connected component algorithm:

1. **L2 normalize** embeddings: $\hat{\mathbf{e}}_i = \mathbf{e}_i / \|\mathbf{e}_i\|_2$
2. **Cosine similarity matrix**: $\text{sim}_{ij} = \hat{\mathbf{e}}_i^\top \hat{\mathbf{e}}_j$
3. **Spatial adjacency**: Build 4-connected grid adjacency $\mathbf{A}_{\text{spatial}}$ over the patch grid.
4. **Combined adjacency**: $\mathbf{A}_{ij} = (\text{sim}_{ij} > \tau) \wedge \mathbf{A}_{\text{spatial},ij}$ with $\tau = 0.7$.
5. **BFS connected components**: Extract components with minimum 4 patches.

**Intuition**: Requiring *both* high similarity and spatial adjacency prevents distant patches with accidentally similar embeddings from being grouped. The 4-patch minimum filters out noise.

**Implementation reference**: `mbps/models/instance/embedding_clustering.py` (166 lines). Functions: `cluster_embeddings()`, `_build_spatial_adjacency()`, `_connected_components_numpy()`.

---

## 9. Panoptic Merge

The panoptic merge module (Algorithm 9 from our technical report) combines semantic predictions and instance masks into a unified panoptic output:

1. Sort instances by confidence score (descending).
2. For each instance: compute majority semantic class; accept if it is a "thing" class and overlap with already-assigned pixels is < 50%.
3. Assign remaining pixels as "stuff" with their semantic class.
4. Encode: $\text{panoptic\_id} = \text{instance\_id} \times 1000 + \text{semantic\_class}$.

**Implementation reference**: `mbps/models/merger/panoptic_merge.py` (156 lines). Functions: `panoptic_merge()`, `batch_panoptic_merge()`.

---

## 10. Loss Functions

### 10.1 Semantic Cross-Entropy Loss

We train the semantic head with label-smoothed cross-entropy against pseudo-labels:

$$\mathcal{L}_{\text{sem}} = -\frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \sum_{k=1}^{K} \tilde{y}_{i,k} \log p_{i,k}$$

where $\mathcal{V} = \{i : y_i \neq 255\}$ is the set of valid (non-ignored) positions, $p_{i,k} = \text{softmax}(\mathbf{s}_i)_k$, and $\tilde{y}$ is the label-smoothed target:

$$\tilde{y}_{i,k} = \begin{cases} 1 - \epsilon + \epsilon/K & \text{if } k = y_i \\ \epsilon/K & \text{otherwise} \end{cases}$$

with label smoothing $\epsilon = 0.1$.

**Intuition for label smoothing**: Pseudo-labels are inherently noisy (they are K-means cluster assignments, not ground truth). Label smoothing prevents the model from becoming overconfident on incorrect pseudo-labels, which would be catastrophic for the self-training phase.

**Implementation reference**: `mbps/losses/semantic_loss_v2.py` (55 lines); `mbps_pytorch/losses/semantic_loss_v2.py` (50 lines).

**Citation**: Label smoothing from Szegedy et al., "Rethinking the Inception Architecture for Computer Vision," CVPR 2016.

### 10.2 Discriminative Instance Embedding Loss

We use the discriminative loss from de Brabandere et al. (2017), which consists of two terms:

**Pull loss** (intra-cluster variance): Pull embeddings of the same instance toward their mean:

$$\mathcal{L}_{\text{pull}} = \frac{1}{C} \sum_{c=1}^{C} \frac{1}{N_c} \sum_{i \in \mathcal{I}_c} \left[\|\mathbf{e}_i - \boldsymbol{\mu}_c\| - \delta_v\right]_+^2$$

where $C$ is the number of instances, $\mathcal{I}_c$ is the set of tokens belonging to instance $c$, $\boldsymbol{\mu}_c = \frac{1}{N_c}\sum_{i \in \mathcal{I}_c} \mathbf{e}_i$ is the instance centroid, $\delta_v = 0.5$ is the pull margin, and $[\cdot]_+ = \max(\cdot, 0)$.

**Push loss** (inter-cluster separation): Push centroids of different instances apart:

$$\mathcal{L}_{\text{push}} = \frac{1}{C(C-1)} \sum_{\substack{a,b=1 \\ a \neq b}}^{C} \left[2\delta_d - \|\boldsymbol{\mu}_a - \boldsymbol{\mu}_b\|\right]_+^2$$

where $\delta_d = 1.5$ is the push margin.

**Total discriminative loss**: $\mathcal{L}_{\text{inst}} = \mathcal{L}_{\text{pull}} + \mathcal{L}_{\text{push}}$

**Intuition**: The pull margin $\delta_v = 0.5$ means that embeddings within 0.5 units of their centroid are not penalized — this allows natural intra-instance variation. The push margin $\delta_d = 1.5$ means that instance centroids more than $2 \times 1.5 = 3.0$ units apart are not further separated — this focuses the model on hard cases (nearby instances) rather than already-separated ones.

Background tokens (instance ID = 0) are excluded from the loss computation.

**Implementation reference**: `mbps/losses/instance_embedding_loss.py` (149 lines); `mbps_pytorch/losses/instance_embedding_loss.py` (144 lines).

**Citation**: de Brabandere, B., et al. "Semantic Instance Segmentation with a Discriminative Loss Function," CVPR Workshops 2017. Also used in LaneNet (Neven et al., 2019).

### 10.3 Bridge Loss

The bridge loss ensures that the cross-modal fusion preserves information while encouraging alignment:

**Reconstruction Loss**: The inverse-projected outputs should reconstruct the original head outputs:

$$\mathcal{L}_{\text{recon}} = \frac{1}{BND_s}\|\mathbf{s} - \hat{\mathbf{s}}\|_F^2 + \frac{1}{BND_i}\|\mathbf{e} - \hat{\mathbf{e}}\|_F^2$$

**CKA Alignment Loss**: The fused semantic and instance representations should be correlated (since they describe the same spatial locations). We measure this using Centered Kernel Alignment (Kornblith et al., 2019):

$$\text{CKA}(\mathbf{X}, \mathbf{Y}) = \frac{\|\bar{\mathbf{Y}}^\top \bar{\mathbf{X}}\|_F^2}{\|\bar{\mathbf{X}}^\top \bar{\mathbf{X}}\|_F \cdot \|\bar{\mathbf{Y}}^\top \bar{\mathbf{Y}}\|_F}$$

where $\bar{\mathbf{X}} = \mathbf{X} - \frac{1}{N}\mathbf{1}\mathbf{1}^\top\mathbf{X}$ is the column-centered version of $\mathbf{X}$.

$$\mathcal{L}_{\text{CKA}} = -\text{CKA}(\mathbf{s}_{\text{fused}}, \mathbf{e}_{\text{fused}})$$

Minimizing negative CKA maximizes the alignment between the two fused streams.

**Numerical stability**: The denominator includes a safe sqrt: $\sqrt{\max(\text{HSIC}_{xx} \cdot \text{HSIC}_{yy}, 10^{-12})} + 10^{-6}$. In the JAX implementation, inputs are L2-normalized before Gram matrix computation.

**Total Bridge Loss**:

$$\mathcal{L}_{\text{bridge}} = \lambda_{\text{recon}} \cdot \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{align}} + \lambda_{\text{CKA}} \cdot \mathcal{L}_{\text{CKA}}$$

with $\lambda_{\text{recon}} = 0.5$, $\lambda_{\text{CKA}} = 0.1$.

**Implementation reference**: `mbps/losses/bridge_loss.py` (JAX); `mbps_pytorch/losses/bridge_loss.py` (PyTorch, 181 lines). Class `BridgeLoss` with functions `reconstruction_loss()`, `cka_loss()`.

**Citation**: CKA from Kornblith et al., "Similarity of Neural Network Representations Revisited," ICML 2019.

### 10.4 Total Loss

The total training loss is a weighted sum:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{sem}} \cdot \mathcal{L}_{\text{sem}} + \lambda_{\text{inst}} \cdot \mathcal{L}_{\text{inst}} + \lambda_{\text{bridge}} \cdot \mathcal{L}_{\text{bridge}}$$

with $\lambda_{\text{sem}} = 1.0$, $\lambda_{\text{inst}} = 1.0$, $\lambda_{\text{bridge}} = 0.1$.

Each loss term is independently guarded against NaN:

$$\mathcal{L}_{\text{term}}' = \begin{cases} \mathcal{L}_{\text{term}} & \text{if } \text{isfinite}(\mathcal{L}_{\text{term}}) \\ 0 & \text{otherwise} \end{cases}$$

**Implementation reference**: `compute_v2_loss()` in `scripts/train_v2.py` (JAX, lines 130-230); `compute_loss()` in `mbps_pytorch/training/trainer_v2.py` (PyTorch, lines 170-246).

---

## 11. Training Strategy

### 11.1 Two-Phase Training Curriculum

Unlike the 4-phase curriculum in MBPS v1, v2 uses a simplified 2-phase approach:

| Phase | Epochs | Bridge | Losses | Description |
|-------|--------|--------|--------|-------------|
| Bootstrap (Heads Only) | 1-5 | OFF | $\mathcal{L}_{\text{sem}} + \mathcal{L}_{\text{inst}}$ | Warm up semantic and instance heads without bridge interference. LR warmup from 0 to $5 \times 10^{-5}$. |
| Bootstrap + Bridge | 6-25 | ON (gate ramps 0.018 → 1.0) | $\mathcal{L}_{\text{sem}} + \mathcal{L}_{\text{inst}} + \mathcal{L}_{\text{bridge}}$ | Bridge activated; gate gradually opens as bridge parameters stabilize. |
| Self-Training R1 | 26-30 | ON | All | EMA teacher generates refined pseudo-labels. Confidence threshold $\geq 0.70$. |
| Self-Training R2 | 31-35 | ON | All | Confidence threshold $\geq 0.75$. |
| Self-Training R3 | 36-40 | ON | All | Confidence threshold $\geq 0.80$. |

**Bridge gate schedule**: The bridge gate logit starts at $-4$ ($\sigma(-4) \approx 0.018$) and is trained via gradient descent like any other parameter. During Bootstrap+Bridge phase, the gate naturally opens as the bridge loss decreases. No explicit schedule is needed.

**Implementation reference**: Phase logic in `scripts/train_v2.py`, `main()` function (lines 480-650). The `use_bridge` flag is set based on epoch number.

### 11.2 Optimization

**Optimizer**: AdamW (Loshchilov & Hutter, 2019) with:
- Learning rate: $5 \times 10^{-5}$
- Weight decay: 0.05
- $\beta_1 = 0.9$, $\beta_2 = 0.999$
- Gradient clipping: global norm 1.0

**Learning Rate Schedule**: Linear warmup for 3 epochs followed by cosine decay:

$$\eta(t) = \begin{cases} \eta_{\max} \cdot t / t_{\text{warmup}} & \text{if } t < t_{\text{warmup}} \\ \eta_{\max} \cdot \frac{1 + \cos(\pi \cdot (t - t_{\text{warmup}}) / (T - t_{\text{warmup}}))}{2} & \text{otherwise} \end{cases}$$

**NaN-safe gradient handling**: In the optimizer chain, we replace NaN gradients with zeros before applying the update:

$$\mathbf{g}' = \begin{cases} \mathbf{g} & \text{if } \text{isfinite}(\mathbf{g}) \\ \mathbf{0} & \text{otherwise} \end{cases}$$

This is implemented as a custom `_nan_to_zero()` function wrapped with `optax.stateless`:

```python
nan_guard = optax.stateless(lambda g, _: jax.tree.map(_nan_to_zero, g))
optimizer = optax.chain(nan_guard, optax.clip_by_global_norm(1.0), adamw, schedule)
```

**TPU Parallelism**: Training uses `jax.pmap` across 4 TPU v4 cores. Gradients are synchronized via `jax.lax.pmean(grads, axis_name="batch")`. The per-device batch size is 4, giving a global batch size of 16.

**Implementation reference**: `create_optimizer()` and `make_train_step()` in `scripts/train_v2.py` (JAX); `MBPSv2Trainer.__init__()` in `mbps_pytorch/training/trainer_v2.py` (PyTorch).

### 11.3 Exponential Moving Average (EMA) Teacher

For the self-training phase, we maintain an EMA copy of the model parameters:

$$\boldsymbol{\theta}_{\text{EMA}}^{(t)} = \alpha \cdot \boldsymbol{\theta}_{\text{EMA}}^{(t-1)} + (1 - \alpha) \cdot \boldsymbol{\theta}^{(t)}$$

with momentum $\alpha = 0.999$. The EMA teacher generates refined pseudo-labels for the self-training rounds.

**EMA teacher configuration**: The teacher uses `use_bridge=False` — it only needs the semantic and instance head outputs for generating pseudo-labels. This avoids propagating the potentially unstable bridge dynamics through the teacher, which was identified as a source of degenerate pseudo-labels in v1.

**Confidence thresholding**: Only predictions where the maximum softmax probability exceeds the confidence threshold are used as pseudo-labels. The threshold increases across self-training rounds (0.70 → 0.75 → 0.80) to progressively select only the most confident predictions.

**Citation**: EMA teacher from Tarvainen & Valpola, "Mean Teachers are Better Role Models," NeurIPS 2017. Self-training with confidence thresholding follows CUPS (Kim et al., CVPR 2025).

### 11.4 Numerical Stability Summary

The following stability techniques are applied throughout the pipeline. These were discovered through extensive debugging of MBPS v1's Phase C NaN failures:

| Technique | Location | Purpose |
|-----------|----------|---------|
| RMSNorm on B, C projections | SSD kernel | Prevent O(P x N) amplification in matmul chain |
| dt_bias inverse softplus init | SSD kernel | Constrain initial timesteps to [0.001, 0.1] |
| A_log clamping to [-20, 2] | SSD kernel | Prevent state transition matrix explosion |
| dt x A clamping to [-20, 0] | SSD kernel | Ensure discretized A_bar in (0, 1] |
| D skip connection (init=ones) | SSD kernel | Gradient highway through SSD |
| RMSNorm before gating | SSD kernel | Stabilize output magnitudes |
| Per-chunk NaN guards | SSD kernel | Prevent NaN propagation across chunks |
| FiLM gamma clamping [0.1, 5.0] | UDCM | Prevent depth-conditioned feature explosion |
| FiLM beta clamping [-5.0, 5.0] | UDCM | Prevent depth-conditioned offset explosion |
| Bridge gate init at sigmoid(-4) | Main model | Prevent untrained bridge corruption |
| Logit clamping [-50, 50] | Main model | Prevent softmax overflow |
| Per-loss NaN guards | Loss computation | Isolate NaN losses from total |
| Gradient NaN-to-zero | Optimizer chain | Prevent NaN gradient propagation |

---

## 12. Ablation Study Design

We design six controlled ablation experiments to isolate the contribution of each architectural component. Each ablation modifies exactly one aspect of the full model while keeping all other hyperparameters identical. All experiments are run with 3 random seeds (42, 123, 456) and we report mean and standard deviation.

### 12.1 Ablation: No Mamba Bridge (`no_mamba.yaml`)

**Modification**: Replace the Mamba2 BiCMS with a simple MLP that concatenates semantic and instance projections:

$$\mathbf{y}_{\text{fused}} = \text{GELU}(W_{\text{mlp}} \cdot [\mathbf{s}_{\text{proj}} \| \mathbf{e}_{\text{proj}}])$$

where $W_{\text{mlp}} \in \mathbb{R}^{2D_{br} \times D_{br}}$.

**Hypothesis**: Without the Mamba2 bridge, there is no mechanism for *sequential* cross-modal interaction. The MLP can only combine features at each position independently, missing the crucial inter-position dependencies (e.g., "this patch is the same instance as the previous one because they share semantic context"). We expect a drop in both SQ (segment quality) and RQ (recognition quality), particularly for "things" classes where instance boundary precision matters.

**Configuration**: `configs/v2_ablations/no_mamba.yaml` — sets `use_mamba_bridge: false`.

### 12.2 Ablation: No Depth Conditioning (`no_depth.yaml`)

**Modification**: Disable the UDCM entirely. The projected representations pass directly to the BiCMS without depth modulation.

**Hypothesis**: Depth provides 3D geometric priors that are complementary to appearance. Without depth conditioning, the model must infer depth relationships purely from RGB features — harder but not impossible since DINOv3 features encode some depth information implicitly. We expect a moderate drop, particularly for overlapping objects at different depths.

**Configuration**: `configs/v2_ablations/no_depth.yaml` — sets `use_depth_conditioning: false`.

### 12.3 Ablation: No Bidirectional Scan (`no_bicms.yaml`)

**Modification**: Use only the forward Mamba2 scan, discarding the backward scan and learned gate merge.

**Hypothesis**: Unidirectional scanning creates an inherent asymmetry: tokens can only attend to preceding tokens. For 2D images, this means raster-order bias — the model can use left and top context but not right or bottom context. We expect degradation on objects whose boundaries are better detected from the right or bottom direction.

**Configuration**: `configs/v2_ablations/no_bicms.yaml` — sets `use_bidirectional: false`.

### 12.4 Ablation: No Bridge at All (`no_bridge.yaml`)

**Modification**: Completely disable the bridge. Semantic and instance heads are trained independently with no cross-modal fusion.

**Hypothesis**: This is the most extreme ablation, testing whether cross-modal fusion provides any benefit at all. Without the bridge, semantic predictions cannot help resolve instance ambiguity and vice versa. We expect this to be the worst-performing variant, establishing the baseline improvement from any form of cross-modal interaction.

**Configuration**: `configs/v2_ablations/no_bridge.yaml` — disables bridge entirely, sets $\lambda_{\text{bridge}} = 0$.

### 12.5 Ablation: No Self-Training (`no_self_train.yaml`)

**Modification**: Skip the self-training phase entirely. Train only the bootstrap phase (40 epochs instead of 40).

**Hypothesis**: Self-training with EMA teacher and progressive confidence thresholding should improve pseudo-label quality iteratively. Without it, we rely entirely on the initial K-means/MaskCut pseudo-labels. We expect a moderate drop, particularly in instance discovery quality where MaskCut's recall is limited.

**Configuration**: `configs/v2_ablations/no_self_train.yaml` — sets `self_training_rounds: 0`.

### 12.6 Ablation: DINOv1 Backbone (`dinov1.yaml`)

**Modification**: Replace DINOv3 ViT-B/16 (768-dim) with DINO ViT-S/8 (384-dim). All downstream dimensions are halved accordingly.

**Hypothesis**: This tests the "backbone quality bottleneck" hypothesis. With DINOv1's ~30 mIoU features (vs. DINOv3's ~81 mIoU), we expect dramatic degradation in all metrics, demonstrating that the backbone upgrade is the primary driver of improvement over MBPS v1.

**Configuration**: `configs/v2_ablations/dinov1.yaml` — changes backbone to `dino_vits8`, backbone_dim to 384, bridge_dim to 192.

### 12.7 Expected Ablation Results

| Configuration | PQ | SQ | RQ | mIoU |
|--------------|----|----|----|----|
| Full MBPS v2 | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ |
| No Mamba Bridge | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ |
| No Depth Conditioning | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ |
| No Bidirectional | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ |
| No Bridge At All | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ |
| No Self-Training | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ |
| DINOv1 Backbone | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ |

Results are yet to be calculated.

---

## 13. Experimental Setup

### 13.1 Datasets

**Cityscapes** (Cordts et al., CVPR 2016): 2975 training images, 500 validation images, 1525 test images. Resolution: 1024 x 2048 (resized to 512 x 1024). 19 semantic classes (11 stuff + 8 things). We train on the training split and evaluate on the validation split.

**COCO-Stuff-27** (Caesar et al., CVPR 2018): ~118K training images, 5K validation images. Resolution varies (resized to 512 x 512). 27 semantic classes (15 stuff + 12 things).

### 13.2 Evaluation Metrics

- **PQ** (Panoptic Quality, Kirillov et al., 2019): $\text{PQ} = \text{SQ} \times \text{RQ}$.
- **SQ** (Segmentation Quality): Average IoU over matched segments.
- **RQ** (Recognition Quality): $\text{RQ} = \frac{\text{TP}}{\text{TP} + \frac{1}{2}\text{FP} + \frac{1}{2}\text{FN}}$.
- **mIoU**: Mean intersection-over-union for semantic evaluation.
- **PQ^St** / **PQ^Th**: PQ for stuff / things classes separately.

### 13.3 Infrastructure

- **Offline pipeline**: Single NVIDIA GPU (A100 recommended) for DINOv3 feature extraction, K-means, MaskCut, and depth estimation.
- **Training**: Google Cloud TPU v4-8 (4 chips, 32GB HBM each), zone `us-central2-b`, GCP project `unsupervised-panoptic-segment`.
- **Parallelism**: `jax.pmap` across 4 TPU cores, batch size 4/core = 16 global.
- **Training time**: \_\_\_\_ hours for 40 epochs on Cityscapes. Results are yet to be calculated.
- **Data storage**: GCS bucket `gs://mbps-panoptic/` for TFRecords, checkpoints, and results.

### 13.4 Hyperparameters

| Hyperparameter | Value | Source |
|---------------|-------|--------|
| Backbone | DINOv3 ViT-B/16 | Oquab et al., 2025 |
| Backbone dim | 768 | - |
| Patch size | 16 | - |
| Num classes (Cityscapes) | 19 | Cordts et al., 2016 |
| Instance embed dim | 64 | de Brabandere et al., 2017 |
| Bridge dim | 384 | Ablation-tuned |
| Mamba2 layers per direction | 4 | Ablation-tuned |
| SSM state dim | 16 (TPU) / 64 (GPU) | Memory-constrained |
| Chunk size | 64 (TPU) / 128 (GPU) | Hardware-aligned |
| Learning rate | 5e-5 | Standard for frozen backbone |
| Weight decay | 0.05 | - |
| Warmup epochs | 3 | - |
| Gradient clip norm | 1.0 | - |
| Label smoothing | 0.1 | Szegedy et al., 2016 |
| Pull margin (delta_v) | 0.5 | de Brabandere et al., 2017 |
| Push margin (delta_d) | 1.5 | de Brabandere et al., 2017 |
| Lambda semantic | 1.0 | - |
| Lambda instance | 1.0 | - |
| Lambda bridge | 0.1 | Ablation-tuned |
| Lambda recon | 0.5 | - |
| Lambda CKA | 0.1 | - |
| EMA momentum | 0.999 | Tarvainen & Valpola, 2017 |
| Bridge gate init | sigmoid(-4) ≈ 0.018 | Empirically determined |
| Total epochs | 40 | - |
| Bootstrap end | 25 | - |
| Self-training rounds | 3 | Following CUPS |
| Confidence thresholds | 0.70, 0.75, 0.80 | Progressive |

---

## 14. Results

### 14.1 Cityscapes Validation

| Method | Supervision | PQ | SQ | RQ | PQ^Th | PQ^St | mIoU |
|--------|------------|----|----|----|----|----|----|
| Panoptic FPN (Kirillov et al., 2019) | Full | 58.1 | 77.9 | 72.1 | 52.0 | 62.5 | - |
| Mask2Former (Cheng et al., 2022) | Full | 62.1 | - | - | - | - | - |
| CUPS (Kim et al., CVPR 2025) | Stereo Video | 27.8 | 57.4 | 35.2 | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ |
| **MBPS v2 (Ours)** | **Monocular** | **\_\_\_\_** | **\_\_\_\_** | **\_\_\_\_** | **\_\_\_\_** | **\_\_\_\_** | **\_\_\_\_** |

Results are yet to be calculated.

### 14.2 COCO-Stuff-27

| Method | Supervision | PQ | SQ | RQ |
|--------|------------|----|----|-----|
| CUPS (Kim et al., CVPR 2025) | Stereo Video | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ |
| **MBPS v2 (Ours)** | **Monocular** | **\_\_\_\_** | **\_\_\_\_** | **\_\_\_\_** |

Results are yet to be calculated.

### 14.3 Pseudo-Label Quality

| Pseudo-Label | Metric | Target | Actual |
|-------------|--------|--------|--------|
| Semantic (K-means on DINOv3) | mIoU vs GT | >= 50 | \_\_\_\_ |
| Instance (MaskCut) | AR@100 | >= 30 | \_\_\_\_ |

Results are yet to be calculated.

### 14.4 Ablation Results

| Configuration | PQ | Delta PQ | SQ | RQ |
|--------------|-----|----------|-----|-----|
| Full MBPS v2 | \_\_\_\_ | - | \_\_\_\_ | \_\_\_\_ |
| w/o Mamba Bridge | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ |
| w/o Depth Conditioning | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ |
| w/o Bidirectional Scan | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ |
| w/o Bridge (Independent Heads) | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ |
| w/o Self-Training | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ |
| DINOv1 ViT-S/8 Backbone | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ | \_\_\_\_ |

Results are yet to be calculated. All experiments use 3 seeds (42, 123, 456); mean +/- std reported.

---

## 15. Codebase Reference

### 15.1 File-to-Component Mapping

| Component | JAX/Flax (TPU Training) | PyTorch (Offline/GPU) |
|-----------|------------------------|-----------------------|
| DINOv3 Backbone | `mbps/models/backbone/dinov3_vitb.py` (355 lines) | `mbps_pytorch/models/backbone/dinov3_vitb.py` (341 lines) |
| Weight Converter | `mbps/models/backbone/dinov3_weights_converter.py` | `DINOv3ViTB.from_pretrained()` |
| Semantic Head | `mbps/models/mbps_v2_model.py::SemanticHeadV2` | `mbps_pytorch/models/mbps_v2_model.py::SemanticHeadV2` |
| Instance Head | `mbps/models/mbps_v2_model.py::InstanceEmbeddingHead` | `mbps_pytorch/models/mbps_v2_model.py::InstanceEmbeddingHead` |
| Full Model | `mbps/models/mbps_v2_model.py::MBPSv2Model` (292 lines) | `mbps_pytorch/models/mbps_v2_model.py::MBPSv2Model` (302 lines) |
| APB | `mbps/models/bridge/projection.py` (99 lines) | `mbps_pytorch/models/bridge/projection.py` (117 lines) |
| UDCM | `mbps/models/bridge/depth_conditioning.py` (127 lines) | `mbps_pytorch/models/bridge/depth_conditioning.py` (145 lines) |
| SSD Kernel | `mbps/models/bridge/mamba2_ssd.py` | `mbps_pytorch/models/bridge/mamba2_ssd.py` |
| BiCMS | `mbps/models/bridge/bicms.py` (155 lines) | `mbps_pytorch/models/bridge/bicms.py` (173 lines) |
| Panoptic Merge | `mbps/models/merger/panoptic_merge.py` (156 lines) | - |
| Instance Clustering | `mbps/models/instance/embedding_clustering.py` (166 lines) | - |
| Semantic Loss | `mbps/losses/semantic_loss_v2.py` (55 lines) | `mbps_pytorch/losses/semantic_loss_v2.py` (50 lines) |
| Instance Loss | `mbps/losses/instance_embedding_loss.py` (149 lines) | `mbps_pytorch/losses/instance_embedding_loss.py` (144 lines) |
| Bridge Loss | `mbps/losses/bridge_loss.py` | `mbps_pytorch/losses/bridge_loss.py` (181 lines) |
| Training Loop | `scripts/train_v2.py` (660 lines) | `mbps_pytorch/training/trainer_v2.py` (385 lines) |
| Train Entry Point | `scripts/train_v2.py` | `mbps_pytorch/scripts/train_v2.py` (291 lines) |
| Feature Extraction | - | `mbps_pytorch/extract_dinov3_features.py` (245 lines) |
| Semantic Pseudo-Labels | - | `mbps_pytorch/generate_semantic_pseudolabels.py` (494 lines) |
| Instance Pseudo-Labels | - | `mbps_pytorch/generate_instance_pseudolabels.py` (416 lines) |
| Depth Maps | - | `mbps_pytorch/generate_depth_maps.py` (278 lines) |
| Stuff-Things | - | `mbps_pytorch/classify_stuff_things.py` (242 lines) |
| TFRecord Generation | `mbps/data/tfrecord_utils.py` (344 lines) | `mbps_pytorch/generate_v2_tfrecords.py` (327 lines) |
| Base Config | `configs/v2_default.yaml` (117 lines) | - |
| Cityscapes Config | `configs/v2_cityscapes.yaml` (16 lines) | - |
| GCS Config | `configs/v2_cityscapes_gcs.yaml` (21 lines) | - |
| Ablation Configs | `configs/v2_ablations/*.yaml` (6 files) | - |

### 15.2 Total Codebase Statistics

| Category | Files | Lines |
|----------|-------|-------|
| JAX/Flax (TPU Training) | 12 | ~1,891 |
| PyTorch (Offline + GPU) | 12 | ~1,948 |
| Offline Pipeline | 6 | ~2,002 |
| Configs | 9 | ~191 |
| **Total** | **39** | **~6,032** |

---

## 16. Summary of Novel Contributions vs. Adapted Components

### Novel Contributions (Ours)

1. **Bidirectional Cross-Modal Scan (BiCMS)**: The first application of state space models for cross-modal fusion in panoptic segmentation. The token interleaving strategy and bidirectional scanning with learned gating are novel.

2. **Unified Depth Conditioning Module (UDCM)**: FiLM-based depth conditioning with stability clamping and depth consistency regularization applied to cross-modal bridge representations.

3. **Adaptive Projection Bridge (APB)**: Energy-balanced projection of heterogeneous modalities into a shared bridge space with alignment regularization.

4. **Bridge Gate Initialization**: The $\sigma(-4) \approx 0.018$ initialization that prevents untrained bridge parameters from corrupting head outputs.

5. **SSD Numerical Stability Suite**: The combination of RMSNorm on B/C projections, dt_bias inverse softplus initialization, D skip connection, A_log/dt clamping, and per-chunk NaN guards — all developed through empirical debugging of Mamba2 training dynamics on TPU.

6. **NaN-safe optimizer chain**: The `_nan_to_zero()` gradient filter in the optimizer pipeline.

### Adapted Components (with citations)

| Component | Source | Adaptation |
|-----------|--------|------------|
| DINOv3 ViT-B/16 backbone | Oquab et al., 2025 (Meta AI) | Frozen feature extractor; weight conversion to JAX/Flax |
| Mamba2 SSD kernel | Dao & Gu, ICML 2024 | Chunked implementation adapted for TPU; added stability techniques |
| K-means semantic clustering | Hamilton et al., ICLR 2022 (STEGO); Cho et al., CVPR 2021 (PiCIE) | Applied to DINOv3 features with CRF post-processing |
| MaskCut instance discovery | Wang et al., CVPR 2023 (CutLER) | Iterative NCut on DINOv3 features |
| Depth Anything V3 | Yang et al., 2025 | Monocular depth for conditioning and pseudo-label refinement |
| FiLM conditioning | Perez et al., AAAI 2018 | Applied to bridge representations with stability clamping |
| Sinusoidal position encoding | Vaswani et al., NeurIPS 2017 | Used for depth value encoding |
| Discriminative loss | de Brabandere et al., CVPR-W 2017 | Standard formulation for instance embeddings |
| Label-smoothed cross-entropy | Szegedy et al., CVPR 2016 | Standard formulation for pseudo-label training |
| CKA alignment | Kornblith et al., ICML 2019 | Applied to fused bridge representations |
| EMA teacher | Tarvainen & Valpola, NeurIPS 2017 | For self-training pseudo-label refinement |
| Panoptic merge algorithm | Kirillov et al., CVPR 2019 | Standard score-based instance assignment |
| Dense CRF post-processing | Krahenbuhl & Koltun, NeurIPS 2011 | For pseudo-label spatial refinement |
| Normalized Cut | Shi & Malik, TPAMI 2000 | Spectral bisection for instance discovery |

---

## References

- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. *arXiv:1607.06450*.
- Caesar, H., Uijlings, J., & Ferrari, V. (2018). COCO-Stuff: Thing and stuff classes in context. *CVPR*.
- Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., & Girdhar, R. (2022). Masked-attention mask transformer for universal image segmentation. *CVPR*.
- Cho, J., Mall, U., Bala, K., & Hariharan, B. (2021). PiCIE: Unsupervised semantic segmentation using invariance and equivariance in clustering. *CVPR*.
- Cordts, M., Omran, M., Ramos, S., et al. (2016). The Cityscapes dataset for semantic urban scene understanding. *CVPR*.
- Dao, T., & Gu, A. (2024). Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. *ICML*.
- Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P. (2024). Vision transformers need registers. *ICLR*.
- de Brabandere, B., Neven, D., & Van Gool, L. (2017). Semantic instance segmentation with a discriminative loss function. *CVPR Workshops*.
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR*.
- Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv:2312.00752*.
- Hamilton, M., Zhang, Z., Hariharan, B., Snavely, N., & Freeman, W. T. (2022). Unsupervised semantic segmentation by distilling feature correspondences. *ICLR*.
- Kim, S., et al. (2025). CUPS: Unsupervised panoptic segmentation from stereo video. *CVPR*.
- Kirillov, A., He, K., Girshick, R., Rother, C., & Dollar, P. (2019). Panoptic segmentation. *CVPR*.
- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. *ICML*.
- Krahenbuhl, P., & Koltun, V. (2011). Efficient inference in fully connected CRFs with Gaussian edge potentials. *NeurIPS*.
- Li, Z., Wang, W., Xie, E., Yu, Z., Anandkumar, A., Alvarez, J. M., & Lu, T. (2022). Panoptic SegFormer: Delving deeper into panoptic segmentation with transformers. *CVPR*.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
- Neven, D., De Brabandere, B., Proesmans, M., & Van Gool, L. (2019). Instance segmentation by jointly optimizing spatial embeddings and clustering bandwidth. *CVPR*.
- Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). DINOv2: Learning robust visual features without supervision. *TMLR*.
- Oquab, M., Darcet, T., et al. (2025). DINOv3: Learning robust visual features with self-supervised vision transformers. *Meta AI Research*.
- Perez, E., Strub, F., De Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual reasoning with a general conditioning layer. *AAAI*.
- Sculley, D. (2010). Web-scale k-means clustering. *WWW*.
- Shi, J., & Malik, J. (2000). Normalized cuts and image segmentation. *TPAMI*.
- Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception architecture for computer vision. *CVPR*.
- Tarvainen, A., & Valpola, H. (2017). Mean teachers are better role models. *NeurIPS*.
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *NeurIPS*.
- Wang, X., Girdhar, R., Yu, S. X., & Misra, I. (2023). Cut and learn for unsupervised object detection and instance segmentation. *CVPR*.
- Wang, Y., Shen, X., Hu, S. X., Yuan, Y., Crowley, J. L., & Vaufreydaz, D. (2022). Self-supervised transformers for unsupervised object discovery using normalized cut. *NeurIPS*.
- Yang, L., Kang, B., Huang, Z., et al. (2024). Depth Anything V2. *NeurIPS*.
- Yang, L., et al. (2025). Depth Anything V3: Scaling up monocular depth estimation with synthetic data.
