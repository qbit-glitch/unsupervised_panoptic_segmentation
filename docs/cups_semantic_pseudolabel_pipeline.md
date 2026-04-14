# CUPS Semantic Pseudo-Label Generation Pipeline

> Deep analysis of the CUPS (CVPR 2025) codebase at `refs/cups/`.
> Date: 2026-04-14

## Overview

CUPS generates semantic pseudo-labels by **retraining DepthG** (CVPR 2024) with overclustering,
then running a depth-guided sliding window inference + CRF post-processing pipeline.
No ground-truth labels are used — the entire process is unsupervised.

---

## 1. Backbone: DINO v1 ViT-Small/8

- **Architecture**: ViT-Small, patch_size=8, embed_dim=384, depth=12, num_heads=6
- **Weights**: `dino_deitsmall8_300ep_pretrain.pth` from fbaipublicfiles (DINO v1, NOT DINOv2)
- **Status**: Completely frozen (`requires_grad = False`), eval mode
- **Output**: 384-dim patch features at (H/8, W/8) resolution
- **Source**: `refs/cups/external/depthg/src/modules.py:19-137` (`DinoFeaturizer`)

CUPS added a DINOv2 code path (line 105) that handles register tokens differently,
but the default DepthG model uses DINO v1.

## 2. Segmentation Head

A lightweight trainable projection on top of frozen DINO features:

```
DINO features (384-dim) → 1x1 Conv → code (dim-dimensional)
                        ↘ (optional) nonlinear path: 1x1 Conv → ReLU → 1x1 Conv → added to code
```

- When `cfg.continuous = True`: `dim = cfg.dim` (continuous code space, typically 90)
- When `cfg.continuous = False`: `dim = n_classes` (each dim = a class)
- Source: `modules.py:75-89` (`make_clusterer`, `make_nonlinear_clusterer`)

## 3. Two Cluster Probes (Critical Design)

```python
self.train_cluster_probe = ClusterLookup(dim, n_classes)                    # 27 centroids
self.cluster_probe = ClusterLookup(dim, n_classes + cfg.extra_clusters)     # 27+ centroids (overclustered)
```

**ClusterLookup** (`modules.py:647-672`):
- Learnable centroid vectors: `self.clusters = nn.Parameter(torch.randn(n_classes, dim))`
- Forward: L2-normalize both features and centroids → cosine similarity via einsum
- `inner_products = einsum("bchw,nc->bnhw", normed_features, normed_clusters)`
- Returns `softmax(inner_products * alpha)` or `log_softmax(inner_products * alpha)`
- Cluster loss: `-(cluster_probs * inner_products).sum(1).mean()`

**Key**: Both probes are trained on **detached codes** (line 447: `detached_code = torch.clone(code.detach())`).
They do NOT backpropagate gradients to the segmentation head. Only contrastive losses train the head.

### Overclustering

- CUPS paper Table 7b: k=27 → PQ=27.8, k=40 → PQ=30.3, k=54 → PQ=30.6
- More clusters → finer semantic distinctions → better panoptic quality
- At inference time, the overclustered `cluster_probe` is used (with `alpha=2` as temperature)
- CUPS integrates Hungarian matching metric (`PanopticQualitySemanticMatching`) to evaluate overclustered outputs

## 4. Training Losses

### What trains the segmentation head (contrastive losses):

| Loss | Formula | Purpose |
|------|---------|---------|
| Positive intra-image | `-cd * (fd - shift)` on same-image pairs | Self-correlation: backbone similarity → code similarity |
| Positive inter-image | `-cd * (fd - shift)` on k-NN pairs | Cross-image correspondence: similar features → similar codes |
| Negative inter-image | `-cd * (fd - shift)` on random pairs | Different images → dissimilar codes |
| Depth feature correlation | `-cd * (dd - shift)` | Codes correlate where depth maps correlate |

Where:
- `fd` = cosine correlation of frozen DINO features (no gradient)
- `cd` = cosine correlation of learned codes (gradient flows here)
- `dd` = cosine correlation of depth maps
- `shift` = learnable margin parameter

Source: `modules.py:1221-1261` (`ContrastiveCorrelationLoss`)
Source: `modules.py:1370-1463` (`DepthContrastiveCorrelationLoss`)

### LHP (Local Hidden Positive Projection)

- Propagates codes to nearby spatial locations based on depth similarity
- Strategy `"depth"`: converts depth to 3D points, computes pairwise distances, creates soft neighbor mask
- Strategy `"attn"`: uses DINO self-attention maps for propagation
- Applied as additional contrastive loss with weight `cfg.lhp_weight`
- Source: `modules.py:140-199` (`LocalHiddenPositiveProjection`)

### What trains the cluster probes (on detached codes):

| Loss | Purpose |
|------|---------|
| Cluster loss | Centroids align to feature clusters (only updates centroid parameters) |
| Linear probe loss | Cross-entropy with GT labels (monitoring only, doesn't affect head) |

### Other optional losses:

- `rec_loss`: Reconstruction loss (weight=0.0 by default)
- `crf_loss`: Contrastive CRF spatial smoothness
- `aug_alignment_loss`: Augmentation consistency

Source: `train_segmentation.py:190-469` (`training_step`)

## 5. Depth Usage — Training vs Inference

| Stage | Depth Source | How Used |
|-------|-------------|----------|
| DepthG training | Precomputed monocular depth (ZoeDepth PNGs) | Depth correlation loss, LHP propagation, FPS sampling |
| CUPS inference | Stereo disparity via RAFT | Depth-guided sliding window fusion |

**Training depth sampling strategies** (`cfg.depth_sampling`):
- `"fps"`: Farthest Point Sampling on depth point clouds
- `"simple"`: Proportional to depth distribution
- `"none"`: Random uniform

## 6. CUPS-Specific Modifications to DepthG

The code in `refs/cups/external/depthg/` is **NOT vanilla DepthG**. Four concrete changes:

1. **Overclustering + CUPS metric** (lines 164-177): Imports `cups.metrics.PanopticQualitySemanticMatching`
   for Hungarian-matching evaluation during validation when `extra_clusters > 0`

2. **CUPS data import** (data.py line 28): `from cups.utils import normalize`

3. **mIoU replaced** (lines 536-539): When `extra_clusters > 0`, standard mIoU is replaced
   by Hungarian-matched mIoU from the CUPS metric

4. **DINOv2 code path** (modules.py line 105): Original DepthG only supports DINO v1;
   CUPS added handling for DINOv2 register tokens

**Conclusion**: CUPS likely **retrains DepthG** with their own overclustering setup,
since Table 7b shows results for different k values and the training loop is integrated
with CUPS-specific metrics. The semantic model is still trained unsupervised.

## 7. Architecture Variants

`LitUnsupervisedSegmenter` supports three architectures:

| `cfg.arch` | Class | Depth in backbone? |
|------------|-------|--------------------|
| `"dino"` | `DinoFeaturizer` | No — depth only in losses |
| `"dino_depth"` | `DinoFeaturizerWithDepth` | Yes — cross-attention fusion |
| `"feature-pyramid"` | `FeaturePyramidNet` | No — legacy ResNet option |

Note: `DinoFeaturizerWithDepth` uses depth during training but creates a **zero depth tensor**
at inference time (line 613), using a learned `no_depth_embed` as query for cross-attention.
So depth conditioning is active during training but effectively neutralized at inference.

## 8. Complete Inference Pipeline (gen_pseudo_labels.py)

Per image, step by step:

### Step 1: Load stereo video data
- `CityscapesStereoVideo`: left/right images at times t and t+1
- Camera intrinsics + baseline
- Resolution: 640×1280

### Step 2: Compute optical flow + stereo disparity (for instances)
```python
optical_flow_l_forward = raft(image_0_l, image_1_l)
optical_flow_l_backward = raft(image_1_l, image_0_l)
disparity_1_forward = raft(image_0_l, image_0_r, disparity=True)
# ... 3 more RAFT calls
```

### Step 3: SE(3) object proposals (instance masks)
```python
object_proposals = get_object_proposals(
    image_1_l, optical_flow_l_forward, optical_flow_l_backward,
    disparity_1_forward, disparity_2_forward,
    disparity_1_backward, disparity_2_backward,
    intrinsics, baseline, valid_pixels,
)
```

### Step 4: Compute depth for semantic guidance
```python
depth = fB / (disp.abs() + 1e-10) * disp.sign()   # fB = focal_length × baseline
depth_weight = 1 / (depth + 1)                      # close → high weight
```

### Step 5: Normalize image
- ImageNet normalization: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)

### Step 6: Depth-guided sliding window prediction

**Pass A — Sliding window** (`slide_segment`):
- Crops: 320×320 with stride (160, 160)
- Each crop: `net(crop)` → feature codes → upsample to 320×320
- `cluster_probe(code, alpha=2, log_probs=True)` → log-probabilities
- Fold back + average overlapping regions
- Output: `out_slidingw` at 640×1280

**Pass B — Single image + flip**:
- Downscale to 320×640
- Forward pass + horizontally flipped forward pass → average codes
- Upsample to 640×1280
- `cluster_probe(code, alpha=2, log_probs=True)` → log-probabilities
- Output: `out_singleimg` at 640×1280

**Depth-weighted fusion**:
```python
weight = depth_weight.expand_as(out_slidingw)
out = out_singleimg * weight + out_slidingw * (1 - weight)
```
- Near objects (small depth → large weight): single-image (global context for large nearby things)
- Far objects (large depth → small weight): sliding window (local detail for small distant objects)

### Step 7: Dense CRF post-processing
```python
cluster_pred = batched_crf(pool, img, out).argmax(1).long()
```
CRF parameters (`crf.py`):
- MAX_ITER = 10
- Gaussian pairwise: POS_W=3, POS_XY_STD=1
- Bilateral pairwise: Bi_W=4, Bi_XY_STD=67, Bi_RGB_STD=3

### Step 8: Semantic-instance alignment
```python
panoptic_pred[..., 0] = align_semantic_to_instance(
    panoptic_pred[..., 0], panoptic_pred[..., 1].unsqueeze(0)
)["aligned_semantics"]
```
For each instance mask: find majority semantic class → assign to all pixels in that instance.
Ensures each instance has a single, consistent semantic label.

### Step 9: Thing/stuff statistics
```python
thingstuff_split.update(panoptic_pred)
```
Accumulates three distributions across all images:
1. Class distribution of pixels inside instance masks (pixel-weighted)
2. Class distribution per instance mask (one vote per mask)
3. Overall class distribution

Classes frequently inside instance masks → "thing"; rarely → "stuff".

### Step 10: Save
- Semantic: 8-bit PNG (values 0 to n_classes+extra_clusters-1)
- Instance: 8-bit PNG

## 9. Missing Information / Ambiguities

1. **Exact config values** (dim, extra_clusters, lhp_weight, etc.) live in a missing Hydra config
   file (`src/configs/local_config.yml`). Actual values are baked into the checkpoint
   via `save_hyperparameters()`.

2. **Which arch is used**: Default DepthG uses `arch="dino"` (no depth in backbone).
   The `dino_depth` variant exists but may be experimental.

3. **Whether CUPS retrains**: Strong evidence they do (CUPS metrics in training loop,
   Table 7b with different k values), but it could also be the official DepthG checkpoint
   with post-hoc overclustering.

## 10. Summary Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  DepthG Training (Unsupervised)              │
│                                                             │
│  DINO v1 ViT-S/8 ──────→ 384-dim features (frozen)         │
│         │                                                   │
│         ↓                                                   │
│  1x1 Conv head ──────→ dim-dim codes (trainable)            │
│         │                                                   │
│  Trained by:                                                │
│    • Contrastive correspondence (STEGO-style)               │
│    • Depth feature correlation (DepthG addition)            │
│    • LHP depth propagation (optional)                       │
│    • CRF spatial smoothness (optional)                      │
│                                                             │
│  ClusterLookup(dim, 27+extra) trained on detached codes     │
│  + CUPS Hungarian metric for overclustering evaluation       │
└─────────────────────────────────────────────────────────────┘
                          ↓ checkpoint
┌─────────────────────────────────────────────────────────────┐
│               CUPS Pseudo-Label Inference                    │
│                                                             │
│  Stereo pair → RAFT → disparity → depth                     │
│  depth_weight = 1/(depth+1)                                 │
│                                                             │
│  Image → [Sliding window 320×320] → cluster_probe(α=2)     │
│       → [Single image + flip]     → cluster_probe(α=2)     │
│                                                             │
│  Fusion: single_img × depth_weight + sliding × (1-weight)  │
│                                                             │
│  → Dense CRF (10 iter, bilateral)                           │
│  → argmax → semantic labels                                 │
│  → align to instance masks (majority vote)                  │
│  → accumulate thing/stuff statistics                        │
│  → save as PNG                                              │
└─────────────────────────────────────────────────────────────┘
```
