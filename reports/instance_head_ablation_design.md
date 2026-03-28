# Lightweight Instance Heads for Mobile Unsupervised Panoptic Segmentation

## Abstract

We address the problem of instance segmentation within a lightweight mobile panoptic model trained exclusively on unsupervised pseudo-labels. Our baseline model (RepViT-M0.9 + SimpleFPN, 4.9M params) predicts only semantic classes and derives thing instances post-hoc via connected components, achieving PQ_things=1.55 at epoch 4---far below the input pseudo-label quality of PQ_things=19.41. We attribute this to the fundamental inability of connected components to separate adjacent same-class objects. To close this gap, we propose three lightweight instance head designs: (A) a discriminative embedding head trained with contrastive loss, (B) a center-offset head following Panoptic-DeepLab [Cheng et al., CVPR 2020], and (C) a boundary prediction head. We design a systematic ablation: 3 individual runs, 3 pairwise combinations, and 1 full combination, totaling 7 experiments. All training uses unsupervised pseudo-labels only---no ground-truth annotations. We analyze expected complementarity between heads and recommend a prioritized execution order.

## 1. Introduction

Panoptic segmentation [Kirillov et al., CVPR 2019] requires both semantic classification and instance-level grouping. For "stuff" classes (road, sky, vegetation), connected components of the semantic map suffice---each contiguous region of a stuff class forms one segment. For "thing" classes (car, person, bicycle), individual objects must be distinguished even when they share the same semantic class and are spatially adjacent.

Our mobile model currently treats panoptic segmentation as a purely semantic problem, applying `scipy.ndimage.label()` connected components to derive thing instances at inference. This approach has a structural ceiling: touching or overlapping objects of the same class inevitably merge. In Cityscapes driving scenes, this is common---parked car rows, pedestrian groups, and bicycle clusters all violate the connected-component assumption.

CUPS [Hahn et al., CVPR 2025] addresses this with a Cascade Mask R-CNN [Cai and Vasconcelos, CVPR 2018] that predicts explicit bounding boxes and instance masks. However, this architecture is prohibitively large (~36M params for the detection heads alone) and requires Detectron2 infrastructure, making it unsuitable for mobile deployment.

We seek a middle ground: lightweight instance heads (<0.2M params each) that can be appended to our existing SimpleFPN decoder and trained on unsupervised instance pseudo-labels. The pseudo-labels are generated via depth-guided splitting of overclustered semantic segments (PQ_things=19.41, detailed in our pseudo-label pipeline report). No ground-truth annotations are used at any stage.

## 2. Instance Pseudo-Labels

Our instance pseudo-labels reside in `cups_pseudo_labels_v3/` as uint16 PNG images at 1024x2048 resolution. Each image contains instance IDs (0 = background/stuff, 1-N = individual thing instances), with a corresponding semantic PNG providing class labels.

These labels are generated through an unsupervised pipeline:
1. Overclustered semantic segmentation (k=80 K-means on CAUSE features)
2. Depth-guided splitting: within each thing-class region, segment by monocular depth discontinuities (SPIdepth, tau=0.20, A_min=1000)
3. Instance-semantic fusion with thing-stuff classification (threshold psi=0.08)

The resulting pseudo-labels achieve PQ=26.74, PQ_stuff=32.08, PQ_things=19.41 on Cityscapes val---surpassing CUPS's own pseudo-labels (PQ_things=17.70). The instance labels provide per-pixel instance IDs that serve as training targets for all three proposed heads.

## 3. Instance Head Designs

All three heads attach to the finest-scale SimpleFPN output (128-dim feature map at 1/4 input resolution). Each head consists of one depthwise-separable convolution followed by a 1x1 projection---matching the existing semantic head's design pattern for architectural consistency.

### 3.1 Option A: Discriminative Embedding Head

**Architecture.** A depthwise-separable conv block projects FPN features to a 16-dimensional embedding space per pixel.

**Training.** We employ the discriminative loss function [De Brabandere et al., 2017]:

```
L_embed = L_pull + L_push + L_reg

L_pull = (1/K) * sum_k [ (1/N_k) * sum_i ||e_i - mu_k||^2 ]        (pull to cluster mean)
L_push = (1/K(K-1)) * sum_{k!=l} max(0, 2*delta - ||mu_k - mu_l||)^2  (push cluster means apart)
L_reg  = (1/K) * sum_k ||mu_k||                                       (regularize means)
```

where K is the number of instances in the crop, N_k is the number of pixels in instance k, e_i is the embedding of pixel i, mu_k is the mean embedding of instance k, and delta is the margin (typically 1.5-2.0).

The loss operates only on thing-class pixels (trainIDs 11-18). Stuff pixels receive no embedding supervision. Instance IDs for computing pull/push groups come directly from the pseudo-instance labels.

**Inference.** For each thing-class region in the semantic map, apply mean-shift clustering [Comaniciu and Meer, TPAMI 2002] on the predicted 16-dim embeddings. Each resulting cluster constitutes one instance segment.

**Parameters.** +0.10M (128 -> 128 DW conv + 128 -> 16 pointwise).

**Strengths.** Flexible---can represent arbitrary instance shapes without bounding box assumptions. Robust to occlusion since each pixel independently encodes identity.

**Weaknesses.** Mean-shift clustering introduces a bandwidth hyperparameter and is computationally expensive at inference (~50ms at 512x1024 on mobile CPU). Non-differentiable clustering prevents end-to-end training of the grouping step.

### 3.2 Option B: Center-Offset Head

**Architecture.** Two parallel projections from FPN features:
- Center heatmap head: 1-channel output, sigmoid activation. Predicts the probability that each pixel is near an object center.
- Offset head: 2-channel output (dx, dy). Predicts a 2D vector from each pixel to its instance center.

**Training target generation.** From each pseudo-instance label:
1. Compute the centroid (y_c, x_c) of each instance mask.
2. Generate a Gaussian heatmap: H(y,x) = exp(-((y-y_c)^2 + (x-x_c)^2) / (2 * sigma^2)), where sigma is proportional to instance size (sigma = max(sqrt(area)/10, 4)).
3. Generate offset maps: O(y,x) = (y_c - y, x_c - x) for all pixels belonging to instance k.

**Loss.**
```
L_center = MSE(predicted_heatmap, target_heatmap)   (weighted: 10x on positive pixels)
L_offset = SmoothL1(predicted_offset, target_offset)  (only on thing-class pixels with valid instances)
L_instance_B = lambda_c * L_center + lambda_o * L_offset
```

**Inference.** (1) Find local maxima in the center heatmap above confidence threshold gamma_c. Each peak = one object center. (2) For each pixel, add predicted offset to pixel coordinates to get the "voted center." (3) Assign each pixel to the nearest detected center. This produces instance IDs without any clustering algorithm.

**Parameters.** +0.15M (128 -> 128 DW conv + 128 -> 1 center + 128 -> 2 offset).

**Strengths.** Fully deterministic inference (<2ms on mobile). The center detection provides an implicit "objectness" score. Can discover objects not in pseudo-labels if the model learns a general notion of "object center."

**Weaknesses.** Centroid representation struggles with highly non-convex instances (e.g., L-shaped objects). Small objects may produce weak center responses. Requires careful sigma tuning per instance size.

### 3.3 Option C: Boundary Head

**Architecture.** A single depthwise-separable conv block projecting to 1 channel with sigmoid activation. Predicts per-pixel probability of lying on an instance boundary.

**Training target generation.** From each pseudo-instance label:
1. Compute binary boundary map: B(y,x) = 1 if any 4-connected neighbor of (y,x) has a different instance ID.
2. Dilate boundaries by 1 pixel for robustness.
3. Apply only to thing-class pixels.

**Loss.**
```
L_boundary = BCE(predicted_boundary, target_boundary)
```
with class balancing (positive weight = N_neg / N_pos, typically ~20x) since boundary pixels are sparse (~2-5% of thing pixels).

**Inference.** (1) Predict semantic map + boundary map. (2) For each thing-class region, subtract predicted boundaries (threshold > 0.5). (3) Apply connected components to the boundary-subtracted mask. Adjacent same-class objects separated by a predicted boundary become distinct instances.

Alternatively, apply watershed transform using the boundary map as the ridge surface and the semantic map as the seed.

**Parameters.** +0.05M (128 -> 128 DW conv + 128 -> 1 pointwise).

**Strengths.** Extremely lightweight. Boundary maps are interpretable and can be visualized for debugging. Complements other heads by providing sharp edge information.

**Weaknesses.** Fragile---a single broken boundary pixel causes two instances to merge catastrophically. Watershed is sensitive to noise. Cannot separate objects that share a boundary with zero gap (e.g., tightly packed cars in a parking lot).

## 4. Ablation Design

### 4.1 Experimental Protocol

All runs share:
- **Backbone**: RepViT-M0.9, ImageNet-pretrained, frozen 5 epochs then unfrozen
- **Decoder**: SimpleFPN (128-dim)
- **Semantic head**: 19-class CE on `pseudo_semantic_mapped_k80` (always present)
- **Augmentations**: Full photometric pipeline (ColorJitter, GaussianBlur, Grayscale) + multi-scale RandomResizedCrop (0.5-1.5x) + horizontal flip --- as prescribed by our gap analysis
- **Optimizer**: AdamW, decoder LR=1e-3, backbone LR=1e-4, cosine decay
- **Epochs**: 50 (backbone frozen first 5)
- **Batch size**: 4 (single GTX 1080 Ti, 11GB VRAM)
- **Evaluation**: PQ, PQ_stuff, PQ_things, mIoU, SQ, RQ every 2 epochs against Cityscapes GT

Instance head losses are weighted relative to the semantic CE loss with tunable lambda parameters (default lambda=1.0 for each, tuned if needed).

### 4.2 Phase 1: Individual Ablations (3 runs)

| Run ID | Instance Head | Extra Params | Instance Loss | Inference |
|--------|--------------|-------------|---------------|-----------|
| **I-A** | 16-dim embedding | +0.10M (5.00M total) | Discriminative (pull/push, delta=1.5) | Mean-shift (bandwidth=1.5) |
| **I-B** | Center (1-ch) + Offset (2-ch) | +0.15M (5.05M total) | MSE (center) + SmoothL1 (offset) | Peak finding + offset grouping |
| **I-C** | Boundary (1-ch) | +0.05M (4.95M total) | Weighted BCE | Boundary subtraction + CC |

**Baseline**: Semantic-only model with connected components (current training run).

### 4.3 Phase 2: Pairwise Combinations (top 2-3 runs)

| Run ID | Heads | Rationale | Inference Strategy |
|--------|-------|-----------|-------------------|
| **I-AB** | Embedding + Center/Offset | Centers provide initial object proposals; embeddings refine ambiguous pixels at cluster boundaries | Offset grouping first, then embedding distance to resolve conflicts |
| **I-AC** | Embedding + Boundary | Embeddings cluster instances; boundaries prevent cluster leakage | Mean-shift with boundary-aware kernel that respects predicted edges |
| **I-BC** | Center/Offset + Boundary | Centers localize objects; boundaries sharpen their edges | Offset grouping with boundary-refined masks: split offset-grouped segments along predicted boundaries |

Phase 2 runs only the 2-3 most promising pairs based on Phase 1 results. If one head clearly dominates (e.g., I-B >> I-A and I-B >> I-C), we run only I-BC and I-AB.

### 4.4 Phase 3: Full Combination (1 run, conditional)

| Run ID | Heads | Total Params | Inference |
|--------|-------|-------------|-----------|
| **I-ABC** | All three | 5.20M | Center peaks → offset grouping → boundary splitting → embedding conflict resolution |

Phase 3 runs only if Phase 2 shows clear complementarity (i.e., I-XY > max(I-X, I-Y) by >1 PQ_things).

### 4.5 Expected Outcomes

| Run | Est. PQ_things | Reasoning |
|-----|---------------|-----------|
| Baseline (CC) | 2-5 | Merges adjacent same-class objects |
| I-A (embedding) | 8-12 | Good separation, noisy clustering |
| I-B (center/offset) | 10-15 | Strong object centers, deterministic |
| I-C (boundary) | 5-8 | Fragile to broken boundaries |
| I-BC (center + boundary) | 12-17 | Most complementary: "where" + "where it ends" |
| I-ABC (all) | 13-18 | Marginal over I-BC if embedding adds value |

The pseudo-label ceiling for PQ_things is 19.41 (input quality). A well-trained model should approach but may not exceed this ceiling without self-training, which enables the model to correct pseudo-label errors.

## 5. Complementarity Analysis

The three heads encode fundamentally different geometric properties:

| Head | Encodes | Failure Mode | Complementary With |
|------|---------|-------------|-------------------|
| Embedding | Per-pixel identity (which object?) | Clustering noise, bandwidth sensitivity | Boundary (prevents leakage) |
| Center/Offset | Object location (where is the center?) | Non-convex shapes, weak small-object centers | Boundary (sharpens edges) |
| Boundary | Instance edges (where do objects end?) | Broken boundaries, false positives at texture edges | Center (provides seed to anchor segments) |

**Most complementary pair: Center + Boundary (I-BC).** Centers provide robust object localization that boundaries alone cannot (a boundary without a seed is useless). Boundaries provide sharp edge delineation that offset grouping alone produces blurrily (offsets near object boundaries are ambiguous). Together they form a complete instance representation: "an object is here (center), and it extends to here (boundary)."

**Least complementary pair: Embedding + Center (I-AB).** Both encode "which object this pixel belongs to" --- embeddings via similarity, centers via offset proximity. They solve the same problem differently, so combining them yields diminishing returns.

## 6. Inference-Time Analysis

| Method | Time (512x1024, mobile CPU) | Time (512x1024, GPU) | Deterministic? |
|--------|---------------------------|---------------------|----------------|
| Connected components | ~5ms | <1ms | Yes |
| Mean-shift (16-dim) | ~50ms | ~10ms | No (kernel bandwidth) |
| Center peak + offset | ~2ms | <1ms | Yes (threshold only) |
| Watershed | ~15ms | ~5ms | Yes |
| Center + boundary split | ~5ms | ~1ms | Yes |

For real-time mobile inference (targeting 30fps = 33ms budget), center/offset-based methods are strongly preferred. Mean-shift is feasible only at reduced resolution (256x512: ~15ms) or with approximations (grid-based mean-shift). Watershed is marginal but acceptable.

## 7. Relation to Prior Work

**Panoptic-DeepLab** [Cheng et al., CVPR 2020] introduced the center-offset paradigm for bottom-up panoptic segmentation. They use a dual-ASPP decoder and train on fully-supervised Cityscapes GT. We adapt their instance head to a lightweight FPN decoder trained on unsupervised pseudo-labels.

**Panoptic-FPN** [Kirillov et al., CVPR 2019] uses a top-down instance branch with Mask R-CNN. We avoid this due to parameter cost and Detectron2 dependency.

**Real-Time Panoptic** [Hou et al., CVPR 2020] demonstrates that center-offset heads can run at real-time speeds. They achieve 30fps at 1024x2048 with a ResNet-50 backbone---our RepViT-M0.9 is 4.8x smaller and should be significantly faster.

**Discriminative embeddings** for instance segmentation were pioneered by [De Brabandere et al., 2017] and applied to panoptic segmentation by [Neven et al., CVPR 2019]. The mean-shift inference cost has motivated recent work on differentiable clustering [Liang et al., ECCV 2022], which we may explore if embedding heads prove effective.

## 8. Conclusion

We propose three lightweight instance head designs totaling <0.2M extra parameters each, all trainable on unsupervised pseudo-labels. Our ablation plan systematically evaluates individual heads, pairwise combinations, and the full ensemble. The center-offset head (Option B) is our expected top performer due to its deterministic inference and strong localization, while the center-boundary combination (I-BC) is expected to achieve the highest PQ_things through complementary geometric reasoning. All experiments remain fully unsupervised---pseudo-labels from depth-guided splitting serve as training targets without any ground-truth annotations.

## References

- [Hahn et al., 2025] Hahn, O., et al. "Scene-Centric Unsupervised Panoptic Segmentation." CVPR 2025.
- [Cheng et al., 2020] Cheng, B., et al. "Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation." CVPR 2020.
- [Kirillov et al., 2019] Kirillov, A., et al. "Panoptic Segmentation." CVPR 2019.
- [Wang et al., 2024] Wang, A., et al. "RepViT: Revisiting Mobile CNN From ViT Perspective." CVPR 2024.
- [De Brabandere et al., 2017] De Brabandere, B., et al. "Semantic Instance Segmentation with a Discriminative Loss Function." arXiv 2017.
- [Cai and Vasconcelos, 2018] Cai, Z. and Vasconcelos, N. "Cascade R-CNN: Delving into High Quality Object Detection." CVPR 2018.
- [Neven et al., 2019] Neven, D., et al. "Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth." CVPR 2019.
- [Hou et al., 2020] Hou, R., et al. "Real-Time Panoptic Segmentation from Dense Detections." CVPR 2020.
- [Comaniciu and Meer, 2002] Comaniciu, D. and Meer, P. "Mean Shift: A Robust Approach Toward Feature Space Analysis." TPAMI 2002.
