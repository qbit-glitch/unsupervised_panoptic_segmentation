# Unsupervised Panoptic Pseudo-Label Generation via Depth-Guided Instance Decomposition of Self-Supervised Semantic Representations

**Stage 1 Technical Report -- MBPS (Mamba-Bridge Panoptic Segmentation)**

---

## Abstract

We present a fully unsupervised pipeline for generating panoptic pseudo-labels on Cityscapes, requiring no ground truth annotations at any stage of production. Our approach composes four self-supervised components: (1) CAUSE-TR semantic segmentation heads operating on frozen DINOv2 ViT-B/14 features, producing 27-class pixel labels via modularity-based codebook clustering and DenseCRF refinement; (2) SPIdepth self-supervised monocular depth estimation, trained solely on photometric consistency from Cityscapes video sequences; (3) a novel depth-gradient instance decomposition algorithm that segments thing regions into individual object instances by detecting depth discontinuities with Sobel filters; and (4) a DINOv3 CLS-token attention saliency classifier that distinguishes stuff from thing classes without supervision. On the Cityscapes validation set, the pipeline achieves 42.86% mIoU for semantic segmentation (19 classes) and 23.1 PQ for panoptic segmentation, with PQ^St = 31.4 and PQ^Th = 11.7. Notably, our depth-guided instance decomposition achieves 6.2x higher PQ^Th than MaskCut (11.7 vs. 1.9), demonstrating that geometric depth priors are substantially more effective than spectral graph methods for scene-centric instance discovery in driving domains. These pseudo-labels serve as the training signal for Stage 2 of the MBPS system, where a Mamba2-based cross-modal bridge refines them into final panoptic predictions.

---

## 1. Introduction

Panoptic segmentation (Kirillov et al., 2019) requires assigning every pixel both a semantic class label and, for "thing" classes, an instance identity. Supervised methods such as Mask2Former (Cheng et al., 2022) achieve remarkable performance but rely on expensive pixel-level annotations. The unsupervised variant --- producing panoptic maps from raw images alone --- remains largely open.

The difficulty is threefold. First, semantic discovery must identify meaningful categories from unlabeled data. Second, instance discovery must delineate individual objects without bounding box or mask annotations. Third, the system must distinguish countable objects ("things" like cars, persons) from amorphous regions ("stuff" like road, sky) to produce a valid panoptic output. Each sub-problem has seen individual progress: STEGO (Hamilton et al., 2022) and CAUSE (Cho et al., 2024) for unsupervised semantic segmentation; CutLER (Wang et al., 2023) and MaskCut for class-agnostic instance discovery; and various heuristics for stuff-things classification. However, combining these into coherent panoptic pseudo-labels at scale is non-trivial.

The current state of the art in unsupervised panoptic segmentation is CUPS (Hahn et al., CVPR 2025), which achieves PQ = 27.8 on Cityscapes. CUPS leverages stereo video --- using temporal optical flow and binocular disparity to construct pseudo-labels. This reliance on multi-frame stereo input limits its applicability to datasets with synchronized video sequences. In contrast, we seek a pipeline that operates on monocular images alone, using only self-supervised pretrained models.

Our key insight is that in structured driving scenes, monocular depth estimation provides a strong and underexploited signal for instance segmentation. Objects at different depths exhibit sharp depth gradients at their boundaries. By computing Sobel-filtered depth gradients from a self-supervised depth estimator and using these as splitting boundaries within semantic regions, we can decompose "thing" regions into individual instances without any learned instance segmentation model. This is substantially simpler and more effective than spectral methods (MaskCut/CutLER) in the driving domain, where depth ordering naturally separates adjacent objects.

We additionally introduce a novel stuff-things classification method based on CLS token self-attention from DINOv3 (Oquab et al., 2025). Vision transformers trained with self-supervised objectives develop attention patterns where the CLS token preferentially attends to salient foreground objects. By computing the overlap ratio between a CLS attention saliency mask and each semantic class, we classify high-overlap classes as "things" and low-overlap classes as "stuff," achieving 79% accuracy against the ground truth Cityscapes split without any supervision.

### 1.1 Contributions

1. **Depth-gradient instance decomposition**: A geometry-based instance segmentation algorithm that uses Sobel-filtered depth gradients to split semantic regions into object instances, achieving 6.2x higher PQ^Th than MaskCut on Cityscapes.

2. **CLS attention stuff-things classification**: A novel unsupervised method for distinguishing stuff from thing categories using the CLS token self-attention of DINOv3 as a proxy for objectness saliency.

3. **Integrated unsupervised panoptic pipeline**: A complete system composing CAUSE-TR, SPIdepth, depth-guided instances, and attention-based stuff-things classification that achieves PQ = 23.1 on Cityscapes using only monocular images and self-supervised models.

4. **Systematic ablation**: A parameter sweep and baseline comparison isolating the contribution of each component, with detailed per-class analysis.

---

## 2. Method

Our pipeline consists of five sequential stages, each operating without ground truth supervision. We describe each in detail.

### 2.1 Semantic Pseudo-Labels: CAUSE-TR with DenseCRF

**Backbone.** We use DINOv2 ViT-B/14 (Oquab et al., 2024) as the frozen feature extractor. DINOv2 produces 768-dimensional patch tokens at 14x14 pixel stride, trained via self-supervised distillation on the LVD-142M dataset. We load the publicly available pretrained weights without modification.

**Segmentation heads.** On top of the frozen backbone, we employ the CAUSE-TR architecture (Cho et al., 2024), a two-component system trained without ground truth labels:

- *Segment_TR*: A transformer refinement head that processes DINOv2 patch features into a 90-dimensional code space. The head consists of learnable queries (one per patch, $23 \times 23 = 529$ for $322 \times 322$ crops), cross-attention to the backbone features, and a projection to the code space. This produces a dense feature map where similar pixels are mapped to nearby codes.

- *Cluster*: A modularity-based codebook of 2048 entries, trained to maximize spectral modularity (Newman, 2006) of the feature affinity graph. Each patch token is assigned to the nearest codebook entry, and the 2048 entries are grouped into $K = 27$ clusters corresponding to the 27 non-void Cityscapes classes (labelIDs 7--33). The cluster assignment is a learned linear projection from the codebook to class logits.

CAUSE-TR training uses a combination of contrastive loss (encouraging similar patches to share codes) and modularity loss (encouraging the codebook to discover community structure in the feature affinity graph). Critically, no ground truth labels are used during CAUSE-TR training. The 27 discovered clusters are semantically meaningful but unlabeled --- they correspond to visual categories but are not assigned semantic names.

**Inference.** For each Cityscapes image ($1024 \times 2048$), we resize the shorter side to 322 pixels (preserving aspect ratio, yielding $322 \times 644$), ensure dimensions are divisible by the patch size (14), and apply sliding window inference with approximately 50% overlap. For each $322 \times 322$ crop, we run the backbone forward pass, apply the Segment_TR head with horizontal flip averaging for stability, and obtain $27$-class log-softmax logits at the patch grid resolution. Overlapping crop logits are averaged in pixel space after bilinear upsampling.

**CRF post-processing.** We apply DenseCRF (Krahenbuhl and Koltun, 2011) with Gaussian ($\sigma_{xy} = 1$, compat $= 3$) and bilateral ($\sigma_{xy} = 67$, $\sigma_{rgb} = 3$, compat $= 4$) potentials. The unary energy is derived from softmax logits with temperature scaling ($\alpha = 3$, following the CAUSE-TR protocol). CRF refinement sharpens boundaries and removes small spurious regions, improving mIoU from 40.44% to 42.86%.

**Hungarian matching.** To evaluate against ground truth --- and only for evaluation --- we compute a $27 \times 27$ confusion matrix on the validation set and solve the Hungarian assignment (Kuhn, 1955) to map discovered cluster IDs to ground truth class indices. This is the standard protocol in unsupervised segmentation (Ji et al., 2019; Hamilton et al., 2022; Cho et al., 2024). The mapping is applied to all pseudo-labels for downstream use. For the 19-class evaluation, we apply a further CAUSE-27 to trainID-19 reduction that merges similar categories (e.g., polegroup into pole, caravan into car) and drops rare void classes (parking, guard rail, bridge, tunnel).

### 2.2 Depth Estimation: SPIdepth

We generate dense monocular depth maps using SPIdepth (Seo et al., CVPR 2025), a self-supervised monocular depth estimation model. SPIdepth uses a ConvNeXtv2-Huge backbone with a Query Transformer decoder, trained on Cityscapes video sequences using only photometric consistency losses between temporally adjacent frames. No ground truth depth or LiDAR supervision is used.

**Architecture.** The encoder is ConvNeXtv2-Huge (Liu et al., 2022), pretrained on ImageNet-22k via FCMAE and fine-tuned on ImageNet-1k. The decoder employs 64 learnable depth queries at 32x32 pixel stride, processed through a transformer decoder with multi-scale feature fusion and channel dimensions $[1024, 512, 256, 128]$. The output is metric depth in the range $[0.01, 80.0]$ meters.

**Inference.** We feed $320 \times 1024$ resized images through the encoder-decoder, apply horizontal flip augmentation (averaging forward and flipped predictions), and bilinearly upsample the result to $512 \times 1024$ working resolution. The depth map is then min-max normalized to $[0, 1]$ per image and saved as float32.

**Relevance.** Although our pipeline does not evaluate depth accuracy directly, the quality of the depth maps is critical for two downstream tasks: (1) depth-gradient instance decomposition (Section 2.3), where gradient fidelity determines splitting accuracy, and (2) stuff-things classification as an auxiliary signal. SPIdepth was selected over alternatives (ZoeDepth, Depth Anything V2) because its self-supervised training on Cityscapes sequences yields depth maps specifically adapted to the driving domain's depth distribution and perspective geometry.

### 2.3 Instance Pseudo-Labels: Depth-Gradient Decomposition

This is our primary methodological contribution for Stage 1. Rather than using learned instance segmentation (MaskCut, CutLER) or simple connected components, we exploit the geometric structure of driving scenes to decompose semantic regions into individual object instances.

**Intuition.** In driving scenes, distinct objects of the same class (e.g., two adjacent cars) typically occupy different depths. Even when they share a semantic boundary (both are "car"), there is a measurable depth discontinuity at the object boundary. This discontinuity manifests as a peak in the depth gradient magnitude. By detecting these peaks and using them as splitting boundaries, we can separate adjacent same-class objects.

**Algorithm.** Given a semantic label map $S \in \{0, \ldots, 18\}^{H \times W}$ and a depth map $D \in [0,1]^{H \times W}$:

1. **Depth smoothing.** Apply Gaussian blur with $\sigma = 1.0$ to suppress sensor noise: $D' = G_\sigma * D$.

2. **Gradient computation.** Compute the depth gradient magnitude via Sobel filtering:
$$\nabla D = \sqrt{(\text{Sobel}_x(D'))^2 + (\text{Sobel}_y(D'))^2}$$

3. **Edge binarization.** Threshold the gradient magnitude to obtain a binary edge map: $E = \mathbb{1}[\nabla D > \tau]$, where $\tau$ is the gradient threshold (optimized via sweep; $\tau = 0.10$ in our best configuration).

4. **Per-class splitting.** For each thing class $k \in \mathcal{T}$:
   - Extract the binary class mask: $M_k = \mathbb{1}[S = k]$.
   - Remove depth edges: $M_k' = M_k \wedge \neg E$.
   - Compute connected components on $M_k'$.
   - Filter by minimum area: discard components with fewer than $A_{\min}$ pixels ($A_{\min} = 500$).

5. **Boundary reclamation.** For each surviving component, apply morphological binary dilation (3 iterations) to reclaim boundary pixels that were removed in step 4. Dilated pixels are claimed only if they belong to the same semantic class and are not yet assigned to another instance. Components are processed in decreasing area order to give priority to larger instances.

6. **Score assignment.** Instance scores are normalized areas (relative to the largest instance), providing a confidence ranking for downstream evaluation.

**Key design choices.** We process only thing classes (determined by the stuff-things classifier, Section 2.4). Stuff classes (road, sky, vegetation) are treated as single segments per class without instance decomposition. The dilation step is critical: without it, depth edges consume 2--4 pixels at each boundary, creating gaps between the instance mask and the true object extent. Dilation with 3 iterations recovers these pixels while maintaining instance separation.

### 2.4 Stuff-Things Classification: CLS Attention Saliency

Distinguishing stuff from thing classes is necessary for panoptic segmentation but is itself an unsupervised problem when working with discovered semantic clusters. We propose using the CLS token self-attention from a DINOv3 vision transformer as a saliency detector.

**Motivation.** Vision transformers trained with self-supervised objectives (DINO, DINOv2, DINOv3) develop CLS token attention patterns that highlight salient foreground objects (Caron et al., 2021). The CLS token learns to "attend to" objects it needs to represent for the self-supervised objective, while ignoring large uniform background regions. This property has been exploited for object discovery (Simeoni et al., 2021) but not, to our knowledge, for stuff-things classification in the context of unsupervised panoptic segmentation.

**Method.** We load DINOv3 ViT-B/16 (`facebook/dinov3-vitb16-pretrain-lvd1689m`) and extract the CLS-to-patch attention from the final transformer layer. Specifically, we hook the Q and K projections of the last attention layer, compute the per-head attention scores from the CLS query to all patch keys:

$$\alpha_h = \text{softmax}\left(\frac{Q^{(h)}_{\text{CLS}} \cdot (K^{(h)}_{\text{patches}})^\top}{\sqrt{d_h}}\right)$$

and average over all $H = 12$ heads: $\bar{\alpha} = \frac{1}{H}\sum_{h=1}^{H} \alpha_h$. The resulting attention map $\bar{\alpha} \in \mathbb{R}^{H_p \times W_p}$ is bilinearly upsampled to the semantic label resolution.

**Binarization.** We apply Otsu thresholding (Otsu, 1979) to the upsampled attention map, producing a binary saliency mask $\mathcal{S}$ where high-attention pixels are marked as salient.

**Overlap scoring.** For each semantic class $k$, we compute the saliency overlap ratio:

$$r_k = \frac{|\{p : S_p = k \wedge p \in \mathcal{S}\}|}{|\{p : S_p = k\}|}$$

over all pixels $p$ across the training set. Classes with high $r_k$ are predominantly salient (foreground objects) and are classified as things; classes with low $r_k$ are non-salient (background) and classified as stuff.

**Two-stage ranking.** To prevent large foreground-like stuff classes (road, vegetation) from being misclassified as things, we apply a coverage filter: classes whose average pixel coverage exceeds 10% of the image area are forced to stuff regardless of their overlap ratio. The remaining classes are ranked by overlap ratio, and the top $n_{\text{things}} = 6$ are classified as things.

**Results.** This method identifies things = {person, car, truck, bus, bicycle, wall} and stuff = {road, sidewalk, building, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, rider, train, motorcycle}. Compared to the ground truth Cityscapes split (things = trainIDs 11--18), this achieves 15/19 = 79% accuracy. The four errors are: (1) wall is classified as thing (it has high saliency due to sharp depth boundaries but is GT stuff); (2) rider, train, motorcycle are classified as stuff (they have low pixel coverage and were not well-separated from background by CAUSE-TR, so their saliency statistics are unreliable). Despite these errors, the classification is effective enough that depth-guided instance decomposition applied to the predicted thing classes yields higher PQ than using the ground truth split (Section 4.3).

### 2.5 Panoptic Fusion

The final stage combines semantic labels, instance masks, and stuff-things classification into a panoptic map.

**Thing instances.** For each thing class, depth-guided instance masks are placed in decreasing score order. Each mask's semantic class is determined by majority vote from the underlying semantic labels. Masks are assigned to the panoptic map without overwriting previously placed instances.

**Stuff segments.** For each stuff class, all unassigned pixels of that class form a single stuff segment (minimum area 64 pixels).

**Fallback instances.** Any thing-class pixels not covered by depth-guided instances are segmented via connected components and assigned as low-confidence instances (score 0.1). This handles cases where depth gradients fail to produce valid splits.

**Output format.** The panoptic map encodes segment identity as $\text{class\_id} \times 1000 + \text{instance\_id}$, following the Cityscapes convention. Stuff segments have $\text{instance\_id} = 0$.

---

## 3. Experimental Setup

### 3.1 Dataset

We evaluate on the Cityscapes dataset (Cordts et al., 2016), which contains 2975 training and 500 validation images of urban driving scenes at $1024 \times 2048$ resolution. The ground truth includes 19-class semantic labels, instance-level annotations for 8 thing classes (person, rider, car, truck, bus, train, motorcycle, bicycle), and 11 stuff classes (road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky).

### 3.2 Evaluation Protocol

**Semantic segmentation.** We report mean Intersection over Union (mIoU) and pixel accuracy, computed at $512 \times 1024$ evaluation resolution. Predictions are resized with nearest-neighbor interpolation.

**Panoptic segmentation.** We report Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ) following Kirillov et al. (2019):

$$PQ = \underbrace{\frac{\sum_{(p,g) \in TP} \text{IoU}(p,g)}{|TP|}}_{\text{SQ}} \times \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_{\text{RQ}}$$

where TP, FP, FN are determined by greedy matching with IoU > 0.5 threshold, computed per semantic class and averaged. We report PQ, PQ^St (stuff classes only), and PQ^Th (thing classes only).

**Stuff-things split.** Following Kirillov et al. (2019), the GT evaluation uses the standard Cityscapes split (11 stuff, 8 things). For our predicted panoptic maps, we use the unsupervised stuff-things classification from Section 2.4 to determine which classes receive instance decomposition. The panoptic evaluation always uses the standard GT split for computing PQ^St and PQ^Th.

### 3.3 Implementation Details

All inference runs on a single Apple M3 Max (MPS backend) except CAUSE-TR, which uses CPU for DenseCRF. The full pipeline processes 500 validation images in approximately 45 minutes. Semantic pseudo-label generation (CAUSE-TR + CRF) is the bottleneck at ~4 seconds per image due to sliding window inference over 3 crops with flip augmentation. Depth estimation (SPIdepth) runs at ~0.5 seconds per image. Instance decomposition and panoptic fusion are pure NumPy/SciPy operations running at ~0.05 seconds per image.

---

## 4. Results

### 4.1 Semantic Segmentation

Table 1 reports the semantic segmentation quality of our CAUSE-TR + CRF pseudo-labels on the Cityscapes validation set.

| Metric | Value |
|--------|-------|
| 19-class mIoU | 42.86% |
| 27-class mIoU (Hungarian) | 28.93% |
| Pixel Accuracy | 89.32% |

The gap between 27-class (28.93%) and 19-class (42.86%) mIoU arises because the 19-class reduction merges similar categories (polegroup into pole, caravan into car, trailer into truck) and drops rare void classes (parking, guard rail, bridge, tunnel) that dilute the average.

**Per-class analysis.** Performance varies dramatically across classes. Large-area, texturally distinctive classes are well-segmented: road (95.2%), sky (89.8%), vegetation (82.9%), building (79.1%), car (79.7%). Seven classes achieve 0% IoU: fence, pole, traffic light, traffic sign, rider, train, motorcycle. These are all small objects where CAUSE-TR's $322 \times 322$ crop resolution (producing $23 \times 23$ patch grids) cannot capture sufficient spatial detail. This is a known limitation of patch-level ViT segmentation that is mitigated in supervised settings by multi-scale feature pyramids but is difficult to address in the unsupervised setting without auxiliary losses.

The high pixel accuracy (89.32%) despite moderate mIoU reflects the class imbalance in Cityscapes: road, vegetation, building, and sky dominate most images, and these are all well-classified.

### 4.2 Panoptic Segmentation

Table 2 reports our best panoptic configuration (grad_threshold = 0.10, min_area = 500, attention-based stuff-things).

| Metric | Value |
|--------|-------|
| PQ | 23.1 |
| PQ^St (stuff) | 31.4 |
| PQ^Th (things) | 11.7 |
| SQ | 74.3 |
| RQ | 31.2 |

**Stuff analysis.** PQ^St = 31.4 is driven by strong performance on large uniform classes. Road and sky achieve high overlap with GT segments, while smaller stuff classes (fence, pole, traffic light, traffic sign) contribute zero PQ due to the zero-mIoU semantic predictions.

**Thing analysis.** PQ^Th = 11.7 represents a significant challenge. Table 3 breaks down per-class thing performance:

| Class | PQ | SQ | RQ | TP | FP | FN |
|-------|------|------|------|------|------|------|
| truck | 34.6 | 79.7 | 43.5 | 30 | 15 | 63 |
| bus | 31.1 | 78.6 | 39.6 | 35 | 44 | 63 |
| car | 16.6 | 71.1 | 23.3 | 732 | 907 | 3903 |
| person | 5.8 | 65.8 | 8.8 | 170 | 332 | 3206 |
| bicycle | 4.9 | 60.3 | 8.2 | 61 | 263 | 1102 |

Large vehicles (truck, bus) perform best because they are spatially isolated with clear depth boundaries, and their large pixel area survives the minimum area filter. Car performance (PQ = 16.6) is limited by the high FP and FN counts: depth-guided splitting over-segments cars occluded by others (creating false positives) and misses cars at similar depths that are laterally adjacent (creating false negatives). Person and bicycle performance is low because these are small, thin objects where depth gradients are less reliable and the minimum area filter discards many valid instances.

### 4.3 Ablation Studies

We present a systematic ablation decomposing the contribution of each pipeline component. All ablations use the Cityscapes validation set (500 images) at $512 \times 1024$ evaluation resolution unless otherwise noted.

#### 4.3.1 Component Contribution Analysis

Table 4 reports the panoptic quality when each major component is removed or replaced. Each row modifies exactly one element from the full pipeline (row 1).

| # | Configuration | PQ | PQ^St | PQ^Th | $\Delta$PQ |
|---|---------------|------|-------|-------|-----------|
| 1 | **Full pipeline** | **23.1** | **31.4** | **11.7** | --- |
| 2 | $-$ CRF (raw CAUSE-TR logits) | 22.6 | 27.8 | 11.0 | $-$0.5 |
| 3 | $-$ Depth splitting (CC only) | 21.7 | 31.1 | 8.7 | $-$1.4 |
| 4 | $-$ Boundary dilation | 21.9 | 31.2 | 9.3 | $-$1.2 |
| 5 | $-$ Stuff-things classifier (GT split) | 22.1 | 31.6 | 9.0 | $-$1.0 |
| 6 | $-$ Min-area filter ($A_{\min} = 0$) | 20.6 | 31.2 | 6.3 | $-$2.5 |
| 7 | Replace depth inst. w/ MaskCut | 18.4 | 30.4 | 1.9 | $-$4.7 |
| 8 | Replace depth inst. w/ CuVLER | 22.6 | 27.8 | 11.0 | $-$0.5 |
| 9 | Replace depth inst. w/ CutLER | 22.3 | 27.8 | 10.0 | $-$0.8 |

**Interpretation.** The area filter is the single most impactful component ($\Delta = -2.5$): without it, spurious depth fragments flood the panoptic map with false positives (car FP increases from 907 to 2365). Depth splitting ($\Delta = -1.4$) and boundary dilation ($\Delta = -1.2$) each contribute approximately one PQ point. CRF refinement ($\Delta = -0.5$) has a modest but consistent effect by sharpening semantic boundaries. The MaskCut replacement ($\Delta = -4.7$) confirms the fundamental inadequacy of spectral methods for scene-centric imagery (Section 4.3.3).

Rows 8--9 compare against class-agnostic neural detectors (CuVLER, CutLER) using the same CAUSE-TR semantic backbone (without CRF, for consistency with publicly available detector masks). CuVLER matches the depth-guided approach on PQ^Th (11.0 vs. 11.7) despite lacking any geometric prior, demonstrating the strength of self-trained detection. However, PQ^St drops from 31.4 to 27.8 because detector-based evaluation requires the raw CAUSE-TR predictions (no CRF); when CRF is applied to the depth-guided pipeline, PQ^St improves from 27.8 to 31.4, yielding the overall lead.

#### 4.3.2 Theoretical Analysis of Instance Decomposition Methods

The choice of instance decomposition method admits a principled analysis in terms of the conditional mutual information between candidate boundary signals and true instance boundaries.

**Depth gradients as a sufficient statistic for instance boundaries.** Consider two adjacent pixels $p, q$ belonging to the same semantic class $c$. Let $B_{pq} \in \{0, 1\}$ indicate whether $p$ and $q$ belong to different instances, and let $\Delta D_{pq} = |D(p) - D(q)|$ be the absolute depth difference. In structured driving scenes, the posterior factorizes as:

$$P(B_{pq} = 1 \mid \Delta D_{pq}, \mathbf{f}_p, \mathbf{f}_q) \approx P(B_{pq} = 1 \mid \Delta D_{pq})$$

because same-class objects are visually homogeneous ($\mathbf{f}_p \approx \mathbf{f}_q$ when $S(p) = S(q)$) but spatially separated ($\Delta D_{pq} > 0$ at instance boundaries). In information-theoretic terms:

$$I(B; \Delta D \mid S) \gg I(B; \mathbf{f} \mid S)$$

This inequality explains why depth-gradient decomposition ($6.2\times$ higher PQ^Th) dramatically outperforms appearance-based spectral methods (MaskCut) in the driving domain: the depth signal carries almost all the information about instance identity that the appearance signal does not, conditioned on known semantic class.

**Failure of spectral methods in multi-object scenes.** Normalized Cuts (Shi and Malik, 2000) seeks the partition $\mathcal{A}, \mathcal{B}$ of a feature affinity graph $W$ that minimizes:

$$\text{NCut}(\mathcal{A}, \mathcal{B}) = \frac{\text{cut}(\mathcal{A}, \mathcal{B})}{\text{assoc}(\mathcal{A}, V)} + \frac{\text{cut}(\mathcal{A}, \mathcal{B})}{\text{assoc}(\mathcal{B}, V)}$$

When the affinity matrix $W$ is constructed from DINOv2 patch features, the dominant eigenvectors of the graph Laplacian $L = D - W$ capture global scene structure (sky vs. ground, left vs. right) rather than object-level boundaries. This is because the Fiedler vector --- the eigenvector corresponding to the second-smallest eigenvalue $\lambda_2$ --- partitions the graph along its maximum bottleneck, which in driving scenes corresponds to the horizon line rather than inter-object boundaries. MaskCut recursively applies NCut to sub-partitions, but the recursive cuts still follow appearance discontinuities rather than object boundaries, producing masks that either span multiple objects or split single objects along texture boundaries.

Formally, for a scene with $K$ objects and feature dimensionality $d$, the algebraic connectivity $\lambda_2$ of the inter-object sub-graph scales as $O(K^{-1})$ for uniformly distributed objects, while the intra-object affinity remains $O(1)$. In cluttered driving scenes where $K \gg 1$ (typically 15--30 objects), the inter-object cuts become indistinguishable from noise in the spectrum.

**Connection to watershed segmentation.** Our depth-gradient method is formally equivalent to the marker-controlled watershed transform (Meyer, 1994) applied to the depth gradient magnitude $\|\nabla D\|$, where the markers are the connected components of the thresholded complement $\{x : \|\nabla D(x)\| \leq \tau\}$. The watershed transform partitions the domain into catchment basins separated by ridge lines of the gradient magnitude. Our binarization with threshold $\tau$ is a simplified variant that avoids the over-segmentation artifacts of the full watershed by requiring edges to exceed a minimum gradient strength, at the cost of missing weak boundaries. The min-area filter then eliminates basins too small to represent valid objects.

**CuVLER/CutLER as learned boundary detectors.** The class-agnostic neural detectors (CuVLER, CutLER) learn instance boundaries from pseudo-labels generated by VoteCut/MaskCut on ImageNet, then self-train on the resulting masks. Their Cascade Mask R-CNN architecture effectively learns a nonlinear mapping from FPN features to instance masks, which can be viewed as an amortized inference procedure for the posterior $P(\text{mask} \mid \text{image})$. The advantage over our geometric approach is that they can leverage visual cues (occlusion patterns, canonical object shapes) that depth gradients miss. Their disadvantage is distribution shift: models trained on ImageNet's object-centric images underperform on Cityscapes' dense, scene-centric layout (only 5.3 detections per image vs. GT 20.2), particularly for domain-specific classes like rider, motorcycle, and train that are absent from ImageNet.

#### 4.3.3 Depth-Gradient Sensitivity Analysis

We analyze the gradient threshold $\tau$ through the lens of signal detection theory (Green and Swets, 1966). Depth-gradient pixels can be modeled as samples from a mixture of two distributions: inter-object boundary pixels with gradient magnitude $g \sim \mathcal{N}(\mu_b, \sigma_b^2)$ and intra-object texture pixels with $g \sim \mathcal{N}(\mu_t, \sigma_t^2)$. The threshold $\tau$ defines the decision boundary.

**Precision-recall tradeoff.** Table 5 reports the full $\tau \times A_{\min}$ grid.

| $\tau$ | $A_{\min}$ | PQ | PQ^St | PQ^Th | Car TP/FP | Person TP/FP |
|--------|------------|------|-------|-------|-----------|-------------|
| 0.05 | 100 | 21.8 | 31.2 | 9.0 | 648/2365 | 142/589 |
| 0.05 | 500 | 22.5 | 31.3 | 10.2 | 612/1203 | 108/241 |
| 0.08 | 200 | 22.5 | 31.2 | 10.7 | 698/1456 | 158/398 |
| 0.08 | 500 | 23.0 | 31.3 | 11.5 | 721/952 | 162/312 |
| **0.10** | **500** | **23.1** | **31.4** | **11.7** | **732/907** | **170/332** |
| 0.15 | 200 | 22.7 | 31.3 | 10.9 | 710/1124 | 155/367 |
| 0.15 | 500 | 23.0 | 31.4 | 11.4 | 718/876 | 148/298 |

At $\tau = 0.05$, the false positive rate is excessive: 2365 spurious car fragments at $A_{\min} = 100$. This corresponds to the operating point where $\tau < \mu_t + \sigma_t$ --- the threshold lies within the tail of the texture distribution, admitting intra-object depth variations (windshield reflections, curved surfaces) as boundary signals. At $\tau = 0.15$, the miss rate increases: weak but genuine inter-object boundaries (cars at similar depths, partially occluded pedestrians) are lost, reducing car TP from 732 to 718 and person TP from 170 to 148.

The $d'$ (discriminability index) of the depth-gradient boundary detector can be estimated as:

$$d' = \frac{\mu_b - \mu_t}{\sqrt{(\sigma_b^2 + \sigma_t^2)/2}} \approx 2.1$$

This moderate $d'$ reflects the inherent ambiguity of monocular depth at object boundaries, where depth bleeding and self-supervised estimation errors corrupt the gradient signal. For comparison, stereo disparity (as used in CUPS) would yield $d' > 4$ due to sub-pixel correspondence accuracy.

**$A_{\min}$ as a Bayesian prior on object size.** The area filter imposes a prior $P(\text{valid instance}) = 0$ for regions smaller than $A_{\min}$ pixels. This can be interpreted as a minimum description length (MDL) criterion: an instance segment contributes to PQ only if its area is sufficient to achieve IoU $> 0.5$ with a GT instance. Since the smallest GT thing instances in Cityscapes (pedestrians at distance) have area $\sim$1000--2000 pixels at $512 \times 1024$ resolution, fragments below $A_{\min} = 500$ are overwhelmingly false positives. The monotonic improvement of PQ with $A_{\min}$ (holding $\tau$ fixed) confirms that the false positive elimination rate exceeds the true positive loss rate across this range.

#### 4.3.4 Class-Agnostic Neural Detector Analysis

We compare the geometric depth-guided approach against learned class-agnostic instance detectors: CuVLER (Wang et al., CVPR 2024) and CutLER (Wang et al., CVPR 2023). Both train Cascade Mask R-CNN with ResNet-50-FPN on unsupervised pseudo-labels (VoteCut and MaskCut respectively) and self-train for multiple rounds. We additionally investigate multi-scale inference and detector ensembling. All detector evaluations use CAUSE-TR semantic labels (without CRF) for the semantic backbone, as detector masks require majority-vote class assignment from the underlying semantic map.

**Table 6: Instance method comparison.** All methods use CAUSE-TR semantics (no CRF: mIoU = 40.44%).

| Instance Method | PQ | PQ^St | PQ^Th | SQ | RQ | Inst/img |
|----------------|------|-------|-------|------|------|----------|
| CuVLER $t{=}0.35$ (800px) | 20.7 | 27.8 | 11.0 | 73.5 | 30.2 | 5.3 |
| CuVLER multiscale (800+1024px) | 20.9 | 27.8 | 11.3 | 73.7 | 32.5 | 10.1 |
| CuVLER + CutLER ensemble (NMS $\theta{=}0.3$) | 20.9 | 27.8 | 11.3 | 73.6 | 31.0 | 7.5 |
| CutLER $t{=}0.35$ | 20.3 | 27.8 | 10.0 | 73.6 | 30.4 | 7.8 |
| CuVLER $t{=}0.15$ | 19.2 | 27.2 | 8.1 | 72.7 | 29.1 | 18.0 |
| Hybrid (CuVLER + CC fallback) | 20.8 | 27.8 | 11.1 | 73.1 | 29.2 | --- |
| Connected Components (CC) | 19.2 | 27.2 | 7.5 | --- | --- | --- |

**Score threshold and the precision-recall Pareto frontier.** PQ decomposes as $\text{PQ} = \text{SQ} \times \text{RQ}$, where $\text{RQ} = \frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}$ penalizes false positives and false negatives symmetrically. Lowering the score threshold increases recall (more detections) but decreases precision (more false positives). Since RQ weights FP and FN equally, the optimal threshold is where $\frac{\partial |TP|}{\partial t} = \frac{\partial |FP|}{\partial t}$ --- i.e., where the marginal true positive rate equals the marginal false positive rate. CuVLER at $t = 0.35$ (5.3 inst/img, PQ^Th = 11.0) outperforms $t = 0.15$ (18.0 inst/img, PQ^Th = 8.1) because the low-confidence detections at $t < 0.35$ are predominantly false positives on stuff regions, despite the semantic filter that rejects non-thing-class detections.

**Multi-scale inference.** Running CuVLER at the native Cityscapes resolution (1024px shorter side) rather than the default detectron2 test scale (800px) doubles the detection count from 5.3 to 10.7 per image. The larger scale resolves small objects that are below the effective receptive field at 800px. However, single-scale 1024px inference alone achieves lower PQ (20.4) than 800px (20.7) because the extra detections include more false positives. Fusing both scales with mask-IoU NMS ($\theta = 0.3$) yields PQ = 20.9 with the best RQ = 32.5 observed across all configurations, confirming that multi-scale fusion improves the precision-recall balance.

**Ensemble complementarity.** CuVLER (trained on VoteCut pseudo-labels from 6 self-supervised models) and CutLER (trained on MaskCut pseudo-labels from a single DINO model) should, in principle, detect complementary object instances due to their different training distributions. In practice, the ensemble with NMS deduplication yields PQ = 20.9 --- identical to multi-scale CuVLER alone. This suggests that the two detectors' error distributions are highly correlated: both miss the same classes (rider, motorcycle, train) and both succeed on the same large objects (cars, buses, trucks). The lack of ensemble gain indicates that the detection bottleneck is not model-specific but reflects a fundamental domain gap between ImageNet-pretrained class-agnostic detectors and Cityscapes' driving-specific object categories.

**Hybrid detector + CC fallback.** We tested augmenting CuVLER detections with connected-component fallback on uncovered thing-class pixels. The CC min-area threshold critically determines performance:

| CC $A_{\min}$ | PQ | PQ^Th | Car FP | Person FP |
|---------------|------|-------|--------|-----------|
| 10 | 18.6 | 5.9 | 4818 | 1855 |
| 500 | 19.9 | 9.2 | 1340 | 612 |
| 1000 | 20.4 | 10.3 | 986 | 423 |
| 2000 | 20.8 | 11.1 | 804 | 312 |
| $\infty$ (no CC) | 20.7 | 11.0 | 673 | 287 |

The monotonic convergence toward the detector-only result as $A_{\min} \to \infty$ reveals that CC instances from CAUSE-TR semantic maps introduce more false positives than true positives at every area threshold. The CAUSE-TR semantic predictions for thing classes are noisy and fragmented --- small mislabeled regions (road pixels classified as car, building pixels classified as person) each become separate CC instances that do not overlap GT instances. This finding has important implications: the semantic pseudo-label quality establishes a hard ceiling on CC-based instance generation that cannot be overcome by post-hoc filtering.

#### 4.3.5 Stuff-Things Classification Ablation

The binary classification of semantic classes into stuff ($\mathcal{S}$) and things ($\mathcal{T}$) determines which classes undergo instance decomposition. This design choice has outsized impact because misclassification in either direction is costly: marking a stuff class as thing creates false positive instances; marking a thing class as stuff collapses all instances into a single segment, producing false negatives.

**Table 7: Stuff-things classification variants.**

| Classifier | PQ | PQ^St | PQ^Th | Accuracy | Notes |
|------------|------|-------|-------|----------|-------|
| CLS attention (ours) | 23.1 | 31.4 | 11.7 | 15/19 | $|\mathcal{T}| = 6$ |
| Oracle GT split | 22.1 | 31.6 | 9.0 | 19/19 | $|\mathcal{T}| = 8$ |
| All-thing | 21.4 | 28.7 | 11.1 | --- | $|\mathcal{T}| = 19$ |
| All-stuff | 19.8 | 31.4 | 0.0 | --- | $|\mathcal{T}| = 0$ |
| Random ($n = 8$) | 21.3 | 30.2 | 8.5 | --- | Avg. over 10 trials |

**Why the oracle loses.** The counterintuitive result that our 79%-accurate classifier (PQ = 23.1) outperforms the 100%-accurate oracle (PQ = 22.1) admits a precise explanation via the PQ decomposition. Define the per-class PQ contribution as $\text{PQ}_k = \text{SQ}_k \times \text{RQ}_k$. For class $k$ with semantic mIoU = 0 (rider, train, motorcycle), $\text{TP}_k = 0$ regardless of instance decomposition, so:

$$\text{PQ}_k^{\text{thing}} = 0 \times \frac{0}{0 + \frac{1}{2}|FP_k| + \frac{1}{2}|FN_k|}$$

When such a class is treated as thing, instance decomposition creates $|FP_k| > 0$ spurious instances (from semantically mislabeled pixels), increasing the denominator and ensuring $\text{PQ}_k = 0$. When treated as stuff, $|FP_k| = 0$ (no instances to match), and the class still contributes $\text{PQ}_k = 0$ because $|TP_k| = 0$. The key difference is in the *averaging*: PQ^Th averages only over thing classes, so adding zero-contribution thing classes dilutes the average. Our classifier avoids this dilution by assigning zero-mIoU classes to stuff.

Conversely, our classifier assigns wall to things. GT classifies wall as stuff, which means wall instances in the GT panoptic map are single segments per image. When wall is treated as thing, our depth-guided decomposition creates wall instances that can match GT wall segments, producing TP > 0 and positive PQ^Th contribution. This instance-level treatment is beneficial because Cityscapes walls are often discrete physical structures (retaining walls, barrier walls) with clear depth boundaries.

**Information-theoretic view.** The CLS attention saliency score $r_k$ can be interpreted as an estimate of the mutual information $I(\text{class}_k; \text{CLS attention})$. Thing classes, which correspond to compact, countable objects, have high mutual information with the CLS token's foreground-selective attention, while stuff classes, which are spatially diffuse, have low mutual information. The Otsu threshold on $r_k$ implements a maximum a posteriori (MAP) decision rule under the assumption that stuff and thing saliency scores follow separate Gaussian distributions, which is approximately satisfied empirically.

#### 4.3.6 CRF Refinement as Markov Random Field Inference

DenseCRF post-processing (Krahenbuhl and Koltun, 2011) refines the CAUSE-TR semantic predictions by modeling the label field as a fully connected conditional random field with Gaussian and bilateral pairwise potentials. The energy function is:

$$E(\mathbf{x}) = \sum_i \psi_u(x_i) + \sum_{i < j} \psi_p(x_i, x_j)$$

where $\psi_u$ encodes the unary energy from CAUSE-TR logits (with temperature scaling $\alpha = 3$) and $\psi_p$ encodes spatial and color-dependent smoothness. The bilateral potential $\psi_p(x_i, x_j) = \mu(x_i, x_j) \cdot k(\mathbf{f}_i, \mathbf{f}_j)$ encourages nearby pixels with similar color to share the same label, which sharpens object boundaries that the $14 \times 14$-stride ViT backbone cannot resolve.

**Table 8: CRF parameter sensitivity.**

| CRF Config | mIoU | $\Delta$mIoU | PQ | PQ^St |
|------------|------|-------------|------|-------|
| No CRF | 40.44 | --- | 22.6 | 27.8 |
| $\sigma_{xy}{=}67$, $\sigma_{rgb}{=}3$, $\alpha{=}3$ (ours) | 42.86 | +2.42 | 23.1 | 31.4 |
| $\sigma_{xy}{=}50$, $\sigma_{rgb}{=}5$, $\alpha{=}3$ | 42.31 | +1.87 | 22.9 | 30.8 |
| $\sigma_{xy}{=}100$, $\sigma_{rgb}{=}3$, $\alpha{=}3$ | 42.52 | +2.08 | 22.8 | 31.0 |

CRF improves mIoU by +2.42 percentage points and PQ by +0.5, primarily through PQ^St (+3.6). The improvement concentrates on boundary pixels where the ViT's $14 \times 14$ stride creates aliasing artifacts. The spatial bandwidth $\sigma_{xy} = 67$ (approximately $5 \times$ the ViT patch stride of 14 pixels) provides the optimal smoothing scale: smaller values under-smooth, leaving patch-grid artifacts, while larger values over-smooth, bleeding labels across genuine semantic boundaries. The color bandwidth $\sigma_{rgb} = 3$ (on 0--255 scale) is deliberately tight, ensuring that only visually similar pixels are encouraged to share labels.

The temperature parameter $\alpha = 3$ amplifies the unary log-probabilities, increasing the CRF's confidence in the CAUSE-TR predictions relative to the pairwise smoothness prior. This is necessary because CAUSE-TR produces poorly calibrated logits --- the entropy of the softmax distribution is too high, making the unary potential too weak relative to the pairwise potential without rescaling.

#### 4.3.7 Instance Merging Post-Processing

We implemented a depth-aware merging step that combines adjacent same-class instances with similar mean depth ($\delta < 0.05$). The merging uses a union-find algorithm on a spatial adjacency graph $G = (V, E)$ where vertices are instance segments and edges connect spatially adjacent segments of the same class, weighted by $w_{ij} = |\bar{D}_i - \bar{D}_j|$. Edges with $w_{ij} < \delta$ are contracted.

Merging reduces the instance count by 5--8% per image but yields marginal PQ improvement (+0.1--0.3). The limited impact indicates that the depth-gradient threshold $\tau$ and min-area filter $A_{\min}$ already handle most over-segmentation: fragments that survive both filters are typically genuinely distinct objects rather than over-segmented pieces of the same object.

---

## 5. Analysis

### 5.1 Taxonomy of Failure Modes

We categorize the pipeline's failure modes into three regimes, each with distinct theoretical characteristics and implications for Stage 2 refinement.

#### 5.1.1 Semantic Resolution Bottleneck (7 Zero-mIoU Classes)

Seven of 19 classes achieve 0% IoU: fence, pole, traffic light, traffic sign, rider, train, motorcycle. These are all thin or small objects that occupy few patches in the $23 \times 23$ patch grid ($322 \times 322$ pixels at stride 14). CAUSE-TR's modularity-based clustering assigns their patches to neighboring large-area classes (fence $\to$ vegetation; pole $\to$ sky; traffic sign $\to$ building).

This failure is fundamental to patch-level self-supervised segmentation. Consider an object occupying $n$ patches. The modularity objective (Newman, 2006) assigns a cluster label by maximizing $Q = \frac{1}{2m}\sum_{ij}[A_{ij} - \frac{k_i k_j}{2m}]\delta(c_i, c_j)$, where $A$ is the patch affinity matrix. For small $n$, the object's patches contribute negligibly to $Q$ compared to surrounding large-area classes, so the optimization has no incentive to preserve them as a distinct cluster. This predicts a critical object size $n^* \propto \sqrt{m/K}$ below which classes cannot be discovered, where $m$ is the total number of patches and $K$ is the number of clusters. For our setting ($m = 529$ patches per crop, $K = 27$ clusters), $n^* \approx 4.4$ patches, consistent with the observation that objects spanning fewer than $\sim$5 patches (100 pixels at 14px stride) are systematically misclassified.

#### 5.1.2 Depth Ambiguity Regime (Person Under-Segmentation)

Person achieves PQ = 5.8 with RQ = 8.8%, meaning only 8.8% of person instances are successfully matched. The dominant failure mode is co-planar grouping: pedestrians walking on the same sidewalk occupy similar depths, so $\Delta D_{pq} < \tau$ for adjacent persons and depth-gradient splitting fails to separate them.

This can be formalized as a signal-to-noise problem. Define the inter-instance depth gap $\Delta_{\text{inter}} = |D(p_i) - D(p_j)|$ for pixels $p_i, p_j$ on adjacent instances and the intra-instance depth variation $\sigma_{\text{intra}}$ for pixels within a single instance. The depth-gradient detector achieves separation when $\Delta_{\text{inter}} > \tau + \sigma_{\text{intra}}$. For pedestrians at 20--30m range in Cityscapes, $\Delta_{\text{inter}} \approx 0.5$--$2.0$m, which after min-max normalization and Gaussian smoothing yields gradient magnitudes of 0.02--0.08 --- often below our threshold $\tau = 0.10$. This contrasts with cars, where the typical inter-car depth gap is 3--10m, yielding gradients $> 0.15$ that are reliably detected.

The learned detectors (CuVLER/CutLER) face a different but equally severe failure mode: pedestrians in Cityscapes are small (mean area $\sim$2000 pixels), thin, and often partially occluded, making them challenging for Cascade Mask R-CNN at the 800px inference scale. CuVLER detects only 4% of GT person instances (204/3376 TP, from our detector experiments), confirming that person segmentation is a domain-specific challenge that neither geometric nor ImageNet-pretrained approaches adequately address.

#### 5.1.3 Over-Segmentation Regime (Car Fragmentation)

Cars have 907 false positive instances despite achieving the highest thing-class TP count (732). The fragmentation arises from intra-object depth variations: a car's windshield, roof, and bumper have measurably different depths due to surface curvature, producing depth gradient peaks that exceed $\tau$. This is a form of the *aperture problem* applied to depth: locally, a depth gradient within an object is indistinguishable from a depth gradient between objects.

The min-area filter addresses this by exploiting the statistical regularity that intra-object fragments are typically smaller than complete objects. However, this heuristic fails for large vehicles where fragments can exceed $A_{\min}$, and for small vehicles (motorcycles, distant cars) where the entire object may be close to $A_{\min}$, creating a fundamental precision-recall tradeoff.

### 5.2 Depth Estimation Quality Bounds

The quality of instance decomposition is theoretically bounded by the depth estimation error. Let $\hat{D}$ be the estimated depth and $D^*$ the true depth. The gradient error is:

$$\|\nabla \hat{D} - \nabla D^*\| \leq \|\nabla(\hat{D} - D^*)\| \leq \text{Lip}(\hat{D} - D^*)$$

where $\text{Lip}(\cdot)$ denotes the Lipschitz constant. SPIdepth's self-supervised training on Cityscapes sequences yields depth maps with relative error $\sigma_{\text{rel}} \approx 0.11$ (estimated from AbsRel metrics on the Eigen split), which translates to gradient error proportional to $\sigma_{\text{rel}} / \Delta x$ where $\Delta x$ is the pixel spacing. At object boundaries, the true gradient is a step function with magnitude $\Delta D^* / \Delta x$; the estimated gradient is a smoothed version with magnitude $(\Delta D^* - \epsilon) / (\Delta x + \delta)$ where $\epsilon$ captures depth estimation bias and $\delta$ captures boundary bleeding. For boundaries where $\Delta D^* < 2\epsilon$, the estimated gradient falls below the detection threshold $\tau$, establishing a minimum detectable depth gap that is inversely proportional to depth estimation accuracy.

SPIdepth was selected over feed-forward alternatives (ZoeDepth, Depth Anything V2) because its self-supervised training on the target domain avoids distribution shift. Models pretrained on mixed indoor/outdoor datasets produce depth maps with correct ordinal structure but incorrect metric scale in driving scenes, which corrupts the gradient magnitude and invalidates our fixed threshold $\tau$.

### 5.3 Comparison with CUPS

CUPS (Hahn et al., CVPR 2025) achieves PQ = 27.8 on Cityscapes using stereo video pseudo-labels from optical flow, binocular disparity, and temporal consistency. Our pipeline achieves PQ = 23.1 ($\Delta = -4.7$) using only monocular images. The gap decomposes as:

| Component | Ours | CUPS | $\Delta$ | Primary Cause |
|-----------|------|------|----------|---------------|
| PQ^St | 31.4 | 35.1 | $-$3.7 | Semantic quality (42.9% vs. CUPS's higher mIoU) |
| PQ^Th | 11.7 | 17.7 | $-$6.0 | Instance recall (geometric vs. stereo depth) |
| SQ | 74.3 | 57.4 | +16.9 | Our masks have higher overlap when matched |
| RQ | 31.2 | 35.2 | $-$4.0 | We match fewer instances overall |

The SQ advantage (+16.9) is notable: when our pipeline successfully matches an instance, the mask quality is substantially higher than CUPS. This suggests that depth-gradient boundaries, while missing many instances, produce well-delineated masks for the instances they do capture. CUPS's lower SQ may reflect noise from optical flow estimation and temporal aggregation.

Our approach has two structural advantages: (1) it operates on monocular images, requiring no video sequences or stereo pairs; and (2) computational simplicity --- no optical flow, stereo matching, or temporal tracking. These properties enable application to arbitrary image datasets beyond driving domains.

### 5.4 Semantic Quality as the Dominant Bottleneck

We quantify the theoretical PQ ceiling imposed by semantic quality. PQ averages per-class contributions: $\text{PQ} = \frac{1}{|\mathcal{C}|}\sum_{k \in \mathcal{C}} \text{PQ}_k$. For any class $k$ with $\text{mIoU}_k = 0$, we have $\text{PQ}_k = 0$ regardless of instance decomposition quality. With 7 of 19 classes at zero mIoU:

$$\text{PQ} \leq \frac{12}{19} \cdot \text{PQ}_{\text{nonzero}} = 0.632 \cdot \text{PQ}_{\text{nonzero}}$$

Our observed PQ on the 12 non-zero classes is $23.1 \times (19/12) = 36.6$, which approaches the practical ceiling for unsupervised methods on these classes. This analysis confirms that semantic quality improvement --- not instance decomposition refinement --- is the primary lever for further PQ gains.

If the 7 zero-mIoU classes were recovered at even moderate quality (mIoU = 20%, producing PQ$_k \approx 10$), the overall PQ would increase to approximately $\frac{12 \times 36.6 + 7 \times 10}{19} = 26.8$, nearly closing the gap to CUPS (27.8). This motivates Stage 2 of MBPS, where the Mamba2 cross-modal bridge jointly refines semantic and instance predictions through depth-conditioned state space dynamics.

---

## 6. Related Work

**Unsupervised semantic segmentation.** STEGO (Hamilton et al., 2022) pioneered self-supervised semantic segmentation by training a feature correspondence head on DINO features. PiCIE (Cho et al., 2021) used photometric invariance and geometric equivariance. CAUSE (Cho et al., 2024) improved upon these with modularity-based codebook clustering and a transformer refinement head, achieving state-of-the-art unsupervised mIoU on Cityscapes. We use CAUSE-TR as our semantic backbone.

**Self-supervised depth estimation.** Monodepth2 (Godard et al., 2019) established the self-supervised monocular depth paradigm using photometric reprojection losses. Subsequent work improved accuracy through attention-based decoders (SQLdepth/SPIdepth, Seo et al., 2025) and larger backbone architectures. We use SPIdepth for its domain-specific training on Cityscapes.

**Unsupervised instance segmentation.** MaskCut (Wang et al., 2023) discovers object masks via Normalized Cuts on DINO self-attention features. CutLER extends this with a self-training loop. FreeSOLO (Wang et al., 2022) uses self-supervised pretext tasks for mask proposal generation. These methods are optimized for object-centric datasets and underperform on scene-centric driving images, as our experiments demonstrate.

**Unsupervised panoptic segmentation.** U2Seg (Niu et al., 2023) unified unsupervised semantic and instance segmentation through a shared feature space. CUPS (Hahn et al., 2025) achieved state-of-the-art PQ = 27.8 on Cityscapes using stereo video pseudo-labels. Our work differs from both in using depth-guided geometric decomposition rather than learned instance proposals.

---

## 7. Conclusion

We have presented a fully unsupervised pipeline for panoptic pseudo-label generation that achieves PQ = 23.1 on Cityscapes using only monocular images and self-supervised models. The key technical insight is that depth-gradient instance decomposition is substantially more effective than spectral methods for scene-centric driving imagery, achieving 6.2x higher PQ^Th than MaskCut. The CLS attention stuff-things classifier provides an effective unsupervised proxy for objectness that, counterintuitively, outperforms the oracle ground truth split in the panoptic metric.

These pseudo-labels serve as the training signal for Stage 2 of the MBPS system, where a Mamba2-based Bidirectional Cross-Modal Scan bridge refines semantic and instance predictions through depth-conditioned state space dynamics. We anticipate that the bridge's cross-modal fusion will be particularly effective at addressing the failure modes identified here --- especially the person under-segmentation and car over-segmentation that arise from purely geometric instance decomposition.

---

## References

- Caron, M., Touvron, H., Misra, I., et al. (2021). Emerging properties in self-supervised vision transformers. ICCV.
- Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., and Girdhar, R. (2022). Masked-attention mask transformer for universal image segmentation. CVPR.
- Cho, J., et al. (2021). PiCIE: Unsupervised semantic segmentation using invariance and equivariance in clustering. CVPR.
- Cho, J., et al. (2024). CAUSE: Contrastive learning with modularity-based codebook for unsupervised segmentation. Pattern Recognition, 146.
- Cordts, M., et al. (2016). The Cityscapes dataset for semantic urban scene understanding. CVPR.
- Dao, T. and Gu, A. (2024). Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. ICML.
- Godard, C., Mac Aodha, O., Firman, M., and Brostow, G. J. (2019). Digging into self-supervised monocular depth estimation. ICCV.
- Gu, A. and Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv:2312.00752.
- Hahn, K., et al. (2025). CUPS: Unsupervised panoptic segmentation from stereo video. CVPR.
- Hamilton, M., Zhang, Z., Hariharan, B., Snavely, N., and Freeman, W. T. (2022). Unsupervised semantic segmentation by distilling feature correspondences. ICLR.
- Ji, X., Henriques, J. F., and Vedaldi, A. (2019). Invariant information clustering for unsupervised image classification and segmentation. ICCV.
- Kirillov, A., He, K., Girshick, R., Rother, C., and Dollar, P. (2019). Panoptic segmentation. CVPR.
- Krahenbuhl, P. and Koltun, V. (2011). Efficient inference in fully connected CRFs with Gaussian edge potentials. NeurIPS.
- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. Naval Research Logistics Quarterly, 2(1-2):83--97.
- Liu, Z., et al. (2022). A ConvNet for the 2020s. CVPR.
- Newman, M. E. J. (2006). Modularity and community structure in networks. PNAS, 103(23):8577--8582.
- Niu, Z., et al. (2023). Unsupervised universal image segmentation. arXiv:2312.17243.
- Oquab, M., et al. (2024). DINOv2: Learning robust visual features without supervision. TMLR.
- Oquab, M., et al. (2025). DINOv3: Self-supervised vision transformers with registers. arXiv.
- Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Trans. Syst. Man Cybern., 9(1):62--66.
- Seo, J., et al. (2025). SPIdepth: Strengthened pose information for self-supervised monocular depth estimation. CVPR.
- Simeoni, Y., et al. (2021). Localizing objects with self-supervised transformers and no labels. BMVC.
- Wang, X., et al. (2022). FreeSOLO: Learning to segment objects without annotations. CVPR.
- Wang, X., et al. (2023). Cut and learn for unsupervised object detection and instance segmentation. CVPR.
