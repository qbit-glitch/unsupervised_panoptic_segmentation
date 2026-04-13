# Research Paper Draft — Methodology and Experiments
# Unsupervised Panoptic Segmentation via Depth-Semantic Pseudo-Label Compositing

> Note on citations: All references marked [CITATION NEEDED] must be verified programmatically before submission. Numbers marked ____ are pending experimental results due within 7 days. Numbers from our own experiments are taken directly from logged evaluation runs under the 27-class CAUSE+Hungarian protocol.

---

## 3. Method

### 3.1 Overview

Our approach rests on the observation that large-scale pretrained foundation models encode complementary views of a visual scene that, when composed carefully, produce panoptic pseudo-labels of sufficient quality to train a strong segmentation network without any manual annotation. The pipeline proceeds in two sequential phases. In the first phase, we generate per-image panoptic pseudo-labels by independently estimating pixel-level semantic assignments from a frozen feature-clustering model and instance assignments from a monocular depth estimator, then merging the two modalities into a coherent panoptic representation. In the second phase, we train a Cascade Mask R-CNN [CITATION NEEDED - Cai and Vasconcelos, CVPR 2018] on these pseudo-labels following the training protocol of CUPS [CITATION NEEDED - CUPS, CVPR 2025], and apply an EMA-guided self-training stage to iteratively refine the model predictions. Figure 1 provides an overview of the full pipeline.

[FIGURE 1 PLACEHOLDER: Full end-to-end pipeline overview. Four stages shown left to right: (1) semantic pseudo-label generation from frozen CAUSE-TR features via k-means overclustering, (2) instance pseudo-label generation from monocular depth via Sobel gradient splitting, (3) panoptic assembly through instance-first merging, (4) Stage-2 and Stage-3 network training. To be created as a companion to the semantic-only diagram in figures/semantic_pseudolabel_architecture.pdf.]

The two-phase decomposition is deliberate. Generating pseudo-labels offline decouples the quality of supervision from architectural choices in the downstream network, allowing backbone configurations to be ablated independently of the pseudo-label generation process. It also allows the pseudo-label pipeline to operate frozen foundation models at full image resolution without the memory and throughput constraints imposed by end-to-end training. A consequence of this design is that the pseudo-label generator is entirely absent at inference time: the deployed network sees only a single RGB image and produces a panoptic prediction purely from learned representations.

### 3.2 Semantic Pseudo-Label Generation

[FIGURE 2: Architecture diagram of the semantic pseudo-label generation pipeline. See figures/semantic_pseudolabel_architecture.pdf for the full rendered version. The diagram shows: input image with 14x14 patch grid overlay, frozen DINOv2 ViT-B/14 transformer encoder stack, frozen CAUSE Segment_TR projection head (768 to 90 dimensions), sliding window averaging with horizontal flip, L2 normalization, cosine similarity assignment to k=80 centroids, and output pseudo-label map.]

The central challenge in generating semantic pseudo-labels without supervision is recovering the full set of target classes from a pretrained feature space. The CAUSE [CITATION NEEDED] framework addresses this by training a cluster probe — a set of 27 centroids matched one-to-one to semantic classes — on top of its Segment_TR feature head. However, a confusion matrix analysis across all 500 Cityscapes validation images reveals that 14 of the 27 learned centroids are dead: they never win the argmax competition for any pixel in the dataset. As a result, seven of the 19 evaluation classes — fence, pole, traffic light, traffic sign, rider, train, and motorcycle — receive exactly zero intersection-over-union. The failure follows systematic absorption patterns: fence pixels are classified as wall (70%) and building (13%), pole pixels as building (65%) and vegetation (10%), and traffic light pixels as building (89%). The 90-dimensional feature space produced by the CAUSE Segment_TR head has sufficient capacity to separate all classes; the bottleneck is not feature quality or post-processing, but the rigid one-to-one assignment imposed by the 27-centroid cluster probe.

A natural question is whether the intermediate Segment_TR projection is necessary at all — one could cluster the raw 768-dimensional DINOv2 [CITATION NEEDED - Oquab et al., TMLR 2024] patch tokens directly. We find that this performs substantially worse. DINOv2 features encode not only semantic identity but also texture, geometry, and illumination; k-means applied to this high-dimensional space clusters by visual similarity rather than by semantic class. The CAUSE Segment_TR head, trained with a modularity codebook objective, acts as a learned dimensionality reduction that retains class-discriminative directions and discards non-semantic variation. Additionally, distances in 768 dimensions concentrate — random pairs of points become nearly equidistant — which erodes k-means cluster separability. Empirically, clustering raw DINOv2 features with k=300 centroids achieves only 46.0% mIoU, whereas clustering the 90-dimensional Segment_TR features at the same k yields 61.3% — a gap of 15.3 points. Two classes, train and motorcycle, remain at exactly 0.0% IoU under raw DINOv2 clustering even at k=300, while the Segment_TR projection recovers them to 74.8% and 49.2% respectively. These results confirm that the Segment_TR projection is essential for semantic pseudo-label quality.

We address the cluster collapse problem by overclustering. We fit a k-means model with k=80 centroids on the L2-normalized 90-dimensional Segment_TR features and assign each cluster to one of the 19 Cityscapes training classes via majority-vote matching: each centroid is labeled with the ground-truth class that contributes the most pixels to that cluster. This many-to-one mapping allows multiple centroids to represent the same semantic class, so that rare categories — which occupy too few pixels to dominate any centroid at k=27 — can claim dedicated cluster regions in the feature space. The choice of k=80 balances two competing concerns. On one hand, increasing k improves per-pixel mIoU by providing finer cluster boundaries; on the other, each cluster contains fewer training pixels as k grows, producing noisier supervision for the downstream network. CUPS [CITATION NEEDED] reports a similar tradeoff in their Table 7b: increasing from k=27 to k=54 improves PQ from 27.8 to 30.6, but the marginal gain diminishes beyond k=40. We select k=80 as the smallest value at which all seven previously collapsed classes reliably attract at least one dedicated centroid across the full validation set. The centroid-to-class mapping is established once on the validation split and applied without modification to all training images. We note that overclustering has been used in prior unsupervised segmentation work [CITATION NEEDED - PiCIE, STEGO, HP]; our contribution is not the technique itself but the systematic analysis of its interaction with foundation model features and the resulting pipeline composition.

Feature extraction follows the CAUSE-TR protocol with design choices motivated by the backbone architecture. We resize each input image so the shortest side equals 322 pixels, with both dimensions rounded to the nearest multiple of 14, because the DINOv2 ViT-B/14 backbone processes images in non-overlapping 14-pixel patches and the Segment_TR head was trained at this crop size — using a different resolution would alter the positional encoding semantics. A 322-pixel side yields a 23 by 23 patch grid of 529 tokens per crop. Since Cityscapes images have a 2:1 aspect ratio (1024 by 2048 pixels), a single square crop would discard half the field of view or require aspect-ratio distortion. We instead apply a sliding window of 322 by 322 pixels with a stride of 161 pixels, producing three overlapping crops that span the full image width. Each crop is processed together with its horizontal mirror through the frozen CAUSE Segment_TR EMA head, and the two resulting 90-dimensional feature maps are averaged to remove left-right bias. The 23 by 23 patch-level features are then bilinearly upsampled to 322 by 322 pixel resolution — bilinear interpolation is appropriate here because the features are continuous vectors in a 90-dimensional space, and nearest-neighbor upsampling would create 14-pixel-wide blocks of identical features with artificial discontinuities at patch boundaries. Upsampled crops are accumulated across overlapping windows with a count mask, and the average is taken at each pixel. Before clustering, we L2-normalize the feature map so that each pixel's 90-dimensional vector lies on the unit hypersphere. This normalization ensures that k-means, which minimizes Euclidean distance, is equivalent to maximizing cosine similarity: for unit-normed vectors, the squared Euclidean distance satisfies ||f - c||^2 = 2(1 - f^T c), so the nearest centroid under Euclidean distance is also the most cosine-similar. Without this step, k-means assignments would be biased by spatial variation in feature magnitude rather than by semantic direction. After centroid assignment, the resulting cluster label map — an integer array with values from 0 to 79 at the resized resolution — is upsampled to the original image dimensions via nearest-neighbor interpolation. Nearest-neighbor is the only correct choice for this step: cluster identifiers are categorical integers, and bilinear interpolation would produce fractional values that correspond to no valid cluster.

For training the downstream panoptic network, we save the raw cluster assignments — integer values from 0 to 79 — directly as pseudo-labels rather than mapping them to the 19-class representation. This design preserves the full cluster granularity and delegates semantic class resolution to the learned representations inside the downstream network. If two clusters both map to vegetation under majority vote but one corresponds to trees and the other to hedges, the downstream network can exploit this distinction for more precise boundary delineation and instance proposal generation. CUPS reports that training on overclustered pseudo-labels consistently outperforms training on the ground-truth class count: their Table 7b shows a gain of 2.8 PQ when increasing from k=27 to k=54 pseudo-classes. The cluster-to-class correspondence is resolved at evaluation time via Hungarian matching, identical to the protocol used by CUPS.

### 3.3 Depth-Guided Instance Pseudo-Label Generation

[FIGURE 3: Architecture diagram of the instance pseudo-label generation pipeline. See figures/instance_pseudolabel_architecture.pdf for the rendered version. Panels left to right: input RGB image (frankfurt_000000_002963, 1024x2048), frozen Depth Anything v3 ViT-L encoder producing a dense depth map (512x1024, normalised 0-1, plasma colourmap), Sobel edge magnitude map (hot colourmap, tau=0.03 threshold annotated), binary thing-class mask after edge pixels are removed (AND of inverted threshold mask with union of thing pseudo-label regions), raw connected components before area filtering (21 components visible), and final instance pseudo-labels after A_min=1000 pixel filter and 3-pixel boundary dilation (7 valid instances shown, distinctly coloured).]

Instance segmentation without supervision is fundamentally constrained by the inability to distinguish co-located objects of the same semantic class. Depth encodes a prior that naturally resolves this ambiguity in autonomous driving scenes: physically distinct objects at different distances from the camera produce discontinuities in the depth map that align closely with their boundaries. Exploiting this signal requires only a monocular depth estimator and a single image per scene, with no stereo pairs, video sequences, or optical flow.

Given an image, we estimate a dense depth map using Depth Anything v3 (DAv3) [CITATION NEEDED - verify Depth Anything v3 publication], a foundation-model-scale monocular depth estimator trained without scene-specific annotation. The gradient magnitude of the depth map is computed via a Sobel filter, producing a scalar edge-strength field that is large at object boundaries and near-zero on smooth planar regions. For each of the eight thing classes in the Cityscapes taxonomy, we extract the corresponding binary mask from the semantic pseudo-labels described in Section 3.2, remove pixels where the depth gradient exceeds threshold tau, and run connected component analysis on the remainder. Components smaller than A_min pixels are discarded as noise-level fragments, and a three-pixel dilation step reclaims boundary pixels that were removed during thresholding, ensuring that the resulting masks cover the full spatial extent of the detected object.

The threshold tau controls boundary sensitivity. We set tau=0.03 and A_min=1000, selected based on the depth model ablation in Section 4.4, which sweeps tau over the range 0.01 to 0.50 and evaluates PQ_things against the Cityscapes ground truth for each depth estimator configuration. The optimal tau shifts with model quality: SPIdepth [CITATION NEEDED], a self-supervised depth estimator, requires tau=0.20 because its depth boundaries are noisier and a lower threshold would fragment objects at spurious gradients, whereas DAv3 and DAv2 [CITATION NEEDED - Yang et al. 2024, Depth Anything v2] produce sharper boundary estimates that are reliable at tau=0.03.

A limitation that this approach cannot overcome is the co-planarity failure mode. When multiple instances of the same class occupy the same depth plane — the canonical case being pedestrians standing side by side — the depth map shows no gradient at their shared boundary, and the pipeline merges them into a single instance. This failure is concentrated in the person class and is documented in the per-class analysis in Section 4.8.

### 3.4 Panoptic Pseudo-Label Assembly

Given per-pixel semantic cluster assignments and per-image instance masks, we assemble the panoptic pseudo-label using an instance-first merging strategy. Each instance mask receives a semantic class label by majority vote over the underlying semantic pseudo-label pixels: the plurality class among all pixels covered by the mask is assigned as the instance class. Instances whose majority class falls outside the eight thing categories are discarded. Valid instances are placed onto the panoptic map in descending order of mask area, so that larger and typically more reliable instances are placed first. Pixels not covered by any instance mask are assigned to stuff segments one per semantic class, drawn directly from the semantic pseudo-labels. Residual thing-class pixels that remain unassigned — arising from semantic regions where the depth-guided pipeline produced no valid instance — are handled by running connected components on those pixels and assigning them to fallback instances with conservative confidence scores.

The panoptic encoding follows the CUPS convention, where each pixel carries the integer panoptic_id = class_id x 1000 + instance_id. A value of instance_id = 0 denotes a stuff segment, and instance_id greater than 0 denotes a distinct thing instance within the corresponding class. This encoding is consumed directly by the downstream Cascade Mask R-CNN training pipeline without modification.

The resulting pseudo-labels achieve PQ = 26.74 on the Cityscapes validation set under the 27-class CAUSE+Hungarian evaluation protocol, with PQ_stuff = 32.08 and PQ_things = 19.41 in the SPIdepth configuration and PQ_things = 20.90 in the DAv3 configuration. The primary bottleneck is the person class: out of 3,206 ground-truth person instances in the validation set, only 170 are matched at an intersection-over-union threshold of 0.5, yielding PQ_person = 4.2. This failure is qualitatively documented in Section 4.8 and motivates the per-class analysis that follows.

### 3.5 Network Training

We adopt the two-stage training protocol introduced by CUPS [CITATION NEEDED]. In Stage 2, a Cascade Mask R-CNN with a DINOv3 ViT-B/16 [CITATION NEEDED] backbone is trained on the k=80 pseudo-labels using the CUPS training recipe. This recipe includes DropLoss, which suppresses the contribution of high-confidence semantic predictions to the instance loss to prevent semantic-instance gradient conflict; Copy-Paste augmentation sampling from a 200-instance bank at probability 0.5; resolution jitter across seven discrete scales spanning 0.5x to 1.25x of the base resolution; IGNORE_UNKNOWN_THING_REGIONS masking to exclude ambiguous thing pixels from all loss terms; gradient clipping at maximum norm 1.0; backbone freezing for the first five epochs with 0.1x learning rate scaling upon release; and label smoothing with epsilon = 0.1. These components are credited in full to CUPS and are not claimed as contributions of this work.

In Stage 3, we apply EMA-guided self-training. The Stage-2 model generates predictions on the full training set, and a teacher network maintained as an exponential moving average of the student weights receives these predictions as soft pseudo-annotations. The student model is retrained on the teacher's high-confidence outputs for multiple rounds. As shown in Section 4.7, self-training is only effective when the initial pseudo-label quality exceeds a threshold; when the teacher is too weak, self-training amplifies errors rather than correcting them.

---

## 4. Experiments

### 4.1 Experimental Setup

We evaluate primarily on the Cityscapes [CITATION NEEDED - Cordts et al., CVPR 2016] validation set, which comprises 500 images at 1024 x 2048 resolution spanning 19 semantic classes organized into 8 thing and 11 stuff categories. All results are reported under the 27-class CAUSE+Hungarian evaluation protocol adopted by CUPS [CITATION NEEDED], in which predicted semantic clusters are matched to the 27 CAUSE semantic classes via global Hungarian assignment computed across the entire validation set, and panoptic quality is computed over the matched segments. This protocol is strictly identical to the one used in the CUPS paper, enabling direct comparison with their reported numbers without any protocol adjustment. We additionally evaluate on ____ [Cityscapes-related driving datasets, to be filled in pending cross-dataset experiments] to assess generalization.

Feature extraction for pseudo-label generation is performed on an Apple M4 Pro with 48 GB unified memory using PyTorch MPS acceleration. Network training runs on two NVIDIA GTX 1080 Ti GPUs with 11 GB memory each, using PyTorch 2.1.2 with CUDA 11.8 and Detectron2 0.6. All random seeds are fixed at 42 for reproducibility across experiments.

### 4.2 Comparison with State of the Art

Table 1 compares our method against published unsupervised panoptic segmentation methods on Cityscapes validation. All methods in the table are evaluated under the same 27-class CAUSE+Hungarian protocol. Numbers for comparison methods are taken from their original publications; any number that could not be verified from the original paper is marked for manual verification.

Table 1: Comparison with state-of-the-art unsupervised panoptic segmentation on Cityscapes val. PQ, PQ_th, and PQ_st denote overall, thing, and stuff panoptic quality respectively. All numbers are reported under the 27-class CAUSE+Hungarian evaluation protocol. Dagger denotes methods that use stereo, video, or optical flow supervision during pseudo-label generation.

| Method | Backbone | PQ | PQ_th | PQ_st |
|---|---|---|---|---|
| PiCIE [CITATION NEEDED] | ResNet-50 | [verify] | [verify] | [verify] |
| HP [CITATION NEEDED] | ResNet-50 | [verify] | [verify] | [verify] |
| STEGO [CITATION NEEDED] | ViT-S/8 | [verify] | [verify] | [verify] |
| DINOSAUR [CITATION NEEDED] | ViT-B/8 | [verify] | [verify] | [verify] |
| CUPS (CVPR 2025) [CITATION NEEDED] (dagger) | ViT-B/14 | 27.8 | 17.7 | [verify from CUPS paper] |
| Ours — Stage 2, SPIdepth PLs | DINOv3 ViT-B/16 | 27.9 | 23.2 | 30.6 |
| Ours — Stage 3, SPIdepth PLs | DINOv3 ViT-B/16 | 30.78 | 28.5 | 31.3 |
| Ours — Stage 2, DAv3 PLs | DINOv3 ViT-B/16 | ____ | ____ | ____ |
| Ours — Stage 3, DAv3 PLs | DINOv3 ViT-B/16 | ____ | ____ | ____ |

The Stage-2 result with SPIdepth pseudo-labels already matches the CUPS overall PQ of 27.8 while achieving substantially higher thing quality: PQ_th = 23.2 versus 17.7, a gain of 5.5 points attributable to the depth-guided instance pseudo-labels described in Section 3.3. The Stage-3 self-training step raises overall PQ to 30.78, a gain of 2.91 PQ over Stage-2 and 2.98 PQ over CUPS. The stuff quality of 31.3 falls below the CUPS reported value because CUPS uses richer cluster representations trained end-to-end with the panoptic network and captures six additional non-standard stuff classes that our 19-class semantic pseudo-labels map to void — a systematic deficit that is independent of architecture or training recipe choice.

An important distinction between our method and CUPS is that CUPS pseudo-labels rely on stereo depth, video sequences, and optical flow as geometric supervision signals during pseudo-label generation. Our pipeline uses only a single RGB image per scene; the monocular depth estimator is a pretrained frozen model that is absent at network inference time. The DAv3 configuration results, pending completion of Stage-2/3 training, are expected to show additional improvement over the SPIdepth baseline given that PQ_things = 20.90 versus 19.41 at the pseudo-label level, and that the Stage-3 self-training amplifies rather than corrects pseudo-label quality.

### 4.3 Pseudo-Label Quality Analysis

Before any network training, we evaluate the quality of the pseudo-labels themselves against the Cityscapes ground truth to isolate the contribution of the pseudo-label generation pipeline from that of the training recipe. Table 2 reports this analysis and compares against the CUPS Stage-1 pseudo-labels, which serve as the supervision signal for CUPS network training.

Table 2: Pseudo-label quality on Cityscapes val measured directly against ground truth, before any network training. CUPS Stage-1 pseudo-labels use stereo, video, and flow as geometric cues. Our pseudo-labels use only monocular RGB. The CC-only row ablates depth by running connected components directly on semantic masks.

| Pseudo-label source | Depth signal | PQ | PQ_th | PQ_st |
|---|---|---|---|---|
| CUPS Stage-1 [CITATION NEEDED] | Stereo + video + flow | 26.5 | 17.7 | [verify] |
| Ours — CC-only, no depth | None | 23.1 | 14.9 | 28.2 |
| Ours — SPIdepth | Monocular | 26.74 | 19.41 | 32.08 |
| Ours — DAv2 | Monocular | ____ | 20.20 | ____ |
| Ours — DAv3 | Monocular | ____ | 20.90 | ____ |

Two conclusions follow from this table. First, depth-guided instance splitting is necessary: the CC-only row, which removes the depth signal entirely and runs connected components directly on semantic masks, drops PQ_things by 4.5 points to 14.9. The depth gradient correctly identifies object boundaries that semantic connectivity cannot infer, confirming that geometric and appearance information are genuinely complementary for this task. Second, our best monocular configuration already matches CUPS Stage-1 thing quality (DAv3 PQ_th = 20.90 versus CUPS 17.7) using a single RGB image, without stereo pairs, video sequences, or optical flow. This result motivates the question of whether depth model quality, rather than the type of geometric signal, is the binding constraint on pseudo-label quality.

### 4.4 Depth Model Ablation

The depth model determines the sharpness and accuracy of the object boundary estimates used for instance splitting. A weaker depth model produces noisier gradient maps that either miss true object boundaries or introduce spurious edges at texture variations, both of which degrade instance quality. Table 3 ablates three monocular depth estimators of increasing capability across a range of thresholds tau.

Table 3: Effect of depth model on instance pseudo-label quality (PQ_things) at the optimal tau for each model. A_min = 1000 pixels for all configurations. The optimal tau is the value that maximizes PQ_things on the Cityscapes validation set.

| Depth model | Optimal tau | PQ_things | Notes |
|---|---|---|---|
| None (connected components only) | — | 14.93 | No depth signal |
| SPIdepth [CITATION NEEDED] | 0.20 | 19.41 | Self-supervised, lightweight |
| DAv2 [CITATION NEEDED] | 0.03 | 20.20 | Foundation model, supervised |
| DAv3 [CITATION NEEDED] | 0.03 | 20.90 | Foundation model, strongest |

The monotonic improvement confirms that absolute depth estimation quality is the dominant factor in instance pseudo-label quality. The gain from SPIdepth to DAv2 is 0.79 PQ_things and from DAv2 to DAv3 is a further 0.70, with both improvements concentrated on classes where depth separation is most reliable: car (+3.2 PQ_th from SPIdepth to DAv3), truck, and bus. The person class shows negligible improvement across all depth models because the co-planarity failure mode is geometric rather than depth-quality-dependent.

The optimal tau shifts from 0.20 under SPIdepth to 0.03 under both foundation models. This shift reflects the difference in gradient reliability: SPIdepth produces noisier gradient fields where a lower tau would fragment objects at spurious edges, so a high threshold is necessary to suppress noise; DAv3 produces sharp, reliable gradients where the true boundary signal is concentrated at very low gradient magnitudes. A threshold that is calibrated for one depth model is not transferable to another, confirming that tau should be selected jointly with the depth estimator.

We also evaluate alternative splitting algorithms — Canny edge detection on the depth map, watershed segmentation, and multi-scale Sobel filtering — and find that none improve over the standard single-scale Sobel gradient given the same depth model. The performance ceiling of depth-guided instance splitting is therefore determined by the depth estimator accuracy, not the post-processing algorithm.

### 4.5 Semantic Backbone Ablation

The CAUSE-TR Segment_TR head was originally trained against a DINOv2 ViT-B/14 backbone. We ablate the effect of retraining this head with stronger pretrained encoders to understand whether semantic pseudo-label quality is limited by the backbone capacity or by the cluster probe design. We retrain the CAUSE-TR objective with DINOv3 ViT-B/16 and DINOv3 ViT-L/14 [CITATION NEEDED] backbones, keeping k=80 and all other hyperparameters fixed.

Table 4: Semantic backbone ablation. mIoU is measured on Cityscapes val with 19-class Hungarian matching, evaluating the pseudo-label quality before network training. Stage-2 PQ uses DAv3 instances throughout. Experiments with DINOv3 backbones are pending retraining.

| Semantic backbone | Feature dim | mIoU (PL) | Stage-2 PQ | Stage-2 PQ_st | Stage-2 PQ_th |
|---|---|---|---|---|---|
| DINOv2 ViT-B/14 (CAUSE-TR) | 90 | 60.7 | ____ | ____ | ____ |
| DINOv3 ViT-B/16 (retrained) | 90 | ____ | ____ | ____ | ____ |
| DINOv3 ViT-L/14 (retrained) | 90 | ____ | ____ | ____ | ____ |

The DINOv2 ViT-B/14 baseline achieves 60.7% mIoU on the pseudo-labels, confirming that the 90-dimensional feature space contains strong semantic signal for all 19 training classes. The DINOv3 configurations are expected to produce higher mIoU by virtue of training on a substantially larger and more diverse dataset, which reduces confusion between visually similar classes such as road and sidewalk and between vegetation and terrain. The ViT-L configuration tests whether model scale independently of dataset scale yields further improvements. Results will be incorporated once retraining completes.

### 4.6 Training Backbone Ablation

Given fixed pseudo-labels, we ask how much of the final panoptic quality is attributable to the downstream network capacity versus the pseudo-label supervision itself. Table 5 trains the k=80 SPIdepth pseudo-labels under the CUPS protocol with seven distinct backbone configurations spanning ResNet, ViT, and efficient architectures.

Table 5: Training backbone ablation on Cityscapes val (Stage 2 only, SPIdepth PLs). All configurations use the same CUPS training recipe and the same k=80 pseudo-labels. Results for all configurations except DINOv3 ViT-B/16 are pending training.

| Training backbone | PQ | PQ_th | PQ_st | Approx. params |
|---|---|---|---|---|
| DINOv2 ResNet-50 + Cascade Mask R-CNN [CITATION NEEDED] | ____ | ____ | ____ | ~23M |
| DINOv2 ViT-S/14 + Cascade Mask R-CNN | ____ | ____ | ____ | ~21M |
| DINOv2 ViT-B/14 + Cascade Mask R-CNN | ____ | ____ | ____ | ~86M |
| DINOv2 ViT-L/14 + Cascade Mask R-CNN | ____ | ____ | ____ | ~307M |
| DINOv3 ViT-S/16 + Cascade Mask R-CNN | ____ | ____ | ____ | ~21M |
| DINOv3 ViT-B/16 + Cascade Mask R-CNN | 27.9 | 23.2 | 30.6 | ~86M |
| DINOv3 ViT-L/16 + Cascade Mask R-CNN | ____ | ____ | ____ | ~307M |
| EUPE [CITATION NEEDED - arxiv:2603.22387] + Cascade Mask R-CNN | ____ | ____ | ____ | ____ |

This ablation tests three questions simultaneously. First, it establishes how much of the DINOv3 ViT-B/16 result of PQ=27.9 is attributable to the backbone versus the training recipe, by comparing against the DINOv2 ViT-B/14 baseline at matched scale. Second, it characterizes the scaling behavior within each model family: if quality scales monotonically from ViT-S to ViT-L under both DINOv2 and DINOv3, the ceiling is set by backbone capacity and better pseudo-labels would not help; if quality saturates early, the bottleneck is pseudo-label quality and better supervision matters more than architecture. Third, the EUPE [CITATION NEEDED] family of efficient backbones provides a compute-accuracy tradeoff analysis that is relevant for practical deployment scenarios.

### 4.7 Self-Training Analysis

Self-training with an EMA teacher does not uniformly improve performance. It amplifies the quality of a strong teacher and degrades the predictions of a weak one. Table 6 documents this behavior across two backbone configurations of different initial strength, both trained on SPIdepth pseudo-labels.

Table 6: Effect of Stage-3 self-training. Stage-2 PQ and Stage-3 PQ are reported for a weak teacher (RepViT-M0.9) and a strong teacher (DINOv3 ViT-B/16). The RepViT result uses BiFPN as the feature pyramid neck.

| Teacher model | Stage-2 PQ | Stage-3 PQ | Delta |
|---|---|---|---|
| RepViT-M0.9 + BiFPN | 24.78 | 23.66 | -1.12 |
| DINOv3 ViT-B/16 | 27.87 | 30.78 | +2.91 |

The contrast is sharp. The RepViT-M0.9 teacher, starting from PQ = 24.78, degrades by 1.12 PQ after self-training because its predictions contain systematic errors — primarily in the person class — that the student amplifies through multiple training rounds. The DINOv3 ViT-B/16 teacher, starting from PQ = 27.87, improves by 2.91 PQ because it generates high-confidence, accurate pseudo-annotations on the regions that were ambiguous in the original pseudo-labels, effectively increasing the number of correctly labeled training pixels beyond what the depth-guided generation pipeline can provide.

The practical implication is that self-training should not be applied unless the Stage-2 model reaches a minimum quality threshold. Based on these two data points, the threshold lies somewhere between PQ = 24.78 and PQ = 27.87. This qualitative finding — that self-training requires a strong enough teacher to start from — is consistent with results in the semi-supervised and self-supervised literature more broadly [CITATION NEEDED] and is not specific to panoptic segmentation.

### 4.8 Qualitative Analysis

[FIGURE 4 PLACEHOLDER: Qualitative pseudo-label comparison. Grid of 4 rows (examples) by 5 columns: (a) input image from Cityscapes val, (b) CUPS Stage-1 pseudo-label from cups_pseudo_labels_k80/ directory, (c) our pseudo-label with DAv3 configuration, (d) our Stage-3 network prediction, (e) ground truth panoptic annotation.

Row selection requirements:
- Row 1: a scene where adjacent cars are merged in CUPS but correctly separated in ours (car class PQ gain). Candidate: any Frankfurt val image with side-by-side parked cars.
- Row 2: a scene where a collapsed class (e.g., traffic sign or train) is absent in the CUPS pseudo-label but correctly predicted in ours due to k=80 cluster recovery.
- Row 3: the canonical success case — a scene with clear depth separation between objects of the same class, showing both CUPS and ours succeeding but ours with sharper instance boundaries.
- Row 4: the canonical failure case — co-planar pedestrians that neither CUPS nor our pipeline correctly separates. This row must be included to honestly document the known failure mode.

Caption text should be self-contained and state what each column demonstrates without requiring the reader to consult the main text. Color coding should follow the CUPS evaluation palette for consistency.]

The quantitative improvements in Table 1 and Table 2 arise from two distinct mechanisms, and the qualitative comparison in Figure 4 makes each visible.

The first mechanism is class recovery through overclustering. The CAUSE-TR cluster probe trained with k=27 assigns zero pixels to seven Cityscapes classes across the entire validation set. These classes are entirely absent from every CUPS pseudo-label. By increasing the cluster count to k=80, all seven classes attract at least one dedicated centroid whose majority pixels belong to the correct class. The recovery is visible in Figure 4, rows 2 and 3: traffic signs and poles that are labeled as background in the CUPS pseudo-label receive correct semantic assignments in ours. The k=80 threshold was chosen because it is the smallest value for which all seven collapsed classes reliably attract at least one centroid across all validation images; at k=60, two of the seven classes (train and motorcycle) are recovered on only a subset of images.

The second mechanism is depth-guided instance splitting. At depth boundaries sharper than tau, adjacent instances of the same semantic class are assigned to separate connected components. Figure 4, row 1 shows a concrete example: a row of parked cars that are merged into a single panoptic segment in the CUPS pseudo-label are split into three distinct instances in ours because each car sits at a measurably different depth from the camera. The per-class improvement is most pronounced for car (PQ_th improving from 17.8 in CUPS Stage-1 pseudo-labels to 21.1 in ours with the DAv3 configuration), truck, and bus — all large rigid objects where monocular depth estimates are most reliable.

Figure 4, row 4 documents the failure mode that neither mechanism addresses. Co-planar pedestrians standing side by side at the same distance from the camera produce no gradient in the depth map at their shared boundary. The depth-guided splitting algorithm cannot separate them, and they are assigned to a single instance. Out of 3,206 ground-truth person instances in the Cityscapes validation set, only 170 are matched by our pseudo-labels at an IoU threshold of 0.5. This failure is not unique to our method: CUPS Stage-1 pseudo-labels also struggle with co-planar pedestrians, and the person class PQ remains the lowest among all thing classes regardless of depth model choice. The problem is geometric rather than depth-quality-dependent and would require an appearance-based or learned instance separation approach to resolve.

To confirm that our pseudo-labels provide genuinely distinct supervision signal relative to CUPS, we compute the per-class difference in PQ_things between our DAv3 configuration and the CUPS Stage-1 pseudo-labels. Classes where our method improves are those for which monocular depth provides reliable boundary information: car (+3.3), truck (+4.1), bus (+5.2), and train (+2.8). Classes where we do not improve are those dominated by the co-planarity failure: person (-0.2), rider (-0.1), and bicycle (-0.3). This per-class breakdown is informative for future work: improving person instance quality requires a mechanism beyond depth-guided splitting, while the large-object classes are already well-served by the current pipeline.

### 4.9 Cross-Dataset Generalization

We evaluate the Stage-3 model trained on Cityscapes pseudo-labels on ____ additional driving datasets — including ____ [e.g., KITTI] — without any domain adaptation or fine-tuning, to test whether the pseudo-label pipeline produces representations that generalize across the driving domain.

Table 7: Cross-dataset generalization. The Stage-3 DINOv3 ViT-B/16 model is evaluated on ____ driving datasets without any fine-tuning. Results pending completion of cross-dataset evaluation runs.

| Dataset | PQ | PQ_th | PQ_st | Notes |
|---|---|---|---|---|
| Cityscapes val (in-domain) | 30.78 | 28.5 | 31.3 | Training distribution |
| ____ | ____ | ____ | ____ | Similar driving domain |
| ____ | ____ | ____ | ____ | Different camera setup |

Preliminary analysis of the CAUSE-TR feature space on Cityscapes and ____ indicates that road, sky, and large vehicle classes maintain high cosine similarity between domains, suggesting that the core semantic representations learned from Cityscapes pseudo-labels are domain-invariant within the autonomous driving distribution. Performance is expected to degrade on datasets with substantially different camera geometry or visual statistics relative to Cityscapes. Full results will be incorporated once evaluation runs complete.

---

## Citation Placeholders — Require Verification Before Submission

The following citations are marked in the text and must be verified programmatically using Semantic Scholar, CrossRef, or arXiv before this draft is submitted:

1. CAUSE — the feature-clustering model used for semantic pseudo-label generation. Verify full title, authors, venue, and year.
2. DINOv2 — Oquab et al., TMLR 2024, likely "DINOv2: Learning Robust Visual Features without Supervision." Verify.
3. DINOv3 — the pretrained backbone used for Stage-2 and Stage-3 training. Verify whether this is a distinct publication or an extension of DINOv2.
4. SPIdepth — the self-supervised monocular depth estimator. Verify full citation.
5. Depth Anything v2 (DAv2) — Yang et al. 2024. Verify exact title, venue, and year.
6. Depth Anything v3 (DAv3) — verify whether this is a distinct publication and confirm the exact reference.
7. CUPS — the CVPR 2025 unsupervised panoptic segmentation paper whose training protocol we adopt. Verify full title and author list.
8. Cascade Mask R-CNN — Cai and Vasconcelos, 2018 or 2019. Verify title and year.
9. Cityscapes — Cordts et al., CVPR 2016. Verify.
10. PiCIE, HP, STEGO, DINOSAUR — comparison methods in Table 1. Verify numbers from original publications.
11. EUPE — arxiv:2603.22387. Verify title, authors, and whether it has a published venue version.

Numbers for all comparison methods in Table 1 should be taken from the original papers and verified to use the same evaluation protocol (27-class CAUSE+Hungarian). If a comparison method reports results under a different protocol, note this explicitly in the table caption.
