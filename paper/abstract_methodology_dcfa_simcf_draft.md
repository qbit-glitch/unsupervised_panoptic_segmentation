# Abstract and Methodology Draft: Depth + DCFA + SIMCF-ABC

## Verification Notes

This draft is written against the code paths and result artifacts currently present in the repository. The final-method narrative should focus on Depth + DCFA + SIMCF-ABC. LoRA, DoRA, Lambda layers, and the older Mamba-bridge architecture appear in experimental files and older drafts, but they are not part of the method described here.

Key verified points:

- DCFA in the final narrative is the residual feature adapter in `mbps_pytorch/models/semantic/depth_adapter.py`: a small MLP over 90-dimensional frozen semantic codes plus a 16-dimensional sinusoidal depth encoding.
- SIMCF-ABC is implemented in `scripts/refine_simcf.py`: Step A majority-votes semantics within instances, Step B merges adjacent same-class instances using normalized foundation-model feature similarity, and Step C masks semantic depth outliers.
- Depth-guided instance generation is implemented in `mbps_pytorch/generate_depth_guided_instances.py`: Sobel depth gradients, edge removal, connected components, area filtering, and dilation-based boundary reclamation.
- The 35.83 PQ number is a trained Stage-3 model result on Cityscapes val, not the raw pseudo-label PQ. The raw pseudo-label report gives 24.54 -> 25.85 PQ for the DCFA + DepthPro + SIMCF-ABC refinement pipeline.
- The repository contains both DINOv2/CAUSE-TR and DINOv3 artifacts. For the clean paper claim, phrase the semantic source as frozen CAUSE-TR semantic codes backed by DINOv2 unless the specific table/result uses DINOv3 features or a DINOv3 downstream backbone.
- Some generation/refinement paths use a GT-derived cluster-to-class mapping for thing selection, evaluation, and SIMCF. A strict unsupervised claim needs either an unsupervised replacement for this mapping or explicit framing that the mapping is used only for protocol alignment/evaluation.

## Abstract

Unsupervised panoptic segmentation for scene-centric images has recently depended on stereo pairs, video, or motion cues to convert self-supervised representations into panoptic pseudo-labels. We show that this multi-view requirement is not fundamental: a single image, paired with frozen semantic and monocular depth foundation priors, contains enough complementary structure to produce competitive panoptic supervision. Our method decomposes pseudo-label generation into three levels. First, frozen DINOv2/CAUSE-TR features are projected into a 90-dimensional semantic code space and overclustered with \(K=80\), recovering rare and previously collapsed semantic regions without retraining the semantic encoder. Second, monocular depth maps from DepthPro or Depth Anything are converted into thing instances by thresholding Sobel depth discontinuities, applying per-class connected components, and reclaiming boundary pixels. Third, we introduce DCFA and SIMCF-ABC: DCFA is a 40K-parameter residual adapter that conditions semantic codes on a 16-dimensional sinusoidal depth encoding, while SIMCF-ABC enforces mutual consistency between semantic clusters, depth-derived instances, and class-level depth statistics. On Cityscapes, DCFA improves semantic clustering mIoU from 52.69 to 55.29, and the full DCFA + DepthPro + SIMCF-ABC pseudo-label pipeline raises raw pseudo-label PQ from 24.54 to 25.85. When these refined pseudo-labels supervise the CUPS-style detector and self-training pipeline, the resulting model reaches 35.83 PQ on Cityscapes val, exceeding the published CUPS baseline of 27.8 PQ while requiring no stereo or video at pseudo-label generation time. The results suggest that scene-centric panoptic structure factorizes into semantic appearance, monocular geometry, and cross-modal label consistency, and that improving pseudo-label quality at each factor can be amplified by downstream self-training.

## Methodology

### Problem Setup

Let \(x \in \mathbb{R}^{3 \times H \times W}\) be a monocular RGB image over pixel domain \(\Omega\), and let \(D: \Omega \to \mathbb{R}_{+}\) be a normalized monocular depth map. The goal is to produce a panoptic map

\[
P(p) = (y(p), i(p)), \qquad p \in \Omega,
\]

where \(y(p)\) is a semantic label and \(i(p)\) is an instance identifier, with \(i(p)=0\) for stuff regions and \(i(p)>0\) for thing objects. The method constructs \(P\) without human labels by composing semantic pseudo-labels, depth-derived instances, and cross-modal consistency filtering.

### Frozen Semantic Codes and Overclustering

We first extract frozen semantic features using a DINOv2/CAUSE-TR pipeline. A frozen ViT backbone produces patch tokens, and the CAUSE-TR Segment_TR head maps them to a compact 90-dimensional code:

\[
z_u = g_{\mathrm{CAUSE}}(\phi_{\mathrm{DINO}}(x))_u \in \mathbb{R}^{90},
\]

for patch or upsampled pixel location \(u\). The codes are L2-normalized before clustering:

\[
\tilde{z}_u = \frac{z_u}{\|z_u\|_2 + \epsilon}.
\]

We fit \(K=80\) k-means centroids \(\{\mu_k\}_{k=1}^{K}\) and assign each location to its nearest centroid:

\[
s_0(u) = \arg\min_{k \in \{1,\ldots,K\}} \|\tilde{z}_u - \mu_k\|_2^2.
\]

For evaluation and for cross-modal filtering, clusters are mapped to train IDs by a fixed cluster-to-class function

\[
\pi: \{1,\ldots,K\} \to \{0,\ldots,C-1,255\},
\]

where \(255\) denotes ignore. In the training pseudo-labels, the cluster identity can be retained to preserve overclustered granularity; in evaluation, \(\pi\) or global matching resolves clusters to canonical classes.

### DCFA: Depth-Conditioned Feature Adapter

DCFA adapts the frozen 90-dimensional semantic code using local depth while preserving the original feature geometry. For a normalized depth value \(d_u \in [0,1]\), we use eight octave-spaced frequencies

\[
\omega_m = 2^m,\qquad m=0,\ldots,7,
\]

and define a 16-dimensional sinusoidal encoding:

\[
e(d_u) =
\left[
\sin(\pi \omega_0 d_u), \cos(\pi \omega_0 d_u),
\ldots,
\sin(\pi \omega_7 d_u), \cos(\pi \omega_7 d_u)
\right] \in \mathbb{R}^{16}.
\]

The adapter is a residual MLP:

\[
h_u^{(1)} = \sigma\!\left(\mathrm{LN}(W_1[z_u;e(d_u)] + b_1)\right),
\]

\[
h_u^{(2)} = \sigma\!\left(\mathrm{LN}(W_2 h_u^{(1)} + b_2)\right),
\]

\[
\hat{z}_u = A_\theta(z_u,d_u) = z_u + W_o h_u^{(2)} + b_o.
\]

The final projection \(W_o,b_o\) is zero-initialized, so \(A_\theta\) begins as the identity map. The adapter is trained with depth-guided correlation and preservation:

\[
\mathcal{L}_{\mathrm{DCFA}}
=
\mathcal{L}_{\mathrm{depth}}
+ \lambda_{\mathrm{preserve}}
\|\hat{z}_u - z_u\|_2^2.
\]

For sampled patch pairs \((u,v)\), the depth-correlation term is

\[
\mathcal{L}_{\mathrm{depth}}
=
\mathbb{E}_{(u,v)}
\left[
w_{uv}
\left(1 -
\frac{\hat{z}_u^\top \hat{z}_v}
{\|\hat{z}_u\|_2 \|\hat{z}_v\|_2 + \epsilon}
\right)^2
\right],
\]

where

\[
w_{uv} =
\exp\left(
-\frac{(D(u)-D(v))^2}{2\sigma_d^2}
\right).
\]

Thus, locations with similar depth are encouraged to remain close in the adapted semantic code space, while the preservation term prevents large semantic drift. In the final label-generation path, the k-means input can concatenate the adapted semantic code and a scaled depth encoding:

\[
\bar{z}_u = [\hat{z}_u;\alpha e(d_u)],
\]

with \(\alpha=0.1\) in the inspected final-generation notes. K-means is then applied to \(\bar{z}\) to produce depth-aware semantic clusters \(s_{\mathrm{DCFA}}\).

### Depth-Guided Instance Generation

Semantic clusters assign category-like labels but do not separate individual objects. We obtain instance proposals from monocular depth. Given depth map \(D\), optionally smoothed by a Gaussian kernel, we compute Sobel gradients:

\[
G_x = S_x * D, \qquad G_y = S_y * D,
\]

\[
G(p) = \sqrt{G_x(p)^2 + G_y(p)^2}.
\]

Depth discontinuities are thresholded as

\[
E(p) = \mathbf{1}[G(p) > \tau_d].
\]

For each thing class \(c \in \mathcal{T}\), where \(\mathcal{T}\) contains the countable Cityscapes thing classes, we form a semantic mask

\[
M_c = \{p \in \Omega : \pi(s(p)) = c\}.
\]

Depth edges are removed from the mask:

\[
M'_c = M_c \setminus \{p : E(p)=1\}.
\]

Connected components are then extracted:

\[
\{C_{c,j}\}_{j=1}^{n_c} = \mathrm{CC}(M'_c).
\]

Small regions are discarded:

\[
C_{c,j} \text{ is kept iff } |C_{c,j}| \ge A_{\min}.
\]

Finally, each kept component is dilated and allowed to reclaim unassigned pixels from its original semantic class mask. This produces an instance map \(I_0: \Omega \to \mathbb{N}_0\). The final scripts use \(\tau_d=0.20\), \(A_{\min}=1000\), dilation radius/iterations \(r=3\), and no Gaussian smoothing in the inspected DepthPro generation path.

### SIMCF-ABC: Semantic-Instance Mutual Consistency Filtering

The semantic map and the instance map are produced by different signals and therefore make different errors. SIMCF-ABC refines them by enforcing three forms of cross-modal agreement.

#### Step A: Instances Validate Semantics

For each instance \(I_k = \{p : I_0(p)=k\}\), we compute the majority train ID:

\[
c_k^* =
\arg\max_{c}
\sum_{p \in I_k}
\mathbf{1}[\pi(s(p))=c].
\]

Pixels inside the instance whose semantic class disagrees with \(c_k^*\) are reassigned to the most frequent cluster within the instance that maps to \(c_k^*\):

\[
s_k^* =
\arg\max_{q:\pi(q)=c_k^*}
\sum_{p \in I_k}
\mathbf{1}[s(p)=q],
\]

\[
s(p) \leftarrow s_k^*
\quad \text{if } p \in I_k
\text{ and } \pi(s(p)) \ne c_k^*.
\]

This step enforces semantic uniformity within each proposed object.

#### Step B: Semantics Validate Instances

Raw depth splitting can over-fragment a single object when depth gradients arise from object surface curvature rather than object boundaries. SIMCF-B builds an adjacency graph over instances. Two instances are adjacent if their masks overlap after dilation:

\[
(i,j) \in \mathcal{E}
\iff
\mathrm{dilate}(I_i,r) \cap I_j \ne \emptyset.
\]

For each instance, we compute a normalized mean appearance feature from frozen foundation-model features \(F(p)\). In the final implementation, this Step-B feature source is DINOv3, while the 90-dimensional semantic codes come from DINOv2/CAUSE-TR:

\[
\bar{F}_i =
\frac{1}{|\tilde{I}_i|}
\sum_{p \in \tilde{I}_i}
\frac{F(p)}{\|F(p)\|_2+\epsilon},
\qquad
\hat{F}_i =
\frac{\bar{F}_i}{\|\bar{F}_i\|_2+\epsilon}.
\]

Adjacent instances are merged if they share the same semantic majority class and their feature similarity is high:

\[
\mathrm{merge}(i,j)
\iff
c_i^*=c_j^*
\;\wedge\;
\hat{F}_i^\top \hat{F}_j > \tau_{\mathrm{sim}}.
\]

The implementation uses \(\tau_{\mathrm{sim}}=0.85\) and resolves transitive merges with union-find.

#### Step C: Depth Validates Semantics

For each semantic class \(c\), we estimate global depth statistics over the pseudo-labeled training set:

\[
\mu_c =
\frac{1}{N_c}
\sum_{p:\pi(s(p))=c}
D(p),
\]

\[
\sigma_c^2 =
\frac{1}{N_c}
\sum_{p:\pi(s(p))=c}
(D(p)-\mu_c)^2.
\]

A pixel assigned to class \(c\) is marked ignore if its depth is a statistical outlier:

\[
s(p) \leftarrow 255
\quad \text{if} \quad
|D(p)-\mu_c| > \lambda_\sigma \sigma_c.
\]

The implementation uses \(\lambda_\sigma=3.0\). This step does not invent new labels; it only removes geometrically implausible supervision.

### Panoptic Assembly

After SIMCF, the refined semantic map \(s'\) and instance map \(I'\) are assembled into a panoptic target:

\[
P(p)=
\begin{cases}
(\pi(s'(p)), I'(p)), & \pi(s'(p)) \in \mathcal{T},\ I'(p)>0,\\
(\pi(s'(p)), 0), & \pi(s'(p)) \notin \mathcal{T},\\
255, & s'(p)=255.
\end{cases}
\]

The resulting pseudo-labels are exported in the CUPS-compatible format with semantic PNGs, instance PNGs, and class distribution tensors.

### Downstream Training and Self-Training

The refined pseudo-labels supervise the downstream panoptic network using the CUPS-style Cascade Mask R-CNN training and self-training recipe. This stage is best described as an evaluation amplifier rather than the core contribution: the method contributes the pseudo-label construction, while the detector and EMA self-training convert cleaner supervision into stronger predictions. In the current artifacts, refined pseudo-labels improve the trained Stage-3 model from a DepthPro-only baseline of 31.62 PQ to 35.83 PQ on Cityscapes val.

### What Not To Claim

- Do not claim the final method uses LoRA, DoRA, or Lambda adapters. Those exist in experimental code paths, not in this paper narrative.
- Do not present 35.83 PQ as raw pseudo-label quality. It is the trained model result.
- Do not claim a strictly annotation-free pseudo-label pipeline if the GT-derived `cluster_to_class` mapping remains in the final generation/refinement path. Either replace it or state precisely where mapping supervision enters.
- Do not imply a fully fair apples-to-apples comparison against CUPS unless the same backbone and self-training controls are included. The safer wording is "exceeds the published CUPS baseline" and then state the remaining control experiment in limitations.
- Do not claim a theorem of orthogonality unless a formal proof is added. The evidence currently supports "empirically near-additive" or "empirically complementary."
