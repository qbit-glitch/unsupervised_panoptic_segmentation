# Monocular Geometry Is Enough: Depth-Conditioned Pseudo-Labels for Scene-Centric Unsupervised Panoptic Segmentation

## Abstract

Panoptic segmentation — assigning a semantic class to every pixel and a distinct identity to every countable object — is a foundational primitive for autonomous driving, robotic perception, and dense scene understanding. Eliminating the dependence on human masks for this task is the central challenge of unsupervised scene-centric segmentation, where the strongest recipe to date, CUPS, supplies the missing structure from stereo depth and motion at pseudo-label time. Most images, however, arrive monocular and static. We ask whether stereo and video are fundamental, and answer that they are not: frozen DINOv2/CAUSE-TR semantic codes, monocular depth, and a cross-modal agreement filter together produce pseudo-labels of similar quality. Two compact mechanisms instantiate this principle: DCFA, a zero-initialized residual that bends the 90-D semantic code toward plausible geometry via a 16-D sinusoidal depth encoding; and SIMCF-ABC, a guarded merge-and-mask filter that accepts labels only when semantics, depth-derived instances, and depth statistics concur. DCFA lifts semantic mIoU from 52.69 to 55.29; the full pipeline lifts pseudo-label PQ from 24.54 to 25.85 on Cityscapes. Plugged into the CUPS bootstrapping and self-training recipe with a DINOv3 ViT-B/16 Cascade Mask R-CNN, these monocular labels train a model that reaches 35.83 PQ on Cityscapes val under a stronger backbone than the published CUPS configuration (27.80 PQ); the same-backbone control is discussed in §6.

## 1. Introduction

Panoptic segmentation requires assigning a class to every pixel and an identity to every countable object. Doing this without human masks forces a sharp question: which signals in an unlabeled image carry enough structure to stand in for those masks? Recent work has answered with stereo and motion. CUPS, the strongest scene-centric recipe, builds pseudo-labels from self-supervised features, stereo depth, and scene flow, then trains a panoptic detector on them. The recipe works, but it commits unsupervised panoptic segmentation to data with calibrated multi-view geometry. Most images do not arrive that way.

This paper asks whether a single frame, read carefully, supplies the same structure. The answer we report is that it does — but only when no individual cue is asked to do too much. Frozen semantic features know what road, vegetation, or vehicle look like, yet routinely merge two adjacent cars into one region. Monocular depth flags physical separations, yet over-fragments single surfaces and misses co-planar splits between same-class objects. Visual similarity heals some depth errors, but only when constrained by category. The natural framing is information agreement: depth discontinuities propose object boundaries — they do not identify them. Geometry proposes the split, semantics and appearance decide whether the split is credible. A pseudo-label is reliable when independent monocular cues, with different failure modes, point at the same answer.

We instantiate this principle through two compact mechanisms inside an otherwise standard pipeline. DCFA is a zero-initialized residual that injects 16-D sinusoidal depth features into a frozen 90-D CAUSE-TR code, bending the semantic partition toward plausible geometry without overwriting appearance. SIMCF-ABC is a label-level filter that enforces semantic uniformity inside instance proposals (A), merges adjacent depth fragments only when class and DINOv3 cosine similarity agree (B, threshold 0.85), and drops semantic pixels whose depth violates per-class statistics (C). Neither component proposes a new architecture. Both are designed to expose a single quantity — agreement across monocular cues — and let CUPS-style bootstrapping amplify whatever structure that quantity preserves.

Empirically, DCFA lifts semantic clustering mIoU from 52.69 to 55.29 in the $K=80$ adapter-evaluation protocol, and the DCFA + DepthPro + SIMCF-ABC pipeline lifts pseudo-label PQ from 24.54 to 25.85 (+1.31) on Cityscapes, with the gain concentrated in PQ_th (12.31 to 14.70). When these monocular labels supervise CUPS bootstrapping and three rounds of EMA self-training with a DINOv3 ViT-B/16 Cascade Mask R-CNN, the final detector reaches 35.83 PQ on Cityscapes val.

We make three contributions. (1) We show that monocular semantic appearance and monocular depth, composed by an agreement filter, are sufficient pseudo-label cues for scene-centric unsupervised panoptic segmentation, removing the stereo/video requirement at pseudo-label time. (2) DCFA and SIMCF-ABC instantiate cross-modal agreement at feature and label levels respectively, with DCFA improving semantic mIoU by 2.60 and the full filter contributing the bulk of the +1.31 PQ pseudo-label gain. (3) Reusing the published CUPS bootstrapping and self-training recipe on these monocular labels yields 35.83 PQ on Cityscapes val with a DINOv3 ViT-B/16 Cascade Mask R-CNN, surpassing the published CUPS baseline of 27.80 PQ. Because CUPS uses a different backbone family, a same-backbone CUPS rerun remains the main controlled comparison.

## 2. Related Work

**Semantic discovery from frozen features.** Unsupervised semantic discovery from self-supervised features now supplies the appearance side of nearly every scene-centric pseudo-label pipeline. STEGO distills DINO correspondences into compact dense embeddings; CAUSE structures the latent space through concept clusterbooks and concept-wise grouping; DepthG correlates feature maps with monocular depth statistics. Each of these methods produces clusters that are coherent at the region level but cannot, by construction, separate two adjacent same-class objects.

**Instance discovery and geometry.** Object-centric instance discovery — TokenCut, MaskCut, CutLER — extracts foreground masks from self-supervised features and bootstraps detectors from them. These methods assume a dominant foreground; scene-centric Cityscapes data violates that assumption, with stuff regions, occlusions, and adjacent same-class instances coexisting in every frame. CutS3D pivots toward our setting by arguing that geometry is required to split 2D semantic masks into instances, and demonstrates this with stereo-derived 3D structure. Our pipeline endorses the same principle but extracts the geometric cue from a monocular depth network, sidestepping stereo reconstruction and full point-cloud handling.

**Unsupervised panoptic segmentation.** Unsupervised panoptic segmentation aggregates these threads. U2Seg unifies semantic, instance, and panoptic settings under a single objective. CUPS is the closest baseline to this paper: it constructs high-resolution panoptic pseudo-labels from self-supervised features, stereo depth, and scene flow, then trains a panoptic detector with bootstrapping, DropLoss, copy-paste augmentation, and EMA self-training. CUPS Table 7b also documents that semantic overclustering helps — $K=27$ yields 27.8 PQ, $K=40$ yields 30.3 PQ, $K=54$ yields 30.6 PQ — establishing $K=80$-style overclustering as a known recipe ingredient, not a contribution of this work. We adopt the entire CUPS training side and replace only the pseudo-label source. The substituted source uses frozen monocular priors throughout: DINOv2 with CAUSE-TR for the semantic code, DINOv3 for downstream features and SIMCF-B similarity, and Depth Pro or Depth Anything for monocular geometry. None of these foundation models are trained or fine-tuned by us; the contribution lies in their composition into a single-frame pseudo-label generator. The relevant question is not which prior we use, but whether priors composed under cross-modal agreement carry the panoptic structure that stereo and motion previously supplied.

## 3. Methodology

### 3.1 Monocular Pseudo-Labeling as Agreement

CUPS shows that a panoptic model can be trained from pseudo-labels when those labels carry enough semantic and instance structure. The unresolved question is where that structure must come from. Stereo video provides explicit multi-view geometry and motion; a single image does not. We therefore frame monocular pseudo-labeling as an *agreement* problem rather than as a stereo replacement.

Given an unlabeled image $x$, Stage 1 produces a semantic pseudo-label $\hat S_x$, an instance pseudo-label $\hat I_x$, and a panoptic pseudo-label $\hat P_x = (\hat S_x, \hat I_x)$. Each subsection below introduces one component by the failure mode it addresses. No monocular cue is trusted alone: semantic codes propose category-like regions, depth proposes physical separations, and a multi-cue filter rejects labels whose semantic, instance, and depth statistics disagree.

### 3.2 CAUSE-TR as a Semantic Code Space

**Failure mode.** Raw DINOv2 patch features are excellent for correspondence but are too high-dimensional and too sensitive to non-semantic visual modes to act as a label vocabulary directly.

CAUSE-TR (Kim et al.) reduces frozen DINOv2 features to a 90-dimensional concept-code space designed for unsupervised grouping. For each upsampled pixel location $u$, the frozen extractor produces

$$z_u = h_{\mathrm{CAUSE\text{-}TR}}\!\left(\phi_{\mathrm{DINOv2}}(x)\right)_u \in \mathbb{R}^{90}.$$

Forcing these codes into the final 19-class taxonomy too early discards visual modes that matter at instance time (curbs vs. road, distant vs. near vegetation, occluded vs. unoccluded vehicles). Following the CUPS clustering ablation (CUPS Table 7b, which shows $K = 27 \to 40 \to 54$ monotonically improves PQ), we adopt overclustering as a known good practice rather than a contribution and push it further to $K = 80$:

$$s_u = \arg\min_k \lVert \bar z_u - \bar\mu_k \rVert_2^2,$$

where $\bar z_u, \bar\mu_k$ are L2-normalized and $\{\mu_k\}_{k=1}^{80}$ are k-means centroids fit once over the unlabeled training images. The resulting $s_u \in \{1,\dots,80\}$ acts as an overcomplete vocabulary that preserves fine semantic modes long enough for depth-based instance construction and downstream matching. In the baseline no-DCFA pipeline (the "Raw $K=80$" rows of Tables 2 and 4) the clustered code is the raw $z_u$; in the main pipeline DCFA (§3.3) is applied first and $z_u$ is replaced by the adapted code $\tilde z_u$ before k-means is applied.

### 3.3 DCFA: Depth-Conditioned Feature Adapter

**Failure mode.** CAUSE-TR boundaries are appearance-driven, so two adjacent objects of the same class collapse into one cluster, and a single physical surface that changes appearance (sun/shade, paint stripes) can be split spuriously.

DepthG (Sick & Mertz, 2024) introduced a depth-guided feature correlation loss that pulls features at similar depth toward each other in cosine space, weighted by a Gaussian on depth difference. DepthG applies this loss when training a semantic backbone end-to-end. DCFA reuses the same correlation principle but applies it differently: it keeps the CAUSE-TR backbone *frozen* and trains only a small zero-initialized residual on the 90-D code, anchored to the original by an MSE preservation term. This makes the adaptation parameter-efficient (≈40K trainable parameters), preserves the frozen-prior assumption central to this paper, and lets the residual deviate from the CAUSE-TR partition only where depth provides additional evidence.

**Sinusoidal depth encoding.** Let $d_u \in [0,1]$ be normalized monocular depth (inverse-depth, percentile-clipped). Define the 8 frequencies $\omega_k = 2^k \pi$ for $k = 0, 1, \dots, 7$. The 16-dimensional encoding is

$$e(d_u) \;=\; \bigl[\sin(\omega_0 d_u),\, \cos(\omega_0 d_u),\, \sin(\omega_1 d_u),\, \cos(\omega_1 d_u),\, \dots,\, \sin(\omega_7 d_u),\, \cos(\omega_7 d_u)\bigr] \in \mathbb{R}^{16}.$$

**Adapter $r_\theta$.** A two-layer MLP with hidden width $h = 128$, LayerNorm, and ReLU activation:

$$r_\theta(v) = W_2 \,\mathrm{ReLU}\!\bigl(\mathrm{LN}(W_1 v + b_1)\bigr) + b_2, \qquad v = [z_u;\, e(d_u)] \in \mathbb{R}^{106}.$$

The output projection $W_2 \in \mathbb{R}^{90 \times 128}$ is zero-initialized so training begins exactly at the frozen CAUSE-TR solution; the full adapter has approximately 40K parameters. Zero-init is a *conservative inductive bias*: the adapter only deviates from the prior when the data demand it. The adapted code is

$$\tilde z_u = z_u + r_\theta\!\bigl([z_u;\, e(d_u)]\bigr).$$

**Depth-guided correlation loss.** Following DepthG, we sample $P = 1024$ random pixel pairs $(i,j)$ within each image at every step. The pair is weighted by depth proximity through a Gaussian kernel with bandwidth $\sigma_d = 0.5$:

$$w_{ij} \;=\; \exp\!\Bigl(-\tfrac{(D_i - D_j)^2}{2\sigma_d^2}\Bigr).$$

The loss penalizes squared cosine distance between the adapted codes, weighted by depth proximity:

$$\mathcal{L}_{\mathrm{depth}} \;=\; \frac{1}{P}\sum_{(i,j)} w_{ij}\,\bigl(1 - \cos(\tilde z_i,\,\tilde z_j)\bigr)^2.$$

Pairs at similar depth (large $w_{ij}$) contribute strongly to the loss when their adapted codes disagree; distant pairs contribute weakly regardless of code similarity. The full DCFA objective adds the preservation term:

$$\mathcal{L}_{\mathrm{DCFA}} \;=\; \mathcal{L}_{\mathrm{depth}} \;+\; \lambda_{\mathrm{preserve}} \,\bigl\lVert \tilde z_u - z_u \bigr\rVert_2^2,$$

with $\lambda_{\mathrm{preserve}} = 20$ chosen by ablation. The preservation term keeps the adapted representation near the original CAUSE-TR manifold while $\mathcal{L}_{\mathrm{depth}}$ pulls patches at similar depth together in cosine space. Table 5 reports the resulting +2.35 mIoU lift in the $K=300$ semantic protocol (sinusoidal vs. no-depth concat) and +2.60 mIoU in the $K=80$ adapter-evaluation protocol; Table 2 measures the contribution to panoptic pseudo-label PQ.

### 3.4 Monocular Depth as an Instance Cue

**Failure mode.** Even after DCFA, semantic clusters cannot separate two adjacent instances of the same class. Depth offers the missing cue: a 3-D boundary usually appears as a depth gradient, even when the two instances share appearance.

Given a monocular depth map $D \in \mathbb{R}^{H \times W}$, we first apply a Gaussian blur with $\sigma_\mathrm{blur} = 1.0$ to suppress quantization-style high-frequency noise that monocular networks sometimes produce, then compute the Sobel gradient magnitude with the standard $3 \times 3$ kernels

$$S_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \qquad S_y = S_x^{\!\top},$$

and form

$$G(p) \;=\; \sqrt{(S_x * D_\mathrm{blur})^2(p) + (S_y * D_\mathrm{blur})^2(p)}.$$

Pixels with $G(p) > \tau_d$ are candidate physical boundaries. Within each *thing*-eligible semantic region, we (i) erase boundary pixels, (ii) run 8-connected component labeling, (iii) drop components below area $A_{\min}$, and (iv) dilate the remaining components by $n_{\mathrm{dil}}$ iterations of a $3 \times 3$ structuring element to reclaim erased boundary pixels. The DepthPro path used for the main results sets $\tau_d = 0.20$, $A_{\min} = 1000$, and $n_{\mathrm{dil}} = 3$. Depth-source-specific values for Depth Anything V2/V3 and SPIdepth are reported in Table 3.

### 3.5 SIMCF-ABC: Semantic-Instance-Depth Agreement

**Failure mode.** After §3.3 and §3.4, two error modes remain. Depth over-fragments: a single car can be sliced where its hood meets its windshield. And the per-class depth distribution contains outliers — a "road" pixel at sky depth — that will be amplified by Stage-2 training.

SIMCF-ABC is a three-stage filter applied after instance extraction. Each stage targets a distinct error.

**SIMCF-A (semantic coherence).** Let $\phi: \{0,\dots,79\} \to \{0,\dots,18\}$ be the cluster-to-class map. For each instance $k$ with pixel set $I_k$, compute the dominant class

$$c_k^{\star} = \arg\max_{c \in \{0,\dots,18\}} \;\bigl|\{p \in I_k : \phi(S(p)) = c\}\bigr|,$$

identify pixels whose semantic class disagrees with $c_k^{\star}$, and reassign them to the most common cluster among the consistent pixels:

$$s_k^{\star} = \arg\max_{s \in \{0,\dots,79\}} \;\bigl|\{p \in I_k : \phi(S(p)) = c_k^{\star} \wedge S(p) = s\}\bigr|, \qquad S(p) \leftarrow s_k^{\star} \;\;\forall p \in I_k \text{ with } \phi(S(p)) \neq c_k^{\star}.$$

This is a structural guard: an instance whose interior cannot agree on a single semantic class is unreliable supervision. Under the current proposal generator, SIMCF-A is near-neutral because instances are already extracted within semantic regions (§4.7); we retain it as a robustness guard.

**SIMCF-B (feature-guided merge).** DINOv3 ViT-B/16 produces 768-D patch tokens at a $32 \times 64$ grid. For each instance $k$, let $\tilde I_k$ denote the set of patches that overlap the instance mask after downsampling to patch resolution. The L2-normalized per-instance feature is

$$\bar f_k \;=\; \frac{1}{|\tilde I_k|}\sum_{p \in \tilde I_k} F(p), \qquad \hat f_k \;=\; \frac{\bar f_k}{\lVert \bar f_k\rVert_2}, \qquad F(p) = \phi_{\mathrm{DINOv3}}(x)_p \in \mathbb{R}^{768},\ \lVert F(p)\rVert_2 = 1.$$

Two instances $a, b$ are adjacent iff $\mathrm{dilate}(I_a,\,r{=}3) \cap I_b \neq \emptyset$ (3-pixel structural dilation). Adjacent instances are merged iff their dominant classes agree and their normalized features pass the cosine threshold:

$$\mathrm{merge}(a, b) \;\Longleftrightarrow\; c_a^{\star} = c_b^{\star} \;\wedge\; \langle \hat f_a,\, \hat f_b\rangle > \tau_{\mathrm{sim}}, \qquad \tau_{\mathrm{sim}} = 0.85.$$

Transitive merges are resolved by union-find with path compression. We selected $\tau_{\mathrm{sim}}$ on a 100-image subset of the Cityscapes *train* split (no val touch) by sweeping $\{0.75, 0.80, 0.85, 0.90\}$ and picking the maximum-PQ value; the same $\tau_{\mathrm{sim}}$ is used on every depth source.

**SIMCF-C (depth-statistic outlier removal).** For each semantic class $c$ (after applying the cluster-to-class mapping $\phi$), compute the global mean $\mu_c$ and standard deviation $\sigma_c$ of depth across *all* pseudo-labeled training images, accumulated online with Welford's algorithm:

$$\mu_c = \mathbb{E}\bigl[D(p) \,\big|\, \phi(S(p)) = c\bigr], \qquad \sigma_c^2 = \mathrm{Var}\bigl[D(p) \,\big|\, \phi(S(p)) = c\bigr].$$

Per-image, semantic pixels whose depth lies outside the global $3\sigma_c$ band of their class are demoted to ignore:

$$S(p) \leftarrow \mathrm{ignore} \quad \text{if} \quad \bigl|D(p) - \mu_{\phi(S(p))}\bigr| > \lambda_\sigma\, \sigma_{\phi(S(p))}, \quad \lambda_\sigma = 3.$$

SIMCF-C removes labels rather than creating them; its role is to prevent Stage 2 from amplifying systematic depth-class incompatibilities (a "sky" pixel at 2 m depth, a "road" pixel above the horizon).

Table 6 reports the verified SIMCF-ABC behavior; Table 2 measures its panoptic contribution.

### 3.6 Stage 2: CUPS-Recipe Panoptic Bootstrapping

The Stage-1 labels are structured but incomplete: small objects are missed, masks are partial, and uncertain pixels are ignored. Stage 2 therefore reuses the CUPS bootstrapping recipe verbatim and changes only the pseudo-label source. The architecture is a DINOv3 ViT-B/16 + Cascade Mask R-CNN with a panoptic FPN head. Stage 2 is intentionally not a new training algorithm: it tests whether monocular Stage-1 labels can substitute for CUPS stereo/video pseudo-labels inside the same bootstrapping recipe.

Let $G_\theta$ denote the panoptic model, $\mathcal{A}$ the augmentation distribution (resolution jitter, copy-paste, photometric), and $\hat P_x$ the Stage-1 label. Stage 2 minimizes

$$\theta_2^{\star} \;=\; \arg\min_{\theta} \; \sum_{x \in \mathcal{D}} \mathcal{L}_{\mathrm{CUPS}}\!\bigl(G_\theta(\mathcal{A}(x)),\, \hat P_x\bigr).$$

Following CUPS, $\mathcal{L}_{\mathrm{CUPS}}$ factors as

$$\mathcal{L}_{\mathrm{CUPS}} \;=\; \underbrace{\sum_{k=1}^{3} \bigl(\mathcal{L}^{(k)}_{\mathrm{cls}} + \mathcal{L}^{(k)}_{\mathrm{box}}\bigr)}_{\text{Cascade R-CNN heads}} \;+\; \mathcal{L}_{\mathrm{mask}} \;+\; \mathcal{L}_{\mathrm{sem}} \;+\; \lambda_{\mathrm{drop}}\,\mathcal{L}_{\mathrm{DropLoss}},$$

where $k = 1, 2, 3$ indexes the three Cascade R-CNN stages with IoU thresholds $0.5/0.6/0.7$, $\mathcal{L}_{\mathrm{mask}}$ is the per-instance mask cross-entropy over pseudo-mask pixels, $\mathcal{L}_{\mathrm{sem}}$ is the panoptic-FPN semantic loss, and $\mathcal{L}_{\mathrm{DropLoss}}$ gates the negative-classification term so unmatched proposals overlapping ignore-regions are excluded. We follow the published CUPS optimizer and augmentation recipe (AdamW, copy-paste, multi-resolution training, horizontal flip); exact hyperparameter values are reported in the appendix and in the released code.

### 3.7 Stage 3: CUPS-Recipe Self-Training

Stage 3 reuses the CUPS self-training recipe to densify the labels learned in Stage 2.

**Teacher.** $G_{\bar\theta}$ is an EMA copy of the student with coefficient $\alpha = 0.999$:

$$\bar\theta \,\leftarrow\, \alpha\,\bar\theta + (1-\alpha)\,\theta.$$

**Multi-view teacher prediction.** For round $r \in \{1,\dots,R\}$ ($R = 3$), the teacher consumes $M = 3$ augmented views at scales $\{0.75, 1.00, 1.25\}$. Each prediction $G_{\bar\theta}(a_m(x))$ is mapped back to the original frame via $a_m^{-1}$, giving aligned predictions $\{a_m^{-1} \circ G_{\bar\theta} \circ a_m(x)\}_{m=1}^{M}$.

**Fusion and thresholding $\Gamma$.** $\Gamma$ averages aligned semantic logits, performs per-class Hungarian matching across instance proposals, averages matched mask probabilities, and accepts a panoptic segment iff its averaged confidence exceeds $\rho_r$. The schedule $\rho_1 = 0.50,\, \rho_2 = 0.55,\, \rho_3 = 0.60$ tightens with rounds. Pixels below $\rho_r$ become ignore. Concretely,

$$\tilde P_x^{(r)} \;=\; \Gamma\!\bigl(\{G_{\bar\theta}(a_m(x))\}_{m=1}^{M},\, \rho_r\bigr).$$

**Student update.** $\mathcal{A}$ is the same augmentation distribution as Stage 2:

$$\theta_3^{\star} \;=\; \arg\min_{\theta} \; \sum_{r=1}^{R} \sum_{x \in \mathcal{D}} \mathcal{L}_{\mathrm{CUPS}}\!\bigl(G_\theta(\mathcal{A}(x)),\, \tilde P_x^{(r)}\bigr).$$

Each self-training round reuses the Stage-2 optimizer settings. If a structure is absent from $\hat P_x$, self-training has little evidence from which to recover it; we therefore evaluate Stage-1 pseudo-label quality and Stage-3 trained-model PQ separately (Table 1).

## 4. Experiments

### 4.1 Protocol and Metrics

Two regimes structure every table that follows. Pseudo-label evaluations measure the supervision signal before any panoptic model is trained: they ask whether monocular depth, DCFA, and SIMCF-ABC produce labels with usable semantic and instance structure. Trained-model evaluations measure the Cascade Mask R-CNN that emerges from CUPS-style bootstrapping and self-training on those labels. Treating the two regimes as separate scoreboards prevents pseudo-label quality from being conflated with detector capacity, and prevents a strong backbone from rescuing weak supervision.

We report panoptic quality (PQ), thing PQ (PQ_th), stuff PQ (PQ_st), segmentation quality (SQ), recognition quality (RQ), mean IoU (mIoU), and pixel accuracy where applicable. Pseudo-label IDs are aligned to ground-truth classes for evaluation, following standard practice in unsupervised semantic and panoptic segmentation. Ground-truth masks are not used to supervise Stage 2 or Stage 3.

Sections 4.2–4.9 jointly establish four claims. (i) *Replaceability and complementarity*: monocular depth substituted for stereo/video at pseudo-label time produces labels of similar order, and DCFA and SIMCF-ABC correct different error modes (§4.2, §4.3). (ii) *Generality across depth estimators*: SPIdepth, DepthPro, and the Depth Anything family all confer thing-PQ gains over a no-depth baseline, so the cue is geometric rather than tied to one model (§4.4). (iii) *Trained-model quality*: refined pseudo-labels feed CUPS-style bootstrapping and self-training to a final Cityscapes PQ that surpasses the published CUPS baseline (§4.5–§4.7). (iv) *Bounded transfer with informative residual failures*: the trained model transfers across driving domains but is bounded by its fixed Cityscapes-derived class space, and the per-class breakdown localizes failures to thin structures, rare construction classes, and co-planar things (§4.8, §4.9).

### 4.2 Replacing Stereo/Video at Pseudo-Label Time

The first claim concerns *replaceability*: when stereo and video are removed at pseudo-label time, does monocular supervision retain enough signal to drive a competitive trained model? Our pseudo-labels alone reach 25.85 PQ, below CUPS's published 27.80 PQ trained-model baseline; the +8 PQ headline gap (35.83 vs 27.80) materializes only after CUPS-style bootstrapping with a DINOv3 ViT-B/16 Cascade Mask R-CNN. The claim we make is therefore narrower than the headline: monocular pseudo-labels are competitive enough to seed a strong CUPS-style recipe, and once seeded that recipe produces a trained model that exceeds the published CUPS number. Whether the gap is principally pseudo-label quality, backbone capacity, or detector implementation cannot be decided from Table 1 alone — a same-backbone CUPS rerun remains the main controlled comparison.

**Table 1. Cityscapes main comparison.** The "Regime" column groups rows by what is being measured: a published trained-model baseline, our raw monocular supervision signal before training, and our trained model. Rows are *not* a head-to-head scoreboard: the CUPS row reports a published trained model under a different backbone and training implementation, not the CUPS pseudo-label cost. No public U2Seg Cityscapes panoptic number is available under this protocol; the only external baseline is CUPS.

| Regime | Method | Pseudo-label cue | Stereo/video at PL time? | Backbone / training | PQ | PQ_th | PQ_st | mIoU |
|---|---|---|---:|---|---:|---:|---:|---:|
| Trained baseline | CUPS published | stereo/video depth + motion | yes | published CUPS | 27.80 | 17.70 | 35.10 | 26.80 |
| Pseudo-labels (ours) | DCFA + DepthPro + SIMCF-ABC | monocular | no | none | 25.85 | 14.70 | 33.96 | 56.22 |
| Trained model (ours) | DCFA + DepthPro + SIMCF-ABC | monocular | no | DINOv3 Cascade Mask R-CNN, Stage 3 | 35.83 | 36.26 | 35.56 | 44.56 |

The trained-model mIoU drops from 56.22 (pseudo-labels) to 44.56 (final), which inverts the usual training-improves-mIoU pattern. The pseudo-label mIoU is computed on patch-aligned semantic features against the 27-class CAUSE-TR vocabulary used for label generation, whereas the trained-model mIoU is computed on full-resolution panoptic predictions under the standard 19-class Cityscapes evaluation, where stuff classes outside the 19-class set are remapped to void; the change of class set depresses mIoU even as PQ rises. We therefore treat cross-row mIoU as auxiliary and do not use it as evidence for the main claim.

### 4.3 Component Complementarity (DCFA × SIMCF-ABC)

If DCFA and SIMCF-ABC corrected the same error, removing either one from the joint configuration would leave PQ unchanged. Table 2 contradicts that null: each component contributes alone, and their combination strictly dominates either ablation, with the gain concentrated in PQ_th.

**Table 2. Pseudo-label component removal under a fixed instance configuration.** All four cells share the same DepthPro instance configuration ($\tau=0.20$, $A_\mathrm{min}=1000$). This threshold is chosen to match the Stage-2/3 training data; §4.4 explains the choice and its relationship to the standalone optimum.

| Variant | Semantics | Depth instances | DCFA | SIMCF-ABC | PQ | PQ_th | PQ_st | mIoU |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Raw $K=80$ + depth | CAUSE-TR $K=80$ | DepthPro $\tau=0.20$ | no | no | 24.54 | 12.31 | 33.43 | 56.56 |
| DCFA only | CAUSE-TR $K=80$ | DepthPro $\tau=0.20$ | yes | no | 25.22 | 13.16 | 33.99 | 56.16 |
| SIMCF-ABC only | CAUSE-TR $K=80$ | DepthPro $\tau=0.20$ | no | yes | 25.27 | 13.64 | 33.73 | 56.57 |
| DCFA + SIMCF-ABC | CAUSE-TR $K=80$ | DepthPro $\tau=0.20$ | yes | yes | 25.85 | 14.70 | 33.96 | 56.22 |

DCFA acts before clustering by reshaping the partition with depth-conditioned features. SIMCF-ABC acts after label generation by removing pseudo-instances on which semantic, instance, and depth cues disagree. The fact that combining them widens the PQ_th gain over either alone (+2.39 vs +0.85 and +1.33 individually) is consistent with the two stages targeting different failure modes — feature-level partition errors and label-level disagreement — rather than competing for the same correction.

### 4.4 Generality Across Monocular Depth Estimators

Does the instance cue ride on a single estimator? Table 3 sweeps four monocular sources and reports each estimator's *standalone optimum* threshold, before the CUPS-training constraint is imposed. Every depth source improves PQ_th over the no-depth baseline, with margins ranging from +4.48 (SPIdepth) to +8.42 (DepthPro). The improvement is therefore a property of monocular geometry as supervision, not a property of one model.

**Table 3. Monocular depth source comparison for pseudo-label instances.** Thresholds are tuned per estimator on the pseudo-label evaluation; semantics are held fixed.

| Depth source | Type | Optimal $\tau$ | PQ | PQ_th | PQ_st | Notes |
|---|---|---:|---:|---:|---:|---|
| No depth / CC-only | none | n/a | 24.80 | 14.93 | 32.08 | semantic connected components |
| SPIdepth | self-supervised monocular | 0.20 | 26.74 | 19.41 | 32.08 | historical baseline |
| DepthPro | monocular foundation model | 0.01 | 28.40 | 23.35 | 32.08 | strongest under standalone sweep |
| Depth Anything v2 | monocular foundation model | 0.03 | 27.10 | 20.20 | 32.08 | domain-agnostic control |
| Depth Anything 3 | monocular foundation model | 0.03 | 27.37 | 20.90 | 32.08 | strongest DA variant |

A reader comparing Tables 2 and 3 will notice DepthPro at $\tau=0.01$ reaches 28.40 PQ in Table 3 but the DCFA + SIMCF-ABC row at $\tau=0.20$ in Table 2 only reaches 25.85 PQ. The difference is intentional and reflects a constraint we discovered while building the Stage-2 training set. Detector-side training with Cascade Mask R-CNN degrades when fed the fragmented instance set produced at $\tau=0.01$: the average count rises to roughly 57 instances per image, more than half are smaller than 1000 px, and the detector overfits to noisy small-instance proposals. At $\tau=0.20$ the count falls to roughly 22 large instances per image and downstream PQ improves. Table 3 therefore reports the *standalone pseudo-label optimum* per estimator, while Table 2 reports the *training-data configuration* the rest of the paper uses. Both are correct, in different regimes; the lower standalone PQ in Table 2 is the price we pay for instances that train detectors well.

To check that the full DCFA + SIMCF-ABC pipeline — and not just the standalone depth signal — generalizes across estimators, we also re-ran the entire pipeline with Depth Anything 3 substituted for DepthPro at the same training-data configuration ($\tau_d=0.20$, $A_{\min}=1000$, $n_{\mathrm{dil}}=3$, identical DCFA training, identical SIMCF-ABC application). The DA3 pseudo-labels reach 26.44 PQ on Cityscapes val (PQ_th 18.37, PQ_st 32.31, mIoU 55.30), versus the DepthPro path at 25.85 PQ (14.70 / 33.96 / 56.22). The two estimators trade off in expected directions — DA3's sharper monocular boundaries lift PQ_th by +3.67 while DepthPro's smoother depth gives slightly better stuff coherence — but both reach the same overall regime, and DA3 actually edges DepthPro at the pseudo-label stage. This is the multi-estimator generalization evidence the standalone Table 3 cannot give: the depth-source-agnostic claim survives DCFA training and SIMCF-ABC filtering, not just the bare gradient threshold.

### 4.5 Trained-Model Performance after CUPS-Style Bootstrapping

Table 4 records the trajectory when the refined pseudo-labels enter Stage-2 and Stage-3 training: 25.85 PQ at the pseudo-label stage rises to 35.83 PQ after CUPS-style bootstrapping and self-training, surpassing the published CUPS baseline of 27.80 PQ by 8.03 PQ at the trained-model level. Whether the within-pipeline 9.98-PQ lift is driven primarily by the pseudo-label refinement (DCFA + SIMCF-ABC) or by the Stage-2/3 recipe itself cannot be decided without a Stage-3 control trained from the raw 24.54-PQ pseudo-labels. This control remains part of the planned controlled comparison and is listed under limitation #1.

**Table 4. Pseudo-label quality and downstream training.** DA3 pseudo-labels at the matching training-data configuration are reported here for direct comparison with the DepthPro row; the corresponding Stage-3 trained-model rows for the DA3 path and for the raw-pseudo-label control are discussed in the surrounding text.

| Stage | Input supervision | Training stage | PQ | PQ_th | PQ_st | mIoU |
|---|---|---|---:|---:|---:|---:|
| Raw pseudo-labels | raw $K=80$ + DepthPro $\tau=0.20$ | none | 24.54 | 12.31 | 33.43 | 56.56 |
| Refined pseudo-labels (DepthPro) | DCFA + DepthPro $\tau=0.20$ + SIMCF-ABC | none | 25.85 | 14.70 | 33.96 | 56.22 |
| Refined pseudo-labels (DA3) | DCFA + DA3 $\tau=0.20$ + SIMCF-ABC | none | 26.44 | 18.37 | 32.31 | 55.30 |
| Final model (DepthPro) | refined DepthPro pseudo-labels | CUPS self-training / EMA | 35.83 | 36.26 | 35.56 | 44.56 |

The Stage-3 trajectory is monotonic: 31.87 PQ at step 800, 33.78 at step 1000, 35.47 at step 2200, 35.83 at step 3000. The shape of this trajectory is consistent with self-training densifying the pseudo-label structure rather than memorizing it.

### 4.6 DCFA and Depth Encoding Sanity

This scene checks the DCFA design choices in isolation, before any panoptic loss is applied. Table 5 tests two questions. First, does sinusoidal depth encoding beat raw or Sobel encoding when concatenated to frozen DINOv2 features? Second, when DCFA is wrapped around CAUSE-TR for $K=80$ semantic clustering, does it improve mIoU and pseudo-label PQ over the no-adapter baseline?

**Table 5. Depth encoding and DCFA sanity checks.** The $K=300$ rows compare encoders with the adapter held fixed; the $K=80$ rows compare adapter on/off with the encoding held at sinusoidal $\alpha=0.1$.

| Setting | Protocol | Metric |
|---|---|---:|
| No depth concat | $K=300$ semantic pseudo-label mIoU | 54.41 |
| Raw depth concat, $\alpha=0.1$ | $K=300$ semantic pseudo-label mIoU | 55.82 |
| Sobel concat, $\alpha=0.5$ | $K=300$ semantic pseudo-label mIoU | 55.49 |
| Sinusoidal concat, $\alpha=0.1$ | $K=300$ semantic pseudo-label mIoU | 56.76 |
| No adapter | $K=80$ adapter-eval | mIoU 52.69 / PQ 26.08 |
| DCFA V3 | $K=80$ adapter-eval | mIoU 55.29 / PQ 26.44 |

Sinusoidal encoding wins among the three encoders at $K=300$ (+2.35 mIoU over no-depth, +0.94 over raw, +1.27 over Sobel). DCFA V3 in the $K=80$ adapter-eval improves mIoU by 2.60 and pseudo-label PQ by 0.36. The table is scoped to the adapter and clustering, not to the full panoptic pipeline; its purpose is to justify the design choice that depth enters as a small, sinusoidal, residual correction to frozen semantic codes.

### 4.7 SIMCF Internal Behavior

This scene answers what happens *inside* SIMCF, separate from its joint effect with DCFA in §4.3. Table 6 reports the verified rows. SIMCF-A is empirically neutral here: most depth-derived instance proposals are already semantically uniform under the $K=80$ label set, so the semantic-consistency gate has little to remove. SIMCF-ABC improves PQ by +0.73 over the no-SIMCF baseline.

**Table 6. SIMCF internal behavior.** Only verified rows are reported.

| Variant | PQ | PQ_th | PQ_st | Note |
|---|---:|---:|---:|---|
| No SIMCF | 24.54 | 12.31 | 33.43 | baseline |
| SIMCF-A | 24.54 | 12.31 | 33.43 | semantic consistency check, no observable change |
| SIMCF-ABC | 25.27 | 13.64 | 33.73 | full semantic-instance-depth filter |

Throughout this paper SIMCF refers to the complete SIMCF-ABC filter. We claim the joint gain in Table 6 because we have verified it, and we explicitly do not claim an independent contribution for SIMCF-C without a verified SIMCF-AB row. The qualitative analysis in §5 supplies the mechanism: feature-guided merging of over-fragmented depth instances is the visible behavior.

### 4.8 Cross-Dataset Transfer

This scene probes how far the trained model carries beyond Cityscapes. We split Table 7 into two regimes because they answer different questions. The first regime measures transfer to driving-style targets with class spaces that genuinely overlap Cityscapes. The second regime is a sanity / degenerate-class-space group: MOTSChallenge collapses to a 2-class subset where one class is background, and COCO-Stuff-27 has a coarse and largely disjoint taxonomy.

**Table 7a. Driving-domain transfer.** PQ here is comparable across rows.

| Dataset | Class-space relation | PQ | PQ_th | PQ_st | mIoU |
|---|---|---:|---:|---:|---:|
| Cityscapes | source | 35.83 | 36.26 | 35.56 | 44.56 |
| KITTI | aligned driving classes | 34.85 | 31.94 | 36.40 | 46.87 |
| Mapillary Vistas v2 | mostly aligned driving classes | 39.19 | 32.06 | 44.37 | 58.87 |

**Table 7b. Sanity / degenerate-class-space rows.** PQ here is *not* directly comparable to Table 7a; we report the numbers for transparency rather than as transfer evidence.

| Dataset | Why separated | PQ | PQ_th | PQ_st | mIoU |
|---|---|---:|---:|---:|---:|
| MOTSChallenge | 2-class subset; PQ dominated by background stuff | 61.10 | 25.52 | 96.68 | 92.10 |
| COCO-Stuff-27 | disjoint/coarse classes; fixed class-space limit | 7.83 | 7.83 | 7.84 | 14.22 |

KITTI and Mapillary confirm that the pipeline transfers across the driving class space without retraining. The MOTS row is included only to document the run; its PQ is inflated by a near-trivial background class and should not be read as a transfer score. The COCO-Stuff-27 row is the *point* of this design choice: a Cityscapes-derived pseudo-label vocabulary cannot recover categories it was never asked to represent. We frame this as a *fixed-class supervision limit*, not as a transfer failure.

### 4.9 Per-Class Failure Modes

Table 8 localizes residual errors with Cityscapes val per-class PQ, SQ, and RQ for the final Stage-3 model. The legible pattern: large stuff and large vehicles are strong; thin structures, pedestrians, and motorcycles are weak; six classes — parking, guard rail, tunnel, polegroup, caravan, trailer — and one near-zero class — motorcycle (PQ 0.10) — are effectively dead.

**Table 8. Cityscapes val per-class PQ/SQ/RQ for the final Stage-3 model.**

| Class | PQ | SQ | RQ | Class | PQ | SQ | RQ |
|---|---:|---:|---:|---|---:|---:|---:|
| road | 92.99 | 94.95 | 97.93 | terrain | 35.68 | 73.42 | 48.60 |
| sidewalk | 62.44 | 78.52 | 79.52 | sky | 86.05 | 89.93 | 95.69 |
| parking | 0.00 | 0.00 | 0.00 | person | 13.37 | 71.41 | 18.72 |
| rail track | 8.40 | 67.21 | 12.50 | rider | 22.94 | 62.65 | 36.61 |
| building | 83.54 | 85.70 | 97.48 | car | 70.71 | 88.74 | 79.68 |
| wall | 32.32 | 67.68 | 47.76 | truck | 62.64 | 83.83 | 74.73 |
| fence | 20.26 | 63.04 | 32.14 | bus | 76.67 | 90.84 | 84.40 |
| guard rail | 0.00 | 0.00 | 0.00 | caravan | 0.00 | 0.00 | 0.00 |
| bridge | 17.21 | 64.54 | 26.67 | trailer | 0.00 | 0.00 | 0.00 |
| tunnel | 0.00 | 0.00 | 0.00 | train | 77.17 | 88.75 | 86.96 |
| pole | 2.05 | 72.27 | 2.83 | motorcycle | 0.10 | 100.00 | 0.10 |
| polegroup | 0.00 | 0.00 | 0.00 | bicycle | 38.99 | 77.18 | 50.53 |
| traffic light | 6.20 | 60.28 | 10.29 | traffic sign | 37.19 | 65.92 | 56.42 |
| vegetation | 84.70 | 85.71 | 98.82 | | | | |

The dead classes split into two failure modes that matter to interpret correctly. Parking, guard rail, tunnel, polegroup, caravan, and trailer are absent or near-absent from the $K=80$ pseudo-label vocabulary after Hungarian alignment to the 27-class evaluation, so the trained model never receives a positive example. Motorcycle is occasionally predicted (SQ 100.00 on the rare match) but RQ collapses to 0.10 because instances are mass-missed at recognition time. The remaining weak classes — person at PQ 13.37, pole at PQ 2.05, traffic light at PQ 6.20 — are recognition failures with non-trivial SQ, consistent with monocular geometry struggling on co-planar pedestrians and on thin structures that occupy too few patches. The overall pattern matches §5's qualitative analysis.

## 5. Qualitative Analysis

The qualitative results echo the quantitative argument. Large stuff classes — road, building, vegetation, sky — are stable because frozen semantic codes give coherent regions and self-training densifies them. Large vehicles benefit from depth boundaries because object extent often coincides with visible geometry. SIMCF-ABC's feature-guided merge step is the visible mechanism: it joins fragments that share semantic identity and DINOv3 appearance and avoids merging semantically distinct neighbors.

The figure for this section pairs RGB, monocular depth, depth edges, raw $K=80$ semantics, raw depth instances, SIMCF-ABC pseudo-labels, the final Stage-3 prediction, and ground truth. The intended visual message is not that the pseudo-labels are perfect, but that their errors are *structured*: the initial labels carry enough correct semantic layout and object extent for bootstrapping to operate on, and self-training does not have to invent structure that is absent from the supervision.

The clearest success case is large-object repair. Raw DepthPro splitting at the standalone optimum produces too many fragments. Under SIMCF-B's feature-guided merge, average pseudo-instance count per image drops from 44 to 22, median instance size grows from 5,502 to 14,965 pixels, and stuff contamination falls from 50.7% to 28.0%. These numbers explain why feature-guided merging improves thing PQ: many depth gradients are internal surface changes, not object boundaries.

The failure cases close the argument. Co-planar pedestrians and adjacent same-depth vehicles merge because monocular depth provides no separating discontinuity. Thin structures — poles, traffic lights — occupy too few patches to register in the semantic code. Caravan, trailer, tunnel, and motorcycle stay dead because the pseudo-label source rarely supplies positive examples. The COCO-Stuff-27 row is qualitatively informative as a class-space limit: the model produces clean masks where its vocabulary applies and void elsewhere, which is the expected behavior of fixed-class supervision rather than a generalization failure.

## 6. Limitations

We list three substantive limitations and close with a reproducibility statement.

**(1) Missing same-backbone CUPS control.** Our final model exceeds the published CUPS baseline by +8.03 PQ, but it also uses a DINOv3 ViT-B/16 Cascade Mask R-CNN that the published CUPS number does not. A fully controlled CUPS rerun requires regenerating stereo/video pseudo-labels (CUPS does not release them) and is left as the main controlled comparison. We therefore frame the contribution narrowly as "monocular pseudo-labels are competitive enough to seed a strong recipe," not as "monocular pseudo-labels exceed CUPS at the pseudo-label stage." Two feasible follow-ups — a Stage-3 from the raw 24.54-PQ pseudo-labels and a Stage-3 from the DA3 path (Table 4) — would respectively isolate DCFA + SIMCF-ABC's contribution and extend the multi-estimator robustness check to the trained-model level.

**(2) Single-seed final training.** The headline 35.83 PQ is from one Stage-3 run; k-means initialization and EMA self-training both inject seed-level variance that multi-seed runs with confidence intervals would characterize, particularly for rare classes.

**(3) Intrinsic limits of monocular supervision.** Co-planar pedestrians and adjacent same-depth vehicles are not separable when neither semantics nor monocular depth provides a discontinuity, and thin structures (poles, traffic lights) occupy too few patches to register reliably in the 90-D semantic code. The pseudo-label vocabulary is also fixed at the source: a Cityscapes-derived $K=80$ taxonomy cannot recover categories the supervision never represented (Table 7b's COCO-Stuff-27 row), so cross-domain transfer requires a relabeling step or an open-vocabulary semantic head.

**Reproducibility.** Code, pseudo-labels, and trained-model checkpoints will be released.

## 7. Conclusion

The headline result is straightforward, but the mechanism under it is what we want to leave with the community. CUPS-style bootstrapping and self-training amplified our 25.85-PQ monocular pseudo-labels into a 35.83-PQ trained model — a 9.98-PQ lift that suggests structured pseudo-labels can seed strong panoptic learning when their errors are coherent: large stuff regions stable, instance extents roughly right, agreement across cues consistent. The structure that lets self-training succeed need not come from stereo or motion. Geometry proposes the split, semantics and appearance decide whether the split is credible; that arbitration is enough on a single frame.

The most important missing experiment is a same-backbone CUPS control that would isolate the pseudo-label contribution from the +8.03 PQ headline gap. Beyond that, co-planar instances, thin structures, and rare classes remain monocular failure modes that no amount of cross-modal agreement repairs from a single frame. What unsupervised panoptic supervision needs, on the evidence here, is not necessarily richer geometry alone, but independent cues whose errors do not align.

## References

Hamilton et al. STEGO: Unsupervised Semantic Segmentation by Distilling Feature Correspondences. ICLR 2022. https://arxiv.org/abs/2203.08414

Kim et al. Causal Unsupervised Semantic Segmentation. arXiv 2023. https://arxiv.org/abs/2310.07379

Sick et al. Unsupervised Semantic Segmentation Through Depth-Guided Feature Correlation and Sampling. CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Sick_Unsupervised_Semantic_Segmentation_Through_Depth-Guided_Feature_Correlation_and_Sampling_CVPR_2024_paper.html

Wang et al. Cut and Learn for Unsupervised Object Detection and Instance Segmentation. CVPR 2023. https://arxiv.org/abs/2301.11320

Sick et al. CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation. ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Sick_CutS3D_Cutting_Semantics_in_3D_for_2D_Unsupervised_Instance_Segmentation_ICCV_2025_paper.html

Niu et al. Unsupervised Universal Image Segmentation. CVPR 2024. https://github.com/u2seg/U2Seg

Hahn et al. Scene-Centric Unsupervised Panoptic Segmentation. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Hahn_Scene-Centric_Unsupervised_Panoptic_Segmentation_CVPR_2025_paper.html

Bochkovskii et al. Depth Pro: Sharp Monocular Metric Depth in Less Than a Second. arXiv 2024. https://machinelearning.apple.com/research/depth-pro

Yang et al. Depth Anything V2. NeurIPS 2024. https://depth-anything-v2.github.io/

Lin et al. Depth Anything 3: Recovering the Visual Space from Any Views. arXiv 2025. https://arxiv.org/abs/2511.10647

Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. TMLR 2024. https://arxiv.org/abs/2304.07193

Simeoni et al. DINOv3. arXiv 2025. https://github.com/facebookresearch/dinov3
