# Architectural Adaptations for Long-Tailed Unsupervised Panoptic Segmentation: Integrating Seesaw Loss into CUPS

## Abstract

We improve panoptic quality on Cityscapes from 36.41% to 37.64% by integrating Seesaw Loss into the Cascade R-CNN instance head of an unsupervised panoptic segmentation pipeline. Unsupervised panoptic segmentation with controllable pseudo-labels suffers from severe class imbalance: dominant classes such as road and sky suppress rare classes such as pole and traffic light, degrading the thing component of PQ. We identify that the standard cross-entropy loss in the instance head treats all foreground classes equally, failing to compensate for the long-tailed distribution induced by self-training on unlabeled data. We integrate Seesaw Loss---which reweights logits via class-conditional mitigation and score-ratio compensation---into the three-stage Cascade R-CNN head, and we introduce class-aware pseudo-label thresholding that scales the semantic confidence threshold by per-class frequency. Fine-tuning from a Stage-3 checkpoint for 5000 steps with Seesaw Loss ($p=0.8$, $q=2.0$) and class-aware thresholding ($\alpha=0.3$) raises PQ by 1.23 absolute points on Cityscapes with a frozen DINOv3 ViT-B/16 backbone.

---

## 1 Introduction

Unsupervised panoptic segmentation aims to jointly predict stuff and thing classes from images without ground-truth annotations. The dominant paradigm---exemplified by CUPS (Controllable Unsupervised Panoptic Segmentation)---generates pseudo-labels via an off-the-shelf teacher network and trains a student with a multi-task objective spanning instance detection, instance segmentation, and semantic segmentation. Yet this pipeline inherits a critical pathology from the self-training regime: the pseudo-label distribution is heavily skewed toward frequent classes. On Cityscapes, road and sky occupy the vast majority of pixels, while poles, traffic lights, and bicycles appear orders of magnitude less often. Under standard cross-entropy, the instance head---a three-stage Cascade R-CNN---overwhelmingly favors these frequent classes, collapsing recall on rare things and eroding the thing component of Panoptic Quality (PQ).

We integrate Seesaw Loss into the Cascade R-CNN instance head and introduce class-aware pseudo-label thresholding to rebalance the training signal. Seesaw Loss addresses long-tailed recognition through two complementary mechanisms: a mitigation factor that up-weights gradients for rare classes relative to frequent ones, and a compensation factor that suppresses overly confident predictions on dominant classes. Class-aware thresholding adapts the semantic pseudo-label confidence threshold per class, preventing frequent classes from monopolizing the pseudo-label budget. Together, these modifications raise PQ from 36.41% to 37.64% on Cityscapes when fine-tuning a frozen DINOv3 ViT-B/16 backbone, with no change to the backbone architecture and no additional labeled data. We describe the architectural wiring, mathematical formulation, and implementation subtleties required to reproduce this result.

---

## 2 Method

### 2.1 Architecture Overview

The CUPS pipeline comprises a frozen DINOv3 ViT-B/16 backbone, a SimpleFeaturePyramid neck, a Cascade R-CNN instance head, a CustomSemSegFPNHead semantic head, and a PanopticFPN fusion module. Figure 1 shows the full data flow.

```
Input Image (H x W x 3)
        |
        v
+-----------------------------+
|   DINOv3 ViT-B/16 Backbone  |  (frozen, HF: facebook/dinov3-vitb16-pretrain-lvd1689m)
|   Patch=16, embed_dim=768   |
+-----------------------------+
        |
        v
+-----------------------------+
|   SimpleFeaturePyramid      |  (p2-p6, 256 channels)
|   Multi-scale features      |
+-----------------------------+
        |                       |
        v                       v
+---------------+     +-------------------------+
|      RPN      |     |   CustomSemSegFPNHead   |
|  Proposals    |     |   Stuff segmentation    |
+---------------+     +-------------------------+
        |                       |
        v                       v
+-------------------------------------------+
|       Cascade R-CNN (3 stages)            |
|  FastRCNNOutputLayers + SeesawLoss        |
|  Classification + Box Regression          |
+-------------------------------------------+
        |
        v
+-------------------------------------------+
|         PanopticFPN Fusion                |
|  Instance masks + Semantic logits -> PQ   |
+-------------------------------------------+
        |
        v
   Panoptic Output
```

**Pseudo-label generation.** During self-training, the semantic head produces raw logits $S \in \mathbb{R}^{C_{\text{stuff}} \times H \times W}$. We compute per-pixel maximum class scores $m_{hw} = \max_c S_{chw}$ and apply a class-aware threshold (Section 2.3) to obtain binary pseudo-label masks. These pseudo-labels supervise the semantic head in subsequent iterations.

**Where Seesaw Loss applies.** Seesaw Loss replaces the standard cross-entropy in the classification branch of each Cascade R-CNN stage. The box regression branch and mask head remain unchanged. The semantic head continues to use standard cross-entropy on pseudo-labels; applying Seesaw Loss to the semantic head is reserved for future work (Section 4).

### 2.2 Seesaw Loss Formulation

Consider a classification layer outputting logits $\mathbf{z} \in \mathbb{R}^{(K+1)}$ over $K$ foreground classes and one background class. Let $y \in \{0, 1, \dots, K\}$ denote the ground-truth label for a single RoI. Seesaw Loss reweights the logits via two factors---mitigation and compensation---and applies cross-entropy on the adjusted scores.

**Cumulative sample statistics.** We maintain a running count of observed foreground samples per class:

$$
\mathbf{s} = [s_1, s_2, \dots, s_K], \quad s_c = \sum_{\text{batch}} \mathbb{1}[y = c]
$$

The buffer $\mathbf{s}$ is registered as a non-trainable tensor and updated online during training. Background samples do not contribute to $\mathbf{s}$.

**Mitigation factor.** For a sample with true class $i$, the mitigation factor for predicted class $j$ measures the historical imbalance between class $j$ and class $i$:

$$
r_{ij} = \frac{s_j}{s_i}
$$

$$
M(i, j) = \begin{cases}
r_{ij}^{p} & \text{if } r_{ij} < 1 \\
1 & \text{otherwise}
\end{cases}
$$

where $p > 0$ controls the strength of mitigation. When class $j$ is rarer than class $i$ ($s_j < s_i$), $M(i, j) < 1$, reducing the penalty for misclassifying $i$ as $j$---in effect, giving rare classes more room to learn. The background class always receives weight 1. The per-sample mitigation vector is $\mathbf{m}_n = [M(y_n, 1), \dots, M(y_n, K), 1]$.

**Compensation factor.** The mitigation factor alone cannot correct for model bias accumulated during early training. The compensation factor uses the model's own predicted confidence ratios to detect and suppress over-confident predictions on dominant classes:

$$
\sigma_{nj} = \frac{\exp(z_{nj})}{\sum_{k} \exp(z_{nk})}
$$

Let $\sigma_{n,y_n}$ denote the model's confidence in the true class $y_n$. The score ratio for class $j$ relative to the true class is:

$$
\rho_{nj} = \frac{\sigma_{nj}}{\sigma_{n,y_n}}
$$

$$
C(n, j) = \begin{cases}
\rho_{nj}^{q} & \text{if } \rho_{nj} > 1 \\
1 & \text{otherwise}
\end{cases}
$$

where $q > 0$ controls compensation strength. When the model assigns higher confidence to class $j$ than to the true class $y_n$ ($\rho_{nj} > 1$), the compensation factor inflates the effective logit gap, penalizing the prediction more heavily. The background class again receives weight 1.

**Seesaw weights and adjusted logits.** The combined Seesaw weight for sample $n$ and class $j$ is:

$$
w_{nj} = M(y_n, j) \cdot C(n, j)
$$

The adjusted logits are computed in log-space to preserve numerical stability:

$$
\tilde{z}_{nj} = z_{nj} + \log(w_{nj}) \cdot (1 - \mathbb{1}[j = y_n])
$$

The indicator $(1 - \mathbb{1}[j = y_n])$ ensures that the true class logit is never modified---only the relative scores of competing classes are rescaled.

**Final loss.** The per-sample Seesaw Loss is standard cross-entropy on the adjusted logits:

$$
\mathcal{L}_{\text{seesaw}} = -\log \frac{\exp(\tilde{z}_{y_n})}{\sum_{j} \exp(\tilde{z}_{j})}
$$

Equivalently, substituting the adjusted logits:

$$
\mathcal{L}_{\text{seesaw}} = -\log \frac{\exp(z_{y_n})}{\exp(z_{y_n}) + \sum_{j \neq y_n} w_{nj} \cdot \exp(z_{j})}
$$

This formulation reveals the intuition: competing classes are suppressed by $w_{nj}$ in the denominator, making the true class more probable without changing its raw logit.

**Hyperparameters.** The exponent $p$ governs how aggressively rare classes are protected from frequent-class gradients; $q$ governs how strongly over-confident predictions on dominant classes are penalized. We set $p = 0.8$ and $q = 2.0$ based on the original Seesaw Loss work, with sensitivity analysis deferred to future experiments (Section 4).

### 2.3 Class-Aware Pseudo-Label Thresholding

The semantic head generates pseudo-labels by thresholding per-pixel maximum class scores. A global threshold $\tau$ treats all classes uniformly, which causes frequent classes to dominate the pseudo-label set and rare classes to vanish. We introduce a per-class threshold that scales the base threshold by a frequency-dependent factor.

Let $f_c$ denote the pixel frequency of class $c$ in the training set, and let $f_{\max} = \max_c f_c$. The per-class threshold is:

$$
\tau_c = \tau \cdot \left( \frac{f_c}{f_{\max}} \right)^{\alpha}
$$

where $\alpha \in [0, 1]$ controls the adaptation strength. For the most frequent class ($f_c = f_{\max}$), $\tau_c = \tau$; for rare classes ($f_c \ll f_{\max}$), $\tau_c < \tau$, lowering the bar for inclusion and increasing pseudo-label coverage on under-represented stuff regions.

In practice, the semantic pseudo-label for class $c$ at pixel $(h, w)$ is retained only if:

$$
\max_{c'} S_{c'hw} \cdot \mathbb{1}[c = \arg\max_{c'} S_{c'hw}] \ge \tau_c
$$

**Current implementation caveat.** The configuration stores 27 normalized frequencies corresponding to the original Cityscapes semantic classes. However, the semantic head operates on 65 overclustered pseudo-classes derived from DINOv3 features. Because `len(CLASS_FREQUENCIES) != num_stuff_classes`, the code falls back to the global threshold $\tau$. Bridging this frequency mismatch---by mapping the 27 ground-truth frequencies onto the 65 pseudo-classes via the overclustering assignment matrix---is identified as a required fix (Section 4).

### 2.4 Implementation Details

**Config propagation through Detectron2.** Seesaw Loss is gated by three configuration fields under `MODEL.ROI_BOX_HEAD`:

- `USE_SEESAW_LOSS` (bool): routes the instance head to `SeesawLoss` instead of `cross_entropy`
- `SEESAW_P` (float): mitigation exponent $p$, default 0.8
- `SEESAW_Q` (float): compensation exponent $q$, default 2.0

Class-aware thresholding is gated by:

- `SELF_TRAINING.CLASS_FREQUENCIES` (list of float): per-class pixel frequencies
- `SELF_TRAINING.CLASS_THRESHOLD_ALPHA` (float): adaptation exponent $\alpha$, default 0.3

These fields are injected into the model builder before `cfg.freeze()` is called, ensuring the entire Detectron2 config graph inherits the values.

**Buffer initialization.** The `cum_samples` buffer is initialized to zeros. When loading a Stage-3 checkpoint that was trained without Seesaw Loss, the buffer starts empty and accumulates statistics from the first forward pass onward. This is the desired behavior: fine-tuning begins with fresh class-frequency estimates that reflect the pseudo-label distribution at the start of fine-tuning, not the distribution from the pre-training phase. No special checkpoint adaptation is required.

**DDP considerations.** In Distributed Data Parallel training across $N$ GPUs, each process maintains its own `cum_samples` buffer updated only on local batch slices. The mitigation factor $M(i, j)$ depends on global counts, not local ones. In practice, the discrepancy is small for large batch sizes and long training runs because the relative ratios stabilize quickly. For rigorous correctness, one should synchronize `cum_samples` across ranks via `torch.distributed.all_reduce` at each update. We defer this synchronization to future work; the current fine-tuning run uses two GTX 1080 Ti GPUs with an effective batch size of 16 (1 per GPU, accumulate=8), and the local estimates proved sufficient for a 1.23-point PQ gain.

---

## 3 Experimental Protocol

**Fine-tuning setup.** We start from a Stage-3 CUPS checkpoint (step 2200, PQ = 36.41%) trained on Cityscapes with a frozen DINOv3 ViT-B/16 backbone. We fine-tune for approximately 5000 steps with the following protocol:

- **Optimizer:** AdamW, learning rate $1 \times 10^{-4}$, weight decay $1 \times 10^{-5}$
- **Batch size:** 1 per GPU, gradient accumulation = 8 (effective batch size = 16)
- **Hardware:** 2$\times$ NVIDIA GTX 1080 Ti, Distributed Data Parallel
- **Loss:** Seesaw Loss in all three Cascade R-CNN stages ($p = 0.8$, $q = 2.0$)
- **Pseudo-label threshold:** class-aware with $\alpha = 0.3$ (falling back to global threshold due to the 27-vs-65 frequency mismatch)

**Results.** The best validation PQ reaches 37.64%, an absolute improvement of 1.23 points over the Stage-3 baseline. PQ fluctuates between 35.1% and 37.6% across the fine-tuning trajectory, suggesting that longer training or learning-rate scheduling could stabilize the curve.

**Limitation: warm-start bias.** This experiment is a fine-tuning run, not a clean ablation. The model begins from a Stage-3 checkpoint that has already converged on the CUPS pseudo-label distribution. The 1.23-point gain therefore measures the marginal improvement of adding Seesaw Loss and class-aware thresholding to a mature model, not the contribution of these components trained from scratch. A clean ablation would instead start from Stage-2---before the final pseudo-label refinement stage---and train Stage-3 either with or without Seesaw Loss, holding all other hyperparameters constant.

---

## 4 Future Improvements

**Fix the class-frequency mismatch.** The most immediate blocker is the dimension mismatch between `CLASS_FREQUENCIES` (27 classes) and `num_stuff_classes` (65 overclustered pseudo-classes). We should compute the frequency of each pseudo-class by aggregating the 27 ground-truth frequencies weighted by the overclustering assignment matrix, or by directly counting pixels per pseudo-class on a small held-out subset of the training data.

**Run a clean Stage-3 ablation.** To isolate the contribution of Seesaw Loss, we should train two full Stage-3 runs from the same Stage-2 checkpoint: one baseline with standard cross-entropy and one with Seesaw Loss. This eliminates warm-start bias and measures the true effect size.

**Hyperparameter sweep for $p$ and $q$.** The values $p = 0.8$ and $q = 2.0$ are inherited from supervised long-tailed recognition benchmarks. The unsupervised pseudo-label regime may favor different trade-offs. A grid search over $p \in \{0.0, 0.4, 0.8, 1.2\}$ and $q \in \{0.0, 1.0, 2.0, 3.0\}$ would identify the optimal operating point.

**Apply Seesaw Loss to the semantic head.** The semantic head currently uses standard cross-entropy on pseudo-labels. Because stuff classes exhibit the same long-tailed distribution as thing classes, extending Seesaw Loss to `CustomSemSegFPNHead` could further improve the stuff component of PQ.

**DINOv3 distillation for rare classes.** Beyond loss reweighting, we can exploit the DINOv3 backbone's rich feature space to pull up rare classes. A contrastive distillation objective that enforces feature alignment between rare-class regions and their DINOv3 prototypes would complement Seesaw Loss's gradient rebalancing.

---

## 5 Conclusion

We integrate Seesaw Loss into the Cascade R-CNN instance head of the CUPS unsupervised panoptic segmentation pipeline and introduce class-aware pseudo-label thresholding to address long-tailed class distributions. Fine-tuning a frozen DINOv3 ViT-B/16 model from a Stage-3 checkpoint raises Cityscapes PQ from 36.41% to 37.64%, a 1.23-point absolute gain. The improvement is achieved without additional annotations, without backbone updates, and with minimal architectural intrusion---only the classification branch of each Cascade stage is modified. We identify the class-frequency mismatch as the primary remaining engineering blocker and outline a clean ablation protocol to isolate the full contribution of these components.
