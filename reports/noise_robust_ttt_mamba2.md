# Noise-Robust Training for Test-Time Trainable Mamba2 Pseudo-Label Refinement

## 1. Problem Statement

Training a TTT-Mamba2 refiner on overclustered semantic pseudo-labels (19-class, mIoU=60.7%, PQ=25.6) with class-balanced cross-entropy and information maximization---the IMFocalLoss configuration---peaked at PQ=23.63 (epoch 4) and plateaued around PQ=23.0 over 20 epochs, consistently *below* the input pseudo-label baseline. The model changed ~9% of pixels but the net effect was harmful: it correctly identified some errors in the pseudo-labels but introduced more new errors than it fixed. Per-class analysis revealed that stuff classes (road, building, vegetation) were largely preserved while thing classes (person, car, rider) suffered from inconsistent corrections. The fundamental limitation is that standard cross-entropy treats all pseudo-labels as ground truth, providing the student network no mechanism to distinguish correct labels from the ~39% of pixels that are incorrectly labeled. A static CE student can approach but never exceed its noisy teacher's accuracy---a ceiling documented theoretically by Jiang et al. (2023) and empirically confirmed by our 20-epoch training trajectory showing monotonic convergence toward the teacher distribution rather than away from it.

## 2. Diagnosis: Three Sources of Failure

**2.1. Cross-entropy memorization of label noise.** Standard CE loss minimizes $-\sum_k q_k \log p_k$ where $q$ is the one-hot pseudo-label distribution. When $q$ is incorrect for a pixel, CE still pushes $p$ toward the wrong class with full gradient magnitude. Over training, the model memorizes these errors, particularly for underrepresented classes where the effective sample size is small and a few noisy examples dominate the gradient. This is the well-characterized "memorization" failure mode of deep networks trained on noisy labels (Arpit et al., ICML 2017; Zhang et al., ICLR 2017), exacerbated in our setting because the noise rate (~39%) substantially exceeds the clean-label regime where CE remains empirically robust.

**2.2. Spatially inconsistent pseudo-label noise.** Overclustered pseudo-labels exhibit two types of spatial noise: (i) boundary artifacts where downsampling from 1024x2048 to 32x64 creates class-confused pixels at object edges, and (ii) isolated mislabeled pixels where the overclustering codebook assigns incorrect categories. These spatially inconsistent pixels are particularly harmful because the model cannot learn a coherent spatial pattern from them---they inject contradictory gradient signals that destabilize nearby correctly-labeled regions.

**2.3. No self-supervised signal beyond the noisy teacher.** The IMFocalLoss loss function derives all its supervision from the pseudo-labels (via CE) and from the model's own predictions (via IM). There is no external self-supervised signal that could provide information about visual structure independent of the label quality. This means the model's learning is fundamentally bounded by the information content of the noisy labels plus the weak inductive bias from IM's class-diversity constraint. Jiang et al. (2023, "Student as Inherent Denoiser") showed that students can exceed noisy teachers *only when they have access to additional information beyond the labels*---such as data augmentation invariances or architectural inductive biases that the teacher cannot exploit.

## 3. Proposed Solution: Three Complementary Noise-Robust Mechanisms

We replace the `IMFocalLoss` with a `NoiseRobustLoss` that addresses each failure source with a targeted, literature-grounded intervention. The three mechanisms are complementary: SCE handles systematic label noise, confidence masking removes spatially inconsistent noise, and cross-augmentation consistency provides the additional self-supervised signal necessary for the student to potentially exceed the teacher.

### 3.1. Symmetric Cross-Entropy (SCE)

**Mechanism.** Following Wang et al. (ICCV 2019), we replace standard CE with the symmetric cross-entropy objective:

$$\mathcal{L}_{\text{SCE}} = \alpha \cdot \text{CE}(p, q) + \beta \cdot \text{RCE}(p, q)$$

where $\text{CE}(p, q) = -\sum_k q_k \log p_k$ is the standard forward cross-entropy and $\text{RCE}(p, q) = -\sum_k p_k \log \bar{q}_k$ is the reverse cross-entropy with $\bar{q}_k = \max(q_k, \epsilon)$ for numerical stability ($\epsilon = 10^{-4}$).

**Why it helps.** The forward CE provides standard learning signal but is not noise-tolerant---it can converge to the noisy label distribution. The reverse CE is provably noise-tolerant under symmetric and asymmetric noise models (Wang et al., 2019, Theorem 1): its loss landscape preserves the global minimum at the clean distribution even when computed against noisy labels. However, RCE alone has poor convergence properties. The combination achieves fast convergence (from CE) with noise tolerance (from RCE). For one-hot labels, $\text{RCE} = -\log(\epsilon) \cdot (1 - p_{\text{target}})$, which is bounded by $-\log(\epsilon) \approx 9.21$. This bounding prevents the model from investing unbounded gradient effort to memorize individual noisy labels---the RCE loss saturates, creating an implicit early-stopping effect per pixel.

**Class balancing.** Both forward and reverse CE terms are weighted by the inverse-sqrt-frequency class weights from the overclustered pseudo-labels, preserving the 19x gradient amplification for rare classes (rider: $\alpha=2.30$) over dominant classes (road: $\alpha=0.12$).

**Hyperparameters.** $\alpha = 1.0$, $\beta = 1.0$, following the recommendation of Wang et al. for asymmetric noise at ~40% noise rate.

### 3.2. Local-Consistency Confidence Masking

**Mechanism.** For each pixel in the 32x64 pseudo-label map, we count how many of its 8 spatial neighbors share the same class label. Pixels where fewer than $\tau = 6$ of 8 neighbors agree are marked as uncertain and excluded from the SCE loss (target set to 255, the ignore index).

**Why it helps.** Spatially inconsistent pixels---boundary artifacts from downsampling and isolated overclustering errors---are precisely those where the pseudo-label is most likely wrong. By removing them from supervision, we reduce the effective noise rate in the training signal. This is conceptually similar to the confidence-based sample selection in DivideMix (Li et al., ICLR 2020), but uses a purely spatial criterion that requires no model predictions and can be computed once at dataset loading time.

**Expected masking rate.** At 32x64 resolution with 19 classes, approximately 10-15% of pixels are expected to be masked, corresponding to class boundaries and isolated noise. This leaves the vast majority of confidently-labeled interior pixels as supervision, ensuring the model still receives strong learning signal while removing the most harmful training examples.

**Edge handling.** Border pixels use edge-replicated padding, ensuring they are not artificially penalized for having fewer neighbors.

### 3.3. Cross-Augmentation Consistency

**Mechanism.** For each training batch, we perform two forward passes: one with the original features $(f, d, \nabla d)$ and one with horizontally flipped features $(f^{\text{flip}}, d^{\text{flip}}, \nabla d^{\text{flip}})$, where the depth x-gradient is negated after flipping to preserve physical consistency. The flipped logits are then un-flipped to align with the original spatial layout. We enforce consistency between the two views via symmetric KL divergence on pixels where either view's maximum softmax probability exceeds 0.5:

$$\mathcal{L}_{\text{consist}} = \frac{1}{|\mathcal{M}|} \sum_{(i,j) \in \mathcal{M}} \frac{1}{2} \left[ \text{KL}(p_{ij} \| p_{ij}^{\text{flip}}) + \text{KL}(p_{ij}^{\text{flip}} \| p_{ij}) \right]$$

where $\mathcal{M} = \{(i,j) : \max(\max_c p_{c,ij}, \max_c p_{c,ij}^{\text{flip}}) > 0.5\}$.

**Why it helps.** This provides self-supervised signal that is *completely independent of pseudo-label quality*. The consistency objective encodes a visual invariance (horizontal flip equivariance) that the Mamba2 architecture does not trivially satisfy---the cross-scan state-space model processes patches in direction-dependent sequences, so its predictions are not automatically flip-invariant. By enforcing this invariance, we provide the "additional information beyond the labels" that Jiang et al. (2023) identified as necessary for the student to exceed the noisy teacher. The consistency loss also acts as a regularizer, preventing the model from learning direction-dependent artifacts in the Mamba2 scan patterns.

**Confidence gating.** The 0.5 threshold ensures consistency is only enforced on pixels where the model has formed a reasonable prediction. On highly uncertain pixels (near-uniform softmax), enforcing consistency would propagate noise between views.

## 4. Combined Loss Function

The total loss combines all three components:

$$\mathcal{L} = \mathcal{L}_{\text{SCE}} + 0.01 \cdot \mathcal{L}_{\text{IM}} + 1.0 \cdot \mathcal{L}_{\text{consist}}$$

**Design rationale for weights:**

| Component | Weight | Magnitude | Role |
|-----------|--------|-----------|------|
| SCE (fwd+rev CE) | $\alpha=1, \beta=1$ | ~0.2--1.0 | Primary supervision from pseudo-labels |
| Information Maximization | 0.01 | ~0.01--0.02 | Gentle class-diversity regularizer |
| Cross-aug consistency | 1.0 | ~0.01--0.5 | Self-supervised flip equivariance |

The hierarchy is deliberate: SCE provides the dominant learning signal from (noise-tolerant) pseudo-labels, the consistency loss adds self-supervised structure, and IM acts as a lightweight anti-collapse regularizer. This follows the proven principle from CSCMRefineNet training: keep distillation dominant ($\geq 0.5$ of total), self-supervised losses as gentle nudges only.

## 5. Implementation Details

**Dataset.** `TTTDataset.__getitem__()` now returns a `conf_mask` tensor (H, W) computed via 8-neighbor agreement on the pseudo-label map at 32x64 resolution. The `mask_threshold` parameter (default: 6) controls the minimum agreement level.

**Training loop.** Each batch requires two forward passes (original + flipped), doubling compute per step. On 2x GTX 1080 Ti with batch_size=4 and bridge_dim=192, estimated training time is ~10 minutes per epoch (up from ~5 minutes with single forward pass). Total training: 20 epochs, ~3.5 hours.

**TTT loss.** The test-time training self-supervised loss (`_ttt_loss` in `TTTMamba2Refiner`) retains $\mathcal{L}_{\text{IM}} + 0.3 \cdot \mathcal{L}_{\text{align}}$ since it operates at inference time without access to pseudo-labels or augmentation pairs.

**Evaluation.** Full panoptic evaluation (PQ, SQ, RQ, PQ_stuff, PQ_things, mIoU, changed_pct) every 2 epochs. Best model selected by PQ. Final TTT evaluation with 1, 3, 5 adaptation steps.

## 6. Expected Outcomes

| Metric | Baseline (k=300) | IMFocalLoss (best) | NoiseRobust (expected) |
|--------|------------------|--------------------|----------------------|
| PQ | 25.6 | 23.63 | 25.5--27.0 |
| mIoU | 60.7 | 59.30 | 60.0--62.0 |
| changed_pct | 0% | ~9% | 5--15% |

The SCE loss should prevent the model from memorizing pseudo-label errors, the confidence mask removes the most harmful training examples, and the consistency loss provides the additional self-supervised signal needed to potentially exceed the teacher. The combined effect should shift the change distribution: fewer correct-to-incorrect changes (currently the majority) and more incorrect-to-correct changes (the desired refinement).

## 7. References

- Wang, Y., Ma, X., Chen, Z., Luo, Y., Yi, J., & Bailey, J. (2019). Symmetric cross entropy for robust learning with noisy labels. *ICCV*.
- Liang, J., Hu, D., & Feng, J. (2020). Do we really need to access the source data? Source hypothesis transfer for unsupervised domain adaptation. *ICML* (SHOT).
- Jiang, Z., et al. (2023). Student as inherent denoiser. *arXiv preprint*.
- Li, J., Socher, R., & Hoi, S. C. H. (2020). DivideMix: Learning with noisy labels as semi-supervised learning. *ICLR*.
- Arpit, D., et al. (2017). A closer look at memorization in deep networks. *ICML*.
- Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-balanced loss based on effective number of samples. *CVPR*.
- Niu, S., et al. (2022). Efficient test-time model adaptation without forgetting. *ICML* (EATA).
- Hahn, M., et al. (2025). Completely unsupervised panoptic segmentation. *CVPR* (CUPS).
