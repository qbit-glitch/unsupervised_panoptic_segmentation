# Training Pipeline Gap Analysis: Mobile Panoptic Distillation vs. CUPS

## Abstract

We present a systematic gap analysis between our lightweight mobile panoptic distillation pipeline and the CUPS [Hahn et al., CVPR 2025] training methodology. Our current approach performs naive cross-entropy distillation from pre-computed pseudo-labels into a 4.9M-parameter RepViT-M0.9 backbone with a SimpleFPN decoder, achieving PQ=13.87 after 2 epochs of training on Cityscapes. We identify five critical training components present in CUPS but absent from our pipeline: (1) DropLoss for selective thing-class supervision, (2) self-enhanced copy-paste augmentation, (3) comprehensive photometric augmentations, (4) multi-scale resolution jittering, and (5) EMA-based self-training. CUPS's own ablation (Table 7c) demonstrates that these components collectively contribute +3.7 PQ over vanilla training. We analyze the architectural and methodological constraints that prevent direct adoption of each component and propose adaptations suitable for our semantic-only lightweight model.

## 1. Introduction

Knowledge distillation from unsupervised pseudo-labels into lightweight models is an emerging paradigm for deploying panoptic segmentation on mobile devices. The key challenge lies not in the model architecture---modern lightweight backbones such as RepViT [Wang et al., CVPR 2024] achieve strong feature quality at <5M parameters---but in the *training recipe* that transfers pseudo-label knowledge effectively.

CUPS [Hahn et al., CVPR 2025] establishes the state of the art for unsupervised panoptic segmentation at PQ=27.8 on Cityscapes. Critically, CUPS achieves this through a three-stage training pipeline where each stage introduces specific augmentation and loss innovations. Their ablation study (Table 7c) reveals that the training recipe alone accounts for +3.7 PQ over vanilla pseudo-label training:

| Training Configuration | PQ | Delta |
|----------------------|------|-------|
| Vanilla training | 24.1 | --- |
| + DropLoss | 25.3 | +1.2 |
| + Copy-paste augmentation | 26.3 | +1.0 |
| + Self-enhanced copy-paste | 26.6 | +0.3 |
| + Self-training (CUPS) | 27.8 | +1.2 |

Our current mobile distillation pipeline omits all five of these innovations, relying solely on basic cross-entropy with minimal augmentation. This report quantifies the gap, analyzes the root causes, and outlines an adaptation strategy.

## 2. Current Implementation

### 2.1 Architecture

Our mobile panoptic model comprises:

- **Backbone**: RepViT-M0.9 [Wang et al., CVPR 2024] (4.7M params), pretrained on ImageNet-1K, with `features_only=True` producing 4-scale features at channels [48, 96, 192, 384].
- **Decoder**: SimpleFPN with depthwise separable convolutions (0.2M params), producing single-scale features at the finest backbone resolution.
- **Head**: Semantic-only---a single 19-class classification head. No instance detection head, no mask prediction head.
- **Total**: 4.9M parameters (well within 8M mobile budget).

### 2.2 Training Configuration

| Component | Our Implementation | CUPS |
|-----------|-------------------|------|
| **Loss** | CE (ignore_index=255, label_smoothing=0.1) | CE (semantic) + DropLoss (instance) + detection losses |
| **Augmentation** | Random flip, random crop 384x768, brightness jitter (0.8-1.2x) | RandomResizedCrop (0.7-1.0x), RandomHorizontalFlip, ColorJitter (B=0.4, C=0.4, S=0.4, H=0.1), GaussianBlur (7x7, sigma 0.1-2.0), RandomGrayscale (p=0.2), CopyPaste (up to 7 objects), ResolutionJitter (0.6-1.1x) |
| **Self-training** | None | EMA momentum network, multi-scale + flip ensembling, confidence thresholding (per-class semantic + IoU instance) |
| **Instance handling** | None (semantic-only) | Cascade Mask R-CNN with 3-stage refinement, box + mask heads |
| **Optimizer** | AdamW (decoder LR=1e-3, backbone LR=1e-4), cosine decay, 50 epochs | AdamW, 4000 steps (stage 2) + 1500 steps (stage 3) |
| **Backbone freezing** | First 5 epochs frozen, then full unfreeze | Frozen backbone + FPN, only heads trained |
| **Batch size** | 4 (single GTX 1080 Ti) | 4 per GPU x 4 A100 GPUs = 16 effective |

### 2.3 Early Results

Training with RepViT-M0.9 on pseudo-label mapped k=80 predictions (input PQ=26.74):

| Epoch | Loss | mIoU | Acc | PQ | PQ_stuff | PQ_things |
|-------|------|------|-----|-----|----------|-----------|
| 1 | 1.285 | --- | --- | --- | --- | --- |
| 2 | 1.171 | 31.38% | 82.88% | 13.87 | 23.12 | 1.16 |
| 3 | 1.135 | --- | --- | --- | --- | --- |
| 4 | 1.109 | 30.35% | 82.48% | 13.57 | 22.32 | 1.55 |

The PQ of 13.87 at epoch 2 represents a 12.87-point gap below the input pseudo-labels (PQ=26.74), and a 13.93-point gap below CUPS (PQ=27.8). PQ_things=1.16 is particularly concerning---the model produces almost no valid thing-class instances through connected components alone.

## 3. Gap Analysis

We decompose the training gap into five categories, ordered by estimated PQ impact.

### 3.1 Missing: DropLoss (+1.2 PQ in CUPS ablation)

**What CUPS does.** DropLoss [De Brabandere et al., 2021] selectively supervises thing-class predictions. For each predicted "thing" region, DropLoss computes IoU with all pseudo-label instance masks. Only predictions overlapping a pseudo mask above threshold tau_IoU receive detection loss. Predictions without corresponding pseudo masks are *not penalized*---this is critical because pseudo-labels are sparse (they miss many static objects). The selective supervision allows the network to discover objects beyond the pseudo-label set.

Formally (CUPS Eq. 4):
```
L_drop(R_j, R_hat_i) = 1(IoU_j^max > tau_IoU) * L_Th(R_j, R_hat_i)
```

**Why we lack it.** Our model has no instance head. We predict only a 19-class semantic map; thing instances are derived post-hoc via `scipy.ndimage.label()` connected components. DropLoss requires a region-proposal network and mask prediction head, which our SimpleFPN architecture does not support.

**Impact.** Without DropLoss, our semantic CE loss penalizes *every* pixel equally. Thing-class pixels that the pseudo-labels mark as "ignore" (no instance mask) receive no gradient signal. Worse, the CE loss may actively suppress correct thing predictions if the pseudo-label is incorrect or missing at those locations. This explains PQ_things=1.16 at epoch 2.

**Adaptation path.** Implement a lightweight instance head (embedding-based, not Cascade Mask R-CNN) and adapt DropLoss for semantic predictions: only backpropagate thing-class CE gradients when the predicted thing region overlaps a pseudo-label instance.

### 3.2 Missing: Copy-Paste Augmentation (+1.0 PQ standard, +0.3 PQ self-enhanced)

**What CUPS does.** CUPS implements batch-wise copy-paste augmentation [Ghiasi et al., ICCV 2017] that:
1. Extracts instance crops (image + mask) from source images in the batch.
2. Randomly scales (0.25-1.5x), flips, and pastes up to 7 objects per image.
3. Updates semantic maps, instance masks, and bounding boxes accordingly.
4. In the self-enhanced variant, the source objects come from the model's *own predictions* rather than pseudo-labels, enabling discovery of objects not in the training set.

**Why we lack it.** Copy-paste requires instance-level masks to define which pixels to cut and paste. Our pseudo-labels are semantic-only (19-class maps without instance IDs). While we have depth-guided instance pseudo-labels (`pseudo_instance_spidepth/`), they are not integrated into the mobile training pipeline.

**Impact.** Copy-paste is particularly important for *thing* classes because:
- Thing objects are small and sparse in Cityscapes (cars, pedestrians, cyclists).
- Without copy-paste, the model sees each thing instance only in its original context, limiting generalization.
- Self-enhanced copy-paste further bootstraps coverage of static objects (parked cars, stopped pedestrians) that motion-based pseudo-labels miss.

**Adaptation path.** Load instance pseudo-labels alongside semantic labels. Implement simplified copy-paste that crops thing-class connected components and pastes them into training images. This does not require Detectron2's `Instances` format---numpy-based crop-and-paste suffices for our pipeline.

### 3.3 Missing: Photometric Augmentations (indirect PQ contribution)

**What CUPS does.** CUPS applies a comprehensive photometric pipeline via Kornia:
- `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5)`
- `RandomGaussianBlur(kernel_size=7x7, sigma=(0.1, 2.0), p=1.0)`
- `RandomGrayscale(p=0.2)`

Additionally, stage 3 self-training applies photometric perturbations to create augmented views for self-label generation.

**What we do.** Brightness jitter only (0.8-1.2x), applied with p=0.5. No contrast, saturation, hue, blur, or grayscale augmentation.

**Impact.** Weak photometric augmentation reduces robustness to lighting variations, weather conditions, and sensor noise---particularly important for mobile deployment where camera quality varies. The absence of Gaussian blur means the model never sees defocused regions, limiting generalization on real-world mobile captures.

**Adaptation path.** Replace our manual brightness jitter with `torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)` plus `GaussianBlur` and `RandomGrayscale`. This is a trivial change with no computational overhead.

### 3.4 Missing: Multi-Scale Resolution Jittering (indirect PQ contribution)

**What CUPS does.** `ResolutionJitter` randomly rescales inputs to one of 11 scales (0.6x to 1.1x of base resolution) at each training step. Combined with `RandomResizedCrop(scale=(0.7, 1.0))`, this exposes the model to objects at varying apparent sizes.

During self-training (stage 3), multi-scale inference with horizontal flipping generates self-labels by ensembling predictions at multiple resolutions.

**What we do.** Fixed crop of 384x768 from 1024x2048 images. No resolution variation during training.

**Impact.** Fixed-scale training causes the model to overfit to objects at a single scale. Thing classes are particularly affected because their apparent size varies dramatically (nearby vs. distant cars span 10x in pixel area). Multi-scale training acts as implicit size augmentation.

**Adaptation path.** Add `RandomResizedCrop` with scale range (0.5, 1.5) instead of fixed 384x768 crop. This matches CUPS's approach and requires no architectural changes.

### 3.5 Missing: EMA Self-Training (+1.2 PQ in CUPS ablation)

**What CUPS does.** Stage 3 maintains an exponential moving average (EMA) "momentum network" as a teacher:
1. Apply multi-scale + flip augmentations to input image.
2. Momentum network generates predictions at each augmented view.
3. Predictions are fused, averaged, and confidence-thresholded to create self-labels.
4. Student network trains on photometrically perturbed image with self-labels.
5. EMA updates momentum network weights from student.

Self-labels are filtered by:
- **Semantic**: per-class confidence threshold zeta_k, derived from dataset-wide statistics.
- **Instance**: IoU threshold gamma on instance predictions; only high-confidence instances become self-labels.

**What we do.** No self-training. The model trains exclusively on pre-computed pseudo-labels that remain fixed throughout training.

**Impact.** Self-training is responsible for +1.2 PQ in CUPS (24.1->27.8 includes self-training). It enables:
- Correction of pseudo-label errors (the model can learn to disagree with noisy labels).
- Discovery of objects missed by pseudo-label generation.
- Progressive refinement of decision boundaries through iterative self-distillation.

Without self-training, the model's performance is upper-bounded by pseudo-label quality. Given that our pseudo-labels have PQ=26.74, the model cannot exceed this ceiling---and in practice falls well below it due to distillation loss.

**Adaptation path.** Implement EMA teacher-student training after initial convergence (epoch 30+). Use multi-scale inference for self-label generation. Apply per-class confidence thresholding. This requires approximately 2x forward passes per step (teacher + student) but no architectural changes.

## 4. Structural Limitations

Beyond missing training components, our pipeline has two fundamental structural differences from CUPS that limit direct comparison:

### 4.1 Semantic-Only vs. Full Panoptic Model

CUPS trains a Cascade Mask R-CNN---a full panoptic architecture with:
- Region Proposal Network (RPN) for object detection.
- Three-stage cascade box refinement.
- Mask prediction head for instance segmentation.
- Semantic segmentation head (FPN-based).

Our model has only a semantic head. Thing instances are derived *post-hoc* via connected components. This means:
- No learned instance separation---overlapping or adjacent objects of the same class merge into single segments.
- No explicit object detection---small or thin objects may be missed entirely.
- PQ_things is structurally capped by the quality of connected component segmentation, which cannot separate touching instances.

### 4.2 Backbone Capacity

| Property | CUPS | Ours |
|----------|------|------|
| Backbone | DINO ResNet-50 (23M params) | RepViT-M0.9 (4.7M params) |
| Pretraining | Self-supervised DINO | Supervised ImageNet |
| Feature dim | 256 (FPN) | 128 (FPN) |
| Total params | ~41M (R50 + Cascade Mask R-CNN) | 4.9M (RepViT + SimpleFPN) |

The 8.4x parameter gap means our model has significantly less capacity to represent complex panoptic predictions. However, this is by design---the goal is mobile deployment at <5ms inference, not matching CUPS's desktop-class performance.

## 5. Prioritized Adaptation Roadmap

Based on the analysis above, we prioritize adaptations by expected PQ gain per implementation effort:

| Priority | Component | Est. PQ Gain | Effort | Dependency |
|----------|-----------|-------------|--------|------------|
| **P0** | Photometric augmentations | +0.5-1.0 | Trivial (5 lines) | None |
| **P1** | Multi-scale crop + resolution jitter | +0.5-1.0 | Low (20 lines) | None |
| **P2** | Instance pseudo-label loading + copy-paste | +1.0-1.5 | Medium (100 lines) | Instance labels |
| **P3** | EMA self-training | +1.0-2.0 | Medium (150 lines) | P0, P1 |
| **P4** | Embedding head + adapted DropLoss | +1.0-1.5 | High (300 lines) | Architectural change |

**Recommended execution order**: P0 -> P1 -> P3 -> P2 -> P4.

P0 and P1 are trivial augmentation improvements with no architectural changes. P3 (EMA self-training) provides the highest PQ gain and is architecture-agnostic. P2 requires loading instance pseudo-labels but is straightforward. P4 requires adding an embedding head and is the most invasive change.

## 6. Expected Outcome

With all adaptations implemented, we estimate the mobile model can achieve:

| Configuration | Est. PQ | PQ_stuff | PQ_things |
|--------------|---------|----------|-----------|
| Current (vanilla CE) | 13-15 | 22-24 | 1-3 |
| + P0 + P1 (augmentations) | 16-18 | 25-27 | 3-5 |
| + P3 (EMA self-training) | 19-22 | 28-30 | 5-8 |
| + P2 (copy-paste) | 21-24 | 29-31 | 8-12 |
| + P4 (embedding + DropLoss) | 23-26 | 30-32 | 12-16 |

These estimates assume the RepViT-M0.9 backbone reaches convergence at 50 epochs. The final PQ target of 23-26 would represent a competitive mobile panoptic model at 4.9M parameters---approximately 8.4x smaller than CUPS while retaining 83-94% of its panoptic quality.

## 7. Conclusion

Our current mobile distillation pipeline suffers from a significant training methodology gap relative to CUPS. The five missing components---DropLoss, copy-paste augmentation, photometric augmentations, multi-scale training, and EMA self-training---collectively account for an estimated +3.7 PQ in CUPS's own ablation. Adapting these techniques to our semantic-only lightweight architecture requires careful modification but is feasible. The immediate priorities are P0 (photometric augmentations) and P1 (multi-scale training), which require minimal code changes and no architectural modifications.

## References

- [Hahn et al., 2025] Hahn, O., et al. "Scene-Centric Unsupervised Panoptic Segmentation." CVPR 2025.
- [Wang et al., 2024] Wang, A., et al. "RepViT: Revisiting Mobile CNN From ViT Perspective." CVPR 2024.
- [Ghiasi et al., 2017] Ghiasi, G., et al. "Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection." ICCV 2017.
- [De Brabandere et al., 2021] De Brabandere, B., et al. "DropLoss for Long-Tail Instance Segmentation." AAAI 2021.
- [Chen et al., 2020] Chen, X., et al. "Improved Baselines with Momentum Contrastive Learning." arXiv 2020.
