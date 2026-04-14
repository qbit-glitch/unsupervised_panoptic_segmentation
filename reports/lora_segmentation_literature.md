# LoRA/DoRA for Segmentation: Literature Survey

**Date**: 2026-04-13
**Context**: Evaluating parameter-efficient adaptation for DINOv3 ViT-B/16 backbone in CUPS panoptic segmentation pipeline with noisy unsupervised pseudo-labels.

## Key Publications

### 1. Conv-LoRA (ICLR 2024)
**"Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model"**
- Zhu et al., ICLR 2024
- arxiv: 2401.17868

**Core idea**: Inserts ultra-lightweight depthwise convolutions inside the LoRA bottleneck to inject spatial inductive bias into plain ViT encoders. Uses Mixture-of-Experts (MoE) with multiple parallel convolutional experts at different spatial scales.

**Architecture**: Standard LoRA path `h = W0x + B(Ax)` becomes `h = W0x + B(MoE_Conv(reshape_2d(Ax)))`. The intermediate r-dim features are reshaped to a 2D spatial grid, processed by depthwise conv experts at different scales, then flattened back.

**Results**: On SAM adaptation for semantic segmentation across medical, remote sensing, and natural image domains. Only 4.02M trainable params. Kvasir dataset: Sα=92.0, Eϕ=94.7.

**Relevance to us**: Directly applicable to DINOv3 ViT backbone. Addresses the key weakness of plain DoRA — lack of spatial awareness in linear LoRA for dense prediction tasks.

### 2. PDoRA (Nature Scientific Reports, Jul 2025)
**"A new low-rank adaptation method for brain structure and metastasis segmentation via decoupled principal weight direction and magnitude"**
- Zhu, Yang, Wang et al., Nature Sci Rep 15, 27388 (2025)
- doi: 10.1038/s41598-025-11632-4

**Core idea**: Extends DoRA by first decomposing pretrained weights via truncated SVD into principal (top-k singular values) and residual components. DoRA (magnitude-direction decomposition) is applied only to the principal weights; residual weights stay frozen and are fused back after training.

**Architecture**: W0 = W_principal + W_residual (via SVD). Then W' = m * (W_principal + BA) / ||W_principal + BA|| + W_residual. Only the most informative weight directions are adapted.

**Results**: Applied to SwinUNETR for hippocampus and brain metastasis segmentation. Outperforms standard DoRA and LoRA on all metrics.

**Relevance to us**: The SVD principal selection is especially relevant for noisy pseudo-labels — focuses adaptation capacity on important directions, avoids corrupting low-importance weights with noisy gradients. Candidate for ablation experiment.

### 3. MoE-LoRA for SAM (Dec 2024)
**"Customize Segment Anything Model for Multi-Modal Semantic Segmentation with Mixture of LoRA Experts"**
- arxiv: 2412.04220

**Core idea**: Mixture of LoRA Experts adapts SAM for multi-modal input (camera + event camera + LiDAR). Different LoRA experts specialize in different modalities.

**Tasks**: Semantic segmentation, panoptic segmentation, uncertainty-aware panoptic segmentation.

**Relevance to us**: **Closest to our use case** — LoRA specifically for panoptic segmentation. Demonstrates that LoRA adaptation can improve panoptic quality. However, uses clean GT labels, not pseudo-labels.

### 4. ECLIPSE (CVPR 2024)
**"Efficient Continual Learning in Panoptic Segmentation with Visual Prompt Tuning"**
- CVPR 2024, github: clovaai/ECLIPSE

**Core idea**: Freezes Mask2Former base model and fine-tunes only small prompt embeddings for continual panoptic segmentation. Related to LoRA in spirit (parameter-efficient adaptation of frozen backbone).

**Relevance to us**: Shows parameter-efficient adaptation works for panoptic segmentation at scale.

### 5. CLIP-DoRA (2025)
**"CLIP-DoRA: Weight-decomposed Low-rank Adaptation for Efficient Vision-Language Models"**
- ScienceDirect, 2025

**Core idea**: Applies DoRA to CLIP for vision-language fine-tuning. Shows competitive results on medical image segmentation with only 1.5% trainable parameters.

**Results**: +0.28% average over previous PEFT SOTA across 11 few-shot datasets and 4 domain generalization benchmarks.

**Relevance to us**: Validates DoRA specifically for vision segmentation tasks.

### 6. LoRA for Stable Domain Adaptation (Springer, 2023)
**"Low Rank Adaptation for Stable Domain Adaptation of Vision Transformers"**
- Optical Memory and Neural Networks, 2023
- doi: 10.3103/S1060992X2306005X

**Core idea**: Uses LoRA for unsupervised domain adaptation in semantic segmentation. Key finding: **LoRA stabilizes the self-training process**, achieving similar dynamics to EMA teacher.

**Relevance to us**: **Most relevant for our noise concern.** Shows LoRA helps with pseudo-label-based training. However, their pseudo-labels come from a teacher trained on clean source-domain GT — still less noisy than our fully unsupervised k=80 clusters.

### 7. Semantic Library Adaptation (CVPR 2025)
**"Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation"**
- Qorbani et al., CVPR 2025

**Core idea**: LoRA retrieval and fusion for open-vocabulary semantic segmentation. Multiple LoRA adapters trained on different domains, dynamically selected/fused at inference.

### 8. PEFT Benchmark for SAM (Feb 2025)
**"Parameter Efficient Fine-Tuning of Segment Anything Model"**
- arxiv: 2502.00418

**Key finding**: Systematic benchmark of PEFT methods for SAM. **LoRA is the best overall PEFT method.** However, PEFT generally doesn't beat full fine-tuning — only marginal efficiency gains for smaller models.

### 9. TextSAM-LoRA (ICDAR 2025)
**"TextSAM-LoRA: Efficient Fine-Tuning of SAM for Text Detection with Low-Rank Adaptation"**
- Springer, ICDAR 2025

**Results**: CTW1500 F1=90.4% (SOTA). Demonstrates LoRA-adapted SAM outperforms fully fine-tuned approaches on specialized segmentation.

## Key Gap

**No publication uses LoRA/DoRA with genuinely unsupervised pseudo-labels.** All validated settings use:
- Clean GT labels (Conv-LoRA, PDoRA, TextSAM-LoRA)
- GT labels with domain shift (LoRA UDA)
- Few-shot human annotations (CLIP-DoRA)
- Teacher models trained on clean GT (MoE-LoRA)

Our scenario — adapting a frozen ViT backbone with k=80 unsupervised cluster pseudo-labels (~30% noise) — is **unprecedented**. This is both a risk and a potential contribution for the BMVC paper.

## Proposed Novel Combination: Conv-DoRA

Combine the best ideas for our use case:
1. **DoRA** (ICML 2024): Protect pretrained feature magnitudes from noisy gradient corruption
2. **Conv-LoRA** (ICLR 2024): Add spatial inductive bias to ViT linear layers for dense prediction
3. **LoRA+** (ICML 2024): Differential LR for faster convergence in limited steps
4. **PDoRA** (Nature 2025): SVD principal weight selection to focus capacity (ablation candidate)

This combination — **Conv-DoRA for unsupervised panoptic segmentation** — has not been published.

## Confidence Assessment

**5/10** for metric improvement. Literature validates the components individually, but the noisy pseudo-label regime is untested. Key mitigants: magnitude regularization, delayed activation, conv spatial bias, low learning rates.
