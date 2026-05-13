---
type: codebase-module
title: Evaluation Scripts
project: mbps-panoptic-segmentation
language: en
tags: [codebase, evaluation, metrics, scripts]
paths:
  - mbps_pytorch/eval_*.py
  - mbps_pytorch/evaluate_*.py
  - scripts/evaluate.py
related:
  - "[[PyTorch-Evaluation]]"
  - "[[JAX-mbps-Evaluation]]"
  - "[[Reports-Index]]"
---

# Evaluation Scripts

| Script | What it evaluates |
|--------|-------------------|
| `mbps_pytorch/eval_cause_k80.py` | CAUSE k=80 pseudo-labels. |
| `mbps_pytorch/eval_contrastive_learned.py` | Learned contrastive embeddings. |
| `mbps_pytorch/eval_learned_merge.py` | Learned merge predictor. |
| `mbps_pytorch/eval_picl.py` | PICL embeddings. |
| `mbps_pytorch/evaluate_alignment.py` | Alignment between depth- and feature-based instances. |
| `mbps_pytorch/evaluate_cascade_pseudolabels.py` | DINOv3 + MaskCut Cascade Mask R-CNN. |
| `mbps_pytorch/evaluate_coconut_pseudolabels.py` | COCONUT benchmark. |
| `mbps_pytorch/evaluate_cross_dataset.py` | Cross-dataset eval (Cityscapes → MOTS / KITTI / Mapillary / COCO-Stuff-27). |
| `mbps_pytorch/evaluate_k50_pseudolabels.py` | k=50 subset of CAUSE. |
| `mbps_pytorch/evaluate_mobile_on_coconut.py` | Mobile model on COCONUT. |
| `mbps_pytorch/evaluate_novel_ablation.py` | Novel-component ablations. |
| `mbps_pytorch/evaluate_panoptic_combined.py` | Panoptic from semantic + depth-guided instance. |
| `mbps_pytorch/evaluate_pseudolabels.py` | Comprehensive pseudo-label QA. |
| `mbps_pytorch/evaluate_semantic_pseudolabels.py` | Semantic accuracy. |
| `scripts/evaluate.py` | JAX TPU evaluation harness with optional CRF. |
| `scripts/evaluate_pseudolabel_quality.py` | Stand-alone pseudo-label QA. |

## Cross-dataset highlights (DINOv3 Stage-3)

| Dataset | PQ | mIoU | Notes |
|---------|----|------|-------|
| MOTS (2,862 imgs) | 63.38 | 91.43 | Best transfer (driving domain). |
| KITTI (200 imgs) | 29.32 | 44.45 | Smaller / sparser. |
| COCO-Stuff-27 (5,000 imgs) | 8.05 | 15.24 | Large domain gap. |
| BDD-10K | — | — | Script ready, data blocked. |

Reports: `reports/dinov3_stage3_cross_dataset_evaluation.md`, `reports/cross_dataset_evaluation_report.md`.

## Reminder

Use the CUPS-standard global Hungarian over 27 classes. Never per-image argmax.
