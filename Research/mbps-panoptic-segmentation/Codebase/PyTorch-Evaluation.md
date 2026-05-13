---
type: codebase-module
title: PyTorch — mbps_pytorch/evaluation/
project: mbps-panoptic-segmentation
module: mbps_pytorch.evaluation
framework: pytorch
paths:
  - mbps_pytorch/evaluation/
tags: [codebase, pytorch, evaluation, metrics]
related:
  - "[[PyTorch-Overview]]"
  - "[[Evaluation-Scripts]]"
  - "[[JAX-mbps-Evaluation]]"
---

# PyTorch — `mbps_pytorch/evaluation/`

| File | Role |
|------|------|
| `panoptic_quality.py` | PQ + per-class breakdown. |
| `instance_metrics.py` | mAP / AP50 / AP75. |
| `semantic_metrics.py` | mIoU, pixel accuracy. |
| `hungarian_matching.py` | Bipartite matching for instance association — also used by Mask2Former post-processing. |
| `visualizer.py` | Visualization of masks, predictions, overlays. |

## Reminder — eval protocol

The CUPS-standard global Hungarian over **27 classes** is the protocol of record. Per-image argmax is **not** comparable to CUPS reports. See the project-memory rule "CUPS eval protocol" and [[Reports-Index]].

## Companion CLI scripts

For batch evaluation across datasets and pseudo-label variants, see [[Evaluation-Scripts]].
