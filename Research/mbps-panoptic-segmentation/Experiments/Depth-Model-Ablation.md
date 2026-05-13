---
type: experiment
title: Depth Model Ablation Study
project: mbps-panoptic-segmentation
language: en
status: complete
updated: 2026-03-29T13:40:24Z
tags:
  - depth-estimation
  - ablation
---

# Depth Model Ablation Study

**Date**: 2026-03-28 | **Status**: COMPLETE

## Key Findings

### Cityscapes
DA3 (Depth Anything V3) is best: PQ_things=20.90 (tau=0.03) > DA2-L (20.20) > SPIdepth (19.41). Depth model quality matters more than splitting algorithm — alternative splitting algorithms (multiscale Sobel, Canny, watershed) provide negligible improvement.

### COCO-Stuff-27
DA2-Large slightly edges DA3 (PQ_things=14.04 vs 13.76). Depth contribution is minimal (+0.77 over CC-only) because semantic quality (mIoU=18.3%) is the ceiling.

## Results

| Depth Model | Cityscapes PQ_things | COCO PQ_things | Opt tau |
|-------------|---------------------|----------------|---------|
| DA3 | **20.90** | 13.76 | 0.03 |
| DA2-Large | 20.20 | **14.04** | 0.03 |
| SPIdepth | 19.41 | 13.27 | 0.20 |
| CC-only | 14.93 | 13.27 | - |

## Person Instance Failure Taxonomy
- Semantic miss: 65.0% — k=80 labels don't cover the object
- Co-planar merge: 30.2% — adjacent same-depth persons merge
- Matched: 3.6%
- Over-split: 1.2%

## Reports
- Full: `reports/depth_model_ablation_study.md`
