---
type: experiment
title: UNet Decoder Architecture Ablation
project: mbps-panoptic-segmentation
language: en
status: complete
updated: 2026-03-29T13:40:24Z
tags:
  - unet
  - semantic-segmentation
  - ablation
---

# UNet Decoder Architecture Ablation

**Date**: 2026-03-08 | **Status**: COMPLETE

## Best Result
**P2-B 2-stage attention**: PQ=28.00 (ep8), PQ_stuff=35.04, PQ_things=18.32, mIoU=57%, 5.45M params.

> **WARNING**: PQ=28.00 is 19-class standard metric. NOT comparable to CUPS 27.8 (27-class CAUSE + Hungarian matching). On same metric: CUPS=38.59 >> UNet=28.00.

## Key Findings
- **Attention blocks >> conv blocks** at every resolution (+0.27 PQ, 11.5x slower)
- **Block type >> resolution ~ capacity**: P2-A ~ P2-D (27.65 ~ 27.64)
- **Focal loss gamma=1.0** best (+0.12 PQ over CE)
- **Universal overfitting after ep6-8**: All runs decline 0.41-0.57 PQ post-peak
- **Feature noise delays PQ_things peak**: Feature aug peaks at ep14 with PQ_things=18.09

## Reports
- `reports/unet_ablation_study.md`
- `reports/unet_phase2_architecture_ablation.md`
- `reports/unet_unified_ablation_study.md`
