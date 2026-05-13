---
type: codebase-module
title: Panoptic-DeepLab + Mobile pipeline
project: mbps-panoptic-segmentation
module: mbps_pytorch.panoptic_deeplab
framework: pytorch
paths:
  - mbps_pytorch/panoptic_deeplab.py
  - mbps_pytorch/train_panoptic_deeplab.py
  - mbps_pytorch/train_mobile_panoptic.py
  - scripts/run_panoptic_deeplab_ablations.sh
  - scripts/run_mobile_ablations.sh
tags: [codebase, pytorch, panoptic, mobile, repvit, bifpn]
related:
  - "[[PyTorch-Overview]]"
  - "[[PyTorch-Mask2Former]]"
  - "[[Refs-Backbones]]"
  - "[[Reports-Index]]"
---

# Panoptic-DeepLab + Mobile pipeline

`panoptic_deeplab.py` exposes 4 panoptic architectures × 3 FPN types (BiFPN, Simple FPN, PANet). Trained via `train_panoptic_deeplab.py` with 12 CUPS-derived components (EMA, SWA, LSJ, Color Jitter, Dense CRF, etc.).

`train_mobile_panoptic.py` is the lightweight RepViT-M0.9 + BiFPN pipeline; the production-style mobile baseline.

## Best results (semantic-only, no instance heads)

| Setting | PQ | mIoU | Notes |
|---------|----|------|-------|
| Mobile baseline (RepViT-M0.9 + Simple FPN) | 23.73 | — | semantic-only |
| BiFPN + 8 CUPS tricks (remote, bs=4, ep46) | **24.78** | 53.11 | best mobile |
| BiFPN + 8 CUPS tricks (local, bs=8, ep46) | 24.04 | 52.17 | matches with smaller batch |
| Stage-3 self-training | 23.66 | — | hurt (-1.12 PQ); too noisy |

5.05M total params (backbone 4.72M + BiFPN decoder 0.33M).

Script: `mbps_pytorch/train_mobile_panoptic.py` (with `--resume`).

## Ablations

- `scripts/run_panoptic_deeplab_ablations.sh` — 13-run suite (arch / recipe / FPN).
- `scripts/run_mobile_ablations.sh` — earlier instance-head ablations (completed).

> **Excluded:** any T0 / T1 / T2 / Tn enumerated ablations under `ablations/`.

## Reports

- `reports/cups_semantic_ablation_report.md`
- `reports/repvit_cups_adaptation_analysis.md`
- `reports/mobile_distillation_gap_analysis.md`
- See [[Reports-Index]] for the full list.
