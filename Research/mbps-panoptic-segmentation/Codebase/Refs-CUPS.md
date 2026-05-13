---
type: codebase-module
title: refs/cups/ — CUPS CVPR 2025
project: mbps-panoptic-segmentation
language: en
tags: [codebase, refs, cups]
paths:
  - refs/cups/
related:
  - "[[Pseudo-Label-Pipeline]]"
  - "[[Refinement-SIMCF]]"
  - "[[Refs-Backbones]]"
  - "[[Refs-Instance]]"
---

# refs/cups/ — CUPS CVPR 2025

Stage-2 trainer: Cascade Mask R-CNN that consumes pseudo-labels from [[Pseudo-Label-Pipeline]].

## Layout

```
refs/cups/
├── train.py                # Stage-2 trainer
├── train_self.py           # Stage-3 self-training
├── cups/                   # Python package
│   ├── augmentation.py
│   ├── config.py
│   ├── data/
│   ├── metrics/
│   │   └── panoptic_quality.py     # Hungarian-matching PQ (canonical)
│   ├── model/
│   │   ├── model.py                # ResNet variant
│   │   ├── model_vitb.py           # ViT-B variant
│   │   └── modeling/roi_heads/
│   │       ├── custom_cascade_rcnn.py
│   │       ├── fast_rcnn.py
│   │       └── semantic_seg.py
│   ├── pl_model_pseudo.py          # Lightning module — Stage-2 pseudo-label training
│   └── pl_model_self.py            # Lightning module — Stage-3 self-training (EMA)
└── configs/                # Stage-2/3 YAMLs
```

## Headline numbers

- **Stage-2 (DINOv3 ViT-B + DCFA + SIMCF-ABC, santosh)**: best step 744, train PQ=27.88, W&B `v70uy7wv`.
- **Stage-3 self-training**: peak PQ=39.12 at step ~2600 (peak ckpt **lost** due to default `save_top_k=1`); the saved best is PQ=35.83 at step 3000 — see `MEMORY.md`.
- Stage-2 ResNet-50 baseline: PQ=24.68 at step 6500.
- Stage-2 ViT-B/14 (DINOv2): PQ_things ≈ 20.6.

## Why CUPS metrics matter

The `refs/cups/cups/metrics/panoptic_quality.py` Hungarian protocol is the **only valid baseline** to compare against published CUPS PQ=27.8. Anything that uses 19-class per-image argmax is on a different scale. See `MEMORY.md` ("CUPS metric is 27-class CAUSE + Hungarian matching").

## Linked notes

- [[Refinement-SIMCF]] — pseudo-label refinement that feeds Stage-2.
- [[Pseudo-Label-Pipeline]] — generators that produce `cups_pseudo_labels_*` directories.
- [[Configs]] — `cups_cityscapes*.yaml` + `cups_self_cityscapes.yaml`.
