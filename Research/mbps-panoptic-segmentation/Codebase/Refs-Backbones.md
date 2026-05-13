---
type: codebase-module
title: refs/ — Vision Backbones (DINO / DINOv3 / DINOSAUR / sinder)
project: mbps-panoptic-segmentation
language: en
tags: [codebase, refs, backbones, dino]
paths:
  - refs/dino/
  - refs/dinov3/
  - refs/dinosaur/
  - refs/sinder/
related:
  - "[[Refs-CUPS]]"
  - "[[JAX-mbps-Models]]"
  - "[[PyTorch-Models]]"
---

# refs/ — Vision Backbones

| Subdirectory | Model |
|--------------|-------|
| `refs/dino/` | DINO (original, ViT-S/8). Canonical self-supervised ViT. |
| `refs/dinov3/` | DINOv3 (latest) — used as the production backbone for Stage-2 / Stage-3 in [[Refs-CUPS]]. |
| `refs/dinosaur/` | Object-centric learning via slot attention (ICLR '23 unofficial). 30-slot variant evaluated in [[Instance-Methods]]. |
| `refs/sinder/` | DINOv2 singular-defect repair (ECCV '24 Oral). Stabilizes feature extraction. |

## Conversion

Weights are converted to JAX PyTrees by `mbps/models/backbone/weights_converter.py` and `dinov3_weights_converter.py`. PyTorch loaders live in `mbps_pytorch/models/backbone/weights_loader.py`.
