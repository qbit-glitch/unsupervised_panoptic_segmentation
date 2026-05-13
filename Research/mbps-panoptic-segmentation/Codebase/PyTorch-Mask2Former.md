---
type: codebase-module
title: PyTorch — Mask2Former (mbps_pytorch/models/mask2former/)
project: mbps-panoptic-segmentation
module: mbps_pytorch.models.mask2former
framework: pytorch
paths:
  - mbps_pytorch/models/mask2former/
tags: [codebase, pytorch, mask2former, panoptic]
related:
  - "[[PyTorch-Models]]"
  - "[[PyTorch-Training]]"
  - "[[PyTorch-Losses]]"
---

# PyTorch — `mbps_pytorch/models/mask2former/`

| File | Role |
|------|------|
| `mask2former_model.py` | Mask2Former architecture: transformer decoder + pixel decoder. |
| `feature_pyramid.py` | FPN backbone wrapper. |
| `pixel_decoder.py` | ASPP-like pixel-level decoder. |
| `transformer_decoder.py` | Transformer decoder with masked cross-attention. |
| `position_encoding.py` | Sine / learnable positional encoding. |
| `panoptic_postprocessor.py` | Post-processing for panoptic merging. |

## Loss / training pairing

The Mask2Former Hungarian-matching loss lives in [[PyTorch-Losses]] (`mask2former_loss.py`). Training entry points that use this stack: `train_panoptic_deeplab.py` (with `--head mask2former`) — see [[Panoptic-DeepLab]] — and the Stage-2 M2F + ViT-Adapter experiments in [[Reports-Index]].

## Reports

- `reports/stage2_m2f_*.md` — see [[Reports-Index]] (M2F Mask2Former Stage-2 reports).
