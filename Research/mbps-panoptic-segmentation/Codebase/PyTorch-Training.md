---
type: codebase-module
title: PyTorch — mbps_pytorch/training/
project: mbps-panoptic-segmentation
module: mbps_pytorch.training
framework: pytorch
paths:
  - mbps_pytorch/training/
tags: [codebase, pytorch, training, curriculum]
related:
  - "[[PyTorch-Overview]]"
  - "[[PyTorch-Models]]"
  - "[[PyTorch-Losses]]"
  - "[[JAX-mbps-Training]]"
---

# PyTorch — `mbps_pytorch/training/`

| File | Role |
|------|------|
| `trainer.py` | Main MBPS trainer (PyTorch baseline). |
| `trainer_v2.py` | v2 trainer with Mamba bridge — references the excluded `mbps_pytorch/mamba2/` package internally. |
| `cause_modeb_cluster.py` | CAUSE Mode B: clustering-based pseudo-label generation. |
| `cause_modeb_freeze.py` | CAUSE Mode B with frozen backbone. |
| `cause_modeb_trainer.py` | Mode B trainer wrapper. |
| `checkpointing.py` | Gradient checkpointing for memory efficiency. |
| `curriculum.py` | Easy → hard pseudo-label progression. |
| `ema.py` | EMA teacher updates. |
| `pseudo_label_correction.py` | Adaptive pseudo-label correction (Uni-UVPT inspired). |
| `self_training.py` | Self-training with confidence filtering. |

## Excluded from this graph

- All adapter trainers (`train_*adapter*.py`, `train_*lora*.py` at the root of `mbps_pytorch/`) and DoRA scripts — see [[00-Codebase-Map]].
- TTT-Mamba2 refiner trainer (`train_ttt_mamba2_refiner.py`, `ttt_mamba2_refiner.py`).

## Linked entry points

- Refinement training: [[RefineNet-Family]] (`train_refine_net.py`, `train_joint_refine_net.py`).
- Mobile pipeline: [[Panoptic-DeepLab]] (`train_panoptic_deeplab.py`, `train_mobile_panoptic.py`).
- Self / pseudo-label rounds: this folder + [[Refinement-SIMCF]].
