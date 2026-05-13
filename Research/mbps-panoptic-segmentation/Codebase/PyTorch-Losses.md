---
type: codebase-module
title: PyTorch — mbps_pytorch/losses/
project: mbps-panoptic-segmentation
module: mbps_pytorch.losses
framework: pytorch
paths:
  - mbps_pytorch/losses/
tags: [codebase, pytorch, losses]
related:
  - "[[PyTorch-Overview]]"
  - "[[PyTorch-Training]]"
  - "[[PyTorch-Mask2Former]]"
  - "[[JAX-mbps-Losses]]"
---

# PyTorch — `mbps_pytorch/losses/`

| File | Role |
|------|------|
| `bridge_loss.py` | Unified loss for depth-feature bridge conditioning. |
| `confidence_filtering.py` | Confidence-based hard-example mining for pseudo-labels. |
| `consistency_loss.py` | Cross-branch (semantic ↔ instance) consistency. |
| `feature_consistency.py` | Perceptual feature-level consistency. |
| `gradient_balancing.py` | Multi-task gradient-norm balancing. |
| `instance_embedding_loss.py` | Discriminative push/pull for per-pixel embeddings. |
| `instance_loss.py` | General instance segmentation loss. |
| `mae_regularizer.py` | MAE regularization for stability. |
| `mask2former_loss.py` | Hungarian-matching Mask2Former loss; consumed by [[PyTorch-Mask2Former]]. |
| `pq_proxy_loss.py` | Differentiable PQ surrogate. |
| `refiner_loss.py` | Loss for [[RefineNet-Family]] training. |
| `semantic_loss.py`, `semantic_loss_v2.py` | Semantic CE and variants. |

## Cross-language map

These mirror [[JAX-mbps-Losses]]; v2 modules pair with `mbps_v2_model.py`.
