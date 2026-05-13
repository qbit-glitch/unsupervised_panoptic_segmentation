---
type: codebase-module
title: refs/ — Semantic Discovery (CAUSE / DepthG / STEGO)
project: mbps-panoptic-segmentation
language: en
tags: [codebase, refs, semantic]
paths:
  - refs/cause/
  - refs/depthg/
  - refs/stego/
related:
  - "[[Generators-Semantic]]"
  - "[[JAX-mbps-Models]]"
  - "[[PyTorch-Models]]"
---

# refs/ — Semantic Discovery

| Subdirectory | Method | Used for |
|--------------|--------|----------|
| `refs/cause/` | Causal Unsupervised Semantic Segmentation via 90-D codes. | CAUSE-TR pipeline, k=80 / k=300 overclustering. |
| `refs/depthg/` | Depth-guided semantic feature distillation (ICCV '23). | DepthG semantic head (`mbps/models/semantic/depthg_head.py`). |
| `refs/stego/` | Self-supervised semantic learning via clustering. | STEGO loss in `mbps/losses/semantic_loss.py`. |

## DCFA — Depth-Conditioned Feature Adapter

DCFA (V3 internal label) is a 40K-param adapter that adapts CAUSE 90-D codes with depth conditioning. Best result: mIoU=60.65% (+6.24 over baseline).

> **Note**: DCFA is a small linear-probe adapter, structurally distinct from the LoRA/DoRA stack — DCFA *is* documented; the LoRA/DoRA blocks are not.

## Reports

- `reports/depth_semantic_ablation_complete.md`
- `reports/dcfa_v2_ablation_analysis.md`
- `reports/cause_tr_refinement.md`
