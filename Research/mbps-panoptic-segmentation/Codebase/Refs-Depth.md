---
type: codebase-module
title: refs/ — Depth Models
project: mbps-panoptic-segmentation
language: en
tags: [codebase, refs, depth]
paths:
  - refs/zoedepth/
  - refs/spidepth/
  - refs/depth-anything-3/
  - refs/prodepth/
  - refs/adversarial_depth/
related:
  - "[[Generators-Instance]]"
  - "[[Instance-Methods]]"
  - "[[Reports-Index]]"
---

# refs/ — Depth Models

Depth backbones consumed by the depth-guided instance pipeline.

| Subdirectory | Model | Notes |
|--------------|-------|-------|
| `refs/zoedepth/` | ZoeDepth (MiDaS-based) | Default for early experiments. |
| `refs/spidepth/` | SPIdepth (CVPR 2025, pose-informed self-supervised) | Stage-1 best for things via depth-guided splitting (PQ_things=19.41). |
| `refs/depth-anything-3/` | DepthAnything v3 (DPT-based) | Used for DA3 instance generator (PQ_things=20.90). |
| `refs/prodepth/` | Probabilistic depth | Used in `generate_depth_multimodel_instances.py`. |
| `refs/adversarial_depth/` | Robustness hardening | Auxiliary. |
| (DepthPro) | Apple DepthPro | Best on Cityscapes things — PQ_things=23.35 (`cups_pseudo_labels_depthpro/`). Used via Hugging Face / direct loader, not in `refs/`. |

## Reports

- `reports/depth_model_ablation_study.md`
- `reports/depthpro_instance_ablation.md`
- `reports/depth_semantic_ablation_complete.md`
