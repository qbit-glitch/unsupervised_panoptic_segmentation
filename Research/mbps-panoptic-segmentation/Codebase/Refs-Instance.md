---
type: codebase-module
title: refs/ — Instance Discovery (CutLER / CuVLER / DiffNCuts)
project: mbps-panoptic-segmentation
language: en
tags: [codebase, refs, instance]
paths:
  - refs/cutler/
  - refs/cuvler/
  - refs/DiffNCuts/
related:
  - "[[Generators-Instance]]"
  - "[[Instance-Methods]]"
  - "[[Refs-CUPS]]"
---

# refs/ — Instance Discovery

| Subdirectory | Method |
|--------------|--------|
| `refs/cutler/` | Cut-and-Learn: unsupervised object detection / instance segmentation (2.7× AP50 improvement). |
| `refs/cuvler/` | Enhanced Cut-Vote-Learn for class-agnostic discovery (CVPR 2024). |
| `refs/DiffNCuts/` | Differentiable Normalized Cuts for unsupervised dense prediction (ECCV 2024). |

## Empirical ranking (Cityscapes things, k=80 protocol)

| Method | PQ_things |
|--------|-----------|
| Depth-guided (DepthPro) | **23.35** |
| Depth-guided (DA3) | 20.90 |
| Depth-guided (SPIdepth) | 19.41 |
| **CUPS Cascade Mask R-CNN** ([[Refs-CUPS]] Stage-2) | ~20.6 (DINOv2 ViT-B), higher with DINOv3 |
| CutLER | 10.05 |
| CuVLER | 7.55 |
| HDBSCAN v2 | 9.29 |
| DINOSAUR (30-slot) | 8.4 |
| MaskCut | 1.9 |

Only the CUPS trained detector beats the depth-guided generators on this metric.
