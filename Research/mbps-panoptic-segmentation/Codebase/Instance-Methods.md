---
type: codebase-module
title: Instance Methods (mbps_pytorch/instance_methods/)
project: mbps-panoptic-segmentation
module: mbps_pytorch.instance_methods
framework: pytorch
paths:
  - mbps_pytorch/instance_methods/
  - mbps_pytorch/ablate_instance_methods.py
tags: [codebase, instance, decomposition, pytorch]
related:
  - "[[PyTorch-Overview]]"
  - "[[Generators-Instance]]"
  - "[[Refs-Depth]]"
  - "[[Reports-Index]]"
---

# Instance Methods — `mbps_pytorch/instance_methods/`

15 instance-decomposition strategies that take a depth map + DINO features and emit instance masks. The runner in `mbps_pytorch/ablate_instance_methods.py` sweeps through them.

| File | Method |
|------|--------|
| `sobel_cc.py` | Baseline: Sobel edges + connected components. |
| `tda_persistence.py` | Topological data analysis / persistent homology. |
| `mumford_shah.py` | Mumford-Shah energy in joint depth-feature space. |
| `feature_edge_cc.py` | DINOv2 feature gradients + depth fusion + CC. |
| `joint_ncut.py` | Depth-feature joint normalized cut. |
| `learned_edge_cc.py` | Learned depth-edge detector + CC (inference). |
| `contrastive_embed.py` | Contrastive depth-feature embedding + HDBSCAN. |
| `optimal_transport.py` | Sinkhorn OT decomposition. |
| `morse_flow.py` | Morse / gradient-flow decomposition. |
| `adaptive_edge.py` | Adaptive depth-feature edge fusion. |
| `depth_stratified.py` | Depth-stratified DINOv2 spectral clustering. |
| `plane_decomp.py` | Local plane decomposition (GeoDepth-inspired). |
| `picl_embed.py` | PICL: project DINOv2 features through trained head. |
| `learned_merge.py` | Two-stage: depth oversegmentation + learned merge predictor. |
| `utils.py` | Shared utilities (boundary reclamation, CC, filtering). |

## Best on Cityscapes (k=80, 27-class CAUSE protocol)

| Method | PQ_things |
|--------|-----------|
| **DepthPro depth-guided (NEW BEST)** | 23.35 (PQ=28.40) |
| DA3 depth-guided | 20.90 (PQ=27.37) |
| SPIdepth depth-guided | 19.41 (PQ=26.74) |
| CutLER | 10.05 |
| CuVLER | 7.55 |
| HDBSCAN v2 | 9.29 |
| DINOSAUR (30 slot) | 8.4 |
| MaskCut | 1.9 |
| Center+Offset head v2 | 9.79 |

The only thing that beats depth-guided in things PQ is CUPS Cascade Mask R-CNN at Stage-2.

## Generation scripts

These methods are wrapped by various `mbps_pytorch/generate_*_instances.py` scripts — see [[Generators-Instance]].

## Reports

- `reports/novel_instance_ablation_default_results.md`
- `reports/novel_instance_ablation_final_report.md`
- `reports/depthpro_instance_ablation.md`
- `reports/depth_guided_cc_instance_method.md`
- `reports/overclustered_spidepth_sweep.md`
- See [[Reports-Index]].

## ⚠️ Lesson — depth splitting is an illusion for CUPS training

The best CUPS training result (PQ=33.51) used **plain CC**, not depth-split instances. Depth-split instances (~57/img, 51% smaller than 1k pixels) hurt CUPS Cascade Mask R-CNN training even though their pseudo-label PQ looks better. See `MEMORY.md` "CRITICAL: Depth splitting illusion".
