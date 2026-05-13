---
type: experiment
title: Instance Decomposition Ablation Study
project: mbps-panoptic-segmentation
language: en
status: phase-b-running
updated: 2026-03-29T13:40:24Z
tags:
  - instance-segmentation
  - ablation
  - mumford-shah
---

# Instance Decomposition Ablation Study

**Date**: 2026-03-28 to 2026-03-29
**Status**: Phase A COMPLETE (239 configs). Phase B in progress (Mumford-Shah top 5 on 500 images).
**Motivation**: NeurIPS W1 weakness — Sobel+CC has "zero algorithmic novelty"

## Setup
- **Dataset**: Cityscapes val (500 imgs full, 100 imgs Phase A for expensive methods)
- **Semantics**: k=80 overclustered pseudo-labels (fixed across all methods)
- **Depth**: SPIdepth monocular
- **Features**: DINOv2 ViT-B/14 (2048 patches, 768-dim)
- **Eval**: 512x1024, 19-class trainID, PQ with IoU>0.5 matching

## Results Summary

| Method | Configs | Best PQ_things | Delta vs Baseline | Speed |
|--------|---------|----------------|-------------------|-------|
| **Mumford-Shah** | 36 (100 imgs) | **23.27** | **+3.86 (+19.9%)** | 19.75 s/img |
| Sobel+CC (baseline) | 15 | 19.41 | --- | 0.04 s/img |
| TDA | 36 | 16.70 | -2.71 | 1.76 s/img |
| Morse | 56 | 16.66 | -2.75 | 0.30 s/img |
| Contrastive | 24 | 6.78 | -12.63 | 0.09 s/img |
| OT | 72 | 2.45 | -16.96 | 0.09 s/img |

## Key Finding
**Mumford-Shah spectral clustering** with beta=1.0 (DINOv2 feature weight) is the clear winner. Joint depth+feature affinity in a principled energy framework substantially beats depth-only splitting. The beta (feature weight) parameter dominates alpha (depth weight) by orders of magnitude.

### Best Mumford-Shah Config
`alpha=1.0, beta=1.0, n_clusters=10, min_area=1000, resolution=64x128`

### Per-Class Improvements (Mumford-Shah vs Sobel+CC)
- car: 19.17 vs 16.49 (+2.68)
- truck: 48.52 vs 35.52 (+13.0)
- bus: 55.15 vs 47.76 (+7.39)
- train: 49.34 vs 36.43 (+12.91)
- person: 4.45 vs 4.02 (+0.43) — still the bottleneck

## Theoretical Lessons
1. Depth-only methods hit a ceiling at co-planar objects
2. Appearance features (DINOv2) are necessary but insufficient alone (contrastive fails)
3. Joint depth+feature reasoning in energy minimization beats post-hoc fusion
4. Watershed methods are degenerate on smooth monocular depth
5. Uniform mass constraint (OT) is catastrophically wrong for instance decomposition

## Code
- Methods: `mbps_pytorch/instance_methods/` (6 modules)
- Sweep script: `mbps_pytorch/ablate_instance_methods.py`
- Phase B: `mbps_pytorch/mumford_shah_phase_b.py`
- Reports: `reports/novel_instance_ablation_default_results.md`, `reports/novel_instance_ablation_sweep_progress.md`
