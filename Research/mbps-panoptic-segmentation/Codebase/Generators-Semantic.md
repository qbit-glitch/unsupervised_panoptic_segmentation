---
type: codebase-module
title: Semantic Pseudo-Label Generators
project: mbps-panoptic-segmentation
language: en
tags: [codebase, pseudo-labels, semantic]
paths:
  - mbps_pytorch/generate_semantic_pseudolabels.py
  - mbps_pytorch/generate_overclustered_semantics.py
  - mbps_pytorch/generate_depth_overclustered_semantics.py
  - mbps_pytorch/generate_depthg_semantic_pseudolabels.py
  - mbps_pytorch/generate_coco_pseudo_semantics.py
related:
  - "[[Pseudo-Label-Pipeline]]"
  - "[[Generators-Instance]]"
  - "[[Refs-Backbones]]"
  - "[[Refs-Semantic]]"
---

# Semantic Pseudo-Label Generators

| Script | Method |
|--------|--------|
| `generate_semantic_pseudolabels.py` | Generic semantic generator (multiple backends). |
| `generate_overclustered_semantics.py` | Overclustering: k > #classes (e.g. k=80 vs 27 CAUSE classes). |
| `generate_depth_overclustered_semantics.py` | Depth-conditioned overclustering for curriculum learning. |
| `generate_depthg_semantic_pseudolabels.py` | DepthG (depth-guided correlation) labels. |
| `generate_coco_pseudo_semantics.py` | COCO-Stuff-27 variant. |
| `diffcut_pseudo_semantics.py` | DiffCut clustering. |
| `falcon_pseudo_semantics.py` | Falcon segmentation. |
| `matryoshka_pseudo_semantics.py` | Hierarchical (matryoshka) clustering. |
| `sam_consensus_pseudo_semantics.py` | Consensus from SAM. |
| `spectral_pseudo_semantics.py` | Spectral clustering. |
| `generate_refined_semantics.py` | Post-refinement filter. |
| `refine_semantic_pseudolabels.py` | Confidence filtering. |
| `remap_cause27_to_trainid.py` | CAUSE → Cityscapes train-id remap. |

> **Excluded:** any `generate_semantic_pseudolabels_adapted*.py` variants tied to the LoRA/DoRA adapter pipeline.

## Centroid invariance

Different machines re-running k-means **diverge 100%** on cluster IDs. The centroids in `pseudo_semantic_raw_k80/kmeans_centroids.npz` are the single source of truth. See the `a6000_pseudolabel_divergence` memory and `reports/a6000_pseudolabel_reproducibility.md`.

## Companions

- [[Generators-Instance]] — depth-guided / cluster / cut-based instance variants.
- [[Refinement-SIMCF]] — refinement passes on top of these labels.
- [[Scripts-Extract]] — feature pre-extraction (`extract_dinov2_features.py`, `extract_dinov3_features.py`, …).
