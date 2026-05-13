---
type: codebase-module
title: Instance Pseudo-Label Generators
project: mbps-panoptic-segmentation
language: en
tags: [codebase, pseudo-labels, instance]
paths:
  - mbps_pytorch/generate_*_instances.py
related:
  - "[[Pseudo-Label-Pipeline]]"
  - "[[Instance-Methods]]"
  - "[[Refs-Depth]]"
  - "[[Refs-Instance]]"
---

# Instance Pseudo-Label Generators

| Script | Method |
|--------|--------|
| `generate_adaptive_instances.py` | Adaptive depth-feature fusion. |
| `generate_cutler_instances.py`, `generate_cutler_detector_instances.py` | CutLER & CutLER-detector. |
| `generate_cuts3d_instances.py` | CutS3D pseudo-instance generator. |
| `generate_depth_guided_instances.py` | Sobel + depth threshold + connected components. |
| `generate_depth_layer_instances.py` | Depth-stratified layer-wise CC. |
| `generate_depth_multimodel_instances.py` | Combine multiple depth models. |
| `generate_depth_spidepth_instances.py` | SPIdepth-only variant. |
| `generate_depth_weighted_kmeans_instances.py` | k-means in joint depth-feature space. |
| `generate_dino_cluster_instances.py` | DINOv2 clustering instances. |
| `generate_dinosaur_instances.py` | DINOSAUR slot attention. |
| `generate_feature_depth_instances.py` | Feature clustering + depth refinement. |
| `generate_picl_instances.py` | PICL embeddings + clustering. |
| `generate_watershed_instances.py` | Watershed on depth + features. |
| `generate_instance_pseudolabels.py` | Generic wrapper. |
| `generate_instance_targets.py` | Convert pseudo-instances to training targets. |
| `merge_class_based.py`, `merge_depth_instances.py`, `merge_gapfill.py`, `merge_instance_sources.py`, `merge_nms.py` | Post-processing & merging. |
| `postprocess_instances.py` | Final clean-up before export. |
| `convert_to_cups_format.py` | Export to CUPS Stage-2 layout. |

> **Excluded:** any `generate_instance_pseudolabels_adapted.py` tied to the adapter pipeline.

## Outputs (Cityscapes)

| Path | Source |
|------|--------|
| `pseudo_instance_spidepth/` | SPIdepth depth-guided. |
| `cups_pseudo_labels_depthpro/` | DepthPro depth-guided, CUPS-format (8,925 files, ready for Stage-2). |
| `pseudo_instances_cutler/` | CutLER. |
| `pseudo_instances_cuvler/` | CuVLER. |

## Linked notes

- [[Instance-Methods]] — the underlying decomposition algorithms.
- [[Refs-Depth]] — depth backbones consumed by these scripts.
- [[Refs-Instance]] — third-party instance discovery references (CutLER, CuVLER, DiffNCuts).
