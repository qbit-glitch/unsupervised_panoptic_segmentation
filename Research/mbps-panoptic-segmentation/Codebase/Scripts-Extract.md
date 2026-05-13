---
type: codebase-module
title: Extract scripts (features, codes, normals)
project: mbps-panoptic-segmentation
language: en
tags: [codebase, scripts, feature-extraction]
paths:
  - mbps_pytorch/extract_*.py
  - scripts/extract_*.py
related:
  - "[[Pseudo-Label-Pipeline]]"
  - "[[Refs-Backbones]]"
  - "[[Refs-Depth]]"
---

# Extract scripts

Pre-extraction utilities that materialize features / codes / normals to disk so downstream generators can run cheaply.

| Script | What it extracts |
|--------|------------------|
| `mbps_pytorch/extract_dinov2_features.py` | DINOv2 ViT-B/14 patch features. |
| `mbps_pytorch/extract_dinov3_features.py` | DINOv3 ViT-B/16 features. |
| `mbps_pytorch/extract_dinov3_features_coco_hires.py` | DINOv3 high-res features for COCO. |
| `mbps_pytorch/extract_dinov3_features_neco.py` | NECO-pretrained DINOv3 features. |
| `mbps_pytorch/extract_ssd1b_features.py` | SSD1B foundation-model features. |
| `scripts/extract_cause_codes.py` | Frozen CAUSE 90-D codes + depth patches. |
| `scripts/extract_cuts3d_gpu.py` | CutS3D pseudo-masks (GPU, paper-faithful). |
| `scripts/extract_cuts3d_coco_gpu.py` | CutS3D for COCO/ImageNet. |
| `scripts/extract_full_cuts3d.py` | Full CutS3D extraction with GCS batching. |
| `scripts/extract_surface_normals.py` | Surface normals from DepthPro depth. |

## Storage layout

Outputs go under `gs://mbps-panoptic/datasets/<dataset>/<feature_kind>/` for TPU runs and `~/cached_features/` locally. See [[Datasets]] for the bucket schema.
