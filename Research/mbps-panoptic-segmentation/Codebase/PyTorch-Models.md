---
type: codebase-module
title: PyTorch — mbps_pytorch/models/
project: mbps-panoptic-segmentation
module: mbps_pytorch.models
framework: pytorch
paths:
  - mbps_pytorch/models/
tags: [codebase, pytorch, models]
related:
  - "[[PyTorch-Overview]]"
  - "[[PyTorch-Mask2Former]]"
  - "[[RefineNet-Family]]"
  - "[[Panoptic-DeepLab]]"
  - "[[JAX-mbps-Models]]"
  - "[[Refs-Backbones]]"
  - "[[Refs-Semantic]]"
  - "[[Refs-Instance]]"
---

# PyTorch — `mbps_pytorch/models/`

PyTorch ports of the JAX architecture in [[JAX-mbps-Models]], plus PyTorch-specific extensions.

## Top-level

| File | Role |
|------|------|
| `mbps_model.py` | Unified MBPS interface (DINO ViT-S/8 backbone). |
| `mbps_v2_model.py` | DINOv3 + Mamba bridge panoptic segmentation. **The mamba bridge itself is excluded from this graph; the model definition that uses it is documented as a black-box `bridge` block.** |

## Subdirectories

### `models/backbone/` → [[Refs-Backbones]]
- `dino_vits8.py` — DINOv2 ViT-S/8.
- `dinov3_vitb.py` — DINOv3 ViT-B/14.
- `weights_loader.py` — load pretrained weights from HF / torchvision.

### `models/bridge/` (non-mamba parts only)
- `depth_conditioning.py` — Unified Depth Conditioning Module (UDCM).
- `bicms.py` — Bi-directional Cross-Modal Synchronization wrapper. The wrapped Mamba2 SSD blocks are excluded.
- `projection.py` — feature projection / alignment.

### `models/classifier/`
- `cues.py` — DBD / FCC / IDF cue fusion.
- `stuff_things_mlp.py` — MLP classifier for stuff vs thing.

### `models/semantic/` → [[Refs-Semantic]]
- `depthg_head.py` — depth-guided semantic head.
- `depth_adapter.py`, `depth_adapter_v2.py` — **excluded** (LoRA/DoRA-related).
- `stego_loss.py` — STEGO loss; consumed by [[PyTorch-Losses]].

### `models/instance/` → [[Refs-Instance]]
- `cascade_mask_rcnn.py` — class-agnostic Cascade Mask R-CNN for instance detection.
- `cuts3d.py` — CutS3D pseudo-instance integration.
- `instance_loss.py` — instance-specific loss; mirrored in [[PyTorch-Losses]].

### `models/merger/`
- `panoptic_merge.py` — semantic + instance → panoptic.
- `crf_postprocess.py` — CRF boundary refinement.

### `models/refiner/` (excluded)
- `mamba2_panoptic_refiner.py`, `four_dir_scan.py`, `instance_encoder.py`, `geometric_features.py` — all part of the Mamba2 Panoptic Refiner (M2PR) and excluded per the global exclusion banner.

### `models/mask2former/` → [[PyTorch-Mask2Former]]
Broken out into its own note because the file count and complexity warrant it.

## Adapters — fully excluded

`models/adapters/` (LoRA/DoRA layers, `cause_adapter.py`, `dinov2_adapter.py`, `depthpro_adapter.py`, etc.) is **not** documented in this graph. See the exclusion banner in [[00-Codebase-Map]].
