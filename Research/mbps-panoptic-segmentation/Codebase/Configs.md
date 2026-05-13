---
type: codebase-module
title: Configs (configs/)
project: mbps-panoptic-segmentation
language: en
tags: [codebase, configs, yaml]
paths:
  - configs/
related:
  - "[[JAX-mbps-Training]]"
  - "[[Scripts-Orchestration]]"
  - "[[Datasets]]"
---

# Configs — `configs/`

Hydra-style YAML deep-merge: `default.yaml` + dataset config + optional ablation override.

## Default & dataset

| File | Purpose |
|------|---------|
| `default.yaml` | Base hyperparameters. |
| `cityscapes.yaml`, `cityscapes_5pct.yaml`, `cityscapes_full.yaml` | Cityscapes (`_gcs`, `_gpu`, `_gpu_512`, `_multihost`, `_tpu_masks` variants). |
| `coco_stuff27.yaml`, `coco_stuff27_5pct.yaml`, `coco_stuff27_gcs.yaml` | COCO-Stuff-27 |
| `pascal_voc.yaml`, `nyu_depth_v2.yaml` | Other datasets. |

## CUPS variants

| File | Purpose |
|------|---------|
| `cups_cityscapes.yaml` | Stage-2 CUPS Cascade Mask R-CNN. |
| `cups_cityscapes_test100.yaml`, `cups_cityscapes_full8k.yaml` | Subset / full-size runs. |
| `cups_self_cityscapes.yaml` | Stage-3 self-training. |
| `cuts3d_coco.yaml` | CutS3D for COCO. |

## Proxy & multi-scale

| File | Purpose |
|------|---------|
| `proxy_cityscapes.yaml`, `proxy_cityscapes_gcs.yaml` | Cheap proxy runs for ablation screening. |
| `proxy_coco.yaml`, `proxy_coco_gcs.yaml` | COCO equivalents. |

## Ablation suites

`configs/ablations/`:
- `no_bicms.yaml` — forward-only Mamba.
- `no_consistency.yaml` — `δ = 0`.
- `no_depth_cond.yaml` — disable depth FiLM.
- `oracle_stuff_things.yaml` — ground-truth stuff/things.

`configs/v2_ablations/`:
- `dinov1.yaml`, `no_bicms.yaml`, `no_bridge.yaml`, `no_copy_paste.yaml`, `no_depth.yaml`, `no_self_train.yaml`.

> **Excluded from this graph:**
> - Any `T0 / T1 / T2 / Tn` enumerated ablation overrides (and their associated reports).
> - `no_mamba.yaml` ablation **is** referenced (it's in the experiment matrix), but the underlying mamba module isn't.
> - Adapter / DoRA / LoRA YAMLs (e.g. `depth_adapter_baseline.yaml`, `depthpro_adapter_baseline.yaml`, `semantic_adapter_baseline.yaml`).

## V2 defaults

`v2_default.yaml`, `v2_cityscapes.yaml`, `v2_cityscapes_gcs.yaml` — pair with `mbps_v2_model.py`.
