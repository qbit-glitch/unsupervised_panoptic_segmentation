---
type: codebase-module
title: Datasets
project: mbps-panoptic-segmentation
language: en
tags: [codebase, datasets, gcs]
related:
  - "[[JAX-mbps-Data]]"
  - "[[PyTorch-Data]]"
  - "[[Configs]]"
  - "[[Scripts-Orchestration]]"
---

# Datasets

## GCS bucket layout (`gs://mbps-panoptic`)

```
gs://mbps-panoptic/
  datasets/
    cityscapes/
      tfrecords/{train,val}/     # sharded TFRecords
      depth_zoedepth/            # precomputed depth (.npy)
      leftImg8bit/               # raw images
      gtFine/                    # ground truth
    coco/
      tfrecords/{train,val}/
      depth_zoedepth/
      images/
  checkpoints/
    {experiment_name}/{vm_name}/checkpoint_epoch_XXXX/
  results/
    {experiment_name}/{vm_name}/
  weights/
    dino_vits8_flax.npz          # converted DINO weights
  logs/
```

## Active datasets

| Dataset | Used for | Size | Notes |
|---------|----------|------|-------|
| **Cityscapes** | Stage-1/2/3 training, eval | ~20 GB | needs CITYSCAPES_USERNAME/PASSWORD |
| **COCO-Stuff-27** | Stage-1/2 training, cross-dataset | ~25 GB | Big domain gap |
| **NYU Depth V2** | Depth pretraining baseline | — | Less used now |
| **PASCAL VOC** | Auxiliary | — | Older runs |

## Cross-dataset OOD

| Dataset | Sample count | Used in eval |
|---------|--------------|--------------|
| MOTS | 2,862 | DINOv3 Stage-3 OOD |
| KITTI | 200 | DINOv3 Stage-3 OOD |
| COCO-Stuff-27 | 5,000 | DINOv3 Stage-3 OOD |
| Mapillary Vistas | various | RepViT mobile cross-dataset |
| KITTI-STEP | sparse | RepViT (PQ=0 due to sparse GT) |
| COCONUT | 19 → 133 classes | RepViT cross-dataset |
| BDD-10K | — | Script ready, data blocked |
| MUSES | — | Need data |

Reports: `reports/dinov3_stage3_cross_dataset_evaluation.md`, `reports/cross_dataset_evaluation_report.md`.

## Cross-references

- Loaders: [[JAX-mbps-Data]], [[PyTorch-Data]].
- Setup: `scripts/setup_data_pipeline.sh` in [[Scripts-Orchestration]].
- Configs: per-dataset YAML in [[Configs]].
