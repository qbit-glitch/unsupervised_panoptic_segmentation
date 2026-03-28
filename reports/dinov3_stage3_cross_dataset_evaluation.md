# DINOv3 Stage-3 Cross-Dataset Evaluation Report

**Model**: DINOv3 ViT-B/16 + Cascade Mask R-CNN (Stage-3, step 8000)
**Checkpoint**: `checkpoints/dinov3_stage3/dinov3_official_stage3_step8000.ckpt`
**Training data**: Cityscapes (k=80 overclustering pseudo-labels)
**Parameters**: 137.8M total (91.2M backbone frozen, 52.2M trainable)
**Evaluation**: Two-pass memory-safe (O(1) memory), CPU inference, Hungarian matching
**Date**: 2026-03-27

---

## 1. MOTSChallenge (OOD)

**Dataset**: 2,862 frames, 4 driving scenes, 2 classes (background + person)
**Results file**: `results/mots_dinov3_stage3_step8000.json`

| Metric     | Value (%) |
|------------|-----------|
| **PQ**     | 63.375    |
| SQ         | 88.118    |
| RQ         | 69.021    |
| **PQ_th**  | 30.384    |
| SQ_th      | 79.868    |
| RQ_th      | 38.042    |
| **PQ_st**  | 96.367    |
| SQ_st      | 96.367    |
| RQ_st      | 100.000   |
| mIoU       | 91.435    |
| Acc        | 98.282    |

### Per-Class Breakdown

| Class      | Type  | PQ (%)  | SQ (%)  | RQ (%)  |
|------------|-------|---------|---------|---------|
| background | Stuff | 96.367  | 96.367  | 100.000 |
| person     | Thing | 30.384  | 79.868  | 38.042  |

### Analysis

- **Background**: Near-perfect segmentation (PQ=96.4%, RQ=100%). The model reliably identifies background regions across domains.
- **Person**: SQ=79.9% shows good segmentation quality when instances are matched, but RQ=38.0% means only ~38% of person instances are correctly detected. This is expected for OOD — the model was trained on Cityscapes driving scenes and MOTS uses different camera setups and pedestrian distributions.

---

## 2. COCO-Stuff-27

**Dataset**: 5,000 val images, 27 coarse supercategory classes (12 things, 15 stuff)
**Results file**: `results/coco_stuff27_dinov3_stage3_step8000.json`

| Metric     | Value (%) |
|------------|-----------|
| **PQ**     | 8.047     |
| SQ         | 55.337    |
| RQ         | 10.721    |
| **PQ_th**  | 7.347     |
| SQ_th      | 43.157    |
| RQ_th      | 9.220     |
| **PQ_st**  | 8.607     |
| SQ_st      | 65.081    |
| RQ_st      | 11.922    |
| mIoU       | 15.238    |
| Acc        | 37.696    |

### Per-Class Breakdown

| Class           | Type  | PQ (%)  | SQ (%)  | RQ (%)  |
|-----------------|-------|---------|---------|---------|
| electronic      | Thing | 1.582   | 96.003  | 1.648   |
| appliance       | Thing | 0.000   | 0.000   | 0.000   |
| food            | Thing | 0.041   | 61.342  | 0.067   |
| furniture       | Thing | 0.096   | 63.134  | 0.152   |
| indoor          | Thing | 0.000   | 0.000   | 0.000   |
| kitchen         | Thing | 0.000   | 0.000   | 0.000   |
| accessory       | Thing | 0.842   | 71.900  | 1.171   |
| animal          | Thing | 1.734   | 66.255  | 2.618   |
| outdoor         | Thing | 0.000   | 0.000   | 0.000   |
| **person**      | Thing | **49.010** | 81.387  | 60.219  |
| sports          | Thing | 0.000   | 0.000   | 0.000   |
| **vehicle**     | Thing | **34.853** | 77.860  | 44.764  |
| ceiling         | Stuff | 1.760   | 61.910  | 2.843   |
| floor           | Stuff | 1.718   | 64.288  | 2.673   |
| food-stuff      | Stuff | 0.072   | 50.811  | 0.143   |
| furniture-stuff | Stuff | 0.431   | 59.307  | 0.727   |
| raw-material    | Stuff | 2.658   | 70.140  | 3.790   |
| textile         | Stuff | 0.723   | 60.042  | 1.204   |
| wall            | Stuff | 9.194   | 64.383  | 14.280  |
| window          | Stuff | 2.771   | 62.451  | 4.437   |
| building        | Stuff | 8.651   | 65.630  | 13.182  |
| **ground**      | Stuff | **21.144** | 70.845  | 29.845  |
| **plant**       | Stuff | **24.059** | 69.719  | 34.508  |
| **sky**         | Stuff | **38.146** | 84.426  | 45.183  |
| solid           | Stuff | 0.560   | 57.841  | 0.968   |
| structural      | Stuff | 16.562  | 68.892  | 24.040  |
| water           | Stuff | 0.656   | 65.533  | 1.001   |

### Analysis

- **Strong classes** (shared with Cityscapes domain): person (49.0%), vehicle (34.9%), sky (38.1%), plant (24.1%), ground (21.1%) — these are common driving-scene categories that transfer well.
- **Weak classes** (absent from Cityscapes): appliance, indoor, kitchen, outdoor, sports all score 0% — the model has no concept of these indoor/non-driving categories.
- **Domain gap is severe**: Overall PQ=8.0% reflects that most COCO categories (indoor scenes, animals, food) are completely outside the Cityscapes training distribution.
- **SQ is reasonable when matched**: For classes with non-zero PQ, SQ values are 60-96%, indicating the model produces decent masks when it does find a match. The bottleneck is RQ (detection recall).

---

## 3. KITTI Panoptic (OOD)

**Dataset**: 200 validation images, 27 Cityscapes classes (same label format)
**Results file**: `results/kitti_dinov3_stage3_step8000.json`

| Metric     | Value (%) |
|------------|-----------|
| **PQ**     | 29.315    |
| SQ         | 52.712    |
| RQ         | 36.842    |
| **PQ_th**  | 24.946    |
| SQ_th      | 54.880    |
| RQ_th      | 31.000    |
| **PQ_st**  | 32.046    |
| SQ_st      | 51.357    |
| RQ_st      | 40.493    |
| mIoU       | 44.454    |
| Acc        | 82.141    |

### Per-Class Breakdown

| Class           | Type  | PQ (%)  | SQ (%)  | RQ (%)  |
|-----------------|-------|---------|---------|---------|
| **road**        | Stuff | **91.295** | 91.754  | 99.500  |
| **sky**         | Stuff | **89.130** | 90.347  | 98.652  |
| **vegetation**  | Stuff | **75.639** | 78.425  | 96.447  |
| **bus**         | Thing | **63.555** | 95.333  | 66.667  |
| terrain         | Stuff | 57.267  | 75.164  | 76.190  |
| wall            | Stuff | 52.635  | 75.004  | 70.175  |
| bicycle         | Thing | 50.183  | 71.093  | 70.588  |
| building        | Stuff | 40.840  | 66.829  | 61.111  |
| train           | Thing | 40.088  | 86.857  | 46.154  |
| traffic sign    | Stuff | 39.068  | 76.834  | 50.847  |
| truck           | Thing | 38.606  | 90.080  | 42.857  |
| sidewalk        | Stuff | 35.416  | 69.448  | 50.996  |
| fence           | Stuff | 27.113  | 74.162  | 36.559  |
| rider           | Thing | 23.085  | 65.407  | 35.294  |
| person          | Thing | 18.378  | 70.808  | 25.954  |
| car             | Thing | 15.567  | 69.222  | 22.489  |
| traffic light   | Stuff | 3.668   | 57.306  | 6.400   |
| pole            | Stuff | 0.668   | 66.433  | 1.005   |
| parking         | Stuff | 0.000   | 0.000   | 0.000   |
| rail track      | Stuff | 0.000   | 0.000   | 0.000   |
| guard rail      | Stuff | 0.000   | 0.000   | 0.000   |
| bridge          | Stuff | 0.000   | 0.000   | 0.000   |
| tunnel          | Stuff | 0.000   | 0.000   | 0.000   |
| polegroup       | Stuff | 0.000   | 0.000   | 0.000   |
| caravan         | Thing | 0.000   | 0.000   | 0.000   |
| trailer         | Thing | 0.000   | 0.000   | 0.000   |
| motorcycle      | Thing | 0.000   | 0.000   | 0.000   |

### Analysis

- **Strong transfer for driving-scene stuff**: road (91.3%), sky (89.1%), vegetation (75.6%) transfer near-perfectly — same domain as Cityscapes training.
- **Large vehicles detected well**: bus (63.6%, SQ=95.3%), truck (38.6%, SQ=90.1%), train (40.1%, SQ=86.9%) — high SQ shows excellent mask quality when matched.
- **Car underperforms** (PQ=15.6%, RQ=22.5%): surprising given cars dominate Cityscapes. KITTI has different camera viewpoint (dashboard-mounted, lower resolution) causing distribution shift.
- **Rare/small classes fail**: pole (0.7%), traffic light (3.7%), parking/guard rail/bridge/tunnel all 0% — either absent from KITTI val or too small to detect at this resolution.

---

## Summary

| Dataset        | Images | PQ (%)  | PQ_th (%) | PQ_st (%) | mIoU (%) |
|----------------|--------|---------|-----------|-----------|----------|
| **MOTS**       | 2,862  | 63.375  | 30.384    | 96.367    | 91.435   |
| **KITTI**      | 200    | 29.315  | 24.946    | 32.046    | 44.454   |
| **COCO-St-27** | 5,000  | 8.047   | 7.347     | 8.607     | 15.238   |

**Key takeaway**: Cross-dataset generalization correlates strongly with domain similarity to Cityscapes. MOTS (driving, 2 classes) transfers best. KITTI (driving, 27 classes) is moderate. COCO (diverse indoor/outdoor, 27 classes) shows the largest domain gap.

---

## Evaluation Setup

- **Hardware**: M4 Pro MacBook 48GB, CPU-only inference
- **Method**: Two-pass memory-safe evaluation
  - Pass 1: Accumulate cost matrix [num_clusters x num_target_classes], no prediction caching
  - Pass 2: Stream PQ + mIoU with pre-computed Hungarian assignments
- **Memory**: ~3-4 GB peak (vs ~80 GB with default CUPS `trainer.validate()`)
- **Scripts**: `refs/cups/evaluate_mots.py`, `refs/cups/evaluate_coco_stuff27.py`, `refs/cups/evaluate_kitti.py`
