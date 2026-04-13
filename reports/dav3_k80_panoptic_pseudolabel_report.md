# DA3 + k=80 Panoptic Pseudo-Label Quality Report

**Date**: 2026-04-04  
**Eval split**: Cityscapes val (500 images)  
**Semantic source**: `pseudo_semantic_raw_k80` (CAUSE-TR k=80 overclustering, mapped to 19 trainIDs)  
**Instance source**: `pseudo_instance_dav3` (Depth Anything v3, Sobel+CC, τ=0.03, A_min=1000, dilation=3)  
**Metric**: Panoptic Quality (19-class Cityscapes trainID metric, global Hungarian matching)  
**Script**: `mbps_pytorch/ablate_instance_methods.py --method sobel_cc --depth_subdir depth_dav3`  
**Result file**: `results/ablation_da3_baseline_k80/ablation_sobel_cc_default_val.json`

---

## Summary

| Metric | Value |
|--------|-------|
| **PQ** | **27.37** |
| PQ_stuff | 32.08 |
| PQ_things | 20.90 |
| SQ | 73.44 |
| RQ | 35.66 |
| avg instances/img | 4.4 |
| eval time | 0.035 s/img |
| images | 500 |
| errors | 0 |

### Comparison to Baselines

| Configuration | PQ | PQ_stuff | PQ_things |
|---------------|-----|----------|-----------|
| DA3 + k=80 (this report) | **27.37** | **32.08** | **20.90** |
| SPIdepth + k=80 (prev best) | 26.74 | 32.08 | 19.41 |
| CUPS CVPR 2025 (27-class metric) | 27.8 | 29.4 | 17.7 |
| DINOv3 Stage-3 (27-class metric) | 30.3 | 31.3 | 28.5 |

> ⚠️ Note: CUPS and DINOv3 results use the 27-class CAUSE+Hungarian metric. This report uses the 19-class standard Cityscapes metric. Direct comparison is only valid within the same metric.

---

## Per-Class Results

### Stuff Classes (11 classes)

| Class | PQ | SQ | RQ | TP | FP | FN |
|-------|----|----|----|----|-----|-----|
| road | 77.05 | 78.89 | 97.66 | 480 | 20 | 3 |
| sidewalk | 41.05 | 68.35 | 60.06 | 288 | 207 | 176 |
| building | 66.69 | 75.95 | 87.80 | 432 | 61 | 59 |
| wall | 10.91 | 65.60 | 16.63 | 39 | 229 | 162 |
| fence | 9.22 | 59.96 | 15.38 | 31 | 183 | 158 |
| pole | 0.00 | 0.00 | 0.00 | 0 | 393 | 489 |
| traffic light | 0.00 | 0.00 | 0.00 | 0 | 0 | 260 |
| traffic sign | 14.69 | 61.31 | 23.96 | 107 | 317 | 362 |
| vegetation | 65.73 | 73.53 | 89.39 | 438 | 56 | 48 |
| terrain | 13.08 | 63.83 | 20.48 | 55 | 251 | 176 |
| sky | 54.41 | 71.29 | 76.32 | 348 | 127 | 89 |

### Thing Classes (8 classes)

| Class | PQ | SQ | RQ | TP | FP | FN | Note |
|-------|----|----|----|----|-----|-----|------|
| person | 6.36 | 66.03 | 9.63 | 179 | 161 | 3197 | Co-planar failure |
| rider | 12.12 | 58.61 | 20.68 | 70 | 66 | 471 | |
| car | 26.79 | 77.22 | 34.69 | 1029 | 269 | 3606 | Adjacent same-depth cars merge |
| truck | 34.82 | 79.75 | 43.66 | 31 | 18 | 62 | |
| bus | 47.73 | 80.74 | 59.12 | 47 | 14 | 51 | |
| train | 32.69 | 73.55 | 44.44 | 10 | 12 | 13 | |
| motorcycle | 0.00 | 0.00 | 0.00 | 0 | 0 | 149 | Too small for depth gradients |
| bicycle | 6.66 | 60.59 | 10.98 | 81 | 231 | 1082 | Small + co-planar |

---

## Train-Split Instance Statistics

Generated via `generate_depth_guided_instances.py` (τ=0.03, A_min=1000):

| Stat | Value |
|------|-------|
| Images | 2975 |
| Total instances | 12,044 |
| Avg instances/img | 4.0 |
| Generation time | 127.8s (0.043s/img) |

**Per-class instance counts (train split):**

| Class | Count | Note |
|-------|-------|------|
| person | 2,032 | Low — co-planar crowd merging |
| rider | 432 | |
| car | 7,866 | Dominant class |
| truck | 269 | |
| bus | 220 | |
| train | 141 | |
| motorcycle | 0 | **Zero** — too small for DA3 gradients |
| bicycle | 1,084 | |

---

## Failure Analysis

### Critical Bottlenecks

**1. Person: FN=3197 (95% miss rate)**  
- Root cause: pedestrians at the same depth (co-planar) produce zero depth gradient → merged into single region → one instance where GT has many
- Person PQ=6.36 despite SQ=66.03 — shape quality is fine when a person IS detected, but RQ=9.63 means only 9.6% of GT persons are matched

**2. Car: FN=3606 (78% miss rate)**  
- Root cause: adjacent cars in parking lots / traffic queues at similar depth → merged into one large region
- PQ=26.79 is the best thing class but still misses 3/4 of GT cars

**3. Motorcycle: PQ=0.00 (100% miss rate)**  
- Root cause: motorcycles are small (~100-500px) — DA3 depth gradient edges are too coarse to separate them from background/rider
- Motorcycle=0 in both val (TP=0) and train (count=0)

**4. Bicycle: FN=1082 (93% miss rate)**  
- Same as motorcycle but slightly larger; DA3 occasionally detects isolated bicycles

### What Works Well

- **Bus** (PQ=47.73): Large, distinct depth from surroundings → clean separation
- **Truck** (PQ=34.82): Similar to bus
- **Train** (PQ=32.69): Distinct depth profile on tracks
- **Car** (PQ=26.79): Isolated cars work well; failure is co-planar clusters

---

## Next Steps

Given the analysis, the depth-only ceiling is ~PQ=27.4 with PQ_things=20.90. The FN bottleneck is co-planar instances (person, car, bicycle). Possible paths forward:

1. **CUPS Stage-2 on DA3 instances**: Train Cascade Mask R-CNN on `pseudo_instance_dav3` train split — learned detector should generalise beyond depth gradients
2. **Post-hoc splitting**: Apply appearance-based (colour/texture) splitting to over-merged instances before evaluation
3. **Accept ceiling and focus on semantics**: PQ_stuff=32.08 has room to improve (pole=0, traffic_light=0); improving semantics directly lifts overall PQ
