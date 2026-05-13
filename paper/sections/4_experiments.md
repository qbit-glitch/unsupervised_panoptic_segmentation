# 4 Experiments

We report experiments on Cityscapes and zero-shot transfer datasets, separating pseudo-label quality from trained-network performance. The evaluation asks whether monocular pseudo-labels can seed the adopted CUPS training procedure, how much DCFA and SIMCF contribute, and where the fixed Cityscapes-derived pseudo-class vocabulary fails.

## 4.1 Experimental Setup

Pseudo-label generation and panoptic network training use only Cityscapes train [Cordts et al., 2016] (2,975 images, no annotations); evaluation reports Cityscapes val (500 images) under the 27-class CAUSE pseudo-vocabulary with a Hungarian map to Cityscapes' 19 evaluation classes [Kirillov et al., 2019], following CUPS. The Stage-1 generator uses frozen DINOv2 ViT-B/14, the frozen CAUSE-TR head, DCFA, k-means over-clustering at k = 80, the monocular instance branch with tau_d = 0.20, A_min = 1000, and three iterations of dilation, and SIMCF with tau_sim = 0.85 and eta = 2.5. DepthPro supplies the monocular depth map.

We follow CUPS for the training side. Panoptic bootstrapping denotes the initial training of a panoptic network on Stage-1 pseudo-labels with DropLoss and self-enhanced copy-paste augmentation. Panoptic network training denotes the subsequent EMA self-training rounds, where teacher predictions relabel the training set and the student continues training on the refined labels. In both stages, the panoptic network is a Cascade Mask R-CNN with three cascade stages on a frozen DINOv3 ViT-B/16 backbone; we use 8K bootstrapping steps followed by three 5K-step EMA rounds with EMA coefficient 0.999 and a three-scale teacher.

## 4.2 Main Cityscapes Results

Table 1 records the main comparison on Cityscapes val. Rows 4 and 5 use the same panoptic bootstrapping and panoptic network training implementation defined above; only the Stage-1 pseudo-label source differs. DCFA and SIMCF add 3.07 PQ in this controlled comparison.

| # | Method | Training | PL inputs | PQ | SQ | RQ | mIoU |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | U2Seg [Niu et al., 2024] | published | RGB monocular | 18.4 | 55.8 | 22.7 | -- |
| 2 | S2-UniSeg [Xu et al., 2025] | published | RGB monocular | 25.4 | 66.7 | 35.0 | -- |
| 3 | CUPS [Hahn et al., 2025] | published | RGB + stereo + optical flow | 27.80 | 57.4 | 35.2 | 26.80 |
| 4 | Monocular baseline (no DCFA, no SIMCF) | ours | RGB monocular | 32.76 | 62.57 | 40.75 | 45.14 |
| 5 | Ours (DCFA + SIMCF) | ours | RGB monocular | 35.83 | 62.79 | 43.78 | 44.56 |

Table 1. Cityscapes panoptic results. PL inputs are the inputs available at pseudo-label time. Published rows use metrics reported by the original papers; rows 4 and 5 isolate DCFA and SIMCF under the same panoptic bootstrapping and panoptic network training implementation.

The final model is obtained after three EMA self-training rounds. Cityscapes validation improves from 31.87/60.84/39.94 (PQ/SQ/RQ) at step 800 to 33.77/61.75/42.04 at step 1000, 35.47/62.57/43.21 at step 2200, and 35.83/62.79/43.78 at step 3000. Most of the gain comes from recognition rather than mask quality: RQ rises by 3.84 points across this trajectory, while SQ rises by 1.95 points. The cross-row mIoU change (56.22 pseudo-label vs 44.56 trained model) is not directly comparable because pseudo-label mIoU is computed against the 27-class CAUSE-TR vocabulary, while trained-model mIoU is computed under the 19-class Cityscapes evaluation.

## 4.3 Ablations

Component contribution. Table 2 reports DCFA by SIMCF at the fixed training-data instance configuration (tau_d = 0.20). All four cells share the same depth instances; only the pre-clustering adapter and the post-instance filter are toggled. The logged single-component runs separate the behavior of the two corrections: DCFA raises SQ, while SIMCF raises RQ. The combined run has the best recorded pseudo-label PQ, consistent with the two stages targeting different failure modes.

| Variant | DCFA | SIMCF | PQ | SQ | RQ | mIoU |
|---|---:|---:|---:|---:|---:|---:|
| Raw k=80 | no | no | 24.54 | 60.0 | 33.6 | 56.56 |
| DCFA only | yes | no | 25.22 | 61.0 | 34.0 | 56.16 |
| SIMCF only | no | yes | 25.27 | 57.5 | 34.4 | 56.57 |
| Both | yes | yes | 25.85 | 58.3 | 34.9 | 56.22 |

Table 2. Pseudo-label component complementarity at fixed tau_d = 0.20 DepthPro instances. Raw k=80 is the no-DCFA, no-SIMCF baseline.

Depth model. Table 3 reports representative depth-source and threshold runs for which the evaluations provide aggregate SQ and RQ. The rows are not used as a single leaderboard because some runs use the raw k=80 semantic map and others use the DCFA semantic map; they are used to separate mask quality from recognition quality while choosing the training-data configuration.

| Depth | Semantics | tau | PQ | SQ | RQ |
|---|---|---:|---:|---:|---:|
| SPIdepth | raw k=80 | 0.20 | 26.74 | 71.88 | 31.41 |
| DA v3 | raw k=80 | 0.03 | 27.37 | 73.44 | 35.66 |
| DA v3 | DCFA | 0.20 | 26.44 | 61.37 | 35.83 |
| DepthPro | DCFA | 0.01 | 27.81 | 63.07 | 37.40 |
| DepthPro | DCFA | 0.20 | 26.13 | 60.63 | 35.80 |

Table 3. Depth and threshold diagnostics with aggregate SQ/RQ.

The table shows why PQ alone is not a sufficient diagnostic for the depth branch. The sharper tau = 0.01 DepthPro setting has stronger standalone PQ/SQ/RQ, but it produces a fragmented training set with roughly 57 instances per image and more than half below 1000 pixels. The tau = 0.20 setting has lower standalone PQ but produces larger training targets, roughly 22 instances per image, which is the configuration used for panoptic bootstrapping. DA v3 at the same training-data threshold reaches a similar RQ, indicating that the choice is governed by the training signal produced by the depth branch rather than by the stuff/thing aggregate split.

## 4.4 Generalization and Failure Analysis

| Dataset | # img | PQ | SQ | RQ | mIoU |
|---|---:|---:|---:|---:|---:|
| Cityscapes val | 500 | 35.83 | 62.79 | 43.78 | 44.56 |
| KITTI [Geiger et al., 2012] | 200 | 34.85 | 62.43 | 42.91 | 46.87 |
| Mapillary v2 | 2,000 | 39.19 | 74.59 | 47.74 | 58.87 |
| MOTS [Voigtlaender et al., 2019] | 2,862 | 61.10 | 88.71 | 65.80 | 92.10 |
| COCO-Stuff-27 [Caesar et al., 2018] | 1,000 | 7.83 | 38.55 | 10.06 | 14.22 |

Table 4. Zero-shot cross-dataset transfer without fine-tuning.

KITTI and Mapillary Vistas v2 show that the trained network transfers across the driving class space without retraining. KITTI stays close to Cityscapes across all three panoptic metrics, while Mapillary reaches higher SQ and RQ under its validation protocol. The MOTS row should not be read as a direct transfer score: MOTS evaluates a much narrower label space and therefore produces unusually high SQ and RQ. COCO-Stuff-27 shows the limit of the fixed pseudo-class vocabulary: a Cityscapes-derived k=80 pseudo-label space cannot recover categories that were not represented during supervision, yielding only 10.06 RQ and 7.83 PQ.

Per-class panoptic quality on Cityscapes' 19 evaluation classes splits into three groups. Large stuff (road 92.99, sky 86.05, vegetation 84.70, building 83.54) and large vehicles (car 70.71, bus 76.67, train 77.17) are strong, because frozen semantic codes give coherent regions and depth boundaries roughly coincide with object extent. Thin structures (pole 2.05, traffic light 6.20) and recognition-bottleneck classes (person 13.37, rider 22.94) are weak: monocular depth provides no separating discontinuity for co-planar pedestrians. Six classes are effectively dead: parking, guard rail, tunnel, polegroup, caravan, and trailer, because the k=80 pseudo-label vocabulary after Hungarian alignment to the 27-class evaluation contains few or no positive examples for them.

Two qualitative metrics from the feature-guided merge step of SIMCF close the analysis. Average pseudo-instance count per image drops from 44 before the merge to 22 after, median instance size grows from 5,502 to 14,965 pixels, and stuff contamination falls from 50.7% to 28.0%. Many depth gradients are intra-object surface changes rather than object boundaries, and merging fragments whose semantic identity and dense appearance agree recovers the single-object structure that monocular depth had over-fragmented.
