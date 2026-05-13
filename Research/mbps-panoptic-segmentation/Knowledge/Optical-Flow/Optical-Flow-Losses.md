---
type: knowledge
title: "Optical Flow Loss Functions"
project: mbps-panoptic-segmentation
status: active
updated: 2026-05-07
tags:
  - optical-flow
  - losses
  - literature-map
---

# Optical Flow Loss Functions

[[Papers/Optical-Flow/Optical-Flow-SMURF-Lineage|SMURF lineage]] | [[Optical-Flow-Algorithms]] | [[Optical-Flow-MBPS-Relevance]]

## Loss Families

| Loss Family | Used By | Formula Sketch | What It Buys | Risk |
| --- | --- | --- | --- | --- |
| Photometric/census reconstruction | [[SMURF-2021]], [[Reviving-Unsupervised-Optical-Flow-2026]], [[U2Flow-2026]], [[Self-Supervised-AutoFlow-2023]] | compare image 1 with image 2 warped by predicted flow | Enables unlabeled training | Fails under occlusion, lighting change, non-Lambertian surfaces |
| Edge-aware smoothness | [[SMURF-2021]], [[Reviving-Unsupervised-Optical-Flow-2026]], [[U2Flow-2026]], [[Self-Supervised-AutoFlow-2023]] | flow derivatives downweighted at RGB edges | Regularizes textureless regions | Over-smooths motion boundaries if edge prior is wrong |
| Teacher-student augmentation self-supervision | [[SMURF-2021]], [[Reviving-Unsupervised-Optical-Flow-2026]], [[Self-Supervised-AutoFlow-2023]] | teacher clean/full prediction supervises student augmented/cropped prediction | Teaches crop borders, occlusion robustness, augmentation invariance | Bad teacher predictions can reinforce errors |
| Augmentation uncertainty NLL | [[U2Flow-2026]] | Laplace NLL from augmentation-induced flow inconsistency | Learns reliability maps without labels | Needs careful gradient detachment/calibration |
| Homography smoothness | [[U2Flow-2026]] | L1 between flow and homography-refined flow | Strong driving-scene planar regularizer | Dataset-specific; can hurt non-planar scenes |
| Masked cost-volume autoencoding | [[FlowFormerPlusPlus-2023]] | MSE between large target cost patch and predicted normalized patch | Pretrains transformer cost memory | Architecture-specific, not final flow supervision |
| RAFT-style supervised sequence L1 | [[VideoFlow-2023]], [[MegaFlow-2026]] | sum_k gamma^(K-k) L1(flow_k, gt) | Strong direct flow supervision | Requires labels |
| Mixture-of-Laplace sequence loss | [[MEMFOF-2025]], [[DPFlow-2025]], [[ARFlow-2026]] | negative log likelihood under per-pixel Laplace mixture | Handles ambiguous flow/outliers better than plain L1 | Requires labels and probability head design |

## SMURF-Style Unsupervised Objective

```text
L = L_photo + omega_smooth L_smooth + omega_self L_self
L_sequence = sum_i gamma^(n-i) L_i
```

This is the core recipe behind [[SMURF-2021]] and a major component in [[Self-Supervised-AutoFlow-2023]]. [[Reviving-Unsupervised-Optical-Flow-2026]] keeps the same family but simplifies the setup and reevaluates masking/augmentation. [[U2Flow-2026]] keeps the unsupervised family but adds uncertainty as a first-class prediction.

## Supervised Teacher Objective

```text
L = sum_t sum_k gamma^(K-k) loss(flow_t,k, gt_t)
```

[[VideoFlow-2023]] uses L1. [[MEMFOF-2025]], [[DPFlow-2025]], and [[ARFlow-2026]] use mixture-of-Laplace. [[MegaFlow-2026]] supervises initial global matching with smooth L1 and recurrent refinement with L1.

## Practical Rule For MBPS

- If the goal is a fair SMURF replacement, prefer unsupervised losses from [[U2Flow-2026]] or [[Reviving-Unsupervised-Optical-Flow-2026]].
- If the goal is stronger pseudo-labels, supervised teacher losses from [[VideoFlow-2023]], [[MEMFOF-2025]], [[ARFlow-2026]], or [[MegaFlow-2026]] are acceptable, but the paper claim must say "teacher generated labels" or "foundation/supervised flow prior", not "pure unsupervised flow".

