---
type: knowledge
title: "Optical Flow Relevance To MBPS"
project: mbps-panoptic-segmentation
status: active
updated: 2026-05-07
tags:
  - optical-flow
  - mbps
  - cups
  - pseudo-labels
---

# Optical Flow Relevance To MBPS

[[Papers/Optical-Flow/Optical-Flow-SMURF-Lineage|SMURF lineage]] | [[Optical-Flow-Algorithms]] | [[Optical-Flow-Losses]]

## Why This Matters

CUPS uses stereo/video/flow-derived motion cues for instance pseudo-labels. MBPS replaces that dependency with monocular depth and frozen single-frame priors. The flow literature remains useful for three reasons:

- It defines the strongest motion-based baseline that MBPS is replacing.
- It provides teacher candidates if we want stronger pseudo-labels for offline training.
- It gives loss and filtering ideas, especially uncertainty and teacher-student consistency, that can transfer to monocular pseudo-label refinement.

## Candidate Uses

| Use Case | Best Paper Starting Point | Reason |
| --- | --- | --- |
| Fair unsupervised CUPS/SMURF replacement | [[U2Flow-2026]] | Best like-for-like successor with improved metrics and uncertainty |
| Reproducible two-frame unsupervised baseline | [[Reviving-Unsupervised-Optical-Flow-2026]] | Open PyTorch, short training, simpler than SMURF |
| Flow reliability mask for instance proposals | [[U2Flow-2026]] | Predicts uncertainty directly |
| Strong multi-frame teacher labels | [[VideoFlow-2023]], [[MEMFOF-2025]], [[ARFlow-2026]] | Better occlusion and temporal reasoning, but supervised |
| High-resolution flow teacher | [[MEMFOF-2025]], [[DPFlow-2025]] | Avoids crop/tiling/downsample artifacts |
| Foundation correspondence teacher | [[MegaFlow-2026]] | DINO-style priors, large displacement, point tracking transfer |
| Target-domain synthetic flow pretraining | [[Self-Supervised-AutoFlow-2023]] | Learns renderer from unlabeled target-domain losses |

## Claim Hygiene

- "Unsupervised flow successor" should be reserved for [[U2Flow-2026]] and [[Reviving-Unsupervised-Optical-Flow-2026]].
- [[VideoFlow-2023]], [[MEMFOF-2025]], [[DPFlow-2025]], [[ARFlow-2026]], [[FlowFormerPlusPlus-2023]], and [[MegaFlow-2026]] are stronger flow estimators, but not fair replacements for SMURF in an unsupervised-method comparison.
- [[MegaFlow-2026]] uses "zero-shot" in the benchmark-transfer sense, not the no-supervision sense.

## Practical MBPS Reading

If the project needs only a literature citation for why MBPS avoids flow, cite [[SMURF-2021]] and CUPS. If it needs a real modern flow baseline, start with [[U2Flow-2026]]. If it needs a teacher for better pseudo-labels rather than a fair unsupervised baseline, evaluate [[MegaFlow-2026]] or [[ARFlow-2026]] first.

