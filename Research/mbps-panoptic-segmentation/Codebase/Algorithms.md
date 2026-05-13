---
type: codebase-module
title: Core Algorithms
project: mbps-panoptic-segmentation
language: en
tags: [codebase, algorithms, theory]
paths:
  - core_algorithms.md
  - cascade_stage_adaptation.md
related:
  - "[[JAX-mbps-Models]]"
  - "[[Refs-CUPS]]"
  - "[[Pseudo-Label-Pipeline]]"
  - "[[Guidelines-Index]]"
---

# Core Algorithms

Pointers to the in-repo algorithm references.

| File | Contents |
|------|----------|
| `core_algorithms.md` | CutS3D core algorithms in CLRS pseudocode (NCut, LocalCut 3D, panoptic merge). |
| `cascade_stage_adaptation.md` | CUPS dissection: PQ contribution breakdown across pipeline stages. |
| `guidelines/mamba_panoptic_technical_report.md` | Loss math and bridge math (mamba content present, but the surrounding loss derivations are still load-bearing for non-mamba parts). |

## In-code algorithm landmarks

- Bipartite matching for instances: `mbps/evaluation/hungarian_matching.py` and `mbps_pytorch/evaluation/hungarian_matching.py`.
- Panoptic merge: `mbps/models/merger/panoptic_merge.py` (Algorithm 9 in the technical report).
- Depth-guided instance splitting: depth Sobel + connected components, see [[Instance-Methods]].
- DCFA depth conditioning: `mbps_pytorch/models/semantic/depth_adapter*.py` (note: this is the *DCFA* adapter, which is documented; the LoRA/DoRA stack is not).
