---
type: codebase-module
title: SIMCF Refinement (refine_simcf*)
project: mbps-panoptic-segmentation
language: en
tags: [codebase, pseudo-labels, refinement, simcf]
paths:
  - scripts/refine_simcf.py
  - scripts/refine_simcf_v2.py
  - scripts/refine_cn_simcf.py
related:
  - "[[Pseudo-Label-Pipeline]]"
  - "[[Generators-Semantic]]"
  - "[[Generators-Instance]]"
  - "[[Refs-CUPS]]"
---

# SIMCF Refinement

**SIMCF** = Semantic-Instance Mutual Consistency Filtering. Three implementations:

| Script | Variant |
|--------|---------|
| `scripts/refine_simcf.py` | 3-step refinement (initial). |
| `scripts/refine_simcf_v2.py` | 5 novel refinement passes (D–H). v2 D–H **all failed** (see memory). |
| `scripts/refine_cn_simcf.py` | Cluster-Native SIMCF — operates on raw k=80 codes, used in DCFA + SIMCF-ABC headline. |

## DCFA + SIMCF-ABC

The headline pseudo-label config used in NeurIPS 2026 draft:

- **Semantic**: DCFA (Depth-Conditioned Feature Adapter, 40K params, V3_dd16_h384_l2) over DINOv2 codes.
- **Instance**: DepthPro depth-guided (τ=0.20, A_min=1000).
- **Refinement**: SIMCF-ABC (cluster-native, 3 passes).

Best train PQ = 25.85, val PQ = 27.81 (mIoU=55.29). See `reports/dcfa_depthpro_simcf_abc_pseudolabel_report.md`.

## Spec

- `reports/simcf_method_specification.md`
- `reports/dcfa_simcf_abc_descriptions.md`

## Related

- [[Refs-CUPS]] — Stage-2 trainer consumes the refined labels.
- [[Pseudo-Label-Pipeline]] — overall dataflow.
- [[Reports-Index]] — list of refinement reports.
