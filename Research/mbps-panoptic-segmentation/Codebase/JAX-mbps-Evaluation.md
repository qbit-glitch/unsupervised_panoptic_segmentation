---
type: codebase-module
title: JAX/Flax — mbps/evaluation/
project: mbps-panoptic-segmentation
module: mbps.evaluation
framework: jax
paths:
  - mbps/evaluation/
tags: [codebase, jax, evaluation, metrics]
related:
  - "[[JAX-mbps]]"
  - "[[JAX-mbps-Training]]"
  - "[[Evaluation-Scripts]]"
  - "[[PyTorch-Evaluation]]"
---

# JAX/Flax — `mbps/evaluation/`

Top-level exports: `compute_panoptic_quality`, `PQResult`, `hungarian_match`, `compute_miou`, `compute_ap`, `compute_ap_range`.

| File | Role |
|------|------|
| `panoptic_quality.py` | `PQ = SQ · RQ`, plus per-class PQ, `PQ^Th` (things), `PQ^St` (stuff). |
| `instance_metrics.py` | Average Precision at IoU 0.5, 0.75, mean over `[0.5:0.05:0.95]`. |
| `hungarian_matching.py` | Bipartite matching for semantic-instance assignment + mIoU. |
| `semantic_metrics.py` | mIoU, per-class IoU. |
| `visualizer.py` | Panoptic-segmentation visualization helpers. |

## Cross-language parity

The same metrics are duplicated in [[PyTorch-Evaluation]] for the PyTorch pipeline. Numbers in reports usually come from one or the other depending on which model produced the prediction. CUPS-style global Hungarian (27-class) is the protocol of record — see [[Reports-Index]] and the rule "CUPS eval protocol" in project memory.
