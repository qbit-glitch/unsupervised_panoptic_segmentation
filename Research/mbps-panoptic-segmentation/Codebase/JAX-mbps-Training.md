---
type: codebase-module
title: JAX/Flax — mbps/training/
project: mbps-panoptic-segmentation
module: mbps.training
framework: jax
paths:
  - mbps/training/
tags: [codebase, jax, training, curriculum]
related:
  - "[[JAX-mbps]]"
  - "[[JAX-mbps-Models]]"
  - "[[JAX-mbps-Losses]]"
  - "[[Scripts-Orchestration]]"
  - "[[Configs]]"
---

# JAX/Flax — `mbps/training/`

Drives the 4-phase curriculum on TPU.

| File | Role |
|------|------|
| `trainer.py` | Main entry: 4-phase curriculum (A semantic, B instance, C bridge, D self-train), `jax.pmap` step, W&B logging. |
| `curriculum.py` | `PhaseConfig` schedule — α / β / γ / δ / ε weights and toggles for gradient projection, bridge, consistency, PQ losses. |
| `checkpointing.py` | Orbax checkpoint save/load: model params, opt state, EMA params, training state. |
| `ema.py` | Exponential moving average teacher (μ=0.999) used by PQ proxy and self-training. |
| `self_training.py` | Phase D: confidence-thresholded pseudo-labels from EMA teacher, ascending threshold across rounds. |

## Curriculum at a glance

| Phase | Epochs | Active losses (in [[JAX-mbps-Losses]]) |
|-------|--------|----------------------------------------|
| A | 1–20 | `L_semantic` (STEGO + DepthG) |
| B | 21–40 | + `L_instance` (Dice + drop + box) |
| C | 41–60 | + `L_bridge`, `L_consistency`, `L_pq` |
| D | 61–75 | self-training rounds with EMA teacher |

## Linked plumbing

- Configuration: see [[Configs]] (`configs/default.yaml` + dataset YAML + optional ablation YAML).
- Orchestration across 16 VMs: see [[Scripts-Orchestration]] (`scripts/orchestrate.py`).
- Coordinator (results aggregator on v4-0): `scripts/coordinate.py`.

## Excluded

- Adapter-based training scripts (`train_*adapter*`, `train_*lora*`) and DoRA recipes are part of the PyTorch tree and are intentionally outside this graph.
