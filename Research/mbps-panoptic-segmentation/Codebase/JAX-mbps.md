---
type: codebase-module
title: JAX/Flax — mbps/ package
project: mbps-panoptic-segmentation
language: en
module: mbps
framework: jax
paths:
  - mbps/
  - mbps/__init__.py
tags: [codebase, jax, flax, tpu]
related:
  - "[[00-Codebase-Map]]"
  - "[[JAX-mbps-Models]]"
  - "[[JAX-mbps-Training]]"
  - "[[JAX-mbps-Losses]]"
  - "[[JAX-mbps-Data]]"
  - "[[JAX-mbps-Evaluation]]"
---

# JAX/Flax — `mbps/` package

The TPU/JAX pipeline used for the NeurIPS 2026 multi-VM run. Trained on Cityscapes and COCO-Stuff-27 across 16 TPU v4 / v5e VMs via [[Scripts-Orchestration]].

## Subpackages

- [[JAX-mbps-Models]] — `mbps/models/` — `MBPSModel`, backbones, heads, merger
- [[JAX-mbps-Training]] — `mbps/training/` — trainer, curriculum, EMA, self-training, checkpointing
- [[JAX-mbps-Losses]] — `mbps/losses/` — semantic / instance / bridge / consistency / PQ proxy / gradient balancer
- [[JAX-mbps-Data]] — `mbps/data/` — datasets, augmentation, copy-paste, TFRecord utils
- [[JAX-mbps-Evaluation]] — `mbps/evaluation/` — PQ, mIoU, AP, Hungarian matching

## Conventions

- `nn.Module` subclasses with explicit `setup()` (not `@nn.compact`).
- Data parallelism via `jax.pmap`, gradient averaging via `lax.pmean`.
- All file I/O through `tf.io.gfile` so paths can be local or `gs://...`.
- Checkpointing: `_save_npy()` / `_load_npy()` via `BytesIO + tf.io.gfile`.
- Configuration: YAML deep-merge of `default.yaml` + dataset config + optional ablation; see [[Configs]].

## Excluded from this graph

The Mamba2 cross-modal bridge (`mbps/models/bridge/mamba2_ssd.py`) and any `*mamba*` ablation are intentionally not surfaced. The bridge orchestration logic in [[JAX-mbps-Models]] still references it as an opaque `bridge` block.
