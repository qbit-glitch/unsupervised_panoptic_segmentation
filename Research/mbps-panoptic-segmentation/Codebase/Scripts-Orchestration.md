---
type: codebase-module
title: Orchestration & Top-level Scripts
project: mbps-panoptic-segmentation
language: en
tags: [codebase, scripts, orchestration, tpu]
paths:
  - scripts/orchestrate.py
  - scripts/train.py
  - scripts/evaluate.py
  - scripts/coordinate.py
  - scripts/setup_data_pipeline.sh
related:
  - "[[JAX-mbps-Training]]"
  - "[[Configs]]"
  - "[[Datasets]]"
  - "[[Evaluation-Scripts]]"
---

# Orchestration & top-level scripts

| Script | Role |
|--------|------|
| `scripts/orchestrate.py` | Single-command lifecycle for 16 TPU VMs across 3 waves: smoke → create → setup → train → monitor. Phases: `smoke`, `create`, `setup`, `launch [--waves N]`, `monitor`, `status`, `cleanup --force`. |
| `scripts/train.py` | Single-VM JAX training entry point. Calls into [[JAX-mbps-Training]]. |
| `scripts/evaluate.py` | Parallel inference (GPU / TPU) with optional CRF. |
| `scripts/coordinate.py` | GCS-polling coordinator that aggregates per-worker checkpoint metrics on `mbps-v4-0`. |
| `scripts/setup_data_pipeline.sh` | Downloads datasets, pre-computes depth, builds TFRecords. Subcommands: `weights`, `cityscapes`, `coco`, `verify`, `all`. |
| `scripts/run_*.sh` (50+) | Training launches for CUPS / DCFA / depth / mobile baselines. **Excluded**: any `T0` / `T1` / `Tn` adapter / mamba launches. |

## Experiment matrix (36 runs in 3 waves)

| Wave | Jobs | Contents |
|------|------|----------|
| 1 | 16 | Full runs + seed-42 ablations (Cityscapes + COCO × 6 configs × seed 42 + full) |
| 2 | 16 | Seeds 123 / 456 ablations |
| 3 | 4 | Remaining seed-456 ablations |

(2 datasets × 6 configurations × 3 seeds = 36 runs. Phase A/B/C: 60 epochs + Phase D self-training: 15 epochs = 75 total.)

The 6 configurations are `full` plus the 5 ablations in [[Configs]] — `no_mamba`, `no_depth_cond`, `no_bicms`, `no_consistency`, `oracle_stuff_things`. The `no_mamba` ablation **does** show up in this graph at the orchestration level (because it's a real run), even though the mamba code itself is excluded.

## Quotas (TRC grant)

| Accelerator | Type | Zone | Chips | VMs (8-chip) |
|-------------|------|------|-------|--------------|
| v4-8 | On-demand | us-central2-b | 32 | 4 |
| v4-8 | Spot | us-central2-b | 32 | 4 |
| v5e-8 | Spot | us-central1-a | 64 | 8 |
| **Total** | | | **128** | **16** |
