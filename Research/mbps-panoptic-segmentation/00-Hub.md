---
type: project
title: MBPS Panoptic Segmentation
project: mbps-panoptic-segmentation
language: en
status: active
tags:
  - research/project
  - bmvc-2026
  - unsupervised-panoptic
updated: 2026-04-13T00:00:00Z
---

# MBPS — Unsupervised Panoptic Segmentation

## Mission
Fully unsupervised panoptic segmentation targeting **NeurIPS 2026**. No ground-truth labels anywhere. Fuses depth-guided semantics with instance decomposition via a Mamba2 state-space bridge over frozen DINOv2/DINOv3 features.

**Targets**: Cityscapes PQ >= 28.0 | COCO-Stuff-27 PQ >= 22.5

## Core Index
- [[01-Plan]] — Active goals, tasks, open questions
- [[Daily/2026-04-13|Today's Daily Note]]
- [[Knowledge/Architecture]] — Pipeline architecture and component map
- [[Knowledge/Codebase-Overview]] — Repo structure and key files
- [[Knowledge/Source-Inventory]] — Imported docs and data sources
- [[Knowledge/Key-Lessons]] — Critical lessons learned (metric pitfalls, training failures)
- [[Codebase/00-Codebase-Map|Codebase Graph]] — Wikilinked module-by-module codebase map (excludes mamba, LoRA/DoRA adapters, T0/T1 ablations)

### Experiments
- [[Experiments/MMGD-Cut-Ablation]] — **Novel model: 45.86% mIoU on COCO (in progress, SSD-1B pending)**
- [[Experiments/Instance-Ablation-Study]] — 239-config sweep, Mumford-Shah wins (+19.9%)
- [[Experiments/Depth-Model-Ablation]] — DA3 > DA2-L > SPIdepth on Cityscapes
- [[Experiments/COCO-Semantic-Ablation]] — K-means overclustering dominates on COCO
- [[Experiments/UNet-Decoder-Ablation]] — P2-B attention = PQ 28.00 (best semantics)
- [[Experiments/NeurIPS-Rebuttal-Ablations]] — **7 rebuttal experiments for NeurIPS review (P0-A backbone control, P0-B seed, P1-A/B sweep, P2-A/C/E)**

### Results
- [[Results/Reports/Best-Results-Summary]] — Current best numbers across all pipelines
- `reports/` directory — 36 detailed experiment reports

## Current State (2026-04-24)

### Best Overall: DINOv3 Stage-3 (Cityscapes, step 8000)
| Metric | Value | Baseline (CUPS) | Delta |
|--------|-------|-----------------|-------|
| PQ (27-class) | **32.76%** | 27.8% | **+5.0** |
| PQ_things | **34.13%** | 17.7% | **+16.4** |
| PQ_stuff | 31.95% | 35.1% | -3.1 |

### Active Work (2026-04-24)
- **NeurIPS advisor meeting: adapter strategy pivot** — DoRA/LoRA restricted to Stage 1 (pseudo-label generation) only. Remove from Stage 2/3 where full backprop already adapts the model. See [[Daily/2026-04-24]] and [[Knowledge/Adapter-Strategy]].
- **Cross-dataset zero-shot evaluation complete** — Mapillary PQ=39.19% (+3.36), KITTI 34.85%, COCO 7.83% (class mismatch). See `reports/cross_dataset_evaluation_report.md`.
- **Silent LoRA drop bug fixed (C213)** — `generate_semantic_pseudolabels_adapted.py` now correctly preserves adapter weights. Prior adapter metrics may be invalid.
- **BMVC 2026 paper draft** — LaTeX at `paper_bmvc2026/main.tex` (background priority behind NeurIPS pivot).

### Key Papers
- [[Papers/CUPS-2025]] — Training protocol (Stage-2 + Stage-3), baseline PQ=27.8%
- [[Papers/CAUSE-2024]] — Semantic pseudo-labels (k=80 CAUSE-TR features)
- [[Papers/DINOv3-2025]] — Primary backbone (ViT-B/16 → ViT-L/16), our best: PQ=30.78%
- [[Papers/DINOv2-2024]] — Original CUPS backbone, used by CAUSE-TR
- [[Papers/DepthAnythingV3]] — Best instance pseudo-labels, PQ_things=20.90%
- [[Papers/SPIdepth]] — Baseline instance depth model, PQ_things=19.41%
- [[Papers/Optical-Flow/Optical-Flow-SMURF-Lineage]] — SMURF and post-SMURF optical-flow graph for CUPS-style motion pseudo-labels

### Writing — BMVC 2026 Paper
- [[Writing/BMVC2026-Paper-Overview]] — Title, abstract, contributions, headline numbers
- [[Writing/BMVC2026-Methodology]] — Method sections §3.1-3.5 (semantic, instance, assembly, training)
- [[Writing/BMVC2026-Experiments]] — SOTA table, 4 ablations, cross-dataset, oracle analysis
- [[Writing/BMVC2026-Related-Work]] — 5 subsections, citation inventory
- [[Writing/BMVC2026-Supplementary]] — Appendices A-D (overclustering, depth, per-class, self-training)
- [[Writing/BMVC2026-Figures]] — 5 figures inventory with files and status
- [[Writing/BMVC2026-Reviewer-Response]] — E1 control experiment, reviewer concerns, decision matrix
- [[Writing/BMVC2026-Mock-Review-Summary]] — 3 simulated brutal BMVC reviewers (mean 5.3/10, borderline)
  - [[Writing/BMVC2026-Reviewer-1]] — R1: novelty, DINOv3 confound, single seed (5/10)
  - [[Writing/BMVC2026-Reviewer-2]] — R2: self-training scaling, cross-dataset, stuff gap (6/10)
  - [[Writing/BMVC2026-Reviewer-3]] — R3: page limit, person/car weakness, narrow eval (5/10)
- [[Writing/paper-discussion-2026-04-06]] — Historical writing discussion log

### Literature Map
- [[Maps/literature]] — Paper relationship canvas
- [[Maps/smurf-optical-flow]] — SMURF/post-SMURF optical-flow canvas

## Folder Layout
- `Knowledge/` — Architecture, lessons, codebase reference
- `Codebase/` — Module-by-module codebase graph (graph-view target)
- `Papers/` — Related work, reading notes
- `Maps/` — Canvas diagrams (literature graph)
- `Experiments/` — Experiment design and results
- `Results/Reports/` — Detailed result summaries
- `Writing/` — Paper drafts, narratives, BMVC 2026 notes
- `Daily/` — Daily research logs

## Codebase Graph (entry points)
- [[Codebase/00-Codebase-Map]] — start here
- JAX: [[Codebase/JAX-mbps]] · [[Codebase/JAX-mbps-Models]] · [[Codebase/JAX-mbps-Training]] · [[Codebase/JAX-mbps-Losses]] · [[Codebase/JAX-mbps-Data]] · [[Codebase/JAX-mbps-Evaluation]]
- PyTorch: [[Codebase/PyTorch-Overview]] · [[Codebase/PyTorch-Models]] · [[Codebase/PyTorch-Mask2Former]] · [[Codebase/PyTorch-Training]] · [[Codebase/PyTorch-Losses]] · [[Codebase/PyTorch-Data]] · [[Codebase/PyTorch-Evaluation]]
- Pipelines: [[Codebase/Pseudo-Label-Pipeline]] · [[Codebase/Generators-Semantic]] · [[Codebase/Generators-Instance]] · [[Codebase/Refinement-SIMCF]] · [[Codebase/Scripts-Extract]]
- Architectures: [[Codebase/RefineNet-Family]] · [[Codebase/Panoptic-DeepLab]] · [[Codebase/Instance-Methods]]
- Orchestration: [[Codebase/Scripts-Orchestration]] · [[Codebase/Configs]] · [[Codebase/Datasets]]
- References: [[Codebase/Refs-CUPS]] · [[Codebase/Refs-Backbones]] · [[Codebase/Refs-Depth]] · [[Codebase/Refs-Semantic]] · [[Codebase/Refs-Instance]] · [[Codebase/Refs-Other]]
- Meta: [[Codebase/Algorithms]] · [[Codebase/Docs-Index]] · [[Codebase/Guidelines-Index]] · [[Codebase/Reports-Index]] · [[Codebase/Project-Meta]] · [[Codebase/Evaluation-Scripts]]
