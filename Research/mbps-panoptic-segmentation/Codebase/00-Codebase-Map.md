---
type: codebase-index
title: Codebase Map
project: mbps-panoptic-segmentation
language: en
updated: 2026-04-28
tags: [codebase, index, graph]
exclusions:
  - mamba2 module and Mamba bridge variants
  - LoRA / DoRA adapters and adapter training scripts
  - T0 / T1 / T2 / Tn enumerated ablations
---

# Codebase Map

Hub: [[00-Hub]] · Plan: [[01-Plan]] · Existing overview: [[Codebase-Overview]]

This map indexes the **non-adapter, non-mamba** parts of the repository so that the Obsidian graph view shows how the modules connect. Each note links to the files it documents using inline backticks for paths and wikilinks for cross-references.

> **Excluded from this graph (per user request):**
> - `mbps_pytorch/mamba2/` and any `*mamba*` bridge / refiner variants
> - `mbps_pytorch/models/adapters/`, `train_*adapter*.py`, `train_*lora*.py`, `LoRA-TTT/`
> - `T0` / `T1` / `T2` / `Tn` enumerated ablations and adapter ablation reports
>
> They still exist on disk; this graph just doesn't surface them.

## Layers (top → bottom)

```
Orchestration ─┐
               ├─ TPU / multi-VM ─→ [[Scripts-Orchestration]]
               └─ Configs ──────── → [[Configs]]

Pseudo-label pipeline ── [[Pseudo-Label-Pipeline]]
   ├─ Semantic generators ── [[Generators-Semantic]]
   ├─ Instance generators ── [[Generators-Instance]]
   ├─ Refinement (SIMCF) ─── [[Refinement-SIMCF]]
   └─ Extract / utilities ── [[Scripts-Extract]]

Models (training) ──────── [[PyTorch-Models]] · [[JAX-mbps-Models]]
   ├─ RefineNet family ──── [[RefineNet-Family]]
   ├─ Panoptic DeepLab ──── [[Panoptic-DeepLab]]
   ├─ Mask2Former / ViT ─── [[PyTorch-Mask2Former]]
   └─ Instance methods ──── [[Instance-Methods]]

Training loops ─────────── [[PyTorch-Training]] · [[JAX-mbps-Training]]
Losses ─────────────────── [[PyTorch-Losses]] · [[JAX-mbps-Losses]]
Data ───────────────────── [[PyTorch-Data]] · [[JAX-mbps-Data]] · [[Datasets]]
Evaluation ─────────────── [[PyTorch-Evaluation]] · [[JAX-mbps-Evaluation]] · [[Evaluation-Scripts]]

References (third-party) ─ [[Refs-CUPS]] · [[Refs-Backbones]] · [[Refs-Depth]]
                           [[Refs-Semantic]] · [[Refs-Instance]] · [[Refs-Other]]

Docs / specs ───────────── [[Docs-Index]] · [[Guidelines-Index]] · [[Reports-Index]]
Top-level meta ─────────── [[Project-Meta]]
```

## Module index

### JAX / Flax (TPU pipeline)
- [[JAX-mbps]] — package root and entry points
- [[JAX-mbps-Models]] — `MBPSModel`, backbones, semantic / instance / classifier / merger heads
- [[JAX-mbps-Training]] — trainer, curriculum, EMA, self-training, checkpointing
- [[JAX-mbps-Losses]] — semantic / instance / bridge / consistency / PQ proxy
- [[JAX-mbps-Data]] — datasets, transforms, copy-paste, TFRecord utilities
- [[JAX-mbps-Evaluation]] — PQ, mIoU, AP, Hungarian matching

### PyTorch (local + remote GPU pipeline)
- [[PyTorch-Overview]] — directory layout for `mbps_pytorch/`
- [[PyTorch-Models]] — backbones, bridge (non-mamba), classifier, semantic, instance, merger
- [[PyTorch-Mask2Former]] — `models/mask2former/` (pixel decoder, transformer decoder, post-processor)
- [[RefineNet-Family]] — `refine_net.py`, `train_refine_net.py`, joint refine net
- [[Panoptic-DeepLab]] — `panoptic_deeplab.py` and `train_panoptic_deeplab.py`
- [[Instance-Methods]] — 15 instance-decomposition methods in `mbps_pytorch/instance_methods/`
- [[PyTorch-Training]] — trainers, curriculum, EMA, self-training, pseudo-label correction
- [[PyTorch-Losses]] — losses for refiner / instance / consistency / PQ proxy
- [[PyTorch-Data]] — panoptic dataset, refiner dataset, depth cache, transforms
- [[PyTorch-Evaluation]] — PQ, mIoU, AP, hungarian matching, visualizer

### Pseudo-label pipeline (Stage 1)
- [[Pseudo-Label-Pipeline]] — high-level dataflow
- [[Generators-Semantic]] — semantic pseudo-labels (DINOv2 / DINOv3 / CAUSE / DiffCut / Falcon / SAM consensus)
- [[Generators-Instance]] — depth-guided / spectral / DINO-cluster / CutLER / DINOSAUR / DepthPro / SPIdepth / watershed
- [[Refinement-SIMCF]] — `refine_simcf.py`, `refine_simcf_v2.py`, `refine_cn_simcf.py`
- [[Scripts-Extract]] — feature / code / surface-normal extraction utilities

### Orchestration & evaluation
- [[Scripts-Orchestration]] — `scripts/orchestrate.py`, `train.py`, `evaluate.py`, `coordinate.py`, `setup_data_pipeline.sh`
- [[Evaluation-Scripts]] — pseudo-label, cross-dataset, panoptic-combined evaluation

### Configuration & references
- [[Configs]] — YAML configs (default, dataset, ablation overrides)
- [[Refs-CUPS]] — CVPR 2025 CUPS Cascade Mask R-CNN reference
- [[Refs-Backbones]] — DINO, DINOv3, DINOSAUR, sinder
- [[Refs-Depth]] — ZoeDepth, SPIdepth, DepthAnything v3, ProDepth, adversarial-depth
- [[Refs-Semantic]] — CAUSE, DepthG, STEGO
- [[Refs-Instance]] — CutLER, CuVLER, DiffNCuts
- [[Refs-Other]] — eomt, hp, mfuser

### Datasets, docs, reports
- [[Datasets]] — Cityscapes, COCO-Stuff-27, NYU, COCONUT, KITTI, MOTS, Mapillary, BDD
- [[Algorithms]] — pointers to `core_algorithms.md` and `cascade_stage_adaptation.md`
- [[Docs-Index]] — `docs/` knowledge base
- [[Guidelines-Index]] — `guidelines/` specifications
- [[Reports-Index]] — `reports/` experiment write-ups
- [[Project-Meta]] — `CLAUDE.md`, `AGENTS.md`, top-level markdowns

## Conventions used by these notes

- **Wikilinks** for cross-references between notes; the Obsidian graph view connects them automatically.
- **Backticked paths** for files in the repo. They are not links — they're identifiers.
- **YAML frontmatter** with `type`, `tags`, `module`, and `paths` keys so Bases / queries can group notes.
- **Exclusion banners** on notes that touch the boundary of the excluded set, naming what was skipped and why.

## How to extend

When you add a new module, create a sibling note here, fill in the frontmatter, and link it from this map and from the related-module notes. The graph will pick up the new node automatically.
