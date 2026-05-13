---
type: codebase-module
title: docs/ — Knowledge Base
project: mbps-panoptic-segmentation
language: en
tags: [codebase, docs]
paths:
  - docs/
  - docs/plans/
related:
  - "[[Guidelines-Index]]"
  - "[[Reports-Index]]"
  - "[[Pseudo-Label-Pipeline]]"
---

# docs/ — Knowledge base

Pipeline architecture, training playbooks, deployment notes, and dated execution plans.

## Pipeline & architecture

- `docs/pseudo_label_pipeline_architecture.md` — DINOv2 + DepthPro + DCFA + SIMCF-ABC end-to-end pipeline.
- `docs/cups_semantic_pseudolabel_pipeline.md` — CUPS codebase analysis for semantic pseudo-labels.
- `docs/pseudo_label_pipeline_dataflow.md` — Data-flow diagram through the pipeline.

## Training & recovery

- `docs/dead_class_recovery_roadmap.md` — strategies for the 5 dead classes.
- `docs/depth_enhanced_pseudolabels_plan.md` — depth-enhanced semantic pseudo-labels.
- `docs/seesaw_loss_finetuning_guide.md` — Seesaw loss fine-tuning for long-tail.

## Infrastructure

- `docs/tpu_baseline_training.md`
- `docs/tpu_deploy_commands.md`
- `docs/full_pipeline_prompt.md`
- `docs/nano_banana_prompt_pseudo_label_pipeline.md`

> **Excluded** from this graph: `docs/research_adapter_distillation_frozen_backbone.md` (LoRA/DoRA-tied; on disk but not surfaced here).

## docs/plans/

Dated execution plans (mobile panoptic, RepViT, hires-refinenet, depth-enhanced, cross-dataset, semantic aux loss, longtail). Excluded: any plan dated for adapter / DoRA / mamba ablations.
