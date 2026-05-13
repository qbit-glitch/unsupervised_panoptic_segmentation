---
type: codebase-module
title: Project Meta (CLAUDE.md, AGENTS.md, root markdowns)
project: mbps-panoptic-segmentation
language: en
tags: [codebase, meta, claude]
paths:
  - CLAUDE.md
  - AGENTS.md
  - core_algorithms.md
  - cascade_stage_adaptation.md
  - training_strategies_dead_classes_report.md
related:
  - "[[00-Codebase-Map]]"
  - "[[Algorithms]]"
  - "[[Reports-Index]]"
---

# Project Meta

Top-level markdowns that aren't directly code or reports.

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Claude-Code project memory: architecture, key paths, GCS layout, TPU quotas, ablation matrix, troubleshooting. |
| `AGENTS.md` | Project agent specifications and CLI tool definitions. |
| `core_algorithms.md` | CutS3D pseudocode (CLRS-style) — see [[Algorithms]]. |
| `cascade_stage_adaptation.md` | CUPS dissection: PQ contribution per stage — see [[Refs-CUPS]] and [[Algorithms]]. |
| `training_strategies_dead_classes_report.md` | 5 novel training strategies for the 5 dead classes. |

These files are the durable "what / why / where" of the project. The Codebase notes in this folder are scoped to non-adapter, non-mamba code; `CLAUDE.md` itself contains the full picture (including the excluded modules).
