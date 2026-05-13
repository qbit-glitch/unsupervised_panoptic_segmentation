---
type: writing
title: "BMVC 2026 — Figures Inventory"
project: mbps-panoptic-segmentation
status: active
venue: BMVC 2026
tags: [paper-writing, bmvc-2026, figures]
updated: 2026-04-13
---

# Figures Inventory

All figures located in `paper_bmvc2026/figures/`.

---

## Figure 1 — Pipeline Overview + Results

- **File**: `fig1_overview_ai.png` (also `fig1_overview.pdf`)
- **Type**: Full-width (`\textwidth`), top of page 1
- **Content**:
  - Top: panoptic predictions (Ours Stage-3 vs CUPS pseudo-labels) on 3 Cityscapes val scenes
  - Bottom left: pseudo-label generation flow (RGB+Depth → Instance/Semantic → Panoptic PL)
  - Bottom right: panoptic network training (PLs → Loss → Network → Self-Train EMA → Output)
- **Caption**: Overview and results. PQ=32.76%, surpassing CUPS (27.8%) by +5.0 PQ using only monocular images.
- **Status**: Done (AI-generated, verified against implementation)
- **Generated**: 2026-04-09

---

## Figure 2 — Pseudo-Label Generation Pipeline

- **File**: `fig2_pl_pipeline_ai.png` (also `fig2_pl_pipeline.pdf`)
- **Type**: Full-width, references §3.2-3.4
- **Content**:
  - 1a (top): Instance PL — RGB → DAv3 depth → Sobel gradient → CC → instance masks
  - 1b (bottom): Semantic PL — RGB → DINOv2+CAUSE-TR → K-Means k=80 → semantic labels
  - 1c (right): Panoptic assembly — stuff/things classifier → align → panoptic PL (PQ=26.74%)
- **Caption**: Stage 1: Pseudo-label generation with monocular depth for instances and CAUSE-TR overclustering for semantics.
- **Status**: Done

---

## Figure 3 — Network Training Pipeline

- **File**: `fig3_training_ai.png` (also `fig3_training.pdf`)
- **Type**: Full-width, references §3.5
- **Content**:
  - Stage-2 (top): PLs from disk → Cascade Mask R-CNN (DropLoss) → PQ=27.87%
  - Stage-3 (bottom): Teacher (EMA+TTA) → online PLs → Student → EMA update loop → PQ=32.76%
- **Caption**: Network training pipeline. Stage-2 trains on pseudo-labels, Stage-3 uses EMA self-training.
- **Status**: Done

---

## Figure 4 — K-Sweep Plot

- **File**: `fig4_ksweep.pdf`
- **Type**: Single-column width (`0.55\columnwidth`), references §3.2
- **Content**: Line plot of PQ vs overclustering count k (50, 60, 80, 100)
  - PQ increases monotonically: 25.78% (k=50) → 27.10% (k=100)
  - PQ_th jumps sharply between k=50 and k=60 (collapsed classes recover)
  - Operating point: k=80
- **Caption**: Pseudo-label quality vs overclustering count k.
- **Status**: Done (matplotlib-generated)

---

## Figure 5 — Qualitative Progression

- **File**: `fig5_qualitative.pdf` (also `.png`)
- **Type**: Full-width, references §4
- **Content**: 7 rows x 3 columns (3 Cityscapes val scenes)
  - (a) RGB input
  - (b) Monocular depth map
  - (c) Depth edges (Sobel thresholding) — visualization uses SPIdepth tau=0.20; final uses DAv3 tau=0.03
  - (d) Semantic pseudo-labels (k=80)
  - (e) Instance pseudo-labels (depth-guided splitting)
  - (f) Stage-2 panoptic prediction (PQ=27.87%)
  - (g) Stage-3 panoptic prediction (PQ=32.76%)
  - Dashed line separates Phase 1 (PL generation) from Phase 2 (network training)
- **Generation script**: `paper_bmvc2026/figures/make_qualitative.py`
- **Caption**: Full pipeline qualitative progression.
- **Status**: Done

---

## Visualization Notebooks

4 notebooks in `notebooks/visualizations/` with real DINOv3 inference:
1. `01_instance_pl_pipeline.ipynb` — 13 cells, instance PL generation
2. `02_panoptic_merge.ipynb` — 10 cells, panoptic assembly
3. `03_stage2_training.ipynb` — 11 cells, Stage-2 inference
4. `04_stage3_selftrain.ipynb` — 11 cells, Stage-3 + S2 vs S3 comparison

Each saves individual sub-images to `figures_*_pipeline/` directories.

---

## Pending / Not in Paper

- Instance PL diagram (DA3 depth → Sobel → CC → masks) — could be standalone figure but covered by Fig 2
- Full pipeline overview figure (4-stage end-to-end for Figure 1) — covered by Fig 1
- Confusion matrix visualization (k=27 absorption patterns) — could strengthen App A

---

## Links

- [[BMVC2026-Paper-Overview]]
- [[BMVC2026-Methodology]]
- [[paper-discussion-2026-04-06]] — Discussion about AI-generated figures
