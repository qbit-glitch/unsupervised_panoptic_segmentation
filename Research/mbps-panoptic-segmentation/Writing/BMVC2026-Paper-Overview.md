---
type: writing
title: "BMVC 2026 Paper — Overview"
project: mbps-panoptic-segmentation
status: active
venue: BMVC 2026
tags: [paper-writing, bmvc-2026, overview]
updated: 2026-04-13
---

# BMVC 2026 Paper Overview

## Title

> **Unsupervised Panoptic Segmentation via Depth-Semantic Pseudo-Label Compositing**

Alternative titles considered:
- *Foundation Model Pseudo-Labels for Unsupervised Panoptic Segmentation*
- *Depth-Guided Instance Splitting and Semantic Overclustering for Unsupervised Panoptic Segmentation*

## Paper Type

**Empirical pipeline paper** — not a theory or architecture paper. Honest framing: careful composition of existing foundation models (CAUSE-TR, DA3, DINOv3) into a pseudo-label pipeline, trained with the CUPS protocol.

## Abstract (5-sentence structure)

1. **Achievement**: Fully unsupervised panoptic segmentation pipeline composing frozen foundation models into pseudo-label framework — surpasses CUPS (CVPR 2025) by +5.0 PQ
2. **Why hard**: Requires jointly recovering semantic categories AND individual instances from unlabeled images; prior work (CUPS) requires stereo video + optical flow
3. **How**: Two streams — semantic overclustering (CAUSE-TR k=80) + monocular depth instance splitting (DA3 Sobel+CC) → Cascade Mask R-CNN + EMA self-training
4. **Evidence**: Cityscapes val, 27-class CAUSE+Hungarian protocol (same as CUPS)
5. **Headline number**: PQ=32.76%, surpassing CUPS 27.8% by +5.0 PQ, using only monocular images

## Four Contributions

1. **Monocular pseudo-label pipeline** — CAUSE-TR k=80 + DA3 depth-guided splitting → PQ=27.37% pseudo-labels (exceeds CUPS PQ=26.5% which needs stereo+flow)
2. **Depth model ablation** — SPIdepth 19.41 → DA2 20.20 → DA3 20.90 PQ_th; splitting algorithm irrelevant — depth quality is the binding constraint
3. **SOTA unsupervised panoptic** — PQ=32.76% with DINOv3+Cascade Mask R-CNN+EMA, surpassing CUPS by +5.0 PQ, PQ_th from 17.7% to 34.1% (+16.4)
4. **Self-training scaling** — EMA gains scale with teacher quality: +1.25 PQ (ResNet-50) vs +4.89 PQ (DINOv3), thing-class quality is primary beneficiary

**Not contributions**: CUPS training recipe, DINOv3, CAUSE-TR, DA3/SPIdepth. Each is prior work, clearly credited.

## Headline Numbers

| Metric | Our Stage-2 | Our Stage-3 | CUPS (CVPR'25) | Delta |
|--------|-------------|-------------|----------------|-------|
| PQ | 27.87% | **32.76%** | 27.8% | **+5.0** |
| PQ_things | 23.2% | **34.1%** | 17.7% | **+16.4** |
| PQ_stuff | 30.6% | 32.0% | 35.1% | -3.1 |
| SQ | 57.8% | 62.6% | 57.4% | +5.2 |

- Dataset: Cityscapes val (500 images, 1024x2048)
- Metric: 27-class CAUSE + Hungarian matching (identical to CUPS paper)
- Input: monocular RGB only (CUPS requires stereo + optical flow)

## Paper Structure

| Section | Content | Pages (est.) |
|---------|---------|-------------|
| Abstract | 5-sentence formula | ~0.3 |
| 1. Introduction | Problem + contributions (4 bullets) | ~1.2 |
| 2. Related Work | 5 subsections (semantic, instance, depth, panoptic, FMs) | ~1.0 |
| 3. Method | 5 subsections (overview, semantic, instance, assembly, training) | ~2.5 |
| 4. Experiments | SOTA table + 4 ablations + cross-dataset + qualitative | ~2.5 |
| 5. Conclusion | Summary + limitations | ~0.5 |
| References | ~25 citations | ~1.0 |

## Key Figures

| # | Description | File | Status |
|---|-------------|------|--------|
| 1 | Pipeline overview + results grid | `fig1_overview_ai.png` | Done |
| 2 | Pseudo-label generation pipeline | `fig2_pl_pipeline_ai.png` | Done |
| 3 | Network training (Stage-2 + Stage-3) | `fig3_training_ai.png` | Done |
| 4 | K-sweep plot (PQ vs overclustering k) | `fig4_ksweep.pdf` | Done |
| 5 | Qualitative progression (7 rows) | `fig5_qualitative.pdf` | Done |

## Venue

**BMVC 2026** — May 2026 deadline. A-grade, CV-focused.
- Upgrade path: ACCV 2026 (~July) if COCO results materialize
- Appropriate for strong empirical pipeline papers
- Results (+5 PQ over CUPS CVPR 2025) justify A-grade submission

## Critical Blocker

**E1 Control Experiment** — both reviewers flagged that +5.0 PQ could be from backbone upgrade (ResNet-50 → DINOv3 ViT-B/16), not pseudo-label quality. See [[BMVC2026-Reviewer-Response]].

## LaTeX Source

- Main: `paper_bmvc2026/main.tex`
- Supplementary: `paper_bmvc2026/supplementary.tex`
- References: `paper_bmvc2026/references.bib`
- Figures: `paper_bmvc2026/figures/`

## Links

- [[BMVC2026-Methodology]] — Method sections §3.1-3.5
- [[BMVC2026-Experiments]] — Results and ablations
- [[BMVC2026-Related-Work]] — Literature positioning
- [[BMVC2026-Supplementary]] — Appendices A-D
- [[BMVC2026-Figures]] — Figure inventory and status
- [[BMVC2026-Reviewer-Response]] — E1 control + reviewer concerns
- [[paper-discussion-2026-04-06]] — Historical writing discussion
- [[00-Hub]]
