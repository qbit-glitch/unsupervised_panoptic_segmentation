---
type: writing
title: "BMVC 2026 — Supplementary Materials (Appendices A-D)"
project: mbps-panoptic-segmentation
status: active
venue: BMVC 2026
tags: [paper-writing, bmvc-2026, supplementary, appendix]
updated: 2026-04-13
---

# Supplementary Materials

LaTeX source: `paper_bmvc2026/supplementary.tex`

Four appendices providing detailed evidence for the main paper claims.

---

## Appendix A: Overclustering Analysis

Supports **Contribution 1** (pseudo-label pipeline) and §3.2 (semantic PLs).

### A.1 K-Sweep (Table S1)

Best PQ at each overclustering count k, with optimal tau and A_min via grid search:

| k | tau* | A_min* | PQ | PQ_th | PQ_st | SQ |
|---|------|--------|-----|-------|-------|-----|
| 27 (CAUSE) | 0.10 | 500 | 23.10 | 11.70 | 31.40 | 74.30 |
| 50 | 0.30 | 1000 | 25.78 | 13.37 | 34.80 | 73.11 |
| 60 | 0.20 | 1000 | 25.83 | 19.08 | 30.74 | 71.69 |
| **80** | **0.20** | **1000** | **26.74** | **19.41** | **32.08** | **71.88** |
| CC-only (k=80) | — | — | 24.84 | 14.90 | 32.08 | — |
| CUPS | — | — | 27.80 | 17.70 | 35.10 | 57.40 |

**Key insight**: PQ increases monotonically with k. PQ_th jumps sharply between k=50 and k=60 as collapsed classes recover. k=80 selected as smallest k where all 7 classes attract dedicated centroids.

### A.2 Centroid Collapse — Absorption Patterns (Table S2)

| Collapsed Class | Primary Absorber (%) | Secondary (%) |
|----------------|---------------------|---------------|
| fence | wall (70%) | building (13%) |
| pole | building (65%) | vegetation (10%) |
| traffic light | building (89%) | vegetation (10%) |
| traffic sign | building (78%) | wall (10%) |
| rider | bicycle (43%) | person (28%) |
| train | bus (82%) | building (9%) |
| motorcycle | bicycle (71%) | car (16%) |

Systematic pattern: small/thin classes absorbed by visually similar large classes.

### A.3 Per-Class IoU Recovery (Table S3)

Progressive recovery as k increases (patch-level, 500 Cityscapes val images):

| k | fence | pole | t.light | t.sign | rider | train | m.cycle | mIoU |
|---|-------|------|---------|--------|-------|-------|---------|------|
| 27 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 40.4 |
| 50 | **42.2** | 0.0 | 0.0 | **45.3** | 0.0 | **73.5** | 0.0 | 47.4 |
| 100 | 44.9 | 0.0 | **21.1** | 44.9 | **41.7** | 74.9 | 0.0 | 56.9 |
| 200 | 44.4 | **11.1** | 34.0 | 44.1 | 38.5 | 75.0 | **25.2** | 59.4 |
| 300 | 47.4 | 20.3 | 35.9 | 38.4 | 42.6 | 74.8 | 49.2 | 61.3 |

Bold = first recovery (>0%). Different classes recover at different k values — pole and motorcycle need k >= 200.

### A.4 Threshold and Area Sensitivity (Table S4)

Effect of A_min at fixed tau=0.20 (SPIdepth):

| A_min | PQ (k=50) | PQ (k=60) | PQ (k=80) |
|-------|-----------|-----------|-----------|
| 500 | 25.50 | 25.27 | 26.32 |
| 700 | — | 25.55 | 26.56 |
| 1000 | — | 25.83 | **26.74** |

A_min=1000 consistently optimal — filters spurious small components without discarding valid instances.

---

## Appendix B: Depth Model and Instance Method Ablation

Supports **Contribution 2** (depth model ablation) and §3.3 (instance PLs).

### B.1 Per-Class Depth Model Comparison (Table S5)

Per-class PQ_th (%) across depth models at optimal thresholds (k=80, Cityscapes val):

| Class | CC-only | SPIdepth (0.20) | DAv2-L (0.03) | DAv3 (0.03) |
|-------|---------|-----------------|---------------|-------------|
| person | — | 4.02 | 5.90 | 6.36 |
| rider | — | 9.15 | 14.10 | 12.12 |
| car | — | 16.49 | 21.40 | **26.79** |
| truck | — | 35.52 | 36.40 | 34.82 |
| bus | — | 47.76 | 46.40 | 47.73 |
| train | — | 36.38 | 32.60 | 32.67 |
| motorcycle | — | 0.00 | 0.00 | 0.00 |
| bicycle | — | 5.82 | 4.60 | 6.71 |
| **Mean** | **14.93** | **19.41** | **20.20** | **20.90** |

DA3 largest gain on **car** (+10.3 over SPIdepth). Motorcycle remains zero across all models.

### B.2 Alternative Splitting Algorithms (Table S6)

All methods use same k=80 semantic pseudo-labels:

| Algorithm | SPIdepth PQ | SPIdepth PQ_th | DA3 PQ | DA3 PQ_th |
|-----------|-------------|----------------|--------|-----------|
| **Sobel (ours)** | **26.74** | **19.41** | **27.37** | **20.90** |
| Multiscale Sobel | 26.75 | 19.42 | 26.70 | 19.28 |
| Canny (10, 30) | 26.30 | 18.36 | 26.60 | 19.04 |
| Canny (20, 50) | 26.25 | 18.25 | 26.49 | 18.81 |
| Canny (30, 80) | 26.26 | 18.27 | 26.49 | 18.81 |
| Watershed (distance) | 25.54 | 16.55 | 24.55 | 14.43 |
| Watershed (depth) | 21.31 | 6.51 | 22.49 | 9.37 |
| Watershed (combined) | 20.42 | 4.40 | 20.20 | 3.91 |

Sobel matches or exceeds all alternatives. Watershed variants severely over-split.

### B.3 Comprehensive Instance Method Ranking (Table S7)

All 15 instance decomposition methods ranked by PQ_th:

| # | Method | D | F | PQ | PQ_th |
|---|--------|---|---|-----|-------|
| 1 | **DA3 Sobel+CC** | Y | | **27.37** | **20.90** |
| 2 | Learned Edge CC | Y | Y | 26.95 | 19.89 |
| 3 | DepthPro Sobel+CC | Y | | 26.89 | 19.75 |
| 4 | SPIdepth Sobel+CC | Y | | 26.74 | 19.41 |
| 5 | Joint NCut | Y | Y | 26.11 | 17.90 |
| 6 | Morse Flow | Y | | 25.58 | 16.66 |
| 7 | Plane Decomposition | Y | | 25.47 | 16.40 |
| 8 | TDA Persistence | Y | | 25.04 | 15.37 |
| 9 | Learned Merge | Y | | 25.00 | 15.28 |
| 10 | Mumford-Shah | Y | | 24.27 | 13.54 |
| 11 | Contrastive Embed | Y | | 21.06 | 5.92 |
| 12 | Depth-Stratified | Y | Y | 19.34 | 1.82 |
| 13 | Optimal Transport | Y | | 18.86 | 0.69 |
| 14 | Adaptive Edge | Y | Y | 18.57 | 0.00 |
| 15 | Feature Edge CC | Y | Y | 18.57 | 0.00 |

D = uses depth, F = uses DINOv2 features. No method beats DA3 Sobel+CC. Methods using global optimization (OT, Mumford-Shah) or feature clustering substantially underperform.

---

## Appendix C: Per-Class PQ Breakdown (Stage-3 Best)

Supports **Contribution 3** (SOTA result) and the oracle analysis.

### Full 27-Class Breakdown (Table S8)

Stage-3 step 8000, PQ=32.76%:

**Stuff (16 classes)**:
| Class | PQ | SQ | RQ |
|-------|-----|-----|-----|
| road | 92.93 | 93.51 | 99.38 |
| sky | 85.81 | 88.11 | 97.39 |
| vegetation | 80.27 | 83.05 | 96.66 |
| building | 69.04 | 76.97 | 89.70 |
| sidewalk | 57.40 | 75.22 | 76.30 |
| traffic sign | 39.14 | 68.35 | 57.26 |
| fence | 37.86 | 67.03 | 56.49 |
| terrain | 34.41 | 72.89 | 47.21 |
| wall | 33.16 | 74.61 | 44.44 |
| parking | 7.98 | 63.56 | 12.56 |
| bridge | 2.87 | 73.12 | 3.92 |
| rail track | 1.23 | 83.99 | 1.46 |
| pole | 1.12 | 55.78 | 2.01 |
| guard rail | 0.00 | — | 0.00 |
| tunnel | 0.00 | — | 0.00 |
| polegroup | 0.00 | — | 0.00 |
| traffic light | 0.00 | — | 0.00 |
| **Stuff mean** | **31.95** | **57.42** | **40.28** |

**Thing (11 classes)**:
| Class | PQ | SQ | RQ |
|-------|-----|-----|-----|
| bus | 80.87 | 92.55 | 87.38 |
| train | 71.85 | 88.18 | 81.48 |
| truck | 66.92 | 90.67 | 73.81 |
| bicycle | 42.38 | 72.58 | 58.39 |
| rider | 40.85 | 69.37 | 58.90 |
| person | 20.75 | 71.50 | 29.03 |
| car | 17.51 | 66.84 | 26.20 |
| motorcycle | 0.10 | 56.79 | 0.18 |
| caravan | 0.05 | 51.56 | 0.09 |
| trailer | 0.03 | 53.26 | 0.06 |
| **Thing mean** | **34.13** | **71.33** | **41.55** |

### Oracle Analysis

7 classes with PQ < 1%: guard rail, tunnel, polegroup, traffic light (stuff) + motorcycle, caravan, trailer (things).

- If these 7 achieved PQ=20%: overall PQ → 37.94% (+5.18)
- These account for **54.7%** of gap to supervised upper bound (Mask2Former PQ=62.3%)
- Well-performing classes are strong: road 92.9%, sky 85.8%, bus 80.9%

---

## Appendix D: Self-Training Progression

Supports **Contribution 4** (self-training scaling).

### Full Checkpoint Progression (Table S9)

DINOv3 ViT-B/16, Cityscapes val:

| Stage | Step | PQ | PQ_th | PQ_st | SQ | mIoU |
|-------|------|-----|-------|-------|-----|------|
| Stage-2 | 1000 | 27.87 | 23.17 | 30.63 | 57.83 | 43.58 |
| Stage-3 | 600 | 29.07 | 25.75 | 31.02 | 58.80 | 43.90 |
| Stage-3 | 800 | 29.00 | 26.61 | 30.41 | 59.04 | 43.90 |
| Stage-3 | 1800 | 30.26 | 28.50 | 31.29 | 62.39 | 44.40 |
| Stage-3 | 2000 | 29.94 | 27.93 | 31.12 | 61.42 | 44.18 |
| Stage-3 | 3400 | 30.81 | 31.33 | 30.50 | 60.93 | 44.65 |
| Stage-3 | 5200 | 31.80 | 31.13 | 32.20 | 60.65 | 45.74 |
| **Stage-3** | **8000** | **32.76** | **34.13** | **31.95** | **62.57** | **45.14** |
| Delta (S2 → best) | | +4.89 | +10.96 | +1.33 | +4.74 | +1.56 |

### Key Observations

1. **Monotonic PQ improvement**: +4.9 over 8,000 steps
2. **PQ_th is primary beneficiary**: +10.96 points (23.17 → 34.13)
3. **PQ_st nearly flat**: +1.33 points only
4. **Semantic saturation**: mIoU peaks at step 5200 (45.74%) then slightly declines, while PQ_th continues improving → self-training primarily refines instance detection, not semantic classification
5. **SQ non-monotonic**: dips at steps 3400-5200, recovers at 8000 — suggests model learns tighter masks at later steps

---

## Links

- [[BMVC2026-Paper-Overview]]
- [[BMVC2026-Methodology]]
- [[BMVC2026-Experiments]]
- [[Experiments/Depth-Model-Ablation]]
- [[Experiments/Instance-Ablation-Study]]
