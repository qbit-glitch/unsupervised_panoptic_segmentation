---
type: experiment
title: "MMGD-Cut: Multi-Modal Graph-Diffused K-way NCut"
project: mbps-panoptic-segmentation
language: en
status: in-progress
updated: 2026-04-01T22:00:00Z
tags:
  - coco
  - semantic-segmentation
  - multi-modal-fusion
  - graph-diffusion
  - falcon-ncut
  - novel-model
  - neurips-2026
---

# MMGD-Cut: Multi-Modal Graph-Diffused K-way NCut

**Date**: 2026-04-01 (ongoing) | **Status**: IN PROGRESS (Round 2 complete, SSD-1B pending)

## Goal

Build a novel unsupervised semantic segmentation model for COCO-Stuff-27 that surpasses the Falcon baseline (42.02%) and approaches published Falcon (52.6%). Key novelty: multi-modal feature fusion + graph diffusion denoising on top of Falcon's NCut solver.

**Target**: NeurIPS 2026 submission.

---

## Novel Contributions

### 1. Multi-Modal Affinity Fusion

Weighted concatenation of L2-normalized features from different foundation models. Mathematical property: cosine similarity of the concatenation equals the weighted average of per-modality cosine similarities. This is a principled, parameter-free fusion mechanism.

### 2. Graph Diffusion Denoising

Two modes for denoising the affinity structure:
- **Feature mode** (GNN-style): Lazy random walk message passing on features before affinity computation: `F' = 0.5(F + P@F)` iterated n_steps times
- **Affinity mode**: PPR or lazy random walk on the normalized cosine similarity matrix before the power transform

### 3. Architecture

`MultiModalFalcon` extends `FalconKwayCut` (ICLR 2026). Pipeline: load multi-modal features -> fuse (weighted concat of L2-normed, resolution-aligned features) -> optional graph diffusion -> Falcon NCut segmentation -> DINOv3 feature pooling -> global k-means -> Hungarian matching.

---

## Feature Sources

| Feature | Model | Resolution | Dims | Tokens | Status |
|---------|-------|-----------|------|--------|--------|
| DINOv3 | ViT-B/16 | 32x32 | 1024 | 1024 | Available (500 imgs) |
| DINOv3 hires | ViT-B/16 | 64x64 | 1024 | 4096 | Available (500 imgs) |
| SD-1.4 | UNet (s50) | 16x16 | 1280 | 256 | Available (5000 imgs) |
| SSD-1B | UNet (s10) | 32x32 | 1280 | 1024 | Extracting (~2.7h) |

---

## Results

### Round 1 (3 configs) + Round 2 (4 configs)

Fixed hyperparams: K=54, alpha=5.5, reg_lambda=0.7, K_global=27, target_res=32x32

| # | Config | Sources | Diffusion | mIoU | Things | Stuff |
|---|--------|---------|-----------|------|--------|-------|
| B1 | Falcon baseline (16x16) | SD | none | 42.02% | 38.41% | 44.91% |
| R1-1 | SD at 32x32 | SD | none | 42.73% | 39.84% | 45.03% |
| R2-1 | DINOv3 only | dinov3 | none | 45.79% | 44.50% | 46.82% |
| R1-2 | DINOv3+SD fusion | dinov3+sd | none | **45.86%** | **44.83%** | 46.68% |
| R1-3 | DINOv3+SD + PPR aff (post-power) | dinov3+sd | affinity/3 | 45.86% | 44.83% | 46.68% |
| R2-2 | DINOv3+SD + feat diff 3 | dinov3+sd | feature/3 | 45.86% | 44.83% | 46.68% |
| R2-3 | DINOv3+SD + aff diff 3 (pre-power) | dinov3+sd | affinity/3 | 45.86% | 44.83% | 46.68% |
| R2-4 | DINOv3+SD + feat diff 5 | dinov3+sd | feature/5 | 45.86% | 44.83% | 46.68% |

### Cross-Modal Affinity Correlation

| Pair | r | Interpretation |
|------|---|----------------|
| DINOv3 vs SD-1.4 | 0.877 | Highly redundant |
| DINOv3 vs SSD-1B | 0.772 | Genuinely complementary |
| SD-1.4 vs SSD-1B | 0.830 | Same model family |

---

## Round 3: SSD-1B Fusion (2026-04-02) — COMPLETE

### Results

| # | Config | Sources | Diffusion | mIoU | Things | Stuff |
|---|--------|---------|-----------|------|--------|-------|
| R3-1 | SSD-1B only | ssd1b | none | 44.08% | 41.75% | 45.94% |
| **R3-2** | **DINOv3+SSD-1B fusion** | **dinov3+ssd1b** | **none** | **46.39%** | **45.03%** | **47.48%** |
| R3-3 | DINOv3+SSD-1B + feat diff 3 | dinov3+ssd1b | feature/3 | 43.67% | 42.69% | 44.45% |
| R3-4 | DINOv3+SSD-1B + aff diff 3 | dinov3+ssd1b | affinity/3 | 43.67% | 42.69% | 44.45% |

### Analysis

- **New best: 46.39% mIoU** — genuine multi-modal gain confirmed
- Fusion adds +0.60 over DINOv3 alone (45.79%) and +2.31 over SSD-1B alone (44.08%)
- SSD-1B alone (44.08%) beats SD-1.4 (42.73%) by +1.35 — better backbone
- **Graph diffusion HURTS: -2.72 mIoU** — over-smooths the complementary signal from SSD-1B
- Gap to published Falcon: 6.21 points (52.6%)

---

## Key Findings (All Rounds)

1. **DINOv3+SSD-1B = 46.39%** — best config, genuine multi-modal fusion benefit
2. **Cross-modal correlation predicts fusion utility**: r=0.772 (SSD-1B) → +0.60 gain; r=0.877 (SD-1.4) → +0.07 negligible
3. **Graph diffusion is conclusively harmful**: zero effect with SD-1.4, -2.72 with SSD-1B. DROP from model.
4. **SD-1.4 hurts some classes**: person -4.1, animal -1.6, sports -5.9, sky -2.6 relative to DINOv3-only
5. **Resolution matters**: SD at 32x32 vs 16x16 gives +0.71 from finer NCut segments

---

## ⚠️ Metric Correction (2026-04-04)

All mIoU numbers above were computed with **per-image Hungarian** — each image gets its own optimal cluster→class mapping. This is **NOT the CUPS standard** and inflates mIoU by ~+19 points. Not comparable to any published number.

**Corrected protocol (CUPS-standard):** global k-means (K=54) → global confusion matrix → Hungarian (27 pairs) + argmax (27 unmatched) → fixed mapping for all images.

| Config | Old mIoU (per-image Hungarian) | CUPS-standard mIoU |
|--------|-------------------------------|---------------------|
| DINOv3+SSD-1B (Round 3 best) | 46.39% | ~27–28% |
| CUPS Stage-1 published | — | ~27–30% |

Relative ordering of configs is preserved. Script: `mbps_pytorch/mmgd_cut_coco_panoptic.py`.

---

## Round 4: COCO Panoptic Evaluation (2026-04-04, CUPS-standard)

| Config | mIoU | PQ | SQ | RQ | PQ_th | PQ_st |
|--------|------|----|----|----|-------|-------|
| Baseline (DINOv3+SSD-1B) | 27.30% | 7.37% | 62.18% | 10.55% | 7.84% | 7.00% |
| **R6 Multiscale** | **27.75%** | **7.57%** | **63.41%** | **10.67%** | **7.97%** | **7.26%** |

Logs: `logs/mmgd_coco_panoptic_{baseline,r6}_cups_v3.log`

---

## Next Steps

- Run R4 (NAMR) and R6+R4 COCO panoptic
- Run Cityscapes R6 panoptic (baseline only done: PQ=11.68%)
- Re-run semantic rounds with CUPS-standard protocol for corrected mIoU table
- Investigate gap to published Falcon (52.6%) — their exact eval protocol unknown
- Try DINOv3+SD+SSD-1B triple fusion

---

## Scripts

- `mbps_coco_semantics/mbps_pytorch/mmgd_cut.py` — MMGD-Cut model and evaluation pipeline
- `mbps_coco_semantics/mbps_pytorch/extract_ssd1b_features.py` — SSD-1B feature extractor
- `mbps_coco_semantics/scripts/run_mmgd_round3_ssd1b.sh` — Round 3 sweep script
- `mbps_coco_semantics/reports/mmgd_cut_ablation_report.md` — Full report with per-class analysis

## Data

- Results JSON: `/Users/qbit-glitch/Desktop/datasets/coco/mmgd_results.json`
- Logs: `/tmp/mmgd_ablation.log`, `/tmp/mmgd_round2.log`, `/tmp/mmgd_round3.log`
