---
type: experiment
title: NeurIPS Rebuttal — Training Ablation Experiments
project: mbps-panoptic-segmentation
status: active
tags: [neurips-2026, rebuttal, ablation, training]
created: 2026-04-09
updated: 2026-04-09
---

# NeurIPS Rebuttal — Training Ablation Experiments

[[00-Hub]] | [[Experiments/UNet-Decoder-Ablation]] | [[Experiments/Depth-Model-Ablation]]

> All experiments address specific reviewer weaknesses (W1-W10) from the NeurIPS 2026 review (score 3, Borderline Reject). Ordered by priority.

---

## P0-A: CUPS+DINOv3 Backbone Control (W1, W5)

**Status**: [ ] In progress — CUPS PLs must be generated, not downloaded
**Priority**: CRITICAL — reviewer's #1 concern
**Machine**: Remote 2x1080Ti (`santosh@172.17.254.146`)
**Est. time**: ~18h total (6h PL generation + 6h Stage-2 + 6h Stage-3)

**Goal**: Isolate backbone vs pseudo-label contribution. Run CUPS pseudo-labels through our DINOv3 ViT-B/16 backbone.

> **IMPORTANT**: CUPS does NOT provide pre-computed pseudo-labels for download. Their pipeline requires stereo pairs + optical flow (SMURF) + DepthG depth. We must run their `cups/pseudo_labels/gen_pseudo_labels.py` ourselves.

**Steps**:
- [ ] Download CUPS supporting checkpoints from TUdatalib:
  - `raft_smurf.pt` (optical flow)
  - `depthg.ckpt` (depth estimation)
  - `dino_RN50_pretrain_d2_format.pkl` (backbone)
- [ ] Verify Cityscapes stereo pairs available (left+right image sequences)
- [ ] Run `cups/pseudo_labels/pseudolabel_gen.sh` on remote GPU (~6h on 2 GPUs)
- [ ] Save to `datasets/cityscapes/cups_official_pseudo_labels/`
- [ ] Create config: `configs/train_cups_official_pls_dinov3.yaml`
- [ ] Train Stage-2: 8K steps, DINOv3 ViT-B/16, CUPS PLs
- [ ] Train Stage-3: 8K steps, EMA self-training
- [ ] Evaluate on Cityscapes val (27-class CAUSE+Hungarian)
- [ ] Add result to Table 1: "CUPS PLs + DINOv3 ViT-B/16"

**Alternative approach** (faster, if stereo data unavailable):
- [ ] Use CUPS final checkpoint (`cups.ckpt`) to generate predictions on train set
- [ ] Use those predictions as pseudo-labels (approximation of CUPS Stage-1 PLs)
- [ ] This avoids running their full stereo+flow pipeline

**Expected outcome**:
- If CUPS+DINOv3 < 30 PQ → our monocular PLs genuinely contribute (+3-5 PQ)
- If CUPS+DINOv3 ≥ 33 PQ → backbone is the main driver, reframe narrative

**Reviewer question answered**: Q1, Q2

---

## P0-B: Extra Seed Evaluation (W2)

**Status**: [ ] Not started
**Priority**: CRITICAL — single-seed undermines all claims
**Machine**: Remote 2x1080Ti
**Est. time**: ~12h

**Goal**: Show PQ=32.76% is reproducible, not a lucky run.

**Steps**:
- [ ] Run Stage-2 with seed=123 (seed=42 already done)
- [ ] Run Stage-3 with seed=123
- [ ] Evaluate on Cityscapes val
- [ ] Report mean ± std of 2 seeds for PQ, PQ_things, PQ_stuff

**Key check**: If mean is ~31-33 ± 1.0, the +3-5 PQ gap over CUPS is robust.

**Config change**: Only `--seed 123` in train command

---

## P1-A: Self-Training Threshold Data Points (W3)

**Status**: [ ] Partially available
**Priority**: MAJOR — N=2 is anecdotal
**Machine**: Remote 2x1080Ti
**Est. time**: ~12h per new model

**Goal**: Strengthen self-training threshold from N=2 to N=4.

**Already have**:
| Model | Stage-2 PQ | Stage-3 PQ | Delta | Status |
|-------|-----------|-----------|-------|--------|
| RepViT-M0.9 + BiFPN | 24.78 | 23.66 | -1.12 | Done |
| DINOv3 ViT-B/16 | 27.87 | 32.76 | +4.89 | Done |

**Need to add**:
| Model | Stage-2 PQ | Stage-3 PQ | Delta | Status |
|-------|-----------|-----------|-------|--------|
| DINOv2 ResNet-50 | 25.93 | ? | ? | Stage-3 ckpt exists (step 600) — check |
| DINOv2 ViT-B/14 | ? | ? | ? | Need to train |

**Steps**:
- [ ] Evaluate existing ResNet-50 Stage-3 checkpoint properly (is PQ=25.93 Stage-2 or Stage-3?)
- [ ] If ResNet-50 Stage-3 degrades: confirms threshold exists between 25-28 PQ
- [ ] Run DINOv2 ViT-B/14 Stage-2 + Stage-3 for 4th data point
- [ ] Plot: self-training delta vs Stage-2 PQ (4 points)

**Reviewer question answered**: Q3

---

## P1-B: k-Sweep Plot (Q5)

**Status**: [ ] Partially available
**Priority**: MAJOR — elevates overclustering contribution
**Machine**: Local M4 Pro
**Est. time**: ~2h

**Goal**: Proper PQ-vs-k curve showing k=80 is the sweet spot.

**Already have**:
| k | PQ (pseudo-labels) | PQ_stuff | PQ_things |
|---|-------------------|----------|-----------|
| 50 | 25.78 | 34.80 | 13.37 |
| 60 | 25.83 | 30.74 | 19.08 |
| 80 | 26.74 | 32.08 | 19.41 |
| 100 | 27.10 | 32.70 | 19.60 |

**Need**: k=20 and k=40

**Steps**:
- [ ] Run K-means with k=20 on CAUSE-TR features
- [ ] Run K-means with k=40 on CAUSE-TR features
- [ ] Generate depth-guided instances for each
- [ ] Evaluate pseudo-label PQ for k=20 and k=40
- [ ] Plot PQ vs k (6 data points)
- [ ] Also plot: per-class recovery curve (at which k does each collapsed class first appear?)
- [ ] Save figure as `figures/k_sweep_pq_vs_k.pdf`

**Reviewer question answered**: Q5

---

## P2-A: Stuff Regression Analysis (W6, Q6)

**Status**: [ ] Not started
**Priority**: MODERATE
**Machine**: Local M4 Pro
**Est. time**: ~1.5h

**Goal**: Understand and address PQ_stuff gap (32.0 vs CUPS 35.1).

**Steps**:
- [ ] Test k=120 and k=150 pseudo-labels
- [ ] Check if more clusters recover the 6 non-standard CAUSE stuff classes
- [ ] Compute per-class stuff PQ breakdown (our 19 standard classes vs CUPS)
- [ ] Write analysis paragraph for Section 4.2

**Hypothesis**: Gap comes entirely from 6 non-standard CAUSE classes that our pipeline maps to void.

**Reviewer question answered**: Q6

---

## P2-C: Mapillary Vistas Cross-Dataset Eval (W7)

**Status**: [ ] Not started
**Priority**: MODERATE
**Machine**: Local M4 or Remote
**Est. time**: ~3h

**Goal**: Add genuinely informative cross-dataset evaluation.

**Data**: Mapillary Vistas v2 available at `~/Desktop/datasets/mapillary-vistas-v2/` (2000 val images, 65 classes)

**Steps**:
- [ ] Write Mapillary evaluation script (class mapping to Cityscapes 19 classes)
- [ ] Run Stage-3 checkpoint (step 8000) on Mapillary val
- [ ] Report PQ, PQ_things, PQ_stuff, mIoU
- [ ] Add row to Table 7

**Reviewer question answered**: Partially addresses W7

---

## P2-E: Oracle Person Analysis (Q4)

**Status**: [ ] Not started
**Priority**: MODERATE
**Machine**: Local M4 Pro
**Est. time**: ~1h

**Goal**: Quantify the cost of person-class failure.

**Steps**:
- [ ] Load GT person instance masks from Cityscapes val
- [ ] Replace predicted person instances with GT person instances
- [ ] Re-evaluate PQ with oracle person
- [ ] Report: "With oracle person instances, PQ = X (+Y over current 32.76)"
- [ ] Compute: fraction of gap to supervised (62.3 PQ) attributable to person failure

**Reviewer question answered**: Q4

---

## Execution Schedule

| Day | Remote 2x1080Ti | Local M4 | Writing |
|-----|-----------------|----------|---------|
| 1 | P0-A Stage-2 (5.7h) | P1-B: k=20,40 sweep (2h) | P2-B: Fix refs (done by agent) |
| 1 cont | P0-A Stage-3 (5.7h) | P2-A: k=120 stuff test (1h) | P2-D: Soften monocular (15min) |
| 2 | P0-B: Seed 123 (12h) | P2-C: Mapillary eval (3h) | P2-E: Oracle person calc (1h) |
| 3 | P1-A: DINOv2-B (12h) | Compile results | Draft rebuttal |
| 4 | — | — | Finalize paper + rebuttal |

**Total**: ~4 days, ~36 GPU-hours

---

## Links

- [[00-Hub]]
- [[Papers/CUPS-2025]]
- [[Papers/DINOv3-2025]]
- [[Experiments/Depth-Model-Ablation]]
- [[Knowledge/Key-Lessons]]
