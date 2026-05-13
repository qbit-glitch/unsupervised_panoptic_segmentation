---
type: writing
title: "BMVC 2026 — Experiments (§4)"
project: mbps-panoptic-segmentation
status: active
venue: BMVC 2026
tags: [paper-writing, bmvc-2026, experiments, results]
updated: 2026-04-13
---

# Experiments (§4)

## Setup

- **Dataset**: Cityscapes val (500 images, 1024x2048, 19 classes: 8 thing + 11 stuff)
- **Metric**: 27-class CAUSE + Hungarian evaluation protocol (identical to CUPS)
- **PQ definition**: PQ = (sum TP IoU) / (|TP| + 0.5|FP| + 0.5|FN|) = SQ x RQ
- **Backbone**: DINOv3 ViT-B/16 (`facebook/dinov3-vitb16-pretrain-lvd1689m`)
- **Hardware**: 2x GTX 1080 Ti (11GB), PyTorch 2.1.2, Detectron2 0.6
- **Seed**: 42
- **Inference speed**: ~2-3 FPS on single GTX 1080 Ti at 640x1280

---

## Table 1: Comparison with State of the Art

| Method | Data | PQ | PQ_th | PQ_st | SQ |
|--------|------|-----|-------|-------|-----|
| DepthG + CutLER | CS+IN | 16.1 | 3.0 | 25.7 | 45.4 |
| U2Seg | COCO+IN | 18.4 | 10.2 | 24.3 | 55.8 |
| CUPS (dagger) | CS (stereo) | 27.8 | 17.7 | 35.1 | 57.4 |
| **Ours — Stage-2** | CS (mono) | 27.87 | 23.2 | 30.6 | 57.8 |
| **Ours — Stage-3** | **CS (mono)** | **32.76** | **34.1** | 32.0 | **62.6** |

(dagger) = requires stereo video or optical flow

### Key Observations

- Stage-2 **matches** CUPS overall PQ (27.87 vs 27.8) with +5.5 PQ_th
- Stage-3 self-training adds +4.9 PQ — total +5.0 over CUPS
- PQ_stuff (32.0) is **below** CUPS (35.1) by -3.1 — attributable to 7 non-standard CAUSE classes where model achieves PQ < 1%
- On the 16 classes with PQ > 10%, average PQ is 49.0%
- **Monocular caveat**: DAv3 was pretrained with multi-view geometry. "Monocular" = deployment requirement, not foundation model pretraining.

---

## Table 2: Ablation Studies (4 sub-tables)

### (a) Pseudo-Label Quality (before network training)

| Source | PQ | PQ_th | PQ_st |
|--------|-----|-------|-------|
| CUPS Stage-1 | 26.5 | 17.7 | — |
| Ours — CC-only | 23.1 | 14.9 | 28.2 |
| Ours — SPIdepth | 26.74 | 19.41 | 32.08 |
| **Ours — DAv3** | **27.37** | **20.90** | 32.08 |

Depth-guided splitting is necessary: removing depth drops PQ_th by 4.5 points. Our best monocular config (DAv3, PQ_th=20.90) exceeds CUPS Stage-1 thing quality (17.7).

### (b) Depth Model Effect on PQ_th

| Depth Model | tau | PQ_th |
|-------------|-----|-------|
| None (CC only) | — | 14.93 |
| SPIdepth | 0.20 | 19.41 |
| DAv2 | 0.03 | 20.20 |
| **DAv3** | **0.03** | **20.90** |

Depth quality is the dominant factor. Alternative splitting algorithms (Canny, watershed, multiscale Sobel) provide no improvement.

### (c) Backbone Ablation (Stage-2 only)

| Backbone | PQ | PQ_th | PQ_st |
|----------|-----|-------|-------|
| ResNet-50 | 24.68 | 19.1 | 28.0 |
| **ViT-B/16** | **27.87** | **23.2** | **30.6** |

DINOv3 outperforms ResNet-50 by +3.19 PQ. But Stage-2 margin over CUPS is only **+0.07 PQ** (27.87 vs 27.8) — suggesting pseudo-label quality and recipe, not backbone, are primary before self-training.

> **Missing control**: Ideal experiment would train CUPS pseudo-labels on DINOv3. Requires ~324 GB Cityscapes sequence data + 4x RTX 4090 — exceeds our infra. Listed as important future work. **This is the E1 control experiment currently in progress — see [[BMVC2026-Reviewer-Response]].**

### (d) Self-Training Scaling

| Teacher | Stage-2 | Stage-3 | Delta |
|---------|---------|---------|-------|
| ResNet-50 | 24.68 | 25.93 | +1.25 |
| **ViT-B/16** | **27.87** | **32.76** | **+4.89** |

PQ_th improvement: +10.9 points for DINOv3 (23.2 → 34.1). PQ_st flat (+1.33). Self-training primarily bootstraps instance quality.

---

## Qualitative Analysis (Figure 5)

7-row progression for 3 Cityscapes val scenes:
1. RGB input
2. Monocular depth map
3. Depth edges (Sobel thresholding)
4. Semantic pseudo-labels (k=80)
5. Instance pseudo-labels (depth-guided splitting)
6. Stage-2 panoptic prediction (PQ=27.87%)
7. Stage-3 panoptic prediction (PQ=32.76%)

Stage-3 produces visibly sharper instance boundaries and recovers absent semantic classes.

---

## Oracle Analysis

7 of 27 CAUSE classes have PQ < 1% (effectively zero):
- Stuff: guard rail, tunnel, polegroup, traffic light
- Things: motorcycle, caravan, trailer

If these achieved moderate PQ=20%: overall PQ → 37.94% (+5.18).
These 7 classes account for **54.7% of total gap** to supervised upper bound (Mask2Former: PQ=62.3%).
Bottleneck is a small set of difficult classes, not the well-performing ones (road PQ=92.9%, sky PQ=85.8%).

---

## Table 3: Cross-Dataset Generalization (no fine-tuning)

| Dataset | Images | PQ | PQ_th | PQ_st | mIoU |
|---------|--------|-----|-------|-------|------|
| Cityscapes (in-domain) | 500 | 32.76 | 34.13 | 31.95 | 45.1 |
| MOTS | 2,862 | 63.38 | 30.38 | 96.37 | 91.4 |
| Mapillary Vistas v2 | 2,000 | 39.85 | 32.10 | 45.48 | 59.8 |
| KITTI | 200 | 29.32 | 24.95 | 32.05 | 44.5 |
| COCO-Stuff-27 | 5,000 | 8.05 | 7.35 | 8.61 | 15.2 |

- **Mapillary PQ=39.85% exceeds in-domain Cityscapes** — reflects 124 → 19 class mapping favoring large classes
- Transfer correlates with domain similarity: driving datasets >> diverse scenes
- COCO (8.1%) confirms driving features don't generalize to diverse scenes

---

## Conclusion Highlights

Three findings with broader implications:
1. Overclustering recovers 7 collapsed classes in frozen CAUSE-TR features
2. Monocular depth quality (not splitting algorithm) is the binding constraint on instance PLs
3. EMA self-training gains scale with teacher quality: +1.25 PQ (RN50) → +4.89 PQ (DINOv3)

## Limitations (stated in paper)

1. Depth-guided splitting cannot separate co-planar objects (person PQ=4.2%)
2. Frozen semantic pipeline cannot recover 7 non-standard CAUSE classes → -3.1 PQ_st ceiling vs CUPS
3. All results from single seed; Stage-2 margin over CUPS (+0.07 PQ) is within noise
4. Self-training analysis rests on two backbone configs only

---

## Links

- [[BMVC2026-Paper-Overview]]
- [[BMVC2026-Methodology]]
- [[BMVC2026-Supplementary]] — Detailed tables supporting these results
- [[BMVC2026-Reviewer-Response]] — E1 control experiment
- [[Results/Reports/Best-Results-Summary]]
