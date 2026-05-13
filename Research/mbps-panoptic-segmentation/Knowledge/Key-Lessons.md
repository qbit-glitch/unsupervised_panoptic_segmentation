---
type: knowledge
title: Key Lessons Learned
project: mbps-panoptic-segmentation
language: en
updated: 2026-03-29T13:40:24Z
---

# Key Lessons Learned

## Metric Pitfalls (CRITICAL)
- **CUPS uses 27-class CAUSE + Hungarian matching**, not 19-class. UNet PQ=28.00 (19-class) is NOT comparable to CUPS PQ=27.8 (27-class). On same metric: CUPS=38.59 >> UNet=28.00.
- **Merge of UNet+CUPS FAILED**: UNet doesn't know 6 non-standard CAUSE classes -> void -> PQ drops. Merged PQ=24.36 < CUPS 27.78.
- **DDP per-rank validation overestimates PQ by ~3 points** — always eval on full val set.
- **Per-image Hungarian inflates mIoU by ~19 points vs global Hungarian** (2026-04-04): `mmgd_cut.py` uses per-image Hungarian (each image gets its own optimal cluster→class mapping). This gave 46.39% but is oracle-like — NOT comparable to any published number. CUPS standard is global: pool all images → one confusion matrix → Hungarian + argmax for unmatched clusters → fixed mapping for all. Correct CUPS-standard mIoU for DINOv3+SSD-1B is ~27-28%.
- **NCut local segment IDs are NOT globally consistent**: Per-image NCut produces IDs 0…K-1 that are independent across images. Must apply global k-means on pooled DINOv3 embeddings first to get globally consistent cluster IDs before building the confusion matrix.
- **CUPS _matching_core for overclustering (K>C)**: Hungarian for the best C pairs, then argmax for the remaining K−C unmatched clusters (see `refs/cups/cups/metrics/panoptic_quality.py` lines 278–288). Never leave unmatched clusters as void — they map to their best-overlap class via argmax.

## Architecture & Training
- **Conv2d >> Mamba for small feature maps**: At 32x64, local convs beat long-range SSMs (11.5x faster, +0.53 PQ).
- **Block type >> resolution ~ capacity**: P2-A ~ P2-D (27.65 ~ 27.64) proves 3rd decoder stage gain is capacity, not resolution.
- **Universal overfitting after ep6-8**: All UNet runs decline 0.41-0.57 PQ post-peak.
- **Self-training with unsupervised teacher HURTS**: EMA teacher from mIoU=53% introduces noise (-1.12 PQ).
- **Never feed target labels as model input** — identity shortcut.

## Instance Segmentation
- **Semantics are NOT "solved"**: Instance quality is THE bottleneck (person PQ=4.2, RQ=8.8%).
- **Depth-guided instances fail for co-planar objects** — same depth = no gradient = merged blob.
- **Mumford-Shah's beta parameter dominates**: Feature weight (beta=1.0) >> depth weight (alpha). Always prioritize feature space tuning.
- **Watershed methods degenerate on monocular depth**: All 56 Morse configs identical. Monocular depth lacks rich local minima.
- **Learned instance heads degrade pseudo-labels by ~50%**: PQ_things 19.41 -> 9.79. Use pre-computed instances when they're good.

## COCO-Specific
- **K-means overclustering is extremely hard to beat** on pre-extracted ViT features.
- **Self-supervised clustering (STEGO, CAUSE) requires real image augmentation** — feature-grid crops insufficient.
- **64x64 features WORSE than 32x32**: ViT-L/16 trained at 518px, 1024px extrapolation hurts.

## Infrastructure
- **CAUSE logits .pt files have WRONG channel order** — use PNG one-hot labels.
- **BCE under fp16 autocast crashes** — use `binary_cross_entropy_with_logits` (logits input).
- **NaN at backbone unfreeze** — rebuild optimizer (stale Adam momentum explodes).
- **MPS virtual memory 434GB VSZ is normal** — actual RSS ~3GB. Use `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7`.
