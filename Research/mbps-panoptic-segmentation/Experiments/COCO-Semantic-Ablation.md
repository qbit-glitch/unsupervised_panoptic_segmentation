---
type: experiment
title: COCO Semantic Pseudo-Label Ablation
project: mbps-panoptic-segmentation
language: en
status: complete
updated: 2026-03-30T10:00:00Z
tags:
  - coco
  - pseudo-labels
  - semantic-segmentation
  - sam
  - spectral
---

# COCO Semantic Pseudo-Label Ablation

**Date**: 2026-03-28 to 2026-03-30 | **Status**: COMPLETE (2 phases)

## Goal
Improve COCO-Stuff-27 pseudo-semantic quality from mIoU=18.3% baseline. Fully unsupervised — no GT labels from COCO. Hungarian matching against GT only for evaluation.

---

## Phase 1: Initial Ablation (2026-03-28)

### Path 1: Higher Resolution Features (64x64)
Extracted 500 images at 1024x1024 (2.62s/image). K-means sweep at k=54,80,150,300,500,1000.
**Result**: 64x64 consistently WORSE than 32x32 at all K values (e.g., k=300: 23.0% vs 27.5%). ViT-L/16 trained at 518px — 1024px extrapolation degrades features.

### Path 2: CAUSE Feature-Mode + Image-Mode
- Feature-mode (pre-extracted, 40 epochs): mIoU=4.4%
- Image-mode (live backbone, RandomResizedCrop, 5000 images): mIoU=4.6%
- Real augmentation did NOT help — 20/27 classes at 0.0% IoU. Severe cluster collapse.

### Path 3: ClusterLookup + STEGO
Lightweight projector + learnable cluster centers with STEGO correlation loss.
**Result**: mIoU=4.6% at k=80. Feature-grid random crops are not real augmentation.

### Phase 1 Conclusion
All learned methods FAILED. K-means overclustering was the only viable method.

---

## Phase 2: Novel Approaches Ablation (2026-03-30)

Informed by EAGLE (CVPR'24), DeepCut++ ('25), NeCo (ICLR'25), Franca ('25), SAM+CLIP ('25).
All code on worktree branch `novel-semantic-ablation` at `mbps_novel_semantics/`.

### Approach 1: Spectral Enrichment + Hierarchical NCut Merge

Per-image graph Laplacian eigenvectors (boundary/manifold info) appended to DINOv3 features, then global k-means. Optionally merge overclusters via NCut or Ward.

| Config | K | mIoU | vs baseline |
|--------|---|------|-------------|
| Feature-only baseline (alpha=1.0, eig=0) | 300 | 26.7% | — |
| **+ 20 eigenvectors, no merge** | 300 | **28.6%** | **+1.9%** |
| + 20 eigenvectors, color mix alpha=0.7 | 300 | 28.3% | +1.6% |
| + 50 eigenvectors | 300 | 25.9% | -0.8% |
| + NCut merge to 27 | 300 | 12.2% | -14.5% |
| + Ward merge to 27 | 300 | 16.5% | -10.2% |
| + 20 eigenvectors, no merge | 500 | 29.6% | +0.0% vs k500 |
| + 20 eigenvectors, no merge | 1000 | 33.5% | +1.6% vs k1000 |

**Verdict**: Modest +1.4-1.9% at K=300, doesn't scale with K. Merge kills performance — use Hungarian matching only.

### Approach 2: SAM Superpixel Consensus Clustering — WINNER

SAM ViT-B automatic mask generation (32 points/side, IoU>0.86) produces boundary-aware segments. DINOv3 features L2-normalized and mean-pooled per segment, then global k-means on segment embeddings. Uncovered pixels filled via nearest-neighbor.

| K | K-means | Spectral | **SAM** | SAM vs k-means |
|---|---------|----------|---------|----------------|
| 54 | 15.4% | — | 17.0% | **+1.6%** |
| 300 | 27.2% | 28.6% | **29.9%** | **+2.7%** |
| 500 | 29.6% | 29.6% | **31.2%** | **+1.6%** |
| 1000 | 31.9% | 33.5% | **34.4%** | **+2.5%** |

**Best: SAM K=1000 = 34.4% mIoU** (Things 41.1%, Stuff 29.1%)
SAM provides the most consistent improvement at ALL K values.

### Approach 3: Matryoshka Multi-Granularity Clustering — FAILED

SwAV-style EMA teacher-student with Sinkhorn-Knopp equipartition. Multi-head projector at K=27,54,150.

- v1 (no augmentation): 1.4% mIoU — degenerate (loss stuck at 1.09)
- v2 (spatial crop + Gaussian noise): 2.0% mIoU — still degenerate
- Confirms: self-distillation on pre-extracted features has no learning signal without real image augmentation.

---

## Complete Leaderboard (501 val images, Hungarian matching)

| Rank | Method | K | mIoU | Things | Stuff |
|------|--------|---|------|--------|-------|
| 1 | k-means | 3000 | **38.4%** | 44.1% | 33.7% |
| 2 | k-means | 2000 | 35.6% | 42.1% | 30.4% |
| 3 | **SAM consensus** | 1000 | **34.4%** | 41.1% | 29.1% |
| 4 | Spectral enrichment | 1000 | 33.5% | 41.6% | 26.9% |
| 5 | k-means | 1000 | 31.9% | 39.6% | 25.9% |
| 6 | **SAM consensus** | 500 | 31.2% | 36.0% | 27.3% |
| 7 | **SAM consensus** | 300 | 29.9% | 34.7% | 26.0% |
| 8 | Spectral enrichment | 300 | 28.6% | 37.6% | 21.3% |
| 9 | k-means | 300 | 27.2% | 35.0% | 20.9% |

## Key Findings

1. **Overclustering is the biggest lever**: k=300 to k=3000 gives +11.2% mIoU
2. **SAM boundaries provide the most consistent secondary improvement**: +1.6 to +2.7% at matched K
3. **Spectral enrichment helps modestly**: +1.4-1.9% at K=300, diminishes at higher K
4. **All learned methods STILL fail** on pre-extracted features: Matryoshka (2.0%), STEGO (4.6%), CAUSE image (4.6%), CAUSE feature (4.4%)
5. **SAM improves Stuff more than Things**: SAM K=300 Stuff=26.0% vs k-means 20.9% (+5.1%), Things=34.7% vs 35.0% (-0.3%)

## Scripts

### Phase 1
- `mbps_pytorch/generate_coco_pseudo_semantics.py` — k-means baseline
- `mbps_pytorch/extract_dinov3_features_coco_hires.py` — 64x64 feature extraction
- `mbps_pytorch/train_cause_coco.py` — CAUSE feature/image mode
- `mbps_pytorch/train_coco_cluster_lookup.py` — ClusterLookup + STEGO

### Phase 2 (worktree: `novel-semantic-ablation`)
- `mbps_pytorch/spectral_pseudo_semantics.py` — Spectral enrichment + NCut merge
- `mbps_pytorch/sam_consensus_pseudo_semantics.py` — SAM superpixel consensus
- `mbps_pytorch/matryoshka_pseudo_semantics.py` — Matryoshka self-distillation
- `mbps_pytorch/evaluate_novel_ablation.py` — Unified evaluation + comparison

### Data
- SAM segments: `coco/sam_segments/val2017/` (500 .npz files)
- Full results JSON: `coco/novel_ablation_results.json`
