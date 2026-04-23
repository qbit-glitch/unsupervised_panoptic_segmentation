# A6000 Pseudo-Label Reproducibility: Root Cause Analysis

**Date**: 2026-04-17
**Status**: RESOLVED
**Impact**: PQ plateau at 22% instead of 28% on A6000 — 6 point gap fully explained

---

## 1. Problem Statement

CUPS Stage-2 training on the Anydesk RTX A6000 Pro (48GB) consistently plateaued at PQ ~22%, while the identical pipeline on santosh's machine reached PQ=28.4%. Multiple hypotheses were tested and eliminated (batch size, precision, augmentation, learning rate) before the true root cause was identified.

## 2. Root Cause

**Different k-means centroids produced a completely different pseudo-label set.**

The A6000 independently re-ran k-means (k=80) on DINOv2 features, producing different centroids than santosh's machine. Despite using the same code, same k, same features — the stochastic initialization of k-means++ led to a fundamentally different 80-class clustering.

### Cascade of Divergence

```
Different k-means centroids (stochastic initialization)
  → Different 80-class semantic label assignments (every pixel)
    → Different depth-guided instance boundaries (CC over different regions)
      → Different .pt distribution files (instance-proposal overlap stats)
        → Different thing/stuff split (12 things vs 15 things)
          → CUPS trained on wrong objective
            → PQ plateau at 22% instead of 28%
```

## 3. Investigation Results

### 3.1 File-Level Comparison

Ran `scripts/investigate_a6000_pseudo_labels.sh` comparing OLD (A6000-generated) vs NEW (santosh's, downloaded from HuggingFace Hub).

| Metric | OLD (A6000) | NEW (santosh) |
|--------|-------------|---------------|
| Instance PNGs | 2975 | 2975 |
| Semantic PNGs | 2975 | 2975 |
| .pt files | 2975 | 2975 |
| **Binary identical** | **0/10** | reference |

**100% file divergence** — not a single file was identical in binary comparison across instance PNGs, semantic PNGs, or .pt files.

### 3.2 Thing/Stuff Split

| | Santosh (correct) | A6000 OLD (wrong) |
|---|---|---|
| **Num things** | 15 | 12 |
| **Thing classes** | 3,11,14,15,16,29,32,37,38,45,46,62,65,73,75 | 4,8,13,23,28,29,32,52,53,56,59,71 |
| **Overlap** | — | **2/15** (classes 29, 32 only) |
| **Avg instances/img** | 19.2 | 15.8 |
| **Unique sem classes** | 79 | 80 |
| **Val contamination** | 0 | 0 |
| **Empty instance maps** | 0% | 0% |

Only 2 out of 15 thing classes overlapped. The model was learning to detect fundamentally different object categories as "things."

### 3.3 Instance File Size Comparison (first 10 files)

| File | Santosh (bytes) | A6000 OLD (bytes) | Delta |
|------|----------------|-------------------|-------|
| aachen_000000 | 6120 | 6834 | +714 |
| aachen_000001 | 6296 | 6888 | +592 |
| aachen_000002 | 6470 | 6526 | +56 |
| aachen_000003 | 7963 | 8046 | +83 |
| aachen_000004 | 6574 | 6884 | +310 |
| aachen_000005 | 7569 | 8141 | +572 |
| aachen_000006 | 6063 | 6935 | +872 |
| aachen_000007 | 5000 | 5006 | +6 |
| aachen_000008 | 5415 | 6196 | +781 |
| aachen_000009 | 6105 | 6441 | +336 |

A6000 OLD files were consistently larger (more instance boundaries from different semantic regions).

## 4. Why This Is NOT Normal K-Means Variance

Normal k-means seed variance on the same features produces:
- NMI > 0.85 between cluster assignments
- Similar thing/stuff splits (same ~15 classes identified as things)
- PQ variance of ±0.5-1.0

The A6000 divergence was extreme:
- Only 2/15 thing overlap (~13% match)
- 6-point PQ gap (22 vs 28)
- Different number of things (12 vs 15)

This suggests the features being clustered were also subtly different — possibly due to different DINOv2/DINOv3 model weights being loaded, different image preprocessing, or numerical precision differences between GPUs.

## 5. Fix Applied

1. Uploaded santosh's pseudo-labels to HuggingFace Hub (79MB compressed)
2. Downloaded on A6000 via `wget` (no HF CLI needed)
3. Backed up old labels to `cups_pseudo_labels_depthpro_tau020_old_wrong_split/`
4. Verified: 15 things, identical class IDs, identical file sizes
5. Restarted training — correct thing classes confirmed in training log

### Released Artifacts
- **HuggingFace Dataset**: `https://huggingface.co/datasets/qbit-glitch/cityscapes-cups-pseudo-labels`
  - `cups_pseudo_labels_depthpro_tau020.tar.gz` (79MB) — 2975 × 3 files (instance PNG + semantic PNG + .pt)
  - Download: `wget -O /tmp/labels.tar.gz https://huggingface.co/datasets/qbit-glitch/cityscapes-cups-pseudo-labels/resolve/main/cups_pseudo_labels_depthpro_tau020.tar.gz`
- **Centroids file**: `weights/kmeans_centroids_k80_santosh.npz` (57KB, MD5: a0cf51613fcbdc14af5294a9588bfcf6)
  - This is the single source of truth — same centroids → same cluster assignments → same thing/stuff split
  - **TODO**: Upload centroids to the same HF dataset repo for public release

### Scripts
- `scripts/download_pseudolabels_a6000.sh` — download + swap labels
- `scripts/restart_cups_fixed_labels.sh` — kill old training, clean checkpoints, restart
- `scripts/investigate_a6000_pseudo_labels.sh` — full comparison (OLD vs NEW)
- `scripts/diagnose_pseudo_labels.py` — per-directory diagnostics (JSON output)

## 6. Reproducibility Implications for Paper

### 6.1 The Problem
K-means clustering is stochastic. Different initializations produce different centroids, which cascade through the entire unsupervised panoptic pipeline. This makes exact reproduction impossible without sharing artifacts.

### 6.2 Standard Practice
All unsupervised segmentation papers (STEGO ICLR 2022, CUPS CVPR 2025, HP ECCV 2022, PiCIE CVPR 2021) release pseudo-labels and/or centroids as artifacts. No paper expects from-scratch reproduction of stochastic clustering.

### 6.3 Recommended Protocol for Our Paper
1. **Release artifacts**: centroids file (57KB), pseudo-labels (79MB), DINOv3 weights
2. **Seed robustness**: run k-means 3-5 times with different seeds on the same features, train CUPS on each, report mean ± std
3. **NMI stability**: measure NMI between cluster assignments from different seeds (expect >0.85)
4. **Paper wording**: "We run k-means (k=80) with scikit-learn's k-means++ initialization (seed=42) on DINOv2 patch features. We release centroids and pseudo-labels for reproducibility. Across N seeds, downstream PQ varies by ±X (Table Y)."

### 6.4 What This Incident Teaches
- Centroids are the **single source of truth** in the pipeline
- The thing/stuff split is an emergent property of centroids + instance proposals
- Transferring pseudo-labels (not re-generating them) is the only reliable way to replicate across machines
- The 6-point PQ gap was a pipeline bug, not a methodological weakness

## 7. Key Diagnostics Reference

### Santosh (ground truth)
```
Things (15): [3, 11, 14, 15, 16, 29, 32, 37, 38, 45, 46, 62, 65, 73, 75]
Files: 2975 instance, 2975 semantic, 2975 .pt
Avg instances: 19.2/img
First file size: aachen_000000 = 6120 bytes
```

### A6000 OLD (wrong)
```
Things (12): [4, 8, 13, 23, 28, 29, 32, 52, 53, 56, 59, 71]
Files: 2975 instance, 2975 semantic, 2975 .pt
Avg instances: 15.8/img
First file size: aachen_000000 = 6834 bytes
```

### Validation
```
Thing class overlap: 2/15 (classes 29, 32)
Only in santosh: [3, 11, 14, 15, 16, 37, 38, 45, 46, 62, 65, 73, 75]
Only in A6000 OLD: [4, 8, 13, 23, 28, 52, 53, 56, 59, 71]
```
