# Seesaw Loss + Class-Aware Thresholds Ablation — Step 400 Results

**Date:** 2026-04-21  
**Experiment:** Stage-4 fine-tuning from Stage-3 best checkpoint (step 2200, PQ=36.41%)  
**Config:** `train_self_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh_ablation_a1_a2.yaml`  
**Checkpoint:** `best_pq_step=000400.ckpt` (1.4 GB)  
**Evaluated on:** Cityscapes validation (500 images), CPU, local machine

---

## Summary

| Metric | Stage-3 Baseline | Stage-4 Step 400 | Δ |
|:---|:---:|:---:|:---:|
| **PQ** | **36.41%** | **35.81%** | −0.60 |
| SQ | — | 65.86% | — |
| RQ | — | 43.39% | — |
| PQ_things | — | 36.93% | — |
| PQ_stuff | — | 35.11% | — |
| mIoU | — | 44.72% | — |
| Acc | — | 86.63% | — |

**Key finding:** At step 400 (early fine-tuning), PQ is slightly below the Stage-3 baseline. However, training logs showed a peak of **39.12%** at step 2600 (GPU1), suggesting significant headroom exists with longer training.

**Dead-class recovery:** 6 zero-PQ classes in baseline → **5 zero-PQ classes** at step 400.
- **Recovered:** `pole` (0% → 1.84% PQ)
- **Still dead:** `guard rail`, `tunnel`, `polegroup`, `caravan`, `trailer`

---

## Per-Class PQ (All 27 Classes)

| Class | PQ | SQ | RQ | Status |
|:---|:---:|:---:|:---:|:---|
| road | 89.11% | 94.20% | 94.61% | Strong |
| sidewalk | 63.35% | 79.33% | 79.86% | Good |
| parking | 1.04% | 74.21% | 1.40% | Weak |
| rail track | 7.63% | 69.98% | 10.91% | Weak |
| building | 82.92% | 85.89% | 96.54% | Strong |
| wall | 34.32% | 69.21% | 49.59% | Moderate |
| fence | 19.37% | 64.30% | 30.13% | Weak |
| **guard rail** | **0.00%** | **0.00%** | **0.00%** | **Dead** |
| bridge | 15.93% | 63.72% | 25.00% | Weak |
| **tunnel** | **0.00%** | **0.00%** | **0.00%** | **Dead** |
| pole | 1.84% | 73.34% | 2.51% | **Recovered** |
| **polegroup** | **0.00%** | **0.00%** | **0.00%** | **Dead** |
| traffic light | 5.68% | 59.97% | 9.47% | Weak |
| traffic sign | 35.85% | 65.93% | 54.37% | Moderate |
| vegetation | 83.92% | 85.29% | 98.39% | Strong |
| terrain | 34.19% | 73.50% | 46.51% | Moderate |
| sky | 86.65% | 90.45% | 95.81% | Strong |
| person | 16.59% | 73.54% | 22.55% | Weak |
| rider | 24.56% | 63.20% | 38.87% | Weak |
| car | 71.27% | 89.48% | 79.65% | Strong |
| truck | 64.93% | 90.06% | 72.09% | Strong |
| bus | 78.69% | 91.80% | 85.71% | Strong |
| **caravan** | **0.00%** | **0.00%** | **0.00%** | **Dead** |
| **trailer** | **0.00%** | **0.00%** | **0.00%** | **Dead** |
| train | 71.01% | 87.15% | 81.48% | Strong |
| motorcycle | 0.11% | 87.79% | 0.13% | Near-dead |
| bicycle | 42.15% | 80.09% | 52.63% | Moderate |

---

## Techniques Applied

### A1: Seesaw Loss
- **Purpose:** Mitigate long-tail class collapse by rebalancing gradients
- **Config:** `USE_SEESAW_LOSS=True`, `SEESAW_P=0.8`, `SEESAW_Q=2.0`
- **Effect:** Increased gradient weight for rare classes (pole, guard rail, etc.)

### A2: Class-Aware Pseudo-Label Thresholds
- **Purpose:** Make pseudo-label filtering more permissive for rare classes
- **Config:** `CLASS_THRESHOLD_ALPHA=0.3`
- **Effect:** Lower thresholds for rare classes → more training samples

### Existing: Copy-Paste Augmentation
- **Config:** `COPY_PASTE=True`, `MAX_NUM_PASTED_OBJECTS=3`, `CONFIDENCE=0.75`
- **Effect:** Increases exposure to rare-class instances

---

## Critical Finding: DDP Validation Discrepancy

During training, `pq_val` was logged with `rank_zero_only=True, sync_dist=False`. This means:
- Only GPU 0's PQ (computed on ~50% of validation data) drove checkpoint selection
- GPU 1's scores were ignored
- The "best" checkpoint may be suboptimal

**Validation PQ per GPU at step 2600:**
- GPU 0: 37.00%
- GPU 1: **39.12%** (ignored by checkpoint callback)

**Recommendation:** Fix by aggregating predictions across GPUs before computing PQ, or use `sync_dist=True` for metric logging.

---

## Proposed Next Steps (Stage-2 Technique Transfer)

The following Stage-2 (Mask2Former) techniques can be ported to Stage-3 (PanopticFPN):

| Technique | Applicable? | Expected Impact |
|:---|:---:|:---|
| **EMA** (DECAY=0.9998) | ✅ Yes | Smoother convergence, better rare-class stability |
| **SWA** (last 25% checkpoints) | ✅ Yes | Improved generalization |
| **LSJ** (Large Scale Jittering) | ✅ Yes | Better rare-class augmentation |
| **Color Jitter** | ✅ Yes | Photometric diversity |
| **Dense CRF** (post-processing) | ✅ Yes | Refine boundaries for thin structures |
| **Longer Schedule** (>4k steps) | ✅ Yes | Allow more time for rare-class learning |
| More Queries | ❌ No | Mask2Former-specific |
| DropPath | ❌ No | Mask2Former-specific |
| Larger Crop | ❌ No | Memory-limited on 2× 1080 Ti |

**Priority combination:** EMA + LSJ + Color Jitter + Longer Schedule + Dense CRF
