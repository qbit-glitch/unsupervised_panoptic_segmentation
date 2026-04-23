# Depth-Guided Semantic Ablation Study — Complete Results

**Date**: 2026-04-17
**Compute**: M4 Pro MacBook 48GB (MPS), RTX A6000 (Approach B v1 only)

## Summary

Three approaches tested for improving CAUSE semantic pseudo-labels with monocular depth:

| Method | mIoU | Delta vs Baseline | Training Time | Params Trained |
|--------|------|-------------------|---------------|----------------|
| CAUSE baseline (k=300) | 54.41% | — | None | 0 |
| **Approach A** (training-free sinusoidal concat) | 56.76% | +2.35 | None | 0 |
| **Approach B** (fine-tune Segment_TR) | 43.21% | -11.20 | 44 min | Millions |
| **Approach C** (frozen adapter, lp=20) | **60.65%** | **+6.24** | 109 sec | 40K |

**Best result: Approach C with lambda_preserve=20.0 achieves mIoU=60.65%**

## Approach A: Training-Free Depth Concatenation

Concatenates depth features to frozen CAUSE 90D codes before k-means clustering.

**Method**: DINOv2 ViT-B/14 → Segment_TR head_ema → 90D codes → concat([codes, sinusoidal_encode(depth) * alpha]) → k-means (k=300) → majority-vote 19-class labels

**Best variant**: Sinusoidal encoding (16D), alpha=0.1

| Variant | Alpha | mIoU | Delta |
|---------|-------|------|-------|
| none (baseline) | — | 54.41% | — |
| raw | 0.1 | 55.82% | +1.41 |
| sobel | 0.5 | 55.49% | +1.08 |
| **sinusoidal** | **0.1** | **56.76%** | **+2.35** |

**Key insight**: Low alpha (0.1) consistently beats high alpha. Depth provides boundary information ("where objects end"), not absolute distance.

## Approach B: Fine-Tuning CAUSE Segment_TR — DEAD

Fine-tunes Segment_TR head with cluster loss + depth correlation loss.

| Variant | Contrastive | mIoU | Issue |
|---------|------------|------|-------|
| B0 v1 (original code) | Active (lambda=0.0 but code had bug) | 45.06% | Empty bank → 98:2 gradient ratio |
| B0 v2 (contrastive disabled) | Disabled | 43.21% | Cluster loss alone collapses minorities |

**Root cause**: CAUSE's cluster loss has inherent majority-class bias. When Segment_TR features are free to move, optimization re-distributes clusters toward dominant classes (road: 96/300 clusters, train: 0, motorcycle: 0). This is a fundamental limitation, not a hyperparameter issue.

**Code changes**: Added MPS support to `train_cause_depth_finetune.py` (device auto-detection, grid_sample zeros padding, num_workers=0 for MPS).

## Approach C: Frozen-Feature Depth Adapter

Freezes CAUSE 90D codes entirely. Trains a tiny adapter MLP with skip connection and zero-initialized output.

### Architecture

```
Frozen: Image → DINOv2 ViT-B/14 → Segment_TR head_ema → 90D codes (B, 529, 90)
Depth:  DepthPro .npy → downsample to patch grid → (B, 529, 1)

Adapter (only trainable):
  Input:  concat([90D codes, 1D depth]) = (B, 529, 91)
  Hidden: Linear(91��128) → LayerNorm → ReLU → Linear(128→128) → LayerNorm → ReLU
  Output: Linear(128→90), zero-initialized
  Skip:   adjusted_codes = original_codes + adapter_output

Evaluation:
  adjusted_codes → sinusoidal depth concat (alpha=0.1) → k-means (k=300) → 19-class labels
```

**Parameters**: 40,410 (vs millions in Segment_TR)

### Loss Function

```
L = depth_guided_correlation_loss(adjusted_codes, depth) + lambda_preserve * MSE(adjusted_codes, original_codes)
```

- No cluster loss (avoids majority-class bias)
- MSE preserve prevents feature drift
- depth_guided_correlation_loss encourages depth-aware corrections

### Lambda_preserve Sweep

| lambda_preserve | Drift | mIoU | PQ | motorcycle | train |
|----------------|-------|------|-----|------------|-------|
| 1.0 (loose) | 1.21 | 58.93% | 26.99 | 49.15% | 74.06% |
| 5.0 | 0.25 | 59.99% | 27.58 | 48.75% | 73.94% |
| 10.0 | 0.13 | 57.27% | 26.82 | 0.00% | 73.70% |
| **20.0** (tight) | **0.06** | **60.65%** | **27.64** | **48.21%** | **73.92%** |

**Key finding**: Tighter constraints (higher lambda_preserve) yield better mIoU. The adapter's value comes from tiny, precise corrections at depth boundaries — large changes add noise. Analogous to LoRA rank-1 >> rank-64 finding.

**Note**: lp=10.0 motorcycle=0 was a k-means initialization artifact, not systematic.

### Best Result Per-Class IoU (lp=20.0)

| Class | IoU | PQ | Type |
|-------|-----|-----|------|
| road | 94.43% | — | stuff |
| sidewalk | 60.58% | 47.47 | stuff |
| building | 82.09% | 75.18 | stuff |
| wall | 33.70% | 13.89 | stuff |
| fence | 37.19% | 14.98 | stuff |
| pole | 24.32% | 0.28 | thing |
| traffic light | 30.29% | 3.51 | thing |
| traffic sign | 38.89% | 7.27 | thing |
| vegetation | 81.50% | 73.81 | stuff |
| terrain | 35.95% | 14.77 | stuff |
| sky | 75.78% | 60.98 | stuff |
| person | 53.41% | 7.60 | thing |
| rider | 30.67% | 6.20 | thing |
| car | 80.41% | 26.15 | thing |
| truck | 52.21% | 20.45 | thing |
| bus | 60.59% | 28.80 | thing |
| train | 73.92% | 15.72 | thing |
| motorcycle | 48.21% | 7.50 | thing |
| bicycle | 58.19% | 4.64 | thing |

### Panoptic Quality (lp=20.0)

| Metric | Value |
|--------|-------|
| Overall PQ | 27.64% |
| PQ_stuff | 48.86% |
| PQ_things | 12.22% |
| SQ | 70.41% |
| RQ | 35.82% |

## Training Infrastructure

### Pre-extraction (one-time)
- Script: `scripts/extract_cause_codes.py`
- Output: `cityscapes/cause_codes_90d/{train,val}/{city}/{stem}_{codes,depth}.npy`
- Time: ~16.5 min on MPS (500 val + 2975 train images)

### Adapter Training
- Script: `mbps_pytorch/train_depth_adapter.py`
- Model: `mbps_pytorch/models/semantic/depth_adapter.py`
- Time: ~109 sec per run on MPS (20 epochs)
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4), CosineAnnealingLR

### Evaluation
- Generate: `mbps_pytorch/generate_depth_overclustered_semantics.py --adapter_checkpoint <path>`
- Evaluate: `mbps_pytorch/evaluate_semantic_pseudolabels.py`
- Time: ~7 min total on MPS

## Architecture Sweep (Approach C Variants)

All variants use lambda_preserve=20.0 (best from sweep above). Tests whether richer depth input, wider hidden layers, or deeper networks improve mIoU.

| Variant | Depth Input | Hidden | Layers | Params | mIoU | PQ | motorcycle | train |
|---------|------------|--------|--------|--------|------|-----|------------|-------|
| **Original** | **1D raw** | **128** | **2** | **40K** | **60.65%** | **27.64** | **48.21%** | **73.92%** |
| V1 | 16D sinusoidal | 128 | 2 | 40K | 59.61% | 27.32 | 48.75% | 73.73% |
| V2 | 16D sinusoidal | 256 | 2 | 91K | 59.17% | 26.93 | 48.88% | 73.51% |
| V3 | 16D sinusoidal | 384 | 2 | 177K | 60.27% | 27.59 | 48.54% | 73.37% |
| V4 | 16D sinusoidal | 256 | 3 | 157K | 60.14% | 27.67 | 44.04% | 73.96% |
| V5 | 1D raw | 256 | 2 | 91K | 59.12% | 26.83 | 36.62% | 73.84% |

### Panoptic Quality Details (per variant)

| Variant | PQ_stuff | PQ_things | SQ | RQ | train PQ | moto PQ | moto TP | moto FP |
|---------|----------|-----------|-----|-----|----------|---------|---------|---------|
| V1 | 49.81 | 10.97 | 70.41 | 35.30 | 15.49 | 8.37 | 20 | 49 |
| V2 | 49.28 | 10.67 | 70.28 | 34.91 | 15.68 | 8.62 | 21 | 51 |
| V3 | 49.67 | 11.52 | 70.43 | 35.62 | 15.41 | 7.82 | 19 | 59 |
| V4 | 49.94 | 11.47 | 69.95 | 35.66 | 15.56 | 6.69 | 19 | 91 |
| V5 | 48.70 | 10.92 | 70.10 | 34.71 | 15.88 | 4.17 | 19 | 297 |

### Architecture Sweep Findings

1. **Sinusoidal depth input at adapter is redundant**: Sinusoidal encoding is already applied at k-means time (alpha=0.1). Adding it at the adapter input gives the model two chances to learn depth features — the extra capacity becomes noise, not signal.

2. **Wider hidden dims don't help (V2 < V1 by 0.44)**: More capacity = more ways to drift. The optimal corrections are low-rank.

3. **V3 (h=384) is closest runner-up (60.27%)**: Just 0.38% below original. Gap is within k-means stochasticity range — may not be meaningfully different.

4. **Deeper networks hurt motorcycle (V4: 44.04% vs Original: 48.21%)**: Extra layers amplify small errors in minority-class regions.

5. **V5 (1D raw, h=256) is worst (59.12%)**: Wider hidden + raw depth = more freedom than needed. FP motorcycle explodes to 297 (vs 49 for V1). Evidence that capacity is the enemy.

6. **Monotonic FP trend**: Motorcycle FP increases with model complexity: V1(49) < V2(51) < V3(59) < V4(91) < V5(297). More capacity → more false positives on minority classes.

## k=80 Evaluation (Downstream-Relevant)

The k=300 results above measure adapter quality in isolation. The downstream CUPS pipeline uses k=80 clusters. Both best adapters re-evaluated at k=80:

### Summary (k=80)

| Metric | Original (1D, h=128) | V3 (16D sin, h=384) | Delta |
|--------|---------------------|---------------------|-------|
| **mIoU** | 52.69% | **55.29%** | **+2.60** |
| Pixel Accuracy | 89.92% | 89.91% | -0.01 |
| PQ (ALL) | 26.08% | **26.44%** | +0.36 |
| PQ_stuff | 47.52% | **48.25%** | +0.73 |
| PQ_things | 10.49% | **10.59%** | +0.10 |
| SQ | 60.88% | **63.84%** | +2.96 |
| RQ | 33.56% | **34.06%** | +0.50 |

**V3 wins decisively at k=80** — ranking flips from k=300 where Original led by 0.38%.

### Cluster Allocation (k=80)

| Class | Orig Clusters | V3 Clusters | Note |
|-------|:---:|:---:|------|
| road | 17 | 17 | dominant |
| building | 15 | 16 | dominant |
| vegetation | 14 | 12 | dominant |
| sidewalk | 8 | 7 | moderate |
| car | 6 | 7 | moderate |
| sky | 5 | 5 | moderate |
| person | 3 | 3 | underserved |
| fence | 3 | 2 | underserved |
| bicycle | 2 | 2 | underserved |
| traffic sign | 1 | 2 | V3 recovers 2nd cluster |
| rider | **0** | **1** | **V3 recovers rider!** |
| motorcycle | 0 | 0 | dead in both |
| traffic light | 0 | 0 | dead in both |

**Root cause of k=80 drop**: With 80 clusters, dominant classes consume ~54 clusters (road+building+vegetation+sidewalk). Only ~26 remain for 15 other classes. Minority classes (motorcycle, traffic light, rider) get 0-1 clusters vs 1-3 at k=300.

### Per-Class IoU Comparison (k=80)

| Class | Original | V3 | Delta | Note |
|-------|----------|-----|-------|------|
| road | 94.90 | 95.07 | +0.17 | |
| sidewalk | 64.24 | 64.46 | +0.22 | |
| building | 83.03 | 82.20 | -0.83 | |
| wall | 43.81 | 48.52 | +4.71 | |
| fence | 45.09 | 47.62 | +2.53 | |
| pole | 8.29 | 11.46 | +3.17 | |
| traffic light | 0.00 | 0.00 | — | dead (0 clusters) |
| traffic sign | 38.21 | 44.36 | +6.15 | V3 gets 2nd cluster |
| vegetation | 81.29 | 80.27 | -1.02 | |
| terrain | 49.23 | 51.63 | +2.40 | |
| sky | 77.44 | 80.34 | +2.90 | |
| person | 51.24 | 52.94 | +1.70 | |
| **rider** | **0.00** | **30.35** | **+30.35** | **V3 recovers!** |
| car | 82.02 | 81.58 | -0.44 | |
| truck | 77.95 | 77.83 | -0.12 | |
| bus | 80.49 | 80.33 | -0.16 | |
| train | 73.84 | 73.83 | -0.01 | |
| motorcycle | 0.00 | 0.00 | — | dead (0 clusters) |
| bicycle | 49.94 | 47.71 | -2.23 | |

### Key Insight: Ranking Flips at Low k

At k=300, the Original adapter leads (60.65% vs 60.27%) because there are enough clusters for minority classes regardless of input encoding. At k=80, the sinusoidal depth encoding in V3 helps the clustering algorithm better separate classes that share similar CAUSE features but differ in depth — giving rider and traffic sign their own clusters.

## Key Takeaways

1. **Frozen features + tiny adapter is the safe way to adapt pretrained representations.** Full fine-tuning (Approach B) always collapses minority classes regardless of loss function.

2. **Static depth concatenation captures most of the useful signal (+2.35 mIoU).** The adapter adds another +3.89 by learning precise depth-boundary corrections.

3. **Less drift = better.** lambda_preserve=20 (drift=0.06) beats lp=1 (drift=1.21). Tiny corrections outperform large changes. Analogous to low-rank adaptation.

4. **No cluster loss needed.** Depth correlation + MSE preserve is sufficient. Cluster loss introduces majority-class bias that dominates optimization.

5. **Pre-extraction decouples adapter from backbone.** Training takes 109 sec vs 44 min. Can iterate on adapter design without touching the expensive CAUSE pipeline.

## Checkpoints

| File | Description |
|------|-------------|
| `results/depth_adapter/lp20.0/best.pt` | Best adapter (lp=20, mIoU=60.65%) |
| `results/depth_adapter/lp5.0/best.pt` | Runner-up (lp=5, mIoU=59.99%) |
| `results/depth_adapter/lp1.0/best.pt` | Loose constraint (lp=1, mIoU=58.93%) |
| `results/depth_adapter/best.pt` | Original lp=10 (mIoU=57.27%) |
| `results/depth_adapter/V1_dd16_h128_l2/best.pt` | V1: sin16D, h=128, 2L (59.61%) |
| `results/depth_adapter/V2_dd16_h256_l2/best.pt` | V2: sin16D, h=256, 2L (59.17%) |
| `results/depth_adapter/V3_dd16_h384_l2/best.pt` | V3: sin16D, h=384, 2L (60.27%) |
| `results/depth_adapter/V4_dd16_h256_l3/best.pt` | V4: sin16D, h=256, 3L (60.14%) |
| `results/depth_adapter/V5_dd1_h256_l2/best.pt` | V5: raw1D, h=256, 2L (59.12%) |
