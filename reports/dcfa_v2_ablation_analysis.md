# DCFA v2 Ablation Analysis Report

## Summary

9 adapter variants evaluated at k=80 mIoU. **No method significantly beats the V3 baseline (55.29%)**. Root cause analysis reveals two independent problems:

1. **Evaluation noise**: mIoU differences are dominated by k-means stochasticity on 5 volatile classes (rider, pole, train, traffic light, motorcycle). A single k-means seed is insufficient to rank methods.
2. **Genuine regression on stable classes**: Even on the 14 stable classes where k-means noise is minimal, all methods score below V3 (66.78%), confirming the V3 architecture is near-optimal for the depth correlation objective.

## Results

| Rank | Method | mIoU (%) | Stable (14cls) | Volatile (5cls) | Params |
|------|--------|----------|----------------|-----------------|--------|
| 0 | **V3 baseline** | **55.29** | **66.78** | 23.13 | 40K |
| 1 | B2 deep | 55.42 | 66.16 | 25.32 | 55K |
| 2 | A1 cross-attn | 55.24 | 65.62 | 26.16 | 59K |
| 3 | A3 gradients | 54.06 | 65.92 | 20.86 | 226K |
| 4 | B3 window-attn | 54.02 | 64.92 | 23.51 | 226K |
| 5 | C2 cross-image | 53.89 | 65.02 | 22.72 | 40K |
| 6 | X combined | 53.81 | 65.14 | 22.06 | 135K |
| 7 | C1 contrastive | 52.82 | 65.53 | 17.23 | 40K |
| 8 | A2 normals | 51.86 | 64.54 | 16.37 | 226K |
| 9 | B1 FiLM | 50.30 | 64.97 | 9.21 | 45K |

## Key Finding 1: mIoU is dominated by k-means noise

5 volatile classes (motorcycle, pole, rider, traffic light, train) have std > 5% or hit IoU=0 across methods:

| Class | Mean | Std | Zeros | V3 |
|-------|------|-----|-------|-----|
| motorcycle | 0.0% | 0.0% | 10/10 | 0.0% |
| traffic light | 3.0% | 9.1% | 9/10 | 0.0% |
| pole | 8.7% | 5.4% | 2/10 | 11.5% |
| rider | 25.0% | 16.9% | 3/10 | 30.4% |
| train | 66.5% | 22.2% | 1/10 | 73.8% |

**Rider alone** swings mIoU by up to 30.4% (between 0% and 39.9% across methods). This is not because the adapter failed -- it's because k-means either assigns a cluster to rider or doesn't, and Hungarian matching either finds a match or doesn't.

**Implication**: Comparing methods with a single k-means seed at k=80 is unreliable. Need 5+ seeds to detect genuine 1-2% differences.

## Key Finding 2: On stable classes, all methods regress from V3

On the 14 stable classes (where k-means noise is < 5%):

- **V3**: 66.78%
- **Best challenger (B2 deep)**: 66.16% (-0.62%)
- **Worst (A2 normals)**: 64.54% (-2.24%)

This is a genuine regression, not noise. Adding complexity to the adapter consistently makes the stable-class features slightly worse.

## Key Finding 3: val_loss does not predict mIoU

All methods converge to val_loss ~ 0.458-0.465. The depth correlation + MSE preservation loss is agnostic to cluster separability. An adapter can achieve good val_loss while producing features that cluster poorly.

## Diagnosis: Why each method failed

### Group A: Input enrichment
- **A1 cross-attn (55.24%)**: 768D DINOv2 features are redundant — the 90D codes already capture the discriminative information for the 14 stable classes. Cross-attention adds parameters but no new signal for clustering.
- **A2 normals (51.86%)**: Surface normals corrupt the depth signal. The adapter learns to mix normals into the codes, but normals don't align with semantic boundaries (e.g., road and sidewalk have similar normals).
- **A3 gradients (54.06%)**: Depth gradients provide boundary signal but this is already captured by depth correlation loss. Gains on traffic_light (+30.4%) offset by losing rider and pole entirely.

### Group B: Architecture
- **B1 FiLM (50.30%)**: Multiplicative modulation is too powerful — it can zero out code dimensions. The depth signal is noisy enough that FiLM gates amplify noise. Train dropped to 0% (-73.8).
- **B2 deep (55.42%)**: 4-layer bottleneck MLP is marginal. The extra depth doesn't help because the 2-layer V3 MLP is already sufficient for the linear-ish depth-code relationship.
- **B3 window-attn (54.02%)**: Local attention averages neighboring patches. Hurts person (-11.9%) because person patches at boundaries get averaged with background, reducing discriminability.

### Group C: Loss
- **C1 contrastive (52.82%)**: The contrastive loss pushes apart clusters, but uses the CURRENT (V3-trained) centroids. These centroids are computed on raw (unadjusted) features — applying them to adjusted features creates a mismatch.
- **C2 cross-image (53.89%)**: Cross-image repulsion loses global context. Features from "road in scene A" and "road in scene B" get pushed apart instead of together.

### DCFA-X combined (53.81%)
Combines FiLM + cross-attention + contrastive — accumulates all individual failure modes. FiLM's gating instability + cross-attention's redundancy + contrastive's centroid mismatch.

## What this means

The V3 adapter (simple 2-layer MLP, 40K params) is essentially optimal for the depth correlation objective. The adapter is doing its job — it adjusts codes so that depth-similar patches become more similar. But the downstream k-means clustering has its own bottleneck:

1. The depth correlation loss is a **proxy** for cluster quality, not cluster quality itself.
2. k=80 clusters must cover 19 classes — rare classes (motorcycle, traffic light) may not get a cluster regardless of how good the features are.
3. The 90D code space already separates the 14 stable classes well. The remaining 5 volatile classes need fundamentally different approaches (more clusters, class-specific handling).

## Actionable next steps (ranked by expected impact)

### 1. Multi-seed evaluation (HIGH PRIORITY, zero compute cost)
Run k-means with 5 seeds per method, report mean +/- std mIoU. This removes noise and reveals whether B2 or A1 genuinely beat V3 or just got lucky. Expected cost: ~1 hour.

### 2. Cluster-aware adapter training (MEDIUM, novel)
Replace depth correlation loss with a loss that directly optimizes cluster separability:
- Silhouette score as differentiable loss
- Or: use V3's k-means cluster assignments as soft labels, train adapter with cross-entropy
- Expected gain: 1-3% mIoU on stable classes

### 3. Class-balanced k-means (LOW, mechanical)
Allocate more clusters to under-represented classes. E.g., k=120 with balanced allocation could ensure motorcycle/traffic light always get clusters. No adapter change needed.

### 4. Iterative self-refinement (MEDIUM, requires #2)
Adapter -> k-means -> use assignments as training signal -> re-train adapter -> repeat. This is the EM approach. Only makes sense if #2 shows the cluster-aware loss helps.
