# DCFA v2 Ablation: Multi-Seed Analysis Report

## Analysis Question

**Do any of the 9 DCFA v2 adapter variants genuinely beat the DCFA baseline (2-layer MLP, 40K params) at k=80 mIoU on Cityscapes val, once k-means stochasticity is controlled?**

Secondary question: Can cluster-aware training (replacing depth correlation loss with cross-entropy toward centroids) improve cluster separability?

## Experimental Setup

- **Dataset**: Cityscapes val (500 images, 1024x2048)
- **Backbone**: DINOv2 ViT-B/14 (frozen) producing 90D CAUSE codes
- **Evaluation**: k-means (k=80) on adapted 90D codes + 16D sinusoidal depth, overclustered pseudo-labels evaluated via 19-class Hungarian-matched mIoU
- **Seeds**: 5 k-means seeds (42, 123, 456, 789, 1024) per method
- **Note**: Training seed is fixed (42); only k-means initialization varies

### Methods Evaluated (Multi-Seed)

| ID | Full Method Name | Architecture | Params |
|----|-----------------|-------------|--------|
| V3 | **DCFA (2-Layer MLP)** | 90D codes + 16D depth -> MLP(384) x2 -> residual 90D | 40K |
| B2 | **Deep Bottleneck (4-Layer MLP)** | 4-layer MLP with 64D bottleneck, h=384 | 55K |
| A1 | **DINOv2 Cross-Attention** | Cross-attention (Q=codes, KV=DINOv2 768D), h=256 | 59K |
| C2 | **Cross-Image Hard Negatives** | V3 architecture + cross-image repulsion loss | 40K |
| X | **DCFA-X (FiLM + Cross-Attn + Fusion)** | FiLM gating + DINOv2 cross-attn + fusion MLP | 135K |
| B1 | **FiLM Depth Conditioning** | Multiplicative depth modulation (gamma, beta) | 45K |

### Cluster-Aware Training (Single-Seed)

| ID | Full Method Name | Loss | Preservation |
|----|-----------------|------|-------------|
| CA1 | **Cluster-Aware Cross-Entropy (lp=20)** | CE toward centroid assignments | lambda_preserve=20 |
| CA2 | **Cluster-Aware Cross-Entropy (lp=10)** | CE toward centroid assignments | lambda_preserve=10 |
| CA3 | **Hybrid Depth-Corr + Cluster CE** | depth_corr + cluster CE (lambda=1.0) | lambda_preserve=20 |

---

## Key Findings

### Finding 1: No method significantly beats the DCFA baseline

One-way ANOVA: F=0.39, p=0.85 — **no significant difference** across methods. Kruskal-Wallis confirms (H=1.82, p=0.87). All pairwise comparisons of DCFA-X vs others fail to reach significance even at uncorrected alpha=0.05 (all p > 0.17).

**The original single-seed comparison was misleading.** V3's reported 55.29% was its best seed. Its true mean is 52.33 +/- 1.87%.

### Finding 2: DCFA-X (FiLM + Cross-Attn + Fusion) is the most stable method

| Metric | DCFA (Baseline) | DCFA-X (Combined) |
|--------|-----------------|-------------------|
| Mean mIoU | 52.33% | 54.08% |
| Std | 1.87% | 0.76% |
| Range | 5.01% | 1.89% |
| Variance | 4.38 | 0.71 |

DCFA-X's variance is **6.1x lower** than the baseline (F-ratio=6.14, p=0.053 — marginal at n=5). Cohen's d for the mean difference is 0.66 (medium effect). The 95% CI for the paired difference is [-1.55, +5.07], which includes zero — the mean gain is not statistically confirmed, but the variance reduction is the real story.

### Finding 3: Stability comes from the "train" and "bus" classes

The per-class analysis reveals WHERE DCFA-X achieves stability:

| Class | DCFA Baseline (mean +/- std) | DCFA-X (mean +/- std) |
|-------|------------------------------|----------------------|
| **train** | 29.5 +/- 36.1 | **73.5 +/- 0.6** |
| **bus** | 74.0 +/- 5.2 | **80.2 +/- 0.4** |
| **person** | 51.1 +/- 2.6 | **53.9 +/- 1.1** |
| rider | 21.3 +/- 17.6 | 19.5 +/- 16.3 |
| motorcycle | 0.0 +/- 0.0 | 0.0 +/- 0.0 |

DCFA-X's "train" class IoU is **73.5% on every seed** (std=0.6). The baseline swings between 0% and 73.8% depending on whether k-means assigns a cluster to train. The FiLM + cross-attention + fusion architecture produces features where train consistently receives its own cluster.

### Finding 4: Cluster-aware training does not improve mIoU

| Method | mIoU (seed=42) | vs DCFA-X mean |
|--------|---------------|----------------|
| CA1: Cluster-Aware CE (lp=20) | 54.41% | +0.33 |
| CA3: Hybrid Depth-Corr + Cluster CE | 53.81% | -0.27 |
| CA2: Cluster-Aware CE (lp=10) | 51.31% | -2.77 |

All three are within seed noise of the DCFA baseline (52.33 +/- 1.87). CA1's 54.41% looks decent but is a single seed — it could easily be a lucky draw (V3 got 55.29% on seed=42). Lower preservation (CA2, lp=10) causes excessive drift (0.24 vs V3's 0.06), degrading stable classes.

**Root cause**: The cluster-aware loss pushes codes toward their *current* centroid assignments. Since centroids were computed from original codes, this reinforces existing structure rather than discovering better structure.

### Finding 5: Five volatile classes dominate mIoU variance

| Class | All-Method Mean | All-Method Std | Problem |
|-------|----------------|----------------|---------|
| motorcycle | 0.0% | 0.0 | Never gets a cluster at k=80 |
| traffic light | 1.9% | 6.4 | Gets a cluster in ~1/5 runs |
| pole | 8.2% | 4.4 | Sometimes absorbed by building |
| rider | 21.8% | 16.8 | Merges with person or bicycle |
| train | 45.2% | 29.6 | Binary: either matched or not |

These 5 classes contribute 15-30% of mIoU variance. On the remaining 14 stable classes, all methods perform within ~1% of each other.

---

## What Changed in Understanding

1. **Before**: V3 baseline appeared to score 55.29%. B2 (55.42%) and A1 (55.24%) appeared comparable.
2. **After**: V3's true mean is 52.33%. All methods are statistically indistinguishable in mean mIoU. DCFA-X has a practically meaningful variance advantage (6.1x lower) but not enough samples to confirm statistically.
3. **Before**: Cluster-aware training was a promising direction.
4. **After**: Pushing toward existing centroids reinforces current structure. The loss is not the bottleneck — the 90D code space already separates stable classes well, and volatile classes need more clusters, not better features.

## Caveats

- **n=5 seeds is underpowered** for detecting 1-2% mIoU differences. At observed effect sizes (d~0.66), would need ~20 seeds for 80% power at alpha=0.05.
- **Training seed is constant (42)**. The multi-seed evaluation varies only k-means initialization, not adapter training. True robustness requires varying training seed too.
- **Cluster-aware experiments are single-seed**. Their mIoU values cannot be compared to 5-seed means. They may appear better or worse purely due to seed luck.
- **Only 6 of 9 methods received multi-seed evaluation**. A2 (normals), A3 (gradients), B3 (window attention) were excluded from multi-seed; their single-seed results suggested they underperform (51.86%, 54.06%, 54.02%).
