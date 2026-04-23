# Statistical Appendix: DCFA v2 Multi-Seed Ablation

## 1. Descriptive Statistics

### 1.1 mIoU Across 5 Seeds

| Method | Mean | Std | Min | Max | Range | Median | Seeds |
|--------|------|-----|-----|-----|-------|--------|-------|
| DCFA-X (FiLM + Cross-Attn + Fusion) | 54.08 | 0.76 | 53.14 | 55.03 | 1.89 | 53.81 | 53.81, 53.14, 55.03, 53.53, 54.91 |
| Deep Bottleneck (4-Layer MLP) | 52.87 | 2.56 | 48.88 | 55.42 | 6.54 | 54.20 | 55.42, 54.20, 55.01, 48.88, 50.86 |
| Cross-Image Hard Negatives | 52.61 | 2.88 | 48.18 | 55.47 | 7.29 | 53.89 | 53.89, 55.21, 50.29, 48.18, 55.47 |
| DINOv2 Cross-Attention | 52.58 | 2.21 | 48.51 | 55.24 | 6.73 | 53.03 | 55.24, 48.51, 53.02, 53.12, 53.03 |
| FiLM Depth Conditioning | 52.35 | 1.75 | 50.30 | 55.55 | 5.25 | 52.20 | 50.30, 52.20, 52.30, 55.55, 51.41 |
| DCFA (2-Layer MLP Baseline) | 52.33 | 1.87 | 50.28 | 55.29 | 5.01 | 51.41 | 55.29, 53.69, 50.28, 50.97, 51.41 |

### 1.2 Per-Class IoU (Mean +/- Std Across 5 Seeds)

| Class | DCFA (Baseline) | Deep Bottleneck | DINOv2 Cross-Attn | Cross-Image Neg. | DCFA-X (Combined) | FiLM Conditioning |
|-------|----------------|----------------|-------------------|-----------------|-------------------|-------------------|
| road | 94.8 +/- 0.4 | 94.4 +/- 0.4 | 94.4 +/- 0.2 | 94.7 +/- 0.2 | 94.4 +/- 0.5 | 94.5 +/- 0.3 |
| sidewalk | 64.0 +/- 3.0 | 63.1 +/- 1.1 | 64.8 +/- 1.9 | 62.2 +/- 2.3 | 64.3 +/- 1.7 | 64.4 +/- 1.0 |
| building | 83.2 +/- 0.9 | 82.8 +/- 0.2 | 83.3 +/- 0.6 | 82.4 +/- 0.5 | 81.9 +/- 0.6 | 82.8 +/- 1.0 |
| wall | 48.5 +/- 1.0 | 49.3 +/- 1.8 | 46.7 +/- 2.5 | 47.5 +/- 2.1 | 48.5 +/- 1.1 | 48.3 +/- 0.4 |
| fence | 46.8 +/- 2.4 | 47.4 +/- 0.8 | 45.4 +/- 4.5 | 46.6 +/- 1.3 | 46.4 +/- 2.8 | 47.9 +/- 1.6 |
| pole | 11.4 +/- 2.3 | 9.2 +/- 5.1 | 6.3 +/- 7.9 | 5.7 +/- 4.7 | 9.3 +/- 1.5 | 9.5 +/- 2.1 |
| traffic light | 5.7 +/- 11.5 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 5.7 +/- 11.5 | 0.0 +/- 0.0 | 0.0 +/- 0.0 |
| traffic sign | 41.0 +/- 1.8 | 39.7 +/- 1.2 | 37.0 +/- 4.4 | 40.2 +/- 0.7 | 40.2 +/- 0.8 | 37.8 +/- 4.5 |
| vegetation | 81.5 +/- 0.9 | 81.8 +/- 0.5 | 81.2 +/- 0.7 | 81.4 +/- 0.2 | 81.0 +/- 0.4 | 81.5 +/- 0.9 |
| terrain | 49.4 +/- 2.2 | 48.6 +/- 2.0 | 49.5 +/- 1.5 | 48.4 +/- 1.3 | 48.1 +/- 2.2 | 49.2 +/- 1.4 |
| sky | 80.4 +/- 2.4 | 79.9 +/- 1.2 | 78.4 +/- 3.7 | 79.0 +/- 2.2 | 77.6 +/- 3.2 | 77.6 +/- 5.6 |
| person | 51.1 +/- 2.6 | 49.6 +/- 2.7 | 51.5 +/- 2.0 | 52.5 +/- 2.8 | 53.9 +/- 1.1 | 51.9 +/- 3.1 |
| rider | 21.3 +/- 17.6 | 28.5 +/- 14.6 | 13.3 +/- 16.6 | 23.0 +/- 18.8 | 19.5 +/- 16.3 | 22.9 +/- 18.7 |
| car | 82.7 +/- 1.1 | 79.6 +/- 1.4 | 80.1 +/- 1.2 | 81.5 +/- 1.3 | 80.3 +/- 1.3 | 81.3 +/- 2.2 |
| truck | 77.9 +/- 0.1 | 78.0 +/- 0.1 | 78.0 +/- 0.0 | 77.9 +/- 0.0 | 77.8 +/- 0.1 | 77.9 +/- 0.1 |
| bus | 74.0 +/- 5.2 | 76.2 +/- 5.2 | 79.4 +/- 1.2 | 76.2 +/- 5.1 | 80.2 +/- 0.4 | 75.5 +/- 4.9 |
| train | 29.5 +/- 36.1 | 44.3 +/- 36.1 | 58.1 +/- 29.1 | 44.2 +/- 36.1 | 73.5 +/- 0.6 | 40.5 +/- 33.3 |
| motorcycle | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 |
| bicycle | 51.2 +/- 1.9 | 52.2 +/- 2.6 | 51.7 +/- 4.0 | 50.4 +/- 2.5 | 50.6 +/- 2.9 | 51.0 +/- 1.4 |

### 1.3 Cluster-Aware Training (Single Seed = 42)

| Method | mIoU | val_loss (depth) | Drift | Key Class Anomalies |
|--------|------|-----------------|-------|-------------------|
| CA1: Cluster-Aware CE (lp=20) | 54.41 | 1.5544 | 0.126 | pole=0.0, traffic_light=0.0, train=73.7 |
| CA2: Cluster-Aware CE (lp=10) | 51.31 | 1.5497 | 0.248 | train=0.0 (collapsed), pole=8.7 |
| CA3: Hybrid Depth-Corr + Cluster CE | 53.81 | 0.5245 | 0.063 | rider=0.0, traffic_light=0.0 |

---

## 2. Assumption Checks

### 2.1 Normality (Shapiro-Wilk)

All six methods pass normality at alpha=0.05:

| Method | W | p | Passes |
|--------|---|---|--------|
| DCFA (Baseline) | 0.9076 | 0.4530 | Yes |
| Deep Bottleneck | 0.8718 | 0.2739 | Yes |
| DINOv2 Cross-Attention | 0.8313 | 0.1423 | Yes |
| Cross-Image Negatives | 0.8752 | 0.2879 | Yes |
| DCFA-X (Combined) | 0.8861 | 0.3379 | Yes |
| FiLM Conditioning | 0.8938 | 0.3766 | Yes |

**Caveat**: With n=5, Shapiro-Wilk has low power. Normality is assumed but not strongly confirmed.

### 2.2 Homogeneity of Variances (Levene's Test)

Levene statistic = 0.6793, p = 0.6433

Equal variances **not rejected**. However, visual inspection shows DCFA-X variance (0.71) is much lower than others (1.75-2.88). The Levene test is underpowered at n=5 per group.

---

## 3. Omnibus Tests

### 3.1 One-Way ANOVA

- F(5, 24) = 0.3858
- p = 0.8536
- **Not significant**. No evidence that methods differ in mean mIoU.

### 3.2 Kruskal-Wallis (Non-Parametric)

- H = 1.8198
- p = 0.8735
- **Not significant**. Confirms ANOVA result.

---

## 4. Pairwise Comparisons (DCFA-X vs Each Other)

Paired t-tests (same seeds) with Bonferroni correction for 5 comparisons:

| Comparison | Mean Diff | t(4) | p_raw | p_bonf | Cohen's d | 95% CI |
|-----------|-----------|------|-------|--------|-----------|--------|
| DCFA-X vs DCFA (Baseline) | +1.76 | 1.473 | 0.2149 | 1.0000 | 0.66 | [-1.55, +5.07] |
| DCFA-X vs Deep Bottleneck | +1.21 | 0.922 | 0.4085 | 1.0000 | 0.41 | [-2.43, +4.85] |
| DCFA-X vs DINOv2 Cross-Attn | +1.50 | 1.501 | 0.2078 | 1.0000 | 0.67 | [-1.27, +4.27] |
| DCFA-X vs Cross-Image Neg. | +1.48 | 0.986 | 0.3799 | 1.0000 | 0.44 | [-2.68, +5.63] |
| DCFA-X vs FiLM Conditioning | +1.73 | 1.652 | 0.1739 | 0.8695 | 0.74 | [-1.18, +4.64] |

**All comparisons non-significant after Bonferroni correction.** Even uncorrected, none reach p < 0.05. Effect sizes are medium (d = 0.41-0.74) but confidence intervals all span zero.

---

## 5. Variance Comparison

### DCFA-X vs DCFA Baseline

- Var(DCFA Baseline) = 4.3774
- Var(DCFA-X) = 0.7126
- F-ratio = 6.14 (baseline / DCFA-X)
- p = 0.0533 (one-tailed F-test, df1=4, df2=4)

**Marginal significance** (p=0.053). At n=5, the F-test for variance has very low power. The 6.1x variance ratio is practically large but not statistically confirmed at conventional thresholds.

---

## 6. Power Analysis

For the observed effect size (d=0.66, paired design):

| Seeds per method | Power (alpha=0.05) |
|------------------|-------------------|
| 5 (current) | ~25% |
| 10 | ~48% |
| 20 | ~80% |
| 30 | ~92% |

To achieve 80% power to detect DCFA-X's advantage, we would need **~20 seeds per method** — approximately 4x the current evaluation budget.

---

## 7. Cluster-Aware Training Dynamics

### CA1 (Cluster-Aware CE, lp=20)

| Epoch | Train Loss | Val Loss | Drift |
|-------|-----------|---------|-------|
| 0 | 1.5591 | 1.5569 | 0.096 |
| 5 | 1.5551 | 1.5551 | 0.119 |
| 10 | 1.5546 | 1.5547 | 0.120 |
| 19 | 1.5542 | 1.5544 | 0.126 |

Loss decreased only 0.005 across 20 epochs. The codes were already near their centroids (loss << log(80)=4.38), so the loss provides very weak gradient signal.

### CA2 (Cluster-Aware CE, lp=10)

| Epoch | Train Loss | Val Loss | Drift |
|-------|-----------|---------|-------|
| 0 | 1.5562 | 1.5541 | 0.176 |
| 10 | 1.5501 | 1.5502 | 0.238 |
| 19 | 1.5495 | 1.5497 | 0.248 |

Higher drift (0.248 vs 0.126) due to lower preservation weight, but loss barely improves.

### CA3 (Hybrid Depth-Corr + Cluster CE, lp=20)

| Epoch | Total Loss | Depth-Corr | Cluster CE | Drift |
|-------|-----------|-----------|-----------|-------|
| 0 | 0.5337 | 0.4674 | 0.0634 | 0.079 |
| 10 | 0.5324 | 0.4681 | 0.0634 | 0.063 |
| 19 | 0.5321 | 0.4677 | 0.0634 | 0.064 |

Cluster CE stays flat at 0.0634 throughout training — it provides no learning signal when the depth-corr loss already constrains codes near their original positions.

---

## 8. Explicit Limitations

1. **n=5 seeds is underpowered** — cannot detect genuine differences of 1-2% mIoU at conventional significance thresholds.
2. **Only k-means seed varies** — adapter training seed is fixed at 42. True robustness requires varying both.
3. **Cluster-aware experiments lack multi-seed evaluation** — their single-seed mIoU cannot be rigorously compared to 5-seed means.
4. **3 methods excluded from multi-seed** — A2 (normals), A3 (gradients), B3 (window attention) only have single-seed data.
5. **No per-class significance tests** — per-class IoU distributions are non-normal (many zeros for volatile classes), and n=5 prohibits meaningful class-level inference.
