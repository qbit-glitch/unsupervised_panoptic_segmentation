# Figure Catalog: DCFA v2 Multi-Seed Analysis

## Figure 1: Main Comparison — Mean mIoU with Seed Points

- **Filename**: `figures/figure-01-main-comparison.pdf`
- **Purpose**: Show that all methods are statistically indistinguishable in mean mIoU, with individual seed points revealing the true spread.
- **Data source**: 6 methods x 5 k-means seeds, `results/depth_adapter/DCFA_v2/*/eval_k80_s{42,123,456,789,1024}.json`
- **Plotted variables**: Horizontal bars = mean mIoU; error bars = 1 std; black dots = individual seed values (jittered vertically)
- **Error bar meaning**: +/- 1 standard deviation across 5 k-means seeds (same adapter weights, different k-means initialization)

### Caption Requirements
- Must state n=5 k-means seeds, k=80, Cityscapes val
- Must mention training seed is constant (42)
- Must note red dashed line = DCFA baseline mean (52.33%)
- Must state error bars = 1 std

### Key Observation
DCFA-X (FiLM + Cross-Attn + Fusion) has both the highest mean (54.08%) and the smallest error bar (std=0.76). All other methods cluster at 52.3-52.9% with much wider spread (std 1.75-2.88). Individual seed points show V3 has one outlier at 55.29 (its "lucky seed") while DCFA-X's 5 seeds are tightly grouped.

### Interpretation Checklist
1. Why does this figure exist? To replace the misleading single-seed comparison with a proper multi-seed view.
2. What should the reader notice? DCFA-X is the only method whose error bar doesn't overlap with all others. The baseline's "55.29%" was seed luck.
3. What does this change? Single-seed mIoU rankings are unreliable for k=80 overclustering. Future adapter comparisons must use 5+ seeds.

---

## Figure 2: Per-Class Stability Heatmap + mIoU Std Comparison

- **Filename**: `figures/figure-02-stability-heatmap.pdf`
- **Purpose**: Identify WHICH classes drive mIoU variance and WHY DCFA-X is more stable.
- **Data source**: Same as Figure 1, per-class IoU breakdown
- **Plotted variables**: Left panel: heatmap of per-class IoU std (19 classes x 6 methods); Right panel: barplot of mIoU std per method
- **Error bar meaning**: N/A (heatmap cells are std values themselves)

### Caption Requirements
- Left panel: color scale = std of per-class IoU across 5 seeds; annotated cells have std > 5%
- Right panel: mIoU std per method (lower is better)
- Must note that red cells indicate classes with high k-means sensitivity

### Key Observation
Three classes dominate variance: **train** (std up to 36.1), **rider** (std up to 18.8), and **traffic light** (std up to 11.5). DCFA-X has the lowest std for train (0.6 vs 29-36 for others) and bus (0.4 vs 4.9-5.2). Right panel shows DCFA-X's mIoU std is 0.76 vs 1.75-2.88 for all others.

### Interpretation Checklist
1. Why? To decompose aggregate variance into class-level contributions.
2. What to notice? Train/rider are the "chaos classes." DCFA-X stabilizes train and bus but not rider.
3. What changes? DCFA-X's advantage is class-specific: it helps large vehicles, not small objects.

### Known Caveats
- Motorcycle (always 0.0 +/- 0.0) and traffic_light (mostly 0.0) appear as cold cells but represent floor effects, not stability.
- n=5 makes per-class std estimates themselves noisy.

---

## Figure 3: Volatile Class Traces Across Seeds

- **Filename**: `figures/figure-03-volatile-classes.pdf`
- **Purpose**: Show seed-by-seed behavior for the three most volatile classes to explain WHY DCFA-X is stable.
- **Data source**: Same as Figure 1, per-class IoU for train, bus, rider
- **Plotted variables**: x-axis = k-means seed; y-axis = class IoU; lines = methods (6 colors)

### Caption Requirements
- Three subplots: train, bus, rider
- Must label all 6 methods in legend
- Must state these are the 3 most decision-relevant volatile classes

### Key Observation
**Train** (left): DCFA-X is a flat line at ~73.5% while all other methods oscillate between 0% and 73%. This is the single most striking result — DCFA-X consistently assigns at least one k-means cluster to train regardless of initialization. **Bus** (center): DCFA-X is flat at ~80% while others oscillate +-5%. **Rider** (right): ALL methods are equally unstable — DCFA-X provides no advantage here.

### Interpretation Checklist
1. Why? To show the mechanism behind DCFA-X's stability.
2. What to notice? Train class is binary (on/off) for 5/6 methods. DCFA-X eliminated this binary behavior.
3. What changes? DCFA-X's combined architecture creates features where large vehicles form sufficiently distinct clusters that k-means always finds them. This does NOT generalize to small objects (rider, motorcycle).

---

## Figure 4: Cluster-Aware Training vs Multi-Seed Baselines

- **Filename**: `figures/figure-04-cluster-aware.pdf`
- **Purpose**: Show that cluster-aware training (3 variants) does not beat the multi-seed baseline.
- **Data source**: `results/depth_adapter/DCFA_v2/CA{1,2,3}_*/eval_k80.json` + multi-seed means
- **Plotted variables**: Horizontal bars = mIoU; annotated with exact values

### Caption Requirements
- Must clearly note CA experiments use single seed (42) only
- Must label baselines as "5-seed mean"
- Must note comparison is unfair (1 seed vs 5-seed mean)

### Key Observation
CA1 (54.41%, single seed) sits between the 5-seed means of DCFA-X (54.08%) and DCFA baseline (52.33%). This looks good but is misleading — V3's seed=42 result was 55.29%. CA2 (lp=10) regresses to 51.31% due to excessive drift. CA3 (hybrid) at 53.81% is unremarkable.

### Interpretation Checklist
1. Why? To evaluate whether the cluster-aware loss direction is worth pursuing.
2. What to notice? No CA variant exceeds the DCFA-X 5-seed mean. The loss direction is a dead end.
3. What changes? Abandon cluster-aware training. The bottleneck is not the loss function but the 90D code space and k=80 cluster budget.

### Known Caveats
- Comparing single-seed CA results to 5-seed means is statistically improper. CA1's 54.41% could be a lucky seed.
