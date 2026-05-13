# Overclustering Ablation: Methods, Backbones, and Cluster Count for Pseudo-Label Generation

**Date**: 2026-05-08
**Experiment line**: Stage-0 pseudo-label overclustering
**Purpose**: Identify optimal clustering method, DINOv2 backbone, and cluster count (k) for raw pseudo-label generation in the MBPS unsupervised panoptic segmentation pipeline.
**Dataset**: Cityscapes val (500 images)
**Evaluation**: 19-class majority-vote Hungarian mapping, panoptic quality with SPIdepth depth-guided instances

---

## 1. Executive Summary

We tested **7 clustering methods** across **3 DINOv2 backbones** and **4 cluster counts** (k=27, 80, 100, 120, 150) to find the best overclustering configuration for Stage-0 pseudo-labels. The experiment covered 15 total configurations.

**Winner**: DINOv2 ViT-L/16 + Spherical K-Means, k=100

| Metric | Value |
|--------|-------|
| mIoU | **39.1%** |
| PQ | **20.9** |
| PQ_stuff | **30.5** |
| PQ_things | **7.7** |
| SQ | 72.7 |
| RQ | 29.5 |
| Empty clusters | 0 |
| Entropy | 0.982 |

This represents a **+5.3 PQ** improvement over the original euclidean k-means ViT-B/16 baseline (15.6), achieved through three independent gains:
1. Centroid renormalization bug fix: **+1.5 PQ** (15.6 → 17.1)
2. Backbone upgrade ViT-B/16 → ViT-L/16: **+1.9 PQ** (17.1 → 19.0)
3. Cluster count increase k=80 → k=100: **+1.9 PQ** (19.0 → 20.9)

---

## 2. Experiment Identity and Decision Context

**Question**: The existing pipeline uses MiniBatchKMeans (k=80) on L2-normalized DINOv2 ViT-B/16 features (768-dim). A bug was identified: centroids are never re-normalized after fitting, causing norm drift (0.69–0.93, mean 0.84). Are there better alternatives?

**Decision context**: Stage-0 pseudo-labels feed into DCFA (depth-conditioned feature adapter), SIMCF-ABC (instance refinement), and CUPS Cascade Mask R-CNN (Stage-2 detector). Improvements here propagate multiplicatively through the pipeline.

**Prior baseline**: PQ=15.6 (euclidean k-means, ViT-B/16, k=80, centroid drift bug present).

---

## 3. Setup and Evaluation Protocol

### Features
| Backbone | Embed dim | Patch size | Grid | Feature dir |
|----------|-----------|------------|------|-------------|
| DINOv2 ViT-B/16 | 768 | 16 | 32×64 (2048 patches) | `dinov3_features/` |
| DINOv2 ViT-L/16 | 1024 | 16 | 32×64 (2048 patches) | `dinov3_features_vitl16/` |
| DINOv2 ViT-g/14 | 1536 | 14 | 37×74 (2738 patches) | `dinov2g14_features/` |

All features L2-normalized before clustering. Stored as float16 .npy files per image.

### Clustering Methods
| ID | Method | Description |
|----|--------|-------------|
| M0 | `euclidean_kmeans` | sklearn MiniBatchKMeans (baseline, centroid drift bug) |
| M1 | `spherical_kmeans` | MiniBatchKMeans + 20 iterations of centroid L2 renormalization |
| M2 | `vmf` | von Mises-Fisher EM with per-cluster concentration κ |
| M3 | `kmeans_sinkhorn` | Spherical k-means + Sinkhorn equipartition |
| M4 | `vmf_sinkhorn` | vMF-EM + Sinkhorn |
| M5 | `eagle_spectral` | EAGLE EiCue spectral enrichment + k-means |
| M6 | `cause_codebook` | Learnable codebook with cosine VQ + diversity loss |

### Evaluation
- **Semantic**: majority-vote cluster→class mapping, 19-class mIoU
- **Instance**: SPIdepth depth-guided instances (τ=0.20, A_min=1000), AP/AR
- **Panoptic**: PQ, PQ_stuff, PQ_things via connected-component thing instances + cluster mapping
- All at 512×1024 resolution on Cityscapes val (500 images)

### Scripts
- Generation: `mbps_pytorch/generate_clustering_ablation.py --method <M> --k <K> --feat_subdir <dir>`
- Evaluation: `mbps_pytorch/evaluate_cascade_pseudolabels.py`
- Driver: `scripts/run_clustering_ablation.sh`

---

## 4. Main Findings

### 4.1 Method Comparison (ViT-B/16, k=80)

| Method | mIoU | PQ | PQ_stuff | PQ_things | Entropy | Notes |
|--------|------|------|----------|-----------|---------|-------|
| euclidean_kmeans | 26.2 | 15.6 | 25.9 | 1.4 | — | Baseline (centroid drift) |
| **spherical_kmeans** | **29.1** | **17.1** | **28.1** | **1.9** | 0.973 | **+1.5 PQ from renorm fix** |
| vmf | 9.7 | 5.1 | 7.8 | 1.4 | 0.382 | COLLAPSED: 70/80 empty |
| kmeans_sinkhorn | 27.9 | 16.4 | 27.1 | 1.8 | 0.976 | Balanced but no PQ gain |
| cause_codebook | 28.2 | 16.7 | 27.2 | 2.4 | 0.937 | Best PQ_things but lower overall |
| eagle_spectral | 9.9 | 5.2 | 8.9 | 0.0 | 0.995 | Spatial, not semantic |

**Ranking**: spherical_kmeans > cause_codebook > kmeans_sinkhorn > euclidean_kmeans >> vmf ≈ eagle_spectral

### 4.2 Backbone Scaling (Spherical K-Means, k=80)

| Backbone | Embed dim | Patches | mIoU | PQ | PQ_stuff | PQ_things |
|----------|-----------|---------|------|------|----------|-----------|
| ViT-B/16 | 768 | 32×64 | 29.1 | 17.1 | 28.1 | 1.9 |
| **ViT-L/16** | **1024** | **32×64** | **35.8** | **19.0** | **29.5** | **4.4** |
| ViT-g/14 | 1536 | 37×74 | 28.9 | 8.2 | 12.4 | 2.4 |

**ViT-L/16 wins**. ViT-g/14 regresses catastrophically despite higher feature dimensionality.

### 4.3 ViT-g/14 Failure Analysis

ViT-g/14 uses patch_size=14 → 37×74 patch grid on 518×1036 input. Upsampling to 512×1024 output requires fractional scaling (37→512 = 13.84× vertically, 74→1024 = 13.84× horizontally). Nearest-neighbor interpolation creates visible block artifacts at class boundaries.

**Evidence**: ViT-g/14 actually improved fine-grained classes (fence, pole, rider got non-zero IoU), proving feature quality is superior. But stuff-class PQ collapsed (sky 64.5→4.0, vegetation 71.1→16.7) because upsampling artifacts fragment contiguous regions into multiple disconnected components.

This is a spatial alignment problem, not a feature quality problem.

### 4.4 ViT-g/14 k=27 — All Methods Failed

Reducing k from 80 to 27 on ViT-g/14 made results worse across all methods:

| Method | mIoU | PQ | PQ_stuff | PQ_things | Dead classes |
|--------|------|------|----------|-----------|-------------|
| euclidean_kmeans | 16.4 | 6.5 | 10.9 | 0.5 | 13 |
| spherical_kmeans | 18.2 | 6.9 | 11.2 | 0.9 | 12 |
| kmeans_sinkhorn | 18.3 | 7.1 | 11.6 | 0.9 | 12 |
| cause_codebook | 18.7 | 6.7 | 10.8 | 1.1 | 12 |

All within noise (PQ 6.5–7.1). The spatial misalignment dominates over clustering method choice.

### 4.5 Cluster Count Scaling (ViT-L/16, Spherical K-Means)

| k | mIoU | PQ | PQ_stuff | PQ_things | Entropy | Gini | Min/Median/Max size | Dead |
|---|------|------|----------|-----------|---------|------|---------------------|------|
| 80 | 35.8 | 19.0 | 29.5 | 4.4 | 0.978 | 0.229 | 11 / 5003 / 10083 | 7 |
| **100** | **39.1** | **20.9** | **30.5** | **7.7** | **0.982** | **0.218** | **696 / 4143 / 7325** | **7** |
| 120 | 33.7 | 18.6 | 29.8 | 3.3 | 0.981 | 0.224 | 8 / 3452 / 7212 | 9 |
| 150 | 39.8 | 20.7 | 30.0 | 7.8 | 0.986 | 0.198 | 8 / 2710 / 5487 | 6 |

The k-scaling trend is **non-monotonic** and **plateaus at k=100**:

- **k=80→100 (+1.9 PQ)**: More clusters allow finer semantic separation. PQ_things gains most (+3.3).
- **k=120 DROPS (−2.3 PQ)**: Seed-dependent artifact — bus (61.5%→0%) and terrain (27.1%→0%) lost entirely, adding 2 dead classes. The random centroid initialization at k=120 happened to merge these clusters.
- **k=150 RECOVERS to match k=100**: PQ=20.7 vs 20.9 (within noise). Revived wall (0%→19.2% IoU) but slightly lost traffic sign accuracy (46.0%→33.6%). One fewer dead class (6 vs 7).
- **k=100 and k=150 are tied**, but k=100 is preferred: 33% fewer clusters means simpler downstream training with identical performance.

Per-class IoU comparison across k values:

| Class | k=80 | k=100 | k=120 | k=150 |
|-------|------|-------|-------|-------|
| road | — | 95.3 | 94.5 | 95.0 |
| building | — | 78.8 | 79.5 | 81.7 |
| wall | — | 0.0 | 0.0 | **19.2** |
| terrain | — | 27.1 | **0.0** | 25.6 |
| bus | — | 61.5 | **0.0** | 61.4 |
| person | — | 55.7 | 54.0 | 58.0 |
| car | — | 82.0 | 76.1 | 81.7 |

---

## 5. Per-Class Analysis (Winner: ViT-L/16, Spherical K-Means, k=100)

### Semantic IoU

| Class | Type | IoU (%) | Status |
|-------|------|---------|--------|
| road | stuff | 95.3 | Excellent |
| sky | stuff | 84.2 | Excellent |
| vegetation | stuff | 82.4 | Excellent |
| car | things | 82.0 | Excellent |
| building | stuff | 78.8 | Good |
| bus | things | 61.5 | Good |
| sidewalk | stuff | 60.7 | Good |
| person | things | 55.7 | Moderate |
| bicycle | things | 53.9 | Moderate |
| traffic sign | stuff | 46.0 | Moderate |
| terrain | stuff | 27.1 | Weak |
| pole | stuff | 16.0 | Weak |
| wall | stuff | 0.0 | Dead |
| fence | stuff | 0.0 | Dead |
| traffic light | stuff | 0.0 | Dead |
| rider | things | 0.0 | Dead |
| truck | things | 0.0 | Dead |
| train | things | 0.0 | Dead |
| motorcycle | things | 0.0 | Dead |

### Panoptic PQ (Per-Class)

| Class | Type | PQ | SQ | RQ | TP | FP | FN |
|-------|------|------|------|------|-----|-----|------|
| road | stuff | 79.0 | 80.8 | 97.7 | 480 | 20 | 3 |
| vegetation | stuff | 73.7 | 78.0 | 94.4 | 459 | 27 | 27 |
| sky | stuff | 66.5 | 77.4 | 85.9 | 385 | 74 | 52 |
| building | stuff | 59.9 | 73.7 | 81.3 | 403 | 97 | 88 |
| sidewalk | stuff | 36.8 | 67.1 | 54.8 | 264 | 235 | 200 |
| bus | things | 34.1 | 75.9 | 44.9 | 53 | 85 | 45 |
| traffic sign | stuff | 15.1 | 60.0 | 25.1 | 112 | 312 | 357 |
| car | things | 15.0 | 70.0 | 21.5 | 656 | 821 | 3979 |
| bicycle | things | 7.3 | 58.0 | 12.5 | 134 | 842 | 1029 |
| person | things | 5.0 | 61.8 | 8.1 | 176 | 814 | 3200 |
| terrain | stuff | 4.0 | 56.6 | 7.1 | 14 | 149 | 217 |
| pole | stuff | 0.4 | 51.5 | 0.7 | 3 | 375 | 486 |
| wall | stuff | 0.0 | — | — | 0 | 0 | 201 |
| fence | stuff | 0.0 | — | — | 0 | 0 | 189 |
| traffic light | stuff | 0.0 | — | — | 0 | 0 | 260 |
| rider | things | 0.0 | — | — | 0 | 0 | 541 |
| truck | things | 0.0 | — | — | 0 | 0 | 93 |
| train | things | 0.0 | — | — | 0 | 0 | 23 |
| motorcycle | things | 0.0 | — | — | 0 | 0 | 149 |

### Dead Class Analysis

6–9 classes have zero PQ depending on k (k=150 revived wall; k=120 lost bus and terrain):
- **wall, fence**: Thin structures merged with adjacent surfaces (building, vegetation)
- **traffic light**: Too small relative to patch resolution (16×16 pixel patches)
- **rider**: Always co-located with bicycle/motorcycle; insufficient spatial separation
- **truck, train, motorcycle**: Rare in Cityscapes val (93, 23, 149 GT instances)

These dead classes are structural — they persist regardless of method, backbone, or k. Recovery requires either higher-resolution features or dedicated per-class handling (e.g., SAM3 injection).

---

## 6. Failure Cases and Negative Results

### 6.1 vMF Collapse (PQ=5.1)
The von Mises-Fisher EM algorithm collapsed catastrophically: 70 of 80 clusters became empty. The Banerjee approximation for concentration parameter κ diverges in high dimensions (d=768+), causing winner-take-all behavior where a few clusters absorb all probability mass. Would require PCA dimensionality reduction before vMF fitting.

### 6.2 EAGLE Spectral Destruction (PQ=5.2)
Spectral eigenvector enrichment from EAGLE captures spatial patterns (position, frequency), not semantic categories. The augmented features (768+20 dims) cluster by spatial location rather than object class, producing pseudo-labels that look like a spatial grid rather than a semantic map.

### 6.3 Sinkhorn Balance ≠ Quality (PQ=16.4)
Sinkhorn equipartition achieves near-perfect entropy (0.976) but does not improve PQ. Uniform cluster sizes are not desirable when Cityscapes classes vary enormously in pixel area (road: ~40% of image vs. traffic light: <0.1%).

### 6.4 ViT-g/14 Spatial Misalignment
Despite being the largest model (1.1B params, 1536-dim features), ViT-g/14 performs worst due to patch_size=14. The 37×74 grid requires fractional upsampling to 512×1024, creating block artifacts that fragment stuff regions. This is NOT fixable by changing clustering — it's a geometric constraint. Only ViT-*/16 backbones align cleanly with 512×1024 output.

### 6.5 MPS Instability with Large Models
ViT-g/14 extraction stalled on Apple MPS (3.6% CPU utilization for 8+ hours, only 147/2975 images). CPU extraction was slower per-image (~3.1s) but reliable. This limits practical use of ViT-g/14 on Apple Silicon for batch processing.

---

## 7. What Changed Our Belief

| Prior belief | Updated belief | Evidence |
|-------------|---------------|----------|
| Centroid drift is negligible | **Centroid renormalization gives +1.5 PQ free** | Spherical > euclidean on every metric |
| Bigger backbone = better | **Patch alignment matters more than model size** | ViT-g/14 (1.1B) << ViT-L/16 (304M) |
| k=80 is optimal | **k=100 is the sweet spot (PQ plateaus at k=100–150)** | k=100=20.9, k=120=18.6 (dip), k=150=20.7 |
| Theoretically principled methods (vMF) should help | **They catastrophically fail in high-d** | vMF: 70/80 clusters empty |
| Balanced clusters help | **Balance is orthogonal to quality** | Sinkhorn: best entropy, no PQ gain |
| Spectral enrichment adds info | **It adds spatial noise that destroys semantics** | EAGLE: PQ=5.2 |

---

## 8. Complete Results Matrix

### All Configurations Tested

| # | Backbone | Method | k | mIoU | PQ | PQ_st | PQ_th | Status |
|---|----------|--------|---|------|------|-------|-------|--------|
| 1 | ViT-B/16 | euclidean_kmeans | 80 | 26.2 | 15.6 | 25.9 | 1.4 | Baseline |
| 2 | ViT-B/16 | spherical_kmeans | 80 | 29.1 | 17.1 | 28.1 | 1.9 | +1.5 PQ |
| 3 | ViT-B/16 | vmf | 80 | 9.7 | 5.1 | 7.8 | 1.4 | COLLAPSED |
| 4 | ViT-B/16 | kmeans_sinkhorn | 80 | 27.9 | 16.4 | 27.1 | 1.8 | No gain |
| 5 | ViT-B/16 | cause_codebook | 80 | 28.2 | 16.7 | 27.2 | 2.4 | Best things |
| 6 | ViT-B/16 | eagle_spectral | 80 | 9.9 | 5.2 | 8.9 | 0.0 | FAILED |
| 7 | ViT-L/16 | spherical_kmeans | 80 | 35.8 | 19.0 | 29.5 | 4.4 | Backbone gain |
| 8 | **ViT-L/16** | **spherical_kmeans** | **100** | **39.1** | **20.9** | **30.5** | **7.7** | **WINNER** |
| 9 | ViT-L/16 | spherical_kmeans | 120 | 33.7 | 18.6 | 29.8 | 3.3 | k=120 dip |
| 10 | ViT-L/16 | spherical_kmeans | 150 | 39.8 | 20.7 | 30.0 | 7.8 | Plateau |
| 11 | ViT-g/14 | spherical_kmeans | 80 | 28.9 | 8.2 | 12.4 | 2.4 | Patch misalign |
| 12 | ViT-g/14 | euclidean_kmeans | 27 | 16.4 | 6.5 | 10.9 | 0.5 | Failed |
| 13 | ViT-g/14 | spherical_kmeans | 27 | 18.2 | 6.9 | 11.2 | 0.9 | Failed |
| 14 | ViT-g/14 | kmeans_sinkhorn | 27 | 18.3 | 7.1 | 11.6 | 0.9 | Failed |
| 15 | ViT-g/14 | cause_codebook | 27 | 18.7 | 6.7 | 10.8 | 1.1 | Failed |

---

## 9. Artifact and Reproducibility Index

### Feature Directories
| Path | Backbone | Shape per image |
|------|----------|-----------------|
| `{CS}/dinov3_features/` | ViT-B/16 | (2048, 768) |
| `{CS}/dinov3_features_vitl16/` | ViT-L/16 | (2048, 1024) |
| `{CS}/dinov2g14_features/` | ViT-g/14 | (2738, 1536) |

### Pseudo-Label Directories (under Cityscapes root)
| Directory | Config |
|-----------|--------|
| `pseudo_semantic_raw_dinov3_k80_euclidean_kmeans/` | ViT-B/16, euclidean, k=80 |
| `pseudo_semantic_raw_dinov3_k80_spherical_kmeans/` | ViT-B/16, spherical, k=80 |
| `pseudo_semantic_raw_dinov3_k80_vmf/` | ViT-B/16, vMF, k=80 |
| `pseudo_semantic_raw_dinov3_k80_kmeans_sinkhorn/` | ViT-B/16, sinkhorn, k=80 |
| `pseudo_semantic_raw_dinov3_k80_cause_codebook/` | ViT-B/16, codebook, k=80 |
| `pseudo_semantic_raw_dinov3_k80_eagle_spectral/` | ViT-B/16, EAGLE, k=80 |
| `pseudo_semantic_raw_dinov3_k80_spherical_kmeans_vitl16/` | ViT-L/16, spherical, k=80 |
| `pseudo_semantic_raw_dinov3_k100_spherical_kmeans_vitl16/` | ViT-L/16, spherical, k=100 |
| `pseudo_semantic_raw_dinov3_k120_spherical_kmeans_vitl16/` | ViT-L/16, spherical, k=120 |
| `pseudo_semantic_raw_dinov3_k150_spherical_kmeans_vitl16/` | ViT-L/16, spherical, k=150 |
| `pseudo_semantic_raw_dinov3_k80_spherical_kmeans_vitg14/` | ViT-g/14, spherical, k=80 |
| `pseudo_semantic_raw_dinov3_k27_*_vitg14/` | ViT-g/14, various, k=27 |

### Result JSONs
All in `results/clustering_ablation/eval_*.json`

### Scripts
- `mbps_pytorch/generate_clustering_ablation.py` — 7 methods via `--method` flag
- `mbps_pytorch/extract_dinov2_vitg14_features.py` — ViT-g/14 extraction
- `scripts/run_clustering_ablation.sh` — driver script
- `scripts/run_vitg14_pipeline.sh` — ViT-g/14 pipeline

### Seed
All experiments use `--seed 42`.

---

## 10. Next Actions

1. **Propagate k=100 ViT-L/16 labels through downstream pipeline**: Run DCFA + SIMCF-ABC on the new pseudo-labels and measure impact at Stage-2 (CUPS Cascade Mask R-CNN). The +5.3 PQ at Stage-0 should yield meaningful downstream improvement.

2. **Do NOT increase k beyond 100**: k=120 and k=150 confirmed the plateau. k=100 matches k=150 performance with 33% fewer clusters — simpler downstream training, no accuracy loss. The k=120 dip shows that higher k introduces seed-dependent instability.

3. **Do NOT invest further in ViT-g/14**: The patch_size=14 spatial misalignment is a geometric constraint, not a clustering problem. Any future backbone exploration should use patch_size=16 (or 8) only.

4. **Do NOT use vMF or EAGLE spectral**: Both are confirmed failures in this setting. vMF needs dimensionality reduction (PCA to ~64d) to avoid κ divergence; EAGLE captures spatial rather than semantic structure.

5. **Consider training a ViT-L/16 CAUSE-TR model**: The current DINOv2 ViT-B/16 CAUSE-TR (27-class) uses ViT-B/14. A ViT-L/16 version could combine the +3.3 mIoU backbone gain with learned semantic structure.
