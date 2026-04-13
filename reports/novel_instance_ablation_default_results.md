# Novel Instance Decomposition — Default Config Ablation Results

Date: 2026-03-28
Dataset: Cityscapes val (500 images)
Semantics: k=80 overclustered pseudo-labels
Depth: SPIdepth
Features: DINOv2 ViT-B/14 (2048 patches, 768-dim)
Eval: 512x1024, 19-class trainID, PQ with IoU>0.5 matching

## Summary Table

| Method          |     PQ |   PQ_st |   PQ_th | inst/img |  s/img | Config |
|-----------------|--------|---------|---------|----------|--------|--------|
| sobel_cc **(baseline)** |  26.74 |   32.08 |   19.41 |      4.3 |   0.03 | grad_threshold=0.20, min_area=1000 |
| morse           |  25.58 |   32.08 |   16.66 |      2.3 |   0.29 | min_basin_depth=0.03, merge_threshold=0.80 |
| tda             |  25.04 |   32.08 |   15.37 |      3.1 |   0.27 | tau_persist=0.05, filtration=depth_direct |
| mumford_shah    |  24.27 |   32.08 |   13.54 |      5.8 |   3.58 | alpha=1.0, beta=0.1, n_clusters=20, res=64x128 |
| contrastive     |  21.06 |   32.08 |    5.92 |      5.7 |   0.06 | hdbscan_min_cluster=5, min_samples=3 |
| ot              |  18.86 |   32.08 |    0.69 |     10.9 |   0.12 | K_proto=15, epsilon=0.1, depth_scale=10.0 |

## Per-Class PQ_things Comparison

| Class        |      sobel_cc |         morse |           tda |  mumford_shah |   contrastive |            ot |
|--------------|---------------|---------------|---------------|---------------|---------------|---------------|
| person       |   4.0 (123/277/3253) |   0.9 (26/190/3350) |   1.0 (31/233/3345) |   4.3 (136/306/3240) |   1.9 (60/401/3316) |   0.6 (19/621/3357) |
| rider        |   9.2 (57/113/484) |   3.8 (22/118/519) |   4.1 (24/116/517) |   7.7 (48/123/493) |   2.6 (15/101/526) |   0.0 (0/76/541) |
| car          |  16.5 (648/386/3987) |   4.2 (159/291/4476) |   8.0 (318/409/4317) |  17.7 (794/872/3841) |  11.6 (545/1242/4090) |   3.4 (244/3786/4391) |
| truck        |  35.5 (33/16/60) |  33.8 (31/13/62) |  29.7 (28/21/65) |  18.5 (24/69/69) |   8.9 (12/77/81) |   0.0 (0/146/93) |
| bus          |  47.8 (48/12/50) |  45.0 (43/7/55) |  40.0 (41/17/57) |  31.3 (43/59/55) |  12.4 (18/86/80) |   0.7 (2/228/96) |
| train        |  36.4 (11/9/12) |  44.2 (12/4/11) |  37.5 (11/7/12) |  22.8 (10/25/13) |   6.1 (2/25/21) |   0.0 (0/60/23) |
| motorcycle   |   0.0 (0/0/149) |   0.0 (0/0/149) |   0.0 (0/0/149) |   0.0 (0/0/149) |   0.0 (0/0/149) |   0.0 (0/0/149) |
| bicycle      |   5.8 (77/323/1086) |   1.5 (18/212/1145) |   2.8 (34/238/1129) |   6.0 (81/325/1082) |   3.9 (48/198/1115) |   0.9 (12/281/1151) |

*Format: PQ (TP/FP/FN)*

## Method Descriptions

1. **sobel_cc** (baseline): Sobel gradient on depth -> threshold -> connected components. Textbook image processing.
2. **morse**: Watershed basins on depth (h-minima suppressed) + DINOv2 feature-based merge of adjacent same-class instances.
3. **tda**: Persistence-guided watershed — oversegment then merge boundaries below persistence threshold tau via union-find on RAG.
4. **mumford_shah**: Spectral clustering on depth+feature affinity graph (4-connected, Gaussian kernel). Energy: alpha*depth_var + beta*feat_var + gamma*boundary.
5. **contrastive** (CE-raw): HDBSCAN clustering on L2-normalized DINOv2 features per thing class. No learned projection (raw features only).
6. **ot**: Optimal transport (Sinkhorn) assigning pixels to K-means prototypes in [feature, depth, position] descriptor space.

## Analysis

- **PQ_stuff is identical (32.08)** across all methods — correct, since only instance decomposition differs.
- **Sobel+CC baseline is hard to beat at default configs** — PQ_things=19.41 vs next best Morse=16.66.
- **Morse and TDA under-segment** — 2.3 and 3.1 inst/img vs baseline 4.3. Default params merge too aggressively.
- **OT produces NaN-free results** after log-domain Sinkhorn fix, but PQ_things=0.69 suggests bad prototype-pixel assignment.
- **Contrastive (raw HDBSCAN)** finds more instances (5.7/img) but many are false positives (car FP=779).
- **Key bottleneck classes**: person (PQ=0.9-4.0), motorcycle (PQ=0.0 everywhere), bicycle (PQ=0-5.8).
- **Hyperparameter sweeps needed** — these are single default configs. Morse/TDA especially should improve with lower merge thresholds.

## Timing

| Method | s/img | Total (500 imgs) |
|--------|-------|-----------------|
| sobel_cc | 0.03 | 15s (0.2min) |
| morse | 0.29 | 147s (2.5min) |
| tda | 0.27 | 133s (2.2min) |
| mumford_shah | 3.58 | 1789s (29.8min) |
| contrastive | 0.06 | 30s (0.5min) |
| ot | 0.12 | 58s (1.0min) |

## Next Steps

1. Run hyperparameter sweeps for all methods (--sweep flag)
2. Morse sweep: min_basin_depth x merge_threshold (56 configs)
3. TDA sweep: tau_persist x filtration_mode x min_area (36 configs)
4. OT sweep: K_proto x epsilon x depth_scale (72 configs)
5. Mumford-Shah sweep: alpha x beta x n_clusters (36 configs)
6. Contrastive sweep: min_cluster x min_samples x min_area (24 configs)
7. Implement slot attention method (Method 3)
8. Implement combined Morse+Contrastive two-stage method

