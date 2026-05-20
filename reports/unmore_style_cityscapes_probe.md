# unMORE-Style Cityscapes Probe

Date: 2026-05-18

## Question

Can unMORE-style center-boundary reasoning improve Cityscapes thing-instance quality before training a detector or changing MBPS?

## Constraint

This is not an official unMORE checkpoint run. The official OneDrive checkpoint downloads currently resolve to blocked 873-byte HTML files under `weights/unmore/`, and the clone under `test-instance-labels/unMORE/` does not include usable `.pth` weights. I therefore ran a training-free proxy: SAM3 fine masks provide object candidates, and the probe scores them with simple center compactness plus boundary support from depth/semantic edges.

## Protocol

- Slice: 15 Cityscapes `val/frankfurt` images with available SAM3 fine masks.
- Semantics: `pseudo_semantic_raw_k80`, mapped to Cityscapes trainIDs using `kmeans_centroids.npz`.
- Ground truth: Cityscapes `gtFine/val`.
- Thing metrics only: person, rider, car, truck, bus, train, motorcycle, bicycle.
- Saved result: `results/unmore_style_probe_val15.json`.
- Repro script: `scripts/probe_unmore_style_cityscapes.py`.
- AP protocol: COCO-style mask AP over thing classes at IoU thresholds 0.50:0.05:0.95, plus AP50/AP75. This is a slice-level diagnostic AP, not the official full Cityscapes instance-eval server.

## Results

| Method | Avg Inst/Image | PQ Things | RQ Things | Mask AP | AP50 | AP75 | Person AP | Person RQ | Person TP/FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current stored depth instances | 3.60 | 4.54 | 6.64 | 3.84 | 8.81 | 4.21 | 0.30 | 4.88 | 1/0/39 |
| Depth Sobel tau=0.20, A_min=1000 | 4.53 | 28.41 | 39.34 | 16.21 | 33.69 | 14.97 | 0.30 | 4.35 | 1/5/39 |
| Depth Sobel tau=0.01, A_min=1000 | 5.47 | 24.95 | 31.34 | 15.90 | 30.49 | 13.86 | 1.19 | 4.76 | 1/1/39 |
| SAM center-boundary proxy | 2.67 | 37.76 | 46.88 | 33.56 | 43.36 | 38.95 | 25.99 | 32.14 | 9/7/31 |
| SAM-first, depth-residual hybrid | 6.20 | 33.33 | 41.56 | 33.59 | 43.48 | 38.95 | 25.99 | 31.58 | 9/8/31 |

## Interpretation

The center-boundary style signal is worth adapting. The clearest evidence is person performance: the depth baselines find only 1 of 40 GT person instances on this slice, while the SAM center-boundary proxy finds 9; person AP rises from 0.30 on the tau=0.20 depth baseline to 25.99. This is exactly the co-planar/crowded-object failure mode the depth-only pipeline struggles with.

The naive hybrid is not the right final design. It improves over pure depth on person/rider/bicycle but adds residual depth false positives, especially for car/train, reducing PQ relative to the SAM-only proxy. The MBPS adaptation should learn a proposal gate or class-aware merge policy instead of blindly unioning center-boundary candidates with depth components.

## Recommendation

Proceed with a MBPS center-boundary adaptation, but make it a learned/gated branch:

1. Use center-boundary candidates to split small/crowded things first, especially person/rider/bicycle.
2. Keep depth/Sobel or DepthPro components as a fallback for large rigid objects, especially car, bus, truck, and train.
3. Train or calibrate a class-aware proposal gate on pseudo targets before replacing the instance pipeline.
4. Before any GPU training run, fix the existing boundary-head training path: CUDA autocast previously crashed on `binary_cross_entropy`; the robust route is boundary logits plus `binary_cross_entropy_with_logits`.
