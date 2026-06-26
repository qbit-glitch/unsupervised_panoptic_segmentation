# DCFA 90D Code-Space Upsampler Ablation Report

Date: 2026-05-20

## Summary

We tested whether a lightweight AnyUp-style upsampler can improve DCFA/CAUSE semantic segmentation without increasing the number of clusters. The constraint was fixed throughout: **K=80 only**. Two 90D code-space upsamplers were trained and evaluated:

1. **Residual 90D upsampler**: bilinear-upsampled DCFA 90D codes plus learned RGB/depth-guided residual correction.
2. **Dynamic-kernel 90D upsampler**: AnyUp-style per-pixel local kernel reassembly of nearby 90D codes, followed by lightweight residual refinement.

The dynamic-kernel upsampler is the clear winner. It improves strict Cityscapes-27 mIoU from the previous DCFA K=80 baseline of **40.114** to **42.796**, a gain of **+2.682 mIoU**, while keeping K fixed at 80.

## Experimental Setup

### Input Representation

The experiment operates on continuous semantic code vectors, not labels:

```text
Cityscapes image
  -> frozen DINOv2 + CAUSE-TR Segment_TR
  -> frozen DCFA depth adapter
  -> low-resolution 90D DCFA code map
  -> learned 90D upsampler
  -> high-resolution 90D code map
  -> K=80 clustering
  -> Cityscapes-27 majority-mapping evaluation
```

The cache contains paired low/high DCFA code maps:

| Split | Cached files |
|---|---:|
| train | 2975 |
| val | 500 |
| total | 3475 |

Cache path:

```text
outputs/code_upsampler/cache_dcfa_v3_90d_64x128/
```

### Training Targets And Losses

The upsampler was trained on a dense frozen-teacher cache. The cache target is produced from sliding-window/crop-derived CAUSE-TR features, but the winning dynamic-kernel baseline **does not** use the later `--crop_teacher` loss. It uses full-map feature supervision against the cached dense teacher.

```text
low_code:      low-resolution adapted DCFA 90D code map
teacher_code:  denser adapted DCFA 90D code map at 64x128
guidance:      RGB + DepthPro depth at 64x128
```

For one training sample, define:

```text
Z_l  in R^{90 x H_l x W_l}       low-resolution DCFA code
Z_t  in R^{90 x H x W}           dense frozen DCFA teacher code
G    in R^{4 x H x W}            RGB/depth guidance, 3 RGB channels + 1 depth channel
Z_p  in R^{90 x H x W}           predicted high-resolution code from the upsampler
```

The winning dynamic-kernel baseline used:

```text
L_total =
  L_teacher_cosine
+ lambda_down * L_downsample_consistency
+ lambda_edge * L_boundary_smoothness
+ lambda_var * L_variance_preservation
```

with the following exact weights:

```text
lambda_down = 0.25
lambda_edge = 0.02
lambda_var  = 0.05
edge_alpha  = 12.0
```

Implementation:

```text
mbps_pytorch/train_code_upsampler.py
  cosine_code_loss()
  downsample_consistency_loss()
  boundary_smoothness_loss()
  variance_preservation_loss()
```

Run config:

```text
outputs/code_upsampler/runs/dynamic_kernel_90d_dcfa_v3_h64w128_k80_seed42/args.json
```

#### 1. Dense Teacher Cosine Loss

The main supervision term matches the predicted 90D code to the frozen dense DCFA teacher code at each high-resolution location:

```text
normalize(z) = z / (||z||_2 + eps)

L_teacher_cosine =
  1 - mean_{u in Omega}
        < normalize(Z_p[:, u]), normalize(Z_t[:, u]) >
```

where `Omega` is the `H x W` spatial grid. In code this is `cosine_code_loss(pred, teacher)`.

Why we use it:

```text
The output is a continuous 90D semantic code map, not a class-logit map.
Therefore the primary target is feature-space agreement with the frozen
DCFA/CAUSE teacher, not cross-entropy against labels.
```

Source / inspiration:

```text
Feature-distillation and feature-upsampling supervision. This is aligned with
FeatUp's and AnyUp's core objective of learning high-resolution features that
preserve the semantics of the original foundation-model feature space.
```

References:

```text
FeatUp: A Model-Agnostic Framework for Features at Any Resolution, ICLR 2024
https://arxiv.org/abs/2403.10516

AnyUp: Universal Feature Upsampling, ICLR 2026
https://wimmerth.github.io/anyup/
```

#### 2. Downsample Consistency Loss

The predicted high-resolution code is downsampled back to the low-resolution DCFA grid and matched to the original low-resolution code:

```text
D(Z_p) = bilinear_downsample(Z_p, size=(H_l, W_l))

L_downsample_consistency =
  1 - mean_{v in Omega_l}
        < normalize(D(Z_p)[:, v]), normalize(Z_l[:, v]) >
```

where `Omega_l` is the low-resolution grid. In code this is `downsample_consistency_loss(pred, low_code)`.

Why we use it:

```text
The upsampler should add spatial detail without changing the original DCFA
semantic identity. This loss anchors the high-resolution output to the original
low-resolution feature field, preventing the learned upsampler from hallucinating
codes that no longer reduce to the input representation.
```

Source / inspiration:

```text
Feature-consistency constraints used by feature upsampling methods. FeatUp uses
multi-view feature consistency, while AnyUp uses feature-consistency
regularization and crop-based supervision. Our version is a lightweight
single-scale consistency anchor: high-res output must collapse back to the
original low-res feature map.
```

References:

```text
FeatUp: A Model-Agnostic Framework for Features at Any Resolution, ICLR 2024
https://arxiv.org/abs/2403.10516

AnyUp: Universal Feature Upsampling, ICLR 2026
https://wimmerth.github.io/anyup/
```

#### 3. RGB/Depth Edge-Aware Boundary Smoothness Loss

The predicted code map is encouraged to be smooth inside homogeneous regions, but the smoothness penalty is reduced at strong RGB/depth edges:

```text
Delta_x Z_p[u] = |Z_p[:, y, x+1] - Z_p[:, y, x]|
Delta_y Z_p[u] = |Z_p[:, y+1, x] - Z_p[:, y, x]|

Delta_x G[u] = mean_c |G[c, y, x+1] - G[c, y, x]|
Delta_y G[u] = mean_c |G[c, y+1, x] - G[c, y, x]|

w_x[u] = exp(-edge_alpha * Delta_x G[u])
w_y[u] = exp(-edge_alpha * Delta_y G[u])

L_boundary_smoothness =
  mean_u w_x[u] * mean_d Delta_x Z_p[d, u]
+ mean_u w_y[u] * mean_d Delta_y Z_p[d, u]
```

Here `G` contains RGB and DepthPro depth, so the edge gates are cross-modal. In code this is `boundary_smoothness_loss(pred, guidance, edge_alpha=12.0)`.

Why we use it:

```text
Raw feature upsampling can create noisy high-frequency code fields. We want
nearby pixels inside the same visual/depth region to share similar 90D codes,
while allowing sharp semantic changes at image/depth boundaries.
```

Source / inspiration:

```text
Edge-aware smoothness from self-supervised monocular depth estimation and
boundary-aware dense prediction. The exact formula is adapted to 90D semantic
codes and uses both RGB and depth gradients rather than only RGB image gradients.
```

Reference:

```text
Godard et al., Digging Into Self-Supervised Monocular Depth Estimation,
ICCV 2019 (Monodepth2)
https://openaccess.thecvf.com/content_ICCV_2019/papers/Godard_Digging_Into_Self-Supervised_Monocular_Depth_Estimation_ICCV_2019_paper.pdf
```

#### 4. Variance Preservation Loss

The predicted 90D code field is prevented from collapsing by requiring each channel's spatial standard deviation to stay above half the teacher's spatial standard deviation:

```text
sigma_p[d] = std_{u in Omega}(Z_p[d, u])
sigma_t[d] = stopgrad(std_{u in Omega}(Z_t[d, u]))

L_variance_preservation =
  mean_d max(0, 0.5 * sigma_t[d] - sigma_p[d])
```

In code this is `variance_preservation_loss(pred, teacher)`.

Why we use it:

```text
The K=80 clustering stage needs a non-collapsed, well-spread feature space.
If the upsampler over-smooths the 90D codes, KMeans loses rare classes and
thin structures. This term keeps the predicted code distribution from becoming
too flat relative to the teacher.
```

Source / inspiration:

```text
Anti-collapse variance regularization. This is inspired by VICReg's variance
term, but simplified for dense 90D feature maps: we keep only a per-channel
variance floor and do not use VICReg's covariance/decorrelation term.
```

Reference:

```text
Bardes, Ponce, LeCun, VICReg: Variance-Invariance-Covariance Regularization
for Self-Supervised Learning, ICLR 2022
https://openreview.net/forum?id=xm6YD62D1Ub
```

#### Reproduction Notes

The original dynamic-kernel baseline did **not** use these later ablation losses:

```text
lambda_mask = 0.0
lambda_neco = 0.0
crop_teacher = false
use_coords = false
```

The exact training command can be reconstructed as:

```bash
python3 mbps_pytorch/train_code_upsampler.py \
  --variant dynamic \
  --cache_dir outputs/code_upsampler/cache_dcfa_v3_90d_64x128 \
  --output_dir outputs/code_upsampler/runs \
  --run_name dynamic_kernel_90d_dcfa_v3_h64w128_k80_seed42 \
  --epochs 20 \
  --batch_size 4 \
  --lr 2e-4 \
  --weight_decay 1e-4 \
  --lambda_down 0.25 \
  --lambda_edge 0.02 \
  --lambda_var 0.05 \
  --edge_alpha 12.0 \
  --hidden_ch 64 \
  --guidance_hidden 32 \
  --num_blocks 3 \
  --kernel_size 3 \
  --residual_scale 0.25 \
  --seed 42 \
  --device mps
```

Both models were trained for 20 epochs. Training used MPS for the upsampler models after the MPS compatibility fix. The train cache was generated with CPU frozen feature extraction.

### Model Sizes

| Model | Trainable params |
|---|---:|
| Residual 90D upsampler | 382,618 |
| Dynamic-kernel 90D upsampler | 385,219 |

## Training Results

| Model | Best epoch | Best val loss | Final val loss |
|---|---:|---:|---:|
| Residual 90D upsampler | 20 | 0.01068 | 0.01068 |
| Dynamic-kernel 90D upsampler | 19 | 0.01039 | 0.01060 |

The dynamic-kernel model had the better feature-reconstruction validation loss, and this translated into the best downstream segmentation score.

## Cityscapes-27 K=80 Evaluation

All rows below use the same strict Cityscapes-27 evaluator and majority cluster-to-class mapping.

| Variant | mIoU | Stuff mIoU | Things mIoU | Pixel Acc |
|---|---:|---:|---:|---:|
| DINOv3 + AnyUp + K=80 | 20.263 | 26.148 | 10.846 | 85.697 |
| Previous DCFA K=80 strict eval | 40.114 | 37.666 | 44.031 | 89.381 |
| Residual 90D upsampler + K=80 | 39.955 | 39.099 | 41.323 | 89.247 |
| Dynamic-kernel 90D upsampler + K=80 | **42.796** | **42.050** | **43.991** | **89.390** |

## Interpretation

The residual upsampler did not improve overall mIoU relative to the previous DCFA K=80 strict evaluation. It improved stuff mIoU from **37.666** to **39.099**, but it reduced things mIoU from **44.031** to **41.323**.

The dynamic-kernel upsampler improved overall mIoU to **42.796**, mainly by improving stuff and rare/non-standard classes while preserving things mIoU almost exactly:

```text
Overall mIoU:  +2.682
Stuff mIoU:    +4.384
Things mIoU:   -0.040
Pixel Acc:     +0.009
```

This supports the core hypothesis: **upsampling the DCFA/CAUSE 90D semantic code space with content-adaptive kernels is more useful than using raw AnyUp-upsampled DINOv3 features alone**.

One nuance: the previous DCFA strict baseline used the existing DCFA generation path, which clusters the DCFA-adjusted semantic code with explicit depth features in the representation. The new upsampler evaluation clusters the learned high-resolution 90D code map. Despite not clustering the extra 16D depth encoding directly, the dynamic-kernel 90D upsampler improves overall mIoU.

## Per-Class Results

| Class | Previous DCFA | Residual 90D | Dynamic-kernel 90D | Dynamic - DCFA |
|---|---:|---:|---:|---:|
| road | 94.563 | 93.693 | 93.958 | -0.605 |
| sidewalk | 62.619 | 60.006 | 61.691 | -0.928 |
| parking | 0.000 | 0.000 | 27.420 | +27.420 |
| rail_track | 0.000 | 0.000 | 0.000 | +0.000 |
| building | 82.182 | 82.090 | 83.497 | +1.315 |
| wall | 48.479 | 48.833 | 48.984 | +0.505 |
| fence | 47.520 | 42.869 | 47.343 | -0.177 |
| guard_rail | 0.000 | 0.000 | 0.000 | +0.000 |
| bridge | 0.000 | 49.318 | 50.318 | +50.318 |
| tunnel | 0.000 | 0.000 | 0.000 | +0.000 |
| pole | 11.436 | 0.000 | 12.695 | +1.259 |
| polegroup | 0.000 | 0.000 | 0.000 | +0.000 |
| traffic_light | 0.000 | 0.000 | 0.000 | +0.000 |
| traffic_sign | 44.238 | 42.958 | 44.301 | +0.063 |
| vegetation | 80.164 | 81.639 | 82.112 | +1.948 |
| terrain | 51.251 | 47.295 | 45.491 | -5.760 |
| sky | 80.202 | 76.891 | 74.985 | -5.217 |
| person | 52.861 | 55.870 | 55.883 | +3.022 |
| rider | 30.339 | 0.000 | 24.841 | -5.498 |
| car | 80.175 | 81.086 | 81.194 | +1.019 |
| truck | 75.510 | 74.895 | 74.078 | -1.432 |
| bus | 80.242 | 80.414 | 80.731 | +0.489 |
| caravan | 0.000 | 0.000 | 0.000 | +0.000 |
| trailer | 0.000 | 0.000 | 0.000 | +0.000 |
| train | 73.749 | 73.683 | 73.850 | +0.101 |
| motorcycle | 0.000 | 0.000 | 0.000 | +0.000 |
| bicycle | 47.435 | 47.285 | 49.334 | +1.899 |

## What Improved

The dynamic-kernel model recovered several classes that were previously zero or weak:

| Class | Previous DCFA | Dynamic-kernel |
|---|---:|---:|
| bridge | 0.000 | 50.318 |
| parking | 0.000 | 27.420 |
| pole | 11.436 | 12.695 |
| person | 52.861 | 55.883 |
| bicycle | 47.435 | 49.334 |
| vegetation | 80.164 | 82.112 |
| building | 82.182 | 83.497 |

The gain is not coming from simply improving head classes. It is mostly from the code upsampler creating better spatially separated high-resolution semantic regions for classes that the previous K=80 clustering either missed or merged into larger classes.

## What Still Fails

The following classes remain at zero IoU even after dynamic-kernel upsampling:

```text
rail_track
guard_rail
tunnel
polegroup
traffic_light
caravan
trailer
motorcycle
```

The most important unresolved failures are:

1. **traffic_light**: still zero, despite being a thin visual class that should benefit from upsampling.
2. **motorcycle**: still zero, likely due to low frequency and cluster absorption by bicycle/rider/car-like clusters.
3. **rider**: dynamic-kernel recovers nonzero IoU, but remains below the previous DCFA score.
4. **sky and terrain**: dynamic-kernel loses some IoU compared with previous DCFA, suggesting the learned high-resolution code field may split or reassign broad stuff regions.

## Artifact Paths

### Code

```text
mbps_pytorch/code_upsampler/models.py
mbps_pytorch/build_dcfa_code_upsampler_cache.py
mbps_pytorch/train_code_upsampler.py
mbps_pytorch/generate_code_upsampler_kmeans.py
scripts/continue_code_upsampler_after_cache.sh
scripts/run_code_upsampler_ablation.sh
```

### Trained Checkpoints

```text
outputs/code_upsampler/runs/residual_90d_dcfa_v3_h64w128_k80_seed42/best.pt
outputs/code_upsampler/runs/dynamic_kernel_90d_dcfa_v3_h64w128_k80_seed42/best.pt
```

### Generated K=80 Pseudo-Labels

```text
outputs/code_upsampler/pseudo_semantic_residual_90d_dcfa_v3_h64w128_k80_seed42/val/
outputs/code_upsampler/pseudo_semantic_dynamic_kernel_90d_dcfa_v3_h64w128_k80_seed42/val/
```

Each directory contains 500 Cityscapes val raw cluster PNGs.

### Evaluation JSONs

```text
outputs/code_upsampler/residual_90d_dcfa_v3_h64w128_k80_cityscapes27.json
outputs/code_upsampler/dynamic_kernel_90d_dcfa_v3_h64w128_k80_cityscapes27.json
```

### KMeans Centers

```text
outputs/code_upsampler/pseudo_semantic_residual_90d_dcfa_v3_h64w128_k80_seed42/kmeans_centers.npz
outputs/code_upsampler/pseudo_semantic_dynamic_kernel_90d_dcfa_v3_h64w128_k80_seed42/kmeans_centers.npz
```

## Commands Used

Generate residual K=80 pseudo-labels:

```bash
python3 mbps_pytorch/generate_code_upsampler_kmeans.py \
  --cache_dir outputs/code_upsampler/cache_dcfa_v3_90d_64x128 \
  --checkpoint outputs/code_upsampler/runs/residual_90d_dcfa_v3_h64w128_k80_seed42/best.pt \
  --output_dir outputs/code_upsampler/pseudo_semantic_residual_90d_dcfa_v3_h64w128_k80_seed42 \
  --k 80 \
  --sample_frac 0.025 \
  --inference_batch_size 8 \
  --device mps
```

Generate dynamic-kernel K=80 pseudo-labels:

```bash
python3 mbps_pytorch/generate_code_upsampler_kmeans.py \
  --cache_dir outputs/code_upsampler/cache_dcfa_v3_90d_64x128 \
  --checkpoint outputs/code_upsampler/runs/dynamic_kernel_90d_dcfa_v3_h64w128_k80_seed42/best.pt \
  --output_dir outputs/code_upsampler/pseudo_semantic_dynamic_kernel_90d_dcfa_v3_h64w128_k80_seed42 \
  --k 80 \
  --sample_frac 0.025 \
  --inference_batch_size 8 \
  --device mps
```

Evaluate residual:

```bash
python3 mbps_pytorch/evaluate_cityscapes27_clusters.py \
  --pred_dir outputs/code_upsampler/pseudo_semantic_residual_90d_dcfa_v3_h64w128_k80_seed42/val \
  --gt_dir /Users/qbit-glitch/Desktop/datasets/cityscapes/gtFine/val \
  --num_clusters 80 \
  --output outputs/code_upsampler/residual_90d_dcfa_v3_h64w128_k80_cityscapes27.json
```

Evaluate dynamic-kernel:

```bash
python3 mbps_pytorch/evaluate_cityscapes27_clusters.py \
  --pred_dir outputs/code_upsampler/pseudo_semantic_dynamic_kernel_90d_dcfa_v3_h64w128_k80_seed42/val \
  --gt_dir /Users/qbit-glitch/Desktop/datasets/cityscapes/gtFine/val \
  --num_clusters 80 \
  --output outputs/code_upsampler/dynamic_kernel_90d_dcfa_v3_h64w128_k80_cityscapes27.json
```

## Conclusion

The lightweight 90D code-space upsampling experiment succeeded. The residual version is useful as a baseline, but the dynamic-kernel AnyUp-style version is the one worth carrying forward. It produces the best strict Cityscapes-27 semantic score so far under the fixed K=80 constraint:

```text
Dynamic-kernel 90D upsampler + K=80:
  mIoU        42.796
  stuff mIoU  42.050
  things mIoU 43.991
  pAcc        89.390
```

The result validates the idea that adaptive, image/depth-guided reassembly of the DCFA 90D semantic code space can improve semantic clustering without increasing cluster count. The next technical bottleneck is not average quality; it is targeted recovery of still-dead classes, especially traffic light and motorcycle.

## Recommended Next Steps

1. Use the dynamic-kernel 90D upsampler output as the new semantic pseudo-label candidate for downstream MBPS/CUPS-style evaluation.
2. Add a rare-class targeted loss or sampling policy during upsampler training, while keeping K=80 fixed.
3. Inspect cluster-to-class support for traffic light and motorcycle to determine whether they receive no clusters or are absorbed by nearby classes.
4. Compare panoptic quality after combining dynamic-kernel semantics with the existing instance pipeline.
5. Save qualitative visualizations for the classes that changed most: bridge, parking, pole, person, bicycle, sky, terrain, and rider.
