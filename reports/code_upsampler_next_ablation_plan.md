# DCFA 90D Code Upsampler Next Ablation Plan

Date: 2026-05-20

## Objective

Test the research-driven upgrades over the current dynamic-kernel 90D upsampler baseline while keeping the semantic clustering constraint fixed at **K=80**.

Current reference:

```text
Dynamic-kernel 90D upsampler + K=80
mIoU        42.796
stuff mIoU  42.050
things mIoU 43.991
pixel acc   89.390
```

## Implemented Ablations

| Run | Paper idea | Implementation |
|---|---|---|
| `jafar_attn_90d_dcfa_v3_h64w128_k80_seed42` | JAFAR-style local cross-attention | New `AttentiveCodeUpsampler`: high-res RGB/depth/coordinate queries attend over local 90D code key/value windows with SFT modulation. |
| `loftup_coord_mask_90d_dcfa_v3_h64w128_k80_seed42` | LoftUp-style coordinates + mask self-distillation | Dynamic-kernel upsampler with Fourier coordinate features plus local class-agnostic mask affinity distillation from teacher/guidance. |
| `anyup_crop_teacher_90d_dcfa_v3_h64w128_k80_seed42` | AnyUp/FeatUp crop-local supervision | Dynamic-kernel upsampler trained with local crop teacher loss instead of only full-map teacher loss. |
| `anyup_crop_teacher_stuff_preserve_90d_dcfa_v3_h64w128_k80_seed42` | Crop teacher + stuff preservation | Crop-local supervision plus a weighted full-map cosine anchor on smooth, teacher-consistent RGB/depth/code regions. |
| `neco_neighbor_90d_dcfa_v3_h64w128_k80_seed42` | NeCo-style patch-neighbor ordering | Dynamic-kernel upsampler with KL matching of teacher patch-neighbor similarity distributions. |

## Shared Setup

```text
Cache:        outputs/code_upsampler/cache_dcfa_v3_90d_64x128
Train split:  2975 Cityscapes images
Val split:    500 Cityscapes images
Device:       MPS
Epochs:       20
Batch size:   2
K:            80
Evaluator:    mbps_pytorch/evaluate_cityscapes27_clusters.py
```

## Script

The full ablation is launched by:

```bash
DEVICE=mps PYTHONUNBUFFERED=1 scripts/run_code_upsampler_next_ablation.sh
```

The script trains each variant, fits K=80 MiniBatchKMeans on train upsampled 90D codes, writes raw val cluster PNGs, runs the strict Cityscapes-27 majority-mapping evaluator, and appends metrics to:

```text
outputs/code_upsampler/next_ablation_summary.csv
```

## Expected Outputs

### Checkpoints

```text
outputs/code_upsampler/runs/{run_name}/best.pt
outputs/code_upsampler/runs/{run_name}/last.pt
outputs/code_upsampler/runs/{run_name}/metrics.csv
```

### Raw K=80 Cluster PNGs

```text
outputs/code_upsampler/pseudo_semantic_{run_name}/val/
```

### Strict Cityscapes-27 JSONs

```text
outputs/code_upsampler/{run_name}_cityscapes27.json
```

## Live Logs

The current long run was launched in tmux:

```text
session: code_up_next_ablation
```

Useful commands:

```bash
tail -f outputs/code_upsampler/logs/next_ablation_tmux.log
tail -f outputs/code_upsampler/logs/next_ablation_20260520_133728/jafar_attn_90d_dcfa_v3_h64w128_k80_seed42_train.log
cat outputs/code_upsampler/next_ablation_summary.csv
```

## Success Criterion

A variant is useful only if it beats the current strict Cityscapes-27 score:

```text
mIoU > 42.796 at K=80
```

Secondary checks:

```text
stuff mIoU should not improve by destroying things mIoU
things mIoU should remain near or above 43.991
zero-IoU thin/rare classes should be inspected in per-class IoU
```

## Completed Results

All four full K=80 runs completed on Cityscapes val.

| Variant | mIoU | Stuff mIoU | Things mIoU | Pixel Acc | Delta vs 42.796 baseline |
|---|---:|---:|---:|---:|---:|
| JAFAR-style attentive upsampler | 41.301 | 40.944 | 41.873 | 89.285 | -1.495 |
| LoftUp-style coord + mask loss | 39.348 | 41.510 | 35.889 | 89.426 | -3.448 |
| AnyUp/FeatUp crop-teacher loss | 42.062 | 39.156 | 46.711 | 89.679 | -0.734 |
| NeCo neighbor-order loss | 36.699 | 39.474 | 32.260 | 89.421 | -6.097 |

Takeaway: none of the isolated upgrades beats the original dynamic-kernel 90D upsampler. The crop-teacher variant is the most interesting failure: it improves things mIoU to **46.711** but loses enough stuff mIoU to miss the overall baseline. This suggests the next run should combine crop-teacher training with a stuff-preserving regularizer or use crop-teacher only on high-edge / thin-object crops.

## Follow-up Prototype

The trainer now includes `--lambda_stuff_preserve`, which adds a stuff-preserving full-map teacher anchor while keeping crop-teacher supervision local. The regularizer builds a label-free smooth-region weight from two cues:

```text
stuff_weight = exp(-alpha_rgbd * RGBD_edge) * exp(-alpha_code * teacher_code_edge)
```

This makes the full-map cosine loss strongest on smooth, teacher-consistent regions, which are the regions most likely to represent road, building, vegetation, sky, sidewalk, and other stuff classes. The first full run to test is:

```bash
DEVICE=mps EPOCHS=20 BATCH_SIZE=2 scripts/run_code_upsampler_next_ablation.sh
```

For a targeted single-run launch without repeating the older variants:

```bash
DEVICE=mps EPOCHS=20 BATCH_SIZE=2 scripts/run_code_upsampler_crop_stuff_ablation.sh
```
