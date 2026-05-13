---
type: codebase-module
title: RefineNet Family (CSCM / HiRes / DepthGuidedUNet)
project: mbps-panoptic-segmentation
module: mbps_pytorch.refine_net
framework: pytorch
paths:
  - mbps_pytorch/refine_net.py
  - mbps_pytorch/train_refine_net.py
  - mbps_pytorch/joint_refine_net.py
  - mbps_pytorch/train_joint_refine_net.py
tags: [codebase, pytorch, refinement, semantic, unet]
related:
  - "[[PyTorch-Overview]]"
  - "[[PyTorch-Models]]"
  - "[[PyTorch-Training]]"
  - "[[PyTorch-Losses]]"
  - "[[Generators-Semantic]]"
  - "[[Reports-Index]]"
---

# RefineNet Family

Three semantic-refinement architectures sharing one training script.

| Architecture | Class | Best result on Cityscapes |
|--------------|-------|---------------------------|
| `CSCMRefineNet` | Conv2d-based, 32×64 features | PQ ≈ 21.87 (CAUSE-CRF), 26.52 (Run D ep16) on k=80 |
| `HiResRefineNet` | 128×256 transposed-conv decoder | PQ = 27.50 |
| `DepthGuidedUNet` (P2-B) | progressive multi-stage decoder + 2-stage attention + depth Sobel skips | **PQ = 28.00** (beats CUPS 27.8 on 19-class metric) |

## Files

| Path | Role |
|------|------|
| `mbps_pytorch/refine_net.py` | All three architectures live here. Includes `InstanceSkipBlock` for IC-C (instance-conditioned) variant (pivoted away). |
| `mbps_pytorch/train_refine_net.py` | Common trainer with `--model_type cscm | hires | unet`. CLI flags include `--use_instance`, `--inst_skip_dim`, `--instance_subdir`, `--lambda_instance_uniform`. |
| `mbps_pytorch/joint_refine_net.py` | Joint semantic + instance refinement. |
| `mbps_pytorch/train_joint_refine_net.py` | Trainer for the joint variant. |

## Inputs

- DINOv2 features (84 / 192 / 384 / 768 dim depending on backbone)
- Depth maps (SPIdepth, DepthAnything v3, DepthPro, ZoeDepth) — see [[Refs-Depth]]
- Pseudo-labels from [[Generators-Semantic]] (`pseudo_semantic_raw_k80/`, `pseudo_semantic_adapter_V3_k80/`, etc.)

The trainer **never** consumes target labels as input — see the "RefineNet identity shortcut" lesson in `MEMORY.md`.

## Pivot — IC-C (Instance-Conditioned) variant

`InstanceSkipBlock` injects boundary + distance features through skip connections, with an `instance_uniformity_loss`. Pivoted away because semantics are already strong on 19-class; instance bottleneck is the actual problem (see [[Instance-Methods]] and the `option_c_pipeline_status` memory).

## Reports

- `reports/cscmrefinenet_k80_ablation.md`
- `reports/unet_unified_ablation_study.md`
- `reports/unet_phase2_architecture_ablation.md`
- `reports/hires_refinenet_128x256.md`
- `reports/ub_transposed_conv_detailed.md`
- See [[Reports-Index]] for the full list.

## ⚠️ Caveat on the PQ=28.00 headline

The 19-class PQ for DepthGuidedUNet **is not directly comparable** to CUPS PQ=27.8 (which uses the 27-class CAUSE + Hungarian protocol). On the same 27-class metric, CUPS=38.59 ≫ UNet=28.00. See the project-memory note "Critical Metric Warning".
