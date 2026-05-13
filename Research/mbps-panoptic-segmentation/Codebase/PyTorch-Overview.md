---
type: codebase-module
title: PyTorch — mbps_pytorch/ overview
project: mbps-panoptic-segmentation
module: mbps_pytorch
framework: pytorch
paths:
  - mbps_pytorch/
tags: [codebase, pytorch, overview]
related:
  - "[[00-Codebase-Map]]"
  - "[[PyTorch-Models]]"
  - "[[PyTorch-Training]]"
  - "[[PyTorch-Losses]]"
  - "[[PyTorch-Data]]"
  - "[[PyTorch-Evaluation]]"
  - "[[PyTorch-Mask2Former]]"
  - "[[RefineNet-Family]]"
  - "[[Panoptic-DeepLab]]"
  - "[[Instance-Methods]]"
---

# PyTorch — `mbps_pytorch/` overview

The PyTorch counterpart to [[JAX-mbps]]. Everything that runs on macOS (MPS / CPU), GTX-1080 Ti remote, and the A6000 Anydesk machine lives here.

## Subpackages

- [[PyTorch-Models]] — `mbps_pytorch/models/` (backbones, bridge, classifier, semantic, instance, merger, mask2former)
- [[PyTorch-Mask2Former]] — `mbps_pytorch/models/mask2former/` (broken out for clarity)
- [[PyTorch-Training]] — `mbps_pytorch/training/`
- [[PyTorch-Losses]] — `mbps_pytorch/losses/`
- [[PyTorch-Data]] — `mbps_pytorch/data/`
- [[PyTorch-Evaluation]] — `mbps_pytorch/evaluation/`
- [[Instance-Methods]] — `mbps_pytorch/instance_methods/` (15 files)

## Major root-level entry points

| Script | Role |
|--------|------|
| `train_refine_net.py` | Train CSCMRefineNet / HiResRefineNet / DepthGuidedUNet. See [[RefineNet-Family]]. |
| `train_mobile_panoptic.py` | Mobile RepViT + BiFPN training (final pipeline used in [[Reports-Index]]). |
| `train_panoptic_deeplab.py` | Panoptic-DeepLab variant — see [[Panoptic-DeepLab]]. |
| `panoptic_deeplab.py` | Architectures (4 panoptic) + 3 FPN types. |
| `refine_net.py` | `CSCMRefineNet`, `HiResRefineNet`, `DepthGuidedUNet`. |
| `joint_refine_net.py` | Joint semantic-instance refinement. |
| `eval_cause_k80.py` | Evaluate CAUSE k=80 pseudo-labels (see also [[Evaluation-Scripts]]). |
| `evaluate_cascade_pseudolabels.py` | Evaluate Cascade Mask R-CNN pseudo-labels. |
| `generate_depth_overclustered_semantics.py` | Depth-conditioned overclustered semantic pseudo-labels. |
| `generate_*` family | See [[Generators-Semantic]] and [[Generators-Instance]]. |
| `train_*` family | Various trainers (panoptic-deeplab, picl, dinosaur, joint refine net, neco, sinder, etc.). |

## Excluded from this graph

- `mbps_pytorch/mamba2/` — pure-PyTorch Mamba2 + GatedDeltaNet, including their vision wrappers and `ttt_mamba2_refiner.py`.
- `mbps_pytorch/models/adapters/` and any `train_*adapter*.py` / `train_*lora*.py` — the entire DoRA / LoRA stack.
- Adapter smoke tests (`tests/smoke_test_*adapter*.py`, `tests/test_adapters.py`, `tests/test_depth_adapters.py`, `tests/test_dora_*`) — included on disk, hidden from this graph.
- `T0`, `T1`, `T2`, … enumerated ablations in `ablations/` and any references to them in `reports/`.

These exclusions are noted at the boundary of each affected note (e.g. [[PyTorch-Training]], [[PyTorch-Models]]).
