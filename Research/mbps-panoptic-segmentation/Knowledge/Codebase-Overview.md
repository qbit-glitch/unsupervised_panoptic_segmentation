---
type: meta
title: Codebase Overview - mbps_panoptic_segmentation
project: mbps-panoptic-segmentation
language: en
updated: 2026-03-29T13:40:24Z
---

# Codebase Overview

- **Repository root**: `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation`
- **Framework**: JAX/Flax (TPU training), PyTorch (pseudo-label generation, local experiments)
- **Python env**: `/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python` (Python 3.10, PyTorch 2.10, MPS)

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `mbps/` | JAX/Flax model code (MBPSModel, losses, training, curriculum) |
| `mbps_pytorch/` | PyTorch experiments (pseudo-labels, RefineNet, UNet, mobile, ablations) |
| `mbps_pytorch/instance_methods/` | 6 instance decomposition methods (sobel_cc, morse, tda, mumford_shah, contrastive, ot) |
| `mbps_pytorch/mamba2/` | Pure PyTorch Mamba2 + GatedDeltaNet + Vision wrappers |
| `configs/` | YAML configs (default, dataset, ablation overrides) |
| `scripts/` | Orchestration, training entry points, evaluation |
| `refs/` | Reference implementations (CUPS, DINOv3, CutLER, CuVLER, etc.) |
| `reports/` | 36+ detailed experiment reports (.md) |
| `results/` | JSON evaluation results |
| `guidelines/` | Architecture specs, implementation guides |

## Critical Files

| File | Purpose |
|------|---------|
| `mbps/models/mbps_model.py` | Main JAX model architecture |
| `mbps/training/trainer.py` | JAX training loop + W&B logging |
| `mbps_pytorch/refine_net.py` | CSCMRefineNet + HiResRefineNet + DepthGuidedUNet |
| `mbps_pytorch/train_refine_net.py` | RefineNet/UNet training (--model_type cscm/hires/unet) |
| `mbps_pytorch/panoptic_deeplab.py` | 4 panoptic architectures + 3 FPN types |
| `mbps_pytorch/train_mobile_panoptic.py` | Mobile pipeline (RepViT+BiFPN) |
| `mbps_pytorch/ablate_instance_methods.py` | Instance decomposition sweep runner |
| `mbps_pytorch/mumford_shah_phase_b.py` | Phase B validation (500 imgs) |
| `refs/cups/` | CUPS CVPR 2025 reference implementation |
| `refs/dinov3/` | Official DINOv3 backbone |
| `scripts/orchestrate.py` | Multi-VM TPU orchestration |

## Conventions
- JAX code: `nn.Module` with `setup()` (not `@nn.compact`), `jax.pmap` for data parallelism
- PyTorch code: Standard `nn.Module`, MPS-compatible
- Config: YAML with deep merge (default + dataset + ablation)
- GCS I/O: Always use `tf.io.gfile` for cloud compatibility
- Logging: W&B for metrics, `absl.logging` for console
