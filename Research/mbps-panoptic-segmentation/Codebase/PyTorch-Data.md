---
type: codebase-module
title: PyTorch — mbps_pytorch/data/
project: mbps-panoptic-segmentation
module: mbps_pytorch.data
framework: pytorch
paths:
  - mbps_pytorch/data/
tags: [codebase, pytorch, data]
related:
  - "[[PyTorch-Overview]]"
  - "[[Datasets]]"
  - "[[JAX-mbps-Data]]"
---

# PyTorch — `mbps_pytorch/data/`

| File | Role |
|------|------|
| `datasets.py` | Generic dataset loader wrapper. |
| `depth_cache.py` | Caches precomputed depth maps. |
| `panoptic_dataset.py` | Cityscapes panoptic dataset with pseudo-labels for [[PyTorch-Mask2Former|Mask2Former]] training. |
| `refiner_dataset.py` | Dataset loader for [[RefineNet-Family]]. |
| `transforms.py` | PyTorch-native augmentation (rotation, crop, color jitter). |

## Pseudo-label sources

The pseudo-labels these loaders consume come from [[Pseudo-Label-Pipeline]] — specifically [[Generators-Semantic]] (e.g. `pseudo_semantic_raw_k80/`) and [[Generators-Instance]] (e.g. `pseudo_instance_spidepth/`).
