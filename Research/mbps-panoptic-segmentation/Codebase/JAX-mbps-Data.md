---
type: codebase-module
title: JAX/Flax — mbps/data/
project: mbps-panoptic-segmentation
module: mbps.data
framework: jax
paths:
  - mbps/data/
tags: [codebase, jax, data]
related:
  - "[[JAX-mbps]]"
  - "[[Datasets]]"
  - "[[Configs]]"
---

# JAX/Flax — `mbps/data/`

Loaders and transforms feeding [[JAX-mbps-Training]].

| File | Role |
|------|------|
| `datasets.py` | Cityscapes / COCO-Stuff-27 / NYU Depth V2 loader returning `(image, depth, labels, meta)`. Local + GCS paths supported. |
| `transforms.py` | JAX-native augmentation (normalize, resize, flip, jitter, …), stochasticity via `jax.random`. |
| `copy_paste.py` | CUPS-style self-enhanced copy-paste — pastes pseudo-label instances from the self-cache. |
| `tfrecord_utils.py` | TFRecord serialize / deserialize for sharded GCS data. |
| `depth_cache.py` | Caches precomputed depth maps under `gs://mbps-panoptic/datasets/<dataset>/depth_zoedepth/`. |

See [[Datasets]] for the full list of datasets and their on-disk / GCS layouts.
