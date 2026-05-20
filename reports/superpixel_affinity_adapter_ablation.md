# Superpixel Affinity Adapter Ablation

This ablation trains a very small learnable instance adapter without changing
the production DCFA/SIMCF pseudo-label path.

## Existing Pipeline References

- DCFA semantic adapter training:
  `mbps_pytorch/train_depth_adapter.py`
- DCFA semantic label handling and k=80-to-trainID mapping:
  `mbps_pytorch/adaptive_instance_semantics.py`
- Current tau/A_min depth connected-component teacher:
  `mbps_pytorch/convert_to_cups_format.py::build_instance_map_depth_cc`
- Depth-guided instance NPZ format:
  `mbps_pytorch/generate_depth_guided_instances.py::save_instances`
- SIMCF post-processing reference:
  `scripts/refine_cn_simcf.py`
- Existing evaluation path:
  `mbps_pytorch/evaluate_cascade_pseudolabels.py`

## New Ablation Files

- Model:
  `mbps_pytorch/models/instance/superpixel_affinity_adapter.py`
- Superpixel graph and NPZ utilities:
  `mbps_pytorch/instance_methods/superpixel_affinity.py`
- Training script:
  `mbps_pytorch/train_superpixel_affinity_adapter.py`
- Generation script:
  `mbps_pytorch/generate_superpixel_affinity_instances.py`

## Method

For each Cityscapes image:

1. Load DCFA/k=80 semantics and map them to Cityscapes trainIDs.
2. Build the existing depth-CC pseudo-instance prior with `tau` and `A_min`.
3. Generate SLIC superpixels.
4. Compute per-superpixel frozen descriptors from RGB/Lab, depth, semantic
   purity, pseudo-instance purity, DINO features, and optional CLIP features.
5. Train a tiny edge MLP to predict `p_merge(u, v)` for adjacent superpixels.
6. At inference, threshold `p_merge`, run connected components inside each
   thing class, and save NPZ instance masks.

The adapter is deliberately graph-level and lightweight. It learns when the
existing depth-CC teacher should be trusted, split, or merged.

The second ablation adds internal negative cut supervision. If two adjacent
superpixels belong to the same depth-CC pseudo-instance but have low
appearance/feature affinity and a strong depth/semantic boundary, the edge is
treated as a weighted negative. This gives the adapter a learnable way to split
merged co-planar objects instead of only reproducing the connected-component
teacher.

## Training

```bash
python mbps_pytorch/train_superpixel_affinity_adapter.py \
  --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
  --semantic_subdir pseudo_semantic_raw_k80 \
  --centroids_path /Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_raw_k80/kmeans_centroids.npz \
  --depth_subdir depth_depthpro \
  --feature_subdir dinov2_features \
  --output_dir checkpoints/superpixel_affinity_adapter \
  --pseudo_tau 0.20 \
  --pseudo_min_area 1000 \
  --max_train_images 200 \
  --max_val_images 50
```

Optional CLIP feature caches can be added with:

```bash
  --clip_feature_subdir clip_features
```

When CLIP features are present, the trainer fits per-class prototype memory and
uses prototype confidence to downweight noisy edges.

### Proposal-Objectness Ablation

The next adapter ablation can consume a proposal bank, such as RAMA, TokenCut,
MaskCut, SAM-style masks, or a fused bank, as optional objectness/support
features. This is intentionally opt-in so previous checkpoints keep the same
input shape.

For each superpixel, the adapter adds proposal support descriptors:

- max normalized proposal score times superpixel coverage
- max proposal coverage
- summed proposal support, clipped to `[0, 1]`
- fraction of top proposals covering the superpixel

For each adjacent-superpixel edge, it adds:

- objectness difference, min objectness, and max objectness
- whether both superpixels share the same strongest proposal
- strongest shared proposal support
- strongest shared proposal coverage

Training command template:

```bash
.venv/bin/python mbps_pytorch/train_superpixel_affinity_adapter.py \
  --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
  --semantic_subdir pseudo_semantic_raw_k80 \
  --centroids_path /Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_raw_k80/kmeans_centroids.npz \
  --depth_subdir depth_depthpro \
  --feature_subdir dinov3_features_vitl16 \
  --proposal_bank_dir <cityscapes_proposal_npz_root> \
  --proposal_objectness \
  --proposal_objectness_top_k 100 \
  --proposal_objectness_support_thresh 0.25 \
  --proposal_soft_affinity_weight 0.0 \
  --output_dir checkpoints/superpixel_affinity_adapter_obj500_k80_depthpro_dinov3 \
  --pseudo_tau 0.20 \
  --pseudo_min_area 1000 \
  --max_train_images 500 \
  --max_val_images 100 \
  --epochs 8 \
  --n_segments 600 \
  --max_edges_per_image 1800 \
  --batch_size 4096 \
  --positive_affinity_min 0.55 \
  --intra_instance_negative_affinity_max 0.35 \
  --intra_instance_negative_boundary_min 0.06 \
  --intra_instance_negative_weight 1.0
```

Generation uses the proposal bank path saved in the checkpoint config, or an
override:

```bash
.venv/bin/python mbps_pytorch/generate_superpixel_affinity_instances.py \
  --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
  --checkpoint checkpoints/superpixel_affinity_adapter_obj500_k80_depthpro_dinov3/best.pth \
  --output_dir /Users/qbit-glitch/Desktop/datasets/cityscapes/superpixel_affinity_instances_obj500 \
  --proposal_bank_dir <cityscapes_proposal_npz_root> \
  --split val
```

Important: the current `test-instance-labels/Superpixels/runs/*train500*`
RAMA/TokenCut banks are COCO 10pct artifacts, not Cityscapes artifacts, so they
are useful for validating the proposal-objectness mechanics but should not be
fed into the Cityscapes adapter as training evidence.

## Generation

```bash
python mbps_pytorch/generate_superpixel_affinity_instances.py \
  --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
  --checkpoint checkpoints/superpixel_affinity_adapter/best.pth \
  --output_dir /Users/qbit-glitch/Desktop/datasets/cityscapes/superpixel_affinity_instances \
  --split val
```

## Evaluation

```bash
python mbps_pytorch/evaluate_cascade_pseudolabels.py \
  --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
  --split val \
  --semantic_subdir pseudo_semantic_raw_k80 \
  --instance_subdir superpixel_affinity_instances \
  --num_clusters 80 \
  --cluster_mapping_path /Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_raw_k80/kmeans_centroids.npz \
  --thing_mode maskcut \
  --output results/superpixel_affinity_adapter_val.json
```

Primary ablation metrics:

- COCO-style instance `AP@[.50:.95]` (`AP`) as the primary ranking metric
- `AP@50`, `AP@75`, and `AR@100`
- average predicted instances per image
- per-class `TP`, `FP`, `FN` for failure diagnosis
- edge hard-label accuracy and positive-edge recall from training logs

`PQ` and `PQ_things` are retained only as panoptic-composition diagnostics.
They should not be used to select the best instance adapter, because they mix
mask quality with semantic mapping and panoptic merge behavior.

For prediction files with `class_ids`, instance AP is evaluated in class-aware
mode. Files without `class_ids` fall back to class-agnostic AP and are flagged
with `"class_aware": false` in the JSON output.

## Expected Failure Checks

- If predicted instances per image collapses, lower `--merge_threshold`.
- If false positives explode, raise `--merge_threshold` or `--min_area`.
- If hard-edge recall is low, lower `--positive_affinity_min`.
- If the adapter simply reproduces the depth-CC teacher, increase superpixel
  count or add CLIP/DINO features with stricter prototype weighting.

## Smoke Run 2026-05-16

Command:

```bash
.venv/bin/python mbps_pytorch/train_superpixel_affinity_adapter.py \
  --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
  --semantic_subdir pseudo_semantic_raw_k80 \
  --centroids_path /Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_raw_k80/kmeans_centroids.npz \
  --depth_subdir depth_depthpro \
  --feature_subdir dinov3_features_vitl16 \
  --output_dir checkpoints/superpixel_affinity_adapter_smoke_k80_depthpro_dinov3 \
  --pseudo_tau 0.20 \
  --pseudo_min_area 1000 \
  --max_train_images 50 \
  --max_val_images 20 \
  --epochs 5 \
  --n_segments 600 \
  --max_edges_per_image 1800 \
  --batch_size 2048
```

Training summary:

| Item | Value |
|---|---:|
| Params | 59,521 |
| Input dim | 330 |
| Train images | 50 |
| Val images | 20 |
| Train edges | 76,409 |
| Train hard edges | 4,072 |
| Train positive hard edges | 3,998 |
| Train negative hard edges | 74 |
| Best val hard accuracy | 98.4 |
| Val hard precision | 98.4 |
| Val hard recall | 100.0 |

The hard labels are strongly positive-biased, so the next training iteration
should increase reliable negative cut supervision instead of scaling this exact
configuration blindly.

50-image validation slice:

| Method | Merge threshold | Inst/img | AP | AR@100 | AP@50 | AP@75 | PQ | PQ_stuff | PQ_things |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Existing `pseudo_instance_depthpro` | n/a | 4.4 | 0.14 | 0.61 | 0.64 | 0.01 | 18.9 | 30.1 | 3.5 |
| Superpixel adapter | 0.55 | 5.3 | 9.72 | 11.31 | 21.22 | 9.51 | 26.2 | 30.1 | 20.8 |
| Superpixel adapter | 0.85 | 6.2 | 9.55 | 11.24 | 21.34 | 9.45 | 26.0 | 30.1 | 20.3 |

Smoke ranking by standard instance metrics: `0.55` has the better mean `AP`;
`0.85` only improves `AP@50` slightly.

Saved artifacts:

- Checkpoint:
  `checkpoints/superpixel_affinity_adapter_smoke_k80_depthpro_dinov3/best.pth`
- Generated instances:
  `results/superpixel_affinity_instances_smoke_k80_depthpro_dinov3/`
- Evaluation JSON:
  `results/superpixel_affinity_adapter_smoke_eval50.json`
- Threshold 0.85 generated instances:
  `results/superpixel_affinity_instances_smoke_k80_depthpro_dinov3_thr085/`
- Threshold 0.85 evaluation JSON:
  `results/superpixel_affinity_adapter_smoke_thr085_eval50.json`

## Negative-Supervision Ablation 2026-05-16

Command:

```bash
.venv/bin/python mbps_pytorch/train_superpixel_affinity_adapter.py \
  --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
  --semantic_subdir pseudo_semantic_raw_k80 \
  --centroids_path /Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_raw_k80/kmeans_centroids.npz \
  --depth_subdir depth_depthpro \
  --feature_subdir dinov3_features_vitl16 \
  --output_dir checkpoints/superpixel_affinity_adapter_neg500_k80_depthpro_dinov3 \
  --pseudo_tau 0.20 \
  --pseudo_min_area 1000 \
  --max_train_images 500 \
  --max_val_images 100 \
  --epochs 8 \
  --n_segments 600 \
  --max_edges_per_image 1800 \
  --batch_size 4096 \
  --positive_affinity_min 0.55 \
  --intra_instance_negative_affinity_max 0.35 \
  --intra_instance_negative_boundary_min 0.06 \
  --intra_instance_negative_weight 1.0
```

Training summary:

| Item | Value |
|---|---:|
| Params | 59,521 |
| Input dim | 330 |
| Train images | 500 |
| Val images | 100 |
| Train edges | 755,551 |
| Train hard edges | 44,934 |
| Train positive hard edges | 40,387 |
| Train negative hard edges | 4,547 |
| Val edges | 149,527 |
| Val hard edges | 5,956 |
| Val positive hard edges | 5,272 |
| Val negative hard edges | 684 |
| Final val loss | 0.2025 |
| Final val hard accuracy | 98.7 |
| Final val hard precision | 98.5 |
| Final val hard recall | 100.0 |

The patch fixed the smoke run's extreme positive-edge skew: negative hard
edges increased from 74 in the 50-image smoke to 4,547 in this 500-image run.

100-image validation slice:

| Method | Merge threshold | Inst/img | AP | AR@100 | AP@50 | AP@75 | PQ | PQ_stuff | PQ_things |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Existing `pseudo_instance_depthpro` | n/a | 4.4 | 0.08 | 0.44 | 0.37 | 0.01 | 19.1 | 29.7 | 4.5 |
| Superpixel adapter | 0.55 | 7.0 | 8.37 | 10.98 | 19.64 | 6.14 | 24.1 | 29.7 | 16.3 |
| Superpixel adapter | 0.85 | 7.9 | 6.78 | 9.85 | 17.01 | 4.20 | 23.1 | 29.7 | 14.0 |

Best 100-image setting by standard instance `AP`: `merge_threshold=0.55`.

Selected thing-class behavior at `merge_threshold=0.55`:

| Class | PQ | SQ | RQ | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| car | 15.0 | 68.2 | 22.0 | 114 | 209 | 598 |
| person | 3.4 | 60.5 | 5.5 | 28 | 161 | 794 |
| rider | 4.6 | 60.1 | 7.6 | 6 | 50 | 95 |
| bicycle | 5.4 | 61.0 | 8.8 | 11 | 77 | 150 |
| bus | 35.6 | 74.8 | 47.6 | 10 | 12 | 10 |
| truck | 28.6 | 76.2 | 37.5 | 6 | 9 | 11 |
| train | 37.7 | 71.5 | 52.6 | 5 | 6 | 3 |

Interpretation:

- The adapter is viable: it substantially improves the learned instance branch
  over the depth-CC teacher on the same first-100 validation slice.
- The best setting remains under-counted (`7.04` predicted instances/image vs
  `18.8` GT instances/image), so small classes still have low recall.
- Raising the merge threshold from `0.55` to `0.85` adds instances, but lowers
  `AP`, `AP@50`, and `AP@75`; thresholding alone is not enough.
- `pseudo_instance_simcf_abc` does not currently expose val instance NPZs in
  the evaluator format, so this table is a direct comparison against
  `pseudo_instance_depthpro`, the teacher used for this adapter ablation.

Saved artifacts:

- Checkpoint:
  `checkpoints/superpixel_affinity_adapter_neg500_k80_depthpro_dinov3/best.pth`
- Training history:
  `checkpoints/superpixel_affinity_adapter_neg500_k80_depthpro_dinov3/history.json`
- Edge cache:
  `checkpoints/superpixel_affinity_adapter_neg500_k80_depthpro_dinov3/edge_training_cache.npz`
- Generated instances at threshold 0.55:
  `results/superpixel_affinity_instances_neg500_k80_depthpro_dinov3/`
- Generated instances at threshold 0.85:
  `results/superpixel_affinity_instances_neg500_k80_depthpro_dinov3_thr085/`
- Evaluation JSONs:
  `results/pseudo_instance_depthpro_eval100.json`,
  `results/superpixel_affinity_adapter_neg500_eval100.json`,
  `results/superpixel_affinity_adapter_neg500_thr085_eval100.json`

Next ablation should add class-aware or CLIP-prototype negative sampling for
person/rider/bicycle, because those failures are dominated by false negatives
rather than boundary quality alone.

## Class-Aware Hard-Negative Ablation 2026-05-16

This ablation adds target-class weighting for high-FN thing classes. The graph
builder now supports:

- `class_aware_negative_classes`
- `class_aware_positive_affinity_min`
- `class_aware_negative_weight`
- `class_aware_positive_weight`
- `class_aware_intra_instance_negative_affinity_max`
- `class_aware_intra_instance_negative_boundary_min`

The generator also supports class-specific minimum output areas with
`--class_min_area`, so small person/rider/bicycle components can be tested
without lowering the global thing-mask area threshold.

Command:

```bash
.venv/bin/python mbps_pytorch/train_superpixel_affinity_adapter.py \
  --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
  --semantic_subdir pseudo_semantic_raw_k80 \
  --centroids_path /Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_raw_k80/kmeans_centroids.npz \
  --depth_subdir depth_depthpro \
  --feature_subdir dinov3_features_vitl16 \
  --output_dir checkpoints/superpixel_affinity_adapter_classaware500_k80_depthpro_dinov3 \
  --pseudo_tau 0.20 \
  --pseudo_min_area 1000 \
  --max_train_images 500 \
  --max_val_images 100 \
  --epochs 8 \
  --n_segments 600 \
  --max_edges_per_image 1800 \
  --batch_size 4096 \
  --positive_affinity_min 0.55 \
  --intra_instance_negative_affinity_max 0.35 \
  --intra_instance_negative_boundary_min 0.06 \
  --intra_instance_negative_weight 1.0 \
  --class_aware_negative_classes 11,12,17,18 \
  --class_aware_positive_affinity_min 0.70 \
  --class_aware_negative_weight 2.5 \
  --class_aware_positive_weight 1.1 \
  --class_aware_intra_instance_negative_affinity_max 0.65 \
  --class_aware_intra_instance_negative_boundary_min 0.04
```

Training summary:

| Item | Value |
|---|---:|
| Params | 59,521 |
| Input dim | 330 |
| Train images | 500 |
| Val images | 100 |
| Train edges | 755,551 |
| Train hard edges | 45,162 |
| Train positive hard edges | 39,963 |
| Train negative hard edges | 5,199 |
| Val edges | 149,527 |
| Val hard edges | 6,022 |
| Val positive hard edges | 5,166 |
| Val negative hard edges | 856 |
| Final val loss | 0.2010 |
| Final val hard accuracy | 98.6 |
| Final val hard precision | 98.5 |
| Final val hard recall | 99.9 |

The class-aware rule did increase target split evidence. Compared with the
previous 500-image run, val negative hard edges increased from 684 to 856.
Person val negatives increased to 222, rider to 76, and bicycle to 36.

Target-class edge labels:

| Class | Train edges | Train pos | Train neg | Val edges | Val pos | Val neg |
|---|---:|---:|---:|---:|---:|---:|
| person | 8,243 | 2,387 | 1,045 | 1,753 | 522 | 222 |
| rider | 1,358 | 287 | 217 | 452 | 82 | 76 |
| motorcycle | 0 | 0 | 0 | 0 | 0 | 0 |
| bicycle | 2,606 | 688 | 117 | 1,150 | 321 | 36 |

100-image validation slice:

| Method | Decode | Inst/img | AP | AR@100 | AP@50 | AP@75 | PQ | PQ_stuff | PQ_things |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Existing `pseudo_instance_depthpro` | n/a | 4.4 | 0.08 | 0.44 | 0.37 | 0.01 | 19.1 | 29.7 | 4.5 |
| Superpixel adapter, negative-supervision run | threshold 0.55 | 7.0 | 8.37 | 10.98 | 19.64 | 6.14 | 24.1 | 29.7 | 16.3 |
| Class-aware adapter | threshold 0.55, min_area 1000 | 8.0 | 8.03 | 10.60 | 19.49 | 5.64 | 23.8 | 29.7 | 15.8 |
| Class-aware adapter | threshold 0.55, class min area 300 | 8.1 | 8.03 | 10.60 | 19.49 | 5.64 | 23.8 | 29.7 | 15.8 |

Selected thing-class comparison:

| Class | Previous TP/FP/FN | Previous PQ | Class-aware TP/FP/FN | Class-aware PQ |
|---|---:|---:|---:|---:|
| car | 114/209/598 | 15.0 | 115/207/597 | 15.2 |
| person | 28/161/794 | 3.4 | 36/231/786 | 4.0 |
| rider | 6/50/95 | 4.6 | 5/64/96 | 3.4 |
| bicycle | 11/77/150 | 5.4 | 10/85/151 | 4.7 |

Interpretation:

- The class-aware loss changes the learned graph behavior and improves person
  recall, but it does not beat the previous adapter on standard instance `AP`.
- Extra target-class splits add false positives for rider/bicycle faster than
  they add true positives.
- Lower class-specific output area does not address the main failure; the
  bottleneck is not only area filtering.
- The best current adapter by standard instance `AP` remains the negative-supervision
  run at threshold 0.55.

Implementation note:

- `edge_training_cache.npz` is now opt-in via `--save_edge_cache`; the
  checkpoint and history are always saved first. This prevents a completed
  training run from failing only because the optional full edge tensor archive
  exhausts local disk.
