# DA3-Trained DCFA Validation Evaluation

Date: 2026-04-27

This folder records the direct validation-set evaluation for the strict DA3 + DCFA ablation. In this run, DCFA was retrained from the DA3-backed cached depth features and then used to generate validation semantic pseudo-labels.

## Fixed Methodology Settings

- Semantic backbone/features: DINOv2 + CAUSE-TR 90D codes
- Adapter: DA3-trained DCFA/V3 adapter
- Code cache: `cause_codes_90d_da3`
- Depth source: `depth_dav3`
- Cluster centroids: existing `pseudo_semantic_adapter_V3_k80/kmeans_centroids.npz`
- Overclustering: `k=80`
- Instance splitting: DA3 depth-guided connected components
- `tau=0.20`
- `A_min=1000`
- `depth_blur_sigma=0.0`
- `dilation_iters=3`

## Validation Metrics

| Metric | Value |
| --- | ---: |
| PQ | 26.44 |
| PQ_stuff | 32.31 |
| PQ_things | 18.37 |
| mIoU | 55.30 |

## Train-Side SIMCF-ABC Quality Check

After generating the full train CUPS labels and applying SIMCF-ABC with DA3 depth statistics, the train-side pseudo-label quality check produced:

| Metric | Value |
| --- | ---: |
| PQ | 25.59 |
| PQ_stuff | 34.20 |
| PQ_things | 13.75 |
| mIoU | 56.98 |
| Ignore pixels | 1.5% |

Direct CUPS label counts:

- `cups_pseudo_labels_adapter_V3_da3dcfa_tau020`: 2975 semantic PNGs, 2975 instance PNGs, 2975 `.pt` files
- `cups_pseudo_labels_dcfa_da3_simcf_abc`: 2975 semantic PNGs, 2975 instance PNGs, 2975 `.pt` files

SIMCF-ABC changed/merge/mask summary:

- Step A: 0 pixels changed (0.00%)
- Step B: 29107 instance merges (9.8/image)
- Step C: 92191464 pixels masked (1.48%)

## Source Artifacts

- `V3_da3_dcfa_k80_da3_tau020_panoptic_eval.json`: full validation metric JSON
- `eval_da3_trained_dcfa_tau020.log`: original validation evaluation log
- `gen_da3_dcfa_semantics_val_mapped.log`: original validation semantic pseudo-label generation log
- `train_dcfa_v3_da3.log`: original DA3-trained DCFA training log
- `build_da3_cause_code_cache.log`: original DA3 cache build log
- `gen_v3_adapter_da3_cups_labels.log`: original full train CUPS/SIMCF generation log
- `gen_v3_adapter_da3_cups_labels_in_progress_tail.log`: train-label generation log tail captured while the train-side CUPS/SIMCF pipeline was still running
- `da3_dcfa_simcf_abc_train_quality.json`: full train-side SIMCF-ABC quality metric JSON
- `eval_da3_dcfa_simcf_abc_train_quality.log`: original train-side SIMCF-ABC quality evaluation log

## Exact Validation Evaluation Command

```bash
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python -u mbps_pytorch/evaluate_panoptic_combined.py \
  --sem_dir /Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_adapter_V3_da3_k80/val \
  --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
  --depth_subdir depth_dav3 \
  --tau 0.20 \
  --min_area 1000 \
  --sigma 0.0 \
  --dilation 3 \
  --output results/depth_adapter/V3_da3_dcfa_k80_da3_tau020_panoptic_eval.json
```
