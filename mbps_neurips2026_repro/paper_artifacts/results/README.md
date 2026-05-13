# Result Artifacts

Small JSON artifacts copied here are used to verify the numerical claims in the paper without bundling checkpoints, data, or generated pseudo-label directories.

Canonical Cityscapes final result:

```text
stage3_dcfa_simcf_abc_step3000_eval.json
PQ        35.83
PQ_things 36.26
PQ_stuff  35.56
mIoU      44.56
```

Cross-dataset artifacts live under `cross_dataset_eval/`. SIMCF threshold-sweep summaries and logs live under `simcf_sensitivity_sweep/`; generated pseudo-label PNG/PT directories were intentionally excluded from this release copy.

