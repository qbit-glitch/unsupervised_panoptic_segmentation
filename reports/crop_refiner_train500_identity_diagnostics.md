# Crop Refiner Train500 Identity Diagnostics

Run:

`test-instance-labels/Superpixels/runs/crop_refiner_fused_train500_val500_e5`

Diagnostic artifact:

`test-instance-labels/Superpixels/runs/crop_refiner_fused_train500_val500_e5/diagnostics/identity/crop_refiner_identity_diagnostics.json`

Exported guard artifacts:

- `test-instance-labels/Superpixels/runs/crop_refiner_fused_train500_val500_e5/pred_eval_guard098`
- `test-instance-labels/Superpixels/runs/crop_refiner_fused_train500_val500_e5/ap_guard098.json`
- `test-instance-labels/Superpixels/runs/crop_refiner_fused_train500_val500_e5/pred_eval_guard098_rs05`
- `test-instance-labels/Superpixels/runs/crop_refiner_fused_train500_val500_e5/ap_guard098_rs05.json`
- `test-instance-labels/Superpixels/runs/crop_refiner_fused_train500_val500_e5/pred_eval_sourceaware_hi075`
- `test-instance-labels/Superpixels/runs/crop_refiner_fused_train500_val500_e5/ap_sourceaware_hi075.json`

## AP-First Result

| Method | AP | AP50 | AP75 | AR100 | Pred/img |
|---|---:|---:|---:|---:|---:|
| Fused teacher | 0.1371 | 0.2733 | 0.1214 | 0.4040 | 21.76 |
| Crop refiner, guard 0.75 | 0.1354 | 0.2738 | 0.1175 | 0.4072 | 21.76 |
| Simulated guard 0.98 | 0.1371 | 0.2731 | 0.1225 | 0.4037 | 21.76 |
| Exported guard 0.98 | 0.1371 | 0.2733 | 0.1214 | 0.4040 | 21.76 |
| Exported guard 0.98 + residual scale 0.5 | 0.1371 | 0.2733 | 0.1214 | 0.4040 | 21.76 |
| Source-aware gate, high-score TokenCut/MaskCut preserve | 0.1386 | 0.2751 | 0.1221 | 0.4082 | 21.76 |
| Source-aware identity-loss training | 0.1286 | 0.2623 | 0.1099 | 0.4082 | 21.76 |
| Source-aware ranker selector | 0.1400 | 0.2785 | 0.1235 | 0.4095 | 21.76 |

The crop refiner is stable, but the default `identity_iou_guard=0.75` allows
too many unnecessary mask edits. It slightly improves recall and AP50, but it
hurts mean AP and AP75. The AP-selected guard is `0.98`, which is effectively a
high-precision residual gate. The exported guard-only artifact restores the
teacher-level AP regime; reducing residual scale to `0.5` does not add value
once the stricter guard is active.

The source-aware gate is the first setting that improves all primary instance
signals over the fused teacher. It keeps the loose global crop-refiner guard
for low-score/RAMA proposals, while forcing high-score TokenCut/MaskCut masks
back to the teacher whenever the residual is not nearly identical.

The source-aware ranker selector is the new AP-selected setting. It preserves
high-score TokenCut/MaskCut proposals, allows low-score RAMA proposals to
choose among residual candidates, and reranks only the residual-eligible pool
with label-free visual, stability, consensus, source-prior, area, compactness,
and border terms.

## Drift Diagnosis

| Statistic | Value |
|---|---:|
| Paired teacher/pred masks | 10,880 |
| Mean identity IoU | 0.9442 |
| Median identity IoU | 0.9966 |
| Fraction below 0.90 | 0.2581 |
| Fraction below 0.95 | 0.3536 |

The learner mostly preserves masks exactly, but the tail of edited masks is
large enough to lower AP. Tightening the guard keeps the small useful edits
while rejecting most harmful drift.

## Bucket Finding

The edits help low-score/RAMA proposals but hurt the highest-score proposals.

| Bucket | Count | Delta GT IoU | Delta AP | Delta AP50 | Delta AP75 |
|---|---:|---:|---:|---:|---:|
| source:tokencut | 500 | -0.0043 | -0.0030 | -0.0018 | -0.0048 |
| score:75-100 | 931 | -0.0028 | -0.0035 | -0.0013 | -0.0055 |
| source:maskcut | 431 | -0.0011 | -0.0013 | +0.0004 | -0.0026 |
| source:rama | 9,949 | +0.0033 | +0.0011 | +0.0005 | +0.0016 |

Conclusion: the next crop-refiner ablation should be quality-gated, not just
stronger. High-score TokenCut/MaskCut masks should be identity-preserved; RAMA
or low-score masks can receive more residual freedom.

## Source-Aware Gate

Command:

```bash
python scripts/export_local_refiner_predictions.py \
  --checkpoint runs/crop_refiner_fused_train500_val500_e5/train/checkpoint_latest.pt \
  --image-root data/coco_10pct_512x512/images/val2017 \
  --coarse-mask-dir runs/crop_refiner_fused_train500_val500_e5/coarse_eval \
  --output-dir runs/crop_refiner_fused_train500_val500_e5/pred_eval_sourceaware_hi075 \
  --image-size 512 512 \
  --identity-iou-guard 0.75 \
  --residual-scale 1.0 \
  --source-aware-identity \
  --source-preserve-ids 0,1 \
  --source-preserve-score-min 0.75 \
  --source-preserve-guard 1.0 \
  --source-free-ids 2 \
  --source-free-score-max 0.25 \
  --source-free-guard 0.75 \
  --pred-score-thresh 0.0 \
  --device cpu
```

Result:

```json
{
  "AP": 0.1386010306334971,
  "AP50": 0.27512481808529754,
  "AP75": 0.1221348493511779,
  "AR100": 0.4082156014045183
}
```

## Source-Aware Identity-Loss Training

Run:

`test-instance-labels/Superpixels/runs/crop_refiner_sourceaware_loss_train500_val500_e5`

This ablation added a train-time identity preservation loss on clean proposals,
weighted by proposal source. High-score TokenCut/MaskCut proposals received
identity pressure; low-score RAMA proposals retained residual freedom.

Command suffix:

```bash
--source-aware-identity-loss \
--source-identity-loss-weight 0.5 \
--source-preserve-identity-weight 1.0 \
--source-free-identity-weight 0.0 \
--source-default-identity-weight 0.0 \
--source-aware-identity \
--source-preserve-ids 0,1 \
--source-preserve-score-min 0.75 \
--source-preserve-guard 1.0 \
--source-free-ids 2 \
--source-free-score-max 0.25 \
--source-free-guard 0.75
```

Training stayed numerically stable:

| Epoch | Loss | Quality | Identity | Identity weight |
|---:|---:|---:|---:|---:|
| 1 | 0.6431 | 0.6604 | 0.0514 | 0.0427 |
| 2 | 0.5548 | 0.5452 | 0.0627 | 0.0427 |
| 3 | 0.5136 | 0.4654 | 0.0633 | 0.0427 |
| 4 | 0.4905 | 0.4409 | 0.0609 | 0.0427 |
| 5 | 0.4753 | 0.4272 | 0.0580 | 0.0427 |

But AP regressed:

| Method | AP | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|
| Coarse fused teacher | 0.1371 | 0.2733 | 0.1214 | 0.4040 |
| Source-aware export gate | 0.1386 | 0.2751 | 0.1221 | 0.4082 |
| Source-aware identity-loss training | 0.1286 | 0.2623 | 0.1099 | 0.4082 |

Conclusion: keep the source-aware decision as an export/inference gate for now.
The train-time identity loss is too blunt: it stabilizes optimization, but
pulls the learned masks away from the AP-optimal teacher/source mixture.

## Source-Aware Ranker Selector

Run:

`test-instance-labels/Superpixels/runs/crop_refiner_fused_train500_val500_e5/unsup_selected_sourceaware_ranker`

This ablation used already-exported teacher and residual candidates, then made
the final mask choice without GT. The source gate was:

- preserve source IDs `0,1` when proposal score >= `0.75`
- allow source ID `2` residual freedom when proposal score <= `0.25`
- apply ranker score only to the residual-eligible pool

Command:

```bash
python test-instance-labels/Superpixels/scripts/select_refiner_candidates_unsup.py \
  --candidate teacher=/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/test-instance-labels/Superpixels/runs/crop_refiner_fused_train500_val500_e5/coarse_eval \
  --candidate residual025=/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/test-instance-labels/Superpixels/runs/crop_refiner_fused_train500_val500_e5/pred_eval_residual025_guard075 \
  --candidate residual050=/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/test-instance-labels/Superpixels/runs/crop_refiner_fused_train500_val500_e5/pred_eval_residual050_guard075 \
  --candidate residual075=/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/test-instance-labels/Superpixels/runs/crop_refiner_fused_train500_val500_e5/pred_eval_residual075_guard075 \
  --candidate sourceaware=/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/test-instance-labels/Superpixels/runs/crop_refiner_fused_train500_val500_e5/pred_eval_sourceaware_hi075 \
  --image-root /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/test-instance-labels/Superpixels/data/coco_10pct_512x512/images/val2017 \
  --gt-instance-dir /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/test-instance-labels/Superpixels/data/coco_10pct_512x512/instanceIds/val2017 \
  --output-dir /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/test-instance-labels/Superpixels/runs/crop_refiner_fused_train500_val500_e5/unsup_selected_sourceaware_ranker \
  --image-size 512 512 \
  --low-score-threshold 0.2 \
  --quality-score-scale 0.0 \
  --residual-bonus 0.03 \
  --selection-margin 0.0 \
  --w-area-delta 0.0 \
  --w-change 0.0 \
  --ranker-score-scale 0.10 \
  --ranker-source-priors 0=1.0,1=0.85,2=0.2 \
  --preserve-source-ids 0,1 \
  --preserve-score-min 0.75 \
  --free-source-ids 2 \
  --free-score-max 0.25
```

Result:

| Method | AP | AP50 | AP75 | AR100 | Delta AP |
|---|---:|---:|---:|---:|---:|
| Coarse fused teacher | 0.1371 | 0.2733 | 0.1214 | 0.4040 | 0.0000 |
| Source-aware export gate | 0.1386 | 0.2751 | 0.1221 | 0.4082 | +0.0015 |
| Source-aware ranker selector | 0.1400 | 0.2785 | 0.1235 | 0.4095 | +0.0030 |

Selection counts:

| Candidate | Count | Fraction |
|---|---:|---:|
| teacher | 931 | 0.0856 |
| residual025 | 4125 | 0.3791 |
| residual050 | 442 | 0.0406 |
| residual075 | 2430 | 0.2233 |
| sourceaware | 2952 | 0.2713 |

Conclusion: this is now the best AP-first local setting. It gives a real gain
over both the fused teacher and the source-aware export gate without using GT
inside the selector.

## Code Change

`test-instance-labels/Superpixels/superpixels_uis/config.py` now defaults
`identity_iou_guard` to `0.98` for future crop-refiner exports/runs. Existing
checkpoints can still be exported with an explicit override:

```bash
python scripts/export_local_refiner_predictions.py \
  --checkpoint runs/crop_refiner_fused_train500_val500_e5/train/checkpoint_latest.pt \
  --image-root data/coco_10pct_512x512/images/val2017 \
  --coarse-mask-dir runs/crop_refiner_fused_train500_val500_e5/coarse_eval \
  --output-dir runs/crop_refiner_fused_train500_val500_e5/pred_eval_guard098 \
  --image-size 512 512 \
  --identity-iou-guard 0.98 \
  --pred-score-thresh 0.0 \
  --device cpu
```
