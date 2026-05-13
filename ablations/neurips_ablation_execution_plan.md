# NeurIPS Ablation Execution Plan

This runbook turns `ablations/neurips_ablation_guide.md` into concrete work items. It is intentionally claim-driven: every run below must fill a table cell, support a paper claim, or document a limitation.

## Common Setup

Use these placeholders in commands:

```bash
export PROJECT_ROOT=/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
export CS_ROOT=/Users/qbit-glitch/Desktop/datasets/cityscapes
export OUT=$PROJECT_ROOT/results/neurips_ablations
mkdir -p "$OUT" "$PROJECT_ROOT/reports/neurips_ablations" "$PROJECT_ROOT/figures/neurips_ablations"
cd "$PROJECT_ROOT"
```

Standard outputs:

- Raw metrics: `results/neurips_ablations/*.json`
- Run logs: `results/neurips_ablations/*.log`
- Paper-ready tables: `reports/neurips_ablations/*.md`
- Qualitative figures: `figures/neurips_ablations/`

Do not mix protocols inside one table. If a result uses Cityscapes train, val, Hungarian mapping, majority mapping, a different depth threshold, or a trained model instead of raw pseudo-labels, label it explicitly.

---

## P0.1 Main Baseline Comparison

### Goal

Fill the main comparison table and make the headline claim precise:

- "Ours exceeds the published CUPS baseline" if no same-backbone CUPS control exists.
- "Ours improves over CUPS under the same backbone/training" only if that control is run.

### Existing Artifacts

| Item | Source |
|---|---|
| Published CUPS PQ=27.80 | CUPS paper / paper bibliography |
| Final ours PQ=35.83, PQ_th=36.26, PQ_st=35.56, mIoU=44.56 | `results/stage3_dcfa_simcf_abc_step3000_eval.json` |
| Stage-3 validation config | `refs/cups/configs/val_stage3_dcfa_simcf_abc_local.yaml` |
| Stage-3 validation script | `refs/cups/evaluate_cityscapes.py` or `scripts/validate_cups_stage3.py` |

### Missing Work

1. Verify final Stage-3 result from checkpoint, or cite the existing JSON if re-evaluation is too expensive.
2. Decide whether to run the same-backbone CUPS control. This is optional for a conservative claim but required for a strongest-possible claim.

### Commands

Re-evaluate the final model locally:

```bash
PYTHONPATH="$PROJECT_ROOT/refs/cups:$PYTHONPATH" \
python refs/cups/evaluate_cityscapes.py \
  --experiment_config_file refs/cups/configs/val_stage3_dcfa_simcf_abc_local.yaml \
  --checkpoint checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt \
  --device cpu \
  --output_json "$OUT/final_stage3_dcfa_simcf_abc_cityscapes_val.json" \
  2>&1 | tee "$OUT/final_stage3_dcfa_simcf_abc_cityscapes_val.log"
```

If running the same-backbone CUPS control, use the CUPS official pseudo-label config family and keep the backbone/training identical to ours:

```bash
bash scripts/run_e1_santosh.sh verify
bash scripts/run_e1_santosh.sh stage2

# After Stage-2 finishes, launch Stage-3 with the best Stage-2 checkpoint:
bash scripts/run_e1_santosh.sh stage3 /path/to/e1_stage2/best_pq_step=XXXXXX.ckpt
```

### Completion Criteria

- Main comparison table has source paths for every number.
- If same-backbone CUPS is missing, the paper text explicitly says "published CUPS baseline" and lists the missing control as a limitation.

---

## P0.2 Component Removal Study

### Goal

Show which proposed components contribute: raw semantics, depth instances, DCFA, SIMCF, and the trained Stage-3 model.

### Existing Artifacts

| Variant | Source |
|---|---|
| Raw k=80 + DepthPro tau=0.20 | `reports/dcfa_depthpro_simcf_abc_pseudolabel_report.md` |
| DCFA + DepthPro only | `reports/dcfa_depthpro_simcf_abc_pseudolabel_report.md` |
| SIMCF-ABC only | `reports/dcfa_depthpro_simcf_abc_pseudolabel_report.md` |
| DCFA + DepthPro + SIMCF-ABC | `reports/dcfa_depthpro_simcf_abc_pseudolabel_report.md` |
| Final trained model | `results/stage3_dcfa_simcf_abc_step3000_eval.json` |

### Missing Work

1. Normalize the component table into one protocol. The existing report uses Cityscapes train for pseudo-labels; the final model result uses Cityscapes val.
2. If the table is intended to be val-only, rerun the pseudo-label variants on val.
3. Add a row for raw semantic baseline with CC/no-depth if not already in the same protocol.

### Commands

Evaluate raw clusters with no depth:

```bash
python scripts/auto/evaluate_with_hungarian.py \
  --sem_dir "$CS_ROOT/pseudo_semantic_raw_k80/val" \
  --cityscapes_root "$CS_ROOT" \
  --no_depth \
  --split val \
  --output "$OUT/component_raw_k80_cc_only_val.json" \
  2>&1 | tee "$OUT/component_raw_k80_cc_only_val.log"
```

Evaluate raw clusters with depth:

```bash
python scripts/auto/evaluate_with_hungarian.py \
  --sem_dir "$CS_ROOT/pseudo_semantic_raw_k80/val" \
  --cityscapes_root "$CS_ROOT" \
  --depth_subdir depth_depthpro \
  --tau 0.20 \
  --min_area 1000 \
  --dilation 3 \
  --split val \
  --output "$OUT/component_raw_k80_depthpro_tau020_val.json" \
  2>&1 | tee "$OUT/component_raw_k80_depthpro_tau020_val.log"
```

Generate DCFA labels if missing:

```bash
bash scripts/gen_v3_adapter_cups_labels.sh \
  2>&1 | tee "$OUT/generate_dcfa_adapter_labels.log"
```

Run SIMCF on a chosen pseudo-label directory:

```bash
python scripts/refine_simcf.py \
  --input_dir "$CS_ROOT/cups_pseudo_labels_adapter_V3_tau020" \
  --output_dir "$CS_ROOT/cups_pseudo_labels_dcfa_simcf_abc" \
  --centroids_path "$CS_ROOT/pseudo_semantic_adapter_V3_k80/kmeans_centroids.npz" \
  --cityscapes_root "$CS_ROOT" \
  --steps A,B,C \
  2>&1 | tee "$OUT/refine_simcf_abc.log"
```

Evaluate CUPS-format pseudo-labels:

```bash
python scripts/evaluate_pseudolabel_quality.py \
  --pseudo_dir "$CS_ROOT/cups_pseudo_labels_dcfa_simcf_abc" \
  --cityscapes_root "$CS_ROOT" \
  --split train \
  --num_clusters 80 \
  2>&1 | tee "$OUT/component_dcfa_simcf_abc_train.log"
```

Note: `scripts/evaluate_pseudolabel_quality.py` currently prints a machine-readable `SUMMARY;...` line but does not write JSON. Either parse the log into `component_dcfa_simcf_abc_train.json` or add JSON export before using it as a paper-table source.

### Completion Criteria

- One component table with matched split/evaluator.
- Separate note if pseudo-label rows use train while trained-model row uses val.
- Paper table does not present 35.83 as raw pseudo-label PQ.

---

## P0.3 Depth Model Comparison

### Goal

Defend the monocular-depth claim with at least no-depth, one historical depth source, and two modern foundation depth sources.

### Existing Artifacts

| Item | Source |
|---|---|
| SPIdepth, DA2, DA3, CC-only sweep | `reports/depth_model_ablation_study.md` |
| DepthPro/DA3 threshold sweeps | `results/auto_fragmentation/*.json` |
| Sweep script | `mbps_pytorch/sweep_depth_model_comparison.py` |
| Fragmentation sweep script | `scripts/auto/run_fragmentation_sweep.py` |

### Missing Work

1. Decide the final depth source for the paper: DepthPro, DA3, or both.
2. Ensure each depth model is evaluated with its own optimal threshold.
3. Add DepthPro to the same table as DA2/DA3/SPIdepth if it is not already protocol-matched.

### Commands

Full depth comparison:

```bash
python mbps_pytorch/sweep_depth_model_comparison.py \
  --cityscapes_root "$CS_ROOT" \
  --depth_subdirs depth_spidepth depth_depthpro depth_da2_large depth_dav3 \
  2>&1 | tee "$OUT/depth_model_comparison.log"
```

DepthPro/DA3 fragmentation sweep on a fixed semantic source:

```bash
python scripts/auto/run_fragmentation_sweep.py \
  --cityscapes_root "$CS_ROOT" \
  --sem_dir "$CS_ROOT/pseudo_semantic_adapter_V3_k80/val" \
  --depth_models depth_depthpro,depth_dav3 \
  --tau_values 0.03,0.05,0.10,0.15,0.20,0.30,0.50 \
  --output_dir "$OUT/depth_fragmentation" \
  2>&1 | tee "$OUT/depth_fragmentation_sweep.log"
```

### Completion Criteria

- Table reports PQ, PQ_th, PQ_st, threshold, and min area.
- The text says depth helps most when semantic coverage is sufficient.
- If COCO is included, explain that poor COCO transfer is mostly class-space/semantic mismatch, not proof that depth fails.

---

## P0.4 Downstream Amplification

### Goal

Show that pseudo-label improvements propagate into trained model performance.

### Existing Artifacts

| Stage | Source |
|---|---|
| Raw/refined pseudo-label deltas | `reports/dcfa_depthpro_simcf_abc_cvpr_report.md` |
| Stage-3 step 800/1000/2200/3000 evals | `results/stage3_dcfa_simcf_abc*.json` |
| Stage-3 launch script | `scripts/run_stage3_dcfa_simcf_abc_santosh.sh` |
| Stage-2/Stage-3 DepthPro scripts | `scripts/run_cups_dinov3_vitb_depthpro_stage2.sh`, `scripts/run_cups_dinov3_vitb_depthpro_stage3.sh` |

### Missing Work

1. Fill Stage-2 result for the exact refined pseudo-label source.
2. Confirm whether the DepthPro-only baseline is 31.62 PQ and cite its artifact.
3. Add a validation trajectory figure/table from checkpoint JSONs.

### Commands

Run Stage-2 for a pseudo-label variant on remote/GPU:

```bash
bash scripts/run_cups_dinov3_vitb_depthpro_stage2.sh
```

Run Stage-3 for the final DCFA+SIMCF labels:

```bash
bash scripts/run_stage3_dcfa_simcf_abc_santosh.sh train
bash scripts/run_stage3_dcfa_simcf_abc_santosh.sh status
```

Re-evaluate checkpoints:

```bash
PYTHONPATH="$PROJECT_ROOT/refs/cups:$PYTHONPATH" \
python refs/cups/evaluate_cityscapes.py \
  --experiment_config_file refs/cups/configs/val_stage3_dcfa_simcf_abc_local.yaml \
  --checkpoint checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt \
  --device cpu \
  --output_json "$OUT/downstream_stage3_step3000.json"
```

### Completion Criteria

- One table with raw pseudo-label, refined pseudo-label, Stage-2, and Stage-3 rows.
- Each row states whether it is a pseudo-label evaluation or trained-model evaluation.
- Include the trajectory: step 800, 1000, 2200, 3000.

---

## P0.5 Per-Class Breakdown

### Goal

Show strengths and remaining failures by class.

### Existing Artifacts

| Item | Source |
|---|---|
| Step-2200 per-class PQ | `results/stage3_dcfa_simcf_abc_step2200_eval.json` |
| Cross-dataset per-class examples | `reports/cross_dataset_evaluation_report.md` |
| Depth model per-class thing breakdown | `reports/depth_model_ablation_study.md` |

### Missing Work

1. Produce final step-3000 per-class PQ/SQ/RQ, not only aggregate metrics.
2. Group classes in the paper table: easy stuff, thin structures, common things, rare things.

### Commands

Re-run final evaluator with per-class output enabled. If `refs/cups/evaluate_cityscapes.py` already writes per-class entries, use:

```bash
PYTHONPATH="$PROJECT_ROOT/refs/cups:$PYTHONPATH" \
python refs/cups/evaluate_cityscapes.py \
  --experiment_config_file refs/cups/configs/val_stage3_dcfa_simcf_abc_local.yaml \
  --checkpoint checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt \
  --device cpu \
  --output_json "$OUT/per_class_stage3_step3000.json"
```

If the output still lacks per-class values, patch the evaluator to include `per_class_pq`, `per_class_sq`, and `per_class_rq` in the JSON.

### Completion Criteria

- Final step-3000 per-class table exists.
- Analysis explicitly calls out person, rider, car, truck, bus, train, motorcycle, bicycle, pole, traffic light, traffic sign, and fence.

---

## P0.6 Failure Case Analysis

### Goal

Create paper-quality qualitative evidence for limitations and successes.

### Existing Artifacts

| Item | Source |
|---|---|
| Visualization utilities | `mbps_pytorch/evaluation/visualizer.py`, `scripts/generate_figures.py` |
| DepthPro instance visualizer | `mbps_pytorch/visualize_depthpro_instances.py` |
| Cross-dataset discussion | `reports/cross_dataset_evaluation_report.md` |

### Missing Work

1. Select 6 to 10 cases, with at least 2 successes and 4 failures.
2. Save deterministic image IDs and captions.
3. Generate a figure panel with RGB, depth, semantic, instance/prediction, GT.

### Suggested Case Types

| Case type | Minimum count |
|---|---:|
| co-planar people/cars merge | 2 |
| depth over-fragments one object | 1 |
| thin structures vanish | 1 |
| rare class collapse | 1 |
| clean success case | 2 |
| COCO/class-space mismatch, if used | 1 |

### Commands

Start with existing visualizers:

```bash
python mbps_pytorch/visualize_depthpro_instances.py \
  --cityscapes_root "$CS_ROOT" \
  --output_dir "$PROJECT_ROOT/figures/neurips_ablations/depthpro_instances" \
  2>&1 | tee "$OUT/visualize_depthpro_instances.log"
```

Then generate final paper panels:

```bash
python scripts/generate_figures.py \
  --output_dir "$PROJECT_ROOT/figures/neurips_ablations" \
  2>&1 | tee "$OUT/generate_neurips_figures.log"
```

### Completion Criteria

- Main paper has 2 to 3 compact qualitative examples.
- Supplement has the full 6 to 10 case set.
- Captions explain the failure mechanism, not just the visual difference.

---

## P1.7 Three-Seed Headline Result

### Goal

Estimate variance for the final Stage-3 number.

### Existing Artifacts

| Item | Source |
|---|---|
| k-means multi-seed DCFA analysis | `reports/dcfa_v2_multiseed_analysis/analysis-report.md` |
| Final single-seed Stage-3 result | `results/stage3_dcfa_simcf_abc_step3000_eval.json` |

### Missing Work

Full training seed variation is missing. The existing DCFA multi-seed analysis varies k-means seed only, not the full Stage-2/Stage-3 training seed.

### Commands

For each seed, set the config seed and use a unique run name/output path:

```bash
for SEED in 42 123 456; do
  cp refs/cups/configs/train_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml \
     refs/cups/configs/train_cityscapes_dinov3_vitb_dcfa_simcf_abc_seed${SEED}.yaml
  cp refs/cups/configs/train_self_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml \
     refs/cups/configs/train_self_cityscapes_dinov3_vitb_dcfa_simcf_abc_seed${SEED}.yaml
  # Edit SYSTEM.SEED and RUN_NAME in both copied configs before launching.
done
```

Launch Stage-2 and Stage-3 for each seed on the remote GPU machine. Save final eval JSONs as:

```text
results/neurips_ablations/final_stage3_seed42.json
results/neurips_ablations/final_stage3_seed123.json
results/neurips_ablations/final_stage3_seed456.json
```

### Completion Criteria

- Report mean +/- std for PQ, PQ_th, PQ_st, mIoU.
- If only one seed is feasible, do not claim seed robustness.

---

## P1.8 DCFA Design Sanity

### Goal

Show the adapter is useful because of depth-conditioned residual adaptation, not just extra parameters.

### Existing Artifacts

| Item | Source |
|---|---|
| DCFA architecture/lambda sweep | `reports/depth_semantic_ablation_complete.md` |
| DCFA v2 multi-seed analysis | `reports/dcfa_v2_multiseed_analysis/analysis-report.md` |
| DCFA implementation | `mbps_pytorch/models/semantic/depth_adapter.py` |
| DCFA training script | `mbps_pytorch/train_depth_adapter.py` |

### Missing Work

1. A same-capacity non-depth adapter baseline, if not already available.
2. One zero-init vs random-init comparison only if the paper claims zero-init matters.

### Commands

Train standard DCFA:

```bash
python mbps_pytorch/train_depth_adapter.py \
  --cityscapes_root "$CS_ROOT" \
  --epochs 20 \
  --lambda_preserve 20.0 \
  --hidden_dim 128 \
  --depth_dim 1 \
  --output_dir "$OUT/dcfa_standard" \
  2>&1 | tee "$OUT/dcfa_standard_train.log"
```

Evaluate adapter outputs with k=80:

```bash
bash scripts/run_adapter_k80_eval.sh \
  2>&1 | tee "$OUT/dcfa_k80_eval.log"
```

If adding the non-depth adapter, use the same MLP capacity but feed constant/noise depth and save as:

```text
results/neurips_ablations/dcfa_non_depth_adapter_eval.json
```

### Completion Criteria

- Table has no adapter, non-depth/random adapter, DCFA.
- Do not add wide/deep activation sweeps to the main paper.

---

## P1.9 Depth Encoding Sanity

### Goal

Show that adding depth information to frozen semantic codes helps.

### Existing Artifacts

| Item | Source |
|---|---|
| no/raw/sobel/sinusoidal depth encoding ablation | `reports/depth_semantic_ablation_complete.md` |
| k=80 adapter evals | `results/depth_adapter/*.json` |

### Missing Work

Only rerun if the existing ablation used a protocol that differs from the final paper protocol.

### Commands

Use the existing depth semantic ablation script:

```bash
bash scripts/run_depth_semantic_ablations.sh \
  2>&1 | tee "$OUT/depth_encoding_sanity.log"
```

If running manually:

```bash
python mbps_pytorch/generate_depth_overclustered_semantics.py \
  --cityscapes_root "$CS_ROOT" \
  --split val \
  --variant sinusoidal \
  --alpha 0.1 \
  --k 80 \
  --output_subdir pseudo_semantic_depth_sinusoidal_k80 \
  --skip_crf
```

### Completion Criteria

- Main paper includes no depth, raw depth, sinusoidal depth.
- Larger encoding sweeps stay out of the main paper.

---

## P1.10 SIMCF Internal Ablation

### Goal

Show the contribution of SIMCF-A/B/C without running every possible subset.

### Existing Artifacts

| Item | Source |
|---|---|
| SIMCF specification | `reports/simcf_method_specification.md` |
| Full SIMCF-ABC report | `reports/dcfa_depthpro_simcf_abc_pseudolabel_report.md` |
| SIMCF implementation | `scripts/refine_simcf.py` |

### Missing Work

The lean table needs no SIMCF, A only, AB, ABC. If time is tight, use no SIMCF, AB, ABC.

### Commands

```bash
for STEPS in A A,B A,B,C; do
  TAG=$(echo "$STEPS" | tr -d ',' | tr 'A-Z' 'a-z')
  python scripts/refine_simcf.py \
    --input_dir "$CS_ROOT/cups_pseudo_labels_adapter_V3_tau020" \
    --output_dir "$CS_ROOT/cups_pseudo_labels_simcf_${TAG}" \
    --centroids_path "$CS_ROOT/pseudo_semantic_adapter_V3_k80/kmeans_centroids.npz" \
    --cityscapes_root "$CS_ROOT" \
    --steps "$STEPS" \
    2>&1 | tee "$OUT/simcf_${TAG}.log"

  python scripts/evaluate_pseudolabel_quality.py \
    --pseudo_dir "$CS_ROOT/cups_pseudo_labels_simcf_${TAG}" \
    --cityscapes_root "$CS_ROOT" \
    --split train \
    --num_clusters 80 \
    2>&1 | tee "$OUT/simcf_${TAG}_eval.log"
done
```

Note: parse each `SUMMARY;...` line into JSON, or patch the evaluator to write JSON before treating these logs as final artifacts.

### Completion Criteria

- Table reports PQ, PQ_th, PQ_st.
- Text identifies Step B as the main instance-quality contributor if reproduced.

---

## P1.11 Cross-Dataset Transfer

### Goal

Show where the trained model transfers and where it does not.

### Existing Artifacts

| Dataset | Source |
|---|---|
| Cityscapes | `results/stage3_dcfa_simcf_abc_step3000_eval.json` |
| KITTI | `results/cross_dataset_eval/kitti_dcfa_simcf_abc.json` |
| Mapillary | `results/cross_dataset_eval/mapillary_dcfa_simcf_abc.json` |
| MOTS | `results/cross_dataset_eval/mots_dcfa_simcf_abc.json` |
| COCO-Stuff-27 | `results/cross_dataset_eval/coco_stuff27_dcfa_simcf_abc.json` |
| Report | `reports/cross_dataset_evaluation_report.md` |

### Missing Work

1. Verify that the same checkpoint and class mapping were used for all datasets.
2. Decide whether COCO belongs in the main paper or only as a limitation.

### Commands

Use CUPS evaluators for the Stage-3 checkpoint where possible:

```bash
cd "$PROJECT_ROOT/refs/cups"

python evaluate_kitti.py \
  --experiment_config_file configs/val_cups_stage3_dinov3_vitb_k80_kitti_local.yaml \
  --checkpoint "$PROJECT_ROOT/checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt" \
  --output_json "$OUT/cross_kitti.json"

python evaluate_mapillary.py \
  --experiment_config_file configs/val_cups_stage3_dinov3_vitb_mapillary_local.yaml \
  --checkpoint "$PROJECT_ROOT/checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt" \
  --mapillary_root /Users/qbit-glitch/Desktop/datasets/mapillary-vistas-v2 \
  --output_json "$OUT/cross_mapillary.json"

python evaluate_coco_stuff27.py \
  --experiment_config_file configs/val_cups_stage3_dinov3_vitb_k80_coco_local.yaml \
  --checkpoint "$PROJECT_ROOT/checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt" \
  --output_json "$OUT/cross_coco_stuff27.json"
```

### Completion Criteria

- Paper claims transfer only on class-aligned datasets.
- COCO is framed as class-space mismatch unless a COCO-trained model is used.

---

## Final Assembly Checklist

After all selected runs finish:

1. Create `reports/neurips_ablations/main_tables.md`.
2. For every table cell, include the source JSON/report path.
3. Create `reports/neurips_ablations/remaining_gaps.md` with anything not run.
4. Copy final qualitative figures into `figures/neurips_ablations/final/`.
5. Update the paper text so claims match the completed evidence.

Minimum deliverable for submission:

- Table 1: main comparison.
- Table 2: component removal.
- Table 3: depth model comparison.
- Table 4: downstream amplification.
- Table 5: per-class breakdown.
- Figure: method overview.
- Figure: success/failure qualitative panel.
