---
type: codebase-module
title: reports/ — Experiment Reports
project: mbps-panoptic-segmentation
language: en
tags: [codebase, reports, experiments]
paths:
  - reports/
related:
  - "[[Pseudo-Label-Pipeline]]"
  - "[[Instance-Methods]]"
  - "[[RefineNet-Family]]"
  - "[[Panoptic-DeepLab]]"
  - "[[Refs-CUPS]]"
---

# reports/ — Experiment Reports

Grouped thematically. Not exhaustive — see `reports/` directly for the full list.

## Semantic refinement & pseudo-label quality
- `cause_tr_refinement.md`
- `cups_semantic_ablation_report.md`
- `depth_semantic_ablation.md`
- `depth_semantic_ablation_complete.md`
- `coco_semantic_ablation_plan.md`
- `pseudolabel_quality_ablation.md`
- `dcfa_depthpro_simcf_abc_pseudolabel_report.md`
- `simcf_method_specification.md`
- `dcfa_simcf_abc_descriptions.md`

## Instance ablation, clustering, merging
- `novel_instance_ablation_default_results.md`
- `novel_instance_ablation_final_report.md`
- `instance_head_ablation_design.md`
- `overclustered_spidepth_sweep.md`
- `overclustering_granularity_sweep.md`
- `mmgd_cut_ablation_report.md`
- `semantic_instance_alignment_proposal.md`

## Cross-dataset evaluation
- `cross_dataset_evaluation_report.md`
- `dinov3_stage3_cross_dataset_evaluation.md`
- `dcfa_v2_ablation_analysis.md`

## Depth integration
- `depth_model_ablation_study.md`
- `depthpro_instance_ablation.md`
- `depth_guided_cc_instance_method.md`

## Mobile / efficient variants
- `mobile_distillation_gap_analysis.md`
- `repvit_cups_adaptation_analysis.md`
- `hires_refinenet_128x256.md`
- `cscmrefinenet_k80_ablation.md`

## Decoder architectures (UNet variants, upsampling)
- `unet_unified_ablation_study.md`
- `ua_bilinear_detailed.md`
- `ub_transposed_conv_detailed.md`
- `uc_pixelshuffle_detailed.md`
- `unet_phase2_architecture_ablation.md`
- `unet_decoder_refinement.md`

## Rare class & dead class
- `stage2_m2f_rare_class_training_brief.md`
- `rare_class_pseudolabel_proposals.md`
- `training_dynamics_dead_class_proposals.md`
- `training_strategies_dead_classes_report.md` (root)

## Stage / method
- `stage_1_report.md`, `stage_1_reproduction_report.md`
- `stage2_spatial_alignment_fix.md`
- `method_section_learned_merge.md`
- `phase4_learned_merge_plan.md`
- `imfocal_loss_refinement.md`
- `negative_loss_analysis.md`
- `pseudoclass_granularity_analysis.md`

## NeurIPS verification
- `neurips_narrative_report.md`
- `neurips_contribution_assessment.md`
- `neurips_improvement_plan.md`
- `neurips_review_audit.md`

## Reproducibility & infra
- `a6000_pseudolabel_reproducibility.md`

## Excluded reports

The following report families are intentionally **not surfaced** in this graph:
- `neurips_final_verify_*`, `neurips_reverify_*`, `neurips_review_*` for `dinov2_cause_tr`, `da2_large`, `da3`, `depthpro` — adapter-architecture verifications.
- `lora_segmentation_literature.md`, `noise_robust_ttt_mamba2.md` — LoRA / mamba-specific.
- Anything matching `T0`, `T1`, `T2`, … enumerated ablation reports.
