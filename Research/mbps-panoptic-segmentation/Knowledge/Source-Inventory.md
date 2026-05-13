---
type: meta
title: Source Inventory - mbps_panoptic_segmentation
project: mbps-panoptic-segmentation
language: en
updated: 2026-03-29T13:40:24Z
---

# Source Inventory

Imported from `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation`.
## Markdown Sources

- `CLAUDE.md`
- `MBPS_V2_TECHNICAL_REPORT.md`
- `README.md`
- `Research/mbps-panoptic-segmentation/00-Hub.md`
- `Research/mbps-panoptic-segmentation/01-Plan.md`
- `Research/mbps-panoptic-segmentation/Daily/2026-03-29.md`
- `ablations/cups-stage-3.md`
- `ablations/depth_sweep_parameters.md`
- `analysis_docs/analysis_1.md`
- `analysis_docs/analysis_2.md`
- `analysis_docs/analysis_3.md`
- `cascade_stage_adaptation.md`
- `core_algorithms.md`
- `docs/plans/2026-03-06-cscmrefinenet-k80.md`
- `docs/plans/2026-03-06-hires-refinenet-design.md`
- `docs/plans/2026-03-06-hires-refinenet.md`
- `docs/plans/2026-03-08-lightweight-backbone-research.md`
- `docs/plans/2026-03-08-mobile-panoptic-training.md`
- `docs/plans/2026-03-09-panoptic-deeplab-repvit.md`
- `docs/tpu_baseline_training.md`
- `docs/tpu_deploy_commands.md`
- `informations/dataset_into.md`
- `paper/narration_style_guide.md`
- `plans/plan_1.md`
- `plans/plan_2.md`
- `plans/plan_3.md`
- `requirements.txt`
- `sample_commands.md`
- `unsupervised-panoptic-segmentation/README.md`
- `unsupervised-panoptic-segmentation/docs/plans/2026-02-25-mask2former-stage2-design.md`
- `unsupervised-panoptic-segmentation/refs/cups/README.md`
- `unsupervised-panoptic-segmentation/refs/cups/cups/scene_flow_2_se3/log/README.md`
- `unsupervised-panoptic-segmentation/refs/cups/requirements.txt`
- `unsupervised-panoptic-segmentation/reports/cause_tr_refinement.md`
- `unsupervised-panoptic-segmentation/reports/majority_vote_cluster_assignment.md`
- `unsupervised-panoptic-segmentation/reports/overclustered_spidepth_sweep.md`
- `unsupervised-panoptic-segmentation/reports/overclustering_granularity_sweep.md`
- `unsupervised-panoptic-segmentation/reports/stage_1_report.md`
- `unsupervised-panoptic-segmentation/requirements.txt`
- `unsupervised-panoptic-segmentation/weights/mask2former-swin-tiny-coco-panoptic/README.md`

## Code and Config Files

- `debug_instance_inference.py`
- `eval_cause_cutler_t15_maskcut_val.json`
- `eval_cause_cutler_t15_val.json`
- `eval_cause_cutler_t35_maskcut_val.json`
- `eval_cause_cutler_val.json`
- `eval_cause_cuvler_t15_maskcut_val.json`
- `eval_cause_cuvler_t15_val.json`
- `eval_cause_cuvler_t35_maskcut_val.json`
- `eval_cause_cuvler_val.json`
- `eval_dinosaur_30slots_dinov2_val.json`
- `eval_semantic_refinement.json`
- `mbps/__init__.py`
- `mbps/losses/__init__.py`
- `mbps/losses/bridge_loss.py`
- `mbps/losses/consistency_loss.py`
- `mbps/losses/gradient_balancing.py`
- `mbps/losses/instance_embedding_loss.py`
- `mbps/losses/instance_loss.py`
- `mbps/losses/pq_proxy_loss.py`
- `mbps/losses/semantic_loss.py`
- `mbps/losses/semantic_loss_v2.py`
- `mbps/models/__init__.py`
- `mbps/models/bridge/__init__.py`
- `mbps/models/bridge/bicms.py`
- `mbps/models/bridge/depth_conditioning.py`
- `mbps/models/bridge/mamba2_ssd.py`
- `mbps/models/bridge/projection.py`
- `mbps/models/mbps_model.py`
- `mbps/models/mbps_v2_model.py`
- `mbps/training/__init__.py`
- `mbps/training/checkpointing.py`
- `mbps/training/curriculum.py`
- `mbps/training/ema.py`
- `mbps/training/self_training.py`
- `mbps/training/trainer.py`
- `package-lock.json`
- `package.json`
- `postproc_results.json`
- `sample.py`
- `setup.py`

## Result and Report Files

- `reports/dinov3_cups_results.md`
- `reports/dinov3_stage3_cross_dataset_evaluation.md`
- `reports/instance_head_ablation_design.md`
- `reports/neurips_narrative_report.md`
- `reports/noise_robust_ttt_mamba2.md`
- `reports/repvit_cups_adaptation_analysis.md`
- `reports/ua_bilinear_detailed.md`
- `reports/unet_unified_ablation_study.md`
- `results/ablation_instance_methods/ablation_contrastive_default_val.json`
- `results/ablation_instance_methods/ablation_contrastive_sweep_val.json`
- `results/ablation_instance_methods/ablation_morse_default_val.json`
- `results/ablation_instance_methods/ablation_morse_sweep_val.json`
- `results/ablation_instance_methods/ablation_mumford_shah_default_val.json`
- `results/ablation_instance_methods/ablation_mumford_shah_sweep_val.json`
- `results/ablation_instance_methods/ablation_ot_default_val.json`
- `results/ablation_instance_methods/ablation_ot_sweep_val.json`
- `results/ablation_instance_methods/ablation_sobel_cc_default_val.json`
- `results/ablation_instance_methods/ablation_sobel_cc_sweep_val.json`
- `results/ablation_instance_methods/ablation_tda_default_val.json`
- `results/ablation_instance_methods/ablation_tda_sweep_val.json`
- `results/ablation_instance_methods/contrastive_default.log`
- `results/ablation_instance_methods/morse_default.log`
- `results/ablation_instance_methods/mumford_shah_default.log`
- `results/ablation_instance_methods/ot_default.log`
- `results/ablation_instance_methods/sobel_cc_baseline_verify.log`
- `results/ablation_instance_methods/tda_default.log`
- `results/coco_stuff27_dinov3_stage3_step8000.json`
- `results/coco_stuff27_smoke_test.json`
- `results/cross_dataset/cross_dataset_results.json`
- `results/cuts3d_training/ap_results.json`
- `results/cuts3d_training/train_5pct.log`
- `results/cuts3d_training/train_5pct_v2.log`
- `results/cuts3d_training/train_5pct_v3.log`
- `results/cuts3d_training/train_5pct_v4.log`
- `results/cuts3d_training/train_5pct_v5.log`
- `results/cuts3d_training_5pct.log`
- `results/kitti_dinov3_stage3_step8000.json`
- `results/kitti_eval.json`
- `results/mots_dinov3_stage3_step8000.json`
- `results/mots_eval.json`
