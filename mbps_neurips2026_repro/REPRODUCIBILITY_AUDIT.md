# Reproducibility Audit

This audit maps the NeurIPS paper to the copied release tree and records the checks performed during assembly.

## Paper-to-Code Map

| Paper component | Copied code path | Notes |
| --- | --- | --- |
| Frozen CAUSE code extraction | `repro_scripts/extract_cause_codes.py`, `third_party/cause/` | Produces `cause_codes_90d/{split}/{city}/*_codes.npy` and depth patches. |
| DCFA adapter | `mbps_pytorch/models/semantic/depth_adapter.py`, `mbps_pytorch/train_depth_adapter.py` | Release smoke test verifies 225,114 parameters for `depth_dim=16, hidden_dim=384, num_layers=2` and identity output at initialization. |
| DepthG-style correlation loss | `mbps_pytorch/models/semantic/stego_loss.py` | Used by `train_depth_adapter.py` as `depth_guided_correlation_loss`. |
| k=80 semantic generation | `mbps_pytorch/generate_depth_overclustered_semantics.py` | Applies the trained adapter and saves raw cluster IDs. |
| DepthPro connected components | `mbps_pytorch/convert_to_cups_format.py` | `--depth_cc_instances --grad_threshold 0.20 --min_instance_area 1000`. |
| SIMCF A/B/C | `repro_scripts/refine_simcf.py` | Step B uses DINOv3 patch features for adjacent same-class merges. |
| Stage 2 CUPS training | `third_party/cups/train.py`, `third_party/cups/configs/train_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml` | DINOv3 ViT-B/16 backbone, 8K training steps. |
| Stage 3 EMA self-training | `third_party/cups/train_self.py`, `third_party/cups/configs/train_self_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml` | Copied exact experiment config. |
| Cityscapes and transfer eval | `third_party/cups/val.py`, `third_party/cups/evaluate_*.py` | Result JSONs copied under `paper_artifacts/results/`. |

## Included Result Artifacts

| Paper table/claim | Artifact |
| --- | --- |
| Cityscapes final 35.83 PQ | `paper_artifacts/results/stage3_dcfa_simcf_abc_step3000_eval.json` |
| Per-class final baseline used in later calibration checks | `paper_artifacts/results/t0_fracal_full_dcfa_simcf_abc_step3000.json` |
| KITTI, Mapillary, MOTS, COCO-Stuff-27 transfer | `paper_artifacts/results/cross_dataset_eval/` |
| SIMCF threshold sensitivity | `paper_artifacts/results/simcf_sensitivity_sweep/` |
| Narrative reports behind tables | `paper_artifacts/reports/` |

## Code Review Findings

1. Full end-to-end reproduction requires external artifacts not copied into the release: Cityscapes, DepthPro depth maps, DINOv2/CAUSE/DINOv3 weights, DINOv3 feature caches, and Lightning checkpoints. This is expected for a GitHub code release; the current machine-specific original paths are recorded in `LOCAL_ASSET_PATHS.md`.
2. The paper text says SIMCF uses `eta=2.5`; `repro_scripts/refine_simcf.py` defaults to `--sigma_threshold 3.0`, and `paper_artifacts/reports/simcf_step_ablation.md` states `eta=3.0`. The release README uses the paper value explicitly in the command. Before final publication, choose one canonical value and make paper text, configs, reports, and scripts agree.
3. The paper setup says Stage 3 uses three 5K-step EMA rounds, but the copied Stage-3 config sets `SELF_TRAINING.ROUND_STEPS: 4000` and `ROUNDS: 3`. This is a reproducibility-critical mismatch to resolve before public release.
4. The Stage-2 config comment says DCFA has 40K parameters, while the paper and copied implementation use the 225K setting. The executable code matches the paper's 225K claim; the comment is stale.
5. Some copied experiment configs and shell scripts contain original absolute paths and remote machine names. They are preserved as evidence of the actual run, but users should override paths locally before execution.

## Tests Performed

The release test suite checks:

1. Required code, configs, paper sources, and result artifacts exist.
2. The local asset manifest exists and records the original Cityscapes, DepthPro, CAUSE, DINOv3, final pseudo-label, checkpoint, transfer-data, and remote GPU references.
3. No copied CUPS experiment/checkpoint/log directories are present.
4. DCFA has the expected parameter count and identity initialization.
5. SIMCF Step A and Step B behave correctly on synthetic inputs.
6. Depth connected-component instance generation creates valid instance IDs.
7. The CUPS config loader parses the paper Stage-2 and Stage-3 configs.
8. The final Cityscapes artifact contains the reported 35.83 PQ, 36.26 PQ-things, and 35.56 PQ-stuff values after percentage conversion.

The full training run was not executed locally because it requires external datasets, weights, and GPU hardware.
