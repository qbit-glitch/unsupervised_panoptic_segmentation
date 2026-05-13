# Lean NeurIPS Ablation Guide

This file replaces the older exhaustive checklist. The paper does not need every possible engineering sweep. It needs ablations that defend the claims made in the main text.

## Core Claims To Defend

The ablation plan should support exactly these claims:

1. Monocular depth can replace stereo/video cues for pseudo-label generation.
2. DCFA improves frozen semantic clustering.
3. SIMCF-ABC cleans semantic-instance-depth inconsistencies.
4. Better pseudo-labels produce a stronger downstream panoptic model.
5. The final system is competitive with relevant unsupervised panoptic baselines.

Experiments that do not defend one of these claims should move to the supplement or be cut.

---

## P0: Required Main-Paper Ablations

These are the ablations that should appear in the main paper.

### 1. Main Baseline Comparison

**Question:** Is the final method competitive with prior unsupervised panoptic segmentation systems?

Minimum table:

| Method | Supervision cue | Stereo/video at pseudo-label time? | Backbone/training | PQ | PQ_th | PQ_st | mIoU |
|---|---|---:|---|---:|---:|---:|---:|
| CUPS published baseline | stereo/video | yes | published | 27.80 | 17.7 | 35.81 | 26.8 |
| Ours pseudo-labels | monocular depth | no | no downstream training | 25.85 | 14.70 | 33.96 | 56.22 |
| Ours final model | monocular depth | no | Stage-3/self-training | 35.83 | 36.26 | 35.56 | 44.56 |

The pseudo-label row is a supervision-quality measurement, not a trained-model comparison. The main claim against CUPS should use the final model row.

If the same-backbone CUPS control is unavailable, phrase the claim as "exceeds the published CUPS baseline" and explicitly mark the same-backbone control as a limitation.

### 2. Component Removal Study

**Question:** Which proposed components actually contribute?

Minimum table:

| Variant | Semantics | Depth instances | DCFA | SIMCF | Training | PQ | PQ_th | PQ_st | mIoU |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| Raw k=80 + depth | CAUSE-TR k=80 | DepthPro tau=0.20 | no | no | none | 24.54 | 12.31 | 33.43 | 56.56 |
| DCFA only | CAUSE-TR k=80 | DepthPro tau=0.20 | yes | no | none | 25.22 | 13.16 | 33.99 | 56.16 |
| SIMCF only | CAUSE-TR k=80 | DepthPro tau=0.20 | no | yes | none | 25.27 | 13.64 | 33.73 | 56.57 |
| DCFA + SIMCF | CAUSE-TR k=80 | DepthPro tau=0.20 | yes | yes | none | 25.85 | 14.70 | 33.96 | 56.22 |
| Final model | CAUSE-TR k=80 | DepthPro or DA3 | yes | yes | Stage-3 | 35.83 | 36.26 | 35.56 | 44.56 |

Rows 1--4 are the protocol-consistent pseudo-label ablation from `reports/dcfa_depthpro_simcf_abc_pseudolabel_report.md`. The final model row is a trained-model anchor and should not be used for pseudo-label deltas. The CC-only/no-depth baseline is reported in Section 3 only. Do not use `results/depth_adapter/orig_k80_eval.json` as the CC-only row here: its PQ=26.08 comes from a separate adapter-eval protocol with an 8-stuff/11-thing split, so it is only useful for the DCFA no-adapter vs adapter sanity check.

This is the central ablation table. It should be internally protocol-consistent: same split, same evaluator, same mapping, same depth threshold policy.

### 3. Depth Model Comparison

**Question:** Is monocular depth generally useful, or did one depth model accidentally fit Cityscapes?

Minimum table:

| Depth source | Type | Optimal threshold | PQ | PQ_th | PQ_st | Notes |
|---|---|---:|---:|---:|---:|---|
| No depth / CC-only | none | n/a | 24.84 | 14.90 | 32.08 | semantic connected components only |
| SPIdepth | self-supervised monocular | 0.20 | 26.74 | 19.41 | 32.08 | historical baseline |
| DepthPro | monocular foundation model | 0.01 | 28.40 | 23.35 | 32.08 | current strongest depth source |
| Depth Anything v2 / DA2 | monocular foundation model | 0.03 | 27.10 | 20.20 | 32.08 | domain-agnostic control |
| Depth Anything v3 / DA3 | monocular foundation model | 0.03 | 27.37 | 20.90 | 32.08 | strongest DA alternative |

This table directly supports the "no stereo/video needed at pseudo-label generation time" claim.

### 4. Downstream Amplification

**Question:** Do pseudo-label improvements survive downstream training?

Minimum table:

| Stage | Input pseudo-labels | Training stage | PQ | PQ_th | PQ_st | mIoU |
|---|---|---|---:|---:|---:|---:|
| Raw pseudo-labels | raw k=80 + DepthPro tau=0.20 | none | 24.54 | 12.31 | 33.43 | 56.56 |
| Refined pseudo-labels | DCFA + DepthPro + SIMCF-ABC | none | 25.85 | 14.70 | 33.96 | 56.22 |
| Stage-2 model | refined pseudo-labels | supervised on pseudo-labels |  |  |  |  |
| Stage-3 model | refined pseudo-labels | self-training / EMA | 35.83 | 36.26 | 35.56 | 44.56 |

This table should not blur raw pseudo-label PQ with trained-model PQ.

### 5. Per-Class Breakdown

**Question:** Which classes improve, and which remain failure modes?

Cityscapes Stage-3 result at `best_pq_step=003000.ckpt`:

| Group | Class | PQ | SQ | RQ | Interpretation |
|---|---|---:|---:|---:|---|
| flat | road | 92.99 | 94.95 | 97.93 | strong |
| flat | sidewalk | 62.44 | 78.52 | 79.52 | strong |
| flat | parking | 0.00 | 0.00 | 0.00 | dead class |
| flat | rail track | 8.40 | 67.21 | 12.50 | weak / rare |
| construction | building | 83.54 | 85.70 | 97.48 | strong |
| construction | wall | 32.32 | 67.68 | 47.76 | moderate |
| construction | fence | 20.26 | 63.04 | 32.14 | weak |
| construction | guard rail | 0.00 | 0.00 | 0.00 | dead class |
| construction | bridge | 17.21 | 64.54 | 26.67 | weak / rare |
| construction | tunnel | 0.00 | 0.00 | 0.00 | dead class |
| object | pole | 2.05 | 72.27 | 2.83 | failure mode |
| object | polegroup | 0.00 | 0.00 | 0.00 | dead class |
| object | traffic light | 6.20 | 60.28 | 10.29 | failure mode |
| object | traffic sign | 37.19 | 65.92 | 56.42 | partially recovered |
| nature | vegetation | 84.70 | 85.71 | 98.82 | strong |
| nature | terrain | 35.68 | 73.42 | 48.60 | moderate |
| sky | sky | 86.05 | 89.93 | 95.69 | strong |
| human | person | 13.37 | 71.41 | 18.72 | weak co-planar/small-object behavior |
| human | rider | 22.94 | 62.65 | 36.61 | weak |
| vehicle | car | 70.71 | 88.74 | 79.68 | strong |
| vehicle | truck | 62.64 | 83.83 | 74.73 | strong |
| vehicle | bus | 76.67 | 90.84 | 84.40 | strong |
| vehicle | caravan | 0.00 | 0.00 | 0.00 | dead class |
| vehicle | trailer | 0.00 | 0.00 | 0.00 | dead class |
| vehicle | train | 77.17 | 88.75 | 86.96 | strong but rare |
| vehicle | motorcycle | 0.00 | 0.00 | 0.00 | dead class |
| vehicle | bicycle | 38.99 | 77.18 | 50.53 | moderate |

Source: `logs/eval_stage3_dcfa_simcf_abc_step3000.log`. Values are percentages over the full 27-class Cityscapes-style evaluation space. The main story is clear: large stuff and large vehicles are strong; thin structures, pedestrians, rare flat/construction classes, and motorcycle remain the honest limitations.

### 6. Failure Case Analysis

**Question:** Are the limitations honest and understandable?

Current status: partially ready. We have full-pipeline qualitative figures and many intermediate panels, but not yet a curated 6--10 example failure set where every failure type is explicitly selected and annotated.

| Failure type | Status | Available artifacts / next action |
|---|---|---|
| co-planar people or vehicles merge | partial | Use existing Stage-3 qualitative panels, then select 1--2 examples with RGB, depth, instance map, prediction, GT |
| depth over-fragments one object | ready as phenomenon, needs final panel | `figures/depthpro_instances_tau020_train.png`; SIMCF Step-B analysis in `reports/dcfa_depthpro_simcf_abc_pseudolabel_report.md` |
| thin objects vanish | supported quantitatively, needs visual examples | pole PQ 2.05, traffic light PQ 6.20; curate RGB, semantic map, prediction, GT |
| rare class collapses | supported quantitatively, needs visual examples | motorcycle PQ 0.10; use per-class table and curate examples |
| class-space mismatch, if using COCO | supported quantitatively, needs visual examples | COCO-Stuff-27 PQ 7.83 in `reports/cross_dataset_evaluation_report.md`; curate RGB, prediction, mapped GT |

Useful existing qualitative artifacts:

| Artifact | What it contains |
|---|---|
| `paper_bmvc2026/figures/fig5_qualitative.png` | 3-image full pipeline: RGB, depth, depth edges, semantic PLs, instance PLs, Stage-2, Stage-3 |
| `figures/depthpro_instances_tau020_train.png` | DepthPro instance pseudo-label visualization |
| `.claude/worktrees/beautiful-fermi/notebooks/visualizations/figures_stage3_pipeline/` | Stage-3 semantic, instance, panoptic, and pseudo-label panels for Frankfurt/Munster/Lindau |
| `.claude/worktrees/beautiful-fermi/notebooks/visualizations/figures_panoptic_merge/` | Panoptic merge/intermediate visualizations |

These figures belong in the main paper if space allows; otherwise include the best 2 to 3 in the main paper and the rest in supplement.

---

## P1: Strongly Recommended If Compute Allows

These are not all mandatory, but they make the paper more robust.

### 7. Three-Seed Headline Result

**Question:** Is the 35.83 PQ result stable?

Minimum table:

| Seed | PQ | PQ_th | PQ_st | mIoU |
|---:|---:|---:|---:|---:|
| seed 1 |  |  |  |  |
| seed 2 |  |  |  |  |
| seed 3 |  |  |  |  |
| mean +/- std |  |  |  |  |

Three full seeds are enough for the main paper if compute is limited. Five seeds are optional.

### 8. DCFA Design Sanity

**Question:** Is DCFA useful because of depth-conditioned residual adaptation, not just extra parameters?

Minimum table:

| Variant | Description | PQ or mIoU |
|---|---|---:|
| no adapter | frozen CAUSE-TR k=80 codes only | mIoU 52.69 / PQ 26.08 (adapter-eval protocol) |
| random/non-depth adapter | same capacity, no meaningful depth signal | missing |
| DCFA | residual depth-conditioned adapter | mIoU 55.29 / PQ 26.44 (adapter-eval protocol) |

If the paper claims zero initialization is important, add one comparison:

| Init | PQ or mIoU | Training stability |
|---|---:|---|
| zero init | mIoU 55.29 / PQ 26.44 (adapter-eval protocol) | implemented default; no isolated init-only comparison |
| standard random init | missing | not run |

No large initialization sweep is needed.

### 9. Depth Encoding Sanity

**Question:** Does depth encoding matter?

Minimum table:

| Encoding | PQ or mIoU |
|---|---:|
| no depth | mIoU 54.41 / PQ 24.40 |
| raw depth | mIoU 55.82 |
| sinusoidal depth | mIoU 56.76 / PQ 24.71 |

Do not run a large 8D/16D/32D/learned/log-depth sweep unless the paper makes depth encoding the core contribution.

### 10. SIMCF Internal Ablation

**Question:** Are SIMCF-A, SIMCF-B, and SIMCF-C each defensible?

Minimum table:

| Variant | PQ | PQ_th | PQ_st | Note |
|---|---:|---:|---:|---|
| no SIMCF | 24.54 | 12.31 | 33.43 | baseline |
| SIMCF-A | 24.54 | 12.31 | 33.43 | semantic consistency; no measured gain |
| SIMCF-AB | missing | missing | missing | not found in local logs/artifacts |
| SIMCF-ABC | 25.27 | 13.64 | 33.73 | full filter |

If compute is tight, use no SIMCF, SIMCF-AB, and SIMCF-ABC only.

### 11. Cross-Dataset Transfer

**Question:** Does the trained model transfer beyond Cityscapes-like data?

Minimum table:

| Dataset | Class-space relation | PQ | PQ_th | PQ_st | Interpretation |
|---|---|---:|---:|---:|---|
| Cityscapes | source | 35.83 | 36.26 | 35.56 | reference |
| KITTI | aligned driving classes | 34.85 | 31.94 | 36.40 | transfer |
| Mapillary | mostly aligned driving classes | 39.19 | 32.06 | 44.37 | transfer |
| COCO-Stuff-27 | disjoint/coarse classes | 7.83 | 7.83 | 7.84 | class-space mismatch limitation |

Only claim generalization where the class space is aligned. COCO should be framed as a class-space mismatch unless a COCO-specific training/evaluation pipeline is run.

---

## Cut Or Move To Supplement

The following are not required for the main paper:

| Item | Recommendation |
|---|---|
| ReLU vs GELU vs SiLU vs Mish | cut or supplement |
| large hidden-dimension sweep | supplement only |
| output dimension 128/256/512 | cut for current DCFA; output should remain 90D unless the method is redesigned |
| full zero-init scale sweep | replace with one zero-init vs random-init comparison |
| EMA alpha sweep | cut unless proposing a new self-training method |
| student/teacher mix sweep | cut unless proposing a new self-training method |
| training duration sweep | replace with a short validation trajectory |
| full hyperparameter sensitivity grid | supplement only |
| five full seeds | optional; three seeds are enough if compute is constrained |
| p-values | usually unnecessary; report mean +/- std |
| DINO layer selection | supplement only unless backbone choice becomes a central claim |
| every possible depth estimator | use CC-only, one historical depth model, and two modern foundation depth models |

---

## Recommended Main-Paper Tables

1. **Table 1:** Comparison to prior unsupervised panoptic methods.
2. **Table 2:** Component removal study.
3. **Table 3:** Depth model comparison.
4. **Table 4:** Downstream amplification from pseudo-labels to Stage-3 model.
5. **Table 5:** Per-class PQ breakdown.
6. **Figure 1:** Method overview.
7. **Figure 2:** Success and failure cases.

This is enough for a focused NeurIPS submission. The supplement can contain extra adapter sweeps, failed variants, implementation details, and additional qualitative results.
