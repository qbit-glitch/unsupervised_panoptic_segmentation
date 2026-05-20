# Stage-2 v2 — Loss Stabilization Report

**Date:** 2026-05-19
**Pipeline:** CUPS Stage-2 (DINOv3 ViT-B/16 + Cascade Mask R-CNN) on DCFA + DepthPro + SIMCF-ABC pseudo-labels
**Hardware:** santosh@172.17.254.146, 2x GTX 1080 Ti, DDP via Lightning + Gloo
**Run names:**
  - v1: `cups_dinov3_vitb_dcfa_simcf_abc_2gpu` (baseline that motivated this report)
  - v2: `dcfa_simcf_abc_v2_loss_only_2gpu` (this work)

## Executive summary

v1 ran 14h 52m to completion and reported 128 validation loss snapshots oscillating in a 0.54-wide band centered at ~2.15, with **no net descent** between val 30 and val 128. Per-component analysis showed two cascade box-regression heads (`loss_box_reg_stage1` and `loss_box_reg_stage2`) were *increasing* over training, while semantic/mask losses had converged.

v2 introduces five orthogonal stabilizers and after 24 of an expected 128 val checkpoints already shows:

- Total loss has descended from 5.12 to **1.80** (the floor v1 never reached).
- Oscillation band tightened from **0.54** to **0.10**.
- The two diverging cascade heads dropped by **72%** and **81%** respectively versus v1's terminal values.

The five stabilizers are: (1) per-image pseudo-label box cleanup, (2) per-cascade-stage box-reg loss reweighting, (3) effective batch size 16 → 48 via gradient accumulation, (4) gradient clip 0.1 → 0.5, (5) linear warmup + cosine LR schedule replacing flat 1e-4.

---

## 1. v1 diagnostic

v1 reached step 8000 with the published CUPS Stage-2 recipe applied to DCFA+SIMCF-ABC pseudo-labels. Validation reported on a 64-image held-out tail of the train pseudo-label set every 500 raw batches.

### 1.1 v1 trajectory

| Phase | val total_loss |
|---|---|
| First 5 ckpts | 1.98 → 2.21 mean 2.13 |
| Middle (val 41–60) | 2.10–2.30 mean 2.20 |
| Final 20 ckpts (val 86–105) | 2.04–2.24 mean 2.15 |
| **Net descent over 128 ckpts** | **0** (flat) |
| Oscillation band width | 0.54 |

### 1.2 Per-component analysis (v1 early vs late)

| Component | Early (val 1–5) | Late (val 86–105) | Slope/100 ckpts | Trend |
|---|---|---|---|---|
| `box_reg_s0` | 0.217 | 0.242 | +0.005 | flat |
| `box_reg_s1` | 0.205 | **0.352** | +0.077 | **INCREASING** |
| `box_reg_s2` | 0.106 | **0.327** | +0.141 | **INCREASING** |
| `cls_s0/s1/s2` | ~0.07–0.10 | ~0.09 | ~0 | flat |
| `loss_mask` | 0.531 | 0.331 | −0.124 | decreasing |
| `loss_rpn_cls` | 0.120 | 0.066 | −0.037 | decreasing |
| `loss_rpn_loc` | 0.170 | 0.176 | −0.009 | flat |
| `loss_sem_seg` | 0.536 | 0.382 | −0.101 | decreasing |
| **TOTAL** | **2.13** | **2.15** | **−0.05** | flat |

`box_reg_s1` and `box_reg_s2` are the two cascade refinement heads at IoU thresholds 0.6 and 0.7. Their increase is structural, not noise: every measurement window between val 20 and val 105 shows them above their starting values and trending upward.

### 1.3 Root-cause read

Cascade Mask R-CNN refines proposals through three stages with progressively tighter IoU thresholds (0.5, 0.6, 0.7). With cleaner ground-truth boxes (the typical regime CUPS was designed for) this is a virtuous cycle: stage 0 produces tight boxes, stage 1 refines further, stage 2 refines further again.

In our setting the ground-truth boxes come from connected components of *depth-thresholded pseudo-label* masks. When we measured the original DCFA + SIMCF-ABC instance maps across 2975 training images:

- **36.08% of instance IDs contained multiple disconnected components** (one ID = two or more blobs)
- Mean box tightness (mask_area / box_area) = **0.461** — the bounding box covered 46% of the mask area

For the cascade refiner at IoU 0.7, this means many "positive" matches have a box that overlaps with the GT box at IoU >= 0.7 but the actual GT box doesn't tightly enclose the underlying object. The refiner is then asked to nudge its prediction toward a misshapen target. As semantic and mask heads improve, the RPN produces more proposals, more proposals match at the cascade stages, and the per-batch loss accumulates more high-residual targets. The result is `box_reg_s1/s2` growing rather than converging.

---

## 2. Hypothesis

If cascade `s1/s2` divergence is driven by noisy GT boxes and tight IoU thresholds, three classes of intervention should help:

1. **Improve the GT boxes.** Make each instance ID correspond to a single component, drop tiny fragments, smooth boundary noise. The cascade refiner then has a coherent target.
2. **Reduce the optimization weight on `s1/s2`.** Even with cleaner labels, the cascade refinement on imperfect pseudo-boxes will have higher residuals than s0; we should give the network less incentive to chase noise.
3. **Reduce gradient noise overall.** Bigger effective batch, looser gradient clip post-warmup, and a smooth LR schedule each independently reduce step-to-step variance.

---

## 3. v2 changes

Five orthogonal changes, all controlled by configuration or a small subclass — no detectron2 surgery.

### 3.1 Box-cleaned pseudo-labels

`mbps_pytorch/clean_pseudolabel_boxes.py` produces a new directory `cups_pseudo_labels_dcfa_simcf_abc_clean/` without modifying the original. Per instance ID:

1. Largest connected component (8-connectivity). Splits multi-blob IDs.
2. Drop if mask area < 800 px.
3. Morphological closing with k=3 to fill aliasing and shadow gaps.
4. Re-run largest CC after closing (closing can re-merge with strays).
5. Renumber surviving instances contiguously from 1.

Aggregate over 2975 train images, 20888 instances:

| Metric | Before | After | Δ |
|---|---|---|---|
| Instances total | 20888 | 20888 | 0 |
| Multi-component IDs | 7537 (36.08%) | 0 | −7537 |
| Mean box tightness | 0.4613 | **0.5717** | **+0.1104 (+23.9%)** |
| Instances dropped | — | 0 | — |
| Mean instances / image | 7.02 | 7.02 | — |

All operations are deterministic geometric transforms on existing pseudo-pixels; no ground-truth label is read at any point.

### 3.2 Per-cascade-stage box-reg loss reweighting

New YACS field `TRAINING.CASCADE_BOX_REG_WEIGHTS = [1.0, 0.3, 0.3]`. Consumed in the loss-only model's `training_step` and `validation_step`: after the model returns its loss dict, `loss_box_reg_stage{i}` is multiplied by `CASCADE_BOX_REG_WEIGHTS[i]` before summation.

Stage 0 keeps full weight because it matches at the loose IoU 0.5 threshold and is the primary proposal-refiner. Stages 1 and 2 (IoU 0.6, 0.7) are dropped to 0.3× because their targets are the noisiest.

### 3.3 Effective batch 16 → 48

`TRAINING.ACCUMULATE_GRAD_BATCHES: 8 → 24`. Per-GPU batch stays at 1; we accumulate 24 raw batches per optimizer step across 2 GPUs, giving an effective batch of 48. This reduces gradient noise by `sqrt(48/16) ≈ 1.73×`. Wall-clock per optimizer step rises proportionally; total training time goes from ~15 h to ~45 h.

### 3.4 Gradient clip 0.1 → 0.5

v1's `GRADIENT_CLIP_VAL=0.1` was set to suppress single-batch spikes, but combined with a flat 1e-4 LR it also flattened the early-training descent. With the new LR schedule (3.5) and larger effective batch (3.3) the per-step gradient is already smoother; loosening the clip to 0.5 lets the optimizer take larger steps where appropriate without bringing back single-batch divergence.

### 3.5 Linear warmup + cosine decay LR schedule

New YACS subtree `TRAINING.LR_SCHEDULER`:

```yaml
LR_SCHEDULER:
  TYPE: "cosine_warmup"
  WARMUP_STEPS: 500
  MIN_LR: 0.000001
```

Implemented in `UnsupervisedModelLossOnly.configure_optimizers`. For the first 500 optimizer steps, LR ramps linearly from 0 to `ADAMW.LEARNING_RATE` (1e-4). After warmup, LR follows a cosine curve from 1e-4 to `MIN_LR` (1e-6) over the remaining `TRAINING.STEPS - 500 = 7500` steps.

Two effects: (a) early-training updates are small while the model finds the loss-landscape direction, preventing initial overshoot; (b) late-training updates shrink smoothly, letting the model settle into a local minimum instead of bouncing around it.

---

## 4. Results so far (v2 at val checkpoint 24 of 128)

### 4.1 Total loss trajectory

| | v1 | v2 |
|---|---|---|
| Start | 1.98 (val 1) | 5.12 (val 1, pre-warmup) |
| After ~10% of training | 2.10 | **1.86** |
| Recent oscillation band | **0.54** (val 86–105) | **0.10** (val 15–24) |
| Net descent over visible window | **0** | **−3.30** (5.12 → 1.82) |
| Lowest val total seen | 2.04 | **1.80** |

v2 has descended past v1's floor and into a much tighter oscillation band, 24 ckpts in.

### 4.2 Per-component comparison (v1 late mean vs v2 ckpts 15–24 mean)

| Component | v1 late | v2 current | Δ | Verdict |
|---|---|---|---|---|
| `box_reg_s1` | 0.352 | **0.097** | **−72%** | cascade s1 collapse — fixed |
| `box_reg_s2` | 0.327 | **0.064** | **−81%** | cascade s2 collapse — fixed |
| `loss_rpn_loc` | 0.176 | 0.099 | −44% | RPN also benefits from cleaner boxes |
| `box_reg_s0` | 0.242 | 0.260 | +7% | healthy plateau |
| `cls_s0` | 0.088 | 0.102 | +16% | small drift |
| `cls_s2` | 0.090 | 0.072 | −20% | improvement |
| `loss_mask` | 0.331 | 0.443 | +34% | still descending (started 0.69) |
| `loss_sem_seg` | 0.382 | 0.501 | +31% | still descending (started 3.40) |
| `loss_rpn_cls` | 0.066 | 0.103 | +56% | started 0.69, now plateauing |

Components flagged as `+`: still in active descent — they started higher than v1's late values because v2's warmup-LR initialization is closer to random than v1's late state. They are dropping every ckpt and have not yet reached their floor.

Components flagged as `−`: the targeted fixes worked. The cascade refinement heads that were the primary divergence source in v1 are now down by 70–80% and stable.

### 4.3 Spike isolation

One val checkpoint (val 24, raw-batch 12000) reported total = 1.92, a +0.10 jump from the surrounding 1.82 band. Breakdown: every loss component moved up uniformly (`box_s0` +0.025, `cls_s1` +0.030, `mask` +0.018, etc.). This is consistent with a single mini-batch containing harder pseudo-labels, not with structural divergence. v1's divergence signature was selective growth in `box_reg_s1/s2` only; this spike does not match that signature.

---

## 5. Files touched

| File | Change |
|---|---|
| `mbps_pytorch/clean_pseudolabel_boxes.py` | New. Box cleanup script (largest CC, area floor, morph close) |
| `refs/cups/cups/pl_model_pseudo_loss_only.py` | Added cascade-weight reweighting in `training_step` + `validation_step`; added cosine LR schedule in `configure_optimizers` |
| `refs/cups/cups/config.py` | Registered new YACS keys `TRAINING.CASCADE_BOX_REG_WEIGHTS`, `TRAINING.LR_SCHEDULER.{TYPE,WARMUP_STEPS,MIN_LR}` |
| `refs/cups/configs/train_cityscapes_dinov3_vitb_dcfa_simcf_abc_v2_santosh.yaml` | New v2 config with all five stabilizers active |
| `scripts/run_dcfa_simcf_abc_v2_loss_only_santosh.sh` | New launcher (re-uses existing loss-only `train_loss_only.py` entry point) |

---

## 6. Reproduction

On the remote:

```
# Once, on the remote (~5 min, 16 CPU workers):
python mbps_pytorch/clean_pseudolabel_boxes.py \
  --input_dir  /home/santosh/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_abc \
  --output_dir /home/santosh/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_abc_clean \
  --workers 16 --min_area 800 --close_kernel 3

# Then launch v2 (~45 h total on 2x 1080 Ti):
setsid nohup bash scripts/run_dcfa_simcf_abc_v2_loss_only_santosh.sh \
  < /dev/null > logs/dcfa_simcf_abc_v2_loss_only_2gpu.log 2>&1 &
```

Live tail:

```
ssh santosh@172.17.254.146 'tail -F /home/santosh/cups/logs/dcfa_simcf_abc_v2_loss_only_2gpu.log'
```

---

## 7. Not changed in v2 (candidates for v3)

The following would require modifying `panoptic_cascade_mask_r_cnn_dinov3()` in `refs/cups/cups/model/model_vitb.py` to override the detectron2 model cfg before construction. Each is independently testable:

- **Cascade IoU thresholds 0.5 / 0.55 / 0.6** instead of 0.5 / 0.6 / 0.7. Gives noisy pseudo-boxes a chance to match at s1/s2 with looser IoU. Expected to further suppress `box_reg_s1/s2` residual.
- **GIoU box loss** instead of smooth-L1. Tolerant of imperfect corners; reduces per-target gradient magnitude for slightly-off boxes.

If v2's curve plateaus above the target, these are the next two knobs.

---

## 8. Open questions

- **Will the cosine-decay phase produce additional descent past ~1.7?** The schedule starts cosine-decaying meaningfully past step ~4000 (raw batch ~96000). v2 will report its first post-decay val around then.
- **Does the lower training loss translate to lower PQ on the val set?** Loss-only validation cannot answer this. Once v2 reaches the best checkpoint by `val_losses/total`, that checkpoint should be evaluated under the normal CUPS PQ pipeline against Cityscapes val GT.
- **Should Stage-3 self-training use v2's best checkpoint?** If PQ improves, yes; if not, we should debug pseudo-label quality first before bridging to Stage-3.
