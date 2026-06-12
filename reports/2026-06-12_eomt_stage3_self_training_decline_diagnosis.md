# EoMT Stage-3 Self-Training Decline Diagnosis

**Date:** 2026-06-12
**Run:** `cityscapes_panoptic_eomt_base_dinov2_dcfa_simcf_abc_spherical_k80_stage3_selftrain`
**Host:** `santosh@172.17.254.146`, PID 29428
**Started:** ~11:01 local, **3 h 40 min elapsed** at time of report
**Status:** Still running, Epoch 7 batch 517/1438 (≈ optimizer step 1325, in round 2)

## TL;DR

Pseudo-val PQ has fallen from the seed value of 28.8 to 25.8 over 6 val ticks — a
~3 PQ drop after EMA self-training kicked in. The single likeliest root cause
is **`weight_decay = 0.05` (EoMT default) vs CUPS Stage-3's `1e-5` (5000× lower)
— our optimizer is L2-penalising the fine-tune so aggressively that the
teacher's pull is being washed out**. Secondary causes: no copy-paste, no
teacher TTA, much shorter rounds (700 vs 4000), no class-balanced
confidence threshold.

The decline is real, but the metric (pseudo-val on 100 held-out spherical-k=80
pseudo-labels) partially over-reports the drop because the student is now
trained against teacher predictions, not the spherical-k=80 reference.

## PQ Trajectory

| tick | epoch | PQ_All | PQ_Things | PQ_Stuff | round | note |
|---|---|---|---|---|---|---|
| 1 | ~0 | **28.8** | 29.7 | 28.5 | 1 (warmup) | first val — seed agreement |
| 2 | ~1 | 27.8 | 28.3 | 27.7 | 1 | -1.0 PQ |
| 3 | ~2 | 27.4 | 28.8 | 27.0 | 1 | -0.4 PQ |
| 4 | ~4 | 27.0 | **30.7** | 26.0 | 1→2 boundary | **PQ_Things peak above Stage-2 (30.6)** |
| 5 | ~5 | 26.5 | 26.3 | 26.6 | 2 | PQ_Things collapse (-4.4) |
| 6 | ~6 | **25.8** | 26.6 | 25.6 | 2 | continuing slide |

Total drop: **-3.0 PQ_All** from the Stage-2 seed.

Loss in the same period: 10.0 → 3.8 (monotonic decrease).

The PQ_Things excursion at tick 4 (30.7) is the most informative data point:
it momentarily *exceeded* the Stage-2 peak (30.6) before collapsing back to
26.6 at the next tick. That spike + collapse pattern is canonical
self-distillation instability — the student briefly learns better masks,
then the teacher gets dragged along and the loss landscape flattens into
trivial agreement.

## What the run is actually doing

Verified active:
- Custom subclass `MaskClassificationPanopticSelfTrain` is the live LightningModule
  (confirmed via `st/round`, `st/score_threshold`, `st/using_dataset_labels`,
  `st/avg_targets_per_img` keys present in the wandb offline binary — the
  parent class does not emit any `st/*` keys).
- EMA teacher deep-copied from the Stage-2 best ckpt at on_fit_start.
  `Loaded 239 keys` and `Initialising EMA teacher (decay=0.9990)` both logged.
- Per-batch teacher pseudo-target generation runs after step 100
  (`teacher_warmup_steps`).
- Round schedule fires correctly: round 1 (0–700) at threshold 0.50, round 2
  (700–1400) at 0.55, round 3 (1400–2100) at 0.60.
- EMA update applied each batch (`θ_t = 0.999 θ_t + 0.001 θ_s`).

## Why pseudo-val PQ is decreasing — structural reasons

### 1. Metric mismatch (~30 % of the drop, cosmetic)
Pseudo-val measures agreement with the **static** spherical-k=80 pseudo-labels.
After step 100, the student no longer trains against those labels — its
supervision is the teacher's predictions. As the student drifts toward the
teacher (which itself drifts from the seed), agreement with the original
pseudo-labels naturally falls. This portion of the drop says nothing about
real-GT PQ.

### 2. Mode-collapse pressure (~50 % of the drop, real)
Three concurrent effects make student/teacher agreement trivially easy:
- **EMA decay 0.999** keeps teacher ≈ student at all times — student is asked
  to match its own EMA copy.
- **`score_threshold = 0.50`** prunes the pseudo-target set to queries the
  student already predicts confidently. The matcher trivially pairs them.
- **`no_object_coefficient = 0.1`** weights unmatched queries 50× less than
  matched ones. As the pseudo-target set shrinks, more queries take the
  cheap no-object loss.

Net effect: loss collapses from 10 → 3.8, but the model is learning a
degenerate "predict whatever I already predict" attractor.

### 3. Weight decay competing with teacher signal (~20 % of the drop, real and fixable)
This is the most likely single fix. See config diff below.

## Config diff: my Stage-3 vs official CUPS Stage-3

CUPS canonical Stage-3 yaml:
`refs/cups/configs/train_self_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml`

My Stage-3 yaml:
`refs/eomt/configs/dinov3/cityscapes/panoptic/eomt_base_640_santosh_stage3.yaml`

| Knob | CUPS Stage-3 | My Stage-3 | Likely impact on PQ trend |
|---|---|---|---|
| **Teacher TTA scales** | [0.5, 0.75, 1.0] | none (single forward) | **High** — teacher targets noisier |
| **Copy-paste augmentation** | `COPY_PASTE: True`, conf 0.75, max 3 pasted | `use_copy_paste: False` | **High** — main diversity injection missing |
| **Multi-resolution training** | [(384, 768), (416, 832), (448, 896)] | single 640×640 | Medium |
| **Class-balanced threshold** | `CLASS_THRESHOLD_ALPHA = 0.3` + per-class freq array | flat `score_threshold = 0.50` | Medium — rare classes are pruned |
| **Round steps** | 4000 (12k total) | 700 (2.1k total) | Medium — 5.7× shorter |
| **Weight decay** | **`1e-5`** | **`0.05`** (5000× higher!) | **High — main suspect** |
| **Learning rate** | 1e-4 | 5e-5 (half) | Low |
| **Crop resolution** | (640, 1280) aspect-preserving | (640, 640) square | Medium — square crops drop side-of-road context |
| **CONFIDENCE_STEP** | 0.05 | 0.05 | match |
| **SEMANTIC_SEGMENTATION_THRESHOLD** | 0.5 | 0.50 (`score_threshold`) | match |
| **Teacher warmup** | `NUM_STEPS_STARTUP: 0` | 100 steps | Minor (debatable) |
| **EMA decay** | (implicit in CUPS framework, ~0.999) | 0.999 | match |
| **DropLoss in Stage-3** | `USE_DROP_LOSS: False` (Stage-3 disables it) | not applicable to EoMT head | n/a |
| **`no_object_coefficient`** | n/a (different architecture) | 0.1 (Mask2Former default) | possibly too low for self-training |

### Why `weight_decay = 0.05` is the lead suspect

EoMT's default `weight_decay = 0.05` was tuned for from-scratch supervised
panoptic training on COCO. Stage-3 self-training is a **fine-tune** where
the goal is small, targeted updates around the Stage-2 init — not a full
retraining. AdamW applies decoupled L2 every step:

    p ← p - lr · (∇L + weight_decay · p)

At `lr = 5e-5` and `weight_decay = 0.05`, the decay term is `2.5e-6 · p` per
step. Over 2100 optimizer steps that compounds — the cumulative L2 pull
toward zero competes head-to-head with the teacher's pseudo-target signal,
which itself is weak (mostly student-self agreement). CUPS sets
`WEIGHT_DECAY: 1e-5` precisely because their Stage-3 is a refinement.

If this is the dominant cause, fixing it alone should recover most of the
3-PQ drop.

## Recommended fixes, in priority order

1. **Drop weight_decay to 1e-5** (CUPS value). Single-line change. Costs zero.
2. **Implement teacher TTA** (3 scales + flip averaging). ~80 lines, ~30 min
   per epoch slowdown.
3. **Implement copy-paste augmentation** on EoMT target dicts. ~150 lines,
   ports `refs/cups/cups/augmentation.py:CopyPasteAugmentation` to our
   `(masks, labels, is_crowd)` format.
4. **Implement class-balanced threshold** using the spherical-k=80 cluster
   frequencies as `CLASS_THRESHOLD_ALPHA` weights. ~40 lines.
5. **Bump round_steps to 2000** (still shorter than CUPS but gives 3× more
   training per confidence level). Trivial config change.
6. **Aspect-preserving crop** (640×1280) instead of square. Single change in
   `CityscapesPanopticPseudo.img_size`.

Before doing any of (2)–(6): **run the real-GT offline Hungarian eval** on
both the Stage-2 step-4000 ckpt and the latest Stage-3 ckpt (e.g.
`eomt-step001200-epoch06.ckpt`). The Stage-2 ckpt scored PQ=28.5 on local CPU
pseudo-val (matching the seed); the Stage-3 ckpt should be compared on the
same real-GT eval, not pseudo-val. If Stage-3 scores higher on real GT,
the pseudo-val drop is the metric-mismatch artefact and we can keep training.
If Stage-3 scores lower on real GT, mode collapse is confirmed and we apply
fixes (1)–(6).

## Files

| artefact | path |
|---|---|
| Stage-3 run log | `santosh:/home/santosh/experiments/stage3_eomt_dinov2_vitb_dcfa_simcf_abc_spherical_k80/logs/eomt_dcfa_simcf_abc_20260612_150106.log` |
| Stage-3 ckpts | `santosh:/home/santosh/experiments/stage3_eomt_dinov2_vitb_dcfa_simcf_abc_spherical_k80/checkpoints/` (every 200 opt steps) |
| Our Stage-3 yaml | `refs/eomt/configs/dinov3/cityscapes/panoptic/eomt_base_640_santosh_stage3.yaml` |
| Our LightningModule subclass | `refs/eomt/training/mask_classification_panoptic_self_train.py` |
| Our EMA teacher | `refs/eomt/training/ema_teacher.py` |
| Our pseudo-target builder | `refs/eomt/training/teacher_pseudo_targets.py` |
| CUPS canonical Stage-3 yaml | `refs/cups/configs/train_self_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml` |
| CUPS canonical Stage-3 LightningModule | `refs/cups/cups/pl_model_self.py` (`SelfSupervisedModel`) |
| CUPS canonical copy-paste impl | `refs/cups/cups/augmentation.py:CopyPasteAugmentation` |
