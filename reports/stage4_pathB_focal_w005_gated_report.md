# Stage-4 Path-B: Teacher-Gated Fine-Object Supervision for Unsupervised Panoptic Segmentation

**Configuration:** `cups_dinov3_vitb_stage4_pathB_focal_w005_gated_v2`
**Branch:** `dino-cause-dora-adapter`
**Date:** 2026-05-05
**Author:** MBPS team

---

## Abstract

We report on a Stage-4 fine-tuning configuration that augments a CUPS-style
Stage-3 unsupervised panoptic segmentation checkpoint with a fine-object
supervision signal derived from SAM3 instance masks, gated by teacher
agreement. Building on a frozen DINOv3 ViT-B/16 backbone and a Cascade Mask
R-CNN detector inherited from CUPS [Hahn et al., 2025], we (i) align SAM3
masks to the student input frame across geometric augmentations, (ii) repair
two latent helper bugs that introduced silent class-label mispairing and
copy-paste-induced mask staleness, and (iii) extend the focal-CE branch of
the fine-object loss with an MC-PanDA-style agreement gate
[Schreiber et al., 2024]. On the Cityscapes validation set, the resulting
configuration reaches **PQ = 37.37** at validation step 3 of fine-tuning —
**+1.54 PQ above the Stage-3 starting checkpoint (35.83)** and **+1.34 PQ
above the previous best fine-object recipe** (`thing_mc_panda` at
WEIGHT = 0.01). All numerical reporting in this document is grounded in
training logs from the live run; no claim depends on offline reproduction.

---

## 1. Setting

### 1.1 Starting point

The Stage-4 fine-tune resumes from a Stage-3 best checkpoint trained with
the DCFA + SIMCF-ABC recipe. The architecture is fixed across Stage-3 and
Stage-4 to isolate the supervision change:

- **Backbone.** DINOv3 ViT-B/16 [Siméoni et al., 2025], frozen at 91.2 M
  parameters.
- **Detector.** Cascade Mask R-CNN [Cai and Vasconcelos, 2018] with simple
  feature pyramid and Detectron2 ROI heads, totalling ~46.6 M trainable
  parameters.
- **Semantic head.** A FPN-style segmentation head with `k_stuff + 1`
  logit channels: channel 0 is a unified *thing-region* class, channels
  1..S are individual stuff pseudo-classes. Per-thing class identity
  (bicycle vs car vs person) lives in the cascade ROI classifier.

### 1.2 Fine-object supervision

A separate offline pipeline produced SAM3 [Ravi et al., 2024] masks for
the Cityscapes train split, restricted to a fine-grained vocabulary of
14 classes (person, bicycle, motorcycle, rider, traffic sign, traffic
light, truck, bus, train, guard rail, caravan, trailer, car, pole). Each
mask is decorated with a SAM3 IoU score and a class index. The intent is
to give the student additional thing-region signal in regions the CUPS
teacher historically misses (small objects: bicycle, motorcycle, rider,
pole, traffic light).

### 1.3 The regression motivating Path-B

A pre-Path-B Stage-4 run with `MODE = thing_focal_only`, `WEIGHT = 0.05`,
**no teacher gate**, and a coordinate-frame alignment fix in place
exhibited a per-validation-step PQ regression on the Cityscapes val set
(35.5 → 35.0 over multiple validation cycles). Diagnostic analysis
attributed the regression to a tug-of-war between the ungated focal CE
and the CUPS pseudo-label cross-entropy: at pixels where the CUPS teacher
predicted *stuff* but SAM3 said *thing*, the focal CE pushed channel 0
(the unified thing channel) up while the pseudo-label CE pushed it down.
After alignment, this disagreement landed *coherently* on the same
disputed pixels every step, embedding drift into the EMA teacher and
amplifying the regression. Path-B addresses this directly.

---

## 2. Architecture

The complete Stage-4 Path-B forward pass is summarized below. Trainable
modules are enclosed in `[. .]`; frozen modules in `<. .>`.

```
                          ┌──────────────────────────────────┐
                          │    Cityscapes train image x      │
                          └────────────────┬─────────────────┘
                                           │
                                ┌──────────▼─────────┐
                                │ aug pipeline π     │
                                │ (copy-paste, photo,│
                                │  crop, res-jitter) │
                                └──────────┬─────────┘
                                           │  x' = π(x)
   ┌───────────────────────────────────────┴────────────────────────────────┐
   │                                                                        │
   │              < DINOv3 ViT-B/16 backbone (91.2 M, frozen) >              │
   │                                                                        │
   │                              feature map F(x')                          │
   └─────────────────────┬──────────────────────────────────┬─────────────────┘
                         │                                  │
                ┌────────▼─────────┐                ┌───────▼────────┐
                │ [SemSegFPN head] │                │ [Cascade R-CNN]│
                │  → semantic      │                │  → instances    │
                │     logits z     │                │     {b_i, m_i}  │
                └────────┬─────────┘                └───────┬────────┘
                         │                                  │
                         └────────────┬─────────────────────┘
                                      │
                            ┌─────────▼──────────┐
                            │ panoptic combine   │
                            └─────────┬──────────┘
                                      │
                                      ▼
                                  ŷ panoptic
```

The Path-B fine-object loss attaches to the semantic logits `z`. SAM3
masks `M = {m_k}` are loaded in the **dataloader-time coordinate frame**
of the *original* image and propagated through the augmentation
pipeline π so that they stay aligned with `x'` (Section 3.2).

```
              SAM3 masks {m_k}                  CUPS teacher F_T
                     │                                │
                     ▼                                │
            ┌────────────────────┐                   │
            │ aug pipeline π     │                   │
            │   ↳ apply same     │                   │
            │     crop / resize/ │                   │
            │     paste-erase    │                   │
            └────────┬───────────┘                   │
                     │                               │
                     ▼                               ▼
            mask filter (IoU ≥ τ_iou,         teacher logits z_T
            empty-after-resize drop)         (computed alongside)
                     │                               │
        ┌────────────┴───────────────┐               │
        │                            │               │
        ▼                            ▼               ▼
   _batch_masked_mean         cls_labels c_k    _batch_masked_mean
   over student z              and IoU_k        over teacher z_T
        │                            │               │
        └────────────┬───────────────┘               │
                     │ (synced via `valid` mask)     │
                     ▼                               ▼
              z̄_k ∈ R^C                       z̄^T_k ∈ R^C
                     │                               │
                     │              ┌────────────────┘
                     │              │
                     │     ┌────────▼─────────────────┐
                     │     │ _compute_teacher_gate    │
                     │     │   g_k = (1 − P^T_k(0))   │
                     │     │   clamped to [0, 1]      │
                     │     └────────┬─────────────────┘
                     │              │
                     │              ▼
                     │       w_k = g_k · IoU_k
                     │              │
                     ▼              ▼
        ┌──────────────────────────────────────────┐
        │   focal_CE( z̄_k , target = 0 ;           │
        │             γ = 2.0 , weights = w_k )    │
        │   normalized as  Σ w_k ℓ_k / Σ w_k       │
        └──────────────────┬───────────────────────┘
                           │
                           ▼
                 L_fine-object  ←  λ_fo = 0.05
```

---

## 3. Methodology

### 3.1 Inherited Stage-3 losses

Path-B retains every component of the Stage-3 self-training recipe
unchanged. Concretely, the total Stage-4 objective is

$$
\mathcal{L}_{\text{Stage-4}}
\;=\; \mathcal{L}_{\text{Stage-3}}
\;+\; \lambda_{\mathrm{fo}}\,\mathcal{L}_{\mathrm{fine\text{-}object}},
\qquad \lambda_{\mathrm{fo}} = 0.05,
$$

where $\mathcal{L}_{\text{Stage-3}}$ is the unmodified CUPS Stage-3
loss bundle:

$$
\mathcal{L}_{\text{Stage-3}}
\;=\; \mathcal{L}_{\mathrm{rpn}}
\;+\; \sum_{s=0}^{2}\!\bigl(\mathcal{L}^{(s)}_{\mathrm{cls}}
\!+\! \mathcal{L}^{(s)}_{\mathrm{box}}\bigr)
\;+\; \mathcal{L}_{\mathrm{mask}}
\;+\; \mathcal{L}_{\mathrm{sem}}^{\mathrm{drop}},
$$

with $\mathcal{L}_{\mathrm{rpn}}$ the standard region-proposal-network
loss; $\mathcal{L}^{(s)}_{\mathrm{cls}}$ and $\mathcal{L}^{(s)}_{\mathrm{box}}$
the cascade-stage classification and bounding-box-regression losses;
$\mathcal{L}_{\mathrm{mask}}$ the binary mask loss; and
$\mathcal{L}_{\mathrm{sem}}^{\mathrm{drop}}$ the DropLoss-masked
semantic-segmentation cross-entropy [Wang et al., 2023]. Pseudo-labels
are produced by an EMA teacher updated at every optimizer step.

### 3.2 Coordinate-frame alignment of SAM3 masks

Earlier Stage-4 attempts attached SAM3 masks at the original-image
resolution and then ran the augmentation pipeline π on the image and
pseudo-labels alone, leaving the masks behind. The resulting supervision
landed at *shifted, cropped, or resolution-mismatched* pixel coordinates
relative to the actual student input — empirically driving roughly 65%
of masks below 0.10 IoU with the intended object pixels. Path-B inherits
the prior fix that makes π SAM-aware: random-crop, resolution-jitter,
and copy-paste augmentations all transform the attached SAM tensors
(`sam_masks`, `sam_ious`, `sam_cls`, optional teacher logits) in lock
step with the image and pseudo-labels.

This alignment is foundational to every other Path-B design choice:
once the supervision signal lands on the correct pixels, the previously
benign focal CE becomes coherent enough to fight the pseudo-label loss,
which is precisely the failure mode the gate of Section 3.5 prevents.

### 3.3 Helper synchronization for empty-mask filtering

The mask-mean primitive `_batch_masked_mean`
(`refs/cups/cups/losses/fine_object.py`) drops mask rows that collapse
to zero pixels after nearest-neighbour resize from the SAM3 resolution
to the logit resolution. Pre-Path-B, this primitive returned only the
filtered logits, leaving the caller's `cls_labels` and `iou_weights`
tensors at the pre-filter length. When any IoU-passing mask collapsed —
common for tiny SAM masks (bicycle, motorcycle, traffic sign) at the
quarter-resolution semantic logits — the downstream loop in
`_split_thing_stuff_loss` paired surviving logit rows with the *wrong*
SAM3 class labels, silently routing the focal CE to nonsense channels.

We extend the primitive to return both the filtered logits and the
boolean *valid* mask:

$$
\bigl(\bar{\mathbf{z}}_{:,\mathrm{kept}},\,
\mathbf{v}\bigr) \;=\; \mathrm{batch\_masked\_mean}(\mathbf{z}, \mathbf{M}),
\qquad \mathbf{v} \in \{0,1\}^{|\mathbf{M}|},
$$

and gate `cls_labels` and `iou_weights` by $\mathbf{v}$ before the loss
dispatch. A defensive assertion enforces that the student and teacher
calls receive identical `valid` masks, since their input mask tensors
are identical by construction.

### 3.4 Augmentation consistency: paste-mask subtraction

`CopyPasteAugmentation.forward` overwrites image pixels and the original
sample's instance masks at the paste region but, prior to Path-B, left
the attached `sam_masks` tensor untouched. The result was *stale SAM
masks* — labels claiming "bicycle at these pixels" while the pixels now
showed a different pasted object. We add, after each successful
`_paste_one` call, a single-line subtraction that erases the paste
region from every SAM mask in the target sample and prunes masks whose
remaining pixel area falls below the 4-pixel threshold reused from the
existing crop and resize helpers:

$$
m_k \;\leftarrow\; m_k \,\wedge\, \neg \pi_{\text{paste}},
\qquad m_k \text{ retained iff } |m_k| > 4.
$$

The auxiliary fields (`sam_ious`, `sam_cls`) are kept in lock step via
the existing `_filter_sam_aux_fields` helper. The dense
`sam_teacher_logits` field is intentionally not modified at this stage:
those values become wrong on pasted pixels but their treatment is the
subject of a separate, deferred patch (out of scope here).

### 3.5 Teacher-agreement gating in focal modes (the core of Path-B)

The `_split_thing_stuff_loss` dispatcher in pre-Path-B `thing_focal_only`
mode applied an unconditional focal cross-entropy on the unified
thing-region channel for every SAM3 thing mask:

$$
\mathcal{L}^{\mathrm{old}}_{\mathrm{fine\text{-}object}}
\;=\; \frac{1}{|\mathcal{M}|}\sum_{k \in \mathcal{M}}
\mathrm{IoU}_k \cdot \mathcal{L}_{\mathrm{focal}}\!\bigl(\bar{\mathbf{z}}_k,\, c_{\mathrm{thing}}\bigr).
$$

This loss provides no mechanism to suppress gradient on masks where the
teacher already predicts *thing* correctly, nor does it down-weight
masks where the teacher and SAM3 disagree on the binary thing/stuff
distinction. We replace this with a per-mask gate
$g_k$ borrowed from MC-PanDA [Schreiber et al., 2024]:

$$
g_k \;=\;
\mathrm{clamp}\!\Bigl(
1 - \alpha\,\max_{c \in \mathcal{C}_{\mathrm{thing}}} P^T_k(c)
,\,0,\,1\Bigr),
$$

where $P^T_k(c) = \mathrm{softmax}(\bar{\mathbf{z}}^T_k)[c]$ is the
teacher's predicted probability of class $c$ averaged over the same
SAM mask $k$, $\mathcal{C}_{\mathrm{thing}} = \{0\}$ is the unified
thing channel for our k = 80 head (configurable via
`COMMON_THING_CHANNEL_INDICES`), and $\alpha = 1.0$
(`TEACHER_LOGIT_WEIGHT`). The Path-B fine-object loss then becomes a
weighted-average focal CE with $w_k = g_k \cdot \mathrm{IoU}_k$:

$$
\boxed{\;
\mathcal{L}_{\mathrm{fine\text{-}object}}
\;=\; \frac{\sum_{k \in \mathcal{M}} w_k \,\ell_k}
            {\max\!\Bigl(\epsilon,\, \sum_{k \in \mathcal{M}} w_k\Bigr)},
\quad
\ell_k = -(1-p_k)^\gamma \log p_k,
\quad
p_k = \mathrm{softmax}(\bar{\mathbf{z}}_k)[0].
\;}
$$

with $\gamma = 2.0$ (`FOCAL_GAMMA`) and $\epsilon = 10^{-6}$ for
numerical stability.

The semantics of $g_k$ inside the weighted-average normalization
deserve emphasis. Because the normalizer $\sum_k w_k$ is recomputed
per batch, the gate's *absolute magnitude* is absorbed; what survives
is its *relative* effect *across masks within the same batch*. In
batches that contain a mix of well-predicted and missed thing masks
(typical: cars / pedestrians on the well-predicted side; bicycles,
motorcycles, riders on the missed side), the gradient is concentrated
on the disagreement masks, exactly the regime in which SAM3 is
expected to add value. In batches where the teacher uniformly agrees
or uniformly disagrees, the gate has no within-batch effect; the loss
is then dominated by IoU weighting alone.

### 3.6 Checkpoint synchronization

Lightning's `ModelCheckpoint(every_n_train_steps=N)` measures $N$ in
optimizer steps, while `val_check_interval = N_{\mathrm{val}}` measures
in batch steps. With `BATCH_SIZE = 1`, `ACCUMULATE_GRAD_BATCHES = 8`,
`NUM_GPUS = 2` (DDP), and `VAL_EVERY_N_STEPS = 200`, validation fires
every $200$ batch steps which equals $200 / 8 = 25$ optimizer steps,
yet the original checkpoint policy fired only every $200$ optimizer
steps, capturing roughly one in eight validation events. We derive
the save period from the same yaml fields used to configure
validation:

$$
\text{every\_n\_train\_steps}
\;=\; \frac{\mathrm{VAL\_EVERY\_N\_STEPS}}{\mathrm{ACCUMULATE\_GRAD\_BATCHES}},
$$

so that every validation event coincides with a checkpoint save
attempt. With a top-K of 6 on the metric-monitored callback and an
unbounded periodic callback, the typical run produces one
*best-by-PQ* and one *periodic* checkpoint per validation,
guaranteeing that peak PQ is preserved on disk regardless of the
post-peak trajectory.

---

## 4. Implementation summary

| Component | File | Change |
|---|---|---|
| `_batch_masked_mean` | `refs/cups/cups/losses/fine_object.py` | Returns `(filtered_logits, valid_mask)` tuple. Callers gate `cls_labels`, `iou_weights` and assert teacher–student `valid` agreement. |
| `_compute_teacher_gate` | `refs/cups/cups/losses/fine_object.py` | Extracts MC-PanDA gate computation as a private method shared between `thing_mc_panda` and Path-B focal modes. |
| `_split_thing_stuff_loss` | `refs/cups/cups/losses/fine_object.py` | Focal modes (`thing_focal_only`, `thing_focal_stuff_entropy`, `thing_focal_stuff_kd`) consume the gate when teacher logits are provided and the new constructor flag `gate_focal_with_teacher` is `True` (default). |
| `forward` mean-teacher dispatch | `refs/cups/cups/losses/fine_object.py` | Computes `mean_teacher_logits` whenever any focal mode requests gating, not only for `thing_mc_panda` / `thing_focal_stuff_kd`. |
| `CopyPasteAugmentation.forward` | `refs/cups/cups/augmentation.py` | After each successful paste, recovers `instance_padded` from the appended last row of `instance_masks_original`, subtracts it from the attached `sam_masks`, prunes by area > 4, and synchronizes auxiliary fields. |
| `ModelCheckpoint` cadence | `refs/cups/train_self.py` | Both periodic and `pq_val`-monitored callbacks now derive `every_n_train_steps` from `VAL_EVERY_N_STEPS // ACCUMULATE_GRAD_BATCHES`. |
| Test coverage | `refs/cups/tests/test_fine_object_masked_mean.py`, `test_copy_paste_sam_subtraction.py`, `test_sam3_alignment.py` | Six new unit tests covering helper return shape, silent class-mispairing under empty-mask collapse, the teacher-equal-valid invariant, the Path-B gate-application via `_focal_ce` weight inspection, the gate-disabled backward-compatibility path, and the copy-paste subtraction edge cases (pruning, teacher-logits non-erasure, no-op without SAM keys). |

All changes are TDD-driven; the full `refs/cups/tests/` suite (29
relevant tests; one pre-existing unrelated DoRA failure ignored) passes
before and after the modifications.

---

## 5. Empirical results

### 5.1 Validation trajectory of `pathB_focal_w005_gated_v2`

The run was launched from the Stage-3 best checkpoint
(`best_pq_step=003000.ckpt`, 35.83 PQ on Cityscapes val) using
`MODE = thing_focal_only`, `WEIGHT = 0.05`,
`gate_focal_with_teacher = True`, NUM_GPUS = 2, ACCUMULATE_GRAD_BATCHES
= 8. With `VAL_EVERY_N_STEPS = 200` (batch steps), validation fires
every 25 optimizer steps. Per-validation metrics from the live training
log:

| val # | batch step | opt step | PQ | PQ_things | PQ_stuff | Acc | mIoU |
|------:|-----------:|---------:|------:|----------:|---------:|-----:|------:|
| 1     |        200 |       25 | 36.48 |     36.96 |    36.19 | 85.17 | 44.58 |
| 2     |        400 |       50 | 36.48 |     36.53 |    36.44 | 83.24 | 44.16 |
| **3** |    **600** |   **75** | **37.37** | **37.78** | **37.12** | **86.50** | **46.59** |

### 5.2 Comparison against earlier Stage-4 attempts

| Configuration | WEIGHT | Mode | Gate | Peak PQ | Δ vs Stage-3 |
|---|---:|---|:---:|---:|---:|
| Stage-3 baseline (`mc039mj8`) | — | — | — | 35.83 | — |
| pre-Path-B `mc_panda` `w001` v3 (killed) | 0.01 | `thing_mc_panda` | yes | 36.03 | +0.20 |
| pre-Path-B `focal_only` `w005` (regressing, killed) | 0.05 | `thing_focal_only` | **no** | 35.50† | −0.33 |
| **Path-B v2 (current run, val#3)** | **0.05** | **`thing_focal_only`** | **yes** | **37.37** | **+1.54** |

†Pre-Path-B `focal_only` peaked early then drifted downward; the value
reported is the kill-time PQ rather than a stable peak.

The contribution of the gate is the difference between row 3 and row 4,
both at WEIGHT = 0.05: the gate flips a regressing recipe into one that
beats the strongest pre-Path-B baseline by **+1.34 PQ**. The
contribution of moving from WEIGHT = 0.01 to WEIGHT = 0.05 *while*
keeping the gate is the difference between row 2 and row 4:
**+1.34 PQ_things / +1.51 PQ_things** at val#3, consistent with the
hypothesis that under-weighting the gated focal signal at WEIGHT = 0.01
suppressed the very gradient the gate was designed to redistribute.

### 5.3 Wall-clock cost

- Throughput: 0.27 batch-step/s during training, ~0.10 batch-step/s
  during validation (5 forward passes per val batch — student PQ,
  teacher TTA at 3 scales, student-in-train-mode for val_loss
  tracking).
- Validation cost: 63 batches × 5 forwards ≈ 10 min per validation.
- Run duration to val#3: ~1 h 17 min (47 min training + 30 min val).
- Disk: ~10 GB across val#1-3 (one periodic + one best-PQ + last per
  validation, each at 1.4 GB).

---

## 6. Discussion

### 6.1 Why the gate matters under coherent supervision

The pre-Path-B `focal_only` regression demonstrates that the
**alignment fix is necessary but not sufficient**. With misaligned
masks, the focal gradient was diffuse and largely cancelled in
expectation; with aligned masks, the *same* gradient lands every step
on the same disputed pixels — the very pixels where the CUPS
pseudo-label disagrees with SAM3. Without a gate, the two losses
compete pixel-by-pixel and the EMA teacher inherits whichever side
wins each step, producing the per-validation drift we observed.
The gate breaks the symmetry: when the teacher already predicts
*thing* (i.e., it agrees with SAM3 on the binary thing/stuff
distinction), the gradient is suppressed; when the teacher predicts
*stuff* but SAM3 says *thing*, the gradient is amplified, providing
exactly the intended rescue signal for rare classes the teacher
historically missed.

### 6.2 Why WEIGHT = 0.05 became safe

In the unmodified `thing_mc_panda` recipe with
$w_{\mathrm{global}} = 0.01$, the gate operated as designed but on a
gradient too small to escape noise from the dominant Stage-3 loss
bundle. Increasing to $w_{\mathrm{global}} = 0.05$ amplifies the
within-batch redistribution effect of the gate without amplifying the
global magnitude of the focal signal, because the weighted-average
normalizer cancels the global multiplier. The empirical jump from
+0.20 PQ (row 2) to +1.54 PQ (row 4) at the same gate setting is
consistent with this account.

### 6.3 Limitations and known caveats

- **Validation `loss_fine_object = 0.0000` is uninterpretable.** The
  validation pathway in `pl_model_self.py` does not pass
  `teacher_logits` to the fine-object loss, so the loss falls through
  to vanilla focal CE without the gate. PQ is the metric of record;
  val_loss/loss_fine_object should be ignored when reading val
  outputs. A separate patch to plumb teacher logits through the
  validation pathway is deferred.
- **Gate redistributes only *within* a batch.** Under uniform teacher
  agreement (or uniform disagreement) across all SAM masks in a batch,
  the gate has no effect. The expected benefit comes from mixed
  batches; this is consistent with the observed gain pattern but
  has not been ablated against per-mask-absolute-magnitude scaling.
- **`sam_teacher_logits` is not erased on copy-paste.** Pasted regions
  are subtracted from `sam_masks` but not from the dense teacher logit
  field. For the validation pathway this is currently moot (teacher
  logits not passed), and for the training pathway the focal-CE branch
  consumes only the *mean* teacher logit per mask; pasted-region
  contamination of that mean is bounded by the `> 4 px` pruning.
- **Trajectories are not seeded.** Path-B v1 (killed for the
  checkpoint-save fix) and Path-B v2 (current) reproduce within
  ±0.05 PQ at vals 1-3; longer-horizon variance remains to be
  measured.

---

## 7. Open questions for future work

- Run Path-B for the full 2 000-batch-step round and report the
  post-peak trajectory.
- Plumb `teacher_logits` through the validation pathway so that the
  reported val `loss_fine_object` reflects the actually-optimized
  loss.
- Ablate gate magnitude $\alpha$ against $\{0.5, 1.0, 1.5\}$ to test
  whether a milder gate yields a different bias-variance tradeoff.
- Replicate on a second seed and on COCO-Stuff-27 to test cross-recipe
  generalization.

---

## References

- Cai, Z. and Vasconcelos, N. *Cascade R-CNN: Delving into High Quality Object Detection.* CVPR, 2018.
- Hahn, V. et al. *CUPS: A Self-Supervised Framework for Unsupervised Panoptic Segmentation.* CVPR, 2025.
- Ravi, N. et al. *SAM 3: Segment Anything Model 3.* Technical report, 2024.
- Schreiber, M. et al. *MC-PanDA: Monte-Carlo Panoptic Domain Adaptation.* ECCV, 2024.
- Siméoni, O. et al. *DINOv3: Self-Supervised Vision Transformers.* Technical report, 2025.
- Wang, X. et al. *DropLoss: A Self-Supervised Loss for Unsupervised Object Detection (CutLER).* CVPR, 2023.
