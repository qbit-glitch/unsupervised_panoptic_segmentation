# Stage-4 SAM3 Alignment Bug Fix Report

## Summary

The Stage-4 SAM3 fine-object supervision was using masks in the wrong coordinate frame. The SAM3 masks were loaded after the CUPS self-training batch had already passed through geometric augmentation: copy-paste, random crop, and resolution jitter. As a result, the student model saw an augmented image, but `loss_fine_object` was computed using masks still aligned to the original unaugmented image.

For small objects, this is a severe training bug. A crop or resize that is harmless for large regions can completely move a bicycle, rider, pole, traffic light, or traffic sign mask away from the intended object. The model can then receive gradients on unrelated background pixels, which explains why Stage-4 did not reliably improve panoptic quality despite using SAM3 supervision.

The fix attaches SAM3 supervision before geometric augmentation, propagates the SAM fields through the same transformations as the image and pseudo-labels, and then computes `loss_fine_object` from those already-augmented tensors.

## Symptom

The training behavior was inconsistent with the intended Stage-4 objective:

- Small objects were still ignored or poorly segmented.
- PQ and PQ-things did not improve reliably.
- The MC-PanDA run with `MODE=thing_mc_panda`, `WEIGHT=0.01` improved briefly but then regressed.
- Earlier CCR history showed a better focal/IoU configuration, but simply changing the loss was not enough if the masks were geometrically misaligned.

The local alignment diagnostic confirmed the issue. Comparing the old resized-only SAM masks against correctly transformed masks gave approximately:

- Mean IoU: `0.15`
- Median IoU: `0.00`
- About `65%` of masks below `0.10` IoU

That means most SAM masks were not covering the object regions they were supposed to supervise.

## Root Cause

In the old flow, the Stage-4 batch was processed roughly like this:

1. Teacher model generated pseudo-labels.
2. Pseudo-labels were passed through copy-paste augmentation.
3. Pseudo-labels were cropped.
4. Pseudo-labels were resolution-jittered.
5. Student model ran on the augmented samples.
6. SAM3 masks were loaded from disk using the original image name.
7. `loss_fine_object` used those original-coordinate SAM3 masks against the augmented student logits.

Step 6 was the problem. The loaded masks had not gone through the same crop and resize as the student image. For fine-object supervision, this made the loss spatially wrong.

Teacher-gated modes had the same class of issue. Teacher logits used for gating were captured in one coordinate frame, while the student logits and augmented samples could be in another. So the teacher gate could also be spatially inconsistent.

## Files Changed

- `refs/cups/cups/augmentation.py`
- `refs/cups/cups/pl_model_self.py`
- `refs/cups/tests/test_sam3_alignment.py`

## Implementation Details

### 1. Attach SAM Supervision Before Augmentation

In `SelfSupervisedModel.training_step`, SAM3 supervision is now attached immediately after teacher pseudo-label generation and optional mask refinement, before copy-paste, crop, and resolution jitter.

The attached fields are:

- `sam_masks`
- `sam_ious`
- `sam_cls`
- `sam_teacher_logits`

This means SAM masks and teacher logits become part of the sample dictionary and travel through the same geometry as `image`, `sem_seg`, and `instances`.

### 2. Transform SAM Masks Through Random Crop

`RandomCrop` now applies the current crop parameters to attached SAM masks:

- Masks are cropped with nearest-neighbor semantics.
- Tiny or empty masks after crop are removed.
- `sam_ious` and `sam_cls` are filtered with the same validity mask.
- `sam_teacher_logits` are cropped with the same crop parameters.

This keeps SAM metadata synchronized with the surviving masks.

### 3. Transform SAM Masks Through Resolution Jitter

`ResolutionJitter` now resizes attached SAM fields using the same geometry as the sample:

- `sam_masks`: nearest-neighbor resize
- `sam_teacher_logits`: bilinear resize
- `sam_ious` and `sam_cls`: filtered if masks become too small or empty

This keeps the SAM supervision aligned to the final image resolution used by the student forward pass.

### 4. Preserve SAM Fields Through Copy-Paste

Copy-paste originally reconstructed the output sample from only core fields. That could silently drop auxiliary keys. The patch refactored the object paste operation into `_paste_one(...)` and preserves non-core sample fields when constructing the output.

The preserved auxiliary fields include:

- `sam_masks`
- `sam_ious`
- `sam_cls`
- `sam_teacher_logits`
- `file_name`

This prevents SAM supervision from disappearing before the fine-object loss is computed.

### 5. Collect Augmented SAM Tensors for Loss

Previously, `loss_fine_object` called `_load_sam_masks(...)` after augmentation. The patched code instead calls:

- `_collect_augmented_sam_supervision(...)`
- `_collect_augmented_sam_teacher_logits(...)`

These functions read the already-transformed fields from the augmented pseudo-label batch. The fine-object loss therefore uses masks, class labels, IoU weights, teacher logits, and student logits in the same coordinate frame.

## Why This Fix Matters

The Stage-4 objective is intended to apply this kind of signal:

> Inside this SAM3 fine-object mask, increase the semantic-head confidence for the correct target, especially the unified thing channel.

Before the fix, the actual signal was often closer to:

> Inside some unrelated crop/resize location, change the prediction.

That can actively harm learning. The problem is worst for small or thin classes:

- bicycle
- motorcycle
- rider
- person
- pole
- traffic sign
- traffic light

These objects occupy few pixels, so a small coordinate mismatch can destroy the supervision signal.

## Verification

The following checks were run locally:

```bash
.venv/bin/python -m pytest refs/cups/tests/test_sam3_alignment.py -q
```

Result: passed.

```bash
.venv/bin/python -m pytest refs/cups/tests/test_sam3_alignment.py refs/cups/tests/test_rare_pool_copy_paste.py -q
```

Result: passed.

```bash
.venv/bin/python -m py_compile refs/cups/cups/augmentation.py refs/cups/cups/pl_model_self.py refs/cups/tests/test_sam3_alignment.py
```

Result: passed.

A real-data CPU probe over 20 images also reported:

```text
all_sam_equal_instance_after_aug: true
```

This confirmed that the transformed SAM masks matched the transformed instance masks after augmentation.

The minimal patch was then applied to the existing remote Santosh checkout at:

```text
/home/santosh/cups
```

Remote syntax validation passed:

```bash
/home/santosh/anaconda3/envs/cups/bin/python -m py_compile \
  /home/santosh/cups/cups/augmentation.py \
  /home/santosh/cups/cups/pl_model_self.py
```

## Smoke-Test Result

A GPU smoke run was launched with the weaker MC-PanDA configuration:

- `MODE=thing_mc_panda`
- `WEIGHT=0.01`
- `USE_IOU_WEIGHTING=True`
- `FOCAL_GAMMA=2.0`

Best observed validation from the smoke run:

- `PQ 36.293`
- `PQ_things 36.928`
- `PQ_stuff 35.897`
- `mIoU 44.065`

This verified that the patched SAM path works on the remote GPU training setup and can recover PQ relative to the regressing run.

## Restarted Training Run

CCR memory identified the previous improving Stage-4 recipe as run `74xt`, logged under:

```text
/home/santosh/cups/logs/stage4_debug.log
```

That run used:

- `MODE=thing_focal_only`
- `WEIGHT=0.05`
- `USE_IOU_WEIGHTING=True`
- `FOCAL_GAMMA=2.0`

The active restarted run combines that CCR-good focal/IoU recipe with the SAM3 alignment fix:

```text
/home/santosh/cups/logs/stage4_alignfix_focal_w005_restart2.log
```

Verified remote active config:

```yaml
MODE: thing_focal_only
WEIGHT: 0.05
USE_IOU_WEIGHTING: True
FOCAL_GAMMA: 2.0
MIN_IOU_SCORE: 0.1
MIN_HARD_IOU: 0.1
```

The run was verified alive on both GTX 1080 Ti GPUs at approximately `99%` utilization.

## Important Caveat

The local YAML is still stale:

```text
refs/cups/configs/train_self_cityscapes_dinov3_vitb_stage4_fine_object_santosh.yaml
```

At the time of this report, the local copy still contains:

```yaml
MODE: thing_mc_panda
WEIGHT: 0.01
```

The remote active config was patched directly on Santosh to:

```yaml
MODE: thing_focal_only
WEIGHT: 0.05
```

Before launching another run from the local machine or syncing configs, the local YAML should be updated to match the intended recipe.

## Expected Outcome

This patch does not guarantee `37+ PQ`. It fixes a real coordinate-frame bug and restores the previously improving focal/IoU loss family.

Current confidence: approximately `75%` that the patched focal/IoU run will beat the broken or weak MC-PanDA path.

The main comparison targets from CCR run `74xt` are:

- `val200 PQ 36.25`
- `val400 PQ 36.40`
- `val600 PQ 36.19`
- `val400 PQ_things 36.67`

If the first validation is near or above `36.2 PQ`, the patch is behaving as expected. If the first validation is below `35.5 PQ` or `PQ_things` collapses, then the next suspect is no longer mask alignment; it is likely loss weight, loss family, or fine-object class filtering.

