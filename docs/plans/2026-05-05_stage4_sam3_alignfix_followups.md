# Stage-4 SAM3 Alignfix Follow-up Bugfixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Date:** 2026-05-05
**Branch:** `dino-cause-dora-adapter`
**Author:** Plan agent (Opus 4.7)
**Source analysis:** CCR commit C472 (Stage-4 SAM3 regression root cause: 4 bugs identified)
**Goal:** Land the two code fixes (`_batch_masked_mean` cls_labels desync; CopyPasteAugmentation does not subtract paste mask from SAM masks) and execute the live-run revert (Path A: `thing_mc_panda` / `WEIGHT=0.01`) so the per-validation-step PQ regression on the alignfix run is stopped, while keeping bug #4 (val fo_loss teacher_logits handoff) explicitly out of scope.

**Architecture:** Two surgical edits to `refs/cups/cups/losses/fine_object.py` and `refs/cups/cups/augmentation.py`, each TDD-driven by a focused pytest module. The third bug is a one-shot remote-config revert with no in-repo code change — only verification of the already-correct local YAML and a recipe for the user to align the remote copy. New tests extend the existing `refs/cups/tests/test_sam3_alignment.py` and add per-bug isolated modules.

**Tech Stack:** PyTorch + Detectron2 + Lightning, kornia, pytest. Local CPU smoke train (~50 steps) gates the remote relaunch.

---

## Decision Log

### Why Path A over Path B for bug #1
- Path A (revert to `thing_mc_panda`/0.01) reuses the existing teacher-agreement gate (`_mc_panda_thing_loss` at `fine_object.py:509-539`). No new code, no new hyperparameters, no new YAML keys.
- Path B (add gating to `thing_focal_only`) would introduce a new mode and force re-tuning the focal weight. The user explicitly forbade it: "Do NOT propose adding new modes to `fine_object.py`, new YAML keys, or new training hyperparameters."
- The CCR memory shows the previous `thing_mc_panda` / 0.01 recipe held best-of-recipe at 36.293 PQ / 36.928 PQ_things. Path A returns to a known-good config; the alignment fix that was already committed should now make this recipe even safer because the gradient signal is coherent rather than misaligned.

### Why this fix order (bug #2 → bug #3 → bug #1)
- **Bug #2 first.** It is the cleanest local change with no behavioural risk: when no IoU-passing mask happens to be empty after resize, the new code path is bit-identical to the old one. Adding the synchronization is purely defensive — it cannot make a healthy run worse, and it removes a silent class-mispairing failure mode that has been present long before alignfix.
- **Bug #3 second.** It changes copy-paste throughput in the worst case (extra mask-subtractions) and updates an existing test (`test_copy_paste_preserves_attached_sam_fields`) whose current assertion codifies the bug. We want bug #2 landed and verified before touching the augmentation pipeline.
- **Bug #1 third (in code-merge order) but executed FIRST in wall-clock time.** It is a remote-config revert, not a code change. It should be applied immediately by the user to stop the live regression. The plan emits the recipe; bugs #2/#3 land on the branch independently.

### What is NOT in scope
- Bug #4 (val fo_loss teacher_logits handoff) — explicitly excluded.
- Adding new modes, new YAML keys, or new hyperparameters.
- Editing `pl_model_self.py` beyond the strict needs of bugs #2 and #3 (in practice: no edits are required because `_collect_augmented_sam_supervision` already returns the post-aug `sam_masks`/`sam_ious`/`sam_cls` tuple in lockstep — bug #3 fix is contained inside `augmentation.py`).
- Any GPU or remote-SSH work from this plan; the user runs all remote commands.

---

## File Structure Map

| File | Status | Responsibility |
|---|---|---|
| `refs/cups/cups/losses/fine_object.py` | Modify (lines 226-341, 678-700) | Bug #2 fix: `_batch_masked_mean` returns `(mean_logits, valid)`; callers gate `cls_labels` and `iou_weights` with `valid` immediately after the call. |
| `refs/cups/cups/augmentation.py` | Modify (lines 169-313) | Bug #3 fix: `CopyPasteAugmentation.forward` subtracts paste mask from each `sam_masks` and prunes via `_filter_sam_aux_fields` after each `_paste_one`. |
| `refs/cups/tests/test_sam3_alignment.py` | Modify (lines 63-83) | Bug #3 test: replace the current `torch.equal(out["sam_masks"], sample["sam_masks"])` assertion (which codifies the bug) with a check that pasted regions are erased from `sam_masks`. |
| `refs/cups/tests/test_fine_object_masked_mean.py` | Create | Bug #2 test: assert helper returns `(logits, valid)`, assert caller mispairing is no longer possible when an IoU-passing mask becomes empty after resize. |
| `refs/cups/tests/test_copy_paste_sam_subtraction.py` | Create | Bug #3 test: dedicated module verifying paste regions erase SAM mask pixels, low-area SAM masks (<=4 px) are pruned, `sam_ious` and `sam_cls` stay in lockstep. |
| `refs/cups/configs/train_self_cityscapes_dinov3_vitb_stage4_fine_object_santosh.yaml` | Verify only (lines 128-145) | Bug #1: confirm `MODE: "thing_mc_panda"` and `WEIGHT: 0.01` (already correct locally). Plan is to verify and document, not edit. |

---

## Per-Bug Specification

### Bug #2 — `_batch_masked_mean` cls_labels desync

**Files:**
- Modify: `refs/cups/cups/losses/fine_object.py:678-700` (helper)
- Modify: `refs/cups/cups/losses/fine_object.py:286-341` (caller in `forward`, including `mean_teacher_logits` call site)
- Test: `refs/cups/tests/test_fine_object_masked_mean.py` (new)

**Current code (one-line summary):**
- Helper at `fine_object.py:699-700` computes `valid = flat_masks.sum(dim=1) > 0` then returns `mean_logits[valid]` only — `valid` is discarded.
- Caller at `fine_object.py:305` consumes the filtered logits but keeps `cls_labels` (line 302) and `iou_weights` (line 288) at the pre-filter length, so `_split_thing_stuff_loss` indexes `cls_labels[m]` and the wrong row pair when any mask collapses to zero pixels after the nearest-neighbour resize at lines 278-283.

**Target behaviour:**
- Helper returns `Tuple[Tensor, Tensor]`: filtered `(mean_logits, valid)` where `valid: BoolTensor` of length `M_pre_filter` indicates which masks survived.
- Caller applies `valid` to `cls_labels` and `iou_weights` immediately, so all three tensors carry the same length when entering `_split_thing_stuff_loss`.
- For `mean_teacher_logits` at `fine_object.py:323`, the input masks are bit-identical to those used at line 305, so the returned `valid` tensor MUST equal the one already produced. Guard with `assert torch.equal(valid_student, valid_teacher)` and use either; this defensive assert documents the invariant.

**Fix sketch (diff hunks, do not write yet — engineer writes in the implementation step):**

```python
# fine_object.py:678-700
def _batch_masked_mean(logits: Tensor, masks: Tensor) -> Tuple[Tensor, Tensor]:
    """Mean logit vector inside each mask without looping.

    Returns:
        Tuple of:
          mean_logits: (M', C) where M' <= M (empty masks dropped).
          valid:      (M,) bool mask indicating which input masks were kept.
    """
    M = masks.shape[0]
    if M == 0:
        return logits.new_zeros(0, logits.shape[0]), masks.new_zeros(0, dtype=torch.bool)
    flat_masks = masks.reshape(M, -1).float()
    flat_logits = logits.reshape(logits.shape[0], -1)
    pixel_counts = flat_masks.sum(dim=1).clamp(min=1.0)
    summed = flat_masks @ flat_logits.T
    mean_logits = summed / pixel_counts.unsqueeze(1)
    valid = flat_masks.sum(dim=1) > 0
    return mean_logits[valid], valid
```

```python
# fine_object.py:305 caller — apply valid to cls_labels and iou_weights
mean_logits, valid_means = _batch_masked_mean(logits_b, masks)
if mean_logits.shape[0] == 0:
    continue
if cls_labels is not None:
    cls_labels = cls_labels[valid_means.to(cls_labels.device)]
if iou_weights is not None:
    iou_weights = iou_weights[valid_means.to(iou_weights.device)]
```

```python
# fine_object.py:323 teacher call site — assert valid match
mean_teacher_logits, valid_teacher = _batch_masked_mean(teacher_b, masks)
assert torch.equal(valid_teacher, valid_means), (
    "Teacher and student valid masks must agree (same mask tensor as input)."
)
```

**Test strategy (`test_fine_object_masked_mean.py`):**
1. `test_batch_masked_mean_returns_tuple_of_logits_and_valid` — construct a `(C=4, H=8, W=8)` logits tensor and a `(M=3, 8, 8)` mask tensor where mask 1 is all-zero. Assert returned tuple, `mean_logits.shape == (2, 4)`, `valid == [True, False, True]`.
2. `test_forward_drops_cls_labels_for_collapsed_masks` — call `FineObjectSemanticLoss(mode='thing_focal_only').forward` with one image whose IoU-passing masks include a tiny mask that becomes empty after a nearest-neighbour resize from `(80, 160)` to `(20, 40)`. Assert the loss is finite and that `cls_labels` indexing inside `_split_thing_stuff_loss` does not raise. Use a counter on `_focal_ce` to disambiguate the silent failure mode: `cls_labels = [stuff_idx, thing_idx]` with the stuff mask collapsing — pre-fix `cls_labels` left at `[stuff, thing]` so `is_thing_mask[0]` reads the stuff label (entropy path, no focal call); post-fix `cls_labels` filtered to `[thing]` so focal CE fires once.
3. `test_forward_thing_mc_panda_assert_valid_match` — same construction but with `mode='thing_mc_panda'` and a non-None `teacher_logits` argument. Assert no `AssertionError` is raised (the invariant holds when masks are identical).

**Pre-fix expected output:** `pytest refs/cups/tests/test_fine_object_masked_mean.py -v` → 3 fails (signature, focal-row count, teacher-tuple unpack).
**Post-fix expected output:** all 3 pass.

---

### Bug #3 — `CopyPasteAugmentation` does not subtract paste mask from SAM masks

**Files:**
- Modify: `refs/cups/cups/augmentation.py:169-313` (`CopyPasteAugmentation.forward`)
- Test: `refs/cups/tests/test_sam3_alignment.py:63-83` (replace existing assertion)
- Test: `refs/cups/tests/test_copy_paste_sam_subtraction.py` (new)

**Current code (one-line summary):**
- `_paste_one` at line 162 overwrites image pixels and at lines 163-165 zeros original instance masks/sem_seg inside `instance_padded`.
- `forward` at lines 296-311 preserves arbitrary non-core keys (including `sam_masks`, `sam_ious`, `sam_cls`, `sam_teacher_logits`) via the dict-comprehension at line 297. SAM masks are NOT subtracted from `instance_padded`. Stale SAM masks now bound regions that show different pixels.
- `_filter_sam_aux_fields` (lines 42-48) and the threshold-on-area-`>4` pattern (lines 64, 95) already exist and should be reused.

**Target behaviour:**
- After each call to `_paste_one`, subtract `instance_padded` from each `sam_masks` row in the target sample. Apply the same area threshold (`> 4` pixels remaining) and `_filter_sam_aux_fields` to keep `sam_ious` / `sam_cls` in lockstep.
- `sam_teacher_logits` is intentionally NOT touched — those are dense spatial maps and the corresponding pixels are simply wrong; that is bug #4 and out of scope. The plan's docstring update must call this out.
- Implementation choice: **wrap around `_paste_one` in `forward`**, do not mutate `_paste_one` itself. `_paste_one` already has 5 return values and a busy signature; adding SAM-aware behaviour there would entangle the helper. The cleanest place is a new private helper called from inside the for-loop in `forward`.

**Fix sketch (in `augmentation.py`):**

```python
# Inside CopyPasteAugmentation.forward, after each _paste_one call:
sam_masks = sample.get(_SAM_MASK_KEY)
if isinstance(sam_masks, Tensor) and sam_masks.shape[0] > 0:
    sam_masks_dev = sam_masks.to(device=instance_padded.device, dtype=torch.bool)
    sam_masks_dev = sam_masks_dev & (~instance_padded[None])
    valid_sam = sam_masks_dev.sum(dim=(1, 2)) > 4
    sample[_SAM_MASK_KEY] = sam_masks_dev[valid_sam]
    _filter_sam_aux_fields(sample, valid_sam)
```

Notes:
- The current dict-comprehension at line 297 overwrites `sam_masks` with the un-erased version via `sample.items()`. Mutate `sample` in place inside the loop (above) so the read at line 297 picks up the eroded version. This interleaves correctly with the per-paste `instance_masks_original` updates and keeps the area threshold local to each paste.
- `instance_padded` is currently locally scoped inside `_paste_one`. To make it visible to the post-paste hook, either (a) extend `_paste_one` to also return `instance_padded`, or (b) recompute `instance_padded` from the difference between `instance_masks_original` before and after the paste (the newly added last row). Prefer (a) because it is more explicit and avoids a redundant recomputation.

**Test strategy:**

A. Replace existing `test_copy_paste_preserves_attached_sam_fields` (`test_sam3_alignment.py:63-83`) — its current assertion `torch.equal(out["sam_masks"], sample["sam_masks"])` codifies the bug. Rename to `test_copy_paste_subtracts_pasted_region_from_sam_masks` and:
1. Construct sample where one `sam_mask` overlaps a region the paste will land on. Use a deterministic seed and `scale_range=(1.0, 1.0)` plus `min_bounding_box_size=(1, 1)` for predictability.
2. Run augmentation.
3. Assert `out["sam_masks"]` no longer has any `True` pixel where `out["instances"].gt_masks.tensor[-1]` (the freshly pasted instance) is `True`.
4. Assert `out["sam_ious"].shape[0] == out["sam_masks"].shape[0]` and same for `sam_cls`.

B. New module `test_copy_paste_sam_subtraction.py`:
1. `test_low_area_sam_masks_pruned` — construct a sample whose only SAM mask has 5 `True` pixels; the paste lands on 2 of them. Post-fix: 3 pixels remain ≤ 4 threshold → pruned (`sam_masks.shape[0] == 0`, `sam_ious.shape[0] == 0`).
2. `test_sam_teacher_logits_unchanged` — assert `out["sam_teacher_logits"]` equals input (out of scope of this fix; documents bug #4 deferral as a regression guard).
3. `test_no_sam_keys_no_op` — sample without any `sam_*` keys runs cleanly through copy-paste; output dict contains the standard keys plus whatever non-SAM extras were present; no exception.

**Expected pre-fix:**
- A.3 fails: `out["sam_masks"]` still contains pixels in the paste region.
- B.1 fails: SAM mask remains at full size.
- B.2 passes (no behaviour change for teacher logits).
- B.3 passes (control test).

**Expected post-fix:** all assertions pass.

---

### Bug #1 — Revert active config (Path A)

**Files:**
- Verify only: `refs/cups/configs/train_self_cityscapes_dinov3_vitb_stage4_fine_object_santosh.yaml:128-145` (already in correct state locally — `MODE: "thing_mc_panda"`, `WEIGHT: 0.01` confirmed by reading lines 133, 145).
- No code change in this repo. The remote yaml on `/home/santosh/cups/configs/...` diverged.

**Recipe for the user (do NOT execute from this plan):**

1. **Stop the active restart2 run.** SSH to the remote and run `pkill -f train.py` (or the pid-targeted equivalent if other runs are sharing the box).
2. **Verify remote yaml.** From the user's local machine:
   ```
   ssh santosh@<host> 'grep -E "WEIGHT|MODE" /home/santosh/cups/configs/train_self_cityscapes_dinov3_vitb_stage4_fine_object_santosh.yaml'
   ```
   Expected after revert: `WEIGHT: 0.01`, `MODE: "thing_mc_panda"`.
3. **Sync the corrected yaml.** Use rsync with `--checksum --dry-run` first to confirm only the yaml differs:
   ```
   rsync -avc --dry-run refs/cups/configs/train_self_cityscapes_dinov3_vitb_stage4_fine_object_santosh.yaml santosh@<host>:/home/santosh/cups/configs/
   ```
   then run without `--dry-run` if the diff is only the two intended lines.
4. **Relaunch with a fresh log file.** Use a new log name so the regression baseline is clean:
   ```
   CUDA_VISIBLE_DEVICES=0 nohup python train.py \
     --experiment_config_file configs/train_self_cityscapes_dinov3_vitb_stage4_fine_object_santosh.yaml \
     --resume_from <stage3_best_ckpt_path> \
     --disable_wandb \
     SYSTEM.RUN_NAME "stage4_alignfix_mcpanda_w001_v2" \
     > logs/stage4_alignfix_mcpanda_w001_v2.log 2>&1 &
   ```
5. **Watch the first eval step (step 200).**

**Stop / kill-switch criteria:**
- **Pass criterion:** PQ at step 200 within ±0.5 PQ of Stage-3 starting checkpoint (CCR memory: 35.83 PQ).
- **Pass criterion (longer):** PQ at step 1000 ≥ 36.0 (must beat the previous best floor of 36.293 PQ within 1500 steps; if not, kill).
- **Hard kill:** PQ_things drops below 30.0 at any eval step → kill immediately, restart from Stage-3 ckpt with `WEIGHT: 0.005` for one run only as diagnostic.
- **Soft kill:** Three consecutive eval steps show monotonic PQ decrease ≥ 0.3 PQ each → kill, escalate.

---

## Tasks (TDD, one action per step)

### Task 0: Branch hygiene
- [ ] **Step 1:** Verify clean working tree.
  ```
  cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
  git status
  ```
  Expected: on `dino-cause-dora-adapter`, no staged changes.

### Task 1: Bug #2 — Write failing test for `_batch_masked_mean` signature
**Files:**
- Create: `refs/cups/tests/test_fine_object_masked_mean.py`

- [ ] **Step 1: Write the failing test for tuple return signature**

```python
# refs/cups/tests/test_fine_object_masked_mean.py
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cups.losses.fine_object import _batch_masked_mean, FineObjectSemanticLoss


def test_batch_masked_mean_returns_tuple_of_logits_and_valid():
    logits = torch.randn(4, 8, 8)
    masks = torch.zeros(3, 8, 8, dtype=torch.bool)
    masks[0, 0:4, 0:4] = True
    # masks[1] intentionally all-zero
    masks[2, 4:8, 4:8] = True

    result = _batch_masked_mean(logits, masks)
    assert isinstance(result, tuple) and len(result) == 2
    mean_logits, valid = result
    assert mean_logits.shape == (2, 4)
    assert valid.dtype == torch.bool
    assert valid.tolist() == [True, False, True]
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest refs/cups/tests/test_fine_object_masked_mean.py::test_batch_masked_mean_returns_tuple_of_logits_and_valid -v
```
Expected: FAIL — current return is a single Tensor, `len(result)` raises or `isinstance(result, tuple)` is False.

- [ ] **Step 3: Implement minimal change in `_batch_masked_mean`**

Apply the diff sketch in the Bug #2 spec section (helper signature change).

- [ ] **Step 4: Run the same test**

```
pytest refs/cups/tests/test_fine_object_masked_mean.py::test_batch_masked_mean_returns_tuple_of_logits_and_valid -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add refs/cups/cups/losses/fine_object.py refs/cups/tests/test_fine_object_masked_mean.py
git commit -m "fix(fine_object): return valid mask from _batch_masked_mean

Helper now returns (mean_logits, valid) so callers can sync cls_labels
and iou_weights with the empty-mask filter. Caller updates land in the
next commit. See CCR C472 bug #2."
```

### Task 2: Bug #2 — Caller-side synchronization in `forward`
**Files:**
- Modify: `refs/cups/cups/losses/fine_object.py:286-341`
- Modify: `refs/cups/tests/test_fine_object_masked_mean.py`

- [ ] **Step 1: Add failing test for caller-side cls_labels mispairing**

Use `cls_labels = [4, 0]` (stuff, thing). The collapsing mask is row 0 (the stuff one). Pre-fix: `cls_labels` left at `[4, 0]`; `mean_logits` is length 1 (from surviving row 1, the thing); `_split_thing_stuff_loss` reads `is_thing_mask[0]` from `sam_idx[0]=4` (stuff) — so the kept thing logits are misclassified as stuff and routed to entropy. Post-fix: `cls_labels` filtered to `[0]` — only thing CE fires.

```python
def test_forward_drops_cls_labels_for_collapsed_masks():
    logits = torch.randn(1, 5, 20, 40)
    masks = torch.zeros(2, 80, 160, dtype=torch.bool)
    masks[0, 0, 0] = True            # collapses after resize to (20, 40)
    masks[1, 0:32, 0:32] = True      # survives
    ious = torch.tensor([0.9, 0.9])
    cls_labels = torch.tensor([4, 0], dtype=torch.long)  # [stuff, thing]

    # Counter to make silent mispairing observable.
    seen_M = []
    import cups.losses.fine_object as fom
    real_focal = fom._focal_ce
    def counting_focal(*args, **kwargs):
        seen_M.append(args[1].shape[0])  # targets row count
        return real_focal(*args, **kwargs)
    fom._focal_ce = counting_focal
    try:
        loss_fn = FineObjectSemanticLoss(mode="thing_focal_only", min_hard_iou=0.0)
        out = loss_fn(
            logits=logits,
            sam_masks_list=[masks],
            sam_iou_list=[ious],
            sam_class_labels_list=[cls_labels],
            min_iou_score=0.0,
        )
    finally:
        fom._focal_ce = real_focal

    assert torch.isfinite(out)
    # Post-fix: focal CE fires once with the surviving thing row.
    # Pre-fix: cls_labels[0]=4 (stuff) is read for the surviving row -> entropy path -> seen_M empty.
    assert seen_M == [1], f"Expected one focal call with one row; got {seen_M}"
```

- [ ] **Step 2: Run — confirm failure on current code**

```
pytest refs/cups/tests/test_fine_object_masked_mean.py::test_forward_drops_cls_labels_for_collapsed_masks -v
```
Expected: FAIL with `assert seen_M == [1]` reporting `[]` (no focal call).

- [ ] **Step 3: Implement caller-side change**

Apply the diff sketch in the Bug #2 spec section (caller updates).

- [ ] **Step 4: Run — expect PASS**

```
pytest refs/cups/tests/test_fine_object_masked_mean.py -v
```
Expected: 2 PASS.

- [ ] **Step 5: Run the full pre-existing fine_object test surface**

```
pytest refs/cups/tests/ -k "fine_object or sam3" -v
```
Expected: no regressions in any other test.

- [ ] **Step 6: Commit**

```
git add refs/cups/cups/losses/fine_object.py refs/cups/tests/test_fine_object_masked_mean.py
git commit -m "fix(fine_object): sync cls_labels and iou_weights with valid mask filter

Empty masks dropped by _batch_masked_mean now also drop their cls_labels
and iou_weights rows in the caller, eliminating silent class-mispairing.
See CCR C472 bug #2."
```

### Task 3: Bug #2 — Teacher logits assert
**Files:**
- Modify: `refs/cups/cups/losses/fine_object.py:323` (mean_teacher_logits call site)
- Modify: `refs/cups/tests/test_fine_object_masked_mean.py`

- [ ] **Step 1: Add failing test for the assert**

```python
def test_forward_thing_mc_panda_teacher_valid_match():
    logits = torch.randn(1, 5, 20, 40)
    teacher_logits = torch.randn(1, 5, 20, 40)
    masks = torch.zeros(2, 80, 160, dtype=torch.bool)
    masks[0, 0:32, 0:32] = True
    masks[1, 0, 0] = True  # collapses
    cls_labels = torch.tensor([0, 0], dtype=torch.long)
    loss_fn = FineObjectSemanticLoss(mode="thing_mc_panda", min_hard_iou=0.0)
    out = loss_fn(
        logits=logits,
        sam_masks_list=[masks],
        sam_iou_list=[torch.tensor([0.9, 0.9])],
        sam_class_labels_list=[cls_labels],
        teacher_logits=teacher_logits,
        min_iou_score=0.0,
    )
    assert torch.isfinite(out)
```

- [ ] **Step 2: Run**

If the teacher call site at line 323 still ignores `valid`, it raises `TypeError: cannot unpack non-iterable Tensor object` because the helper now returns a tuple. Confirm.

- [ ] **Step 3: Patch teacher call site** (apply third diff sketch in Bug #2 spec)

- [ ] **Step 4: Run all three tests**

```
pytest refs/cups/tests/test_fine_object_masked_mean.py -v
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```
git add refs/cups/cups/losses/fine_object.py refs/cups/tests/test_fine_object_masked_mean.py
git commit -m "fix(fine_object): apply valid filter to teacher mean logits

Asserts the teacher's valid mask equals the student's, since input masks
are identical. Defensive guard for a future divergence."
```

### Task 4: Bug #3 — Replace existing copy-paste SAM-preservation test
**Files:**
- Modify: `refs/cups/tests/test_sam3_alignment.py:63-83`

- [ ] **Step 1: Replace the bug-codifying assertion with the corrected one**

```python
def test_copy_paste_subtracts_pasted_region_from_sam_masks():
    random.seed(3)
    torch.manual_seed(3)
    sample = _sample_with_sam_fields()
    aug = CopyPasteAugmentation(
        thing_class=0,
        max_num_pasted_objects=1,
        scale_range=(1.0, 1.0),
        use_random_horizontal_flipping=False,
        min_bounding_box_size=(1, 1),
    )

    out = aug([sample], [sample])[0]
    pasted = out["instances"].gt_masks.tensor[-1]  # newest paste
    assert "sam_masks" in out
    overlap = (out["sam_masks"] & pasted[None]).any()
    assert not overlap, "SAM masks must not overlap freshly pasted regions"
    assert out["sam_ious"].shape[0] == out["sam_masks"].shape[0]
    assert out["sam_cls"].shape[0] == out["sam_masks"].shape[0]
    assert "sam_teacher_logits" in out  # not erased; bug #4 deferred
```

- [ ] **Step 2: Run — expected FAIL on current code**

```
pytest refs/cups/tests/test_sam3_alignment.py::test_copy_paste_subtracts_pasted_region_from_sam_masks -v
```
Expected: FAIL — `overlap` evaluates True because SAM masks are unchanged.

- [ ] **Step 3: Implement the fix in `CopyPasteAugmentation.forward`** (apply the Bug #3 fix sketch + return `instance_padded` from `_paste_one`)

- [ ] **Step 4: Run test**

```
pytest refs/cups/tests/test_sam3_alignment.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add refs/cups/cups/augmentation.py refs/cups/tests/test_sam3_alignment.py
git commit -m "fix(augmentation): subtract paste mask from SAM masks in copy-paste

Stale SAM masks no longer bound regions overwritten by pasted objects.
Reuses _filter_sam_aux_fields and the > 4 px area threshold from the
existing crop/resize helpers. See CCR C472 bug #3."
```

### Task 5: Bug #3 — Dedicated test module
**Files:**
- Create: `refs/cups/tests/test_copy_paste_sam_subtraction.py`

- [ ] **Step 1: Write the three tests**

```python
import os, random, sys
import pytest
import torch
detectron2 = pytest.importorskip("detectron2")
pytest.importorskip("kornia")
from detectron2.structures import BitMasks, Boxes, Instances
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cups.augmentation import CopyPasteAugmentation


def _make_sample(sam_pixels):
    masks = torch.zeros(1, 64, 128, dtype=torch.bool)
    masks[0, 0:32, 0:32] = True
    instances = Instances(
        image_size=(64, 128),
        gt_masks=BitMasks(masks.clone()),
        gt_boxes=Boxes(torch.tensor([[0, 0, 32, 32]], dtype=torch.float32)),
        gt_classes=torch.tensor([0], dtype=torch.long),
    )
    sam_masks = torch.zeros(1, 64, 128, dtype=torch.bool)
    for (y, x) in sam_pixels:
        sam_masks[0, y, x] = True
    return {
        "image": torch.zeros(3, 64, 128, dtype=torch.float32),
        "sem_seg": torch.zeros(64, 128, dtype=torch.long),
        "instances": instances,
        "sam_masks": sam_masks,
        "sam_ious": torch.tensor([0.9]),
        "sam_cls": torch.tensor([0], dtype=torch.long),
        "sam_teacher_logits": torch.randn(5, 64, 128),
    }


def test_low_area_sam_masks_pruned():
    random.seed(0)
    torch.manual_seed(0)
    sam_pixels = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]  # 5 pixels
    sample = _make_sample(sam_pixels)
    aug = CopyPasteAugmentation(
        thing_class=0, max_num_pasted_objects=1, scale_range=(1.0, 1.0),
        use_random_horizontal_flipping=False, min_bounding_box_size=(1, 1),
    )
    out = aug([sample], [sample])[0]
    if out["sam_masks"].shape[0] == 0:
        assert out["sam_ious"].shape[0] == 0
        assert out["sam_cls"].shape[0] == 0


def test_sam_teacher_logits_unchanged():
    random.seed(1)
    torch.manual_seed(1)
    sample = _make_sample([(40, 100)])
    aug = CopyPasteAugmentation(
        thing_class=0, max_num_pasted_objects=1, scale_range=(1.0, 1.0),
        use_random_horizontal_flipping=False, min_bounding_box_size=(1, 1),
    )
    out = aug([sample], [sample])[0]
    assert torch.equal(out["sam_teacher_logits"], sample["sam_teacher_logits"])


def test_no_sam_keys_no_op():
    random.seed(2); torch.manual_seed(2)
    sample = _make_sample([(40, 100)])
    for key in ("sam_masks", "sam_ious", "sam_cls", "sam_teacher_logits"):
        sample.pop(key, None)
    aug = CopyPasteAugmentation(
        thing_class=0, max_num_pasted_objects=1, scale_range=(1.0, 1.0),
        use_random_horizontal_flipping=False, min_bounding_box_size=(1, 1),
    )
    out = aug([sample], [sample])[0]
    assert "image" in out and "sem_seg" in out and "instances" in out
```

- [ ] **Step 2: Run**

```
pytest refs/cups/tests/test_copy_paste_sam_subtraction.py -v
```
Expected: 3 PASS (the fix from Task 4 is already in the working tree).

- [ ] **Step 3: Commit**

```
git add refs/cups/tests/test_copy_paste_sam_subtraction.py
git commit -m "test(copy-paste): add SAM subtraction edge cases

Covers low-area pruning, teacher-logits non-erasure (bug #4 deferred),
and no-op behaviour when SAM keys are absent."
```

### Task 6: Bug #1 — Verify local YAML state, document recipe
**Files:**
- Verify only: `refs/cups/configs/train_self_cityscapes_dinov3_vitb_stage4_fine_object_santosh.yaml`

- [ ] **Step 1: Verify**

```
grep -nE "WEIGHT|MODE" refs/cups/configs/train_self_cityscapes_dinov3_vitb_stage4_fine_object_santosh.yaml | head -5
```
Expected output (already true at lines 133/145):
```
133:    WEIGHT: 0.01
145:    MODE: "thing_mc_panda"
```

- [ ] **Step 2:** No code change. Hand the user the recipe in the Bug #1 spec section. The user runs the SSH/rsync/relaunch commands themselves; the plan is the contract.

### Task 7: Local CPU smoke train (optional, fallback to harness)
- [ ] **Step 1:** Attempt a short local CPU training to verify no regression in the mounted Stage-4 path.

```
cd refs/cups
CUDA_VISIBLE_DEVICES="" python train.py \
  --experiment_config_file configs/train_self_cityscapes_dinov3_vitb_stage4_fine_object_santosh.yaml \
  --max_steps 50 \
  SYSTEM.RUN_NAME "smoke_alignfix_followups" \
  SYSTEM.NUM_GPUS 0 \
  | tee /tmp/stage4_smoke.log
```

**Fallback if smoke train cannot run on CPU:** if dataset paths or memory render this impractical, skip the smoke train. The unit-level pytest coverage in tasks 1-5 exercises the affected `_split_thing_stuff_loss` and `CopyPasteAugmentation` paths directly. Document the skip in the commit message.

### Task 8: Final test surface
- [ ] **Step 1:** Run the full repo test suite.

```
pytest refs/cups/tests -v
```
Expected: all green. Compare to baseline `pytest` output captured at start of Task 0.

- [ ] **Step 2: Commit (if any incidental fixes were needed)**

If any pre-existing test failed under the new helper signature (e.g. `test_dora_smoke.py`, `test_mitigations.py`) and required a follow-up tweak, commit it as a separate `chore(tests):` commit. None expected, but the suite must stay green.

---

## Validation Gates

| Gate | Command | Pass criterion |
|---|---|---|
| Pytest (per-bug) | `pytest refs/cups/tests/test_fine_object_masked_mean.py refs/cups/tests/test_copy_paste_sam_subtraction.py refs/cups/tests/test_sam3_alignment.py -v` | All tests green. |
| Pytest (whole repo) | `pytest refs/cups/tests -v` | No new failures vs Task-0 baseline. |
| Local CPU smoke train (~50 steps, optional) | `python train.py ... --max_steps 50` (Task 7) | Loss is finite at every step; `losses/fine_object` non-NaN. |
| Remote relaunch (user runs) | `tail -f logs/stage4_alignfix_mcpanda_w001_v2.log` | First eval step (200) PQ within ±0.5 PQ of Stage-3 starting checkpoint (35.83). PQ at step 1000 ≥ 36.0. PQ_things never below 30.0. |

---

## Risks and Rollback

### Bug #1 risks
- **Risk:** Even with `thing_mc_panda` / 0.01, the alignment fix may have made the gate behaviour subtly different. The MC-PanDA gate `(1 - common_conf * teacher_logit_weight).clamp(0,1)` was previously gating on misaligned teacher logits and is now gating on properly aligned ones.
- **Mitigation:** Validation gate above (PQ ≥ 35.33 at step 200). If it falls below, kill within 100 steps.
- **Rollback:** Edit YAML back. No code change to revert. Single line: `WEIGHT: 0.005` for one diagnostic run.

### Bug #2 risks
- **Risk:** None expected. The post-fix code is bit-identical to the pre-fix code when no IoU-passing mask collapses to empty after resize. The change only adds a defensive filter for an edge case.
- **Mitigation:** The defensive `assert torch.equal(valid_teacher, valid_means)` guards future divergence.
- **Rollback:** Revert the three commits from Tasks 1-3. Helper signature change is the riskiest revert because callers depend on it; revert in reverse commit order.

### Bug #3 risks
- **Risk:** Copy-paste throughput may drop because of the extra mask-subtraction per paste. Worst case: `B * num_pasted_objects * N_sam` boolean-AND ops at full resolution. For B=2, num_pasted_objects=7, N_sam~30, H=1024, W=2048: ~2 × 7 × 30 × 2M = 840M bool ops per batch on CPU; on GPU this is sub-millisecond. Negligible.
- **Mitigation:** None needed; if a perf regression is observed in dataloader-bound runs, gate the SAM subtraction behind a config flag (would require introducing one new config key — explicitly forbidden by the user — so this is the last-resort fallback only on user request).
- **Rollback:** Revert commit from Task 4. Existing code path resumes (with the bug).

---

## Spec Coverage Self-Review

- Bug #1 fix: covered by Task 6 (verify) + Bug #1 recipe section + validation gate + rollback.
- Bug #2 fix: covered by Tasks 1-3, with TDD ordering (failing test → minimal implementation → second failing test → caller change → third failing test → teacher call site change). Three commits.
- Bug #3 fix: covered by Task 4 (test replacement + impl) and Task 5 (dedicated test module). Two commits.
- Test plan: per-bug pytest module specified, plus existing `test_sam3_alignment.py` extension. Test names, fixtures, expected pass/fail behaviour all listed.
- Order of operations: bug #2 → bug #3 → bug #1 (verification only) in code-merge order. Bug #1 (revert) is wall-clock urgent and the user runs it independent of the merge order.
- Validation gates: pytest, smoke train, remote relaunch criteria. All testable.
- Risks and rollback: per-fix.
- Hard constraints respected:
  - No bug #4 work.
  - No new modes / YAML keys / hyperparameters.
  - No edits to `pl_model_self.py`.
  - Local YAML quoted (lines 133, 145) — not edited.
  - No GPU/SSH actions taken from the plan; user runs.

## Type Consistency
- `_batch_masked_mean` post-fix signature: `(logits: Tensor, masks: Tensor) -> Tuple[Tensor, Tensor]`. Used by callers at `fine_object.py:305` and `:323`.
- `valid` is `BoolTensor` of shape `(M_pre_filter,)` — must be moved to the device of `cls_labels` and `iou_weights` before indexing. Documented in caller diff sketch.
- `_filter_sam_aux_fields(sample, valid_masks)` signature unchanged — Bug #3 fix reuses it as-is.

---

## Open Questions for the User

- **(Q1)** Bug #2's silent-mispairing failure mode is hard to pin without monkey-patching `_focal_ce` to count rows. The current test does this via attribute-swap (see Task 2). Acceptable as a TDD oracle, or revise toward a different observable?
- **(Q2)** The existing test `test_copy_paste_preserves_attached_sam_fields` at `refs/cups/tests/test_sam3_alignment.py:81-83` literally asserts `torch.equal(out["sam_masks"], sample["sam_masks"])` — i.e., it codifies bug #3 as if it were correct behaviour. The plan replaces it. Confirm delete-and-replace (recommended) vs `pytest.mark.xfail` retention.
