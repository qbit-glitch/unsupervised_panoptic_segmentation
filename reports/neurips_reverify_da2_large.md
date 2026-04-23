# NeurIPS Re-Verification: Depth Anything V2 Large DoRA Adapter Implementation

**Previous Review:** reports/neurips_review_da2_large.md (Rated: 3 — Reject)  
**Reviewer:** External - Depth Estimation & Self-Supervised Adaptation  
**Re-Verification Date:** 2026-04-24

---

## Executive Summary

The authors have addressed **12 of 29** identified issues. The most critical show-stoppers—broken ranking loss, missing augmentation, inverted `late_block_start` defaults, and `copy.deepcopy` GPU memory bomb—are now fixed. However, **several critical and major issues remain unresolved**, and the fix for `late_block_start` introduced a **new critical inference-time architecture mismatch** that would silently corrupt pseudo-label quality. The codebase has improved from "untrainable draft" to "trainable but still hazardous," but it is not yet research-grade.

**Bottom line:** The rating improves from **3 (Reject)** to approximately **4 (Weak Reject / Major Revision)**. The core training loop is now conceptually correct, but inference pipelines, architectural edge cases, and engineering hygiene still require significant work before training should commence.

---

## Issue-by-Issue Verification

### Issue 1: `late_block_start=6` catastrophically wrong for DA2-Large
**Previous Severity:** CRITICAL  
**Status:** ⚠️ PARTIALLY FIXED

**Evidence:**
- `train_depth_adapter_lora.py:323` still declares `default=6` in argparse.
- `train_depth_adapter_lora.py:335-341` adds auto-adjustment:
  ```python
  if args.late_block_start == 6:  # user didn't override
      if args.model_type in ("da2", "depthpro"):
          args.late_block_start = 18
          logger.info("Auto-set late_block_start=%d for %s (24-block model)", ...)
      elif args.model_type == "dav3":
          args.late_block_start = 18
  ```
- `configs/depth_adapter_baseline.yaml:15` now correctly sets `late_block_start: 18`.

**Remaining Concerns:**
1. The argparse default is still `6`. If a user passes `--late_block_start 12` for a 24-block model, no auto-adjustment triggers, and they silently get inverted tiering.
2. The auto-adjustment sets **DAv3 to 18 as well**, but the previous review explicitly recommended `DA3: 6, DepthPro: 6`. The test `test_da3_adapter_injection` uses `late_block_start=6` and expects 12-block-style tiering (blocks 0–5 early, 6–23 late). Forcing 18 for DAv3 means only blocks 0–17 are "early" (Q+V only) and 18–23 are "late." If the real DAv3 has 12 blocks, **all blocks become early** and MLP layers are never adapted.
3. The safety-net test `test_late_block_start_defaults` is vacuous—it asserts `18 == 18` and `6 == 6` without exercising the actual CLI logic.

---

### Issue 2: CAUSE-style vs HF-style precedence (greedy first-match)
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

**Evidence:**
- `depth_adapter.py:109-134` still appends CAUSE-style `target_groups` first, then HF-style.
- `depth_adapter.py:161-163` still breaks after the first successful group:
  ```python
  if group_found > 0 and attn_found:
      found_any = True
      break  # Stop after first successful group
  ```
- If a future model variant exposes both `attn.qkv` (fused) and `attention.attention.query` (unfused), CAUSE-style takes precedence and silently skips the HF path. The logic remains architecture-agnostic and greedy.

**Remaining Concerns:** The reviewer explicitly asked for "architecture-aware, not greedy-first-match" selection. No such change was made.

---

### Issue 3: `_m_init_norm` dead code in `DoRALinear`
**Previous Severity:** MINOR  
**Status:** ✅ FIXED

**Evidence:**
- The `_m_init_norm` buffer has been completely removed from `DoRALinear`.
- `lora_layers.py:98-100` now registers only `lora_magnitude` (which is actively used in `forward()`).

---

### Issue 4: `ConvDoRALinear._spatial_dims` never set
**Previous Severity:** CRITICAL  
**Status:** ❌ NOT FIXED

**Evidence:**
- `lora_layers.py:141` declares `self._spatial_dims: Optional[Tuple[int, int]] = None`.
- `lora_layers.py:153` gates the conv path on `h_p is not None and w_p is not None`.
- A helper `set_dinov2_spatial_dims` exists in `models/adapters/dinov2_adapter.py:129-136`, but it is **never imported or called** in `train_depth_adapter_lora.py` or `generate_instance_pseudolabels_adapted.py`.
- The conv path remains unreachable dead code. All claims about "Conv-DoRA activation-space refinement" are still unsubstantiated.

**Remaining Concerns:** Either implement `_spatial_dims` injection in the depth training/inference pipelines or remove the `conv_dora` option to prevent silent degradation to plain DoRA.

---

### Issue 5: `img_aug` computed but never used
**Previous Severity:** CRITICAL  
**Status:** ✅ FIXED

**Evidence:**
- `train_depth_adapter_lora.py:224-226` now retrieves `img_aug` from the batch and moves it to device:
  ```python
  img_aug = batch.get("img_aug")
  if img_aug is not None:
      img_aug = img_aug.to(device)
  ```
- `train_depth_adapter_lora.py:238` routes augmented input to the student:
  ```python
  student_input = img_aug if img_aug is not None else img
  ```
- The teacher still receives the clean `img`, preserving the self-distillation signal.

---

### Issue 6: `inputs` dict reused between teacher and student
**Previous Severity:** MAJOR  
**Status:** ✅ FIXED

**Evidence:**
- `train_depth_adapter_lora.py:233-235` creates `teacher_inputs` separately.
- `train_depth_adapter_lora.py:242-244` creates `student_inputs` separately.
- The two pipelines are now decoupled, enabling future extensions (different resolutions, processor configs).

---

### Issue 7: `copy.deepcopy()` on 300M-parameter GPU model
**Previous Severity:** MAJOR  
**Status:** ✅ FIXED

**Evidence:**
- `train_depth_adapter_lora.py:373-378` loads the teacher from scratch on **CPU**:
  ```python
  if args.model_type == "dav3":
      teacher_model = load_dav3_model(model_name, device="cpu")
  elif args.model_type == "da2":
      teacher_model = load_da2_model(model_name, device="cpu")
  elif args.model_type == "depthpro":
      teacher_model = load_depthpro_model(model_name, device="cpu")
  ```
- `copy.deepcopy` has been eliminated entirely. GPU memory is no longer doubled at initialization.

---

### Issue 8: `teacher_model.to(device)` redundancy
**Previous Severity:** MINOR  
**Status:** ✅ FIXED

**Evidence:**
- The teacher is now loaded on CPU and explicitly moved to device at `train_depth_adapter_lora.py:382`. This `.to(device)` call is now necessary, not redundant.

---

### Issue 9: Ranking loss is self-consistency, not teacher-guided
**Previous Severity:** CRITICAL  
**Status:** ✅ FIXED

**Evidence:**
- `train_depth_adapter_lora.py:102-134` now correctly uses **teacher depth** for targets:
  ```python
  s_i = student_flat[b, idx_i]
  s_j = student_flat[b, idx_j]
  t_i = teacher_flat[b, idx_i]
  t_j = teacher_flat[b, idx_j]
  target = torch.sign(t_i - t_j)
  ```
- Self-pairs are excluded via resampling loop (`lines 117-121`).
- Zero-target pairs (teacher depths equal) are skipped (`lines 129-133`).
- The docstring now accurately describes the behavior: "penalizes when the student's predicted depth order contradicts the teacher's order."

---

### Issue 10: Scale-invariant loss excluded by default
**Previous Severity:** CRITICAL  
**Status:** ✅ FIXED

**Evidence:**
- `train_depth_adapter_lora.py:325` now defaults to `"distillation,ranking,scale_invariant"`.
- `configs/depth_adapter_baseline.yaml:34` lists `names: ["distillation", "ranking", "scale_invariant"]`.
- The one loss the authors themselves labeled "essential" is now included out-of-the-box.

---

### Issue 11: Ranking loss samples only 1,024 pairs
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED (Noted)

**Evidence:**
- `train_depth_adapter_lora.py:102` still declares `num_pairs=1024`.
- For a 512×1024 image, this is still only **0.2%** of all possible pairs.
- While self-pairs are now excluded, the statistical power of the signal remains extremely low.

**Recommendation:** Increase to at least 8,192–16,384 pairs, or adopt a dense/random-sampled hybrid strategy. Document the rationale.

---

### Issue 12: No affine-invariance loss
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED (Noted)

**Evidence:**
- No Pearson correlation, normalized cross-correlation, or MiDaS-style affine-invariant loss has been added.
- The student is still forced to stay unnaturally close to the teacher's exact depth values.

**Recommendation:** Add `scale_invariant_loss` is good; adding an affine-invariant term (e.g., `1 - Pearson(student, teacher)`) would further relax the scale constraint and allow meaningful domain adaptation.

---

### Issue 13: Numerical stability in `scale_invariant_loss` (log near-zero)
**Previous Severity:** MINOR  
**Status:** ✅ FIXED

**Evidence:**
- `train_depth_adapter_lora.py:137-144` now clamps before log:
  ```python
  def scale_invariant_loss(pred, target, lambda_si=0.5, min_depth=1e-3):
      pred_clamped = torch.clamp(pred, min=min_depth)
      target_clamped = torch.clamp(target.detach(), min=min_depth)
      diff = torch.log(pred_clamped) - torch.log(target_clamped)
  ```
- `min_depth=1e-3` prevents `log(1e-6) = -13.8` explosions. The maximum negative log value is now `-6.9`, which is far more stable.

**Minor concern:** Clamping alters the depth map; `log(pred + 1e-3)` would preserve relative ordering of small values. This is acceptable but not optimal.

---

### Issue 14: 10 epochs insufficient for domain adaptation
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED (Noted)

**Evidence:**
- `train_depth_adapter_lora.py:317` still declares `default=10`.
- Cityscapes train (~2,975 images, batch_size=4) = ~7,440 total steps. This is still almost certainly too few for meaningful domain adaptation of a 300M model.

**Recommendation:** Default to at least 25–50 epochs with validation-based early stopping.

---

### Issue 15: No learning-rate warmup
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED (Noted)

**Evidence:**
- `train_depth_adapter_lora.py:208` still uses bare `CosineAnnealingLR` with `T_max=epochs`.
- Freshly initialized DoRA matrices (Kaiming A, zero B) can experience initial loss spikes without warmup.

**Recommendation:** Add 5–10% linear warmup before cosine decay.

---

### Issue 16: 0.28% trainable ratio may be too low
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED (Noted)

**Evidence:**
- `train_depth_adapter_lora.py:320` still defaults to `rank=4`.
- No ablation on rank, block coverage, or trainable ratio is provided.

**Recommendation:** Run ablations on r={4, 8, 16} and provide evidence that 0.28% is sufficient.

---

### Issue 17: `adapt_decoder=False` — is this truly optimal?
**Previous Severity:** QUESTION  
**Status:** ❌ NOT FIXED (Noted)

**Evidence:**
- `train_depth_adapter_lora.py:324` still defaults to `False` (via `action="store_true"`).
- No ablation showing `adapt_decoder=True` vs `False`.

**Recommendation:** Add a decoder-adaptation ablation, even if only on the final fusion layer.

---

### Issue 18: Gradient clipping at `max_norm=1.0`
**Previous Severity:** MINOR  
**Status:** ❌ NOT FIXED (Noted)

**Evidence:**
- `train_depth_adapter_lora.py:276,282` still clips at `1.0`.
- While harmless for adapter training, it may suppress legitimate large updates during early training.

---

### Issue 19: No validation set or depth metrics
**Previous Severity:** MINOR  
**Status:** ❌ NOT FIXED (Noted)

**Evidence:**
- The training loop still logs only total loss.
- No δ1 threshold accuracy, RMSE, edge-F1, or pseudo-label PQ is computed during training.
- "Best" checkpoint is still selected by training loss alone.

---

### Issue 20: Sobel threshold default (`grad_threshold=0.20`) contradicts spec
**Previous Severity:** CRITICAL  
**Status:** ✅ FIXED

**Evidence:**
- `instance_methods/sobel_cc.py:11` now defaults to `grad_threshold=0.03`.
- This aligns with the architecture spec's claimed "Cityscapes optimal" value.

**BUT:** See **New Issue 1** below—`generate_instance_pseudolabels_adapted.py` still defaults to `tau=0.20`, creating a dangerous caller-level discrepancy.

---

### Issue 21: `min_area=1000` too large for Cityscapes at 1024×512
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED (Noted)

**Evidence:**
- `instance_methods/sobel_cc.py:11` still defaults to `min_area=1000`.
- Distant pedestrians, cyclists, and traffic signs still fall below this threshold systematically.

**Recommendation:** Reduce to 100–200 pixels or implement a size-dependent threshold.

---

### Issue 22: Sobel on relative depth lacks per-image normalization
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED (Noted)

**Evidence:**
- `instance_methods/sobel_cc.py:24-27` still uses a fixed global threshold:
  ```python
  grad_mag = np.sqrt(gx ** 2 + gy ** 2)
  depth_edges = grad_mag > grad_threshold
  ```
- No division by depth range, no percentile-based adaptive threshold.

---

### Issue 23: `depth_blur_sigma=1.0` applied unconditionally
**Previous Severity:** MINOR  
**Status:** ❌ NOT FIXED (Noted)

**Evidence:**
- `instance_methods/sobel_cc.py:12` still defaults to `depth_blur_sigma=1.0`.
- No ablation or conditional logic based on image resolution.

---

### Issue 24: Hardcoded HF model ID with no graceful fallback
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED (Noted)

**Evidence:**
- `train_depth_adapter_lora.py:72` still hardcodes `"depth-anything/Depth-Anything-V2-Large-hf"`.
- No `os.path.exists(model_name)` check, no local path fallback, no `snapshot_download` pre-checking.
- Same issue exists in `generate_instance_pseudolabels_adapted.py`.

---

### Issue 25: No shape validation for `predicted_depth`
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED (Noted)

**Evidence:**
- `train_depth_adapter_lora.py:235` assumes `.predicted_depth` returns `(B, H, W)`:
  ```python
  teacher_out = teacher_model(**teacher_inputs).predicted_depth
  ```
- `relative_depth_ranking_loss()` immediately unpacks `B, H, W = student_depth.shape`.
- If HF returns `(B, 1, H, W)` in a future update, the loss crashes with a shape error.

---

### Issue 26: Inefficient tensor→PIL→tensor roundtrip in training loop
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED (Noted)

**Evidence:**
- `train_depth_adapter_lora.py:233,242` still perform lossy roundtrips:
  ```python
  teacher_inputs = processor(images=[Image.fromarray((i.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)) for i in img], return_tensors="pt")
  student_inputs = processor(images=[Image.fromarray((i.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)) for i in student_input], return_tensors="pt")
  ```
- Quantization to uint8, double resize, and 4 PIL creations per batch remain.
- The dataset already resizes to 512×1024; the processor then resizes to 518×518 for DINOv2 patch 14. Double bilinear resize degrades fine boundaries.

---

### Issue 27: No deterministic CUDA settings
**Previous Severity:** MINOR  
**Status:** ✅ FIXED

**Evidence:**
- `train_depth_adapter_lora.py:56-58` now sets:
  ```python
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  os.environ["PYTHONHASHSEED"] = str(seed)
  ```
- `test_depth_adapters.py::test_deterministic_cuda_settings` validates this.

---

### Issue 28: `freeze_non_adapter_params` uses brittle string matching
**Previous Severity:** MINOR  
**Status:** ✅ FIXED

**Evidence:**
- `lora_layers.py:303-306` now uses **suffix matching** instead of substring containment:
  ```python
  ADAPTER_SUFFIXES = (".lora_A", ".lora_B", ".lora_magnitude", ".dwconv.weight", ".conv_gate",
                      ".lora_A.weight", ".lora_B.weight")
  for name, param in model.named_parameters():
      if any(name.endswith(suffix) for suffix in ADAPTER_SUFFIXES):
          param.requires_grad = True
      else:
          param.requires_grad = False
  ```
- `test_depth_adapters.py::test_freeze_non_adapter_params_robust` confirms that a parameter named `something_lora_like` remains frozen while real adapter parameters are unfrozen.

---

### Issue 29: `ConvDoRA` parameter count inflated by unused `dwconv` weights
**Previous Severity:** MINOR  
**Status:** ❌ NOT FIXED (dependent on Issue 4)

**Evidence:**
- `lora_layers.py:173-176` still counts `dwconv.weight` and `conv_gate` in `trainable_count()`.
- Since `_spatial_dims` is never set (Issue 4), these weights are never used but still counted.
- The reported parameter count for `conv_dora` is therefore inflated relative to actual compute.

---

## New Issues Introduced by Fixes

### New Issue 1: CRITICAL — Inference-time architecture mismatch (`late_block_start`)
**Severity:** CRITICAL

**Evidence:**
- `train_depth_adapter_lora.py:335-341` auto-adjusts `late_block_start` to **18** for DA2 during training.
- `generate_instance_pseudolabels_adapted.py:92-94` calls `inject_lora_into_depth_model()` **without** passing `late_block_start`, so it falls back to the function default of **6**:
  ```python
  inject_lora_into_depth_model(model, variant="dora", rank=4, alpha=4.0)
  ```
- This means:
  - **Training:** blocks 0–17 = early (Q+V only); blocks 18–23 = late (Q+K+V+proj+fc1+fc2).
  - **Inference:** blocks 0–5 = early (Q+V only); blocks 6–23 = late (Q+K+V+proj+fc1+fc2).
- When the checkpoint is loaded with `strict=False` (`generate_instance_pseudolabels_adapted.py:97`), trained Q+V weights from blocks 6–17 load correctly, but **K, proj, fc1, fc2 in blocks 6–17 are randomly initialized** (Kaiming/zeros from adapter init). The model therefore produces **degraded pseudo-labels** compared to training.

**Fix required:** Pass `late_block_start` from `ckpt["adapter_config"]` during inference, or read the config before injection.

---

### New Issue 2: Inference script default `tau=0.20` contradicts fixed `sobel_cc_instances` default
**Severity:** MAJOR

**Evidence:**
- `instance_methods/sobel_cc.py:11` fixed the default to `grad_threshold=0.03`.
- `generate_instance_pseudolabels_adapted.py:68,178` still default to `tau=0.20`:
  ```python
  def generate_adapted_instances(..., tau=0.20, ...):
      ...
  parser.add_argument("--tau", type=float, default=0.20)
  ```
- A user running the generation script without `--tau` will use **0.20**, which is **6.7× higher** than the validated optimal value. This directly undermines the fix for Issue 20.

**Fix required:** Change the generation script default to `0.03` and read it from the checkpoint config if available.

---

### New Issue 3: Inference script ignores `adapter_config` from checkpoint
**Severity:** MAJOR

**Evidence:**
- The checkpoint stores `adapter_config` (variant, rank, alpha, late_block_start, adapt_decoder, model_type) at `train_depth_adapter_lora.py:416-424`.
- `generate_instance_pseudolabels_adapted.py:96-97` loads the checkpoint but **never reads** `ckpt["adapter_config"]`:
  ```python
  ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
  model.load_state_dict(ckpt["model"], strict=False)
  ```
- Hardcoded `variant="dora", rank=4, alpha=4.0` are used for injection. If the user trained with `rank=8` or `variant="lora"`, the architecture mismatch is silent and catastrophic.

**Fix required:** Read `ckpt["adapter_config"]` and use its fields to drive `inject_lora_into_depth_model()`.

---

### New Issue 4: `test_late_block_start_defaults` is vacuous
**Severity:** MINOR

**Evidence:**
- `tests/test_depth_adapters.py:478-484`:
  ```python
  def test_late_block_start_defaults():
      assert 18 == 18, "Default late_block_start for 24-block models should be 18"
      assert 6 == 6, "Default late_block_start for 12-block models should be 6"
  ```
- This test asserts tautologies. It does not exercise `main()`, argparse, or the auto-adjustment logic. It gives false confidence.

**Fix required:** Actually invoke the auto-adjustment logic (or refactor it to a testable function) and assert the resulting values for each `model_type`.

---

### New Issue 5: Dataset augmentation performs lossy tensor→PIL→tensor roundtrip
**Severity:** MINOR

**Evidence:**
- `train_depth_adapter_lora.py:178-186`:
  ```python
  img_tensor = self.transform(img)  # (3, H, W)
  img_aug = None
  if self.aug_transform:
      img_aug = T.ToTensor()(self.aug_transform(Image.fromarray(
          (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
      )))
  ```
- The image is loaded as PIL → transformed to tensor → converted back to uint8 numpy → wrapped as PIL → augmented → converted back to tensor. This is lossy (uint8 quantization) and computationally wasteful.
- The augmentation should be applied directly to the PIL image **before** `self.transform`, or torchvision tensor transforms should be used.

---

### New Issue 6: Auto-adjustment forces `late_block_start=18` for DAv3 and DepthPro
**Severity:** MAJOR

**Evidence:**
- `train_depth_adapter_lora.py:335-341`:
  ```python
  if args.model_type in ("da2", "depthpro"):
      args.late_block_start = 18
  elif args.model_type == "dav3":
      args.late_block_start = 18
  ```
- The previous review explicitly recommended: **DA2: 18, DA3: 6, DepthPro: 6**.
- The current code forces **all three** to 18. If DepthPro is a 12-block model (like many DINOv2-based depth models), setting `late_block_start=18` means **all blocks are treated as early** (Q+V only), and MLP layers are never adapted.
- If DAv3 is also 12-block, the same problem occurs.

**Fix required:** Use architecture-specific defaults verified against actual model configs: DA2-Large=18, DA3=6, DepthPro=6. Add assertions that `late_block_start < n_blocks`.

---

## Overall Recommendation

**Rating: 4 — Weak Reject / Major Revision Required**

The authors have made **genuine progress** on the most egregious bugs:
- ✅ Ranking loss now correctly distills from the teacher.
- ✅ Augmentation pipeline is now connected to the student.
- ✅ `late_block_start` defaults are auto-corrected at training time.
- ✅ Scale-invariant loss is now included by default.
- ✅ Teacher loading no longer doubles GPU memory via `deepcopy`.
- ✅ Deterministic CUDA and robust parameter freezing are now in place.

However, **three critical issues prevent this from being training-ready**:

1. **Inference-time architecture mismatch (New Issue 1):** The generation script uses `late_block_start=6` while training uses 18 for DA2. This means pseudo-label generation uses a **different adapter topology** than the one trained. With `strict=False` checkpoint loading, random weights are silently injected into blocks 6–17. This would produce pseudo-labels **worse than the zero-shot baseline**.

2. **ConvDoRA remains unreachable (Issue 4):** `_spatial_dims` is still never populated in the depth pipelines. The `conv_dora` option is a dead code path that inflates parameter counts without adding compute.

3. **Inference defaults still sabotage Sobel threshold (New Issue 2):** The generation script defaults to `tau=0.20`, contradicting the 0.03 fix in `sobel_cc_instances`.

Before any training run, the authors **must**:
1. Fix the inference script to read `adapter_config` from the checkpoint and match training architecture exactly.
2. Align `generate_instance_pseudolabels_adapted.py` defaults with `sobel_cc_instances` (τ=0.03).
3. Either implement `_spatial_dims` injection for ConvDoRA or remove the `conv_dora` CLI option.
4. Fix architecture-specific `late_block_start` defaults (DA2=18, DA3=6, DepthPro=6) with assertions against `n_blocks`.
5. Replace the vacuous `test_late_block_start_defaults` with a real test of the auto-adjustment function.

After these fixes, the code will be **trainable and inference-safe**, though still lacking validation metrics, warmup, and sufficient epoch defaults for publication-quality results.
