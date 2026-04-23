# NeurIPS Re-Verification: Apple DepthPro DoRA Adapter Implementation

**Previous Review:** `reports/neurips_review_depthpro.md` (Rated: ⭐ Reject)  
**Reviewer:** Depth Estimation & Self-Supervised Adaptation Specialist  
**Date:** 2026-04-24

---

## Executive Summary

Of the 22 specific issues flagged in the previous review, **only 8 were fully fixed**, **4 were partially fixed**, and **10 remain completely unaddressed**. The fixes that were applied (ranking loss teacher target, eval-mode base model, augmentation usage, `copy.deepcopy` elimination, deterministic CUDA, clamped SI log loss, and self-pair exclusion) are mechanically correct. However, the most consequential systemic problems—**inappropriate MSE distillation for metric depth**, **absence of any validation loop**, **900 MB memory waste from base-weight cloning**, **silent adapter-injection failure modes**, **no mixed-precision training**, **no LR warmup**, and **PIL reconstruction CPU bottlenecks**—were left untouched. Worse, the fix for `late_block_start` introduces a **new silent override bug** that can corrupt explicit user intent, and the deterministic-seed helper sets `PYTHONHASHSEED` too late to be effective. The code is now *less likely to crash immediately*, but it is still **scientifically unsound for training** and **operationally fragile for inference**.

**Overall Rating:** ⭐ Reject (Major Revision Still Required)

---

## Issue-by-Issue Verification

### Issue 1: Ranking loss compares the student to itself
**Previous Severity:** CRITICAL  
**Status:** ✅ FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:102-134
def relative_depth_ranking_loss(student_depth, teacher_depth, num_pairs=1024, margin=0.1):
    ...
    s_i = student_flat[b, idx_i]
    s_j = student_flat[b, idx_j]
    t_i = teacher_flat[b, idx_i]
    t_j = teacher_flat[b, idx_j]
    # Teacher defines the ground-truth ordering
    target = torch.sign(t_i - t_j)
    # Skip pairs where teacher has equal depth (target=0)
    valid = target != 0
    if valid.any():
        l = F.margin_ranking_loss(s_i[valid], s_j[valid], target[valid], margin=margin)
```

The target is now derived from **`teacher_depth`** (`t_i - t_j`), not the student. Self-pairs are excluded by a resampling loop (lines 118-121), and equal-depth pairs are masked out with `valid = target != 0`. The call site at line 255 correctly passes the teacher output:
```python
l_rank = relative_depth_ranking_loss(student_out, teacher_out)
```

**Remaining Concerns:** Hyperparameters (`num_pairs=1024`, `margin=0.1`) are still unjustified (see Issue 7).

---

### Issue 2: Training script default contradicts the architecture spec
**Previous Severity:** CRITICAL  
**Status:** ⚠️ PARTIALLY FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:323
parser.add_argument("--late_block_start", type=int, default=6)

# train_depth_adapter_lora.py:335-341
if args.late_block_start == 6:  # user didn't override
    if args.model_type in ("da2", "depthpro"):
        args.late_block_start = 18
        logger.info("Auto-set late_block_start=%d for %s ...", args.late_block_start, args.model_type)
    elif args.model_type == "dav3":
        args.late_block_start = 18
        logger.info("Auto-set late_block_start=%d for %s", args.late_block_start, args.model_type)
```

The *default* run (no CLI override) now correctly arrives at `18` for DepthPro, DA2, and DAv3. However:
- The argparse default is still `6`. The code cannot distinguish between "user accepted the default" and "user explicitly passed `--late_block_start 6` for a 12-block ablation." In the latter case, the script **silently overwrites** the user's explicit choice to `18`.
- The auto-adjustment is hardcoded by `model_type` string, not by inspecting the actual number of layers in the loaded model.

**Remaining Concerns:** The fix is a brittle runtime patch, not a corrected default. Make `18` the argparse default and remove the conditional override.

---

### Issue 3: Augmented views are computed and then thrown away
**Previous Severity:** CRITICAL  
**Status:** ✅ FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:223-226
img = batch["img"].to(device)
img_aug = batch.get("img_aug")
if img_aug is not None:
    img_aug = img_aug.to(device)

# train_depth_adapter_lora.py:229-235 (teacher on clean image)
with torch.no_grad():
    ...
    teacher_out = teacher_model(**teacher_inputs).predicted_depth

# train_depth_adapter_lora.py:238 (student on augmented image)
student_input = img_aug if img_aug is not None else img
```

The asymmetric teacher-student pipeline is now correctly wired: teacher sees clean `img`; student sees `img_aug`.

**Remaining Concerns:** Both branches still waste CPU cycles on PIL reconstruction (see Issue 17).

---

### Issue 4: Base model left in `train()` mode may corrupt frozen features
**Previous Severity:** CRITICAL  
**Status:** ✅ FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:212-215
model.eval()  # Frozen base in eval mode
for m in model.modules():
    if isinstance(m, (LoRALinear, DoRALinear, ConvDoRALinear)):
        m.train()  # Only adapters in train mode
```

Dropout / batch-norm in the frozen backbone is now disabled, while adapter dropouts remain active. This is the correct pattern.

**Remaining Concerns:** None.

---

### Issue 5: MSE distillation is inappropriate for metric depth
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:94-99
def self_distillation_loss(student_out, teacher_out, mask=None):
    if mask is not None:
        diff = (student_out - teacher_out.detach()) ** 2
        return (diff * mask).sum() / mask.sum().clamp(min=1)
    return F.mse_loss(student_out, teacher_out.detach())
```

The distillation loss is still **pure MSE** on metric depth values. A 1 m error at 100 m still contributes 10,000× more loss than a 1 m error at 1 m, biasing the student toward close-range accuracy and suppressing far-field boundary learning.

**Required Change:** Replace with log-L1, relative L1, or a scale-aware distillation loss.

---

### Issue 6: MSE and scale-invariant losses are in direct conflict
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# configs/depth_adapter_baseline.yaml:34-38
losses:
  names: ["distillation", "ranking", "scale_invariant"]
  weights:
    distillation: 1.0
    ranking: 0.1
    scale_invariant: 0.5
```

The training script still applies both objectives with their original weights:
```python
# train_depth_adapter_lora.py:248-264
if "distillation" in losses:
    ...
    loss_total = loss_total + w * l_dist
if "scale_invariant" in losses:
    ...
    loss_total = loss_total + w * l_si
```

MSE (weight 1.0) fights to preserve absolute metric scale, while SI log loss (weight 0.5) explicitly discards global scale. No comment, no down-weighting, no reconciliation strategy.

**Required Change:** Either remove SI loss or reduce its weight to ≤ 0.1, and document the rationale.

---

### Issue 7: Ranking loss hyperparameters are unjustified
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:102
def relative_depth_ranking_loss(student_depth, teacher_depth, num_pairs=1024, margin=0.1):
```

- **`num_pairs=1024`** still covers only ~0.2 % of pixels in a 512×1024 image. Uniform random sampling will almost never hit boundary pixels.
- **`margin=0.1`** is still an **absolute** threshold in meters. It is absurdly large for close objects and vanishingly small for far objects.
- **No relative margin** (e.g., `margin = 0.05 * max(d_i, d_j)`) was added.

**Remaining Concerns:** The self-pair and equal-depth bugs are fixed, but the core hyperparameter design is still flawed.

---

### Issue 8: `copy.deepcopy` of a 1B-parameter HF model
**Previous Severity:** MAJOR  
**Status:** ✅ FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:372-382
if args.model_type == "dav3":
    teacher_model = load_dav3_model(model_name, device="cpu")
elif args.model_type == "da2":
    teacher_model = load_da2_model(model_name, device="cpu")
elif args.model_type == "depthpro":
    teacher_model = load_depthpro_model(model_name, device="cpu")
teacher_model.eval()
for p in teacher_model.parameters():
    p.requires_grad = False
teacher_model = teacher_model.to(device)
```

The teacher is now re-instantiated from pretrained weights. No `copy.deepcopy`.

**Remaining Concerns:** Loading on CPU then moving to GPU is slower than `device_map="auto"` or loading directly to GPU, but it is safe and memory-efficient.

---

### Issue 9: DoRA adapters clone base weights, wasting ~900 MB
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# mbps_pytorch/models/adapters/lora_layers.py:40 (LoRALinear)
self.weight = nn.Parameter(wrapped.weight.data.clone(), requires_grad=False)

# mbps_pytorch/models/adapters/lora_layers.py:92 (DoRALinear)
self.weight = nn.Parameter(wrapped.weight.data.clone(), requires_grad=False)

# mbps_pytorch/models/adapters/lora_layers.py:208 (LoRAConv2d)
self.weight = nn.Parameter(wrapped.weight.data.clone(), requires_grad=False)
```

Every adapted layer still stores a **duplicate** of its base weight. Because `wrap_linear_if_match` keeps the original `nn.Linear` alive inside the adapter instance (`wrapped` is not deleted), the clone is pure waste. With ~226M parameters cloned across two encoders, this wastes ~900 MB in fp32.

**Required Change:** Reference `wrapped.weight` directly instead of cloning.

---

### Issue 10: No validation loop or convergence proxy
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

**Evidence:**
The `train_depth_adapter` function (lines 194–301) still contains **no validation protocol**. There is no held-out split, no depth-boundary F-score, no visual logging of sample depth maps, and no PQ_things proxy. The "best" checkpoint is selected by training loss alone, which is meaningless because the losses can decrease while depth quality degrades.

**Required Change:** Add a validation loop with at minimum boundary-recall metrics and periodic depth-map visualization.

---

### Issue 11: Sobel threshold `τ=0.20` is arbitrary and scale-dependent
**Previous Severity:** MAJOR  
**Status:** ⚠️ PARTIALLY FIXED

**Evidence:**
```python
# mbps_pytorch/instance_methods/sobel_cc.py:11
def sobel_cc_instances(..., grad_threshold=0.03, ...):

# mbps_pytorch/generate_instance_pseudolabels_adapted.py:32
def depth_guided_instances(..., tau=0.03, ...):

# mbps_pytorch/generate_instance_pseudolabels_adapted.py:178
parser.add_argument("--tau", type=float, default=0.20)
```

The core library (`sobel_cc.py`) and the internal function default were changed to `0.03`. **However**, the CLI entry-point in `generate_instance_pseudolabels_adapted.py` still defaults to **`0.20`**. A user running the generation script without an explicit `--tau` flag will therefore use the old, criticized threshold.

**Required Change:** Align the CLI default with the library default (`0.03`).

---

### Issue 12: `min_area=1000` pixels suppresses small instances
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# mbps_pytorch/instance_methods/sobel_cc.py:12
def sobel_cc_instances(..., min_area=1000, ...):

# mbps_pytorch/generate_instance_pseudolabels_adapted.py:33
def depth_guided_instances(..., min_area=1000, ...):

# mbps_pytorch/generate_instance_pseudolabels_adapted.py:179
parser.add_argument("--min_area", type=int, default=1000)
```

`min_area=1000` remains hardcoded in both the library and the CLI. At 512×1024, this filters out riders, bicycles, and distant pedestrians.

**Required Change:** Scale by image area or validate against Cityscapes instance-size statistics.

---

### Issue 13: Greedy dilation reclamation merges adjacent instances
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# mbps_pytorch/instance_methods/sobel_cc.py:46-52
cc_list.sort(key=lambda x: -x[1])
for cc_mask, area in cc_list:
    if dilation_iters > 0:
        dilated = ndimage.binary_dilation(cc_mask, iterations=dilation_iters)
        reclaimed = dilated & cls_mask & ~assigned
        final_mask = cc_mask | reclaimed
```

Instances are still processed in **strict descending area order**. A large instance dilates first and reclaims boundary pixels before smaller neighbors get a chance. The systematic bias toward large instances stealing boundaries from small adjacent ones persists.

**Note:** The alternate `depth_guided_instances` function in `generate_instance_pseudolabels_adapted.py` does not use greedy instance dilation; however, the file specifically cited in the audit (`sobel_cc.py`) remains unchanged.

---

### Issue 14: Silent failure if adapter injection misses layers
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# mbps_pytorch/models/adapters/depthpro_adapter.py:51-58
encoder = getattr(dinov2_model, "encoder", None)
if encoder is None:
    logger.warning("No encoder found in %s", prefix)
    return adapted

layers = getattr(encoder, "layer", None)
if layers is None:
    logger.warning("No encoder.layer found in %s", prefix)
    return adapted
```

If HF changes internal attribute names (e.g., `encoder.layer` → `encoder.layers`), the function logs a `warning` and returns an empty dict. `inject_lora_into_depthpro` then logs "0 layers adapted," `freeze_non_adapter_params` freezes everything, and training proceeds with **zero trainable parameters**. The script logs "Trainable adapter params: 0" at line 205, but this is easy to miss in log noise, and training still burns GPU hours uselessly.

**Required Change:** Raise a `RuntimeError` if zero layers were adapted.

---

### Issue 15: `ConvDoRALinear` is non-functional without `_spatial_dims`
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# mbps_pytorch/models/adapters/lora_layers.py:141
self._spatial_dims: Optional[Tuple[int, int]] = None

# mbps_pytorch/models/adapters/lora_layers.py:153-169
h_p, w_p = self._spatial_dims or (None, None)
if h_p is not None and w_p is not None:
    ...  # DWConv path
```

`_spatial_dims` is still initialized to `None` and **no code in the repository sets it**. If a user selects `variant="conv_dora"`, the model silently falls back to standard DoRA while carrying the extra (unused) conv parameters.

**Required Change:** Inject spatial dimensions automatically during model forward, or remove the `conv_dora` variant from the choices.

---

### Issue 16: Batch size 4 is too small for stable adaptation
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:315
parser.add_argument("--batch_size", type=int, default=4)

# train_depth_adapter_lora.py:330
parser.add_argument("--grad_accum_steps", type=int, default=1, ...)

# configs/depth_adapter_baseline.yaml:22
batch_size: 4
```

Effective batch size is still 4. For 1.66M+ parameters and a highly stochastic ranking loss, gradient variance remains extreme.

**Required Change:** Default `grad_accum_steps` to at least 8, or increase `batch_size`.

---

### Issue 17: Teacher inputs are wastefully reconstructed via PIL every forward pass
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:233
teacher_inputs = processor(images=[Image.fromarray((i.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)) for i in img], return_tensors="pt")

# train_depth_adapter_lora.py:242
student_inputs = processor(images=[Image.fromarray((i.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)) for i in student_input], return_tensors="pt")
```

Both teacher and student paths still execute this CPU-bound, quantizing round-trip for every forward pass. The fix for Issue 3 (asymmetric augmentation) **doubled** the number of PIL reconstructions per batch (teacher + student), making the bottleneck worse.

**Required Change:** Pass raw tensors directly to the processor, or pre-cache `pixel_values` in the dataset.

---

### Issue 18: No warmup in the LR schedule
**Previous Severity:** MINOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:208
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

Training still starts at full `lr=1e-4` from step 0. For DoRA magnitude parameters, large initial updates can destabilize pretrained feature directions.

**Required Change:** Add a short linear warmup (e.g., 500 steps) before cosine annealing.

---

### Issue 19: `margin_ranking_loss` receives invalid `target=0`
**Previous Severity:** MINOR  
**Status:** ✅ FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:118-121
# Ensure no self-pairs
mask_same = idx_i == idx_j
while mask_same.any():
    idx_j[mask_same] = torch.randint(0, H * W, (mask_same.sum(),), device=device)
    mask_same = idx_i == idx_j

# train_depth_adapter_lora.py:128-132
target = torch.sign(t_i - t_j)
# Skip pairs where teacher has equal depth (target=0)
valid = target != 0
if valid.any():
    l = F.margin_ranking_loss(s_i[valid], s_j[valid], target[valid], margin=margin)
```

Self-pairs are explicitly resampled until distinct, and equal-depth teacher pairs are filtered by `valid = target != 0` before being passed to `F.margin_ranking_loss`.

**Remaining Concerns:** None.

---

### Issue 20: `+1e-6` epsilon in log-space risks gradient explosion near zero
**Previous Severity:** MINOR  
**Status:** ✅ FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:137-144
def scale_invariant_loss(pred, target, lambda_si=0.5, min_depth=1e-3):
    pred_clamped = torch.clamp(pred, min=min_depth)
    target_clamped = torch.clamp(target.detach(), min=min_depth)
    diff = torch.log(pred_clamped) - torch.log(target_clamped)
```

The dangerous `+1e-6` pattern is gone. Values are clamped to `min_depth=1e-3` before `torch.log`, bounding the maximum gradient to `1e3` instead of `1e6`.

**Remaining Concerns:** `1e-3` meters (1 mm) is smaller than any realistic Cityscapes depth, so the clamp is physically safe.

---

### Issue 21: Missing mixed-precision training support
**Previous Severity:** MINOR  
**Status:** ❌ NOT FIXED

**Evidence:**
No `torch.cuda.amp.autocast` context or `torch.amp.GradScaler` appears anywhere in the training loop. The adapter `forward` methods already include `.to(x.dtype)` casts (e.g., line 55 in `lora_layers.py`), so AMP readiness exists, but it is not utilized.

**Required Change:** Wrap the forward/backward pass in ` autocast()` and use `GradScaler`.

---

### Issue 22: Deterministic CUDA settings
**Previous Severity:** Not previously ranked (added in checklist)  
**Status:** ✅ FIXED (with a bug)

**Evidence:**
```python
# train_depth_adapter_lora.py:50-58
def set_seed(seed=42):
    ...
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
```

`torch.backends.cudnn.deterministic = True` and `benchmark = False` are correctly set. The function is called at line 343 before data loading.

**New Bug Introduced:** `os.environ["PYTHONHASHSEED"] = str(seed)` is set **after the interpreter has already started**. `PYTHONHASHSEED` must be set before Python startup to affect hash randomization; setting it at runtime is a no-op. This gives a false sense of full determinism.

---

## New Issues Introduced by Fixes

### New Issue A: Explicit `--late_block_start 6` is silently overridden
**Severity:** MAJOR  
**Location:** `train_depth_adapter_lora.py:335-341`

The fix for Issue 2 uses `if args.late_block_start == 6` to detect "user didn't override." This conflates the argparse default with an explicit user choice. A researcher running an ablation with `--late_block_start 6` on a custom 12-block model will have their hyperparameter silently rewritten to `18` without warning. The log line says "Auto-set" but does not indicate that a user-provided value was overwritten.

**Fix:** Change the argparse default to `None`, then set the true default inside `main()` only when the argument is `None`.

### New Issue B: `PYTHONHASHSEED` set too late to be effective
**Severity:** MINOR  
**Location:** `train_depth_adapter_lora.py:58`

The `set_seed` helper (added to fix Issue 22) sets `os.environ["PYTHONHASHSEED"]` at runtime. This environment variable is read once by the CPython interpreter at startup. Setting it inside a Python function has zero effect on hash randomization, defeating the purpose of the fix for users who rely on it for full reproducibility.

**Fix:** Set `PYTHONHASHSEED` in the shell / launch script before invoking Python, or remove the line and document the requirement.

### New Issue C: Orphaned YAML configuration file
**Severity:** MAJOR (workflow / reproducibility)  
**Location:** `configs/depth_adapter_baseline.yaml`

The repository contains a carefully structured YAML config (`depth_adapter_baseline.yaml`) that documents the intended hyperparameters (`late_block_start: 18`, loss weights, image size, etc.). However, `train_depth_adapter_lora.py` only accepts argparse CLI arguments and has **no `--config` loader**. The YAML is therefore completely disconnected from the training script. Users reading the config will believe those values are active, but they are ignored unless manually transcribed to the command line. This is a serious reproducibility trap.

**Fix:** Add `--config` argument that loads the YAML and populates `args`, or remove the YAML to avoid confusion.

### New Issue D: Test 8 provides false confidence
**Severity:** MINOR  
**Location:** `mbps_pytorch/tests/test_depth_adapters.py:478-484`

The newly added `test_late_block_start_defaults` contains only tautological assertions:
```python
def test_late_block_start_defaults():
    assert 18 == 18, "Default late_block_start for 24-block models should be 18"
    assert 6 == 6, "Default late_block_start for 12-block models should be 6"
```

This test does **not** exercise the argparse logic, the auto-adjustment condition, or the model-type branching. It will always pass and gives no protection against regression.

**Fix:** Replace with a test that instantiates the argument parser and verifies the parsed result for each `model_type`.

### New Issue E: Student branch doubles the PIL CPU bottleneck
**Severity:** MAJOR (performance)  
**Location:** `train_depth_adapter_lora.py:242`

Before the fix for Issue 3, the student branch used the clean image (same tensor as the teacher), so both branches shared the same wasteful PIL reconstruction path. After the fix, the student receives `img_aug`, which is a **different tensor**, forcing a **second** CPU-round-trip PIL reconstruction per batch. The effective CPU overhead for the non-DAv3 path has doubled.

**Fix:** Cache processed `pixel_values` in the dataset and pass them directly, bypassing the processor entirely.

---

## Overall Recommendation

**Reject — Major Revision Still Required.**

While the authors successfully patched the most egregious crash bugs (self-referential ranking loss, `copy.deepcopy` OOM risk, and eval-mode corruption), the implementation remains **scientifically and operationally unsound** for the following reasons:

1. **The training objective is still broken.** MSE on metric depth is theoretically inappropriate, and its conflict with the scale-invariant loss is unresolved. The ranking loss hyperparameters are still arbitrary.
2. **There is still no validation protocol.** Training blindly minimizes a training loss that can decrease while the adapted depth maps diverge from the teacher. No boundary metrics, no visual inspection, no PQ_things proxy.
3. **Memory waste is unaddressed.** Base-weight cloning wastes ~900 MB, and no AMP means the model runs in full fp32 for no reason.
4. **Silent failure modes persist.** Adapter injection can miss every layer and training will still run. Conv-DoRA is a dead code path. The YAML config is ignored.
5. **Performance bottlenecks were worsened.** The asymmetric augmentation fix doubled the CPU-bound PIL reconstruction overhead.

### Required Changes Before Training

- [ ] **Replace MSE distillation** with log-L1 or relative L1 for metric depth.
- [ ] **Reconcile SI loss:** Remove it or reduce weight to ≤ 0.1 and document the choice.
- [ ] **Add a validation loop:** At minimum log sample depth maps and compute boundary F-score against the teacher on a held-out split.
- [ ] **Remove base-weight cloning:** Reference `wrapped.weight` directly in all adapter layers.
- [ ] **Add AMP:** Use `torch.amp.autocast` and `GradScaler`.
- [ ] **Add LR warmup:** 500-step linear warmup before cosine annealing.
- [ ] **Fix `late_block_start` default properly:** Use `default=None` in argparse and set the real default in code.
- [ ] **Align inference thresholds:** Set `generate_instance_pseudolabels_adapted.py` CLI default `--tau` to `0.03`.
- [ ] **Scale `min_area`:** Make it proportional to image area or dataset-validated.
- [ ] **Fix silent injection failure:** Raise `RuntimeError` if zero adapters are injected.
- [ ] **Fix or remove `conv_dora`:** Either inject `_spatial_dims` automatically or drop the variant.
- [ ] **Connect YAML config:** Add `--config` argument so `depth_adapter_baseline.yaml` is actually loaded.
- [ ] **Eliminate PIL reconstruction bottleneck:** Cache `pixel_values` in the dataset.
- [ ] **Replace vacuous test 8** with a real argparse integration test.
- [ ] **Fix `PYTHONHASHSEED`:** Set it in the launch environment, not at runtime.

Only after these changes can the submission be reconsidered.
