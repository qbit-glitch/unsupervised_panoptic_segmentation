# NeurIPS Re-Verification: DINOv2+CAUSE-TR DoRA Adapter Implementation

**Previous Review:** `reports/neurips_review_dinov2_cause_tr.md` (Rated: ⭐ Reject)  
**Reviewer:** Anonymous Re-Verifier  
**Specialization:** Self-supervised learning, parameter-efficient fine-tuning, unsupervised segmentation  
**Date:** 2026-04-24

---

## Executive Summary

The authors have addressed approximately half of the critical and major issues from the initial review. The three blocking critical issues—(1) nonsensical DINO distillation loss, (2) missing ImageNet normalization, and (3) zero-gradient CAUSE cluster loss—have been correctly fixed. Deterministic CUDA settings, suffix-based parameter freezing, and the dead `_m_init_norm` buffer have also been resolved. However, **the inference checkpoint loading safety (Issue 4) was only partially fixed** and remains dangerously permissive: a checkpoint without LoRA weights will still be loaded silently when `require_lora=True`. The EMA update is technically no longer a latent runtime crash, but `adapt_ema=False` means it remains a no-op for all trainable parameters. Several major concerns (batch size, epoch count, depth loss degeneracy, weak augmentations) were left untouched. Most troubling, **two new issues have been introduced**: (a) the augmented view pipeline in `AdapterTrainingDataset` now denormalizes an already-normalized tensor through `ToPILImage()`, clamping distribution tails and creating a train/augment view mismatch, and (b) the test suite endorses the EMA no-op by testing a physically impossible scenario. The implementation is closer to trainable, but it is **not yet ready for a production training run** without addressing the remaining checkpoint safety and data pipeline bugs.

**New Rating: ⚠️ Major Revision**

---

## Issue-by-Issue Verification

### Issue 1: The DINO Distillation Loss Is Mathematically Nonsensical
**Previous Severity:** CRITICAL  
**Status:** ✅ FIXED  
**Evidence:**
```python
# mbps_pytorch/train_semantic_adapter.py:118-124
def dino_distillation_loss(student_feat, teacher_feat):
    """Cosine-similarity distillation between student and teacher features."""
    student_feat = F.normalize(student_feat, dim=-1)
    teacher_feat = F.normalize(teacher_feat, dim=-1).detach()
    # Per-token cosine similarity; maximize similarity = minimize (1 - cos)
    loss = (1 - (student_feat * teacher_feat).sum(dim=-1)).mean()
    return loss
```
The softmax-over-features KL-divergence has been completely replaced with per-token cosine-similarity distillation, exactly as recommended. The teacher features are correctly detached.  
**Remaining Concerns:** None. The test at `test_dino_distillation_cosine()` (test_dora_adapter_training.py:428-446) additionally verifies the loss range is [0, 2] and behaves correctly for identical/opposite features.

---

### Issue 2: Catastrophic Train/Test Distribution Shift: Missing ImageNet Normalization
**Previous Severity:** CRITICAL  
**Status:** ✅ FIXED  
**Evidence:**
```python
# mbps_pytorch/train_semantic_adapter.py:477-482
img_transform = T.Compose([
    T.Resize(TRAIN_RESOLUTION, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(TRAIN_RESOLUTION),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```
Additionally, the augmented view is normalized after augmentation:
```python
# mbps_pytorch/train_semantic_adapter.py:220-221
img_aug = T.ToTensor()(self.aug_transform(img_pil))
# Normalize to match training distribution
img_aug = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_aug)
```
**Remaining Concerns:** The `img_aug` normalization path introduces a **new bug** (see New Issue 1 below): `item["img"]` is already an ImageNet-normalized tensor from `ContrastiveSegDataset`. Feeding it through `T.ToPILImage()` clamps normalized values to [0, 1] before re-normalizing, destroying distribution tails.

---

### Issue 3: CAUSE Cluster Loss Is Completely Disconnected from Student Learning
**Previous Severity:** CRITICAL  
**Status:** ✅ FIXED  
**Evidence:**
```python
# mbps_pytorch/train_semantic_adapter.py:300-302
if "cause_cluster" in losses:
    logger.warning("cause_cluster loss is deprecated and skipped (zero gradient).")
    totals["cause_cluster"] += 0.0
```
```python
# mbps_pytorch/train_semantic_adapter.py:459-460
if "cause_cluster" in args.losses:
    logger.warning("cause_cluster is deprecated; cluster_probe will remain frozen.")
```
The loss is skipped with a clear deprecation warning, and `cluster_probe` remains frozen. No gradient computation occurs.  
**Remaining Concerns:** None. However, the dead code in `Cluster.forward_centroid` (`.detach()` inside) is still present in the CAUSE submodule. This is outside the scope of the files under review but should be noted.

---

### Issue 4: Inference Can Silently Use Random Adapter Weights
**Previous Severity:** MAJOR  
**Status:** ⚠️ PARTIALLY FIXED  
**Evidence:**
```python
# mbps_pytorch/generate_semantic_pseudolabels_adapted.py:173-207
def _load_state_checked(module, state_dict, component_name, require_lora):
    result = module.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    dropped_lora = [
        k for k in unexpected
        if k.endswith(("lora_A", "lora_B", "lora_magnitude"))
        or ".lora_" in k
        or "dwconv" in k
        or "conv_gate" in k
    ]
    if dropped_lora:
        raise RuntimeError(
            f"[{component_name}] {len(dropped_lora)} LoRA/DoRA parameters were dropped..."
        )
    if missing:
        base_missing = [k for k in missing if ".lora_" not in k]
        if base_missing:
            logger.warning("[%s] %d base params missing...", ...)
    if require_lora:
        lora_loaded = sum(
            1 for k in state_dict.keys()
            if k.endswith(("lora_A", "lora_B", "lora_magnitude"))
            or "dwconv" in k
            or "conv_gate" in k
        )
        logger.info("[%s] %d LoRA-style parameters loaded successfully.", ...)
```
**Remaining Concerns:** The fix catches the case where the **checkpoint has LoRA keys but the model lacks adapter wrappers** (`dropped_lora`). However, it **does NOT catch the opposite and more dangerous case**: the model has adapter wrappers (`require_lora=True`) but the **checkpoint lacks LoRA keys entirely**. In that case:
- `missing` contains all adapter keys
- `base_missing` is empty (all base keys present)
- `lora_loaded = 0`
- The function logs `"0 LoRA-style parameters loaded successfully"` and **returns without error**.

The adapter parameters (`lora_A`, `lora_B`, `lora_magnitude`) remain at their **random initialization**. The subsequent `n_adapter == 0` check at line 330 passes because the wrappers *do* contain parameters—they are just random. **This is exactly the silent failure mode the original review identified.** The required assertion—"every expected adapter key in the model exists in the checkpoint when `require_lora=True`"—is absent.

---

### Issue 5: Cross-View Consistency Lacks Stop-Gradient, Enabling Collapse
**Previous Severity:** MAJOR  
**Status:** ✅ FIXED  
**Evidence:**
```python
# mbps_pytorch/train_semantic_adapter.py:127-130
def cross_view_consistency_loss(feat1, feat2):
    feat1 = F.normalize(feat1, dim=-1)
    feat2 = F.normalize(feat2, dim=-1).detach()  # Stop-gradient on augmented view
    return (1 - (feat1 * feat2).sum(dim=-1)).mean()
```
**Remaining Concerns:** None. The test at `test_cross_view_stop_gradient()` (test_dora_adapter_training.py:492-504) verifies `feat2.grad is None` after backward.

---

### Issue 6: Batch Size 4 Is Far Below Stable Self-Supervised Regime
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED  
**Evidence:**
```python
# mbps_pytorch/train_semantic_adapter.py:342
parser.add_argument("--batch_size", type=int, default=4)
```
**Remaining Concerns:** Default remains 4. For self-supervised distillation with teacher centering dynamics, this is still far below the stable regime (≥32–64). No gradient accumulation was added.

---

### Issue 7: Depth Correlation Loss Is Degenerate
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED  
**Evidence:**
```python
# mbps_pytorch/train_semantic_adapter.py:141-153
def depth_correlation_loss(code, depth, feature_samples=11, shift=0.0):
    ...
    cd = torch.einsum("nchw,ncij->nhwij", norm(code_sampled), norm(code_sampled))
    dd = torch.einsum("nchw,ncij->nhwij", norm(depth_sampled), norm(depth_sampled))
    loss = -cd.clamp(0.0, 0.8) * (dd - shift)
    return loss.mean()
```
Additionally:
```python
# mbps_pytorch/train_semantic_adapter.py:138
def grid_sample(t, coords):
    return F.grid_sample(t, coords, padding_mode="zeros", align_corners=True, mode="bilinear")
```
**Remaining Concerns:** `align_corners=True` is still used. `F.normalize` on 1-channel depth still reduces depth magnitude to signs (±1). The arbitrary `[0.0, 0.8]` clamp on `cd` remains unjustified. Missing depth files still silently return zeros (`torch.zeros(1, 23, 23)`) with no warning logged during training.

---

### Issue 8: EMA Update Is a No-Op
**Previous Severity:** MAJOR  
**Status:** ⚠️ PARTIALLY FIXED  
**Evidence:**
```python
# mbps_pytorch/train_semantic_adapter.py:83-91
def ema_update(student_head, teacher_head, lamb=0.99):
    student_state = dict(student_head.named_parameters())
    with torch.no_grad():
        for name, p_t in teacher_head.named_parameters():
            if name in student_state:
                p_s = student_state[name]
                if p_s.shape == p_t.shape:
                    p_t.data = lamb * p_t.data + (1 - lamb) * p_s.data
```
```python
# mbps_pytorch/train_semantic_adapter.py:452
inject_lora_into_cause_tr(
    segment, ..., adapt_head=True, adapt_projection=False, adapt_ema=False,
)
```
**Remaining Concerns:** The function was fixed from a latent **undefined variable bug** (`p_s` was used before assignment in the original). However, `adapt_ema=False` means the EMA head (`segment.head_ema`) has **no adapter parameters**. The student head's adapter parameters (`lora_A`, `lora_B`, `lora_magnitude`) are present in `student_state` but absent from `teacher_head.named_parameters()`, so they are silently skipped. The base weights in both heads are **frozen** by `freeze_non_adapter_params`. Consequently, the EMA update still copies frozen base weights into frozen base weights—a no-op for all **trainable** parameters. The spec's claim that EMA "prevents representation collapse during clustering" remains unfounded.

Additionally, the test `test_ema_update_correctness` (test_dora_adapter_training.py:275-318) **endorses this no-op behavior** by perturbing frozen base weights (which never change in actual training) and asserting the EMA copies them. This test validates a physically impossible scenario.

---

### Issue 9: `strict=False` Loading Masks Silent Weight Loading Failures
**Previous Severity:** MAJOR  
**Status:** ⚠️ PARTIALLY FIXED  
**Evidence:**
```python
# mbps_pytorch/train_semantic_adapter.py:382-386
result = backbone.load_state_dict(state, strict=False)
if result.missing_keys:
    logger.warning("Backbone missing keys: %s", result.missing_keys[:10])
if result.unexpected_keys:
    logger.warning("Backbone unexpected keys: %s", result.unexpected_keys[:10])
```
**Remaining Concerns:** Missing and unexpected keys are now **logged**, but the code does **not assert or raise** on missing base weights. A corrupted checkpoint with a single renamed key would trigger a warning and proceed with randomly initialized weights. The original review requested: "assert that no base weights are missing after loading." This hard assertion is absent. The same pattern is repeated for the teacher backbone, segment, and teacher segment.

Furthermore, in the **inference script**, the base DINOv2 and CAUSE segment loading still uses naked `strict=False` without any inspection:
```python
# mbps_pytorch/generate_semantic_pseudolabels_adapted.py:256
backbone.load_state_dict(state, strict=False)
# mbps_pytorch/generate_semantic_pseudolabels_adapted.py:278-280
segment.load_state_dict(..., strict=False)
```
These lines do not log missing/unexpected keys at all.

---

### Issue 10: 10-Epoch CosineAnnealing Is Insufficient
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED  
**Evidence:**
```python
# mbps_pytorch/train_semantic_adapter.py:344
parser.add_argument("--epochs", type=int, default=10)
```
**Remaining Concerns:** Default remains 10 epochs. With batch size 4 and ~2,975 images, this is ~7,440 steps. Self-supervised adapter convergence typically requires 50–100 epochs. No Warmup + Cosine with longer horizon was added.

---

### Issue 11: Training Augmentations Are Pathetically Weak
**Previous Severity:** MAJOR  
**Status:** ⚠️ PARTIALLY FIXED  
**Evidence:**
```python
# mbps_pytorch/train_semantic_adapter.py:168-173
self.aug_transform = T.Compose([
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.RandomHorizontalFlip(p=0.5),
    T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
])
```
**Remaining Concerns:** `RandomHorizontalFlip` and `GaussianBlur` were added. However, the pipeline **still lacks**:
- `RandomResizedCrop` (the only spatial sampling is `CenterCrop` in `img_transform`)
- `RandomSolarization`
- Multi-crop (the CAUSE `ContrastiveSegDataset` may provide some, but the adapter wrapper does not leverage it)

For self-supervised distillation, this augmentation strength is still weaker than DINOv2's standard.

---

### Issue 12: Sliding Window Inference Produces Non-Uniform Overlap
**Previous Severity:** MINOR  
**Status:** ❌ NOT FIXED  
**Evidence:**
```python
# mbps_pytorch/generate_semantic_pseudolabels_adapted.py:102-103
if not y_positions or y_positions[-1] + crop_size < H:
    y_positions.append(H - crop_size)
```
**Remaining Concerns:** Unchanged. The last crop still overlaps more than 50% with its neighbor when the image dimensions are not exact multiples of the stride.

---

### Issue 13: Inference Silently Crops up to 13 Pixels
**Previous Severity:** MINOR  
**Status:** ❌ NOT FIXED  
**Evidence:**
```python
# mbps_pytorch/generate_semantic_pseudolabels_adapted.py:74-77
new_H = (H // 14) * 14
new_W = (W // 14) * 14
if new_H != H or new_W != W:
    tensor = F.interpolate(tensor, size=(new_H, new_W), mode="bilinear", align_corners=False)
```
**Remaining Concerns:** Unchanged. For Cityscapes (1024×2048), this becomes 1022×2044. Still undocumented in the script's docstring or output.

---

### Issue 14: Reproducibility: Missing Deterministic CUDA Settings
**Previous Severity:** MINOR  
**Status:** ✅ FIXED  
**Evidence:**
```python
# mbps_pytorch/train_semantic_adapter.py:70-71
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
**Remaining Concerns:** None.

---

### Issue 15: Unused Config YAML File
**Previous Severity:** MINOR  
**Status:** ❌ NOT FIXED  
**Evidence:** `configs/semantic_adapter_baseline.yaml` still exists and is meticulously structured but **never loaded** by `train_semantic_adapter.py`. The YAML still contains stale keys (`temp_student: 0.1`, `temp_teacher: 0.07`) for the removed temperature parameters.  
**Remaining Concerns:** Maintenance hazard remains. The CLI defaults and YAML can diverge silently.

---

### Issue 16: `freeze_non_adapter_params` Uses Fragile Substring Matching
**Previous Severity:** MINOR  
**Status:** ✅ FIXED  
**Evidence:**
```python
# mbps_pytorch/models/adapters/lora_layers.py:297-309
ADAPTER_SUFFIXES = (".lora_A", ".lora_B", ".lora_magnitude", ".dwconv.weight", ".conv_gate",
                    ".lora_A.weight", ".lora_B.weight")
for name, param in model.named_parameters():
    if any(name.endswith(suffix) for suffix in ADAPTER_SUFFIXES):
        param.requires_grad = True
    else:
        param.requires_grad = False
```
**Remaining Concerns:** None. The suffix-based matching eliminates accidental substring collisions.

---

### Issue 17: Dead Code: `_m_init_norm` Buffer
**Previous Severity:** MINOR  
**Status:** ✅ FIXED  
**Evidence:** The `_m_init_norm` buffer has been removed from `DoRALinear.__init__` (`mbps_pytorch/models/adapters/lora_layers.py:69-120`).  
**Remaining Concerns:** None.

---

## New Issues Introduced by Fixes

### New Issue A: Augmented View Pipeline Denormalizes-and-Reclips an Already-Normalized Tensor [CRITICAL]
**Location:** `mbps_pytorch/train_semantic_adapter.py:215-225`

```python
if self.use_augmentation and "img" in item:
    img = item["img"]
    if isinstance(img, torch.Tensor):
        img_pil = T.ToPILImage()(img)                          # Line 218
        img_aug = T.ToTensor()(self.aug_transform(img_pil))    # Line 219
        img_aug = T.Normalize(...)(img_aug)                    # Line 221
        item["img_aug"] = img_aug
```

`item["img"]` originates from `ContrastiveSegDataset`, which applies `img_transform` including `T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`. It is therefore already **ImageNet-normalized** (values outside [0, 1]). Passing this tensor through `T.ToPILImage()` clamps values to [0, 1] before converting to uint8, **destroying the tails of the distribution** (all negative values become 0, all values > 1 become 255). After augmentation and re-normalization, `img_aug` follows a **clipped, bounded distribution** while the clean view `img` sees the full normalized range. This creates an **intra-batch distribution mismatch** between the distillation target (clean, full range) and the cross-view consistency input (clipped, bounded range). The network may learn to map two different input manifolds to the same output, which is an ill-posed objective.

**Fix:** Do not convert the normalized tensor to PIL. Instead, either:
1. Apply augmentation as tensor transforms (e.g., `T.RandomHorizontalFlip`, `T.ColorJitter` accept tensors directly in modern torchvision), or
2. Normalize *after* `ToTensor()` in the base dataset, and apply augmentation to the [0, 1] tensor before normalizing.

---

### New Issue B: `_load_state_checked` Still Permits Random Adapter Initialization [CRITICAL]
**Location:** `mbps_pytorch/generate_semantic_pseudolabels_adapted.py:173-207`

As detailed in Issue 4 verification above, the fix added a check for `dropped_lora` (unexpected keys in checkpoint) but **omitted the mirror check**: when `require_lora=True`, the function must assert that **every adapter key present in the model state_dict also exists in the checkpoint state_dict**. Without this, loading a baseline CAUSE checkpoint into an adapter-wrapped model logs `"0 LoRA-style parameters loaded successfully"` and proceeds with random adapters. This is the exact silent failure mode the original review demanded be hardened.

**Fix:** After `load_state_dict`, when `require_lora=True`:
```python
model_adapter_keys = {k for k in module.state_dict().keys() if k.endswith(ADAPTER_SUFFIXES) or ...}
ckpt_adapter_keys = {k for k in state_dict.keys() if ...}
missing_in_ckpt = model_adapter_keys - ckpt_adapter_keys
if missing_in_ckpt:
    raise RuntimeError(f"...")
```

---

### New Issue C: Inference Script Loads Base Weights Without Validation [MAJOR]
**Location:** `mbps_pytorch/generate_semantic_pseudolabels_adapted.py:256` and `:278-280`

Unlike the training script, the inference script loads the DINOv2 backbone and CAUSE segment baseline with naked `strict=False` and never inspects `missing_keys` or `unexpected_keys`:
```python
backbone.load_state_dict(state, strict=False)
segment.load_state_dict(torch.load(...), strict=False)
```
A corrupted or mismatched baseline checkpoint would silently produce random base weights. At minimum, these should log warnings identical to the training script.

---

### New Issue D: EMA Test Validates a Physically Impossible Scenario [MAJOR]
**Location:** `mbps_pytorch/tests/test_dora_adapter_training.py:275-318`

The `test_ema_update_correctness` test:
1. Creates a student head **with adapters** and an EMA head **without adapters**.
2. Calls `freeze_non_adapter_params` on both.
3. **Perturbs the student's base weights** (which are frozen and never change in training).
4. Asserts the EMA copies these perturbed base weights.

In actual training, `freeze_non_adapter_params(segment)` is called at `train_semantic_adapter.py:454`, so base weights **never change**. The test validates a scenario that literally cannot occur, and by passing, it gives false confidence that the EMA mechanism is correct. In reality, the EMA is a no-op for all trainable parameters because `adapt_ema=False`.

**Fix:** The test should either:
- Test with `adapt_ema=True` and verify adapter parameters are smoothed, OR
- Assert that when `adapt_ema=False`, the EMA head does NOT update for adapter-shaped params (which it currently skips silently).

---

### New Issue E: Unnormalized `img_aug` When Base Dataset Returns PIL Images [MINOR]
**Location:** `mbps_pytorch/train_semantic_adapter.py:223-224`

```python
else:
    item["img_aug"] = self.aug_transform(img)
```

If `item["img"]` is a PIL Image (not a tensor), the `else` branch applies augmentation transforms but **never normalizes** the result. The clean view `img` would have been normalized by `img_transform` in the base dataset, but `img_aug` would remain in [0, 1] or PIL format. This creates a clean/augmented distribution mismatch within the same training step. While `ContrastiveSegDataset` currently returns tensors, this branch is a latent bug for any alternative base dataset.

---

### New Issue F: Config YAML Contains Stale Keys [MINOR]
**Location:** `configs/semantic_adapter_baseline.yaml:41-42`

```yaml
losses:
  ...
  temp_student: 0.1
  temp_teacher: 0.07
```

These temperature parameters no longer exist in `dino_distillation_loss()` (the function signature was simplified to `(student_feat, teacher_feat)`). The YAML is not only unused but now factually incorrect.

---

## Overall Recommendation

**⚠️ Major Revision**

The implementation has materially improved since the initial review. The three blocking critical issues (invalid distillation loss, missing normalization, zero-gradient cluster loss) are resolved. Suffix-based freezing, deterministic CUDA, and dead code removal demonstrate attention to detail.

However, the following block a training recommendation:

1. **Checkpoint loading safety is incomplete.** The `_load_state_checked` function must assert that all model adapter keys exist in the checkpoint when `require_lora=True`. Without this, random adapter initialization at inference is still possible.
2. **The augmented view pipeline corrupts the input distribution** by passing an already-normalized tensor through `ToPILImage()`, which clamps tails. This undermines the cross-view consistency loss.
3. **EMA remains a no-op** for trainable parameters because `adapt_ema=False`, and the test suite misleadingly validates this behavior.
4. **Depth loss degeneracy, batch size, and epoch count** were not addressed and remain major training risks.

The architecture and parameter counts are still correct. With the two critical fixes above (checkpoint validation and augmentation pipeline), the code would be **Borderline → Weak Accept** for an experimental training run. Until then, **Major Revision**.

---

## Required Changes Before Training (if any)

### Must-Fix (Blocking)

1. **Fix `_load_state_checked` to assert all model adapter keys are present in the checkpoint when `require_lora=True`.**
   ```python
   if require_lora:
       model_adapter_keys = {k for k in module.state_dict() if is_adapter_key(k)}
       ckpt_adapter_keys = {k for k in state_dict if is_adapter_key(k)}
       missing_adapters = model_adapter_keys - ckpt_adapter_keys
       if missing_adapters:
           raise RuntimeError(f"Adapter checkpoint missing keys: {missing_adapters}")
   ```

2. **Fix `AdapterTrainingDataset.__getitem__` to avoid denormalizing a normalized tensor.**
   - Option A: Remove `T.ToPILImage()` and apply tensor-native augmentations directly.
   - Option B: Restructure so augmentation happens on [0, 1] tensors *before* normalization.

3. **Add missing/unexpected key logging to inference backbone and segment loading** (`generate_semantic_pseudolabels_adapted.py:256,278-280`).

### Should-Fix (Strongly Recommended)

4. **Decide on EMA behavior.** Either set `adapt_ema=True` so the EMA head tracks adapted parameters (and update `ema_update` to handle shape mismatches gracefully), or remove the EMA update call entirely and document that CAUSE-TR's EMA head is frozen.
5. **Increase default batch size** to at least 32, or add gradient accumulation to an effective batch size of ≥64.
6. **Extend default epochs** to 50–100, or implement Warmup + Cosine with a longer horizon.
7. **Replace or fix the depth correlation loss** to preserve depth magnitude (remove `F.normalize` on depth, or replace with a depth-aware contrastive formulation).
8. **Strengthen augmentations** with `RandomResizedCrop` and `RandomSolarization`.
9. **Delete or load the YAML config.** If keeping it, remove stale `temp_student`/`temp_teacher` keys and add `yaml.safe_load` at the start of `main()`.
10. **Rewrite `test_ema_update_correctness`** to reflect actual training conditions (frozen base weights, optional adapter EMA tracking).

---

*Re-verification completed. The code is improved but not yet ready for a reliable training run.*
