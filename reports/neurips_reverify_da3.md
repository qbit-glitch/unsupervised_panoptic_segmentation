# NeurIPS Re-Verification: Depth Anything V3 DoRA Adapter Implementation

**Previous Review:** reports/neurips_review_da3.md (Rated: 🚫 Reject)  
**Reviewer:** Neural Network Surgery / Custom API Adaptation Specialist  
**Date:** 2026-04-24

---

## Executive Summary

The authors addressed **6 of 23 identified issues** from the previous audit. The two most critical algorithmic bugs—the ranking loss self-target tautology and the silent DA3→DA2 fallback—are fixed. The inference script now correctly branches on `model_type`, and `copy.deepcopy` has been eliminated. Parameter freezing is marginally more robust (suffix matching instead of substring matching).

However, **the majority of critical and major issues remain unfixed**, including the semantically broken generic injection logic, the complete absence of validation, the memory-bomb training configuration, and the lack of adapter execution verification. Several new concerns have emerged from the fixes, most notably tying image augmentation exclusively to the distillation loss presence and adding a tautological test that provides no actual coverage.

**Verdict: Still not acceptable for NeurIPS.** The fixes are patch-level band-aids on a fundamentally fragile architecture. The generic injection strategy remains catastrophically unsound, and the training pipeline still lacks any mechanism to detect divergence or confirm that adapters are actually being executed.

---

## Issue-by-Issue Verification

### Issue 1: `_inject_generic_vit` treats parents as blocks
**Previous Severity:** CRITICAL  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# depth_adapter.py:225-228
for name, module in candidates:
    parent = ".".join(name.split(".")[:-1])
    if parent not in block_groups:
        block_groups[parent] = []
    block_groups[parent].append((name, module))
```

The grouping logic is **completely unchanged**. The same catastrophic fragilities remain:
- A block with no `nn.Linear` children is invisible, shifting all subsequent block indices.
- If attention projections live in `block.attn.qkv`, the parent is `block.attn`, not `block`. Two "blocks" (`block_0.attn` and `block_0.mlp`) are created from one actual transformer block, destroying the tiering philosophy.
- MLP layers from block N can be grouped separately from attention layers from the same block.

**Remaining Concerns:** The authors did not even attempt to fix this. The generic injection is still an approximation so coarse it may merge layers from adjacent blocks or split real blocks. For research code claiming to support "any ViT," this is unacceptable.

---

### Issue 2: Natural sort handles multiple numeric components
**Previous Severity:** MAJOR  
**Status:** ⚠️ PARTIALLY FIXED

**Evidence:**
```python
# depth_adapter.py:231-234
sorted_parents = sorted(
    block_groups.keys(),
    key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)],
)
```

Natural sort is still present and mechanically correct. However, as the previous review stated, this is the *least* of the problems. Sorting parent names correctly does not fix the fact that the parents themselves may not correspond to actual transformer blocks. The sort correctly orders `layer_1_block_2` before `layer_1_block_10`, but if `layer_1_block_2.mlp` and `layer_1_block_2.attn` are separate parents, the "block" count doubles and tiering becomes meaningless.

**Remaining Concerns:** Natural sort is a cosmetic fix on a broken grouping strategy.

---

### Issue 3: `_find_encoder_blocks` omits common wrappers
**Previous Severity:** MINOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# depth_adapter.py:34-41
candidates = [
    "encoder",
    "blocks",
    "vit.blocks",
    "backbone.blocks",
    "encoder.layer",
    "backbone.encoder.layer",  # HF Depth Anything V2
]
```

Still missing:
- `model.model.blocks` (common in wrapper classes)
- `model.vision_model.encoder.layers` (CLIP-style naming)
- `model.transformer.blocks` (Swin-style naming)

**Remaining Concerns:** If DA3 refactors to any of these paths, the code silently falls back to the broken generic injection. No expansion of the candidate list was attempted.

---

### Issue 4: Silent DA3→DA2 fallback on ImportError
**Previous Severity:** CRITICAL  
**Status:** ✅ FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:65-69
def load_dav3_model(model_name="depth-anything/DA3MONO-LARGE", device="cpu"):
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=torch.device(device))
    return model
```

The `try/except ImportError` block and the `return load_da2_model(...)` fallback have been **completely removed**. If `depth_anything_3` is not installed, the script now raises a hard `ImportError` and crashes immediately. This is the correct behavior.

**Remaining Concerns:** None. This fix is clean and complete.

---

### Issue 5: `inference_batch` called twice per step
**Previous Severity:** CRITICAL  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:229-244
with torch.no_grad():
    if model_type == "dav3":
        teacher_out = teacher_model.inference_batch(img)
    ...
student_input = img_aug if img_aug is not None else img
if model_type == "dav3":
    student_out = model.inference_batch(student_input)
    ...
```

The training code still calls `inference_batch()` on two **different** model objects (`teacher_model` and `model`). The review's concerns about random augmentations, non-deterministic CUDA kernels, and internal `torch.no_grad()` contexts remain entirely unaddressed. The authors have not added:
- Forward-hook verification
- Output-diff checks between teacher and student with zeroed adapters
- Any inspection of the DA3 `inference_batch` source to verify gradient safety

**Remaining Concerns:** If `inference_batch` contains resize jitter, random crops, or fused kernels that bypass `DoRALinear.forward()`, the adapters are dead code. The authors still provide **zero empirical evidence** that adapters are actually executed during the forward pass.

---

### Issue 6: Batch size 4 with full-resolution images is a memory bomb
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:315, 327
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--image_size", type=int, nargs=2, default=[512, 1024])
```

No automatic mixed precision (AMP), no gradient checkpointing, no resolution downsampling, and no memory-profiling documentation. A DINOv3-Large forward at 512×1024 with batch size 4 will still OOM on most consumer GPUs. The default configuration remains reckless.

**Remaining Concerns:** The training script is still not runnable on standard hardware without manual tuning. The authors should at least document expected VRAM usage.

---

### Issue 7: `inference_batch` may bypass adapted layers entirely
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

No forward-hook verification, output-difference tests, or gradient-flow checks have been added to the codebase. The test suite (`test_depth_adapters.py`) verifies structural injection but does **not** verify that a forward pass through the model actually invokes the adapter `forward()` methods.

**Remaining Concerns:** The adapters could be decorative. If DA3 uses custom CUDA extensions, TorchScript, or fused attention kernels that operate on raw weight pointers, the `DoRALinear` wrapper's `forward()` is never called. The training loss would still decrease (MSE against teacher), but the adapters would learn nothing. This is a **reproducibility and correctness crisis** that remains completely unaddressed.

---

### Issue 8: Ranking loss uses student output as its own target
**Previous Severity:** CRITICAL  
**Status:** ✅ FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:102-134
def relative_depth_ranking_loss(student_depth, teacher_depth, num_pairs=1024, margin=0.1):
    ...
    t_i = teacher_flat[b, idx_i]
    t_j = teacher_flat[b, idx_j]
    # Teacher defines the ground-truth ordering
    target = torch.sign(t_i - t_j)
    ...
    l = F.margin_ranking_loss(s_i[valid], s_j[valid], target[valid], margin=margin)
```

The function signature now accepts **both** `student_depth` and `teacher_depth`. The target is derived from `teacher_flat`, not `student_flat`. The call site (line 255) correctly passes both tensors:
```python
l_rank = relative_depth_ranking_loss(student_out, teacher_out)
```

**Remaining Concerns:** None. This fix is algorithmically correct.

---

### Issue 9: MSE distillation against frozen teacher is a suffocating constraint
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:94-99
def self_distillation_loss(student_out, teacher_out, mask=None):
    ...
    return F.mse_loss(student_out, teacher_out.detach())
```

```python
# train_depth_adapter_lora.py:250
w = loss_weights.get("distillation", 1.0)
```

MSE weight is still **1.0 by default**. The review's concern—that the optimal solution for the student is to output exactly the teacher's depth map, penalizing any domain-specific adaptation—remains valid. No ablation or evidence has been added to show that adapters learn meaningful shifts rather than converging to a no-op.

**Remaining Concerns:** The MSE distillation still dominates and may suffocate domain adaptation. The authors should either reduce the default weight or provide empirical evidence that meaningful adaptation occurs despite MSE=1.0.

---

### Issue 10: `log(0)` in scale-invariant loss
**Previous Severity:** MAJOR  
**Status:** ✅ FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:137-144
def scale_invariant_loss(pred, target, lambda_si=0.5, min_depth=1e-3):
    pred_clamped = torch.clamp(pred, min=min_depth)
    target_clamped = torch.clamp(target.detach(), min=min_depth)
    diff = torch.log(pred_clamped) - torch.log(target_clamped)
```

The `pred` and `target` tensors are now explicitly clamped to `min_depth=1e-3` before `torch.log()`. This prevents `log(0)` and mitigates gradient explosion near zero.

**Remaining Concerns:** `min_depth=1e-3` is somewhat aggressive for normalized depth maps (clipping ~0.1% of the dynamic range). A robust loss (Huber, Charbonnier) would be preferable to hard clamping, but the critical correctness issue is resolved.

---

### Issue 11: Ranking loss samples self-pairs
**Previous Severity:** MINOR  
**Status:** ✅ FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:115-121
idx_i = torch.randint(0, H * W, (num_pairs // B,), device=device)
idx_j = torch.randint(0, H * W, (num_pairs // B,), device=device)
# Ensure no self-pairs
mask_same = idx_i == idx_j
while mask_same.any():
    idx_j[mask_same] = torch.randint(0, H * W, (mask_same.sum(),), device=device)
    mask_same = idx_i == idx_j
```

Self-pairs are now explicitly avoided via a resampling loop. Additionally, pairs where the teacher has equal depth (`target == 0`) are filtered out before computing the loss.

**Remaining Concerns:** In the extreme case where `num_pairs // B > H * W` (more pairs than pixels), the while loop could theoretically take many iterations. For the default settings this is not a practical concern.

---

### Issue 12: No validation loop, no early stopping
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:194-301
# Entire training function — no validation dataloader, no val metrics.
```

The training loop still saves checkpoints based on training loss only:
```python
if avg_total < best_loss:
    best_loss = avg_total
    ... # save best.pt
```

No validation set, no depth-quality metrics (δ<1.25, RMSE), and no sanity check that adapted depth produces better instance boundaries. The "best" checkpoint selection remains uninformative.

**Remaining Concerns:** Self-supervised training without validation is flying blind. The authors must add at least a held-out validation set with depth-quality metrics or instance PQ evaluation.

---

### Issue 13: Gradient clipping applied per accumulation step
**Previous Severity:** MINOR  
**Status:** ⚠️ PARTIALLY FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:275-278
if step % grad_accum_steps == 0:
    torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
```

The clipping logic is mechanically unchanged (still correct—clip before step). However, the review explicitly asked for **documentation** of the interaction between accumulation steps and clipping. No comment, docstring, or README note has been added.

**Remaining Concerns:** Users may not realize that effective gradient scale depends on `grad_accum_steps` in a non-obvious way. Document this behavior.

---

### Issue 14: `copy.deepcopy` on 307M-parameter model
**Previous Severity:** MINOR  
**Status:** ✅ FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:371-378
if args.model_type == "dav3":
    teacher_model = load_dav3_model(model_name, device="cpu")
elif args.model_type == "da2":
    teacher_model = load_da2_model(model_name, device="cpu")
elif args.model_type == "depthpro":
    teacher_model = load_depthpro_model(model_name, device="cpu")
```

The `copy.deepcopy(model)` pattern has been **completely replaced** with fresh model loading from the same checkpoint path. This eliminates the CPU RAM spike and is the correct pattern.

**Remaining Concerns:** Loading the teacher from scratch assumes deterministic checkpoint loading. If the model loader has any stochastic initialization paths, the teacher and student base weights could theoretically diverge. For standard pretrained models this is not a practical concern.

---

### Issue 15: CAUSE-style vs HF-style greedy first-match
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# depth_adapter.py:106-163
target_groups = []
# Group 1: CAUSE-style (fused qkv)
...
# Group 2: HF-style (separate Q,K,V)
...
for group in target_groups:
    ...
    if group_found > 0 and attn_found:
        found_any = True
        break  # Stop after first successful group
```

The code still tries CAUSE-style first, then HF-style, and breaks after the first match. There is still no architecture fingerprinting (e.g., `isinstance(module, CustomDA3Attention)`). If DA3 happens to have a module named `attn.qkv` that is not a standard fused projection, the code "succeeds" on the wrong group.

**Remaining Concerns:** Blind path probing is fragile. The authors should fingerprint the exact attention class type before deciding which projection pattern to use.

---

### Issue 16: `freeze_non_adapter_params` uses brittle substring matching
**Previous Severity:** MAJOR  
**Status:** ⚠️ PARTIALLY FIXED

**Evidence:**
```python
# lora_layers.py:297-309
def freeze_non_adapter_params(model: nn.Module) -> None:
    ADAPTER_SUFFIXES = (".lora_A", ".lora_B", ".lora_magnitude", ".dwconv.weight", ".conv_gate",
                        ".lora_A.weight", ".lora_B.weight")
    for name, param in model.named_parameters():
        if any(name.endswith(suffix) for suffix in ADAPTER_SUFFIXES):
            param.requires_grad = True
        else:
            param.requires_grad = False
```

The previous substring matching (`any(k in name for k in ...)`) has been replaced with **suffix matching** (`name.endswith(suffix)`). This eliminates false positives like `depth_normalization_lora_param` (which contains `lora_` but does not end with a suffix).

However, the review explicitly recommended **type-based checking**: `isinstance(module, (DoRALinear, LoRALinear, ConvDoRALinear))`. The current implementation still relies on naming conventions. If a future adapter variant renames `lora_A` to `adapter_A`, all parameters would be frozen and training would silently fail.

**Remaining Concerns:** Suffix matching is better than substring matching, but it is still string-based and therefore brittle. The recommended type-based approach was not adopted.

---

### Issue 17: `_adapt_depth_decoder` blind probing
**Previous Severity:** MINOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# depth_adapter.py:298-327
decoder_paths = [
    "head", "decoder", "depth_head", "prediction_head",
]
for path in decoder_paths:
    module = getattr(model, path, None)
    if module is None:
        continue
    ...
```

If `adapt_decoder=True` is passed but none of these paths exist, the function silently skips without logging a warning. The user is not informed that decoder adaptation was requested but not applied.

**Remaining Concerns:** Add a warning when `adapt_decoder=True` but no decoder layers are found.

---

### Issue 18: τ=0.03 claimed as optimal without sensitivity analysis
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# generate_instance_pseudolabels_adapted.py:178
parser.add_argument("--tau", type=float, default=0.20)
```

The default changed from 0.03 to 0.20, but there is still no sensitivity analysis, no grid search documentation, and no evidence that this threshold transfers across:
- Different DA3 checkpoints (DA3MONO-SMALL, DA3STEREO)
- Different datasets (KITTI, COCO, Mapillary)
- Different resolutions
- Different adapter ranks or training durations

**Remaining Concerns:** A single threshold value without sensitivity analysis is not scientifically justified. The authors must either validate per-configuration or report a sensitivity curve.

---

### Issue 19: Inference script unconditionally calls `inject_lora_into_depth_model`
**Previous Severity:** MAJOR  
**Status:** ✅ FIXED

**Evidence:**
```python
# generate_instance_pseudolabels_adapted.py:91-94
if model_type == "depthpro":
    inject_lora_into_depthpro(model, variant="dora", rank=4, alpha=4.0)
else:
    inject_lora_into_depth_model(model, variant="dora", rank=4, alpha=4.0)
```

The inference script now correctly branches on `model_type`, calling `inject_lora_into_depthpro()` for DepthPro and `inject_lora_into_depth_model()` for DA2/DA3. The regression that broke DepthPro inference has been resolved.

**Remaining Concerns:** None. This fix is clean.

---

### Issue 20: Sobel + CC post-processing identical for all models
**Previous Severity:** MINOR  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# sobel_cc.py:10-13
def sobel_cc_instances(semantic, depth, thing_ids=THING_IDS,
                       grad_threshold=0.03, min_area=1000,
                       dilation_iters=3, depth_blur_sigma=1.0,
                       features=None):
```

The post-processing pipeline still uses the same fixed parameters for all depth models. DA3's sharper boundaries might benefit from different blur sigma, CC connectivity, or area thresholds scaled by resolution. No per-model tuning or justification has been added.

**Remaining Concerns:** Using identical post-processing with only a threshold change ignores the fundamentally different depth qualities produced by DA2, DA3, and DepthPro.

---

### Issue 21: DA3 not version-pinned
**Previous Severity:** CRITICAL  
**Status:** ❌ NOT FIXED

**Evidence:**
```python
# train_depth_adapter_lora.py:66
from depth_anything_3.api import DepthAnything3
model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")
```

There is still no `requirements.txt` pinning `depth_anything_3==X.Y.Z`, no Docker image, and no checksum verification. If the DA3 authors update their API or checkpoint, this codebase will break or produce different results.

**Remaining Concerns:** This is not reproducible research. The authors must provide exact commit hashes, wheel URLs, or Docker images.

---

### Issue 22: No validation during training
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

This is the same as Issue 12. No validation dataloader, no depth-quality metrics, and no early stopping have been added.

**Remaining Concerns:** The training pipeline remains a black box with no quality control.

---

### Issue 23: Frozen DA3 already wins—no ablation
**Previous Severity:** MAJOR  
**Status:** ❌ NOT FIXED

No frozen-vs-adapted ablation has been added to the codebase. The authors still have not demonstrated that adapter training improves upon the frozen DA3 baseline (PQ=27.37).

**Remaining Concerns:** Without this ablation, the adapter architecture is an unmotivated engineering addition consuming non-trivial compute for unproven benefit.

---

## New Issues Introduced by Fixes

### New Issue A: Image augmentation tied exclusively to distillation loss
**Severity:** MAJOR  
**Evidence:**
```python
# train_depth_adapter_lora.py:401
dataset = DepthAdapterDataset(
    args.data_dir, image_size=tuple(args.image_size), augment="distillation" in args.losses,
)
```

Augmentation is **only enabled when `"distillation"` is in the loss list**. If a user trains with only `ranking` and `scale_invariant` losses (a perfectly valid configuration), `img_aug` will be `None`, and both teacher and student will see identical inputs. The ranking loss specifically benefits from augmentation (student learns to match teacher's ordering under photometric distortions), but this configuration disables it. This is a **design regression**: the augmentation should be independent of loss composition, or at least enabled for ranking as well.

---

### New Issue B: `test_late_block_start_defaults` is a tautology
**Severity:** MINOR  
**Evidence:**
```python
# tests/test_depth_adapters.py:478-484
def test_late_block_start_defaults():
    assert 18 == 18, "Default late_block_start for 24-block models should be 18"
    assert 6 == 6, "Default late_block_start for 12-block models should be 6"
```

This test asserts that constants equal themselves. It does **not** test the actual default logic in `main()`:
```python
if args.late_block_start == 6:
    if args.model_type in ("da2", "depthpro"):
        args.late_block_start = 18
```

The test provides **zero actual coverage** and gives a false sense of security. A proper test would instantiate the argparse parser and verify the mutation logic.

---

### New Issue C: No test coverage for inference script branching
**Severity:** MAJOR  

The critical fix for Issue 19 (inference script branching on `model_type`) has **no test coverage**. The test suite does not import or exercise `generate_instance_pseudolabels_adapted.py` at all. A future refactor could easily re-introduce the unconditional `inject_lora_into_depth_model` call, and no test would catch it.

---

### New Issue D: Test suite does not verify adapter execution in forward pass
**Severity:** CRITICAL  

Despite the previous review explicitly requesting "forward-hook or output-diff verification" (Questions for Authors #1), the test suite contains **no test** that confirms:
1. A forward pass through an adapted model produces different outputs than the frozen base model.
2. Gradients flow back through adapter parameters during backpropagation.
3. `inference_batch` (or any model-specific API) actually invokes the adapter wrappers.

The tests verify structural presence (`isinstance(module, DoRALinear)`) but not functional execution. This is the most important missing test.

---

### New Issue E: Device loading inconsistency between model loaders
**Severity:** MINOR  
**Evidence:**
```python
# train_depth_adapter_lora.py:65-69
def load_dav3_model(..., device="cpu"):
    model = model.to(device=torch.device(device))  # Explicit conversion

# train_depth_adapter_lora.py:72-78
def load_da2_model(..., device="cpu"):
    model = model.to(device)  # No conversion
```

`load_dav3_model` wraps `device` in `torch.device()`, while `load_da2_model` and `load_depthpro_model` do not. This is a minor API inconsistency. If a `torch.device` object is passed to the latter functions, it works (PyTorch accepts both), but the inconsistency is sloppy.

---

### New Issue F: Potential infinite loop in ranking loss (theoretical)
**Severity:** MINOR  
**Evidence:**
```python
# train_depth_adapter_lora.py:118-121
while mask_same.any():
    idx_j[mask_same] = torch.randint(0, H * W, (mask_same.sum(),), device=device)
    mask_same = idx_i == idx_j
```

If `num_pairs // B > H * W` (more sampled pairs than pixels in the image), this loop could theoretically iterate many times or, in pathological cases, become a near-infinite loop. For the default settings (`num_pairs=1024`, `H*W=524,288`), this is not a practical concern, but the function does not guard against it.

---

## Overall Recommendation

**🚫 Reject — Major Revision Still Required.**

The authors fixed the two most embarrassing bugs (ranking loss tautology and silent model fallback) and patched the inference script branching. These are necessary but not sufficient improvements. The implementation still contains:

1. **Catastrophically fragile generic injection** that misunderstands module hierarchy.
2. **No validation loop**, making training divergence undetectable.
3. **No adapter execution verification**, leaving open the possibility that adapters are dead code.
4. **No frozen-vs-adapted ablation**, leaving the scientific motivation unproven.
5. **No version pinning**, making reproduction impossible.
6. **New design regressions**, including tying augmentation to a specific loss and adding a tautological test.

The code is closer to correctness but still far from NeurIPS standards for reproducibility and robustness.

---

## Required Changes Before Training (if any)

### Must Fix (Blocking)
- [ ] **Add adapter execution verification test** — Forward-pass an adapted mock DA3 model and assert output changes when adapter weights are perturbed. Assert gradients flow to `lora_A`/`lora_B`.
- [ ] **Fix or eliminate generic injection** — Either fingerprint DA3's exact module hierarchy or replace `_inject_generic_vit` with a DA3-specific injector. The current generic fallback is not acceptable for research.
- [ ] **Add validation loop** — At minimum, evaluate depth-quality metrics (δ<1.25, RMSE, or instance PQ) on a held-out validation set every N epochs.
- [ ] **Decouple augmentation from loss composition** — Enable augmentation for all training modes that use student-teacher distillation/ranking, not just when `"distillation"` is in the loss list.

### Should Fix (Strongly Recommended)
- [ ] **Replace `test_late_block_start_defaults` with a real test** — Actually invoke the argparse default logic and verify mutation.
- [ ] **Add test for inference script branching** — Verify `generate_instance_pseudolabels_adapted.py` calls the correct injection function per `model_type`.
- [ ] **Harden parameter freezing to type-based** — Use `isinstance(module, (DoRALinear, LoRALinear, ConvDoRALinear))` instead of name suffix matching.
- [ ] **Add AMP or gradient checkpointing** — Document and support lower-memory training configurations.
- [ ] **Version-pin `depth_anything_3`** — Provide exact commit hash, wheel URL, or Docker image.
- [ ] **Add frozen-vs-adapted ablation** — Demonstrate that adapters improve upon frozen DA3, or remove the adapter claim.
- [ ] **Add sensitivity analysis for τ** — Report PQ as a function of threshold, not a single magic number.
- [ ] **Document gradient clipping + accumulation interaction** — Add a code comment explaining the effective gradient scale.

### Nice to Have
- [ ] **Expand `_find_encoder_blocks` candidates** — Add `model.model.blocks`, `model.vision_model.encoder.layers`, `model.transformer.blocks`.
- [ ] **Add warning in `_adapt_depth_decoder`** — Log when `adapt_decoder=True` but no decoder is found.
- [ ] **Add per-model Sobel+CC tuning** — Justify or tune post-processing parameters per depth model.
- [ ] **Standardize device handling** in model loaders.

---

*Review compiled by: Brutal NeurIPS Reviewer (Custom API & NN Surgery)*  
*Date: 2026-04-24*  
*Verdict: Major revision still required. Do not accept in current form.*
