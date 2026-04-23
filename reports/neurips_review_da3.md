# NeurIPS Review: Depth Anything V3 DoRA Adapter Implementation

**Paper ID:** [Redacted for blind review]  
**Title:** MBPS: Unsupervised Panoptic Segmentation via Depth-Guided Pseudo-Labels  
**Section Reviewed:** Stage 1 — DA3 DoRA Adapter Architecture & Training  
**Reviewer Type:** Neural Network Surgery / Custom API Adaptation Specialist  
**Rating:** 🚫 **Reject** — Multiple critical correctness bugs, fragile generic injection, and questionable scientific motivation.

---

## 1. Summary

This submission proposes integrating Depth Anything V3 (DA3) into an unsupervised panoptic segmentation pipeline by injecting DoRA adapters into its ViT encoder via a "generic fallback" injection strategy. The authors claim DA3 achieves PQ=27.37 (the highest among all depth models tested) and that ~1.2–1.5M adapter parameters can be trained self-supervised in Stage 1 to further improve pseudo-label quality.

**Regrettably, the implementation is not ready for publication.** It contains at least **two critical algorithmic bugs** that invalidate the self-supervised training signal, a **dangerous silent model fallback** that can corrupt experiments, fragile generic injection logic that misunderstands module hierarchy semantics, and a glaring inconsistency between the training and inference adapter injection paths for DepthPro. The claim that adapter training is "low-risk and high-reward" is directly contradicted by the evidence: frozen DA3 already outperforms all adapted competitors, and the proposed training losses are either broken or too conservative to produce meaningful adaptation.

---

## 2. Strengths

- **DoRA layer implementation is mechanically correct.** The `DoRALinear` class in `lora_layers.py` correctly implements weight decomposition with row-wise L2 normalization, magnitude parameters, and low-rank updates. The `V_norm.detach()` trick for gradient stability is sound.
- **Structured injection for HF-style models is well-engineered.** The `inject_lora_into_depthpro()` and structured DA2 paths show understanding of HuggingFace module hierarchies. Tiered adaptation (early Q/V only, late full) is a reasonable heuristic.
- **Test suite covers basic architecture invariants.** The mock-based tests verify adapter presence/absence, teacher-student separation, and checkpoint roundtrips.

---

## 3. Weaknesses

- **Self-supervised training losses are broken.** The relative depth ranking loss derives its targets from the *student* output, not the teacher, making it a tautology that provides no learning signal.
- **Inference pipeline is broken for DepthPro.** `generate_instance_pseudolabels_adapted.py` unconditionally calls `inject_lora_into_depth_model()` for all model types, bypassing the correct `inject_lora_into_depthpro()` path entirely.
- **Generic injection is semantically unsound.** Grouping layers by `".".join(name.split(".")[:-1])` and calling the result a "block" is an approximation so coarse it may merge layers from adjacent blocks or split a single block across multiple groups.
- **Silent fallback from DA3 to DA2 on ImportError is a data integrity disaster.** An experiment intended to study DA3 can silently run with DA2, invalidating all downstream ablations and comparisons.
- **No validation loop.** Training runs for 10 epochs with only training loss monitoring, providing no early detection of training divergence or overfitting.
- **Scientific justification is weak.** Frozen DA3 already achieves the highest PQ reported (27.37). The adapter training is motivated by a hoped-for improvement that is neither quantified nor isolated in ablations.

---

## 4. Specific Issues

### 4.1 Generic Injection Correctness

**[CRITICAL] `_inject_generic_vit` treats parent modules as "blocks" without verifying they are actually blocks.**

The function collects all `nn.Linear` modules matching `"attn"` or `"mlp"`, groups them by their immediate parent, sorts the parent names via natural sort, and calls each parent a "block." This is catastrophically fragile:

- **What if a block has no `nn.Linear` children?** E.g., a block with only normalization layers or custom CUDA ops. The block is *invisible* to the generic walker, shifting all subsequent block indices by −1. Block 7 becomes "block 6" in the tiering logic, causing full adaptation to start one block too early.
- **What if attention projections live in a *sub-sub-module*?** E.g., `block.attn.qkv` where `attn` is itself a container with `qkv` as a child. The parent is `block.attn`, not `block`. Two blocks `block_0.attn` and `block_1.attn` sort correctly, but if `block_0` also contains `block_0.mlp` as a separate parent, the "block" count doubles. The tiering `block_idx >= late_block_start` now applies to half-blocks.
- **What if MLP layers from block N are grouped with attention layers from block N+1?** If DA3 uses a flattened naming scheme like `transformer.layers.0.attn.proj` and `transformer.layers.0.mlp.fc1`, grouping by parent works. But if the naming is non-uniform (e.g., some blocks use `ffn` instead of `mlp`), the block count and ordering become nonsense.
- **The spec claims: "For ViTs, this typically respects depth ordering."** "Typically" is not good enough for research code. The authors provide no empirical validation that DA3's `named_modules()` enumeration order actually correlates with depth. PyTorch's `named_modules()` uses `yield from` DFS traversal; module registration order matters. If DA3 registers blocks dynamically or uses a `ModuleDict`, the order can be arbitrary.

**[MAJOR] Natural sort handles multiple numeric components correctly but the grouping semantics are still broken.**

The natural sort `re.split(r'(\d+)', s)` correctly handles `layer_1_block_2` vs `layer_1_block_10`. However, this is the *least* of the problems. Consider:
- `encoder.blocks.0.attn.qkv` → parent: `encoder.blocks.0.attn`
- `encoder.blocks.0.mlp.fc1` → parent: `encoder.blocks.0.mlp`

These are TWO "blocks" in the generic walker's view. Block index 0 gets only QKV, and block index 1 (which is actually the MLP of block 0) gets the full late-block treatment. The tiering philosophy is destroyed by naming conventions.

**[MINOR] The `_find_encoder_blocks` candidate list omits common wrappers.**

The function checks `model.encoder`, `model.blocks`, `model.vit.blocks`, `model.backbone.blocks`, `model.encoder.layer`, `model.backbone.encoder.layer`. It does NOT check:
- `model.model.blocks` (common in wrapper classes)
- `model.vision_model.encoder.layers` (CLIP-style naming)
- `model.transformer.blocks` (Swin-style naming)

If DA3 refactors to use any of these, the code silently falls back to generic injection, compounding the problems above.

---

### 4.2 DA3 Custom API Risks

**[CRITICAL] `load_dav3_model()` silently falls back to DA2 on ImportError.**

```python
def load_dav3_model(...):
    try:
        from depth_anything_3.api import DepthAnything3
        ...
    except ImportError:
        logger.error("depth_anything_3 not installed. Falling back to DA2.")
        return load_da2_model(device=device)
```

This is an unforgivable design choice. A researcher running an ablation labeled "DA3" will silently get DA2 outputs if the environment is misconfigured. The resulting PQ numbers, training curves, and adapter weights are *meaningless* for the stated claim. **The correct behavior is to raise a hard exception.**

**[CRITICAL] `inference_batch()` is called twice per training step with no guarantee of determinism or gradient support.**

```python
with torch.no_grad():
    teacher_out = teacher_model.inference_batch(img)
student_out = model.inference_batch(img)
```

The spec claims: "Both student and teacher share the same model.inference_batch() call." **They do not.** The training code calls `inference_batch()` on two *different* model objects. If `inference_batch()` contains:
- Random augmentations (resize jitter, random crops)
- Non-deterministic CUDA kernels
- Stateful behavior (running statistics, cached feature banks)
- Internal `torch.no_grad()` contexts that drop gradient history

...then the student and teacher receive *different* effective inputs or the student path may not backpropagate at all. The authors have not inspected the DA3 source code to verify any of these assumptions.

**[MAJOR] Batch size 4 with `inference_batch` on full-resolution images (512×1024) is a memory bomb.**

The training script defaults to `image_size=(512, 1024)` and `batch_size=4` with no gradient checkpointing. DA3's `inference_batch` likely processes at native resolution internally. A DINOv3-Large forward pass at 512×1024 with batch size 4 can exceed 24GB VRAM. There is no automatic mixed precision (AMP) usage, no gradient checkpointing, and no resolution downsampling during training. The script will OOM on consumer GPUs and many cloud instances.

**[MAJOR] The custom `inference_batch` API may bypass adapted layers entirely.**

`inference_batch` is a high-level method that may:
1. Call `model.eval()` internally
2. Use custom CUDA extensions for attention that operate on raw weight pointers
3. Reconstruct the model graph from a TorchScript / ONNX representation
4. Apply fused kernels that directly reference `nn.Linear.weight` buffers, ignoring the `DoRALinear` wrapper's `forward()` method

If any of these occur, the adapters are **decorative** — they exist in the module tree but are never invoked. The training loss will still decrease (MSE against a teacher that also bypasses adapters ≈ MSE against itself), but the adapters receive no gradients and learn nothing. The authors provide no forward-hook verification or output-difference check to confirm adapters are actually executed.

---

### 4.3 Self-Supervised Loss Issues

**[CRITICAL] Relative depth ranking loss uses student output as its own target.**

```python
def relative_depth_ranking_loss(depth_pred, num_pairs=1024):
    ...
    d_i = flat[b, idx_i]
    d_j = flat[b, idx_j]
    target = torch.sign(d_i.detach() - d_j.detach())  # ← STUDENT self-target!
    l = F.margin_ranking_loss(d_i, d_j, target, margin=margin)
```

In the training loop: `l_rank = relative_depth_ranking_loss(student_out)`. The ranking target is derived from `student_out` itself. The loss is literally trying to make the student consistent with its own rankings. Since `d_i` and `d_j` are the *same tensors* used to compute the target, the margin ranking loss becomes a no-op (the "teacher" and "student" are identical). **This loss contributes zero valid learning signal.** It should use `teacher_out` for the ranking targets.

**[MAJOR] MSE distillation against frozen teacher is a suffocating constraint.**

```python
self_distillation_loss(student_out, teacher_out):
    return F.mse_loss(student_out, teacher_out.detach())
```

With weight 1.0, this loss dominates the objective. The optimal solution for the student is to output *exactly* the teacher's depth map. Any domain-specific adaptation that shifts depth values (e.g., adapting to Cityscapes camera geometry) is penalized. The adapter is incentivized to be a **no-op**, not a domain adapter. The authors should report the actual magnitude of depth shifts learned by the adapters; if the MSE is near-zero, the adapters are useless.

**[MAJOR] Scale-invariant loss computes `torch.log(pred + 1e-6)` without guarding against non-positive values.**

```python
def scale_invariant_loss(pred, target, lambda_si=0.5):
    diff = torch.log(pred + 1e-6) - torch.log(target.detach() + 1e-6)
```

Depth Anything models output relative depth that can legitimately be zero (background, sky). `log(1e-6) = -13.8`, but the gradient `1/(pred + 1e-6)` explodes when `pred` approaches zero. A single outlier pixel can produce a gradient spike that destabilizes adapter training. Proper implementations clamp pred to `[eps, inf)` before log and optionally use a robust loss (Huber, Charbonnier).

**[MINOR] Ranking loss samples `num_pairs // B` per batch element with no deduplication.**

```python
idx_i = torch.randint(0, H * W, (num_pairs // B,), device=device)
idx_j = torch.randint(0, H * W, (num_pairs // B,), device=device)
```

It is possible (and likely) that `idx_i == idx_j` for some pairs, producing `target = 0` and a degenerate margin ranking loss. There is no check to exclude identical indices or duplicate pairs.

---

### 4.4 Training Pipeline

**[MAJOR] No validation loop, no early stopping, no metric-based checkpointing.**

The training loop saves checkpoints based on training loss only:
```python
if avg_total < best_loss:
    best_loss = avg_total
    ... # save best.pt
```

Training loss for self-supervised adapter training is **uninformative**: MSE against teacher will always decrease, and the broken ranking loss is tautological. There is no validation on a held-out set, no depth-quality metric (e.g., δ-threshold, RMSE on a small labeled subset), and no sanity check that adapted depth actually produces better instance boundaries. The "best" checkpoint may be overfitted or diverged.

**[MINOR] Gradient clipping at `max_norm=1.0` is applied per accumulation step, not per optimizer step.**

```python
if step % grad_accum_steps == 0:
    torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
    optimizer.step()
```

This is actually correct (clipping before the step), but if `grad_accum_steps > 1`, the accumulated gradients are clipped *after* averaging by `loss_total / grad_accum_steps`. The effective gradient scale depends on accumulation steps in a non-obvious way. Document this.

**[MINOR] `copy.deepcopy(model)` on a 307M-parameter model before adapter injection is memory-inefficient.**

The teacher model is created via `copy.deepcopy(model)`. For DA3, this duplicates the entire model in CPU RAM before moving to GPU. A more efficient pattern is to load the teacher from the same checkpoint path or use `torch.cuda.empty_cache()` between operations. For a TPU or memory-constrained environment, this can cause host OOM.

---

### 4.5 Robustness & Failure Modes

**[MAJOR] If structured block discovery accidentally succeeds on DA3, CAUSE-style qkv fusion may be applied to incompatible layers.**

```python
# In inject_lora_into_depth_model:
target_groups = []
# Group 1: CAUSE-style (fused qkv)
target_groups.append([("attn.qkv", "qkv"), ("attn.proj", "proj"), ...])
# Group 2: HF-style (separate Q,K,V)
target_groups.append([("attention.attention.query", "query"), ...])
```

The code tries CAUSE-style first, then HF-style. If DA3 happens to have a module named `attn.qkv` (even if it's not a standard fused projection), the code "succeeds" on Group 1 and breaks. There is no architecture fingerprinting — just blind path probing. A safer approach is to fingerprint the exact attention class type (e.g., `isinstance(module, CustomDA3Attention)`) before deciding which projection pattern to use.

**[MAJOR] `freeze_non_adapter_params` uses brittle substring matching.**

```python
def freeze_non_adapter_params(model):
    for name, param in model.named_parameters():
        if any(k in name for k in ("lora_", "dwconv", "conv_gate")):
            param.requires_grad = True
        else:
            param.requires_grad = False
```

If the base DA3 model has a parameter named `depth_normalization_lora_param` (hypothetically), it would be incorrectly unfrozen. Conversely, if a future adapter variant renames `lora_A` to `adapter_A`, all parameters would be frozen and training would silently fail (zero gradients, no error). Parameter freezing should be type-based (`isinstance(module, (DoRALinear, LoRALinear, ...))`) or explicitly whitelist adapter class instances.

**[MINOR] `_adapt_depth_decoder` probes decoder paths blindly.**

```python
decoder_paths = ["head", "decoder", "depth_head", "prediction_head"]
```

For DA3, the decoder may not be at any of these top-level attributes. The function will silently skip decoder adaptation without warning. If `adapt_decoder=True` is passed for DA3, the user is not informed that the decoder was not adapted.

---

### 4.6 Inference Pipeline

**[MAJOR] τ=0.03 is overfitted to a specific DA3 checkpoint and dataset.**

The spec states τ=0.03 is "empirically optimal" from "March 2026 ablations." This is a single-grid-search result on Cityscapes with a specific DA3 checkpoint. There is no evidence this transfers to:
- Different DA3 checkpoints (e.g., DA3MONO-SMALL, DA3STEREO variants)
- Different datasets (KITTI, COCO, Mapillary)
- Different resolutions (the ablation is at 512×1024)
- Different adapter ranks or training durations

The threshold should be validated per-configuration or the authors should report sensitivity analysis.

**[MAJOR] Inference script unconditionally calls `inject_lora_into_depth_model` for ALL model types, breaking DepthPro.**

In `generate_instance_pseudolabels_adapted.py`:
```python
inject_lora_into_depth_model(model, variant="dora", rank=4, alpha=4.0)
```

This is called regardless of `model_type`. For `depthpro`, the correct function is `inject_lora_into_depthpro()`. Using the generic function on DepthPro will either:
- Fail to find encoder blocks and fall back to generic injection (wrong architecture, wrong layers)
- Partially match `model.encoder.layer` if it exists at the top level, adapting the wrong module

This is a **regression bug** — the training script correctly branches on `model_type` for injection, but the inference script does not.

**[MINOR] Sobel + CC post-processing is identical for all depth models despite different depth qualities.**

The `sobel_cc_instances` function uses fixed `grad_threshold=0.20` by default (overridden to 0.03 for DA3), `min_area=1000`, and `dilation_iters=3` for all models. DA3's sharper boundaries might benefit from:
- Different blur sigma (smaller σ preserves more detail)
- Different CC connectivity (8-connectivity vs 4-connectivity)
- Area thresholds scaled by dataset resolution

The authors do not justify using the *same* post-processing pipeline with only a threshold change.

---

### 4.7 Reproducibility

**[CRITICAL] DA3 is not on HuggingFace and has no version pinning.**

The model is loaded via a custom package:
```python
from depth_anything_3.api import DepthAnything3
model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")
```

There is no `requirements.txt` pinning `depth_anything_3==X.Y.Z`, no Docker image, and no checksum verification of the downloaded checkpoint. If the DA3 authors update their API (e.g., rename `inference_batch` to `predict`, change internal module names, or release a new checkpoint at the same identifier), this codebase will break silently or produce different results. **This is not reproducible research.**

**[MAJOR] No validation during training means divergence is undetectable.**

Without a validation set and depth-quality metrics, the only signal is training loss, which is uninformative for self-supervised distillation. The adapters could collapse to zero (producing teacher-identical outputs) or explode to noise, and the training loop would not distinguish these outcomes.

---

### 4.8 Comparison & Scientific Motivation

**[MAJOR] Adapter training justification is absent given frozen DA3 already wins.**

The spec states: "DA3 achieves PQ=27.37, the highest among all depth models." But the adapter training is motivated by the assumption that adapters will *improve* upon this. Where is the ablation showing **frozen DA3 vs. adapted DA3**? If frozen DA3 at PQ=27.37 is already the best, and adapter training risks destabilizing it (see MSE distillation critique above), the entire adapter sub-project may be negative-utility.

The training cost (~1.5M params × 10 epochs × full-resolution forward passes) is non-trivial. The authors must demonstrate:
1. Frozen DA3 baseline PQ
2. Adapted DA3 PQ with these exact losses
3. Ablations removing each loss component

Absent these numbers, the adapter architecture is an unmotivated engineering addition.

---

## 5. Questions for the Authors

1. **Have you verified that `model.inference_batch()` actually calls the wrapped `DoRALinear.forward()` methods?** A simple forward-hook or output-diff test (adapter weights = 0 vs. adapter weights = random) would suffice. If `inference_batch` uses fused kernels or JIT compilation, the adapters may be dead code.

2. **What is the actual parameter count for your specific DA3 checkpoint?** "~1.2–1.5M" is a 25% variance. Report the exact count from `count_adapter_params()` after injection.

3. **Does `DepthAnything3.inference_batch()` contain any `torch.no_grad()` guards or non-deterministic preprocessing?** If so, how do you guarantee gradient flow and teacher-student input consistency?

4. **Why does the ranking loss use `student_out` as its own ranking target?** Is this a bug, or is there a theoretical justification for self-consistency ranking that we missed?

5. **Where is the frozen DA3 vs. adapted DA3 ablation?** If adapters do not improve over the frozen baseline, why include them in the pipeline at all?

6. **How do you ensure reproducibility across `depth_anything_3` package versions?** The package is not on PyPI or HuggingFace. What is the installation source, commit hash, and checksum?

---

## 6. Recommendation

**🚫 Reject.**

This work requires fundamental fixes before it can be considered for NeurIPS:

1. **Fix the ranking loss** to use teacher outputs as targets.
2. **Replace the silent DA3→DA2 fallback** with a hard error.
3. **Fix the inference script** to branch on `model_type` and call the correct injection function (`inject_lora_into_depthpro` for DepthPro).
4. **Validate that `inference_batch` executes adapter `forward()`** via output-difference tests or forward hooks.
5. **Add a validation loop** with depth-quality metrics (δ<1.25, RMSE, or at least instance PQ on a small labeled validation set).
6. **Report the frozen DA3 baseline** and prove adapters improve upon it.
7. **Version-pin all non-standard dependencies** (especially `depth_anything_3`) and provide installation instructions with commit hashes.
8. **Make `freeze_non_adapter_params` robust** by checking module types rather than parameter name substrings.

After these changes, a re-review may be warranted. In its current state, the implementation contains too many correctness bugs and too little empirical validation to support the paper's claims.

---

## 7. Required Changes (Checklist for Authors)

- [ ] **Fix ranking loss targets** — use `teacher_out` for pairwise depth ordering, not `student_out`.
- [ ] **Remove silent fallback** in `load_dav3_model()` — raise `ImportError` instead of returning DA2.
- [ ] **Fix inference injection** — `generate_instance_pseudolabels_adapted.py` must branch on `model_type` like the training script does.
- [ ] **Add adapter execution verification** — test that `inference_batch` output changes when adapter weights are perturbed.
- [ ] **Add validation loop** — at minimum, log depth-quality metrics every N steps; ideally evaluate pseudo-label PQ on a validation set.
- [ ] **Report frozen-vs-adapted ablation** — demonstrate that adapter training improves upon frozen DA3.
- [ ] **Pin `depth_anything_3` version** — include exact commit hash or wheel URL in dependencies.
- [ ] **Harden parameter freezing** — use `isinstance(module, (DoRALinear, LoRALinear, ConvDoRALinear))` instead of string matching.
- [ ] **Document memory requirements** — test and report max VRAM usage for batch_size=4 at 512×1024.
- [ ] **Guard against log(0) in SI loss** — use `torch.clamp(pred, min=1e-6)` before `torch.log`.

---

*Review compiled by: Brutal NeurIPS Reviewer (Custom API & NN Surgery)*  
*Date: 2026-04-24*  
*Verdict: Major revision required. Do not accept in current form.*
