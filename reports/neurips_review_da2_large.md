# NeurIPS Review: Depth Anything V2 Large DoRA Adapter Implementation

**Paper ID:** MBPS-DA2-2026  
**Reviewer:** External - Depth Estimation & Self-Supervised Adaptation  
**Rating:** 3 (Reject — Major revision required)  
**Confidence:** 5/5 (Expert in monocular depth estimation, domain adaptation, and parameter-efficient fine-tuning)

---

## Summary

This work proposes to adapt Depth Anything V2 Large (DA2-Large, ~300M parameters) for unsupervised panoptic segmentation pseudo-label generation by injecting tiered DoRA adapters into the DINOv2-Large encoder. The student-teacher self-distillation framework uses MSE distillation, a pairwise ranking loss, and a scale-invariant consistency loss to adapt ~829K parameters (0.28% of total) while keeping the DPT decoder frozen. The adapted model generates instance pseudo-labels via Sobel-gradient connected components.

While the high-level idea is sound, the implementation contains **multiple critical bugs, architectural mismatches between specification and code, and fundamental conceptual errors in the self-supervised objective** that would prevent reliable reproduction and likely degrade pseudo-label quality below the unadapted baseline. The code reads as an engineering draft, not research-grade software. I cannot recommend acceptance without a complete overhaul.

---

## Strengths

1. **Correct DoRA formulation.** The `DoRALinear` implementation (row-wise L2 normalization, `.detach()` on directional norm, magnitude initialization from `||W||`) is mathematically faithful to Liu et al. (ICML 2024).
2. **Structured encoder discovery.** `_find_encoder_blocks()` correctly discovers HF-style DA2 via `model.backbone.encoder.layer`, and the fallback generic injection handles non-standard architectures gracefully.
3. **Parameter count is exact.** The ~829K count for HF DINOv2-Large (dim=1024, MLP=4096, r=4, α=4.0) is correct: 18×2×9,216 + 6×4×9,216 + 6×24,576 + 6×21,504 = 829,440.
4. **Tiered rationale is physically motivated.** Restricting early blocks to Q+V and late blocks to full attention+MLP is a defensible design choice grounded in representation theory.

---

## Weaknesses

The weaknesses are severe and span architecture, training dynamics, loss design, and inference. Several issues are **show-stoppers** that would cause silent training failure or produce worse-than-baseline pseudo-labels.

---

## Specific Issues

### 1. Architecture Correctness

**[CRITICAL]** `late_block_start` default is catastrophically wrong for DA2-Large.
- The specification declares `late_block_start=18` for DA2-Large (24 blocks: 18 early, 6 late).
- The training script (`train_depth_adapter_lora.py`, line 303) sets `default=6`.
- The config file (`configs/depth_adapter_baseline.yaml`, line 15) also sets `late_block_start: 6`.
- **Consequence:** A user running the "baseline" adapts 18 blocks fully and only 6 blocks lightly—the **exact opposite** of the intended tiering. Parameter count balloons, overfitting risk increases, and the early-layer texture prior is destroyed. This is a specification-to-code consistency failure of the highest order.

**[MAJOR]** CAUSE-style paths are checked before HF-style paths per block.
- In `inject_lora_into_depth_model()`, the loop appends the CAUSE-style `target_groups` first, then HF-style. It breaks after the first group with `attn_found=True`.
- For HF DA2, CAUSE-style fails (no `attn.qkv`), so HF-style is used. This is fine in isolation.
- **However**, if a future model variant has *both* `attn.qkv` (fused) and `attention.attention.query` (unfused) in the same block—e.g., during a architecture migration—the CAUSE-style group would take precedence and silently miss the HF path. The logic should be architecture-aware, not greedy-first-match.

**[MINOR]** `_m_init_norm` buffer in `DoRALinear` is dead code.
- Registered at line 100 but never referenced in `forward()`. It wastes a negligible amount of memory but signals incomplete implementation.

**[CRITICAL]** `ConvDoRALinear._spatial_dims` is never set.
- The depthwise convolution path (lines 151–167) is gated on `h_p is not None`, but `_spatial_dims` defaults to `(None, None)` and is never populated by the injection code.
- **Consequence:** The `conv_dora` variant silently degenerates to plain DoRA. All claims about "Conv-DoRA activation-space refinement" are unsubstantiated by the code. The spatial conv path is unreachable dead code.

---

### 2. Teacher-Student Setup

**[CRITICAL]** `img_aug` is computed but **never used** in the training loop.
- `DepthAdapterDataset.__getitem__()` (line 167–176) computes `img_aug` with ColorJitter and RandomHorizontalFlip.
- `train_depth_adapter()` (line 208) reads `batch["img"]` but **never touches `batch["img_aug"]`**.
- **Consequence:** The student receives the **exact same input** as the frozen teacher. The entire self-supervised consistency objective collapses to trivial identity reproduction. There is no view invariance, no augmentation robustness, and no meaningful regularization beyond a tiny MSE penalty. This is not self-supervised learning; it is a very expensive no-op.

**[MAJOR]** The `inputs` dict is reused between teacher and student for HF models.
- Lines 216–224: `inputs = processor(...)` is created inside the teacher's `torch.no_grad()` block, then fed to **both** `teacher_model(**inputs)` and `model(**inputs)`.
- While tensor reuse is not a correctness bug per se, it couples the teacher and student preprocessing pipelines in a way that prevents future extensions (e.g., different resolutions, different processor configs). More importantly, combined with the missing augmentation bug, it means the student has **zero input diversity** across the entire training run.

**[MAJOR]** `copy.deepcopy()` on a 300M-parameter GPU model is memory-inefficient and risky.
- `load_da2_model()` places the model on `device` (potentially GPU). `copy.deepcopy(model)` (line 344) duplicates the full model on the same device, doubling GPU memory usage (~1.2 GB → ~2.4 GB for parameters alone, plus activations).
- **Risk:** Silent OOM on consumer GPUs (e.g., RTX 3090 with 24GB) when batch size is already tight.
- **Better:** Load the teacher from scratch with `AutoModelForDepthEstimation.from_pretrained()` or deep-copy on CPU then `.to(device)`.
- **Additional risk:** `copy.deepcopy` may not properly replicate HF Transformers' internal hooks, cached attributes, or gradient checkpointing state. The code does not verify structural equivalence post-copy.

**[MINOR]** `teacher_model = teacher_model.to(device)` (line 348) is redundant.
- The teacher was already placed on `device` by `copy.deepcopy` of a device-resident model. This line is harmless but reveals sloppiness.

---

### 3. Self-Supervised Losses for RELATIVE Depth

**[CRITICAL]** Ranking loss is **self-consistency, not teacher-guided ranking**.
- `relative_depth_ranking_loss()` (lines 103–125) computes:
  ```python
  target = torch.sign(d_i.detach() - d_j.detach())
  ```
  where `d_i` and `d_j` are sampled from `depth_pred` — the **student's own output**.
- The docstring claims it "penalizes when the predicted depth order contradicts the **teacher's** order." This is false.
- **Consequence:** The ranking loss enforces nothing more than self-consistency (the student should agree with itself). It provides **zero distillation signal** from the teacher. A randomly initialized student would satisfy this loss trivially by being consistent with its own predictions. The loss is conceptually broken.
- **Fix:** The target must be `torch.sign(teacher_flat[b, idx_i] - teacher_flat[b, idx_j])`.

**[CRITICAL]** Scale-invariant loss is **excluded by default**.
- The specification (Section 4) states: "Scale-invariant losses are **critical** during adapter training" and "**Essential** — DA2 has no absolute scale."
- Yet `main()` sets `--losses` default to `"distillation,ranking"` (line 305), omitting `"scale_invariant"`.
- The config file (`depth_adapter_baseline.yaml`, line 34) also lists only `distillation` and `ranking`.
- **Consequence:** Out-of-the-box training runs with **only MSE + broken ranking**, missing the one loss the authors themselves claim is essential for relative depth. MSE on relative depth is scale-sensitive and will penalize valid scale shifts that the frozen DPT decoder might induce.

**[MAJOR]** Ranking loss samples only 1,024 pairs per batch.
- For a 512×1024 image (524,288 pixels), 1,024 pairs covers **0.2%** of all possible pairs.
- Worse, pairs are sampled **with replacement** and can be the **same pixel** (`idx_i == idx_j`). When `d_i == d_j`, `target=0`, and `F.margin_ranking_loss` with `target=0` returns `margin=0.1` unconditionally. Same-pixel pairs always contribute positive loss regardless of correctness.
- **Consequence:** The ranking loss is both woefully undersampled and contaminated by self-pair artifacts. It is statistical noise, not a useful training signal.

**[MAJOR]** No scale-matching or affine-invariance loss.
- DA2 outputs **relative depth** with arbitrary global scale. The student and teacher share the same frozen decoder, but encoder adaptation can still induce relative scale drift.
- MSE heavily penalizes any scale mismatch. The SI loss (when enabled) only handles log-space shifts. There is no explicit affine-invariant loss (e.g., Pearson correlation, normalized cross-correlation, or the affine-invariant loss from MiDaS/DA2 training) to handle both scale and shift.
- **Consequence:** The student is forced to stay unnaturally close to the teacher's exact depth values, limiting the extent of domain adaptation. The adapter cannot learn to rescale depth to better match target-domain boundary statistics.

**[MINOR]** Numerical stability in `scale_invariant_loss`.
- `torch.log(pred + 1e-6)` can produce large negative values when `pred` is near zero. For relative depth maps that can span [0, ~1], `log(1e-6) = -13.8`. A single near-zero pixel dominates the loss. A more robust formulation uses `log(pred + 1)` or clips values before log.

---

### 4. Training Dynamics

**[MAJOR]** 10 epochs is almost certainly insufficient for domain adaptation.
- Cityscapes train has ~2,975 images. At batch_size=4, this is ~744 steps/epoch. Ten epochs = **7,440 total optimization steps**.
- Adapting a 300M-parameter foundation model to a new visual domain (driving scenes, different camera geometry, distinct texture distributions) with only 829K adapters typically requires **20–50+ epochs** or a much larger unlabeled corpus.
- Cosine annealing with `T_max=epochs` (line 197) drives LR to near-zero by step 7,440. With `lr=1e-4`, the model spends half its training below 5e-5. This is far too conservative for adapter warm-up.
- **No learning rate warmup.** AdamW on freshly initialized LoRA/DoRA matrices (Kaiming A, zero B) without warmup can cause initial instability and loss spikes.

**[MAJOR]** 0.28% trainable ratio may be too low for meaningful domain adaptation.
- While parameter efficiency is desirable, 829K parameters to adapt a 300M model across 24 ViT blocks for depth boundary refinement is aggressive. For comparison, standard ViT-LoRA for image classification often uses r=8–16 and adapts all blocks. Here r=4 with selective block adaptation may lack representational capacity to shift the encoder's domain-specific feature distributions meaningfully.
- The authors provide **zero ablation** on rank, block coverage, or trainable ratio. There is no evidence that 0.28% is sufficient or optimal.

**[QUESTION]** `adapt_decoder=False` — is this truly optimal?
- The specification claims the decoder must remain frozen for "zero-shot depth prior stability."
- However, the DPT decoder contains learnable reassembly and fusion layers that are not domain-agnostic. If the encoder adapters shift feature distributions, a frozen decoder may misinterpret them. The authors should ablate `adapt_decoder=True` (even with a tiny adapter on the final fusion layer) to justify this design choice.

**[MINOR]** Gradient clipping at `max_norm=1.0` may be overly aggressive.
- For adapter training where gradients are naturally small (due to frozen backbone), clipping at 1.0 is harmless but may suppress legitimate large updates during early training. A dynamic or higher threshold (e.g., 5.0) is more common in LoRA literature.

**[MINOR]** No validation set, no depth metrics.
- The training loop logs only total loss. There is no validation split, no δ1 threshold accuracy, no RMSE, no edge-F1, and no pseudo-label PQ computed during training. The "best" checkpoint is selected by training loss alone, which is meaningless for relative depth when the loss itself is mis-specified (broken ranking + missing SI).

---

### 5. Inference & Post-Processing

**[CRITICAL]** Sobel threshold default (`grad_threshold=0.20`) contradicts the specification's "optimal" value (`τ=0.03`).
- `sobel_cc_instances()` (line 11) defaults to `grad_threshold=0.20`.
- The architecture spec (Section 5) repeatedly states τ=0.03 is "Cityscapes optimal" and shows a table with PQ_things=20.20 at τ=0.03.
- `ablate_instance_methods.py` (line 120) configures sobel_cc with `grad_threshold: 0.03`, but this is a runtime override. The **function default** is 0.20.
- **Consequence:** Any caller that omits the threshold (e.g., `sobel_cc_instances(sem, depth)`) silently uses τ=0.20, which is **6.7× higher** than the optimal value. This will miss fine-grained boundaries and under-segment instances. The discrepancy between specification, config, and code default is a reproducibility hazard.

**[MAJOR]** `min_area=1000` is too large for Cityscapes at 1024×512.
- 1000 pixels = 0.19% of the image. Distant pedestrians, cyclists, and traffic signs can easily fall below this threshold.
- The architecture spec's diagram (line 304) claims "Filter by minimum area (typically 50–200 pixels)," yet the table (line 329) and function default both use 1000.
- **Consequence:** Small thing-class instances are systematically filtered out, biasing PQ_things upward for large objects and downward for the long tail. The authors should justify 1000 or provide a size-dependent threshold.

**[MAJOR]** Sobel on relative depth lacks per-image normalization.
- Relative depth maps have **arbitrary global scale** per image. The gradient magnitude distribution varies wildly between a close-up street scene and a highway vista.
- A fixed global threshold (τ=0.03 or τ=0.20) cannot generalize across these scale variations. Standard practice is to normalize gradients by the image's depth range (max–min) or use percentile-based adaptive thresholds.
- **Consequence:** The threshold is effectively a hyperparameter that must be re-tuned per dataset, per scene type, and potentially per image. The claim of "Cityscapes optimal τ=0.03" is likely only valid for the specific depth range distribution of the adapted model, not the raw DA2 output.

**[MINOR]** `depth_blur_sigma=1.0` is applied unconditionally.
- Gaussian smoothing before Sobel reduces noise but also blurs thin object boundaries (e.g., pedestrian arms, bicycle frames). For high-resolution depth maps, σ=1.0 may be excessive. No ablation is shown.

---

### 6. Comparison to DA3

**[MAJOR]** The justification for adapting DA2 over DA3 is weak.
- The prompt notes DA3 achieves PQ=27.37 vs DA2's PQ=26.5. The spec claims DA2 beats DA3 on COCO (14.04 vs 13.76) but admits DA3 wins on Cityscapes (20.90 vs 20.20 for PQ_things).
- The code supports all three models (`dav3`, `da2`, `depthpro`), but the **default config targets DepthPro**, not DA2. The DA2 adapter code appears to be a secondary path.
- **Question:** If DA3 is better on the primary benchmark (Cityscapes PQ) and the code already supports it, why invest engineering effort in DA2 adapters? The COCO argument is dataset-specific and does not generalize. The spec should provide a stronger justification (e.g., DA2 inference speed, DA3 licensing, DA2 availability at higher resolutions) rather than a single dataset metric.

**[QUESTION]** Is the DA2-COCO result with or without adapters?
- The spec claims "DA2-Large beats DA3 on COCO (14.04 vs 13.76)." It is unclear whether the DA2 number is the **zero-shot baseline** or the **adapter-refined** result. If it is the zero-shot baseline, then adapter training is irrelevant to the comparison. If it is the adapted result, the improvement over zero-shot must be quantified.

---

### 7. Common Failure Modes

**[MAJOR]** Hardcoded HF model ID with no graceful fallback.
- `load_da2_model()` hardcodes `"depth-anything/Depth-Anything-V2-Large-hf"`. If HuggingFace renames the repo, deprecates the checkpoint, or experiences downtime, the code crashes with no local caching logic or alternative source.
- **Fix:** Support local path fallback (`os.path.exists(model_name)`), cache directory specification, and `snapshot_download` pre-checking.

**[MAJOR]** No shape validation for `predicted_depth`.
- The code assumes `.predicted_depth` returns `(B, H, W)`. HF Depth Anything models currently do this, but the API contract is not guaranteed.
- If a future update returns `(B, 1, H, W)`, `relative_depth_ranking_loss()` will crash with "too many values to unpack" at `B, H, W = depth_pred.shape`.
- **Fix:** Assert or squeeze the tensor shape before loss computation.

**[MAJOR]** Inefficient and lossy tensor→PIL→tensor roundtrip in training loop.
- Lines 216–217: `img` (already a torch tensor [0,1]) is converted to uint8 numpy, wrapped in PIL, then passed to the HF processor which converts back to tensor.
- This is **extremely slow** (4 PIL creations per batch) and **lossy** (quantization to uint8, double resize).
- The dataset already resizes to 512×1024. The processor then resizes to the model's expected input size (518×518 for DINOv2 patch 14). Double bilinear resize degrades fine boundaries.
- **Fix:** Return raw PIL images from the dataset and let the processor handle resize/normalize in one step. Or use the processor's tensor path directly.

**[MINOR]** No deterministic CUDA settings.
- `set_seed()` sets Python, NumPy, and PyTorch seeds but does **not** set `torch.backends.cudnn.deterministic = True` or `torch.backends.cudnn.benchmark = False`. Reproducibility across GPU architectures is not guaranteed.

**[MINOR]** `freeze_non_adapter_params()` uses brittle string matching.
- Line 297: `if any(k in name for k in ("lora_", "dwconv", "conv_gate"))`.
- If a pretrained parameter name happens to contain these substrings (e.g., a base model layer named `conv_gate_fusion`), it would be incorrectly unfrozen. This is unlikely but not impossible.
- **Fix:** Use a dedicated `is_adapter_param` flag or registry instead of string grepping.

**[MINOR]** `inject_lora_into_depth_model` does not return the actual parameter count for `ConvDoRA`.
- Because `_spatial_dims` is never set, the conv path is dead, so the effective parameter count of `conv_dora` equals `dora`. The returned count from `trainable_count()` includes the unused `dwconv` weights, inflating the reported parameter count without corresponding compute.

---

## Questions for the Authors

1. **Why is the default `late_block_start=6` when the specification explicitly requires 18 for DA2-Large?** Was this a copy-paste error from DA3/DepthPro configs? How do you ensure users do not accidentally train with inverted tiering?

2. **The ranking loss uses the student's own predictions as targets.** Was this intentional (e.g., as a self-consistency regularizer alongside augmentation), or is it simply a bug? If intentional, why frame it as "teacher-guided ranking" in the docstring and specification?

3. **Why is scale-invariant loss omitted from the default configuration** despite being labeled "essential" in the architecture document? What empirical result justifies this omission?

4. **The `img_aug` field is computed but never consumed.** Was augmentation ever tested? What are the ablation results with and without ColorJitter/RandomFlip?

5. **ConvDoRA's depthwise convolution is unreachable.** Was ConvDoRA ever benchmarked, or does it always fall back to standard DoRA? If the latter, why expose it as a CLI option?

6. **What is the PQ_things improvement attributable solely to DA2 adapter training** versus the zero-shot DA2 baseline? The spec shows DA2-Large at PQ_things=20.20 but does not break out the adapter contribution.

7. **Have you validated `τ=0.03` on the adapted model's depth output, or only on zero-shot DA2?** If adapter training changes the depth scale distribution, the optimal threshold may shift.

---

## Required Changes

Before this work can be considered for any venue, the authors **must**:

1. **Fix the ranking loss** to use teacher depth for targets, not student self-predictions.
2. **Include scale-invariant loss in the default loss configuration** and justify its weighting empirically.
3. **Use `img_aug` in the training loop** or remove augmentation from the dataset. If removed, explain why input diversity is unnecessary.
4. **Correct the `late_block_start` default** for DA2-Large to 18, and add architecture-specific defaults (DA2: 18, DA3: 6, DepthPro: 6) with clear documentation.
5. **Align `sobel_cc_instances` defaults** with the validated ablation values (τ=0.03, min_area=1000 for Cityscapes), or remove the defaults and require explicit caller specification.
6. **Implement per-image gradient normalization** (e.g., divide by depth range) for the Sobel threshold, or provide evidence that a fixed threshold generalizes.
7. **Add a validation split** with depth metrics (δ1, RMSE) and pseudo-label PQ to enable model selection based on something other than training loss.
8. **Either implement `_spatial_dims` injection for ConvDoRA** or remove the `conv_dora` option to prevent silent degradation.
9. **Eliminate the tensor→PIL→tensor roundtrip** in the training loop; pass PIL images directly from the dataset to the processor.
10. **Provide ablations** on: (a) rank, (b) trainable ratio, (c) decoder adaptation, (d) number of epochs, and (e) augmentation strategy.

---

## Recommendation

**Reject — Major Revision Required.**

The underlying idea (tiered DoRA adapters for relative depth adaptation) is reasonable, but the implementation is riddled with critical bugs, specification-to-code mismatches, and missing empirical validation. The ranking loss is conceptually broken, the default configuration omits the essential scale-invariant term, and the augmentation pipeline is completely disconnected from training. A 10-epoch run with these defaults would likely produce pseudo-labels **worse** than the unadapted zero-shot baseline.

I encourage the authors to address the required changes, run controlled ablations, and resubmit. The codebase shows promise but currently fails the basic standard of research-grade reproducibility.
