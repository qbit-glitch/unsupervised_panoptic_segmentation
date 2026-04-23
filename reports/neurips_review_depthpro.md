# NeurIPS Review: Apple DepthPro DoRA Adapter for Unsupervised Instance Pseudo-Label Generation

**Reviewer:** Depth Estimation & Self-Supervised Adaptation Specialist  
**Rating:** ⭐ Reject (Major Revision Required)  
**Confidence:** 5 / 5 — I have read every line of code and the architecture spec.

---

## Summary

The submission proposes to adapt Apple DepthPro for unsupervised instance pseudo-label generation using tiny DoRA adapters (~1.66M parameters, 0.17% of the model). The goal is to sharpen depth discontinuities at object boundaries by self-supervised distillation from a frozen teacher, augmented with a pairwise ranking loss and a scale-invariant log loss. While the parameter-efficiency narrative is appealing, the implementation is undermined by **critical conceptual errors in the loss functions**, **dangerous default hyperparameters that contradict the architecture specification**, **silent data-pipeline bugs**, and **severe memory inefficiencies**. In its current state, the code is unlikely to train stably, may silently fail, and shows no empirical validation that the resulting pseudo-labels improve downstream PQ_things. A fundamental redesign of the training objectives and data pipeline is required before this work can be considered viable.

---

## Strengths

1. **Parameter efficiency is well-calculated.** The architecture spec provides an exact parameter-count table (1,658,880 trainable parameters) and clearly motivates which layers are adapted and why.
2. **Modular injection code.** `depthpro_adapter.py` and `lora_layers.py` are cleanly structured, easy to test, and the DoRA implementation appears faithful to Liu et al. (ICML 2024).
3. **Tiered adaptation strategy is intellectually coherent.** Restricting early blocks to Q+V and late blocks to full adaptation aligns with the known low-level vs. high-level feature hierarchy of ViTs.
4. **Test coverage for injection logic.** `test_depth_adapters.py` verifies adapter placement, teacher-student separation, and checkpoint round-tripping for mock models.

---

## Weaknesses

1. **The ranking loss is mathematically broken for self-distillation.** It uses the *student’s own prediction* to define the target ordering, making it either a no-op or a self-amplifying instability. It provides no meaningful teacher signal.
2. **Default training hyperparameters directly contradict the architecture specification.** The training script defaults to `late_block_start=6`, which adapts early MLPs and triples the parameter count, violating the spec’s warning that early MLP adaptation causes “representation collapse in self-supervised settings.”
3. **Data augmentation is computed and then silently discarded.** The dataset returns augmented views (`img_aug`), but the training loop never uses them. This defeats regularization and suggests the authors never inspected the data flow.
4. **Teacher instantiation via `copy.deepcopy` on a 1B-parameter HF model is reckless.** It doubles memory, is known to be fragile with Transformers (shared weights, custom attributes), and will OOM on mid-range GPUs.
5. **No validation, no proxy metrics, no PQ_things numbers.** Training blindly minimizes a training loss that can decrease while depth quality degrades. There is no evidence the adapters help downstream segmentation.
6. **Inference thresholds are arbitrary and untested.** `grad_threshold=0.20`, `min_area=1000`, and `dilation_iters=3` are hardcoded without ablation or connection to Cityscapes object statistics.

---

## Specific Issues

### 1. [CRITICAL] Ranking loss compares the student to itself.
**Location:** `train_depth_adapter_lora.py:103-126`

```python
d_i = flat[b, idx_i]
d_j = flat[b, idx_j]
target = torch.sign(d_i.detach() - d_j.detach())
margin = 0.1
l = F.margin_ranking_loss(d_i, d_j, target, margin=margin)
```

`d_i` and `d_j` are sampled from **`depth_pred` (the student)**. The target is the sign of the *student’s own prediction*. This is not self-distillation; it is a self-consistency loss that either:
- Vanishes when `|d_i - d_j| > 0.1` (no gradient), or
- Amplifies existing differences without bound when `|d_i - d_j| < 0.1`.

There is **no teacher signal** here. The correct formulation must compare the student’s ordering to the **teacher’s** depth map, or the loss should be removed entirely. As written, it can destabilize training by pushing the student away from the teacher for no valid reason.

---

### 2. [CRITICAL] Training script default contradicts the architecture spec.
**Location:** `train_depth_adapter_lora.py:303`

The argparse default is `--late_block_start 6`. The architecture specification (`Architecture-DepthPro-Adapters.md`) and the unit tests both demand `late_block_start=18` for DepthPro. With the default value, the script adapts blocks 6–23 fully, injecting **~2.4M parameters per encoder** instead of the intended 0.83M. The spec explicitly states:

> “Adapting MLPs in early layers causes unstable training and representation collapse in self-supervised settings.”

Running the script with zero user overrides will therefore trigger the exact failure mode the authors claim to avoid. **The default must be removed or changed to 18.**

---

### 3. [CRITICAL] Augmented views are computed and then thrown away.
**Location:** `train_depth_adapter_lora.py:140-176` (dataset) vs. `train_depth_adapter_lora.py:206-224` (training loop)

`DepthAdapterDataset` returns both `img` and `img_aug` when `augment=True` (triggered whenever distillation is in the loss list). However, the training loop only consumes `batch["img"]`:

```python
img = batch["img"].to(device)
# ... teacher and student both operate on `img` ...
```

`img_aug` is never referenced. If the design intent is a BYOL/DINO-style asymmetric pipeline where the student sees augmentations and the teacher sees the clean image, the student branch is simply missing. If augmentations are unnecessary, the dataset should not waste compute generating them. **This bug suggests the training pipeline was never executed end-to-end with logging that inspects batch contents.**

---

### 4. [CRITICAL] Base model left in `train()` mode may corrupt frozen features.
**Location:** `train_depth_adapter_lora.py:201`

```python
model.train()
```

This sets **the entire model** to training mode. While DINOv2 blocks lack dropout, DepthPro’s decoder or FOV head may contain dropout or batch-norm layers. Any stochasticity in the frozen base network will inject noise into the teacher-quality features that the adapters are trying to refine. The correct pattern is:

```python
model.eval()
for m in model.modules():
    if isinstance(m, (LoRALinear, DoRALinear, ConvDoRALinear)):
        m.train()
```

Without this, reproducibility and feature stability are compromised.

---

### 5. [MAJOR] MSE distillation is inappropriate for metric depth.
**Location:** `train_depth_adapter_lora.py:95-100`

`F.mse_loss(student_out, teacher_out.detach())` penalizes absolute squared errors in meters. A 1m error at 100m depth contributes **10,000× more loss** than a 1m error at 1m. This heavily biases the student toward close-range accuracy and can suppress learning of far-field boundary discontinuities, which are equally important for instance segmentation. A **log-L1** or **relative L1** loss (`|pred - target| / target`) would be far more appropriate for metric depth.

---

### 6. [MAJOR] MSE and scale-invariant losses are in direct conflict.
**Location:** `train_depth_adapter_lora.py:228-244`

- **MSE (weight 1.0):** “Preserve the exact metric scale of the teacher.”
- **Scale-Invariant Log Loss (weight 0.5):** “Ignore global scale; only match shape.”

These objectives pull the student in opposite directions. Because the SI loss explicitly discards global scale, its gradient can shift the student’s depth distribution away from the teacher’s metric scale, while MSE fights to pull it back. With 10 epochs and no warmup, this conflict wastes optimizer capacity and can cause loss oscillation. **Either drop the SI loss (the frozen decoder already preserves metric scale) or down-weight it to ≤ 0.1.**

---

### 7. [MAJOR] Ranking loss hyperparameters are unjustified.
**Location:** `train_depth_adapter_lora.py:103-126`

- **`num_pairs=1024`** covers only **0.2%** of pixels in a 512×1024 image. Random uniform sampling will almost never hit the critical boundary pixels where depth ordering determines instance separation.
- **`margin=0.1`** is a fixed absolute threshold in **meters**. It is absurdly large for close objects (0.5m apart) and vanishingly small for far objects (100m apart). A **relative margin** (e.g., `margin = 0.05 * max(d_i, d_j)`) is required for metric depth.
- **Sampling with replacement** means identical pixels can be paired, producing `target = sign(0) = 0`, which is invalid input to `F.margin_ranking_loss` (expects `±1`).

---

### 8. [MAJOR] `copy.deepcopy` of a 1B-parameter HF model is dangerous and memory-inefficient.
**Location:** `train_depth_adapter_lora.py:344`

```python
teacher_model = copy.deepcopy(model)
```

This instantiates a second 1B-parameter model in memory (~4 GB in fp32, ~2 GB in fp16). For a 16 GB GPU, this is catastrophic. Worse, `copy.deepcopy` is known to mishandle HF Transformers models: shared weights become unshared, custom `__getstate__` hooks may be skipped, and attached `processor` objects (which contain Python callables) are deep-copied unnecessarily. **Recommended fix:** load the teacher from pretrained weights a second time, or save/load `state_dict` into a fresh instance.

---

### 9. [MAJOR] DoRA adapters clone base weights, wasting ~900 MB of GPU memory.
**Location:** `lora_layers.py:91`

```python
self.weight = nn.Parameter(wrapped.weight.data.clone(), requires_grad=False)
```

Every adapted layer stores a **duplicate** of its base weight matrix. With 144 adapted layers across two encoders (~226M parameters cloned), this wastes roughly **900 MB** in fp32. Since the original `nn.Linear` is replaced but kept alive inside the adapter, the clone is completely unnecessary. Referencing `wrapped.weight` directly would save nearly 1 GB, which is decisive for fitting the model on a single 16 GB or 24 GB GPU.

---

### 10. [MAJOR] No validation loop or convergence proxy.
**Location:** `train_depth_adapter_lora.py:183-281`

Training blindly minimizes the sum of three losses. Without ground-truth labels, the authors provide **no validation protocol whatsoever**: no depth boundary recall, no visual depth-map inspection, no held-out PQ_things evaluation. The “best” checkpoint is selected by training loss alone, which is meaningless because the broken ranking loss can decrease while the model diverges from the teacher. **At minimum, the code should log sample depth maps every epoch and compute a boundary F-score proxy against the teacher.**

---

### 11. [MAJOR] Sobel threshold `τ=0.20` is arbitrary and scale-dependent.
**Location:** `instance_methods/sobel_cc.py:27`

```python
depth_edges = grad_mag > grad_threshold  # grad_threshold=0.20
```

The threshold is applied to the gradient magnitude of the depth map. If depth is **metric** (meters), a gradient of 0.20 m/pixel is enormous and will miss all but the most violent depth discontinuities (e.g., a wall directly in front of the camera). If depth is **normalized** to [0, 1], the threshold is arbitrary and scene-dependent. The authors provide no ablation or dataset-specific justification. **This single hardcoded value could render the entire instance decomposition pipeline useless.**

---

### 12. [MAJOR] `min_area=1000` pixels at 512×1024 suppresses small instances.
**Location:** `instance_methods/sobel_cc.py:11`

At half-resolution Cityscapes (512×1024), 1000 pixels is ~0.2% of the image. Many **riders, bicycles, and distant pedestrians** are smaller than this. The Cityscapes validation set contains hundreds of thing instances below 1000 pixels. Filtering them out will directly and severely degrade PQ_things. The threshold should be scaled by image area or validated against the dataset’s instance-size distribution.

---

### 13. [MAJOR] Greedy dilation reclamation merges adjacent instances.
**Location:** `instance_methods/sobel_cc.py:49-52`

```python
dilated = ndimage.binary_dilation(cc_mask, iterations=dilation_iters)
reclaimed = dilated & cls_mask & ~assigned
final_mask = cc_mask | reclaimed
```

Instances are processed in **descending area order**. A large instance dilates first and reclaims boundary pixels before smaller neighbors get a chance. This creates a **systematic bias toward large instances stealing boundaries from small adjacent ones**, a classic failure mode for greedy connected-components pipelines.

---

### 14. [MAJOR] Silent failure if adapter injection misses layers.
**Location:** `models/adapters/depthpro_adapter.py:29-108`

If the HuggingFace `transformers` library updates its internal attribute names (e.g., `encoder.layer` → `encoder.layers`), `_inject_into_hf_dinov2` logs a `warning` and returns an empty dict. Training proceeds with **zero trainable parameters**, and the user may never notice because the script still runs and the loss still changes (due to the frozen teacher baseline). **This is a silent total failure mode.**

---

### 15. [MAJOR] `ConvDoRALinear` is non-functional without external spatial-dimension injection.
**Location:** `models/adapters/lora_layers.py:122-174`

The `ConvDoRALinear` class requires `_spatial_dims` to be set manually to activate its depthwise-convolution path. **No code in the reviewed repository sets this attribute.** If a user selects `variant="conv_dora"`, the model silently falls back to standard DoRA while carrying the extra (unused) conv parameters. This is misleading and wastes compute.

---

### 16. [MAJOR] Batch size 4 is too small for stable adaptation.
**Location:** `train_depth_adapter_lora.py:295`

With only 4 images per step and default `grad_accum_steps=1`, gradient variance is extremely high. The ranking loss, which samples only 256 pairs per image, is especially noisy. For 1.66M+ parameters, an effective batch size of at least 32–64 is advisable. The default should be `grad_accum_steps=8` or higher.

---

### 17. [MAJOR] Teacher inputs are wastefully reconstructed via PIL every forward pass.
**Location:** `train_depth_adapter_lora.py:216-218`

```python
inputs = processor(images=[Image.fromarray((i.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)) for i in img], return_tensors="pt")
```

For every batch, the code moves GPU tensors to CPU, casts to uint8, wraps in PIL, runs the processor (which often resizes and normalizes), and moves back to GPU. This is a **massive CPU bottleneck** and introduces quantization noise. The processor should accept raw tensors directly, or the dataset should cache `pixel_values`.

---

### 18. [MINOR] No warmup in the LR schedule.
**Location:** `train_depth_adapter_lora.py:197`

`CosineAnnealingLR` starts at full `lr=1e-4`. For DoRA magnitude parameters, large initial updates can destabilize the pretrained feature directions. A short linear warmup (e.g., 500 steps) is standard practice and cheap to add.

---

### 19. [MINOR] `margin_ranking_loss` receives invalid `target=0`.
**Location:** `train_depth_adapter_lora.py:121`

`torch.sign(0)` returns `0`. PyTorch’s `margin_ranking_loss` expects `target ∈ {−1, +1}`. When two sampled pixels have exactly equal depth (guaranteed to happen on flat surfaces or when sampling with replacement), the target is 0, producing undefined gradients.

---

### 20. [MINOR] `+1e-6` epsilon in log-space risks gradient explosion near zero.
**Location:** `train_depth_adapter_lora.py:130`

```python
diff = torch.log(pred + 1e-6) - torch.log(target.detach() + 1e-6)
```

If the adapted encoder pushes the decoder to output values near zero, the gradient `∂loss/∂pred = 1 / (pred + 1e-6)` can reach `1e6`. Clamping `pred` to a physically plausible minimum (e.g., `min_depth = 0.01` meters) before the log would be safer.

---

### 21. [MINOR] Missing mixed-precision training support.
**Location:** `train_depth_adapter_lora.py:183-281`

No `torch.cuda.amp.autocast` context is used. For a 1B-parameter model, mixed precision reduces memory by ~40% and speeds up training significantly. The adapter `forward` methods already include `.to(x.dtype)` casts, so adding autocast is trivial.

---

### 22. [QUESTION] Is the FOV encoder truly irrelevant for boundary quality?
**Location:** Architecture spec, Section 2

The authors freeze the FOV encoder, claiming it “only estimates focal length” and does not affect boundaries. However, DepthPro uses the FOV prediction to **rescale the decoder output to metric depth**. If the FOV encoder predicts an incorrect focal length for the resized 512×1024 input, the entire depth map is scaled incorrectly. While relative boundaries may survive, the **MSE distillation loss operates on absolute metric values**. Has an ablation been run to confirm that freezing FOV does not harm MSE convergence?

---

### 23. [QUESTION] Why 512×1024 instead of native Cityscapes resolution?
**Location:** `train_depth_adapter_lora.py:307`

Cityscapes is natively 1024×2048. Downsampling by 2× loses fine details critical for thin instance boundaries (poles, bicycle wheels, rider limbs). Has the impact of resolution on downstream PQ_things been ablated? Could the adapters simply be trained at native resolution with patch-based processing?

---

### 24. [QUESTION] Where is the empirical PQ_things evaluation?
**Location:** Entire repository (not found in reviewed files)

The entire justification for this adapter system is architectural intuition. **There is no evaluation script, no PQ_things numbers, and no comparison to a frozen DepthPro baseline** in the reviewed files. How do the authors know the adapters improve anything? A simple test-time augmentation baseline (multi-scale + flip averaging) is training-free and might outperform this system at zero cost.

---

## Questions for the Authors

1. What is the exact PQ_things improvement on Cityscapes val when using adapted depth vs. frozen DepthPro depth in the Sobel+CC pipeline?
2. Have you visualized the student depth maps during training? Do they diverge from the teacher in global scale or local boundary sharpness?
3. How does the system perform when `late_block_start` is swept across {6, 12, 18, 20}? Is 18 truly optimal, or is this a post-hoc rationalization?
4. What GPU memory does a single training run consume? Does `copy.deepcopy` cause OOM on 16 GB or 24 GB cards?
5. Why was `num_pairs=1024` chosen? Was it tuned, or was it simply a round number?
6. Have you compared adapter training to test-time augmentation (e.g., multi-scale + horizontal flip depth averaging)?

---

## Recommendation

**Reject — Major Revision Required.**

The code as written cannot be trusted for a first training run. The ranking loss is conceptually invalid, the default hyperparameters guarantee representation collapse, augmentations are silently discarded, and there is no validation protocol to detect failure. Even after bug fixes, the choice of MSE for metric depth, the lack of boundary-aware weighting, and the absence of any downstream PQ_things evaluation leave the scientific contribution unsubstantiated. The authors must:

1. Fix or remove the broken ranking loss.
2. Align training defaults with the architecture spec.
3. Implement a proper asymmetric teacher-student data flow with augmentations.
4. Add a validation loop with depth-boundary proxy metrics.
5. Provide empirical PQ_things results against a frozen baseline and a TTA baseline.

Only then can this submission be reconsidered.

---

## Required Changes (Checklist)

- [ ] **Fix ranking loss:** Use teacher depth to define target ordering, or remove the loss entirely.
- [ ] **Fix default `late_block_start`:** Set to 18 for DepthPro, or make it a required CLI argument with no default.
- [ ] **Use augmented views:** Feed `img_aug` to the student (or teacher) branch; do not compute and discard.
- [ ] **Fix train/eval mode:** Call `model.eval()` and selectively `.train()` only adapter modules.
- [ ] **Replace MSE distillation:** Use log-L1, relative L1, or a scale-aware distillation loss.
- [ ] **Reconcile SI loss:** Either remove it or reduce its weight to ≤ 0.1 to avoid conflicting with MSE.
- [ ] **Eliminate `copy.deepcopy`:** Load teacher from pretrained weights twice, or use `state_dict` round-trip.
- [ ] **Remove base-weight cloning:** Reference `wrapped.weight` directly in `DoRALinear` to save ~900 MB.
- [ ] **Add validation:** Compute depth boundary F-score or visual divergence from teacher on a held-out split.
- [ ] **Validate inference thresholds:** Ablate `grad_threshold`, `min_area`, and `dilation_iters` against PQ_things.
- [ ] **Fix `ConvDoRALinear`:** Inject spatial dimensions automatically, or remove the variant.
- [ ] **Add warmup + mixed precision:** Use `LinearLR` warmup and `torch.cuda.amp` for memory and stability.
- [ ] **Provide PQ_things results:** Include a table comparing frozen, TTA, and adapted DepthPro.
