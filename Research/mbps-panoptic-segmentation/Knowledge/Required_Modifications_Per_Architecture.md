# Required Modifications Per Architecture
## Verified by Cross-Check Between Qbit (Theory) and Puchu (Implementation)

---

## A. Depth Anything V2 Large (DA2-Large) + DoRA
**Confidence: 7/10 | Status: Proceed with minor fixes**

### Modifications Needed:

1. **Add explicit loss weighting strategy**
   - Current: MSE (1.0) + ranking (0.1) + SI (0.5) — weights listed but no tuning rationale
   - Fix: Run a quick ablation on λ_mse, λ_rank, λ_si on a 200-image validation subset
   - Target ratio: MSE should dominate (prevents divergence), ranking should be 5-10% of MSE, SI 20-50%

2. **Monitor double-adaptation risk**
   - DA2 encoder is already fine-tuned for depth; DoRA adds a second adaptation layer
   - Fix: Log gradient norms for adapter params vs base params. If adapter grads are <1% of base param magnitude, increase LR. If >50%, decrease LR or rank.

3. **No structural changes needed**
   - HF loading path (`AutoModelForDepthEstimation`) is correct
   - Tiered injection (`backbone.encoder.layer[i]`) is verified against HF Transformers
   - Parameter count (~829K) is accurate

### What Works Out of the Box:
- Self-supervised training paradigm (student-teacher distillation)
- Sobel + Connected Components for instance pseudo-labels (empirically validated PQ_things = 20.20)
- τ = 0.03 for Cityscapes (document has ablation data)

---

## B. Depth Anything V3 (DA3) + DoRA
**Confidence: 8/10 | Status: Proceed with corrected loading path**

### Modifications Needed:

1. **Replace custom API loading with standard HF loading**
   - Current doc claims: `depth_anything_3.api.DepthAnything3.from_pretrained()`
   - **WRONG.** DA3 is available on HuggingFace as `depth-anything/DA3-LARGE`
   - Fix: Use `AutoModelForDepthEstimation.from_pretrained("depth-anything/DA3-Large")`
   - This eliminates the need for generic fallback injection entirely

2. **Use structured tiered injection (same as DA2)**
   - Current doc proposes: generic `named_modules()` walker with string matching
   - Fix: Use `_find_encoder_blocks()` targeting `model.backbone.encoder.layer` (same HF path as DA2)
   - Tiering: Early (0-17) Q+V only, Late (18-23) full 6 layers

3. **Same loss weighting ablation as DA2**
   - Identical self-supervised losses apply
   - Run same λ tuning

4. **Update τ default**
   - Document claims τ=0.03 is shared with DA2
   - DA3 produces sharper boundaries — τ=0.03 may still work, but verify with ablation
   - Suggested ablation range: {0.02, 0.03, 0.05, 0.08}

### What Works Out of the Box:
- Self-supervised losses (same as DA2)
- Tiered DoRA strategy
- PQ_things 20.90 (document's ablation number)

---

## C. Apple DepthPro + DoRA
**Confidence: 4/10 | Status: Needs significant fixes before proceeding**

### Modifications Needed:

1. **Replace Scale-Invariant loss with MetricDepthLoss**
   - Current: MSE + ranking + SI (Eigen et al.)
   - **SI loss destroys metric scale** — the `-(λ/n²)(Σ(d-d*))²` term removes global mean difference
   - Fix: Drop SI entirely. Replace with:
     - Edge-aware Log-L1 base loss (preserves absolute scale, handles dynamic range)
     - Multi-scale gradient consistency (local geometry, no mean subtraction)
   - See `DepthPro_Patch_Specification.md` for full implementation

2. **Fix Sobel input: raw metric depth, not [0,1]-normalized**
   - Current: `depth_norm = (depth - min) / (max - min)` then Sobel
   - This makes τ physically meaningless (same 20cm jump maps to different values per scene)
   - Fix: Run Sobel directly on metric depth in meters. τ=0.20 means "20cm depth jump = boundary"

3. **Run τ ablation on validation set**
   - Current τ=0.20 has **no justification** in the document
   - Fix: Implement `TauAblationStudy` class
   - Search τ ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50} meters
   - Select τ that maximizes PQ_things

4. **Address memory constraints**
   - Three DINOv2-Large encoders = ~1B parameters
   - Even with 1.66M trainable, forward pass needs full model in memory
   - Fix: Use gradient checkpointing or `torch.cuda.empty_cache()` between encoder passes
   - Minimum GPU: 40GB. Your 48GB RTX Pro covers this with headroom.

5. **Consider dropping ranking loss**
   - MarginRankingLoss was designed for relative depth ordering
   - With metric depth + gradient consistency loss, ranking is redundant
   - Simplified loss stack: `MetricDepthLoss` only (MSE distillation is implicitly handled by Log-L1 base)

6. **Verify FOV encoder freezing assumption**
   - Document freezes FOV encoder because "it only predicts focal length"
   - If camera intrinsics vary across dataset, frozen FOV = systematically wrong depth scale for some images
   - Fix: Check if your dataset has consistent focal length. If not, consider adapting FOV encoder too (adds ~829K params, still <0.3% of total)

### What Works Out of the Box:
- Adapter injection strategy (tiered on Patch + Image encoders, FOV frozen)
- Parameter count (~1.66M trainable = 0.17% of ~1B)
- Three-encoder architecture (real, from Apple paper)

---

## D. DINOv2 + CAUSE-TR + DoRA
**Confidence: 7/10 | Status: Proceed with EMA fix**

### Modifications Needed:

1. **Reduce EMA momentum from 0.999 to 0.95–0.99**
   - Current: `head_ema ← 0.99·ema + 0.01·head` (momentum λ = 0.99)
   - Wait — re-reading: the document says `λ = 0.99` which means EMA weight is 0.99, new weight is 0.01. This is actually standard.
   - BUT the training diagram shows `head_ema ← 0.99·ema + 0.01·head`. If this is meant to be the update rule, then after 100 steps the teacher is still 36% original. After 1000 steps it's still 0.004% original. This is actually fine.
   - However, if the code implements `momentum = 0.999` (as Puchu flagged), that's too high.
   - Fix: Verify the actual EMA momentum value in code. If it's 0.999, reduce to 0.99. If it's already 0.99, no change needed.

2. **Add loss balancing strategy**
   - 4 losses: KL distillation + depth correlation + cross-view consistency + cluster loss
   - Current weights: not specified in document
   - Fix: Start with equal weighting (0.25 each), then monitor gradient norms per loss. If one loss dominates by >5×, downweight it.
   - Recommended initial weights based on typical magnitudes:
     - KL distillation: 1.0 (anchor)
     - Depth correlation: 0.05 (weak signal)
     - Cross-view consistency: 0.1
     - Cluster loss: 0.1

3. **Consider increasing rank from 4 to 8 or 16**
   - Current: ~472K trainable params (0.5% of 86M backbone)
   - For meaningful clustering adaptation, this may be too few
   - Fix: Run ablation on rank ∈ {4, 8, 16} measuring pseudo-label mIoU
   - Cost at rank=16: ~1.9M params (still only 2.2% of backbone)

4. **Verify CAUSE-TR internals independently**
   - CAUSE-TR paper found: "Causal Unsupervised Semantic Segmentation" (arXiv:2310.07379)
   - Confirms: VQ mechanism, 90D reduction, TRDecoder with SA+CA+FFN
   - Fix: Cross-check your `segment.head` implementation against the paper's Appendix B.2

5. **K-Means clustering robustness**
   - K=54 clusters mapped to 19 Cityscapes classes via Hungarian matching
   - This is standard (STEGO, PiCIE) but clustering quality depends on feature distribution
   - Fix: After adapter training, visualize t-SNE of 90D codes. If clusters are not well-separated, increase K to 108 (6 per class) and merge post-hoc.

### What Works Out of the Box:
- Tiered DoRA on DINOv2-B/14 (well-established)
- Sliding window inference (322×322 crops, stride 161)
- K-Means clustering approach (STEGO-inspired, legitimate)
- Student-teacher distillation (standard practice)

---

## Cross-Cutting Modifications for All Architectures

### 1. Loss Weighting Strategy (All 4 architectures)
Every document lists loss formulas but **never explains how weights were chosen** or whether they need tuning. Add a simple validation script:

```python
# After each epoch, evaluate on 200-image val set with different λ combos
for lambda_mse in [0.5, 1.0, 2.0]:
    for lambda_rank in [0.05, 0.1, 0.2]:
        for lambda_si in [0.0, 0.25, 0.5, 1.0]:  # 0.0 for DepthPro
            pq = evaluate_pq(val_loader, model, lambdas=(lambda_mse, lambda_rank, lambda_si))
            log(f"λ=({lambda_mse},{lambda_rank},{lambda_si}) → PQ={pq:.2f}")
```

### 2. Gradient Norm Monitoring (All 4 architectures)
Double adaptation risk (DoRA on already-fine-tuned encoders) requires monitoring:

```python
# In training loop
for name, param in model.named_parameters():
    if param.grad is not None and 'lora' in name:
        grad_norm = param.grad.norm().item()
        wandb.log({f"grad_norm/{name}": grad_norm})
        
# Alert if any adapter grad_norm drops to <1e-6 (dead adapter)
# or exceeds 1.0 (potential explosion)
```

### 3. Checkpoint Validation (All 4 architectures)
The document mentions `adapter_config` dict serialization. Verify this actually reconstructs the exact same injection topology at inference:

```python
# After saving checkpoint
ckpt = torch.load("best.pt")
model_new = load_base_model()
inject_lora_from_config(model_new, ckpt['adapter_config'])
model_new.load_state_dict(ckpt['model'], strict=False)

# Verify: every parameter with 'lora' in name matches original
for (n1, p1), (n2, p2) in zip(model.named_parameters(), model_new.named_parameters()):
    if 'lora' in n1:
        assert torch.allclose(p1, p2), f"Checkpoint mismatch: {n1}"
```

---

## Summary Table: What to Change

| Architecture | Critical Fix | Medium Fix | Minor Fix |
|-------------|------------|-----------|-----------|
| **DA2-Large** | Add loss weighting ablation | Monitor grad norms | — |
| **DA3** | Switch to HF loading (drop custom API) | Update τ ablation | — |
| **DepthPro** | Replace SI with MetricDepthLoss | Run τ ablation | Verify FOV freeze |
| **DepthPro** | Fix Sobel input (raw metric) | Add gradient checkpointing | Consider rank=8 |
| **DINOv2+CAUSE-TR** | Verify EMA momentum value | Add 4-loss balancing | Consider rank=8 |
| **All** | Add gradient norm monitoring | Add checkpoint validation | — |

---

*Cross-verified by Qbit (theoretical) and Puchu (implementation).*
