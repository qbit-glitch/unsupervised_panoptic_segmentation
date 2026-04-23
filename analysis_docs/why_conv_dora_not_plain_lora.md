# Why Are We Using Conv-DoRA Instead of Plain LoRA?

**Date:** 2026-04-21
**Context:** DINOv3 ViT-B Stage-3 Self-Training Underperformance
**Status:** Open question — evidence strongly suggests we should NOT be using Conv-DoRA

---

## 1. The Original Rationale (Why We Chose It)

### Literature-Driven Choice

Conv-DoRA was selected based on the **Conv-LoRA** paper (ICLR 2024), which proposed:

> *"Add spatial inductive bias to ViT linear layers for dense prediction"*

The core idea: plain LoRA only adapts linear projections (W' = W₀ + BA), which maintains the **per-token** nature of ViT attention. For dense prediction tasks (segmentation, detection), the authors argued that injecting a **depthwise 3×3 convolution** inside the LoRA bottleneck adds local spatial correlations that ViTs lack — an "inductive bias" that CNNs naturally have.

### Our Implementation

In `cups/model/lora.py`:

```python
class ConvDoRALinear(nn.Module):
    """DoRA + depthwise 3×3 conv for spatial inductive bias."""
    def forward(self, x):
        # 1. Standard DoRA: low-rank adaptation
        lora_out = self.lora_B(self.lora_A(x))  # (B, N, r)
        # 2. SPATIAL CONV: reshape to 2D grid, apply DWConv
        B, N, C = lora_out.shape
        H = W = int(N ** 0.5)
        lora_out = lora_out.transpose(1, 2).view(B, C, H, W)
        lora_out = self.dwconv(lora_out)  # ← THE KEY ADDITION
        lora_out = lora_out.view(B, C, N).transpose(1, 2)
        # 3. Weight decomposition: magnitude + direction
        ...
```

The `dwconv` is a `nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C)` — a **depthwise separable 3×3 convolution** that operates on the spatial grid of patch tokens.

### The Argument Was:

| Claim | Reasoning |
|-------|-----------|
| ViT lacks spatial bias | Self-attention is permutation-invariant within the sequence; no locality prior |
| Dense prediction needs locality | Segmentation boundaries are local; neighboring pixels should interact |
| Conv-LoRA proved this | ICLR 2024 showed gains on ADE20K semantic segmentation |
| DoRA > LoRA | Weight decomposition (magnitude/direction) is more stable than plain low-rank |

---

## 2. The Empirical Reality (Why It Was Wrong)

### Experiment E2: Clean Conv-DoRA (2026-04-16)

| Configuration | PQ | PQ_Stuff | PQ_Things | Result |
|---------------|-----|----------|-----------|--------|
| **Baseline (frozen backbone)** | **27.87** | 32.4 | 2.9 | Reference |
| **Conv-DoRA r=4, late=6** | **26.40** | 30.6 | 19.4 | **−1.47 PQ regression** |

**Key finding:** Conv-DoRA improved PQ_Things (+16.5!) but **destroyed PQ_Stuff** (−1.8), for net negative.

### Experiment E2-DepthPro: Near-Miss (2026-04-18)

With **better pseudo-labels** (DepthPro τ=0.20 instead of k=80 overclustering):

| Configuration | PQ | Delta |
|---------------|-----|-------|
| Frozen backbone | 28.40 | Baseline |
| Conv-DoRA r=4 | 28.28 | −0.12 (near-miss) |

Better labels helped, but Conv-DoRA **still underperformed** frozen.

### Current Stage-3: The Smoking Gun (2026-04-21)

| Stage | Frozen Backbone | Conv-DoRA (r=32, all blocks) |
|-------|----------------|------------------------------|
| Stage-2 PQ | 28.09 | 29.02 |
| Stage-3 PQ (best) | **37.43** | **32.64** |
| **Stage-3 gain** | **+9.34** | **+3.62** |

**This is the critical result.** Self-training amplifies backbone quality:
- Frozen backbone: +9.3 PQ gain (strong features → strong teacher → strong student)
- Conv-DoRA: +3.6 PQ gain (corrupted features → weak teacher → weak student)

The **+5.7 PQ gap** is the cost of using Conv-DoRA. The EMA teacher cannot escape the low ceiling created by Conv-DoRA-corrupted features.

---

## 3. Root Cause: Spatial Inductive Bias Clash

### The Fundamental Mismatch

| Property | DINOv3 ViT | Conv-DoRA DWConv |
|----------|-----------|------------------|
| Feature type | **Global attention** — each patch attends to ALL other patches | **Local convolution** — 3×3 neighborhood only |
| Spatial structure | Learned implicitly via position embeddings + attention weights | Hard-coded locality via kernel weights |
| Information flow | Long-range from pretraining | Short-range, artificially injected |
| Optimization | Pretrained on 100M+ images with global context | Randomly initialized 3×3 kernels |

**The problem:** DINOv3's features ALREADY encode spatial relationships through **global self-attention**. Adding a 3×3 DWConv:
1. **Constrains** the feature adaptation to local neighborhoods
2. **Conflicts** with the pretrained global attention patterns
3. **Introduces noise** from randomly initialized conv kernels
4. **Disrupts** all 6+ downstream heads simultaneously

### Why It Hurts Stuff More Than Things

| Aspect | Stuff (amorphous regions) | Things (countable objects) |
|--------|---------------------------|---------------------------|
| Spatial scale | **Large** (road, sky, building facades) | **Small/medium** (cars, pedestrians) |
| Needs global context | **Yes** — "sky" is spatially extended | **Less** — local boundaries matter more |
| Conv-DoRA effect | **Damaging** — 3×3 kernel can't capture large regions | **Helpful** — local edge detection improves |
| PQ impact | PQ_Stuff drops | PQ_Things rises |

This explains the E2 pattern: PQ_Things ↑ but PQ_Stuff ↓, net negative.

---

## 4. Why Plain LoRA or Pure DoRA Might Be Better

### Option A: Plain LoRA (No Conv)

```python
# LoRA: W' = W₀ + BA
# - No spatial constraints
# - Per-token adaptation respects pretrained global patterns
# - Fewer parameters (no conv kernels)
```

**Pros:**
- Preserves DINOv3's global attention structure
- No randomly initialized spatial kernels
- Simpler, more stable optimization

**Cons:**
- May not capture local boundary refinement
- Still adds 4.8M parameters

### Option B: Pure DoRA (No Conv, Weight Decomposition)

```python
# DoRA: W' = m ⊙ (W₀ + BA) / ||W₀ + BA||
# - Magnitude/direction decomposition
# - More stable than LoRA (direction changes are normalized)
# - No spatial convolution
```

**Pros:**
- Better training stability than LoRA
- Preserves pretrained feature directions
- No spatial clash

**Cons:**
- Still underperformed frozen in early trials (needs re-testing with good labels)

### Option C: No Adaptation At All (Frozen)

```python
# Frozen: W' = W₀
# - Zero adaptation parameters
# - Maximum preservation of pretrained features
```

**Pros:**
- Best Stage-3 gain (+9.3 PQ)
- Proven by E2 and current experiments
- Simplest

**Cons:**
- No backbone customization for panoptic task
- May limit ceiling if pseudo-labels were perfect

---

## 5. The Label Quality Hypothesis

From C180 (2026-04-18):

> *"Conv-DoRA adaptation is label-quality limited, not fundamentally broken. Old k=80 labels → regression. DepthPro τ=0.20 labels → near-miss (−0.12). When a frozen baseline moves up (28.40 vs 27.87), a previously-failing adaptation technique can become viable on the same architecture."*

**The hypothesis:** With PERFECT pseudo-labels, Conv-DoRA might help. But:
- Current labels (SimCF ABC, DepthPro, k=80) all have noise
- Conv-DoRA **amplifies label noise** by giving the backbone capacity to overfit
- Frozen backbone **is robust to label noise** because it cannot overfit

**Conclusion:** In the unsupervised setting (noisy pseudo-labels), **frozen backbone > any adaptation**.

---

## 6. Recommendations

### Immediate Actions

| Priority | Action | Rationale |
|----------|--------|-----------|
| **P0** | **Stop Conv-DoRA Stage-3** | Current run is wasting GPU hours — best PQ 32.64 vs frozen's 37.43 |
| **P1** | **Run plain LoRA ablation** | Test `VARIANT: "lora"` (no conv) on Stage-2 with SimCF ABC labels |
| **P2** | **Run pure DoRA ablation** | Test `VARIANT: "dora"` (weight decomp, no conv) |
| **P3** | **Try reduced scope** | `late_block_start=6` (only blocks 6-11) instead of 0 (all 12) |
| **P4** | **Try reduced rank** | `r=4, alpha=4.0` instead of `r=32, alpha=32.0` |

### Stage-3 Strategy Going Forward

Given the evidence, the optimal strategy is:

```
Stage-2: Conv-DoRA/plain LoRA/pure DoRA → find best Stage-2 PQ
Stage-3: ALWAYS use frozen backbone (load best Stage-2 ckpt, freeze backbone)
```

The Stage-2 adaptation is just to **find a better initialization** for the heads. The real gains come from self-training with a **frozen, high-quality backbone**.

---

## 7. Summary

| Question | Answer |
|----------|--------|
| Why Conv-DoRA? | Literature (Conv-LoRA ICLR 2024) suggested spatial inductive bias helps dense prediction |
| Was the literature wrong? | **Partially** — it applies to supervised segmentation with GT labels, not unsupervised with noisy pseudo-labels |
| What's the real issue? | DINOv3's global attention features **clash** with Conv-DoRA's local 3×3 conv; also, adaptation amplifies label noise |
| What's the fix? | Test plain LoRA / pure DoRA / frozen; abandon Conv-DoRA for this task |
| What's the ceiling? | Frozen backbone Stage-3 reaches PQ 37.43; that's our target |

---

## 8. Open Questions for Discussion

1. **Should we test plain LoRA on Stage-2 with SimCF ABC labels?** The E2 near-miss was with DepthPro labels; SimCF ABC might behave differently.

2. **Is there a middle ground?** E.g., `ConvDoRA` with `kernel_size=1` (pointwise conv) — adds channel mixing without spatial constraints?

3. **Should we adapt only the LATE blocks?** `late_block_start=6` adapts only blocks 6-11 (higher-level features), leaving early blocks (low-level edges/textures) frozen.

4. **What about adapter placement?** Instead of qkv/proj/fc1/fc2, adapt only specific layers (e.g., only projection layers, not attention qkv)?

5. **Is there ANY scenario where Conv-DoRA wins?** Perhaps with extremely high-quality pseudo-labels (e.g., oracle labels), but that's not our setting.

---

*Document compiled from project memory (commits C142, C170, C180), diagnostic analysis, and current training logs.*
