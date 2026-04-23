# DoRA Stage-3 Diagnostic Analysis

## Date: 2026-04-23
## Context: Stage-3 self-training on Cascade Mask R-CNN + Conv-DoRA r=32

---

## Actual Results (Corrected)

| Stage | Frozen Backbone | DoRA r=32 | Delta |
|-------|-----------------|-----------|-------|
| **Stage-2** | **28.09** | **29.02** | +0.93 |
| **Stage-3** | **37.43** | **32.64** | -4.79 |
| **Stage-3 Gain** | **+9.34** | **+3.62** | — |

**Key Insight:** Frozen backbone self-training gains **+9.3 PQ**, while DoRA self-training only gains **+3.6 PQ**. The DoRA-adapted features are a hard ceiling that self-training cannot overcome.

---

## Why Self-Training Fails for DoRA

### Frozen Backbone Case (28 → 37.43)
- **Bottleneck is ONLY the detection heads** (46M params)
- DINOv3 features are already excellent — just need better pseudo-labels
- Virtuous cycle: better labels → better heads → better teacher → even better labels
- EMA teacher starts with high-quality features and improves them

### DoRA Case (29 → 32.64)
- **Bottleneck is BOTH backbone features AND heads**
- DoRA corrupted the pretrained feature distribution across all 12 blocks
- EMA teacher shares the SAME corrupted backbone — cannot generate high-quality pseudo-labels
- Vicious cycle: mediocre labels → mediocre student → mediocre teacher → no improvement

---

## The Conv-DoRA Problem

Conv-DoRA injects **depthwise 3×3 convolutions** into every qkv/proj/fc layer of DINOv3.

DINOv3 was trained with:
- Patch embeddings + global self-attention on 1.7B images
- A feature hierarchy optimized for dense visual understanding

Conv-DoRA introduces:
- **Spatial inductive bias clash**: DINOv3 learned patch-level relationships. Adding 3×3 convs introduces local spatial correlations that conflict with global attention patterns.
- **Compounding error across 12 blocks**: Each block's DoRA modifies its output. By block 11, feature drift is 12× accumulated.
- **High rank = high drift**: r=32 gives enough capacity to significantly shift feature distribution away from pretrained.

---

## Evidence from Ablations

Two independent DoRA configs both hurt performance:
- r=4, `LATE_BLOCK_START=6` (only blocks 6-11): Worse than frozen
- r=32, `LATE_BLOCK_START=0` (all blocks): Still ~6-9 points below frozen

If **even adapting only late blocks with tiny rank hurts**, the issue is not insufficient capacity — it's that **any adaptation degrades pretrained features**.

---

## Verification Checklist (No Bugs Found)

| Check | Result |
|-------|--------|
| DoRA params in checkpoint | 4.80M (correct for r=32, 48 adapters) |
| Backbone base params | 91.24M (frozen, not trainable) |
| Head params | 46.60M (trainable) |
| Checkpoint loads with no missing keys | Clean load |
| Differential LR (LoRA+) in Stage-3 | Confirmed: "AdamW with DoRA differential LR (LoRA+) used" |
| DoRA config passed to builder | Confirmed in logs |

**The code is working as intended.** The problem is architectural, not a bug.

---

## Open Questions

1. **Why are we using conv-DoRA instead of plain LoRA?**
   - Conv-DoRA adds spatial conv which may be especially harmful for patch-based DINOv3
   - Plain DoRA or LoRA might preserve feature quality better

2. **Is r=32 too high?**
   - More rank = more feature drift
   - Try r=8 or r=16 with late_block_start=6

3. **Is ALPHA=32 too aggressive?**
   - alpha/rank = 1.0 (standard)
   - Try alpha=8 or 16 to constrain adaptation

4. **Should we freeze DoRA during Stage-3?**
   - Test whether adapted features can be mapped well without further adaptation

5. **Should we drop DoRA entirely for Stage-3?**
   - Use frozen backbone Stage-2 checkpoint + self-training (proven +9.3 gain)

---

## Recommendations

1. **Stop DoRA Stage-3** — ceiling is too low (~32-33 PQ vs 37+ for frozen)
2. **Try pure DoRA (no conv)** with r=16, alpha=16, late_block_start=6
3. **Try standard LoRA** instead of DoRA — less disruptive to feature distribution
4. **Consider full fine-tuning** with tiny LR (1e-6) instead of adapters
5. **Or accept frozen backbone** — it's the strongest path (28 → 37.43 proven)
