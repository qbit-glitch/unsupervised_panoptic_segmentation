---
type: knowledge
title: Adapter Strategy — LoRA/DoRA Placement in MBPS
project: mbps-panoptic-segmentation
tags:
  - architecture
  - adapters
  - lora
  - dora
  - neurips
  - professor-advice
updated: 2026-04-24
---

# Adapter Strategy — LoRA/DoRA Placement in MBPS

## Decision (2026-04-24, NeurIPS Professor)

> **DoRA/LoRA adapters are used ONLY in Stage 1 (pseudo-label generation). They are NEVER used in Stage 2 or Stage 3.**

### Rationale

| Stage | Training Mode | Adapter Role | Verdict |
|-------|---------------|--------------|---------|
| **Stage 1** | Frozen pretrained models (CAUSE-TR, DepthPro, DA3, DA2-Large) | **DoRA** provides controlled domain adaptation without ground-truth labels | ✅ Use |
| **Stage 2** | Full backpropagation (Mask2Former end-to-end) | DoRA is redundant — full model already adapts; adds indirection | ❌ Remove |
| **Stage 3** | Full backpropagation + EMA teacher (CUPS self-training) | DoRA corrupts pretrained features, weakening the EMA teacher | ❌ Remove |

### Why Adapters Fail in Stage 2/3

1. **Redundant capacity:** Full backpropagation can already update every weight. Constraining updates to a low-rank subspace (DoRA) adds no benefit — it only limits expressiveness.
2. **Feature corruption:** Empirically, Conv-DoRA in Stage 2/3 cost **~5.7 PQ** compared to frozen backbone. The adapted features drift from the pretrained manifold, producing a weak EMA teacher in Stage 3.
3. **Amplified noise:** With noisy pseudo-labels, adapters overfit to label errors. Frozen backbones are robust because they cannot overfit.

### Why Adapters Work in Stage 1

1. **No ground truth available:** We cannot do full supervised fine-tuning. DoRA provides a small, controlled parameter budget (~0.8% of model) for domain adaptation.
2. **Preserves pretrained features:** The base weights stay frozen. Only low-rank perturbations are learned, keeping the model close to its pretrained manifold.
3. **Self-supervised objectives:** Distillation from frozen teacher + auxiliary losses (depth alignment, ranking) guide adaptation without requiring labels.

## Implementation

### Stage 1 — Adapter Training

| Component | Script | Adapter Target |
|-----------|--------|----------------|
| Semantic pseudo-labels | `mbps_pytorch/train_semantic_adapter.py` | DINOv2 ViT-B/14 + CAUSE-TR head |
| Depth pseudo-labels | `mbps_pytorch/train_depth_adapter_lora.py` | DA2-Large / DA3 / DepthPro encoder |

Recommended config:
- Variant: `dora` (weight decomposition, no conv)
- Rank: `r=4`, alpha `α=4.0`
- Late-block start: `6` (ViT-Base), `18` (ViT-Large / DINOv2-L)
- Losses: distillation + depth_cluster (semantic), distillation + ranking (depth)

### Stage 1 — Adapted Pseudo-Label Generation

| Output | Script |
|--------|--------|
| Adapted semantic pseudo-labels | `mbps_pytorch/generate_semantic_pseudolabels_adapted.py` |
| Adapted instance pseudo-labels | `mbps_pytorch/generate_instance_pseudolabels_adapted.py` |

⚠️ **Critical:** Use the fixed generation scripts (post-C213). Prior versions silently dropped LoRA weights, making adapter metrics invalid.

### Stage 2/3 — Standard Training (No Adapters)

Stage 2/3 configs should:
- Load standard pretrained backbones (no adapter injection)
- Use full backpropagation (backbone trainable or frozen as design choice)
- Point to adapted pseudo-label directories:
  ```yaml
  pseudo_semantic_dir: "cityscapes/pseudo_semantic_adapted/train"
  pseudo_instance_dir: "cityscapes/pseudo_instance_adapted/train"
  ```

## Empirical Evidence

| Experiment | Backbone | Stage-2 PQ | Stage-3 PQ | Gain | Source |
|------------|----------|-----------|------------|------|--------|
| Frozen | Frozen DINOv3 | 28.09 | **37.43** | **+9.34** | `why_conv_dora_not_plain_lora.md` |
| Conv-DoRA | Conv-DoRA DINOv3 | 29.02 | 32.64 | +3.62 | `why_conv_dora_not_plain_lora.md` |

**Gap:** ~5.7 PQ lost by using adapters in Stage 2/3.

## Audit Checklist

Before running any Stage 2/3 experiment, verify:
- [ ] No `inject_lora_into_dinov2()` in training scripts
- [ ] No `inject_lora_into_cause_tr()` in training scripts
- [ ] No `inject_lora_into_depth_model()` in training scripts
- [ ] Checkpoint loading does not expect `lora_A`/`lora_B` keys
- [ ] `count_adapter_params(model) == 0` after model initialization

## Related
- [[Daily/2026-04-24]] — Meeting notes with NeurIPS professor
- `analysis_docs/why_conv_dora_not_plain_lora.md` — Empirical analysis of adapter failure in Stage 2/3
- `analysis_docs/lora_adapter_implementation_plan.md` — Original Stage-1 adapter design
- Commit `C213` — Bug fix for silent LoRA drop in pseudo-label generation
