# Plan: DoRA Integration for DINOv3 + Cascade Mask R-CNN Stage-2

## Context

The MBPS unsupervised panoptic segmentation pipeline uses a **fully frozen** DINOv3 ViT-B/16 backbone (88M params) feeding into a trainable Cascade Mask R-CNN (~20M params). Current best: **PQ=30.255%** (Stage-3), beating CUPS 27.8%. The backbone is frozen because pseudo-labels (generated unsupervised) are noisy and 4000-8000 training steps are insufficient for full fine-tuning.

**Hypothesis**: DoRA (Weight-Decomposed LoRA) can task-adapt DINOv3 features for panoptic segmentation with only ~0.33M additional parameters — enough capacity to improve thing detection (the main bottleneck: PQ_things=28.5%) without overfitting to noisy pseudo-labels. DoRA's magnitude-direction decomposition protects pretrained feature scales while adapting spatial attention patterns.

## Approach: DoRA with LoRA+ Learning Rate Split

**Variant**: DoRA (Liu et al., ICML 2024 Oral) — decomposes weight into magnitude `m` and direction `V/||V||_c`, applies LoRA only to direction. Combined with LoRA+ (Hayou et al., ICML 2024) differential LR: B and magnitude learn faster than A.

**Why DoRA over standard LoRA**: Noisy pseudo-labels can corrupt feature scales. DoRA isolates magnitude from direction — the magnitude vector `m` can be tightly regularized (weight decay 1e-3) while direction adapts via low-rank LoRA. DINOv3's weight magnitudes encode critical information from 1.7B-image distillation training.

## Architecture: Tiered Block Injection

DINOv3 ViT-B/16 has 12 `SelfAttentionBlock` (in `refs/dinov3/dinov3/layers/`), each with:
- `attn.qkv`: `nn.Linear(768, 2304)` — fused Q/K/V
- `attn.proj`: `nn.Linear(768, 768)` — output projection
- `mlp.fc1`: `nn.Linear(768, 3072)` — FFN up-projection
- `mlp.fc2`: `nn.Linear(3072, 768)` — FFN down-projection

**Injection plan (rank r=4)**:

| Blocks | attn.qkv | attn.proj | mlp.fc1 | mlp.fc2 | Rationale |
|--------|:---:|:---:|:---:|:---:|-----------|
| 0–5 (early) | DoRA r=4 | — | — | — | Low-level features; minimal steering via qkv only |
| 6–11 (late) | DoRA r=4 | DoRA r=4 | DoRA r=4 | DoRA r=4 | Semantic features; full adaptation |

**Parameter count**: ~331K DoRA params (0.37% of backbone, 1.7% of current trainable params)

## Files to Create/Modify

### NEW: `refs/cups/cups/model/lora.py`
Core DoRA implementation. Contains:
- `class DoRALinear(nn.Module)` — wraps existing `nn.Linear` (not replace), stores frozen W, trainable A/B/m
- `def inject_dora_into_model(model, config)` — walks model tree, replaces target `nn.Linear` modules in `self.vit.blocks[i].attn.qkv/proj` and `self.vit.blocks[i].mlp.fc1/fc2`
- `def get_dora_param_groups(model, lr_a, lr_b)` — returns optimizer param groups with differential LR

Source: adapt from teaching notebook (`00_lora_complete_guide.ipynb` Cell 12 `DoRALinear`) but modified to wrap in-place.

### MODIFY: `refs/cups/cups/model/backbone_dinov3_vit.py`
3 changes to `DINOv3ViTBackbone`:
1. Add `lora_config: dict | None = None` parameter to `__init__` (line 80-84)
2. After loading model (line 93), inject DoRA if config provided
3. In `forward()` (line 136-140): when LoRA active, don't use `torch.no_grad()` — gradients must flow through DoRA params. Change condition from `if self._freeze` to `if self._freeze and self._lora_config is None`

Same for `build_dinov3_vitb_fpn_backbone()` (line 178-214): add `lora_config` param, pass to backbone.

### MODIFY: `refs/cups/cups/pl_model_pseudo.py`
2 changes:
1. `configure_optimizers()` (line 413-452): add DoRA param groups with differential LR
   - Group 1: detection heads — LR=1e-4, WD=1e-5 (unchanged)
   - Group 2: DoRA B + magnitude — LR=5e-5, WD=0
   - Group 3: DoRA A — LR=1e-5, WD=0
2. `build_model_pseudo()` (line 455+): pass `lora_config` from config to model builder

### NEW: Config YAML (e.g. `refs/cups/configs/train_cityscapes_dinov3_k80_dora.yaml`)
Add LORA section:
```yaml
MODEL:
  LORA:
    ENABLED: true
    VARIANT: "dora"
    RANK: 4
    ALPHA: 4.0
    LATE_BLOCK_START: 6
    LR_A: 1e-5
    LR_B: 5e-5
    DROPOUT: 0.05
    DELAYED_START_STEPS: 500
```

## Training Recipe

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (3 param groups) |
| Head LR | 1e-4 (unchanged) |
| DoRA B + magnitude LR | 5e-5 |
| DoRA A LR | 1e-5 |
| Weight decay (DoRA) | 0 (A/B), 1e-3 (magnitude only) |
| LoRA dropout | 0.05 |
| Delayed activation | First 500 steps: DoRA LR = 0 (heads settle first) |
| Warmup | Steps 500-1000: linear ramp from 0 → target DoRA LR |
| Total steps | 8000 |
| Batch size | 2 (grad_accum=4, effective bs=8) |
| GPU | RTX A6000 48GB |

## Noise Mitigation Strategies

1. **DoRA magnitude regularization**: WD=1e-3 on `m` vector prevents backbone from scaling feature channels based on noisy class assignments
2. **LoRA dropout (p=0.05)**: randomly zeros the adaptation path during training
3. **Delayed activation (500 steps)**: let detection heads settle before backbone adaptation
4. **Low LR (0.5× base for B, 0.1× for A)**: gentle adaptation prevents catastrophic corruption of pretrained features

## Ablation Plan (8 Experiments, Priority Order)

| # | Experiment | Change | Expected PQ |
|---|-----------|--------|:-----------:|
| 1 | Baseline (frozen, no LoRA) | Current config | 27.9 |
| 2 | **DoRA r=4 tiered (full recipe)** | This plan | 29.0-30.0 |
| 3 | LoRA r=4 tiered (no decomposition) | `LoRALinear` instead of `DoRALinear` | 28.5-29.5 |
| 4 | DoRA r=4 all-blocks uniform | All 12 blocks: qkv+proj+fc1+fc2 | 28.5-29.5 |
| 5 | DoRA r=2 tiered | Lower rank | 28.0-29.0 |
| 6 | DoRA r=8 tiered | Higher rank | 28.5-30.0 |
| 7 | DoRA r=4 attention-only | No MLP LoRA | 28.5-29.5 |
| 8 | Best config + Stage-3 self-training | Best from #2-7 | **31.0-32.5** |

## Verification

1. **Smoke test**: Build model, print param counts (confirm ~331K DoRA, ~20M heads, ~88M frozen). Run 50 steps, verify loss decreases and gradients flow to DoRA params
2. **Zero-init check**: At step 0, confirm LoRA-adapted output == frozen output (B=0 means delta_W=0)
3. **Per-experiment**: Track PQ, PQ_things, PQ_stuff every 500 steps; monitor ||B||, ||A||, ||m|| norms
4. **Feature sanity**: CKA between frozen features and LoRA features should stay > 0.85
5. **Final eval**: 27-class CAUSE + Hungarian matching (same metric as CUPS paper)

## Memory / Speed Impact

- **Additional memory**: ~100MB activation storage for LoRA backward (negligible on A6000 48GB)
- **Speed overhead**: <5% wall-clock (one small r=4 matmul per adapted layer)
- **No batch size change needed**
