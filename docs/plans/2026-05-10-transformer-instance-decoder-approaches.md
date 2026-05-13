# Transformer-Based Instance Decoder: Design Approaches

**Date**: 2026-05-10  
**Status**: TODO — Design phase, pending selection  
**Goal**: Replace Sobel+CC instances (PQ_things=23.35) with a learned transformer decoder on DepthPro features  
**Supervision**: Hybrid bootstrap (current pseudo-labels → self-train iteratively)  
**Constraint**: No pretrained segmentation models allowed

---

## Current Pipeline (Baseline)

```
DepthPro → depth map (512×1024) → Sobel gradients → threshold (τ=0.01)
  → per-class connected components (A_min=1000) → instances
  → PQ_things = 23.35 (DepthPro, best config)
```

**Key weakness**: Co-planar objects (persons at same depth) merge because the ONLY signal is depth discontinuity.

---

## Research Gap Identified

After surveying 20 papers (DINOSAUR, SPOT, MUFASA, CutLER, CuVLER, CutS3D, UnSAM, S2-UniSeg, CUPS, SlotDiffusion, GLASS, EAGLE, etc.):

> **No published paper combines depth encoder features + a learned transformer decoder for unsupervised instance segmentation.**

- DINOSAUR/SPOT/MUFASA: learned decoder, NO depth
- CutS3D: uses depth, NOT learned (graph-cut)
- CUPS: uses depth for pseudo-labels, CNN detector (not transformer decoder)

---

## Approach 1: Depth-Conditioned Slot Attention Decoder ⭐ RECOMMENDED

**Inspired by**: DINOSAUR (ICLR 2023), SPOT (CVPR 2024 Highlight), MUFASA (arXiv 2026)

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Feature Extraction (FROZEN)                             │
├─────────────────────────────────────────────────────────┤
│ DINOv2 ViT-B/14 → 768-D features (32×64 patches)       │
│ DepthPro encoder → multi-scale ViT features (L3,L6,L9) │
│ DepthPro → depth map → Sobel grads + surface normals   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Depth-FiLM Fusion                                       │
├─────────────────────────────────────────────────────────┤
│ Input: DINOv2 features (768-D) + geometric (18-D)       │
│ FiLM: γ,β = MLP(depth_sinusoidal + grads + normals)    │
│ Output: depth-modulated features (768-D)                │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Slot Attention Module                                   │
├─────────────────────────────────────────────────────────┤
│ K = 20 learnable slot vectors (256-D)                   │
│ T = 5 iterations of competitive binding                 │
│ Each iteration:                                         │
│   attn = softmax(Q_slots · K_features / √d)            │
│   updates = weighted_mean(V_features, attn)             │
│   slots = GRU(slots, updates)                           │
│ Output: K slots (256-D each) + attention maps (K×H×W)   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Transformer Decoder (Feature Reconstruction)            │
├─────────────────────────────────────────────────────────┤
│ 4-layer autoregressive transformer                      │
│ Input: slot representations + positional encoding       │
│ Cross-attention: slots → spatial positions              │
│ Output: reconstructed feature map (768-D, 32×64)        │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Instance Mask Extraction                                │
├─────────────────────────────────────────────────────────┤
│ Per-slot attention maps → soft masks (K×H×W)            │
│ Argmax across slots → hard instance assignment          │
│ Filter by area (A_min) + thing-class assignment         │
│ Upsample 32×64 → 512×1024                              │
└─────────────────────────────────────────────────────────┘
```

### Training Protocol (Hybrid Bootstrap)

| Phase | Epochs | Loss | Signal |
|-------|--------|------|--------|
| 1 (Warmup) | 1-20 | L_recon = MSE(reconstructed_features, target_features) | Self-supervised only |
| 2 (Bootstrap) | 21-40 | L_recon + λ₁·L_mask (CE with pseudo-label assignments) | Current pseudo-labels |
| 3 (Self-train) | 41-60 | L_recon + λ₁·L_mask + λ₂·L_depth_consistency | Self-generated labels (confidence > 0.7) |

**Depth consistency loss**: slots attending to pixels at DIFFERENT depths should be penalized (encourages depth-aware grouping).

### Key Hyperparameters

| Param | Value | Notes |
|-------|-------|-------|
| K (slots) | 20 | Max instances per image (Cityscapes has ~15 avg) |
| d_slot | 256 | Slot dimension |
| T (iterations) | 5 | Slot attention iterations |
| Decoder layers | 4 | Autoregressive transformer |
| d_model | 768 | Match DINOv2 feature dim |
| Params (total) | ~5-8M | Fits on 1080 Ti |
| Batch size | 4 | Per GPU |

### Why This Is Strongest

1. **Self-supervised reconstruction** provides clean gradient — doesn't depend on noisy pseudo-labels alone
2. **Competitive slot binding** handles variable instance count naturally
3. **Lightest compute** (~5-8M params)
4. **Depth FiLM** addresses co-planar failure: appearance-similar objects still differentiated by feature-level slot competition
5. **Novel contribution**: no paper has depth-conditioned slot attention for instances

### Relevant Papers

- DINOSAUR (ICLR 2023) — slot attention on DINO features, feature reconstruction
- SPOT (CVPR 2024) — self-training with patch-order permutation, SOTA mBO=30% COCO
- MUFASA (arXiv 2026) — multi-layer slot attention, plug-and-play improvement

---

## Approach 2: Query-Based Instance Decoder with Depth Cross-Attention

**Inspired by**: Mask2Former architecture (trained from scratch), S2-UniSeg (PQ=25.4 Cityscapes)

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Multi-Scale Feature Pyramid                             │
├─────────────────────────────────────────────────────────┤
│ DepthPro intermediate features: 1/4, 1/8, 1/16, 1/32   │
│ DINOv2 features: 32×64 (1/16 scale)                    │
│ Fuse: concat + 1×1 conv at each scale                  │
│ Output: 4-level feature pyramid                         │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Transformer Decoder (6 layers)                          │
├─────────────────────────────────────────────────────────┤
│ N = 100 learnable object queries (256-D)                │
│ Each layer:                                             │
│   1. Self-attention among queries                       │
│   2. Masked cross-attention to multi-scale features     │
│   3. FFN                                                │
│   4. Depth-geometric positional encoding (3D sinusoidal)│
│ Output: N refined query embeddings                      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Prediction Heads                                        │
├─────────────────────────────────────────────────────────┤
│ Per query:                                              │
│   - Class head: MLP → logit (thing/stuff/∅)             │
│   - Mask head: dot(query_embed, pixel_features) → mask  │
│ Hungarian matching with pseudo-GT for training          │
└─────────────────────────────────────────────────────────┘
```

### Training Protocol

| Phase | Loss Components |
|-------|-----------------|
| 1 (Bootstrap) | L_mask (BCE+Dice) + L_cls (CE) + DropLoss (from CutLER) |
| 2 (Self-train) | Same + confidence filtering (keep IoU > 0.5 predictions) |
| 3 (Refinement) | + L_depth_boundary (penalize masks crossing depth edges) |

### Key Specs

| Param | Value |
|-------|-------|
| Queries | 100 |
| Decoder layers | 6 |
| d_model | 256 |
| Params | ~30-50M |
| Batch size | 2 (A6000) or 1 (1080 Ti) |

### Pros/Cons

**Pros**: Most powerful; native multi-scale; proven architecture; S2-UniSeg gets PQ=25.4 with similar design.  
**Cons**: Heaviest (30-50M params); Hungarian matching is complex; risk of overfitting to noisy pseudo-labels; needs DropLoss or similar robust training.

### Relevant Papers

- Mask2Former (CVPR 2022) — query-based universal segmentor
- S2-UniSeg (arXiv 2025) — unsupervised panoptic with query self-distillation, PQ=25.4
- CutLER (CVPR 2023) — DropLoss for noisy pseudo-labels

---

## Approach 3: Contrastive Instance Embedding + Transformer Clustering

**Inspired by**: Panoptic-DeepLab embedding, SlotContrast (CVPR 2025 Oral)

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Depth-Conditioned Transformer Encoder (4 layers)        │
├─────────────────────────────────────────────────────────┤
│ Input: DINOv2 (768-D) + DepthPro features + geometry    │
│ 4× self-attention layers with depth positional encoding │
│ Output: enhanced per-pixel features (256-D, 32×64)      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Dual Prediction Heads                                   │
├─────────────────────────────────────────────────────────┤
│ Head 1: Instance Embedding (MLP → 64-D per pixel)       │
│ Head 2: Center Heatmap (MLP → 1-D objectness score)     │
│ Head 3: Offset (MLP → 2-D vector to instance center)    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Training Losses                                         │
├─────────────────────────────────────────────────────────┤
│ L_pull: pull same-instance embeddings together (L2)     │
│ L_push: push different-instance embeddings apart (hinge)│
│ L_center: MSE on Gaussian heatmap targets               │
│ L_offset: L1 on offset vectors                          │
│ L_depth: regularize — same instance → similar depth     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Inference: Mean-Shift Clustering                        │
├─────────────────────────────────────────────────────────┤
│ Seeds from center heatmap peaks (NMS)                   │
│ Mean-shift on 64-D embeddings → cluster assignments     │
│ Filter by area + thing-class from semantic map          │
└─────────────────────────────────────────────────────────┘
```

### Key Specs

| Param | Value |
|-------|-------|
| Encoder layers | 4 |
| d_model | 256 |
| Embedding dim | 64 |
| Params | ~10-15M |
| Batch size | 4 |

### Pros/Cons

**Pros**: Simplest to implement; no Hungarian matching; directly measurable embedding quality; center heatmap provides strong seed points.  
**Cons**: Mean-shift at inference is slow and sensitive to bandwidth; contrastive loss noisy with pseudo-labels; less principled than slot attention.

### Relevant Papers

- Panoptic-DeepLab (CVPR 2020) — instance center + embedding + offset paradigm
- SlotContrast (CVPR 2025 Oral) — contrastive slot learning
- DepthG (CVPR 2024) — depth-feature correlation for contrastive sampling

---

## Comparison Matrix

| Criterion | Approach 1 (Slot) | Approach 2 (Query) | Approach 3 (Embed) |
|-----------|-------------------|--------------------|--------------------|
| **Params** | 5-8M | 30-50M | 10-15M |
| **Compute** | Low (1080 Ti OK) | High (A6000 needed) | Medium |
| **Self-supervised signal** | Yes (reconstruction) | No (needs pseudo-labels) | Partial (contrastive) |
| **Variable instance count** | Natural (slots compete) | Fixed N queries | Natural (clustering) |
| **Multi-scale** | No (single scale) | Yes (FPN) | No (single scale) |
| **Novelty** | High (depth+slots) | Medium (Mask2Former variant) | Low (known paradigm) |
| **Risk** | Medium | High (overfit to noise) | Medium |
| **Expected PQ_things** | 25-28 | 26-30 | 24-27 |

---

## TODO: Next Steps

- [ ] **SELECT** approach (recommended: Approach 1)
- [ ] Extract DepthPro intermediate features (identify which layers, what dims)
- [ ] Implement slot attention module with depth-FiLM conditioning
- [ ] Implement transformer decoder (feature reconstruction)
- [ ] Build training loop with 3-phase curriculum
- [ ] Evaluate Phase 1 (reconstruction-only) — verify slots discover objects
- [ ] Add pseudo-label bootstrap (Phase 2)
- [ ] Self-training loop (Phase 3)
- [ ] Ablation: with/without depth conditioning, num slots, num iterations

---

## Key References

1. DINOSAUR — Seitzer et al., ICLR 2023. Slot attention on DINO features.
2. SPOT — Kakogeorgiou et al., CVPR 2024 Highlight. Self-training + patch permutation.
3. MUFASA — Bock et al., arXiv 2026. Multi-layer slot attention.
4. CutLER — Wang et al., CVPR 2023. DropLoss for noisy pseudo-labels.
5. S2-UniSeg — arXiv 2025. Query self-distillation, PQ=25.4 Cityscapes.
6. CutS3D — Sick et al., ICCV 2025. 3D graph-cut from depth (non-learned).
7. UnSAM — Wang et al., NeurIPS 2024. SAM trained from scratch on pseudo-masks.
8. SlotContrast — Manasyan et al., CVPR 2025 Oral. Temporal contrastive slots.
9. GLASS — Singh et al., CVPR 2025. Diffusion-guided slot attention.
10. DepthG — Sick et al., CVPR 2024. Depth-feature correlation for semantics.
