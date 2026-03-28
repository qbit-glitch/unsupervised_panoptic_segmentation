# IC-C Model Architecture & Data Flow

## Training Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATASET (PseudoLabelDataset)                    │
│                                                                        │
│  Pre-extracted per image:                                              │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ DINOv2 Features  │  │ Depth Maps   │  │ Pseudo Semantic Labels   │  │
│  │ (768, 32, 64)    │  │ (1, 512,1024)│  │ (19-class, 32, 64)      │  │
│  └────────┬─────────┘  └──────┬───────┘  └────────────┬─────────────┘  │
│           │                   │                       │                │
│  ┌────────┴─────────┐  ┌─────┴────────────────┐      │                │
│  │ Depth Gradients   │  │ Instance Masks (NEW) │      │                │
│  │ Sobel dx,dy       │  │ (1, 512, 1024)       │      │                │
│  │ (2, 32, 64)       │  │ pseudo_instance_     │      │                │
│  └────────┬──────────┘  │ spidepth/            │      │                │
│           │             └─────┬────────────────┘      │                │
└───────────┼───────────────────┼────────────────────────┼────────────────┘
            │                   │                        │
            ▼                   ▼                        ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    IC-C MODEL (DepthGuidedUNet)                       │
│                       5,458,219 params                                │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    PROJECTION (32×64)                            │  │
│  │                                                                  │  │
│  │  DINOv2 (768)──→ SemanticProjection ──→ sem (192, 32, 64)      │  │
│  │                  [Conv1x1→GN→GELU]                               │  │
│  │                                                                  │  │
│  │  DINOv2 (768)──→ DepthFeatureProjection ──→ depth_feat          │  │
│  │  + Depth (1)     [Cat→Conv1x1→GN→GELU]     (192, 32, 64)       │  │
│  │  + Grads (2)                                                     │  │
│  └──────────────────────────┬──────────────────────────────────────┘  │
│                             │                                         │
│  ┌──────────────────────────▼──────────────────────────────────────┐  │
│  │              BOTTLENECK (32×64) — 2 blocks                      │  │
│  │                                                                  │  │
│  │  ┌──────────────────────────────────────────────────────────┐   │  │
│  │  │  WindowedAttentionBlock (8×8 windows, 4 heads)           │   │  │
│  │  │  ┌─────────────┐    coupling=0.1    ┌──────────────┐    │   │  │
│  │  │  │ sem stream  │◄──────────────────►│ depth stream │    │   │  │
│  │  │  │ (192,32,64) │   cross-modal mix  │ (192,32,64)  │    │   │  │
│  │  │  └─────────────┘                    └──────────────┘    │   │  │
│  │  └──────────────────────────────────────────────────────────┘   │  │
│  │                          × 2 blocks                             │  │
│  └──────────────────────────┬──────────────────────────────────────┘  │
│                             │                                         │
│  ┌──────────────────────────▼──────────────────────────────────────┐  │
│  │             DECODER STAGE 1 (32×64 → 64×128)                    │  │
│  │                                                                  │  │
│  │  sem ──→ TransConv2d 2× ──→ sem_up (192, 64, 128)              │  │
│  │  depth_feat ──→ TransConv2d 2× ──→ depth_up (192, 64, 128)     │  │
│  │                                                                  │  │
│  │  Depth (full res) ──→ DepthSkipBlock ──→ depth_skip (32,64,128) │  │
│  │                       [Sobel→Conv3x3→GN→GELU]                   │  │
│  │                                                                  │  │
│  │  Instance (full) ──→ InstanceSkipBlock ──→ inst_skip (16,64,128)│  │
│  │                      [Boundary+Dist→Conv3x3→GN→GELU]    (NEW)  │  │
│  │                                                                  │  │
│  │  ┌──────────────────────────────────────────────────────┐       │  │
│  │  │  FUSE: cat([sem_up, depth_skip, inst_skip]) → 1×1    │       │  │
│  │  │        (192 + 32 + 16 = 240) → (192)                 │       │  │
│  │  └──────────────────────────┬───────────────────────────┘       │  │
│  │                             ▼                                    │  │
│  │  WindowedAttentionBlock (8×8, 4 heads, shift=False)             │  │
│  │  sem (192, 64, 128) ◄──coupling──► depth_feat (192, 64, 128)   │  │
│  └──────────────────────────┬──────────────────────────────────────┘  │
│                             │                                         │
│  ┌──────────────────────────▼──────────────────────────────────────┐  │
│  │             DECODER STAGE 2 (64×128 → 128×256)                  │  │
│  │                                                                  │  │
│  │  sem ──→ TransConv2d 2× ──→ sem_up (192, 128, 256)             │  │
│  │  depth_feat ──→ TransConv2d 2× ──→ depth_up (192, 128, 256)    │  │
│  │                                                                  │  │
│  │  Depth (full res) ──→ DepthSkipBlock ──→ depth_skip (32,128,256)│  │
│  │                                                                  │  │
│  │  Instance (full) ──→ InstanceSkipBlock ──→ inst_skip(16,128,256)│  │
│  │                                                         (NEW)   │  │
│  │                                                                  │  │
│  │  FUSE: cat([sem_up, depth_skip, inst_skip]) → 1×1               │  │
│  │        (192 + 32 + 16 = 240) → (192)                            │  │
│  │                                                                  │  │
│  │  WindowedAttentionBlock (8×8, 4 heads, shift=True)              │  │
│  │  sem (192, 128, 256) ◄──coupling──► depth (192, 128, 256)      │  │
│  └──────────────────────────┬──────────────────────────────────────┘  │
│                             │                                         │
│  ┌──────────────────────────▼──────────────────────────────────────┐  │
│  │              CLASSIFICATION HEAD (128×256)                       │  │
│  │                                                                  │  │
│  │  sem (192, 128, 256) ──→ Conv1x1 ──→ logits (19, 128, 256)     │  │
│  └──────────────────────────┬──────────────────────────────────────┘  │
└─────────────────────────────┼─────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         LOSS COMPUTATION                                │
│                                                                         │
│  logits (19, 128, 256)                                                  │
│       │                                                                 │
│       ├──→ Distillation Loss (CE vs pseudo-labels)    × λ_dist (1.0→0.5)│
│       ├──→ Alignment Loss (feature-logit agreement)   × λ_align (0.25) │
│       ├──→ Prototype Loss (class-prototype contrast)  × λ_proto (0.025)│
│       ├──→ Entropy Loss (prediction sharpness)        × λ_ent (0.025)  │
│       │                                                                 │
│       │  instance_map (1, 128, 256) ← nearest-downsample from full-res │
│       │       │                                                         │
│       └──→ Instance Uniformity Loss (NEW)             × λ_iu (0.5)     │
│            ┌─────────────────────────────────────────────────┐          │
│            │ For each instance ID > 0:                       │          │
│            │   mask = (instance_map == id)                   │          │
│            │   probs = softmax(logits)[:, mask]  → (19, N)   │          │
│            │   mean_p = probs.mean(dim=1)        → (19,)     │          │
│            │   loss += var(probs - mean_p)                   │          │
│            │                                                 │          │
│            │ Penalizes different predictions within          │          │
│            │ the same instance → forces class consistency    │          │
│            └─────────────────────────────────────────────────┘          │
│                                                                         │
│  total = λ_dist·L_dist + λ_align·L_align + λ_proto·L_proto             │
│        + λ_ent·L_ent + λ_iu·L_uniform                                  │
└─────────────────────────────────────────────────────────────────────────┘


## InstanceSkipBlock Detail (NEW component)

```
Instance Mask (1, 512, 1024)          full-resolution integer IDs
        │
        ▼
  nearest-neighbor downsample          to decoder stage resolution
        │
        ▼
  (1, H, W) instance IDs
        │
        ├──→ Pixel-diff h/w ──→ Boundary Map (1, H, W)    [0/1 edges]
        │                              │
        │                              ├──→ channel 1
        │                              │
        │                              ▼
        │                     5× max_pool3x3 ──→ dilated
        │                              │
        │                         1 - dilated ──→ Distance Map (1, H, W)
        │                                              │
        │                                              ├──→ channel 2
        │                                              │
        └──────────────────────────────────────────────┘
                                       │
                                 cat → (2, H, W)
                                       │
                              Conv2d(2→16, 3×3)
                              GroupNorm(1, 16)
                              GELU
                                       │
                                       ▼
                           inst_skip (16, H, W)  ──→ fused with sem + depth_skip
```


## Evaluation Pipeline

```
Model Output: logits (19, 128, 256)
        │
        ▼
  argmax → pred_semantic (128, 256)
        │
        ▼
  upsample → pred_semantic (512, 1024)
        │
        │     Pre-computed Instance Masks
        │     pseudo_instance_spidepth/ (512, 1024)
        │              │
        ▼              ▼
  ┌─────────────────────────────────────┐
  │  Majority Voting (per instance):    │
  │                                     │
  │  for each instance_id > 0:          │
  │    mask = (inst_map == id)          │
  │    votes = pred_semantic[mask]      │
  │    class = mode(votes)             │
  │    if class ∈ THING_IDS:           │
  │      → panoptic segment            │
  │    else:                            │
  │      → merge into stuff             │
  └──────────────┬──────────────────────┘
                 │
                 ▼
  Panoptic Map + PQ / PQ_stuff / PQ_things evaluation
```


## Key Differences from Baseline P2-B

| Component | P2-B (baseline) | IC-C (this run) |
|-----------|----------------|-----------------|
| InstanceSkipBlock | ✗ | ✓ (16-dim at each stage) |
| Instance uniformity loss | ✗ | ✓ (λ=0.5) |
| Eval instances | Connected components | Pre-computed depth-guided |
| Instance input to model | None | Full-res mask → boundary+dist features |
| Fuse layer input | 192+32=224 | 192+32+16=240 |
| Parameters | 5,451,435 | 5,458,219 (+6,784 / +0.12%) |
