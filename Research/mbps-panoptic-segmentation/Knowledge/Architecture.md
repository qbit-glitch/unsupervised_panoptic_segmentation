---
type: knowledge
title: Pipeline Architecture
project: mbps-panoptic-segmentation
language: en
updated: 2026-03-29T13:40:24Z
---

# Pipeline Architecture

## High-Level Design

```
DINO ViT-S/8 backbone (frozen, 384-dim)
    |
    +-> DepthG semantic head -> 90-dim code space -> semantic labels
    |
    +-> Adaptive Projection Bridge (384 -> 192-dim)
    |       |
    |       +-> Depth Conditioning (FiLM)
    |       +-> Mamba2 SSD (bidirectional cross-modal scan)
    |
    +-> CutS3D instance head -> NCut + LocalCut 3D -> instance masks
    |
    +-> Stuff-Things Classifier (DBD + FCC + IDF cues)
    |
    +-> Panoptic Merge + CRF post-processing -> final panoptic map
```

## Current Best Pipelines

### DINOv3 Stage-3 (PQ=30.255%)
- DINOv3 ViT-B/16 backbone (official `facebookresearch/dinov3`)
- Cascade Mask R-CNN (CUPS Stage-2) + self-training (Stage-3)
- TTA with max scale 1.0 (1.25x OOM on 11GB GPUs)
- 27-class CAUSE + Hungarian matching metric

### UNet P2-B Attention (PQ_stuff=35.04)
- DINOv2 + depth -> DepthGuidedUNet -> refined semantics
- 2-stage progressive decoder: 32x64 -> 128x256
- Attention blocks >> conv blocks (11.5x slower but +0.27 PQ)
- 5.45M params, peaks at epoch 8

### Pseudo-Label Pipeline
1. **Semantics**: MiniBatchKMeans(k=80) on DINOv2/DINOv3 features -> Hungarian matching -> 19 classes
2. **Instances**: Mumford-Shah spectral clustering on depth+feature affinity (PQ_things=23.27)
   - Fallback: Sobel+CC on depth maps (PQ_things=19.41, 0.03s/img)
3. **Depth**: SPIdepth (self-supervised) or DA3 (foundation model, +1.5 PQ_things)

## Training Curriculum (4 Phases)

| Phase | Epochs | Losses | Key Events |
|-------|--------|--------|------------|
| A (Semantic) | 1-20 | L_semantic (STEGO + DepthG) | LR warmup, clusters form |
| B (Instance) | 21-40 | + L_instance (mask coherence) | Beta ramp, gradient projection |
| C (Bridge) | 41-60 | + L_bridge + L_consistency + L_pq | Full convergence |
| D (Self-train) | 61-75 | Pseudo-label training (3 rounds x 5 epochs) | EMA teacher |

## Key Constraints
- All v5e TPUs are spot-only (preemption recovery required)
- 11GB GPU limit for remote training — constrains TTA scales and batch sizes
- MPS autocast hangs with VisionMamba2 — must use float32
- Self-training with low-quality teacher HURTS (PQ drops -1.12)
