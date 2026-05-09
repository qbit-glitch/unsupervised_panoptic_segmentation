# MSDA: Multi-Scale Dense Adapter for Overclustering

## Problem

Stage 0 overclustering (DINOv3 ViT-L/16 + spherical k-means k=100) achieves PQ=20.9 / mIoU=39.1, but 7 classes are completely dead (0% IoU): wall, fence, traffic light, rider, truck, train, motorcycle.

Two distinct failure modes:
- **Resolution ceiling**: traffic light, fence, wall — smaller than 16×16 patches
- **Rarity ceiling**: motorcycle, truck, train, rider — too few examples for k-means

Six clustering-level interventions tested (PPAP, PCL+Sinkhorn, SDCluster, etc.) — all failed. None transformed the features themselves.

## Solution

Train a self-supervised adapter on frozen DINOv3 features + monocular depth to produce transformed features that cluster better, especially for dead classes.

## Ablation Matrix

Two-phase ablation (8 runs instead of 16):
- **Phase 1**: 4 architectures × 1 fixed loss (hybrid) → find best architecture
- **Phase 2**: 1 best architecture × 4 losses → find best loss

### Architectures

| ID | Name | Key Idea | Params |
|----|------|----------|--------|
| A | Multi-Scale Conv | TransposedConv upsample + residual conv blocks + depth FiLM | ~5-25M |
| B | Transformer Pyramid | Multi-scale feature pyramid + transformer blocks + depth cross-attn | ~10-50M |
| C | Slot Attention | Conv encoder + slot attention decoder (replaces k-means) | ~5-30M |
| D | Conv + Transformer | Conv upsampler + transformer blocks at 32×64 only | ~8-40M |

### Losses

| ID | Name | Signal | Existing Code |
|----|------|--------|---------------|
| 1 | Depth-guided contrastive | DepthPro depth proximity → attract/repel | `stego_loss.py:depth_guided_correlation_loss` (partial) |
| 2 | STEGO correspondence | DINOv3 KNN → feature correlation distillation | `stego_loss.py:stego_loss` (complete) |
| 3 | SwAV + Sinkhorn | Online prototypes + balanced assignment | New |
| 4 | Hybrid | Loss 1 + Loss 3 weighted combination | New |

## Data

Pre-cached on disk (no re-extraction needed):
- DINOv3 ViT-L/16 features: `cityscapes/dinov3_features_vitl16/{train,val}/{city}/*.npy` — shape (2048, 1024)
- DepthPro depth: `cityscapes/depth_depthpro/{train,val}/{city}/*.npy` — shape (512, 1024)
- k=80 centroids: `cityscapes/pseudo_semantic_raw_k80/kmeans_centroids.npz`
- Total: 2975 train, 500 val images

## Architecture Details

### A: Multi-Scale Conv Adapter

```
DINOv3 feats (2048, 1024) → reshape (1024, 32, 64)
    ↓
TransposedConv: (1024, 32, 64) → (512, 64, 128) → (256, 128, 256)
    ↓
Scale 1: 2× ResBlock(1024ch) @ 32×64 + DepthFiLM
Scale 2: 2× ResBlock(512ch) @ 64×128 + DepthFiLM
Scale 3: 2× ResBlock(256ch) @ 128×256 + DepthFiLM
    ↓
Downsample all to 32×64, concat (1024+512+256=1792ch) → 1×1 Conv → D_out
    ↓
L2-normalize → output (D_out, 32, 64)
```

Depth conditioning: sinusoidal encoding (16-dim) → MLP → (scale, shift) per conv block (FiLM).

### B: Transformer Pyramid

```
DINOv3 feats (2048, 1024) → reshape (1024, 32, 64)
    ↓
3-scale pyramid via strided conv + transposed conv
    ↓
Scale 1 (32×64, 2048 tokens): 4× TransformerBlock(dim=1024, heads=16)
Scale 2 (64×128, 8192 tokens): 2× TransformerBlock(dim=512, heads=8)
Scale 3 (128×256, 32768 tokens): SKIP transformer (too expensive on MPS)
    ↓
Depth cross-attention at each scale (depth as K/V)
    ↓
FPN merge → D_out
```

Note: Scale 3 uses conv-only processing (no attention at 32K tokens).

### C: Slot Attention Adapter

```
DINOv3 feats (2048, 1024) → reshape (1024, 32, 64)
    ↓
Conv encoder → (256, 128, 256) via transposed conv
    ↓
Flatten: 32768 tokens × 256-dim
    ↓
Slot Attention: K=200 slots, T=7 iterations
  - slot_dim=256, mlp_hidden=512
  - Depth-modulated attention: add depth PE to keys
    ↓
Per-pixel slot assignment → soft cluster labels
    ↓
(No k-means needed — slot assignment IS the clustering)
```

### D: Conv + Transformer Hybrid

```
DINOv3 feats (2048, 1024) → reshape (1024, 32, 64)
    ↓
Conv upsample: → (512, 64, 128) → (256, 128, 256)
    ↓
High-res conv blocks: 2× ResBlock @ 64×128, 2× ResBlock @ 128×256
    ↓
Low-res transformer: 4× TransformerBlock @ 32×64 (2048 tokens, dim=1024)
    ↓
FPN merge: transformer output + conv features → D_out
```

## Loss Details

### Loss 1: Depth-Guided Contrastive (enhanced)

Extends existing `depth_guided_correlation_loss`:
- **Attract**: patches at similar depth + spatial proximity → high cosine similarity
- **Repel**: patches across depth discontinuities → low similarity
- **Boundary emphasis**: weight pairs near Sobel depth edges 2× higher
- **Scale-aware**: depth sigma varies by scale (coarse=large sigma, fine=small sigma)

### Loss 2: STEGO Correspondence

Reuse existing `stego_loss` with DINOv3 features as the teacher signal.

### Loss 3: SwAV + Sinkhorn

- K=200 learnable prototypes (nn.Parameter)
- Forward: project features → compute similarity with prototypes → Sinkhorn normalize
- Loss: cross-entropy between student assignments (from augmented view) and teacher assignments (from original view)
- Sinkhorn iterations: 3 (balanced assignment, prevents dead prototypes)
- Temperature: 0.1

### Loss 4: Hybrid

`L_hybrid = α * L_depth_contrastive + β * L_swav_sinkhorn`
Default: α=1.0, β=1.0

## Training Configuration

- Optimizer: AdamW, lr=1e-4, weight_decay=0.01
- Scheduler: CosineAnnealing, T_max=50 epochs
- Batch size: 8 (M4 Pro MPS)
- Epochs: 50
- Augmentation: random crop (224×448 from 512×1024), horizontal flip
- Feature augmentation: dropout(0.1), noise(σ=0.01) on input features
- Gradient clipping: 1.0
- Seeds: 42
- Checkpoint: save best (val loss) + every 10 epochs

## Evaluation

After training each adapter:
1. Extract transformed features for all train+val images
2. Run spherical k-means k=100 on train features
3. Map clusters to 19 classes via Hungarian matching (GT)
4. Compute PQ, mIoU, per-class IoU (focus on 7 dead classes)
5. Compare to baseline: PQ=20.9, mIoU=39.1

## Codebase Structure

```
mbps_pytorch/msda/
├── __init__.py              # Factory + registry
├── architectures/
│   ├── __init__.py          # ARCH_REGISTRY
│   ├── base.py              # BaseAdapter ABC
│   ├── conv_adapter.py      # Arch A
│   ├── transformer_pyramid.py # Arch B
│   ├── slot_adapter.py      # Arch C
│   └── conv_transformer.py  # Arch D
├── losses/
│   ├── __init__.py          # LOSS_REGISTRY
│   ├── depth_contrastive.py # Loss 1
│   ├── stego.py             # Loss 2 (wraps stego_loss.py)
│   ├── swav_sinkhorn.py     # Loss 3
│   └── hybrid.py            # Loss 4
├── dataset.py               # CachedFeatureDataset
├── train.py                 # Training entry point
├── evaluate.py              # k-means + PQ/mIoU eval
└── run_ablation.sh          # 8-run ablation script
```

## Success Criteria

- **Minimum**: At least one dead class recovers from 0% to >5% IoU
- **Target**: Overall PQ > 22.0 (baseline 20.9) with ≥2 dead classes recovered
- **Stretch**: PQ > 24.0 with ≥4 dead classes recovered
