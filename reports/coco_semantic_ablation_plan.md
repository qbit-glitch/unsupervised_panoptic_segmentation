# COCO-Stuff-27 Semantic Pseudo-Label Quality Ablation

## Context

The depth ablation study (complete) showed that on COCO-Stuff-27, depth-guided instance splitting barely helps (+0.77 PQ_things over CC-only) because **semantic pseudo-label quality is the bottleneck** (mIoU=18.3%). To train a competitive COCO model for cross-dataset evaluation (vs U2Seg), we need mIoU >= 35%.

Current baseline: MiniBatchKMeans(k=80) on 32x32 DINOv3 ViT-L/16 features (1024-dim, NOT L2-normalized) -> Hungarian matching -> 27 classes -> mIoU=18.3%.

**Goal**: Ablate 3 approaches to improve COCO pseudo-semantic quality, then combine best semantics with depth-guided instances for a full COCO panoptic pipeline.

---

## Quick Win: L2 Normalization (Before any Path)

The current `generate_coco_pseudo_semantics.py` does NOT L2-normalize features before k-means. The Cityscapes version does. This alone could yield +2-5% mIoU.

**File**: `mbps_pytorch/generate_coco_pseudo_semantics.py`
**Change**: Add `feat = feat / (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-8)` before concatenation
**Output**: `pseudo_semantic_k80_norm/val2017/`
**Time**: 5 minutes

---

## Path 3: CUPS-Style Learned ClusterLookup (Start Here — Fastest)

### What
Train a lightweight projector + learnable cluster centers (ClusterLookup) on pre-extracted 32x32 DINOv3 features using STEGO-style feature correlation loss. No live backbone inference needed.

### Why First
- Operates on existing features (no extraction step)
- Training is fast (~30 min on MPS for 501 images)
- Directly tests whether learning beats raw k-means

### Architecture
```
Pre-extracted DINOv3 features (1024, 1024) per image
    -> reshape to (1, 1024, 32, 32)
    -> Projector: Conv1x1(1024, 90) + ReLU + Conv1x1(90, 90)
    -> ClusterLookup: nn.Parameter(K, 90), cosine similarity assignment
```

### Losses
1. **STEGO correlation loss**: Encourage projected code similarity to match backbone feature similarity
   - Use existing `mbps_pytorch/models/semantic/stego_loss.py`
   - `loss = -code_corr * (feat_corr - shift)`
2. **Cluster loss**: `-(cluster_probs * inner_products).sum(1).mean()`
   - From `refs/cups/external/depthg/src/modules.py` ClusterLookup

### Script to Create
`mbps_pytorch/train_coco_cluster_lookup.py`

### Hyperparameter Sweep
| Parameter | Values |
|-----------|--------|
| K (clusters) | 80, 150, 300 |
| Projector dim | 90, 128 |
| LR | 1e-3, 5e-4 |
| STEGO shift | 0.05, 0.10 |
| Cluster temp (alpha) | 2.0, 5.0 |
| Epochs | 20, 30 |
| Training data | val (501), val+train (3502) |

### Output
- `pseudo_semantic_cups_k{K}/val2017/` — PNG labels
- Evaluate: mIoU, per-class IoU, then PQ via `sweep_coco_depth_comparison.py`

### Expected: mIoU 22-28%
### Time: ~4 hours (implement + sweep + eval)

---

## Path 1: Higher Resolution DINOv3 Features

### What
Re-extract DINOv3 features at 64x64 patch grid (1024x1024 input) instead of 32x32 (512x512 input). Then re-cluster.

### Why
32x32 patches = each covers ~14x14 pixels. Very coarse for small COCO objects. 64x64 doubles spatial resolution, giving 4x more patches per image.

### DINOv3 Compatibility
DINOv3 uses RoPE (base=100) which supports variable input sizes. The HuggingFace model accepts arbitrary resolutions.

### Script to Modify
`mbps_pytorch/extract_dinov3_features.py` — create COCO variant

### New Script
`mbps_pytorch/extract_dinov3_features_coco_hires.py`

Key changes:
- Input: resize COCO images to 1024x1024 (pad shorter side)
- Output: (4096, 1024) per image = 64x64 patches x 1024-dim
- Save to `coco/dinov3_features_64x64/val2017/`
- Model: `facebook/dinov3-vitl16-pretrain-lvd1689m` (same ViT-L/16)
- batch_size=1 (memory: ~6-8 GB per image at 1024x1024)
- Storage: ~16 MB/image x 501 = ~8 GB total

### Clustering
Modify `generate_coco_pseudo_semantics.py` for 64x64:
- `--patch_grid 64`
- K values: 80, 150, 300 (more patches may need more clusters)
- L2-normalize features before k-means
- Output: `pseudo_semantic_hires_k{K}/val2017/`

### Combinations
- **1a**: k-means on 64x64 features (baseline for Path 1)
- **1b**: Path 3 ClusterLookup on 64x64 features (Path 1 + Path 3 synergy)

### Expected: mIoU 24-32%
### Time: ~6 hours (3-4h extraction + 1h clustering + 1h eval)

---

## Path 2: CAUSE Contrastive Learning on COCO

### What
Full CAUSE pipeline (codebook + contrastive head + cluster probe) trained on COCO features. This is the most principled approach — learn semantic-aware projections with contrastive self-supervision.

### Why Last
Most complex, longest training, but highest potential ceiling.

### Existing Code
`mbps_pytorch/train_cause_dinov3.py` — complete pipeline for Cityscapes DINOv3 ViT-L/16

### What Needs to Change for COCO
1. **Dataset loader**: Replace `CityscapesCAUSE` with `COCOCAUSE` — load COCO images, apply random crops
2. **n_classes**: Already 27 (same as COCO-Stuff-27!)
3. **Training data**: Use 3001 train images with pre-extracted features, OR live backbone inference on raw images
4. **Backbone**: Same `facebook/dinov3-vitl16-pretrain-lvd1689m` (1024-dim)

### Two Training Modes

**Mode A: Feature-based (faster, no augmentation)**
- Load pre-extracted .npy features directly
- Skip backbone forward pass — huge speed gain
- Loss: spatial sampling of patches within 32x32 grid
- Loses random crop augmentation but 10x faster

**Mode B: Image-based (slower, full augmentation)**
- Load raw COCO images, apply RandomResizedCrop(320)
- Live DINOv3 forward pass each batch (frozen backbone)
- Full CAUSE augmentation pipeline
- Uses 3001 train images (comparable to Cityscapes 2975)

### Script to Create
`mbps_pytorch/train_cause_coco.py` — adapted from `train_cause_dinov3.py`

### Architecture (identical to Cityscapes CAUSE)
```
DINOv3 ViT-L/16 features (1024-dim, frozen)
    -> Codebook: 2048 entries, modularity loss (Stage 2, 10-15 epochs)
    -> Segment head: TRDecoder (1024 -> 90-dim code space)
    -> Projection head: Conv1x1 (90 -> 2048) for contrastive loss
    -> Cluster probe: nn.Parameter(27, 90) — learnable class centroids
    -> Contrastive loss: codebook-quantized anchor + EMA feature bank
    -> Cluster loss: cosine similarity to cluster probe
```

### Training Schedule
| Stage | Epochs | LR | Batch | Est. Time (MPS) |
|-------|--------|-----|-------|-----------------|
| Stage 2 (codebook) | 15 | 1e-3 | 4 | ~1 hour |
| Stage 3 (heads) | 40-60 | 5e-5 | 2-4 | ~4-6 hours |

### Inference
```python
seg_feat = segment.head_ema(feat)  # (B, N, 90)
cluster_logits = (seg_feat @ cluster_probe.T) / temp
pred = argmax(cluster_logits)  # Direct 27-class prediction
```

If using overclustered ClusterLookup instead of cluster_probe:
- Need Hungarian matching (same as k-means baseline)

### Hyperparameters to Sweep
| Parameter | Values |
|-----------|--------|
| num_codebook | 2048, 4096 |
| head_epochs | 40, 60 |
| Training mode | Feature-based, Image-based |
| Training data | val only (501), train (3001), train+val (3502) |

### Combinations
- **2a**: CAUSE on 32x32 features (direct comparison to Path 3)
- **2b**: CAUSE on 64x64 features from Path 1 (best possible)

### Expected: mIoU 30-40%
### Time: ~12-15 hours (implement + train + eval)

---

## Shared Evaluation Framework

### Script to Create
`mbps_pytorch/evaluate_coco_semantic_ablation.py`

For each method:
1. **mIoU** (full 500 val images, all 27 classes, Hungarian matching)
2. **Per-class IoU** breakdown (12 things + 15 stuff)
3. **PQ with depth splitting**: Run with DA2-Large (best COCO depth) at τ=0.08, MA=1000
4. **Quick depth sweep**: τ in {0.03, 0.05, 0.08, 0.15, 0.30}, MA=1000
5. Save JSON results for comparison table

### Evaluation Baseline
| Method | mIoU | PQ | PQ_things |
|--------|------|-----|-----------|
| k-means k=80 (no norm) | 18.3% | 12.60 | 13.76 |

---

## Execution Order

### Day 1: Quick Win + Path 3
1. L2-norm baseline (5 min) — expected mIoU ~20-23%
2. Implement `train_coco_cluster_lookup.py` (~2 hrs)
3. Train + sweep Path 3 configs (~2 hrs)
4. Evaluate all Path 3 variants (~30 min)

### Day 2: Path 1
5. Implement `extract_dinov3_features_coco_hires.py` (~1 hr)
6. Extract 64x64 features for 501 val images (~3-4 hrs)
7. K-means clustering at 64x64 (~30 min)
8. ClusterLookup (Path 3) on 64x64 features (~1 hr)
9. Evaluate all Path 1 variants (~30 min)

### Day 3-4: Path 2
10. Implement `train_cause_coco.py` (~3 hrs)
11. Train codebook Stage 2 (~1 hr)
12. Train heads Stage 3 (~4-6 hrs, run overnight)
13. Generate pseudo-labels + evaluate (~1 hr)
14. Optional: CAUSE on 64x64 features

### Day 5: Analysis
15. Compile comparison table (all methods)
16. Select best method
17. Update `reports/depth_model_ablation_study.md` with COCO semantic ablation
18. If mIoU >= 35%: proceed to CUPS Stage-2 training on COCO

---

## Critical Files

### Existing (to reuse)
| File | Purpose |
|------|---------|
| `mbps_pytorch/generate_coco_pseudo_semantics.py` | Baseline k-means + Hungarian matching + eval |
| `mbps_pytorch/models/semantic/stego_loss.py` | STEGO correlation loss (Path 3) |
| `mbps_pytorch/train_cause_dinov3.py` | Full CAUSE pipeline for DINOv3 (Path 2 template) |
| `mbps_pytorch/extract_dinov3_features.py` | Feature extraction (Path 1 template) |
| `mbps_pytorch/sweep_coco_depth_comparison.py` | PQ evaluation with depth |
| `refs/cause/modules/segment_module.py` | Cluster, contrastive loss, codebook |
| `refs/cause/modules/segment.py` | Segment_TR head architecture |
| `refs/cups/external/depthg/src/modules.py` | ClusterLookup class |

### New files to create
| File | Purpose |
|------|---------|
| `mbps_pytorch/train_coco_cluster_lookup.py` | Path 3: ClusterLookup training |
| `mbps_pytorch/extract_dinov3_features_coco_hires.py` | Path 1: 64x64 feature extraction |
| `mbps_pytorch/train_cause_coco.py` | Path 2: Full CAUSE on COCO |
| `mbps_pytorch/evaluate_coco_semantic_ablation.py` | Unified evaluation for all methods |

### Data locations
| Data | Path |
|------|------|
| COCO root | `/Users/qbit-glitch/Desktop/datasets/coco/` |
| DINOv3 features (32x32) | `coco/dinov3_features/{train,val}2017/` (501 val + 3001 train) |
| DINOv3 features (64x64) | `coco/dinov3_features_64x64/val2017/` (to create) |
| Current pseudo-labels | `coco/pseudo_semantic_k80/val2017/` |
| COCO panoptic GT | `coco/annotations/panoptic_val2017.json` + `panoptic_val2017/` |
| Depth maps | `coco/depth_da2_large/val2017/`, `coco/depth_dav3/val2017/` |

---

## Verification

1. **Quick Win**: L2-norm k-means produces mIoU > 18.3% (should be ~20-23%)
2. **Path 3**: At least one ClusterLookup config beats k-means baseline by >= 3% mIoU
3. **Path 1**: 64x64 features produce higher mIoU than 32x32 at same k
4. **Path 2**: CAUSE achieves mIoU >= 30% (competitive with published results)
5. **Best combo**: Some method reaches mIoU >= 35%, enabling viable COCO panoptic training
6. **PQ improvement**: Best semantics + DA2-Large depth produces PQ > 14 (current best: 12.72)
