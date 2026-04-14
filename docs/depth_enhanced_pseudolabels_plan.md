# Plan: Depth-Enhanced Semantic Pseudo-Labels & CUPS Training Modifications

## Context

**Problem**: CUPS DINOv3 Stage-2/3 training degrades stuff quality. Raw k=80 pseudo-labels have PQ_stuff=32.08, but after Stage-3 training PQ_stuff drops to 31.29 (-0.79). Training excels at instances (+9.1 PQ_things) but trades away semantic precision.

**Goal**: Improve PQ_stuff through two complementary approaches:
- **(A) Better pseudo-labels**: Inject depth information into clustering so CUPS starts from higher-quality stuff boundaries
- **(B) Training modifications**: Preserve/enhance stuff quality during Stage-2/3 training

**Scope**: Full local ablation on M4 Pro Mac. Pseudo-label experiments are training-free or light-train on MPS. CUPS training modifications verified via 100-step mini-train on CPU.

---

## Part 1: Pseudo-Label Ablation (Approach A)

### Experiment Matrix

| ID | Method | Training | Baseline |
|----|--------|----------|----------|
| PL-0 | Baseline k=80 k-means on DINOv3 features | None | PQ_stuff=32.08 (existing) |
| PL-1 | Depth-weighted k-means (λ=0.1) | None | — |
| PL-2 | Depth-weighted k-means (λ=0.3) | None | — |
| PL-3 | Depth-weighted k-means (λ=0.5) | None | — |
| PL-4 | Trained DepthG projector (MLP 768→384→90, STEGO+depth loss) | ~30 min MPS | — |
| PL-5 | DepthG projector codes + depth-weighted k-means (λ=0.3) | ~30 min MPS | — |

### Step 1.1: Create `mbps_pytorch/generate_depth_weighted_kmeans.py`

**What**: Depth-weighted k-means that concatenates scaled depth features with DINOv3 features before standard MiniBatchKMeans.

**Algorithm**:
1. Load DINOv3 features (2048, 768) per image — L2-normalize (same as baseline)
2. Load DepthPro depth (512×1024) → downsample to (32×64) → flatten to (2048,)
3. Encode depth: sinusoidal encoding (12-dim) using existing `sinusoidal_depth_encoding()` from `mbps_pytorch/models/bridge/depth_conditioning.py:21-41`
4. Optionally add Sobel gradients (2-dim: gx, gy) → total depth features: 15-dim
5. Scale depth features by λ: `depth_feat_scaled = λ * depth_features`
6. Concatenate: `combined = np.concatenate([dino_feat, depth_feat_scaled], axis=-1)` → (2048, 768+15)
7. Run MiniBatchKMeans(k=80) on combined features
8. Save cluster IDs as PNG (same format as existing pipeline)

**Reuse from**: `generate_dinov3_kmeans.py` (structure, I/O, k-means fitting, cluster assignment)

**CLI args**: Same as `generate_dinov3_kmeans.py` plus:
- `--depth_subdir` (default: `depth_depthpro`)
- `--depth_weight` (λ, default: 0.3)
- `--depth_encoding` (choices: `sinusoidal`, `raw`, `sobel_sinusoidal`)

### Step 1.2: Create `mbps_pytorch/train_depthg_dinov3.py`

**What**: Train a DepthG projector on DINOv3 768-dim features with STEGO + depth correlation loss.

**Architecture**: Extend existing `DepthGHead` (from `mbps_pytorch/models/semantic/depthg_head.py:16-77`):
- Input: 768-dim DINOv3 features (was 384-dim ViT-S/8)
- MLP: Linear(768→384) → LN → ReLU → Linear(384→384) → LN → ReLU → Linear(384→90)
- `DepthGHead(input_dim=768, hidden_dim=384, code_dim=90)` — no code changes needed, already parameterized

**Loss**: Reuse existing implementations:
- `stego_loss()` from `mbps_pytorch/models/semantic/stego_loss.py:56-113`
- `depth_guided_correlation_loss()` from `mbps_pytorch/models/semantic/stego_loss.py:116-169`
- Combined: `L = L_stego + λ_depthg * L_depth_corr` (λ_depthg=0.3 default)

**Training loop**:
- Load pre-computed DINOv3 features (.npy) + DepthPro depth (.npy) per image
- Batch: 8 images × 2048 patches = 16384 tokens per batch
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Epochs: 20 (on 2975 train images → ~600 steps/epoch)
- Device: MPS (model is small — 3-layer MLP, ~1M params)
- Save: best checkpoint by STEGO loss, plus final centroids (k-means on learned codes)

**Post-training**: Run k=80 MiniBatchKMeans on the learned 90-dim codes → pseudo-labels

### Step 1.3: Evaluate all pseudo-label variants

**Script**: Reuse `mbps_pytorch/evaluate_cascade_pseudolabels.py` for each PL-{0..5}

**Metrics to compare**:
- PQ, PQ_stuff, PQ_things (panoptic quality)
- mIoU, per-class IoU (semantic quality)
- Per-class PQ for stuff classes (road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky)
- Boundary precision (depth-edge alignment with GT boundaries)

**Output**: JSON results per experiment + summary comparison table

---

## Part 2: Training Modification Ablation (Approach B)

### Experiment Matrix

| ID | Modification | On top of |
|----|-------------|-----------|
| TM-0 | Baseline CUPS Stage-2 (100 steps, CPU) | PL-0 labels |
| TM-1 | + Stuff preservation KD loss (λ=0.2) | TM-0 |
| TM-2 | + Depth FiLM conditioning in semantic head | TM-0 |
| TM-3 | + All (KD + FiLM) | TM-0 |

### Step 2.1: Modify `refs/cups/cups/model/modeling/roi_heads/semantic_seg.py`

**What**: Add stuff-preservation KD loss to `CustomSemSegFPNHead.losses()`.

**Modification** (lines 151-204): Add a `stuff_kd_loss` alongside `loss_sem_seg`:
```python
# In losses() method, after computing standard CE loss:
if self.stuff_kd_weight > 0 and pseudo_logits is not None:
    # KD: soft targets from pseudo-labels on stuff pixels only
    stuff_mask = (targets < 11) & (targets != self.ignore_value)  # stuff = trainIDs 0-10
    if stuff_mask.any():
        pred_log_probs = F.log_softmax(predictions[:, :, stuff_mask_h, stuff_mask_w], dim=1)
        target_probs = F.softmax(pseudo_logits[:, :, stuff_mask_h, stuff_mask_w] / self.kd_temperature, dim=1)
        kd_loss = F.kl_div(pred_log_probs, target_probs, reduction="batchmean")
        losses["loss_stuff_kd"] = kd_loss * self.stuff_kd_weight
```

**New `__init__` params**: `stuff_kd_weight` (default 0.0), `kd_temperature` (default 2.0)
**New `from_config` keys**: `cfg.MODEL.SEM_SEG_HEAD.STUFF_KD_WEIGHT`, `cfg.MODEL.SEM_SEG_HEAD.KD_TEMPERATURE`

**How pseudo_logits are provided**: The pseudo-label PNG is one-hot → convert to soft logits in the data loader or pass as one-hot targets. Simplest: use one-hot pseudo-labels with temperature=1.0 (hard KD).

### Step 2.2: Create `refs/cups/cups/model/modeling/roi_heads/depth_film_semantic_head.py`

**What**: Depth-conditioned variant of `CustomSemSegFPNHead` with FiLM modulation.

**Architecture**: Subclass `CustomSemSegFPNHead`, override `layers()`:
```python
class DepthFiLMSemSegHead(CustomSemSegFPNHead):
    """Semantic head with depth FiLM conditioning at each FPN scale."""
    
    def __init__(self, ..., depth_channels=15):
        super().__init__(...)
        # FiLM generators: one per FPN scale
        self.film_generators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(depth_channels, conv_dims, 1),
                nn.ReLU(),
                nn.Conv2d(conv_dims, conv_dims * 2, 1),  # gamma + beta
            )
            for _ in self.in_features
        ])
        # Depth encoder: sinusoidal + Sobel → depth_channels
        self.depth_encoder = DepthEncoder(out_channels=depth_channels)
    
    def layers(self, features, depth=None):
        depth_feat = self.depth_encoder(depth)  # (B, depth_channels, H, W)
        for i, f in enumerate(self.in_features):
            feat_i = self.scale_heads[i](features[f])
            if depth is not None:
                # Resize depth_feat to match feat_i spatial dims
                d = F.interpolate(depth_feat, size=feat_i.shape[-2:], mode="bilinear")
                film_params = self.film_generators[i](d)
                gamma, beta = film_params.chunk(2, dim=1)
                feat_i = feat_i * (1 + gamma) + beta
            if i == 0:
                x = feat_i
            else:
                x = x + feat_i
        return self.predictor(x)
```

**Depth encoder**: Reuse sinusoidal encoding pattern from `mbps_pytorch/models/bridge/depth_conditioning.py:21-41`:
- Input: raw depth (B, 1, H, W)
- Sinusoidal encoding: 6 frequency bands → 12 channels
- Sobel gradients: 2 channels (gx, gy)
- Raw depth: 1 channel
- Total: 15 channels

### Step 2.3: Modify `refs/cups/cups/model/modeling/meta_arch/panoptic_fpn.py`

**What**: Pass depth maps through to the semantic head.

**Changes to `forward()` (line 88-148)**:
1. Extract depth from `batched_inputs`: `depth = [x["depth"].to(self.device) for x in batched_inputs]`
2. Stack into tensor, pad with `ImageList.from_tensors()`
3. Pass to `self.sem_seg_head(features, gt_sem_seg, pixel_weights=pixel_weights, depth=depth_tensor)`

**Changes to `inference()` (line 150-199)**: Same depth extraction, pass to sem_seg_head.

### Step 2.4: Modify data pipeline to load depth maps

**File**: `refs/cups/cups/pl_model_pseudo.py` (Stage-2 data loading)

**Changes**: In the dataset/data loader, for each training image:
1. Load corresponding DepthPro depth .npy file
2. Normalize to [0, 1]
3. Add as `"depth"` key in the batched_inputs dict

### Step 2.5: Add config keys

**File**: `refs/cups/cups/config.py`

**New keys**:
```python
# Stuff preservation
cfg.MODEL.SEM_SEG_HEAD.STUFF_KD_WEIGHT = 0.0  # 0 = disabled
cfg.MODEL.SEM_SEG_HEAD.KD_TEMPERATURE = 2.0

# Depth FiLM
cfg.MODEL.SEM_SEG_HEAD.USE_DEPTH_FILM = False
cfg.MODEL.SEM_SEG_HEAD.DEPTH_CHANNELS = 15

# Depth data
cfg.DATA.DEPTH_SUBDIR = ""  # e.g., "depth_depthpro"
```

### Step 2.6: Mini-train protocol

**Device**: CPU (Detectron2 ROI ops don't support MPS, model_vitb.py falls back to CPU)
**Images**: 50 train images (tiny subset), 256×512 resolution
**Batch size**: 1, no gradient accumulation
**Steps**: 100 per experiment
**Estimated time**: ~15-25 min per experiment (5-15 sec/step on M4 Pro CPU)

**What to monitor**:
- `loss_sem_seg` (standard semantic CE loss)
- `loss_stuff_kd` (stuff preservation KD, TM-1/TM-3 only)
- `loss_cls`, `loss_box_reg`, `loss_mask` (instance losses — should be stable)
- Relative stuff vs things loss magnitude across experiments

**Success criteria**: Stuff-related losses are lower and more stable in TM-1/TM-2/TM-3 than TM-0 baseline. No regression in instance losses.

---

## Files Summary

### New files (5)
| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `mbps_pytorch/generate_depth_weighted_kmeans.py` | Depth-weighted k-means pseudo-labels | ~220 |
| `mbps_pytorch/train_depthg_dinov3.py` | Train DepthG projector on DINOv3+DepthPro | ~300 |
| `refs/cups/cups/model/modeling/roi_heads/depth_film_semantic_head.py` | Depth FiLM semantic head | ~150 |
| `scripts/run_pseudolabel_ablation.sh` | Run all PL experiments + evaluation | ~80 |
| `scripts/run_training_mod_ablation.sh` | Run all TM mini-train experiments | ~60 |

### Modified files (4)
| File | Change | Lines changed (est.) |
|------|--------|---------------------|
| `refs/cups/cups/model/modeling/roi_heads/semantic_seg.py` | Add stuff KD loss in `losses()`, new init params | ~30 |
| `refs/cups/cups/model/modeling/meta_arch/panoptic_fpn.py` | Pass depth to sem_seg_head in `forward()` and `inference()` | ~15 |
| `refs/cups/cups/config.py` | Add STUFF_KD_WEIGHT, DEPTH_FILM, DEPTH_SUBDIR keys | ~10 |
| `refs/cups/cups/pl_model_pseudo.py` | Load depth .npy in data pipeline | ~20 |

### Reused existing code
| Component | Source | Used in |
|-----------|--------|---------|
| `sinusoidal_depth_encoding()` | `mbps_pytorch/models/bridge/depth_conditioning.py:21-41` | depth_weighted_kmeans, depth_film_head |
| `DepthGHead` class | `mbps_pytorch/models/semantic/depthg_head.py:16-77` | train_depthg_dinov3 (input_dim=768) |
| `stego_loss()` | `mbps_pytorch/models/semantic/stego_loss.py:56-113` | train_depthg_dinov3 |
| `depth_guided_correlation_loss()` | `mbps_pytorch/models/semantic/stego_loss.py:116-169` | train_depthg_dinov3 |
| `find_feature_files()` | `mbps_pytorch/generate_dinov3_kmeans.py:39-48` | depth_weighted_kmeans |
| `fit_kmeans()` / `assign_clusters()` | `mbps_pytorch/generate_dinov3_kmeans.py:51-121` | depth_weighted_kmeans |
| Eval pipeline | `mbps_pytorch/evaluate_cascade_pseudolabels.py` | All PL experiments |

---

## Execution Order

```
Phase 1 (Parallel — no dependencies):
├── [1.1] Create generate_depth_weighted_kmeans.py
├── [1.2] Create train_depthg_dinov3.py
└── [2.1-2.5] Implement all Approach B code changes

Phase 2 (Sequential — depends on Phase 1):
├── [PL-1..3] Run depth-weighted k-means (λ=0.1, 0.3, 0.5) — ~10 min each
├── [PL-4] Train DepthG projector — ~30 min MPS
├── [PL-5] Depth-weighted k-means on DepthG codes — ~10 min
└── [PL eval] Evaluate all pseudo-label sets — ~5 min each

Phase 3 (Sequential — depends on Phase 1):
├── [TM-0] Baseline CUPS mini-train — ~20 min CPU
├── [TM-1] + Stuff KD loss — ~20 min CPU
├── [TM-2] + Depth FiLM head — ~25 min CPU
└── [TM-3] + All combined — ~25 min CPU

Phase 4: Analysis
└── Compare all results, identify best combination of (PL-X, TM-Y)
```

## Verification

### Pseudo-labels (Part 1)
```bash
# For each PL variant:
python mbps_pytorch/evaluate_cascade_pseudolabels.py \
    --cityscapes_root /path/to/cityscapes \
    --semantic_subdir <output_subdir> \
    --split val --eval_size 512 1024
# Compare PQ_stuff across PL-0..PL-5
```

### Training modifications (Part 2)
```bash
# Check loss curves converge (no NaN, no explosion):
grep "loss_sem_seg\|loss_stuff_kd" logs/tm_*.log
# Verify stuff KD loss decreases over 100 steps
# Verify instance losses (loss_cls, loss_mask) are not degraded
```

### Smoke tests before full runs
- `generate_depth_weighted_kmeans.py --k 10 --sample_frac 0.01` on 5 images
- `train_depthg_dinov3.py --epochs 1 --max_images 10` for quick sanity check
- CUPS mini-train: verify model builds and runs 1 step on CPU without error

### Pre-requisites to check
- [ ] DINOv3 features exist: `ls cityscapes/dinov3_features/train/`
- [ ] DepthPro depth exists: `ls cityscapes/depth_depthpro/train/`
- [ ] Detectron2 runs on CPU: `python -c "import detectron2; print('OK')"`
- [ ] Existing k=80 baseline results available for comparison
