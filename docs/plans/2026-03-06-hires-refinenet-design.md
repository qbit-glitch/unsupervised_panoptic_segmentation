# HiRes RefineNet: High-Resolution Semantic Refinement at 128x256

## Problem Statement

CSCMRefineNet operating at 32x64 patch resolution improves semantic quality (mIoU 50% -> 55.3%, PQ_stuff 32.08 -> 33.38) but structurally degrades PQ_things (19.41 -> 17.10). Four ablation studies (TAD, BPL, ASPP-lite, TAD+BPL) confirmed this is a resolution bottleneck, not a loss or receptive field problem. At 32x64, small thing instances (bicycles, riders, motorcycles) collapse below the resolution limit.

## Hypothesis

Operating at 128x256 (4x higher resolution) preserves thing-object spatial precision while maintaining the semantic improvement from cross-modal refinement. The 128x256 resolution provides 16x more spatial positions than 32x64, enabling the network to distinguish adjacent thing instances that previously merged.

## Design

### Two-Phase Ablation

**Phase 1: Upsampling Strategy** (3 runs, Conv2d architecture fixed)

Ablate how DINOv2 features (32x64) are upsampled to 128x256 before refinement:

| Run | Strategy | Mechanism | Extra Params |
|-----|----------|-----------|-------------|
| U-B | Transposed Conv | 2-stage ConvTranspose2d (32x64 -> 64x128 -> 128x256) | ~300K |
| U-A | Bilinear | F.interpolate(scale_factor=4, mode='bilinear') | 0 |
| U-C | PixelShuffle | Conv2d(192, 192*16, 1) + PixelShuffle(4) | ~590K |

**Phase 2: Architecture** (5 runs, best upsampling from Phase 1)

Ablate the refinement block architecture at 128x256 resolution:

| Run | Architecture | Description | Source |
|-----|-------------|-------------|--------|
| A-Conv | Conv2d | CoupledConvBlock (depthwise-separable 3x3 + cross-gating) | Existing |
| A-Attn | Efficient Self-Attention | Windowed attention (8x8 windows, Swin-style) | New |
| A-VM2 | VisionMamba2 | Bidirectional Mamba2 SSM scan | Existing |
| A-SM | Spatial Mamba | VMamba SS2D 4-way cross-scan | Existing (cross_scan mode) |
| A-MOut | MambaOut | Gated CNN without SSM (Yu et al., 2024) | New |

### Architecture: HiResRefineNet

```
DINOv2 ViT-B/14 (frozen, 768-dim, 32x64)
     |
     +-- SemanticProjection (768 -> 192, 1x1 conv) -----> sem (192, 32x64)
     |                                                        |
     +-- DepthFeatureProjection (768+depth -> 192, FiLM) --> depth_feat (192, 32x64)
     |                                                        |
     |                                               [Upsampling Module]
     |                                          32x64 -> 128x256 (both streams)
     |                                                        |
     |                                               [Refinement Blocks x4]
     |                                          Conv2d / Attention / Mamba2 / SS2D / MambaOut
     |                                                        |
     |                                               Classification Head
     |                                          Conv2d(192, 19, 1) at 128x256
     |                                                        |
     v                                                        v
  Depth (1024x2048)                              Refined logits (19, 128x256)
  downsample to 128x256                          upsample to 1024x2048 for eval
```

### Upsampling Module Details

**B: Transposed Convolution (recommended)**
```
Input: (B, 192, 32, 64)
  -> ConvTranspose2d(192, 192, kernel=4, stride=2, padding=1) -> (B, 192, 64, 128)
  -> GroupNorm(1, 192) + GELU
  -> ConvTranspose2d(192, 192, kernel=4, stride=2, padding=1) -> (B, 192, 128, 256)
  -> GroupNorm(1, 192) + GELU
Output: (B, 192, 128, 256)
```

**A: Bilinear Interpolation**
```
Input: (B, 192, 32, 64)
  -> F.interpolate(scale_factor=4, mode='bilinear', align_corners=False)
Output: (B, 192, 128, 256)
```

**C: PixelShuffle**
```
Input: (B, 192, 32, 64)
  -> Conv2d(192, 192 * 16, 1)  # expand channels for 4x4 spatial rearrangement
  -> PixelShuffle(4)            # (B, 192, 128, 256)
  -> GroupNorm(1, 192) + GELU
Output: (B, 192, 128, 256)
```

### Refinement Block Details (all at 128x256)

**Conv2d** — Existing CoupledConvBlock, unchanged. 128x256 = 32K spatial positions. Each DW-Conv 3x3 processes 16x more pixels than at 32x64.

**Efficient Self-Attention (Windowed)** — New block:
- Partition 128x256 into 8x8 non-overlapping windows = 512 windows of 64 tokens
- Standard multi-head self-attention within each window (4 heads, head_dim=48)
- Shifted window attention (Swin-style) on alternating blocks
- Cross-modal gating same as CoupledConvBlock (1x1 conv + sigmoid)
- Pre-norm (LayerNorm), residual connection, FFN (expand 2x)

**VisionMamba2 (Bidirectional)** — Existing CoupledMambaBlock with scan_mode="bidirectional". 32K tokens with O(N) linear scan.

**Spatial Mamba (SS2D Cross-Scan)** — Existing CoupledMambaBlock with scan_mode="cross_scan". 4-way directional scan (row-fwd, row-bwd, col-fwd, col-bwd) batched into single Mamba2 call.

**MambaOut (Gated CNN)** — New block inspired by Yu et al. (2024):
- Replaces SSM with identity — keeps only the gated convolution structure from Mamba
- Architecture: LayerNorm -> Linear(192, 384) -> DW-Conv7x7(384) -> SiLU gate -> Linear(384, 192) -> Residual
- Same cross-modal gating as other blocks
- Tests the hypothesis that Mamba's SSM component isn't needed for vision refinement

### Supervision at 128x256

Pseudo-labels (1024x2048 PNGs) are downsampled to 128x256 via nearest-neighbor interpolation. This provides 4x sharper supervision boundaries than the 32x64 baseline. All losses (distillation, alignment, prototype, entropy) computed at 128x256.

Depth maps (512x1024 float32) are downsampled to 128x256 via bilinear interpolation for the FiLM conditioning and boundary alignment loss.

### Training Configuration

Same proven conservative config from Run D baseline:
- distill_min=0.85, lambda_align=0.25, lambda_proto=0.025, lambda_ent=0.025
- label_smoothing=0.1, lr=5e-5, cosine schedule
- 20 epochs, eval every 2 epochs
- batch_size=4 (reduce to 2 if OOM)
- bridge_dim=192, num_blocks=4
- Device: MPS (M4 Pro, 48GB unified memory, float32 only)

### Memory Budget (128x256, batch=4, float32)

| Component | Estimate |
|-----------|----------|
| DINOv2 features (input, 32x64) | 150MB |
| Upsampled features (128x256) | 600MB |
| Conv2d blocks (4x) | ~1.6GB |
| Self-attention blocks (windowed) | ~2.0GB |
| Mamba2 blocks (bidirectional) | ~3.5GB |
| Gradients + optimizer | ~2x model |
| **Total Conv2d** | **~5GB** |
| **Total Attention** | **~6GB** |
| **Total Mamba2** | **~9GB** |

All configurations fit within 48GB unified memory on MPS.

### Success Criteria

- PQ_things recovery: 17.10 -> 19+ (matching or exceeding input pseudo-labels)
- PQ_stuff maintained: >= 33.0
- Overall PQ improvement: > 26.74 (input pseudo-label quality)
- changed_pct in healthy range: 3-8%

### Risks

1. **16x FLOPs increase** — Conv2d at 128x256 is 16x slower per block than 32x64. Each epoch may take 15-25 min vs 1.5 min.
2. **Upsampled features are blurry** — Bilinear/transposed conv can't recover high-frequency detail not present in 32x64 DINOv2 patches. The 128x256 model operates on interpolated features, not native high-res features.
3. **MPS float32 constraint** — No autocast/mixed precision on MPS with Mamba2. Memory usage will be higher than CUDA equivalent.
4. **Mamba2 at 32K tokens** — Pure PyTorch Mamba2 reference implementation may be slow at this sequence length. Monitor per-epoch time.
