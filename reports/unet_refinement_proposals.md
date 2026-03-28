# UNet-Based Semantic Refinement for Unsupervised Panoptic Segmentation: Architecture Proposals

## 1. Motivation

Self-supervised vision transformers (DINOv2 ViT-B/14) produce rich semantic features at coarse patch resolution (32x64 for 448x896 input). Prior work with CSCMRefineNet demonstrated that operating at this native resolution improves semantic quality (mIoU +5.3%, PQ_stuff +1.30) but structurally degrades thing-class panoptic quality (PQ_things 19.41 to 17.10) due to insufficient spatial resolution for small instances.

Initial experiments with direct upsampling (32x64 to 128x256 via transposed convolution) followed by refinement at the target resolution show further semantic gains (PQ_stuff 34.11, mIoU 56.24 at epoch 2) but fail to recover PQ_things (16.01). The fundamental limitation is that upsampled features are spatially smooth --- bilinear interpolation, transposed convolution, and sub-pixel convolution cannot synthesize high-frequency boundary information absent from the 32x64 source. The upsampled 128x256 feature map contains no more spatial detail than the original 32x64 representation; it merely distributes the same information across more spatial positions.

This motivates a UNet-style architecture that injects high-frequency spatial signals via skip connections during progressive upsampling. Unlike direct upsampling, skip connections provide the decoder with boundary-level detail at each scale, enabling the network to refine semantic predictions at high resolution while preserving the sharp object boundaries critical for thing-class panoptic quality.

## 2. Problem Analysis: Why Direct Upsampling Fails for PQ_things

### 2.1 The Information Bottleneck

DINOv2 ViT-B/14 with stride 14 produces a 32x64 feature grid for a 448x896 input image. Each patch token summarizes a 14x14 pixel region, discarding sub-patch spatial structure. When upsampled to 128x256, the resulting features are 4x oversampled but contain no new spatial information:

```
    Original image (1024 x 2048)
    ┌──────────────────────────────────────────────────────────────┐
    │  ┌──┐ ┌──┐                                                  │
    │  │B1│ │B2│  <-- Bicycles: ~40x20 px each = 800 px           │
    │  └──┘ └──┘      = 0.04% of image area                       │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
                            |
                    DINOv2 ViT-B/14 (stride 14)
                            |
                            v
    Patch features (32 x 64)
    ┌──────────────────────────────────────────────────────────────┐
    │  [P] [P]  <-- Both bicycles map to ~2-3 patches             │
    │           Adjacent instances share same patch tokens         │
    │           Boundary between B1 and B2 is LOST                │
    └──────────────────────────────────────────────────────────────┘
                            |
                    Bilinear upsample 4x
                            |
                            v
    Upsampled features (128 x 256)
    ┌──────────────────────────────────────────────────────────────┐
    │  [....] [....] <-- Smooth interpolation of same 2-3 patches │
    │                    No new boundary information               │
    │                    B1/B2 boundary still absent               │
    └──────────────────────────────────────────────────────────────┘
```

### 2.2 What Skip Connections Provide

High-resolution spatial signals (depth maps, RGB edges, or learned features) carry boundary information at the target resolution. Skip connections inject this information directly into the decoder at each scale, bypassing the information bottleneck:

```
    Depth map (512 x 1024)           RGB image (1024 x 2048)
    ┌────────────────────────┐       ┌────────────────────────────┐
    │      ____              │       │      ____                  │
    │     /B1  \  /B2\       │       │     |B1  |  |B2|           │
    │     \____/  \__/       │       │     |____|  |__|           │
    │  Depth discontinuity   │       │  RGB edges preserve        │
    │  at object boundary    │       │  instance boundaries       │
    └────────────────────────┘       └────────────────────────────┘
            |                                |
            v                                v
    Boundary detail at HIGH resolution is available
    from modalities OTHER than DINOv2 features
```

A UNet decoder with skip connections fuses the semantic richness of DINOv2 features with the spatial precision of high-resolution auxiliary signals, potentially recovering the PQ_things regression.

## 3. Proposed Architectures

We propose three UNet-based architectures with increasing complexity. All share the same training objective (distillation + self-supervised losses) and evaluation protocol as the baseline CSCMRefineNet.

### 3.1 Option A: Depth-Guided UNet Decoder

#### Rationale

Monocular depth maps (SPIdepth, 512x1024) are already used for FiLM conditioning in CSCMRefineNet. Depth discontinuities strongly correlate with object boundaries --- the same signal that PQ_things evaluation rewards. By providing depth at each decoder scale as skip connections, the network receives explicit boundary guidance during progressive upsampling.

This is the lightest option: no new feature extraction, no encoder. The depth map is simply downsampled to each decoder scale and projected via learned convolutions before concatenation.

#### Architecture

```
    ┌──────────────────────────────────────────────────────────────────────┐
    │                 Option A: Depth-Guided UNet Decoder                  │
    │                                                                      │
    │                                                                      │
    │  DINOv2 ViT-B/14 (frozen)                SPIdepth (frozen)           │
    │  (768-dim, 32x64)                        (512x1024)                  │
    │       │                                       │                      │
    │       ├──────────────┐                        │                      │
    │       │              │                        │                      │
    │       v              v                        │                      │
    │  ┌─────────┐  ┌──────────────┐                │                      │
    │  │Semantic │  │Depth-Feature │                │                      │
    │  │  Proj   │  │  Proj + FiLM │◄───────────────┤  (depth at 32x64)   │
    │  │768→192  │  │768→192       │                │                      │
    │  └────┬────┘  └──────┬───────┘                │                      │
    │       │              │                        │                      │
    │       v              v                        │                      │
    │  ┌────────────────────────────┐               │                      │
    │  │  Bottleneck: CoupledBlocks │               │                      │
    │  │  × 2 at 32x64             │               │                      │
    │  │  (192-dim, cross-gating)   │               │                      │
    │  └────────────┬───────────────┘               │                      │
    │               │ (192, 32, 64)                 │                      │
    │               v                               │                      │
    │  ┌────────────────────────────────────────────┤                      │
    │  │  Decoder Stage 1: 32x64 → 64x128          │                      │
    │  │                                            │                      │
    │  │  ┌──────────────┐   ┌────────────────┐     │                      │
    │  │  │ ConvTranspose │   │ Depth Skip     │     │                      │
    │  │  │ 2d (stride=2) │   │ (64x128)       │     │                      │
    │  │  │ 192→192      │   │                │     │                      │
    │  │  └──────┬───────┘   │ Depth (1ch)    │     │                      │
    │  │         │           │ + Sobel (2ch)   │     │                      │
    │  │         │           │ → Conv 3→32    │     │                      │
    │  │         │           │ → GN + GELU    │     │                      │
    │  │         │           └───────┬────────┘     │                      │
    │  │         │                   │              │                      │
    │  │         └───────┬───────────┘              │                      │
    │  │                 │ Cat: (192+32, 64, 128)   │                      │
    │  │                 v                          │                      │
    │  │         ┌───────────────┐                  │                      │
    │  │         │ Fuse Conv 1x1 │                  │                      │
    │  │         │ (224→192)     │                  │                      │
    │  │         │ + CoupledBlock│                  │                      │
    │  │         │ × 1 at 64x128│                  │                      │
    │  │         └───────┬───────┘                  │                      │
    │  │                 │ (192, 64, 128)           │                      │
    │  └─────────────────┼──────────────────────────┘                      │
    │                    v                                                 │
    │  ┌─────────────────────────────────────────────┐                     │
    │  │  Decoder Stage 2: 64x128 → 128x256          │                     │
    │  │                                              │                     │
    │  │  ┌──────────────┐   ┌────────────────┐       │                     │
    │  │  │ ConvTranspose │   │ Depth Skip     │       │                     │
    │  │  │ 2d (stride=2) │   │ (128x256)      │       │                     │
    │  │  │ 192→192      │   │                │       │                     │
    │  │  └──────┬───────┘   │ Depth (1ch)    │       │                     │
    │  │         │           │ + Sobel (2ch)   │       │                     │
    │  │         │           │ → Conv 3→32    │       │                     │
    │  │         │           │ → GN + GELU    │       │                     │
    │  │         │           └───────┬────────┘       │                     │
    │  │         │                   │                │                     │
    │  │         └───────┬───────────┘                │                     │
    │  │                 │ Cat: (192+32, 128, 256)    │                     │
    │  │                 v                            │                     │
    │  │         ┌───────────────┐                    │                     │
    │  │         │ Fuse Conv 1x1 │                    │                     │
    │  │         │ (224→192)     │                    │                     │
    │  │         │ + CoupledBlock│                    │                     │
    │  │         │ × 1 at 128x256│                    │                     │
    │  │         └───────┬───────┘                    │                     │
    │  │                 │ (192, 128, 256)            │                     │
    │  └─────────────────┼────────────────────────────┘                     │
    │                    v                                                 │
    │            ┌───────────────┐                                         │
    │            │ Classification│                                         │
    │            │ Head          │                                         │
    │            │ GN + Conv 1x1 │                                         │
    │            │ (192 → 19)   │                                         │
    │            └───────────────┘                                         │
    │                    │                                                 │
    │                    v                                                 │
    │            Refined logits (19, 128, 256)                             │
    │            → nearest upsample to (19, 1024, 2048) for eval           │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘
```

#### Depth Skip Connection Detail

```
    ┌───────────────────────────────────────────────────────┐
    │              Depth Skip Block (per scale)              │
    │                                                       │
    │  Depth map (512, 1024)                                │
    │       │                                               │
    │       v                                               │
    │  Bilinear downsample to target scale                  │
    │  (e.g., 64x128 or 128x256)                           │
    │       │                                               │
    │       ├──► Raw depth ────────────────┐                │
    │       │    (1 channel)               │                │
    │       │                              │                │
    │       ├──► Sobel_x ─────────────────┤                │
    │       │    (1 channel)               │  Cat           │
    │       │                              │  (3 channels)  │
    │       └──► Sobel_y ─────────────────┤                │
    │            (1 channel)               │                │
    │                                      v                │
    │                              ┌──────────────┐         │
    │                              │ Conv2d 3x3   │         │
    │                              │ (3 → 32)     │         │
    │                              │ GN + GELU    │         │
    │                              └──────┬───────┘         │
    │                                     │                 │
    │                                     v                 │
    │                              (32, H, W)               │
    │                              Skip features            │
    │                                                       │
    └───────────────────────────────────────────────────────┘

    Key insight: Sobel gradients of depth encode object boundaries.
    These boundaries are available at HIGH resolution from the depth
    map, regardless of DINOv2 patch resolution. The skip connection
    provides boundary-level spatial detail that upsampled features lack.
```

#### Parameter Budget

| Component | Parameters |
|-----------|-----------|
| Semantic Projection (768 -> 192) | 148K |
| Depth-Feature Projection (768 + FiLM -> 192) | 165K |
| Bottleneck: CoupledConvBlock x 2 at 32x64 | 460K |
| Decoder Stage 1: ConvTranspose2d + DepthSkip + Fuse + CoupledBlock | ~310K |
| Decoder Stage 2: ConvTranspose2d + DepthSkip + Fuse + CoupledBlock | ~310K |
| Classification Head | 3.7K |
| **Total** | **~1.4M** |

#### Advantages and Limitations

**Advantages:**
- Lightest option: no new feature extraction, uses already-loaded depth
- Depth boundaries directly encode the spatial detail PQ_things needs
- Progressive upsampling allows refinement at each scale
- Bottleneck blocks at 32x64 are cheap (proven from prior work)

**Limitations:**
- Depth boundaries are noisy (monocular estimation artifacts)
- Depth provides geometric but not semantic boundary cues (a wall and a person at the same depth have no depth discontinuity)
- Limited to 3 channels of skip information per scale

---

### 3.2 Option B: Lightweight CNN Encoder with UNet Skip Connections

#### Rationale

While depth provides geometric boundaries, RGB images contain richer boundary information: texture edges, color discontinuities, and semantic contours invisible to depth. A lightweight CNN encoder applied to the downsampled input image produces multi-scale feature maps that serve as skip connections, providing both geometric and appearance-level boundary detail to the UNet decoder.

The encoder is intentionally shallow (3 stages, 32/64/128 channels) to minimize computational overhead while extracting boundary-relevant features at 128x256, 64x128, and 32x64 resolutions.

#### Architecture

```
    ┌──────────────────────────────────────────────────────────────────────┐
    │          Option B: Lightweight CNN Encoder + UNet Decoder             │
    │                                                                      │
    │                                                                      │
    │  Input Image (3, 256, 512)         DINOv2 ViT-B/14 (frozen)         │
    │  (bilinear downsampled)            (768, 32, 64)                     │
    │       │                                  │                           │
    │       v                                  │                           │
    │  ┌────────────────┐                      │                           │
    │  │ Encoder Stage 1│                      │                           │
    │  │ Conv 3x3 (s=2) │                      │                           │
    │  │ 3→32, GN, GELU│                      │                           │
    │  │ Conv 3x3       │                      │                           │
    │  │ 32→32, GN, GELU│                      │                           │
    │  └────┬───────────┘                      │                           │
    │       │ skip_1 (32, 128, 256)            │                           │
    │       v                                  │                           │
    │  ┌────────────────┐                      │                           │
    │  │ Encoder Stage 2│                      │                           │
    │  │ Conv 3x3 (s=2) │                      │                           │
    │  │ 32→64, GN, GELU│                      │                           │
    │  │ Conv 3x3       │                      │                           │
    │  │ 64→64, GN, GELU│                      │                           │
    │  └────┬───────────┘                      │                           │
    │       │ skip_2 (64, 64, 128)             │                           │
    │       v                                  │                           │
    │  ┌────────────────┐                      │                           │
    │  │ Encoder Stage 3│                      │                           │
    │  │ Conv 3x3 (s=2) │                      │                           │
    │  │ 64→128, GN,GELU│                      │                           │
    │  │ Conv 3x3       │                      │                           │
    │  │128→128, GN,GELU│                      │                           │
    │  └────┬───────────┘                      │                           │
    │       │ enc_feat (128, 32, 64)           │                           │
    │       │                                  │                           │
    │       v                                  v                           │
    │  ┌────────────────────────────────────────────────┐                  │
    │  │              Bottleneck Fusion                  │                  │
    │  │                                                │                  │
    │  │  DINOv2 proj (192, 32, 64)                     │                  │
    │  │       +                                        │                  │
    │  │  enc_feat (128, 32, 64)                        │                  │
    │  │       │                                        │                  │
    │  │       v                                        │                  │
    │  │  Cat → Conv 1x1 (320→192) + CoupledBlocks × 2 │                  │
    │  │                                                │                  │
    │  └────────────────────┬───────────────────────────┘                  │
    │                       │ (192, 32, 64)                                │
    │                       v                                              │
    │  ┌────────────────────────────────────────────────────┐              │
    │  │  Decoder Stage 1: 32x64 → 64x128                  │              │
    │  │                                                    │              │
    │  │  ConvTranspose2d ──────────┐                       │              │
    │  │  (192→192, s=2)           │                       │              │
    │  │                            │  Cat                  │              │
    │  │  skip_2 (64, 64, 128) ────┘                       │              │
    │  │                            │  (256, 64, 128)      │              │
    │  │                            v                       │              │
    │  │                    Fuse Conv 1x1 (256→192)         │              │
    │  │                    + CoupledBlock × 1              │              │
    │  │                                                    │              │
    │  └────────────────────┬───────────────────────────────┘              │
    │                       │ (192, 64, 128)                               │
    │                       v                                              │
    │  ┌────────────────────────────────────────────────────┐              │
    │  │  Decoder Stage 2: 64x128 → 128x256                │              │
    │  │                                                    │              │
    │  │  ConvTranspose2d ──────────┐                       │              │
    │  │  (192→192, s=2)           │                       │              │
    │  │                            │  Cat                  │              │
    │  │  skip_1 (32, 128, 256) ───┘                       │              │
    │  │                            │  (224, 128, 256)     │              │
    │  │                            v                       │              │
    │  │                    Fuse Conv 1x1 (224→192)         │              │
    │  │                    + CoupledBlock × 1              │              │
    │  │                                                    │              │
    │  └────────────────────┬───────────────────────────────┘              │
    │                       │ (192, 128, 256)                              │
    │                       v                                              │
    │               ┌───────────────┐                                      │
    │               │ Classification│                                      │
    │               │ GN + Conv 1x1 │                                      │
    │               │ (192 → 19)   │                                      │
    │               └───────────────┘                                      │
    │                       │                                              │
    │                       v                                              │
    │               Refined logits (19, 128, 256)                          │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘
```

#### Encoder Detail

```
    ┌───────────────────────────────────────────────────────┐
    │              Lightweight CNN Encoder                    │
    │                                                       │
    │  Input: RGB image downsampled to (3, 256, 512)        │
    │                                                       │
    │  Stage 1:                                             │
    │    Conv2d(3, 32, 3, stride=2, padding=1)              │
    │    GroupNorm(1, 32) + GELU                            │
    │    Conv2d(32, 32, 3, stride=1, padding=1)             │
    │    GroupNorm(1, 32) + GELU                            │
    │    Output: (32, 128, 256) ──────── skip_1             │
    │                                                       │
    │  Stage 2:                                             │
    │    Conv2d(32, 64, 3, stride=2, padding=1)             │
    │    GroupNorm(1, 64) + GELU                            │
    │    Conv2d(64, 64, 3, stride=1, padding=1)             │
    │    GroupNorm(1, 64) + GELU                            │
    │    Output: (64, 64, 128) ──────── skip_2              │
    │                                                       │
    │  Stage 3:                                             │
    │    Conv2d(64, 128, 3, stride=2, padding=1)            │
    │    GroupNorm(1, 128) + GELU                           │
    │    Conv2d(128, 128, 3, stride=1, padding=1)           │
    │    GroupNorm(1, 128) + GELU                           │
    │    Output: (128, 32, 64) ──────── bottleneck fusion   │
    │                                                       │
    │  Total encoder parameters: ~110K                      │
    │  Note: Encoder weights are LEARNED (not frozen).      │
    │  The encoder learns to extract boundary-relevant       │
    │  features from raw RGB that complement DINOv2.        │
    │                                                       │
    └───────────────────────────────────────────────────────┘

    Design choice: The encoder intentionally uses small channel
    widths (32/64/128) because its role is to provide spatial
    structure, not semantic understanding. DINOv2 handles semantics.
```

#### Parameter Budget

| Component | Parameters |
|-----------|-----------|
| CNN Encoder (3 stages) | 110K |
| DINOv2 Projections (sem + depth-FiLM) | 313K |
| Bottleneck fusion + CoupledBlocks x 2 | 530K |
| Decoder Stage 1: up + fuse + CoupledBlock | 310K |
| Decoder Stage 2: up + fuse + CoupledBlock | 260K |
| Classification Head | 3.7K |
| **Total** | **~1.5M** |

#### Advantages and Limitations

**Advantages:**
- Richer skip features than depth alone (texture, color, semantic edges)
- Multi-scale encoder aligns naturally with UNet decoder
- Learned encoder adapts to task-specific boundary detection
- Small channel widths keep parameter count low

**Limitations:**
- Requires loading raw images (currently not loaded --- only pre-extracted features)
- Encoder training adds computational overhead
- Risk of encoder overfitting to pseudo-label noise (RGB patterns that correlate with pseudo-label errors)
- 256x512 input is 4x downsampled from original --- some boundary detail still lost

---

### 3.3 Option C: Progressive Refinement Decoder (Decoder-Only UNet)

#### Rationale

The simplest approach: no encoder, no skip connections. Instead of refining at a single resolution, distribute computation across a multi-scale cascade. Process features at 32x64 first (where convolutions are cheap and capture global context), then progressively upsample and refine at each scale with fewer blocks.

This exploits the observation that global semantic decisions (road vs. building) are best made at low resolution, while boundary precision (car vs. adjacent car) requires high resolution. Allocating more blocks at 32x64 and fewer at 128x256 is both computationally efficient and architecturally appropriate.

#### Architecture

```
    ┌──────────────────────────────────────────────────────────────────────┐
    │        Option C: Progressive Refinement Decoder (No Encoder)         │
    │                                                                      │
    │                                                                      │
    │  DINOv2 ViT-B/14 (frozen)                                           │
    │  (768, 32, 64)                                                       │
    │       │                                                              │
    │       ├──────────────┐                                               │
    │       v              v                                               │
    │  ┌─────────┐  ┌──────────────┐                                       │
    │  │Semantic │  │Depth-Feature │                                       │
    │  │  Proj   │  │  Proj + FiLM │◄── Depth (32x64)                     │
    │  │768→192  │  │768→192       │                                       │
    │  └────┬────┘  └──────┬───────┘                                       │
    │       │              │                                               │
    │       v              v                                               │
    │  ┌────────────────────────────────────────────────┐                  │
    │  │  Scale 1: Processing at 32x64 (2,048 tokens)   │                  │
    │  │                                                │                  │
    │  │  CoupledConvBlock × 2                          │                  │
    │  │  (global context, semantic decisions)           │                  │
    │  │                                                │                  │
    │  │  Cost: 2 blocks × ~230K FLOPs = 460K FLOPs    │                  │
    │  │  Role: Establish class-level predictions        │                  │
    │  │                                                │                  │
    │  └────────────────────┬───────────────────────────┘                  │
    │                       │ (192, 32, 64)                                │
    │                       v                                              │
    │  ┌────────────────────────────────────────────────┐                  │
    │  │  Upsample 1: 32x64 → 64x128                   │                  │
    │  │                                                │                  │
    │  │  ConvTranspose2d(192, 192, k=4, s=2, p=1)     │                  │
    │  │  GroupNorm + GELU                              │                  │
    │  │                                                │                  │
    │  └────────────────────┬───────────────────────────┘                  │
    │                       │ (192, 64, 128)                               │
    │                       v                                              │
    │  ┌────────────────────────────────────────────────┐                  │
    │  │  Scale 2: Processing at 64x128 (8,192 tokens)  │                  │
    │  │                                                │                  │
    │  │  CoupledConvBlock × 1                          │                  │
    │  │  (medium-scale boundary refinement)             │                  │
    │  │                                                │                  │
    │  │  Cost: 1 block × ~920K FLOPs = 920K FLOPs     │                  │
    │  │  Role: Refine stuff-thing boundaries           │                  │
    │  │                                                │                  │
    │  └────────────────────┬───────────────────────────┘                  │
    │                       │ (192, 64, 128)                               │
    │                       v                                              │
    │  ┌────────────────────────────────────────────────┐                  │
    │  │  Upsample 2: 64x128 → 128x256                 │                  │
    │  │                                                │                  │
    │  │  ConvTranspose2d(192, 192, k=4, s=2, p=1)     │                  │
    │  │  GroupNorm + GELU                              │                  │
    │  │                                                │                  │
    │  └────────────────────┬───────────────────────────┘                  │
    │                       │ (192, 128, 256)                              │
    │                       v                                              │
    │  ┌────────────────────────────────────────────────┐                  │
    │  │  Scale 3: Processing at 128x256 (32,768 tokens)│                  │
    │  │                                                │                  │
    │  │  CoupledConvBlock × 1                          │                  │
    │  │  (fine-grained instance boundary precision)     │                  │
    │  │                                                │                  │
    │  │  Cost: 1 block × ~3.7M FLOPs = 3.7M FLOPs    │                  │
    │  │  Role: Sharpen thing instance boundaries       │                  │
    │  │                                                │                  │
    │  └────────────────────┬───────────────────────────┘                  │
    │                       │ (192, 128, 256)                              │
    │                       v                                              │
    │               ┌───────────────┐                                      │
    │               │ Classification│                                      │
    │               │ GN + Conv 1x1 │                                      │
    │               │ (192 → 19)   │                                      │
    │               └───────────────┘                                      │
    │                       │                                              │
    │                       v                                              │
    │               Refined logits (19, 128, 256)                          │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘
```

#### Multi-Scale Computation Distribution

```
    ┌────────────────────────────────────────────────────────────┐
    │              Computation Pyramid                            │
    │                                                            │
    │  Resolution   Blocks   Tokens   FLOPs/block   Total FLOPs │
    │  ──────────   ──────   ──────   ──────────    ─────────── │
    │  32 × 64        2      2,048      230K          460K      │
    │  64 × 128       1      8,192      920K          920K      │
    │  128 × 256      1     32,768    3,700K        3,700K      │
    │                                                            │
    │  Total: 4 blocks, 5,080K FLOPs                             │
    │                                                            │
    │  Compare: 4 blocks all at 128×256 = 14,800K FLOPs          │
    │  Savings: 66% FLOPs reduction with multi-scale cascade     │
    │                                                            │
    │  Visualization of block allocation:                        │
    │                                                            │
    │     32×64:    [Block] [Block]    ← most blocks here       │
    │                  │                  (cheap, global context) │
    │                  v                                         │
    │     64×128:  [  Block  ]         ← one block              │
    │                  │                  (medium cost)           │
    │                  v                                         │
    │    128×256:  [    Block    ]      ← one block             │
    │                                     (expensive, fine-grain)│
    │                                                            │
    └────────────────────────────────────────────────────────────┘
```

#### Parameter Budget

| Component | Parameters |
|-----------|-----------|
| Semantic Projection (768 -> 192) | 148K |
| Depth-Feature Projection (768 + FiLM -> 192) | 165K |
| Scale 1: CoupledConvBlock x 2 at 32x64 | 460K |
| Upsample 1: ConvTranspose2d | 148K |
| Scale 2: CoupledConvBlock x 1 at 64x128 | 230K |
| Upsample 2: ConvTranspose2d | 148K |
| Scale 3: CoupledConvBlock x 1 at 128x256 | 230K |
| Classification Head | 3.7K |
| **Total** | **~1.5M** |

#### Advantages and Limitations

**Advantages:**
- Simplest architecture: no encoder, no skip connections, no additional data loading
- 66% FLOPs reduction vs. all-blocks-at-128x256 approach
- Multi-scale processing is architecturally principled (global to local)
- Compatible with any block type (conv, attention, Mamba2, MambaOut)

**Limitations:**
- No external high-frequency signal --- still limited by DINOv2 patch resolution
- Progressive upsampling via transposed conv produces smooth interpolations
- The 128x256 block refines upsampled features, not original high-res detail
- May behave similarly to direct upsampling (the current approach) since no skip information is injected

---

## 4. Comparative Summary

| Property | A: Depth-Guided UNet | B: CNN Encoder UNet | C: Progressive Decoder |
|----------|---------------------|--------------------|-----------------------|
| Parameters | ~1.4M | ~1.5M | ~1.5M |
| Skip connections | Depth + Sobel grads | Learned RGB features | None |
| Encoder required | No | Yes (lightweight CNN) | No |
| Additional data | None (depth already loaded) | Raw images (new) | None |
| Boundary signal | Geometric (depth edges) | Geometric + appearance | None (implicit) |
| FLOPs (relative) | 1.0x | 1.2x | 0.66x |
| Implementation complexity | Medium | High | Low |
| Risk | Depth noise | Encoder overfitting | Same as direct upsample |

### Recommended Ablation Order

1. **Option A (Depth-Guided UNet)** --- highest expected impact, minimal infrastructure change. Depth boundaries directly address the PQ_things bottleneck.
2. **Option C (Progressive Decoder)** --- simplest baseline. If this matches Option A, skip connections are unnecessary.
3. **Option B (CNN Encoder UNet)** --- only if A significantly outperforms C, validating that skip connections matter. Then test whether learned RGB features improve over depth-only skips.

## 5. Expected Outcomes

### 5.1 Hypothesis Testing

| Hypothesis | Test | Success Criterion |
|-----------|------|-------------------|
| Higher resolution recovers PQ_things | All options vs. 32x64 baseline | PQ_things > 17.10 |
| Skip connections provide boundary detail | Option A vs. C | PQ_things(A) > PQ_things(C) |
| Depth boundaries sufficient for things | Option A vs. B | PQ_things(A) >= PQ_things(B) |
| Progressive refinement saves compute | Option C timing | < 50% wall-clock vs. flat 128x256 |
| Overall PQ exceeds input pseudo-labels | Best option vs. 26.74 | PQ > 26.74 |

### 5.2 Results (Pending)

| Config | PQ | PQ_stuff | PQ_things | mIoU | Time/Epoch |
|--------|-----|----------|-----------|------|------------|
| Input pseudo-labels | 26.74 | 32.08 | 19.41 | ~50% | --- |
| 32x64 Conv2d (Run D) | 26.52 | 33.38 | 17.10 | 55.31 | ~96s |
| 128x256 direct upsample (U-B) | --- | --- | --- | --- | ~948s |
| A: Depth-Guided UNet | --- | --- | --- | --- | --- |
| B: CNN Encoder UNet | --- | --- | --- | --- | --- |
| C: Progressive Decoder | --- | --- | --- | --- | --- |
