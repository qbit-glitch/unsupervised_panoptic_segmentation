# UNet Decoder Architectures for High-Resolution Semantic Refinement in Unsupervised Panoptic Segmentation

## Abstract

Direct upsampling of frozen DINOv2 ViT-B/14 features from 32x64 to 128x256 followed by convolutional refinement achieves record semantic quality (mIoU=56.58, PQ_stuff=34.64) but fails to fully recover thing-class panoptic quality (PQ_things=17.33 vs 19.41 input). We identify the root cause: upsampled features are spatially smooth --- bilinear interpolation, transposed convolution, and sub-pixel convolution redistribute existing information across more spatial positions without synthesizing the high-frequency boundary detail critical for distinguishing adjacent thing instances. We propose three UNet-style decoder architectures that inject boundary-level spatial signals via skip connections during progressive upsampling. Option A uses monocular depth maps (already available) as skip connections, providing geometric boundary cues at each decoder scale. Option B introduces a lightweight CNN encoder on RGB images, providing both geometric and appearance-level boundary features. Option C serves as a decoder-only baseline with multi-scale processing but no skip connections. Experiments evaluate whether skip-connected decoders recover PQ_things while preserving the semantic gains of high-resolution refinement.

---

## 1. Introduction

Self-supervised vision transformers produce semantically rich but spatially coarse feature representations. DINOv2 ViT-B/14 with stride 14 compresses a 1024x2048 Cityscapes image into a 32x64 patch grid, discarding sub-patch spatial structure. Our prior work with CSCMRefineNet demonstrated that cross-modal refinement at this native resolution improves semantic segmentation (mIoU 50% to 55.3%) but structurally degrades thing-class panoptic quality (PQ_things 19.41 to 17.10). Four architectural ablations (TAD, BPL, ASPP-lite, TAD+BPL) confirmed this is a resolution bottleneck, not a loss or receptive field problem.

Moving to 128x256 resolution via direct upsampling (HiRes RefineNet) partially addresses this limitation. At epoch 10 of training with transposed convolution upsampling:

| Metric | Input Pseudo-labels | 32x64 RefineNet | 128x256 HiRes (epoch 10) |
|--------|--------------------|-----------------|-----------------------|
| PQ | 26.74 | 26.52 | **27.35** |
| PQ_stuff | 32.08 | 33.38 | **34.64** |
| PQ_things | **19.41** | 17.10 | 17.33 |
| mIoU | ~50% | 55.31 | **56.58** |

The 128x256 resolution achieves record PQ (27.35), PQ_stuff (34.64), and mIoU (56.58). However, PQ_things recovers only partially (17.33 vs 19.41 input) --- a 2.08-point gap remains. The fundamental limitation is that upsampled features contain no more spatial information than the original 32x64 representation. Transposed convolution learns to sharpen features during upsampling, but cannot synthesize boundary detail that was never captured by the ViT encoder.

This paper proposes UNet-style decoder architectures that address this information bottleneck by injecting high-frequency spatial signals from auxiliary modalities (depth, RGB) via skip connections at each decoder scale.

## 2. The Information Bottleneck: Why Direct Upsampling Fails for PQ_things

### 2.1 Spatial Resolution and Instance Separability

PQ_things measures the quality of thing-class predictions (person, car, bicycle, etc.), which requires correctly identifying individual instances. Adjacent instances must be spatially separated in the prediction map. At 32x64, small thing instances collapse:

```
    Original image (1024 x 2048)
    +--------------------------------------------------------------+
    |  +--+ +--+                                                    |
    |  |B1| |B2|  <-- Two bicycles: ~40x20 px each                 |
    |  +--+ +--+      Gap between them: ~10 px                     |
    |                  Total footprint: 90x20 = 1,800 px            |
    |                  = 0.09% of image area                        |
    +--------------------------------------------------------------+
                            |
                    DINOv2 ViT-B/14 (stride 14)
                            |
                            v
    Patch features (32 x 64)
    +--------------------------------------------------------------+
    |  [P] [P]  <-- Both bicycles share 2-3 patches                |
    |           10px gap / 14px stride < 1 patch                    |
    |           B1 and B2 are MERGED in feature space               |
    +--------------------------------------------------------------+
                            |
                    Transposed Conv 4x upsample
                            |
                            v
    Upsampled features (128 x 256)
    +--------------------------------------------------------------+
    |  [........] [........] <-- Smooth interpolation               |
    |                           of 2-3 merged patches               |
    |                           No boundary between B1/B2           |
    |                           4x more pixels, same information    |
    +--------------------------------------------------------------+
```

The 10-pixel gap between bicycles is sub-patch (< 14 pixels), so it is irreversibly lost during ViT encoding. No upsampling strategy can recover it from the patch features alone.

### 2.2 What Skip Connections Provide

High-resolution auxiliary signals (depth maps, RGB images) retain boundary information at the target resolution. Depth discontinuities mark object boundaries; RGB edges mark texture and color transitions. Skip connections inject these signals directly into the decoder at each scale, providing the spatial detail that upsampled features lack:

```
    Depth map (512 x 1024)              RGB image (1024 x 2048)
    +------------------------+          +----------------------------+
    |      ____              |          |      ____                  |
    |     /B1  \  /B2\       |          |     |B1  |  |B2|           |
    |     \____/  \__/       |          |     |____|  |__|           |
    |  Depth discontinuity   |          |  Color/texture edges       |
    |  at each object        |          |  at each object            |
    |  boundary preserved    |          |  boundary preserved        |
    +------------------------+          +----------------------------+
            |                                   |
            v                                   v
    Boundary information available at HIGH resolution
    from modalities INDEPENDENT of DINOv2 patch encoding
```

A UNet decoder fuses the semantic richness of DINOv2 features (what is here?) with the spatial precision of auxiliary signals (where are the boundaries?), potentially closing the PQ_things gap.

### 2.3 Quantitative Gap Analysis

The PQ_things deficit concentrates on small, frequently adjacent thing classes:

| Class | Input PQ | 128x256 HiRes PQ | Delta |
|-------|----------|-------------------|-------|
| person | 3.58 | 3.99 | +0.41 |
| rider | 7.45 | 8.29 | +0.84 |
| car | 14.46 | 14.92 | +0.46 |
| bicycle | 6.96 | 6.12 | -0.84 |
| motorcycle | 0.00 | 0.00 | 0.00 |
| truck | 27.53 | 33.56 | +6.03 |
| bus | 40.17 | 40.49 | +0.32 |
| train | 28.13 | 33.31 | +5.18 |

Input PQ: k=80 overclustered pseudo-labels with depth-guided instance splitting. HiRes PQ: U-B transposed conv run, epoch 16 (best PQ_things=17.58). Large things (truck +6.03, train +5.18) benefit most from 128x256 resolution; small things (person +0.41, bicycle -0.84) show negligible or negative improvement.

---

## 3. Method

We propose three UNet-based decoder architectures with increasing complexity, all sharing the same training objective and evaluation protocol. The key variable is the source and mechanism of skip connections.

### 3.1 Common Components

All three options share:
- **Backbone**: DINOv2 ViT-B/14 (frozen, 768-dim, 32x64 patches)
- **Semantic projection**: Conv2d 1x1, 768 to 192 channels
- **Depth-feature projection**: Conv2d 1x1 + FiLM conditioning from depth, 768 to 192 channels
- **Target resolution**: 128x256 (4x upsampling from patch grid)
- **Classification head**: GroupNorm + Conv2d 1x1, 192 to 19 classes
- **Refinement blocks**: CoupledConvBlock with depthwise-separable 3x3 convolutions and cross-modal gating
- **Output**: Nearest-neighbor upsampled to 1024x2048 for panoptic evaluation

### 3.2 Option A: Depth-Guided UNet Decoder

#### 3.2.1 Rationale

Monocular depth maps (SPIdepth, 512x1024) are already computed and loaded for FiLM conditioning. Depth discontinuities strongly correlate with object boundaries --- the same signal that PQ_things evaluation rewards. By providing depth at each decoder scale as skip connections, the network receives explicit geometric boundary guidance during progressive upsampling. This is the lightest option: no new feature extraction, no encoder training.

#### 3.2.2 Architecture

```
+------------------------------------------------------------------------+
|                  Option A: Depth-Guided UNet Decoder                    |
|                                                                        |
|                                                                        |
|  DINOv2 ViT-B/14 (frozen)                SPIdepth (frozen)             |
|  (768-dim, 32x64)                        (1, 512, 1024)               |
|       |                                       |                        |
|       +---------------+                       |                        |
|       |               |                       |                        |
|       v               v                       |                        |
|  +---------+   +--------------+                |                        |
|  |Semantic |   |Depth-Feature |                |                        |
|  |  Proj   |   |Proj + FiLM   |<----- depth ---+--- (bilinear to       |
|  |768->192 |   |768->192      |      (32x64)   |     each scale)       |
|  +----+----+   +------+-------+                |                        |
|       |               |                        |                        |
|       v               v                        |                        |
|  +----------------------------+                |                        |
|  | Bottleneck: CoupledBlocks  |                |                        |
|  | x 2 at 32x64              |                |                        |
|  | (192-dim, cross-gating)    |                |                        |
|  +------------+---------------+                |                        |
|               | (192, 32, 64)                  |                        |
|               v                                |                        |
|  +---------------------------------------------+----+                  |
|  | Decoder Stage 1: 32x64 -> 64x128                 |                  |
|  |                                                   |                  |
|  |  +--------------+    +------------------+         |                  |
|  |  | ConvTranspose |    | Depth Skip       |         |                  |
|  |  | 2d (stride=2) |    | (64x128)         |         |                  |
|  |  | 192->192     |    |                  |         |                  |
|  |  +------+-------+    | depth (1ch)      |         |                  |
|  |         |             | + Sobel_x (1ch)  |         |                  |
|  |         |             | + Sobel_y (1ch)  |         |                  |
|  |         |             | -> Conv 3x3      |         |                  |
|  |         |             |    (3->32)       |         |                  |
|  |         |             |    GN + GELU     |         |                  |
|  |         |             +--------+---------+         |                  |
|  |         |                      |                   |                  |
|  |         +----------+-----------+                   |                  |
|  |                    | Cat: (192+32, 64, 128)        |                  |
|  |                    v                               |                  |
|  |            +---------------+                       |                  |
|  |            | Fuse Conv 1x1 |                       |                  |
|  |            | (224->192)    |                       |                  |
|  |            | + CoupledBlock|                       |                  |
|  |            | x 1 at 64x128|                       |                  |
|  |            +-------+-------+                       |                  |
|  |                    | (192, 64, 128)                |                  |
|  +--------------------+-------------------------------+                  |
|                       v                                                 |
|  +---------------------------------------------+----+                  |
|  | Decoder Stage 2: 64x128 -> 128x256               |                  |
|  |                                                   |                  |
|  |  +--------------+    +------------------+         |                  |
|  |  | ConvTranspose |    | Depth Skip       |         |                  |
|  |  | 2d (stride=2) |    | (128x256)        |         |                  |
|  |  | 192->192     |    |                  |         |                  |
|  |  +------+-------+    | depth (1ch)      |         |                  |
|  |         |             | + Sobel_x (1ch)  |         |                  |
|  |         |             | + Sobel_y (1ch)  |         |                  |
|  |         |             | -> Conv 3x3      |         |                  |
|  |         |             |    (3->32)       |         |                  |
|  |         |             |    GN + GELU     |         |                  |
|  |         |             +--------+---------+         |                  |
|  |         |                      |                   |                  |
|  |         +----------+-----------+                   |                  |
|  |                    | Cat: (192+32, 128, 256)       |                  |
|  |                    v                               |                  |
|  |            +---------------+                       |                  |
|  |            | Fuse Conv 1x1 |                       |                  |
|  |            | (224->192)    |                       |                  |
|  |            | + CoupledBlock|                       |                  |
|  |            | x 1 at 128x256|                       |                  |
|  |            +-------+-------+                       |                  |
|  |                    | (192, 128, 256)               |                  |
|  +--------------------+-------------------------------+                  |
|                       v                                                 |
|               +---------------+                                         |
|               | Classification|                                         |
|               | GN + Conv 1x1 |                                         |
|               | (192 -> 19)  |                                         |
|               +-------+-------+                                         |
|                       |                                                 |
|                       v                                                 |
|               Refined logits (19, 128, 256)                             |
|               -> nearest upsample to (19, 1024, 2048) for eval          |
|                                                                        |
+------------------------------------------------------------------------+
```

#### 3.2.3 Depth Skip Connection Detail

The depth skip block at each scale extracts boundary-relevant features from the monocular depth map. Sobel gradients of depth explicitly encode object boundaries --- surfaces at different depths produce strong gradient responses at their boundaries, regardless of texture or color similarity.

```
+-------------------------------------------------------+
|              Depth Skip Block (per scale)              |
|                                                        |
|  Depth map (1, 512, 1024)                              |
|       |                                                |
|       v                                                |
|  Bilinear downsample to target scale                   |
|  (e.g., 1x64x128 or 1x128x256)                        |
|       |                                                |
|       +---> depth_raw (1ch) --------+                  |
|       |                             |                  |
|       +---> Sobel_x kernel:        |                  |
|       |     [-1, 0, 1]             |                  |
|       |     [-2, 0, 2]      Cat    |                  |
|       |     [-1, 0, 1]    (3ch)    |                  |
|       |     -> grad_x (1ch) -------+                  |
|       |                             |                  |
|       +---> Sobel_y kernel:        |                  |
|             [-1,-2,-1]             |                  |
|             [ 0, 0, 0]             |                  |
|             [ 1, 2, 1]             |                  |
|             -> grad_y (1ch) -------+                  |
|                                     |                  |
|                                     v                  |
|                             +---------------+          |
|                             | Conv2d 3x3    |          |
|                             | (3 -> 32)     |          |
|                             | padding=1     |          |
|                             | GroupNorm(1,32)|          |
|                             | + GELU        |          |
|                             +-------+-------+          |
|                                     |                  |
|                                     v                  |
|                             (32, H, W)                 |
|                             Skip features              |
|                                                        |
+-------------------------------------------------------+

Key insight: Sobel gradients of depth encode object boundaries
at HIGH resolution from the depth map. These boundaries are
available regardless of DINOv2 patch resolution.

Depth gradient magnitude at boundaries:
  - car/road boundary:       strong (depth discontinuity)
  - person/building boundary: strong (foreground/background)
  - bicycle/bicycle boundary: moderate (similar depth, slight offset)
  - road/sidewalk boundary:  weak (similar depth, slight step)
```

#### 3.2.4 Parameter Budget

| Component | Parameters |
|-----------|-----------|
| Semantic Projection (768 -> 192) | 148K |
| Depth-Feature Projection (768 + FiLM -> 192) | 174K |
| Bottleneck: CoupledConvBlock x 2 at 32x64 | 751K |
| Decoder Stage 1: 2x ConvTranspose2d + DepthSkip + Fuse + CoupledBlock | 1,600K |
| Decoder Stage 2: 2x ConvTranspose2d + DepthSkip + Fuse + CoupledBlock | 1,600K |
| Classification Head | 4K |
| **Total** | **4.28M** |

*Note: Each decoder stage requires separate transposed convolutions for both semantic and depth streams (590K each), bringing the total well above the initial estimate. The parameter count (4.28M) is comparable to HiResRefineNet with transposed conv (4.19M).*

---

### 3.3 Option B: Lightweight CNN Encoder with UNet Skip Connections

#### 3.3.1 Rationale

While depth provides geometric boundaries, RGB images contain richer spatial information: texture edges, color discontinuities, and semantic contours invisible to depth. A wall and a person at the same depth produce no depth discontinuity, but their RGB appearance differs markedly. A lightweight CNN encoder applied to downsampled RGB produces multi-scale feature maps that serve as skip connections, providing both geometric and appearance-level boundary detail.

The encoder is intentionally shallow (3 stages, 32/64/128 channels) and uses small receptive fields. Its role is spatial boundary extraction, not semantic understanding --- DINOv2 handles semantics. The encoder weights are learned end-to-end, allowing the network to discover which visual features best complement the frozen DINOv2 representation.

#### 3.3.2 Architecture

```
+------------------------------------------------------------------------+
|          Option B: Lightweight CNN Encoder + UNet Decoder               |
|                                                                        |
|                                                                        |
|  Input Image (3, 256, 512)         DINOv2 ViT-B/14 (frozen)           |
|  (bilinear from 1024x2048)         (768, 32, 64)                      |
|       |                                  |                             |
|       v                                  |                             |
|  +----------------+                      |                             |
|  | Encoder Stage 1|                      |                             |
|  | Conv 3x3 (s=2) |                      |                             |
|  | 3->32, GN, GELU|                      |                             |
|  | Conv 3x3       |                      |                             |
|  | 32->32,GN,GELU |                      |                             |
|  +----+-----------+                      |                             |
|       | skip_1 (32, 128, 256)            |                             |
|       v                                  |                             |
|  +----------------+                      |                             |
|  | Encoder Stage 2|                      |                             |
|  | Conv 3x3 (s=2) |                      |                             |
|  | 32->64,GN,GELU |                      |                             |
|  | Conv 3x3       |                      |                             |
|  | 64->64,GN,GELU |                      |                             |
|  +----+-----------+                      |                             |
|       | skip_2 (64, 64, 128)             |                             |
|       v                                  |                             |
|  +----------------+                      |                             |
|  | Encoder Stage 3|                      |                             |
|  | Conv 3x3 (s=2) |                      |                             |
|  | 64->128,GN,GELU|                      |                             |
|  | Conv 3x3       |                      |                             |
|  |128->128,GN,GELU|                      |                             |
|  +----+-----------+                      |                             |
|       | enc_feat (128, 32, 64)           |                             |
|       |                                  |                             |
|       v                                  v                             |
|  +------------------------------------------------+                   |
|  |              Bottleneck Fusion                  |                   |
|  |                                                 |                   |
|  |  DINOv2 proj (192, 32, 64)                      |                   |
|  |       +                                         |                   |
|  |  enc_feat (128, 32, 64)                         |                   |
|  |       |                                         |                   |
|  |       v                                         |                   |
|  |  Cat -> Conv 1x1 (320->192) + CoupledBlocks x 2|                   |
|  |                                                 |                   |
|  +------------------------+------------------------+                   |
|                           | (192, 32, 64)                              |
|                           v                                            |
|  +----------------------------------------------------+               |
|  |  Decoder Stage 1: 32x64 -> 64x128                  |               |
|  |                                                     |               |
|  |  ConvTranspose2d -------------+                     |               |
|  |  (192->192, s=2)             |                     |               |
|  |                               |  Cat                |               |
|  |  skip_2 (64, 64, 128) ------+                     |               |
|  |                               |  (256, 64, 128)    |               |
|  |                               v                     |               |
|  |                       Fuse Conv 1x1 (256->192)      |               |
|  |                       + CoupledBlock x 1             |               |
|  |                                                     |               |
|  +------------------------+----------------------------+               |
|                           | (192, 64, 128)                             |
|                           v                                            |
|  +----------------------------------------------------+               |
|  |  Decoder Stage 2: 64x128 -> 128x256                |               |
|  |                                                     |               |
|  |  ConvTranspose2d -------------+                     |               |
|  |  (192->192, s=2)             |                     |               |
|  |                               |  Cat                |               |
|  |  skip_1 (32, 128, 256) -----+                     |               |
|  |                               |  (224, 128, 256)   |               |
|  |                               v                     |               |
|  |                       Fuse Conv 1x1 (224->192)      |               |
|  |                       + CoupledBlock x 1             |               |
|  |                                                     |               |
|  +------------------------+----------------------------+               |
|                           | (192, 128, 256)                            |
|                           v                                            |
|                   +---------------+                                    |
|                   | Classification|                                    |
|                   | GN + Conv 1x1 |                                    |
|                   | (192 -> 19)  |                                    |
|                   +-------+-------+                                    |
|                           |                                            |
|                           v                                            |
|                   Refined logits (19, 128, 256)                        |
|                                                                        |
+------------------------------------------------------------------------+
```

#### 3.3.3 Encoder Design

```
+-------------------------------------------------------+
|              Lightweight CNN Encoder                   |
|                                                        |
|  Input: RGB image bilinear-downsampled to (3, 256, 512)|
|                                                        |
|  Stage 1:                                              |
|    Conv2d(3, 32, 3, stride=2, padding=1)               |
|    GroupNorm(1, 32) + GELU                              |
|    Conv2d(32, 32, 3, stride=1, padding=1)              |
|    GroupNorm(1, 32) + GELU                              |
|    Output: (32, 128, 256) ---------- skip_1             |
|                                                        |
|  Stage 2:                                              |
|    Conv2d(32, 64, 3, stride=2, padding=1)              |
|    GroupNorm(1, 64) + GELU                              |
|    Conv2d(64, 64, 3, stride=1, padding=1)              |
|    GroupNorm(1, 64) + GELU                              |
|    Output: (64, 64, 128) ----------- skip_2             |
|                                                        |
|  Stage 3:                                              |
|    Conv2d(64, 128, 3, stride=2, padding=1)             |
|    GroupNorm(1, 128) + GELU                             |
|    Conv2d(128, 128, 3, stride=1, padding=1)            |
|    GroupNorm(1, 128) + GELU                             |
|    Output: (128, 32, 64) --------- bottleneck fusion    |
|                                                        |
|  Total encoder parameters: ~110K                       |
|  Encoder weights are LEARNED (not frozen).             |
|  Learns boundary-relevant features that complement     |
|  DINOv2's semantic representation.                     |
|                                                        |
|  Design choice: Small channel widths (32/64/128)       |
|  because the encoder provides spatial structure,       |
|  not semantic understanding. DINOv2 handles semantics. |
|                                                        |
+-------------------------------------------------------+
```

#### 3.3.4 Parameter Budget

| Component | Parameters |
|-----------|-----------|
| CNN Encoder (3 stages) | 110K |
| DINOv2 Projections (sem + depth-FiLM) | 313K |
| Bottleneck fusion + CoupledBlocks x 2 | 530K |
| Decoder Stage 1: up + fuse + CoupledBlock | 310K |
| Decoder Stage 2: up + fuse + CoupledBlock | 260K |
| Classification Head | 3.7K |
| **Total** | **~1.5M** |

---

### 3.4 Option C: Progressive Refinement Decoder (Decoder-Only Baseline)

#### 3.4.1 Rationale

Options A and B hypothesize that skip connections are necessary for PQ_things recovery. Option C tests this hypothesis by serving as a no-skip baseline with the same multi-scale progressive structure. Instead of refining at a single resolution (as in direct upsampling), computation is distributed across a cascade: more blocks at 32x64 (where convolutions are cheap and global context is captured), fewer blocks at higher resolutions (where fine-grained boundary precision is refined).

If Option C matches Options A/B, skip connections are unnecessary and the multi-scale cascade alone provides sufficient inductive bias. If Options A/B significantly outperform C, the skip connections contribute essential boundary information.

#### 3.4.2 Architecture

```
+------------------------------------------------------------------------+
|        Option C: Progressive Refinement Decoder (No Encoder)           |
|                                                                        |
|                                                                        |
|  DINOv2 ViT-B/14 (frozen)                                             |
|  (768, 32, 64)                                                         |
|       |                                                                |
|       +---------------+                                                |
|       v               v                                                |
|  +---------+   +--------------+                                        |
|  |Semantic |   |Depth-Feature |                                        |
|  |  Proj   |   |Proj + FiLM   |<-- Depth (32x64)                      |
|  |768->192 |   |768->192      |                                        |
|  +----+----+   +------+-------+                                        |
|       |               |                                                |
|       v               v                                                |
|  +------------------------------------------------+                   |
|  | Scale 1: Processing at 32x64 (2,048 tokens)    |                   |
|  |                                                 |                   |
|  | CoupledConvBlock x 2                            |                   |
|  | (global context, semantic decisions)             |                   |
|  |                                                 |                   |
|  | Role: Establish class-level predictions          |                   |
|  +------------------------+------------------------+                   |
|                           | (192, 32, 64)                              |
|                           v                                            |
|  +------------------------------------------------+                   |
|  | Upsample 1: 32x64 -> 64x128                    |                   |
|  |                                                 |                   |
|  | ConvTranspose2d(192, 192, k=4, s=2, p=1)       |                   |
|  | GroupNorm + GELU                                |                   |
|  +------------------------+------------------------+                   |
|                           | (192, 64, 128)                             |
|                           v                                            |
|  +------------------------------------------------+                   |
|  | Scale 2: Processing at 64x128 (8,192 tokens)   |                   |
|  |                                                 |                   |
|  | CoupledConvBlock x 1                            |                   |
|  | (medium-scale boundary refinement)               |                   |
|  |                                                 |                   |
|  | Role: Refine stuff-thing boundaries              |                   |
|  +------------------------+------------------------+                   |
|                           | (192, 64, 128)                             |
|                           v                                            |
|  +------------------------------------------------+                   |
|  | Upsample 2: 64x128 -> 128x256                  |                   |
|  |                                                 |                   |
|  | ConvTranspose2d(192, 192, k=4, s=2, p=1)       |                   |
|  | GroupNorm + GELU                                |                   |
|  +------------------------+------------------------+                   |
|                           | (192, 128, 256)                            |
|                           v                                            |
|  +------------------------------------------------+                   |
|  | Scale 3: Processing at 128x256 (32,768 tokens)  |                   |
|  |                                                 |                   |
|  | CoupledConvBlock x 1                            |                   |
|  | (fine-grained instance boundary precision)       |                   |
|  |                                                 |                   |
|  | Role: Sharpen thing instance boundaries          |                   |
|  +------------------------+------------------------+                   |
|                           | (192, 128, 256)                            |
|                           v                                            |
|                   +---------------+                                    |
|                   | Classification|                                    |
|                   | GN + Conv 1x1 |                                    |
|                   | (192 -> 19)  |                                    |
|                   +-------+-------+                                    |
|                           |                                            |
|                           v                                            |
|                   Refined logits (19, 128, 256)                        |
|                                                                        |
+------------------------------------------------------------------------+
```

#### 3.4.3 Multi-Scale Computation Distribution

```
+------------------------------------------------------------+
|              Computation Pyramid                            |
|                                                             |
|  Resolution   Blocks   Tokens   FLOPs/block   Total FLOPs  |
|  ----------   ------   ------   ----------    -----------   |
|  32 x 64        2      2,048      230K          460K        |
|  64 x 128       1      8,192      920K          920K        |
|  128 x 256      1     32,768    3,700K        3,700K        |
|                                                             |
|  Total: 4 blocks, 5,080K FLOPs                              |
|                                                             |
|  Compare: 4 blocks all at 128x256 = 14,800K FLOPs           |
|  Savings: 66% FLOPs reduction with multi-scale cascade      |
|                                                             |
|  Block allocation visualization:                            |
|                                                             |
|     32x64:    [Block] [Block]    <- most blocks here        |
|                  |                  (cheap, global context)  |
|                  v                                           |
|     64x128:  [  Block  ]         <- one block               |
|                  |                  (medium cost)            |
|                  v                                           |
|    128x256:  [    Block    ]     <- one block               |
|                                    (expensive, fine-grain)   |
|                                                             |
+------------------------------------------------------------+
```

#### 3.4.4 Parameter Budget

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

---

## 4. Comparative Analysis

### 4.1 Architecture Comparison

| Property | A: Depth-Guided UNet | B: CNN Encoder UNet | C: Progressive Decoder |
|----------|---------------------|--------------------|-----------------------|
| Parameters | ~1.4M | ~1.5M | ~1.5M |
| Skip connections | Depth + Sobel grads | Learned RGB features | None |
| Encoder required | No | Yes (lightweight CNN) | No |
| Additional data loading | None (depth already loaded) | Raw RGB images (new) | None |
| Boundary signal type | Geometric (depth edges) | Geometric + appearance | None (implicit only) |
| Relative FLOPs | 1.0x | 1.2x | 0.66x |
| Implementation complexity | Medium | High | Low |
| Primary risk | Depth estimation noise | Encoder overfitting | No new boundary info |

### 4.2 Skip Connection Information Content

```
+----------------------------------------------------------------+
|           Information available at each decoder scale            |
|                                                                  |
|  Option A (Depth):                                               |
|    Stage 1 (64x128):  depth value + dx + dy = 3 raw channels    |
|                       -> 32 learned features                     |
|    Stage 2 (128x256): depth value + dx + dy = 3 raw channels    |
|                       -> 32 learned features                     |
|    Total skip info:   geometric boundaries only                  |
|    Blind spot:        same-depth adjacent objects (e.g., two     |
|                       cars side-by-side at same distance)         |
|                                                                  |
|  Option B (RGB):                                                 |
|    Stage 1 (64x128):  64 learned features from RGB              |
|    Stage 2 (128x256): 32 learned features from RGB              |
|    Total skip info:   texture + color + geometric boundaries     |
|    Blind spot:        similarly-textured adjacent objects         |
|                       (rare in practice)                          |
|                                                                  |
|  Option C (None):                                                |
|    Stage 1 (64x128):  none --- upsampled features only           |
|    Stage 2 (128x256): none --- upsampled features only           |
|    Total skip info:   zero external boundary information         |
|    Blind spot:        all sub-patch boundaries                   |
|                                                                  |
+----------------------------------------------------------------+
```

### 4.3 Theoretical Advantage of Depth Skips for Thing Classes

Thing instances in driving scenes are typically at different depths from their surroundings. The depth skip provides a natural instance separation signal:

```
  Depth profile along horizontal scanline through two adjacent cars:

  Depth
    ^
    |          +-----+     +-----+
    |          | Car |     | Car |
    |          |  A  |     |  B  |
    |  --------+     +-----+     +--------  background
    |
    +-------------------------------------------> x

  Depth gradient (Sobel_x):
    ^
    |     +                 +
    |     |                 |     <- positive edges (near->far)
    |     |  -         -    |
    |     |  |         |    |     <- negative edges (far->near)
    +-------------------------------------------> x

  The depth gradient produces FOUR strong responses at the
  boundaries of two adjacent cars, even when the cars have
  similar RGB appearance. This is exactly the boundary
  information needed for PQ_things.
```

---

## 5. Training Configuration

All options share the same training setup as the HiRes RefineNet baseline for fair comparison.

| Parameter | Value |
|-----------|-------|
| Output resolution | 128x256 |
| bridge_dim | 192 |
| Bottleneck blocks | 2 (at 32x64) |
| Decoder blocks per stage | 1 |
| batch_size | 4 (2 if OOM) |
| lr | 5e-5 |
| scheduler | Cosine annealing |
| epochs | 20 |
| eval_interval | 2 |
| distill_min | 0.85 |
| lambda_align | 0.25 |
| lambda_proto | 0.025 |
| lambda_ent | 0.025 |
| label_smoothing | 0.1 |
| num_classes | 19 |
| device | MPS (Apple M4 Pro, 48GB) |

### 5.1 Data Requirements

| Option | DINOv2 Features | Depth Maps | RGB Images | Pseudo-labels |
|--------|----------------|------------|------------|---------------|
| A | Yes (32x64) | Yes (512x1024) | No | Yes (128x256) |
| B | Yes (32x64) | Yes (32x64 for FiLM) | **Yes (256x512)** | Yes (128x256) |
| C | Yes (32x64) | Yes (32x64 for FiLM) | No | Yes (128x256) |

Option B requires loading raw RGB images at 256x512, which are not currently part of the data pipeline. This adds I/O overhead and requires augmenting the dataset class.

---

## 6. Experimental Design

### 6.1 Ablation Order

We follow a sequential ablation strategy that maximizes information gain:

1. **Option C (Progressive Decoder)** --- the no-skip baseline. Establishes whether multi-scale cascading alone (without skip connections) improves over flat 128x256 processing. If C matches or exceeds HiRes RefineNet, skip connections are unnecessary.

2. **Option A (Depth-Guided UNet)** --- tests the skip connection hypothesis with minimal infrastructure change. If A significantly outperforms C, skip connections provide essential boundary information. If A matches C, depth boundaries do not help beyond what the refinement blocks already capture.

3. **Option B (CNN Encoder UNet)** --- only if A outperforms C, validating that skip connections matter. Tests whether learned RGB features provide additional boundary detail beyond depth-only skips. If B matches A, depth is sufficient. If B outperforms A, appearance-level boundaries contribute.

### 6.2 Hypothesis Testing

| Hypothesis | Comparison | Success Criterion |
|-----------|------------|-------------------|
| H1: Multi-scale cascading improves over flat processing | C vs HiRes RefineNet | PQ(C) > PQ(HiRes) |
| H2: Skip connections recover PQ_things | A vs C | PQ_things(A) > PQ_things(C) by >= 1.0 |
| H3: Depth boundaries sufficient for things | A vs B | PQ_things(A) >= PQ_things(B) - 0.5 |
| H4: Learned features add to depth skips | B vs A | PQ_things(B) > PQ_things(A) by >= 0.5 |
| H5: Best UNet recovers input PQ_things | Best vs input | PQ_things >= 19.0 |
| H6: UNet preserves semantic gains | All vs 32x64 | mIoU >= 55.0 for all options |

---

## 7. Results

### 7.1 Main Results

| Config | PQ | PQ_stuff | PQ_things | mIoU | changed_pct | Time/Epoch |
|--------|-----|----------|-----------|------|-------------|------------|
| Input pseudo-labels | 26.74 | 32.08 | 19.41 | ~50% | --- | --- |
| 32x64 Conv2d (Run D) | 26.52 | 33.38 | 17.10 | 55.31 | 5.5% | ~96s |
| 128x256 HiRes RefineNet (U-B) | ___ | ___ | ___ | ___ | ___ | ~948s |
| C: Progressive Decoder | ___ | ___ | ___ | ___ | ___ | ___ |
| A: Depth-Guided UNet | ___ | ___ | ___ | ___ | ___ | ___ |
| B: CNN Encoder UNet | ___ | ___ | ___ | ___ | ___ | ___ |

### 7.2 PQ_things Recovery Analysis

| Config | PQ_things | Delta vs Input (19.41) | Delta vs 32x64 (17.10) |
|--------|-----------|----------------------|----------------------|
| 128x256 HiRes (U-B) | ___ | ___ | ___ |
| C: Progressive Decoder | ___ | ___ | ___ |
| A: Depth-Guided UNet | ___ | ___ | ___ |
| B: CNN Encoder UNet | ___ | ___ | ___ |

### 7.3 Per-Class Thing PQ Comparison

| Class | Input | 32x64 | HiRes U-B | Option C | Option A | Option B |
|-------|-------|-------|-----------|----------|----------|----------|
| person | ___ | ___ | ___ | ___ | ___ | ___ |
| rider | ___ | ___ | ___ | ___ | ___ | ___ |
| car | ___ | ___ | ___ | ___ | ___ | ___ |
| truck | ___ | ___ | ___ | ___ | ___ | ___ |
| bus | ___ | ___ | ___ | ___ | ___ | ___ |
| train | ___ | ___ | ___ | ___ | ___ | ___ |
| motorcycle | ___ | ___ | ___ | ___ | ___ | ___ |
| bicycle | ___ | ___ | ___ | ___ | ___ | ___ |

### 7.4 Semantic Quality (Per-Class mIoU)

| Class | Input | 32x64 | HiRes U-B | Option C | Option A | Option B |
|-------|-------|-------|-----------|----------|----------|----------|
| road | ___ | ___ | ___ | ___ | ___ | ___ |
| sidewalk | ___ | ___ | ___ | ___ | ___ | ___ |
| building | ___ | ___ | ___ | ___ | ___ | ___ |
| wall | ___ | ___ | ___ | ___ | ___ | ___ |
| fence | ___ | ___ | ___ | ___ | ___ | ___ |
| pole | ___ | ___ | ___ | ___ | ___ | ___ |
| traffic light | ___ | ___ | ___ | ___ | ___ | ___ |
| traffic sign | ___ | ___ | ___ | ___ | ___ | ___ |
| vegetation | ___ | ___ | ___ | ___ | ___ | ___ |
| terrain | ___ | ___ | ___ | ___ | ___ | ___ |
| sky | ___ | ___ | ___ | ___ | ___ | ___ |
| person | ___ | ___ | ___ | ___ | ___ | ___ |
| rider | ___ | ___ | ___ | ___ | ___ | ___ |
| car | ___ | ___ | ___ | ___ | ___ | ___ |
| truck | ___ | ___ | ___ | ___ | ___ | ___ |
| bus | ___ | ___ | ___ | ___ | ___ | ___ |
| train | ___ | ___ | ___ | ___ | ___ | ___ |
| motorcycle | ___ | ___ | ___ | ___ | ___ | ___ |
| bicycle | ___ | ___ | ___ | ___ | ___ | ___ |

### 7.5 Computational Efficiency

| Config | Parameters | Time/Epoch | Memory (BS=4) | PQ per 100K Params |
|--------|-----------|------------|---------------|-------------------|
| 32x64 Conv2d | 1.83M | ~96s | ~3GB | ___ |
| 128x256 HiRes (U-B) | 3.44M | ~948s | ___ | ___ |
| C: Progressive Decoder | ~1.5M | ___ | ___ | ___ |
| A: Depth-Guided UNet | ~1.4M | ___ | ___ | ___ |
| B: CNN Encoder UNet | ~1.5M | ___ | ___ | ___ |

### 7.6 Training Curves

*Learning dynamics for PQ, PQ_stuff, PQ_things, and mIoU across epochs for all configurations. To be plotted after experiments.*

---

## 8. Discussion

### 8.1 Do Skip Connections Help?

*Analysis of Option A vs Option C. If depth skip connections provide measurable PQ_things improvement, this validates the information bottleneck hypothesis --- upsampled features lack boundary detail that auxiliary modalities can provide.*

### 8.2 Depth vs RGB Skip Connections

*Analysis of Option A vs Option B. Depth provides geometric boundaries; RGB provides geometric + appearance boundaries. The question is whether appearance-level detail (texture, color) adds meaningful boundary information beyond depth discontinuities for unsupervised panoptic segmentation.*

### 8.3 Multi-Scale Cascade vs Flat Processing

*Analysis of Option C vs HiRes RefineNet (flat 128x256). The progressive decoder distributes blocks across scales (2 at 32x64, 1 at 64x128, 1 at 128x256) while HiRes RefineNet applies all 4 blocks at 128x256. If the cascade performs comparably with 66% fewer FLOPs, multi-scale is more efficient.*

### 8.4 The PQ_things Ceiling

*Even with skip connections, PQ_things may be limited by the quality of instance pseudo-labels (SPIdepth depth-guided splitting) rather than semantic refinement quality. If all UNet variants plateau at a similar PQ_things, the bottleneck has shifted from semantic resolution to instance segmentation quality.*

### 8.5 Limitations

- All options still rely on upsampled DINOv2 features for semantic content. Skip connections add boundary detail but cannot introduce new semantic information.
- Monocular depth estimation (SPIdepth) introduces systematic errors: reflective surfaces, transparent objects, and textureless regions produce unreliable depth boundaries.
- Option B requires RGB image loading, adding ~20% I/O overhead to the training pipeline.
- MPS float32 constraint prevents mixed-precision training, inflating memory usage and training time relative to CUDA baselines.
- Panoptic evaluation uses connected-component analysis, which may interact unpredictably with resolution and boundary sharpness.

---

## 9. Conclusion

*To be completed after all experiments. Expected narrative: whether UNet-style skip connections from auxiliary modalities can close the PQ_things gap that direct upsampling leaves open, and which source of boundary information (depth vs RGB) is most effective for unsupervised panoptic refinement.*

---

## Appendix A: Relationship to Prior HiRes RefineNet Experiments

This work builds on the HiRes RefineNet upsampling ablation (Phase 1). The UNet decoder replaces the flat upsampling + refinement pipeline with a structured multi-scale decoder. Key differences:

| Property | HiRes RefineNet | UNet Decoder |
|----------|----------------|--------------|
| Upsampling | Single-stage (32x64 -> 128x256) | Progressive (32x64 -> 64x128 -> 128x256) |
| Skip connections | None | Depth (A), RGB (B), or None (C) |
| Block distribution | All 4 blocks at 128x256 | 2 at 32x64, 1 at 64x128, 1 at 128x256 |
| FLOPs | High (all blocks at max resolution) | Lower (pyramid distribution) |
| Boundary information | Implicit (from upsampled features) | Explicit (from depth/RGB skips) |

## Appendix B: Memory Budget Estimates (MPS, float32, batch=4)

| Component | Option A | Option B | Option C |
|-----------|----------|----------|----------|
| DINOv2 features (input) | 150MB | 150MB | 150MB |
| Depth maps (multi-scale) | 50MB | 10MB | 10MB |
| RGB images (256x512) | --- | 25MB | --- |
| CNN encoder activations | --- | 200MB | --- |
| Bottleneck (32x64) | 200MB | 250MB | 200MB |
| Decoder stage 1 (64x128) | 400MB | 400MB | 400MB |
| Decoder stage 2 (128x256) | 800MB | 800MB | 800MB |
| Gradients + optimizer | ~2x model | ~2x model | ~2x model |
| **Estimated total** | **~4GB** | **~5GB** | **~4GB** |

All configurations fit well within 48GB unified memory on MPS.
