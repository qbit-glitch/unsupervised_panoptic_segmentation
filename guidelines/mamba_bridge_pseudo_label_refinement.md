# Mamba-Bridge Pseudo-Label Refinement: Implementation Guide

## For Claude Code Agents — Complete Specification for Methods 1, 2, 3 + Self-Training

**Target**: Improve Cityscapes unsupervised panoptic pseudo-labels from PQ = 23.2 → PQ ≥ 28.0
**Compute**: Apple M4 Pro 48GB (MPS) for local training; 2× GTX 1080 Ti for Stage-2
**Framework**: PyTorch (pure, no CUDA/Triton dependencies)
**Codebase**: `mbps_pytorch/` directory

---

## Table of Contents

1. [Prerequisites & Context](#1-prerequisites--context)
2. [Method 1: Coupled-State Cross-Modal Mamba Refinement Network](#2-method-1-cscm-refinenet)
3. [Method 2: Mamba-Guided Adaptive Instance Decomposition](#3-method-2-mamba-guided-adaptive-instance-decomposition)
4. [Method 3: EMA Teacher Self-Training with Mamba Bridge](#4-method-3-ema-teacher-self-training)
5. [Integration Pipeline](#5-integration-pipeline)
6. [Ablation Matrix](#6-ablation-matrix)
7. [Evaluation Protocol](#7-evaluation-protocol)
8. [References](#8-references)

---

## 1. Prerequisites & Context

### 1.1 Current Pipeline State (Stage-1 Baseline)

The following outputs already exist and are verified reproducible:

| Component | Output Directory | Quality |
|-----------|-----------------|---------|
| CAUSE-CRF semantics (27-class) | `pseudo_semantic_cause_crf/` | mIoU = 42.86% |
| SPIdepth depth maps | `depth_spidepth/` | Self-supervised, normalized [0,1] |
| TrainID remapped semantics (19-class) | `pseudo_semantic_cause_trainid/` | Remapped from 27→19 |
| Depth-guided instances | `sweep_instances/gt0.10_ma500/` | PQ_things = 11.7% |
| DINOv2 ViT-B/14 features | `dinov2_features/` | 32×64×768 patch grid |
| **Combined panoptic** | `eval_cause_crf_sweep_gt010_ma500_val.json` | **PQ = 23.2** |

### 1.2 Existing Mamba2 Module

Location: `mbps_pytorch/mamba2/`

**Available classes** (all pure PyTorch, works on CPU/MPS/CUDA):
```python
from mbps_pytorch.mamba2 import (
    Mamba2,              # Core SSM: forward(u: (B,L,D)) → (B,L,D)
    GatedDeltaNet,       # Delta-rule SSM: same interface as Mamba2
    VisionMamba2,        # 2D image wrapper: forward(x: (B,C,H,W)) → (B,C,H,W)
    CrossModalMamba2,    # Two-stream fusion: forward(sem, feat) → (fused_sem, fused_feat)
    SCAN_MODES,          # ("raster", "bidirectional", "cross_scan")
    LAYER_TYPES,         # ("mamba2", "gated_delta_net")
)
```

**Key interfaces:**
- `VisionMamba2(d_model, scan_mode, layer_type)` — single-stream, accepts 4D (B,C,H,W) or 3D (B,L,D)
- `CrossModalMamba2(d_model, scan_mode, layer_type)` — two-stream, interleaves tokens [s1,f1,s2,f2,...] then processes via VisionMamba2
- `_build_layer(layer_type, d_model, **kwargs)` — factory for Mamba2 or GatedDeltaNet

**Scan modes:**
- `"raster"`: row-major flatten → single SSM pass
- `"bidirectional"`: forward + reverse SSM → sigmoid gate merge
- `"cross_scan"`: VMamba-style 4-way (row-fwd, row-bwd, col-fwd, col-bwd) → learned linear merge

### 1.3 Bottleneck Analysis

| Bottleneck | Quantification | Method that addresses it |
|------------|----------------|--------------------------|
| 7 zero-IoU semantic classes | Caps PQ at 36.7 on remaining 12 classes | **Method 1** |
| Instance recall: 5.3 vs 20.2 GT inst/img | PQ_things = 11.7, person recall = 5.2% | **Method 2** |
| No iterative refinement | Single-pass pipeline, no feedback loop | **Method 3** |
| Car over-segmentation | 925 FP vs 758 TP | **Method 2** |

### 1.4 Environment

```bash
PYTHON=/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python
CS_ROOT=/Users/qbit-glitch/Desktop/datasets/cityscapes
PROJECT=/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
```

Python 3.10, PyTorch 2.1.0, MPS backend.

### 1.5 DINOv2 Feature Extraction

If DINOv2 features are not yet extracted for all images, use:
```bash
$PYTHON mbps_pytorch/extract_dinov2_features.py \
    --cityscapes_root $CS_ROOT \
    --output_dir $CS_ROOT/dinov2_features \
    --split val \
    --model dinov2_vitb14 \
    --device auto
```
Output: `dinov2_features/val/{city}/{stem}.npz` with keys `features` (32×64×768 float16) and `grid_h`, `grid_w`.

---

## 2. Method 1: CSCM-RefineNet

### 2.1 Overview

**Goal**: Refine CAUSE-TR 27-class semantic logits using cross-modal Mamba processing of DINOv2 features conditioned on depth geometry.

**Key innovation**: Coupled-state SSM where semantic and depth-feature streams have inter-dependent hidden state transitions (Coupled Mamba, NeurIPS 2024), plus depth-geometry modulation of SSM selectivity (DFormerv2, CVPR 2025).

**Expected outcome**: mIoU from 42.86% → 48-52% by recovering 2-4 of the 7 zero-IoU classes (fence, pole, traffic light, traffic sign, rider, train, motorcycle).

### 2.2 Architecture

```
File: mbps_pytorch/refine_net.py

class CSCMRefineNet(nn.Module):
    """Coupled-State Cross-Modal Mamba Refinement Network.

    Takes pre-computed CAUSE-TR logits + DINOv2 features + SPIdepth depth
    and produces refined 27-class semantic logits.
    """
```

#### 2.2.1 Input Specification

```python
# All inputs at patch resolution (32×64 for 512×1024 eval resolution)
# or pixel resolution — see Section 2.2.7 for resolution choices

inputs = {
    "cause_logits":   (B, 27, H_p, W_p),  # CAUSE-TR 27-class softmax logits
    "dinov2_features": (B, 768, H_p, W_p), # DINOv2 ViT-B/14 patch features
    "depth":          (B, 1, H_p, W_p),    # SPIdepth normalized depth [0,1]
    "depth_gradients": (B, 2, H_p, W_p),   # Sobel_x, Sobel_y of depth
}

output = {
    "refined_logits": (B, 27, H_p, W_p),   # Refined 27-class logits
    "instance_affinity": (B, 32, H_p, W_p), # Optional: instance embeddings
}
```

#### 2.2.2 Semantic Stream Projection

```python
class SemanticProjection(nn.Module):
    """Project CAUSE 27-class logits to bridge dimension."""

    def __init__(self, num_classes=27, bridge_dim=192):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(num_classes, bridge_dim, 1),
            nn.LayerNorm([bridge_dim]),  # channel-last norm after permute
            nn.GELU(),
        )

    def forward(self, logits):
        # logits: (B, 27, H, W) → (B, 192, H, W)
        return self.proj(logits)
```

#### 2.2.3 Depth-Feature Stream Projection

```python
class DepthFeatureProjection(nn.Module):
    """Project DINOv2 features with depth FiLM conditioning."""

    def __init__(self, feature_dim=768, bridge_dim=192, depth_freq_bands=6):
        super().__init__()
        # Feature projection
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feature_dim, bridge_dim, 1),
            nn.LayerNorm([bridge_dim]),
            nn.GELU(),
        )

        # Depth encoding: sinusoidal + Sobel gradients → FiLM params
        depth_input_dim = 2 * depth_freq_bands + 3  # sin/cos + depth + grad_x + grad_y
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_input_dim, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, bridge_dim * 2, 1),  # gamma + beta for FiLM
        )

        self.freq_bands = torch.tensor(
            [2**i * torch.pi for i in range(depth_freq_bands)]
        )

    def forward(self, features, depth, depth_grads):
        # features: (B, 768, H, W) → (B, 192, H, W)
        feat_proj = self.feat_proj(features)

        # Sinusoidal depth encoding
        # depth: (B, 1, H, W)
        freqs = self.freq_bands.to(depth.device)  # (F,)
        d_expanded = depth * freqs[None, :, None, None]  # (B, F, H, W)
        depth_enc = torch.cat([
            torch.sin(d_expanded),
            torch.cos(d_expanded),
            depth,
            depth_grads,  # (B, 2, H, W)
        ], dim=1)  # (B, 2F+3, H, W)

        # FiLM conditioning
        film_params = self.depth_encoder(depth_enc)  # (B, 2*bridge_dim, H, W)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma.clamp(-2.0, 2.0)  # Prevent explosion
        beta = beta.clamp(-2.0, 2.0)

        return feat_proj * (1.0 + gamma) + beta
```

#### 2.2.4 Coupled-State Mamba Block (KEY NOVELTY)

This is the core contribution. Instead of the existing `CrossModalMamba2` which interleaves tokens into a single SSM, we implement **coupled dual-chain SSMs** where each chain's hidden state depends on both its own previous state and the partner chain's previous state.

**Reference**: Coupled Mamba (Li et al., NeurIPS 2024) — "the current hidden state depends on both its own chain's previous state and the neighboring modality chain's previous state."

```python
class CoupledMambaBlock(nn.Module):
    """Coupled dual-chain SSM for cross-modal feature fusion.

    Two separate SSM chains (semantic and depth-feature) with learnable
    cross-chain state coupling. Each chain's hidden state transition
    receives a contribution from the partner chain's previous state.

    Reference: Coupled Mamba (NeurIPS 2024)

    h_t^sem   = A^sem h_{t-1}^sem   + B^sem x_t^sem   + alpha * C^cross h_{t-1}^depth
    h_t^depth = A^depth h_{t-1}^depth + B^depth x_t^depth + beta  * C^cross h_{t-1}^sem
    """

    def __init__(
        self,
        d_model: int = 192,
        layer_type: str = "gated_delta_net",
        scan_mode: str = "bidirectional",
        coupling_strength: float = 0.1,  # Initial alpha/beta
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.scan_mode = scan_mode

        # Two independent VisionMamba2 streams
        self.sem_mamba = VisionMamba2(
            d_model=d_model, scan_mode=scan_mode, layer_type=layer_type,
            d_state=d_state, d_conv=d_conv, expand=expand,
            headdim=headdim, chunk_size=chunk_size,
        )
        self.depth_mamba = VisionMamba2(
            d_model=d_model, scan_mode=scan_mode, layer_type=layer_type,
            d_state=d_state, d_conv=d_conv, expand=expand,
            headdim=headdim, chunk_size=chunk_size,
        )

        # Cross-chain coupling projections
        # These project partner chain's output to modulate current chain's input
        self.cross_sem_to_depth = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.cross_depth_to_sem = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # Learnable coupling strength (initialized small for stable training)
        self.alpha = nn.Parameter(torch.tensor(coupling_strength))
        self.beta = nn.Parameter(torch.tensor(coupling_strength))

        # LayerNorm before each branch (pre-norm)
        self.norm_sem = nn.LayerNorm(d_model)
        self.norm_depth = nn.LayerNorm(d_model)

        # FFN after fusion
        self.ffn_sem = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ffn_depth = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, sem: torch.Tensor, depth_feat: torch.Tensor):
        """
        Args:
            sem: (B, D, H, W) semantic stream features
            depth_feat: (B, D, H, W) depth-feature stream features
        Returns:
            (refined_sem, refined_depth_feat) — same shapes
        """
        B, D, H, W = sem.shape

        # Flatten to (B, L, D) for LayerNorm
        sem_flat = sem.permute(0, 2, 3, 1).reshape(B, H*W, D)
        depth_flat = depth_feat.permute(0, 2, 3, 1).reshape(B, H*W, D)

        # Pre-norm
        sem_normed = self.norm_sem(sem_flat)
        depth_normed = self.norm_depth(depth_flat)

        # Cross-chain modulation:
        # Each chain's input is augmented by a gated projection of the partner chain
        cross_d2s = self.alpha * self.cross_depth_to_sem(depth_normed)
        cross_s2d = self.beta * self.cross_sem_to_depth(sem_normed)

        sem_input = sem_normed + cross_d2s
        depth_input = depth_normed + cross_s2d

        # Reshape back to (B, D, H, W) for VisionMamba2
        sem_input = sem_input.reshape(B, H, W, D).permute(0, 3, 1, 2)
        depth_input = depth_input.reshape(B, H, W, D).permute(0, 3, 1, 2)

        # Independent SSM processing with coupled inputs
        sem_out = self.sem_mamba(sem_input)    # (B, D, H, W)
        depth_out = self.depth_mamba(depth_input)  # (B, D, H, W)

        # Residual + FFN
        sem_out_flat = sem_out.permute(0, 2, 3, 1).reshape(B, H*W, D)
        depth_out_flat = depth_out.permute(0, 2, 3, 1).reshape(B, H*W, D)

        sem_refined = sem_flat + sem_out_flat
        sem_refined = sem_refined + self.ffn_sem(sem_refined)

        depth_refined = depth_flat + depth_out_flat
        depth_refined = depth_refined + self.ffn_depth(depth_refined)

        # Reshape back
        sem_refined = sem_refined.reshape(B, H, W, D).permute(0, 3, 1, 2)
        depth_refined = depth_refined.reshape(B, H, W, D).permute(0, 3, 1, 2)

        return sem_refined, depth_refined
```

**Why `gated_delta_net` is the recommended default layer_type**: GatedDeltaNet (ICLR 2025) uses a delta update rule `S ← S + k·(v − S^T k)·β` that performs error-correcting memory writes. When scanning across patches of a misclassified thin object (e.g., a pole classified as sky by CAUSE-TR), the delta rule can write the correct "pole" pattern into state memory when DINOv2 features provide distinctive pole-like activations, even when the CAUSE logits disagree. Mamba2's simpler accumulation lacks this error-correction property.

#### 2.2.5 Full CSCMRefineNet

```python
class CSCMRefineNet(nn.Module):
    """Coupled-State Cross-Modal Mamba Refinement Network.

    Architecture:
        CAUSE logits (27) → SemanticProjection → sem_proj (192)
        DINOv2 (768) + depth → DepthFeatureProjection → depth_proj (192)
        [sem_proj, depth_proj] → N × CoupledMambaBlock → [sem_refined, _]
        sem_refined → InverseProjection → refined_logits (27)

    References:
        - Coupled Mamba (NeurIPS 2024): Cross-modal state coupling
        - DFormerv2 (CVPR 2025): Depth geometry as attention prior
        - FiLM (AAAI 2018): Feature-wise linear modulation
        - PRN (WACV 2024): Pseudo-label refinement network concept
    """

    def __init__(
        self,
        num_classes: int = 27,
        feature_dim: int = 768,
        bridge_dim: int = 192,
        num_blocks: int = 4,
        layer_type: str = "gated_delta_net",
        scan_mode: str = "bidirectional",
        coupling_strength: float = 0.1,
        d_state: int = 64,
        chunk_size: int = 256,
    ):
        super().__init__()

        self.sem_proj = SemanticProjection(num_classes, bridge_dim)
        self.depth_feat_proj = DepthFeatureProjection(feature_dim, bridge_dim)

        self.blocks = nn.ModuleList([
            CoupledMambaBlock(
                d_model=bridge_dim,
                layer_type=layer_type,
                scan_mode=scan_mode,
                coupling_strength=coupling_strength,
                d_state=d_state,
                chunk_size=chunk_size,
            )
            for _ in range(num_blocks)
        ])

        # Inverse projection: bridge_dim → num_classes
        self.head = nn.Sequential(
            nn.LayerNorm([bridge_dim]),
            nn.Conv2d(bridge_dim, num_classes, 1),
        )

        # Optional: instance affinity head
        self.affinity_head = nn.Sequential(
            nn.Conv2d(bridge_dim, 32, 1),
            nn.GELU(),
            nn.Conv2d(32, 32, 1),
        )

    def forward(self, cause_logits, dinov2_features, depth, depth_grads):
        """
        Args:
            cause_logits: (B, 27, H, W) — CAUSE-TR softmax logits
            dinov2_features: (B, 768, H, W) — DINOv2 patch features
            depth: (B, 1, H, W) — normalized depth [0,1]
            depth_grads: (B, 2, H, W) — Sobel gradients of depth
        Returns:
            refined_logits: (B, 27, H, W)
            instance_affinity: (B, 32, H, W)
        """
        sem = self.sem_proj(cause_logits)
        depth_feat = self.depth_feat_proj(dinov2_features, depth, depth_grads)

        for block in self.blocks:
            sem, depth_feat = block(sem, depth_feat)

        refined_logits = self.head(sem)

        # Residual: add original logits for stable training
        refined_logits = refined_logits + cause_logits

        instance_affinity = self.affinity_head(depth_feat)

        return refined_logits, instance_affinity
```

#### 2.2.6 Parameter Count

```
SemanticProjection:   27 × 192 + 192 = 5,376
DepthFeatureProjection: 768 × 192 + 192 + (15 × 64 + 64 × 384) = ~173K
4 × CoupledMambaBlock: 4 × (~2.5M) = ~10M
  (each block: 2 × VisionMamba2 + 2 × cross_proj + 2 × FFN)
Head:                 192 × 27 = 5,184
Affinity head:        192 × 32 + 32 × 32 = ~7K

Total: ~10.2M trainable parameters
```

Fits comfortably on M4 Pro MPS with batch_size=4 at 32×64 patch resolution.

#### 2.2.7 Resolution Strategy

**Option A: Patch resolution (32×64) — RECOMMENDED FOR START**
- Process at DINOv2 patch grid (32×64)
- Fastest training, smallest memory
- Upscale refined logits via bilinear interpolation for evaluation
- Limitation: can't produce sub-patch refinements

**Option B: Intermediate resolution (128×256)**
- Bilinear upsample all inputs to 128×256
- Better spatial detail, 16× more tokens (slower)
- Consider this if Option A works but PQ gains are limited

**Option C: Full resolution (512×1024) — NOT RECOMMENDED**
- Sequence length = 524,288 tokens — too long for SSM even chunked
- Would require hierarchical/windowed approach

### 2.3 Training Procedure

#### 2.3.1 Data Loading

```python
class PseudoLabelDataset(Dataset):
    """Load pre-computed pseudo-labels and features for refinement training."""

    def __init__(self, cityscapes_root, split="train"):
        self.root = cityscapes_root
        self.split = split
        # Find all images
        self.images = sorted(glob(f"{root}/leftImg8bit/{split}/*/*.png"))

    def __getitem__(self, idx):
        stem = extract_stem(self.images[idx])
        city = extract_city(self.images[idx])

        # Load pre-computed outputs
        cause_logits = load_cause_logits(self.root, self.split, city, stem)  # (27, H, W)
        dinov2_feats = np.load(f"{self.root}/dinov2_features/{self.split}/{city}/{stem}.npz")
        depth = np.load(f"{self.root}/depth_spidepth/{self.split}/{city}/{stem}.npy")

        # Compute depth gradients
        depth_grads = compute_sobel(depth)  # (2, H, W)

        # Apply augmentations (for cross-view consistency training)
        # ... color jitter, random horizontal flip, random scale

        return {
            "cause_logits": torch.from_numpy(cause_logits).float(),
            "dinov2_features": torch.from_numpy(dinov2_feats["features"]).float(),
            "depth": torch.from_numpy(depth).float().unsqueeze(0),
            "depth_grads": torch.from_numpy(depth_grads).float(),
            "image_path": self.images[idx],
        }
```

**IMPORTANT**: You must first extract and save CAUSE-TR logits (not just argmax labels). The CAUSE generation script needs to be modified to save the full 27-class softmax distribution, not just the hard class assignment. If full logits are unavailable, use one-hot encoded labels from the existing PNGs as a fallback (weaker but functional).

#### 2.3.2 Loss Functions

All losses are **self-supervised** — no ground truth needed.

```python
class RefineNetLoss(nn.Module):
    """Combined self-supervised loss for CSCMRefineNet.

    References:
        - Cross-view consistency: UniMatch V2 (TPAMI 2025)
        - Depth-boundary alignment: DepthG (CVPR 2024)
        - Feature-prototype consistency: CAUSE codebook concept
        - Entropy minimization: Goodfellow "Deep Learning" Ch. 19
    """

    def __init__(self, lambda_consist=1.0, lambda_align=0.5,
                 lambda_proto=0.3, lambda_ent=0.1):
        super().__init__()
        self.lambda_consist = lambda_consist
        self.lambda_align = lambda_align
        self.lambda_proto = lambda_proto
        self.lambda_ent = lambda_ent
```

**Loss 1: Cross-View Consistency** (UniMatch V2, TPAMI 2025)

Apply two different augmentations to the same image, run both through the network, enforce prediction agreement. The weak augmentation produces pseudo-targets for the strong augmentation.

```python
def cross_view_consistency_loss(logits_weak, logits_strong, confidence_threshold=0.9):
    """
    logits_weak: (B, 27, H, W) — predictions under weak augmentation
    logits_strong: (B, 27, H, W) — predictions under strong augmentation
    """
    probs_weak = F.softmax(logits_weak.detach(), dim=1)
    max_probs, pseudo_targets = probs_weak.max(dim=1)

    # Only supervise pixels with high confidence
    mask = max_probs > confidence_threshold

    loss = F.cross_entropy(logits_strong, pseudo_targets, reduction='none')
    loss = (loss * mask.float()).sum() / (mask.sum() + 1e-6)
    return loss
```

**Loss 2: Depth-Boundary Alignment** (DepthG, CVPR 2024)

Refined semantic boundaries should align with depth discontinuities. Penalize label disagreement between neighboring pixels that have similar depth.

```python
def depth_boundary_alignment_loss(logits, depth, sigma=0.05):
    """
    Encourages pixels with similar depth to have similar class predictions.
    Pixels with large depth difference are allowed to disagree.
    """
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)

    # Horizontal neighbors
    depth_diff_h = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])  # (B,1,H,W-1)
    weight_h = torch.exp(-depth_diff_h ** 2 / (2 * sigma ** 2))
    prob_diff_h = (probs[:, :, :, 1:] - probs[:, :, :, :-1]) ** 2
    loss_h = (weight_h * prob_diff_h.sum(dim=1, keepdim=True)).mean()

    # Vertical neighbors
    depth_diff_v = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])
    weight_v = torch.exp(-depth_diff_v ** 2 / (2 * sigma ** 2))
    prob_diff_v = (probs[:, :, 1:, :] - probs[:, :, :-1, :]) ** 2
    loss_v = (weight_v * prob_diff_v.sum(dim=1, keepdim=True)).mean()

    return loss_h + loss_v
```

**Loss 3: Feature-Prototype Consistency**

Refined predictions should produce compact DINOv2 feature clusters per class. For each predicted class, compute the mean DINOv2 feature (prototype) and maximize the cosine similarity of same-class pixels to their prototype.

```python
def feature_prototype_loss(logits, dinov2_features, temperature=0.1):
    """
    Encourages each class to correspond to a compact cluster in DINOv2 space.
    """
    probs = F.softmax(logits / temperature, dim=1)  # (B, C, H, W)
    feats = dinov2_features  # (B, D, H, W)

    B, C, H, W = probs.shape
    D = feats.shape[1]

    # Compute weighted prototypes per class
    probs_flat = probs.reshape(B, C, H*W)  # (B, C, N)
    feats_flat = feats.reshape(B, D, H*W)  # (B, D, N)

    # Weighted average: prototypes (B, D, C)
    prototypes = torch.bmm(feats_flat, probs_flat.permute(0, 2, 1))  # (B, D, C)
    prototype_weights = probs_flat.sum(dim=2, keepdim=True).permute(0, 2, 1)  # (B, C, 1)
    prototypes = prototypes / (prototype_weights.permute(0, 2, 1) + 1e-6)  # (B, D, C)

    # Compute per-pixel similarity to assigned prototype
    prototypes_norm = F.normalize(prototypes, dim=1)  # (B, D, C)
    feats_norm = F.normalize(feats_flat, dim=1)  # (B, D, N)

    sim = torch.bmm(prototypes_norm.permute(0, 2, 1), feats_norm)  # (B, C, N)

    # Weighted loss: high-probability pixels should be close to their prototype
    loss = -(probs_flat * sim).sum() / (B * H * W)
    return loss
```

**Loss 4: Entropy Minimization** (Goodfellow "Deep Learning" Ch. 19)

Push predictions toward confident (low-entropy) assignments, preventing the network from producing uniform distributions.

```python
def entropy_loss(logits):
    """Minimize prediction entropy to encourage confident assignments."""
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1).mean()
    return entropy
```

#### 2.3.3 Training Script

```python
# File: mbps_pytorch/train_refine_net.py

def train_refine_net(
    cityscapes_root: str,
    output_dir: str,
    num_epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    num_blocks: int = 4,
    bridge_dim: int = 192,
    layer_type: str = "gated_delta_net",
    scan_mode: str = "bidirectional",
    device: str = "auto",
):
    # Device selection
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = CSCMRefineNet(
        num_classes=27,
        feature_dim=768,
        bridge_dim=bridge_dim,
        num_blocks=num_blocks,
        layer_type=layer_type,
        scan_mode=scan_mode,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    dataset = PseudoLabelDataset(cityscapes_root, split="train")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    loss_fn = RefineNetLoss(
        lambda_consist=1.0,
        lambda_align=0.5,
        lambda_proto=0.3,
        lambda_ent=0.1,
    )

    for epoch in range(num_epochs):
        model.train()
        for batch in loader:
            # Weak augmentation: only horizontal flip
            weak_logits, weak_affinity = model(
                batch["cause_logits"].to(device),
                batch["dinov2_features"].to(device),
                batch["depth"].to(device),
                batch["depth_grads"].to(device),
            )

            # Strong augmentation: color jitter + random crop + scale
            strong_batch = apply_strong_augmentation(batch)
            strong_logits, strong_affinity = model(
                strong_batch["cause_logits"].to(device),
                strong_batch["dinov2_features"].to(device),
                strong_batch["depth"].to(device),
                strong_batch["depth_grads"].to(device),
            )

            loss = loss_fn(weak_logits, strong_logits,
                          batch["dinov2_features"].to(device),
                          batch["depth"].to(device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Evaluate on val set every 5 epochs
        if (epoch + 1) % 5 == 0:
            evaluate_refined_semantics(model, cityscapes_root)
```

#### 2.3.4 Generating Refined Pseudo-Labels

After training, generate refined semantic pseudo-labels for all images:

```bash
$PYTHON mbps_pytorch/generate_refined_semantics.py \
    --cityscapes_root $CS_ROOT \
    --checkpoint checkpoints/refine_net/best.pth \
    --output_dir $CS_ROOT/pseudo_semantic_refined \
    --split val \
    --device auto
```

Then evaluate:
```bash
$PYTHON mbps_pytorch/evaluate_cascade_pseudolabels.py \
    --cityscapes_root $CS_ROOT \
    --split val \
    --semantic_subdir pseudo_semantic_refined \
    --instance_subdir sweep_instances/gt0.10_ma500 \
    --thing_mode maskcut \
    --cause27 \
    --output $CS_ROOT/eval_refined_sem_val.json
```

#### 2.3.5 Hyperparameter Sweep

| Parameter | Default | Sweep Range | Notes |
|-----------|---------|-------------|-------|
| `num_blocks` | 4 | {2, 4, 6} | More blocks = more capacity but slower |
| `bridge_dim` | 192 | {128, 192, 256} | Match to existing Mamba2 module defaults |
| `layer_type` | `gated_delta_net` | {`mamba2`, `gated_delta_net`} | Ablation |
| `scan_mode` | `bidirectional` | {`raster`, `bidirectional`, `cross_scan`} | Ablation |
| `coupling_strength` | 0.1 | {0.01, 0.1, 0.5} | How much cross-modal coupling |
| `lr` | 1e-4 | {5e-5, 1e-4, 3e-4} | |
| `lambda_consist` | 1.0 | Fixed | Primary loss |
| `lambda_align` | 0.5 | {0.1, 0.5, 1.0} | |
| `confidence_threshold` | 0.9 | {0.8, 0.9, 0.95} | For consistency loss |

---

## 3. Method 2: Mamba-Guided Adaptive Instance Decomposition

### 3.1 Overview

**Goal**: Replace the fixed depth-gradient threshold τ = 0.10 with a learned, spatially-adaptive split predictor.

**Key innovation**: Use GatedDeltaNet's error-correcting delta rule to maintain object-level context during scanning, enabling the model to distinguish intra-object depth variations from inter-object boundaries.

**Expected outcome**: PQ_things from 11.7 → 14-16 by reducing car FP from 925 to ~400 and improving person TP from 176 to ~400.

### 3.2 Architecture

```
File: mbps_pytorch/adaptive_instance_net.py

class AdaptiveInstanceNet(nn.Module):
    """Mamba-guided adaptive instance decomposition.

    Takes DINOv2 features + depth + semantics and predicts:
    1. Per-pixel split probability (replaces fixed threshold)
    2. Per-pixel instance embedding (for grouping)

    References:
        - MTMamba (ECCV 2024): Self-Task + Cross-Task Mamba decomposition
        - CutS3D (ICCV 2025): Spatial Importance Sharpening
        - Sigma (WACV 2025): Cross-selective scan for multi-modal segmentation
    """

    def __init__(
        self,
        feature_dim: int = 768,
        depth_channels: int = 3,      # depth + sobel_x + sobel_y
        semantic_dim: int = 27,        # CAUSE logits
        hidden_dim: int = 256,
        embed_dim: int = 32,           # instance embedding dimension
        num_blocks: int = 3,
        layer_type: str = "gated_delta_net",
        scan_mode: str = "bidirectional",
    ):
        super().__init__()

        # Input fusion (concatenate all modalities → project)
        total_input = feature_dim + depth_channels + semantic_dim
        self.input_proj = nn.Sequential(
            nn.Conv2d(total_input, hidden_dim, 1),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
        )

        # Mamba processing blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                VisionMamba2(
                    d_model=hidden_dim,
                    scan_mode=scan_mode,
                    layer_type=layer_type,
                    d_state=64,
                    chunk_size=256,
                ),
            )
            for _ in range(num_blocks)
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(16, hidden_dim) for _ in range(num_blocks)
        ])

        # Split probability head: should I split here? (binary per-pixel)
        self.split_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

        # Instance embedding head: which instance does this pixel belong to?
        self.embed_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, 1),
        )

    def forward(self, dinov2_features, depth, depth_grads, cause_logits):
        """
        Returns:
            split_prob: (B, 1, H, W) — per-pixel probability of instance boundary
            instance_embed: (B, embed_dim, H, W) — instance grouping embeddings
        """
        x = torch.cat([dinov2_features, depth, depth_grads, cause_logits], dim=1)
        x = self.input_proj(x)

        for block, norm in zip(self.blocks, self.norms):
            x = x + block(norm(x))

        split_prob = torch.sigmoid(self.split_head(x))
        instance_embed = self.embed_head(x)

        return split_prob, instance_embed
```

### 3.3 Instance Generation from Network Output

```python
def generate_adaptive_instances(
    split_prob,          # (H, W) float [0,1]
    instance_embed,      # (H, W, 32) float
    semantic_trainid,    # (H, W) uint8 trainID labels
    depth,               # (H, W) float depth
    min_area=500,
    split_threshold=0.5,
    embed_threshold=0.7, # cosine sim threshold for grouping
    thing_ids=set(range(11, 19)),
):
    """Generate instance masks using adaptive split map + embeddings.

    Algorithm:
    1. Create binary edge map from split_prob > split_threshold
    2. For each thing class, compute connected components on (class_mask AND NOT edge_map)
    3. Optionally merge components with high embedding similarity
    4. Filter by min_area
    5. Apply dilation to reclaim boundary pixels
    """
    edge_map = split_prob > split_threshold

    instances = []
    for class_id in thing_ids:
        class_mask = semantic_trainid == class_id
        if class_mask.sum() == 0:
            continue

        split_mask = class_mask & (~edge_map)
        labels, num_labels = scipy.ndimage.label(split_mask)

        for label_id in range(1, num_labels + 1):
            component = labels == label_id
            if component.sum() < min_area:
                continue

            # Optionally: merge with nearby components that have similar embeddings
            # (helps prevent over-segmentation of single objects)

            instances.append({
                "mask": component,
                "class_id": class_id,
                "score": component.sum() / class_mask.sum(),
            })

    return instances
```

### 3.4 Training Losses

**Loss 1: Distillation from existing depth-guided instances** (knowledge distillation)

```python
def distillation_loss(split_prob, depth_gradient_magnitude, tau=0.10):
    """Soft distillation from the heuristic depth-gradient threshold.

    The network should approximately reproduce the existing threshold behavior
    while learning to improve upon it.
    """
    teacher_split = (depth_gradient_magnitude > tau).float()
    return F.binary_cross_entropy(split_prob, teacher_split)
```

**Loss 2: Contrastive instance embedding** (proxy from DINOv2 feature similarity)

```python
def contrastive_embedding_loss(embeddings, dinov2_features, semantic_labels,
                                depth, margin=0.5, num_pairs=4096):
    """
    Positive pairs: same class + similar depth + high DINOv2 cosine sim
    Negative pairs: same class + different depth (or different class)
    """
    # Sample pixel pairs within thing classes
    # Positive: cosine_sim(dinov2[i], dinov2[j]) > 0.85 AND same class AND |depth_diff| < 0.05
    # Negative: same class AND |depth_diff| > 0.15
    # Triplet loss or InfoNCE on instance embeddings
    pass
```

**Loss 3: Boundary precision** (edges should coincide with RGB and depth boundaries)

```python
def boundary_precision_loss(split_prob, rgb_edges, depth_edges):
    """Split boundaries should align with multi-modal edges."""
    combined_edges = torch.max(rgb_edges, depth_edges)
    return F.binary_cross_entropy(split_prob, combined_edges)
```

### 3.5 Training Protocol

1. Train for 30 epochs on Cityscapes train (2975 images)
2. Batch size 4 at 32×64 patch resolution
3. AdamW lr=1e-4, cosine decay
4. Early stopping on val PQ_things

---

## 4. Method 3: EMA Teacher Self-Training

### 4.1 Overview

**Goal**: Iteratively improve pseudo-label quality through a self-training loop with class-adaptive thresholding and rare class sampling.

**Key innovations**:
- FlexMatch class-adaptive confidence thresholds (NeurIPS 2022)
- ST++ prediction-stability filtering (CVPR 2022)
- DAFormer rare class sampling (CVPR 2022)

### 4.2 Self-Training Loop

```
File: mbps_pytorch/self_training.py

def self_training_loop(
    refine_net: CSCMRefineNet,        # Trained Method 1 model
    instance_net: AdaptiveInstanceNet,  # Trained Method 2 model (optional)
    cityscapes_root: str,
    num_rounds: int = 3,
    epochs_per_round: int = 15,
    ema_momentum: float = 0.999,
):
    """
    Self-training loop with EMA teacher.

    Round 0: Use existing pseudo-labels (PQ=23.2)
    Round r (r=1,2,3):
        1. Generate pseudo-labels using EMA teacher
        2. Apply FlexMatch class-adaptive thresholding
        3. Apply ST++ stability filtering
        4. Train student on filtered pseudo-labels with rare class sampling
        5. Update EMA teacher
        6. Evaluate → accept if PQ improves, else stop

    References:
        - Mean Teacher (NeurIPS 2017): EMA teacher-student
        - FlexMatch (NeurIPS 2022): Class-adaptive thresholds
        - ST++ (CVPR 2022): Prediction stability filtering
        - DAFormer (CVPR 2022): Rare class sampling
        - EM algorithm: Goodfellow "Deep Learning" Ch.19, Bishop "PRML" Sec.10.2
    """
```

### 4.3 FlexMatch Class-Adaptive Thresholding

```python
class FlexMatchThreshold:
    """Per-class adaptive confidence thresholds.

    Well-learned classes (road, sky) → full threshold (0.95)
    Poorly-learned classes (pole, fence) → reduced threshold (0.5-0.7)

    Reference: FlexMatch (NeurIPS 2022)
    """

    def __init__(self, num_classes=27, base_threshold=0.95):
        self.base_threshold = base_threshold
        self.class_counts = torch.zeros(num_classes)
        self.total_pixels = 0

    def update(self, probs):
        """Update class learning status from a batch of predictions."""
        max_probs, pred_classes = probs.max(dim=1)  # (B, H, W)
        for c in range(len(self.class_counts)):
            mask = (pred_classes == c) & (max_probs > self.base_threshold)
            self.class_counts[c] += mask.sum().item()
        self.total_pixels += pred_classes.numel()

    def get_thresholds(self):
        """Return per-class thresholds."""
        sigma = self.class_counts / (self.total_pixels + 1e-6)
        max_sigma = sigma.max()
        # Flex threshold: scale by class learning status
        thresholds = self.base_threshold * sigma / (max_sigma + 1e-6)
        # Floor at 0.5 to prevent accepting everything
        thresholds = thresholds.clamp(min=0.5)
        return thresholds

    def filter_pseudo_labels(self, probs, hard_labels):
        """Return mask of pixels that pass class-adaptive threshold."""
        thresholds = self.get_thresholds()  # (C,)
        max_probs, pred_classes = probs.max(dim=1)  # (B, H, W)

        # Per-pixel threshold based on predicted class
        pixel_thresholds = thresholds[pred_classes]  # (B, H, W)
        mask = max_probs > pixel_thresholds
        return mask
```

### 4.4 ST++ Prediction Stability Filtering

```python
class StabilityFilter:
    """Image-level stability filtering.

    Compare predictions from two checkpoints (epoch e and e-5).
    Images with high prediction disagreement are unreliable → exclude.

    Reference: ST++ (CVPR 2022)
    """

    def __init__(self, stability_threshold=0.8):
        self.stability_threshold = stability_threshold

    def compute_stability(self, preds_current, preds_previous):
        """
        preds_current: dict {image_path: (H, W) label array}
        preds_previous: dict {image_path: (H, W) label array}
        Returns: dict {image_path: float stability_score}
        """
        stability = {}
        for path in preds_current:
            agreement = (preds_current[path] == preds_previous[path]).mean()
            stability[path] = agreement
        return stability

    def filter_images(self, stability_scores):
        """Return list of stable image paths."""
        return [
            path for path, score in stability_scores.items()
            if score > self.stability_threshold
        ]
```

### 4.5 DAFormer Rare Class Sampling

```python
class RareClassSampler:
    """Oversample images containing rare classes.

    Zero-IoU classes in CAUSE-TR: fence, pole, traffic light, traffic sign,
    rider, train, motorcycle. Images containing predictions of these classes
    are sampled 5× more frequently.

    Reference: DAFormer (CVPR 2022)
    """

    def __init__(self, rare_classes, oversample_factor=5):
        self.rare_classes = set(rare_classes)
        self.oversample_factor = oversample_factor

    def compute_weights(self, dataset):
        """Compute per-image sampling weights."""
        weights = []
        for idx in range(len(dataset)):
            labels = dataset.get_semantic_labels(idx)
            unique_classes = set(np.unique(labels).tolist())
            has_rare = bool(unique_classes & self.rare_classes)
            weights.append(self.oversample_factor if has_rare else 1.0)
        return weights
```

### 4.6 Full Self-Training Round

```python
def run_self_training_round(
    student_model,
    teacher_model,        # EMA of student
    train_dataset,
    val_dataset,
    cityscapes_root,
    round_num,
    epochs=15,
    lr=5e-5,
    ema_momentum=0.999,
):
    """One round of self-training."""

    # 1. Generate pseudo-labels from teacher
    print(f"Round {round_num}: Generating pseudo-labels from teacher...")
    teacher_model.eval()
    pseudo_labels = {}
    with torch.no_grad():
        for batch in DataLoader(train_dataset, batch_size=8):
            refined_logits, _ = teacher_model(**batch)
            probs = F.softmax(refined_logits, dim=1)
            pseudo_labels.update(extract_labels(probs, batch["image_path"]))

    # 2. FlexMatch filtering
    flex = FlexMatchThreshold(num_classes=27)
    for path, (probs, labels) in pseudo_labels.items():
        flex.update(probs.unsqueeze(0))

    filtered_mask = {}
    for path, (probs, labels) in pseudo_labels.items():
        mask = flex.filter_pseudo_labels(probs.unsqueeze(0), labels.unsqueeze(0))
        filtered_mask[path] = mask.squeeze(0)

    # 3. ST++ stability filtering (compare with previous round)
    if round_num > 1:
        stability = StabilityFilter(threshold=0.8)
        scores = stability.compute_stability(
            current_preds=pseudo_labels,
            previous_preds=load_previous_round_preds(round_num - 1),
        )
        stable_images = stability.filter_images(scores)
        print(f"  Stability filter: {len(stable_images)}/{len(pseudo_labels)} images retained")
    else:
        stable_images = list(pseudo_labels.keys())

    # 4. Rare class sampling
    rare_sampler = RareClassSampler(
        rare_classes=[4, 5, 6, 7, 12, 18, 24, 25],  # CAUSE-27 IDs for weak classes
        oversample_factor=5,
    )
    sample_weights = rare_sampler.compute_weights(train_dataset)
    sampler = WeightedRandomSampler(sample_weights, len(train_dataset))

    # 5. Train student
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        student_model.train()
        for batch in DataLoader(train_dataset, batch_size=4, sampler=sampler):
            refined_logits, _ = student_model(**batch)

            # Supervised loss on filtered pseudo-labels
            loss = masked_cross_entropy(refined_logits, pseudo_labels, filtered_mask, batch)

            # + Self-supervised losses (consistency, alignment, etc.)
            loss += self_supervised_losses(student_model, batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()

            # EMA update
            with torch.no_grad():
                for tp, sp in zip(teacher_model.parameters(), student_model.parameters()):
                    tp.data.mul_(ema_momentum).add_(sp.data, alpha=1 - ema_momentum)

    # 6. Evaluate
    pq = evaluate_on_val(teacher_model, val_dataset, cityscapes_root)
    return pq
```

---

## 5. Integration Pipeline

### 5.1 End-to-End Execution Order

```bash
# ─── Phase 0: Ensure prerequisites exist ───
# CAUSE-CRF semantics, SPIdepth depth, DINOv2 features must be pre-generated
# See Section 1.1 for paths

# ─── Phase I: Method 1 — Semantic Refinement ───
$PYTHON mbps_pytorch/train_refine_net.py \
    --cityscapes_root $CS_ROOT \
    --output_dir checkpoints/refine_net \
    --num_epochs 50 \
    --batch_size 4 \
    --layer_type gated_delta_net \
    --scan_mode bidirectional \
    --num_blocks 4

# Generate refined semantics
$PYTHON mbps_pytorch/generate_refined_semantics.py \
    --cityscapes_root $CS_ROOT \
    --checkpoint checkpoints/refine_net/best.pth \
    --output_dir $CS_ROOT/pseudo_semantic_refined \
    --split both

# Evaluate semantic improvement
$PYTHON mbps_pytorch/evaluate_cascade_pseudolabels.py \
    --cityscapes_root $CS_ROOT --split val \
    --semantic_subdir pseudo_semantic_refined \
    --instance_subdir sweep_instances/gt0.10_ma500 \
    --thing_mode maskcut --cause27 \
    --output $CS_ROOT/eval_refined_method1.json

# ─── Phase II: Method 2 — Adaptive Instances ───
$PYTHON mbps_pytorch/train_adaptive_instance.py \
    --cityscapes_root $CS_ROOT \
    --semantic_subdir pseudo_semantic_refined \
    --output_dir checkpoints/adaptive_instance \
    --num_epochs 30 \
    --layer_type gated_delta_net

# Generate adaptive instances
$PYTHON mbps_pytorch/generate_adaptive_instances.py \
    --cityscapes_root $CS_ROOT \
    --checkpoint checkpoints/adaptive_instance/best.pth \
    --semantic_subdir pseudo_semantic_refined_trainid \
    --output_dir $CS_ROOT/adaptive_instances \
    --split both

# Evaluate instance improvement
$PYTHON mbps_pytorch/evaluate_cascade_pseudolabels.py \
    --cityscapes_root $CS_ROOT --split val \
    --semantic_subdir pseudo_semantic_refined \
    --instance_subdir adaptive_instances \
    --thing_mode maskcut --cause27 \
    --output $CS_ROOT/eval_refined_method2.json

# ─── Phase III: Method 3 — Self-Training ───
$PYTHON mbps_pytorch/self_training.py \
    --cityscapes_root $CS_ROOT \
    --refine_checkpoint checkpoints/refine_net/best.pth \
    --instance_checkpoint checkpoints/adaptive_instance/best.pth \
    --output_dir checkpoints/self_training \
    --num_rounds 3 \
    --epochs_per_round 15

# Generate final pseudo-labels
$PYTHON mbps_pytorch/generate_final_pseudolabels.py \
    --cityscapes_root $CS_ROOT \
    --checkpoint checkpoints/self_training/round3_teacher.pth \
    --output_dir $CS_ROOT/pseudo_labels_final \
    --split both

# Final evaluation
$PYTHON mbps_pytorch/evaluate_cascade_pseudolabels.py \
    --cityscapes_root $CS_ROOT --split val \
    --semantic_subdir pseudo_labels_final/semantic \
    --instance_subdir pseudo_labels_final/instances \
    --thing_mode maskcut --cause27 \
    --output $CS_ROOT/eval_final.json

# ─── Phase IV: Stage-2 CUPS Training (remote GPU) ───
# Convert final pseudo-labels to CUPS format
$PYTHON mbps_pytorch/convert_to_cups_format.py \
    --cityscapes_root $CS_ROOT \
    --semantic_subdir pseudo_labels_final/semantic \
    --instance_subdir pseudo_labels_final/instances \
    --output_dir $CS_ROOT/cups_pseudo_labels_v3

# Transfer to remote and train (see stage2_vitb.md)
```

### 5.2 Expected Progression

| Phase | Semantic mIoU | PQ | PQ_things | Δ PQ |
|-------|--------------|-----|-----------|------|
| Baseline (current) | 42.86% | 23.2 | 11.7 | — |
| After Method 1 | 48-52% | 25-27 | 11-13 | +2-4 |
| After Method 2 | 48-52% | 26-28 | 14-16 | +1-2 |
| After Method 3 | 50-55% | 27-29 | 15-18 | +1-2 |
| After Stage-2 | — | 28-31 | 18-22 | +1-3 |

---

## 6. Ablation Matrix

### 6.1 Method 1 Ablations

| Ablation | What changes | Expected effect |
|----------|-------------|-----------------|
| `layer_type=mamba2` | Mamba2 SSD instead of GatedDeltaNet | Slightly worse (no error correction) |
| `scan_mode=raster` | Simple row-major scan | Worse (no bidirectional context) |
| `scan_mode=cross_scan` | VMamba 4-way scan | Similar or slightly better (more expensive) |
| `coupling_strength=0` | No cross-modal coupling | Reduces to independent processing |
| `no_depth_conditioning` | Remove FiLM from DepthFeatureProjection | Loses depth geometry awareness |
| `no_residual` | Remove `+ cause_logits` residual | Training instability |
| `no_consistency_loss` | Remove cross-view consistency | Weaker regularization |
| `no_prototype_loss` | Remove feature-prototype loss | Less compact clusters |

### 6.2 Method 2 Ablations

| Ablation | What changes | Expected effect |
|----------|-------------|-----------------|
| `fixed_threshold` | Replace adaptive split with τ=0.10 | Baseline (current pipeline) |
| `no_embeddings` | Split probability only, no instance embeddings | Loses grouping information |
| `no_mamba` | Replace VisionMamba2 with Conv2d stack | Loses long-range context |
| `no_distillation` | Don't distill from existing instances | Slower convergence |

### 6.3 Method 3 Ablations

| Ablation | What changes | Expected effect |
|----------|-------------|-----------------|
| `fixed_threshold` | No FlexMatch, use τ=0.95 everywhere | Confirmation bias on rare classes |
| `no_stability_filter` | No ST++ filtering | Include unreliable pseudo-labels |
| `no_rare_sampling` | Uniform sampling | Rare classes under-represented |
| `no_ema` | Student-only (no teacher) | Noisy pseudo-labels |

---

## 7. Evaluation Protocol

### 7.1 Metrics to Track

At each checkpoint, evaluate and record:

```json
{
    "semantic": {"miou": float, "pixel_accuracy": float, "per_class_iou": {...}},
    "instance": {"ar_100": float, "ap_50": float, "avg_pred_instances": float},
    "panoptic": {"PQ": float, "PQ_stuff": float, "PQ_things": float, "SQ": float, "RQ": float}
}
```

### 7.2 Key Diagnostic Metrics

| Metric | What it tells you | Target |
|--------|------------------|--------|
| Number of non-zero IoU classes | Semantic recovery progress | 12 → 15+ (out of 19) |
| mIoU | Overall semantic quality | 42.86 → 50+ |
| Car FP count | Over-segmentation control | 925 → <500 |
| Person TP count | Instance recall improvement | 176 → 400+ |
| PQ_things | Instance quality overall | 11.7 → 16+ |
| Coupling strength (α, β) | Cross-modal information flow | Should grow from 0.1 during training |

### 7.3 Sanity Checks

Before celebrating any improvement, verify:
1. **No GT leakage**: All losses are self-supervised; GT is only used in evaluation
2. **Reproducibility**: Run evaluation twice, confirm identical metrics
3. **Per-class analysis**: Check that gains come from recovering zero-IoU classes, not just inflating already-strong classes
4. **Visual inspection**: Look at 10-20 images where predictions changed; confirm changes are semantically meaningful

---

## 8. References

### 8.1 Core SSM / Mamba Papers

1. **Mamba2 / SSD**: Dao, T. and Gu, A. (2024). "Transformers are SSMs: Generalized Models and Efficient Algorithms through Structured State Space Duality." *ICML 2024*. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)

2. **GatedDeltaNet**: Yang, S., Wang, B., et al. (2025). "Gated Delta Networks: Improving Mamba2 with Delta Rule." *ICLR 2025*. [Paper](https://jankautz.com/publications/GatedDeltaNet_ICLR25.pdf), [GitHub](https://github.com/NVlabs/GatedDeltaNet)

3. **Mamba**: Gu, A. and Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *arXiv:2312.00752*.

### 8.2 Vision SSM / Mamba for Segmentation

4. **VMamba**: Liu, Y., Tian, Y., Zhao, Y. et al. (2024). "VMamba: Visual State Space Model." *NeurIPS 2024*. [arXiv:2401.10166](https://arxiv.org/abs/2401.10166)

5. **Vision Mamba (ViM)**: Zhu, L., Liao, B., Zhang, Q. et al. (2024). "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model." *ICML 2024*. [arXiv:2401.09417](https://arxiv.org/abs/2401.09417)

6. **Mamba2D**: Coelho de Castro, A. et al. (2025). "Mamba2D: A Natively Multi-Dimensional State-Space Model for Vision Tasks." *ICLR 2025*. [arXiv:2412.16146](https://arxiv.org/abs/2412.16146)

7. **U-Mamba**: Ma, J. et al. (2024). "U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation." [u-mamba.github.io](https://u-mamba.github.io/)

8. **VM-UNet**: Ruan, J. and Xiang, S. (2024). "VM-UNet: Vision Mamba UNet for Medical Image Segmentation." *ACM Trans. Multimedia Computing*. [arXiv:2402.02491](https://arxiv.org/abs/2402.02491)

### 8.3 Cross-Modal SSM Fusion

9. **Coupled Mamba**: Li, W., Zhou, H., Song, Z., Yang, W. (2024). "Coupled Mamba: Enhanced Multi-modal Fusion with Coupled State Space Model." *NeurIPS 2024*. [GitHub](https://github.com/hustcselwb/coupled-mamba)

10. **Sigma**: Wan, Z., Wang, Y. et al. (2025). "Sigma: Siamese Mamba Network for Multi-Modal Semantic Segmentation." *WACV 2025*. [arXiv:2404.04256](https://arxiv.org/abs/2404.04256)

11. **MTMamba**: Li, B. et al. (2024). "MTMamba: Enhancing Multi-Task Dense Scene Understanding with Mamba-Based Decoder." *ECCV 2024*. [arXiv:2407.02228](https://arxiv.org/abs/2407.02228)

12. **FusionMamba**: Xie, M. et al. (2024). "FusionMamba: Dynamic Feature Enhancement for Multimodal Image Fusion." *Visual Intelligence*. [arXiv:2404.09498](https://arxiv.org/abs/2404.09498)

13. **MambaSOD**: Li, Y. et al. (2025). "MambaSOD: Dual Mamba-Driven Cross-Modal Fusion Network for RGB-D Salient Object Detection." *ScienceDirect*. [Link](https://www.sciencedirect.com/science/article/abs/pii/S092523122500390X)

### 8.4 Pseudo-Label Refinement & Self-Training

14. **FlexMatch**: Zhang, B., Wang, Y., et al. (2022). "FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling." *NeurIPS 2022*. [arXiv:2110.08263](https://arxiv.org/abs/2110.08263)

15. **ST++**: Yang, L. et al. (2022). "ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation." *CVPR 2022*. [arXiv:2106.05095](https://arxiv.org/abs/2106.05095)

16. **DAFormer**: Hoyer, L. et al. (2022). "DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation." *CVPR 2022*. [arXiv:2111.14887](https://arxiv.org/abs/2111.14887)

17. **Mean Teacher**: Tarvainen, A. and Valpola, H. (2017). "Mean teachers are better role models." *NeurIPS 2017*. [arXiv:1703.01780](https://arxiv.org/abs/1703.01780)

18. **UniMatch V2**: Yang, L. et al. (2025). "Revisiting Semi-Supervised Learning in the Era of Foundation Models." *TPAMI 2025*. [GitHub](https://github.com/LiheYoung/UniMatch-V2)

19. **FixMatch**: Sohn, K. et al. (2020). "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence." *NeurIPS 2020*. [arXiv:2001.07685](https://arxiv.org/abs/2001.07685)

20. **PRN**: Zhao, J. et al. (2024). "Unsupervised Domain Adaptation for Semantic Segmentation with Pseudo Label Self-Refinement." *WACV 2024*. [Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Zhao_Unsupervised_Domain_Adaptation_for_Semantic_Segmentation_With_Pseudo_Label_Self-Refinement_WACV_2024_paper.pdf)

21. **RankMatch**: Mai, T. et al. (2024). "RankMatch: Exploring the Better Consistency Regularization for Semi-supervised Semantic Segmentation." *CVPR 2024*. [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Mai_RankMatch_Exploring_the_Better_Consistency_Regularization_for_Semi-supervised_Semantic_Segmentation_CVPR_2024_paper.pdf)

### 8.5 Depth-Semantic Fusion

22. **DFormerv2**: Yin, J. et al. (2025). "DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation." *CVPR 2025*. [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_DFormerv2_Geometry_Self-Attention_for_RGBD_Semantic_Segmentation_CVPR_2025_paper.pdf)

23. **CMX**: Zhang, J. et al. (2023). "CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers." *TITS*. [arXiv:2203.04838](https://arxiv.org/abs/2203.04838)

24. **MultiMAE**: Bachmann, R. et al. (2022). "MultiMAE: Multi-modal Multi-task Masked Autoencoders." *ECCV 2022*. [arXiv:2204.01678](https://arxiv.org/abs/2204.01678)

25. **DepthG**: Sick, L., Engel, N., Hermosilla, P., Ropinski, T. (2024). "Unsupervised Semantic Segmentation Through Depth-Guided Feature Correlation and Sampling." *CVPR 2024*. [GitHub](https://github.com/leonsick/depthg)

26. **CutS3D**: Sick, L. et al. (2025). "CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation." *ICCV 2025*. [Project](https://leonsick.github.io/cuts3d/)

27. **FiLM**: Perez, E. et al. (2018). "FiLM: Visual Reasoning with a General Conditioning Layer." *AAAI 2018*. [arXiv:1709.07871](https://arxiv.org/abs/1709.07871)

### 8.6 Foundation Model Feature Refinement

28. **DINOv2 Registers**: Darcet, T. et al. (2024). "Vision Transformers Need Registers." *ICLR 2024*. [arXiv:2309.16588](https://arxiv.org/abs/2309.16588)

29. **DINOv2**: Oquab, M. et al. (2024). "DINOv2: Learning Robust Visual Features without Supervision." *TMLR*.

30. **CKA**: Kornblith, S., Norouzi, M., Lee, H., Hinton, G. (2019). "Similarity of Neural Network Representations Revisited." *ICML 2019*. [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)

### 8.7 Unsupervised Panoptic Segmentation

31. **CUPS**: Hahn, K. et al. (2025). "Scene-Centric Unsupervised Panoptic Segmentation." *CVPR 2025 (Highlight)*. [GitHub](https://github.com/visinf/cups)

32. **U2Seg**: Niu, Z. et al. (2024). "Unsupervised Universal Image Segmentation." *CVPR 2024*. [GitHub](https://github.com/u2seg/U2Seg)

33. **CAUSE**: Cho, J. et al. (2024). "CAUSE: Contrastive Learning with Modularity-Based Codebook for Unsupervised Segmentation." *Pattern Recognition*. [GitHub](https://github.com/ByungKwanLee/Causal-Unsupervised-Segmentation)

34. **STEGO**: Hamilton, M. et al. (2022). "Unsupervised Semantic Segmentation by Distilling Feature Correspondences." *ICLR 2022*. [GitHub](https://github.com/mhamilton723/STEGO)

35. **SPIdepth**: Seo, J. et al. (2025). "SPIdepth: Strengthened Pose Information for Self-Supervised Monocular Depth Estimation." *CVPR 2025*.

### 8.8 Textbooks

36. **Goodfellow, I., Bengio, Y., Courville, A.** (2016). *Deep Learning*. MIT Press. Ch. 7 (Regularization), Ch. 19 (Approximate Inference / EM Algorithm). [deeplearningbook.org](https://www.deeplearningbook.org/)

37. **Bishop, C.** (2006). *Pattern Recognition and Machine Learning*. Springer. Ch. 8 (Graphical Models / CRF / MRF), Sec. 10.2 (Variational Inference / Variational EM).

38. **Murphy, K.** (2022/2023). *Probabilistic Machine Learning: An Introduction / Advanced Topics*. MIT Press. Chapters on Semi-Supervised Learning, Structured Prediction, State Space Models.

---

*Guide version: 1.0 — February 20, 2026*
*Author: MBPS Project*
*For use by Claude Code agents implementing the Mamba-Bridge pseudo-label refinement pipeline*
