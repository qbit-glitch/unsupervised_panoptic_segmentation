# HiRes RefineNet 128x256 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a high-resolution (128x256) semantic refinement network with three upsampling strategies and five architecture variants, then run systematic ablation studies.

**Architecture:** Upsample DINOv2 32x64 features to 128x256 via configurable upsampling module (bilinear/transposed-conv/pixel-shuffle), then refine with configurable blocks (Conv2d/WindowedAttention/VisionMamba2/SpatialMamba/MambaOut). Reuses existing training pipeline with resolution parameterization.

**Tech Stack:** PyTorch, DINOv2 ViT-B/14 (frozen features), SPIdepth depth, MPS (Apple M4 Pro 48GB)

**Key context:**
- Existing model: `mbps_pytorch/refine_net.py` — CSCMRefineNet with CoupledConvBlock and CoupledMambaBlock
- Existing training: `mbps_pytorch/train_refine_net.py` — full pipeline (dataset, losses, eval, CLI)
- Existing Mamba2: `mbps_pytorch/mamba2/vision.py` — VisionMamba2 with raster/bidirectional/cross_scan modes
- Current best (32x64): PQ=26.52, PQ_stuff=33.38, PQ_things=17.10 (Run D)
- Input pseudo-labels: PQ=26.74, PQ_stuff=32.08, PQ_things=19.41
- Pseudo-labels at: `/Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_mapped_k80/`
- Python env: `/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python`
- Design doc: `docs/plans/2026-03-06-hires-refinenet-design.md`
- Report (blank): `reports/hires_refinenet_128x256.md`

---

### Task 1: Add upsampling modules to refine_net.py

**Files:**
- Modify: `mbps_pytorch/refine_net.py`

**Step 1: Add the three upsampling module classes**

Add these classes before `CoupledConvBlock` (around line 120) in `mbps_pytorch/refine_net.py`:

```python
class UpsampleBilinear(nn.Module):
    """Strategy A: Parameter-free bilinear 4x upsampling."""

    def __init__(self, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale_factor,
                             mode='bilinear', align_corners=False)


class UpsampleTransposedConv(nn.Module):
    """Strategy B: 2-stage learnable transposed convolution (2x + 2x = 4x)."""

    def __init__(self, channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.ConvTranspose2d(channels, channels, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class UpsamplePixelShuffle(nn.Module):
    """Strategy C: Sub-pixel convolution (PixelShuffle) 4x upsampling."""

    def __init__(self, channels: int, scale_factor: int = 4):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * scale_factor ** 2, 1, bias=False),
            nn.PixelShuffle(scale_factor),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)
```

**Step 2: Add factory function for upsampling**

```python
def _build_upsample(strategy: str, channels: int) -> nn.Module:
    """Factory: build upsampling module by strategy name."""
    if strategy == "bilinear":
        return UpsampleBilinear(scale_factor=4)
    elif strategy == "transposed_conv":
        return UpsampleTransposedConv(channels)
    elif strategy == "pixel_shuffle":
        return UpsamplePixelShuffle(channels)
    elif strategy == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown upsample strategy: {strategy}")
```

**Step 3: Verify syntax**

```bash
cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python -c "
from mbps_pytorch.refine_net import UpsampleBilinear, UpsampleTransposedConv, UpsamplePixelShuffle, _build_upsample
import torch
x = torch.randn(2, 192, 32, 64)
for s in ['bilinear', 'transposed_conv', 'pixel_shuffle']:
    m = _build_upsample(s, 192)
    print(f'{s}: {x.shape} -> {m(x).shape}, params={sum(p.numel() for p in m.parameters()):,}')
"
```

Expected output:
```
bilinear: torch.Size([2, 192, 32, 64]) -> torch.Size([2, 192, 128, 256]), params=0
transposed_conv: torch.Size([2, 192, 32, 64]) -> torch.Size([2, 192, 128, 256]), params=~590K
pixel_shuffle: torch.Size([2, 192, 32, 64]) -> torch.Size([2, 192, 128, 256]), params=~590K
```

---

### Task 2: Add WindowedAttentionBlock and MambaOutBlock to refine_net.py

**Files:**
- Modify: `mbps_pytorch/refine_net.py`

**Step 1: Add WindowedAttentionBlock**

Add after `CoupledMambaBlock` class:

```python
class WindowedAttentionBlock(nn.Module):
    """Windowed self-attention block with cross-modal gating (Swin-style).

    Partitions spatial dims into non-overlapping windows, applies MHSA within
    each window. Alternating blocks use shifted windows for cross-window
    communication. Relative position bias encodes intra-window spatial structure.
    """

    def __init__(
        self,
        d_model: int = 192,
        window_size: int = 8,
        num_heads: int = 4,
        coupling_strength: float = 0.1,
        shift: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift = shift
        self.head_dim = d_model // num_heads

        # Pre-norm
        self.norm_sem = nn.LayerNorm(d_model)
        self.norm_depth = nn.LayerNorm(d_model)

        # Cross-modal gating
        self.cross_d2s = nn.Linear(d_model, d_model)
        self.cross_s2d = nn.Linear(d_model, d_model)
        self.alpha = nn.Parameter(torch.tensor(coupling_strength))
        self.beta = nn.Parameter(torch.tensor(coupling_strength))

        # Self-attention (sem stream)
        self.qkv_sem = nn.Linear(d_model, 3 * d_model)
        self.proj_sem = nn.Linear(d_model, d_model)

        # Self-attention (depth stream)
        self.qkv_depth = nn.Linear(d_model, 3 * d_model)
        self.proj_depth = nn.Linear(d_model, d_model)

        # Relative position bias
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(num_heads, (2 * window_size - 1) * (2 * window_size - 1))
        )
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        # Compute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, ws, ws)
        coords_flat = coords.reshape(2, -1)  # (2, ws*ws)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, ws*ws, ws*ws)
        rel = rel.permute(1, 2, 0).contiguous()  # (ws*ws, ws*ws, 2)
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        rel_idx = rel.sum(-1)  # (ws*ws, ws*ws)
        self.register_buffer("rel_pos_index", rel_idx)

        # FFN
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

    def _window_partition(self, x, H, W):
        """(B, H*W, C) -> (B*nW, ws*ws, C)"""
        B, N, C = x.shape
        ws = self.window_size
        x = x.view(B, H, W, C)
        # Pad if needed
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
        return x, Hp, Wp

    def _window_unpartition(self, x, B, Hp, Wp, H, W):
        """(B*nW, ws*ws, C) -> (B, H*W, C)"""
        ws = self.window_size
        C = x.shape[-1]
        nH, nW = Hp // ws, Wp // ws
        x = x.view(B, nH, nW, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        if Hp > H or Wp > W:
            x = x[:, :H, :W, :]
        return x.reshape(B, H * W, C)

    def _windowed_attention(self, x, qkv_proj, out_proj, H, W):
        """Apply windowed MHSA."""
        B_orig = x.shape[0]
        ws = self.window_size

        # Shifted window
        if self.shift:
            shift = ws // 2
            x_2d = x.view(B_orig, H, W, -1)
            x_2d = torch.roll(x_2d, shifts=(-shift, -shift), dims=(1, 2))
            x = x_2d.reshape(B_orig, H * W, -1)

        # Partition into windows
        x_win, Hp, Wp = self._window_partition(x, H, W)  # (B*nW, ws*ws, C)
        BnW, L, C = x_win.shape

        # QKV
        qkv = qkv_proj(x_win).reshape(BnW, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, BnW, heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (BnW, heads, L, L)

        # Add relative position bias
        bias = self.rel_pos_bias[:, self.rel_pos_index.view(-1)].view(
            self.num_heads, ws * ws, ws * ws)
        attn = attn + bias.unsqueeze(0)

        # Attention mask for shifted windows (simplified: no mask for non-shifted)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(BnW, L, C)
        out = out_proj(out)

        # Unpartition
        out = self._window_unpartition(out, B_orig, Hp, Wp, H, W)

        # Reverse shift
        if self.shift:
            out_2d = out.view(B_orig, H, W, -1)
            out_2d = torch.roll(out_2d, shifts=(shift, shift), dims=(1, 2))
            out = out_2d.reshape(B_orig, H * W, -1)

        return out

    def forward(self, sem, depth_feat):
        """
        Args:
            sem: (B, D, H, W) semantic stream
            depth_feat: (B, D, H, W) depth-feature stream
        Returns:
            (refined_sem, refined_depth_feat) — same shapes
        """
        B, D, H, W = sem.shape

        # Flatten to (B, H*W, D)
        sem_flat = sem.permute(0, 2, 3, 1).reshape(B, H * W, D)
        depth_flat = depth_feat.permute(0, 2, 3, 1).reshape(B, H * W, D)

        # Pre-norm + cross-modal gating
        sem_n = self.norm_sem(sem_flat)
        depth_n = self.norm_depth(depth_flat)
        sem_input = sem_n + self.alpha * torch.sigmoid(self.cross_d2s(depth_n))
        depth_input = depth_n + self.beta * torch.sigmoid(self.cross_s2d(sem_n))

        # Windowed attention
        sem_attn = self._windowed_attention(sem_input, self.qkv_sem, self.proj_sem, H, W)
        depth_attn = self._windowed_attention(depth_input, self.qkv_depth, self.proj_depth, H, W)

        # Residual + FFN
        sem_out = sem_flat + sem_attn
        sem_out = sem_out + self.ffn_sem(sem_out)
        depth_out = depth_flat + depth_attn
        depth_out = depth_out + self.ffn_depth(depth_out)

        # Reshape back
        sem_out = sem_out.reshape(B, H, W, D).permute(0, 3, 1, 2)
        depth_out = depth_out.reshape(B, H, W, D).permute(0, 3, 1, 2)
        return sem_out, depth_out


class MambaOutBlock(nn.Module):
    """MambaOut block: Gated CNN without SSM (Yu et al., 2024).

    Keeps Mamba's gated convolution structure but replaces the SSM with a
    larger depthwise conv. Tests whether SSM is necessary for vision.
    Uses cross-modal gating consistent with other block types.
    """

    def __init__(
        self,
        d_model: int = 192,
        expand: int = 2,
        kernel_size: int = 7,
        coupling_strength: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        inner_dim = d_model * expand

        # Pre-norm
        self.norm_sem = nn.GroupNorm(1, d_model)
        self.norm_depth = nn.GroupNorm(1, d_model)

        # Cross-modal gating
        self.cross_d2s = nn.Conv2d(d_model, d_model, 1)
        self.cross_s2d = nn.Conv2d(d_model, d_model, 1)
        self.alpha = nn.Parameter(torch.tensor(coupling_strength))
        self.beta = nn.Parameter(torch.tensor(coupling_strength))

        # Gated conv (sem)
        self.sem_gate_proj = nn.Conv2d(d_model, inner_dim, 1, bias=False)
        self.sem_value_proj = nn.Conv2d(d_model, inner_dim, 1, bias=False)
        self.sem_dw_conv = nn.Conv2d(
            inner_dim, inner_dim, kernel_size,
            padding=kernel_size // 2, groups=inner_dim, bias=False)
        self.sem_out_proj = nn.Conv2d(inner_dim, d_model, 1, bias=False)
        self.sem_gate_norm = nn.GroupNorm(1, inner_dim)

        # Gated conv (depth)
        self.depth_gate_proj = nn.Conv2d(d_model, inner_dim, 1, bias=False)
        self.depth_value_proj = nn.Conv2d(d_model, inner_dim, 1, bias=False)
        self.depth_dw_conv = nn.Conv2d(
            inner_dim, inner_dim, kernel_size,
            padding=kernel_size // 2, groups=inner_dim, bias=False)
        self.depth_out_proj = nn.Conv2d(inner_dim, d_model, 1, bias=False)
        self.depth_gate_norm = nn.GroupNorm(1, inner_dim)

    def _gated_conv(self, x, gate_proj, value_proj, dw_conv, gate_norm, out_proj):
        gate = F.silu(gate_norm(dw_conv(gate_proj(x))))
        value = value_proj(x)
        return out_proj(gate * value)

    def forward(self, sem, depth_feat):
        """
        Args:
            sem: (B, D, H, W) semantic stream
            depth_feat: (B, D, H, W) depth-feature stream
        Returns:
            (refined_sem, refined_depth_feat)
        """
        sem_n = self.norm_sem(sem)
        depth_n = self.norm_depth(depth_feat)

        # Cross-modal gating
        sem_input = sem_n + self.alpha * torch.sigmoid(self.cross_d2s(depth_n))
        depth_input = depth_n + self.beta * torch.sigmoid(self.cross_s2d(sem_n))

        # Gated conv + residual
        sem_out = sem + self._gated_conv(
            sem_input, self.sem_gate_proj, self.sem_value_proj,
            self.sem_dw_conv, self.sem_gate_norm, self.sem_out_proj)
        depth_out = depth_feat + self._gated_conv(
            depth_input, self.depth_gate_proj, self.depth_value_proj,
            self.depth_dw_conv, self.depth_gate_norm, self.depth_out_proj)

        return sem_out, depth_out
```

**Step 2: Verify new blocks compile and run**

```bash
cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python -c "
import torch
from mbps_pytorch.refine_net import WindowedAttentionBlock, MambaOutBlock

x_sem = torch.randn(2, 192, 128, 256)
x_dep = torch.randn(2, 192, 128, 256)

# Test windowed attention (non-shifted and shifted)
attn = WindowedAttentionBlock(192, window_size=8, num_heads=4, shift=False)
out_s, out_d = attn(x_sem, x_dep)
print(f'WindowedAttn: in={x_sem.shape}, out={out_s.shape}, params={sum(p.numel() for p in attn.parameters()):,}')

attn_shift = WindowedAttentionBlock(192, window_size=8, num_heads=4, shift=True)
out_s2, out_d2 = attn_shift(x_sem, x_dep)
print(f'WindowedAttn(shifted): out={out_s2.shape}')

# Test MambaOut
mout = MambaOutBlock(192, expand=2, kernel_size=7)
out_s3, out_d3 = mout(x_sem, x_dep)
print(f'MambaOut: in={x_sem.shape}, out={out_s3.shape}, params={sum(p.numel() for p in mout.parameters()):,}')
print('All new blocks OK')
"
```

Expected: All shapes (2, 192, 128, 256), no errors.

---

### Task 3: Add HiResRefineNet model class

**Files:**
- Modify: `mbps_pytorch/refine_net.py`

**Step 1: Add HiResRefineNet class after CSCMRefineNet**

This is the main model that chains: projection → upsampling → refinement blocks → head.

```python
class HiResRefineNet(nn.Module):
    """High-Resolution Cross-Modal Refinement Network.

    Upsamples DINOv2 features from 32x64 to a configurable target resolution
    (default 128x256), then refines with configurable block architecture.

    Supports 5 block types: conv, attention, mamba_bidir, mamba_spatial, mambaout
    Supports 3 upsampling strategies: bilinear, transposed_conv, pixel_shuffle
    """

    BLOCK_TYPES = ("conv", "attention", "mamba_bidir", "mamba_spatial", "mambaout")
    UPSAMPLE_STRATEGIES = ("bilinear", "transposed_conv", "pixel_shuffle", "none")

    def __init__(
        self,
        num_classes: int = 19,
        feature_dim: int = 768,
        bridge_dim: int = 192,
        num_blocks: int = 4,
        block_type: str = "conv",
        upsample_strategy: str = "transposed_conv",
        coupling_strength: float = 0.1,
        window_size: int = 8,
        num_heads: int = 4,
        # Mamba-specific
        layer_type: str = "mamba2",
        d_state: int = 64,
        chunk_size: int = 64,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        assert block_type in self.BLOCK_TYPES, f"block_type must be one of {self.BLOCK_TYPES}"
        assert upsample_strategy in self.UPSAMPLE_STRATEGIES

        self.gradient_checkpointing = gradient_checkpointing
        self.block_type = block_type

        # Input projections (at 32x64)
        self.sem_proj = SemanticProjection(feature_dim, bridge_dim)
        self.depth_feat_proj = DepthFeatureProjection(feature_dim, bridge_dim)

        # Upsampling (32x64 → target resolution)
        self.upsample_sem = _build_upsample(upsample_strategy, bridge_dim)
        self.upsample_depth = _build_upsample(upsample_strategy, bridge_dim)

        # Refinement blocks
        blocks = []
        for i in range(num_blocks):
            if block_type == "conv":
                blocks.append(CoupledConvBlock(
                    d_model=bridge_dim,
                    coupling_strength=coupling_strength,
                ))
            elif block_type == "attention":
                # Alternate non-shifted and shifted windows
                blocks.append(WindowedAttentionBlock(
                    d_model=bridge_dim,
                    window_size=window_size,
                    num_heads=num_heads,
                    coupling_strength=coupling_strength,
                    shift=(i % 2 == 1),
                ))
            elif block_type == "mamba_bidir":
                blocks.append(CoupledMambaBlock(
                    d_model=bridge_dim,
                    layer_type=layer_type,
                    scan_mode="bidirectional",
                    coupling_strength=coupling_strength,
                    d_state=d_state,
                    chunk_size=chunk_size,
                ))
            elif block_type == "mamba_spatial":
                blocks.append(CoupledMambaBlock(
                    d_model=bridge_dim,
                    layer_type=layer_type,
                    scan_mode="cross_scan",
                    coupling_strength=coupling_strength,
                    d_state=d_state,
                    chunk_size=chunk_size,
                ))
            elif block_type == "mambaout":
                blocks.append(MambaOutBlock(
                    d_model=bridge_dim,
                    coupling_strength=coupling_strength,
                ))
        self.blocks = nn.ModuleList(blocks)

        # Classification head (at upsampled resolution)
        self.head = nn.Sequential(
            nn.GroupNorm(1, bridge_dim),
            nn.Conv2d(bridge_dim, num_classes, 1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, dinov2_features, depth, depth_grads):
        """
        Args:
            dinov2_features: (B, 768, 32, 64) DINOv2 patch features
            depth: (B, 1, H_depth, W_depth) normalized depth
            depth_grads: (B, 2, H_depth, W_depth) Sobel gradients
        Returns:
            refined_logits: (B, num_classes, H_up, W_up) — at upsampled resolution
        """
        # Project at 32x64
        sem = self.sem_proj(dinov2_features)
        depth_feat = self.depth_feat_proj(dinov2_features, depth, depth_grads)

        # Upsample to target resolution
        sem = self.upsample_sem(sem)
        depth_feat = self.upsample_depth(depth_feat)

        # Refinement
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                sem, depth_feat = checkpoint(
                    block, sem, depth_feat, use_reentrant=False,
                )
            else:
                sem, depth_feat = block(sem, depth_feat)

        return self.head(sem)
```

**Step 2: Verify HiResRefineNet compiles with all block types**

```bash
cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python -c "
import torch
from mbps_pytorch.refine_net import HiResRefineNet

feat = torch.randn(1, 768, 32, 64)
depth = torch.randn(1, 1, 32, 64)
grads = torch.randn(1, 2, 32, 64)

for bt in ['conv', 'attention', 'mambaout']:
    model = HiResRefineNet(num_classes=19, block_type=bt, upsample_strategy='transposed_conv',
                           num_blocks=2, gradient_checkpointing=False)
    out = model(feat, depth, grads)
    params = sum(p.numel() for p in model.parameters())
    print(f'{bt}: output={out.shape}, params={params:,}')

# Mamba variants (smaller for quick test)
for bt in ['mamba_bidir', 'mamba_spatial']:
    model = HiResRefineNet(num_classes=19, block_type=bt, upsample_strategy='bilinear',
                           num_blocks=1, gradient_checkpointing=False,
                           d_state=16, chunk_size=64)
    out = model(feat, depth, grads)
    params = sum(p.numel() for p in model.parameters())
    print(f'{bt}: output={out.shape}, params={params:,}')

print('All HiResRefineNet block types OK')
"
```

Expected: All output shape `(1, 19, 128, 256)`.

---

### Task 4: Adapt training script for HiResRefineNet

**Files:**
- Modify: `mbps_pytorch/train_refine_net.py`

**Step 1: Update dataset to support configurable target resolution**

In `PseudoLabelDataset.__init__`, add `target_h` and `target_w` parameters. When set, the dataset returns depth and pseudo-labels at the target resolution instead of 32x64.

Modify `__init__` signature:
```python
def __init__(
    self,
    cityscapes_root: str,
    split: str = "train",
    semantic_subdir: str = "pseudo_semantic_cause_crf",
    feature_subdir: str = "dinov2_features",
    depth_subdir: str = "depth_spidepth",
    logits_subdir: str = None,
    num_classes: int = 27,
    target_h: int = None,  # NEW: target spatial height (e.g. 128)
    target_w: int = None,  # NEW: target spatial width (e.g. 256)
):
    ...
    self.target_h = target_h
    self.target_w = target_w
```

In `__getitem__`, AFTER loading features (which stay at 32x64), conditionally resize depth and pseudo-labels:

```python
# Determine spatial resolution for depth and labels
out_h = self.target_h if self.target_h else PATCH_H
out_w = self.target_w if self.target_w else PATCH_W

# Load depth: (512, 1024) float32 → downsample to (1, out_h, out_w)
depth_path = os.path.join(
    self.root, self.depth_subdir, self.split, city, f"{stem}.npy",
)
depth_full = np.load(depth_path)  # (512, 1024)
depth_patch = torch.from_numpy(depth_full).unsqueeze(0).unsqueeze(0)
depth_patch = F.interpolate(
    depth_patch, size=(out_h, out_w),
    mode="bilinear", align_corners=False,
).squeeze(0)  # (1, out_h, out_w)
depth_np = depth_patch.numpy()

# Compute Sobel gradients at target resolution
depth_grads = self._sobel_gradients(depth_np[0])  # (2, out_h, out_w)
```

And update `_load_onehot_semantics` to use target resolution:

```python
def _load_onehot_semantics(self, city, stem):
    """Load argmax PNG and convert to smoothed one-hot at target resolution."""
    sem_path = os.path.join(
        self.root, self.semantic_subdir, self.split, city, f"{stem}.png",
    )
    sem_full = np.array(Image.open(sem_path))  # (1024, 2048) uint8

    out_h = self.target_h if self.target_h else PATCH_H
    out_w = self.target_w if self.target_w else PATCH_W

    # Downsample to target resolution via nearest neighbor
    sem_pil = Image.fromarray(sem_full)
    sem_patch = np.array(
        sem_pil.resize((out_w, out_h), Image.NEAREST)
    )  # (out_h, out_w)

    nc = self.num_classes
    smooth = 0.1
    onehot = np.zeros((nc, out_h, out_w), dtype=np.float32)
    onehot[:] = smooth / nc
    for c in range(nc):
        mask = sem_patch == c
        onehot[c][mask] = 1.0 - smooth + smooth / nc

    ignore_mask = sem_patch == 255
    if ignore_mask.any():
        onehot[:, ignore_mask] = 1.0 / nc

    return np.log(np.clip(onehot, 1e-7, None))
```

**Step 2: Update `train()` to support HiResRefineNet**

Import HiResRefineNet:
```python
from mbps_pytorch.refine_net import CSCMRefineNet, HiResRefineNet
```

In `train()`, choose model based on `args.model_type`:

```python
if args.model_type == "hires":
    model = HiResRefineNet(
        num_classes=num_classes,
        feature_dim=768,
        bridge_dim=args.bridge_dim,
        num_blocks=args.num_blocks,
        block_type=args.block_type,
        upsample_strategy=args.upsample_strategy,
        coupling_strength=args.coupling_strength,
        window_size=args.window_size,
        num_heads=args.num_heads,
        layer_type=args.layer_type,
        d_state=args.d_state,
        chunk_size=args.chunk_size,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)
else:
    model = CSCMRefineNet(
        num_classes=num_classes,
        feature_dim=768,
        bridge_dim=args.bridge_dim,
        num_blocks=args.num_blocks,
        block_type=args.block_type,
        layer_type=args.layer_type,
        scan_mode=args.scan_mode,
        coupling_strength=args.coupling_strength,
        d_state=args.d_state,
        chunk_size=args.chunk_size,
        gradient_checkpointing=args.gradient_checkpointing,
        use_aspp=getattr(args, 'use_aspp', False),
    ).to(device)
```

Pass target resolution to datasets:
```python
target_h = args.target_h if args.model_type == "hires" else None
target_w = args.target_w if args.model_type == "hires" else None

train_dataset = PseudoLabelDataset(
    args.cityscapes_root, split="train",
    semantic_subdir=args.semantic_subdir,
    logits_subdir=args.logits_subdir,
    num_classes=num_classes,
    target_h=target_h,
    target_w=target_w,
)
val_dataset = PseudoLabelDataset(
    args.cityscapes_root, split="val",
    semantic_subdir=args.semantic_subdir,
    logits_subdir=args.logits_subdir,
    num_classes=num_classes,
    target_h=target_h,
    target_w=target_w,
)
```

**Step 3: Update `feature_prototype_loss` to handle resolution mismatch**

When using HiResRefineNet, logits are at 128x256 but `dinov2_features` are at 32x64 (768-dim). The prototype loss needs features at the same resolution as logits. Add interpolation:

In `RefineNetLoss.forward`, before calling `feature_prototype_loss`:
```python
if self.lambda_proto > 0:
    # Match feature resolution to logits resolution
    feat_for_proto = dinov2_features
    if feat_for_proto.shape[2:] != logits.shape[2:]:
        feat_for_proto = F.interpolate(
            feat_for_proto, size=logits.shape[2:],
            mode='bilinear', align_corners=False)
    l_proto = feature_prototype_loss(logits, feat_for_proto)
    losses["proto"] = l_proto
```

**Step 4: Add new CLI arguments**

```python
# HiRes model
parser.add_argument("--model_type", type=str, default="cscm",
                    choices=["cscm", "hires"],
                    help="Model type: cscm (original 32x64) or hires (upsampled)")
parser.add_argument("--upsample_strategy", type=str, default="transposed_conv",
                    choices=["bilinear", "transposed_conv", "pixel_shuffle", "none"])
parser.add_argument("--target_h", type=int, default=128,
                    help="Target spatial height for HiRes model")
parser.add_argument("--target_w", type=int, default=256,
                    help="Target spatial width for HiRes model")
parser.add_argument("--window_size", type=int, default=8,
                    help="Window size for windowed attention blocks")
parser.add_argument("--num_heads", type=int, default=4,
                    help="Number of attention heads")
```

Also update `--block_type` choices:
```python
parser.add_argument("--block_type", type=str, default="conv",
                    choices=["conv", "mamba", "attention", "mamba_bidir", "mamba_spatial", "mambaout"])
```

**Step 5: Test HiRes training compiles (dry run)**

```bash
cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python -c "
from mbps_pytorch.train_refine_net import PseudoLabelDataset
ds = PseudoLabelDataset(
    '/Users/qbit-glitch/Desktop/datasets/cityscapes',
    split='val',
    semantic_subdir='pseudo_semantic_mapped_k80',
    num_classes=19,
    target_h=128,
    target_w=256,
)
sample = ds[0]
print('cause_logits:', sample['cause_logits'].shape)  # (19, 128, 256)
print('features:', sample['dinov2_features'].shape)     # (768, 32, 64)
print('depth:', sample['depth'].shape)                  # (1, 128, 256)
print('depth_grads:', sample['depth_grads'].shape)      # (2, 128, 256)
"
```

Expected: `cause_logits: torch.Size([19, 128, 256])`, features stay at 32x64, depth/grads at 128x256.

---

### Task 5: Run Phase 1 — Upsampling ablation (3 runs)

**Files:**
- Use: `mbps_pytorch/train_refine_net.py`
- Output: `checkpoints/hires_upsample_{bilinear,transposed_conv,pixel_shuffle}/`

Run all 3 upsampling strategies with Conv2d blocks (the proven baseline architecture), 20 epochs each. Run sequentially (not parallel) since each run will be memory-intensive at 128x256.

**Step 1: Run U-B (Transposed Conv)**

```bash
cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python mbps_pytorch/train_refine_net.py \
    --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
    --semantic_subdir pseudo_semantic_mapped_k80 \
    --output_dir checkpoints/hires_upsample_transposed_conv \
    --model_type hires \
    --block_type conv \
    --upsample_strategy transposed_conv \
    --target_h 128 --target_w 256 \
    --num_classes 19 \
    --num_epochs 20 \
    --batch_size 4 \
    --lr 5e-5 \
    --eval_interval 2 \
    --lambda_distill 1.0 \
    --lambda_distill_min 0.85 \
    --lambda_align 0.25 \
    --lambda_proto 0.025 \
    --lambda_ent 0.025 \
    --label_smoothing 0.1 \
    --device auto
```

Run in background. Monitor: `tail -f checkpoints/hires_upsample_transposed_conv/metrics_history.jsonl`

**Step 2: Run U-A (Bilinear)**

```bash
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python mbps_pytorch/train_refine_net.py \
    --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
    --semantic_subdir pseudo_semantic_mapped_k80 \
    --output_dir checkpoints/hires_upsample_bilinear \
    --model_type hires \
    --block_type conv \
    --upsample_strategy bilinear \
    --target_h 128 --target_w 256 \
    --num_classes 19 \
    --num_epochs 20 \
    --batch_size 4 \
    --lr 5e-5 \
    --eval_interval 2 \
    --lambda_distill 1.0 \
    --lambda_distill_min 0.85 \
    --lambda_align 0.25 \
    --lambda_proto 0.025 \
    --lambda_ent 0.025 \
    --label_smoothing 0.1 \
    --device auto
```

**Step 3: Run U-C (PixelShuffle)**

```bash
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python mbps_pytorch/train_refine_net.py \
    --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
    --semantic_subdir pseudo_semantic_mapped_k80 \
    --output_dir checkpoints/hires_upsample_pixel_shuffle \
    --model_type hires \
    --block_type conv \
    --upsample_strategy pixel_shuffle \
    --target_h 128 --target_w 256 \
    --num_classes 19 \
    --num_epochs 20 \
    --batch_size 4 \
    --lr 5e-5 \
    --eval_interval 2 \
    --lambda_distill 1.0 \
    --lambda_distill_min 0.85 \
    --lambda_align 0.25 \
    --lambda_proto 0.025 \
    --lambda_ent 0.025 \
    --label_smoothing 0.1 \
    --device auto
```

**Step 4: Collect Phase 1 results**

```bash
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python -c "
import torch
for name, path in [
    ('U-B: TransposedConv', 'checkpoints/hires_upsample_transposed_conv/best.pth'),
    ('U-A: Bilinear', 'checkpoints/hires_upsample_bilinear/best.pth'),
    ('U-C: PixelShuffle', 'checkpoints/hires_upsample_pixel_shuffle/best.pth'),
]:
    ckpt = torch.load(path, weights_only=False, map_location='cpu')
    m = ckpt.get('metrics', {})
    print(f'{name} | ep={ckpt[\"epoch\"]} | PQ={m.get(\"PQ\",0):.2f} | PQ_stuff={m.get(\"PQ_stuff\",0):.2f} | PQ_things={m.get(\"PQ_things\",0):.2f} | mIoU={m.get(\"mIoU\",0):.2f}')
"
```

**Step 5: Select best upsampling and update report**

Update `reports/hires_refinenet_128x256.md` Section 4 with Phase 1 results. Carry the best upsampling forward to Phase 2.

---

### Task 6: Run Phase 2 — Architecture ablation (5 runs)

**Files:**
- Use: `mbps_pytorch/train_refine_net.py`
- Output: `checkpoints/hires_arch_{conv,attention,mamba_bidir,mamba_spatial,mambaout}/`

Use the best upsampling strategy from Phase 1. Run all 5 architecture variants, 20 epochs each.

**Step 1: Run A-Conv (Conv2d baseline at 128x256)**

(Note: this may be the same as the best Phase 1 run — skip if so, just copy the checkpoint)

```bash
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python mbps_pytorch/train_refine_net.py \
    --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
    --semantic_subdir pseudo_semantic_mapped_k80 \
    --output_dir checkpoints/hires_arch_conv \
    --model_type hires \
    --block_type conv \
    --upsample_strategy BEST_FROM_PHASE1 \
    --target_h 128 --target_w 256 \
    --num_classes 19 \
    --num_epochs 20 \
    --batch_size 4 \
    --lr 5e-5 \
    --eval_interval 2 \
    --lambda_distill 1.0 \
    --lambda_distill_min 0.85 \
    --lambda_align 0.25 \
    --lambda_proto 0.025 \
    --lambda_ent 0.025 \
    --label_smoothing 0.1 \
    --device auto
```

**Step 2: Run A-Attn (Windowed Self-Attention)**

```bash
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python mbps_pytorch/train_refine_net.py \
    --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
    --semantic_subdir pseudo_semantic_mapped_k80 \
    --output_dir checkpoints/hires_arch_attention \
    --model_type hires \
    --block_type attention \
    --upsample_strategy BEST_FROM_PHASE1 \
    --target_h 128 --target_w 256 \
    --num_classes 19 \
    --num_epochs 20 \
    --batch_size 4 \
    --lr 5e-5 \
    --eval_interval 2 \
    --lambda_distill 1.0 \
    --lambda_distill_min 0.85 \
    --lambda_align 0.25 \
    --lambda_proto 0.025 \
    --lambda_ent 0.025 \
    --label_smoothing 0.1 \
    --window_size 8 \
    --num_heads 4 \
    --device auto
```

**Step 3: Run A-VM2 (VisionMamba2 Bidirectional)**

```bash
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python mbps_pytorch/train_refine_net.py \
    --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
    --semantic_subdir pseudo_semantic_mapped_k80 \
    --output_dir checkpoints/hires_arch_mamba_bidir \
    --model_type hires \
    --block_type mamba_bidir \
    --upsample_strategy BEST_FROM_PHASE1 \
    --target_h 128 --target_w 256 \
    --num_classes 19 \
    --num_epochs 20 \
    --batch_size 2 \
    --lr 5e-5 \
    --eval_interval 2 \
    --lambda_distill 1.0 \
    --lambda_distill_min 0.85 \
    --lambda_align 0.25 \
    --lambda_proto 0.025 \
    --lambda_ent 0.025 \
    --label_smoothing 0.1 \
    --d_state 64 \
    --chunk_size 64 \
    --device auto
```

Note: `batch_size=2` for Mamba variants (higher memory usage at 32K tokens).

**Step 4: Run A-SM (Spatial Mamba SS2D)**

```bash
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python mbps_pytorch/train_refine_net.py \
    --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
    --semantic_subdir pseudo_semantic_mapped_k80 \
    --output_dir checkpoints/hires_arch_mamba_spatial \
    --model_type hires \
    --block_type mamba_spatial \
    --upsample_strategy BEST_FROM_PHASE1 \
    --target_h 128 --target_w 256 \
    --num_classes 19 \
    --num_epochs 20 \
    --batch_size 2 \
    --lr 5e-5 \
    --eval_interval 2 \
    --lambda_distill 1.0 \
    --lambda_distill_min 0.85 \
    --lambda_align 0.25 \
    --lambda_proto 0.025 \
    --lambda_ent 0.025 \
    --label_smoothing 0.1 \
    --d_state 64 \
    --chunk_size 64 \
    --device auto
```

**Step 5: Run A-MOut (MambaOut)**

```bash
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python mbps_pytorch/train_refine_net.py \
    --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
    --semantic_subdir pseudo_semantic_mapped_k80 \
    --output_dir checkpoints/hires_arch_mambaout \
    --model_type hires \
    --block_type mambaout \
    --upsample_strategy BEST_FROM_PHASE1 \
    --target_h 128 --target_w 256 \
    --num_classes 19 \
    --num_epochs 20 \
    --batch_size 4 \
    --lr 5e-5 \
    --eval_interval 2 \
    --lambda_distill 1.0 \
    --lambda_distill_min 0.85 \
    --lambda_align 0.25 \
    --lambda_proto 0.025 \
    --lambda_ent 0.025 \
    --label_smoothing 0.1 \
    --device auto
```

**Step 6: Collect Phase 2 results**

```bash
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python -c "
import torch
for name, path in [
    ('A-Conv', 'checkpoints/hires_arch_conv/best.pth'),
    ('A-Attn', 'checkpoints/hires_arch_attention/best.pth'),
    ('A-VM2', 'checkpoints/hires_arch_mamba_bidir/best.pth'),
    ('A-SM', 'checkpoints/hires_arch_mamba_spatial/best.pth'),
    ('A-MOut', 'checkpoints/hires_arch_mambaout/best.pth'),
]:
    try:
        ckpt = torch.load(path, weights_only=False, map_location='cpu')
        m = ckpt.get('metrics', {})
        print(f'{name} | ep={ckpt[\"epoch\"]} | PQ={m.get(\"PQ\",0):.2f} | PQ_stuff={m.get(\"PQ_stuff\",0):.2f} | PQ_things={m.get(\"PQ_things\",0):.2f} | mIoU={m.get(\"mIoU\",0):.2f}')
    except Exception as e:
        print(f'{name} | ERROR: {e}')
"
```

---

### Task 7: Update report with results

**Files:**
- Modify: `reports/hires_refinenet_128x256.md`
- Modify: `MEMORY.md` (auto-memory)

**Step 1: Fill in Phase 1 and Phase 2 result tables**

Update Section 4 and Section 5 in the report with actual numbers from the checkpoint metrics.

**Step 2: Write analysis sections**

Fill in Sections 5.1 (PQ_things recovery), 5.2 (computational efficiency), 5.3 (analysis), 6 (discussion), and 7 (conclusion) based on the experimental results.

**Step 3: Update MEMORY.md**

Add HiRes RefineNet results and key findings to the auto-memory.

**Step 4: Commit**

```bash
git add mbps_pytorch/refine_net.py mbps_pytorch/train_refine_net.py reports/hires_refinenet_128x256.md docs/plans/
git commit -m "feat: HiRes RefineNet 128x256 with 3 upsampling + 5 architecture ablations"
```
