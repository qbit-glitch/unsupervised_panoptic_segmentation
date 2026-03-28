# Mobile Panoptic Training Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the training gap between our naive distillation pipeline and CUPS by adding proper augmentations, instance heads, and self-training to our lightweight mobile panoptic model.

**Architecture:** RepViT-M0.9 (4.7M) + SimpleFPN (0.2M) + semantic head + instance heads, trained on unsupervised pseudo-labels (semantic k=80 mapped + depth-guided instances).

**Tech Stack:** PyTorch 2.x, timm, torchvision, scipy, PIL. Runs on GTX 1080 Ti (11GB).

**Reports:**
- Gap analysis: `reports/mobile_distillation_gap_analysis.md`
- Instance head design: `reports/instance_head_ablation_design.md`
- Backbone research: `docs/plans/2026-03-08-lightweight-backbone-research.md`

**Remote:** `santosh@100.93.203.100`, datasets at `/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/`

---

## Phase 0: Training Recipe Fixes (P0 + P1 from gap analysis)

These are free PQ gains requiring no architectural changes. Must be done before any ablation run.

### Task 1: Add Full Photometric Augmentations

**Files:**
- Modify: `mbps_pytorch/train_mobile_panoptic.py` (class `CityscapesPseudoLabelDataset.__getitem__`)

**Step 1: Write the augmentation code**

Replace the current brightness-only jitter (lines 127-130) with a full photometric pipeline matching CUPS:

```python
# In __init__, add:
self.color_jitter = transforms.ColorJitter(
    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
)
self.gaussian_blur = transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))

# In __getitem__, replace brightness block with:
if self.is_train:
    # ... after crop ...
    # Photometric augmentations (applied to PIL image before numpy conversion)
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray((img * 255).astype(np.uint8))
    if np.random.rand() > 0.5:
        img_pil = self.color_jitter(img_pil)
    if np.random.rand() > 0.5:
        img_pil = self.gaussian_blur(img_pil)
    if np.random.rand() > 0.8:
        img_pil = transforms.functional.rgb_to_grayscale(img_pil, num_output_channels=3)
    img = np.array(img_pil, dtype=np.float32) / 255.0
```

**Step 2: Verify augmentations don't crash**

Run: `python mbps_pytorch/train_mobile_panoptic.py --cityscapes_root <path> --num_epochs 1 --eval_interval 1`
Expected: Completes 1 epoch without error. Verify images are augmented (loss should be similar or slightly higher due to harder inputs).

**Step 3: Commit**

```bash
git add mbps_pytorch/train_mobile_panoptic.py
git commit -m "feat: add full photometric augmentations (ColorJitter, GaussianBlur, Grayscale)"
```

---

### Task 2: Add Multi-Scale RandomResizedCrop

**Files:**
- Modify: `mbps_pytorch/train_mobile_panoptic.py` (class `CityscapesPseudoLabelDataset.__getitem__`)

**Step 1: Replace fixed crop with RandomResizedCrop**

Replace the fixed 384x768 crop with scale-jittered cropping matching CUPS's `RandomResizedCrop(scale=(0.5, 1.5))`:

```python
if self.is_train:
    # Random scale factor (multi-scale training)
    scale = np.random.uniform(0.5, 1.5)
    ch, cw = self.crop_size  # (384, 768)
    scaled_h = int(ch * scale)
    scaled_w = int(cw * scale)

    # Resize image and label to scaled size, then crop to target
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray((img * 255).astype(np.uint8))
    label_pil = PILImage.fromarray(label.astype(np.uint8))

    # Scale
    new_h = int(H * scale)
    new_w = int(W * scale)
    img_pil = img_pil.resize((new_w, new_h), PILImage.BILINEAR)
    label_pil = label_pil.resize((new_w, new_h), PILImage.NEAREST)

    img = np.array(img_pil, dtype=np.float32) / 255.0
    label = np.array(label_pil, dtype=np.int64)
    H, W = img.shape[:2]

    # Random crop to target size
    if H >= ch and W >= cw:
        y = np.random.randint(0, H - ch + 1)
        x = np.random.randint(0, W - cw + 1)
        img = img[y:y+ch, x:x+cw]
        label = label[y:y+ch, x:x+cw]
    else:
        # Pad if scaled image is smaller than crop
        img_pil = PILImage.fromarray((img * 255).astype(np.uint8)).resize((cw, ch), PILImage.BILINEAR)
        label_pil = PILImage.fromarray(label.astype(np.uint8)).resize((cw, ch), PILImage.NEAREST)
        img = np.array(img_pil, dtype=np.float32) / 255.0
        label = np.array(label_pil, dtype=np.int64)
```

**Step 2: Test scale jittering**

Run 1 epoch, verify loss and mIoU are reasonable.

**Step 3: Commit**

```bash
git add mbps_pytorch/train_mobile_panoptic.py
git commit -m "feat: add multi-scale RandomResizedCrop (0.5-1.5x) for training"
```

---

### Task 3: Run Augmented Baseline (semantic-only)

**Files:**
- No new files. Use modified `train_mobile_panoptic.py`.

**Step 1: Upload and run on GPU-1**

```bash
scp mbps_pytorch/train_mobile_panoptic.py santosh@100.93.203.100:/media/santosh/Kuldeep/panoptic_segmentation/mbps_pytorch/

ssh santosh@100.93.203.100
export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
cd /media/santosh/Kuldeep/panoptic_segmentation

nohup python mbps_pytorch/train_mobile_panoptic.py \
    --cityscapes_root datasets/cityscapes \
    --semantic_subdir pseudo_semantic_mapped_k80 \
    --backbone repvit_m0_9.dist_450e_in1k \
    --num_classes 19 --fpn_dim 128 \
    --num_epochs 50 --batch_size 4 --lr 1e-3 \
    --backbone_lr_ratio 0.1 --freeze_backbone_epochs 5 \
    --crop_h 384 --crop_w 768 --eval_interval 2 \
    --label_smoothing 0.1 --num_workers 4 \
    --output_dir checkpoints/mobile_repvit_m09_augmented \
    > logs/mobile_repvit_m09_augmented.log 2>&1 &
```

**Step 2: Monitor**

```bash
cat logs/mobile_repvit_m09_augmented.log | tr '\r' '\n' | grep -E '(loss=.*time|mIoU|PQ|best|unfrozen)'
```

Expected: PQ should exceed the current vanilla run (PQ=13.87 at epoch 2) by epoch 10 with augmentations.

**Step 3: Record results in report**

Update `reports/mobile_distillation_gap_analysis.md` Section 2.3 with augmented baseline results.

---

## Phase 1: Instance Head Infrastructure

### Task 4: Pre-Compute Instance Training Targets

**Files:**
- Create: `mbps_pytorch/generate_instance_targets.py`

This script reads instance pseudo-labels and generates center heatmaps + offset maps + boundary maps for all training images. These are saved as .npy files for efficient loading during training.

**Step 1: Write the target generation script**

```python
#!/usr/bin/env python3
"""Generate instance training targets from pseudo-instance labels.

Reads: cups_pseudo_labels_v3/*_instance.png (uint16, instance IDs 0-N)
Writes:
  instance_targets/{split}/{city}/{stem}_center.npy   (H, W) float32 heatmap
  instance_targets/{split}/{city}/{stem}_offset.npy   (2, H, W) float32 (dy, dx)
  instance_targets/{split}/{city}/{stem}_boundary.npy  (H, W) uint8 binary boundary

Usage:
  python mbps_pytorch/generate_instance_targets.py \
    --cityscapes_root /path/to/cityscapes \
    --instance_subdir cups_pseudo_labels_v3 \
    --output_subdir instance_targets \
    --split train
"""

import argparse
import os
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage

def generate_targets_for_image(instance_path, semantic_path, thing_ids):
    """Generate center heatmap, offset map, boundary map for one image."""
    inst = np.array(Image.open(instance_path), dtype=np.int32)
    H, W = inst.shape

    center_map = np.zeros((H, W), dtype=np.float32)
    offset_map = np.zeros((2, H, W), dtype=np.float32)
    boundary_map = np.zeros((H, W), dtype=np.uint8)

    # Load semantic to identify thing pixels
    if semantic_path and os.path.exists(semantic_path):
        sem = np.array(Image.open(semantic_path), dtype=np.int32)
    else:
        sem = None

    for uid in np.unique(inst):
        if uid == 0:
            continue  # background / stuff

        mask = inst == uid
        area = mask.sum()
        if area < 50:
            continue

        # Centroid
        ys, xs = np.where(mask)
        cy, cx = ys.mean(), xs.mean()

        # Center heatmap (Gaussian)
        sigma = max(np.sqrt(area) / 10.0, 4.0)
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        gaussian = np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2))
        center_map = np.maximum(center_map, gaussian * mask)

        # Offset map (dy, dx from each pixel to its center)
        offset_map[0][mask] = cy - ys  # dy
        offset_map[1][mask] = cx - xs  # dx

        # Boundary: pixels where a neighbor has different instance ID
        eroded = ndimage.binary_erosion(mask, iterations=1)
        boundary = mask & ~eroded
        boundary_map[boundary] = 1

    return center_map, offset_map, boundary_map
```

**Step 2: Run on train + val splits**

```bash
python mbps_pytorch/generate_instance_targets.py \
    --cityscapes_root datasets/cityscapes \
    --instance_subdir cups_pseudo_labels_v3 \
    --output_subdir instance_targets \
    --split train

python mbps_pytorch/generate_instance_targets.py \
    --cityscapes_root datasets/cityscapes \
    --instance_subdir cups_pseudo_labels_v3 \
    --output_subdir instance_targets \
    --split val
```

Expected: ~2975 train + 500 val sets of (center.npy, offset.npy, boundary.npy). ~5GB total.

**Step 3: Commit**

```bash
git add mbps_pytorch/generate_instance_targets.py
git commit -m "feat: add instance target generation (center heatmaps, offsets, boundaries)"
```

---

### Task 5: Extend Dataset to Load Instance Targets

**Files:**
- Modify: `mbps_pytorch/train_mobile_panoptic.py` (class `CityscapesPseudoLabelDataset`)

**Step 1: Add instance target loading**

Add `instance_subdir` parameter to the dataset. When provided, load center.npy, offset.npy, boundary.npy alongside each image. Apply the same spatial transforms (flip, crop, resize) to these targets.

```python
class CityscapesPseudoLabelDataset(Dataset):
    def __init__(self, ..., instance_subdir=None):
        self.instance_subdir = instance_subdir
        # ... existing init ...

    def __getitem__(self, idx):
        # ... existing image + semantic label loading ...

        center, offset, boundary = None, None, None
        if self.instance_subdir:
            city = self.images[idx].parent.name
            stem = self.images[idx].name.replace("_leftImg8bit.png", "")
            base = self.root / self.instance_subdir / self.split / city
            center = np.load(base / f"{stem}_center.npy")    # (H, W)
            offset = np.load(base / f"{stem}_offset.npy")    # (2, H, W)
            boundary = np.load(base / f"{stem}_boundary.npy") # (H, W)

            # Apply same spatial transforms as image/label
            if flip:
                center = center[:, ::-1].copy()
                offset = offset[:, :, ::-1].copy()
                offset[1] = -offset[1]  # flip dx
                boundary = boundary[:, ::-1].copy()

            # Apply same crop
            center = center[y:y+ch, x:x+cw]
            offset = offset[:, y:y+ch, x:x+cw]
            boundary = boundary[y:y+ch, x:x+cw]

        # Return as tensors
        result = (img_t, label_t)
        if center is not None:
            result += (
                torch.from_numpy(center).float(),
                torch.from_numpy(offset.copy()).float(),
                torch.from_numpy(boundary.copy()).float(),
            )
        return result
```

**Step 2: Verify loading works**

Quick test: load one sample, print shapes, verify transforms are correct.

**Step 3: Commit**

```bash
git add mbps_pytorch/train_mobile_panoptic.py
git commit -m "feat: extend dataset to load instance targets (center, offset, boundary)"
```

---

### Task 6: Implement Three Instance Heads

**Files:**
- Modify: `mbps_pytorch/train_mobile_panoptic.py` (add head modules + MobilePanopticModel changes)

**Step 1: Implement EmbeddingHead, CenterOffsetHead, BoundaryHead**

```python
class EmbeddingHead(nn.Module):
    """16-dim discriminative embedding head."""
    def __init__(self, in_dim=128, embed_dim=16):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim),
            nn.Conv2d(in_dim, in_dim, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, embed_dim, 1),
        )
    def forward(self, x):
        return self.head(x)  # (B, embed_dim, H, W)


class CenterOffsetHead(nn.Module):
    """Center heatmap (1-ch) + offset (2-ch) head."""
    def __init__(self, in_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim),
            nn.Conv2d(in_dim, in_dim, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )
        self.center = nn.Conv2d(in_dim, 1, 1)
        self.offset = nn.Conv2d(in_dim, 2, 1)

    def forward(self, x):
        feat = self.shared(x)
        return torch.sigmoid(self.center(feat)), self.offset(feat)


class BoundaryHead(nn.Module):
    """Instance boundary prediction (1-ch) head."""
    def __init__(self, in_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim),
            nn.Conv2d(in_dim, in_dim, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, 1, 1),
        )
    def forward(self, x):
        return torch.sigmoid(self.head(x))
```

**Step 2: Add `--instance_head` CLI argument**

```python
parser.add_argument("--instance_head", type=str, default="none",
                    choices=["none", "embedding", "center_offset", "boundary",
                             "embed_center", "embed_boundary", "center_boundary", "all"],
                    help="Instance head type for ablation")
```

**Step 3: Modify MobilePanopticModel to conditionally create heads**

Based on `--instance_head`, create the appropriate head(s) and add their parameters to the optimizer.

**Step 4: Verify model creation**

```python
model = MobilePanopticModel("repvit_m0_9.dist_450e_in1k", instance_head="center_offset")
# Should print: Total params: 5.05M
```

**Step 5: Commit**

```bash
git add mbps_pytorch/train_mobile_panoptic.py
git commit -m "feat: add three instance head options (embedding, center_offset, boundary)"
```

---

### Task 7: Implement Instance Losses

**Files:**
- Modify: `mbps_pytorch/train_mobile_panoptic.py` (training loop)

**Step 1: Implement discriminative embedding loss**

```python
def discriminative_loss(embeddings, instance_labels, delta_v=0.5, delta_d=1.5):
    """Discriminative loss for embedding head.

    Args:
        embeddings: (B, E, H, W) embedding predictions
        instance_labels: (B, H, W) instance IDs (0=ignore)
    """
    B = embeddings.shape[0]
    loss = 0.0
    for b in range(B):
        emb = embeddings[b]  # (E, H, W)
        inst = instance_labels[b]  # (H, W)
        ids = inst.unique()
        ids = ids[ids > 0]  # exclude background
        if len(ids) < 2:
            continue

        means = []
        pull_loss = 0.0
        for uid in ids:
            mask = inst == uid
            emb_k = emb[:, mask]  # (E, N_k)
            mu_k = emb_k.mean(dim=1, keepdim=True)  # (E, 1)
            means.append(mu_k.squeeze())
            pull_loss += torch.clamp(torch.norm(emb_k - mu_k, dim=0) - delta_v, min=0).mean()
        pull_loss /= len(ids)

        push_loss = 0.0
        means = torch.stack(means)  # (K, E)
        K = means.shape[0]
        for i in range(K):
            for j in range(i+1, K):
                dist = torch.norm(means[i] - means[j])
                push_loss += torch.clamp(2*delta_d - dist, min=0) ** 2
        push_loss /= max(K * (K-1) / 2, 1)

        loss += pull_loss + push_loss
    return loss / B
```

**Step 2: Implement center + offset losses**

```python
def center_offset_loss(pred_center, pred_offset, target_center, target_offset, thing_mask):
    """MSE for center heatmap + SmoothL1 for offset vectors.

    Args:
        pred_center: (B, 1, H, W) predicted center heatmap
        pred_offset: (B, 2, H, W) predicted offsets
        target_center: (B, H, W) target center heatmap
        target_offset: (B, 2, H, W) target offsets
        thing_mask: (B, H, W) bool mask for thing-class pixels
    """
    # Center loss (weighted MSE: 10x on positive pixels)
    weight = torch.where(target_center > 0.01, 10.0, 1.0)
    center_loss = (weight * (pred_center.squeeze(1) - target_center) ** 2).mean()

    # Offset loss (SmoothL1, only on thing pixels with valid instances)
    valid = thing_mask & (target_center > 0.01)
    if valid.sum() > 0:
        offset_loss = F.smooth_l1_loss(
            pred_offset[:, :, valid.squeeze() if valid.dim() == 4 else valid],
            target_offset[:, :, valid],
        )
    else:
        offset_loss = torch.tensor(0.0, device=pred_center.device)

    return center_loss + offset_loss
```

**Step 3: Implement boundary loss**

```python
def boundary_loss(pred_boundary, target_boundary, thing_mask):
    """Weighted BCE for boundary prediction.

    Args:
        pred_boundary: (B, 1, H, W) predicted boundary probability
        target_boundary: (B, H, W) target boundary (0 or 1)
        thing_mask: (B, H, W) bool mask for thing-class pixels
    """
    pred = pred_boundary.squeeze(1)[thing_mask]
    target = target_boundary[thing_mask].float()
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred_boundary.device)

    # Class-balanced weight
    pos_weight = (target == 0).sum().float() / max((target == 1).sum().float(), 1)
    return F.binary_cross_entropy(pred, target,
                                   weight=torch.where(target == 1, pos_weight, 1.0))
```

**Step 4: Integrate into training loop**

Add instance loss computation after semantic loss, weighted by `--lambda_instance` (default 1.0):
```python
total_loss = semantic_loss + args.lambda_instance * instance_loss
```

**Step 5: Verify losses produce valid gradients**

Run 10 training steps, check that all losses decrease and no NaNs appear.

**Step 6: Commit**

```bash
git add mbps_pytorch/train_mobile_panoptic.py
git commit -m "feat: add instance losses (discriminative, center_offset, boundary)"
```

---

### Task 8: Implement Instance-Aware Panoptic Inference

**Files:**
- Modify: `mbps_pytorch/train_mobile_panoptic.py` (evaluate function)

**Step 1: Add inference methods for each instance head**

For evaluation, replace connected-components with head-specific inference:
- **Embedding**: mean-shift clustering on predicted 16-dim embeddings within each thing-class region.
- **Center/Offset**: find center peaks (NMS on heatmap), group pixels by offset-voted center.
- **Boundary**: subtract boundary pixels from thing-class mask, then connected components.

**Step 2: Integrate into evaluate() function**

The evaluate function should detect which instance head is active and use the corresponding inference method.

**Step 3: Verify PQ computation is correct**

Run on 10 val images, manually inspect panoptic maps.

**Step 4: Commit**

```bash
git add mbps_pytorch/train_mobile_panoptic.py
git commit -m "feat: add instance-aware panoptic inference for evaluation"
```

---

## Phase 2: Instance Head Ablation (3 individual runs)

### Task 9: Run I-A (Embedding Head)

```bash
nohup python mbps_pytorch/train_mobile_panoptic.py \
    --cityscapes_root datasets/cityscapes \
    --semantic_subdir pseudo_semantic_mapped_k80 \
    --instance_subdir instance_targets \
    --backbone repvit_m0_9.dist_450e_in1k \
    --instance_head embedding \
    --lambda_instance 1.0 \
    --num_epochs 50 --batch_size 4 --lr 1e-3 \
    --output_dir checkpoints/mobile_instance_embedding \
    > logs/mobile_instance_embedding.log 2>&1 &
```

Record: PQ, PQ_stuff, PQ_things, mIoU at best epoch.

### Task 10: Run I-B (Center/Offset Head)

```bash
nohup python mbps_pytorch/train_mobile_panoptic.py \
    --cityscapes_root datasets/cityscapes \
    --semantic_subdir pseudo_semantic_mapped_k80 \
    --instance_subdir instance_targets \
    --backbone repvit_m0_9.dist_450e_in1k \
    --instance_head center_offset \
    --lambda_instance 1.0 \
    --num_epochs 50 --batch_size 4 --lr 1e-3 \
    --output_dir checkpoints/mobile_instance_center_offset \
    > logs/mobile_instance_center_offset.log 2>&1 &
```

### Task 11: Run I-C (Boundary Head)

```bash
nohup python mbps_pytorch/train_mobile_panoptic.py \
    --cityscapes_root datasets/cityscapes \
    --semantic_subdir pseudo_semantic_mapped_k80 \
    --instance_subdir instance_targets \
    --backbone repvit_m0_9.dist_450e_in1k \
    --instance_head boundary \
    --lambda_instance 1.0 \
    --num_epochs 50 --batch_size 4 --lr 1e-3 \
    --output_dir checkpoints/mobile_instance_boundary \
    > logs/mobile_instance_boundary.log 2>&1 &
```

### Task 12: Analyze Phase 2 Results

Compare I-A, I-B, I-C against augmented baseline. Select top 2 for pairwise combination.

Update `reports/instance_head_ablation_design.md` with results.

---

## Phase 3: Pairwise Combinations (2-3 runs)

### Task 13: Run Top Pairwise Combination(s)

Based on Phase 2 results, run the most promising pair(s):
- If I-B wins: run I-BC (center + boundary)
- If I-A wins: run I-AB (embedding + center)
- Always run the top-2 combination

```bash
nohup python mbps_pytorch/train_mobile_panoptic.py \
    --instance_head center_boundary \
    ...
```

### Task 14: Analyze Phase 3 Results

Compare pairwise vs individual. If combination > best individual by >1 PQ_things, proceed to Phase 4. Otherwise, select best individual.

---

## Phase 4: EMA Self-Training (P3 from gap analysis)

### Task 15: Implement EMA Teacher-Student

**Files:**
- Modify: `mbps_pytorch/train_mobile_panoptic.py` (add EMA model, self-label generation)

After initial training converges (epoch 30+), switch to self-training:
1. Create EMA copy of model (momentum=0.999)
2. Each step: EMA model generates predictions on augmented views → confidence threshold → self-labels
3. Student trains on photometrically perturbed image with self-labels
4. Update EMA weights from student

Add `--self_training_start_epoch` (default 30) and `--ema_momentum` (default 0.999).

### Task 16: Run Self-Training Ablation

Run the best model from Phase 3 with self-training enabled. Compare PQ before/after self-training.

---

## Phase 5: Final Integration + Copy-Paste

### Task 17: Implement Copy-Paste Augmentation

Using instance pseudo-labels, implement simplified copy-paste:
1. Extract thing-instance crops from batch
2. Random scale (0.25-1.5x) + flip
3. Paste up to 5 objects per image
4. Update semantic + instance targets

### Task 18: Run Final Model with All Components

Full pipeline: augmentations + best instance head + copy-paste + EMA self-training.

Record final PQ and compare against:
- Vanilla baseline (Phase 0 before augmentations)
- Augmented baseline (Task 3)
- Best instance head (Phase 2/3)
- CUPS (PQ=27.8)

### Task 19: Write Final Report

Update both reports with complete results tables. Add per-class PQ analysis.

---

## Summary of Runs

| Run | Phase | Config | GPU | Est. Time |
|-----|-------|--------|-----|-----------|
| Augmented baseline | 0 | Semantic-only + full aug | GPU-1 | ~3h |
| I-A | 2 | + Embedding head | GPU-1 | ~3.5h |
| I-B | 2 | + Center/Offset head | GPU-0 or sequential | ~3.5h |
| I-C | 2 | + Boundary head | Sequential | ~3h |
| I-XY | 3 | Top-2 pairwise | GPU-1 | ~3.5h |
| Self-train | 4 | + EMA self-training | GPU-1 | ~5h |
| Full | 5 | All components | GPU-1 | ~5h |

Total estimated: ~26 hours of GPU time across 7 runs.
