# CSCMRefineNet on k=80 Overclustered Pseudo-Labels — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Retrain CSCMRefineNet (Conv2d blocks) on k=80 overclustered semantic pseudo-labels (80 raw classes), targeting PQ_stuff improvement (32.08 → 35+) while keeping instances untouched.

**Architecture:** Reuse the proven CSCMRefineNet (DINOv2 768-dim features + depth → CoupledConvBlocks → 80-class logits). Change num_classes from 27 to 80. Train on raw k=80 cluster IDs (0-79). At evaluation, map 80 predictions → 19 trainIDs via cluster_to_class LUT from kmeans_centroids.npz. CUPS Table 7b shows overclustering helps — do NOT collapse to 19 classes for training. Same self-supervised losses with proven conservative weights (distill=1.0→0.5, align=0.5, proto=0.05, ent=0.05).

**Tech Stack:** PyTorch, DINOv2 ViT-B/14 (frozen features), SPIdepth depth maps, CSCMRefineNet (CoupledConvBlock)

**Key context:**
- k=80 raw clusters at: `/Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_raw_k80/` (uint8, values 0-79)
- Centroids + mapping: `pseudo_semantic_raw_k80/kmeans_centroids.npz` → `cluster_to_class` (80,) maps to 17 of 19 trainIDs (missing: 6=traffic_light, 17=motorcycle)
- DINOv2 features: `dinov2_features/{train,val}/{city}/{stem}_leftImg8bit.npy` (2048, 768) float16
- Depth: `depth_spidepth/{train,val}/{city}/{stem}.npy` (512, 1024) float32
- CSCMRefineNet model: `mbps_pytorch/refine_net.py`
- Training script: `mbps_pytorch/train_refine_net.py`
- Previous best (on CAUSE-CRF 27-class): PQ=21.87, mIoU=41.78 at epoch 14
- Proven loss weights: `lambda_distill=1.0, lambda_distill_min=0.5, lambda_align=0.5, lambda_proto=0.05, lambda_ent=0.05, label_smoothing=0.1`

---

### Task 1: Generate 19-class mapped labels from k=80 raw clusters

**Files:**
- Use existing: `unsupervised-panoptic-segmentation/pseudo_labels/remap_raw_clusters_to_trainid.py`
- Output: `pseudo_semantic_mapped_k80/{train,val}/{city}/{stem}.png` (uint8, values 0-18 + 255=ignore)

**Step 1: Run remap for train split**

```bash
cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
python unsupervised-panoptic-segmentation/pseudo_labels/remap_raw_clusters_to_trainid.py \
    --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
    --semantic_subdir pseudo_semantic_raw_k80 \
    --centroids_path /Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_raw_k80/kmeans_centroids.npz \
    --split train
```

Expected: Creates `pseudo_semantic_mapped_k80/train/` with 2975 PNGs.

**Step 2: Run remap for val split**

```bash
python unsupervised-panoptic-segmentation/pseudo_labels/remap_raw_clusters_to_trainid.py \
    --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
    --semantic_subdir pseudo_semantic_raw_k80 \
    --centroids_path /Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_raw_k80/kmeans_centroids.npz \
    --split val
```

Expected: Creates `pseudo_semantic_mapped_k80/val/` with 500 PNGs.

**Step 3: Verify output**

```bash
python3 -c "
from PIL import Image; import numpy as np, os
img = np.array(Image.open('/Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_mapped_k80/val/frankfurt/frankfurt_000000_000294.png'))
print('shape:', img.shape, 'dtype:', img.dtype)
print('unique values:', sorted(np.unique(img).tolist()))
print('max valid:', img[img < 255].max())
n_train = sum(len(os.listdir(os.path.join('/Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_mapped_k80/train', c))) for c in os.listdir('/Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_mapped_k80/train') if os.path.isdir(os.path.join('/Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_mapped_k80/train', c)))
n_val = sum(len(os.listdir(os.path.join('/Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_mapped_k80/val', c))) for c in os.listdir('/Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_mapped_k80/val') if os.path.isdir(os.path.join('/Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_mapped_k80/val', c)))
print(f'train: {n_train}, val: {n_val}')
"
```

Expected: values 0-18 (no 6 or 17) + 255, train ~2975, val ~500.

---

### Task 2: Adapt training script for 19-class k=80 labels

**Files:**
- Modify: `mbps_pytorch/train_refine_net.py`

The key changes are:
1. `NUM_CLASSES = 27` → parameterize to support 19
2. `_CAUSE27_TO_TRAINID` mapping in evaluation — skip when already in 19-class space
3. `_load_onehot_semantics` — handle 19-class labels (values 0-18 + 255=ignore)

**Step 1: Modify `PseudoLabelDataset._load_onehot_semantics` to handle 19-class labels**

In `mbps_pytorch/train_refine_net.py`, the current `_load_onehot_semantics` assumes 27-class CAUSE PNGs. When `num_classes=19` and the labels are already in trainID space (0-18 + 255), we need to:
- Use `NUM_CLASSES` from the dataset (not hardcoded 27)
- Handle `255` as ignore (don't include in one-hot)

Edit `PseudoLabelDataset.__init__` to accept `num_classes` parameter:

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
):
    ...
    self.num_classes = num_classes
```

Edit `_load_onehot_semantics` to use `self.num_classes` and handle 255 as ignore:

```python
def _load_onehot_semantics(self, city, stem):
    """Load argmax PNG and convert to smoothed one-hot at patch resolution."""
    sem_path = os.path.join(
        self.root, self.semantic_subdir, self.split, city, f"{stem}.png",
    )
    sem_full = np.array(Image.open(sem_path))  # (1024, 2048) uint8

    # Downsample to patch resolution via nearest neighbor
    sem_pil = Image.fromarray(sem_full)
    sem_patch = np.array(
        sem_pil.resize((PATCH_W, PATCH_H), Image.NEAREST)
    )  # (32, 64)

    nc = self.num_classes
    smooth = 0.1
    onehot = np.zeros((nc, PATCH_H, PATCH_W), dtype=np.float32)
    onehot[:] = smooth / nc
    for c in range(nc):
        mask = sem_patch == c
        onehot[c][mask] = 1.0 - smooth + smooth / nc

    # Pixels with value 255 (ignore) get uniform distribution
    ignore_mask = sem_patch == 255
    if ignore_mask.any():
        onehot[:, ignore_mask] = 1.0 / nc

    return np.log(np.clip(onehot, 1e-7, None))
```

**Step 2: Modify `evaluate_panoptic` to skip the 27→19 mapping when already in 19-class space**

The current `evaluate_panoptic` applies `_CAUSE27_TO_TRAINID` to convert 27-class predictions to 19-class trainIDs. When `num_classes=19`, predictions are already in trainID space — skip the mapping.

Add `num_classes` parameter to `evaluate_panoptic`:

```python
def evaluate_panoptic(model, val_loader, device, cityscapes_root,
                      eval_hw=(512, 1024), cc_min_area=50, num_classes=27):
```

Inside the evaluation loop, replace:
```python
pred_tid_patch = _CAUSE27_TO_TRAINID[pred_27[i]]
```
with:
```python
if num_classes == 27:
    pred_tid_patch = _CAUSE27_TO_TRAINID[pred_27[i]]
else:
    # Already in trainID space (0-18 + 255)
    pred_tid_patch = pred_27[i].astype(np.uint8)
```

**Step 3: Update `train()` to pass `num_classes` through to dataset, model, and evaluation**

In the `train()` function:
- Pass `args.num_classes` to `PseudoLabelDataset` constructor
- Pass `args.num_classes` to `CSCMRefineNet` constructor (it already has the parameter)
- Pass `num_classes` to `evaluate_panoptic`
- Change `NUM_CLASSES` reference in `RefineNetLoss` to not matter (it doesn't use it)

Add CLI argument:
```python
parser.add_argument("--num_classes", type=int, default=27,
                    help="Number of semantic classes (27 for CAUSE, 19 for mapped overclusters)")
```

**Step 4: Test the changes compile and load**

```bash
cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
python3 -c "
from mbps_pytorch.train_refine_net import PseudoLabelDataset
ds = PseudoLabelDataset(
    '/Users/qbit-glitch/Desktop/datasets/cityscapes',
    split='val',
    semantic_subdir='pseudo_semantic_mapped_k80',
    num_classes=19,
)
sample = ds[0]
print('cause_logits shape:', sample['cause_logits'].shape)  # should be (19, 32, 64)
print('dinov2_features shape:', sample['dinov2_features'].shape)  # (768, 32, 64)
print('depth shape:', sample['depth'].shape)  # (1, 32, 64)
"
```

Expected: `cause_logits shape: torch.Size([19, 32, 64])`

**Step 5: Commit**

```bash
git add mbps_pytorch/train_refine_net.py
git commit -m "feat: support 19-class k=80 overclustered labels in CSCMRefineNet training"
```

---

### Task 3: Run CSCMRefineNet training on k=80 mapped labels

**Files:**
- Use: `mbps_pytorch/train_refine_net.py`
- Output: `checkpoints/refine_net_k80/`

**Step 1: Launch training with proven loss weights**

Use the same conservative loss weights that worked for CAUSE-CRF (distill dominant, gentle self-supervised):

```bash
cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
python3 mbps_pytorch/train_refine_net.py \
    --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
    --semantic_subdir pseudo_semantic_mapped_k80 \
    --output_dir checkpoints/refine_net_k80 \
    --num_classes 19 \
    --num_epochs 30 \
    --batch_size 4 \
    --lr 1e-4 \
    --eval_interval 2 \
    --lambda_distill 1.0 \
    --lambda_distill_min 0.5 \
    --lambda_align 0.5 \
    --lambda_proto 0.05 \
    --lambda_ent 0.05 \
    --label_smoothing 0.1 \
    --device auto
```

Expected: ~30 epochs, evaluating every 2 epochs. Each epoch ~3-5 min on MPS. Watch for:
- `changed_pct` should be 3-8% (too low = identity trap, too high = divergence)
- PQ should improve over baseline k=80 PQ=26.74 (at least PQ_stuff)
- mIoU at 19-class level

**Step 2: Monitor training**

Key metrics to watch at each eval checkpoint:
- `PQ` — should be >= 26.74 (the input pseudo-label quality)
- `PQ_stuff` — primary target, currently 32.08, goal 35+
- `PQ_things` — should stay >= 19.41 (don't degrade)
- `changed_pct` — healthy range is 3-8%
- `mIoU` — should be comparable to k=80 baseline

**Step 3: If PQ degrades below input quality (26.74)**

This is the known risk — previous CSCMRefineNet on CAUSE-CRF degraded PQ from 23.1 to 21.87. If this happens:
- Check `changed_pct` — if too high (>15%), reduce self-supervised weights
- Try `lambda_distill_min=0.7` (keep distillation even stronger)
- Try `lambda_align=0.2` (reduce depth-boundary pressure)
- The depth-boundary loss may fight the overclustered boundaries which are already good

---

### Task 4: Evaluate best checkpoint with full panoptic pipeline

**Files:**
- Use: `mbps_pytorch/train_refine_net.py` (built-in eval)
- Optional: generate refined pseudo-labels for Stage-2

**Step 1: Check best checkpoint metrics**

```bash
python3 -c "
import torch
ckpt = torch.load('checkpoints/refine_net_k80/best.pth', weights_only=False, map_location='cpu')
print('Best epoch:', ckpt['epoch'])
print('Metrics:', ckpt['metrics'])
"
```

**Step 2: Compare against baseline**

| Metric | k=80 baseline | CSCMRefineNet k=80 | Delta |
|--------|--------------|-------------------|-------|
| PQ | 26.74 | ? | ? |
| PQ_stuff | 32.08 | ? | ? |
| PQ_things | 19.41 | ? | ? |
| mIoU | ~50% | ? | ? |

**Decision point:** If PQ_stuff improves and PQ_things doesn't degrade significantly, proceed to generate refined pseudo-labels for Stage-2. Otherwise, fall back to Approach B (joint refinement).

---

### Task 5: Generate refined pseudo-labels (if Task 4 shows improvement)

**Files:**
- Modify: `mbps_pytorch/generate_refined_semantics.py` (or create similar script)
- Output: `pseudo_semantic_refined_k80/{train,val}/{city}/{stem}.png`

**Step 1: Generate refined labels for train and val**

Write an inference script that:
1. Loads best CSCMRefineNet k=80 checkpoint
2. For each image: run model on DINOv2 features + depth → 19-class logits → argmax
3. Upsample from 32×64 to 1024×2048 via nearest neighbor
4. Save as uint8 PNG (values 0-18)

```bash
python3 mbps_pytorch/generate_refined_semantics.py \
    --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
    --checkpoint checkpoints/refine_net_k80/best.pth \
    --output_subdir pseudo_semantic_refined_k80 \
    --num_classes 19 \
    --split train

python3 mbps_pytorch/generate_refined_semantics.py \
    --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
    --checkpoint checkpoints/refine_net_k80/best.pth \
    --output_subdir pseudo_semantic_refined_k80 \
    --num_classes 19 \
    --split val
```

**Step 2: Re-evaluate refined labels with depth-guided instances**

Run the panoptic evaluation pipeline on the refined labels + depth instances (τ=0.20, A_min=1000) to get the final PQ with the combined pipeline.

**Step 3: Commit**

```bash
git add mbps_pytorch/generate_refined_semantics.py checkpoints/refine_net_k80/config.json
git commit -m "feat: CSCMRefineNet trained on k=80 overclustered labels"
```
