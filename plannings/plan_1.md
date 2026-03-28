# Plan: 5 Unsupervised Algorithms to Improve PQ_things

**Date**: 2026-02-19
**Objective**: Increase PQ_things from current best 11.3% (CuVLER multiscale) toward 18%+
**Constraint**: All methods must be fully unsupervised (no GT labels, no supervised backbones like SAM2)

## Current Baseline

| Method | PQ | PQ_stuff | PQ_things | SQ | RQ | Inst/img |
|---|---|---|---|---|---|---|
| CuVLER t=0.35 multiscale (best) | 20.9 | 27.8 | 11.3 | 73.7 | 32.5 | 10.1 |
| Depth-guided (CAUSE-TR + CRF) | 23.1 | 31.4 | 11.7 | 74.3 | 31.2 | ~17 |
| CUPS CVPR 2025 (target) | 27.8 | 35.1 | 17.7 | 57.4 | 35.2 | ? |

**Key bottlenecks**:
- Detector recall: 5.3-10.1 detections/img vs GT 20.2
- 5/8 thing classes have zero or near-zero IoU (rider, train, motorcycle, person, bicycle)
- Person under-segmentation: co-planar pedestrians not split by depth gradients
- Car over-segmentation: intra-object depth variation creates 907 FP vs 732 TP

---

## Environment & Paths

```
Python env:     /Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python
Project root:   /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/
Cityscapes:     /Users/qbit-glitch/Desktop/datasets/cityscapes/
DINOv3 feats:   /Users/qbit-glitch/Desktop/datasets/cityscapes/dinov3_features/{split}/{city}/{stem}.npy
                Shape: (2048, 768) float16 — 32x64 patch grid for 512x1024 input, DINOv3 ViT-B/16
CAUSE semantics:/Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_cause/{split}/{city}/{stem}.png
                1024x2048 uint8, class IDs 0-26 (CAUSE 27-class)
CAUSE+CRF:      /Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_cause_crf/{split}/{city}/{stem}.png
Depth (SPIdepth):/Users/qbit-glitch/Desktop/datasets/cityscapes/depth_spidepth/{split}/{city}/{stem}.npy
Depth (DAv3):   /Users/qbit-glitch/Desktop/datasets/cityscapes/depth_dav3/{split}/{city}/{stem}.npy
CuVLER masks:   /Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_instances_cuvler/{split}/{city}/{stem}.npz
Video sequences:/Users/qbit-glitch/Desktop/datasets/cityscapes/leftImg8bit_sequence/{split}/{city}/
Refs:           /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/refs/
CUPS code:      refs/cups/cups/ (has optical_flow/, scene_flow_2_se3/, pseudo_labels/)
```

**NPZ output format** (all instance scripts must produce this):
```python
np.savez_compressed(path,
    masks=masks,      # (N, H, W) bool — H=512 or 1024, W=1024 or 2048
    scores=scores,    # (N,) float32 — confidence, higher = better
    boxes=boxes,      # (N, 4) float32 — [x1, y1, x2, y2] (optional, can be zeros)
    num_valid=N,      # int
)
```

**Evaluation command** (use for all experiments):
```bash
cd /Users/qbit-glitch/Desktop/datasets/cityscapes

/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python \
  /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/mbps_pytorch/evaluate_cascade_pseudolabels.py \
  --cityscapes_root . --split val --cause27 \
  --semantic_subdir pseudo_semantic_cause \
  --instance_subdir <INSTANCE_DIR_NAME> \
  --thing_mode maskcut \
  --output <EVAL_OUTPUT_NAME>.json
```

---

## Algorithm 1: DINOv2 Intra-Class Feature Clustering

**Priority**: 1 (implement first)
**Expected gain**: PQ_things +3-5 points
**Effort**: 1-2 days
**New script**: `mbps_pytorch/generate_dino_cluster_instances.py`

### 1.1 Core Idea

DINOv2 patch features encode not just "what class" but "which specific object." Two adjacent cars produce different 768-dim feature vectors due to different colors, viewing angles, and occlusion patterns. Within each thing-class region from CAUSE-TR semantics, cluster the DINOv2 features to discover individual object instances.

**Key advantages over depth-guided splitting**:
- Solves co-planar person under-segmentation (different people have different features even at same depth)
- Reduces car over-segmentation (single car has coherent features despite depth variation across its surface)
- No new model needed — DINOv3 ViT-B/16 features already extracted at `dinov3_features/`

### 1.2 Algorithm

```
Input: CAUSE-TR semantic map S (1024x2048, classes 0-26)
       DINOv3 features F (2048x768 = 32x64 patches, float16)
       Depth map D (512x1024, float32)

For each thing class k in {person, rider, car, truck, bus, train, motorcycle, bicycle}:
  1. Get class mask M_k from semantic map S (resize to 512x1024)
  2. Find connected components of M_k → regions R_1, R_2, ...
  3. For each region R_i with area > min_area:
     a. Extract patch indices that overlap with R_i
        - Patches are on a 32x64 grid (stride 16 at 512x1024 resolution)
        - A patch at grid position (r, c) covers pixels [r*16:(r+1)*16, c*16:(c+1)*16]
     b. Get DINOv2 features for those patches → F_i (N_patches x 768)
     c. Get spatial coordinates for those patches → XY_i (N_patches x 2), normalized [0,1]
     d. Get mean depth per patch → D_i (N_patches x 1), normalized [0,1]
     e. Build feature matrix: X_i = concat([F_i, α*XY_i, β*D_i]) — (N_patches x 771)
        - α = spatial weight (try 5.0, 10.0)
        - β = depth weight (try 2.0, 5.0)
     f. Estimate number of clusters:
        - If N_patches < 5: treat as single instance
        - Otherwise: use HDBSCAN(min_cluster_size=3) or
          eigengap heuristic on cosine similarity matrix
     g. Cluster using HDBSCAN or spectral clustering
     h. For each cluster c:
        - Reconstruct pixel mask from patch assignments
        - Smooth with morphological operations (close gaps between patches)
        - Filter by min_area (500 pixels)
        - Compute score = cluster_size / total_region_size
  4. Collect all instances across all classes
  5. Save as NPZ in standard format
```

### 1.3 Implementation Instructions

Create `mbps_pytorch/generate_dino_cluster_instances.py`:

**Required imports**:
```python
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.cluster import HDBSCAN  # or from hdbscan import HDBSCAN
from sklearn.preprocessing import normalize
```

**Key design decisions**:
- Feature resolution: DINOv3 features are 32x64 patches (stride 16 at 512x1024). The semantic map at 1024x2048 must be downsampled to 512x1024 to align with features, or patches mapped to full-res.
- CAUSE 27-class → 19 trainID mapping is needed to identify thing classes. Use `_CAUSE27_TO_TRAINID` from `evaluate_cascade_pseudolabels.py` (lines 48-62).
- Feature normalization: L2-normalize DINOv2 features before clustering (cosine similarity → Euclidean distance).
- The existing `dinov3_features/` files are float16 — cast to float32 for clustering.

**CLI interface** (match existing script patterns):
```
python mbps_pytorch/generate_dino_cluster_instances.py \
    --cityscapes_root /path/to/cityscapes \
    --split val \
    --feature_subdir dinov3_features \
    --semantic_subdir pseudo_semantic_cause \
    --depth_subdir depth_spidepth \
    --output_dir pseudo_instances_dino_cluster \
    --spatial_weight 5.0 \
    --depth_weight 2.0 \
    --min_area 500 \
    --min_cluster_size 3 \
    --cause27 \
    --visualize \
    --limit 10
```

**Output directory structure** (must match existing pattern):
```
pseudo_instances_dino_cluster/
  val/
    frankfurt/
      frankfurt_000000_000294_leftImg8bit.npz
      ...
    lindau/
    munster/
  stats_val.json
```

**Stuff-things mapping for CAUSE 27-class**:
- CAUSE27 IDs for thing classes (after Hungarian matching): need to map CAUSE27 → trainID19 → check if trainID in {11-18}
- Use the same `_CAUSE27_TO_TRAINID` table from `evaluate_cascade_pseudolabels.py`
- Thing trainIDs: 11=person, 12=rider, 13=car, 14=truck, 15=bus, 16=train, 17=motorcycle, 18=bicycle

### 1.4 Evaluation

```bash
cd /Users/qbit-glitch/Desktop/datasets/cityscapes

# Quick test (3 images)
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python \
  /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/mbps_pytorch/generate_dino_cluster_instances.py \
  --cityscapes_root . --split val --cause27 \
  --feature_subdir dinov3_features \
  --semantic_subdir pseudo_semantic_cause \
  --depth_subdir depth_spidepth \
  --output_dir pseudo_instances_dino_cluster \
  --limit 3 --visualize

# Full val set
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python \
  /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/mbps_pytorch/generate_dino_cluster_instances.py \
  --cityscapes_root . --split val --cause27 \
  --feature_subdir dinov3_features \
  --semantic_subdir pseudo_semantic_cause \
  --depth_subdir depth_spidepth \
  --output_dir pseudo_instances_dino_cluster

# Evaluate
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python \
  /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/mbps_pytorch/evaluate_cascade_pseudolabels.py \
  --cityscapes_root . --split val --cause27 \
  --semantic_subdir pseudo_semantic_cause \
  --instance_subdir pseudo_instances_dino_cluster \
  --thing_mode maskcut \
  --output eval_cause_dino_cluster_val.json
```

### 1.5 Parameter Sweep

Run grid search over:
- `spatial_weight`: [2.0, 5.0, 10.0, 20.0]
- `depth_weight`: [0.0, 2.0, 5.0, 10.0]
- `min_cluster_size`: [2, 3, 5]
- `min_area`: [200, 500, 1000]

### 1.6 Dependencies to Install

```bash
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/pip install hdbscan scikit-learn
```

---

## Algorithm 2: Self-Supervised Optical Flow Instance Splitting

**Priority**: 2 (highest ceiling, medium effort)
**Expected gain**: PQ_things +3-6 points
**Effort**: 2-3 days
**New script**: `mbps_pytorch/generate_flow_instances.py`

### 2.1 Core Idea

Cityscapes has video sequences (30 frames per snippet at `leftImg8bit_sequence/`). Different objects move differently — even two cars at the same depth have different velocities if one is parked and one is moving. Self-supervised optical flow models (RAFT pretrained on synthetic data, or SMURF from CUPS) compute dense pixel correspondences between consecutive frames. Within each thing-class region, cluster pixels by flow vector similarity to discover individual instances.

This is fundamentally what makes CUPS achieve PQ_things=17.7 — their SF2SE3 module clusters scene flow into rigid SE(3) motions.

**Key advantages**:
- Solves co-planar person problem (walking person has different flow from standing person)
- Detects rider/motorcycle/bicycle (moving objects have distinct flow from static background)
- Complementary to depth — depth separates objects at different distances, flow separates objects with different velocities

### 2.2 Algorithm — Two Variants

#### Variant A: Simple Flow-Magnitude Thresholding (Quick, 1 day)

```
Input: Annotated frame I_t (1024x2048)
       Adjacent frame I_{t+1} from leftImg8bit_sequence/
       CAUSE-TR semantic map S

1. Compute optical flow F = RAFT(I_t, I_{t+1}) → (2, H, W) flow field [dx, dy]
2. Compute flow magnitude M = sqrt(dx^2 + dy^2)
3. For each thing class k:
   a. Get class mask M_k from S
   b. Compute "moving" mask: M_k & (M > flow_thresh)
   c. Connected components on moving mask → moving instances
   d. Remaining M_k pixels → "static" instances via CC
4. Save as NPZ
```

This is a simplified version that just separates moving vs. static objects within each class. Quick to implement but doesn't separate multiple moving objects at different speeds.

#### Variant B: Flow-Vector Clustering (Full, 2-3 days)

```
Input: Annotated frame I_t (1024x2048)
       Adjacent frame I_{t+1} from leftImg8bit_sequence/
       CAUSE-TR semantic map S
       Depth map D (optional, for scene flow)

1. Compute forward/backward optical flow:
   F_fwd = RAFT(I_t, I_{t+1})    → (2, H, W)
   F_bwd = RAFT(I_{t+1}, I_t)    → (2, H, W)
2. Forward-backward consistency check:
   valid = ||F_fwd(p) + F_bwd(p + F_fwd(p))|| < ε
3. For each thing class k:
   a. Get class mask M_k from S
   b. Extract flow vectors for all valid pixels in M_k
   c. Build feature: [flow_dx, flow_dy, normalized_x, normalized_y, depth]
   d. Cluster using HDBSCAN → instance groups
   e. Each cluster = one instance (with contiguous pixel mask)
4. Save as NPZ
```

#### Variant C: SF2SE3 (Full CUPS approach, 3-5 days)

Replicate CUPS's approach using their code at `refs/cups/cups/scene_flow_2_se3/`:

```
1. Compute stereo disparity (or use monocular depth as proxy)
2. Compute forward/backward optical flow
3. Construct scene flow from optical flow + disparity
4. Run SF2SE3 RANSAC clustering → rigid SE(3) motion groups
5. Each motion group = one instance
6. Ensemble: run N=5 times, keep masks in ≥80% of runs
```

### 2.3 Implementation Instructions

**Step 1: Find adjacent frames**

Cityscapes `leftImg8bit_sequence/` has the same naming convention as `leftImg8bit/`. The annotated frames in `leftImg8bit/` correspond to specific frames in the sequence. The sequence directory contains ALL frames from the 30-frame snippets (the annotated frame is frame 19, with frames 0-29 at frame indices spaced by ~1/17s).

The frame ID is the second number in the filename: `{city}_{seq}_{frame}_leftImg8bit.png`

To find adjacent frame for `frankfurt_000000_000294_leftImg8bit.png`:
- Same city and sequence: `frankfurt_000000_*`
- Frame ID 000294 → look for frame 000293 or 000295 in `leftImg8bit_sequence/`
- If exact adjacent doesn't exist, use nearest available frame

Actually, looking at the data, `leftImg8bit_sequence/val/frankfurt/` has 267 files — same count as `leftImg8bit/val/frankfurt/` (267). So these ARE the annotated frames only, not the full sequences. Need to check if the full sequence data is downloaded.

**IMPORTANT**: Check if full 30-frame sequences exist. If only annotated frames are available, you need to download `leftImg8bit_sequence_trainvaltest.zip` from Cityscapes (requires login):
```bash
# Check if we have adjacent frames
ls /Users/qbit-glitch/Desktop/datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/ | sort | head -3
# Compare with annotated frames
ls /Users/qbit-glitch/Desktop/datasets/cityscapes/leftImg8bit/val/frankfurt/ | sort | head -3
# If they're identical, we need the full sequence package
```

**Step 2: Optical flow model**

Use torchvision's RAFT (pretrained on FlyingThings3D synthetic data — no real GT labels):
```python
import torchvision.models.optical_flow as flow_models
raft = flow_models.raft_large(weights=flow_models.Raft_Large_Weights.DEFAULT)
raft.eval()
```

Or use CUPS's SMURF wrapper at `refs/cups/cups/optical_flow/raft.py`.

**Step 3: Create the script**

`mbps_pytorch/generate_flow_instances.py`:

```
CLI args:
  --cityscapes_root     Path to cityscapes
  --split               val or train
  --sequence_subdir     leftImg8bit_sequence (for adjacent frames)
  --semantic_subdir     pseudo_semantic_cause
  --depth_subdir        depth_spidepth (optional, for variant B/C)
  --output_dir          pseudo_instances_flow
  --flow_model          raft_large (default)
  --variant             simple|cluster|sf2se3
  --flow_thresh         Flow magnitude threshold for variant A (default: 2.0)
  --min_area            Minimum instance area (default: 500)
  --cause27             Flag for CAUSE 27-class mapping
  --device              cpu/mps/cuda
  --visualize           Save flow + instance overlays
  --limit               Process first N images only
```

### 2.4 Evaluation

```bash
cd /Users/qbit-glitch/Desktop/datasets/cityscapes

# Generate flow instances (simple variant first)
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python \
  /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/mbps_pytorch/generate_flow_instances.py \
  --cityscapes_root . --split val --cause27 \
  --semantic_subdir pseudo_semantic_cause \
  --output_dir pseudo_instances_flow_simple \
  --variant simple --flow_thresh 2.0 \
  --limit 10 --visualize

# Evaluate
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python \
  /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/mbps_pytorch/evaluate_cascade_pseudolabels.py \
  --cityscapes_root . --split val --cause27 \
  --semantic_subdir pseudo_semantic_cause \
  --instance_subdir pseudo_instances_flow_simple \
  --thing_mode maskcut \
  --output eval_cause_flow_simple_val.json
```

### 2.5 Dependencies

```bash
# torchvision RAFT should already be available
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python -c "import torchvision.models.optical_flow; print('RAFT available')"
```

### 2.6 Critical Note on Video Data

If `leftImg8bit_sequence/` only contains the annotated frames (no adjacent frames), this algorithm CANNOT run. You must either:
1. Download the full sequence package from Cityscapes website, OR
2. Use the CUPS code which expects `CityscapesStereoVideo` dataset format with specific directory structure

Check first by comparing file counts and frame IDs between `leftImg8bit/` and `leftImg8bit_sequence/`.

---

## Algorithm 3: DINOSAUR Slot Attention on DINOv2 Features

**Priority**: 3 (most novel for NeurIPS paper, requires training)
**Expected gain**: PQ_things +4-8 points
**Effort**: 3-5 days
**New scripts**: `mbps_pytorch/train_dinosaur.py`, `mbps_pytorch/generate_dinosaur_instances.py`

### 3.1 Core Idea

DINOSAUR (ICLR 2023) trains slot attention on frozen DINOv2 features to decompose scenes into object "slots." Each slot learns to bind to one object. The training objective is purely unsupervised — reconstruct the DINOv2 features from the slot representations. No GT labels of any kind are needed.

After training, each image is decomposed into K slots, each with an attention mask showing which pixels belong to that slot. These masks directly serve as instance segmentation.

**Key advantages**:
- Learns an object-level prior — discovers that scenes decompose into discrete entities
- Handles occlusion, co-planarity, appearance similarity by learning compositional structure
- The most theoretically novel approach for a top-venue paper
- Fully unsupervised training on Cityscapes data

### 3.2 Algorithm

#### Training Phase

```
Input: DINOv2 features for all Cityscapes training images
       F_i ∈ R^{N_patches x D} for each image i (N_patches=2048, D=768)

Architecture:
  Encoder: Linear(768, 256) — project DINOv2 features to slot dim
  Slot Attention: K=20 slots, D_slot=256, T=3 iterations
    - slots_init: K learnable slot embeddings (K x D_slot)
    - At each iteration t:
      1. Compute attention: A = softmax(Q·K^T / sqrt(D_slot))
         Q = slot queries, K = encoded features
      2. Update slots: slot_i += GRU(slot_i, weighted_sum(V, A))
  Decoder: Linear(256, 768) — reconstruct features from slots

Loss: L = MSE(Decoder(slots) @ attention_masks, F_original)
      + entropy regularization on attention maps (encourage sharp masks)

Training: ~200 epochs on Cityscapes train set (2975 images)
          Adam, lr=4e-4, batch_size=32 (patch sequences, not full images)
          ~2-4 hours on MPS (features are pre-extracted, no backbone forward pass)
```

#### Inference Phase

```
Input: DINOv2 features F (2048x768) for one image
       CAUSE-TR semantic map S (1024x2048)

1. Project features: F' = Linear(F) → (2048, 256)
2. Run slot attention: slots, attn_masks = SlotAttention(F')
   → attn_masks: (K, 2048) — attention weight per slot per patch
3. For each slot k:
   a. Reshape attention to spatial: A_k (32x64)
   b. Binarize: mask_k = (A_k > threshold) — e.g., threshold=0.5
   c. Upsample to 512x1024 via nearest-neighbor
   d. Compute score = max(A_k) * mask_area (confidence × coverage)
4. Assign semantic class via majority vote from CAUSE-TR:
   class_k = mode(S[mask_k])
5. Keep only thing-class instances, filter by min_area
6. Save as NPZ
```

### 3.3 Implementation Instructions

**Step 1: Implement Slot Attention module**

Create `mbps_pytorch/train_dinosaur.py`:

```python
# Key components to implement:

class SlotAttention(nn.Module):
    """Slot Attention module (Locatello et al., 2020)."""
    def __init__(self, num_slots=20, dim=256, iters=3, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.dim = dim
        self.eps = eps
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, num_slots, dim)))
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_inputs = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*2), nn.ReLU(), nn.Linear(dim*2, dim))

    def forward(self, inputs):
        # inputs: (B, N, D)
        B, N, D = inputs.shape
        mu = self.slots_mu.expand(B, -1, -1)
        sigma = self.slots_sigma.expand(B, -1, -1)
        slots = mu + sigma * torch.randn_like(mu)

        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs)  # (B, N, D)
        v = self.to_v(inputs)  # (B, N, D)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)  # (B, K, D)
            attn = torch.einsum('bkd,bnd->bkn', q, k) / (D ** 0.5)
            attn = attn.softmax(dim=1) + self.eps  # (B, K, N) — softmax over slots
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bkn,bnd->bkd', attn, v)
            slots = self.gru(updates.reshape(-1, D), slots_prev.reshape(-1, D))
            slots = slots.reshape(B, -1, D)
            slots = slots + self.mlp(slots)

        # Final attention for mask extraction
        slots = self.norm_slots(slots)
        q = self.to_q(slots)
        attn = torch.einsum('bkd,bnd->bkn', q, k) / (D ** 0.5)
        attn = attn.softmax(dim=1)  # (B, K, N) — normalize over slots (competition)

        return slots, attn
```

**Step 2: Training loop**

```python
class DINOSAUR(nn.Module):
    def __init__(self, input_dim=768, slot_dim=256, num_slots=20):
        super().__init__()
        self.encoder = nn.Linear(input_dim, slot_dim)
        self.slot_attention = SlotAttention(num_slots, slot_dim)
        self.decoder = nn.Linear(slot_dim, input_dim)

    def forward(self, features):
        # features: (B, N, 768)
        x = self.encoder(features)           # (B, N, 256)
        slots, attn = self.slot_attention(x)  # slots: (B, K, 256), attn: (B, K, N)
        # Decode each slot
        decoded = self.decoder(slots)         # (B, K, 768)
        # Reconstruct: weighted combination of decoded slots
        recon = torch.einsum('bkn,bkd->bnd', attn, decoded)  # (B, N, 768)
        return recon, attn, slots

# Loss: MSE reconstruction + entropy regularization
loss = F.mse_loss(recon, features) + lambda_ent * entropy_reg(attn)
```

**Step 3: Dataset**

```python
class CityscapesDINOFeatureDataset(Dataset):
    """Load pre-extracted DINOv3 features."""
    def __init__(self, feature_dir):
        self.files = sorted(Path(feature_dir).rglob("*.npy"))

    def __getitem__(self, idx):
        features = np.load(self.files[idx]).astype(np.float32)  # (2048, 768)
        return torch.from_numpy(features)
```

**CLI interface for training**:
```
python mbps_pytorch/train_dinosaur.py \
    --feature_dir /path/to/cityscapes/dinov3_features/train \
    --output_dir checkpoints/dinosaur/ \
    --num_slots 20 \
    --slot_dim 256 \
    --epochs 200 \
    --batch_size 32 \
    --lr 4e-4 \
    --device mps
```

**CLI interface for inference**:
```
python mbps_pytorch/generate_dinosaur_instances.py \
    --cityscapes_root /path/to/cityscapes \
    --split val \
    --feature_subdir dinov3_features \
    --semantic_subdir pseudo_semantic_cause \
    --checkpoint checkpoints/dinosaur/best.pth \
    --output_dir pseudo_instances_dinosaur \
    --num_slots 20 \
    --attn_threshold 0.5 \
    --min_area 500 \
    --cause27 \
    --visualize \
    --limit 10
```

### 3.4 Hyperparameter Search

Key hyperparameters to sweep:
- `num_slots`: [10, 15, 20, 30] — more slots = more instances, risk of over-segmentation
- `slot_dim`: [128, 256] — capacity per slot
- `iters`: [3, 5] — slot attention iterations
- `attn_threshold`: [0.3, 0.5, 0.7] — binarization threshold for mask extraction
- `entropy_weight`: [0.0, 0.01, 0.1] — encourages sharp slot assignments

### 3.5 Evaluation

```bash
cd /Users/qbit-glitch/Desktop/datasets/cityscapes

/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python \
  /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/mbps_pytorch/evaluate_cascade_pseudolabels.py \
  --cityscapes_root . --split val --cause27 \
  --semantic_subdir pseudo_semantic_cause \
  --instance_subdir pseudo_instances_dinosaur \
  --thing_mode maskcut \
  --output eval_cause_dinosaur_val.json
```

### 3.6 Dependencies

```bash
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/pip install torch  # already installed
# No additional dependencies — pure PyTorch implementation
```

### 3.7 References

- Seitzer et al., "Bridging the Gap to Real-World Object-Centric Learning" (ICLR 2023) — DINOSAUR
- Locatello et al., "Object-Centric Learning with Slot Attention" (NeurIPS 2020) — Slot Attention

---

## Algorithm 4: VoteCut on Cityscapes (Domain-Adapted Discovery)

**Priority**: 4 (medium effort, addresses domain gap)
**Expected gain**: PQ_things +2-6 points
**Effort**: 2-4 days
**New script**: `mbps_pytorch/generate_votecut_instances.py`

### 4.1 Core Idea

CuVLER's VoteCut discovers objects by running Normalized Cuts on features from 6 different self-supervised ViTs and aggregating votes. Currently VoteCut is run on ImageNet (object-centric), and the resulting pseudo-masks train a Cascade Mask R-CNN. The domain gap from ImageNet → Cityscapes is severe (5.3 detections vs 20.2 GT).

**Novel approach**: Run VoteCut directly on Cityscapes images. This produces domain-specific pseudo-masks that capture driving-scene objects. Then either:
- (a) Use VoteCut masks directly as instance proposals, or
- (b) Self-train a Cascade Mask R-CNN on VoteCut-Cityscapes masks (domain-specific CuVLER)

### 4.2 Algorithm

```
Input: Cityscapes image I (1024x2048)
       N self-supervised ViTs: DINOv1-B/8, DINOv2-B/14, MAE-B/16, iBOT-B/16, ...

For each ViT model m in {1..N}:
  1. Extract patch features F_m (H_p x W_p x D_m)
  2. Compute affinity matrix W_m = F_m @ F_m^T (cosine similarity)
  3. Run Normalized Cut on W_m:
     - Compute degree matrix D = diag(W1)
     - Solve generalized eigenvalue: (D - W)v = λDv
     - Take Fiedler vector (2nd smallest eigenvalue)
     - Binarize by sign → foreground/background mask
  4. If foreground has multiple connected components, keep largest
  5. Recursively apply NCut to foreground (up to 3 levels)
  → Produces K_m masks per image from model m

Pixel Voting:
  1. For each pixel p, count how many models assign it to an object mask
  2. Vote map V(p) = (1/N) * sum_m(is_object(p, m))
  3. Threshold: object pixels = V(p) > 0.5 (majority vote)
  4. Connected components on voted object mask → instance proposals
  5. Score each proposal by average vote confidence

Save as NPZ.
```

### 4.3 Implementation Instructions

**Step 1: Install self-supervised ViT models**

CuVLER uses 6 models. For efficiency, start with 3:
```python
# DINOv1 ViT-B/8 (from torch hub)
dino_v1 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')

# DINOv2 ViT-B/14 (already used by CAUSE-TR)
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

# MAE ViT-B/16 (from torch hub)
mae = # load from HuggingFace or timm
```

**Step 2: NCut implementation**

Reference CuVLER's NCut at `refs/cuvler/` or implement directly:
```python
from scipy.sparse.linalg import eigsh

def normalized_cut(affinity, num_eig=2):
    """Compute Normalized Cut on affinity matrix."""
    D = np.diag(affinity.sum(axis=1))
    L = D - affinity  # Laplacian
    # Solve generalized eigenvalue problem
    eigenvalues, eigenvectors = eigsh(L, k=num_eig, M=D, which='SM')
    # Fiedler vector is second eigenvector
    fiedler = eigenvectors[:, 1]
    # Binarize by sign
    mask = fiedler > 0
    return mask
```

**Step 3: Create script**

`mbps_pytorch/generate_votecut_instances.py`:

```
CLI args:
  --cityscapes_root       Path to cityscapes
  --split                 val or train
  --output_dir            pseudo_instances_votecut
  --models                dino_v1,dinov2,mae (comma-separated)
  --ncut_depth            Max recursion depth for NCut (default: 3)
  --vote_threshold        Minimum vote fraction (default: 0.5)
  --min_area              Minimum mask area (default: 500)
  --image_size            Resize shorter side (default: 480)
  --device                mps/cpu
  --visualize
  --limit
```

**Step 4 (optional): Self-train detector on VoteCut masks**

If VoteCut masks are promising, convert them to COCO format and fine-tune CuVLER:
```python
# Convert NPZ masks to COCO-format JSON
# Then use detectron2 training (requires CUDA GPU / TPU VM)
```

### 4.4 Evaluation

```bash
cd /Users/qbit-glitch/Desktop/datasets/cityscapes

/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python \
  /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/mbps_pytorch/generate_votecut_instances.py \
  --cityscapes_root . --split val \
  --output_dir pseudo_instances_votecut \
  --models dino_v1,dinov2 \
  --limit 10 --visualize

/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python \
  /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/mbps_pytorch/evaluate_cascade_pseudolabels.py \
  --cityscapes_root . --split val --cause27 \
  --semantic_subdir pseudo_semantic_cause \
  --instance_subdir pseudo_instances_votecut \
  --thing_mode maskcut \
  --output eval_cause_votecut_val.json
```

### 4.5 Computational Cost

VoteCut is expensive: NCut eigendecomposition on a 2048x2048 affinity matrix takes ~2-5s per model per image. With 3 models and 500 val images: ~1-2 hours on MPS. With 6 models: ~3-5 hours.

**Optimization**: Downsample images to 480x960 (fewer patches → smaller affinity matrix). Use sparse eigensolvers.

### 4.6 Dependencies

```bash
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/pip install timm  # for MAE models
# torch hub models download automatically
```

---

## Algorithm 5: Adaptive Depth-Layered Decomposition

**Priority**: 5 (simplest, quick improvement over gradient thresholding)
**Expected gain**: PQ_things +1-3 points
**Effort**: 0.5-1 day
**New script**: `mbps_pytorch/generate_depth_layer_instances.py`

### 5.1 Core Idea

Replace brittle gradient thresholding (single global τ) with principled depth quantization. Instead of detecting edges in continuous depth, discretize depth into adaptive layers and run connected components per (class, layer) pair.

**Key insight**: One global threshold τ can't handle both close-range (where 0.5m depth difference matters) and far-range (where 5m depth difference matters). Adaptive quantile-based depth bins are naturally denser where more objects cluster (near range) and sparser at far range.

### 5.2 Algorithm

```
Input: CAUSE-TR semantic map S (1024x2048, classes 0-26)
       Depth map D (512x1024, float32, min-max normalized to [0,1])

1. Compute adaptive depth bins:
   - Extract all valid depth values from thing-class pixels
   - Compute K equal-mass quantiles (K=15-20)
   - Bin edges: q_0=0, q_1, q_2, ..., q_K=1
   This ensures each bin contains roughly the same number of thing pixels,
   giving finer resolution where objects are dense.

2. Assign each pixel to a depth bin:
   bin_map[p] = digitize(D[p], bin_edges)

3. For each thing class k:
   For each depth bin b:
     a. Get mask: M_{k,b} = (S == k) & (bin_map == b)
     b. Connected components on M_{k,b} → instances
     c. Filter by min_area

4. Merge across adjacent bins:
   For each pair of instances (i in bin b, j in bin b+1):
     - If same class AND spatially adjacent (dilated masks overlap)
       AND DINOv2 feature similarity > merge_threshold:
       → merge into single instance

5. Post-process:
   - Morphological closing to fill small holes
   - Remove instances smaller than min_area
   - Score = area / max_area_in_class

6. Save as NPZ
```

### 5.3 Implementation Instructions

Create `mbps_pytorch/generate_depth_layer_instances.py`:

```python
# Key function
def depth_layer_instances(semantic, depth, thing_ids,
                          num_bins=15, min_area=500,
                          merge_threshold=0.8):
    """Split thing regions using adaptive depth binning."""
    # 1. Compute adaptive bins from thing-class depth values
    thing_mask = np.isin(semantic, list(thing_ids))
    thing_depths = depth[thing_mask]
    if len(thing_depths) < 100:
        return []
    bin_edges = np.quantile(thing_depths, np.linspace(0, 1, num_bins + 1))
    bin_edges = np.unique(bin_edges)  # remove duplicates

    # 2. Digitize depth into bins
    bin_map = np.digitize(depth, bin_edges) - 1  # 0-indexed

    # 3. Per (class, bin) connected components
    instances = []
    for cls in sorted(thing_ids):
        cls_mask = semantic == cls
        if cls_mask.sum() < min_area:
            continue
        for b in range(len(bin_edges) - 1):
            region = cls_mask & (bin_map == b)
            if region.sum() < min_area:
                continue
            labeled, n_cc = ndimage.label(region)
            for cc_id in range(1, n_cc + 1):
                cc_mask = labeled == cc_id
                if cc_mask.sum() >= min_area:
                    instances.append((cc_mask, cls, float(cc_mask.sum())))

    # 4. Optional: merge across adjacent bins (union-find)
    # ... (see merge step in algorithm)

    return instances
```

**CLI interface**:
```
python mbps_pytorch/generate_depth_layer_instances.py \
    --cityscapes_root /path/to/cityscapes \
    --split val \
    --semantic_subdir pseudo_semantic_cause \
    --depth_subdir depth_spidepth \
    --output_dir pseudo_instances_depth_layer \
    --num_bins 15 \
    --min_area 500 \
    --cause27 \
    --merge_adjacent \
    --visualize \
    --limit 10
```

### 5.4 Evaluation

```bash
cd /Users/qbit-glitch/Desktop/datasets/cityscapes

# Generate
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python \
  /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/mbps_pytorch/generate_depth_layer_instances.py \
  --cityscapes_root . --split val --cause27 \
  --semantic_subdir pseudo_semantic_cause \
  --depth_subdir depth_spidepth \
  --output_dir pseudo_instances_depth_layer \
  --num_bins 15 --min_area 500

# Evaluate
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python \
  /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/mbps_pytorch/evaluate_cascade_pseudolabels.py \
  --cityscapes_root . --split val --cause27 \
  --semantic_subdir pseudo_semantic_cause \
  --instance_subdir pseudo_instances_depth_layer \
  --thing_mode maskcut \
  --output eval_cause_depth_layer_val.json
```

### 5.5 Parameter Sweep

```
num_bins: [5, 10, 15, 20, 30]
min_area: [200, 500, 1000]
merge_adjacent: [true, false]
```

### 5.6 Dependencies

None — pure NumPy/SciPy (already available).

---

## Combination Strategy

After implementing all 5 algorithms individually, combine the best performers:

### Ensemble via `merge_instance_sources.py`

The existing `mbps_pytorch/merge_instance_sources.py` merges NPZ files from multiple sources with NMS deduplication:

```bash
# Example: combine DINOv2 clustering + flow instances
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python \
  /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/mbps_pytorch/merge_instance_sources.py \
  --sources \
    pseudo_instances_dino_cluster/val \
    pseudo_instances_flow_simple/val \
  --output pseudo_instances_combined/val \
  --nms_iou 0.3

# Evaluate combined
/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python \
  /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/mbps_pytorch/evaluate_cascade_pseudolabels.py \
  --cityscapes_root . --split val --cause27 \
  --semantic_subdir pseudo_semantic_cause \
  --instance_subdir pseudo_instances_combined \
  --thing_mode maskcut \
  --output eval_cause_combined_val.json
```

### Recommended Combination Priority

```
Step 1: Algorithm 1 (DINOv2 clustering) alone → baseline improvement
Step 2: Algorithm 5 (depth layers) alone → compare with current depth-guided
Step 3: Ensemble Algorithm 1 + best of {current CuVLER, Algorithm 5}
Step 4: Algorithm 2 (flow) alone → if video data available
Step 5: Ensemble Algorithm 1 + Algorithm 2 (features + motion)
Step 6: Algorithm 3 (DINOSAUR) alone → if results from 1-5 are insufficient
Step 7: Best ensemble from all available sources
```

---

## Success Criteria

| Milestone | PQ_things | PQ | Status |
|---|---|---|---|
| Current best (CuVLER multiscale) | 11.3 | 20.9 | Done |
| Depth-guided + CRF best | 11.7 | 23.1 | Done |
| Target: beat CUPS | 17.7+ | 27.8+ | Target |
| Intermediate target | 15.0 | 24.0 | Milestone |
| Minimum viable improvement | 13.0 | 22.0 | Milestone |

## Quick Reference: What to Ask Claude Code

**To implement Algorithm 1 (DINOv2 clustering)**:
> "Implement Algorithm 1 from plannings/plan_1.md — create generate_dino_cluster_instances.py that clusters DINOv2 features within CAUSE-TR thing-class regions to produce instance masks. Follow the NPZ output format and CLI interface specified in the plan."

**To implement Algorithm 2 (optical flow)**:
> "Implement Algorithm 2 Variant A (simple flow-magnitude thresholding) from plannings/plan_1.md — create generate_flow_instances.py using torchvision RAFT to compute optical flow between consecutive Cityscapes frames and split thing regions by motion. Check first if leftImg8bit_sequence has adjacent frames."

**To implement Algorithm 3 (DINOSAUR slot attention)**:
> "Implement Algorithm 3 from plannings/plan_1.md — create train_dinosaur.py and generate_dinosaur_instances.py implementing slot attention on pre-extracted DINOv3 features. The training uses MSE reconstruction loss. Follow the architecture spec in section 3.3."

**To implement Algorithm 4 (VoteCut on Cityscapes)**:
> "Implement Algorithm 4 from plannings/plan_1.md — create generate_votecut_instances.py that runs Normalized Cuts on multiple self-supervised ViT features and aggregates votes. Start with 2 models (DINOv1-B/8 and DINOv2-B/14)."

**To implement Algorithm 5 (adaptive depth layers)**:
> "Implement Algorithm 5 from plannings/plan_1.md — create generate_depth_layer_instances.py that replaces gradient thresholding with adaptive depth quantile binning. This is the simplest algorithm — pure NumPy/SciPy."

**To run ensemble combination**:
> "Run the ensemble combination from plannings/plan_1.md — merge the best instance sources using merge_instance_sources.py with NMS=0.3 and evaluate."






Use CuVLER instances as the base, supplement with global feature clustering for objects CuVLER misses.