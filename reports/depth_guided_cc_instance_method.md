# Depth-Guided Connected Component Instance Generation for CUPS Training

**Date**: 2026-04-10
**Script**: `mbps_pytorch/convert_to_cups_format.py --depth_cc_instances`
**Function**: `build_instance_map_depth_cc()`

---

## Problem Statement

CUPS Stage-2 training requires instance pseudo-labels where **every thing-class pixel** has an instance ID. Two existing methods fail to achieve this with depth information:

1. **Plain CC** (`--cc_instances`): Full pixel coverage but **no depth splitting** — adjacent same-class objects at different depths merge into one instance. This is what `cups_pseudo_labels_k80/` uses.

2. **Depth-split instances** (`generate_depth_guided_instances.py`): Uses DepthPro Sobel edges to split at depth boundaries, but only assigns IDs to fragments passing `min_area=1000`, covering **~7% of thing pixels**. CUPS training fails because Cascade Mask R-CNN needs full-coverage instance maps.

| Method | Instance Coverage | Depth Splitting | CUPS Compatible |
|--------|------------------|-----------------|-----------------|
| Plain CC | 100% of thing pixels | No | Yes |
| Depth-split (sparse) | ~7% of thing pixels | Yes (tau=0.01) | No — PQ_things=0 |
| **Depth-guided CC (hybrid)** | **100% of thing pixels** | **Yes (tau=0.01)** | **Yes** |

## Algorithm: Depth-Guided CC (`build_instance_map_depth_cc`)

The hybrid method combines full pixel coverage with depth-guided splitting:

```
Input:
  - semantic: (H, W) uint8, raw k=80 cluster IDs
  - depth: (H, W) float32, DepthPro depth map [0, 1]
  - thing_ids: set of cluster IDs classified as things (15 out of 80)

Step 1: Compute depth edges (once per image)
  depth_smooth = gaussian_filter(depth, sigma=0.0)  # No blur for DepthPro
  gx = sobel(depth_smooth, axis=1)
  gy = sobel(depth_smooth, axis=0)
  grad_mag = sqrt(gx^2 + gy^2)
  depth_edges = (grad_mag > 0.01)  # tau=0.01 optimal for DepthPro

Step 2: For each thing cluster ID:
  a. Get full cluster mask: cls_mask = (semantic == cluster_id)
  b. Remove depth edges: split_mask = cls_mask & ~depth_edges
  c. Connected components on split_mask → each CC is a separate instance
  d. Sort CCs by area (largest first for dilation priority)

Step 3: Assign instance IDs with boundary reclamation
  For each CC (largest first):
    - Dilate CC mask by 3 iterations
    - Claim pixels that belong to this cluster AND are not yet assigned
    - Assign unique instance_id to all claimed pixels

Step 4: Reclaim remaining unassigned thing pixels
  For any thing pixel still unassigned (fell on a depth edge and wasn't
  reclaimed by dilation):
    - Use distance_transform_edt to find nearest assigned instance
    - Assign to that instance

Output:
  instance_map: (H, W) uint16, every thing pixel has instance_id > 0
```

## Key Properties

### Full Coverage Guarantee
Every pixel where `semantic[y,x]` is a thing cluster ID gets an `instance_id > 0`. No thing pixel is left as background. This is critical for CUPS training where the Cascade Mask R-CNN instance heads learn from the instance map.

### Depth Boundary Preservation
Adjacent same-class objects at different depths are split into separate instances. For example, two cars side by side that plain CC would merge are separated by the depth edge between them.

### Comparison: Plain CC vs Depth-Guided CC

For a typical Cityscapes image (`aachen_000000_000019`):

| Property | Plain CC | Depth-Guided CC |
|----------|---------|-----------------|
| Thing pixel coverage | 72,995 px | 73,083 px |
| Number of instances | 26 | 43 |
| Avg instance size | 2,808 px | 1,700 px |
| Adjacent car separation | No (merged) | Yes (split at depth edges) |
| Adjacent person separation | No (merged) | Yes (if different depths) |

The depth-guided method produces **65% more instances** from the same thing pixels, because depth edges split merged objects.

## Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `grad_threshold` | 0.01 | Optimal for DepthPro (from Phase 1 sweep, PQ_things=23.35) |
| `depth_blur_sigma` | 0.0 | DepthPro needs no blur (from Phase 2 sweep) |
| `dilation_iters` | 3 | Reclaims boundary pixels (same as sparse instances) |
| `min_area` | 50 | Low threshold to preserve small instances (CUPS handles filtering) |

## Usage

```bash
python mbps_pytorch/convert_to_cups_format.py \
    --cityscapes_root /path/to/cityscapes \
    --semantic_subdir pseudo_semantic_raw_k80 \
    --output_subdir cups_pseudo_labels_depthpro_v2 \
    --split train \
    --num_classes 80 \
    --depth_cc_instances \
    --depth_subdir depth_depthpro \
    --centroids_path /path/to/pseudo_semantic_raw_k80/kmeans_centroids.npz \
    --grad_threshold 0.01 --depth_blur_sigma 0.0 --dilation_iters 3 \
    --min_instance_area 50
```

## Output Format

CUPS-compatible flat directory with 3 files per image:

| File | Format | Content |
|------|--------|---------|
| `{stem}_leftImg8bit_semantic.png` | uint8, 1024x2048 | Raw k=80 cluster IDs (0-79) |
| `{stem}_leftImg8bit_instance.png` | uint16, 1024x2048 | Instance IDs (0=stuff, 1+=instances) |
| `{stem}_leftImg8bit.pt` | PyTorch dict | `distribution all pixels` + `distribution inside object proposals` |

The distribution tensors enable CUPS to automatically classify clusters as thing/stuff based on the ratio of pixels inside instances (threshold=5%).

## Why This Matters for CUPS Stage-2

CUPS Cascade Mask R-CNN has three instance heads that learn to:
1. **Detect** objects (predict bounding boxes)
2. **Classify** objects (predict semantic class)
3. **Segment** objects (predict instance masks)

All three heads are supervised by the instance map. If the instance map doesn't split adjacent objects, the model learns that merged blobs are single instances — it never learns to separate them. With depth-guided splitting, the training signal teaches the model where object boundaries are, even when the objects share the same semantic class and are adjacent in the image.

## Data Locations

| Resource | Path |
|----------|------|
| v1 labels (CC-only, broken) | `cups_pseudo_labels_depthpro/` |
| **v2 labels (depth-guided CC)** | **`cups_pseudo_labels_depthpro_v2/`** |
| DepthPro depth maps | `depth_depthpro/{train,val}/` |
| k=80 semantics | `pseudo_semantic_raw_k80/{train,val}/` |
| Centroids mapping | `pseudo_semantic_raw_k80/kmeans_centroids.npz` |

## Timeline

- **v1 (broken)**: Used `assemble_cups_pseudo_da3.py` → sparse depth-split instances (7% coverage) → CUPS PQ_things=0
- **v1 fix (wrong)**: Regenerated with `--cc_instances` → full coverage but NO depth → 96.7% identical to k80 baseline
- **v2 (correct)**: `--depth_cc_instances` → full coverage WITH depth splitting → 65% more instances
