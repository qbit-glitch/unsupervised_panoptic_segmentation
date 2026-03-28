"""Post-process depth-guided instances: merge small adjacent same-class fragments.

Depth-guided instance segmentation over-fragments objects when depth gradients
create spurious splits within a single object. This script merges fragments
that are:
  1. Same semantic class (from pseudo-labels)
  2. Spatially adjacent (touching in 4-connected sense)
  3. Have similar mean depth (below merge_depth_delta threshold)

Usage:
    python mbps_pytorch/merge_depth_instances.py \
        --instance_dir .../sweep_instances/gt0.10_ma500/val \
        --semantic_dir .../pseudo_semantic_cause_trainid/val \
        --depth_dir .../depth_spidepth/val \
        --output_dir .../sweep_instances/gt0.10_ma500_merged/val \
        --stuff_things .../stuff_things_attention.json \
        --merge_depth_delta 0.05
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WORK_H, WORK_W = 512, 1024
DEFAULT_THING_IDS = {11, 12, 13, 14, 15, 16, 17, 18}


def load_thing_ids(stuff_things_path):
    with open(stuff_things_path) as f:
        data = json.load(f)
    return set(data["thing_ids"])


def load_instances(npz_path):
    """Load instance masks from NPZ."""
    data = np.load(str(npz_path))
    masks = data["masks"]  # (M, H*W) bool
    scores = data["scores"]
    num_valid = int(data["num_valid"])
    h = int(data["h_patches"])
    w = int(data["w_patches"])
    masks = masks[:num_valid]
    scores = scores[:num_valid]
    return masks, scores, h, w


def save_instances(masks, scores, output_path, h=WORK_H, w=WORK_W):
    """Save merged instances in same NPZ format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = masks.shape[0] if masks.ndim == 2 else 0
    np.savez_compressed(
        str(output_path),
        masks=masks,
        scores=scores,
        num_valid=n,
        h_patches=h,
        w_patches=w,
    )

    # Visualization
    vis = np.zeros((h, w), dtype=np.uint16)
    for i in range(n):
        m = masks[i].reshape(h, w)
        vis[m] = i + 1
    vis_path = str(output_path).replace(".npz", "_instance.png")
    Image.fromarray(vis).save(vis_path)


def merge_instances(masks, scores, semantic, depth, thing_ids,
                    merge_depth_delta=0.05, min_area=50):
    """Merge adjacent same-class fragments with similar depth.

    Algorithm:
      1. Build instance-to-class mapping via majority vote from semantic map
      2. Build spatial adjacency graph between instances (dilate each mask by 1px,
         check overlap with other masks)
      3. For each pair of adjacent same-class instances:
         - Compute mean depth difference
         - If delta < merge_depth_delta, mark for merging
      4. Use union-find to group transitive merges
      5. Merge groups: combine masks, keep max score, remove duplicates

    Returns:
        merged_masks: (M', H*W) bool
        merged_scores: (M',) float
    """
    h, w = semantic.shape
    n = masks.shape[0]

    if n == 0:
        return masks, scores

    # 1. Assign class to each instance via majority vote
    inst_classes = np.zeros(n, dtype=int)
    inst_depths = np.zeros(n, dtype=float)
    inst_areas = np.zeros(n, dtype=int)

    for i in range(n):
        m = masks[i].reshape(h, w)
        sem_vals = semantic[m]
        if len(sem_vals) == 0:
            inst_classes[i] = 255
            continue
        inst_classes[i] = np.bincount(sem_vals, minlength=256).argmax()
        inst_depths[i] = depth[m].mean()
        inst_areas[i] = m.sum()

    # Only process thing instances
    thing_mask = np.array([c in thing_ids for c in inst_classes])
    if thing_mask.sum() == 0:
        return masks, scores

    # 2. Build adjacency graph using dilated overlap
    # Create instance label map for efficient adjacency detection
    inst_map = np.zeros((h, w), dtype=np.int32)  # 0 = background
    for i in range(n):
        if not thing_mask[i]:
            continue
        m = masks[i].reshape(h, w)
        inst_map[m] = i + 1  # 1-indexed

    # Find adjacencies: dilate each instance by 1 pixel, check what other instances it touches
    adjacency = set()  # set of (i, j) pairs where i < j
    struct = ndimage.generate_binary_structure(2, 1)  # 4-connected

    for i in range(n):
        if not thing_mask[i]:
            continue
        m = masks[i].reshape(h, w)
        dilated = ndimage.binary_dilation(m, structure=struct, iterations=1)
        border = dilated & ~m  # just the 1-pixel border
        neighbors = np.unique(inst_map[border])
        for nb in neighbors:
            if nb == 0:  # background
                continue
            j = nb - 1  # convert back to 0-indexed
            if j == i:
                continue
            if not thing_mask[j]:
                continue
            pair = (min(i, j), max(i, j))
            adjacency.add(pair)

    # 3. Filter adjacencies by same-class AND similar depth
    merge_pairs = []
    for i, j in adjacency:
        if inst_classes[i] != inst_classes[j]:
            continue
        depth_diff = abs(inst_depths[i] - inst_depths[j])
        if depth_diff < merge_depth_delta:
            merge_pairs.append((i, j))

    if not merge_pairs:
        return masks, scores

    # 4. Union-Find to group transitive merges
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            # Merge smaller into larger
            if inst_areas[px] >= inst_areas[py]:
                parent[py] = px
            else:
                parent[px] = py

    for i, j in merge_pairs:
        union(i, j)

    # 5. Group instances by their root
    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    # Build merged output
    merged_masks = []
    merged_scores = []

    for root, members in groups.items():
        if len(members) == 1:
            # No merge needed
            merged_masks.append(masks[members[0]])
            merged_scores.append(scores[members[0]])
        else:
            # Merge: OR all masks, take max score
            combined = np.zeros(h * w, dtype=bool)
            best_score = 0.0
            for idx in members:
                combined |= masks[idx]
                best_score = max(best_score, scores[idx])
            if combined.sum() >= min_area:
                merged_masks.append(combined)
                merged_scores.append(best_score)

    if not merged_masks:
        return np.zeros((0, h * w), dtype=bool), np.zeros(0, dtype=np.float32)

    merged_masks = np.stack(merged_masks)
    merged_scores = np.array(merged_scores, dtype=np.float32)

    # Sort by score descending
    order = np.argsort(-merged_scores)
    merged_masks = merged_masks[order]
    merged_scores = merged_scores[order]

    return merged_masks, merged_scores


def find_npz_files(instance_dir):
    """Find all NPZ files in instance directory."""
    return sorted(Path(instance_dir).rglob("*.npz"))


def main():
    parser = argparse.ArgumentParser(
        description="Merge small adjacent same-class depth-guided instances"
    )
    parser.add_argument("--instance_dir", type=str, required=True,
                        help="Input directory with depth-guided instance NPZ files")
    parser.add_argument("--semantic_dir", type=str, required=True,
                        help="Directory with semantic pseudo-label PNGs")
    parser.add_argument("--depth_dir", type=str, required=True,
                        help="Directory with depth map NPY files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for merged instance NPZ files")
    parser.add_argument("--stuff_things", type=str, default=None,
                        help="Path to stuff_things.json")
    parser.add_argument("--merge_depth_delta", type=float, default=0.05,
                        help="Max mean depth difference to allow merging (default: 0.05)")
    parser.add_argument("--min_area", type=int, default=50,
                        help="Min area for merged instance to survive (default: 50)")

    args = parser.parse_args()

    if args.stuff_things:
        thing_ids = load_thing_ids(args.stuff_things)
    else:
        thing_ids = DEFAULT_THING_IDS

    logger.info(f"Thing IDs: {sorted(thing_ids)}")
    logger.info(f"Merge depth delta: {args.merge_depth_delta}")

    instance_dir = Path(args.instance_dir)
    semantic_dir = Path(args.semantic_dir)
    depth_dir = Path(args.depth_dir)
    output_dir = Path(args.output_dir)

    npz_files = find_npz_files(instance_dir)
    logger.info(f"Found {len(npz_files)} instance NPZ files")

    t0 = time.time()
    total_before = 0
    total_after = 0
    total_merges = 0

    for npz_path in npz_files:
        # Load instances
        masks, scores, h, w = load_instances(npz_path)
        n_before = masks.shape[0]
        total_before += n_before

        if n_before == 0:
            # Copy empty file
            rel = npz_path.relative_to(instance_dir)
            out_path = output_dir / rel
            save_instances(masks, scores, out_path, h, w)
            continue

        # Find corresponding semantic and depth files
        rel = npz_path.relative_to(instance_dir)
        stem = npz_path.stem
        city = rel.parent

        sem_path = semantic_dir / city / f"{stem}.png"
        if not sem_path.exists():
            # Try without _leftImg8bit suffix
            alt_stem = stem.replace("_leftImg8bit", "")
            sem_path = semantic_dir / city / f"{alt_stem}.png"

        # Depth: try .npy
        depth_path = depth_dir / city / f"{stem}.npy"
        if not depth_path.exists():
            alt_stem = stem.replace("_leftImg8bit", "")
            depth_path = depth_dir / city / f"{alt_stem}.npy"
        if not depth_path.exists():
            # Try with _leftImg8bit
            depth_path = depth_dir / city / f"{stem}_leftImg8bit.npy"

        if not sem_path.exists() or not depth_path.exists():
            logger.warning(f"Missing semantic ({sem_path.exists()}) or depth ({depth_path.exists()}) for {npz_path.name}")
            # Copy as-is
            rel = npz_path.relative_to(instance_dir)
            out_path = output_dir / rel
            save_instances(masks, scores, out_path, h, w)
            total_after += n_before
            continue

        # Load semantic and depth
        semantic = np.array(Image.open(sem_path))
        if semantic.shape != (h, w):
            semantic = np.array(
                Image.fromarray(semantic).resize((w, h), Image.NEAREST)
            )

        depth = np.load(str(depth_path))
        if depth.shape != (h, w):
            depth = np.array(
                Image.fromarray(depth).resize((w, h), Image.BILINEAR)
            )

        # Merge
        merged_masks, merged_scores = merge_instances(
            masks, scores, semantic, depth, thing_ids,
            merge_depth_delta=args.merge_depth_delta,
            min_area=args.min_area,
        )

        n_after = merged_masks.shape[0]
        total_after += n_after
        total_merges += (n_before - n_after)

        # Save
        out_path = output_dir / rel
        save_instances(merged_masks, merged_scores, out_path, h, w)

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s")
    logger.info(f"Instances: {total_before} -> {total_after} ({total_merges} merged)")
    logger.info(f"Avg reduction: {total_merges / max(len(npz_files), 1):.1f} merges/image")


if __name__ == "__main__":
    main()
