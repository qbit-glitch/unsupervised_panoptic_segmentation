#!/usr/bin/env python3
"""Generate v2 TFRecords with pseudo-labels for MBPS v2 training.

Combines:
  - Original images (RGB, float32)
  - Depth maps (DA V3, float32)
  - Pseudo semantic labels (from DINOv3 k-means)
  - Pseudo instance masks (from MaskCut)
  - Stuff-things mapping

Output TFRecord schema:
  - image: (H, W, 3) float32
  - depth: (H, W) float32
  - image_id: string
  - height, width: int64
  - pseudo_semantic: (H, W) int32 — semantic cluster labels
  - pseudo_instance: (H, W) int32 — instance ID map (0=bg)
  - pseudo_masks: (M, N_patches) float32 — patch-level masks
  - num_masks: int64
  - mask_dim_m, mask_dim_k: int64

Usage:
    python mbps_pytorch/generate_v2_tfrecords.py \
        --image_dir /data/cityscapes/leftImg8bit/train \
        --depth_dir /data/cityscapes/depth_dav3/train \
        --semantic_dir /data/cityscapes/pseudo_semantic/train \
        --instance_dir /data/cityscapes/pseudo_instance/train \
        --output_dir /data/cityscapes/tfrecords_v2/train \
        --num_shards 12 \
        --image_size 512 1024
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_v2_example(
    image: np.ndarray,
    depth: np.ndarray,
    image_id: str,
    pseudo_semantic: np.ndarray,
    pseudo_instance: np.ndarray,
    pseudo_masks: np.ndarray = None,
    num_valid_masks: int = 0,
) -> bytes:
    """Serialize a v2 training example.

    Args:
        image: (H, W, 3) float32 RGB image, normalized to [0, 1].
        depth: (H, W) float32 depth map, normalized to [0, 1].
        image_id: Unique identifier.
        pseudo_semantic: (H, W) int32 semantic cluster labels.
        pseudo_instance: (H, W) int32 instance ID map.
        pseudo_masks: (M, N_patches) float32 patch-level instance masks.
        num_valid_masks: Number of valid masks in pseudo_masks.

    Returns:
        Serialized tf.train.Example as bytes.
    """
    h, w = image.shape[:2]

    feature = {
        "image": _bytes_feature(image.astype(np.float32).tobytes()),
        "depth": _bytes_feature(depth.astype(np.float32).tobytes()),
        "image_id": _bytes_feature(image_id.encode("utf-8")),
        "height": _int64_feature(h),
        "width": _int64_feature(w),
        "pseudo_semantic": _bytes_feature(
            pseudo_semantic.astype(np.int32).tobytes()
        ),
        "pseudo_instance": _bytes_feature(
            pseudo_instance.astype(np.int32).tobytes()
        ),
    }

    if pseudo_masks is not None and num_valid_masks > 0:
        feature["pseudo_masks"] = _bytes_feature(
            pseudo_masks.astype(np.float32).tobytes()
        )
        feature["num_masks"] = _int64_feature(num_valid_masks)
        feature["mask_dim_m"] = _int64_feature(pseudo_masks.shape[0])
        feature["mask_dim_k"] = _int64_feature(pseudo_masks.shape[1])

    example = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example.SerializeToString()


def get_image_list(image_dir: str) -> list:
    """Get sorted list of image paths."""
    image_dir = Path(image_dir)
    paths = sorted(image_dir.rglob("*_leftImg8bit.png"))
    if not paths:
        paths = sorted(image_dir.rglob("*.png"))
    return paths


def find_matching_file(
    image_path: Path,
    image_dir: Path,
    target_dir: Path,
    suffix: str = ".npy",
    alt_suffix: str = None,
) -> Path:
    """Find file in target_dir matching the image path structure."""
    rel = image_path.relative_to(image_dir)
    # Try direct match (same relative path, different extension)
    candidate = target_dir / rel.with_suffix(suffix)
    if candidate.exists():
        return candidate

    # Try without _leftImg8bit suffix
    stem = rel.stem.replace("_leftImg8bit", "")
    candidate = target_dir / rel.parent / (stem + suffix)
    if candidate.exists():
        return candidate

    if alt_suffix:
        candidate = target_dir / rel.with_suffix(alt_suffix)
        if candidate.exists():
            return candidate
        candidate = target_dir / rel.parent / (stem + alt_suffix)
        if candidate.exists():
            return candidate

    return None


# ImageNet normalization stats
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def generate_tfrecords(
    image_dir: str,
    depth_dir: str,
    semantic_dir: str,
    instance_dir: str,
    output_dir: str,
    num_shards: int = 12,
    image_size: tuple = (512, 1024),
    max_instances: int = 20,
):
    """Generate sharded TFRecords from all pseudo-label outputs.

    Args:
        image_dir: Cityscapes leftImg8bit directory.
        depth_dir: DA V3 depth maps (.npy).
        semantic_dir: Pseudo semantic labels (.png).
        instance_dir: Instance masks (.npz) and instance maps (*_instance.png).
        output_dir: Where to write TFRecord files.
        num_shards: Number of TFRecord shards.
        image_size: (H, W) target resolution.
        max_instances: Max instances per image for mask tensor.
    """
    image_dir = Path(image_dir)
    depth_dir = Path(depth_dir)
    semantic_dir = Path(semantic_dir)
    instance_dir = Path(instance_dir)
    os.makedirs(output_dir, exist_ok=True)

    image_paths = get_image_list(str(image_dir))
    print(f"Found {len(image_paths)} images")

    # Open TFRecord writers
    writers = []
    for shard_id in range(num_shards):
        path = os.path.join(output_dir, f"shard_{shard_id:04d}.tfrecord")
        writers.append(tf.io.TFRecordWriter(path))

    n_patches = (image_size[0] // 16) * (image_size[1] // 16)  # 2048
    written = 0
    skipped = 0

    for idx, img_path in enumerate(tqdm(image_paths, desc="Writing TFRecords")):
        # Load image
        img = Image.open(img_path).convert("RGB")
        img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
        img_array = np.array(img).astype(np.float32) / 255.0  # [0, 1]

        # Load depth
        depth_path = find_matching_file(img_path, image_dir, depth_dir, ".npy")
        if depth_path is None:
            skipped += 1
            continue
        depth = np.load(str(depth_path)).astype(np.float32)
        if depth.shape != (image_size[0], image_size[1]):
            depth_img = Image.fromarray(depth, mode="F")
            depth = np.array(
                depth_img.resize((image_size[1], image_size[0]), Image.BILINEAR)
            )

        # Load semantic pseudo-labels
        sem_path = find_matching_file(
            img_path, image_dir, semantic_dir, ".png"
        )
        if sem_path is None:
            skipped += 1
            continue
        semantic = np.array(Image.open(sem_path)).astype(np.int32)
        if semantic.shape != (image_size[0], image_size[1]):
            semantic = np.array(
                Image.fromarray(semantic.astype(np.uint8)).resize(
                    (image_size[1], image_size[0]), Image.NEAREST
                )
            ).astype(np.int32)

        # Load instance pseudo-labels
        inst_map_path = find_matching_file(
            img_path, image_dir, instance_dir, "_instance.png"
        )
        inst_masks_path = find_matching_file(
            img_path, image_dir, instance_dir, ".npz"
        )

        if inst_map_path is not None:
            instance = np.array(Image.open(inst_map_path)).astype(np.int32)
            if instance.shape != (image_size[0], image_size[1]):
                instance = np.array(
                    Image.fromarray(instance.astype(np.uint16)).resize(
                        (image_size[1], image_size[0]), Image.NEAREST
                    )
                ).astype(np.int32)
        else:
            instance = np.zeros((image_size[0], image_size[1]), dtype=np.int32)

        # Load patch-level masks if available
        pseudo_masks = None
        num_valid = 0
        if inst_masks_path is not None:
            data = np.load(str(inst_masks_path))
            masks = data["masks"]  # (num_valid, N_patches) or empty
            num_valid = int(data["num_valid"])
            if num_valid > 0:
                # Pad to max_instances
                M, K = masks.shape
                pseudo_masks = np.zeros(
                    (max_instances, n_patches), dtype=np.float32
                )
                n_copy = min(M, max_instances)
                if K == n_patches:
                    pseudo_masks[:n_copy] = masks[:n_copy].astype(np.float32)
                num_valid = n_copy

        # Image ID from filename
        image_id = img_path.stem.replace("_leftImg8bit", "")

        # Serialize and write to shard
        serialized = serialize_v2_example(
            image=img_array,
            depth=depth,
            image_id=image_id,
            pseudo_semantic=semantic,
            pseudo_instance=instance,
            pseudo_masks=pseudo_masks,
            num_valid_masks=num_valid,
        )

        shard_id = idx % num_shards
        writers[shard_id].write(serialized)
        written += 1

    # Close writers
    for w in writers:
        w.close()

    print(f"\nDone! Wrote {written} examples to {num_shards} shards in {output_dir}")
    if skipped > 0:
        print(f"Skipped {skipped} images (missing depth or pseudo-labels)")

    # Save metadata
    metadata = {
        "num_examples": written,
        "num_shards": num_shards,
        "image_size": list(image_size),
        "n_patches": n_patches,
        "max_instances": max_instances,
        "skipped": skipped,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate v2 TFRecords with pseudo-labels"
    )
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--semantic_dir", type=str, required=True)
    parser.add_argument("--instance_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_shards", type=int, default=12)
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 1024])
    parser.add_argument("--max_instances", type=int, default=20)
    args = parser.parse_args()

    generate_tfrecords(
        image_dir=args.image_dir,
        depth_dir=args.depth_dir,
        semantic_dir=args.semantic_dir,
        instance_dir=args.instance_dir,
        output_dir=args.output_dir,
        num_shards=args.num_shards,
        image_size=tuple(args.image_size),
        max_instances=args.max_instances,
    )


if __name__ == "__main__":
    main()
