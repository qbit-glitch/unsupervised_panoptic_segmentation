"""TFRecord utilities for TPU-efficient data loading.

Provides serialization/deserialization of image+depth+label data
as TFRecords, and tf.data pipeline construction for TPU training.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from absl import logging


def _bytes_feature(value: bytes) -> tf.train.Feature:
    """Create a bytes feature for TFRecord."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> tf.train.Feature:
    """Create an int64 feature for TFRecord."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(
    image: np.ndarray,
    depth: np.ndarray,
    image_id: str,
    semantic_label: Optional[np.ndarray] = None,
    instance_label: Optional[np.ndarray] = None,
    pseudo_masks: Optional[np.ndarray] = None,
    spatial_confidence: Optional[np.ndarray] = None,
    num_valid_masks: int = 0,
    pseudo_semantic: Optional[np.ndarray] = None,
    pseudo_instance: Optional[np.ndarray] = None,
) -> bytes:
    """Serialize a single sample to TFRecord format.

    Args:
        image: RGB image as float32 array of shape (H, W, 3).
        depth: Depth map as float32 array of shape (H, W).
        image_id: Unique identifier string.
        semantic_label: Optional semantic label of shape (H, W).
        instance_label: Optional instance label of shape (H, W).
        pseudo_masks: Optional CutS3D masks of shape (M, K).
        spatial_confidence: Optional SC maps of shape (M, K).
        num_valid_masks: Number of valid masks.
        pseudo_semantic: v2 pseudo-semantic labels, int32 (N,) or (H, W).
        pseudo_instance: v2 pseudo-instance labels, int32 (N,) or (H, W).

    Returns:
        Serialized tf.train.Example as bytes.
    """
    h, w = image.shape[:2]

    feature = {
        "image": _bytes_feature(image.tobytes()),
        "depth": _bytes_feature(depth.tobytes()),
        "image_id": _bytes_feature(image_id.encode("utf-8")),
        "height": _int64_feature(h),
        "width": _int64_feature(w),
    }

    if semantic_label is not None:
        feature["semantic_label"] = _bytes_feature(
            semantic_label.astype(np.int32).tobytes()
        )

    if instance_label is not None:
        feature["instance_label"] = _bytes_feature(
            instance_label.astype(np.int32).tobytes()
        )

    if pseudo_masks is not None:
        feature["pseudo_masks"] = _bytes_feature(
            pseudo_masks.astype(np.float32).tobytes()
        )
        feature["spatial_confidence"] = _bytes_feature(
            spatial_confidence.astype(np.float32).tobytes()
        )
        feature["num_masks"] = _int64_feature(num_valid_masks)
        feature["mask_dim_m"] = _int64_feature(pseudo_masks.shape[0])
        feature["mask_dim_k"] = _int64_feature(pseudo_masks.shape[1])

    # v2 pseudo-labels (from K-means semantic + MaskCut instance)
    if pseudo_semantic is not None:
        feature["pseudo_semantic"] = _bytes_feature(
            pseudo_semantic.astype(np.int32).tobytes()
        )
        feature["pseudo_semantic_len"] = _int64_feature(pseudo_semantic.size)

    if pseudo_instance is not None:
        feature["pseudo_instance"] = _bytes_feature(
            pseudo_instance.astype(np.int32).tobytes()
        )
        feature["pseudo_instance_len"] = _int64_feature(pseudo_instance.size)

    example = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example.SerializeToString()


def parse_example(
    serialized: tf.Tensor,
    image_size: Tuple[int, int] = (512, 512),
    has_semantic: bool = True,
    has_instance: bool = False,
    has_pseudo_masks: bool = False,
    has_pseudo_labels: bool = False,
    max_instances: int = 5,
    num_patches: int = 4096,
    patch_size: int = 8,
) -> Dict[str, tf.Tensor]:
    """Parse a serialized TFRecord example.

    Args:
        serialized: Serialized tf.train.Example.
        image_size: Expected (H, W) of stored images.
        has_semantic: Whether semantic labels are included.
        has_instance: Whether instance labels are included.
        has_pseudo_masks: Whether CutS3D pseudo masks are included.
        has_pseudo_labels: Whether v2 pseudo-labels are included
            (pseudo_semantic and pseudo_instance from K-means/MaskCut).
        max_instances: Max instances M for pseudo mask shape.
        num_patches: Number of patches K for pseudo mask shape.
        patch_size: Patch size for computing token count (8 for v1, 16 for v2).

    Returns:
        Dictionary of parsed tensors.
    """
    feature_spec = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "depth": tf.io.FixedLenFeature([], tf.string),
        "image_id": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
    }

    if has_semantic:
        feature_spec["semantic_label"] = tf.io.FixedLenFeature([], tf.string)
    if has_instance:
        feature_spec["instance_label"] = tf.io.FixedLenFeature([], tf.string)
    if has_pseudo_masks:
        feature_spec["pseudo_masks"] = tf.io.FixedLenFeature([], tf.string)
        feature_spec["spatial_confidence"] = tf.io.FixedLenFeature([], tf.string)
        feature_spec["num_masks"] = tf.io.FixedLenFeature([], tf.int64)
        feature_spec["mask_dim_m"] = tf.io.FixedLenFeature([], tf.int64)
        feature_spec["mask_dim_k"] = tf.io.FixedLenFeature([], tf.int64)
    if has_pseudo_labels:
        feature_spec["pseudo_semantic"] = tf.io.FixedLenFeature([], tf.string)
        feature_spec["pseudo_semantic_len"] = tf.io.FixedLenFeature([], tf.int64)
        feature_spec["pseudo_instance"] = tf.io.FixedLenFeature([], tf.string)
        feature_spec["pseudo_instance_len"] = tf.io.FixedLenFeature([], tf.int64)

    example = tf.io.parse_single_example(serialized, feature_spec)

    h = tf.cast(example["height"], tf.int32)
    w = tf.cast(example["width"], tf.int32)

    # Decode image
    image = tf.io.decode_raw(example["image"], tf.float32)
    image = tf.reshape(image, [image_size[0], image_size[1], 3])

    # Decode depth
    depth = tf.io.decode_raw(example["depth"], tf.float32)
    depth = tf.reshape(depth, [image_size[0], image_size[1]])

    result = {
        "image": image,
        "depth": depth,
        "image_id": example["image_id"],
    }

    if has_semantic:
        label = tf.io.decode_raw(example["semantic_label"], tf.int32)
        label = tf.reshape(label, [image_size[0], image_size[1]])
        result["semantic_label"] = label

    if has_instance:
        inst = tf.io.decode_raw(example["instance_label"], tf.int32)
        inst = tf.reshape(inst, [image_size[0], image_size[1]])
        result["instance_label"] = inst

    if has_pseudo_masks:
        masks = tf.io.decode_raw(example["pseudo_masks"], tf.float32)
        masks = tf.reshape(masks, [max_instances, num_patches])
        result["pseudo_masks"] = masks

        sc = tf.io.decode_raw(example["spatial_confidence"], tf.float32)
        sc = tf.reshape(sc, [max_instances, num_patches])
        result["spatial_confidence"] = sc

        result["num_valid_masks"] = tf.cast(example["num_masks"], tf.int32)

    if has_pseudo_labels:
        # v2 pseudo-labels: stored as flat int32 arrays of length N
        # where N = (H/patch_size) * (W/patch_size)
        n_tokens = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        ps = tf.io.decode_raw(example["pseudo_semantic"], tf.int32)
        ps = tf.reshape(ps, [n_tokens])
        result["pseudo_semantic"] = ps

        pi = tf.io.decode_raw(example["pseudo_instance"], tf.int32)
        pi = tf.reshape(pi, [n_tokens])
        result["pseudo_instance"] = pi

    return result


def create_tfrecord_dataset(
    tfrecord_pattern: str,
    batch_size: int,
    image_size: Tuple[int, int] = (512, 512),
    shuffle: bool = True,
    shuffle_buffer: int = 1000,
    prefetch_buffer: int = 16,
    num_parallel_calls: int = 8,
    has_semantic: bool = True,
    has_instance: bool = False,
    has_pseudo_masks: bool = False,
    has_pseudo_labels: bool = False,
    max_instances: int = 5,
    num_patches: Optional[int] = None,
    patch_size: int = 8,
    num_shards: int = 1,
    shard_index: int = 0,
) -> tf.data.Dataset:
    """Create a tf.data pipeline for TPU training.

    Args:
        tfrecord_pattern: Glob pattern for TFRecord files.
        batch_size: Batch size (total, will be split across TPU cores).
        image_size: Target image size (H, W).
        shuffle: Whether to shuffle.
        shuffle_buffer: Shuffle buffer size.
        prefetch_buffer: Prefetch buffer size.
        num_parallel_calls: Parallelism for map operations.
        has_semantic: Whether records contain semantic labels.
        has_instance: Whether records contain instance labels.
        has_pseudo_masks: Whether records contain CutS3D pseudo masks.
        has_pseudo_labels: Whether records contain v2 pseudo-labels.
        max_instances: Max instances M for pseudo mask shape.
        num_patches: Number of patches K. Defaults to (H/patch_size)*(W/patch_size).
        patch_size: Backbone patch size (8 for v1, 16 for v2).

    Returns:
        tf.data.Dataset yielding batched dictionaries.
    """
    if num_patches is None:
        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)

    files = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=shuffle)

    # Multi-host sharding: each host reads a disjoint subset of files
    if num_shards > 1:
        files = files.shard(num_shards=num_shards, index=shard_index)
        logging.info(
            f"Data sharding: process {shard_index} of {num_shards} "
            f"(file-level shard)"
        )

    dataset = files.interleave(
        lambda f: tf.data.TFRecordDataset(f, compression_type="GZIP"),
        cycle_length=num_parallel_calls,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)

    parse_fn = lambda x: parse_example(
        x, image_size, has_semantic, has_instance,
        has_pseudo_masks, has_pseudo_labels, max_instances, num_patches,
        patch_size,
    )
    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Repeat infinitely for multi-host training: ensures all workers run
    # the same number of steps per epoch (controlled by steps_per_epoch
    # in the trainer). Without this, uneven file-level sharding causes
    # workers to desync at epoch boundaries.
    if num_shards > 1:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(prefetch_buffer)

    return dataset


def write_tfrecords(
    dataset,
    output_dir: str,
    shard_size: int = 1000,
    compression: str = "GZIP",
) -> None:
    """Write dataset to TFRecord shards.

    Args:
        dataset: BaseDataset instance with __getitem__ and __len__.
        output_dir: Output directory for TFRecord files (local or GCS).
        shard_size: Number of examples per shard.
        compression: Compression type ('GZIP' or '').
    """
    tf.io.gfile.makedirs(output_dir)
    n = len(dataset)
    num_shards = (n + shard_size - 1) // shard_size

    options = tf.io.TFRecordOptions(compression_type=compression)

    for shard_idx in range(num_shards):
        shard_name = f"shard-{shard_idx:05d}-of-{num_shards:05d}.tfrecord"
        # Use "/" join for GCS compatibility (os.path.join works too)
        shard_path = output_dir.rstrip("/") + "/" + shard_name

        with tf.io.TFRecordWriter(shard_path, options=options) as writer:
            start = shard_idx * shard_size
            end = min(start + shard_size, n)

            for i in range(start, end):
                sample = dataset[i]
                serialized = serialize_example(
                    image=sample["image"],
                    depth=sample["depth"],
                    image_id=sample.get("image_id", str(i)),
                    semantic_label=sample.get("semantic_label"),
                    instance_label=sample.get("instance_label"),
                    pseudo_masks=sample.get("pseudo_masks"),
                    spatial_confidence=sample.get("spatial_confidence"),
                    num_valid_masks=int(sample.get("num_valid_masks", 0)),
                )
                writer.write(serialized)

        logging.info(
            f"Wrote shard {shard_idx + 1}/{num_shards} "
            f"({end - start} samples): {shard_name}"
        )

    logging.info(f"Finished writing {n} examples to {num_shards} shards.")
