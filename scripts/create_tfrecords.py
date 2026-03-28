"""Create TFRecords from Dataset.

Usage:
    python scripts/create_tfrecords.py \
        --config configs/cityscapes.yaml \
        --output_dir /path/to/tfrecords \
        --split train

    # Write directly to GCS:
    python scripts/create_tfrecords.py \
        --config configs/cityscapes.yaml \
        --output_dir gs://mbps-panoptic/datasets/cityscapes/tfrecords/train \
        --split train

Converts dataset images + depth maps + labels into sharded
TFRecord files for efficient TPU data loading. Streams samples
one at a time to avoid loading entire dataset into memory.
"""

from __future__ import annotations

import argparse
import os
import sys

import tensorflow as tf
import yaml
from absl import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbps.data.datasets import get_dataset
from mbps.data.tfrecord_utils import write_tfrecords


def main():
    parser = argparse.ArgumentParser(description="Create TFRecords")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--shard_size", type=int, default=256,
        help="Number of samples per TFRecord shard",
    )
    # Allow overriding data/depth dirs from CLI (useful for GCS paths)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--depth_dir", type=str, default=None)
    parser.add_argument(
        "--pseudo_mask_dir", type=str, default=None,
        help="Directory with CutS3D pseudo masks (.npz files)",
    )
    parser.add_argument(
        "--max_instances", type=int, default=5,
        help="Max instances per image for padding",
    )
    args = parser.parse_args()

    logging.set_verbosity(logging.INFO)

    # Load config with deep merge
    config_dir = os.path.dirname(os.path.abspath(args.config))
    default_path = os.path.join(config_dir, "default.yaml")
    with open(default_path) as f:
        config = yaml.safe_load(f)
    with open(args.config) as f:
        override = yaml.safe_load(f)
    if override:
        for k, v in override.items():
            if isinstance(v, dict) and k in config:
                config[k].update(v)
            else:
                config[k] = v

    # Resolve dataset name (configs use "dataset" or "dataset_name")
    data_cfg = config["data"]
    dataset_name = data_cfg.get("dataset_name", data_cfg.get("dataset"))
    if not dataset_name:
        logging.error("Config must have data.dataset or data.dataset_name")
        sys.exit(1)

    data_dir = args.data_dir or data_cfg["data_dir"]
    depth_dir = args.depth_dir or data_cfg.get("depth_dir", "")

    logging.info(f"Dataset: {dataset_name}")
    logging.info(f"Data dir: {data_dir}")
    logging.info(f"Depth dir: {depth_dir}")
    logging.info(f"Split: {args.split}")
    logging.info(f"Output: {args.output_dir}")

    # Resolve pseudo mask dir
    pseudo_mask_dir = args.pseudo_mask_dir or data_cfg.get("pseudo_mask_dir")
    gcs_pseudo = config.get("gcs", {}).get("pseudo_mask_dir")
    if gcs_pseudo and not pseudo_mask_dir:
        pseudo_mask_dir = gcs_pseudo
    if pseudo_mask_dir:
        logging.info(f"Pseudo mask dir: {pseudo_mask_dir}")

    max_instances = args.max_instances or config.get(
        "architecture", {}
    ).get("max_instances", 5)

    # Create dataset (does NOT load samples into memory — just collects paths)
    dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        depth_dir=depth_dir,
        split=args.split,
        image_size=tuple(data_cfg["image_size"]),
        pseudo_mask_dir=pseudo_mask_dir,
        max_instances=max_instances,
    )

    logging.info(f"Found {len(dataset)} samples. Writing TFRecords (streaming)...")

    # Write TFRecords — streams one sample at a time via dataset[i]
    write_tfrecords(
        dataset=dataset,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
    )

    logging.info(
        f"Created TFRecords: {len(dataset)} samples in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
