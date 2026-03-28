"""Dataset loaders for MBPS.

Supports Cityscapes, COCO-Stuff-27, and NYU Depth V2.
All datasets return (image, depth_map, labels, metadata) tuples.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from absl import logging


def _open_image(path: str) -> Image.Image:
    """Open an image from local or GCS path."""
    if path.startswith("gs://"):
        with tf.io.gfile.GFile(path, "rb") as f:
            return Image.open(io.BytesIO(f.read()))
    return Image.open(path)


def _file_exists(path: str) -> bool:
    """Check if a file exists (local or GCS)."""
    if path.startswith("gs://"):
        return tf.io.gfile.exists(path)
    return os.path.exists(path)


class BaseDataset:
    """Base dataset class for MBPS.

    Args:
        data_dir: Root directory of the dataset.
        depth_dir: Directory containing pre-computed ZoeDepth maps.
        split: Dataset split ('train', 'val', 'test').
        image_size: Target image size (H, W).
        transforms: Optional transform functions.
    """

    def __init__(
        self,
        data_dir: str,
        depth_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        transforms: Optional[Any] = None,
        subset_fraction: Optional[float] = None,
        pseudo_mask_dir: Optional[str] = None,
        max_instances: int = 5,
    ):
        self.data_dir = data_dir
        self.depth_dir = depth_dir
        self.split = split
        self.image_size = image_size
        self.transforms = transforms
        self.subset_fraction = subset_fraction
        self.pseudo_mask_dir = pseudo_mask_dir
        self.max_instances = max_instances
        self.samples: List[Dict[str, str]] = []
        self._load_samples()

        # Apply subset sampling if requested
        if self.subset_fraction is not None and 0 < self.subset_fraction < 1.0:
            rng = np.random.RandomState(42)
            n_subset = max(1, int(len(self.samples) * self.subset_fraction))
            indices = rng.choice(len(self.samples), size=n_subset, replace=False)
            self.samples = [self.samples[i] for i in sorted(indices)]
            logging.info(
                f"Subset: using {len(self.samples)} samples "
                f"({self.subset_fraction*100:.0f}%)"
            )

    def _load_samples(self) -> None:
        """Load sample file paths. Override in subclass."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Load a single sample.

        Returns:
            Dictionary with keys: 'image', 'depth', 'image_id',
            and optionally 'semantic_label', 'instance_label'.
        """
        sample = self.samples[idx]

        # Load image (supports local and GCS paths)
        image = np.array(_open_image(sample["image"]).convert("RGB"))
        image = np.array(
            Image.fromarray(image).resize(
                (self.image_size[1], self.image_size[0]),
                Image.BILINEAR,
            )
        ).astype(np.float32) / 255.0

        # Load depth (supports local and GCS paths)
        depth_path = sample.get("depth")
        depth_loaded = False
        if depth_path:
            try:
                if depth_path.startswith("gs://"):
                    with tf.io.gfile.GFile(depth_path, "rb") as f:
                        depth = np.load(io.BytesIO(f.read())).astype(np.float32)
                    depth_loaded = True
                elif os.path.exists(depth_path):
                    depth = np.load(depth_path).astype(np.float32)
                    depth_loaded = True
            except Exception:
                pass
        if depth_loaded:
            depth = np.array(
                Image.fromarray(depth).resize(
                    (self.image_size[1], self.image_size[0]),
                    Image.BILINEAR,
                )
            )
            # Normalize depth to [0, 1] to prevent overflow in sinusoidal
            # encoding and depth conditioning exponentials
            d_min, d_max = depth.min(), depth.max()
            depth = (depth - d_min) / (d_max - d_min + 1e-8)
        else:
            depth = np.zeros(self.image_size, dtype=np.float32)

        result = {
            "image": image,
            "depth": depth,
            "image_id": sample.get("image_id", str(idx)),
        }

        # Load semantic labels if available
        if "semantic_label" in sample and _file_exists(sample["semantic_label"]):
            label = np.array(
                _open_image(sample["semantic_label"]).resize(
                    (self.image_size[1], self.image_size[0]),
                    Image.NEAREST,
                )
            ).astype(np.int32)
            result["semantic_label"] = label

        # Load instance labels if available
        if "instance_label" in sample and _file_exists(sample["instance_label"]):
            inst_label = np.array(
                _open_image(sample["instance_label"]).resize(
                    (self.image_size[1], self.image_size[0]),
                    Image.NEAREST,
                )
            ).astype(np.int32)
            result["instance_label"] = inst_label

        # Load pseudo masks if available
        if self.pseudo_mask_dir:
            result.update(
                self._load_pseudo_masks(idx, sample.get("image_id", str(idx)))
            )

        if self.transforms is not None:
            result = self.transforms(result)

        return result

    def _load_pseudo_masks(
        self, idx: int, image_id: str
    ) -> Dict[str, np.ndarray]:
        """Load CutS3D pseudo masks from .npz file.

        Args:
            idx: Sample index (used for filename).
            image_id: Image identifier.

        Returns:
            Dict with 'pseudo_masks', 'spatial_confidence', 'num_valid_masks'.
        """
        mask_path = os.path.join(
            self.pseudo_mask_dir, f"masks_{idx:08d}.npz"
        )
        M = self.max_instances

        try:
            if mask_path.startswith("gs://"):
                with tf.io.gfile.GFile(mask_path, "rb") as f:
                    data = np.load(io.BytesIO(f.read()))
            else:
                data = np.load(mask_path)

            masks = data["masks"].astype(np.float32)      # (num_valid, K)
            sc = data["spatial_confidence"].astype(np.float32)  # (num_valid, K)
            num_valid = int(data["num_valid"])

            # Pad to max_instances
            K = masks.shape[1]
            padded_masks = np.zeros((M, K), dtype=np.float32)
            padded_sc = np.zeros((M, K), dtype=np.float32)
            n = min(num_valid, M)
            padded_masks[:n] = masks[:n]
            padded_sc[:n] = sc[:n]

            return {
                "pseudo_masks": padded_masks,
                "spatial_confidence": padded_sc,
                "num_valid_masks": np.int32(n),
            }
        except Exception:
            # Fallback: empty masks
            patch_h = self.image_size[0] // 8
            patch_w = self.image_size[1] // 8
            K = patch_h * patch_w
            return {
                "pseudo_masks": np.zeros((M, K), dtype=np.float32),
                "spatial_confidence": np.zeros((M, K), dtype=np.float32),
                "num_valid_masks": np.int32(0),
            }


# Cityscapes original label IDs → train IDs (0-18), 255=ignore
CITYSCAPES_ID_TO_TRAINID = {
    7: 0,    # road
    8: 1,    # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}


def remap_cityscapes_labels(label: np.ndarray) -> np.ndarray:
    """Remap Cityscapes original label IDs to train IDs (0-18).

    Args:
        label: Label array with original Cityscapes IDs.

    Returns:
        Label array with train IDs (0-18), unmapped classes set to 255.
    """
    remapped = np.full_like(label, 255)
    for orig_id, train_id in CITYSCAPES_ID_TO_TRAINID.items():
        remapped[label == orig_id] = train_id
    return remapped


class CityscapesDataset(BaseDataset):
    """Cityscapes dataset loader.

    Expected directory structure:
        data_dir/
            leftImg8bit/
                train/city/*.png
                val/city/*.png
            gtFine/
                train/city/*_labelIds.png
                val/city/*_instanceIds.png
        depth_dir/
            train/city/*.npy
            val/city/*.npy
    """

    NUM_CLASSES = 19
    STUFF_CLASSES = list(range(11))
    THING_CLASSES = list(range(11, 19))

    def _load_samples(self) -> None:
        img_dir = os.path.join(self.data_dir, "leftImg8bit", self.split)
        gt_dir = os.path.join(self.data_dir, "gtFine", self.split)

        # Use tf.io.gfile for GCS compatibility
        if not tf.io.gfile.isdir(img_dir):
            logging.warning(f"Cityscapes image dir not found: {img_dir}")
            return

        for city in sorted(tf.io.gfile.listdir(img_dir)):
            city_img_dir = os.path.join(img_dir, city)
            if not tf.io.gfile.isdir(city_img_dir):
                continue

            for fname in sorted(tf.io.gfile.listdir(city_img_dir)):
                if not fname.endswith("_leftImg8bit.png"):
                    continue

                base = fname.replace("_leftImg8bit.png", "")
                sample = {
                    "image": os.path.join(city_img_dir, fname),
                    "image_id": base,
                }

                # Semantic label
                sem_path = os.path.join(
                    gt_dir, city, f"{base}_gtFine_labelIds.png"
                )
                if tf.io.gfile.exists(sem_path):
                    sample["semantic_label"] = sem_path

                # Instance label
                inst_path = os.path.join(
                    gt_dir, city, f"{base}_gtFine_instanceIds.png"
                )
                if tf.io.gfile.exists(inst_path):
                    sample["instance_label"] = inst_path

                # Depth
                depth_path = os.path.join(
                    self.depth_dir, self.split, city, f"{base}.npy"
                )
                sample["depth"] = depth_path

                self.samples.append(sample)

        logging.info(
            f"Cityscapes {self.split}: loaded {len(self.samples)} samples"
        )

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Load sample with Cityscapes label remapping."""
        result = super().__getitem__(idx)
        if "semantic_label" in result:
            result["semantic_label"] = remap_cityscapes_labels(
                result["semantic_label"]
            )
        return result


class COCOStuff27Dataset(BaseDataset):
    """COCO-Stuff-27 dataset loader.

    Expected directory structure:
        data_dir/
            images/
                train2017/*.jpg
                val2017/*.jpg
            annotations/
                stuff_train2017_pixelmaps/*.png
                stuff_val2017_pixelmaps/*.png
        depth_dir/
            train2017/*.npy
            val2017/*.npy
    """

    NUM_CLASSES = 27

    def _load_samples(self) -> None:
        split_name = f"{self.split}2017"
        img_dir = os.path.join(self.data_dir, "images", split_name)
        ann_dir = os.path.join(
            self.data_dir, "annotations", f"stuff_{split_name}_pixelmaps"
        )

        if not os.path.isdir(img_dir):
            logging.warning(f"COCO image dir not found: {img_dir}")
            return

        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith((".jpg", ".png")):
                continue

            base = os.path.splitext(fname)[0]
            sample = {
                "image": os.path.join(img_dir, fname),
                "image_id": base,
            }

            # Semantic label
            sem_path = os.path.join(ann_dir, f"{base}.png")
            if os.path.exists(sem_path):
                sample["semantic_label"] = sem_path

            # Depth
            depth_path = os.path.join(self.depth_dir, split_name, f"{base}.npy")
            sample["depth"] = depth_path

            self.samples.append(sample)

        logging.info(
            f"COCO-Stuff-27 {self.split}: loaded {len(self.samples)} samples"
        )


class NYUDepthV2Dataset(BaseDataset):
    """NYU Depth V2 dataset loader for prototyping.

    Expected directory structure:
        data_dir/
            images/
                *.png
            depth/
                *.npy
            labels/
                *.png
    """

    NUM_CLASSES = 40

    def _load_samples(self) -> None:
        img_dir = os.path.join(self.data_dir, "images")

        if not os.path.isdir(img_dir):
            logging.warning(f"NYU image dir not found: {img_dir}")
            return

        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith((".png", ".jpg")):
                continue

            base = os.path.splitext(fname)[0]
            sample = {
                "image": os.path.join(img_dir, fname),
                "image_id": base,
                "depth": os.path.join(self.data_dir, "depth", f"{base}.npy"),
                "semantic_label": os.path.join(
                    self.data_dir, "labels", f"{base}.png"
                ),
            }
            self.samples.append(sample)

        logging.info(
            f"NYU Depth V2: loaded {len(self.samples)} samples"
        )


class PASCALVOCDataset(BaseDataset):
    """PASCAL VOC 2012 dataset loader for prototyping.

    Expected directory structure:
        data_dir/  (= VOCdevkit/VOC2012)
            JPEGImages/*.jpg
            SegmentationClass/*.png    (semantic)
            SegmentationObject/*.png   (instance)
            ImageSets/Segmentation/train.txt
            ImageSets/Segmentation/val.txt
        depth_dir/
            *.npy  (pre-computed ZoeDepth)
    """

    NUM_CLASSES = 21  # 20 classes + background

    def _load_samples(self) -> None:
        split_file = os.path.join(
            self.data_dir, "ImageSets", "Segmentation", f"{self.split}.txt"
        )

        if not os.path.exists(split_file):
            logging.warning(f"PASCAL VOC split file not found: {split_file}")
            return

        with open(split_file) as f:
            image_ids = [line.strip() for line in f if line.strip()]

        for img_id in image_ids:
            sample = {
                "image": os.path.join(self.data_dir, "JPEGImages", f"{img_id}.jpg"),
                "image_id": img_id,
            }

            # Semantic label
            sem_path = os.path.join(
                self.data_dir, "SegmentationClass", f"{img_id}.png"
            )
            if os.path.exists(sem_path):
                sample["semantic_label"] = sem_path

            # Instance label
            inst_path = os.path.join(
                self.data_dir, "SegmentationObject", f"{img_id}.png"
            )
            if os.path.exists(inst_path):
                sample["instance_label"] = inst_path

            # Depth
            depth_path = os.path.join(self.depth_dir, f"{img_id}.npy")
            sample["depth"] = depth_path

            self.samples.append(sample)

        logging.info(
            f"PASCAL VOC 2012 {self.split}: loaded {len(self.samples)} samples"
        )


def get_dataset(
    dataset_name: str,
    data_dir: str,
    depth_dir: str,
    split: str = "train",
    image_size: Tuple[int, int] = (512, 512),
    transforms: Optional[Any] = None,
    subset_fraction: Optional[float] = None,
    pseudo_mask_dir: Optional[str] = None,
    max_instances: int = 5,
) -> BaseDataset:
    """Factory function to create dataset by name.

    Args:
        dataset_name: One of 'cityscapes', 'coco_stuff27', 'nyu_depth_v2', 'pascal_voc'.
        data_dir: Root data directory.
        depth_dir: Pre-computed depth directory.
        split: Dataset split.
        image_size: Target image size.
        transforms: Optional transforms.
        subset_fraction: Optional fraction for random subsetting (e.g., 0.05).
        pseudo_mask_dir: Directory with CutS3D pseudo masks (.npz).
        max_instances: Maximum instances per image for padding.

    Returns:
        Dataset instance.

    Raises:
        ValueError: If dataset_name is not recognized.
    """
    datasets = {
        "cityscapes": CityscapesDataset,
        "coco_stuff27": COCOStuff27Dataset,
        "nyu_depth_v2": NYUDepthV2Dataset,
        "pascal_voc": PASCALVOCDataset,
    }

    if dataset_name not in datasets:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(datasets.keys())}"
        )

    return datasets[dataset_name](
        data_dir=data_dir,
        depth_dir=depth_dir,
        split=split,
        image_size=image_size,
        transforms=transforms,
        subset_fraction=subset_fraction,
        pseudo_mask_dir=pseudo_mask_dir,
        max_instances=max_instances,
    )


def collate_batch(
    samples: List[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Collate a list of samples into a batched dictionary.

    Args:
        samples: List of sample dictionaries.

    Returns:
        Batched dictionary with stacked arrays.
    """
    batch = {}
    keys = samples[0].keys()

    for key in keys:
        values = [s[key] for s in samples]
        if isinstance(values[0], np.ndarray):
            batch[key] = np.stack(values, axis=0)
        else:
            batch[key] = values

    return batch
