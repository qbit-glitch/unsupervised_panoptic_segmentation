"""Dataset loaders for MBPS.

Supports Cityscapes, COCO-Stuff-27, and NYU Depth V2.
All datasets return (image, depth_map, labels, metadata) tuples.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """Base dataset class for MBPS.

    Inherits from ``torch.utils.data.Dataset`` so it can be used directly
    with ``torch.utils.data.DataLoader``.

    Args:
        data_dir: Root directory of the dataset.
        depth_dir: Directory containing pre-computed ZoeDepth maps.
        split: Dataset split ('train', 'val', 'test').
        image_size: Target image size (H, W).
        transforms: Optional transform functions.
        subset_fraction: Optional fraction for random subsetting (e.g., 0.05).
    """

    def __init__(
        self,
        data_dir: str,
        depth_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        transforms: Optional[Any] = None,
        subset_fraction: Optional[float] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.depth_dir = depth_dir
        self.split = split
        self.image_size = image_size
        self.transforms = transforms
        self.subset_fraction = subset_fraction
        self.samples: List[Dict[str, str]] = []
        self._load_samples()

        # Apply subset sampling if requested
        if self.subset_fraction is not None and 0 < self.subset_fraction < 1.0:
            rng = np.random.RandomState(42)
            n_subset = max(1, int(len(self.samples) * self.subset_fraction))
            indices = rng.choice(len(self.samples), size=n_subset, replace=False)
            self.samples = [self.samples[i] for i in sorted(indices)]
            logger.info(
                f"Subset: using {len(self.samples)} samples "
                f"({self.subset_fraction*100:.0f}%)"
            )

    def _load_samples(self) -> None:
        """Load sample file paths. Override in subclass."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample.

        Returns:
            Dictionary with keys: 'image', 'depth', 'image_id',
            and optionally 'semantic_label', 'instance_label'.
            Images are returned as (C, H, W) float32 tensors (NCHW convention).
            Depth maps are returned as (H, W) float32 tensors.
            Labels are returned as (H, W) int64 tensors.
        """
        sample = self.samples[idx]

        # Load image
        image = np.array(Image.open(sample["image"]).convert("RGB"))
        image = np.array(
            Image.fromarray(image).resize(
                (self.image_size[1], self.image_size[0]),
                Image.BILINEAR,
            )
        ).astype(np.float32) / 255.0

        # Load depth (supports local paths)
        depth_path = sample.get("depth")
        depth_loaded = False
        if depth_path:
            try:
                if os.path.exists(depth_path):
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
        else:
            depth = np.zeros(self.image_size, dtype=np.float32)

        result: Dict[str, Any] = {
            "image": image,
            "depth": depth,
            "image_id": sample.get("image_id", str(idx)),
        }

        # Load semantic labels if available
        if "semantic_label" in sample and os.path.exists(sample["semantic_label"]):
            label = np.array(
                Image.open(sample["semantic_label"]).resize(
                    (self.image_size[1], self.image_size[0]),
                    Image.NEAREST,
                )
            ).astype(np.int32)
            result["semantic_label"] = label

        # Load instance labels if available
        if "instance_label" in sample and os.path.exists(sample["instance_label"]):
            inst_label = np.array(
                Image.open(sample["instance_label"]).resize(
                    (self.image_size[1], self.image_size[0]),
                    Image.NEAREST,
                )
            ).astype(np.int32)
            result["instance_label"] = inst_label

        if self.transforms is not None:
            result = self.transforms(result)

        # Convert numpy arrays to torch tensors
        result = _numpy_to_torch(result)

        return result


def _numpy_to_torch(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy arrays in a sample dict to torch tensors.

    Images are transposed from (H, W, C) to (C, H, W) for PyTorch convention.
    Labels are converted to int64 tensors.

    Args:
        result: Sample dictionary with numpy arrays.

    Returns:
        Sample dictionary with torch tensors.
    """
    out: Dict[str, Any] = {}
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            if key == "image":
                # (H, W, C) -> (C, H, W) for PyTorch NCHW convention
                out[key] = torch.from_numpy(val.transpose(2, 0, 1)).float()
            elif "label" in key:
                out[key] = torch.from_numpy(val).long()
            else:
                out[key] = torch.from_numpy(val).float()
        else:
            out[key] = val
    return out


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

        if not os.path.isdir(img_dir):
            logger.warning(f"Cityscapes image dir not found: {img_dir}")
            return

        for city in sorted(os.listdir(img_dir)):
            city_img_dir = os.path.join(img_dir, city)
            if not os.path.isdir(city_img_dir):
                continue

            for fname in sorted(os.listdir(city_img_dir)):
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
                if os.path.exists(sem_path):
                    sample["semantic_label"] = sem_path

                # Instance label
                inst_path = os.path.join(
                    gt_dir, city, f"{base}_gtFine_instanceIds.png"
                )
                if os.path.exists(inst_path):
                    sample["instance_label"] = inst_path

                # Depth
                depth_path = os.path.join(
                    self.depth_dir, self.split, city, f"{base}.npy"
                )
                sample["depth"] = depth_path

                self.samples.append(sample)

        logger.info(
            f"Cityscapes {self.split}: loaded {len(self.samples)} samples"
        )


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
            logger.warning(f"COCO image dir not found: {img_dir}")
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

        logger.info(
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
            logger.warning(f"NYU image dir not found: {img_dir}")
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

        logger.info(
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
            logger.warning(f"PASCAL VOC split file not found: {split_file}")
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

        logger.info(
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
    )


def collate_batch(
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Collate a list of samples into a batched dictionary.

    Compatible with ``torch.utils.data.DataLoader`` as a custom
    ``collate_fn``.  Torch tensors are stacked along a new leading
    batch dimension; non-tensor values (e.g., image IDs) are gathered
    into a list.

    Args:
        samples: List of sample dictionaries.

    Returns:
        Batched dictionary with stacked tensors.
    """
    batch: Dict[str, Any] = {}
    keys = samples[0].keys()

    for key in keys:
        values = [s[key] for s in samples]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], np.ndarray):
            batch[key] = torch.from_numpy(np.stack(values, axis=0))
        else:
            batch[key] = values

    return batch
