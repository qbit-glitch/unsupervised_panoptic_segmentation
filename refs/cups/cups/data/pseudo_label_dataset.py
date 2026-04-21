from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.structures import BitMasks, Boxes, Instances
from kornia.augmentation import AugmentationSequential, CenterCrop, PadTo
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from cups.data.utils import (
    CITYSCAPES_TRAINING_FILES,
    get_bounding_boxes,
    instances_to_masks,
    load_image,
    load_label,
)
from cups.scene_flow_2_se3 import remap_ids

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

__all__: Tuple[str, ...] = ("PseudoLabelDataset",)


class PseudoLabelDataset(Dataset):
    """This class implements the panoptic pseudo label dataset."""

    def __init__(
        self,
        root: str,
        root_pseudo: str,
        ground_truth_root: str | None = None,
        crop_resolution: Tuple[int, int] = (608, 1104),
        return_detectron2_format: bool = True,
        return_ground_truth: bool = False,
        dataset: str = "cityscapes",
        ground_truth_scale: float = 0.625,
        thing_stuff_threshold: float = 0.05,
        void_id: int = 255,
        augmentations: AugmentationSequential | None = None,
        only_use_training_samples: bool = False,
        ignore_unknown_thing_regions: bool = True,
        ignore_object_proposal_stuff_regions: bool = True,
        only_use_non_empty_samples: bool = False,
        depth_subdir: str = "",
        num_pseudo_classes: int = 80,
        load_pseudo_onehot: bool = False,
        instance_aware_crop: bool = False,
        instance_aware_crop_prob: float = 0.5,
    ) -> None:
        """Constructor method.

        Notes:
            return_ground_truth is not supported for Detectron2 format.

        Args:
            root (str): Path to dataset.
            root_pseudo (str): Path to pseudo labels.
            ground_truth_root (str | None): Path to GT labels.
            crop_resolution (Tuple[int, int]): Crop target resolution.
            return_detectron2_format (bool): If true we return the Detectron2 training format.
            return_ground_truth (bool): If true GT is also loaded.
            dataset (str): Dataset to be used. Either cityscapes or kitti. Default: cityscapes.
            ground_truth_scale (float): Scale used to rescale the ground truth labels to image resolution.
            void_id (int): Void ID to be used. Default 255.0
            augmentations (AugmentationSequential): Standard photometric augmentations.
            only_use_training_samples (bool): If true we only use the samples from the Cityscapes training set.
            ignore_unknown_thing_regions (bool): If true thing regions that are not occupied by a OP is ignored.
            ignore_object_proposal_stuff_regions (bool): If true stuff object proposal regions are set to seam. ignore.
            only_use_non_empty_samples (bool): If true only samples that have > 0 object proposals are loaded.
            instance_aware_crop (bool): If true bias random crops toward thing instances.
            instance_aware_crop_prob (float): Probability of using a thing-biased crop vs uniform random.
        """
        # Call super constructor
        super(PseudoLabelDataset, self).__init__()
        # Check parameters
        if return_ground_truth:
            assert ground_truth_root is not None, "If ground truth should be loaded GT path must be given."
        assert dataset in ("cityscapes", "cityscapes_val", "kitti"), "Provided dataset is not supported."
        if return_ground_truth:
            assert augmentations is None, "If GT should be returned no augmentations are supported."
        # Save parameters
        self.return_detectron2_format: bool = return_detectron2_format
        self.return_ground_truth: bool = return_ground_truth
        self.ground_truth_root: str | None = ground_truth_root
        self.dataset: str = dataset
        self.ground_truth_scale: float = ground_truth_scale
        self.thing_stuff_threshold: float = thing_stuff_threshold
        self.void_id: int = void_id
        self.augmentations: AugmentationSequential | None = augmentations
        self.ignore_unknown_thing_regions: bool = ignore_unknown_thing_regions
        self.only_use_training_samples: bool = only_use_training_samples
        self.ignore_object_proposal_stuff_regions: bool = ignore_object_proposal_stuff_regions
        self.instance_aware_crop: bool = instance_aware_crop
        self.instance_aware_crop_prob: float = instance_aware_crop_prob
        # Init crop module
        self.crop_module: nn.Module = CenterCrop(size=crop_resolution, keepdim=True)
        self.pad_module: nn.Module = (
            PadTo(size=crop_resolution, pad_mode="constant", pad_value=0) if dataset == "kitti" else None
        )
        # Get paths
        if dataset == "cityscapes":
            self.image_paths, self.instance_paths, self.semantic_paths = self.get_cityscapes_paths(
                root=root, root_pseudo=root_pseudo
            )
        elif dataset == "cityscapes_val":
            self.image_paths, self.instance_paths, self.semantic_paths = self.get_cityscapes_paths(
                root=root, root_pseudo=root_pseudo, split="val"
            )
        else:
            self.image_paths, self.instance_paths, self.semantic_paths = self.get_kitti_paths(
                root=root, root_pseudo=root_pseudo
            )
        # Only use training samples if utilized
        if only_use_training_samples and (self.dataset == "cityscapes"):
            self.image_paths = [
                path
                for path in self.image_paths
                if path.split("/")[-1].replace("_leftImg8bit.png", "") in CITYSCAPES_TRAINING_FILES
            ]
            self.instance_paths = [
                path
                for path in self.instance_paths
                if path.split("/")[-1].replace("_leftImg8bit_instance.png", "") in CITYSCAPES_TRAINING_FILES
            ]
            self.semantic_paths = [
                path
                for path in self.semantic_paths
                if path.split("/")[-1].replace("_leftImg8bit_semantic.png", "") in CITYSCAPES_TRAINING_FILES
            ]
        # Get thing stuff class split
        tensor_files = []
        for p in os.listdir(root_pseudo):
            # Skip AppleDouble resource forks and hidden files
            if p.startswith(".") or not p.endswith(".pt"):
                continue
            path = os.path.join(root_pseudo, p)
            try:
                tensor_files.append(torch.load(path, weights_only=False))
            except Exception as e:
                log.error("Failed to load pseudo-label file %s: %s", path, e)
                raise RuntimeError(
                    f"Corrupted pseudo-label file: {path}. "
                    f"Re-download the pseudo-labels and re-extract. "
                    f"Original error: {e}"
                ) from e
        class_distribution_instances = torch.stack(
            [t["distribution inside object proposals"] for t in tensor_files]
        ).sum(dim=0)
        class_distribution = torch.stack([t["distribution all pixels"] for t in tensor_files]).sum(dim=0)
        # Compute split
        distribution, indices = torch.sort(class_distribution_instances / (class_distribution + 1e-06), descending=True)
        distribution = distribution / distribution.sum()
        above_threshold = distribution > thing_stuff_threshold
        if above_threshold.all():
            num_instance_pseudo_classes = len(distribution)
        else:
            num_instance_pseudo_classes = above_threshold.float().argmin()
        # Get thing and stuff split
        self.things_classes: Tuple[int, ...] = tuple(indices[:num_instance_pseudo_classes].tolist())
        self.stuff_classes: Tuple[int, ...] = tuple(indices[num_instance_pseudo_classes:].tolist())
        # Log thing and stuff pseudo classes
        print(f"Thing classes: {self.things_classes}")
        print(f"Stuff classes: {self.stuff_classes}")
        # Get class distribution [stuff classes + 1 (thing class frequency)].
        # This aggregated form is what the Cascade Mask R-CNN branch expects —
        # its semantic head has S+1 outputs (stuff + "thing placeholder") and
        # per-thing classification is delegated to the instance head.
        class_distribution = class_distribution / (class_distribution.sum() * class_distribution.numel())
        class_distribution_stuff = class_distribution[torch.tensor(self.stuff_classes, dtype=torch.long)]
        if len(self.things_classes) > 0:
            class_distribution_thing_sum = class_distribution[torch.tensor(self.things_classes, dtype=torch.long)].sum().view(1)
            class_distribution_thing_per = class_distribution[torch.tensor(self.things_classes, dtype=torch.long)]
        else:
            log.warning(
                "THING_STUFF_THRESHOLD=%.3f produced zero thing classes — "
                "all pseudo classes treated as stuff. Instance head will receive no supervision.",
                thing_stuff_threshold,
            )
            class_distribution_thing_sum = torch.zeros(1)
            class_distribution_thing_per = torch.tensor([])
        self.class_distribution: Tuple[float, ...] = tuple(
            torch.cat((class_distribution_stuff, class_distribution_thing_sum))
        )
        # Per-class distribution of length S+T in SetCriterion class_id order
        # (stuff first in stuff_classes order, thing next in things_classes order).
        # M2F's per-query classification needs a weight per real class — there is
        # no thing-placeholder — so the aggregated form above is the wrong shape
        # for it. This full-dim vector matches Mask2FormerPanoptic's combined
        # class space built by `_collect_targets`.
        self.class_distribution_full: Tuple[float, ...] = tuple(
            torch.cat((class_distribution_stuff, class_distribution_thing_per))
        )
        # Approach B: depth + pseudo-onehot for stuff KD
        self.root = root
        self.depth_subdir = depth_subdir
        self.num_pseudo_classes = num_pseudo_classes
        self.load_pseudo_onehot = load_pseudo_onehot
        self._split = "val" if "val" in self.dataset else "train"
        if only_use_non_empty_samples:
            self.omit_empty_labels()

    def omit_empty_labels(self) -> None:
        """Loads each instance label and omits the samples that do not include at least a single proposal."""
        # Init new paths
        image_paths = []
        instance_paths = []
        semantic_paths = []
        # Show info
        log.info("Find non-empty samples")
        # Iterate over samples
        for image_path, instance_path, semantic_path in zip(self.image_paths, self.instance_paths, self.semantic_paths):
            # Load instance label
            instance_pseudo_label: Tensor = load_label(instance_path)[0].long()
            # Check if we have at least one object proposal
            if instance_pseudo_label.unique().shape[0] > 1:
                image_paths.append(image_path)
                instance_paths.append(instance_path)
                semantic_paths.append(semantic_path)
        # Show info
        log.info("Find non-empty samples finished")
        # Overwrite paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths
        self.semantic_paths = semantic_paths

    def _add_depth_and_onehot(self, output: Dict[str, Any], index: int) -> None:
        """Add depth map and pseudo-onehot to output dict (in-place).

        Args:
            output: Detectron2-format output dict (must contain 'sem_seg').
            index: Sample index for resolving file paths.
        """
        if self.depth_subdir:
            stem = os.path.basename(self.semantic_paths[index]).replace(
                "_leftImg8bit_semantic.png", ""
            )
            city = stem.split("_")[0]
            depth_path = os.path.join(
                self.root, self.depth_subdir, self._split, city, f"{stem}.npy"
            )
            if os.path.exists(depth_path):
                depth_raw = np.load(depth_path).astype(np.float32)
                depth_t = torch.from_numpy(depth_raw).unsqueeze(0)  # (1, H, W)
                d_min, d_max = depth_t.min(), depth_t.max()
                if d_max > d_min:
                    depth_t = (depth_t - d_min) / (d_max - d_min)
                # Resize to match crop resolution, then center-crop
                crop_h, crop_w = self.crop_module.size
                depth_t = F.interpolate(
                    depth_t.unsqueeze(0),
                    size=(crop_h, crop_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                depth_t = self.crop_module(depth_t.unsqueeze(0)).squeeze(0)
                output["depth"] = depth_t

        if self.load_pseudo_onehot:
            sem_labels = output["sem_seg"]  # (H, W), values 0=thing, 1..S=stuff, 255=void
            num_classes = len(self.stuff_classes) + 1  # match model output channels
            valid_mask = sem_labels < num_classes
            clamped = sem_labels.clone()
            clamped[~valid_mask] = 0
            onehot = F.one_hot(
                clamped.long(), num_classes=num_classes
            ).permute(2, 0, 1).float()  # (C, H, W)
            onehot[:, ~valid_mask] = 0.0
            output["pseudo_onehot"] = onehot

    def get_kitti_paths(self, root: str, root_pseudo: str) -> Tuple[List[str], List[str], List[str]]:
        """Loads the KITTI path for both the pseudo labels and images.

        Args:
            root (str): Path to dataset.
            root_pseudo (str): Path to pseudo labels.

        Returns:
            image_paths (List[str]): List of image paths.
            instance_paths (List[str]): List of pseudo instance paths.
            semantic_paths (List[str]): List of pseudo semantic paths.
        """
        # Get both semantic and instance pseudo label path
        # Skip AppleDouble resource forks (._*) and hidden files
        instance_paths = [
            file for file in sorted(os.listdir(root_pseudo))
            if "_instance.png" in file and not file.startswith(".")
        ]
        semantic_paths = [
            file for file in sorted(os.listdir(root_pseudo))
            if "_semantic.png" in file and not file.startswith(".")
        ]
        # 2011_09_26_drive_0001_sync_0000000000_instance.png
        # Get corresponding image path
        image_paths = []
        for file in instance_paths:
            # Get day
            day: str = file[:10]
            # Get clip
            clip: str = file[:26]
            # Make image path with day and clip name
            image_paths.append(
                os.path.join(root, day, clip, "image_02", "data", file.replace("_instance.png", ".png")[-14:])
            )
        # Add pseudo root to paths
        instance_paths = [os.path.join(root_pseudo, file) for file in instance_paths]
        semantic_paths = [os.path.join(root_pseudo, file) for file in semantic_paths]
        return image_paths, instance_paths, semantic_paths

    def get_cityscapes_paths(
        self, root: str, root_pseudo: str, split: str = "train"
    ) -> Tuple[List[str], List[str], List[str]]:
        """Loads the Cityscapes path for both the pseudo labels and images.

        Args:
            root (str): Path to dataset.
            root_pseudo (str): Path to pseudo labels.

        Returns:
            image_paths (List[str]): List of image paths.
            instance_paths (List[str]): List of pseudo instance paths.
            semantic_paths (List[str]): List of pseudo semantic paths.
        """
        # Get both semantic and instance pseudo label path
        # Skip AppleDouble resource forks (._*) and hidden files
        instance_paths = [
            file for file in sorted(os.listdir(root_pseudo))
            if "_instance.png" in file and not file.startswith(".")
        ]
        semantic_paths = [
            file for file in sorted(os.listdir(root_pseudo))
            if "_semantic.png" in file and not file.startswith(".")
        ]
        # Get corresponding image path
        image_paths = []
        for file in instance_paths:
            # Get city name
            city: str = file.split("_")[0]
            # Make image path with city
            image_paths.append(
                os.path.join(root, "leftImg8bit_sequence", split, city, file.replace("_instance.png", ".png"))
            )
        # Add pseudo root to paths
        instance_paths = [os.path.join(root_pseudo, file) for file in instance_paths]
        semantic_paths = [os.path.join(root_pseudo, file) for file in semantic_paths]
        return image_paths, instance_paths, semantic_paths

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            length (int): Length of the dataset.
        """
        length: int = len(self.image_paths)
        return length

    def load_cityscapes_ground_truth(self, image_path: str) -> Tuple[Tensor | None, Tensor | None]:
        """Function loads ground truth Cityscapes label if available.

        Args:
            image_path (str): Image path.

        Returns:
            semantic_gt (Tensor): Semantic segmentation GT of the shape [H, W].
            instance_gt (Tensor): Instance segmentation GT of the shape [H, W].
        """
        # Get unique file name
        file_name: str = image_path.split("/")[-1].replace("_leftImg8bit_semantic.png", "")
        # Get city name
        city: str = file_name.split("_")[0]
        # Get split based on city
        split: str = "train"
        train_cities: List[str] = os.listdir(os.path.join(self.ground_truth_root, "gtFine", "train"))  # type: ignore
        if city not in train_cities:
            split = "val"
            val_cities: List[str] = os.listdir(os.path.join(self.ground_truth_root, "gtFine", "val"))  # type: ignore
            if city not in val_cities:
                split = "test"
        # Get all labels for the city
        label_paths: List[str] = os.listdir(os.path.join(self.ground_truth_root, "gtFine", split, city))  # type: ignore
        # Check if label is available
        if not any([file_name in file for file in label_paths]):
            return None, None
        # Make file paths
        semantic_label_path: str = os.path.join(
            self.ground_truth_root, "gtFine", split, city, file_name + "_gtFine_labelIds.png"  # type: ignore
        )
        instance_label_path: str = os.path.join(
            self.ground_truth_root, "gtFine", split, city, file_name + "_gtFine_instanceIds.png"  # type: ignore
        )
        # Load semantic and instance label
        semantic_label = to_tensor(Image.open(semantic_label_path))
        instance_label = to_tensor(Image.open(instance_label_path))
        # Scale labels
        semantic_label = (255.0 * semantic_label).long()
        instance_label = instance_label.long()
        # Remap semantic classes to 27 classes plus void class
        weight: Tensor = torch.ones(34, dtype=torch.long) * self.void_id
        weight[7:] = torch.arange(start=0, end=27)
        semantic_label = torch.embedding(indices=semantic_label, weight=weight[..., None]).squeeze(dim=0)[..., 0]
        # Omit non-instance IDs, instance IDs are constructed by semantic ID (two digits and >= 24)
        # + instance ID (three digits starting from 000). We shift the instance label by 23999 to prevent building a
        # large weight matrix
        instance_label = torch.where(instance_label > 24000, instance_label - 23999, 0)
        # Resize labels
        semantic_label = F.interpolate(
            semantic_label[None, None].float(), scale_factor=self.ground_truth_scale, mode="nearest"
        ).long()
        instance_label = F.interpolate(
            instance_label[None].float(), scale_factor=self.ground_truth_scale, mode="nearest"
        ).long()
        return semantic_label, instance_label

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Returns on instance of the dataset.

        Args:
            index (int): Index of the sample to be loaded.

        Returns:
            output (Dict[str, Tensor]): Dict of loaded data (optionally in Detectron2 format).
        """
        # Load data
        image: Tensor = load_image(self.image_paths[index])
        semantic_pseudo_label: Tensor = load_label(self.semantic_paths[index])[0].long()
        instance_pseudo_label: Tensor = load_label(self.instance_paths[index])[0].long()
        # Scale image and pseudo-labels to training resolution
        image = F.interpolate(image[None], scale_factor=self.ground_truth_scale, mode="bilinear")[0]
        semantic_pseudo_label = F.interpolate(
            semantic_pseudo_label[None, None].float(), scale_factor=self.ground_truth_scale, mode="nearest"
        )[0, 0].long()
        instance_pseudo_label = F.interpolate(
            instance_pseudo_label[None, None].float(), scale_factor=self.ground_truth_scale, mode="nearest"
        )[0, 0].long()
        # Spatial crop (standard center/random or instance-aware)
        if self.pad_module is not None:
            image = self.pad_module(image[None])[0]
        if self.instance_aware_crop:
            image, semantic_pseudo_label, instance_pseudo_label = self._instance_aware_crop(
                image, semantic_pseudo_label, instance_pseudo_label
            )
        else:
            image = self.crop_module(image[None])
            semantic_pseudo_label = self.crop_module(semantic_pseudo_label[None, None].float())
            instance_pseudo_label = self.crop_module(instance_pseudo_label[None, None].float())
        # Perform augmentations
        if self.augmentations is not None:
            image, semantic_pseudo_label, instance_pseudo_label, valid_mask = self.augmentations(
                image, semantic_pseudo_label, instance_pseudo_label, torch.ones_like(instance_pseudo_label)
            )
            # Clip image to valid pixel range
            image = image.clip(min=0.0, max=1.0)
            # Ignore invalid regions
            semantic_pseudo_label[valid_mask != 1.0] = self.void_id
        semantic_pseudo_label = semantic_pseudo_label.long()
        instance_pseudo_label = instance_pseudo_label.long()
        # Remap ids
        instance_pseudo_label = remap_ids(instance_pseudo_label[0, 0])[None, None]
        # If utilized load GT
        if self.return_ground_truth and not self.return_detectron2_format and self.only_use_training_samples:
            # Load GT
            semantic_gt, instance_gt = self.load_cityscapes_ground_truth(image_path=self.semantic_paths[index])
            # Crop GT
            if semantic_gt is not None:
                semantic_gt = self.crop_module(semantic_gt.float()).long().squeeze()
                instance_gt = self.crop_module(instance_gt.float()).long().squeeze()  # type: ignore
                # Remap instance IDs to 0, 1, 2, ..., N
                instance_gt = remap_ids(instance_gt)
            output: Dict[str, Tensor] = {
                "image": image.squeeze(),
                "semantic_pseudo": semantic_pseudo_label.squeeze(),
                "instance_pseudo": instance_pseudo_label.squeeze(),
                "semantic_gt": semantic_gt,  # type: ignore
                "instance_gt": instance_gt,  # type: ignore
                "image_path": self.image_paths[index],  # type: ignore
            }
            return output
        # Return data in standard format
        if not self.return_detectron2_format:
            output = {
                "image": image.squeeze(),
                "semantic_pseudo": semantic_pseudo_label.squeeze(),
                "instance_pseudo": instance_pseudo_label.squeeze(),
                "image_path": self.image_paths[index],  # type: ignore
            }
            return output
        # Get stuff semantic segmentation
        weight = torch.ones(256, dtype=torch.long) * self.void_id
        weight[torch.tensor(self.things_classes)] = 0
        weight[torch.tensor(self.stuff_classes)] = torch.arange(len(self.stuff_classes), dtype=torch.long) + 1
        semantic_pseudo_label_stuff: Tensor = torch.embedding(weight.reshape(-1, 1), semantic_pseudo_label).squeeze()
        # Squeeze instance label
        instance_pseudo_label = instance_pseudo_label.squeeze()
        # Set semantic to ignore if we have a thing region without any object proposal
        if self.ignore_unknown_thing_regions:
            semantic_pseudo_label_stuff[
                torch.logical_and(instance_pseudo_label == 0, semantic_pseudo_label_stuff == 0)
            ] = self.void_id
        # Get binary instance masks
        instance_masks: Tensor = instances_to_masks(instance_pseudo_label)
        # Get semantic classes of objects
        object_semantics: Tensor = (semantic_pseudo_label[0] * instance_masks).amax(dim=(-1, -2))
        # Check if object semantics are in things classes
        object_semantics_valid: Tensor = torch.isin(object_semantics, torch.tensor(self.things_classes))
        # If there is no valid object we return empty instances
        if not object_semantics_valid.any().item():
            output = {
                "image": image.squeeze(),
                "sem_seg": semantic_pseudo_label_stuff,
                "instances": Instances(
                    image_size=tuple(image.shape[1:]),
                    gt_masks=BitMasks(torch.zeros(0, *image.shape[2:]).bool()),
                    gt_boxes=Boxes(torch.zeros(0, 4).long()),
                    gt_classes=torch.zeros(0).long(),
                ),
            }
            self._add_depth_and_onehot(output, index)
            return output
        # Omit invalid instance masks and get invalid masks
        instance_masks, invalid_instance_masks = (
            instance_masks[object_semantics_valid],
            instance_masks[~object_semantics_valid],
        )
        object_semantics = object_semantics[object_semantics_valid]
        # For invalid object proposals we set the semantic to ignore
        if self.ignore_object_proposal_stuff_regions:
            semantic_pseudo_label_stuff[invalid_instance_masks.any(dim=0)] = 255
        # Remap object semantics
        weight = torch.zeros(len(self.stuff_classes) + len(self.things_classes), dtype=torch.long)
        weight[torch.tensor(self.things_classes)] = torch.arange(len(self.things_classes), dtype=torch.long)
        object_semantics = torch.embedding(weight.reshape(-1, 1), object_semantics)[..., 0]
        # Remap instance pseudo label since we need it for getting the bounding boxes
        instance_pseudo_label = instance_pseudo_label.squeeze()
        instance_pseudo_label[~instance_masks.any(dim=0)] = 0
        # Make Detectron2 dict
        output = {
            "image": image.squeeze(),
            "sem_seg": semantic_pseudo_label_stuff,
            "instances": Instances(
                image_size=tuple(image.shape[1:]),
                gt_masks=BitMasks(instance_masks),
                gt_boxes=Boxes(get_bounding_boxes(instance_pseudo_label)),
                gt_classes=object_semantics,
            ),
        }
        self._add_depth_and_onehot(output, index)
        return output


    def _instance_aware_crop(
        self,
        image: Tensor,
        semantic: Tensor,
        instance: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Crop a random region, biasing toward areas containing thing instances.

        With probability ``instance_aware_crop_prob`` the crop is centered on
        a random thing instance's bounding box (with jitter). Otherwise a
        uniform random crop is used. The crop is then resized to
        ``self.crop_resolution``.

        Returns:
            (image, semantic, instance) tensors after crop+resize.
        """
        H, W = image.shape[-2:]
        crop_h, crop_w = self.crop_module.size
        # Determine crop box size (same as RandomResizedCrop default scale)
        scale_min, scale_max = 0.9, 1.0
        scale = torch.empty(1).uniform_(scale_min, scale_max).item()
        ch, cw = int(crop_h * scale), int(crop_w * scale)
        ch, cw = min(ch, H), min(cw, W)

        # Decide whether to bias toward a thing instance
        use_thing_bias = (
            torch.rand(1).item() < self.instance_aware_crop_prob
            and instance.unique().numel() > 1  # at least one non-zero instance id
        )

        if use_thing_bias:
            # Get bbox of a random thing instance
            inst_ids = instance.unique()
            inst_ids = inst_ids[inst_ids != 0]  # remove background
            if len(inst_ids) > 0:
                tid = int(inst_ids[torch.randint(0, len(inst_ids), (1,)).item()].item())
                ys, xs = (instance == tid).nonzero(as_tuple=True)
                y_min, y_max = int(ys.min().item()), int(ys.max().item())
                x_min, x_max = int(xs.min().item()), int(xs.max().item())
                # Center crop on instance bbox with jitter
                cy = (y_min + y_max) // 2
                cx = (x_min + x_max) // 2
                jitter_y = torch.randint(-ch // 4, ch // 4 + 1, (1,)).item()
                jitter_x = torch.randint(-cw // 4, cw // 4 + 1, (1,)).item()
                y0 = max(0, min(cy - ch // 2 + jitter_y, H - ch))
                x0 = max(0, min(cx - cw // 2 + jitter_x, W - cw))
            else:
                y0 = torch.randint(0, H - ch + 1, (1,)).item() if H > ch else 0
                x0 = torch.randint(0, W - cw + 1, (1,)).item() if W > cw else 0
        else:
            y0 = torch.randint(0, H - ch + 1, (1,)).item() if H > ch else 0
            x0 = torch.randint(0, W - cw + 1, (1,)).item() if W > cw else 0

        # Slice and resize
        image_c = image[:, y0 : y0 + ch, x0 : x0 + cw]
        sem_c = semantic[y0 : y0 + ch, x0 : x0 + cw]
        inst_c = instance[y0 : y0 + ch, x0 : x0 + cw]
        image_c = F.interpolate(image_c[None], size=(crop_h, crop_w), mode="bilinear", align_corners=False)[0]
        sem_c = F.interpolate(sem_c[None, None].float(), size=(crop_h, crop_w), mode="nearest")[0, 0].long()
        inst_c = F.interpolate(inst_c[None, None].float(), size=(crop_h, crop_w), mode="nearest")[0, 0].long()
        return image_c, sem_c, inst_c


if __name__ == "__main__":
    dataset = PseudoLabelDataset(
        root="/path_to_datasets/datasets/KITTI-raw",
        root_pseudo="/path_to_pseudo_labels/",
        dataset="cityscapes",
        ground_truth_scale=1.0,
        crop_resolution=(368, 1104),
        return_detectron2_format=False,
        return_ground_truth=False,
        only_use_training_samples=False,
        augmentations=None,
        thing_stuff_threshold=0.08,
    )

    dataset = PseudoLabelDataset(
        root="/path_to_datasets/datasets/Cityscapes",
        root_pseudo="/path_to_pseudo_labels/",
        dataset="cityscapes",
        ground_truth_scale=0.625,
        crop_resolution=(640, 1280),
        return_detectron2_format=False,
        return_ground_truth=False,
        only_use_training_samples=False,
        augmentations=None,
        thing_stuff_threshold=0.08,
    )
