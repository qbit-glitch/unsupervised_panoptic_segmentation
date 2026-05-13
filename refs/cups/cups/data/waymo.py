import os
import sys
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import CenterCrop, PadTo
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

sys.path.append(os.getcwd())
from cups.data.utils import load_image
from cups.scene_flow_2_se3.utils import remap_ids

WAYMO_19_MISSING_CS_CLASSES: List[bool] = [
    True,
    True,
    True,
    False,
    False,
    True,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
]
WAYMO_7_MISSING_CS_CLASSES: List[bool] = [True, True, True, True, True, True, True]

# 27-class evaluation mask. True = class is present in Waymo GT (or has a
# proxy class that maps to it); False = absent from Waymo and excluded from
# the per-class average. Of the 8 Cityscapes-27 extras (parking, rail track,
# guard rail, bridge, tunnel, polegroup, caravan, trailer), only "trailer"
# (Waymo id 8) has GT support; the other seven are absent from Waymo's
# 30-class taxonomy.
WAYMO_27_MISSING_CS_CLASSES: List[bool] = [
    True,   # 0  road           (Waymo: road, lanemarker, road marker)
    True,   # 1  sidewalk       (Waymo: sidewalk)
    False,  # 2  parking        - absent
    False,  # 3  rail track     - absent
    True,   # 4  building       (Waymo: building)
    False,  # 5  wall           - absent
    False,  # 6  fence          - absent
    False,  # 7  guard rail     - absent
    False,  # 8  bridge         - absent
    False,  # 9  tunnel         - absent
    True,   # 10 pole           (Waymo: pole)
    False,  # 11 polegroup      - absent
    True,   # 12 traffic light  (Waymo: traffic light)
    True,   # 13 traffic sign   (Waymo: sign)
    True,   # 14 vegetation     (Waymo: vegetation)
    False,  # 15 terrain        - absent
    True,   # 16 sky            (Waymo: sky)
    True,   # 17 person         (Waymo: pedestrian)
    True,   # 18 rider          (Waymo: cyclist, motorcyclist)
    True,   # 19 car            (Waymo: car)
    True,   # 20 truck          (Waymo: truck)
    True,   # 21 bus            (Waymo: bus)
    False,  # 22 caravan        - absent
    True,   # 23 trailer        (Waymo: trailer - NEW vs. CS19)
    True,   # 24 train          (Waymo: other large vehicle proxy)
    True,   # 25 motorcycle     (Waymo: motorcycle)
    True,   # 26 bicycle        (Waymo: bicycle)
]

# CS19 train_id -> CS27 train_id (used to lift Waymo's to_cs19 to CS27 via a
# table lookup; trailer is handled as an explicit override below).
_CS19_TO_CS27: Dict[int, int] = {
    0: 0,   1: 1,   2: 4,   3: 5,   4: 6,   5: 10,  6: 12,  7: 13,
    8: 14,  9: 15,  10: 16, 11: 17, 12: 18, 13: 19, 14: 20, 15: 21,
    16: 24, 17: 25, 18: 26,
    255: 255,
}


def _paired_label_path(image_path: str, label_kind: str) -> str:
    """Return the paired semantic/instance label path for a Waymo image path."""
    direct_path = image_path.replace("image", label_kind)
    if os.path.exists(direct_path):
        return direct_path
    stem_path = Path(direct_path)
    for suffix in (".png", ".tif", ".tiff"):
        candidate = stem_path.with_suffix(suffix)
        if candidate.exists():
            return str(candidate)
    return direct_path


def waymo2cs_class_mapping(num_classes: int = 19, void_id: int = 255) -> Tensor:
    """Function returns the mapping from raw Waymo classes to the supported
    Cityscapes evaluation taxonomy (27, 19, or 7).

    Args:
        num_classes (int): Number of classes to be utilized. Supported: 27, 19, 7.
        void_id (int): Void id class to ignore parts.

    Returns:
        mapping_weights (Tensor): Mapping weights of the shape [num_waymo_labels, 1]
    """
    # Check input
    assert num_classes in (27, 19, 7), f"{num_classes} classes is not supported."
    # 27 classes case (CUPS in-domain protocol). Lift each Waymo label's CS19
    # train_id to CS27 via the _CS19_TO_CS27 table, with a special case for
    # "trailer" which is mapped to CS27 id 23 even though it is void in CS19.
    if num_classes == 27:
        cs27_ids = []
        for c in WAYMO_LABEL:
            if c.name == "trailer":
                cs27_ids.append(23)
            else:
                cs27_ids.append(_CS19_TO_CS27.get(int(c.to_cs19), void_id))
        weights: Tensor = torch.Tensor(cs27_ids).int()
        weights[weights == 254] = void_id
    # 19 classes case
    elif num_classes == 19:
        weights = torch.Tensor([c.to_cs19 for c in WAYMO_LABEL]).int()
        weights[weights == 254] = void_id
    # 7 parent classes
    else:
        weights = torch.Tensor([c.categoryId - 1 for c in WAYMO_LABEL]).int()
        weights[weights == 254] = void_id

    # Reshape weights to [num_waymo_labels, 1]
    weights = weights[..., None]
    return weights


# label and all meta information
# Code inspired by Cityscapes https://github.com/mcordts/cityscapesScripts
Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class
        "id",  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images An ID
        # of -1 means that this label does not have an ID and thus is ignored
        # when creating ground truth images (e.g. license plate). Do not modify
        # these IDs, since exactly these IDs are expected by the evaluation
        # server.
        "category",  # The name of the category that this label belongs to
        "categoryId",
        # The ID of this category. Used to create ground truth images
        # on category level.
        "color",  # The color of this label
        "to_cs19",  # mapping to 27 classes of cityscapes
    ],
)

WAYMO_LABEL = [
    Label("undefined", 0, "void", 0, [0, 0, 0], 255),
    Label("ego vehicle", 1, "void", 0, [102, 102, 102], 255),
    Label("car", 2, "vehicle", 7, [0, 0, 142], 13),
    Label("truck", 3, "vehicle", 7, [0, 0, 70], 14),
    Label("bus", 4, "vehicle", 7, [0, 60, 100], 15),
    Label("other large vehicle", 5, "vehicle", 7, [61, 133, 198], 16),
    Label("bicycle", 6, "vehicle", 7, [119, 11, 32], 18),
    Label("motorcycle", 7, "vehicle", 7, [0, 0, 230], 17),
    Label("trailer", 8, "vehicle", 7, [111, 168, 220], 255),
    Label("pedestrian", 9, "human", 6, [220, 20, 60], 11),
    Label("cyclist", 10, "human", 6, [255, 0, 0], 12),
    Label("motorcyclist", 11, "human", 6, [180, 0, 0], 12),
    Label("bird", 12, "void", 0, [127, 96, 0], 255),
    Label("ground animal", 13, "void", 0, [91, 15, 0], 255),
    Label("construction cone pole", 14, "void", 0, [230, 145, 56], 255),
    Label("pole", 15, "object", 3, [153, 153, 153], 5),
    Label("pedestrian object", 16, "void", 0, [234, 153, 153], 255),
    Label("sign", 17, "object", 3, [246, 178, 107], 7),
    Label("traffic light", 18, "object", 3, [250, 170, 30], 6),
    Label("building", 19, "construction", 2, [70, 70, 70], 2),
    Label("road", 20, "flat", 1, [128, 64, 128], 0),
    Label("lanemarker", 21, "void", 0, [234, 209, 220], 0),
    Label("road marker", 22, "void", 0, [217, 210, 233], 0),
    Label("sidewalk", 23, "flat", 1, [244, 35, 232], 1),
    Label("vegetation", 24, "nature", 4, [107, 142, 35], 8),
    Label("sky", 25, "sky", 5, [70, 130, 180], 10),
    Label("ground", 26, "void", 0, [102, 102, 102], 255),
    Label("dynamic", 27, "void", 0, [102, 102, 102], 255),
    Label("static", 28, "void", 0, [102, 102, 102], 255),
    Label("void", 29, "void", 0, [0, 0, 0], 255),
]


class WaymoPanopticValidation(Dataset):
    """This class implements the KITTI panoptic validation dataset."""

    def __init__(
        self,
        root: str,
        resize_scale: float = 0.6,
        crop_resolution: Tuple[int, int] = (640, 960),
        void_id: int = 255,
        num_classes: int = 19,
    ) -> None:
        """Constructor method.

        Args:
            root (str): Path to the dataset.
            resize_scale (float): Scale factor to be applied to the images and labels.
            crop_resolution (Tuple[int, int]): Crop resolution to be utilized after resizing. Default (368, 1240).
            void_id (int): Void ID to be utilized. Default 255.
            num_classes (int): Number of classes to be utilized. Default 27.
        """
        # Call super constructor
        super(WaymoPanopticValidation, self).__init__()
        # Save parameters
        self.resize_scale: float = resize_scale
        self.void_id: int = void_id
        # Init crop module
        self.crop_module: nn.Module = CenterCrop(size=crop_resolution, keepdim=True)
        # Init padding module
        self.pad_module: nn.Module = PadTo(size=crop_resolution, pad_mode="constant", pad_value=0)
        self.pad_module_semantic: nn.Module = PadTo(size=crop_resolution, pad_mode="constant", pad_value=self.void_id)
        # Get class mapping
        self.class_mapping: Tensor = waymo2cs_class_mapping(num_classes=num_classes, void_id=self.void_id)
        # Init list to store paths
        self.images = []
        # Get image paths
        scene_paths = [os.path.join(root, "validation", d) for d in os.listdir(os.path.join(root, "validation"))]
        for scene_path in scene_paths:
            frames_paths = [i for i in os.listdir(scene_path) if "image" in i]
            # frames_paths = sorted(frames_paths)
            for frame_path in frames_paths:
                # ignore frames without labels
                image_path = os.path.join(scene_path, frame_path)
                if not os.path.exists(_paired_label_path(image_path, "semantic")):
                    continue
                self.images.append(image_path)

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            length (int): Length of the dataset.
        """
        length: int = len(self.images)
        return length

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Method returns an instances of the dataset given its index.

        Args:
            index (int): Index of the sample.

        Returns:
            output (Dict[str, Tensor]): Sample containing the image, semantic, and instance segmentation.
        """
        # Get image left t=0 path
        image_0_l_path = self.images[index]
        # Load images
        image_0_l: Tensor = load_image(path=image_0_l_path)[None]
        # Resize images
        image_0_l = F.interpolate(image_0_l, scale_factor=self.resize_scale, mode="bilinear")
        # Load labels
        semantic_path = _paired_label_path(image_0_l_path, "semantic")
        instance_path = _paired_label_path(image_0_l_path, "instance")
        semantic_label = torch.Tensor(
            np.array(Image.open(semantic_path), dtype=int)
        ).long()
        instance_label = torch.Tensor(
            np.array(Image.open(instance_path), dtype=int)
        ).long()
        semantic_label = self.class_mapping[semantic_label].squeeze()
        # Resize labels
        semantic_label = F.interpolate(
            semantic_label[None][None].float(), scale_factor=self.resize_scale, mode="nearest"
        ).long()
        instance_label = F.interpolate(
            instance_label[None][None].float(), scale_factor=self.resize_scale, mode="nearest"
        ).long()
        # Pad data to ensure min size
        image_0_l = self.pad_module(image_0_l)
        semantic_label = self.pad_module_semantic(semantic_label.float()).long()
        instance_label = self.pad_module(instance_label.float()).long()
        # Crop data
        image_0_l = self.crop_module(image_0_l)
        semantic_label = self.crop_module(semantic_label.float()).long()
        instance_label = self.crop_module(instance_label.float()).long()
        # Ensure we don't have an object mask for ignore regions
        instance_label[semantic_label == self.void_id] = 0
        # Remap instance IDs to 0, 1, 2, ..., N
        instance_label = remap_ids(instance_label[0, 0])[None, None]
        # Make output dict
        output: Dict[str, Tensor] = {
            "image_0_l": image_0_l,
            "semantic_gt": semantic_label,
            "instance_gt": instance_label,
            "image_name": self.images[index].split("/")[-2]  # type: ignore
            + Path(self.images[index]).stem,  # type: ignore
        }
        return output


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from cups.visualization import plot_panoptic_segmentation_overlay

    # Visualize labels
    dataset = WaymoPanopticValidation(
        root="/path_to_datasets/waymo_preprocessed",
        resize_scale=0.5,
        crop_resolution=(640, 960),
        void_id=255,
        num_classes=7,
    )
    print("Length Dataset: ", len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for idx, output in enumerate(dataloader):
        plot_panoptic_segmentation_overlay(
            torch.stack((output["semantic_gt"][0, 0].squeeze(), output["instance_gt"][0, 0].squeeze()), dim=-1),
            output["image_0_l"][0].squeeze(),
            dataset="cityscapes_19",
        )
