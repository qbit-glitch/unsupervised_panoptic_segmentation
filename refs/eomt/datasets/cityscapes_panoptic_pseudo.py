# ---------------------------------------------------------------
# Cityscapes panoptic LightningDataModule for EoMT, backed by CUPS-style
# pseudo-labels (DepthPro + DCFA + SIMCF-ABC, k=80 pseudo-classes).
#
# Reads images and pseudo-labels directly from disk (no zip), to match the
# layout produced by mbps_pytorch/generate_depth_guided_instances.py and the
# CUPS pseudo-label generator at refs/cups/cups/pseudo_labels/.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms


class _CityscapesPseudoDataset(torch.utils.data.Dataset):
    """Directory-backed Cityscapes panoptic pseudo-label dataset.

    Expected layout (relative to ``path``):
        path/leftImg8bit/{split}/<city>/<stem>_leftImg8bit.png
        pseudo_path/<stem>_leftImg8bit_semantic.png   (uint8, values in [0, num_classes))
        pseudo_path/<stem>_leftImg8bit_instance.png   (uint16, 0 = no instance)

    The instance map carries depth-guided thing instances; semantic IDs are
    the k=80 pseudo-classes after DCFA + SIMCF-ABC. We convert the (semantic,
    instance) pair into EoMT's per-segment target via :meth:`target_parser`.
    """

    def __init__(
        self,
        image_paths: List[Path],
        pseudo_path: Path,
        target_parser: Callable,
        min_mask_pixels: int = 64,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.pseudo_path = pseudo_path
        self.target_parser = target_parser
        self.min_mask_pixels = min_mask_pixels
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def _pseudo_paths(self, image_path: Path) -> Tuple[Path, Path]:
        stem = image_path.stem  # e.g. aachen_000000_000019_leftImg8bit
        return (
            self.pseudo_path / f"{stem}_semantic.png",
            self.pseudo_path / f"{stem}_instance.png",
        )

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        sem_path, inst_path = self._pseudo_paths(image_path)

        img = tv_tensors.Image(Image.open(image_path).convert("RGB"))
        # Pseudo-label PNGs are uint8 (semantic) and uint16 (instance). Convert
        # both via numpy → int64 before wrapping as a tv_tensors.Mask: torch
        # cannot ingest uint16 directly, and torchvision's PIL path goes via
        # `numpy.asarray` so it hits the same restriction.
        sem_np = np.asarray(Image.open(sem_path), dtype=np.int64)
        inst_np = np.asarray(Image.open(inst_path), dtype=np.int64)
        sem = tv_tensors.Mask(torch.from_numpy(sem_np).unsqueeze(0))
        inst = tv_tensors.Mask(torch.from_numpy(inst_np).unsqueeze(0))

        if img.shape[-2:] != sem.shape[-2:]:
            sem = F.resize(sem, list(img.shape[-2:]), interpolation=F.InterpolationMode.NEAREST)
            inst = F.resize(inst, list(img.shape[-2:]), interpolation=F.InterpolationMode.NEAREST)

        masks, labels, is_crowd = self.target_parser(
            semantic=sem[0],
            instance=inst[0],
            min_mask_pixels=self.min_mask_pixels,
        )

        if len(masks) == 0:
            # Should not happen — fall back to the next sample to avoid empty batches.
            return self[(index + 1) % len(self)]

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class _ResizeOnlyTransform:
    """Resize (img, target) to a fixed size — no augmentation.

    Used in CUPS-exact Stage-3 self-training: the dataloader must hand the
    teacher a CLEAN image (CUPS CityscapesSelfTraining resizes 1024x2048 by
    0.625 to 640x1280); all augmentation happens inside training_step after
    teacher labelling.
    """

    def __init__(self, img_size: tuple[int, int]) -> None:
        self.img_size = list(img_size)

    def __call__(self, img, target):
        img = F.resize(img, self.img_size, interpolation=F.InterpolationMode.BILINEAR)
        masks = F.resize(
            target["masks"],
            self.img_size,
            interpolation=F.InterpolationMode.NEAREST,
        )
        valid = masks.flatten(1).sum(dim=1) > 0
        target = {
            "masks": tv_tensors.Mask(masks[valid]),
            "labels": target["labels"][valid],
            "is_crowd": target["is_crowd"][valid],
        }
        return img, target


class CityscapesPanopticPseudo(LightningDataModule):
    """EoMT-compatible data module for CUPS-style Cityscapes pseudo-labels.

    Train split is the full pseudo-label set (typically 2975 images across the
    18 train cities). Validation is a deterministic held-out slice of the same
    set, so loss-based val tracks training quality without requiring real GT.
    Final PQ evaluation against Cityscapes GT must be done offline with a
    Hungarian mapping (see scripts/evaluate_*.py).
    """

    def __init__(
        self,
        path: str,
        pseudo_path: str,
        num_classes: int = 80,
        stuff_classes: Optional[List[int]] = None,
        num_workers: int = 4,
        batch_size: int = 2,
        img_size: tuple[int, int] = (640, 640),
        color_jitter_enabled: bool = True,
        scale_range: tuple[float, float] = (0.5, 2.0),
        train_cities: Optional[List[str]] = None,
        val_holdout_count: int = 100,
        min_mask_pixels: int = 64,
        check_empty_targets: bool = True,
        self_train_clean: bool = False,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.pseudo_path = Path(pseudo_path)
        self.num_classes = num_classes
        self.stuff_classes = list(stuff_classes or [])
        self.train_cities = train_cities
        self.val_holdout_count = val_holdout_count
        self.min_mask_pixels = min_mask_pixels

        if self_train_clean:
            # CUPS-exact Stage-3: clean resize-only train images; augmentation
            # happens inside training_step after the teacher labels the batch.
            self.transforms: Callable = _ResizeOnlyTransform(img_size)
        else:
            self.transforms = Transforms(
                img_size=img_size,
                color_jitter_enabled=color_jitter_enabled,
                scale_range=scale_range,
            )

    def _list_train_images(self) -> List[Path]:
        train_root = Path(self.path) / "leftImg8bit" / "train"
        cities = sorted(p.name for p in train_root.iterdir() if p.is_dir())
        if self.train_cities:
            cities = [c for c in cities if c in self.train_cities]

        images: List[Path] = []
        for city in cities:
            for img in sorted((train_root / city).glob("*_leftImg8bit.png")):
                sem = self.pseudo_path / f"{img.stem}_semantic.png"
                inst = self.pseudo_path / f"{img.stem}_instance.png"
                if sem.exists() and inst.exists():
                    images.append(img)
        return images

    def target_parser(
        self,
        semantic: torch.Tensor,
        instance: torch.Tensor,
        min_mask_pixels: int,
    ) -> Tuple[List[torch.Tensor], List[int], List[bool]]:
        """Convert a (semantic, instance) pseudo-label pair into per-segment masks.

        Each nonzero instance ID contributes one thing-mask whose label is the
        majority pseudo-class inside the mask. Each unique pseudo-class in the
        instance==0 region contributes one stuff-mask covering that class only.

        We deliberately filter masks below ``min_mask_pixels`` to avoid teaching
        the model to imitate tiny depth-guided artefacts.
        """
        masks: List[torch.Tensor] = []
        labels: List[int] = []

        stuff_region = instance == 0

        for inst_id in instance.unique().tolist():
            if inst_id == 0:
                continue
            m = instance == inst_id
            if int(m.sum()) < min_mask_pixels:
                continue
            sem_in_mask = semantic[m]
            label = int(torch.mode(sem_in_mask).values.item())
            if not 0 <= label < self.num_classes:
                continue
            masks.append(m)
            labels.append(label)

        for cls in semantic[stuff_region].unique().tolist():
            cls_int = int(cls)
            if not 0 <= cls_int < self.num_classes:
                continue
            m = stuff_region & (semantic == cls_int)
            if int(m.sum()) < min_mask_pixels:
                continue
            masks.append(m)
            labels.append(cls_int)

        return masks, labels, [False] * len(masks)

    def setup(self, stage: Union[str, None] = None) -> "CityscapesPanopticPseudo":
        all_images = self._list_train_images()
        if len(all_images) == 0:
            raise RuntimeError(
                f"No training images found under {self.path}/leftImg8bit/train "
                f"with matching pseudo-labels under {self.pseudo_path}"
            )

        # Deterministic held-out val slice: every Nth image, distributed across cities.
        n = max(1, len(all_images) // max(1, self.val_holdout_count))
        val_images = all_images[::n][: self.val_holdout_count]
        val_set = set(map(str, val_images))
        train_images = [p for p in all_images if str(p) not in val_set]

        self.train_dataset = _CityscapesPseudoDataset(
            image_paths=train_images,
            pseudo_path=self.pseudo_path,
            target_parser=self.target_parser,
            min_mask_pixels=self.min_mask_pixels,
            transforms=self.transforms,
        )
        self.val_dataset = _CityscapesPseudoDataset(
            image_paths=val_images,
            pseudo_path=self.pseudo_path,
            target_parser=self.target_parser,
            min_mask_pixels=self.min_mask_pixels,
            transforms=None,
        )
        return self

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
