"""
COCO-Stuff Dataset Loader for SpectralDiffusion

Scene-centric segmentation with 171 classes (80 things + 91 stuff).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict
from pathlib import Path
import json
import numpy as np
from PIL import Image
import os


class COCOStuffDataset(Dataset):
    """
    COCO-Stuff 164K dataset for panoptic segmentation.
    
    Contains 164K images with pixel-level annotations for
    80 thing classes and 91 stuff classes.
    
    Args:
        root_dir: Path to COCO-Stuff directory
        split: "train" or "val"
        image_size: Target image size
        num_classes: Number of segmentation classes
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train2017",
        image_size: Tuple[int, int] = (518, 518),
        num_classes: int = 171,
    ):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Paths
        self.images_dir = self.root_dir / split
        self.annotations_file = self.root_dir / f"stuff_{split}.json"
        
        # Load annotations
        if self.annotations_file.exists():
            print(f"Loading annotations from {self.annotations_file}...")
            with open(self.annotations_file) as f:
                self.coco_data = json.load(f)
            
            # Index by image id
            self.images = {img['id']: img for img in self.coco_data['images']}
            
            # Group annotations by image
            self.anns_by_image = {}
            for ann in self.coco_data.get('annotations', []):
                img_id = ann['image_id']
                if img_id not in self.anns_by_image:
                    self.anns_by_image[img_id] = []
                self.anns_by_image[img_id].append(ann)
            
            self.image_ids = list(self.images.keys())
        else:
            # Fallback: just use available images
            print(f"Annotations not found, using image directory: {self.images_dir}")
            if self.images_dir.exists():
                self.image_files = sorted([
                    f for f in os.listdir(self.images_dir)
                    if f.endswith(('.jpg', '.png'))
                ])
                self.image_ids = list(range(len(self.image_files)))
            else:
                self.image_files = []
                self.image_ids = []
            self.images = {}
            self.anns_by_image = {}
        
        print(f"COCO-Stuff {split}: {len(self.image_ids)} images")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a COCO-Stuff sample.
        
        Returns:
            Dictionary with:
            - image: [3, H, W] normalized image
            - segmentation: [H, W] class labels
            - idx: Sample index
        """
        img_id = self.image_ids[idx]
        
        # Load image
        if img_id in self.images:
            img_info = self.images[img_id]
            img_path = self.images_dir / img_info['file_name']
        else:
            img_path = self.images_dir / self.image_files[idx]
        
        image = Image.open(img_path).convert('RGB')
        orig_size = image.size
        
        # Resize
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        
        # Convert to tensor
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1)
        
        output = {
            'image': image,
            'idx': idx,
            'image_id': img_id,
        }
        
        # Load segmentation
        seg = self._load_segmentation(img_id, orig_size)
        if seg is not None:
            output['segmentation'] = seg
        
        return output
    
    def _load_segmentation(
        self,
        img_id: int,
        orig_size: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        """Load segmentation mask for an image."""
        # Check for pre-computed masks
        if img_id in self.images:
            img_info = self.images[img_id]
            mask_name = img_info['file_name'].replace('.jpg', '.png')
        else:
            mask_name = self.image_files[img_id].replace('.jpg', '.png')
        
        mask_path = self.root_dir / 'annotations' / mask_name
        
        if mask_path.exists():
            mask = Image.open(mask_path)
            mask = mask.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
            mask = torch.from_numpy(np.array(mask)).long()
            return mask
        
        return None


class COCOPanopticDataset(Dataset):
    """
    COCO Panoptic dataset with instance and stuff segmentation.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train2017",
        image_size: Tuple[int, int] = (518, 518),
        max_instances: int = 24,
    ):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.max_instances = max_instances
        
        # Paths
        self.images_dir = self.root_dir / split
        self.panoptic_dir = self.root_dir / f"panoptic_{split}"
        self.annotations_file = self.root_dir / f"panoptic_{split}.json"
        
        # Load annotations
        self.annotations = {}
        if self.annotations_file.exists():
            with open(self.annotations_file) as f:
                data = json.load(f)
            
            self.images = {img['id']: img for img in data['images']}
            self.annotations = {ann['image_id']: ann for ann in data['annotations']}
            self.categories = {cat['id']: cat for cat in data['categories']}
            self.image_ids = list(self.images.keys())
        else:
            if self.images_dir.exists():
                self.image_files = sorted([
                    f for f in os.listdir(self.images_dir)
                    if f.endswith(('.jpg', '.png'))
                ])
                self.image_ids = list(range(len(self.image_files)))
            else:
                self.image_ids = []
        
        print(f"COCO Panoptic {split}: {len(self.image_ids)} images")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_id = self.image_ids[idx]
        
        # Load image
        if hasattr(self, 'images') and img_id in self.images:
            img_info = self.images[img_id]
            img_path = self.images_dir / img_info['file_name']
        else:
            img_path = self.images_dir / self.image_files[idx]
        
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1)
        
        output = {
            'image': image,
            'idx': idx,
        }
        
        # Load panoptic segmentation
        if img_id in self.annotations:
            panoptic_mask = self._load_panoptic(img_id)
            if panoptic_mask is not None:
                output['panoptic_mask'] = panoptic_mask
        
        return output
    
    def _load_panoptic(self, img_id: int) -> Optional[torch.Tensor]:
        """Load panoptic segmentation mask."""
        ann = self.annotations.get(img_id)
        if ann is None:
            return None
        
        mask_path = self.panoptic_dir / ann['file_name']
        
        if mask_path.exists():
            # Panoptic masks use RGB encoding: R + G*256 + B*256*256
            mask_img = Image.open(mask_path)
            mask_img = mask_img.resize(
                (self.image_size[1], self.image_size[0]),
                Image.NEAREST
            )
            mask = np.array(mask_img)
            
            # Decode segment IDs
            segment_ids = mask[:, :, 0] + mask[:, :, 1] * 256 + mask[:, :, 2] * 256 * 256
            
            # Convert to instance masks
            segments = ann.get('segments_info', [])
            instance_masks = torch.zeros(self.max_instances, *self.image_size)
            
            for i, seg in enumerate(segments[:self.max_instances]):
                seg_id = seg['id']
                instance_masks[i] = torch.from_numpy(segment_ids == seg_id).float()
            
            return instance_masks
        
        return None


def create_coco_dataloader(
    root_dir: str,
    split: str = "train2017",
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (518, 518),
) -> DataLoader:
    """Create COCO-Stuff dataloader."""
    dataset = COCOStuffDataset(
        root_dir=root_dir,
        split=split,
        image_size=image_size,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train2017"),
        num_workers=num_workers,
        pin_memory=True,
    )
