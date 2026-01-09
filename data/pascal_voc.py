"""
PASCAL VOC Dataset Loader for SpectralDiffusion
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict
from pathlib import Path
import numpy as np
from PIL import Image
import os


class PASCALVOCDataset(Dataset):
    """
    PASCAL VOC 2012 dataset for semantic segmentation.
    
    Args:
        root_dir: Path to VOCdevkit/VOC2012 directory
        split: "train", "val", or "trainval"
        image_size: Target image size
        num_classes: Number of classes (21 for VOC)
    """
    
    # VOC class names
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
        'train', 'tvmonitor'
    ]
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (518, 518),
        num_classes: int = 21,
    ):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Paths
        self.images_dir = self.root_dir / "JPEGImages"
        self.masks_dir = self.root_dir / "SegmentationClass"
        self.sets_dir = self.root_dir / "ImageSets" / "Segmentation"
        
        # Load split file
        split_file = self.sets_dir / f"{split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                self.image_ids = [line.strip() for line in f.readlines()]
        else:
            # Fallback: use all images
            if self.images_dir.exists():
                self.image_ids = [
                    f.replace('.jpg', '') 
                    for f in os.listdir(self.images_dir)
                    if f.endswith('.jpg')
                ]
            else:
                self.image_ids = []
        
        print(f"PASCAL VOC {split}: {len(self.image_ids)} images")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_id = self.image_ids[idx]
        
        # Load image
        img_path = self.images_dir / f"{img_id}.jpg"
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1)
        
        output = {
            'image': image,
            'idx': idx,
            'image_id': img_id,
        }
        
        # Load segmentation mask
        mask_path = self.masks_dir / f"{img_id}.png"
        if mask_path.exists():
            mask = Image.open(mask_path)
            mask = mask.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
            mask = np.array(mask)
            
            # Handle boundary pixels (255 in VOC)
            mask[mask == 255] = 0
            
            output['segmentation'] = torch.from_numpy(mask).long()
        
        return output


def create_pascal_voc_dataloader(
    root_dir: str,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (518, 518),
) -> DataLoader:
    """Create PASCAL VOC dataloader."""
    dataset = PASCALVOCDataset(
        root_dir=root_dir,
        split=split,
        image_size=image_size,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )
