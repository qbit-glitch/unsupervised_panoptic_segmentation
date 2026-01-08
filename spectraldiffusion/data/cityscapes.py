"""
Cityscapes Dataset Loader for SpectralDiffusion

Urban scene panoptic segmentation dataset.
Target: mIoU > 37.2
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import numpy as np
from PIL import Image
import os
import json


class CityscapesDataset(Dataset):
    """
    Cityscapes dataset for panoptic segmentation.
    
    Contains 5,000 high-resolution urban scene images with fine annotations.
    
    Args:
        root_dir: Path to Cityscapes directory (base level)
        split: "train", "val", or "test"
        image_size: Target image size
        fine_annotations: Use fine (True) or coarse (False) annotations
    """
    
    # 19 evaluation classes for Cityscapes
    CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 'fence',
        'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
        'sky', 'person', 'rider', 'car', 'truck',
        'bus', 'train', 'motorcycle', 'bicycle'
    ]
    
    # Class ID mapping from Cityscapes to training IDs (0-18)
    LABEL_MAP = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4,
        17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
        23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
        28: 15, 31: 16, 32: 17, 33: 18
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (512, 1024),
        fine_annotations: bool = True,
        return_masks: bool = True,
    ):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.fine_annotations = fine_annotations
        self.return_masks = return_masks
        
        # Paths (standard Cityscapes structure)
        self.images_dir = self.root_dir / "leftImg8bit" / split
        if fine_annotations:
            self.labels_dir = self.root_dir / "gtFine" / split
        else:
            self.labels_dir = self.root_dir / "gtCoarse" / split
        
        # Get all images
        self.samples = self._load_samples()
        print(f"Cityscapes {split}: {len(self.samples)} images")
    
    def _load_samples(self) -> List[Dict]:
        """Load sample paths."""
        samples = []
        
        if not self.images_dir.exists():
            print(f"Warning: Images dir not found: {self.images_dir}")
            return samples
        
        # Iterate through city subdirs
        for city_dir in sorted(self.images_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            
            city = city_dir.name
            
            for img_file in sorted(city_dir.glob("*.png")):
                # Image: {city}_{seq}_{frame}_leftImg8bit.png
                # Label: {city}_{seq}_{frame}_gtFine_labelIds.png
                base_name = img_file.stem.replace("_leftImg8bit", "")
                
                label_suffix = "gtFine" if self.fine_annotations else "gtCoarse"
                label_file = self.labels_dir / city / f"{base_name}_{label_suffix}_labelIds.png"
                instance_file = self.labels_dir / city / f"{base_name}_{label_suffix}_instanceIds.png"
                
                samples.append({
                    'image': str(img_file),
                    'label': str(label_file) if label_file.exists() else None,
                    'instance': str(instance_file) if instance_file.exists() else None,
                    'city': city,
                    'name': base_name,
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image']).convert('RGB')
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1)  # [3, H, W]
        
        output = {
            'image': image,
            'idx': idx,
            'name': sample['name'],
            'city': sample['city'],
        }
        
        # Load semantic label
        if sample['label'] and os.path.exists(sample['label']):
            label = Image.open(sample['label'])
            label = label.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
            label = np.array(label)
            
            # Map to training IDs
            mapped_label = np.full_like(label, 255)  # 255 = ignore
            for orig_id, train_id in self.LABEL_MAP.items():
                mapped_label[label == orig_id] = train_id
            
            output['segmentation'] = torch.from_numpy(mapped_label).long()
        
        # Load instance mask for panoptic
        if self.return_masks and sample['instance'] and os.path.exists(sample['instance']):
            instance = Image.open(sample['instance'])
            instance = instance.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
            instance = np.array(instance).astype(np.int32)
            
            # Instance IDs: class_id * 1000 + instance_id
            unique_instances = np.unique(instance)
            max_instances = 24
            
            masks = torch.zeros(max_instances, *self.image_size)
            for i, inst_id in enumerate(unique_instances[:max_instances]):
                if inst_id > 0:  # Skip background
                    masks[i] = torch.from_numpy(instance == inst_id).float()
            
            output['mask'] = masks
        
        return output


def create_cityscapes_dataloader(
    root_dir: str,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 1024),
) -> DataLoader:
    """Create Cityscapes dataloader."""
    dataset = CityscapesDataset(
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
        drop_last=(split == "train"),
    )
