"""
nuScenes Dataset Loader for SpectralDiffusion

Autonomous driving panoptic segmentation dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import numpy as np
from PIL import Image
import os
import json


class NuScenesPanopticDataset(Dataset):
    """
    nuScenes panoptic segmentation dataset.
    
    Contains driving scenes with panoptic annotations.
    
    Args:
        root_dir: Path to nuScenes-panoptic directory
        split: "train" or "val"
        image_size: Target image size
        max_instances: Maximum number of instances
    """
    
    # nuScenes-panoptic classes (16 stuff + 10 thing = 26 total)
    STUFF_CLASSES = [
        'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
        'manmade', 'vegetation'
    ]
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (448, 800),
        max_instances: int = 32,
    ):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.max_instances = max_instances
        
        # Find panoptic data directory
        self.panoptic_dir = self._find_panoptic_dir()
        
        # Load samples
        self.samples = self._load_samples()
        print(f"nuScenes {split}: {len(self.samples)} samples")
    
    def _find_panoptic_dir(self) -> Path:
        """Find the panoptic data directory."""
        # Try common locations
        candidates = [
            self.root_dir / "nuScenes-panoptic-v1.0-all",
            self.root_dir / "panoptic",
            self.root_dir,
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        return self.root_dir
    
    def _load_samples(self) -> List[Dict]:
        """Load sample paths."""
        samples = []
        
        # nuScenes panoptic structure varies - handle both formats
        # Format 1: panoptic/{scene_id}/{frame}.png
        # Format 2: flat directory with all PNGs
        
        panoptic_files = list(self.panoptic_dir.rglob("*.png"))
        
        if not panoptic_files:
            print(f"Warning: No panoptic files found in {self.panoptic_dir}")
            return samples
        
        # Filter by split (train uses 75%, val uses 25% based on scene order)
        num_total = len(panoptic_files)
        split_idx = int(0.75 * num_total)
        
        if self.split == "train":
            panoptic_files = panoptic_files[:split_idx]
        else:
            panoptic_files = panoptic_files[split_idx:]
        
        for panoptic_file in panoptic_files:
            samples.append({
                'panoptic': str(panoptic_file),
                'name': panoptic_file.stem,
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load panoptic mask
        panoptic_img = Image.open(sample['panoptic'])
        panoptic_img = panoptic_img.resize(
            (self.image_size[1], self.image_size[0]), 
            Image.NEAREST
        )
        panoptic = np.array(panoptic_img)
        
        # nuScenes panoptic is RGB encoded: R + G*256 + B*256*256
        if len(panoptic.shape) == 3:
            panoptic_ids = (
                panoptic[:, :, 0].astype(np.int32) +
                panoptic[:, :, 1].astype(np.int32) * 256 +
                panoptic[:, :, 2].astype(np.int32) * 256 * 256
            )
        else:
            panoptic_ids = panoptic.astype(np.int32)
        
        # Create instance masks
        unique_ids = np.unique(panoptic_ids)
        masks = torch.zeros(self.max_instances, *self.image_size)
        
        for i, seg_id in enumerate(unique_ids[:self.max_instances]):
            masks[i] = torch.from_numpy(panoptic_ids == seg_id).float()
        
        # Create semantic segmentation (category = id // 1000)
        semantic = panoptic_ids // 1000
        
        # Generate synthetic image from masks (placeholder for testing)
        # In real use, you'd load the actual camera image
        image = self._generate_synthetic_image(masks)
        
        return {
            'image': image,
            'mask': masks,
            'segmentation': torch.from_numpy(semantic).long(),
            'idx': idx,
            'name': sample['name'],
        }
    
    def _generate_synthetic_image(self, masks: torch.Tensor) -> torch.Tensor:
        """Generate a synthetic colored image from masks for testing."""
        H, W = self.image_size
        image = torch.zeros(3, H, W)
        
        # Random colors for each slot
        colors = torch.rand(self.max_instances, 3)
        
        for i in range(self.max_instances):
            mask = masks[i]
            for c in range(3):
                image[c] += mask * colors[i, c]
        
        # Normalize
        image = image.clamp(0, 1)
        
        return image


def create_nuscenes_dataloader(
    root_dir: str,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (448, 800),
) -> DataLoader:
    """Create nuScenes panoptic dataloader."""
    dataset = NuScenesPanopticDataset(
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
