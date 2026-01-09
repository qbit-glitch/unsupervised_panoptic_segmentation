"""
CLEVR Dataset Loader for SpectralDiffusion

Synthetic dataset for validating object discovery and slot attention.
Target: ARI > 0.92
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import json
import numpy as np
from PIL import Image
import os


class CLEVRDataset(Dataset):
    """
    CLEVR dataset loader for object discovery evaluation.
    
    CLEVR contains simple 3D scenes with primitive objects (spheres, cubes, cylinders).
    Each scene has 3-10 objects with ground truth masks.
    
    Args:
        root_dir: Path to CLEVR_v1.0 directory
        split: "train" or "val"
        image_size: Target image size
        max_objects: Maximum number of objects to consider
        return_masks: Whether to return object masks
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (240, 320),
        max_objects: int = 11,  # Max objects in CLEVR + background
        return_masks: bool = True,
    ):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.max_objects = max_objects
        self.return_masks = return_masks
        
        # Paths
        self.images_dir = self.root_dir / "images" / split
        self.scenes_dir = self.root_dir / "scenes"
        self.masks_dir = self.root_dir / "masks" / split  # If available
        
        # Load scene descriptions
        scenes_file = self.scenes_dir / f"CLEVR_{split}_scenes.json"
        if scenes_file.exists():
            with open(scenes_file) as f:
                scenes_data = json.load(f)
            self.scenes = scenes_data['scenes']
        else:
            self.scenes = None
        
        # Get image list
        if self.images_dir.exists():
            self.image_files = sorted([
                f for f in os.listdir(self.images_dir) 
                if f.endswith('.png')
            ])
        else:
            # Fallback: search for images
            self.image_files = []
            print(f"Warning: Images directory not found at {self.images_dir}")
        
        print(f"CLEVR {split}: {len(self.image_files)} images")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a CLEVR sample.
        
        Returns:
            Dictionary with:
            - image: [3, H, W] normalized image
            - mask: [K, H, W] per-object masks (if return_masks)
            - num_objects: Number of objects in scene
            - scene_info: Optional scene metadata
        """
        # Load image
        img_name = self.image_files[idx]
        img_path = self.images_dir / img_name
        
        image = Image.open(img_path).convert('RGB')
        
        # Resize
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        
        # Convert to tensor and normalize to [0, 1]
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
        
        output = {
            'image': image,
            'idx': torch.tensor(idx),
        }
        
        # Scene info - only store num_objects as tensor (collation-safe)
        if self.scenes is not None and idx < len(self.scenes):
            scene = self.scenes[idx]
            output['num_objects'] = torch.tensor(len(scene['objects']))
        else:
            output['num_objects'] = torch.tensor(0)
        
        # Load masks if available
        if self.return_masks:
            mask = self._load_masks(idx)
            if mask is not None:
                output['mask'] = mask
        
        return output
    
    def _load_masks(self, idx: int) -> Optional[torch.Tensor]:
        """Load object masks for a sample."""
        # Check for pre-computed masks
        img_name = self.image_files[idx]
        mask_name = img_name.replace('.png', '_mask.npy')
        mask_path = self.masks_dir / mask_name
        
        if mask_path.exists():
            masks = np.load(mask_path)
            masks = torch.from_numpy(masks).float()
            
            # Resize to target size
            masks = torch.nn.functional.interpolate(
                masks.unsqueeze(0),
                size=self.image_size,
                mode='nearest',
            ).squeeze(0)
            
            # Pad to max_objects
            K = masks.shape[0]
            if K < self.max_objects:
                padding = torch.zeros(self.max_objects - K, *self.image_size)
                masks = torch.cat([masks, padding], dim=0)
            elif K > self.max_objects:
                masks = masks[:self.max_objects]
            
            return masks
        
        return None


class CLEVRWithMasksDataset(Dataset):
    """
    CLEVR with ground truth masks from multi-object datasets.
    
    Uses the clevr_with_masks TFRecords format.
    """
    
    def __init__(
        self,
        tfrecord_path: str,
        image_size: Tuple[int, int] = (128, 128),
        max_objects: int = 11,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.max_objects = max_objects
        
        # Parse TFRecord
        self.samples = self._parse_tfrecord(tfrecord_path, max_samples)
        print(f"Loaded {len(self.samples)} samples from TFRecord")
    
    def _parse_tfrecord(
        self, 
        path: str, 
        max_samples: Optional[int],
    ) -> List[Dict]:
        """Parse TFRecord file (simplified, requires tensorflow)."""
        samples = []
        
        try:
            import tensorflow as tf
            
            dataset = tf.data.TFRecordDataset(path)
            
            feature_desc = {
                'image': tf.io.FixedLenFeature([], tf.string),
                'mask': tf.io.FixedLenFeature([], tf.string),
            }
            
            for i, record in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                
                example = tf.io.parse_single_example(record, feature_desc)
                
                image = tf.io.decode_raw(example['image'], tf.uint8)
                mask = tf.io.decode_raw(example['mask'], tf.uint8)
                
                samples.append({
                    'image': image.numpy(),
                    'mask': mask.numpy(),
                })
            
        except ImportError:
            print("TensorFlow not available. Using placeholder data.")
            # Create synthetic placeholder samples
            for i in range(max_samples or 1000):
                samples.append({
                    'image': np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
                    'mask': np.random.randint(0, 11, (128, 128), dtype=np.uint8),
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Process image
        image = sample['image']
        if image.ndim == 1:
            image = image.reshape(128, 128, 3)
        
        image = Image.fromarray(image)
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1)
        
        # Process mask
        mask = sample['mask']
        if mask.ndim == 1:
            mask = mask.reshape(128, 128)
        
        # Convert to one-hot
        mask_tensor = torch.from_numpy(mask).long()
        num_objects = mask_tensor.max().item() + 1
        
        masks = torch.zeros(self.max_objects, *self.image_size)
        for obj_id in range(num_objects):
            obj_mask = (mask_tensor == obj_id).float()
            # Resize
            obj_mask = torch.nn.functional.interpolate(
                obj_mask.unsqueeze(0).unsqueeze(0),
                size=self.image_size,
                mode='nearest',
            ).squeeze()
            if obj_id < self.max_objects:
                masks[obj_id] = obj_mask
        
        return {
            'image': image,
            'mask': masks,
            'num_objects': num_objects,
            'idx': idx,
        }


class SyntheticShapesDataset(Dataset):
    """
    Synthetic shapes dataset for quick validation.
    
    Generates simple 2D scenes with geometric shapes.
    Useful for debugging slot attention without CLEVR download.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: Tuple[int, int] = (128, 128),
        max_objects: int = 6,
        min_objects: int = 2,
    ):
        super().__init__()
        
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_objects = max_objects
        self.min_objects = min_objects
        
        # Pre-generate scenes for reproducibility
        np.random.seed(42)
        self.scenes = [self._generate_scene() for _ in range(num_samples)]
    
    def _generate_scene(self) -> Dict:
        """Generate a random scene with shapes."""
        H, W = self.image_size
        num_objects = np.random.randint(self.min_objects, self.max_objects + 1)
        
        image = np.zeros((H, W, 3), dtype=np.float32)
        masks = np.zeros((self.max_objects, H, W), dtype=np.float32)
        
        # Background
        bg_color = np.random.rand(3) * 0.3
        image[:, :] = bg_color
        masks[0, :, :] = 1.0  # Background mask
        
        for obj_idx in range(1, num_objects + 1):
            if obj_idx >= self.max_objects:
                break
            
            # Random shape: circle, square, triangle
            shape = np.random.choice(['circle', 'square', 'triangle'])
            
            # Random position
            cx = np.random.randint(20, W - 20)
            cy = np.random.randint(20, H - 20)
            size = np.random.randint(15, 30)
            
            # Random color
            color = np.random.rand(3) * 0.7 + 0.3
            
            # Draw shape
            y, x = np.ogrid[:H, :W]
            
            if shape == 'circle':
                mask = ((x - cx) ** 2 + (y - cy) ** 2) <= size ** 2
            elif shape == 'square':
                mask = (np.abs(x - cx) <= size) & (np.abs(y - cy) <= size)
            else:  # triangle
                mask = (y >= cy - size) & (y <= cy + size) & \
                       (np.abs(x - cx) <= (cy + size - y) * size / (2 * size))
            
            mask = mask.astype(np.float32)
            
            # Update image and masks
            for c in range(3):
                image[:, :, c] = image[:, :, c] * (1 - mask) + color[c] * mask
            
            masks[obj_idx] = mask
            masks[0] *= (1 - mask)  # Remove from background
        
        return {
            'image': image,
            'masks': masks,
            'num_objects': num_objects,
        }
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene = self.scenes[idx]
        
        image = torch.from_numpy(scene['image']).permute(2, 0, 1)
        masks = torch.from_numpy(scene['masks'])
        
        return {
            'image': image,
            'mask': masks,
            'num_objects': scene['num_objects'],
            'idx': idx,
        }


def create_clevr_dataloader(
    root_dir: str,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (240, 320),
) -> DataLoader:
    """Create CLEVR dataloader."""
    dataset = CLEVRDataset(
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


def create_synthetic_dataloader(
    num_samples: int = 1000,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (128, 128),
) -> DataLoader:
    """Create synthetic shapes dataloader for quick testing."""
    dataset = SyntheticShapesDataset(
        num_samples=num_samples,
        image_size=image_size,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    # Test synthetic dataset
    print("Testing SyntheticShapesDataset...")
    dataset = SyntheticShapesDataset(num_samples=100, image_size=(128, 128))
    
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Num objects: {sample['num_objects']}")
    
    # Test dataloader
    loader = create_synthetic_dataloader(num_samples=100, batch_size=8)
    batch = next(iter(loader))
    print(f"\nBatch image shape: {batch['image'].shape}")
    print(f"Batch mask shape: {batch['mask'].shape}")
