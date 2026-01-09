"""
VIPSeg Video Dataset Loader for SpectralDiffusion

Video panoptic segmentation dataset for temporal consistency.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import json
import numpy as np
from PIL import Image
import os


class VIPSegDataset(Dataset):
    """
    VIPSeg dataset for video panoptic segmentation.
    
    Contains 3,536 video clips with panoptic annotations.
    
    Args:
        root_dir: Path to VIPSeg directory
        split: "train", "val", or "test"
        clip_length: Number of frames per clip
        image_size: Target image size
        max_instances: Maximum instances per frame
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        clip_length: int = 4,
        image_size: Tuple[int, int] = (480, 720),
        max_instances: int = 24,
        sample_rate: int = 1,
    ):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.clip_length = clip_length
        self.image_size = image_size
        self.max_instances = max_instances
        self.sample_rate = sample_rate
        
        # Paths
        self.videos_dir = self.root_dir / "VIPSeg" / split / "images"
        self.masks_dir = self.root_dir / "VIPSeg" / split / "panomasksRGB"
        self.json_dir = self.root_dir / "VIPSeg" / split / "panoptic_gt_VIPSeg"
        
        # Load video list
        self.clips = self._load_clips()
        print(f"VIPSeg {split}: {len(self.clips)} video clips")
    
    def _load_clips(self) -> List[Dict]:
        """Load video clips metadata."""
        clips = []
        
        if not self.videos_dir.exists():
            print(f"Warning: Videos directory not found at {self.videos_dir}")
            return clips
        
        # Get all videos
        video_dirs = sorted([d for d in os.listdir(self.videos_dir) if 
                            (self.videos_dir / d).is_dir()])
        
        for video_id in video_dirs:
            video_path = self.videos_dir / video_id
            frames = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
            
            # Sample clips
            step = self.clip_length * self.sample_rate
            for start in range(0, len(frames) - step + 1, step):
                clip_frames = frames[start:start + step:self.sample_rate]
                if len(clip_frames) == self.clip_length:
                    clips.append({
                        'video_id': video_id,
                        'frames': clip_frames,
                    })
        
        return clips
    
    def __len__(self) -> int:
        return len(self.clips)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clip = self.clips[idx]
        video_id = clip['video_id']
        
        images = []
        masks = []
        
        for frame_name in clip['frames']:
            # Load image
            img_path = self.videos_dir / video_id / frame_name
            image = Image.open(img_path).convert('RGB')
            image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image.permute(2, 0, 1)
            images.append(image)
            
            # Load mask
            mask_name = frame_name.replace('.jpg', '.png')
            mask_path = self.masks_dir / video_id / mask_name
            
            if mask_path.exists():
                mask_img = Image.open(mask_path)
                mask_img = mask_img.resize(
                    (self.image_size[1], self.image_size[0]),
                    Image.NEAREST
                )
                mask = np.array(mask_img)
                
                # Decode RGB to segment IDs
                segment_ids = mask[:, :, 0] + mask[:, :, 1] * 256 + mask[:, :, 2] * 256 * 256
                
                # Convert to instance masks
                unique_ids = np.unique(segment_ids)
                instance_masks = torch.zeros(self.max_instances, *self.image_size)
                
                for i, seg_id in enumerate(unique_ids[:self.max_instances]):
                    instance_masks[i] = torch.from_numpy(segment_ids == seg_id).float()
                
                masks.append(instance_masks)
            else:
                masks.append(torch.zeros(self.max_instances, *self.image_size))
        
        images = torch.stack(images)  # [T, 3, H, W]
        masks = torch.stack(masks)    # [T, K, H, W]
        
        return {
            'images': images,
            'masks': masks,
            'video_id': video_id,
            'idx': idx,
        }


def create_vipseg_dataloader(
    root_dir: str,
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 4,
    clip_length: int = 4,
    image_size: Tuple[int, int] = (480, 720),
) -> DataLoader:
    """Create VIPSeg video dataloader."""
    dataset = VIPSegDataset(
        root_dir=root_dir,
        split=split,
        clip_length=clip_length,
        image_size=image_size,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )
