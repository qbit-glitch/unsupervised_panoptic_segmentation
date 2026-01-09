"""
Data Augmentation Pipeline for SpectralDiffusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class RandomHorizontalFlip:
    """Random horizontal flip for images and masks."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if torch.rand(1).item() < self.p:
            image = torch.flip(image, dims=[-1])
            if mask is not None:
                mask = torch.flip(mask, dims=[-1])
        return image, mask


class RandomResizedCrop:
    """Random resized crop for images and masks."""
    
    def __init__(
        self,
        size: Tuple[int, int],
        scale: Tuple[float, float] = (0.8, 1.0),
        ratio: Tuple[float, float] = (0.9, 1.1),
    ):
        self.size = size
        self.scale = scale
        self.ratio = ratio
    
    def __call__(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        C, H, W = image.shape
        
        # Random scale and ratio
        scale = np.random.uniform(*self.scale)
        ratio = np.random.uniform(*self.ratio)
        
        new_h = int(H * scale)
        new_w = int(new_h * ratio)
        new_w = min(new_w, W)
        
        # Random crop position
        top = np.random.randint(0, max(H - new_h + 1, 1))
        left = np.random.randint(0, max(W - new_w + 1, 1))
        
        # Crop
        image = image[:, top:top+new_h, left:left+new_w]
        
        # Resize to target
        image = F.interpolate(
            image.unsqueeze(0),
            size=self.size,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask[top:top+new_h, left:left+new_w]
                mask = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=self.size,
                    mode='nearest',
                ).squeeze(0).squeeze(0).long()
            else:
                mask = mask[:, top:top+new_h, left:left+new_w]
                mask = F.interpolate(
                    mask.unsqueeze(0).float(),
                    size=self.size,
                    mode='nearest',
                ).squeeze(0)
        
        return image, mask


class ColorJitter:
    """Random color jittering for images."""
    
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Brightness
        factor = 1 + np.random.uniform(-self.brightness, self.brightness)
        image = image * factor
        
        # Contrast
        factor = 1 + np.random.uniform(-self.contrast, self.contrast)
        gray = image.mean(dim=0, keepdim=True)
        image = (image - gray) * factor + gray
        
        # Clip to valid range
        image = image.clamp(0, 1)
        
        return image, mask


class RandomGaussianBlur:
    """Random Gaussian blur for images."""
    
    def __init__(
        self,
        kernel_size: int = 5,
        sigma: Tuple[float, float] = (0.1, 2.0),
        p: float = 0.5,
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
    
    def __call__(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if torch.rand(1).item() < self.p:
            sigma = np.random.uniform(*self.sigma)
            
            # Create Gaussian kernel
            k = self.kernel_size
            x = torch.arange(k) - k // 2
            gauss = torch.exp(-x**2 / (2 * sigma**2))
            kernel = gauss.outer(gauss)
            kernel = kernel / kernel.sum()
            kernel = kernel.view(1, 1, k, k).expand(3, 1, k, k)
            kernel = kernel.to(image.device)
            
            # Apply blur
            image = F.pad(image.unsqueeze(0), (k//2, k//2, k//2, k//2), mode='reflect')
            image = F.conv2d(image, kernel, groups=3)
            image = image.squeeze(0)
        
        return image, mask


class Normalize:
    """Normalize images with ImageNet statistics."""
    
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    
    def __call__(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mean = self.mean.to(image.device)
        std = self.std.to(image.device)
        image = (image - mean) / std
        return image, mask


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


def get_train_transforms(
    image_size: Tuple[int, int] = (518, 518),
) -> Compose:
    """Get training transforms."""
    return Compose([
        RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        RandomGaussianBlur(p=0.3),
    ])


def get_val_transforms(
    image_size: Tuple[int, int] = (518, 518),
) -> Compose:
    """Get validation transforms (minimal)."""
    return Compose([])  # No augmentation for validation
