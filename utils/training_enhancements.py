#!/usr/bin/env python
"""
Training Enhancement Utilities for SlotAttention
Based on guides/more_enhancement/ recommendations

Includes:
1. CurriculumLearning - Start with simple scenes, increase complexity
2. TestTimeAugmentation - Multi-scale inference
3. ContrastiveSlotLoss - Push slots apart
4. AdaptiveLossWeighting - Auto-balance loss components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import numpy as np


class CurriculumLearning:
    """
    Curriculum Learning: Start with simpler scenes (fewer objects).
    
    As training progresses, gradually include more complex scenes.
    This helps the model learn basic object discovery before hard cases.
    """
    
    def __init__(
        self,
        max_epochs: int = 100,
        min_complexity: float = 0.3,  # Start with 30% complexity
        max_complexity: float = 1.0,
        strategy: str = 'linear',  # 'linear', 'cosine', 'step'
    ):
        self.max_epochs = max_epochs
        self.min_complexity = min_complexity
        self.max_complexity = max_complexity
        self.strategy = strategy
    
    def get_complexity(self, epoch: int) -> float:
        """Get curriculum complexity for current epoch."""
        progress = min(1.0, epoch / self.max_epochs)
        
        if self.strategy == 'linear':
            return self.min_complexity + progress * (self.max_complexity - self.min_complexity)
        elif self.strategy == 'cosine':
            # Cosine annealing from min to max
            return self.min_complexity + 0.5 * (self.max_complexity - self.min_complexity) * (
                1 - np.cos(np.pi * progress)
            )
        elif self.strategy == 'step':
            # Step function: 30%, 50%, 70%, 100%
            if progress < 0.25:
                return 0.3
            elif progress < 0.5:
                return 0.5
            elif progress < 0.75:
                return 0.7
            else:
                return 1.0
        else:
            return self.max_complexity
    
    def filter_batch_by_complexity(
        self,
        batch: Dict,
        epoch: int,
        max_objects_key: str = 'num_objects'
    ) -> Dict:
        """
        Filter batch to match curriculum complexity.
        
        Complexity = number of objects / max_objects
        Include samples with complexity <= current threshold
        """
        complexity = self.get_complexity(epoch)
        
        # If no object count info, return full batch
        if max_objects_key not in batch:
            return batch
        
        num_objects = batch[max_objects_key]
        max_objects = num_objects.max().item()
        
        if max_objects == 0:
            return batch
        
        # Filter: keep samples with relative complexity <= threshold
        sample_complexity = num_objects.float() / max_objects
        mask = sample_complexity <= complexity
        
        if mask.sum() == 0:
            return batch  # Keep at least some samples
        
        filtered_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.shape[0] == mask.shape[0]:
                filtered_batch[key] = value[mask]
            else:
                filtered_batch[key] = value
        
        return filtered_batch


class TestTimeAugmentation:
    """
    Test-Time Augmentation (TTA) for improved inference.
    
    Run inference at multiple scales and average predictions.
    Expected improvement: +2-3 PQ
    """
    
    def __init__(
        self,
        scales: List[float] = [0.75, 1.0, 1.25],
        flip: bool = True,
    ):
        self.scales = scales
        self.flip = flip
    
    @torch.no_grad()
    def __call__(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply TTA and average predictions.
        
        Args:
            model: Trained model
            images: [B, 3, H, W] input images
            
        Returns:
            dict with averaged masks and other outputs
        """
        B, C, H, W = images.shape
        all_masks = []
        
        for scale in self.scales:
            # Scale image
            if scale != 1.0:
                new_H, new_W = int(H * scale), int(W * scale)
                scaled = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=False)
            else:
                scaled = images
            
            # Forward pass
            outputs = model(scaled, return_loss=False)
            masks = outputs['masks']  # [B, K, H', W']
            
            # Resize masks back to original size
            if scale != 1.0:
                masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
            
            all_masks.append(masks)
            
            # Horizontal flip
            if self.flip:
                flipped = torch.flip(scaled, dims=[3])
                outputs_flip = model(flipped, return_loss=False)
                masks_flip = torch.flip(outputs_flip['masks'], dims=[3])
                
                if scale != 1.0:
                    masks_flip = F.interpolate(masks_flip, size=(H, W), mode='bilinear', align_corners=False)
                
                all_masks.append(masks_flip)
        
        # Average all predictions
        avg_masks = torch.stack(all_masks, dim=0).mean(dim=0)
        
        return {
            'masks': avg_masks,
            'masks_all': all_masks,  # For debugging
        }


class ContrastiveSlotLoss(nn.Module):
    """
    Contrastive loss for slot representations.
    
    Encourages slots representing same object (across augmentations) 
    to be similar, and different objects to be dissimilar.
    
    Uses InfoNCE loss formulation.
    Expected improvement: +2-3 PQ
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        weight: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.weight = weight
    
    def forward(
        self,
        slots1: torch.Tensor,
        slots2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between two slot sets.
        
        Args:
            slots1: [B, K, D] slots from view 1
            slots2: [B, K, D] slots from view 2 (augmented)
            
        Returns:
            loss: Contrastive loss encouraging consistency
        """
        B, K, D = slots1.shape
        
        # Normalize
        slots1_norm = F.normalize(slots1, dim=-1)  # [B, K, D]
        slots2_norm = F.normalize(slots2, dim=-1)  # [B, K, D]
        
        # Compute similarity matrix
        # We want diagonal (same slot index) to have high similarity
        sim = torch.bmm(slots1_norm, slots2_norm.transpose(1, 2))  # [B, K, K]
        sim = sim / self.temperature
        
        # InfoNCE: positive pairs are diagonal
        labels = torch.arange(K, device=slots1.device).unsqueeze(0).expand(B, -1)  # [B, K]
        
        # Cross entropy loss (each slot should match itself across views)
        loss = F.cross_entropy(sim.reshape(B * K, K), labels.reshape(B * K))
        
        return self.weight * loss


class AdaptiveLossWeighting(nn.Module):
    """
    Adaptive loss weighting based on uncertainty.
    
    Learns optimal weights for multiple loss components.
    Based on "Multi-Task Learning Using Uncertainty" (Kendall et al.)
    """
    
    def __init__(
        self,
        num_losses: int = 3,
        init_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        
        if init_weights is None:
            init_weights = [0.0] * num_losses
        
        # Log variance as learnable parameters
        self.log_vars = nn.Parameter(torch.tensor(init_weights))
    
    def forward(
        self,
        losses: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted sum of losses with uncertainty weighting.
        
        Args:
            losses: List of [scalar] loss tensors
            
        Returns:
            total_loss: Weighted sum
            weights: Dict of weight values for logging
        """
        total = 0.0
        weights = {}
        
        for i, (loss, log_var) in enumerate(zip(losses, self.log_vars)):
            # Weight = 1 / (2 * variance) = 1 / (2 * exp(log_var))
            precision = torch.exp(-log_var)
            
            # Weighted loss + regularizer
            weighted = precision * loss + log_var
            total = total + weighted
            
            weights[f'weight_{i}'] = precision.item()
        
        return total, weights
