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


class MeanTeacherFramework(nn.Module):
    """
    Mean Teacher Framework for semi-supervised learning.
    Based on Tarvainen & Valpola (NeurIPS 2017) + 2025 improvements.
    
    Key idea: Teacher model provides more stable pseudo-labels than student.
    Teacher weights are EMA of student weights.
    
    Expected improvement: +4-5 PQ
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        ema_decay: float = 0.999,
        consistency_weight: float = 1.0,
        confidence_threshold: float = 0.95,
    ):
        super().__init__()
        self.student = student_model
        self.teacher = deepcopy(student_model)
        self.ema_decay = ema_decay
        self.consistency_weight = consistency_weight
        self.confidence_threshold = confidence_threshold
        
        # Freeze teacher - updated via EMA only
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update_teacher(self):
        """EMA update of teacher weights."""
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1 - self.ema_decay)
    
    def strong_augment(self, images: torch.Tensor) -> torch.Tensor:
        """Strong augmentation for consistency training."""
        # Color jitter
        if torch.rand(1).item() > 0.5:
            # Random brightness, contrast
            images = images * (0.8 + 0.4 * torch.rand(1, device=images.device))
            images = images.clamp(0, 1)
        
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            images = torch.flip(images, dims=[3])
        
        return images
    
    def forward(
        self,
        images: torch.Tensor,
        return_loss: bool = True,
    ) -> Dict:
        """
        Forward pass with consistency loss between student and teacher.
        
        Args:
            images: [B, 3, H, W] input images
            
        Returns:
            dict with outputs and consistency loss
        """
        # Student forward (with augmentation)
        images_aug = self.strong_augment(images)
        student_outputs = self.student(images_aug, return_loss=return_loss)
        
        # Teacher forward (no augmentation, no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(images, return_loss=False)
        
        # Consistency loss: KL divergence between student and teacher masks
        student_masks = student_outputs['masks']  # [B, K, H, W]
        teacher_masks = teacher_outputs['masks']  # [B, K, H, W]
        
        # Flatten spatial dims for KL
        B, K, H, W = student_masks.shape
        student_flat = student_masks.reshape(B, K, -1)  # [B, K, H*W]
        teacher_flat = teacher_masks.reshape(B, K, -1)  # [B, K, H*W]
        
        # Softmax over slots (competition)
        student_prob = F.softmax(student_flat, dim=1)  # [B, K, H*W]
        teacher_prob = F.softmax(teacher_flat, dim=1)  # [B, K, H*W]
        
        # KL divergence (student should match teacher)
        consistency_loss = F.kl_div(
            student_prob.log(),
            teacher_prob,
            reduction='batchmean'
        )
        
        # Confidence mask: only use high-confidence teacher predictions
        teacher_confidence = teacher_prob.max(dim=1)[0]  # [B, H*W]
        confidence_mask = (teacher_confidence > self.confidence_threshold).float()
        
        # Weighted consistency loss
        if confidence_mask.sum() > 0:
            consistency_loss = (consistency_loss * confidence_mask.mean())
        
        student_outputs['consistency_loss'] = self.consistency_weight * consistency_loss
        student_outputs['teacher_masks'] = teacher_masks
        
        return student_outputs


class UncertaintyAwareLoss(nn.Module):
    """
    Uncertainty-Aware Loss for pseudo-labeling.
    
    Weights pseudo-labels by prediction uncertainty (entropy or KL variance).
    Low uncertainty = high weight, high uncertainty = low weight.
    
    Based on CVPR 2025 semi-supervised segmentation methods.
    Expected improvement: +2-3 PQ
    """
    
    def __init__(
        self,
        uncertainty_threshold: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.uncertainty_threshold = uncertainty_threshold
        self.temperature = temperature
    
    def compute_uncertainty(
        self,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute uncertainty from prediction.
        
        Args:
            pred: [B, K, H, W] slot predictions (logits or probs)
            
        Returns:
            uncertainty: [B, H, W] uncertainty map (0 = certain, 1 = uncertain)
        """
        # Softmax over slots
        if pred.min() < 0:  # Logits
            prob = F.softmax(pred / self.temperature, dim=1)
        else:  # Already probabilities
            prob = pred
        
        # Entropy as uncertainty: H = -sum(p * log(p))
        entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1)  # [B, H, W]
        
        # Normalize by max entropy (log(K))
        K = pred.shape[1]
        max_entropy = np.log(K)
        normalized_entropy = entropy / max_entropy  # [0, 1]
        
        return normalized_entropy
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'mean',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute uncertainty-weighted loss.
        
        Args:
            pred: [B, K, H, W] predictions (logits)
            target: [B, H, W] ground truth or pseudo-labels
            
        Returns:
            loss: Weighted cross-entropy
            uncertainty: [B, H, W] uncertainty map
        """
        B, K, H, W = pred.shape
        
        # Compute uncertainty
        uncertainty = self.compute_uncertainty(pred)  # [B, H, W]
        
        # Weight: low uncertainty = high weight
        # weight = 1 - uncertainty, but clamp at threshold
        reliable_mask = (uncertainty < self.uncertainty_threshold).float()
        weight = (1 - uncertainty) * reliable_mask
        
        # Cross entropy loss
        loss = F.cross_entropy(pred, target.long(), reduction='none')  # [B, H, W]
        
        # Weighted loss
        if reliable_mask.sum() > 0:
            weighted_loss = (loss * weight).sum() / (weight.sum() + 1e-8)
        else:
            weighted_loss = loss.mean()
        
        return weighted_loss, uncertainty

