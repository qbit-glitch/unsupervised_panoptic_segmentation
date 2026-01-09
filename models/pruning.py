"""
Adaptive Slot Pruning for SpectralDiffusion

Dynamically removes underutilized slots to reduce computation
and improve efficiency during inference.

Key features:
- Utilization-based pruning with threshold τ
- Dynamic K: K_effective ∈ [K_min, K_max]
- Soft and hard pruning modes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AdaptiveSlotPruning(nn.Module):
    """
    Adaptive slot pruning based on utilization.
    
    Removes slots with low attention mass to reduce computation
    while maintaining segmentation quality.
    
    Args:
        num_slots: Maximum number of slots
        min_slots: Minimum slots to keep
        threshold: Utilization threshold for pruning (τ)
        mode: "soft" (reweight) or "hard" (remove)
        learnable_threshold: Whether threshold is learnable
    """
    
    def __init__(
        self,
        num_slots: int = 12,
        min_slots: int = 4,
        threshold: float = 0.05,
        mode: str = "soft",
        learnable_threshold: bool = False,
    ):
        super().__init__()
        
        self.num_slots = num_slots
        self.min_slots = min_slots
        self.mode = mode
        
        if learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(threshold))
        else:
            self.register_buffer('threshold', torch.tensor(threshold))
        
        # For soft pruning: learnable gating
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
    
    def compute_utilization(
        self,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute utilization score for each slot.
        
        Args:
            masks: [B, N, K] soft attention masks
            
        Returns:
            utilization: [B, K] utilization per slot
        """
        # Sum of attention mass per slot
        utilization = masks.sum(dim=1)  # [B, K]
        
        # Normalize by number of features
        N = masks.shape[1]
        utilization = utilization / N
        
        return utilization
    
    def soft_prune(
        self,
        slots: torch.Tensor,
        utilization: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft pruning: scale slots by learned gate.
        
        Args:
            slots: [B, K, D] slot representations
            utilization: [B, K] utilization scores
            
        Returns:
            pruned_slots: [B, K, D] reweighted slots
            gates: [B, K] gate values
        """
        B, K, D = slots.shape
        
        # Compute gates from utilization
        util_input = utilization.unsqueeze(-1)  # [B, K, 1]
        gates = self.gate_mlp(util_input).squeeze(-1)  # [B, K]
        
        # Apply gate-based reweighting
        pruned_slots = slots * gates.unsqueeze(-1)
        
        return pruned_slots, gates
    
    def hard_prune(
        self,
        slots: torch.Tensor,
        masks: torch.Tensor,
        utilization: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hard pruning: remove low-utilization slots.
        
        Args:
            slots: [B, K, D] slot representations
            masks: [B, N, K] attention masks
            utilization: [B, K] utilization scores
            
        Returns:
            active_slots: [B, K_active, D] active slots (padded)
            active_masks: [B, N, K_active] active masks (padded)
            active_mask: [B, K] boolean mask of active slots
        """
        B, K, D = slots.shape
        N = masks.shape[1]
        
        # Determine active slots (above threshold)
        active_mask = utilization > self.threshold  # [B, K]
        
        # Ensure minimum number of slots
        # If too few are active, keep top-min_slots by utilization
        num_active = active_mask.sum(dim=1)  # [B]
        
        for b in range(B):
            if num_active[b] < self.min_slots:
                # Keep top min_slots
                _, top_indices = utilization[b].topk(self.min_slots)
                active_mask[b] = False
                active_mask[b, top_indices] = True
        
        # For batch processing, we need same K_active across batch
        # Use maximum active slots
        K_active = active_mask.sum(dim=1).max().item()
        K_active = max(K_active, self.min_slots)
        
        # Gather active slots (with padding)
        active_slots = torch.zeros(B, K_active, D, device=slots.device, dtype=slots.dtype)
        active_masks_out = torch.zeros(B, N, K_active, device=masks.device, dtype=masks.dtype)
        
        for b in range(B):
            indices = active_mask[b].nonzero(as_tuple=True)[0]
            n_active = len(indices)
            
            active_slots[b, :n_active] = slots[b, indices]
            active_masks_out[b, :, :n_active] = masks[b, :, indices]
        
        return active_slots, active_masks_out, active_mask
    
    def forward(
        self,
        slots: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Apply adaptive pruning.
        
        Args:
            slots: [B, K, D] slot representations
            masks: [B, N, K] attention masks
            
        Returns:
            pruned_slots: Pruned slot representations
            pruned_masks: Pruned attention masks  
            info: Dictionary with pruning statistics
        """
        # Compute utilization
        utilization = self.compute_utilization(masks)
        
        info = {
            'utilization': utilization,
            'threshold': self.threshold.item() if isinstance(self.threshold, torch.Tensor) else self.threshold,
        }
        
        if self.mode == "soft":
            pruned_slots, gates = self.soft_prune(slots, utilization)
            
            # Reweight masks by gates
            pruned_masks = masks * gates.unsqueeze(1)
            
            # Renormalize masks
            mask_sum = pruned_masks.sum(dim=-1, keepdim=True)
            pruned_masks = pruned_masks / (mask_sum + 1e-8)
            
            info['gates'] = gates
            info['num_active'] = (gates > 0.5).sum(dim=1).float().mean().item()
            
        elif self.mode == "hard":
            pruned_slots, pruned_masks, active_mask = self.hard_prune(
                slots, masks, utilization
            )
            
            info['active_mask'] = active_mask
            info['num_active'] = active_mask.sum(dim=1).float().mean().item()
            
        else:
            raise ValueError(f"Unknown pruning mode: {self.mode}")
        
        return pruned_slots, pruned_masks, info


class DynamicSlotCount(nn.Module):
    """
    Dynamically predict optimal number of slots from image content.
    
    Based on image complexity estimation.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        min_slots: int = 4,
        max_slots: int = 24,
    ):
        super().__init__()
        
        self.min_slots = min_slots
        self.max_slots = max_slots
        self.num_classes = max_slots - min_slots + 1
        
        # Predict slot count from pooled features
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes),
        )
    
    def forward(
        self,
        features: torch.Tensor,
    ) -> Tuple[int, torch.Tensor]:
        """
        Predict optimal slot count.
        
        Args:
            features: [B, N, D] image features
            
        Returns:
            slot_count: Predicted number of slots
            probs: [B, num_classes] probability distribution
        """
        # Global average pooling
        pooled = features.mean(dim=1)  # [B, D]
        
        # Predict
        logits = self.predictor(pooled)  # [B, num_classes]
        probs = F.softmax(logits, dim=-1)
        
        # Get prediction (argmax in eval, sample in training)
        if self.training:
            # Gumbel-softmax for differentiable sampling
            slot_idx = F.gumbel_softmax(logits, tau=1.0, hard=True)
            slot_idx = slot_idx.argmax(dim=-1)
        else:
            slot_idx = probs.argmax(dim=-1)
        
        # Convert to slot count
        slot_count = slot_idx + self.min_slots
        
        # Use mode across batch for simplicity
        slot_count = slot_count.mode()[0].item()
        
        return slot_count, probs


if __name__ == "__main__":
    # Test adaptive pruning
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    # Create module
    print("\n=== Testing AdaptiveSlotPruning (soft) ===")
    pruner_soft = AdaptiveSlotPruning(
        num_slots=12,
        min_slots=4,
        threshold=0.05,
        mode="soft",
    ).to(device)
    
    # Test inputs
    B, N, K, D = 2, 256, 12, 768
    slots = torch.randn(B, K, D, device=device)
    
    # Create masks with varying utilization
    masks = torch.rand(B, N, K, device=device)
    masks = masks / masks.sum(dim=-1, keepdim=True)  # Normalize
    
    # Make some slots underutilized
    masks[:, :, -4:] *= 0.01  # Last 4 slots have low utilization
    masks = masks / masks.sum(dim=-1, keepdim=True)  # Renormalize
    
    # Apply soft pruning
    pruned_slots, pruned_masks, info = pruner_soft(slots, masks)
    print(f"Soft pruning:")
    print(f"  Utilization: {info['utilization'][0].tolist()}")
    print(f"  Gates: {info['gates'][0].tolist()}")
    print(f"  Avg active slots: {info['num_active']:.1f}")
    
    # Test hard pruning
    print("\n=== Testing AdaptiveSlotPruning (hard) ===")
    pruner_hard = AdaptiveSlotPruning(
        num_slots=12,
        min_slots=4,
        threshold=0.05,
        mode="hard",
    ).to(device)
    
    pruned_slots_h, pruned_masks_h, info_h = pruner_hard(slots, masks)
    print(f"Hard pruning:")
    print(f"  Active mask: {info_h['active_mask'][0].tolist()}")
    print(f"  Pruned slots shape: {pruned_slots_h.shape}")
    print(f"  Avg active slots: {info_h['num_active']:.1f}")
    
    # Test dynamic slot count
    print("\n=== Testing DynamicSlotCount ===")
    dynamic_counter = DynamicSlotCount(
        feature_dim=768,
        min_slots=4,
        max_slots=24,
    ).to(device)
    
    features = torch.randn(B, N, D, device=device)
    slot_count, probs = dynamic_counter(features)
    print(f"Predicted slot count: {slot_count}")
    print(f"Probability distribution: {probs[0].max().item():.3f} (max)")
