"""
Temporal Consistency Loss for SpectralDiffusion

Slot matching across video frames for temporal coherence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from scipy.optimize import linear_sum_assignment


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for video panoptic segmentation.
    
    L_temp = E_{t,t+1} ||S_t - Match(S_{t+1})||²
    
    Uses Hungarian algorithm for optimal slot matching across frames.
    
    Args:
        reduction: "mean", "sum", or "none"
        cost_type: "l2" or "cosine" for matching cost
    """
    
    def __init__(
        self,
        reduction: str = "mean",
        cost_type: str = "l2",
    ):
        super().__init__()
        self.reduction = reduction
        self.cost_type = cost_type
    
    def compute_cost_matrix(
        self,
        slots_t: torch.Tensor,
        slots_t1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cost matrix for Hungarian matching.
        
        Args:
            slots_t: [B, K, D] slots at time t
            slots_t1: [B, K, D] slots at time t+1
            
        Returns:
            cost: [B, K, K] cost matrix
        """
        if self.cost_type == "l2":
            # L2 distance
            cost = torch.cdist(slots_t, slots_t1)  # [B, K, K]
        elif self.cost_type == "cosine":
            # Cosine distance (1 - similarity)
            slots_t_norm = F.normalize(slots_t, p=2, dim=-1)
            slots_t1_norm = F.normalize(slots_t1, p=2, dim=-1)
            sim = torch.bmm(slots_t_norm, slots_t1_norm.transpose(1, 2))  # [B, K, K]
            cost = 1 - sim
        else:
            raise ValueError(f"Unknown cost type: {self.cost_type}")
        
        return cost
    
    def hungarian_match(
        self,
        cost: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Hungarian algorithm for optimal matching.
        
        Args:
            cost: [B, K, K] cost matrix
            
        Returns:
            row_indices: [B, K] row indices
            col_indices: [B, K] column indices (matched)
        """
        B, K, _ = cost.shape
        
        row_indices = []
        col_indices = []
        
        for b in range(B):
            cost_np = cost[b].cpu().detach().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            row_indices.append(torch.tensor(row_ind, device=cost.device))
            col_indices.append(torch.tensor(col_ind, device=cost.device))
        
        row_indices = torch.stack(row_indices)  # [B, K]
        col_indices = torch.stack(col_indices)  # [B, K]
        
        return row_indices, col_indices
    
    def forward(
        self,
        slots_t: torch.Tensor,
        slots_t1: torch.Tensor,
        masks_t: Optional[torch.Tensor] = None,
        masks_t1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            slots_t: [B, K, D] slots at time t
            slots_t1: [B, K, D] slots at time t+1
            masks_t: Optional [B, N, K] masks at time t
            masks_t1: Optional [B, N, K] masks at time t+1
            
        Returns:
            loss: Temporal consistency loss
        """
        B, K, D = slots_t.shape
        
        # Compute cost matrix
        cost = self.compute_cost_matrix(slots_t, slots_t1)
        
        # Hungarian matching
        _, match_indices = self.hungarian_match(cost)
        
        # Reorder slots_t1 according to matching
        matched_slots = torch.gather(
            slots_t1,
            dim=1,
            index=match_indices.unsqueeze(-1).expand(-1, -1, D),
        )  # [B, K, D]
        
        # L2 loss on matched slots
        diff = slots_t - matched_slots
        loss = (diff ** 2).sum(dim=-1)  # [B, K]
        
        # Optional: weight by slot utilization
        if masks_t is not None:
            utilization = masks_t.sum(dim=1)  # [B, K]
            utilization = utilization / utilization.sum(dim=1, keepdim=True)
            loss = loss * utilization
        
        # Reduce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FlowGuidedTemporalLoss(nn.Module):
    """
    Flow-guided temporal consistency using optical flow.
    
    Warps previous frame masks using flow and compares.
    """
    
    def __init__(
        self,
        reduction: str = "mean",
    ):
        super().__init__()
        self.reduction = reduction
    
    def warp_masks(
        self,
        masks: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """
        Warp masks using optical flow.
        
        Args:
            masks: [B, K, H, W] masks to warp
            flow: [B, 2, H, W] optical flow (x, y)
            
        Returns:
            warped: [B, K, H, W] warped masks
        """
        B, K, H, W = masks.shape
        
        # Create meshgrid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=masks.device),
            torch.linspace(-1, 1, W, device=masks.device),
            indexing='ij'
        )
        
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
        
        # Normalize flow to [-1, 1] range
        flow_normalized = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
        flow_normalized[..., 0] = flow_normalized[..., 0] / (W / 2)
        flow_normalized[..., 1] = flow_normalized[..., 1] / (H / 2)
        
        # Add flow to grid
        grid_warped = grid + flow_normalized
        
        # Warp each slot mask
        warped = F.grid_sample(
            masks,
            grid_warped,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        
        return warped
    
    def forward(
        self,
        masks_t: torch.Tensor,
        masks_t1: torch.Tensor,
        flow_t_to_t1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute flow-guided temporal loss.
        
        Args:
            masks_t: [B, K, H, W] masks at time t
            masks_t1: [B, K, H, W] masks at time t+1
            flow_t_to_t1: [B, 2, H, W] flow from t to t+1
            
        Returns:
            loss: Temporal consistency loss
        """
        # Warp masks from t using flow
        warped_masks = self.warp_masks(masks_t, flow_t_to_t1)
        
        # L1 loss between warped and actual t+1 masks
        diff = (warped_masks - masks_t1).abs()
        loss = diff.mean(dim=[1, 2, 3])  # [B]
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
