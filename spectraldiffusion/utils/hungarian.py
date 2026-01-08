"""
Hungarian Algorithm Utilities for SpectralDiffusion

Optimal slot matching using linear sum assignment.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from scipy.optimize import linear_sum_assignment
import numpy as np


def hungarian_matching(
    cost_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute optimal bipartite matching using Hungarian algorithm.
    
    Args:
        cost_matrix: [B, M, N] cost matrix (minimize)
        
    Returns:
        row_indices: [B, min(M,N)] matched row indices
        col_indices: [B, min(M,N)] matched column indices
    """
    B, M, N = cost_matrix.shape
    device = cost_matrix.device
    
    row_indices = []
    col_indices = []
    
    for b in range(B):
        cost_np = cost_matrix[b].cpu().detach().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        row_indices.append(torch.tensor(row_ind, device=device, dtype=torch.long))
        col_indices.append(torch.tensor(col_ind, device=device, dtype=torch.long))
    
    # Pad to same length
    max_len = max(len(r) for r in row_indices)
    row_indices = torch.stack([
        F.pad(r, (0, max_len - len(r)), value=-1) for r in row_indices
    ])
    col_indices = torch.stack([
        F.pad(c, (0, max_len - len(c)), value=-1) for c in col_indices
    ])
    
    return row_indices, col_indices


def compute_slot_matching_loss(
    slots_pred: torch.Tensor,
    slots_target: torch.Tensor,
    cost_type: str = "l2",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute slot matching loss with Hungarian algorithm.
    
    Args:
        slots_pred: [B, K, D] predicted slots
        slots_target: [B, K, D] target slots
        cost_type: "l2" or "cosine"
        
    Returns:
        loss: Matching loss
        matched_indices: [B, K] indices showing matching
    """
    B, K, D = slots_pred.shape
    
    # Compute cost matrix
    if cost_type == "l2":
        cost = torch.cdist(slots_pred, slots_target)  # [B, K, K]
    elif cost_type == "cosine":
        pred_norm = F.normalize(slots_pred, p=2, dim=-1)
        target_norm = F.normalize(slots_target, p=2, dim=-1)
        cos_sim = torch.bmm(pred_norm, target_norm.transpose(1, 2))
        cost = 1 - cos_sim  # [B, K, K]
    else:
        raise ValueError(f"Unknown cost type: {cost_type}")
    
    # Hungarian matching
    row_indices, col_indices = hungarian_matching(cost)
    
    # Compute loss on matched pairs
    losses = []
    for b in range(B):
        mask = (row_indices[b] >= 0) & (col_indices[b] >= 0)
        rows = row_indices[b][mask]
        cols = col_indices[b][mask]
        
        if len(rows) > 0:
            pred_matched = slots_pred[b, rows]
            target_matched = slots_target[b, cols]
            loss_b = F.mse_loss(pred_matched, target_matched)
        else:
            loss_b = torch.tensor(0.0, device=slots_pred.device)
        
        losses.append(loss_b)
    
    loss = torch.stack(losses).mean()
    
    return loss, col_indices


def compute_mask_matching_loss(
    masks_pred: torch.Tensor,
    masks_target: torch.Tensor,
    use_dice: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mask matching loss with Hungarian algorithm.
    
    Args:
        masks_pred: [B, K, H, W] predicted masks
        masks_target: [B, K, H, W] target masks
        use_dice: Whether to use Dice loss (vs BCE)
        
    Returns:
        loss: Matching loss
        matched_indices: [B, K] indices showing matching
    """
    B, K, H, W = masks_pred.shape
    
    # Flatten masks for cost computation
    pred_flat = masks_pred.view(B, K, -1)  # [B, K, H*W]
    target_flat = masks_target.view(B, K, -1)  # [B, K, H*W]
    
    # Compute cost matrix using IoU
    # For each pred-target pair, compute 1 - IoU as cost
    costs = []
    for b in range(B):
        cost_b = torch.zeros(K, K, device=masks_pred.device)
        for i in range(K):
            for j in range(K):
                # IoU
                intersection = (pred_flat[b, i] * target_flat[b, j]).sum()
                union = pred_flat[b, i].sum() + target_flat[b, j].sum() - intersection
                iou = intersection / (union + 1e-8)
                cost_b[i, j] = 1 - iou
        costs.append(cost_b)
    cost = torch.stack(costs)  # [B, K, K]
    
    # Hungarian matching
    row_indices, col_indices = hungarian_matching(cost)
    
    # Compute loss on matched pairs
    losses = []
    for b in range(B):
        mask = (row_indices[b] >= 0) & (col_indices[b] >= 0)
        rows = row_indices[b][mask]
        cols = col_indices[b][mask]
        
        if len(rows) > 0:
            pred_matched = masks_pred[b, rows]  # [M, H, W]
            target_matched = masks_target[b, cols]  # [M, H, W]
            
            if use_dice:
                # Dice loss
                intersection = (pred_matched * target_matched).sum(dim=[1, 2])
                union = pred_matched.sum(dim=[1, 2]) + target_matched.sum(dim=[1, 2])
                dice = 2 * intersection / (union + 1e-8)
                loss_b = (1 - dice).mean()
            else:
                # BCE loss
                loss_b = F.binary_cross_entropy(
                    pred_matched.clamp(1e-6, 1-1e-6),
                    target_matched,
                    reduction='mean'
                )
        else:
            loss_b = torch.tensor(0.0, device=masks_pred.device)
        
        losses.append(loss_b)
    
    loss = torch.stack(losses).mean()
    
    return loss, col_indices


def greedy_matching(
    cost_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast greedy matching (approximate, but faster than Hungarian).
    
    Args:
        cost_matrix: [B, M, N] cost matrix
        
    Returns:
        row_indices: [B, min(M,N)] matched row indices
        col_indices: [B, min(M,N)] matched column indices
    """
    B, M, N = cost_matrix.shape
    num_matches = min(M, N)
    device = cost_matrix.device
    
    row_indices = []
    col_indices = []
    
    for b in range(B):
        cost_b = cost_matrix[b].clone()
        rows = []
        cols = []
        
        for _ in range(num_matches):
            # Find minimum
            min_idx = cost_b.argmin()
            row = min_idx // N
            col = min_idx % N
            
            rows.append(row)
            cols.append(col)
            
            # Mask out row and column
            cost_b[row, :] = float('inf')
            cost_b[:, col] = float('inf')
        
        row_indices.append(torch.tensor(rows, device=device))
        col_indices.append(torch.tensor(cols, device=device))
    
    row_indices = torch.stack(row_indices)
    col_indices = torch.stack(col_indices)
    
    return row_indices, col_indices
