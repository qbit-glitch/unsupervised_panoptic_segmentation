"""
Panoptic Segmentation Metrics for SpectralDiffusion

Implements:
- Adjusted Rand Index (ARI) for object discovery
- Panoptic Quality (PQ), Segmentation Quality (SQ), Recognition Quality (RQ)
- Mean Intersection over Union (mIoU)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_ari(
    pred_masks: torch.Tensor,
    true_masks: torch.Tensor,
    ignore_background: bool = False,
) -> torch.Tensor:
    """
    Compute Adjusted Rand Index for clustering evaluation.
    
    ARI measures similarity between predicted and ground truth segmentations,
    adjusted for chance.
    
    Args:
        pred_masks: [B, H, W] predicted segmentation (integer labels)
        true_masks: [B, H, W] ground truth segmentation
        ignore_background: Whether to ignore background (label 0)
        
    Returns:
        ari: [B] ARI scores
    """
    B, H, W = pred_masks.shape
    N = H * W
    
    aris = []
    
    for b in range(B):
        pred = pred_masks[b].flatten().cpu().numpy()
        true = true_masks[b].flatten().cpu().numpy()
        
        if ignore_background:
            # Only evaluate on foreground pixels
            fg_mask = true > 0
            pred = pred[fg_mask]
            true = true[fg_mask]
            N = len(pred)
        
        if N == 0:
            aris.append(0.0)
            continue
        
        # Compute contingency table
        pred_labels = np.unique(pred)
        true_labels = np.unique(true)
        
        contingency = np.zeros((len(pred_labels), len(true_labels)))
        pred_map = {l: i for i, l in enumerate(pred_labels)}
        true_map = {l: i for i, l in enumerate(true_labels)}
        
        for p, t in zip(pred, true):
            contingency[pred_map[p], true_map[t]] += 1
        
        # ARI formula
        a = np.sum(contingency, axis=1)  # Row sums
        b = np.sum(contingency, axis=0)  # Column sums
        n = N
        
        # Sum of C(n_ij, 2)
        sum_comb_c = np.sum(contingency * (contingency - 1)) / 2
        
        # Sum of C(a_i, 2) and C(b_j, 2)
        sum_comb_a = np.sum(a * (a - 1)) / 2
        sum_comb_b = np.sum(b * (b - 1)) / 2
        
        # C(n, 2)
        comb_n = n * (n - 1) / 2
        
        # Expected index
        expected = sum_comb_a * sum_comb_b / comb_n
        
        # Max index
        max_idx = (sum_comb_a + sum_comb_b) / 2
        
        # ARI
        if max_idx - expected == 0:
            ari = 1.0 if sum_comb_c == expected else 0.0
        else:
            ari = (sum_comb_c - expected) / (max_idx - expected)
        
        aris.append(ari)
    
    return torch.tensor(aris, device=pred_masks.device, dtype=torch.float32)


def compute_foreground_ari(
    pred_masks: torch.Tensor,
    true_masks: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Foreground-only ARI (FG-ARI).
    
    Only evaluates on foreground regions.
    """
    return compute_ari(pred_masks, true_masks, ignore_background=True)


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute IoU between binary masks.
    
    Args:
        pred: [H, W] or [N, H, W] predicted mask
        target: [H, W] or [N, H, W] target mask
        
    Returns:
        iou: IoU score
    """
    pred = pred.bool()
    target = target.bool()
    
    intersection = (pred & target).sum(dim=(-2, -1)).float()
    union = (pred | target).sum(dim=(-2, -1)).float()
    
    iou = intersection / (union + 1e-8)
    return iou


def compute_pq(
    pred_masks: torch.Tensor,
    true_masks: torch.Tensor,
    pred_labels: Optional[torch.Tensor] = None,
    true_labels: Optional[torch.Tensor] = None,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute Panoptic Quality (PQ) and its components.
    
    PQ = (ΣTP IoU(p,g)) / (|TP| + 0.5|FP| + 0.5|FN|)
       = SQ × RQ
    
    where:
    - SQ (Segmentation Quality) = ΣTP IoU(p,g) / |TP|
    - RQ (Recognition Quality) = |TP| / (|TP| + 0.5|FP| + 0.5|FN|)
    
    Args:
        pred_masks: [B, K, H, W] predicted instance masks
        true_masks: [B, K, H, W] ground truth instance masks
        pred_labels: Optional [B, K] class labels
        true_labels: Optional [B, K] class labels
        iou_threshold: IoU threshold for matching (default 0.5)
        
    Returns:
        Dictionary with PQ, SQ, RQ, TP, FP, FN
    """
    B = pred_masks.shape[0]
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0
    
    for b in range(B):
        # Get valid (non-empty) masks
        pred_valid = pred_masks[b].sum(dim=[1, 2]) > 0
        true_valid = true_masks[b].sum(dim=[1, 2]) > 0
        
        pred_b = pred_masks[b][pred_valid]  # [Kp, H, W]
        true_b = true_masks[b][true_valid]  # [Kt, H, W]
        
        Kp = pred_b.shape[0]
        Kt = true_b.shape[0]
        
        if Kp == 0 and Kt == 0:
            continue
        elif Kp == 0:
            total_fn += Kt
            continue
        elif Kt == 0:
            total_fp += Kp
            continue
        
        # Compute IoU matrix
        iou_matrix = torch.zeros(Kp, Kt, device=pred_masks.device)
        for i in range(Kp):
            for j in range(Kt):
                iou_matrix[i, j] = compute_iou(pred_b[i], true_b[j])
        
        # Hungarian matching on valid pairs (IoU >= threshold)
        cost = -iou_matrix.cpu().numpy()  # Negative for minimization
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Count TP, FP, FN
        matched_pred = set()
        matched_true = set()
        iou_sum = 0.0
        
        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] >= iou_threshold:
                matched_pred.add(i)
                matched_true.add(j)
                iou_sum += iou_matrix[i, j].item()
        
        tp = len(matched_pred)
        fp = Kp - tp
        fn = Kt - len(matched_true)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_iou += iou_sum
    
    # Compute metrics
    if total_tp == 0:
        sq = 0.0
        rq = 0.0
        pq = 0.0
    else:
        sq = total_iou / total_tp
        rq = total_tp / (total_tp + 0.5 * total_fp + 0.5 * total_fn)
        pq = sq * rq
    
    return {
        'PQ': pq,
        'SQ': sq,
        'RQ': rq,
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
    }


def compute_sq_rq(
    pred_masks: torch.Tensor,
    true_masks: torch.Tensor,
    iou_threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute Segmentation Quality and Recognition Quality.
    
    Returns:
        sq: Segmentation Quality
        rq: Recognition Quality
    """
    result = compute_pq(pred_masks, true_masks, iou_threshold=iou_threshold)
    return result['SQ'], result['RQ']


def compute_miou(
    pred_masks: torch.Tensor,
    true_masks: torch.Tensor,
    num_classes: Optional[int] = None,
) -> float:
    """
    Compute mean Intersection over Union (mIoU).
    
    Args:
        pred_masks: [B, H, W] predicted segmentation (class labels)
        true_masks: [B, H, W] ground truth segmentation
        num_classes: Number of classes (auto-detect if None)
        
    Returns:
        miou: Mean IoU across classes
    """
    if num_classes is None:
        num_classes = max(pred_masks.max().item(), true_masks.max().item()) + 1
    
    ious = []
    
    for c in range(num_classes):
        pred_c = pred_masks == c
        true_c = true_masks == c
        
        intersection = (pred_c & true_c).sum().float()
        union = (pred_c | true_c).sum().float()
        
        if union > 0:
            ious.append((intersection / union).item())
    
    if len(ious) == 0:
        return 0.0
    
    return sum(ious) / len(ious)


class PanopticMetrics:
    """
    Aggregated panoptic metrics tracker.
    
    Accumulates predictions over batches and computes final metrics.
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.5,
        ignore_background: bool = True,
    ):
        self.iou_threshold = iou_threshold
        self.ignore_background = ignore_background
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_iou = 0.0
        self.total_ari = 0.0
        self.num_samples = 0
    
    def update(
        self,
        pred_masks: torch.Tensor,
        true_masks: torch.Tensor,
    ):
        """
        Update with a batch of predictions.
        
        Args:
            pred_masks: [B, K, H, W] predicted masks
            true_masks: [B, K, H, W] ground truth masks
        """
        B = pred_masks.shape[0]
        
        # PQ metrics
        pq_result = compute_pq(
            pred_masks, true_masks, iou_threshold=self.iou_threshold
        )
        self.total_tp += pq_result['TP']
        self.total_fp += pq_result['FP']
        self.total_fn += pq_result['FN']
        if pq_result['TP'] > 0:
            self.total_iou += pq_result['SQ'] * pq_result['TP']
        
        # ARI
        pred_labels = pred_masks.argmax(dim=1)  # [B, H, W]
        true_labels = true_masks.argmax(dim=1)  # [B, H, W]
        ari = compute_ari(pred_labels, true_labels, self.ignore_background)
        self.total_ari += ari.sum().item()
        
        self.num_samples += B
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary with all metrics
        """
        if self.total_tp == 0:
            sq = 0.0
            rq = 0.0
            pq = 0.0
        else:
            sq = self.total_iou / self.total_tp
            rq = self.total_tp / (self.total_tp + 0.5 * self.total_fp + 0.5 * self.total_fn)
            pq = sq * rq
        
        ari = self.total_ari / max(self.num_samples, 1)
        
        return {
            'PQ': pq,
            'SQ': sq,
            'RQ': rq,
            'ARI': ari,
            'TP': self.total_tp,
            'FP': self.total_fp,
            'FN': self.total_fn,
            'num_samples': self.num_samples,
        }
    
    def __str__(self) -> str:
        metrics = self.compute()
        return (
            f"PQ: {metrics['PQ']:.4f}, "
            f"SQ: {metrics['SQ']:.4f}, "
            f"RQ: {metrics['RQ']:.4f}, "
            f"ARI: {metrics['ARI']:.4f}"
        )
