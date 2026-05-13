"""Path-C AuxThingAdapter: GT-free 14-channel SAM3-thing classifier on FPN-P4 features.

A small MLP that maps frozen Stage-3 FPN-P4 features (256-dim) to 14 SAM3
fine-grained thing classes. Trained with SAM3 cross-entropy supervision
*only* — no Cityscapes labels, no LUT, fully GT-free.

At inference, paired with the existing 80-cluster head via per-pixel
confidence-thresholded fusion. The cluster head handles stuff via standard
CUPS Hungarian; the adapter handles things where SAM3 is confident.

See plan: docs/plans/2026-05-05_dead_class_cascade_recovery_plan.md (Task 7).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxThingAdapter(nn.Module):
    """MLP head: FPN-P4 features → 14 SAM3-thing-class logits.

    Init: Kaiming-normal for weights, zero biases. No information from any
    GT-derived source (no LUT, no Cityscapes class names).

    Args:
        in_dim: input feature dimension (default 256, matching CUPS FPN P4).
        hidden_dim: bottleneck width (default 64).
        num_sam3_classes: output dim (default 14, the SAM3 fine vocab size).
        dropout: optional dropout between fc1 and fc2 (default 0.0).
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 64,
        num_sam3_classes: int = 14,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_sam3_classes = num_sam3_classes

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, num_sam3_classes)

        # Kaiming init (no LUT, no GT)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: ``(B, C, H, W)`` per-pixel FPN-P4 features, OR
                ``(N, C)`` per-pixel feature vectors.

        Returns:
            ``(B, num_sam3_classes, H, W)`` or ``(N, num_sam3_classes)``
            class logits.
        """
        if features.dim() == 4:
            # (B, C, H, W) → (B, H, W, C) → linear → (B, H, W, K) → (B, K, H, W)
            x = features.permute(0, 2, 3, 1)
            x = self.fc1(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x.permute(0, 3, 1, 2).contiguous()
        # (N, C) → linear → (N, K)
        x = self.fc1(features)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Structural mapping from SAM3 fine-grained class index to Cityscapes trainID.
# This mapping is DEFINITIONAL, not learned: SAM3's "motorcycle" concept
# refers to the same real-world object as Cityscapes' "motorcycle" trainID.
# No GT labels are consulted.
SAM3_TO_CITYSCAPES_TRAINID: dict[int, int] = {
    0: 11,   # person → trainID 11 (person)
    1: 18,   # bicycle → trainID 18 (bicycle)
    2: 17,   # motorcycle → trainID 17 (motorcycle)  ◄── previously dead
    3: 12,   # rider → trainID 12 (rider)
    4: 7,    # traffic sign → trainID 7 (traffic sign)
    5: 6,    # traffic light → trainID 6 (traffic light)  ◄── previously dead
    6: 14,   # truck → trainID 14 (truck)
    7: 15,   # bus → trainID 15 (bus)
    8: 16,   # train → trainID 16 (train)
    # 9: guard rail — Cityscapes doesn't have it as a thing; map to ignore.
    # 10: caravan — Cityscapes void/thing, no clean trainID.
    # 11: trailer — Cityscapes void/thing, no clean trainID.
    12: 13,  # car → trainID 13 (car)
    13: 5,   # pole → trainID 5 (pole)
}


def fuse_predictions(
    cluster_head_27_pred: torch.Tensor,
    adapter_logits: torch.Tensor,
    confidence_threshold: float = 0.5,
    sam3_to_trainid: dict[int, int] | None = None,
) -> torch.Tensor:
    """Per-pixel fusion of cluster-head (Hungarian-collapsed) + adapter.

    Args:
        cluster_head_27_pred: ``(B, H, W)`` 27-class trainID prediction from
            the existing 80-cluster head after CUPS Hungarian collapse.
            Used for stuff and as fallback for low-confidence adapter pixels.
        adapter_logits: ``(B, 14, H, W)`` raw logits from AuxThingAdapter.
        confidence_threshold: τ. Adapter prediction is used only when
            its softmax max exceeds this. Default 0.5.
        sam3_to_trainid: optional mapping override. Defaults to the global
            SAM3_TO_CITYSCAPES_TRAINID dict.

    Returns:
        ``(B, H, W)`` fused trainID prediction.
    """
    if sam3_to_trainid is None:
        sam3_to_trainid = SAM3_TO_CITYSCAPES_TRAINID

    probs = F.softmax(adapter_logits, dim=1)  # (B, 14, H, W)
    conf, c_star = probs.max(dim=1)  # both (B, H, W)

    # Build a lookup from sam3 class index → trainID (-1 for unmapped).
    lut = torch.full((adapter_logits.shape[1],), -1, dtype=torch.long,
                     device=adapter_logits.device)
    for k, v in sam3_to_trainid.items():
        if 0 <= k < lut.shape[0]:
            lut[k] = v

    adapter_trainid = lut[c_star]  # (B, H, W) with -1 where unmapped
    adapter_active = (conf > confidence_threshold) & (adapter_trainid >= 0)

    fused = torch.where(adapter_active, adapter_trainid, cluster_head_27_pred)
    return fused
