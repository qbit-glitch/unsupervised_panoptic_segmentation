"""MPS-safe Cluster subclass for Mode B CAUSE-TR training.

Patches `contrastive_ema_with_codebook_bank` to replace
`tensor[torch.where(mask != 0)]` with `tensor.masked_select(mask)`, which is
correctly implemented on MPS. The two are mathematically identical on CPU/CUDA;
the MPS implementation of advanced indexing via `torch.where` produces
out-of-range indices once the codebook bank grows past ~10k entries.

Inherits all bank logic from `DeviceAwareCluster` (mbps_pytorch/train_cause_dinov3.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure CAUSE modules importable.
_CAUSE_DIR = Path(__file__).resolve().parent.parent.parent / "refs" / "cause"
if str(_CAUSE_DIR) not in sys.path:
    sys.path.insert(0, str(_CAUSE_DIR))

from modules.segment_module import flatten, vqt  # noqa: E402

from mbps_pytorch.train_cause_dinov3 import DeviceAwareCluster  # noqa: E402


def _safe_masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over `tensor` where `mask` is True. MPS-safe.

    On MPS, both `tensor[torch.where(mask)]` and `tensor.masked_select(mask)`
    fail with out-of-range indices once the contrastive bank fills (mask becomes
    a 2D bool tensor with shape (N, K) where K can be in the tens of thousands).
    We compute the masked mean as `(tensor * mask) / mask.sum()` which is purely
    elementwise + reduction — both are MPS-safe.

    Returns a 0-d tensor. If the mask selects nothing, returns a zero scalar.
    """
    mask_f = mask.to(tensor.dtype)
    denom = mask_f.sum()
    if denom.item() == 0:
        return tensor.new_zeros(())
    return (tensor * mask_f).sum() / denom


class ModeBCluster(DeviceAwareCluster):
    """Cluster with MPS-safe contrastive_ema_with_codebook_bank."""

    def contrastive_ema_with_codebook_bank(
        self,
        feat: torch.Tensor,
        proj_feat: torch.Tensor,
        proj_feat_ema: torch.Tensor,
        temp: float = 0.07,
        pos_thresh: float = 0.3,
        neg_thresh: float = 0.1,
    ) -> torch.Tensor:
        """Identical math to refs/cause/modules/segment_module.py:152, but uses
        `masked_select` instead of `tensor[torch.where(mask)]` to dodge the MPS
        advanced-indexing OOB bug."""
        vq_feat = vqt(feat, self.codebook)
        norm_vq_feat = F.normalize(vq_feat, dim=2)
        flat_norm_vq_feat = flatten(norm_vq_feat)

        norm_proj_feat = F.normalize(proj_feat, dim=2)
        norm_proj_feat_ema = F.normalize(proj_feat_ema, dim=2)
        flat_norm_proj_feat_ema = flatten(norm_proj_feat_ema)

        loss_NCE_list = []
        N_per_batch = norm_vq_feat.shape[1]

        for batch_ind in range(proj_feat.shape[0]):
            anchor_vq_feat = norm_vq_feat[batch_ind]                              # (N, D)
            anchor_proj_feat = norm_proj_feat[batch_ind]                          # (N, D2)

            cs_st = anchor_proj_feat @ flat_norm_proj_feat_ema.T                  # (N, B*N)
            codebook_distance = anchor_vq_feat @ flat_norm_vq_feat.T              # (N, B*N)
            bank_codebook_distance = anchor_vq_feat @ self.flat_norm_bank_vq_feat.T  # (N, bank)

            pos_mask = codebook_distance > pos_thresh                             # (N, B*N) bool
            neg_mask = codebook_distance < neg_thresh

            auto_mask = torch.ones_like(pos_mask)
            auto_mask[
                :,
                batch_ind * N_per_batch:(batch_ind + 1) * N_per_batch,
            ].fill_diagonal_(0)
            pos_mask = pos_mask & auto_mask.bool()

            cs_teacher = cs_st / temp
            shifted = cs_teacher - cs_teacher.max(dim=1, keepdim=True)[0].detach()
            denom = (shifted.exp() * (pos_mask | neg_mask).float()).sum(dim=1, keepdim=True)
            pos_neg_loss = -shifted + torch.log(denom + 1e-12)
            loss_NCE_list.append(_safe_masked_mean(pos_neg_loss, pos_mask))

            # bank path
            if self.flat_norm_bank_proj_feat_ema.shape[0] != 0:
                cs_st_bank = anchor_proj_feat @ self.flat_norm_bank_proj_feat_ema.T  # (N, bank)

                bank_pos_mask = bank_codebook_distance > pos_thresh
                bank_neg_mask = bank_codebook_distance < neg_thresh

                cs_teacher_bank = cs_st_bank / temp
                shifted_b = cs_teacher_bank - cs_teacher_bank.max(dim=1, keepdim=True)[0].detach()
                denom_b = (shifted_b.exp() * (bank_pos_mask | bank_neg_mask).float()).sum(
                    dim=1, keepdim=True,
                )
                pos_neg_loss_bank = -shifted_b + torch.log(denom_b + 1e-12)
                loss_NCE_list.append(_safe_masked_mean(pos_neg_loss_bank, bank_pos_mask))

        return sum(loss_NCE_list) / float(len(loss_NCE_list))


__all__ = ["ModeBCluster"]
