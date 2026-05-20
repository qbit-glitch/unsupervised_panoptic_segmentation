"""Unsupervised losses for the JPC-Up panoptic pseudo-label generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from mbps_pytorch.models.panoptic.jpc_up import JPCUpOutput


Tensor = torch.Tensor


@dataclass(frozen=True)
class JPCUpLossConfig:
    """Loss weights and margins for :func:`compute_jpc_up_losses`."""

    lambda_semantic: float = 1.0
    lambda_code_keep: float = 0.2
    lambda_up_down: float = 1.0
    lambda_up_edge: float = 0.05
    lambda_up_var: float = 0.05
    lambda_objectness: float = 0.5
    lambda_center: float = 0.5
    lambda_boundary: float = 0.5
    lambda_proposal_mask: float = 1.0
    lambda_embedding: float = 0.5
    lambda_semantic_purity: float = 0.3
    lambda_overlap: float = 0.1
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    embedding_margin: float = 1.0
    min_proposal_score: float = 0.0


def _zero_like_loss(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def _masked_mean(loss: Tensor, valid: Optional[Tensor]) -> Tensor:
    if valid is None:
        return loss.mean()
    valid = valid.to(dtype=loss.dtype, device=loss.device)
    if valid.dim() == loss.dim() + 1 and valid.shape[1] == 1:
        valid = valid[:, 0]
    elif valid.dim() == loss.dim() - 1:
        valid = valid.unsqueeze(1)
    while valid.dim() < loss.dim():
        valid = valid.unsqueeze(1)
    return (loss * valid).sum() / valid.sum().clamp(min=1.0)


def focal_bce_with_logits(logits: Tensor, target: Tensor, valid: Optional[Tensor], alpha: float, gamma: float) -> Tensor:
    """Binary focal BCE with optional spatial validity mask."""
    target = target.to(dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return _masked_mean(alpha_t * ((1.0 - p_t) ** gamma) * bce, valid)


def soft_semantic_loss(logits: Tensor, target: Optional[Tensor], valid: Optional[Tensor]) -> Tensor:
    """Cross-entropy for hard ``BHW`` targets or soft ``BKHW`` targets."""
    if target is None:
        return _zero_like_loss(logits)
    if target.dim() == 3:
        loss = F.cross_entropy(logits, target.long().to(logits.device), reduction="none")
        return _masked_mean(loss, valid)
    if target.dim() == 4:
        target = target.to(dtype=logits.dtype, device=logits.device)
        target = target / target.sum(dim=1, keepdim=True).clamp(min=1e-6)
        loss = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
        return _masked_mean(loss, valid)
    raise ValueError("semantic_target must be BxHxW hard labels or BxKxHxW soft labels")


def code_preservation_loss(refined_codes: Tensor, teacher_codes: Optional[Tensor], valid: Optional[Tensor]) -> Tensor:
    """Cosine preservation against DCFA/CAUSE teacher codes."""
    if teacher_codes is None:
        return _zero_like_loss(refined_codes)
    teacher_codes = teacher_codes.to(refined_codes.device, refined_codes.dtype)
    teacher_codes = F.interpolate(teacher_codes, size=refined_codes.shape[-2:], mode="bilinear", align_corners=False)
    cos = F.cosine_similarity(refined_codes, teacher_codes, dim=1)
    return _masked_mean(1.0 - cos, valid)


def downsample_consistency_loss(high_codes: Tensor, low_codes: Tensor) -> Tensor:
    """High-resolution codes should downsample back to the low-resolution seed."""
    high_down = F.interpolate(high_codes, size=low_codes.shape[-2:], mode="bilinear", align_corners=False)
    return F.l1_loss(high_down, low_codes)


def edge_aware_smoothness(codes: Tensor, boundary_logits: Tensor) -> Tensor:
    """Encourage smooth codes away from predicted boundaries."""
    boundary = torch.sigmoid(boundary_logits.detach())
    dx = (codes[:, :, :, 1:] - codes[:, :, :, :-1]).abs()
    dy = (codes[:, :, 1:, :] - codes[:, :, :-1, :]).abs()
    bx = 1.0 - boundary[:, :, :, 1:].clamp(0.0, 1.0)
    by = 1.0 - boundary[:, :, 1:, :].clamp(0.0, 1.0)
    return (dx * bx).mean() + (dy * by).mean()


def variance_preservation_loss(high_codes: Tensor, low_codes: Tensor) -> Tensor:
    """Keep upsampling from collapsing code-channel variance."""
    high_std = high_codes.flatten(2).std(dim=-1)
    low_std = low_codes.flatten(2).std(dim=-1)
    return F.l1_loss(high_std, low_std)


def dice_loss_from_probs(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    pred = pred.flatten(-2)
    target = target.flatten(-2)
    numer = 2.0 * (pred * target).sum(dim=-1)
    denom = pred.sum(dim=-1) + target.sum(dim=-1) + eps
    return 1.0 - (numer + eps) / denom


def proposal_mask_loss(mask_logits: Tensor, proposal_masks: Optional[Tensor], proposal_scores: Optional[Tensor], config: JPCUpLossConfig) -> Tensor:
    """Greedy best-query matching to class-agnostic proposal masks."""
    if proposal_masks is None or proposal_masks.numel() == 0:
        return _zero_like_loss(mask_logits)
    proposals = proposal_masks.to(dtype=mask_logits.dtype, device=mask_logits.device)
    proposals = F.interpolate(proposals, size=mask_logits.shape[-2:], mode="nearest")
    pred = torch.sigmoid(mask_logits)
    bsz, num_queries, height, width = pred.shape
    num_props = proposals.shape[1]
    pred_flat = pred.reshape(bsz, num_queries, -1)
    prop_flat = proposals.reshape(bsz, num_props, -1)
    inter = torch.einsum("bqn,bpn->bqp", pred_flat, prop_flat)
    union = pred_flat.sum(dim=-1).unsqueeze(-1) + prop_flat.sum(dim=-1).unsqueeze(1)
    dice = (2.0 * inter + 1e-6) / (union + 1e-6)
    best_q = dice.argmax(dim=1)

    losses = []
    weights = []
    for b in range(bsz):
        pred_b = pred[b, best_q[b]]
        prop_b = proposals[b]
        bce = F.binary_cross_entropy(pred_b.clamp(1e-6, 1.0 - 1e-6), prop_b, reduction="none").mean(dim=(-2, -1))
        dice_b = dice_loss_from_probs(pred_b, prop_b)
        loss_b = bce + dice_b
        if proposal_scores is None:
            weight_b = torch.ones(num_props, dtype=mask_logits.dtype, device=mask_logits.device)
        else:
            weight_b = proposal_scores[b].to(dtype=mask_logits.dtype, device=mask_logits.device)
            weight_b = torch.where(weight_b >= config.min_proposal_score, weight_b, torch.zeros_like(weight_b))
        losses.append(loss_b)
        weights.append(weight_b)
    loss = torch.stack(losses)
    weight = torch.stack(weights)
    return (loss * weight).sum() / weight.sum().clamp(min=1.0)


def objectness_target_from_proposals(proposal_masks: Optional[Tensor], output_hw: tuple[int, int], reference: Tensor) -> Optional[Tensor]:
    """Union proposal masks into a soft class-agnostic objectness target."""
    if proposal_masks is None or proposal_masks.numel() == 0:
        return None
    masks = proposal_masks.to(dtype=reference.dtype, device=reference.device)
    masks = F.interpolate(masks, size=output_hw, mode="nearest")
    return masks.max(dim=1, keepdim=True).values.clamp(0.0, 1.0)


def embedding_pull_push_loss(embeddings: Tensor, proposal_masks: Optional[Tensor], proposal_scores: Optional[Tensor], config: JPCUpLossConfig) -> Tensor:
    """Pull pixels inside each proposal together and push proposal means apart."""
    if proposal_masks is None or proposal_masks.numel() == 0:
        return _zero_like_loss(embeddings)
    masks = proposal_masks.to(dtype=embeddings.dtype, device=embeddings.device)
    masks = F.interpolate(masks, size=embeddings.shape[-2:], mode="nearest")
    bsz, num_props, _, _ = masks.shape
    losses = []
    for b in range(bsz):
        emb = embeddings[b].flatten(1).T
        mask_flat = masks[b].flatten(1)
        if proposal_scores is None:
            keep = torch.ones(num_props, dtype=torch.bool, device=embeddings.device)
            score = torch.ones(num_props, dtype=embeddings.dtype, device=embeddings.device)
        else:
            score = proposal_scores[b].to(dtype=embeddings.dtype, device=embeddings.device)
            keep = score >= config.min_proposal_score
        means = []
        pull_terms = []
        for p in range(num_props):
            if not bool(keep[p]):
                continue
            weights = mask_flat[p]
            denom = weights.sum().clamp(min=1.0)
            mean = (emb * weights.unsqueeze(-1)).sum(dim=0) / denom
            means.append(F.normalize(mean, dim=0))
            dist = ((emb - mean.unsqueeze(0)) ** 2).sum(dim=-1)
            pull_terms.append(score[p] * (dist * weights).sum() / denom)
        if pull_terms:
            pull = torch.stack(pull_terms).mean()
        else:
            pull = _zero_like_loss(embeddings[b])
        if len(means) > 1:
            mean_stack = torch.stack(means)
            sim = mean_stack @ mean_stack.T
            margin = 1.0 - config.embedding_margin
            pair_loss = F.relu(sim - margin)
            pair_mask = torch.triu(torch.ones_like(pair_loss, dtype=torch.bool), diagonal=1)
            push = pair_loss[pair_mask].mean()
        else:
            push = _zero_like_loss(embeddings[b])
        losses.append(pull + push)
    return torch.stack(losses).mean()


def semantic_purity_loss(semantic_logits: Tensor, mask_logits: Tensor, query_scores: Tensor) -> Tensor:
    """Low semantic entropy inside confident predicted object masks."""
    sem_prob = F.softmax(semantic_logits, dim=1)
    masks = torch.sigmoid(mask_logits)
    scores = torch.sigmoid(query_scores).detach()
    bsz, num_queries, height, width = masks.shape
    sem_flat = sem_prob.flatten(2)
    mask_flat = masks.flatten(2)
    weighted_sem = torch.einsum("bkn,bqn->bqk", sem_flat, mask_flat)
    denom = mask_flat.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    dist = weighted_sem / denom
    entropy = -(dist.clamp(min=1e-6) * dist.clamp(min=1e-6).log()).sum(dim=-1)
    return (entropy * scores).sum() / scores.sum().clamp(min=1.0)


def overlap_loss(mask_logits: Tensor, query_scores: Tensor) -> Tensor:
    """Penalize confident predicted masks that overlap each other."""
    masks = torch.sigmoid(mask_logits)
    scores = torch.sigmoid(query_scores).detach()
    bsz, num_queries, _, _ = masks.shape
    flat = masks.flatten(2)
    norm = flat.sum(dim=-1, keepdim=True).clamp(min=1.0)
    flat = flat / norm
    overlap = torch.bmm(flat, flat.transpose(1, 2))
    eye = torch.eye(num_queries, dtype=torch.bool, device=mask_logits.device).unsqueeze(0)
    pair_weight = scores.unsqueeze(1) * scores.unsqueeze(2)
    overlap = overlap.masked_fill(eye, 0.0)
    pair_weight = pair_weight.masked_fill(eye, 0.0)
    return (overlap * pair_weight).sum() / pair_weight.sum().clamp(min=1.0)


def compute_jpc_up_losses(
    output: JPCUpOutput,
    *,
    semantic_target: Optional[Tensor] = None,
    semantic_valid: Optional[Tensor] = None,
    teacher_codes: Optional[Tensor] = None,
    proposal_masks: Optional[Tensor] = None,
    proposal_scores: Optional[Tensor] = None,
    objectness_target: Optional[Tensor] = None,
    center_target: Optional[Tensor] = None,
    boundary_target: Optional[Tensor] = None,
    valid_mask: Optional[Tensor] = None,
    config: JPCUpLossConfig = JPCUpLossConfig(),
) -> Dict[str, Tensor]:
    """Compute optional unsupervised losses for available evidence sources.

    All target arguments are optional. Missing sources contribute zero, making
    the function useful across warmup, proposal-bank, and self-training phases.
    """
    high_hw = output.semantic_logits.shape[-2:]
    if semantic_target is not None and semantic_target.dim() >= 3:
        if semantic_target.shape[-2:] != high_hw:
            if semantic_target.dim() == 3:
                semantic_target = F.interpolate(
                    semantic_target.unsqueeze(1).float(),
                    size=high_hw,
                    mode="nearest",
                )[:, 0].long()
            else:
                semantic_target = F.interpolate(
                    semantic_target.float(),
                    size=high_hw,
                    mode="bilinear",
                    align_corners=False,
                )
    if semantic_valid is not None and semantic_valid.shape[-2:] != high_hw:
        semantic_valid = F.interpolate(semantic_valid.float().unsqueeze(1), size=high_hw, mode="nearest")[:, 0]
    if valid_mask is not None and valid_mask.shape[-2:] != high_hw:
        valid_mask = F.interpolate(valid_mask.float(), size=high_hw, mode="nearest")

    if objectness_target is None:
        objectness_target = objectness_target_from_proposals(proposal_masks, high_hw, output.objectness_logits)
    if objectness_target is not None and objectness_target.shape[-2:] != high_hw:
        objectness_target = F.interpolate(objectness_target.float(), size=high_hw, mode="nearest")
    if center_target is not None and center_target.shape[-2:] != high_hw:
        center_target = F.interpolate(center_target.float(), size=high_hw, mode="bilinear", align_corners=False)
    if boundary_target is not None and boundary_target.shape[-2:] != high_hw:
        boundary_target = F.interpolate(boundary_target.float(), size=high_hw, mode="bilinear", align_corners=False)

    losses: Dict[str, Tensor] = {}
    losses["L_semantic"] = soft_semantic_loss(output.semantic_logits, semantic_target, semantic_valid)
    losses["L_code_keep"] = code_preservation_loss(output.refined_codes, teacher_codes, valid_mask)
    losses["L_up_down"] = downsample_consistency_loss(output.refined_codes, output.low_codes)
    losses["L_up_edge"] = edge_aware_smoothness(output.refined_codes, output.boundary_logits)
    losses["L_up_var"] = variance_preservation_loss(output.refined_codes, output.low_codes)

    losses["L_objectness"] = (
        focal_bce_with_logits(output.objectness_logits, objectness_target, valid_mask, config.focal_alpha, config.focal_gamma)
        if objectness_target is not None else _zero_like_loss(output.objectness_logits)
    )
    losses["L_center"] = (
        focal_bce_with_logits(output.center_logits, center_target, valid_mask, config.focal_alpha, config.focal_gamma)
        if center_target is not None else _zero_like_loss(output.center_logits)
    )
    losses["L_boundary"] = (
        focal_bce_with_logits(output.boundary_logits, boundary_target, valid_mask, config.focal_alpha, config.focal_gamma)
        if boundary_target is not None else _zero_like_loss(output.boundary_logits)
    )
    losses["L_proposal_mask"] = proposal_mask_loss(output.mask_logits, proposal_masks, proposal_scores, config)
    losses["L_embedding"] = embedding_pull_push_loss(output.instance_embeddings, proposal_masks, proposal_scores, config)
    losses["L_semantic_purity"] = semantic_purity_loss(output.semantic_logits, output.mask_logits, output.query_scores)
    losses["L_overlap"] = overlap_loss(output.mask_logits, output.query_scores)

    weighted = (
        config.lambda_semantic * losses["L_semantic"]
        + config.lambda_code_keep * losses["L_code_keep"]
        + config.lambda_up_down * losses["L_up_down"]
        + config.lambda_up_edge * losses["L_up_edge"]
        + config.lambda_up_var * losses["L_up_var"]
        + config.lambda_objectness * losses["L_objectness"]
        + config.lambda_center * losses["L_center"]
        + config.lambda_boundary * losses["L_boundary"]
        + config.lambda_proposal_mask * losses["L_proposal_mask"]
        + config.lambda_embedding * losses["L_embedding"]
        + config.lambda_semantic_purity * losses["L_semantic_purity"]
        + config.lambda_overlap * losses["L_overlap"]
    )
    losses["loss"] = weighted
    return losses
