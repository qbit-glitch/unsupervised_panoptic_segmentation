"""Unsupervised losses for the JPC-Up panoptic pseudo-label generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

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
    lambda_mask_coverage: float = 1.0
    lambda_query_score: float = 0.5
    lambda_direct_proposal_mask: float = 1.0
    lambda_direct_mask_coverage: float = 0.5
    lambda_direct_query_score: float = 0.5
    lambda_query_class: float = 0.0
    lambda_proposal_residual: float = 0.05
    lambda_mask_background: float = 0.25
    lambda_embedding: float = 0.5
    lambda_graph_edge: float = 0.25
    lambda_semantic_purity: float = 0.05
    lambda_overlap: float = 0.1
    mask_pos_weight: float = 0.75
    mask_neg_weight: float = 0.25
    query_score_pos_weight: float = 1.0
    query_score_neg_weight: float = 0.1
    query_class_pos_weight: float = 1.0
    query_class_no_object_weight: float = 0.1
    no_object_class: int = 19
    matcher_score_cost: float = 0.25
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


def balanced_mask_bce_with_logits(logits: Tensor, target: Tensor, config: JPCUpLossConfig) -> Tensor:
    """Foreground-balanced BCE for sparse object masks."""
    target = target.to(dtype=logits.dtype, device=logits.device).clamp(0.0, 1.0)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pos = target
    neg = 1.0 - target
    pos_loss = (loss * pos).sum(dim=(-2, -1)) / pos.sum(dim=(-2, -1)).clamp(min=1.0)
    neg_loss = (loss * neg).sum(dim=(-2, -1)) / neg.sum(dim=(-2, -1)).clamp(min=1.0)
    return config.mask_pos_weight * pos_loss + config.mask_neg_weight * neg_loss


def _valid_proposal_scores(
    proposal_masks: Tensor,
    proposal_scores: Optional[Tensor],
    config: JPCUpLossConfig,
) -> Tensor:
    bsz, num_props = proposal_masks.shape[:2]
    if proposal_scores is None:
        score = torch.ones((bsz, num_props), dtype=proposal_masks.dtype, device=proposal_masks.device)
    else:
        score = proposal_scores.to(dtype=proposal_masks.dtype, device=proposal_masks.device)
        if score.dim() == 1:
            score = score.unsqueeze(0)
        if score.shape[1] != num_props:
            aligned = torch.zeros((bsz, num_props), dtype=score.dtype, device=score.device)
            n = min(num_props, score.shape[1])
            aligned[:, :n] = score[:, :n]
            score = aligned
    area = proposal_masks.flatten(2).sum(dim=-1)
    return torch.where((score >= config.min_proposal_score) & (area > 0), score, torch.zeros_like(score))


def _pairwise_mask_dice(pred: Tensor, target: Tensor) -> Tensor:
    """Pairwise soft Dice between ``B x Q x H x W`` and ``B x P x H x W`` masks."""
    pred_flat = pred.flatten(2)
    target_flat = target.flatten(2)
    inter = torch.einsum("bqn,bpn->bqp", pred_flat, target_flat)
    denom = pred_flat.sum(dim=-1).unsqueeze(-1) + target_flat.sum(dim=-1).unsqueeze(1)
    return (2.0 * inter + 1e-6) / (denom + 1e-6)


def _fallback_greedy_assignment(cost: Tensor, proposal_scores: Tensor) -> list[tuple[int, int]]:
    """One-to-one query/proposal matching without a scipy dependency."""
    if cost.numel() == 0:
        return []
    num_queries, num_props = cost.shape
    prop_order = torch.argsort(proposal_scores, descending=True)
    available = torch.ones(num_queries, dtype=torch.bool, device=cost.device)
    pairs: list[tuple[int, int]] = []
    for p in prop_order.tolist():
        ranked_q = torch.argsort(cost[:, p], descending=False)
        chosen = None
        for q in ranked_q.tolist():
            if bool(available[q]):
                chosen = q
                break
        if chosen is None:
            break
        available[chosen] = False
        pairs.append((int(chosen), int(p)))
        if not bool(available.any()):
            break
    return pairs


def _linear_sum_assignment_pairs(cost: Tensor, proposal_scores: Tensor) -> list[tuple[int, int]]:
    """Hungarian assignment when scipy is present, with a deterministic fallback."""
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
    except Exception:
        return _fallback_greedy_assignment(cost, proposal_scores)

    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    pairs = [(int(r), int(c)) for r, c in zip(row_ind.tolist(), col_ind.tolist())]
    pairs.sort(key=lambda item: float(proposal_scores[item[1]].detach().cpu()), reverse=True)
    return pairs


def _query_proposal_assignments(
    mask_logits: Tensor,
    query_scores: Optional[Tensor],
    proposal_masks: Optional[Tensor],
    proposal_scores: Optional[Tensor],
    config: JPCUpLossConfig,
) -> tuple[Tensor, Tensor, list[list[tuple[int, int]]]]:
    """Return resized proposals, valid scores, and one-to-one query/proposal pairs.

    This is the DETR-style supervision hinge for JPC-Up: each static teacher
    proposal can supervise at most one query, and each query can be claimed by
    at most one proposal.  Matching is detached; gradients flow only through the
    matched BCE/Dice and confidence losses.
    """
    if proposal_masks is None or proposal_masks.numel() == 0:
        empty = mask_logits.new_zeros((mask_logits.shape[0], 0, *mask_logits.shape[-2:]))
        return empty, mask_logits.new_zeros((mask_logits.shape[0], 0)), [[] for _ in range(mask_logits.shape[0])]

    proposals = proposal_masks.to(dtype=mask_logits.dtype, device=mask_logits.device)
    proposals = F.interpolate(proposals, size=mask_logits.shape[-2:], mode="nearest").clamp(0.0, 1.0)
    scores = _valid_proposal_scores(proposals, proposal_scores, config)
    pred = torch.sigmoid(mask_logits.detach())
    dice = _pairwise_mask_dice(pred, proposals)
    cost = 1.0 - dice
    if query_scores is not None:
        query_prob = torch.sigmoid(query_scores.detach()).to(dtype=cost.dtype, device=cost.device)
        cost = cost - config.matcher_score_cost * query_prob.unsqueeze(-1)

    assignments: list[list[tuple[int, int]]] = []
    for b in range(mask_logits.shape[0]):
        valid_props = torch.nonzero(scores[b] > 0, as_tuple=False).flatten()
        if valid_props.numel() == 0:
            assignments.append([])
            continue
        sub_cost = cost[b, :, valid_props]
        pairs = _linear_sum_assignment_pairs(sub_cost, scores[b, valid_props])
        assignments.append([(q, int(valid_props[p].item())) for q, p in pairs])
    return proposals, scores, assignments


def proposal_mask_loss(mask_logits: Tensor, proposal_masks: Optional[Tensor], proposal_scores: Optional[Tensor], config: JPCUpLossConfig) -> Tensor:
    """One-to-one DETR-style query matching to class-agnostic proposals."""
    proposals, score, assignments = _query_proposal_assignments(
        mask_logits,
        None,
        proposal_masks,
        proposal_scores,
        config,
    )
    if proposals.shape[1] == 0:
        return _zero_like_loss(mask_logits)
    losses = []
    weights = []
    for b, pairs in enumerate(assignments):
        for chosen, p in pairs:
            target = proposals[b, p]
            if score[b, p] <= 0:
                continue
            logit = mask_logits[b, chosen]
            bce = balanced_mask_bce_with_logits(logit.unsqueeze(0), target.unsqueeze(0), config)[0]
            dice_b = dice_loss_from_probs(torch.sigmoid(logit).unsqueeze(0), target.unsqueeze(0))[0]
            losses.append(bce + dice_b)
            weights.append(score[b, p])
    if not losses:
        return _zero_like_loss(mask_logits)
    loss = torch.stack(losses)
    weight = torch.stack(weights)
    return (loss * weight).sum() / weight.sum().clamp(min=1.0)


def mask_coverage_loss(mask_logits: Tensor, proposal_masks: Optional[Tensor], proposal_scores: Optional[Tensor], config: JPCUpLossConfig) -> Tensor:
    """Force the query set to cover foreground evidence somewhere."""
    if proposal_masks is None or proposal_masks.numel() == 0:
        return _zero_like_loss(mask_logits)
    proposals = proposal_masks.to(dtype=mask_logits.dtype, device=mask_logits.device)
    proposals = F.interpolate(proposals, size=mask_logits.shape[-2:], mode="nearest").clamp(0.0, 1.0)
    score = _valid_proposal_scores(proposals, proposal_scores, config)
    valid = (score > 0).to(dtype=proposals.dtype).unsqueeze(-1).unsqueeze(-1)
    target = (proposals * valid).max(dim=1, keepdim=True).values
    if target.sum() <= 0:
        return _zero_like_loss(mask_logits)
    coverage_logits = torch.logsumexp(mask_logits, dim=1, keepdim=True)
    cover_prob = 1.0 - torch.prod(1.0 - torch.sigmoid(mask_logits).clamp(1e-6, 1.0 - 1e-6), dim=1, keepdim=True)
    bce = balanced_mask_bce_with_logits(coverage_logits, target, config).mean()
    dice = dice_loss_from_probs(cover_prob, target).mean()
    return bce + dice


def query_score_loss(
    query_scores: Tensor,
    mask_logits: Tensor,
    proposal_masks: Optional[Tensor],
    proposal_scores: Optional[Tensor],
    config: JPCUpLossConfig,
) -> Tensor:
    """Calibrate object-query confidence from one-to-one proposal assignment."""
    if proposal_masks is None or proposal_masks.numel() == 0:
        return _zero_like_loss(query_scores)
    proposals, score, assignments = _query_proposal_assignments(
        mask_logits,
        query_scores,
        proposal_masks,
        proposal_scores,
        config,
    )
    if score.sum() <= 0:
        return _zero_like_loss(query_scores)

    target = torch.zeros_like(query_scores)
    for b, pairs in enumerate(assignments):
        for chosen, p in pairs:
            target[b, chosen] = score[b, p].clamp(0.0, 1.0)
    loss = F.binary_cross_entropy_with_logits(query_scores, target, reduction="none")
    pos = (target > 0).to(dtype=loss.dtype)
    neg = 1.0 - pos
    if pos.sum() <= 0:
        return loss.mean()
    pos_loss = (loss * pos).sum() / pos.sum().clamp(min=1.0)
    neg_loss = (loss * neg).sum() / neg.sum().clamp(min=1.0)
    return config.query_score_pos_weight * pos_loss + config.query_score_neg_weight * neg_loss


def query_class_loss(
    query_class_logits: Tensor,
    mask_logits: Tensor,
    query_scores: Tensor,
    proposal_masks: Optional[Tensor],
    proposal_scores: Optional[Tensor],
    proposal_class_ids: Optional[Tensor],
    config: JPCUpLossConfig,
) -> Tensor:
    """Supervise matched object queries with proposal-derived Cityscapes trainIDs."""
    if proposal_masks is None or proposal_masks.numel() == 0 or proposal_class_ids is None:
        return _zero_like_loss(query_class_logits)
    _, score, assignments = _query_proposal_assignments(
        mask_logits,
        query_scores,
        proposal_masks,
        proposal_scores,
        config,
    )
    if score.sum() <= 0:
        return _zero_like_loss(query_class_logits)

    num_classes = query_class_logits.shape[-1]
    no_object = min(int(config.no_object_class), num_classes - 1)
    class_ids = proposal_class_ids.to(device=query_class_logits.device, dtype=torch.long)
    if class_ids.dim() == 1:
        class_ids = class_ids.unsqueeze(0)
    if class_ids.shape[1] != score.shape[1]:
        aligned = torch.full(
            (score.shape[0], score.shape[1]),
            fill_value=no_object,
            dtype=torch.long,
            device=query_class_logits.device,
        )
        n = min(class_ids.shape[1], score.shape[1])
        aligned[:, :n] = class_ids[:, :n]
        class_ids = aligned

    target = torch.full(
        query_class_logits.shape[:2],
        fill_value=no_object,
        dtype=torch.long,
        device=query_class_logits.device,
    )
    weights = query_class_logits.new_full(
        query_class_logits.shape[:2],
        fill_value=float(config.query_class_no_object_weight),
    )
    positives = 0
    for b, pairs in enumerate(assignments):
        for chosen, p in pairs:
            cls = int(class_ids[b, p].detach().item())
            if cls < 0 or cls >= no_object or score[b, p] <= 0:
                continue
            target[b, chosen] = cls
            weights[b, chosen] = config.query_class_pos_weight * score[b, p].clamp(0.0, 1.0)
            positives += 1
    if positives == 0:
        return _zero_like_loss(query_class_logits)

    loss = F.cross_entropy(
        query_class_logits.reshape(-1, num_classes),
        target.reshape(-1),
        reduction="none",
    ).reshape_as(weights)
    return (loss * weights).sum() / weights.sum().clamp(min=1.0)


def _anchor_scores_from_indices(
    output: JPCUpOutput,
    proposal_scores: Optional[Tensor],
    config: JPCUpLossConfig,
) -> Optional[Tensor]:
    anchor_valid = getattr(output, "proposal_anchor_valid", None)
    anchor_indices = getattr(output, "proposal_anchor_indices", None)
    if anchor_valid is None or anchor_indices is None:
        return None
    valid = anchor_valid.to(device=output.mask_logits.device)
    if proposal_scores is None:
        scores = output.mask_logits.new_ones(valid.shape)
    else:
        scores_src = proposal_scores.to(device=output.mask_logits.device, dtype=output.mask_logits.dtype)
        if scores_src.dim() == 1:
            scores_src = scores_src.unsqueeze(0)
        bsz = valid.shape[0]
        if scores_src.shape[0] != bsz:
            scores_src = scores_src[:1].expand(bsz, -1)
        if scores_src.shape[1] == 0:
            return output.mask_logits.new_zeros(valid.shape)
        safe_idx = anchor_indices.clamp(min=0)
        in_range = safe_idx < scores_src.shape[1]
        safe_idx = safe_idx.clamp(max=max(scores_src.shape[1] - 1, 0))
        scores = torch.gather(scores_src, 1, safe_idx)
        valid = valid & in_range
    scores = torch.where(valid, scores.clamp(0.0, 1.0), torch.zeros_like(scores))
    return torch.where(scores >= config.min_proposal_score, scores, torch.zeros_like(scores))


def anchored_proposal_mask_loss(
    output: JPCUpOutput,
    proposal_scores: Optional[Tensor],
    config: JPCUpLossConfig,
) -> Tensor:
    """Conservative mask-refinement loss for proposal-conditioned queries."""
    anchor_masks = getattr(output, "proposal_anchor_masks", None)
    if anchor_masks is None:
        return _zero_like_loss(output.mask_logits)
    scores = _anchor_scores_from_indices(output, proposal_scores, config)
    if scores is None or scores.sum() <= 0:
        return _zero_like_loss(output.mask_logits)
    bce = balanced_mask_bce_with_logits(output.mask_logits, anchor_masks.to(output.mask_logits.device), config)
    dice = dice_loss_from_probs(torch.sigmoid(output.mask_logits), anchor_masks.to(output.mask_logits.device))
    loss = bce + dice
    return (loss * scores).sum() / scores.sum().clamp(min=1.0)


def anchored_query_score_loss(
    output: JPCUpOutput,
    proposal_scores: Optional[Tensor],
    config: JPCUpLossConfig,
) -> Tensor:
    """Score anchored proposal queries from their teacher proposal confidence."""
    scores = _anchor_scores_from_indices(output, proposal_scores, config)
    if scores is None:
        return _zero_like_loss(output.query_scores)
    target = scores.to(dtype=output.query_scores.dtype, device=output.query_scores.device)
    loss = F.binary_cross_entropy_with_logits(output.query_scores, target, reduction="none")
    pos = (target > 0).to(dtype=loss.dtype)
    neg = 1.0 - pos
    if pos.sum() <= 0:
        return (loss * neg).sum() / neg.sum().clamp(min=1.0)
    pos_loss = (loss * pos).sum() / pos.sum().clamp(min=1.0)
    neg_loss = (loss * neg).sum() / neg.sum().clamp(min=1.0)
    return config.query_score_pos_weight * pos_loss + config.query_score_neg_weight * neg_loss


def anchored_query_class_loss(
    output: JPCUpOutput,
    proposal_scores: Optional[Tensor],
    proposal_class_ids: Optional[Tensor],
    config: JPCUpLossConfig,
) -> Tensor:
    """Class-aware gating loss for proposal-conditioned queries."""
    anchor_indices = getattr(output, "proposal_anchor_indices", None)
    if anchor_indices is None or proposal_class_ids is None:
        return _zero_like_loss(output.query_class_logits)
    scores = _anchor_scores_from_indices(output, proposal_scores, config)
    if scores is None:
        return _zero_like_loss(output.query_class_logits)

    num_classes = output.query_class_logits.shape[-1]
    no_object = min(int(config.no_object_class), num_classes - 1)
    class_src = proposal_class_ids.to(device=output.query_class_logits.device, dtype=torch.long)
    if class_src.dim() == 1:
        class_src = class_src.unsqueeze(0)
    if class_src.shape[0] != anchor_indices.shape[0]:
        class_src = class_src[:1].expand(anchor_indices.shape[0], -1)
    if class_src.shape[1] == 0:
        return _zero_like_loss(output.query_class_logits)
    safe_idx = anchor_indices.clamp(min=0)
    in_range = safe_idx < class_src.shape[1]
    safe_idx = safe_idx.clamp(max=max(class_src.shape[1] - 1, 0))
    gathered = torch.gather(class_src, 1, safe_idx)

    target = torch.full_like(anchor_indices, fill_value=no_object)
    valid_pos = (scores > 0) & in_range & (gathered >= 0) & (gathered < no_object)
    target = torch.where(valid_pos, gathered, target)
    weights = output.query_class_logits.new_full(target.shape, fill_value=float(config.query_class_no_object_weight))
    weights = torch.where(valid_pos, config.query_class_pos_weight * scores, weights)
    if not bool(valid_pos.any()):
        return _zero_like_loss(output.query_class_logits)
    loss = F.cross_entropy(
        output.query_class_logits.reshape(-1, num_classes),
        target.reshape(-1),
        reduction="none",
    ).reshape_as(weights)
    return (loss * weights).sum() / weights.sum().clamp(min=1.0)


def proposal_residual_regularizer(output: JPCUpOutput) -> Tensor:
    """Keep residual refinement small around the static unMORE proposal prior."""
    delta = getattr(output, "proposal_mask_delta", None)
    valid = getattr(output, "proposal_anchor_valid", None)
    if delta is None or valid is None:
        return _zero_like_loss(output.mask_logits)
    weight = valid.to(device=delta.device, dtype=delta.dtype).unsqueeze(-1).unsqueeze(-1)
    if weight.sum() <= 0:
        return _zero_like_loss(delta)
    return (delta.abs() * weight).sum() / (weight.sum() * delta.shape[-1] * delta.shape[-2]).clamp(min=1.0)


def mask_background_suppression_loss(
    mask_logits: Tensor,
    proposal_masks: Optional[Tensor],
    proposal_scores: Optional[Tensor],
    config: JPCUpLossConfig,
) -> Tensor:
    """Suppress broad query masks outside the static object-evidence union."""
    if proposal_masks is None or proposal_masks.numel() == 0:
        return _zero_like_loss(mask_logits)
    proposals = proposal_masks.to(dtype=mask_logits.dtype, device=mask_logits.device)
    proposals = F.interpolate(proposals, size=mask_logits.shape[-2:], mode="nearest").clamp(0.0, 1.0)
    score = _valid_proposal_scores(proposals, proposal_scores, config)
    valid = (score > 0).to(dtype=proposals.dtype).unsqueeze(-1).unsqueeze(-1)
    union = (proposals * valid).max(dim=1, keepdim=True).values
    outside = 1.0 - union
    if outside.sum() <= 0:
        return _zero_like_loss(mask_logits)
    prob = torch.sigmoid(mask_logits)
    return (prob * outside).sum() / (outside.sum().clamp(min=1.0) * mask_logits.shape[1])


def mask_probability_regularizer(mask_logits: Tensor) -> Tensor:
    """Keep mask logits away from an all-zero probability floor."""
    prob = torch.sigmoid(mask_logits)
    return F.relu(1e-4 - prob.mean(dim=(-2, -1))).mean()


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
    inter = torch.bmm(flat, flat.transpose(1, 2))
    area = flat.sum(dim=-1)
    union = area.unsqueeze(1) + area.unsqueeze(2) - inter
    overlap = inter / union.clamp(min=1.0)
    eye = torch.eye(num_queries, dtype=torch.bool, device=mask_logits.device).unsqueeze(0)
    pair_weight = scores.unsqueeze(1) * scores.unsqueeze(2)
    overlap = overlap.masked_fill(eye, 0.0)
    pair_weight = pair_weight.masked_fill(eye, 0.0)
    return (overlap * pair_weight).sum() / pair_weight.sum().clamp(min=1.0)


def _node_embedding_means(emb: Tensor, superpixels: Tensor, num_nodes: int) -> Tensor:
    flat_sp = superpixels.reshape(-1)
    flat_emb = emb.flatten(1).T
    means = emb.new_zeros((num_nodes, emb.shape[0]))
    counts = emb.new_zeros((num_nodes,))
    means.index_add_(0, flat_sp, flat_emb)
    counts.index_add_(0, flat_sp, torch.ones_like(flat_sp, dtype=emb.dtype))
    return F.normalize(means / counts.clamp(min=1.0).unsqueeze(1), dim=1)


def superpixel_edge_affinity_loss(
    embeddings: Tensor,
    superpixel_ids: Optional[Tensor],
    edge_indices: Optional[Sequence[Tensor]],
    edge_targets: Optional[Sequence[Tensor]],
    edge_weights: Optional[Sequence[Tensor]],
) -> Tensor:
    """BCE on adjacent-superpixel affinities predicted from instance embeddings."""
    if superpixel_ids is None or edge_indices is None or edge_targets is None:
        return _zero_like_loss(embeddings)
    if superpixel_ids.dim() == 4:
        superpixel_ids = superpixel_ids[:, 0]
    if superpixel_ids.shape[-2:] != embeddings.shape[-2:]:
        superpixel_ids = F.interpolate(
            superpixel_ids.unsqueeze(1).float(),
            size=embeddings.shape[-2:],
            mode="nearest",
        )[:, 0].long()
    else:
        superpixel_ids = superpixel_ids.long().to(embeddings.device)

    losses = []
    for b, edges in enumerate(edge_indices):
        if edges is None or edges.numel() == 0:
            continue
        target = edge_targets[b].to(device=embeddings.device, dtype=embeddings.dtype)
        if target.numel() == 0:
            continue
        edges = edges.to(device=embeddings.device, dtype=torch.long)
        sp_b = superpixel_ids[b]
        num_nodes = int(sp_b.max().item()) + 1 if sp_b.numel() else 0
        if num_nodes <= 0:
            continue
        node_emb = _node_embedding_means(embeddings[b], sp_b, num_nodes)
        u = edges[:, 0].clamp(0, num_nodes - 1)
        v = edges[:, 1].clamp(0, num_nodes - 1)
        pred = ((node_emb[u] * node_emb[v]).sum(dim=1) + 1.0) * 0.5
        weight = (
            edge_weights[b].to(device=embeddings.device, dtype=embeddings.dtype)
            if edge_weights is not None and edge_weights[b] is not None
            else torch.ones_like(target)
        )
        bce = F.binary_cross_entropy(pred.clamp(1e-6, 1.0 - 1e-6), target.clamp(0.0, 1.0), reduction="none")
        losses.append((bce * weight).sum() / weight.sum().clamp(min=1.0))
    if not losses:
        return _zero_like_loss(embeddings)
    return torch.stack(losses).mean()


def compute_jpc_up_losses(
    output: JPCUpOutput,
    *,
    semantic_target: Optional[Tensor] = None,
    semantic_valid: Optional[Tensor] = None,
    teacher_codes: Optional[Tensor] = None,
    proposal_masks: Optional[Tensor] = None,
    proposal_scores: Optional[Tensor] = None,
    direct_proposal_masks: Optional[Tensor] = None,
    direct_proposal_scores: Optional[Tensor] = None,
    direct_proposal_class_ids: Optional[Tensor] = None,
    objectness_target: Optional[Tensor] = None,
    center_target: Optional[Tensor] = None,
    boundary_target: Optional[Tensor] = None,
    superpixel_ids: Optional[Tensor] = None,
    edge_indices: Optional[Sequence[Tensor]] = None,
    edge_targets: Optional[Sequence[Tensor]] = None,
    edge_weights: Optional[Sequence[Tensor]] = None,
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
    losses["L_mask_coverage"] = mask_coverage_loss(output.mask_logits, proposal_masks, proposal_scores, config)
    losses["L_query_score"] = query_score_loss(output.query_scores, output.mask_logits, proposal_masks, proposal_scores, config)
    has_anchors = getattr(output, "proposal_anchor_valid", None) is not None
    losses["L_direct_proposal_mask"] = (
        anchored_proposal_mask_loss(output, direct_proposal_scores, config)
        if has_anchors
        else proposal_mask_loss(
            output.mask_logits,
            direct_proposal_masks,
            direct_proposal_scores,
            config,
        )
    )
    losses["L_direct_mask_coverage"] = mask_coverage_loss(
        output.mask_logits,
        direct_proposal_masks,
        direct_proposal_scores,
        config,
    )
    losses["L_direct_query_score"] = (
        anchored_query_score_loss(output, direct_proposal_scores, config)
        if has_anchors
        else query_score_loss(
            output.query_scores,
            output.mask_logits,
            direct_proposal_masks,
            direct_proposal_scores,
            config,
        )
    )
    losses["L_query_class"] = (
        anchored_query_class_loss(output, direct_proposal_scores, direct_proposal_class_ids, config)
        if has_anchors
        else query_class_loss(
            output.query_class_logits,
            output.mask_logits,
            output.query_scores,
            direct_proposal_masks,
            direct_proposal_scores,
            direct_proposal_class_ids,
            config,
        )
    )
    losses["L_proposal_residual"] = proposal_residual_regularizer(output)
    losses["L_mask_background"] = mask_background_suppression_loss(output.mask_logits, proposal_masks, proposal_scores, config)
    losses["L_mask_prob_floor"] = mask_probability_regularizer(output.mask_logits)
    losses["L_embedding"] = embedding_pull_push_loss(output.instance_embeddings, proposal_masks, proposal_scores, config)
    losses["L_graph_edge"] = superpixel_edge_affinity_loss(
        output.instance_embeddings,
        superpixel_ids,
        edge_indices,
        edge_targets,
        edge_weights,
    )
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
        + config.lambda_mask_coverage * losses["L_mask_coverage"]
        + config.lambda_query_score * losses["L_query_score"]
        + config.lambda_direct_proposal_mask * losses["L_direct_proposal_mask"]
        + config.lambda_direct_mask_coverage * losses["L_direct_mask_coverage"]
        + config.lambda_direct_query_score * losses["L_direct_query_score"]
        + config.lambda_query_class * losses["L_query_class"]
        + config.lambda_proposal_residual * losses["L_proposal_residual"]
        + config.lambda_mask_background * losses["L_mask_background"]
        + 0.01 * losses["L_mask_prob_floor"]
        + config.lambda_embedding * losses["L_embedding"]
        + config.lambda_graph_edge * losses["L_graph_edge"]
        + config.lambda_semantic_purity * losses["L_semantic_purity"]
        + config.lambda_overlap * losses["L_overlap"]
    )
    losses["loss"] = weighted
    return losses
