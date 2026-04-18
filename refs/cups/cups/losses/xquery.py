"""N3 XQuery: cross-image query correspondence loss.

Given a batch with B>=2 images and their per-query embeddings
``q: (B, Q, C)``, pull queries that match across images together and
push non-matching queries apart using an InfoNCE-style symmetric loss.

The "match" between images is approximated by nearest-neighbor in
embedding space (a soft anchor). Returns 0 for B=1.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

__all__ = ["xquery_loss"]


def xquery_loss(dec_out: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], ctx: Dict) -> torch.Tensor:
    q = dec_out.get("query_embeds", None)
    if q is None:
        raise KeyError("dec_out must include 'query_embeds' for xquery_loss")
    B, Q, C = q.shape
    if B < 2:
        return q.sum() * 0.0
    t = float(ctx.get("temperature", 0.1))
    q_norm = F.normalize(q, dim=-1)
    # Pair (b, b+1) for simplicity; could be all-pairs but O(B^2) memory.
    loss = q.new_zeros([])
    count = 0
    for a, b in zip(range(B), list(range(1, B)) + [0]):
        if a == b:
            continue
        sim = q_norm[a] @ q_norm[b].T          # Q x Q
        # Symmetric InfoNCE with diagonal as positives.
        targets_ = torch.arange(Q, device=q.device)
        loss = loss + 0.5 * (F.cross_entropy(sim / t, targets_) + F.cross_entropy(sim.T / t, targets_))
        count += 1
    return loss / max(count, 1)
