"""N4 Query-consistency: EMA-teacher query-embedding alignment.

Given student query embeddings ``q_s: (B, Q, C)`` and teacher query
embeddings ``q_t`` (EMA of student, stored in ctx['teacher_query_embeds']),
minimize cosine distance on a per-image per-query basis.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

__all__ = ["query_consistency_loss"]


def query_consistency_loss(dec_out: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], ctx: Dict) -> torch.Tensor:
    q_s = dec_out.get("query_embeds", None)
    if q_s is None:
        raise KeyError("dec_out must include 'query_embeds' for query_consistency_loss")
    q_t = ctx.get("teacher_query_embeds", None)
    if q_t is None:
        return q_s.sum() * 0.0  # silently zero if no teacher yet
    B, Q, C = q_s.shape
    q_s_n = F.normalize(q_s.reshape(-1, C), dim=-1)
    q_t_n = F.normalize(q_t.reshape(-1, C), dim=-1)
    cos = (q_s_n * q_t_n).sum(-1)     # (B*Q,)
    return (1.0 - cos).mean()
