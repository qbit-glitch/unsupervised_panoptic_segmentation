"""QueryPool factory: standard / decoupled (N1) / depth_bias (N2)."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

__all__ = ["build_query_pool", "register_query_pool"]

_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_query_pool(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    def decorator(cls: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        _REGISTRY[name] = cls
        return cls
    return decorator


def build_query_pool(kind: str, **kwargs) -> nn.Module:
    if kind not in _REGISTRY:
        raise KeyError(f"unknown query pool kind={kind}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[kind](**kwargs)


@register_query_pool("standard")
class StandardQueryPool(nn.Module):
    def __init__(self, num_queries: int = 100, embed_dim: int = 256) -> None:
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.query_feat = nn.Embedding(num_queries, embed_dim)

    def forward(self, batch_size: int, **kwargs) -> torch.Tensor:
        return self.query_feat.weight.unsqueeze(0).expand(batch_size, -1, -1)


@register_query_pool("decoupled")
class DecoupledQueryPool(nn.Module):
    """N1: Two learnable pools (stuff + thing) concatenated as one sequence."""

    def __init__(self, num_queries_stuff: int = 150, num_queries_thing: int = 50, embed_dim: int = 256) -> None:
        super().__init__()
        self.num_stuff = num_queries_stuff
        self.num_thing = num_queries_thing
        self.num_queries = num_queries_stuff + num_queries_thing
        self.stuff = nn.Embedding(num_queries_stuff, embed_dim)
        self.thing = nn.Embedding(num_queries_thing, embed_dim)

    def forward(self, batch_size: int, **kwargs) -> torch.Tensor:
        q = torch.cat([self.stuff.weight, self.thing.weight], dim=0)
        return q.unsqueeze(0).expand(batch_size, -1, -1)


@register_query_pool("depth_bias")
class DepthBiasQueryPool(nn.Module):
    """N2: FiLM modulation of a standard pool using mean image depth."""

    def __init__(self, num_queries: int = 100, embed_dim: int = 256) -> None:
        super().__init__()
        self.base = nn.Embedding(num_queries, embed_dim)
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2 * embed_dim),
        )

    def forward(self, batch_size: int, depth: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        q = self.base.weight.unsqueeze(0).expand(batch_size, -1, -1).clone()
        if depth is None:
            return q
        # depth: B, 1, H, W -> B, 1 (mean) -> B, 2*embed -> (gamma, beta)
        d = depth.flatten(1).mean(-1, keepdim=True)          # B, 1
        gamma_beta = self.depth_mlp(d)                       # B, 2*C
        C = q.shape[-1]
        gamma, beta = gamma_beta[:, :C].unsqueeze(1), gamma_beta[:, C:].unsqueeze(1)
        return q * (1 + gamma) + beta
