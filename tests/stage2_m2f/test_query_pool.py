from __future__ import annotations

import torch

from cups.model.modeling.mask2former.query_pool import build_query_pool


def test_standard_pool_shape() -> None:
    pool = build_query_pool(kind="standard", num_queries=100, embed_dim=256)
    q = pool(batch_size=2)
    assert q.shape == (2, 100, 256)


def test_decoupled_pool_concatenates() -> None:
    pool = build_query_pool(kind="decoupled", num_queries_stuff=150, num_queries_thing=50, embed_dim=256)
    q = pool(batch_size=2)
    assert q.shape == (2, 200, 256)


def test_depth_bias_pool_uses_depth() -> None:
    pool = build_query_pool(kind="depth_bias", num_queries=100, embed_dim=256)
    depth = torch.rand(2, 1, 64, 128)
    q = pool(batch_size=2, depth=depth)
    assert q.shape == (2, 100, 256)
    q_no_depth = pool(batch_size=2)
    # Depth-biased pool differs from no-depth pool.
    assert not torch.allclose(q, q_no_depth)


def test_unknown_kind_raises() -> None:
    import pytest
    with pytest.raises(KeyError):
        build_query_pool(kind="bogus", num_queries=10, embed_dim=16)
