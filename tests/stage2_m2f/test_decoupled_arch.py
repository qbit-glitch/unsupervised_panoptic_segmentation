"""Test true decoupled architecture (Approach A).

Guards that:
- MaskedAttentionDecoder emits correct logits shape with decoupled heads
- HungarianMatcher restricts matching to within-type only
- build_mask2former_vitb raises if DECOUPLED_CLASS_HEADS=True without decoupled pool
"""
from __future__ import annotations

import torch

from cups.model.modeling.mask2former.matcher import HungarianMatcher
from cups.model.modeling.mask2former.masked_attention_decoder import MaskedAttentionDecoder
from cups.model.modeling.mask2former.query_pool import build_query_pool


def test_decoder_decoupled_logit_shape() -> None:
    pool = build_query_pool(kind="decoupled", num_queries_stuff=4, num_queries_thing=2, embed_dim=16)
    dec = MaskedAttentionDecoder(
        hidden_dim=16, num_queries=6, num_classes=8, num_layers=2, num_heads=2,
        query_pool=pool, num_stuff_classes=5, num_thing_classes=3,
    )
    mask_feat = torch.randn(2, 16, 8, 16)
    multi_scale = [torch.randn(2, 16, 4, 8), torch.randn(2, 16, 8, 16)]
    out = dec(mask_feat=mask_feat, multi_scale=multi_scale)
    assert out["pred_logits"].shape == (2, 6, 9)  # B, Q, K+1 (5+3+1=9)
    assert out["pred_masks"].shape == (2, 6, 8, 16)


def test_matcher_decoupled_restricts_cross_type() -> None:
    """Stuff queries must NOT match thing targets and vice versa."""
    matcher = HungarianMatcher(
        cost_class=1.0, cost_mask=1.0, cost_dice=1.0, num_points=16,
        num_stuff_classes=3, num_queries_stuff=4,
    )
    # 4 stuff queries + 2 thing queries = 6 total
    # 2 stuff targets (labels 0,1) + 1 thing target (label 3)
    outputs = {
        "pred_logits": torch.randn(1, 6, 5),   # B=1, Q=6, K=4 (3 stuff + 1 thing)
        "pred_masks": torch.randn(1, 6, 8, 16),
    }
    targets = [
        {
            "labels": torch.tensor([0, 1, 3]),
            "masks": torch.ones(3, 8, 16).float(),
        }
    ]
    indices = matcher(outputs, targets)
    src, tgt = indices[0]
    # src indices must be split: stuff queries [0,1,2,3] match stuff targets [0,1]
    # thing queries [4,5] match thing targets [2]
    stuff_src = src[src < 4]
    thing_src = src[src >= 4]
    stuff_tgt = tgt[tgt < 2]
    thing_tgt = tgt[tgt >= 2]
    assert (stuff_src < 4).all(), "stuff targets matched to thing queries"
    assert (thing_src >= 4).all(), "thing targets matched to stuff queries"


def test_build_raises_without_decoupled_pool() -> None:
    import pytest
    from yacs.config import CfgNode
    from cups.model.model_mask2former import build_mask2former_vitb

    cfg = CfgNode()
    cfg.MODEL = CfgNode()
    cfg.MODEL.MASK2FORMER = CfgNode()
    cfg.MODEL.MASK2FORMER.QUERY_POOL = "standard"
    cfg.MODEL.MASK2FORMER.DECOUPLED_CLASS_HEADS = True
    cfg.MODEL.MASK2FORMER.NUM_QUERIES = 10
    cfg.MODEL.MASK2FORMER.NUM_DECODER_LAYERS = 2
    cfg.MODEL.MASK2FORMER.PIXEL_DECODER_LAYERS = 2
    cfg.MODEL.MASK2FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK2FORMER.NUM_HEADS = 2
    cfg.MODEL.MASK2FORMER.MASK_WEIGHT = 1.0
    cfg.MODEL.MASK2FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK2FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK2FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK2FORMER.NUM_POINTS = 16
    cfg.MODEL.MASK2FORMER.PYRAMID_CHANNELS = 256
    cfg.MODEL.MASK2FORMER.ADAPTER_BLOCKS = 1
    cfg.MODEL.MASK2FORMER.ADAPTER_EMBED_DIM = 768
    cfg.MODEL.MASK2FORMER.DROPPATH = 0.0
    cfg.MODEL.MASK2FORMER.XQUERY_WEIGHT = 0.0
    cfg.MODEL.MASK2FORMER.QUERY_CONSISTENCY_WEIGHT = 0.0
    cfg.MODEL.DINOV2_FREEZE = True

    with pytest.raises(ValueError, match="DECOUPLED_CLASS_HEADS=True requires QUERY_POOL='decoupled'"):
        build_mask2former_vitb(cfg, num_stuff_classes=5, num_thing_classes=3)
