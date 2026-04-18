from __future__ import annotations

import torch

from cups.model.modeling.mask2former.masked_attention_decoder import MaskedAttentionDecoder
from cups.model.modeling.mask2former.query_pool import build_query_pool


def test_decoder_shapes() -> None:
    torch.manual_seed(0)
    B, C, Q = 2, 256, 100
    multi_scale = [
        torch.randn(B, C, 8, 16),
        torch.randn(B, C, 4, 8),
        torch.randn(B, C, 2, 4),
    ]
    mask_feat = torch.randn(B, C, 16, 32)
    pool = build_query_pool(kind="standard", num_queries=Q, embed_dim=C)
    decoder = MaskedAttentionDecoder(
        hidden_dim=C, num_queries=Q, num_classes=20, num_layers=3, num_heads=8, query_pool=pool,
    ).eval()
    with torch.no_grad():
        out = decoder(mask_feat=mask_feat, multi_scale=multi_scale)
    # Main + 2 aux = 3 entries in aux_outputs? No — we surface main + aux_outputs separately.
    assert out["pred_logits"].shape == (B, Q, 21)   # K + 1 (no-object)
    assert out["pred_masks"].shape == (B, Q, 16, 32)
    assert len(out["aux_outputs"]) == 2
    for aux in out["aux_outputs"]:
        assert aux["pred_logits"].shape == (B, Q, 21)
        assert aux["pred_masks"].shape == (B, Q, 16, 32)


def test_decoder_returns_query_embeds() -> None:
    torch.manual_seed(1)
    B, C, Q = 1, 128, 50
    multi_scale = [torch.randn(B, C, 8, 16)]
    mask_feat = torch.randn(B, C, 16, 32)
    pool = build_query_pool(kind="standard", num_queries=Q, embed_dim=C)
    decoder = MaskedAttentionDecoder(
        hidden_dim=C, num_queries=Q, num_classes=10, num_layers=1, num_heads=4, query_pool=pool,
    ).eval()
    with torch.no_grad():
        out = decoder(mask_feat=mask_feat, multi_scale=multi_scale, return_query_embeds=True)
    assert out["query_embeds"].shape == (B, Q, C)


def test_decoder_uses_depth_bias_pool() -> None:
    """N2 hook: depth-biased query init must produce different output on same input
    when depth differs."""
    torch.manual_seed(2)
    B, C, Q = 2, 128, 32
    multi_scale = [torch.randn(B, C, 4, 8)]
    mask_feat = torch.randn(B, C, 8, 16)
    pool = build_query_pool(kind="depth_bias", num_queries=Q, embed_dim=C)
    decoder = MaskedAttentionDecoder(
        hidden_dim=C, num_queries=Q, num_classes=5, num_layers=1, num_heads=4, query_pool=pool,
    ).eval()
    depth_a = torch.full((B, 1, 64, 128), 0.1)
    depth_b = torch.full((B, 1, 64, 128), 0.9)
    with torch.no_grad():
        out_a = decoder(mask_feat=mask_feat, multi_scale=multi_scale, depth=depth_a)
        out_b = decoder(mask_feat=mask_feat, multi_scale=multi_scale, depth=depth_b)
    assert not torch.allclose(out_a["pred_masks"], out_b["pred_masks"])
