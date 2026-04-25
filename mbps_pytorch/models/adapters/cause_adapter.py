"""LoRA/DoRA injection for CAUSE-TR Segment_TR head.

Adapts the TRDecoder linear layers and optional 1x1 conv projections.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch.nn as nn

from mbps_pytorch.models.adapters.lora_layers import (
    ADAPTER_CLASSES,
    wrap_linear_if_match,
    wrap_conv2d_if_match,
    count_total_params,
)

logger = logging.getLogger(__name__)


def inject_lora_into_cause_tr(
    segment_tr: nn.Module,
    variant: str = "dora",
    rank: int = 4,
    alpha: float = 4.0,
    dropout: float = 0.05,
    adapt_head: bool = True,
    adapt_projection: bool = False,
    adapt_ema: bool = False,
) -> Dict[str, int]:
    """Inject LoRA/DoRA adapters into CAUSE-TR Segment_TR head.

    Targets:
        - TRDecoder: self_attn.out_proj, multihead_attn.out_proj,
                     linear1, linear2
        - Optional: f1, f2 Conv2d 1x1 projections
        - Optional: projection_head Conv2d

    Args:
        segment_tr: Segment_TR module.
        variant: "lora", "dora", or "conv_dora".
        rank: LoRA rank.
        alpha: Scaling factor.
        dropout: Dropout probability.
        adapt_head: Adapt the TRDecoder (self_attn, multihead_attn, FFN).
        adapt_projection: Adapt projection_head and linear (1x1 convs).
        adapt_ema: Also adapt the EMA head (head_ema, projection_head_ema).
            Usually False: EMA should remain frozen as teacher.

    Returns:
        Dict mapping adapted layer names to trainable param counts.
    """
    if variant not in ADAPTER_CLASSES:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(ADAPTER_CLASSES)}")
    adapter_cls = ADAPTER_CLASSES[variant]

    adapted: Dict[str, int] = {}

    def _adapt_decoder(decoder: nn.Module, prefix: str) -> None:
        """Adapt a TRDecoder inside Decoder."""
        tr = getattr(decoder, "tr", None)
        if tr is None:
            return

        # self_attn.out_proj
        if hasattr(tr, "self_attn"):
            count = wrap_linear_if_match(
                tr.self_attn, "out_proj", adapter_cls, rank, alpha, dropout
            )
            if count is not None:
                adapted[f"{prefix}.tr.self_attn.out_proj"] = count

        # multihead_attn.out_proj
        if hasattr(tr, "multihead_attn"):
            count = wrap_linear_if_match(
                tr.multihead_attn, "out_proj", adapter_cls, rank, alpha, dropout
            )
            if count is not None:
                adapted[f"{prefix}.tr.multihead_attn.out_proj"] = count

        # FFN layers
        for attr in ("linear1", "linear2"):
            count = wrap_linear_if_match(tr, attr, adapter_cls, rank, alpha, dropout)
            if count is not None:
                adapted[f"{prefix}.tr.{attr}"] = count

        # 1x1 conv projections f1, f2
        if adapt_projection:
            for attr in ("f1", "f2"):
                conv_mod = getattr(tr, attr, None)
                if isinstance(conv_mod, nn.Conv2d):
                    count = wrap_conv2d_if_match(tr, attr, rank, alpha, dropout)
                    if count is not None:
                        adapted[f"{prefix}.tr.{attr}"] = count
                elif isinstance(conv_mod, nn.Sequential):
                    for idx, layer in enumerate(conv_mod):
                        if isinstance(layer, nn.Conv2d):
                            wrapped = wrap_conv2d_if_match(
                                conv_mod, str(idx), rank, alpha, dropout
                            )
                            if wrapped is not None:
                                adapted[f"{prefix}.tr.{attr}[{idx}]"] = wrapped

    # Adapt main head
    if adapt_head and hasattr(segment_tr, "head"):
        _adapt_decoder(segment_tr.head, "head")

    # Adapt projection head (1x1 conv)
    if adapt_projection and hasattr(segment_tr, "projection_head"):
        proj = segment_tr.projection_head
        if hasattr(proj, "f") and isinstance(proj.f, nn.Conv2d):
            count = wrap_conv2d_if_match(proj, "f", rank, alpha, dropout)
            if count is not None:
                adapted["projection_head.f"] = count

    # Adapt linear (classifier) head
    if adapt_projection and hasattr(segment_tr, "linear"):
        lin = segment_tr.linear
        if hasattr(lin, "f") and isinstance(lin.f, nn.Conv2d):
            count = wrap_conv2d_if_match(lin, "f", rank, alpha, dropout)
            if count is not None:
                adapted["linear.f"] = count

    # Adapt EMA head (usually disabled)
    if adapt_ema:
        if hasattr(segment_tr, "head_ema"):
            _adapt_decoder(segment_tr.head_ema, "head_ema")
        if adapt_projection and hasattr(segment_tr, "projection_head_ema"):
            proj_ema = segment_tr.projection_head_ema
            if hasattr(proj_ema, "f") and isinstance(proj_ema.f, nn.Conv2d):
                count = wrap_conv2d_if_match(proj_ema, "f", rank, alpha, dropout)
                if count is not None:
                    adapted["projection_head_ema.f"] = count

    total_adapted = sum(adapted.values())
    total_model = count_total_params(segment_tr)
    logger.info(
        "CAUSE-TR %s injection: %d layers adapted, +%d trainable params (%.3f%% of head)",
        variant.upper(), len(adapted), total_adapted,
        total_adapted / max(total_model, 1) * 100,
    )
    return adapted
