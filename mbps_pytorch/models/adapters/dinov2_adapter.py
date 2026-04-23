"""LoRA/DoRA injection for DINOv2 ViT backbone.

Handles the CAUSE-repo DINOv2 implementation which uses BlockChunk wrappers.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch.nn as nn

from mbps_pytorch.models.adapters.lora_layers import (
    ADAPTER_CLASSES,
    wrap_linear_if_match,
    count_adapter_params,
    count_total_params,
)

logger = logging.getLogger(__name__)


def _flatten_blocks(model: nn.Module) -> List[nn.Module]:
    """Flatten DINOv2 blocks, handling BlockChunk wrappers.

    The CAUSE DINOv2 may wrap blocks in BlockChunk (nn.ModuleList).
    We flatten to a simple list of Block modules.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return []

    flat_blocks: List[nn.Module] = []
    for blk in blocks:
        if hasattr(blk, "__iter__") and not isinstance(blk, nn.ModuleList):
            # BlockChunk or similar
            for inner in blk:
                if isinstance(inner, nn.Identity):
                    continue
                flat_blocks.append(inner)
        elif isinstance(blk, nn.ModuleList):
            for inner in blk:
                if isinstance(inner, nn.Identity):
                    continue
                flat_blocks.append(inner)
        elif isinstance(blk, nn.Identity):
            continue
        else:
            flat_blocks.append(blk)
    return flat_blocks


def inject_lora_into_dinov2(
    model: nn.Module,
    variant: str = "dora",
    rank: int = 4,
    alpha: float = 4.0,
    dropout: float = 0.05,
    late_block_start: int = 6,
) -> Dict[str, int]:
    """Inject LoRA/DoRA adapters into a DINOv2 ViT backbone.

    Tiered strategy:
        - Blocks 0..late_block_start-1: qkv only (minimal early-block steering)
        - Blocks late_block_start..N: qkv + proj + fc1 + fc2 (full adaptation)

    Args:
        model: DINOv2 model (e.g., from dinov2_vit_base_14).
        variant: "lora", "dora", or "conv_dora".
        rank: LoRA rank r.
        alpha: Scaling factor alpha.
        dropout: Dropout on LoRA path.
        late_block_start: First block index for full adaptation.

    Returns:
        Dict mapping adapted layer names to trainable param counts.
    """
    if variant not in ADAPTER_CLASSES:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(ADAPTER_CLASSES)}")
    adapter_cls = ADAPTER_CLASSES[variant]

    blocks = _flatten_blocks(model)
    n_blocks = len(blocks)
    if n_blocks == 0:
        logger.warning("No transformer blocks found in model. No adapters injected.")
        return {}

    logger.info(
        "Injecting %s (r=%d, alpha=%.1f) into %d DINOv2 blocks "
        "(early[0:%d]=qkv-only, late[%d:%d]=full)",
        variant.upper(), rank, alpha, n_blocks,
        late_block_start, late_block_start, n_blocks,
    )

    adapted: Dict[str, int] = {}

    for block_idx, block in enumerate(blocks):
        is_late = block_idx >= late_block_start
        targets = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"] if is_late else ["attn.qkv"]

        for target_path in targets:
            parts = target_path.split(".")
            parent = block
            for part in parts[:-1]:
                parent = getattr(parent, part, None)
                if parent is None:
                    break
            if parent is None:
                continue

            attr_name = parts[-1]
            count = wrap_linear_if_match(
                parent, attr_name, adapter_cls, rank, alpha, dropout
            )
            if count is not None:
                full_name = f"blocks.{block_idx}.{target_path}"
                adapted[full_name] = count

    total_adapted = sum(adapted.values())
    total_model = count_total_params(model)
    logger.info(
        "%s injection complete: %d layers adapted, +%d trainable params (%.3f%% of %dM backbone)",
        variant.upper(), len(adapted), total_adapted,
        total_adapted / total_model * 100, total_model // 1_000_000,
    )
    return adapted


def set_dinov2_spatial_dims(model: nn.Module, h_patches: int, w_patches: int) -> None:
    """Set spatial dimensions on all ConvDoRALinear layers for spatial conv path.

    Call this before each forward pass if using conv_dora variant.
    """
    for module in model.modules():
        if hasattr(module, "_spatial_dims"):
            module._spatial_dims = (h_patches, w_patches)
