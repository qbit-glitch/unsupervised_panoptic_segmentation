"""LoRA/DoRA injection for depth estimation models (DAv3 / DepthPro).

Both use ViT-based encoders (DINOv2/DINOv3) + DPT/decoder heads.
We inject adapters into the encoder blocks and optionally the decoder.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

import torch.nn as nn

from mbps_pytorch.models.adapters.lora_layers import (
    ADAPTER_CLASSES,
    wrap_linear_if_match,
    count_total_params,
)

logger = logging.getLogger(__name__)


def _find_encoder_blocks(model: nn.Module) -> List[nn.Module]:
    """Heuristic to find transformer encoder blocks in depth models.

    Checks common paths:
        - model.encoder (DAv3 / DepthPro HF)
        - model.blocks
        - model.vit.blocks
        - model.backbone.blocks
        - model.encoder.layer (HF BERT-style)
    """
    candidates = [
        "encoder",
        "blocks",
        "vit.blocks",
        "backbone.blocks",
        "encoder.layer",
        "backbone.encoder.layer",  # HF Depth Anything V2
    ]
    for path in candidates:
        parts = path.split(".")
        obj = model
        for p in parts:
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if obj is not None:
            if isinstance(obj, nn.ModuleList):
                return list(obj)
            if hasattr(obj, "__iter__"):
                blocks = []
                for b in obj:
                    if isinstance(b, nn.Identity):
                        continue
                    blocks.append(b)
                return blocks
    return []


def inject_lora_into_depth_model(
    model: nn.Module,
    variant: str = "dora",
    rank: int = 4,
    alpha: float = 4.0,
    dropout: float = 0.05,
    late_block_start: int = 6,
    adapt_decoder: bool = False,
) -> Dict[str, int]:
    """Inject LoRA/DoRA adapters into a depth estimation model encoder.

    Args:
        model: Depth model (DAv3 or DepthPro).
        variant: "lora", "dora", or "conv_dora".
        rank: LoRA rank.
        alpha: Scaling factor.
        dropout: Dropout probability.
        late_block_start: First block for full (attn+MLP) adaptation.
        adapt_decoder: Also adapt decoder head final layers (not recommended
            for pseudo-label stability; encoder adaptation is usually enough).

    Returns:
        Dict mapping adapted layer names to trainable param counts.
    """
    if variant not in ADAPTER_CLASSES:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(ADAPTER_CLASSES)}")
    adapter_cls = ADAPTER_CLASSES[variant]

    blocks = _find_encoder_blocks(model)
    if not blocks:
        logger.warning("Could not find encoder blocks. Trying generic module walk.")
        # Fallback: walk all modules looking for attention + mlp patterns
        return _inject_generic_vit(model, adapter_cls, rank, alpha, dropout, late_block_start)

    n_blocks = len(blocks)
    logger.info(
        "Injecting %s (r=%d, alpha=%.1f) into %d depth encoder blocks",
        variant.upper(), rank, alpha, n_blocks,
    )

    adapted: Dict[str, int] = {}
    for block_idx, block in enumerate(blocks):
        is_late = block_idx >= late_block_start

        # Try both CAUSE-style and HF-style paths
        target_groups = []

        # Group 1: CAUSE-style (fused qkv)
        if is_late:
            target_groups.append([
                ("attn.qkv", "qkv"),
                ("attn.proj", "proj"),
                ("mlp.fc1", "fc1"),
                ("mlp.fc2", "fc2"),
            ])
        else:
            target_groups.append([("attn.qkv", "qkv")])

        # Group 2: HF-style (separate Q,K,V)
        if is_late:
            target_groups.append([
                ("attention.attention.query", "query"),
                ("attention.attention.value", "value"),
                ("attention.attention.key", "key"),
                ("attention.output.dense", "proj"),
                ("mlp.fc1", "fc1"),
                ("mlp.fc2", "fc2"),
            ])
        else:
            target_groups.append([
                ("attention.attention.query", "query"),
                ("attention.attention.value", "value"),
            ])

        found_any = False
        for group in target_groups:
            group_found = 0
            attn_found = False
            for target_path, short_name in group:
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
                    adapted[f"encoder.block_{block_idx}.{short_name}"] = count
                    group_found += 1
                    if "attn" in target_path or "attention" in target_path:
                        attn_found = True
            # Only claim success if we found at least one attention layer.
            # This prevents MLP-only partial matches from one architecture
            # style blocking the correct attention+MLP matches of another style.
            if group_found > 0 and attn_found:
                found_any = True
                break  # Stop after first successful group

        if not found_any and is_late:
            logger.warning("Block %d: no layers adapted", block_idx)

    # Optionally adapt decoder head
    if adapt_decoder:
        decoder_adapted = _adapt_depth_decoder(model, adapter_cls, rank, alpha, dropout)
        adapted.update(decoder_adapted)

    total_adapted = sum(adapted.values())
    total_model = count_total_params(model)
    logger.info(
        "Depth model %s injection: %d layers adapted, +%d trainable params (%.4f%% of model)",
        variant.upper(), len(adapted), total_adapted,
        total_adapted / max(total_model, 1) * 100,
    )
    return adapted


def _inject_generic_vit(
    model: nn.Module,
    adapter_cls,
    rank: int,
    alpha: float,
    dropout: float,
    late_block_start: int,
) -> Dict[str, int]:
    """Fallback generic injection with approximate tiering for custom ViTs.

    Groups attention-related linear layers by their parent module to
    approximate transformer blocks, then applies tiered adaptation:
        - Early blocks: Q / V / qkv only
        - Late blocks:  Q + K + V + proj + fc1 + fc2
    """
    adapted: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # 1. Collect candidate layers
    # ------------------------------------------------------------------
    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        name_lower = name.lower()
        # Attention projections
        if "attn" in name_lower or "attention" in name_lower:
            candidates.append((name, module))
        # MLP layers (only adapted in late blocks)
        elif "mlp" in name_lower and ("fc1" in name_lower or "fc2" in name_lower):
            candidates.append((name, module))

    if not candidates:
        logger.warning("Generic injection: no candidate layers found")
        return adapted

    # ------------------------------------------------------------------
    # 2. Group by parent module to approximate blocks
    # ------------------------------------------------------------------
    from collections import OrderedDict
    block_groups = OrderedDict()
    for name, module in candidates:
        parent = ".".join(name.split(".")[:-1])
        if parent not in block_groups:
            block_groups[parent] = []
        block_groups[parent].append((name, module))

    # Sort parent names to approximate depth ordering (natural sort)
    sorted_parents = sorted(
        block_groups.keys(),
        key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)],
    )
    n_blocks = len(sorted_parents)

    if n_blocks == 0:
        return adapted

    logger.info(
        "Generic injection: %d candidate layers across %d approximate blocks",
        len(candidates), n_blocks,
    )

    # ------------------------------------------------------------------
    # 3. Tiered adaptation
    # ------------------------------------------------------------------
    for block_idx, parent in enumerate(sorted_parents):
        is_late = block_idx >= min(late_block_start, max(n_blocks - 1, 1))
        layers = block_groups[parent]

        for full_name, module in layers:
            attr_name = full_name.split(".")[-1]
            name_lower = full_name.lower()

            # Identify layer type
            attr_lower = attr_name.lower()
            is_q = "query" in name_lower or attr_lower in {"q", "query", "attn_q"}
            is_v = "value" in name_lower or attr_lower in {"v", "value", "attn_v"}
            is_qkv = "qkv" in name_lower
            is_k = ("key" in name_lower and not is_qkv) or attr_lower in {"k", "key", "attn_k"}
            is_proj = any(k in name_lower for k in ["proj", "dense", "output"])
            is_fc1 = "fc1" in name_lower
            is_fc2 = "fc2" in name_lower

            # Tiered rules
            should_adapt = False
            if is_q or is_v or is_qkv:
                should_adapt = True  # Q, V, qkv always adapted
            elif is_late:
                if is_k or is_proj or is_fc1 or is_fc2:
                    should_adapt = True

            if not should_adapt:
                continue

            # Navigate to parent module
            parent_parts = parent.split(".")
            parent_module = model
            for p in parent_parts:
                if p:
                    parent_module = getattr(parent_module, p)

            count = wrap_linear_if_match(
                parent_module, attr_name, adapter_cls, rank, alpha, dropout
            )
            if count is not None:
                adapted[full_name] = count

    total_adapted = sum(adapted.values())
    logger.info(
        "Generic %s injection: %d layers adapted, +%d params",
        adapter_cls.__name__, len(adapted), total_adapted,
    )
    return adapted


def _adapt_depth_decoder(
    model: nn.Module,
    adapter_cls,
    rank: int,
    alpha: float,
    dropout: float,
) -> Dict[str, int]:
    """Adapt final decoder projection layers."""
    adapted: Dict[str, int] = {}
    # Common decoder head paths
    decoder_paths = [
        "head", "decoder", "depth_head", "prediction_head",
    ]
    for path in decoder_paths:
        module = getattr(model, path, None)
        if module is None:
            continue
        for name, child in module.named_modules():
            if isinstance(child, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                attr_name = name.split(".")[-1]
                parent = module
                for p in parent_name.split("."):
                    if not p:
                        continue
                    parent = getattr(parent, p)
                count = wrap_linear_if_match(parent, attr_name, adapter_cls, rank, alpha, dropout)
                if count is not None:
                    adapted[f"{path}.{name}"] = count
    return adapted
