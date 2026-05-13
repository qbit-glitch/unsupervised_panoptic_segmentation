"""LoRA/DoRA injection for Apple DepthPro.

DepthPro contains THREE DINOv2-Large models:
  1. depth_pro.encoder.patch_encoder.model   (24 layers, 1024-dim)
  2. depth_pro.encoder.image_encoder.model   (24 layers, 1024-dim)
  3. fov_model.fov_encoder.model             (24 layers, 1024-dim)

Each uses HuggingFace Dinov2Layer architecture with separate Q/K/V projections.

We adapt (1) and (2) for depth quality. (3) is for focal length estimation — frozen.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch.nn as nn

from mbps_pytorch.models.adapters.lora_layers import (
    ADAPTER_CLASSES,
    wrap_linear_if_match,
    count_total_params,
)

logger = logging.getLogger(__name__)


def _inject_into_hf_dinov2(
    dinov2_model: nn.Module,
    adapter_cls,
    rank: int,
    alpha: float,
    dropout: float,
    late_block_start: int,
    prefix: str = "dinov2",
) -> Dict[str, int]:
    """Inject adapters into a HuggingFace Dinov2Model.

    HF DINOv2 structure per layer:
        layer.attention.attention.query  (Linear)
        layer.attention.attention.key    (Linear)
        layer.attention.attention.value  (Linear)
        layer.attention.output.dense     (Linear)
        layer.mlp.fc1                    (Linear)
        layer.mlp.fc2                    (Linear)
    """
    adapted: Dict[str, int] = {}

    encoder = getattr(dinov2_model, "encoder", None)
    if encoder is None:
        logger.warning("No encoder found in %s", prefix)
        return adapted

    layers = getattr(encoder, "layer", None)
    if layers is None:
        logger.warning("No encoder.layer found in %s", prefix)
        return adapted

    n_blocks = len(layers)
    logger.info(
        "Injecting %s (r=%d, alpha=%.1f) into %s — %d blocks (late=%d+)",
        adapter_cls.__name__, rank, alpha, prefix, n_blocks, late_block_start,
    )

    for block_idx, layer in enumerate(layers):
        is_late = block_idx >= late_block_start

        # Always adapt Q, V; optionally K, proj, MLP
        targets = []
        if hasattr(layer, "attention") and hasattr(layer.attention, "attention"):
            targets.append(("attention.attention.query", "query"))
            targets.append(("attention.attention.value", "value"))
            if is_late:
                targets.append(("attention.attention.key", "key"))

        if is_late:
            if hasattr(layer, "attention") and hasattr(layer.attention, "output"):
                targets.append(("attention.output.dense", "proj"))
            if hasattr(layer, "mlp"):
                targets.append(("mlp.fc1", "fc1"))
                targets.append(("mlp.fc2", "fc2"))

        for path, short_name in targets:
            parts = path.split(".")
            parent = layer
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
                adapted[f"{prefix}.layer.{block_idx}.{short_name}"] = count

    total_adapted = sum(adapted.values())
    total_model = count_total_params(dinov2_model)
    logger.info(
        "%s: %d layers adapted, +%d trainable params (%.4f%% of %dM)",
        prefix, len(adapted), total_adapted,
        total_adapted / max(total_model, 1) * 100, total_model // 1_000_000,
    )
    return adapted


def inject_lora_into_depthpro(
    depthpro_model: nn.Module,
    variant: str = "dora",
    rank: int = 4,
    alpha: float = 4.0,
    dropout: float = 0.05,
    late_block_start: int = 18,  # 24 layers total; adapt last 6
    adapt_patch_encoder: bool = True,
    adapt_image_encoder: bool = True,
    adapt_fov_encoder: bool = False,
) -> Dict[str, int]:
    """Inject LoRA/DoRA into Apple DepthPro's internal DINOv2 models.

    Args:
        depthpro_model: DepthProForDepthEstimation (HF transformers).
        variant: "lora", "dora", or "conv_dora".
        rank: LoRA rank.
        alpha: Scaling factor.
        dropout: Dropout probability.
        late_block_start: First block index for full adaptation (default 18 of 24).
        adapt_patch_encoder: Adapt the patch encoder DINOv2.
        adapt_image_encoder: Adapt the image encoder DINOv2.
        adapt_fov_encoder: Adapt the FOV encoder DINOv2 (usually not needed).

    Returns:
        Dict mapping adapted layer names to trainable param counts.
    """
    if variant not in ADAPTER_CLASSES:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(ADAPTER_CLASSES)}")
    adapter_cls = ADAPTER_CLASSES[variant]

    adapted: Dict[str, int] = {}

    # 1. Patch encoder
    if adapt_patch_encoder:
        patch_dino = depthpro_model.depth_pro.encoder.patch_encoder.model
        adapted.update(_inject_into_hf_dinov2(
            patch_dino, adapter_cls, rank, alpha, dropout,
            late_block_start, prefix="depth_pro.patch_encoder",
        ))

    # 2. Image encoder
    if adapt_image_encoder:
        image_dino = depthpro_model.depth_pro.encoder.image_encoder.model
        adapted.update(_inject_into_hf_dinov2(
            image_dino, adapter_cls, rank, alpha, dropout,
            late_block_start, prefix="depth_pro.image_encoder",
        ))

    # 3. FOV encoder (optional, usually frozen)
    if adapt_fov_encoder:
        fov_dino = depthpro_model.fov_model.fov_encoder.model
        adapted.update(_inject_into_hf_dinov2(
            fov_dino, adapter_cls, rank, alpha, dropout,
            late_block_start, prefix="fov_model.fov_encoder",
        ))

    total_adapted = sum(adapted.values())
    total_model = count_total_params(depthpro_model)
    logger.info(
        "DepthPro %s injection complete: %d layers adapted, +%d trainable params (%.4f%% of %dM total)",
        variant.upper(), len(adapted), total_adapted,
        total_adapted / max(total_model, 1) * 100, total_model // 1_000_000,
    )
    set_depthpro_spatial_dims(depthpro_model)
    return adapted


def set_depthpro_spatial_dims(model, image_size=(518, 518), patch_size=14):
    from mbps_pytorch.models.adapters.lora_layers import ConvDoRALinear
    h_patches = image_size[0] // patch_size
    w_patches = image_size[1] // patch_size
    count = 0
    for module in model.modules():
        if isinstance(module, ConvDoRALinear):
            module._spatial_dims = (h_patches, w_patches)
            count += 1
    if count > 0:
        logger.info("Set _spatial_dims=(%d, %d) on %d ConvDoRALinear layers", h_patches, w_patches, count)
