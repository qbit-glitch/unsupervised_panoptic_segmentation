"""Adapter injection utilities for LoRA/DoRA in MBPS.

Supports:
    - DINOv2 ViT backbone (semantic feature extraction)
    - CAUSE-TR Segment_TR head (semantic decoder)
    - Depth models: Depth Anything V3, DepthPro (monocular depth)
"""

from mbps_pytorch.models.adapters.lora_layers import (
    LoRALinear,
    DoRALinear,
    ConvDoRALinear,
    LoRAConv2d,
    freeze_non_adapter_params,
    count_adapter_params,
    count_total_params,
)

from mbps_pytorch.models.adapters.dinov2_adapter import (
    inject_lora_into_dinov2,
    set_dinov2_spatial_dims,
)

from mbps_pytorch.models.adapters.cause_adapter import (
    inject_lora_into_cause_tr,
)

from mbps_pytorch.models.adapters.depth_adapter import (
    inject_lora_into_depth_model,
)

from mbps_pytorch.models.adapters.depthpro_adapter import (
    inject_lora_into_depthpro,
)

__all__ = [
    "LoRALinear",
    "DoRALinear",
    "ConvDoRALinear",
    "LoRAConv2d",
    "freeze_non_adapter_params",
    "count_adapter_params",
    "count_total_params",
    "inject_lora_into_dinov2",
    "set_dinov2_spatial_dims",
    "inject_lora_into_cause_tr",
    "inject_lora_into_depth_model",
    "inject_lora_into_depthpro",
]
