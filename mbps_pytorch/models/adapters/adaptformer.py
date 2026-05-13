"""AdaptFormer adapters for DINOv2 ViT backbones.

AdaptFormer is a bottleneck residual adapter placed beside the transformer MLP.
Unlike LoRA/DoRA, it does not modify the linear layer weights directly: each
late-block MLP becomes ``base_mlp(x) + adapter(x)``.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _flatten_blocks(model: nn.Module) -> List[nn.Module]:
    """Flatten DINOv2 blocks, including BlockChunk/ModuleList wrappers."""
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return []

    flat_blocks: List[nn.Module] = []
    for block in blocks:
        if isinstance(block, nn.Identity):
            continue
        if isinstance(block, nn.ModuleList) or (
            hasattr(block, "__iter__") and not isinstance(block, nn.Module)
        ):
            for inner in block:
                if not isinstance(inner, nn.Identity):
                    flat_blocks.append(inner)
        else:
            flat_blocks.append(block)
    return flat_blocks


class AdaptFormerAdapter(nn.Module):
    """Small bottleneck residual adapter used beside a transformer MLP."""

    def __init__(
        self,
        dim: int = 768,
        bottleneck_dim: int = 64,
        dropout: float = 0.05,
        init_scale: float = 1e-3,
        learnable_scale: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Linear(bottleneck_dim, dim)

        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        scale = torch.tensor(float(init_scale))
        if learnable_scale:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.down(x)
        z = self.act(z)
        z = self.dropout(z)
        z = self.up(z)
        return self.scale.to(dtype=x.dtype) * z

    def trainable_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdaptFormerMlpWrapper(nn.Module):
    """Wrap a ViT MLP with an AdaptFormer residual branch."""

    def __init__(
        self,
        mlp: nn.Module,
        dim: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.05,
        init_scale: float = 1e-3,
        learnable_scale: bool = True,
    ) -> None:
        super().__init__()
        self.base_mlp = mlp
        self.adaptformer = AdaptFormerAdapter(
            dim=dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            init_scale=init_scale,
            learnable_scale=learnable_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_mlp(x) + self.adaptformer(x)


def inject_adaptformer_into_dinov2(
    model: nn.Module,
    bottleneck_dim: int = 64,
    dropout: float = 0.05,
    init_scale: float = 1e-3,
    late_block_start: int = 6,
    learnable_scale: bool = True,
) -> Dict[str, int]:
    """Inject AdaptFormer MLP-side adapters into late DINOv2 blocks."""
    blocks = _flatten_blocks(model)
    if not blocks:
        logger.warning("No transformer blocks found in model. No AdaptFormer injected.")
        return {}

    adapted: Dict[str, int] = {}
    for block_idx, block in enumerate(blocks):
        if block_idx < late_block_start or not hasattr(block, "mlp"):
            continue
        mlp = block.mlp
        if isinstance(mlp, AdaptFormerMlpWrapper):
            continue

        dim = None
        fc1 = getattr(mlp, "fc1", None)
        if fc1 is not None and hasattr(fc1, "in_features"):
            dim = int(fc1.in_features)
        if dim is None:
            logger.warning("Skipping block %d: could not infer MLP input dim.", block_idx)
            continue

        wrapper = AdaptFormerMlpWrapper(
            mlp,
            dim=dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            init_scale=init_scale,
            learnable_scale=learnable_scale,
        )
        block.mlp = wrapper
        adapted[f"blocks.{block_idx}.mlp.adaptformer"] = wrapper.adaptformer.trainable_count()

    logger.info(
        "AdaptFormer injection complete: %d blocks, +%d trainable params",
        len(adapted),
        sum(adapted.values()),
    )
    return adapted


def freeze_non_adaptformer_params(model: nn.Module) -> None:
    """Freeze all parameters except AdaptFormer adapter parameters."""
    for name, param in model.named_parameters():
        param.requires_grad = ".adaptformer." in name


def has_adaptformer_keys(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Return True when a checkpoint state dict contains AdaptFormer weights."""
    return any(".adaptformer." in key for key in state_dict.keys())
