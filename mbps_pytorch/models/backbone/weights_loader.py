"""PyTorch weight loader for DINO ViT-S/8.

Loads pretrained DINO weights from the official PyTorch checkpoint
into the PyTorch DINOViTS8 model, handling key name mapping between
the original DINO format and our model's state dict.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _remap_checkpoint_keys(
    checkpoint_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Remap keys from the official DINO checkpoint to our model's state dict.

    The official DINO ViT-S/8 checkpoint uses key names like:
        blocks.0.attn.qkv.weight  ->  blocks.0.attn.qkv.weight
        blocks.0.mlp.fc1.weight   ->  blocks.0.mlp.fc1.weight
        blocks.0.norm1.weight     ->  blocks.0.norm1.weight

    Our model uses the same names via nn.ModuleList, so most keys
    map directly. This function handles any prefix stripping.

    Args:
        checkpoint_state: Raw state dict from the DINO checkpoint.

    Returns:
        Remapped state dict compatible with DINOViTS8.
    """
    remapped = {}

    for key, value in checkpoint_state.items():
        # Remove 'backbone.' prefix if present (e.g., from DINO teacher)
        new_key = key.replace("backbone.", "")

        # Map patch_embed.proj -> patch_embed.proj (same)
        # Map cls_token -> cls_token (same)
        # Map pos_embed -> pos_embed (same)
        # Map blocks.X.* -> blocks.X.* (same via ModuleList)
        # Map norm.* -> norm.* (same)

        remapped[new_key] = value

    return remapped


def load_dino_weights(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> nn.Module:
    """Load DINO ViT-S/8 pretrained weights into the model.

    Supports multiple checkpoint formats:
    - Direct state dict
    - Checkpoint with 'teacher' key (DINO self-distillation format)
    - Checkpoint with 'state_dict' key

    Args:
        model: DINOViTS8 model instance.
        checkpoint_path: Path to the PyTorch checkpoint (.pth file).
        strict: Whether to enforce strict key matching. Default False
            to allow missing/extra keys gracefully.

    Returns:
        The model with loaded weights.

    Raises:
        FileNotFoundError: If checkpoint path doesn't exist.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # Load PyTorch checkpoint
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    # Handle different checkpoint formats
    if "teacher" in checkpoint:
        pt_state = checkpoint["teacher"]
    elif "state_dict" in checkpoint:
        pt_state = checkpoint["state_dict"]
    else:
        pt_state = checkpoint

    # Remap key names
    pt_state = _remap_checkpoint_keys(pt_state)

    # Load into model
    missing, unexpected = model.load_state_dict(pt_state, strict=strict)

    if missing:
        logger.warning(f"Missing keys when loading weights: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading weights: {unexpected}")

    logger.info(
        f"Loaded DINO ViT-S/8 weights from {checkpoint_path}"
    )

    return model


def load_dino_from_url(
    model: nn.Module,
    url: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> nn.Module:
    """Load DINO ViT-S/8 weights from a URL (e.g., official Facebook hub).

    Uses torch.hub.load_state_dict_from_url for automatic download and caching.

    Args:
        model: DINOViTS8 model instance.
        url: URL to the checkpoint file. Defaults to the official
            DINO ViT-S/8 checkpoint URL.
        cache_dir: Directory for caching downloads. Defaults to
            torch hub cache.

    Returns:
        The model with loaded weights.
    """
    if url is None:
        url = (
            "https://dl.fbaipublicfiles.com/dino/"
            "dino_deit_small_8_pretrain/"
            "dino_deit_small_8_pretrain.pth"
        )

    checkpoint = torch.hub.load_state_dict_from_url(
        url,
        model_dir=cache_dir,
        map_location="cpu",
    )

    # Remap and load
    pt_state = _remap_checkpoint_keys(checkpoint)
    missing, unexpected = model.load_state_dict(pt_state, strict=False)

    if missing:
        logger.warning(f"Missing keys when loading weights: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading weights: {unexpected}")

    logger.info(f"Loaded DINO ViT-S/8 weights from URL: {url}")

    return model


def save_checkpoint(
    model: nn.Module,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model checkpoint.

    Args:
        model: Model to save.
        output_path: Output .pth file path.
        metadata: Optional metadata dict to include.
    """
    save_dict: Dict[str, Any] = {
        "state_dict": model.state_dict(),
    }
    if metadata is not None:
        save_dict["metadata"] = metadata

    torch.save(save_dict, output_path)
    logger.info(f"Saved checkpoint to {output_path}")


def load_checkpoint(
    model: nn.Module,
    input_path: str,
    strict: bool = False,
) -> nn.Module:
    """Load model checkpoint.

    Args:
        model: Model to load weights into.
        input_path: Path to .pth checkpoint file.
        strict: Whether to enforce strict key matching.

    Returns:
        Model with loaded weights.
    """
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    logger.info(f"Loaded checkpoint from {input_path}")
    return model
