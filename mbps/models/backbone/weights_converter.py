"""PyTorch to JAX/Flax weight converter for DINO ViT-S/8.

Converts pretrained DINO weights from PyTorch state_dict format
to Flax parameter dictionaries, handling reshaping conventions.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np
from absl import logging


def convert_attention_weights(
    pt_state: Dict[str, np.ndarray],
    block_idx: int,
) -> Dict[str, Any]:
    """Convert attention weights from PyTorch to Flax format.

    PyTorch stores QKV as a single (3*D, D) matrix.
    Flax uses the same convention so we just transpose.

    Args:
        pt_state: PyTorch state dict (numpy arrays).
        block_idx: Transformer block index.

    Returns:
        Flax attention parameter dict.
    """
    prefix = f"blocks.{block_idx}.attn"

    attn_params = {
        "qkv": {
            "kernel": pt_state[f"{prefix}.qkv.weight"].T,
            "bias": pt_state[f"{prefix}.qkv.bias"],
        },
        "proj": {
            "kernel": pt_state[f"{prefix}.proj.weight"].T,
            "bias": pt_state[f"{prefix}.proj.bias"],
        },
    }
    return attn_params


def convert_mlp_weights(
    pt_state: Dict[str, np.ndarray],
    block_idx: int,
) -> Dict[str, Any]:
    """Convert MLP weights from PyTorch to Flax format.

    Args:
        pt_state: PyTorch state dict (numpy arrays).
        block_idx: Transformer block index.

    Returns:
        Flax MLP parameter dict.
    """
    prefix = f"blocks.{block_idx}.mlp"

    mlp_params = {
        "fc1": {
            "kernel": pt_state[f"{prefix}.fc1.weight"].T,
            "bias": pt_state[f"{prefix}.fc1.bias"],
        },
        "fc2": {
            "kernel": pt_state[f"{prefix}.fc2.weight"].T,
            "bias": pt_state[f"{prefix}.fc2.bias"],
        },
    }
    return mlp_params


def convert_layernorm_weights(
    pt_state: Dict[str, np.ndarray],
    key: str,
) -> Dict[str, np.ndarray]:
    """Convert LayerNorm weights.

    Args:
        pt_state: PyTorch state dict.
        key: Weight key prefix.

    Returns:
        Flax LayerNorm parameter dict.
    """
    return {
        "scale": pt_state[f"{key}.weight"],
        "bias": pt_state[f"{key}.bias"],
    }


def convert_dino_weights(
    pytorch_checkpoint_path: str,
) -> Dict[str, Any]:
    """Convert full DINO ViT-S/8 checkpoint from PyTorch to Flax.

    Args:
        pytorch_checkpoint_path: Path to the PyTorch checkpoint (.pth file).

    Returns:
        Flax parameter dictionary compatible with DINOViTS8.

    Raises:
        FileNotFoundError: If checkpoint path doesn't exist.
    """
    if not os.path.exists(pytorch_checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {pytorch_checkpoint_path}"
        )

    # Load PyTorch checkpoint
    try:
        import torch

        checkpoint = torch.load(
            pytorch_checkpoint_path, map_location="cpu", weights_only=False
        )
    except ImportError:
        raise ImportError(
            "PyTorch required for weight conversion. "
            "Install with: pip install torch"
        )

    # Handle different checkpoint formats
    if "teacher" in checkpoint:
        pt_state = checkpoint["teacher"]
    elif "state_dict" in checkpoint:
        pt_state = checkpoint["state_dict"]
    else:
        pt_state = checkpoint

    # Remove 'backbone.' prefix if present
    pt_state = {
        k.replace("backbone.", ""): v.numpy() if hasattr(v, "numpy") else v
        for k, v in pt_state.items()
    }

    flax_params = {}

    # Patch embedding
    patch_weight = pt_state["patch_embed.proj.weight"]
    # PyTorch Conv2D: (out_channels, in_channels, kH, kW)
    # Flax Conv: (kH, kW, in_channels, out_channels)
    patch_weight = np.transpose(patch_weight, (2, 3, 1, 0))

    flax_params["patch_embed"] = {
        "proj": {
            "kernel": patch_weight,
            "bias": pt_state["patch_embed.proj.bias"],
        }
    }

    # CLS token
    flax_params["cls_token"] = pt_state["cls_token"]

    # Position embedding
    flax_params["pos_embed"] = pt_state["pos_embed"]

    # Transformer blocks
    num_blocks = 12
    for i in range(num_blocks):
        block_params = {
            "norm1": convert_layernorm_weights(pt_state, f"blocks.{i}.norm1"),
            "attn": convert_attention_weights(pt_state, i),
            "norm2": convert_layernorm_weights(pt_state, f"blocks.{i}.norm2"),
            "mlp": convert_mlp_weights(pt_state, i),
        }
        flax_params[f"blocks_{i}"] = block_params

    # Final LayerNorm
    flax_params["norm"] = convert_layernorm_weights(pt_state, "norm")

    # Convert all to jnp arrays
    flax_params = _convert_to_jnp(flax_params)

    logging.info(
        f"Converted DINO ViT-S/8 weights from {pytorch_checkpoint_path}"
    )

    return {"params": flax_params}


def _convert_to_jnp(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert numpy arrays to jax arrays.

    Args:
        d: Nested dictionary potentially containing numpy arrays.

    Returns:
        Same structure with jnp.ndarray values.
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _convert_to_jnp(v)
        elif isinstance(v, np.ndarray):
            result[k] = jnp.array(v)
        else:
            result[k] = v
    return result


def save_flax_params(
    params: Dict[str, Any],
    output_path: str,
) -> None:
    """Save Flax parameters to a numpy file.

    Args:
        params: Flax parameter dictionary.
        output_path: Output .npz file path.
    """
    flat = {}

    def _flatten(d: Dict, prefix: str = ""):
        for k, v in d.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, key)
            else:
                flat[key] = np.array(v)

    _flatten(params)
    np.savez(output_path, **flat)
    logging.info(f"Saved Flax params to {output_path}")


def load_flax_params(input_path: str) -> Dict[str, Any]:
    """Load Flax parameters from a numpy file.

    Args:
        input_path: Path to .npz file.

    Returns:
        Nested Flax parameter dictionary.
    """
    data = np.load(input_path)
    result: Dict[str, Any] = {}

    for key in data:
        parts = key.split("/")
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = jnp.array(data[key])

    return result
