"""PyTorch to JAX/Flax weight converter for DINOv3 ViT-B/16.

Converts pretrained DINOv3 weights from HuggingFace format to Flax
parameter dictionaries. Handles register tokens and 768-dim weights.

Usage:
    # From Python:
    from mbps.models.backbone.dinov3_weights_converter import convert_dinov3_weights
    params = convert_dinov3_weights("facebook/dinov3-vitb16-pretrain-lvd1689m")

    # From CLI:
    python -m mbps.models.backbone.dinov3_weights_converter \
        --model_name facebook/dinov3-vitb16-pretrain-lvd1689m \
        --output weights/dinov3_vitb16_flax.npz
"""

from __future__ import annotations

import os
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np
from absl import logging


def convert_dinov3_weights(
    model_name_or_path: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
) -> Dict[str, Any]:
    """Convert DINOv3 ViT-B/16 weights from HuggingFace to Flax.

    Supports loading from:
      1. HuggingFace model name (requires transformers + torch)
      2. Local safetensors/pytorch checkpoint

    Args:
        model_name_or_path: HuggingFace model name or path to checkpoint.

    Returns:
        Flax parameter dictionary compatible with DINOv3ViTB.
    """
    pt_state = _load_hf_state_dict(model_name_or_path)

    flax_params = {}

    # Patch embedding
    # HF DINOv3: dinov3.embeddings.patch_embeddings.projection.weight (768, 3, 16, 16)
    patch_weight = pt_state["dinov3.embeddings.patch_embeddings.projection.weight"]
    # PyTorch Conv2D: (out, in, kH, kW) -> Flax: (kH, kW, in, out)
    patch_weight = np.transpose(patch_weight, (2, 3, 1, 0))
    flax_params["patch_embed"] = {
        "proj": {
            "kernel": patch_weight,
            "bias": pt_state["dinov3.embeddings.patch_embeddings.projection.bias"],
        }
    }

    # CLS token: (1, 1, 768)
    flax_params["cls_token"] = pt_state["dinov3.embeddings.cls_token"]

    # Position embedding: (1, 1+N_patches, 768)
    flax_params["pos_embed"] = pt_state["dinov3.embeddings.position_embeddings"]

    # Register tokens: (1, 4, 768) — DINOv3 specific
    if "dinov3.embeddings.register_tokens" in pt_state:
        flax_params["register_tokens"] = pt_state["dinov3.embeddings.register_tokens"]
    else:
        logging.warning("No register tokens found in checkpoint, initializing to zeros")
        flax_params["register_tokens"] = np.zeros((1, 4, 768), dtype=np.float32)

    # Transformer blocks
    num_blocks = 12
    for i in range(num_blocks):
        hf_prefix = f"dinov3.encoder.layer.{i}"
        block_params = {
            "norm1": _convert_layernorm(pt_state, f"{hf_prefix}.norm1"),
            "attn": _convert_attention_hf(pt_state, hf_prefix),
            "norm2": _convert_layernorm(pt_state, f"{hf_prefix}.norm2"),
            "mlp": _convert_mlp_hf(pt_state, hf_prefix),
        }
        flax_params[f"blocks_{i}"] = block_params

    # Final LayerNorm
    flax_params["norm"] = _convert_layernorm(pt_state, "dinov3.layernorm")

    # Convert all to jnp arrays
    flax_params = _to_jnp(flax_params)

    logging.info(f"Converted DINOv3 ViT-B/16 weights from {model_name_or_path}")
    return {"params": flax_params}


def _load_hf_state_dict(model_name_or_path: str) -> Dict[str, np.ndarray]:
    """Load state dict from HuggingFace model or local checkpoint.

    Returns numpy arrays (not torch tensors) for framework-agnostic processing.
    """
    try:
        import torch
        from transformers import AutoModel
    except ImportError:
        raise ImportError(
            "Requires transformers and torch. Install with:\n"
            "pip install transformers torch"
        )

    logging.info(f"Loading DINOv3 model: {model_name_or_path}")
    model = AutoModel.from_pretrained(model_name_or_path)
    state_dict = model.state_dict()

    # Convert to numpy
    np_state = {}
    for key, tensor in state_dict.items():
        np_state[key] = tensor.cpu().numpy()

    logging.info(f"Loaded {len(np_state)} weight tensors")
    return np_state


def _convert_attention_hf(
    pt_state: Dict[str, np.ndarray],
    block_prefix: str,
) -> Dict[str, Any]:
    """Convert HF-format attention weights to Flax format.

    HuggingFace DINOv3 stores QKV as separate matrices:
      attention.attention.query.weight/bias
      attention.attention.key.weight/bias
      attention.attention.value.weight/bias
      attention.output.dense.weight/bias

    Flax uses a single QKV dense:
      qkv.kernel (D, 3*D)
      qkv.bias (3*D,)
    """
    prefix = f"{block_prefix}.attention.attention"

    q_w = pt_state[f"{prefix}.query.weight"]  # (D, D)
    k_w = pt_state[f"{prefix}.key.weight"]
    v_w = pt_state[f"{prefix}.value.weight"]
    q_b = pt_state[f"{prefix}.query.bias"]  # (D,)
    k_b = pt_state[f"{prefix}.key.bias"]
    v_b = pt_state[f"{prefix}.value.bias"]

    # Stack QKV: (3*D, D) then transpose for Flax: (D, 3*D)
    qkv_weight = np.concatenate([q_w, k_w, v_w], axis=0)  # (3*D, D)
    qkv_bias = np.concatenate([q_b, k_b, v_b], axis=0)  # (3*D,)

    proj_prefix = f"{block_prefix}.attention.output.dense"

    return {
        "qkv": {
            "kernel": qkv_weight.T,  # (D, 3*D) — Flax convention
            "bias": qkv_bias,
        },
        "proj": {
            "kernel": pt_state[f"{proj_prefix}.weight"].T,  # (D, D)
            "bias": pt_state[f"{proj_prefix}.bias"],
        },
    }


def _convert_mlp_hf(
    pt_state: Dict[str, np.ndarray],
    block_prefix: str,
) -> Dict[str, Any]:
    """Convert HF-format MLP weights to Flax format.

    HuggingFace: intermediate.dense (fc1) + output.dense (fc2)
    """
    return {
        "fc1": {
            "kernel": pt_state[f"{block_prefix}.intermediate.dense.weight"].T,
            "bias": pt_state[f"{block_prefix}.intermediate.dense.bias"],
        },
        "fc2": {
            "kernel": pt_state[f"{block_prefix}.output.dense.weight"].T,
            "bias": pt_state[f"{block_prefix}.output.dense.bias"],
        },
    }


def _convert_layernorm(
    pt_state: Dict[str, np.ndarray],
    prefix: str,
) -> Dict[str, np.ndarray]:
    """Convert LayerNorm weights."""
    return {
        "scale": pt_state[f"{prefix}.weight"],
        "bias": pt_state[f"{prefix}.bias"],
    }


def _to_jnp(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert numpy arrays to jax arrays."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _to_jnp(v)
        elif isinstance(v, np.ndarray):
            result[k] = jnp.array(v)
        else:
            result[k] = v
    return result


def save_flax_params(
    params: Dict[str, Any],
    output_path: str,
) -> None:
    """Save Flax parameters to a numpy file."""
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
    logging.info(f"Saved Flax params to {output_path} ({len(flat)} arrays)")


def load_flax_params(input_path: str) -> Dict[str, Any]:
    """Load Flax parameters from a numpy file."""
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert DINOv3 ViT-B/16 weights to Flax format"
    )
    parser.add_argument(
        "--model_name", type=str,
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output", type=str,
        default="weights/dinov3_vitb16_flax.npz",
        help="Output .npz file path"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    params = convert_dinov3_weights(args.model_name)
    save_flax_params(params, args.output)
    print(f"Done! Saved to {args.output}")
