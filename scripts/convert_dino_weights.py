#!/usr/bin/env python3
"""Convert DINO / DINOv2 ViT PyTorch weights to Flax format.

Usage:
    # DINOv2 ViT-B/14 (default)
    python scripts/convert_dino_weights.py \
        --input weights/dinov2_vitb14_pretrain.pth \
        --output weights/dinov2_vitb14_flax.npz \
        --model dinov2_vitb14

    # DINO ViT-S/8 (legacy)
    python scripts/convert_dino_weights.py \
        --input weights/dino_deitsmall8_pretrain.pth \
        --output weights/dino_vits8_flax.npz \
        --model dino_vits8
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# PyTorch is only needed for loading .pth files
try:
    import torch
except ImportError:
    print("ERROR: PyTorch is required for weight conversion.")
    print("Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)


# Model configs: (embed_dim, num_heads, depth, patch_size, has_layerscale)
MODEL_CONFIGS = {
    "dino_vits8":    (384,  6, 12,  8, False),
    "dino_vits16":   (384,  6, 12, 16, False),
    "dino_vitb8":    (768, 12, 12,  8, False),
    "dino_vitb16":   (768, 12, 12, 16, False),
    "dinov2_vits14":  (384,  6, 12, 14, True),
    "dinov2_vitb14":  (768, 12, 12, 14, True),
    "dinov2_vitl14": (1024, 16, 24, 14, True),
    "dinov2_vitg14": (1536, 24, 40, 14, True),
}


def convert_dino_weights(input_path: str, output_path: str, model_name: str = "dinov2_vitb14"):
    """Convert PyTorch DINO/DINOv2 ViT checkpoint to Flax .npz."""
    if model_name not in MODEL_CONFIGS:
        print(f"ERROR: Unknown model '{model_name}'. Choose from: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    embed_dim, num_heads, depth, patch_size, has_ls = MODEL_CONFIGS[model_name]
    head_dim = embed_dim // num_heads
    print(f"Model: {model_name}")
    print(f"  embed_dim={embed_dim}, heads={num_heads}, depth={depth}, "
          f"patch={patch_size}, layerscale={has_ls}")

    print(f"Loading PyTorch checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    # DINOv2 checkpoints may nest state dict under various keys
    if isinstance(checkpoint, dict):
        for key in ("model", "teacher", "state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                print(f"  Extracted state dict from '{key}' key")
                break

    state_dict = checkpoint
    print(f"  State dict keys: {len(state_dict)}")

    # Build Flax params dict
    flax_params = {}

    # 1. Patch embedding conv: PyTorch (out, in, H, W) → Flax (H, W, in, out)
    patch_weight = state_dict["patch_embed.proj.weight"].numpy()
    patch_bias = state_dict["patch_embed.proj.bias"].numpy()
    flax_params["params/patch_embed/proj/kernel"] = np.transpose(patch_weight, (2, 3, 1, 0))
    flax_params["params/patch_embed/proj/bias"] = patch_bias

    # 2. CLS token
    flax_params["params/cls_token"] = state_dict["cls_token"].numpy()

    # 3. Position embedding
    flax_params["params/pos_embed"] = state_dict["pos_embed"].numpy()

    # 4. Register tokens (DINOv2 with registers, if present)
    if "register_tokens" in state_dict:
        flax_params["params/register_tokens"] = state_dict["register_tokens"].numpy()
        print(f"  Found register_tokens: {state_dict['register_tokens'].shape}")

    # 5. Mask token (DINOv2, if present)
    if "mask_token" in state_dict:
        flax_params["params/mask_token"] = state_dict["mask_token"].numpy()

    # 6. Transformer blocks
    for i in range(depth):
        prefix_pt = f"blocks.{i}"
        prefix_flax = f"params/blocks_{i}"

        # Layer norm 1
        flax_params[f"{prefix_flax}/norm1/scale"] = state_dict[f"{prefix_pt}.norm1.weight"].numpy()
        flax_params[f"{prefix_flax}/norm1/bias"] = state_dict[f"{prefix_pt}.norm1.bias"].numpy()

        # Self-attention QKV
        qkv_weight = state_dict[f"{prefix_pt}.attn.qkv.weight"].numpy()
        qkv_bias = state_dict[f"{prefix_pt}.attn.qkv.bias"].numpy()

        # Split into Q, K, V
        q_w, k_w, v_w = np.split(qkv_weight, 3, axis=0)
        q_b, k_b, v_b = np.split(qkv_bias, 3, axis=0)

        # Transpose for Flax Dense: PyTorch (out, in) → Flax (in, out)
        flax_params[f"{prefix_flax}/attn/query/kernel"] = q_w.T
        flax_params[f"{prefix_flax}/attn/query/bias"] = q_b
        flax_params[f"{prefix_flax}/attn/key/kernel"] = k_w.T
        flax_params[f"{prefix_flax}/attn/key/bias"] = k_b
        flax_params[f"{prefix_flax}/attn/value/kernel"] = v_w.T
        flax_params[f"{prefix_flax}/attn/value/bias"] = v_b

        # Attention output projection
        proj_weight = state_dict[f"{prefix_pt}.attn.proj.weight"].numpy()
        proj_bias = state_dict[f"{prefix_pt}.attn.proj.bias"].numpy()
        flax_params[f"{prefix_flax}/attn/out/kernel"] = proj_weight.T
        flax_params[f"{prefix_flax}/attn/out/bias"] = proj_bias

        # LayerScale (DINOv2 only)
        if has_ls:
            ls1_key = f"{prefix_pt}.ls1.gamma"
            ls2_key = f"{prefix_pt}.ls2.gamma"
            if ls1_key in state_dict:
                flax_params[f"{prefix_flax}/ls1/gamma"] = state_dict[ls1_key].numpy()
                flax_params[f"{prefix_flax}/ls2/gamma"] = state_dict[ls2_key].numpy()

        # Layer norm 2
        flax_params[f"{prefix_flax}/norm2/scale"] = state_dict[f"{prefix_pt}.norm2.weight"].numpy()
        flax_params[f"{prefix_flax}/norm2/bias"] = state_dict[f"{prefix_pt}.norm2.bias"].numpy()

        # MLP (PyTorch Linear: weight is (out, in), Flax Dense: kernel is (in, out))
        flax_params[f"{prefix_flax}/mlp/fc1/kernel"] = state_dict[f"{prefix_pt}.mlp.fc1.weight"].numpy().T
        flax_params[f"{prefix_flax}/mlp/fc1/bias"] = state_dict[f"{prefix_pt}.mlp.fc1.bias"].numpy()
        flax_params[f"{prefix_flax}/mlp/fc2/kernel"] = state_dict[f"{prefix_pt}.mlp.fc2.weight"].numpy().T
        flax_params[f"{prefix_flax}/mlp/fc2/bias"] = state_dict[f"{prefix_pt}.mlp.fc2.bias"].numpy()

    # 7. Final LayerNorm
    flax_params["params/norm/scale"] = state_dict["norm.weight"].numpy()
    flax_params["params/norm/bias"] = state_dict["norm.bias"].numpy()

    # Save
    print(f"Saving Flax weights: {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, **flax_params)

    # Verify
    data = np.load(output_path)
    total_params = sum(v.size for v in data.values())
    print(f"Total parameters: {total_params:,}")
    print(f"Keys: {len(data.files)}")

    # Print shape summary
    print("\nKey shapes:")
    for k in sorted(data.files)[:10]:
        print(f"  {k}: {data[k].shape}")
    if len(data.files) > 10:
        print(f"  ... and {len(data.files) - 10} more")

    print(f"\nConversion complete: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Convert DINO/DINOv2 weights to Flax")
    parser.add_argument("--input", type=str, required=True, help="Path to PyTorch .pth file")
    parser.add_argument("--output", type=str, required=True, help="Output path for Flax .npz file")
    parser.add_argument("--model", type=str, default="dinov2_vitb14",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model architecture (default: dinov2_vitb14)")
    args = parser.parse_args()

    convert_dino_weights(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
