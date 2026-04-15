#!/usr/bin/env python3
"""Convert DINOv3 weights from HuggingFace key format to official repo format.

HF keys use: embeddings.*, layer.{i}.attention.{q,k,v}_proj.*, layer.{i}.mlp.*
Official keys use: cls_token, blocks.{i}.attn.qkv.*, blocks.{i}.mlp.*

The critical difference: HF has separate Q, K, V projections while official
fuses them into a single QKV linear layer.

Run on A6000: python scripts/convert_dinov3_hf_to_official.py
"""
import sys
import os
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DINOV3_ROOT = os.path.join(PROJECT_ROOT, "refs", "dinov3")
sys.path.insert(0, DINOV3_ROOT)

import torch

HF_PATH = os.path.join(PROJECT_ROOT, "weights", "dinov3_vitb16_official.pth")
OUT_PATH = os.path.join(PROJECT_ROOT, "weights", "dinov3_vitb16_official.pth")
BACKUP_PATH = os.path.join(PROJECT_ROOT, "weights", "dinov3_vitb16_hf_original.pth")

print(f"Loading HF weights from: {HF_PATH}")
hf_sd = torch.load(HF_PATH, map_location="cpu", weights_only=True)
print(f"HF keys: {len(hf_sd)}")

# Print all HF keys for debugging
print("\nAll HF keys:")
for k in sorted(hf_sd.keys()):
    print(f"  {k}: {hf_sd[k].shape}")

# Create model to get target keys
print("\nCreating DINOv3 ViT-B/16 model...")
from dinov3.hub.backbones import dinov3_vitb16
model = dinov3_vitb16(pretrained=False)
model_sd = model.state_dict()

print("\nAll model keys:")
for k in sorted(model_sd.keys()):
    print(f"  {k}: {model_sd[k].shape}")

# Build the converted state dict
new_sd = {}

# --- Simple renames ---
simple_map = {
    "embeddings.cls_token": "cls_token",
    "embeddings.mask_token": "mask_token",
    "embeddings.register_tokens": "register_tokens",
    "embeddings.position_embeddings": "pos_embed",
    "embeddings.patch_embeddings.weight": "patch_embed.proj.weight",
    "embeddings.patch_embeddings.bias": "patch_embed.proj.bias",
    "layernorm.weight": "norm.weight",
    "layernorm.bias": "norm.bias",
}

for hf_key, official_key in simple_map.items():
    if hf_key in hf_sd:
        if official_key in model_sd:
            if hf_sd[hf_key].shape == model_sd[official_key].shape:
                new_sd[official_key] = hf_sd[hf_key]
                print(f"  Mapped: {hf_key} -> {official_key}")
            else:
                print(f"  SHAPE MISMATCH: {hf_key} {hf_sd[hf_key].shape} vs {official_key} {model_sd[official_key].shape}")
        else:
            print(f"  Target key not found in model: {official_key}")
    else:
        print(f"  Source key not found in HF: {hf_key}")

# --- Per-block mappings ---
num_blocks = 12
for i in range(num_blocks):
    prefix_hf = f"layer.{i}"
    prefix_official = f"blocks.{i}"

    # Direct renames within each block
    block_map = {
        f"{prefix_hf}.attention.o_proj.weight": f"{prefix_official}.attn.proj.weight",
        f"{prefix_hf}.attention.o_proj.bias": f"{prefix_official}.attn.proj.bias",
        f"{prefix_hf}.norm1.weight": f"{prefix_official}.norm1.weight",
        f"{prefix_hf}.norm1.bias": f"{prefix_official}.norm1.bias",
        f"{prefix_hf}.norm2.weight": f"{prefix_official}.norm2.weight",
        f"{prefix_hf}.norm2.bias": f"{prefix_official}.norm2.bias",
        f"{prefix_hf}.mlp.fc1.weight": f"{prefix_official}.mlp.fc1.weight",
        f"{prefix_hf}.mlp.fc1.bias": f"{prefix_official}.mlp.fc1.bias",
        f"{prefix_hf}.mlp.fc2.weight": f"{prefix_official}.mlp.fc2.weight",
        f"{prefix_hf}.mlp.fc2.bias": f"{prefix_official}.mlp.fc2.bias",
    }

    # LayerScale: HF might use layer.{i}.ls1 (scalar or vector) -> blocks.{i}.ls1.gamma
    for ls_name in ["ls1", "ls2"]:
        hf_key = f"{prefix_hf}.{ls_name}"
        official_key = f"{prefix_official}.{ls_name}.gamma"
        if hf_key in hf_sd and official_key in model_sd:
            block_map[hf_key] = official_key

    for hf_key, official_key in block_map.items():
        if hf_key in hf_sd:
            if official_key in model_sd:
                if hf_sd[hf_key].shape == model_sd[official_key].shape:
                    new_sd[official_key] = hf_sd[hf_key]
                else:
                    print(f"  SHAPE MISMATCH block {i}: {hf_key} {hf_sd[hf_key].shape} vs {official_key} {model_sd[official_key].shape}")
            else:
                print(f"  Target key not found: {official_key}")

    # --- QKV fusion ---
    q_w_key = f"{prefix_hf}.attention.q_proj.weight"
    k_w_key = f"{prefix_hf}.attention.k_proj.weight"
    v_w_key = f"{prefix_hf}.attention.v_proj.weight"
    qkv_w_key = f"{prefix_official}.attn.qkv.weight"

    if all(k in hf_sd for k in [q_w_key, k_w_key, v_w_key]) and qkv_w_key in model_sd:
        q_w = hf_sd[q_w_key]
        k_w = hf_sd[k_w_key]
        v_w = hf_sd[v_w_key]
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
        if qkv_w.shape == model_sd[qkv_w_key].shape:
            new_sd[qkv_w_key] = qkv_w
            print(f"  Fused QKV weight block {i}: {q_w.shape} x3 -> {qkv_w.shape}")
        else:
            print(f"  QKV SHAPE MISMATCH block {i}: {qkv_w.shape} vs {model_sd[qkv_w_key].shape}")

    # QKV bias fusion
    q_b_key = f"{prefix_hf}.attention.q_proj.bias"
    k_b_key = f"{prefix_hf}.attention.k_proj.bias"
    v_b_key = f"{prefix_hf}.attention.v_proj.bias"
    qkv_b_key = f"{prefix_official}.attn.qkv.bias"

    if qkv_b_key in model_sd:
        dim = model_sd[qkv_b_key].shape[0] // 3
        q_b = hf_sd.get(q_b_key, torch.zeros(dim))
        k_b = hf_sd.get(k_b_key, torch.zeros(dim))
        v_b = hf_sd.get(v_b_key, torch.zeros(dim))
        qkv_b = torch.cat([q_b, k_b, v_b], dim=0)
        if qkv_b.shape == model_sd[qkv_b_key].shape:
            new_sd[qkv_b_key] = qkv_b
            has_q = q_b_key in hf_sd
            has_k = k_b_key in hf_sd
            has_v = v_b_key in hf_sd
            print(f"  Fused QKV bias block {i}: q={has_q} k={has_k} v={has_v}")

    # QKV bias_mask: indicates which biases are active
    # DINOv3 typically has bias on Q and V but not K
    bm_key = f"{prefix_official}.attn.qkv.bias_mask"
    if bm_key in model_sd:
        dim = model_sd[bm_key].shape[0] // 3
        has_q = q_b_key in hf_sd
        has_k = k_b_key in hf_sd
        has_v = v_b_key in hf_sd
        q_mask = torch.ones(dim) if has_q else torch.zeros(dim)
        k_mask = torch.ones(dim) if has_k else torch.zeros(dim)
        v_mask = torch.ones(dim) if has_v else torch.zeros(dim)
        bias_mask = torch.cat([q_mask, k_mask, v_mask], dim=0)
        new_sd[bm_key] = bias_mask
        print(f"  Created bias_mask block {i}: q={has_q} k={has_k} v={has_v}")

# --- Verify coverage ---
print(f"\n--- Conversion summary ---")
print(f"Converted keys: {len(new_sd)} / {len(model_sd)} model keys")

missing = set(model_sd.keys()) - set(new_sd.keys())
if missing:
    print(f"\nMissing {len(missing)} keys (will use random init):")
    for k in sorted(missing):
        print(f"  {k}: {model_sd[k].shape}")

extra_hf = set(hf_sd.keys()) - set()  # keys we didn't use
used_hf_keys = set()
for hf_key in simple_map:
    if hf_key in hf_sd:
        used_hf_keys.add(hf_key)
for i in range(num_blocks):
    for suffix in ["attention.q_proj.weight", "attention.q_proj.bias",
                    "attention.k_proj.weight", "attention.k_proj.bias",
                    "attention.v_proj.weight", "attention.v_proj.bias",
                    "attention.o_proj.weight", "attention.o_proj.bias",
                    "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
                    "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias",
                    "ls1", "ls2"]:
        k = f"layer.{i}.{suffix}"
        if k in hf_sd:
            used_hf_keys.add(k)

unused_hf = set(hf_sd.keys()) - used_hf_keys
if unused_hf:
    print(f"\nUnused HF keys ({len(unused_hf)}):")
    for k in sorted(unused_hf):
        print(f"  {k}: {hf_sd[k].shape}")

# Verify shapes match
shape_ok = True
for k in new_sd:
    if new_sd[k].shape != model_sd[k].shape:
        print(f"  SHAPE ERROR: {k}: converted {new_sd[k].shape} vs model {model_sd[k].shape}")
        shape_ok = False

if not shape_ok:
    print("\nERROR: Shape mismatches found! Not saving.")
    sys.exit(1)

# Test loading
print("\n--- Testing load with strict=True ---")
result = model.load_state_dict(new_sd, strict=False)
if result.missing_keys:
    print(f"Missing keys after load: {len(result.missing_keys)}")
    for k in result.missing_keys[:5]:
        print(f"  {k}")
if result.unexpected_keys:
    print(f"Unexpected keys: {len(result.unexpected_keys)}")

match_pct = (len(model_sd) - len(result.missing_keys)) / len(model_sd) * 100
print(f"\nMatch rate: {match_pct:.1f}%")

if match_pct < 90:
    print(f"\nERROR: Only {match_pct:.1f}% matched. Conversion incomplete.")
    print("Saving partial conversion for inspection but DO NOT USE for training.")
    debug_path = os.path.join(PROJECT_ROOT, "weights", "dinov3_vitb16_debug.pth")
    torch.save(new_sd, debug_path)
    print(f"Debug weights saved to: {debug_path}")
    sys.exit(1)

# Backup original and save converted
print(f"\nBacking up original HF weights to: {BACKUP_PATH}")
os.rename(HF_PATH, BACKUP_PATH)

print(f"Saving converted weights to: {OUT_PATH}")
torch.save(new_sd, OUT_PATH)
print(f"Saved {len(new_sd)} keys")

# Final verification
print("\n--- Final verification ---")
sd_verify = torch.load(OUT_PATH, map_location="cpu", weights_only=True)
model2 = dinov3_vitb16(pretrained=False)
result2 = model2.load_state_dict(sd_verify, strict=False)
print(f"Missing keys: {len(result2.missing_keys)}")
print(f"Unexpected keys: {len(result2.unexpected_keys)}")
if len(result2.missing_keys) == 0:
    print("\n*** SUCCESS: All keys matched! Weights are now in official format. ***")
    print("Re-run training with these weights.")
else:
    print(f"\n*** WARNING: {len(result2.missing_keys)} keys still missing ***")
