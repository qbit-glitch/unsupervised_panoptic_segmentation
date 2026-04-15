#!/usr/bin/env python3
"""Convert DINOv3 weights from HuggingFace key format to official repo format.

Key mapping differences discovered:
  HF                                    -> Official
  embeddings.cls_token                  -> cls_token
  embeddings.mask_token [1,1,768]       -> mask_token [1,768] (squeeze)
  embeddings.register_tokens            -> storage_tokens
  embeddings.patch_embeddings.*         -> patch_embed.proj.*
  norm.weight/bias                      -> norm.weight/bias (same)
  layer.{i}.attention.q/k/v_proj.*      -> blocks.{i}.attn.qkv.* (fuse)
  layer.{i}.attention.o_proj.*          -> blocks.{i}.attn.proj.*
  layer.{i}.mlp.up_proj.*              -> blocks.{i}.mlp.fc1.*
  layer.{i}.mlp.down_proj.*            -> blocks.{i}.mlp.fc2.*
  layer.{i}.layer_scale1.lambda1        -> blocks.{i}.ls1.gamma
  layer.{i}.layer_scale2.lambda1        -> blocks.{i}.ls2.gamma
  layer.{i}.norm1/2.*                   -> blocks.{i}.norm1/2.*
  (no equivalent)                       -> rope_embed.periods (use default init)

Run on A6000: python scripts/convert_dinov3_hf_to_official.py
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DINOV3_ROOT = os.path.join(PROJECT_ROOT, "refs", "dinov3")
sys.path.insert(0, DINOV3_ROOT)

import torch

HF_PATH = os.path.join(PROJECT_ROOT, "weights", "dinov3_vitb16_official.pth")
OUT_PATH = os.path.join(PROJECT_ROOT, "weights", "dinov3_vitb16_official.pth")
BACKUP_PATH = os.path.join(PROJECT_ROOT, "weights", "dinov3_vitb16_hf_original.pth")

# Check if backup already exists (script was run before partially)
if os.path.exists(BACKUP_PATH) and not os.path.exists(HF_PATH):
    HF_PATH = BACKUP_PATH
elif os.path.exists(BACKUP_PATH):
    # Previous partial run — load from backup (the original HF weights)
    HF_PATH = BACKUP_PATH

print(f"Loading HF weights from: {HF_PATH}")
hf_sd = torch.load(HF_PATH, map_location="cpu", weights_only=True)
print(f"HF keys: {len(hf_sd)}")

# Verify these are HF-format keys (not already converted)
if "blocks.0.attn.qkv.weight" in hf_sd:
    print("Weights are already in official format! Nothing to do.")
    sys.exit(0)

# Create model to get target keys
print("Creating DINOv3 ViT-B/16 model...")
from dinov3.hub.backbones import dinov3_vitb16
model = dinov3_vitb16(pretrained=False)
model_sd = model.state_dict()
print(f"Model keys: {len(model_sd)}")

new_sd = {}
num_blocks = 12

# --- Global keys ---

# cls_token
if "embeddings.cls_token" in hf_sd:
    new_sd["cls_token"] = hf_sd["embeddings.cls_token"]
    print("  cls_token: mapped")

# mask_token: HF [1,1,768] -> official [1,768]
if "embeddings.mask_token" in hf_sd:
    mt = hf_sd["embeddings.mask_token"]
    if mt.dim() == 3 and mt.shape[0] == 1 and mt.shape[1] == 1:
        mt = mt.squeeze(1)
    new_sd["mask_token"] = mt
    print(f"  mask_token: {hf_sd['embeddings.mask_token'].shape} -> {mt.shape}")

# register_tokens -> storage_tokens
if "embeddings.register_tokens" in hf_sd:
    new_sd["storage_tokens"] = hf_sd["embeddings.register_tokens"]
    print("  register_tokens -> storage_tokens: mapped")

# patch_embed
for suffix in ["weight", "bias"]:
    hf_k = f"embeddings.patch_embeddings.{suffix}"
    off_k = f"patch_embed.proj.{suffix}"
    if hf_k in hf_sd:
        new_sd[off_k] = hf_sd[hf_k]
        print(f"  {hf_k} -> {off_k}: mapped")

# Final norm (same key names in both formats)
for suffix in ["weight", "bias"]:
    hf_k = f"norm.{suffix}"
    if hf_k in hf_sd:
        new_sd[hf_k] = hf_sd[hf_k]
        print(f"  norm.{suffix}: direct copy")

# rope_embed.periods: no HF equivalent, use model default
if "rope_embed.periods" in model_sd:
    new_sd["rope_embed.periods"] = model_sd["rope_embed.periods"]
    print("  rope_embed.periods: using model default init")

# --- Per-block keys ---
for i in range(num_blocks):
    hf_p = f"layer.{i}"
    off_p = f"blocks.{i}"

    # Attention output projection
    for suffix in ["weight", "bias"]:
        hf_k = f"{hf_p}.attention.o_proj.{suffix}"
        off_k = f"{off_p}.attn.proj.{suffix}"
        if hf_k in hf_sd:
            new_sd[off_k] = hf_sd[hf_k]

    # Norms
    for norm_name in ["norm1", "norm2"]:
        for suffix in ["weight", "bias"]:
            hf_k = f"{hf_p}.{norm_name}.{suffix}"
            off_k = f"{off_p}.{norm_name}.{suffix}"
            if hf_k in hf_sd:
                new_sd[off_k] = hf_sd[hf_k]

    # MLP: up_proj -> fc1, down_proj -> fc2
    mlp_map = {
        f"{hf_p}.mlp.up_proj.weight": f"{off_p}.mlp.fc1.weight",
        f"{hf_p}.mlp.up_proj.bias": f"{off_p}.mlp.fc1.bias",
        f"{hf_p}.mlp.down_proj.weight": f"{off_p}.mlp.fc2.weight",
        f"{hf_p}.mlp.down_proj.bias": f"{off_p}.mlp.fc2.bias",
    }
    for hf_k, off_k in mlp_map.items():
        if hf_k in hf_sd:
            new_sd[off_k] = hf_sd[hf_k]

    # LayerScale: layer_scale1.lambda1 -> ls1.gamma
    ls_map = {
        f"{hf_p}.layer_scale1.lambda1": f"{off_p}.ls1.gamma",
        f"{hf_p}.layer_scale2.lambda1": f"{off_p}.ls2.gamma",
    }
    for hf_k, off_k in ls_map.items():
        if hf_k in hf_sd:
            new_sd[off_k] = hf_sd[hf_k]

    # QKV fusion: separate q/k/v -> fused qkv
    q_w = hf_sd.get(f"{hf_p}.attention.q_proj.weight")
    k_w = hf_sd.get(f"{hf_p}.attention.k_proj.weight")
    v_w = hf_sd.get(f"{hf_p}.attention.v_proj.weight")
    if q_w is not None and k_w is not None and v_w is not None:
        new_sd[f"{off_p}.attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

    # QKV bias fusion
    dim = 768
    q_b = hf_sd.get(f"{hf_p}.attention.q_proj.bias", torch.zeros(dim))
    k_b = hf_sd.get(f"{hf_p}.attention.k_proj.bias", torch.zeros(dim))
    v_b = hf_sd.get(f"{hf_p}.attention.v_proj.bias", torch.zeros(dim))
    new_sd[f"{off_p}.attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

    # QKV bias_mask
    has_q_b = f"{hf_p}.attention.q_proj.bias" in hf_sd
    has_k_b = f"{hf_p}.attention.k_proj.bias" in hf_sd
    has_v_b = f"{hf_p}.attention.v_proj.bias" in hf_sd
    q_mask = torch.ones(dim) if has_q_b else torch.zeros(dim)
    k_mask = torch.ones(dim) if has_k_b else torch.zeros(dim)
    v_mask = torch.ones(dim) if has_v_b else torch.zeros(dim)
    new_sd[f"{off_p}.attn.qkv.bias_mask"] = torch.cat([q_mask, k_mask, v_mask], dim=0)

    print(f"  Block {i}: attn(qkv fused, q_b={has_q_b} k_b={has_k_b} v_b={has_v_b}), "
          f"mlp(up->fc1, down->fc2), ls(lambda1->gamma), norms")

# --- Verify ---
print(f"\n--- Conversion summary ---")
print(f"Converted: {len(new_sd)} / {len(model_sd)} model keys")

missing = set(model_sd.keys()) - set(new_sd.keys())
if missing:
    print(f"\nStill missing {len(missing)} keys:")
    for k in sorted(missing):
        print(f"  {k}: {model_sd[k].shape}")

# Shape verification
shape_errors = []
for k in new_sd:
    if k in model_sd and new_sd[k].shape != model_sd[k].shape:
        shape_errors.append(f"  {k}: got {new_sd[k].shape}, expected {model_sd[k].shape}")

if shape_errors:
    print(f"\nSHAPE ERRORS ({len(shape_errors)}):")
    for e in shape_errors:
        print(e)
    print("\nAborting — shape mismatches found.")
    sys.exit(1)

# Test load
print("\n--- Testing load ---")
result = model.load_state_dict(new_sd, strict=False)
n_loaded = len(model_sd) - len(result.missing_keys)
pct = n_loaded / len(model_sd) * 100
print(f"Loaded: {n_loaded}/{len(model_sd)} ({pct:.1f}%)")

if result.missing_keys:
    print(f"Missing ({len(result.missing_keys)}):")
    for k in result.missing_keys:
        print(f"  {k}")

if pct < 95:
    print(f"\nERROR: Only {pct:.1f}% loaded. Aborting.")
    sys.exit(1)

# Save
if not os.path.exists(BACKUP_PATH):
    # Only backup if we haven't already (from a previous partial run)
    orig_path = os.path.join(PROJECT_ROOT, "weights", "dinov3_vitb16_official.pth")
    if os.path.exists(orig_path) and orig_path != BACKUP_PATH:
        print(f"\nBacking up HF weights to: {BACKUP_PATH}")
        os.rename(orig_path, BACKUP_PATH)

print(f"Saving converted weights to: {OUT_PATH}")
torch.save(new_sd, OUT_PATH)

# Final verification
print("\n--- Final verification ---")
sd_check = torch.load(OUT_PATH, map_location="cpu", weights_only=True)
model2 = dinov3_vitb16(pretrained=False)
r2 = model2.load_state_dict(sd_check, strict=False)
n2 = len(model_sd) - len(r2.missing_keys)
print(f"Final: {n2}/{len(model_sd)} keys loaded ({100*n2/len(model_sd):.1f}%)")
if r2.missing_keys:
    print(f"Still missing: {r2.missing_keys}")
if n2 == len(model_sd):
    print("\n*** SUCCESS: All 188/188 keys matched! Weights correctly converted. ***")
    print("Now re-run training.")
elif n2 >= len(model_sd) - 1:
    print(f"\n*** NEAR-SUCCESS: {n2}/{len(model_sd)} keys. Missing keys use default init. ***")
    print("This should work — re-run training.")
else:
    print(f"\n*** WARNING: {len(r2.missing_keys)} keys still missing. ***")
