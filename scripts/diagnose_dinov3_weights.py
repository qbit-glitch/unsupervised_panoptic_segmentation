#!/usr/bin/env python3
"""Diagnose DINOv3 weight loading: check if safetensors keys match model keys.

Run on A6000: python scripts/diagnose_dinov3_weights.py
"""
import sys
import os

# Add dinov3 repo to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DINOV3_ROOT = os.path.join(PROJECT_ROOT, "refs", "dinov3")
sys.path.insert(0, DINOV3_ROOT)

import torch

WEIGHT_PATH = os.path.join(PROJECT_ROOT, "weights", "dinov3_vitb16_official.pth")

print(f"Project root: {PROJECT_ROOT}")
print(f"Weight file:  {WEIGHT_PATH}")
print(f"File exists:  {os.path.exists(WEIGHT_PATH)}")

if not os.path.exists(WEIGHT_PATH):
    print("ERROR: Weight file not found!")
    sys.exit(1)

file_size = os.path.getsize(WEIGHT_PATH) / 1e6
print(f"File size:    {file_size:.1f} MB")

if file_size < 10:
    print("ERROR: File too small — likely corrupt or empty!")
    sys.exit(1)

# Load saved weights
print("\n--- Loading saved weights ---")
sd_saved = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=True)
saved_keys = set(sd_saved.keys())
print(f"Saved keys: {len(saved_keys)}")
for k in sorted(saved_keys)[:10]:
    print(f"  {k}: {sd_saved[k].shape}")
print(f"  ... ({len(saved_keys)} total)")

# Create model and get expected keys
print("\n--- Creating DINOv3 ViT-B/16 model (no weights) ---")
from dinov3.hub.backbones import dinov3_vitb16
model = dinov3_vitb16(pretrained=False)
model_keys = set(model.state_dict().keys())
print(f"Model keys: {len(model_keys)}")
for k in sorted(model_keys)[:10]:
    print(f"  {k}: {model.state_dict()[k].shape}")
print(f"  ... ({len(model_keys)} total)")

# Compare
matched = saved_keys & model_keys
only_in_saved = saved_keys - model_keys
only_in_model = model_keys - saved_keys

print(f"\n--- Key comparison ---")
print(f"Matched:        {len(matched)} / {len(model_keys)} ({100*len(matched)/len(model_keys):.1f}%)")
print(f"Only in saved:  {len(only_in_saved)}")
print(f"Only in model:  {len(only_in_model)}")

if only_in_saved:
    print(f"\nKeys in saved file but NOT in model (first 10):")
    for k in sorted(only_in_saved)[:10]:
        print(f"  {k}")

if only_in_model:
    print(f"\nKeys in model but NOT in saved file (first 10):")
    for k in sorted(only_in_model)[:10]:
        print(f"  {k}")

# Verdict
if len(matched) == len(model_keys):
    print("\n*** PASS: All model keys matched. Weights loaded correctly. ***")
elif len(matched) == 0:
    print("\n*** FAIL: ZERO keys matched! Backbone is running with RANDOM weights! ***")
    print("This explains PQ=12%. The safetensors keys don't match the official repo format.")
    print("Fix: Download weights using dinov3_vitb16(pretrained=True) on a node with internet.")
elif len(matched) < len(model_keys) * 0.5:
    print(f"\n*** FAIL: Only {len(matched)}/{len(model_keys)} keys matched. Backbone is partially random! ***")
else:
    print(f"\n*** WARNING: {len(only_in_model)} model keys missing. Some layers may have random weights. ***")

# Try to detect key prefix pattern
if len(matched) == 0 and len(only_in_saved) > 0 and len(only_in_model) > 0:
    print("\n--- Attempting to detect key mapping ---")
    saved_sample = sorted(only_in_saved)[:5]
    model_sample = sorted(only_in_model)[:5]
    print("Saved keys sample:")
    for k in saved_sample:
        print(f"  {k}")
    print("Model keys sample:")
    for k in model_sample:
        print(f"  {k}")

    # Check if adding/removing a prefix fixes it
    for prefix in ["model.", "backbone.", "encoder.", "dinov3.", "vit."]:
        prefixed = {prefix + k for k in saved_keys}
        if len(prefixed & model_keys) > len(matched):
            print(f"\n  Adding prefix '{prefix}' matches {len(prefixed & model_keys)} keys!")
        stripped = {k.replace(prefix, "", 1) for k in saved_keys if k.startswith(prefix)}
        if stripped and len(stripped & model_keys) > len(matched):
            print(f"\n  Removing prefix '{prefix}' matches {len(stripped & model_keys)} keys!")
