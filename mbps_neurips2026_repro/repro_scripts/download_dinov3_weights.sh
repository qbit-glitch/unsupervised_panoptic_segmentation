#!/bin/bash
# Download DINOv3 ViT-B/16 weights via HuggingFace Hub and convert to PyTorch format.
# Requires: hf auth login (DINOv3 is a gated model)
# Saves to: weights/dinov3_vitb16_official.pth (loaded by backbone_dinov3_vit.py)
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$PROJ_ROOT/weights"
mkdir -p "$DEST"
OUTFILE="$DEST/dinov3_vitb16_official.pth"

if [ -f "$OUTFILE" ] && [ -s "$OUTFILE" ]; then
    echo "Weights already exist: $OUTFILE ($(du -h "$OUTFILE" | cut -f1))"
    exit 0
fi

python -c "
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch

print('Downloading DINOv3 ViT-B/16 from HuggingFace...')
sf_path = hf_hub_download('facebook/dinov3-vitb16-pretrain-lvd1689m', 'model.safetensors')
print('Converting safetensors -> PyTorch state_dict...')
sd = load_file(sf_path)
torch.save(sd, '$OUTFILE')
print(f'Saved {len(sd)} keys to $OUTFILE')
" && echo "Done: $(ls -lh "$OUTFILE")"
