#!/bin/bash
# Download DINOv3 ViT-B/16 weights to torch hub cache
DEST=/home/cvpr_ug_5/.cache/torch/hub/checkpoints
mkdir -p "$DEST"
URL=https://huggingface.co/facebook/dinov3-base/resolve/main/model.safetensors
OUTFILE="$DEST/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
wget -O "$OUTFILE" "$URL"
echo "Downloaded to: $OUTFILE"
ls -lh "$OUTFILE"
