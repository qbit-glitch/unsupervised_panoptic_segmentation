#!/bin/bash
# Download CAUSE-TR pretrained checkpoints for Cityscapes with DINOv2 ViT-B/14
# Reference: https://github.com/cactus-blossom-studio/CAUSE

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CAUSE_DIR="$PROJECT_ROOT/refs/cause"

echo "=== Downloading CAUSE-TR Checkpoints ==="

# 1. DINOv2 ViT-B/14 backbone
mkdir -p "$CAUSE_DIR/checkpoint"
if [ ! -f "$CAUSE_DIR/checkpoint/dinov2_vit_base_14.pth" ]; then
    echo "Downloading DINOv2 ViT-B/14 backbone..."
    wget -q --show-progress -O "$CAUSE_DIR/checkpoint/dinov2_vit_base_14.pth" \
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
else
    echo "DINOv2 backbone already exists, skipping."
fi

# 2. CAUSE-TR Cityscapes Seg Head + Cluster (from Google Drive)
CKPT_DIR="$CAUSE_DIR/CAUSE/cityscapes/dinov2_vit_base_14/2048"
MOD_DIR="$CAUSE_DIR/CAUSE/cityscapes/modularity/dinov2_vit_base_14/2048"
mkdir -p "$CKPT_DIR"
mkdir -p "$MOD_DIR"

# Check if gdown is available
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown for Google Drive downloads..."
    pip install gdown
fi

# Seg Head
if [ ! -f "$CKPT_DIR/segment_tr.pth" ]; then
    echo "Downloading CAUSE-TR Seg Head..."
    gdown --folder "https://drive.google.com/drive/folders/1fi_DvMD3CLaZEozEgrGhIh6nH7WFq_Sj" -O /tmp/cause_seg_head
    # Find and copy the segment_tr.pth file
    find /tmp/cause_seg_head -name "segment_tr.pth" -exec cp {} "$CKPT_DIR/" \;
    echo "Seg head saved to $CKPT_DIR/segment_tr.pth"
else
    echo "Seg head already exists, skipping."
fi

# Cluster + Modularity codebook
if [ ! -f "$CKPT_DIR/cluster_tr.pth" ]; then
    echo "Downloading CAUSE-TR Cluster..."
    gdown --folder "https://drive.google.com/drive/folders/1t66yv8_otlAMwy-QQyff-6fiwP58kCvV" -O /tmp/cause_cluster
    # Find and copy files
    find /tmp/cause_cluster -name "cluster_tr.pth" -exec cp {} "$CKPT_DIR/" \;
    find /tmp/cause_cluster -name "modular.npy" -exec cp {} "$MOD_DIR/" \;
    echo "Cluster saved to $CKPT_DIR/cluster_tr.pth"
    echo "Modularity codebook saved to $MOD_DIR/modular.npy"
else
    echo "Cluster already exists, skipping."
fi

echo ""
echo "=== Checkpoint Status ==="
echo "DINOv2 backbone:   $([ -f "$CAUSE_DIR/checkpoint/dinov2_vit_base_14.pth" ] && echo 'OK' || echo 'MISSING')"
echo "Seg head:          $([ -f "$CKPT_DIR/segment_tr.pth" ] && echo 'OK' || echo 'MISSING')"
echo "Cluster:           $([ -f "$CKPT_DIR/cluster_tr.pth" ] && echo 'OK' || echo 'MISSING')"
echo "Modularity:        $([ -f "$MOD_DIR/modular.npy" ] && echo 'OK' || echo 'MISSING')"
