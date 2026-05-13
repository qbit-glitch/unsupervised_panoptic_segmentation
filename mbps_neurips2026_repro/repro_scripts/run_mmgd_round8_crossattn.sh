#!/bin/bash
# MMGD-Cut Round 8: Cross-Attention Fusion
# Trains a lightweight cross-attention module to learn per-token fusion weights
# between DINOv3 and SSD-1B features (replacing uniform weighted concat).
# No GPU training required — runs locally on MPS in ~10-30 minutes.
# Usage: nohup bash scripts/run_mmgd_round8_crossattn.sh > /tmp/mmgd_round8.log 2>&1 &

set -e

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
COCO="/Users/qbit-glitch/Desktop/datasets/coco"
DEVICE="mps"
CKPT_DIR="checkpoints/crossattn_fusion"

# Config key of the best baseline NCut segments (used as pseudo-labels)
# This must match the segment cache directory created by Round 3
SEG_KEY="mmgd_K54_a5.5_reg0.7_dinov3+ssd1b_nodiff_r32"

echo "=== MMGD-Cut Round 8: Cross-Attention Fusion ==="
echo "Start: $(date)"
echo "Using pseudo-labels from: $SEG_KEY"

# ── Step 1: Train CrossAttentionFusion ──
echo ""
echo "=== Step 1: Train CrossAttentionFusion (20 epochs on 500 val images) ==="
echo "Start: $(date)"
$PYTHON -u mbps_pytorch/train_cross_attention_fusion.py \
    --coco_root $COCO \
    --seg_config_key "$SEG_KEY" \
    --output_dir $CKPT_DIR \
    --device $DEVICE \
    --epochs 20 \
    --lr 1e-4 \
    --temperature 0.07 \
    --min_seg_size 4 \
    --d_dino 1024 \
    --d_ssd 1280 \
    --d_proj 256 \
    --n_heads 4 \
    --d_out 512 \
    --log_interval 50
echo "Step 1 done: $(date)"

# ── Step 2: Evaluate cross-attention fusion via mmgd_cut.py ──
# Note: mmgd_cut.py currently doesn't have --fusion_ckpt support.
# After training, use FusionWrapper manually or extend mmgd_cut.py.
# For now: the training output + alpha statistics indicate fusion quality.

echo ""
echo "=== Step 2: Quick evaluation instructions ==="
echo "After training, load the checkpoint in a notebook or script:"
echo "  from cross_attention_fusion import CrossAttentionFusion, FusionWrapper"
echo "  ckpt = torch.load('$CKPT_DIR/best.pth')"
echo "  model = CrossAttentionFusion(**ckpt['config'])"
echo "  model.load_state_dict(ckpt['model_state'])"
echo ""
echo "  Then replace fuse_features() in mmgd_cut.py with FusionWrapper.fuse()"
echo "  and re-run the pipeline evaluation."
echo ""

# ── Step 2b: Alternate — evaluate with higher-epoch training ──
echo "=== Step 2b: Optional longer training (50 epochs) ==="
echo "To train longer, run:"
echo "  $PYTHON -u mbps_pytorch/train_cross_attention_fusion.py \\"
echo "      --coco_root $COCO --seg_config_key '$SEG_KEY' \\"
echo "      --output_dir ${CKPT_DIR}_50ep --epochs 50 --device $DEVICE"

echo ""
echo "=== Round 8 complete ==="
echo "End: $(date)"
