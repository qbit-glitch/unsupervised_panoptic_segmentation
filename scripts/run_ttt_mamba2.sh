#!/bin/bash
# Launch Noise-Robust TTT-Mamba2 DDP training with EMA Teacher on 2x GTX 1080 Ti
# Effective batch size: 16 per GPU × 2 GPUs = 32 (FP16/AMP)
# Loss: SCE + IM (0.01) + EMA Teacher-Student consistency (1.0)
# EMA: alpha=0.999, temp=0.5, conf=0.7, warmup=2 epochs, feat_drop=0.5

set -e

export PATH=/home/santosh/anaconda3/envs/cups/bin:$PATH
export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/media/santosh/Kuldeep/panoptic_segmentation:$PYTHONPATH

cd /media/santosh/Kuldeep/panoptic_segmentation

CITYSCAPES_ROOT="/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes"
OUTPUT_DIR="checkpoints/ttt_mamba2_ema_featdrop"
LOG_FILE="training_ttt_mamba2_ema_featdrop.log"

echo "Starting EMA Teacher TTT-Mamba2 DDP training at $(date)" | tee "$LOG_FILE"
echo "Cityscapes: $CITYSCAPES_ROOT" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "GPUs: 2x GTX 1080 Ti, batch_size=16/GPU, effective=32, FP16/AMP" | tee -a "$LOG_FILE"

torchrun --nproc_per_node=2 --master_port=29500 \
    mbps_pytorch/train_ttt_mamba2.py \
    --cityscapes_root "$CITYSCAPES_ROOT" \
    --semantic_subdir pseudo_semantic_overclustered_k300_nocrf \
    --feature_subdir dinov2_features \
    --depth_subdir depth_spidepth \
    --num_classes 19 \
    --bridge_dim 192 \
    --num_blocks 3 \
    --scan_mode cross_scan \
    --layer_type mamba2 \
    --context_dropout 0.5 \
    --d_state 64 \
    --num_epochs 20 \
    --batch_size 16 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --num_workers 4 \
    --eval_every 2 \
    --sce_alpha 1.0 \
    --sce_beta 1.0 \
    --im_weight 0.01 \
    --consist_weight 1.0 \
    --mask_threshold 6 \
    --ema_alpha 0.999 \
    --teacher_temp 0.5 \
    --teacher_conf 0.7 \
    --ema_warmup_epochs 2 \
    --feat_drop_rate 0.5 \
    --ttt_lr 0.01 \
    --ttt_steps 3 \
    --output_dir "$OUTPUT_DIR" \
    --device cuda \
    2>&1 | tee -a "$LOG_FILE"

echo "Training completed at $(date)" | tee -a "$LOG_FILE"
