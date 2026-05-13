#!/bin/bash
# UNet Phase 2 Ablation Study — Architecture Variants
# Baseline: PQ=27.73 (2-stage conv, ep8)
# 4 ablations: P2-A (3-stage conv), P2-B (2-stage attn), P2-C (3-stage attn), P2-D (2-stage conv +extra)
# Run on remote: santosh@100.93.203.100, 2x GTX 1080 Ti
# Schedule: GPU-0: P2-A then P2-C | GPU-1: P2-B then P2-D

PYTHON="python"
ROOT="/media/santosh/Kuldeep/panoptic_segmentation"
CS="/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes"
export PYTHONPATH="$ROOT"
export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:$LD_LIBRARY_PATH"

# Common base args (matching UNet baseline config)
BASE_ARGS="--cityscapes_root $CS \
  --semantic_subdir pseudo_semantic_mapped_k80 \
  --num_classes 19 \
  --model_type unet \
  --bridge_dim 192 \
  --num_bottleneck_blocks 2 \
  --skip_dim 32 \
  --coupling_strength 0.1 \
  --num_epochs 20 \
  --batch_size 4 \
  --lr 5e-5 \
  --weight_decay 1e-4 \
  --num_workers 4 \
  --eval_interval 2 \
  --lambda_distill 1.0 \
  --lambda_distill_min 0.85 \
  --lambda_align 0.25 \
  --lambda_proto 0.025 \
  --lambda_ent 0.025 \
  --label_smoothing 0.1"

RUN=$1
GPU=${2:-0}

case "$RUN" in
  P2-A|3stage_conv)
    echo "=== P2-A: 3-Stage Conv (256x512) on GPU $GPU ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --batch_size 2 \
      --num_decoder_stages 3 \
      --block_type conv \
      --output_dir checkpoints/unet_p2a_3stage_conv
    ;;
  P2-B|2stage_attn)
    echo "=== P2-B: 2-Stage Attention (128x256) on GPU $GPU ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --num_decoder_stages 2 \
      --block_type attention \
      --window_size 8 \
      --num_heads 4 \
      --output_dir checkpoints/unet_p2b_2stage_attn
    ;;
  P2-C|3stage_attn)
    echo "=== P2-C: 3-Stage Attention (256x512) on GPU $GPU ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --batch_size 2 \
      --num_decoder_stages 3 \
      --block_type attention \
      --window_size 8 \
      --num_heads 4 \
      --output_dir checkpoints/unet_p2c_3stage_attn
    ;;
  P2-D|2stage_extra)
    echo "=== P2-D: 2-Stage Conv + Extra Block (128x256) on GPU $GPU ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --num_decoder_stages 2 \
      --block_type conv \
      --num_final_blocks 1 \
      --output_dir checkpoints/unet_p2d_2stage_extra
    ;;
  all)
    echo "=== Phase 2: GPU-0 [P2-A -> P2-C] | GPU-1 [P2-B -> P2-D] ==="
    # GPU 0: P2-A then P2-C (both 256x512, batch=2)
    (bash "$0" P2-A 0 && bash "$0" P2-C 0) > "$ROOT/logs/gpu0_p2a_p2c.log" 2>&1 &
    PID_GPU0=$!
    # GPU 1: P2-B then P2-D (both 128x256, batch=4)
    (bash "$0" P2-B 1 && bash "$0" P2-D 1) > "$ROOT/logs/gpu1_p2b_p2d.log" 2>&1 &
    PID_GPU1=$!
    echo "GPU-0 chain PID=$PID_GPU0 | GPU-1 chain PID=$PID_GPU1"
    echo "Logs: logs/gpu0_p2a_p2c.log, logs/gpu1_p2b_p2d.log"
    wait $PID_GPU0
    echo "GPU-0 chain complete (P2-A + P2-C)"
    wait $PID_GPU1
    echo "GPU-1 chain complete (P2-B + P2-D)"
    echo "=== All Phase 2 ablations complete ==="
    ;;
  *)
    echo "Usage: $0 {P2-A|P2-B|P2-C|P2-D|3stage_conv|2stage_attn|3stage_attn|2stage_extra|all} [gpu_id]"
    exit 1
    ;;
esac
