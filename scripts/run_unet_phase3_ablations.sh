#!/bin/bash
# UNet Phase 3 Ablation Study — Gradient Accumulation + Extended Decoder
# Baseline: PQ=27.73 (2-stage conv, ep8), P2-B: PQ=28.00 (2-stage attn, ep8)
# 6 ablations: P3-A through P3-F
# Run on remote: santosh@100.93.203.100, 2x GTX 1080 Ti
# Schedule: Priority order P3-F -> P3-B -> P3-A -> P3-D -> P3-C -> P3-E

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
  P3-A|3stage_conv_acc2)
    echo "=== P3-A: 3-Stage Conv, accum=2, eff_batch=4 (256x512) on GPU $GPU ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --batch_size 2 \
      --gradient_accumulation_steps 2 \
      --num_decoder_stages 3 \
      --block_type conv \
      --output_dir checkpoints/unet_p3a_3stage_conv_acc2
    ;;
  P3-B|3stage_attn_acc2)
    echo "=== P3-B: 3-Stage Attention, accum=2, eff_batch=4 (256x512) on GPU $GPU ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --batch_size 2 \
      --gradient_accumulation_steps 2 \
      --num_decoder_stages 3 \
      --block_type attention \
      --window_size 8 \
      --num_heads 4 \
      --output_dir checkpoints/unet_p3b_3stage_attn_acc2
    ;;
  P3-C|3stage_conv_acc4)
    echo "=== P3-C: 3-Stage Conv, accum=4, eff_batch=8 (256x512) on GPU $GPU ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --batch_size 2 \
      --gradient_accumulation_steps 4 \
      --num_decoder_stages 3 \
      --block_type conv \
      --output_dir checkpoints/unet_p3c_3stage_conv_acc4
    ;;
  P3-D|3stage_attn_acc4)
    echo "=== P3-D: 3-Stage Attention, accum=4, eff_batch=8 (256x512) on GPU $GPU ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --batch_size 2 \
      --gradient_accumulation_steps 4 \
      --num_decoder_stages 3 \
      --block_type attention \
      --window_size 8 \
      --num_heads 4 \
      --output_dir checkpoints/unet_p3d_3stage_attn_acc4
    ;;
  P3-E|4stage_attn_acc4)
    echo "=== P3-E: 4-Stage Attention, accum=4, eff_batch=4 (512x1024) on GPU $GPU ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --batch_size 1 \
      --gradient_accumulation_steps 4 \
      --num_decoder_stages 4 \
      --block_type attention \
      --window_size 8 \
      --num_heads 4 \
      --output_dir checkpoints/unet_p3e_4stage_attn_acc4
    ;;
  P3-F|attn_focal)
    echo "=== P3-F: 2-Stage Attention + Focal Loss (128x256) on GPU $GPU ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --batch_size 4 \
      --num_decoder_stages 2 \
      --block_type attention \
      --window_size 8 \
      --num_heads 4 \
      --focal_gamma 1.0 \
      --output_dir checkpoints/unet_p3f_attn_focal
    ;;
  priority)
    echo "=== Phase 3: Priority order — GPU-0: P3-F -> P3-B | GPU-1: P3-A -> P3-D ==="
    mkdir -p "$ROOT/logs"
    # GPU 0: P3-F (fast, 128x256) then P3-B (256x512, accum=2)
    (bash "$0" P3-F 0 && bash "$0" P3-B 0) > "$ROOT/logs/gpu0_p3f_p3b.log" 2>&1 &
    PID_GPU0=$!
    # GPU 1: P3-A (256x512, accum=2) then P3-D (256x512, accum=4)
    (bash "$0" P3-A 1 && bash "$0" P3-D 1) > "$ROOT/logs/gpu1_p3a_p3d.log" 2>&1 &
    PID_GPU1=$!
    echo "GPU-0 chain PID=$PID_GPU0 | GPU-1 chain PID=$PID_GPU1"
    echo "Logs: logs/gpu0_p3f_p3b.log, logs/gpu1_p3a_p3d.log"
    wait $PID_GPU0
    echo "GPU-0 chain complete (P3-F + P3-B)"
    wait $PID_GPU1
    echo "GPU-1 chain complete (P3-A + P3-D)"
    echo "=== Priority Phase 3 ablations complete ==="
    ;;
  all)
    echo "=== Phase 3: All runs — GPU-0: P3-F -> P3-B -> P3-E | GPU-1: P3-A -> P3-D -> P3-C ==="
    mkdir -p "$ROOT/logs"
    # GPU 0: P3-F (2.3h) -> P3-B (16.7h) -> P3-E (67h)
    (bash "$0" P3-F 0 && bash "$0" P3-B 0 && bash "$0" P3-E 0) > "$ROOT/logs/gpu0_p3_all.log" 2>&1 &
    PID_GPU0=$!
    # GPU 1: P3-A (16.7h) -> P3-D (33h) -> P3-C (33h)
    (bash "$0" P3-A 1 && bash "$0" P3-D 1 && bash "$0" P3-C 1) > "$ROOT/logs/gpu1_p3_all.log" 2>&1 &
    PID_GPU1=$!
    echo "GPU-0 chain PID=$PID_GPU0 | GPU-1 chain PID=$PID_GPU1"
    echo "Logs: logs/gpu0_p3_all.log, logs/gpu1_p3_all.log"
    wait $PID_GPU0
    echo "GPU-0 chain complete"
    wait $PID_GPU1
    echo "GPU-1 chain complete"
    echo "=== All Phase 3 ablations complete ==="
    ;;
  *)
    echo "Usage: $0 {P3-A|P3-B|P3-C|P3-D|P3-E|P3-F|priority|all} [gpu_id]"
    echo ""
    echo "Runs:"
    echo "  P3-A  3-stage conv, accum=2, eff_batch=4 (256x512)"
    echo "  P3-B  3-stage attn, accum=2, eff_batch=4 (256x512)  [HIGH PRIORITY]"
    echo "  P3-C  3-stage conv, accum=4, eff_batch=8 (256x512)"
    echo "  P3-D  3-stage attn, accum=4, eff_batch=8 (256x512)  [HIGH POTENTIAL]"
    echo "  P3-E  4-stage attn, accum=4, eff_batch=4 (512x1024) [YOUR REQUEST]"
    echo "  P3-F  2-stage attn + focal, batch=4 (128x256)       [HIGHEST PRIORITY]"
    echo ""
    echo "Modes:"
    echo "  priority  GPU-0: P3-F->P3-B | GPU-1: P3-A->P3-D"
    echo "  all       GPU-0: P3-F->P3-B->P3-E | GPU-1: P3-A->P3-D->P3-C"
    exit 1
    ;;
esac
