#!/bin/bash
# Instance-Conditioned UNet Ablation Study — Push PQ_Things > 20
# Baseline: P2-B PQ=28.00 (2-stage attn, ep8), PQ_things=18.32 (CC)
# Stage-1 pseudo-labels: PQ_things=19.41 (depth-guided instances)
# 6 ablations: IC-0 through IC-E
# Run on remote: santosh@100.93.203.100, 2x GTX 1080 Ti

PYTHON="python"
ROOT="/media/santosh/Kuldeep/panoptic_segmentation"
CS="/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes"
export PYTHONPATH="$ROOT"
export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:$LD_LIBRARY_PATH"

# P2-B best checkpoint (to fine-tune from or evaluate)
P2B_CKPT="checkpoints/unet_p2b_2stage_attn/best.pth"

# Common base args (matching P2-B config, but fine-tuning LR)
BASE_ARGS="--cityscapes_root $CS \
  --semantic_subdir pseudo_semantic_mapped_k80 \
  --num_classes 19 \
  --model_type unet \
  --bridge_dim 192 \
  --num_bottleneck_blocks 2 \
  --skip_dim 32 \
  --coupling_strength 0.1 \
  --block_type attention \
  --window_size 8 \
  --num_heads 4 \
  --num_decoder_stages 2 \
  --num_epochs 12 \
  --lr 2e-5 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --num_workers 4 \
  --eval_interval 2 \
  --lambda_distill 1.0 \
  --lambda_distill_min 0.9 \
  --lambda_align 0.25 \
  --lambda_proto 0.025 \
  --lambda_ent 0.025 \
  --label_smoothing 0.1"

RUN=$1
GPU=${2:-0}

case "$RUN" in
  IC-0|eval_only)
    echo "=== IC-0: Eval-only — P2-B checkpoint with pre-computed instances ==="
    echo "This is a zero-cost baseline: re-evaluate P2-B with depth-guided instances."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --num_epochs 0 \
      --eval_interval 1 \
      --resume $P2B_CKPT \
      --eval_instance_subdir pseudo_instance_spidepth \
      --output_dir checkpoints/unet_ic0_eval_only
    ;;
  IC-A|loss_only)
    echo "=== IC-A: Loss-only — P2-B + instance uniformity loss, CC eval ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --resume $P2B_CKPT \
      --instance_subdir pseudo_instance_spidepth \
      --lambda_instance_uniform 0.5 \
      --output_dir checkpoints/unet_ica_loss_only
    ;;
  IC-B|arch_only)
    echo "=== IC-B: Arch-only — P2-B + InstanceSkipBlock, precomputed eval ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --resume $P2B_CKPT \
      --use_instance \
      --inst_skip_dim 16 \
      --instance_subdir pseudo_instance_spidepth \
      --eval_instance_subdir pseudo_instance_spidepth \
      --output_dir checkpoints/unet_icb_arch_only
    ;;
  IC-C|arch_loss)
    echo "=== IC-C: Arch + Loss — InstanceSkipBlock + uniformity, precomputed eval ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --resume $P2B_CKPT \
      --use_instance \
      --inst_skip_dim 16 \
      --instance_subdir pseudo_instance_spidepth \
      --lambda_instance_uniform 0.5 \
      --eval_instance_subdir pseudo_instance_spidepth \
      --output_dir checkpoints/unet_icc_arch_loss
    ;;
  IC-D|full)
    echo "=== IC-D: Full combo — InstanceSkipBlock + uniform + boundary, precomputed eval ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --resume $P2B_CKPT \
      --use_instance \
      --inst_skip_dim 16 \
      --instance_subdir pseudo_instance_spidepth \
      --lambda_instance_uniform 0.5 \
      --lambda_instance_boundary 0.1 \
      --eval_instance_subdir pseudo_instance_spidepth \
      --output_dir checkpoints/unet_icd_full
    ;;
  IC-E|focal)
    echo "=== IC-E: Best combo — InstanceSkipBlock + uniform + focal, precomputed eval ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --resume $P2B_CKPT \
      --use_instance \
      --inst_skip_dim 16 \
      --instance_subdir pseudo_instance_spidepth \
      --lambda_instance_uniform 0.5 \
      --focal_gamma 1.0 \
      --eval_instance_subdir pseudo_instance_spidepth \
      --output_dir checkpoints/unet_ice_focal
    ;;
  *)
    echo "Usage: $0 {IC-0|IC-A|IC-B|IC-C|IC-D|IC-E} [GPU_ID]"
    echo ""
    echo "Ablation runs (priority order):"
    echo "  IC-0  Eval-only: P2-B checkpoint with pre-computed instances (FREE)"
    echo "  IC-A  Loss-only: P2-B + instance uniformity loss (CC eval)"
    echo "  IC-B  Arch-only: P2-B + InstanceSkipBlock (precomputed eval)"
    echo "  IC-C  Arch+Loss: InstanceSkipBlock + uniformity (precomputed eval)"
    echo "  IC-D  Full:      InstanceSkipBlock + uniform + boundary (precomputed eval)"
    echo "  IC-E  Focal:     InstanceSkipBlock + uniform + focal (precomputed eval)"
    exit 1
    ;;
esac
