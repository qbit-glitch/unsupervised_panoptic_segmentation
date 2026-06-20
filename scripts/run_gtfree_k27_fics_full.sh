#!/usr/bin/env bash
# Full GT-free DINOv3 k27 + DepthPro pipeline on fics-lab:
#   1. extract DINOv3 ViT-L/16 train features (GPU, offline weights)
#   2. spherical k-means k=27 GT-free train labels
#   3. seeded DepthGuidedUNet training
# Each step is guarded (idempotent) so the script is resumable.
set -euo pipefail
CS_ROOT="${CS_ROOT:?set CS_ROOT (e.g. /mnt/HDD_16TB/datasets/Cityscapes)}"
PY="${PY:-/mnt/HDD_16TB/umesh/envs/gadepthg/bin/python}"
SEM=pseudo_semantic_raw_dinov3_k27_spherical_kmeans_vitl16
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 1. DINOv3 ViT-L/16 train features (1024-D, 32x64) — skip if already done
if [ ! -f "$CS_ROOT/dinov3_features_vitl16/train/metadata.json" ]; then
  echo "[1/3] extracting DINOv3 ViT-L/16 train features..."
  "$PY" mbps_pytorch/extract_dinov3_features.py \
    --data_dir "$CS_ROOT/leftImg8bit/train" \
    --output_dir "$CS_ROOT/dinov3_features_vitl16/train" \
    --model_name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --image_height 512 --image_width 1024 --batch_size 16 --device cuda
else
  echo "[1/3] features present, skip"
fi

# 2. GT-free k=27 spherical labels (fits on train, assigns train) — skip if done
if [ ! -d "$CS_ROOT/$SEM/train" ]; then
  echo "[2/3] generating GT-free k27 spherical labels..."
  "$PY" mbps_pytorch/generate_clustering_ablation.py \
    --cityscapes_root "$CS_ROOT" \
    --feat_subdir dinov3_features_vitl16 \
    --method spherical_kmeans --k 27 \
    --splits train --seed 42 --output_suffix vitl16
else
  echo "[2/3] labels present, skip"
fi

# 3. Seeded GT-free training
echo "[3/3] training DepthGuidedUNet (k=27, DINOv3 1024-D, DepthPro)..."
"$PY" mbps_pytorch/train_refine_net.py \
  --cityscapes_root "$CS_ROOT" \
  --model_type unet --num_classes 27 \
  --feature_subdir dinov3_features_vitl16 --feature_dim 1024 \
  --semantic_subdir "$SEM" \
  --depth_subdir depth_depthpro \
  --block_type attention --num_decoder_stages 2 --num_bottleneck_blocks 2 \
  --bridge_dim 192 --skip_dim 32 --window_size 8 --num_heads 4 \
  --num_epochs 12 --batch_size 4 --lr 1e-4 --seed 42 \
  --eval_interval 999 \
  --output_dir checkpoints/gtfree_k27 \
  --device cuda --gpu 0
echo "PIPELINE DONE"
