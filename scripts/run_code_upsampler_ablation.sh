#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/Users/qbit-glitch/Desktop/datasets/cityscapes}"
CACHE_DIR="${CACHE_DIR:-outputs/code_upsampler/cache_dcfa_v3_90d_64x128}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LIMIT_IMAGES="${LIMIT_IMAGES:-0}"
VAL_LIMIT_IMAGES="${VAL_LIMIT_IMAGES:-0}"
DEVICE="${DEVICE:-auto}"
NUM_WORKERS="${NUM_WORKERS:-0}"

mkdir -p logs outputs/code_upsampler/runs

echo "[code-upsampler] cache train split -> ${CACHE_DIR}"
python3 mbps_pytorch/build_dcfa_code_upsampler_cache.py \
  --cityscapes_root "$CITYSCAPES_ROOT" \
  --split train \
  --output_dir "$CACHE_DIR" \
  --device "$DEVICE" \
  ${LIMIT_IMAGES:+--limit_images "$LIMIT_IMAGES"}

echo "[code-upsampler] cache val split -> ${CACHE_DIR}"
python3 mbps_pytorch/build_dcfa_code_upsampler_cache.py \
  --cityscapes_root "$CITYSCAPES_ROOT" \
  --split val \
  --output_dir "$CACHE_DIR" \
  --device "$DEVICE" \
  ${VAL_LIMIT_IMAGES:+--limit_images "$VAL_LIMIT_IMAGES"}

echo "[code-upsampler] train residual 90D upsampler"
python3 mbps_pytorch/train_code_upsampler.py \
  --variant residual \
  --run_name residual_90d_dcfa_v3_h64w128_k80_seed42 \
  --cache_dir "$CACHE_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  ${LIMIT_IMAGES:+--limit_images "$LIMIT_IMAGES"} \
  ${VAL_LIMIT_IMAGES:+--val_limit_images "$VAL_LIMIT_IMAGES"}

echo "[code-upsampler] train dynamic-kernel 90D upsampler"
python3 mbps_pytorch/train_code_upsampler.py \
  --variant dynamic \
  --run_name dynamic_kernel_90d_dcfa_v3_h64w128_k80_seed42 \
  --cache_dir "$CACHE_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  ${LIMIT_IMAGES:+--limit_images "$LIMIT_IMAGES"} \
  ${VAL_LIMIT_IMAGES:+--val_limit_images "$VAL_LIMIT_IMAGES"}

echo "[code-upsampler] done"
