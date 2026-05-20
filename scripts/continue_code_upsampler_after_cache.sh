#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/Users/qbit-glitch/Desktop/datasets/cityscapes}"
CACHE_DIR="${CACHE_DIR:-outputs/code_upsampler/cache_dcfa_v3_90d_64x128}"
EXPECTED_TRAIN="${EXPECTED_TRAIN:-2975}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-auto}"
NUM_WORKERS="${NUM_WORKERS:-0}"

count_cache() {
  find "$CACHE_DIR/train" -type f -name '*.npz' 2>/dev/null | wc -l | tr -d ' '
}

mkdir -p logs outputs/code_upsampler/runs
echo "[code-upsampler-cont] waiting for train cache: ${CACHE_DIR}/train"

while true; do
  count="$(count_cache)"
  echo "[code-upsampler-cont] train cache ${count}/${EXPECTED_TRAIN}"
  if [ "$count" -ge "$EXPECTED_TRAIN" ]; then
    break
  fi
  sleep 120
done

echo "[code-upsampler-cont] cache val split -> ${CACHE_DIR}"
python3 mbps_pytorch/build_dcfa_code_upsampler_cache.py \
  --cityscapes_root "$CITYSCAPES_ROOT" \
  --split val \
  --output_dir "$CACHE_DIR" \
  --device "$DEVICE"

echo "[code-upsampler-cont] train residual 90D upsampler"
python3 mbps_pytorch/train_code_upsampler.py \
  --variant residual \
  --run_name residual_90d_dcfa_v3_h64w128_k80_seed42 \
  --cache_dir "$CACHE_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --device "$DEVICE"

echo "[code-upsampler-cont] train dynamic-kernel 90D upsampler"
python3 mbps_pytorch/train_code_upsampler.py \
  --variant dynamic \
  --run_name dynamic_kernel_90d_dcfa_v3_h64w128_k80_seed42 \
  --cache_dir "$CACHE_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --device "$DEVICE"

echo "[code-upsampler-cont] done"
