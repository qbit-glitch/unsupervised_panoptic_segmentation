#!/usr/bin/env bash
# Launch EoMT Stage-3 self-training with the EXACT official CUPS protocol
# (teacher TTA, copy-paste, photometric, random crop, resolution jitter,
# wd=1e-5, lr=1e-4, 3x4000 steps, clean-teacher/augmented-student).
#
# Target host: santosh@172.17.254.146, 2x GTX 1080 Ti (11 GB each, sm_61).
# Run ON the remote (after rsyncing the repo):
#     bash scripts/run_eomt_stage3_cupsexact_santosh.sh

set -eo pipefail

EOMT_ROOT="${EOMT_ROOT:-/home/santosh/mbps_panoptic_segmentation/refs/eomt}"
CONDA_ROOT="${CONDA_ROOT:-/home/santosh/anaconda3}"
CONDA_ENV="${CONDA_ENV:-cups}"
CONFIG="${CONFIG:-configs/dinov3/cityscapes/panoptic/eomt_base_640_santosh_stage3_cupsexact.yaml}"
EXP_DIR="/home/santosh/experiments/stage3_eomt_dinov2_vitb_dcfa_simcf_abc_spherical_k80_cupsexact"
LOG_DIR="${LOG_DIR:-${EXP_DIR}/logs}"

mkdir -p "${LOG_DIR}"

eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
conda activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${EOMT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_ROOT}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# Pascal sm_61: no Flash Attention / bfloat16; disable torch.compile.
export TORCH_DYNAMO_DISABLE=1
export PYTORCH_SDP_BACKEND=math

if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
  export WANDB_MODE="${WANDB_MODE:-offline}"
  echo "[run] No W&B credential found; running with WANDB_MODE=${WANDB_MODE}"
fi

cd "${EOMT_ROOT}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[run] config not found: ${EOMT_ROOT}/${CONFIG}" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/eomt_stage3_cupsexact_${STAMP}.log"

# Resume from last.ckpt if present (preempt/restart safety).
CKPT_LAST="${CKPT_LAST:-${EXP_DIR}/checkpoints/last.ckpt}"
RESUME_ARGS=()
if [[ -f "${CKPT_LAST}" ]]; then
  echo "[run] Resuming from ${CKPT_LAST}"
  RESUME_ARGS=("--ckpt_path" "${CKPT_LAST}")
fi

echo "[run] CONFIG   = ${CONFIG}"
echo "[run] LOG_FILE = ${LOG_FILE}"

setsid nohup python -u main.py fit \
    -c "${CONFIG}" \
    --compile_disabled \
    "${RESUME_ARGS[@]}" \
    > "${LOG_FILE}" 2>&1 < /dev/null &
PID=$!

echo "[run] PID=${PID}"
echo "[run] tail -f ${LOG_FILE}"
echo "${PID}" > "${LOG_DIR}/eomt_stage3_cupsexact_${STAMP}.pid"
disown ${PID} || true
