#!/usr/bin/env bash
# Stage-3 self-training: ORIGINAL CUPS pipeline (train_self.py recipe) with
# the DINOv2 + EoMT network via the D2 adapter.
#
# Target host: santosh@172.17.254.146, 2x GTX 1080 Ti.
# Run ON the remote (after rsyncing the repo):
#     bash scripts/run_cups_stage3_eomt_santosh.sh
#
# Pass SMOKE=1 for a 1-GPU quick start check (no detach, 5 min timeout).

set -eo pipefail

REPO="${REPO:-/home/santosh/mbps_panoptic_segmentation}"
CUPS_ROOT="${REPO}/refs/cups"
EOMT_ROOT="${REPO}/refs/eomt"
CONDA_ROOT="${CONDA_ROOT:-/home/santosh/anaconda3}"
CONDA_ENV="${CONDA_ENV:-cups}"
CONFIG="${CONFIG:-configs/train_self_cityscapes_eomt_dinov2_dcfa_simcf_abc_santosh.yaml}"
EXP_DIR="/home/santosh/experiments/stage3_cups_orig_eomt_dinov2_dcfa_simcf_abc_spherical_k80"
LOG_DIR="${LOG_DIR:-${EXP_DIR}/logs}"

mkdir -p "${LOG_DIR}"

eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
conda activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
# BOTH trees: cups (pipeline) + eomt (models.eomt / training.mask_classification_loss)
export PYTHONPATH="${CUPS_ROOT}:${EOMT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_ROOT}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# Pascal sm_61: no Flash Attention / bfloat16; disable torch.compile.
export TORCH_DYNAMO_DISABLE=1
export PYTORCH_SDP_BACKEND=math

if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
  export WANDB_MODE="${WANDB_MODE:-offline}"
  echo "[run] No W&B credential found; running with WANDB_MODE=${WANDB_MODE}"
fi

cd "${CUPS_ROOT}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[run] config not found: ${CUPS_ROOT}/${CONFIG}" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/cups_stage3_eomt_${STAMP}.log"

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "[run] SMOKE mode: 1 GPU, foreground, SIGKILL after 360s"
  timeout -s KILL 360 python -u train_self_eomt.py \
      --experiment_config_file "${CONFIG}" \
      --disable_wandb \
      SYSTEM.NUM_GPUS 1 || true
  exit 0
fi

RESUME_ARGS=()
CKPT_LAST="$(ls -t ${EXP_DIR}/experiments/*/Unsupervised\ Panoptic\ Segmentation/*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)"
if [[ -n "${CKPT_LAST}" && -f "${CKPT_LAST}" ]]; then
  echo "[run] Resuming from ${CKPT_LAST}"
  RESUME_ARGS=(--ckpt_path "${CKPT_LAST}")
fi

echo "[run] CONFIG   = ${CONFIG}"
echo "[run] LOG_FILE = ${LOG_FILE}"

setsid nohup python -u train_self_eomt.py \
    --experiment_config_file "${CONFIG}" \
    "${RESUME_ARGS[@]}" \
    > "${LOG_FILE}" 2>&1 < /dev/null &
PID=$!

echo "[run] PID=${PID}"
echo "[run] tail -f ${LOG_FILE}"
echo "${PID}" > "${LOG_DIR}/cups_stage3_eomt_${STAMP}.pid"
disown ${PID} || true
