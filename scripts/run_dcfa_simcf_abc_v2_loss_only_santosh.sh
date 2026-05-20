#!/usr/bin/env bash
# Launch Stage-2 v2 training of DCFA+SIMCF-ABC + DINOv3 + Cascade Mask R-CNN
# on the Santosh remote, DDP across both GTX 1080 Ti GPUs. Uses box-cleaned
# pseudo-labels + cascade s1/s2 reweighting + cosine LR schedule + eff batch 48.
# Validation reports loss metrics only.
#
# Run ON the remote:
#   bash scripts/run_dcfa_simcf_abc_v2_loss_only_santosh.sh
#
# Required env:
#   CUPS_ROOT     (default: /home/santosh/cups)
#   CONDA_ROOT    (default: /home/santosh/anaconda3)
#   CONDA_ENV     (default: cups)
#   RUN_NAME      (default: dcfa_simcf_abc_v2_loss_only_2gpu)
#   LOG_PATH      (default: /home/santosh/experiments/stage2_dcfa_simcf_abc_v2)
#   SEED          (default: 43)

set -eo pipefail

CUPS_ROOT="${CUPS_ROOT:-/home/santosh/cups}"
CONDA_ROOT="${CONDA_ROOT:-/home/santosh/anaconda3}"
CONDA_ENV="${CONDA_ENV:-cups}"
RUN_NAME="${RUN_NAME:-dcfa_simcf_abc_v2_loss_only_2gpu}"
LOG_PATH="${LOG_PATH:-/home/santosh/experiments/stage2_dcfa_simcf_abc_v2}"
SEED="${SEED:-43}"

eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
conda activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${CONDA_ROOT}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# WandB: prefer online; fall back to offline so training never blocks.
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
  export WANDB_MODE="${WANDB_MODE:-offline}"
  echo "[run] WandB credential not found; running with WANDB_MODE=${WANDB_MODE}"
else
  echo "[run] WandB credential found; running with online sync"
fi

cd "${CUPS_ROOT}"
mkdir -p logs "${LOG_PATH}"

CONFIG="configs/train_cityscapes_dinov3_vitb_dcfa_simcf_abc_v2_santosh.yaml"
if [[ ! -f "${CONFIG}" ]]; then
  echo "[run] config not found: ${CUPS_ROOT}/${CONFIG}" >&2
  exit 2
fi

python -u train_loss_only.py \
  --experiment_config_file "${CONFIG}" \
  SYSTEM.SEED "${SEED}" \
  SYSTEM.RUN_NAME "${RUN_NAME}" \
  SYSTEM.LOG_PATH "${LOG_PATH}"
