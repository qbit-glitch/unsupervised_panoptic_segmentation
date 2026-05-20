#!/usr/bin/env bash
# Wait until both GTX 1080 Ti GPUs on the Santosh remote are idle, then
# launch the DCFA+SIMCF-ABC loss-only Stage-2 training run.
#
# Idle means: each GPU has <512 MiB used AND <3% utilization for
# IDLE_CONFIRM consecutive polls spaced POLL_INTERVAL seconds apart.
# This avoids racing the previous job's teardown.
#
# Run ON the remote (e.g. via `nohup bash scripts/wait_for_gpus_and_launch_loss_only.sh &`).

set -euo pipefail

POLL_INTERVAL="${POLL_INTERVAL:-30}"
IDLE_CONFIRM="${IDLE_CONFIRM:-3}"
MEM_THRESHOLD_MIB="${MEM_THRESHOLD_MIB:-512}"
UTIL_THRESHOLD_PCT="${UTIL_THRESHOLD_PCT:-3}"

CUPS_ROOT="${CUPS_ROOT:-/home/santosh/cups}"
RUN_NAME="${RUN_NAME:-dcfa_simcf_abc_loss_only_2gpu}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${CUPS_ROOT}/logs}"
TRAIN_LOG="${TRAIN_LOG_DIR}/${RUN_NAME}.log"

mkdir -p "${TRAIN_LOG_DIR}"

echo "[wait] polling nvidia-smi every ${POLL_INTERVAL}s; need ${IDLE_CONFIRM} consecutive idle polls"
echo "[wait] threshold: mem < ${MEM_THRESHOLD_MIB} MiB AND util < ${UTIL_THRESHOLD_PCT}% on BOTH GPUs"
echo "[wait] target run name: ${RUN_NAME}"
echo "[wait] training log will be: ${TRAIN_LOG}"

idle_count=0

while (( idle_count < IDLE_CONFIRM )); do
  mapfile -t gpu_lines < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  all_idle=1
  status_csv=""
  for line in "${gpu_lines[@]}"; do
    IFS=',' read -r idx mem util <<<"${line}"
    status_csv+="${idx}:${mem}MiB/${util}%  "
    if (( mem >= MEM_THRESHOLD_MIB )) || (( util >= UTIL_THRESHOLD_PCT )); then
      all_idle=0
    fi
  done

  if (( all_idle == 1 )); then
    idle_count=$((idle_count + 1))
    echo "[wait] $(date '+%F %T')  idle ${idle_count}/${IDLE_CONFIRM}  ${status_csv}"
  else
    if (( idle_count > 0 )); then
      echo "[wait] $(date '+%F %T')  busy — resetting confirm counter  ${status_csv}"
    else
      echo "[wait] $(date '+%F %T')  busy  ${status_csv}"
    fi
    idle_count=0
  fi

  if (( idle_count < IDLE_CONFIRM )); then
    sleep "${POLL_INTERVAL}"
  fi
done

echo "[wait] GPUs idle confirmed at $(date '+%F %T'). Launching training."

cd "${CUPS_ROOT}"
nohup bash "scripts/run_dcfa_simcf_abc_loss_only_santosh.sh" \
  > "${TRAIN_LOG}" 2>&1 &
TRAIN_PID=$!
echo "${TRAIN_PID}" > "${TRAIN_LOG_DIR}/${RUN_NAME}.pid"
echo "[wait] training launched: PID=${TRAIN_PID}"
echo "[wait] tail with: tail -f ${TRAIN_LOG}"
