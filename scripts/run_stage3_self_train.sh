#!/usr/bin/env bash
# Usage: scripts/run_stage3_self_train.sh [config] [machine]
#   config:  configs/stage2_m2f/Stage3_self_train.yaml (default)
#   machine: anydesk (default) | santosh | $STAGE2_MACHINE env var
#
# Stage-3 uses refs/cups/train_self.py (not train.py). Self-training runs for
# SELF_TRAINING.ROUND_STEPS * SELF_TRAINING.ROUNDS optimizer steps. EMA teacher
# updates happen every step via on_train_batch_end (pl_model_pseudo.py).
#
# The machine override file configs/stage2_m2f/_machines/<machine>.env supplies
# DATA.ROOT*, TRAINING.BATCH_SIZE/ACCUMULATE_GRAD_BATCHES, SYSTEM.NUM_WORKERS,
# LOG_ROOT (consumed by bash, not yacs).
#
# PREREQ:
#   anydesk:  module load gcc-9.3.0 && source ~/umesh/ups_env/bin/activate
#   santosh:  conda activate cups

set -euo pipefail

CFG="${1:-configs/stage2_m2f/Stage3_self_train.yaml}"
MACHINE="${2:-${STAGE2_MACHINE:-anydesk}}"
RUN_NAME="$(basename "$CFG" .yaml)"

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MACHINE_ENV="$PROJ_ROOT/configs/stage2_m2f/_machines/${MACHINE}.env"

[[ -f "$CFG" ]]         || { echo "FATAL: config $CFG not found" >&2; exit 1; }
[[ -f "$MACHINE_ENV" ]] || { echo "FATAL: machine override $MACHINE_ENV not found" >&2; exit 1; }

LOG_ROOT=""
declare -a OVERRIDE_ARGS=()
while IFS= read -r line; do
  line="${line%%#*}"
  line="${line#"${line%%[![:space:]]*}"}"
  [[ -z "$line" ]] && continue
  KEY="${line%%=*}"
  VAL="${line#*=}"
  KEY="${KEY%"${KEY##*[![:space:]]}"}"
  VAL="${VAL#"${VAL%%[![:space:]]*}"}"
  if [[ "$KEY" == "LOG_ROOT" ]]; then
    LOG_ROOT="$VAL"
    continue
  fi
  OVERRIDE_ARGS+=("$KEY" "$VAL")
done < "$MACHINE_ENV"

[[ -n "$LOG_ROOT" ]] || { echo "FATAL: LOG_ROOT missing in $MACHINE_ENV" >&2; exit 1; }

LOG_DIR="$LOG_ROOT/$RUN_NAME"
LOGFILE="$LOG_DIR/selftrain_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

OVERRIDE_ARGS+=("SYSTEM.LOG_PATH" "$LOG_DIR")
OVERRIDE_ARGS+=("SYSTEM.RUN_NAME" "$RUN_NAME")

export WANDB_MODE=disabled
export PYTHONPATH="$PROJ_ROOT/refs/cups:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJ_ROOT"

echo "=== Stage-3 Self-Training launcher ==="
echo "Config:    $CFG"
echo "Machine:   $MACHINE ($MACHINE_ENV)"
echo "Run name:  $RUN_NAME"
echo "Log dir:   $LOG_DIR"
echo "Log file:  $LOGFILE"
echo "Overrides: ${OVERRIDE_ARGS[*]}"
echo "========================================"

nohup python -u refs/cups/train_self.py \
  --experiment_config_file "$CFG" \
  --disable_wandb \
  "${OVERRIDE_ARGS[@]}" \
  > "$LOGFILE" 2>&1 &

PID=$!
echo "PID=$PID"
echo "LOG=$LOGFILE"
echo ""
echo "Monitor: tail -f $LOGFILE"
echo "Kill:    kill $PID"
