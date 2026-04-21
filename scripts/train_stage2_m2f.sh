#!/usr/bin/env bash
# Usage: scripts/train_stage2_m2f.sh <config-path> [machine]
#   config-path: configs/stage2_m2f/G1_EMA.yaml
#   machine:     anydesk (default) | santosh | $STAGE2_MACHINE env var
#
# The machine override file configs/stage2_m2f/_machines/<machine>.env
# supplies DATA.ROOT*, TRAINING.BATCH_SIZE/ACCUMULATE_GRAD_BATCHES,
# SYSTEM.NUM_WORKERS, and LOG_ROOT (used to build SYSTEM.LOG_PATH).
# Each KEY=VALUE line becomes a "KEY VALUE" pair appended to the yacs
# positional CLI overrides. LOG_ROOT is consumed by bash, not yacs.
#
# PREREQ (run BEFORE this launcher):
#   anydesk:  module load gcc-9.3.0 && source ~/umesh/ups_env/bin/activate
#   santosh:  conda activate cups

set -euo pipefail

CFG="${1:?config path required}"
MACHINE="${2:-${STAGE2_MACHINE:-anydesk}}"
RUN_NAME="$(basename "$CFG" .yaml)"

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MACHINE_ENV="$PROJ_ROOT/configs/stage2_m2f/_machines/${MACHINE}.env"

[[ -f "$CFG" ]]         || { echo "FATAL: config $CFG not found" >&2; exit 1; }
[[ -f "$MACHINE_ENV" ]] || { echo "FATAL: machine override $MACHINE_ENV not found" >&2; exit 1; }

# Parse machine env: KEY=VALUE lines become a flat yacs CLI list "KEY VALUE ...".
# LOG_ROOT is extracted separately (not a yacs key).
LOG_ROOT=""
declare -a OVERRIDE_ARGS=()
while IFS= read -r line; do
  # Skip blanks and comments
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
LOGFILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

# Append LOG_PATH + RUN_NAME yacs overrides
OVERRIDE_ARGS+=("SYSTEM.LOG_PATH" "$LOG_DIR")
OVERRIDE_ARGS+=("SYSTEM.RUN_NAME" "$RUN_NAME")

# Required environment
export WANDB_MODE=disabled
export PYTHONPATH="$PROJ_ROOT/refs/cups:${PYTHONPATH:-}"
# ViT-Adapter c2 cross-attn at stride-4 fragments CUDA allocator; expandable
# segments lets the allocator grow on demand instead of failing on a
# contiguous 1.5 GiB request inside the 44 GiB working set.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJ_ROOT"

echo "=== Stage-2 M2F+ViT-Adapter launcher ==="
echo "Config:    $CFG"
echo "Machine:   $MACHINE ($MACHINE_ENV)"
echo "Run name:  $RUN_NAME"
echo "Log dir:   $LOG_DIR"
echo "Log file:  $LOGFILE"
echo "Overrides: ${OVERRIDE_ARGS[*]}"
echo "========================================"

nohup python -u refs/cups/train.py \
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
