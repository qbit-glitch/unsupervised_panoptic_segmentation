#!/usr/bin/env bash
# Usage: scripts/run_stage2_P_passes.sh <pass> [machine]
#   pass:    P0 | P1_loce | P2_featmirror | P3_dglr | P4_daff
#            (shorthand: P1, P2, P3, P4 map to the canonical suffixes)
#   machine: anydesk (default) | santosh | $STAGE2_MACHINE env var
#
# Plan A (`.claude/plans/okay-let-s-do-a-lovely-cerf.md`): P0-P4 semantic
# auxiliary-loss passes on top of frozen DINOv3 ViT-B/16 + Cascade Mask R-CNN +
# k=80 semantic + DepthPro tau=0.20 instance pseudo-labels.
#
# Each pass uses the same Plan A entrypoint (refs/cups/train.py) with a
# derived yaml. The machine override file refs/cups/configs/_machines/<machine>.env
# supplies DATA.ROOT*, TRAINING.BATCH_SIZE/ACCUMULATE_GRAD_BATCHES,
# SYSTEM.NUM_WORKERS, and LOG_ROOT (used to build SYSTEM.LOG_PATH).
# Each KEY=VALUE line becomes a "KEY VALUE" pair appended to the yacs
# positional CLI overrides. LOG_ROOT is consumed by bash, not yacs.
#
# PREREQ (run BEFORE this launcher):
#   anydesk:  module load gcc-9.3.0 && source ~/umesh/ups_env/bin/activate
#   santosh:  conda activate cups

set -euo pipefail

PASS="${1:?pass required (P0|P1|P2|P3|P4|P1_loce|P2_featmirror|P3_dglr|P4_daff)}"
MACHINE="${2:-${STAGE2_MACHINE:-anydesk}}"

# Normalize shorthand -> canonical suffix
case "$PASS" in
  P0)              PASS_SUFFIX="" ;;
  P1|P1_loce)      PASS_SUFFIX="_P1_loce" ;;
  P2|P2_featmirror) PASS_SUFFIX="_P2_featmirror" ;;
  P3|P3_dglr)      PASS_SUFFIX="_P3_dglr" ;;
  P4|P4_daff)      PASS_SUFFIX="_P4_daff" ;;
  *) echo "FATAL: unknown pass '$PASS' (expected P0|P1|P2|P3|P4 or canonical suffix)" >&2; exit 1 ;;
esac

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$PROJ_ROOT/refs/cups/configs/train_cityscapes_dinov3_vitb_k80_depthpro_tau020${PASS_SUFFIX}.yaml"
RUN_NAME="stage2_${PASS}_k80_depthpro_tau020"
MACHINE_ENV="$PROJ_ROOT/refs/cups/configs/_machines/${MACHINE}.env"

[[ -f "$CFG" ]]         || { echo "FATAL: config $CFG not found" >&2; exit 1; }
[[ -f "$MACHINE_ENV" ]] || { echo "FATAL: machine override $MACHINE_ENV not found" >&2; exit 1; }

# Parse machine env: KEY=VALUE lines become a flat yacs CLI list "KEY VALUE ...".
# LOG_ROOT is extracted separately (not a yacs key).
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
LOGFILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

# The P0 yaml hardcodes SYSTEM.LOG_PATH to "results/stage2_P0_baseline/"
# (relative, santosh-flavoured). Override to machine-scoped absolute path.
OVERRIDE_ARGS+=("SYSTEM.LOG_PATH" "$LOG_DIR")
OVERRIDE_ARGS+=("SYSTEM.RUN_NAME" "$RUN_NAME")

# Required environment
export WANDB_MODE=disabled
export PYTHONPATH="$PROJ_ROOT/refs/cups:${PYTHONPATH:-}"
# Some aux losses (Gated-CRF at K=5) unfold 640x1280 feature maps into
# (B, C*25, H*W) tensors which fragment the CUDA allocator. Expandable
# segments lets the allocator grow on demand instead of failing on a
# contiguous allocation inside the working set.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJ_ROOT"

echo "=== Stage-2 Plan A P-pass launcher ==="
echo "Pass:      $PASS"
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
