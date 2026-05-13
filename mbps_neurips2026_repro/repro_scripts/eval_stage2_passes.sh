#!/usr/bin/env bash
# Evaluate a Stage-2 checkpoint from the loss-augmentation plan and emit
# a pq.json next to the checkpoint so P0/P1/P2/P3/P4 gates can diff.
#
# Usage:
#   bash scripts/eval_stage2_passes.sh <path/to/checkpoint.ckpt> [val_split]
#   val_split defaults to "val" (500 images).
#
# Output:
#   <checkpoint_dir>/pq.json   (PQ / SQ / RQ / mIoU + per-class breakdown)
#   <checkpoint_dir>/eval.txt  (human-readable summary)
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <checkpoint> [split]" >&2
  exit 2
fi

CKPT="$1"
SPLIT="${2:-val}"
CKPT_DIR="$(dirname "$CKPT")"

if [[ ! -f "$CKPT" ]]; then
  echo "error: checkpoint not found: $CKPT" >&2
  exit 3
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Resolve the python used for the other CUPS evals; fall back to project venv.
if [[ -z "${PY:-}" ]]; then
  if [[ -x "/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python" ]]; then
    PY="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
  else
    PY="python3"
  fi
fi

export PYTHONPATH="$REPO_ROOT/refs/cups:${PYTHONPATH:-}"

EVAL_LOG="$CKPT_DIR/eval.txt"
echo "[eval_stage2_passes] ckpt=$CKPT split=$SPLIT" | tee "$EVAL_LOG"

# Reuse the CUPS Cityscapes evaluator; it writes per-class PQ/SQ/RQ + mIoU.
"$PY" -u refs/cups/evaluate_cityscapes.py \
  --checkpoint "$CKPT" \
  --split "$SPLIT" \
  --output_json "$CKPT_DIR/pq.json" \
  2>&1 | tee -a "$EVAL_LOG"

echo "[eval_stage2_passes] wrote $CKPT_DIR/pq.json"
