#!/usr/bin/env bash
# Stage-2 EoMT A/B -- AnyUp vs bilinear pseudo-labels. Run on santosh (2x1080Ti).
# Both arms: ablA config (no DropLoss/CopyPaste), 8000 steps, seed 42. ONLY DATA.ROOT_PSEUDO differs
# (baked into each config). Bilinear (control) runs first so a crash still leaves the baseline.
#
# LAUNCH (single line, from the repo root on santosh):
#   nohup bash scripts/run_anyup_vs_bilinear_stage2.sh > logs/ab_stage2.log 2>&1 & echo "PID $!"
#   tail -f logs/ab_stage2.log
#
# Prereqs on santosh (see runbook): git pull this branch; rsync the bilinear label dir
# (incl. pseudo_classes_split_1.pt); confirm the anyup dir + cityscapes are present.
set -u
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
CONDA="${CONDA:-$HOME/anaconda3/envs/cups}"
PY="$CONDA/bin/python"
export LD_LIBRARY_PATH="$CONDA/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/refs/cups:$REPO/refs/eomt"
cd "$REPO" || { echo "cannot cd to REPO=$REPO"; exit 1; }
mkdir -p logs

run_arm () {
  local name="$1" cfg="$2" logp="$3"
  echo "########## [$(date)] TRAIN arm=$name cfg=$cfg ##########"
  CUDA_VISIBLE_DEVICES=0,1 "$PY" -u refs/cups/train_eomt.py \
      --experiment_config_file "$cfg" \
    || { echo "!!! TRAIN $name FAILED (see above) -- continuing to next arm"; return 1; }

  echo "########## [$(date)] EVAL arm=$name (every retained best_pq ckpt, NUM_GPUS=1) ##########"
  local found=0 ck
  while IFS= read -r ck; do
    [ -z "$ck" ] && continue
    found=1
    echo "---- eval $name : $ck ----"
    "$PY" -u refs/cups/eval_eomt_checkpoint.py \
        --experiment_config_file "$cfg" --ckpt "$ck" SYSTEM.NUM_GPUS 1 \
      2>&1 | tee "logs/eval_${name}_$(basename "$ck" .ckpt).txt"
  done < <(find "$logp" -name 'best_pq_step*.ckpt' 2>/dev/null | sort -V)
  [ "$found" -eq 0 ] && echo "!!! no best_pq ckpt found under $logp for arm=$name"
}

# control arm first
run_arm bilinear refs/cups/configs/stage2_ab_bilinear.yaml /home/santosh/experiments/anyup_vs_bilinear/bilinear
run_arm anyup    refs/cups/configs/stage2_ab_anyup.yaml    /home/santosh/experiments/anyup_vs_bilinear/anyup

echo "########## [$(date)] DONE -- PQ SUMMARY (max over ckpts per arm is the arm's result) ##########"
grep -H -E '^[[:space:]]+(PQ|PQ_things|PQ_stuff|SQ|RQ)[[:space:]]*=' logs/eval_bilinear_*.txt logs/eval_anyup_*.txt 2>/dev/null \
  || echo "no eval metric lines parsed -- read logs/eval_*_*.txt directly"
