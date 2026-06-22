#!/usr/bin/env bash
# Launch N parallel shards of build_cause_tr_27_seq.py
# Usage: bash scripts/launch_cause_tr_27_seq.sh [N_SHARDS]
# Default: 4 shards (~10.8h at 1.74s/frame on MPS)

set -euo pipefail

N=${1:-4}
LOGDIR="logs/cause_tr_27_seq"
mkdir -p "$LOGDIR"

echo "Launching $N shards → logs at $LOGDIR/shard_{0..$((N-1))}.log"

for i in $(seq 0 $((N-1))); do
    log="$LOGDIR/shard_${i}.log"
    setsid .venv/bin/python scripts/build_cause_tr_27_seq.py \
        --shard "$i" --total "$N" \
        > "$log" 2>&1 &
    echo "  shard $i → PID $! → $log"
done

echo "All $N shards launched. Monitor with:"
echo "  tail -f $LOGDIR/shard_0.log"
echo "  watch 'find /Volumes/code_files_2/mbps_instances_seq/cause_tr_27_seq -name \"*.png\" | wc -l'"
