#!/usr/bin/env bash
set -euo pipefail

RUN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! "$RUN" =~ ^[ABC]$ ]]; then
  echo "Usage: $0 --run A|B|C" >&2
  exit 2
fi

REMOTE="${REMOTE:-santosh@172.17.254.146}"
REMOTE_REPO="${REMOTE_REPO:-/media/santosh/Kuldeep/panoptic_segmentation}"
REMOTE_DATA="${REMOTE_DATA:-/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes}"
# shellcheck disable=SC2089
# Tight rsync excludes: refs/cups/ contains experiments/ (~53 GB checkpoints),
# results/ (~4.5 GB), and assets/ (~27 MB) which must NOT be uploaded. We
# only push source code and configs.
RSYNC="${RSYNC:-rsync -avz \
  --exclude=__pycache__ \
  --exclude=*.pyc \
  --exclude=experiments/ \
  --exclude=results/ \
  --exclude=assets/ \
  --exclude=logs/ \
  --exclude=outputs/ \
  --exclude=checkpoints/ \
  --exclude=*.ckpt \
  --exclude=*.pth \
  --exclude=*.npz \
  --exclude=*.tar.gz \
  --exclude=.git/}"

# Guard against duplicate launches: refuse if a python train.py is already
# running on remote. We require "python" in the command line so the guard
# does not false-positive on bash -c wrappers and ssh diagnostic commands
# that happen to mention the config path.
if ssh "$REMOTE" "pgrep -u santosh -af 'python.*train\\.py.*configs/train_cityscapes_t0_longtail' >/dev/null"; then
  echo "ERROR: A t0_longtail training is already running on $REMOTE. Kill it first:" >&2
  echo "       ssh $REMOTE 'pkill -u santosh -9 -f train.py'" >&2
  exit 3
fi
LOCAL_DATA="${LOCAL_DATA:-$HOME/Desktop/datasets/cityscapes}"
POOL_PATH="$LOCAL_DATA/rare_instance_pool/pool_t0.pkl"

if [[ "$RUN" != "A" && ! -f "$POOL_PATH" ]]; then
  python scripts/build_rare_instance_pool.py \
    --pseudo-dir "$LOCAL_DATA/cups_pseudo_labels_dcfa_simcf_depthpro/train" \
    --image-dir "$LOCAL_DATA/leftImg8bit/train" \
    --depth-dir "$LOCAL_DATA/depth_pro/train" \
    --centroids "$LOCAL_DATA/pseudo_semantic_raw_dinov3_k80/kmeans_centroids.npz" \
    --out "$POOL_PATH" \
    --max-per-class 3000 \
    --workers 8
fi

ssh "$REMOTE" "test -d '$REMOTE_DATA/cups_pseudo_labels_dcfa_simcf_depthpro'" || \
  $RSYNC "$LOCAL_DATA/cups_pseudo_labels_dcfa_simcf_depthpro/" \
    "$REMOTE:$REMOTE_DATA/cups_pseudo_labels_dcfa_simcf_depthpro/"

$RSYNC refs/cups/ "$REMOTE:$REMOTE_REPO/refs/cups/"
$RSYNC scripts/build_rare_instance_pool.py "$REMOTE:$REMOTE_REPO/scripts/build_rare_instance_pool.py"

if [[ "$RUN" != "A" ]]; then
  ssh "$REMOTE" "mkdir -p '$REMOTE_DATA/rare_instance_pool'"
  $RSYNC "$POOL_PATH" "$REMOTE:$REMOTE_DATA/rare_instance_pool/"
fi

CONFIG="train_cityscapes_t0_longtail_run${RUN}_santosh.yaml"
LOGDIR="$REMOTE_REPO/experiments/t0_longtail_run${RUN}"
# Match the working santosh env activation: anaconda3 (not miniconda3),
# LD_LIBRARY_PATH for cuda/cudnn, WANDB disabled because remote has no API key.
# Lightning DDPStrategy spawns child processes from the single python invocation
# so CUDA_VISIBLE_DEVICES=0,1 + Trainer(devices=2, strategy=DDPStrategy) IS
# proper DDP — no torchrun needed.
ssh "$REMOTE" "bash -lc 'cd $REMOTE_REPO/refs/cups && \
  source /home/santosh/anaconda3/etc/profile.d/conda.sh && conda activate cups && \
  export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:\${LD_LIBRARY_PATH:-} && \
  export WANDB_MODE=disabled && \
  mkdir -p $LOGDIR && \
  if [ -f $LOGDIR/run.log ]; then mv $LOGDIR/run.log $LOGDIR/run_\$(date +%Y%m%d_%H%M%S).log; fi && \
  CUDA_VISIBLE_DEVICES=0,1 nohup python -u train.py --disable_wandb \
    --config-file configs/$CONFIG \
    > $LOGDIR/run.log 2>&1 < /dev/null & \
  echo PID=\$!'"

echo "Monitor: ssh $REMOTE 'tail -f $LOGDIR/run.log'"
