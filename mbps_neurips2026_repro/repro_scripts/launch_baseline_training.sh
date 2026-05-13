#!/bin/bash
# Launch all baseline training jobs in parallel on TPUs
#
# Usage on GCP:
#   1. Create 3 TPU VMs (v4-8 each):
#      - tpu-depthg     (for DepthG training)
#      - tpu-cuts3d     (for CutS3D training)
#      - tpu-eval       (for evaluation + naive panoptic)
#
#   2. Deploy code to all TPUs:
#      gcloud compute tpus tpu-vm scp --recurse \
#          mbps_panoptic_segmentation/ tpu-depthg:~/ \
#          --zone=us-central2-b
#
#   3. Run this script:
#      bash scripts/launch_baseline_training.sh
#
# The script will:
# - SSH into each TPU
# - Start training in background with nohup
# - Save logs to logs/
#

set -e

CONFIG="configs/cityscapes_5pct.yaml"
ZONE="us-central2-b"
PROJECT="unsupervised-panoptic-segment"

echo "=========================================="
echo " Launching Baseline Training on TPUs"
echo "=========================================="
echo ""

# TPU hostnames
DEPTHG_TPU="tpu-depthg"
CUTS3D_TPU="tpu-cuts3d"
EVAL_TPU="tpu-eval"

# Training commands
DEPTHG_CMD="cd mbps_panoptic_segmentation && nohup python scripts/train_depthg.py --config $CONFIG --epochs 30 --output checkpoints/depthg_baseline.npz > logs/depthg_train.log 2>&1 &"

CUTS3D_CMD="cd mbps_panoptic_segmentation && nohup python scripts/train_cuts3d.py --config $CONFIG --epochs 30 --output checkpoints/cuts3d_baseline.npz > logs/cuts3d_train.log 2>&1 &"

# Function to launch training on a TPU
launch_training() {
    local tpu_name=$1
    local command=$2
    local job_name=$3
    
    echo "[$job_name] Launching on $tpu_name..."
    
    gcloud compute tpus tpu-vm ssh $tpu_name \
        --zone=$ZONE \
        --project=$PROJECT \
        --command="$command"
    
    if [ $? -eq 0 ]; then
        echo "[$job_name] ✓ Started successfully"
    else
        echo "[$job_name] ✗ Failed to start"
        return 1
    fi
    echo ""
}

# Launch DepthG training
launch_training $DEPTHG_TPU "$DEPTHG_CMD" "DepthG"

# Launch CutS3D training
launch_training $CUTS3D_TPU "$CUTS3D_CMD" "CutS3D"

echo "=========================================="
echo " All training jobs launched!"
echo "=========================================="
echo ""
echo "Monitor progress:"
echo "  DepthG:  gcloud compute tpus tpu-vm ssh $DEPTHG_TPU --zone=$ZONE --command='tail -f mbps_panoptic_segmentation/logs/depthg_train.log'"
echo "  CutS3D:  gcloud compute tpus tpu-vm ssh $CUTS3D_TPU --zone=$ZONE --command='tail -f mbps_panoptic_segmentation/logs/cuts3d_train.log'"
echo ""
echo "Fetch checkpoints when complete:"
echo "  gcloud compute tpus tpu-vm scp $DEPTHG_TPU:~/mbps_panoptic_segmentation/checkpoints/depthg_baseline.npz ./checkpoints/ --zone=$ZONE"
echo "  gcloud compute tpus tpu-vm scp $CUTS3D_TPU:~/mbps_panoptic_segmentation/checkpoints/cuts3d_baseline.npz ./checkpoints/ --zone=$ZONE"
echo ""
echo "Run naive panoptic evaluation (after training completes):"
echo "  python scripts/evaluate_naive_panoptic.py \\"
echo "      --config $CONFIG \\"
echo "      --depthg-checkpoint checkpoints/depthg_baseline.npz \\"
echo "      --cuts3d-checkpoint checkpoints/cuts3d_baseline.npz"
echo ""
