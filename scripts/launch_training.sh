#!/bin/bash
# Simple script to launch training on both TPUs after dependencies are installed

PROJECT_ID="unsupervised-panoptic-segment"
ZONE="us-central2-b"

echo "=========================================="
echo " Launching Training Jobs"
echo "=========================================="

# Launch DepthG training
echo "Starting DepthG training on panoptic-tpu-depthg..."
gcloud compute tpus tpu-vm ssh panoptic-tpu-depthg --zone=$ZONE --project=$PROJECT_ID << 'EOF'
cd ~/mbps
export PATH="$HOME/.local/bin:$PATH"
nohup python3 scripts/train_depthg.py \
    --config configs/cityscapes_5pct.yaml \
    --epochs 30 \
    --output checkpoints/depthg_baseline.npz \
    > logs/depthg_train.log 2>&1 &
echo "DepthG training started (PID: $!)"
EOF

echo ""

# Launch CutS3D training
echo "Starting CutS3D training on panoptic-tpu-cuts3d..."
gcloud compute tpus tpu-vm ssh panoptic-tpu-cuts3d --zone=$ZONE --project=$PROJECT_ID << 'EOF'
cd ~/mbps
export PATH="$HOME/.local/bin:$PATH"
nohup python3 scripts/train_cuts3d.py \
    --config configs/cityscapes_5pct.yaml \
    --epochs 30 \
    --output checkpoints/cuts3d_baseline.npz \
    > logs/cuts3d_train.log 2>&1 &
echo "CutS3D training started (PID: $!)"
EOF

echo ""
echo "=========================================="
echo " Both training jobs launched!"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  gcloud compute tpus tpu-vm ssh panoptic-tpu-depthg --zone=$ZONE --project=$PROJECT_ID --command='tail -f ~/mbps/logs/depthg_train.log'"
echo "  gcloud compute tpus tpu-vm ssh panoptic-tpu-cuts3d --zone=$ZONE --project=$PROJECT_ID --command='tail -f ~/mbps/logs/cuts3d_train.log'"
