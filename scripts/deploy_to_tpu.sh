#!/bin/bash
# Deploy and run baseline training on existing TPUs
# Customized for your TPU setup

set -e

PROJECT_ID="unsupervised-panoptic-segment"
ZONE="us-central2-b"

echo "=========================================="
echo " Deploying Baseline Training to TPUs"
echo "=========================================="
echo ""

# First, let's check what TPUs you have
echo "Checking existing TPUs..."
gcloud compute tpus tpu-vm list --zone=$ZONE --project=$PROJECT_ID

echo ""
echo "Which TPUs would you like to use?"
echo "Enter TPU names (space-separated) or press Enter to use default names:"
read -p "TPU names (e.g., tpu-1 tpu-2 tpu-3): " TPU_NAMES

if [ -z "$TPU_NAMES" ]; then
    # Use first 3 TPUs from the list
    TPU_NAMES=$(gcloud compute tpus tpu-vm list --zone=$ZONE --project=$PROJECT_ID --format="value(name)" | head -n 3 | tr '\n' ' ')
    echo "Using TPUs: $TPU_NAMES"
fi

# Convert to array
IFS=' ' read -r -a TPU_ARRAY <<< "$TPU_NAMES"
DEPTHG_TPU="${TPU_ARRAY[0]}"
CUTS3D_TPU="${TPU_ARRAY[1]}"
EVAL_TPU="${TPU_ARRAY[2]}"

echo ""
echo "TPU Assignment:"
echo "  DepthG training:  $DEPTHG_TPU"
echo "  CutS3D training:  $CUTS3D_TPU"
echo "  Evaluation:       $EVAL_TPU"
echo ""

# Step 1: Deploy code
echo "=========================================="
echo " Step 1: Deploying code to TPUs"
echo "=========================================="

for TPU in "${TPU_ARRAY[@]}"; do
    echo "Deploying to $TPU..."
    
    # Create directory
    gcloud compute tpus tpu-vm ssh $TPU --zone=$ZONE --project=$PROJECT_ID \
        --command="mkdir -p ~/mbps_panoptic_segmentation/logs ~/mbps_panoptic_segmentation/checkpoints" || true
    
    # Copy code
    gcloud compute tpus tpu-vm scp --recurse \
        --zone=$ZONE \
        --project=$PROJECT_ID \
        . $TPU:~/mbps_panoptic_segmentation/
    
    echo "✓ $TPU ready"
done

echo ""
echo "=========================================="
echo " Step 2: Installing dependencies"
echo "=========================================="

for TPU in "${TPU_ARRAY[@]}"; do
    echo "Installing on $TPU..."
    
    gcloud compute tpus tpu-vm ssh $TPU --zone=$ZONE --project=$PROJECT_ID << 'EOF'
# Check if JAX is already installed
if python3 -c "import jax" 2>/dev/null; then
    echo "JAX already installed"
    python3 -c "import jax; print(f'TPU devices: {jax.device_count()}')"
else
    echo "Installing JAX and dependencies..."
    pip install --upgrade pip
    pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install flax optax orbax-checkpoint tensorflow tensorflow-datasets
    pip install einops pyyaml absl-py ml-collections scikit-learn scipy pillow matplotlib
    python3 -c "import jax; print(f'TPU devices: {jax.device_count()}')"
fi
EOF
    
    echo "✓ $TPU ready"
done

echo ""
echo "=========================================="
echo " Step 3: Launching training jobs"
echo "=========================================="

# Launch DepthG
echo "Launching DepthG on $DEPTHG_TPU..."
gcloud compute tpus tpu-vm ssh $DEPTHG_TPU --zone=$ZONE --project=$PROJECT_ID << 'EOF'
cd ~/mbps_panoptic_segmentation
nohup python3 scripts/train_depthg.py \
    --config configs/cityscapes_5pct.yaml \
    --epochs 30 \
    --output checkpoints/depthg_baseline.npz \
    > logs/depthg_train.log 2>&1 &
echo "DepthG started (PID: $!)"
EOF

# Launch CutS3D
echo "Launching CutS3D on $CUTS3D_TPU..."
gcloud compute tpus tpu-vm ssh $CUTS3D_TPU --zone=$ZONE --project=$PROJECT_ID << 'EOF'
cd ~/mbps_panoptic_segmentation
nohup python3 scripts/train_cuts3d.py \
    --config configs/cityscapes_5pct.yaml \
    --epochs 30 \
    --output checkpoints/cuts3d_baseline.npz \
    > logs/cuts3d_train.log 2>&1 &
echo "CutS3D started (PID: $!)"
EOF

echo ""
echo "=========================================="
echo " ✓ Training jobs launched!"
echo "=========================================="
echo ""
echo "Monitor progress:"
echo "  DepthG:  gcloud compute tpus tpu-vm ssh $DEPTHG_TPU --zone=$ZONE --project=$PROJECT_ID --command='tail -f ~/mbps_panoptic_segmentation/logs/depthg_train.log'"
echo "  CutS3D:  gcloud compute tpus tpu-vm ssh $CUTS3D_TPU --zone=$ZONE --project=$PROJECT_ID --command='tail -f ~/mbps_panoptic_segmentation/logs/cuts3d_train.log'"
echo ""
echo "Check job status:"
echo "  gcloud compute tpus tpu-vm ssh $DEPTHG_TPU --zone=$ZONE --project=$PROJECT_ID --command='ps aux | grep python'"
echo ""
