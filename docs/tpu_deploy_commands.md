# TPU Deployment Commands - Quick Reference

## Step 1: Create 3 TPU VMs (v4-8 each)

```bash
# Set your project ID
export PROJECT_ID="unsupervised-panoptic-segment"
export ZONE="us-central2-b"

# Create DepthG Training TPU
gcloud compute tpus tpu-vm create tpu-depthg \
    --zone=$ZONE \
    --accelerator-type=v4-8 \
    --version=tpu-ubuntu2204-base \
    --project=$PROJECT_ID

# Create CutS3D Training TPU
gcloud compute tpus tpu-vm create tpu-cuts3d \
    --zone=$ZONE \
    --accelerator-type=v4-8 \
    --version=tpu-ubuntu2204-base \
    --project=$PROJECT_ID

# Create Evaluation TPU
gcloud compute tpus tpu-vm create tpu-eval \
    --zone=$ZONE \
    --accelerator-type=v4-8 \
    --version=tpu-ubuntu2204-base \
    --project=$PROJECT_ID
```

## Step 2: Install Dependencies on All TPUs

```bash
# Function to install on a TPU
install_deps() {
    local TPU=$1
    echo "Installing dependencies on $TPU..."
    
    gcloud compute tpus tpu-vm ssh $TPU --zone=$ZONE --project=$PROJECT_ID << 'EOF'
# Update pip
pip install --upgrade pip

# Install JAX with TPU support
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install ML dependencies
pip install flax optax orbax-checkpoint tensorflow tensorflow-datasets

# Install utilities
pip install einops pyyaml absl-py ml-collections scikit-learn scipy pillow matplotlib

# Verify TPU visibility
python3 -c "import jax; print(f'TPU devices: {jax.device_count()}')"
EOF
}

# Install on all TPUs
install_deps tpu-depthg
install_deps tpu-cuts3d
install_deps tpu-eval
```

## Step 3: Deploy Code to All TPUs

```bash
# From your local machine (in project root)
cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation

# Deploy to all TPUs
for TPU in tpu-depthg tpu-cuts3d tpu-eval; do
    echo "Deploying code to $TPU..."
    
    gcloud compute tpus tpu-vm scp --recurse \
        --zone=$ZONE \
        --project=$PROJECT_ID \
        . $TPU:~/mbps_panoptic_segmentation/
    
    # Create logs directory
    gcloud compute tpus tpu-vm ssh $TPU --zone=$ZONE --project=$PROJECT_ID \
        --command="mkdir -p ~/mbps_panoptic_segmentation/logs ~/mbps_panoptic_segmentation/checkpoints"
done
```

## Step 4: Launch Training Jobs

```bash
# Launch DepthG training
echo "Launching DepthG training..."
gcloud compute tpus tpu-vm ssh tpu-depthg --zone=$ZONE --project=$PROJECT_ID << 'EOF'
cd ~/mbps_panoptic_segmentation
nohup python scripts/train_depthg.py \
    --config configs/cityscapes_5pct.yaml \
    --epochs 30 \
    --output checkpoints/depthg_baseline.npz \
    > logs/depthg_train.log 2>&1 &
echo "DepthG training started (PID: $!)"
EOF

# Launch CutS3D training
echo "Launching CutS3D training..."
gcloud compute tpus tpu-vm ssh tpu-cuts3d --zone=$ZONE --project=$PROJECT_ID << 'EOF'
cd ~/mbps_panoptic_segmentation
nohup python scripts/train_cuts3d.py \
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
```

## Step 5: Monitor Training Progress

```bash
# Watch DepthG training (in terminal 1)
gcloud compute tpus tpu-vm ssh tpu-depthg --zone=$ZONE --project=$PROJECT_ID \
    --command="tail -f ~/mbps_panoptic_segmentation/logs/depthg_train.log"

# Watch CutS3D training (in terminal 2)
gcloud compute tpus tpu-vm ssh tpu-cuts3d --zone=$ZONE --project=$PROJECT_ID \
    --command="tail -f ~/mbps_panoptic_segmentation/logs/cuts3d_train.log"

# Check if jobs are still running
gcloud compute tpus tpu-vm ssh tpu-depthg --zone=$ZONE --project=$PROJECT_ID \
    --command="ps aux | grep python | grep train"
```

## Step 6: Fetch Checkpoints (After Training Completes)

Training will take approximately 2-3 hours. Once complete:

```bash
# Fetch DepthG checkpoint
gcloud compute tpus tpu-vm scp \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    tpu-depthg:~/mbps_panoptic_segmentation/checkpoints/depthg_baseline.npz \
    ./checkpoints/

# Fetch CutS3D checkpoint
gcloud compute tpus tpu-vm scp \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    tpu-cuts3d:~/mbps_panoptic_segmentation/checkpoints/cuts3d_baseline.npz \
    ./checkpoints/

# Verify checkspoints
ls -lh checkpoints/*.npz
```

## Step 7: Run Naive Panoptic Evaluation

```bash
# Run locally (if you have the dataset and checkpoints)
python scripts/evaluate_naive_panoptic.py \
    --config configs/cityscapes_5pct.yaml \
    --depthg-checkpoint checkpoints/depthg_baseline.npz \
    --cuts3d-checkpoint checkpoints/cuts3d_baseline.npz \
    --output results/naive_panoptic_cityscapes5pct.json

# OR run on tpu-eval
gcloud compute tpus tpu-vm ssh tpu-eval --zone=$ZONE --project=$PROJECT_ID << 'EOF'
cd ~/mbps_panoptic_segmentation
python scripts/evaluate_naive_panoptic.py \
    --config configs/cityscapes_5pct.yaml \
    --depthg-checkpoint checkpoints/depthg_baseline.npz \
    --cuts3d-checkpoint checkpoints/cuts3d_baseline.npz \
    --output results/naive_panoptic_cityscapes5pct.json
EOF

# Fetch results
gcloud compute tpus tpu-vm scp \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    tpu-eval:~/mbps_panoptic_segmentation/results/naive_panoptic_cityscapes5pct.json \
    ./results/
```

## Step 8: View Results

```bash
# View JSON results
cat results/naive_panoptic_cityscapes5pct.json

# Expected output:
# {
#   "PQ": 12.34,        # Target: 10-15%
#   "PQ_Th": 9.87,      # Target: 8-12%
#   "PQ_St": 14.56,     # Target: 12-18%
#   "SQ": 68.23,
#   "RQ": 18.76,
#   "mIou": 28.45,      # Target: 25-35%
#   "n_samples": 50
# }
```

## Step 9: Cleanup (When Done)

```bash
# Delete TPUs to avoid charges
gcloud compute tpus tpu-vm delete tpu-depthg --zone=$ZONE --project=$PROJECT_ID --quiet
gcloud compute tpus tpu-vm delete tpu-cuts3d --zone=$ZONE --project=$PROJECT_ID --quiet
gcloud compute tpus tpu-vm delete tpu-eval --zone=$ZONE --project=$PROJECT_ID --quiet

echo "All TPUs deleted. No further charges will be incurred."
```

## Troubleshooting

### Issue: JAX not finding TPUs

```bash
# SSH into TPU and check devices
gcloud compute tpus tpu-vm ssh tpu-depthg --zone=$ZONE --project=$PROJECT_ID

# On the TPU:
python3 -c "import jax; print(jax.devices())"
# Should show: [TpuDevice(id=0), TpuDevice(id=1), ..., TpuDevice(id=7)]
```

### Issue: Training not starting

```bash
# Check if Python process is running
gcloud compute tpus tpu-vm ssh tpu-depthg --zone=$ZONE --project=$PROJECT_ID \
    --command="ps aux | grep python"

# Check log for errors
gcloud compute tpus tpu-vm ssh tpu-depthg --zone=$ZONE --project=$PROJECT_ID \
    --command="cat ~/mbps_panoptic_segmentation/logs/depthg_train.log | tail -n 50"
```

### Issue: Out of memory

```bash
# Reduce batch size in config
# Edit configs/cityscapes_5pct.yaml:
#   batch_size: 1  # Reduce from 2 to 1
```

## Expected Timeline

- **Step 1-3:** TPU creation and setup (~10-15 minutes)
- **Step 4-5:** Training (~2-3 hours total)
  - DepthG: ~2 hours for 30 epochs
  - CutS3D: ~2-3 hours for 30 epochs (runs in parallel)
- **Step 6-7:** Evaluation (~5 minutes)
- **Total:** ~2.5-3.5 hours

## Cost Estimate

With TRC quota, TPUs are **free for 30 days** in us-central2-b. You only pay for:
- GCS storage (minimal)
- Network egress (minimal for checkpoints <100MB)

**Total estimated cost:** <$5 USD (likely covered by $300 free credit)
