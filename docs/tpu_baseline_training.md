# Baseline Model Training - TPU Setup Guide

## Quick Start

### 1. Create TPU VMs

You have access to **32 v4 chips in us-central2-b**. Create 3 TPU VMs (v4-8 each, using 24/32 chips):

```bash
# DepthG Training TPU
gcloud compute tpus tpu-vm create tpu-depthg \
    --zone=us-central2-b \
    --accelerator-type=v4-8 \
    --version=tpu-ubuntu2204-base

# CutS3D Training TPU  
gcloud compute tpus tpu-vm create tpu-cuts3d \
    --zone=us-central2-b \
    --accelerator-type=v4-8 \
    --version=tpu-ubuntu2204-base

# Evaluation TPU (optional, for naive panoptic)
gcloud compute tpus tpu-vm create tpu-eval \
    --zone=us-central2-b \
    --accelerator-type=v4-8 \
    --version=tpu-ubuntu2204-base
```

### 2. Install Dependencies on Each TPU

SSH into each TPU and install dependencies:

```bash
# For each TPU (depthg, cuts3d, eval)
gcloud compute tpus tpu-vm ssh tpu-depthg --zone=us-central2-b

# On the TPU:
pip install --upgrade pip
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax optax orbax-checkpoint tensorflow tensorflow-datasets
pip install einops pyyaml absl-py ml-collections
pip install scikit-learn scipy pillow matplotlib
```

### 3. Deploy Code to TPUs

From your local machine:

```bash
# Create archive
tar -czf mbps.tar.gz mbps_panoptic_segmentation/

# Copy to all TPUs
for TPU in tpu-depthg tpu-cuts3d tpu-eval; do
    gcloud compute tpus tpu-vm scp mbps.tar.gz $TPU:~/ --zone=us-central2-b
    gcloud compute tpus tpu-vm ssh $TPU --zone=us-central2-b \
        --command="tar -xzf mbps.tar.gz"
done
```

### 4. Launch Parallel Training

```bash
# From your local machine
bash scripts/launch_baseline_training.sh
```

This will:
- Start DepthG training on `tpu-depthg` (30 epochs, ~2 hours)
- Start CutS3D training on `tpu-cuts3d` (30 epochs, ~2-3 hours)
- Both run in background with logs saved

### 5. Monitor Progress

```bash
# Watch DepthG training
gcloud compute tpus tpu-vm ssh tpu-depthg --zone=us-central2-b \
    --command="tail -f mbps_panoptic_segmentation/logs/depthg_train.log"

# Watch CutS3D training
gcloud compute tpus tpu-vm ssh tpu-cuts3d --zone=us-central2-b \
    --command="tail -f mbps_panoptic_segmentation/logs/cuts3d_train.log"
```

### 6. Fetch Checkpoints

After training completes (~2-3 hours):

```bash
# Download checkpoints
gcloud compute tpus tpu-vm scp \
    tpu-depthg:~/mbps_panoptic_segmentation/checkpoints/depthg_baseline.npz \
    ./checkpoints/ --zone=us-central2-b

gcloud compute tpus tpu-vm scp \
    tpu-cuts3d:~/mbps_panoptic_segmentation/checkpoints/cuts3d_baseline.npz \
    ./checkpoints/ --zone=us-central2-b
```

### 7. Run Naive Panoptic Baseline

```bash
python scripts/evaluate_naive_panoptic.py \
    --config configs/cityscapes_5pct.yaml \
    --depthg-checkpoint checkpoints/depthg_baseline.npz \
    --cuts3d-checkpoint checkpoints/cuts3d_baseline.npz \
    --output results/naive_panoptic_cityscapes5pct.json
```

## Expected Results (5% Cityscapes)

Since we're training on only 5% of Cityscapes (149 train images), expect **lower performance** than paper baselines:

| Model | Metric | Expected (5% data) | Paper Baseline (full data) |
|-------|--------|-------------------|-----------------------------|
| DepthG | mIoU | 25-35% | 44.8% (COCO-Stuff-27) |
| CutS3D | AP | 5-8% | 10.7% (COCO val2017) |
| Naive Panoptic | PQ | 10-15% | 18-22% (Cityscapes) |

These are reasonable "reproduction" targets for prototyping before scaling to full datasets.

## Manual Training (Alternative)

If the launch script doesn't work, train manually:

```bash
# On tpu-depthg
python scripts/train_depthg.py \
    --config configs/cityscapes_5pct.yaml \
    --epochs 30 \
    --output checkpoints/depthg_baseline.npz

# On tpu-cuts3d
python scripts/train_cuts3d.py \
    --config configs/cityscapes_5pct.yaml \
    --epochs 30 \
    --output checkpoints/cuts3d_baseline.npz
```

## Cleanup

When done, delete TPUs to avoid charges:

```bash
gcloud compute tpus tpu-vm delete tpu-depthg --zone=us-central2-b --quiet
gcloud compute tpus tpu-vm delete tpu-cuts3d --zone=us-central2-b --quiet
gcloud compute tpus tpu-vm delete tpu-eval --zone=us-central2-b --quiet
```

## Troubleshooting

### JAX not finding TPUs
```bash
# Check TPU visibility
python -c "import jax; print(jax.devices())"
# Should show 8 TPU devices
```

### Out of memory
- Reduce batch size in `configs/cityscapes_5pct.yaml` (currently 2)
- Reduce image size from [256, 512] to [128, 256]

### Dataset not found
- Ensure Cityscapes data is accessible on TPU
- Check paths in config: `data_dir`, `depth_dir`
- You may need to copy dataset to GCS bucket and mount it

## Next Steps

After reproducing baselines:
1. Compare results with paper baselines
2. Train full MBPS model with Mamba bridge
3. Run ablation studies
4. Scale to full Cityscapes dataset
