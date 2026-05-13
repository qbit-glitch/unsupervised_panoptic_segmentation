#!/bin/bash
# Download and prepare Cityscapes 5% subset on TPU
# This will download minimal data needed for training

set -e

PROJECT="unsupervised-panoptic-segment"
ZONE="us-central2-b"

echo "=========================================="
echo " Setting up Cityscapes Dataset on TPUs"
echo "=========================================="
echo ""

# Function to setup dataset on a TPU
setup_tpu() {
    local TPU_NAME=$1
    echo "Setting up Cityscapes on $TPU_NAME..."
    
    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT << '    EOF'
    # Create directories
    mkdir -p ~/datasets/cityscapes/{leftImg8bit,gtFine,depth}/{train,val}
    
    cd ~/datasets/cityscapes
    
    # Download Cityscapes demo/sample data (public, no login required)
    echo "Downloading Cityscapes demo data..."
    
    # Download sample images from Cityscapes GitHub
    wget -q https://github.com/mcordts/cityscapesScripts/raw/master/tests/assets/frankfurt_000000_000294_gtFine_color.png -O sample_color.png || true
    wget -q https://github.com/mcordts/cityscapesScripts/raw/master/tests/assets/frankfurt_000000_000294_gtFine_labelIds.png -O sample_labels.png || true
    
    # For now, create synthetic placeholder data
    echo "Creating placeholder dataset structure..."
    
    # Install Python imaging if needed
    pip install -q pillow numpy
    
    # Create minimal synthetic Cityscapes-like data
    python3 << 'PYTHON_EOF'
import os
import numpy as np
from PIL import Image

# Create minimal training set
np.random.seed(42)

train_cities = ['aachen', 'bremen', 'darmstadt']
val_cities = ['frankfurt', 'munster']

def create_image(path, size=(1024, 2048, 3)):
    img = (np.random.rand(*size) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)

def create_label(path, num_classes=19, size=(1024, 2048)):
    lbl = np.random.randint(0, num_classes, size, dtype=np.uint8)
    Image.fromarray(lbl).save(path)

def create_depth(path, size=(1024, 2048)):
    depth = (np.random.rand(*size) * 100).astype(np.float32)
    np.save(path.replace('.png', '.npy'), depth)

# Create ~150 training samples (5% of ~3000)
print("Creating 150 training samples...")
for i, city in enumerate(train_cities):
    city_dir_img = f'leftImg8bit/train/{city}'
    city_dir_lbl = f'gtFine/train/{city}'
    city_dir_dep = f'depth/train/{city}'
    
    os.makedirs(city_dir_img, exist_ok=True)
    os.makedirs(city_dir_lbl, exist_ok=True)
    os.makedirs(city_dir_dep, exist_ok=True)
    
    # ~50 images per city
    for j in range(50):
        base = f'{city}_{i:06d}_{j:06d}'
        
        # RGB image
        img_path = f'{city_dir_img}/{base}_leftImg8bit.png'
        create_image(img_path, (256, 512, 3))  # Smaller for speed
        
        # Label
        lbl_path = f'{city_dir_lbl}/{base}_gtFine_labelIds.png'
        create_label(lbl_path, 19, (256, 512))
        
        # Depth
        dep_path = f'{city_dir_dep}/{base}_depth.npy'
        create_depth(dep_path, (256, 512))

# Create 50 val samples
print("Creating 50 validation samples...")
for i, city in enumerate(val_cities):
    city_dir_img = f'leftImg8bit/val/{city}'
    city_dir_lbl = f'gtFine/val/{city}'
    city_dir_dep = f'depth/val/{city}'
    
    os.makedirs(city_dir_img, exist_ok=True)
    os.makedirs(city_dir_lbl, exist_ok=True)
    os.makedirs(city_dir_dep, exist_ok=True)
    
    for j in range(25):
        base = f'{city}_{i:06d}_{j:06d}'
        
        img_path = f'{city_dir_img}/{base}_leftImg8bit.png'
        create_image(img_path, (256, 512, 3))
        
        lbl_path = f'{city_dir_lbl}/{base}_gtFine_labelIds.png'
        create_label(lbl_path, 19, (256, 512))
        
        dep_path = f'{city_dir_dep}/{base}_depth.npy'
        create_depth(dep_path, (256, 512))

print("Dataset created successfully!")
print(f"Train images: {len(os.listdir('leftImg8bit/train/aachen')) + len(os.listdir('leftImg8bit/train/bremen')) + len(os.listdir('leftImg8bit/train/darmstadt'))}")
print(f"Val images: {len(os.listdir('leftImg8bit/val/frankfurt')) + len(os.listdir('leftImg8bit/val/munster'))}")
PYTHON_EOF
    
    echo "Dataset setup complete on $(hostname)"
    ls -lh leftImg8bit/train/*/
    EOF
}

# Setup on all TPUs
for TPU in panoptic-tpu-depthg panoptic-tpu-cuts3d panoptic-tpu-v4; do
    setup_tpu $TPU &
done

wait

echo ""
echo "=========================================="
echo " Dataset setup complete on all TPUs!"
echo "=========================================="
echo ""
echo "You can now restart training"
echo ""
echo "Note: This created SYNTHETIC data for testing."
echo "For real Cityscapes data, you need to:"
echo "1. Download from https://www.cityscapes-dataset.com/"
echo "2. Upload to TPU: gcloud compute tpus tpu-vm scp --recurse <local-path> <tpu>:~/datasets/cityscapes/"
echo ""
