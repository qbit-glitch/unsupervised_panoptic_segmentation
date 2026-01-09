#!/bin/bash
# =============================================================================
# SpectralDiffusion Setup & Training Script for Multi-GPU
# Target: Linux machine with 2x NVIDIA RTX 1080 Ti
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "SpectralDiffusion Multi-GPU Setup Script"
echo "=============================================="

# Configuration - MODIFY THESE PATHS
REPO_URL="https://github.com/qbit-glitch/unsupervised_panoptic_segmentation.git"
PROJECT_DIR="$HOME/unsupervised_panoptic_segmentation"
DATASETS_DIR="$HOME/datasets"
CLEVR_DIR="$DATASETS_DIR/clevr/CLEVR_v1.0"

# =============================================================================
# Step 1: Clone/Update Repository
# =============================================================================
echo ""
echo "[1/6] Setting up repository..."

if [ -d "$PROJECT_DIR" ]; then
    echo "Repository exists, pulling latest..."
    cd "$PROJECT_DIR"
    git pull origin main
else
    echo "Cloning repository..."
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# =============================================================================
# Step 2: Create Virtual Environment
# =============================================================================
echo ""
echo "[2/6] Setting up Python environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Created virtual environment"
fi

source venv/bin/activate
echo "Activated virtual environment: $(which python)"

# Upgrade pip
pip install --upgrade pip

# =============================================================================
# Step 3: Install Dependencies
# =============================================================================
echo ""
echo "[3/6] Installing dependencies..."

# PyTorch with CUDA 11.8 (compatible with GTX 1080 Ti)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Core dependencies (skip wandb to avoid build issues)
pip install numpy scipy scikit-learn opencv-python pillow tqdm pyyaml einops timm transformers

echo "Dependencies installed successfully"

# =============================================================================
# Step 4: Verify GPU Setup
# =============================================================================
echo ""
echo "[4/6] Verifying GPU setup..."

python << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
EOF

# =============================================================================
# Step 5: Generate CLEVR Masks (if needed)
# =============================================================================
echo ""
echo "[5/6] Checking CLEVR dataset and masks..."

if [ -d "$CLEVR_DIR" ]; then
    echo "CLEVR dataset found at: $CLEVR_DIR"
    
    # Check if masks already exist
    if [ -d "$CLEVR_DIR/masks" ]; then
        echo "Masks directory already exists"
    else
        echo "Generating masks for CLEVR dataset..."
        python generate_clevr_masks.py --clevr-dir "$CLEVR_DIR" --split train
        python generate_clevr_masks.py --clevr-dir "$CLEVR_DIR" --split val
        echo "Masks generated successfully"
    fi
else
    echo "WARNING: CLEVR dataset not found at $CLEVR_DIR"
    echo "Please download and extract CLEVR dataset first"
    echo "Expected structure: $CLEVR_DIR/images/{train,val}/"
fi

# =============================================================================
# Step 6: Print Training Commands
# =============================================================================
echo ""
echo "[6/6] Setup complete!"
echo ""
echo "=============================================="
echo "TRAINING COMMANDS"
echo "=============================================="
echo ""
echo "# Quick test (small subset):"
echo "python train.py \\"
echo "    --dataset clevr \\"
echo "    --data-dir $DATASETS_DIR \\"
echo "    --image-size 128 128 \\"
echo "    --num-slots 11 \\"
echo "    --batch-size 32 \\"
echo "    --epochs 5 \\"
echo "    --device cuda \\"
echo "    --multi-gpu \\"
echo "    --gpu-ids 0,1 \\"
echo "    --num-workers 8 \\"
echo "    --subset-percent 0.1"
echo ""
echo "# Full training (recommended):"
echo "python train.py \\"
echo "    --dataset clevr \\"
echo "    --data-dir $DATASETS_DIR \\"
echo "    --image-size 240 320 \\"
echo "    --num-slots 11 \\"
echo "    --batch-size 64 \\"
echo "    --epochs 50 \\"
echo "    --lr 2e-4 \\"
echo "    --device cuda \\"
echo "    --multi-gpu \\"
echo "    --gpu-ids 0,1 \\"
echo "    --num-workers 8 \\"
echo "    --output-dir ./outputs"
echo ""
echo "=============================================="
echo "To start training, run one of the commands above"
echo "=============================================="
