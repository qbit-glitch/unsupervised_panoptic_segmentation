#!/bin/bash
set -e
echo "=== Fixing remaining dependencies ==="

# Fix opencv - install system lib
sudo apt-get update -qq && sudo apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 > /dev/null 2>&1
echo "System libs installed"

# Reinstall opencv headless
pip3 install --no-cache-dir opencv-python-headless --force-reinstall -q 2>&1 | tail -1
echo "opencv fixed"

# Fix panopticapi - install from git
pip3 install --no-cache-dir "git+https://github.com/cocodataset/panopticapi.git" -q 2>&1 | tail -1
echo "panopticapi installed"

# Ensure wandb is there
pip3 install --no-cache-dir wandb -q 2>&1 | tail -1
echo "wandb verified"

# Reinstall MBPS in editable mode (skip deps since they're installed)
cd ~/mbps_panoptic_segmentation && pip3 install -e . --no-deps -q 2>&1 | tail -1
echo "MBPS project installed"

# Final verification
echo ""
echo "=== Final Verification ==="
python3 -c "
import cv2; print(f'  opencv: {cv2.__version__}')
import wandb; print(f'  wandb: {wandb.__version__}')
try:
    from panopticapi.utils import IdGenerator; print('  panopticapi: OK')
except: print('  panopticapi: FAILED')
import jax; print(f'  JAX devices: {len(jax.devices())} TPUs')
import torch; print(f'  torch: {torch.__version__}')
import flax; print(f'  flax: {flax.__version__}')
import timm; print(f'  timm: {timm.__version__}')
import kornia; print(f'  kornia: {kornia.__version__}')
import transformers; print(f'  transformers: {transformers.__version__}')
print('  ALL OK')
"
