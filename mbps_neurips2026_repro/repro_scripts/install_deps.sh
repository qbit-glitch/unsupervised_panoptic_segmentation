#!/bin/bash
# MBPS Full Dependency Installation Script for TPU VMs
# Installs all deps needed for MBPS core + reference baselines in refs/
set -e

echo "=== MBPS Dependency Installation ==="
echo "Started at: $(date)"
echo "Python: $(python3 --version)"
echo "Pip: $(pip3 --version)"
echo ""

# Upgrade pip first
pip3 install --upgrade pip setuptools wheel 2>&1 | tail -1

# ─── 1. MBPS CORE DEPENDENCIES (not already on TPU VM) ───
echo ""
echo "=== [1/6] Installing MBPS core dependencies ==="
pip3 install --no-cache-dir \
    opencv-python-headless>=4.8.0 \
    wandb>=0.15.0 \
    panopticapi>=0.1 \
    2>&1 | tail -5

# ─── 2. PYTORCH CPU (for weight conversion & running ref baselines) ───
echo ""
echo "=== [2/6] Installing PyTorch CPU (for reference baselines) ==="
pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu \
    2>&1 | tail -5

# ─── 3. PYTORCH ECOSYSTEM (needed by refs: cups, depthg, cuts3d, mfuser, etc.) ───
echo ""
echo "=== [3/6] Installing PyTorch ecosystem packages ==="
pip3 install --no-cache-dir \
    timm>=0.9.0 \
    einops>=0.7.0 \
    kornia>=0.6.11 \
    torchmetrics>=0.11.4 \
    pytorch-lightning>=2.0.0 \
    transformers \
    huggingface-hub \
    2>&1 | tail -5

# ─── 4. COMPUTER VISION & DATA (needed by ref repos) ───
echo ""
echo "=== [4/6] Installing CV, data, and evaluation packages ==="
pip3 install --no-cache-dir \
    scikit-image>=0.21.0 \
    imageio>=2.31.0 \
    pycocotools>=2.0.6 \
    cityscapesscripts>=2.2.0 \
    networkx>=3.1 \
    pandas>=2.0.0 \
    seaborn>=0.12.0 \
    shapely>=2.0.0 \
    open3d \
    2>&1 | tail -5

# ─── 5. CONFIGURATION & UTILITIES (needed by ref repos) ───
echo ""
echo "=== [5/6] Installing config/utility packages ==="
pip3 install --no-cache-dir \
    omegaconf>=2.3.0 \
    hydra-core>=1.3.0 \
    easydict>=1.11 \
    yacs>=0.1.8 \
    colored \
    prettytable \
    ttach \
    grad-cam \
    faster-coco-eval \
    2>&1 | tail -5

# ─── 6. INSTALL MBPS PROJECT IN EDITABLE MODE ───
echo ""
echo "=== [6/6] Installing MBPS project (editable mode) ==="
cd ~/mbps_panoptic_segmentation
pip3 install -e . 2>&1 | tail -5

# ─── VERIFICATION ───
echo ""
echo "=== Verification ==="
python3 -c "
import sys
results = []
packages = {
    'jax': 'jax',
    'flax': 'flax',
    'optax': 'optax',
    'tensorflow': 'tensorflow',
    'torch': 'torch',
    'torchvision': 'torchvision',
    'timm': 'timm',
    'cv2': 'opencv',
    'einops': 'einops',
    'kornia': 'kornia',
    'wandb': 'wandb',
    'sklearn': 'scikit-learn',
    'scipy': 'scipy',
    'skimage': 'scikit-image',
    'PIL': 'Pillow',
    'yaml': 'PyYAML',
    'omegaconf': 'omegaconf',
    'pycocotools': 'pycocotools',
    'transformers': 'transformers',
    'pandas': 'pandas',
    'networkx': 'networkx',
}
for mod, name in packages.items():
    try:
        m = __import__(mod)
        v = getattr(m, '__version__', 'OK')
        results.append(f'  ✓ {name}: {v}')
    except ImportError as e:
        results.append(f'  ✗ {name}: MISSING ({e})')

# Check JAX TPU
import jax
devs = jax.devices()
tpu_count = sum(1 for d in devs if 'Tpu' in str(type(d).__name__) or 'tpu' in str(d).lower())
results.append(f'  ✓ JAX TPU devices: {tpu_count}')

for r in results:
    print(r)
print()
print(f'Total packages: {len([r for r in results if \"✓\" in r])}/{len(packages)+1} OK')
"

echo ""
echo "=== Installation complete at: $(date) ==="
