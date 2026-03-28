#!/bin/bash
# MBPS Dependency Installation for TPU v6e (Trillium) VMs
# v6e requires newer JAX (>=0.4.30) with v6e-specific libtpu
set -e

echo "=== MBPS v6e Dependency Installation ==="
echo "Started at: $(date)"
echo "Python: $(python3 --version)"
echo ""

# Upgrade pip
pip3 install --upgrade pip setuptools wheel 2>&1 | tail -1

# ─── 1. JAX for v6e (must install BEFORE other deps to avoid version conflicts) ───
echo ""
echo "=== [1/7] Installing JAX for TPU v6e ==="
pip3 install --no-cache-dir \
    "jax[tpu]>=0.4.30" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    2>&1 | tail -5

# ─── 2. Flax/Optax/Orbax (match JAX version) ───
echo ""
echo "=== [2/7] Installing Flax, Optax, Orbax ==="
pip3 install --no-cache-dir \
    "flax>=0.8.0" \
    "optax>=0.2.0" \
    "orbax-checkpoint>=0.5.0" \
    2>&1 | tail -5

# ─── 3. TensorFlow (for tf.io.gfile GCS access + TFRecord pipeline) ───
echo ""
echo "=== [3/7] Installing TensorFlow ==="
pip3 install --no-cache-dir \
    "tensorflow>=2.15.0" \
    2>&1 | tail -5

# ─── 4. MBPS core dependencies ───
echo ""
echo "=== [4/7] Installing MBPS core dependencies ==="
pip3 install --no-cache-dir \
    opencv-python-headless>=4.8.0 \
    "wandb>=0.15.0" \
    "panopticapi>=0.1" \
    "einops>=0.7.0" \
    2>&1 | tail -5

# ─── 5. PyTorch CPU (for weight conversion only) ───
echo ""
echo "=== [5/7] Installing PyTorch CPU ==="
pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu \
    2>&1 | tail -5

# ─── 6. CV, data, and config packages ───
echo ""
echo "=== [6/7] Installing CV/data/config packages ==="
pip3 install --no-cache-dir \
    "timm>=0.9.0" \
    "kornia>=0.6.11" \
    "scikit-image>=0.21.0" \
    "pycocotools>=2.0.6" \
    "cityscapesscripts>=2.2.0" \
    "omegaconf>=2.3.0" \
    "hydra-core>=1.3.0" \
    "pandas>=2.0.0" \
    "networkx>=3.1" \
    "transformers" \
    "huggingface-hub" \
    2>&1 | tail -5

# ─── 7. Install MBPS project ───
echo ""
echo "=== [7/7] Installing MBPS project (editable) ==="
cd ~/mbps_panoptic_segmentation
pip3 install -e . 2>&1 | tail -5

# ─── VERIFICATION ───
echo ""
echo "=== Verification ==="
python3 -c "
import sys

# Check JAX + TPU
import jax
print(f'  JAX version: {jax.__version__}')
devs = jax.devices()
tpu_count = len(devs)
print(f'  JAX devices: {tpu_count} ({devs[0].platform if devs else \"none\"})')

# Check key packages
for pkg in ['flax', 'optax', 'tensorflow', 'torch', 'cv2', 'wandb', 'einops']:
    try:
        m = __import__(pkg)
        v = getattr(m, '__version__', 'OK')
        print(f'  {pkg}: {v}')
    except ImportError as e:
        print(f'  {pkg}: MISSING ({e})')
        sys.exit(1)

print()
if tpu_count >= 4:
    print(f'All checks PASSED ({tpu_count} TPU devices)')
else:
    print(f'WARNING: Only {tpu_count} TPU devices found')
"

echo ""
echo "=== Installation complete at: $(date) ==="
