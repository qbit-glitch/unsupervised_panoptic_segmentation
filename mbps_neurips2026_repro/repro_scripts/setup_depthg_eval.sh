#!/bin/bash
# DepthG Evaluation Setup & Run Script for TPU VM (CPU-based PyTorch inference)
set -e

echo "============================================="
echo "  DepthG Evaluation Setup on TPU VM (CPU)"
echo "============================================="
echo ""

PROJ_DIR=~/mbps_panoptic_segmentation
DEPTHG_DIR=$PROJ_DIR/refs/depthg
DATA_DIR=$PROJ_DIR/datasets
CKPT_DIR=$PROJ_DIR/saved_models

# ─── 1. DIRECTORY SETUP ───
echo "=== [1/5] Setting up directories ==="
mkdir -p $DATA_DIR
mkdir -p $CKPT_DIR
mkdir -p $PROJ_DIR/results/depthg_eval

# Create cityscapes symlink in datasets dir (DepthG expects data_dir/cityscapes/)
# Use the proxy data if no full cityscapes available
if [ -d "$PROJ_DIR/data/cityscapes_proxy" ]; then
    echo "  Found cityscapes_proxy data, linking..."
    ln -sfn $PROJ_DIR/data/cityscapes_proxy $DATA_DIR/cityscapes
elif [ -d "$PROJ_DIR/data/cityscapes" ]; then
    echo "  Found full cityscapes data, linking..."
    ln -sfn $PROJ_DIR/data/cityscapes $DATA_DIR/cityscapes
else
    echo "  ERROR: No Cityscapes data found!"
    exit 1
fi

echo "  Cityscapes data at: $(readlink -f $DATA_DIR/cityscapes)"
echo "  Val images: $(find $DATA_DIR/cityscapes/leftImg8bit/val -name '*.png' 2>/dev/null | wc -l)"
echo "  Val labels: $(find $DATA_DIR/cityscapes/gtFine/val -name '*labelIds.png' 2>/dev/null | wc -l)"
echo ""

# ─── 2. DOWNLOAD CHECKPOINTS ───
echo "=== [2/5] Downloading checkpoints ==="

# Download STEGO-base Cityscapes checkpoint from Azure (known working URL)
STEGO_CKPT="$CKPT_DIR/cityscapes_vit_base_1.ckpt"
if [ ! -f "$STEGO_CKPT" ]; then
    echo "  Downloading cityscapes_vit_base_1.ckpt from Azure..."
    wget -q --show-progress -O "$STEGO_CKPT" \
        "https://marhamilresearch4.blob.core.windows.net/stego-public/saved_models/cityscapes_vit_base_1.ckpt"
    echo "  Downloaded: $(ls -lh $STEGO_CKPT | awk '{print $5}')"
else
    echo "  Checkpoint already exists: $STEGO_CKPT"
fi

# Try downloading DepthG-specific checkpoint via gdown (Google Drive)
pip3 install -q gdown 2>/dev/null
DEPTHG_CKPT="$CKPT_DIR/cityscapes_vitb_depthg.ckpt"
if [ ! -f "$DEPTHG_CKPT" ]; then
    echo "  Attempting DepthG checkpoint download from Google Drive..."
    # The Google Drive folder: 1vaSsTbpObcWygw1NJ8INltiM2PuKyPYj
    gdown --folder "1vaSsTbpObcWygw1NJ8INltiM2PuKyPYj" -O "$CKPT_DIR/" --remaining-ok 2>&1 | tail -5 || \
        echo "  Warning: gdown failed (may need manual download). Falling back to STEGO checkpoint."
fi

# Pick best available checkpoint
if [ -f "$DEPTHG_CKPT" ]; then
    EVAL_CKPT="$DEPTHG_CKPT"
    echo "  Using DepthG checkpoint: $EVAL_CKPT"
else
    EVAL_CKPT="$STEGO_CKPT"
    echo "  Using STEGO checkpoint: $EVAL_CKPT"
fi
echo ""

# ─── 3. INSTALL ADDITIONAL DEPS ───
echo "=== [3/5] Installing additional DepthG dependencies ==="
pip3 install -q pydensecrf 2>/dev/null || pip3 install -q cython && pip3 install -q pydensecrf 2>/dev/null || echo "  pydensecrf install failed (CRF will be disabled)"
echo ""

# ─── 4. CREATE CPU-COMPATIBLE EVAL SCRIPT ───
echo "=== [4/5] Creating CPU-compatible evaluation script ==="
cat > $PROJ_DIR/scripts/eval_depthg_cpu.py << 'PYEOF'
"""
DepthG/STEGO evaluation script - CPU-compatible version for TPU VMs.
Patches .cuda() calls to work on CPU when no GPU is available.
"""
import os
import sys
import warnings

# Add DepthG src to path
DEPTHG_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "refs", "depthg", "src")
sys.path.insert(0, DEPTHG_SRC)

# ─── CPU COMPATIBILITY PATCHES ───
import torch

# Determine device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

if not torch.cuda.is_available():
    print("No CUDA available - patching for CPU execution...")

    # Patch torch.Tensor.cuda to be a no-op on CPU
    _original_cuda = torch.Tensor.cuda
    def _cpu_cuda(self, *args, **kwargs):
        return self
    torch.Tensor.cuda = _cpu_cuda

    # Patch nn.Module.cuda to be a no-op
    _original_module_cuda = torch.nn.Module.cuda
    def _cpu_module_cuda(self, *args, **kwargs):
        return self
    torch.nn.Module.cuda = _cpu_module_cuda

    # Patch torch.load to always map to CPU
    _original_load = torch.load
    def _cpu_load(*args, **kwargs):
        kwargs['map_location'] = 'cpu'
        return _original_load(*args, **kwargs)
    torch.load = _cpu_load

# ─── Now import DepthG modules ───
from modules import *
from data import *
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from train_segmentation import LitUnsupervisedSegmenter, prep_for_plot, get_class_labels

try:
    from crf import dense_crf
    HAS_CRF = True
except ImportError:
    HAS_CRF = False
    print("Warning: pydensecrf not available, disabling CRF post-processing")

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless server
import matplotlib.pyplot as plt
import seaborn as sns


def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def batched_crf(img_tensor, prob_tensor):
    if not HAS_CRF:
        return prob_tensor
    outputs = []
    for img, prob in zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()):
        outputs.append(torch.from_numpy(dense_crf(img, prob)).unsqueeze(0))
    return torch.cat(outputs, dim=0)


def run_eval(data_dir, model_path, output_dir, batch_size=4, num_workers=2, res=320, run_crf=True):
    """Run DepthG/STEGO evaluation on Cityscapes."""

    print(f"\n{'='*60}")
    print(f"  DepthG Evaluation")
    print(f"  Model: {model_path}")
    print(f"  Data:  {data_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {DEVICE}")
    print(f"  CRF: {run_crf and HAS_CRF}")
    print(f"{'='*60}\n")

    os.makedirs(os.path.join(output_dir, "img"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "label"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cluster"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "linear"), exist_ok=True)

    # Load model
    print("Loading model checkpoint...")
    model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)
    model_cfg = model.cfg
    print(f"  Model config: dataset={model_cfg.dataset_name}, arch={model_cfg.arch}, "
          f"model_type={model_cfg.model_type}, dim={model_cfg.dim}")

    # Create dataset
    print("Loading validation dataset...")
    test_dataset = ContrastiveSegDataset(
        data_dir=data_dir,
        dataset_name=model_cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(res, False, "center"),
        target_transform=get_transform(res, True, "center"),
        cfg=model_cfg,
    )
    print(f"  Validation set size: {len(test_dataset)} images")

    test_loader = DataLoader(
        test_dataset, batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=False, collate_fn=flexible_collate
    )

    # Eval mode
    model.eval()
    if DEVICE.type == 'cuda':
        model = model.cuda()

    par_model = model.net

    # Metrics tracking
    saved_data = defaultdict(list)
    all_good_images = list(range(min(20, len(test_dataset))))  # Save first 20 for visualization
    batch_nums = torch.tensor([n // batch_size for n in all_good_images])
    batch_offsets = torch.tensor([n % batch_size for n in all_good_images])

    print(f"\nRunning inference on {len(test_dataset)} images...")
    for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        with torch.no_grad():
            img = batch["img"]
            label = batch["label"]

            if DEVICE.type == 'cuda':
                img = img.cuda()
                label = label.cuda()

            # Forward pass (with horizontal flip augmentation)
            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2

            # Upsample to label resolution
            code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

            # Get predictions
            linear_probs = torch.log_softmax(model.linear_probe(code), dim=1)
            cluster_probs = model.cluster_probe(code, 2, log_probs=True)

            # CRF post-processing
            if run_crf and HAS_CRF:
                linear_preds = batched_crf(img, linear_probs).argmax(1)
                cluster_preds = batched_crf(img, cluster_probs).argmax(1)
            else:
                linear_preds = linear_probs.argmax(1)
                cluster_preds = cluster_probs.argmax(1)

            # Update metrics
            model.test_linear_metrics.update(linear_preds.cpu(), label.cpu())
            model.test_cluster_metrics.update(cluster_preds.cpu(), label.cpu())

            # Save visualization data
            if i in batch_nums:
                matching_offsets = batch_offsets[torch.where(batch_nums == i)]
                for offset in matching_offsets:
                    saved_data["linear_preds"].append(linear_preds.cpu()[offset].unsqueeze(0))
                    saved_data["cluster_preds"].append(cluster_preds.cpu()[offset].unsqueeze(0))
                    saved_data["label"].append(label.cpu()[offset].unsqueeze(0))
                    saved_data["img"].append(img.cpu()[offset].unsqueeze(0))

    # Compute final metrics
    print("\nComputing metrics...")
    tb_metrics = {
        **model.test_linear_metrics.compute(),
        **model.test_cluster_metrics.compute(),
    }

    # Print results
    print(f"\n{'='*60}")
    print("  EVALUATION RESULTS")
    print(f"{'='*60}")
    for k, v in sorted(tb_metrics.items()):
        if isinstance(v, torch.Tensor):
            v = v.item()
        print(f"  {k}: {v:.4f}")

    # Save results to file
    import json
    results_file = os.path.join(output_dir, "metrics.json")
    results = {k: float(v) if isinstance(v, (torch.Tensor, float)) else v
               for k, v in tb_metrics.items()}
    results["model_path"] = model_path
    results["data_dir"] = data_dir
    results["device"] = str(DEVICE)
    results["num_images"] = len(test_dataset)
    results["crf_enabled"] = run_crf and HAS_CRF
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nMetrics saved to: {results_file}")

    # Save visualizations
    if saved_data:
        print("Saving visualizations...")
        saved_data = {k: torch.cat(v, dim=0) for k, v in saved_data.items()}
        for idx in range(min(len(saved_data.get("img", [])), 20)):
            plot_img = (prep_for_plot(saved_data["img"][idx]) * 255).numpy().astype(np.uint8)
            plot_label = (model.label_cmap[saved_data["label"][idx]]).astype(np.uint8)
            plot_cluster = (model.label_cmap[
                model.test_cluster_metrics.map_clusters(
                    saved_data["cluster_preds"][idx])]).astype(np.uint8)

            Image.fromarray(plot_img).save(os.path.join(output_dir, "img", f"{idx}.jpg"))
            Image.fromarray(plot_label).save(os.path.join(output_dir, "label", f"{idx}.png"))
            Image.fromarray(plot_cluster).save(os.path.join(output_dir, "cluster", f"{idx}.png"))

        print(f"Visualizations saved to: {output_dir}")

    print(f"\n{'='*60}")
    print("  EVALUATION COMPLETE")
    print(f"{'='*60}")
    return tb_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DepthG/STEGO evaluation (CPU-compatible)")
    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory containing cityscapes/")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .ckpt checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/depthg_eval", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--res", type=int, default=320, help="Evaluation resolution")
    parser.add_argument("--no_crf", action="store_true", help="Disable CRF post-processing")
    args = parser.parse_args()

    run_eval(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        res=args.res,
        run_crf=not args.no_crf,
    )
PYEOF

echo "  Created: $PROJ_DIR/scripts/eval_depthg_cpu.py"
echo ""

# ─── 5. RUN EVALUATION ───
echo "=== [5/5] Running DepthG evaluation ==="
cd $PROJ_DIR

python3 scripts/eval_depthg_cpu.py \
    --data_dir "$DATA_DIR" \
    --model_path "$EVAL_CKPT" \
    --output_dir "results/depthg_eval" \
    --batch_size 8 \
    --num_workers 4 \
    --res 320

echo ""
echo "=== DONE ==="
echo "Results at: $PROJ_DIR/results/depthg_eval/metrics.json"
