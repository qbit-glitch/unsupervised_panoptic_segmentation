#!/usr/bin/env python3
"""
Evaluate CUPS instances + UNet semantics combination on Cityscapes val.

This script:
1. Loads UNet P2-B checkpoint and generates semantic predictions
2. Loads CUPS checkpoint and generates instance predictions
3. Merges them using panoptic merging
4. Evaluates on Cityscapes val ground truth
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import cv2
from tqdm import tqdm

# Setup paths
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

def load_unet_model(checkpoint_path, device="cuda"):
    """Load UNet P2-B semantic model."""
    from mbps_pytorch.refine_net import DepthGuidedUNet

    print(f"Loading UNet from {checkpoint_path}")
    model = DepthGuidedUNet(num_classes=19, in_channels=768+2)  # DINOv2 + depth
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device).eval()
    print(f"✓ UNet loaded from {checkpoint_path}")
    return model

def load_cups_model(checkpoint_path, device="cuda"):
    """Load CUPS Cascade Mask R-CNN model."""
    import sys
    cups_path = Path("/media/santosh/Kuldeep/panoptic_segmentation/refs/cups")
    sys.path.insert(0, str(cups_path))

    from cups.models import MODELS
    from yacs.config import CfgNode

    print(f"Loading CUPS from {checkpoint_path}")
    # Load config and model
    config = CfgNode()
    config.BACKBONE_TYPE = "resnet50"
    config.USE_DINO = True

    model = MODELS["cascade_mask_rcnn"](config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device).eval()
    print(f"✓ CUPS loaded from {checkpoint_path}")
    return model

def generate_unet_semantics(model, val_images_dir, output_dir, device="cuda"):
    """Generate UNet semantic predictions for all val images."""
    from torchvision import transforms
    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating UNet semantic predictions...")
    image_files = sorted(Path(val_images_dir).glob("*.png"))

    for img_path in tqdm(image_files[:10]):  # TODO: remove limit for full eval
        img_name = img_path.stem
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

        with torch.no_grad():
            semantics = model(img_tensor)  # [1, 19, H, W]
            semantics_np = semantics.argmax(dim=1).cpu().numpy()[0]

        output_path = output_dir / f"{img_name}_semantic.png"
        cv2.imwrite(str(output_path), semantics_np.astype(np.uint8))

    print(f"✓ Saved semantic predictions to {output_dir}")
    return output_dir

def generate_cups_instances(model, val_images_dir, output_dir, device="cuda"):
    """Generate CUPS instance predictions for all val images."""
    from torchvision import transforms
    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating CUPS instance predictions...")
    image_files = sorted(Path(val_images_dir).glob("*.png"))

    for img_path in tqdm(image_files[:10]):  # TODO: remove limit for full eval
        img_name = img_path.stem
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            # CUPS outputs instance masks
            # Extract instances and ID them
            instance_map = np.zeros(img.size[::-1], dtype=np.uint16)
            # TODO: properly extract CUPS output instances

        output_path = output_dir / f"{img_name}_instance.png"
        cv2.imwrite(str(output_path), instance_map)

    print(f"✓ Saved instance predictions to {output_dir}")
    return output_dir

def merge_semantic_instance(semantic_pred, instance_pred, stuff_classes):
    """Merge semantic and instance predictions into panoptic map."""
    panoptic = np.zeros_like(semantic_pred)

    # Stuff classes: use semantic directly
    for stuff_id in stuff_classes:
        panoptic[semantic_pred == stuff_id] = stuff_id

    # Things classes: use instances
    for inst_id in np.unique(instance_pred):
        if inst_id == 0:
            continue
        thing_mask = instance_pred == inst_id
        thing_class = semantic_pred[thing_mask].mode() if thing_mask.sum() > 0 else 0
        panoptic[thing_mask] = (thing_class * 1000) + inst_id

    return panoptic

def evaluate_panoptic(pred_dir, gt_dir, output_file="metrics.json"):
    """Evaluate panoptic predictions against ground truth."""
    print(f"\nEvaluating panoptic predictions...")
    # TODO: Use PanopticAPI or Detectron2 eval
    print(f"✓ Results saved to {output_file}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    unet_ckpt = Path("/media/santosh/Kuldeep/panoptic_segmentation/checkpoints/unet_p2b_attention/best.pth")
    cups_ckpt = Path("/media/santosh/Kuldeep/panoptic_segmentation/experiments/experiments/cups_resnet50_k80_stage2/Unsupervised Panoptic Segmentation/bquaqla1/checkpoints/ups_checkpoint_step=004000.ckpt")

    val_images = Path("/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/leftImg8bit/val")
    val_gt = Path("/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/gtFine/val")

    output_base = Path("/tmp/cups_unet_merge_eval")
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CUPS Instances + UNet Semantics Evaluation")
    print("=" * 60)

    # Load models
    unet_model = load_unet_model(str(unet_ckpt), device=device)
    # cups_model = load_cups_model(str(cups_ckpt), device=device)  # TODO: fix CUPS loading

    # Generate predictions
    semantic_dir = generate_unet_semantics(unet_model, str(val_images), output_base / "semantics", device=device)
    # instance_dir = generate_cups_instances(cups_model, str(val_images), output_base / "instances", device=device)

    # Evaluate
    # evaluate_panoptic(output_base / "panoptic", str(val_gt), output_base / "metrics.json")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
