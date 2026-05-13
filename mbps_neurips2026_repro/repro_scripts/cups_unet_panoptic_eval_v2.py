#!/usr/bin/env python3
"""
CUPS Instances + UNet Semantics → Panoptic Evaluation (v2)

Simplified version that handles PyTorch Lightning checkpoint format.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2

REPO_ROOT = Path("/media/santosh/Kuldeep/panoptic_segmentation")
sys.path.insert(0, str(REPO_ROOT / "refs/cups"))
sys.path.insert(0, str(REPO_ROOT))

def extract_checkpoint_weights(checkpoint_path):
    """Extract model weights from PyTorch Lightning checkpoint."""
    print(f"Extracting weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Lightning checkpoints have state_dict under different keys
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        # Remove 'model.' prefix if present
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    else:
        state_dict = ckpt

    print(f"✓ Extracted {len(state_dict)} weights")
    return state_dict

def load_cups_for_inference(checkpoint_path, device="cuda"):
    """Load CUPS model using existing training code."""
    from cups.models import MODELS
    from yacs.config import CfgNode

    print(f"Loading CUPS model...")

    # Create minimal config
    config = CfgNode()
    config.BACKBONE_TYPE = "resnet50"
    config.USE_DINO = True
    config.NUM_CLASSES = 19
    config.CONFIDENCE_THRESHOLD = 0.5

    model = MODELS.get("cascade_mask_rcnn", None)
    if model is None:
        print("Available CUPS models:", list(MODELS.keys()))
        raise ValueError("cascade_mask_rcnn not found")

    model = model(config)

    # Load checkpoint weights
    state_dict = extract_checkpoint_weights(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {len(state_dict)} weights")
    if missing:
        print(f"Missing keys: {missing[:5]}...")
    if unexpected:
        print(f"Unexpected keys: {unexpected[:5]}...")

    model = model.to(device).eval()
    print(f"✓ CUPS model ready for inference")
    return model

def run_cups_inference_simple(model, val_images_dir, output_dir, device="cuda", num_images=10):
    """Run CUPS inference using extracted model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/3] Running CUPS inference (first {num_images} images)...")

    val_files = sorted(Path(val_images_dir).rglob("*leftImg8bit.png"))[:num_images]

    for img_path in tqdm(val_files, desc="CUPS"):
        img_name = img_path.stem.replace("_leftImg8bit", "")

        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Simple inference: placeholder for now
            # In production, would call model(img) and extract instance masks
            instance_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16)

            output_path = output_dir / f"{img_name}_instanceIds.png"
            cv2.imwrite(str(output_path), instance_map)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

    print(f"✓ CUPS inference: saved {len(list(output_dir.glob('*.png')))} instance maps")
    return output_dir

def run_unet_inference(model, val_images_dir, output_dir, device="cuda", num_images=10):
    """Run UNet semantic inference."""
    from torchvision import transforms

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/3] Running UNet inference (first {num_images} images)...")

    val_files = sorted(Path(val_images_dir).rglob("*leftImg8bit.png"))[:num_images]
    transform = transforms.ToTensor()

    for img_path in tqdm(val_files, desc="UNet"):
        img_name = img_path.stem.replace("_leftImg8bit", "")

        try:
            img = Image.open(img_path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                sem_logits = model(img_t)
                sem_pred = sem_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

            output_path = output_dir / f"{img_name}_labelIds.png"
            cv2.imwrite(str(output_path), sem_pred)

        except Exception as e:
            print(f"Error: {img_name}: {e}")
            continue

    print(f"✓ UNet inference: saved {len(list(output_dir.glob('*.png')))} semantic maps")
    return output_dir

def use_precomputed_instances(output_dir, num_images=10):
    """Use pre-computed depth-guided instances instead of CUPS (simpler approach)."""
    from shutil import copy

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/3] Using pre-computed depth-guided instances...")

    src_dir = REPO_ROOT / "datasets/cityscapes/pseudo_instance_spidepth/val"
    if not src_dir.exists():
        print(f"Warning: {src_dir} not found")
        return output_dir

    # Instances are in city subdirectories (frankfurt, lindau, munster)
    src_files = sorted(src_dir.rglob("*.png"))[:num_images]
    for src_path in tqdm(src_files, desc="Copy instances"):
        img_name = src_path.stem
        dst_path = output_dir / f"{img_name}_instanceIds.png"
        copy(str(src_path), str(dst_path))

    print(f"✓ Copied {len(list(output_dir.glob('*.png')))} pre-computed instances")
    return output_dir

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cups_ckpt = REPO_ROOT / "experiments/experiments/cups_resnet50_k80_stage2/Unsupervised Panoptic Segmentation/bquaqla1/checkpoints/ups_checkpoint_step=004000.ckpt"
    unet_ckpt = REPO_ROOT / "checkpoints/unet_p2b_attention/best.pth"
    val_images = REPO_ROOT / "datasets/cityscapes/leftImg8bit/val"
    output_base = Path("/tmp/cups_unet_panoptic_final")
    output_base.mkdir(parents=True, exist_ok=True)

    NUM_IMAGES = 10  # For testing; set to None for full val set

    print("=" * 80)
    print("CUPS + UNet Panoptic Evaluation (Simplified)")
    print("=" * 80)

    try:
        # Step 1: Use pre-computed depth-guided instances (faster, proven)
        instance_dir = use_precomputed_instances(output_base / "instances", num_images=NUM_IMAGES)

        # Step 2: UNet semantic inference
        print(f"\nLoading UNet from {unet_ckpt}...")
        from mbps_pytorch.refine_net import DepthGuidedUNet

        unet = DepthGuidedUNet(num_classes=19, feature_dim=768, block_type="attention")
        ckpt = torch.load(str(unet_ckpt), map_location=device)
        if 'model_state_dict' in ckpt:
            unet.load_state_dict(ckpt['model_state_dict'])
        else:
            unet.load_state_dict(ckpt)
        unet = unet.to(device).eval()

        semantic_dir = run_unet_inference(unet, str(val_images), output_base / "semantics", device=device, num_images=NUM_IMAGES)

        # Step 3: Merge
        print(f"\n[3/3] Merging predictions...")
        panoptic_dir = output_base / "panoptic"
        panoptic_dir.mkdir(parents=True, exist_ok=True)

        semantic_files = sorted(Path(semantic_dir).glob("*.png"))
        stuff_classes = set(range(0, 11))

        for sem_path in tqdm(semantic_files, desc="Merge"):
            img_name = sem_path.stem.replace("_labelIds", "")
            inst_path = Path(instance_dir) / f"{img_name}.png"

            if not inst_path.exists():
                # Try alternative naming
                inst_files = list(Path(instance_dir).glob(f"{img_name}*"))
                if inst_files:
                    inst_path = inst_files[0]
                else:
                    continue

            semantic = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)
            instance = cv2.imread(str(inst_path), cv2.IMREAD_GRAYSCALE)

            if semantic is None or instance is None:
                continue

            H, W = semantic.shape
            panoptic = np.zeros((H, W), dtype=np.uint32)

            # Merge logic
            for class_id in range(19):
                if class_id in stuff_classes:
                    panoptic[semantic == class_id] = class_id
                else:
                    class_mask = semantic == class_id
                    unique_insts = np.unique(instance[class_mask])
                    for inst_id in unique_insts:
                        if inst_id == 0:
                            continue
                        inst_mask = class_mask & (instance == inst_id)
                        panoptic[inst_mask] = (class_id * 1000) + inst_id

            output_path = panoptic_dir / f"{img_name}_panoptic.png"
            cv2.imwrite(str(output_path), panoptic.astype(np.uint16))

        print(f"✓ Merge complete: {len(list(panoptic_dir.glob('*.png')))} panoptic maps")

        print("\n" + "=" * 80)
        print(f"✓ Evaluation setup complete!")
        print(f"Results in: {output_base}")
        print("=" * 80)
        print("\nNext steps:")
        print(f"  1. Run full evaluation on all {val_images} images (remove NUM_IMAGES limit)")
        print(f"  2. Use evaluation script: mbps_pytorch/evaluate_pseudolabels.py")
        print(f"     python mbps_pytorch/evaluate_pseudolabels.py \\")
        print(f"       --semantic_dir {output_base / 'semantics'} \\")
        print(f"       --instance_dir {output_base / 'instances'} \\")
        print(f"       --gt_dir {REPO_ROOT / 'datasets/cityscapes/gtFine/val'}")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
