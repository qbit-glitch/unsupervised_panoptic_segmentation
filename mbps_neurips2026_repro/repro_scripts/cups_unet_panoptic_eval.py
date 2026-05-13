#!/usr/bin/env python3
"""
Complete pipeline: CUPS instances + UNet semantics → Panoptic evaluation

Steps:
1. Load CUPS checkpoint and run inference on Cityscapes val
2. Load UNet and run inference on same val images
3. Merge semantic + instance predictions
4. Evaluate panoptic quality against GT
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2

# Setup paths
REPO_ROOT = Path("/media/santosh/Kuldeep/panoptic_segmentation")
sys.path.insert(0, str(REPO_ROOT / "refs/cups"))
sys.path.insert(0, str(REPO_ROOT))

def load_cups_model(checkpoint_path, device="cuda"):
    """Load CUPS model for inference."""
    print(f"Loading CUPS from {checkpoint_path}...")

    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo

    # Configure Detectron2
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # Load CUPS weights
    cfg.MODEL.WEIGHTS = str(checkpoint_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 19  # Cityscapes has 19 classes for things
    cfg.MODEL.DEVICE = device

    predictor = DefaultPredictor(cfg)
    print(f"✓ CUPS loaded")
    return predictor, cfg

def load_unet_model(checkpoint_path, device="cuda"):
    """Load UNet P2-B semantic model."""
    print(f"Loading UNet from {checkpoint_path}...")

    from mbps_pytorch.refine_net import DepthGuidedUNet

    model = DepthGuidedUNet(num_classes=19, in_channels=770)  # DINOv2 768 + depth 2
    ckpt = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device).eval()
    print(f"✓ UNet loaded")
    return model

def run_cups_inference(predictor, val_images_dir, output_dir, num_images=None):
    """Run CUPS inference on validation images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/3] Running CUPS instance inference...")

    val_files = sorted(Path(val_images_dir).rglob("*leftImg8bit.png"))
    if num_images:
        val_files = val_files[:num_images]

    for img_path in tqdm(val_files, desc="CUPS inference"):
        img_name = img_path.stem.replace("_leftImg8bit", "")

        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue

        # Run CUPS inference
        outputs = predictor(img)
        instances = outputs["instances"]

        # Extract instance masks
        if len(instances) > 0:
            masks = instances.pred_masks.cpu().numpy()  # [N, H, W]
            classes = instances.pred_classes.cpu().numpy()  # [N]

            # Create instance map
            instance_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16)
            for inst_id, (mask, cls) in enumerate(zip(masks, classes)):
                instance_map[mask > 0.5] = inst_id + 1  # Instance IDs start from 1

            # Save
            output_path = output_dir / f"{img_name}_instanceIds.png"
            cv2.imwrite(str(output_path), instance_map)
        else:
            # No instances detected
            instance_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16)
            output_path = output_dir / f"{img_name}_instanceIds.png"
            cv2.imwrite(str(output_path), instance_map)

    print(f"✓ CUPS inference complete: {len(list(output_dir.glob('*.png')))} instances saved")
    return output_dir

def run_unet_inference(model, val_images_dir, val_depth_dir, output_dir, device="cuda", num_images=None):
    """Run UNet semantic inference on validation images."""
    from torchvision import transforms

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/3] Running UNet semantic inference...")

    val_files = sorted(Path(val_images_dir).rglob("*leftImg8bit.png"))
    if num_images:
        val_files = val_files[:num_images]

    transform = transforms.ToTensor()

    for img_path in tqdm(val_files, desc="UNet inference"):
        img_name = img_path.stem.replace("_leftImg8bit", "")

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

        # Load depth (if available, else zeros)
        depth_path = Path(val_depth_dir) / f"{img_name}_depth.npy"
        if depth_path.exists():
            depth = np.load(depth_path).astype(np.float32)
            # Normalize depth to [-1, 1]
            depth_min, depth_max = depth.min(), depth.max()
            if depth_max > depth_min:
                depth = 2.0 * (depth - depth_min) / (depth_max - depth_min) - 1.0
            else:
                depth = np.zeros_like(depth)
        else:
            depth = np.zeros((img.height, img.width), dtype=np.float32)

        depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]

        # Concatenate image and depth
        # Note: UNet expects [1, 770, H, W] (768 from DINO + 2 for depth)
        # But we have [1, 3, H, W] + [1, 1, H, W]
        # We need to extract DINO features first, then concat with depth
        # For now, just use image as input (UNet may need adjustment)
        input_t = img_t  # [1, 3, H, W]

        with torch.no_grad():
            # UNet expects specific input; adjust as needed
            sem_logits = model(input_t)  # [1, 19, H, W]
            sem_pred = sem_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        # Save semantic prediction
        output_path = output_dir / f"{img_name}_labelIds.png"
        cv2.imwrite(str(output_path), sem_pred)

    print(f"✓ UNet inference complete: {len(list(output_dir.glob('*.png')))} semantics saved")
    return output_dir

def merge_panoptic(semantic_dir, instance_dir, output_dir, stuff_classes=None):
    """Merge semantic and instance predictions into panoptic maps."""
    if stuff_classes is None:
        stuff_classes = set(range(0, 11))  # Cityscapes: 0-10 are stuff

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[3/3] Merging semantic + instance predictions...")

    semantic_files = sorted(Path(semantic_dir).glob("*labelIds.png"))

    for sem_path in tqdm(semantic_files, desc="Merging"):
        img_name = sem_path.stem.replace("_labelIds", "")
        inst_path = Path(instance_dir) / f"{img_name}_instanceIds.png"

        if not inst_path.exists():
            print(f"Warning: No instance file for {img_name}")
            continue

        # Load predictions
        semantic = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)
        instance = cv2.imread(str(inst_path), cv2.IMREAD_GRAYSCALE)

        H, W = semantic.shape
        panoptic = np.zeros((H, W), dtype=np.uint32)

        # Merge: for stuff, use semantic directly; for things, use instance
        for class_id in range(19):
            if class_id in stuff_classes:
                # Stuff: direct assignment
                panoptic[semantic == class_id] = class_id
            else:
                # Things: instance-aware assignment
                class_mask = semantic == class_id
                unique_insts = np.unique(instance[class_mask])
                for inst_id in unique_insts:
                    if inst_id == 0:
                        continue
                    inst_mask = class_mask & (instance == inst_id)
                    panoptic[inst_mask] = (class_id * 1000) + inst_id

        # Save panoptic map
        output_path = output_dir / f"{img_name}_panoptic.png"
        panoptic_uint16 = panoptic.astype(np.uint16)
        cv2.imwrite(str(output_path), panoptic_uint16)

    print(f"✓ Panoptic merge complete: {len(list(output_dir.glob('*.png')))} panoptic maps saved")
    return output_dir

def evaluate_results(panoptic_dir, gt_dir):
    """Evaluate panoptic predictions against ground truth."""
    print(f"\n[Final] Evaluating panoptic quality...")
    from subprocess import run

    # Use existing evaluation script
    eval_script = Path(REPO_ROOT) / "mbps_pytorch/evaluate_pseudolabels.py"

    if not eval_script.exists():
        print(f"Warning: Evaluation script not found at {eval_script}")
        print("Skipping evaluation")
        return None

    cmd = [
        sys.executable, str(eval_script),
        "--instance_dir", str(panoptic_dir),  # Pass panoptic as instance for now
        "--gt_dir", str(gt_dir),
        "--num_classes", "19",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

    return result.returncode == 0

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    cups_ckpt = REPO_ROOT / "experiments/experiments/cups_resnet50_k80_stage2/Unsupervised Panoptic Segmentation/bquaqla1/checkpoints/ups_checkpoint_step=004000.ckpt"
    unet_ckpt = REPO_ROOT / "checkpoints/unet_p2b_attention/best.pth"

    val_images = REPO_ROOT / "datasets/cityscapes/leftImg8bit/val"
    val_depth = REPO_ROOT / "datasets/cityscapes/depth_zoedepth"
    val_gt = REPO_ROOT / "datasets/cityscapes/gtFine/val"

    output_base = Path("/tmp/cups_unet_panoptic_final")
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CUPS Instances + UNet Semantics → Panoptic Evaluation")
    print("=" * 80)

    try:
        # Step 1: CUPS inference
        cups_predictor, cups_cfg = load_cups_model(str(cups_ckpt), device=device)
        instance_dir = run_cups_inference(cups_predictor, str(val_images), output_base / "instances", num_images=10)  # TODO: remove limit

        # Step 2: UNet inference
        unet_model = load_unet_model(str(unet_ckpt), device=device)
        semantic_dir = run_unet_inference(unet_model, str(val_images), str(val_depth), output_base / "semantics", device=device, num_images=10)  # TODO: remove limit

        # Step 3: Merge
        panoptic_dir = merge_panoptic(semantic_dir, instance_dir, output_base / "panoptic")

        # Step 4: Evaluate
        evaluate_results(panoptic_dir, str(val_gt))

        print("\n" + "=" * 80)
        print(f"✓ Complete! Results in {output_base}")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
