#!/usr/bin/env python3
"""
Proper CUPS inference from PyTorch Lightning checkpoint.

Extracts instance predictions from the trained CUPS model (step 4000)
to combine with UNet semantics.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm

REPO_ROOT = Path("/media/santosh/Kuldeep/panoptic_segmentation")
sys.path.insert(0, str(REPO_ROOT / "refs/cups"))
sys.path.insert(0, str(REPO_ROOT))

def extract_cups_checkpoint_weights(checkpoint_path):
    """Extract model weights from PyTorch Lightning checkpoint.

    Lightning saves under 'state_dict' with 'model.' prefix on keys.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Lightning checkpoint structure
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        # Remove 'model.' prefix from keys
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        print(f"✓ Extracted {len(state_dict)} weights from Lightning checkpoint")
    else:
        state_dict = ckpt
        print(f"✓ Extracted {len(state_dict)} weights")

    return state_dict

def load_cups_detectron2(checkpoint_path, device="cuda"):
    """Load CUPS as Detectron2 model using extracted weights."""
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer

    print("Setting up Detectron2 configuration...")
    cfg = get_cfg()

    # Use Mask R-CNN config as base (CUPS extends this)
    from detectron2 import model_zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # CUPS checkpoint was trained with 15 classes (k=80 things mapped to 15)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15
    cfg.MODEL.DEVICE = device

    # Build model
    print("Building model...")
    model = build_model(cfg)

    # Load weights
    print("Loading extracted weights...")
    state_dict = extract_cups_checkpoint_weights(checkpoint_path)

    # Try to load with strict=False since we may have mismatches
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")

    model = model.to(device).eval()
    print(f"✓ CUPS model loaded and ready")

    return model, cfg

def run_cups_inference(model, val_images_dir, output_dir, device="cuda", num_images=None):
    """Run CUPS inference to extract instance predictions."""
    from detectron2.structures import Instances

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[CUPS Inference] Extracting instances from trained model...")

    val_files = sorted(Path(val_images_dir).rglob("*leftImg8bit.png"))
    if num_images:
        val_files = val_files[:num_images]

    for img_path in tqdm(val_files, desc="CUPS inference"):
        img_name = img_path.stem.replace("_leftImg8bit", "")

        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Prepare input (BGR to RGB, add batch dim if needed)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # CUPS forward pass
            with torch.no_grad():
                # Convert to tensor format expected by model
                img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().to(device)
                img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

                # Get predictions
                outputs = model([{"image": img_tensor}])

                if "instances" in outputs[0]:
                    instances = outputs[0]["instances"]

                    # Extract masks and classes
                    if len(instances) > 0:
                        masks = instances.pred_masks.cpu().numpy()  # [N, H, W]
                        classes = instances.pred_classes.cpu().numpy()  # [N]

                        # Create instance map
                        instance_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16)
                        for inst_id, (mask, cls_id) in enumerate(zip(masks, classes)):
                            instance_map[mask > 0.5] = inst_id + 1
                    else:
                        instance_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16)
                else:
                    instance_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16)

            # Save instance map
            output_path = output_dir / f"{img_name}_instanceIds.png"
            cv2.imwrite(str(output_path), instance_map)

        except Exception as e:
            print(f"Error on {img_name}: {e}")
            # Create empty instance map
            instance_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16)
            output_path = output_dir / f"{img_name}_instanceIds.png"
            cv2.imwrite(str(output_path), instance_map)
            continue

    print(f"✓ CUPS inference complete: {len(list(output_dir.glob('*.png')))} instances")
    return output_dir

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cups_ckpt = REPO_ROOT / "experiments/experiments/cups_resnet50_k80_stage2/Unsupervised Panoptic Segmentation/bquaqla1/checkpoints/ups_checkpoint_step=004000.ckpt"
    val_images = REPO_ROOT / "datasets/cityscapes/leftImg8bit/val"
    output_dir = Path("/tmp/cups_instances_extracted")

    print("=" * 80)
    print("CUPS Instance Extraction from Trained Model")
    print("=" * 80)

    try:
        # Load CUPS model
        model, cfg = load_cups_detectron2(str(cups_ckpt), device=device)

        # Run inference
        instance_dir = run_cups_inference(
            model,
            str(val_images),
            output_dir,
            device=device,
            num_images=10  # Test with 10, remove for full
        )

        print("\n" + "=" * 80)
        print(f"✓ CUPS instances extracted to: {instance_dir}")
        print("\nNext: Combine with UNet semantics for final evaluation")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
