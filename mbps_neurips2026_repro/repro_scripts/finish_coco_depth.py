#!/usr/bin/env python3
"""Compute remaining COCO depth maps not yet on GCS, then upload."""

import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

COCO_DIR = os.path.expanduser("~/mbps_panoptic_segmentation/data/coco")
DEPTH_DIR = os.path.join(COCO_DIR, "depth_zoedepth")
BUCKET_DEPTH = "gs://mbps-panoptic/datasets/coco/depth_zoedepth"


def get_gcs_stems():
    """Get set of depth map stems already on GCS."""
    print("Listing existing depth maps on GCS...")
    result = subprocess.run(
        ["gsutil", "ls", f"{BUCKET_DEPTH}/train2017/"],
        capture_output=True, text=True, timeout=120,
    )
    stems = set()
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.endswith(".npy"):
            stems.add(Path(line).stem)
    # Also check val2017
    result2 = subprocess.run(
        ["gsutil", "ls", f"{BUCKET_DEPTH}/val2017/"],
        capture_output=True, text=True, timeout=120,
    )
    for line in result2.stdout.strip().split("\n"):
        line = line.strip()
        if line.endswith(".npy"):
            stems.add(Path(line).stem)
    return stems


def main():
    # Collect all images
    images = sorted(glob.glob(os.path.join(COCO_DIR, "images", "**", "*.jpg"), recursive=True))
    print(f"Total COCO images: {len(images)}")

    # Find what's already on GCS
    done = get_gcs_stems()
    print(f"Already on GCS: {len(done)} depth maps")

    # Filter to missing
    missing = []
    for img_path in images:
        stem = Path(img_path).stem
        if stem not in done:
            missing.append(img_path)
    print(f"Missing: {len(missing)} depth maps to compute")

    if not missing:
        print("All depth maps are on GCS. Nothing to do.")
        return

    # Load ZoeDepth with compatibility patches
    import torch

    _orig_load = torch.nn.Module.load_state_dict
    torch.nn.Module.load_state_dict = (
        lambda self, state_dict, strict=True, **kw:
        _orig_load(self, state_dict, strict=False, **kw)
    )

    # Patch cached MiDaS beit.py
    midas_beit = os.path.expanduser(
        "~/.cache/torch/hub/intel-isl_MiDaS_master/midas/backbones/beit.py"
    )
    if os.path.exists(midas_beit):
        with open(midas_beit, "r") as f:
            src = f.read()
        if "self.drop_path(" in src and "self.drop_path1(" not in src:
            src = src.replace("self.drop_path(", "self.drop_path1(")
            with open(midas_beit, "w") as f:
                f.write(src)
            import shutil
            pycache = os.path.join(os.path.dirname(midas_beit), "__pycache__")
            if os.path.exists(pycache):
                shutil.rmtree(pycache, ignore_errors=True)
            print("Patched MiDaS beit.py: drop_path -> drop_path1")

    try:
        model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)
    finally:
        torch.nn.Module.load_state_dict = _orig_load

    model.eval()
    print("ZoeDepth loaded (CPU mode)")

    from PIL import Image
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor

    transform = Compose([
        Resize((512, 512)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    os.makedirs(DEPTH_DIR, exist_ok=True)
    start = time.time()

    for idx, img_path in enumerate(missing):
        rel = os.path.relpath(img_path, os.path.join(COCO_DIR, "images"))
        depth_path = os.path.join(DEPTH_DIR, str(Path(rel).with_suffix(".npy")))
        os.makedirs(os.path.dirname(depth_path), exist_ok=True)

        img = Image.open(img_path).convert("RGB")
        inp = transform(img).unsqueeze(0)
        with torch.no_grad():
            depth = model.infer(inp)
        np.save(depth_path, depth.squeeze().cpu().numpy())

        if (idx + 1) % 100 == 0 or (idx + 1) == len(missing):
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed
            remaining = (len(missing) - idx - 1) / rate if rate > 0 else 0
            print(f"[{idx+1}/{len(missing)}] {rate:.1f} img/s, ETA: {remaining:.0f}s")

    print(f"Computed {len(missing)} depth maps. Uploading to GCS...")
    subprocess.run(
        ["gsutil", "-m", "rsync", "-r", DEPTH_DIR + "/", BUCKET_DEPTH + "/"],
        check=True,
    )
    print("Upload complete.")

    # Cleanup local depth maps
    import shutil
    shutil.rmtree(DEPTH_DIR, ignore_errors=True)
    print("Local depth maps cleaned up.")


if __name__ == "__main__":
    main()
